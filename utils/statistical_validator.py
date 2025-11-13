"""
Advanced statistical validation checks for generated data.
Implements the Tier 2 validation suite from spec §§9.2-9.5.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalValidator:
    """Runs statistical distribution, correlation, and plausibility checks."""

    def __init__(self, distributions: Dict[str, Any], reference_date: str = '2024-01-01'):
        self.distributions = distributions
        self.reference_date = pd.to_datetime(reference_date)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.report: List[Dict[str, Any]] = []
        self.members = pd.DataFrame()
        self.member_conditions = None
        self.condition_lookup = None
        self.condition_name_map: Dict[str, int] = {}

    def run(self, tables: Dict[str, pd.DataFrame], sample_size: int | None = None) -> bool:
        """Execute the statistical validation suite."""
        self.errors.clear()
        self.warnings.clear()
        self.report.clear()

        if 'MEMBERS' not in tables:
            self.errors.append("Statistical validation requires MEMBERS table.")
            return False

        members = tables['MEMBERS'].copy()
        if sample_size and len(members) > sample_size:
            members = members.sample(sample_size, random_state=42)

        self.members = self._prepare_members(members)
        self.member_conditions = tables.get('MEMBER_CONDITION')
        self.condition_lookup = tables.get('CONDITION')
        self._prepare_condition_lookup()

        self._run_ks_tests()
        self._run_sex_chi_square()
        self._run_correlation_checks()
        self._run_condition_prevalence_checks()

        return len(self.errors) == 0

    # ------------------------------------------------------------------ #
    # Preparation helpers
    # ------------------------------------------------------------------ #
    def _prepare_members(self, members: pd.DataFrame) -> pd.DataFrame:
        members = members.copy()
        members['DOB'] = pd.to_datetime(members['DOB'], errors='coerce')
        ages = (self.reference_date - members['DOB']).dt.days / 365.25
        members['_age_years'] = ages
        members['_age_group'] = pd.cut(
            ages,
            bins=[-np.inf, 18, 35, 45, 55, 65, np.inf],
            labels=['0-17', '18-34', '35-44', '45-54', '55-64', '65+'],
            right=False
        )

        numeric_cols = [
            'height_cm', 'weight_kg', 'exercise_minutes_per_week',
            'sleep_hours_per_night', 'heart_rate', 'blood_oxygen'
        ]
        for col in numeric_cols:
            if col in members.columns:
                members[col] = pd.to_numeric(members[col], errors='coerce')

        for col in ['smoker', 'drinker', 'housing insecurity', 'employment status']:
            if col in members.columns:
                members[col] = pd.to_numeric(members[col], errors='coerce')

        return members

    def _prepare_condition_lookup(self):
        if self.member_conditions is None or self.condition_lookup is None:
            self.condition_name_map = {}
            return

        cond_df = self.condition_lookup[['Condition_ID', 'Condition_name']].dropna()

        def _normalize(name: str) -> str:
            return ''.join(ch.lower() for ch in name if ch.isalnum())

        self.condition_name_map = {
            _normalize(name): cond_id
            for cond_id, name in zip(cond_df['Condition_ID'], cond_df['Condition_name'])
        }

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #
    def _run_ks_tests(self):
        anthropometrics = self.distributions.get('anthropometrics', {})
        sex_map = {'M': 'male', 'F': 'female'}

        for sex_code, config_key in sex_map.items():
            params = anthropometrics.get(config_key)
            subset = self.members[self.members['Sex'] == sex_code]
            if params and len(subset) >= 200:
                self._ks_normal_test(
                    subset['height_cm'].dropna(),
                    params['height_mean_cm'],
                    params['height_std_cm'],
                    f"Height ({sex_code})"
                )
                self._ks_normal_test(
                    subset['weight_kg'].dropna(),
                    params['weight_mean_kg'],
                    params['weight_std_kg'],
                    f"Weight ({sex_code})"
                )

        # Exercise: log-normal => log transform then KS vs normal
        if 'exercise_minutes_per_week' in self.members.columns:
            exercise = self.members['exercise_minutes_per_week']
            exercise = exercise[exercise > 0]
            if len(exercise) >= 200:
                log_vals = np.log(exercise)
                mu = np.log(self.distributions['lifestyle']['exercise']['median_minutes_per_week'])
                sigma = self.distributions['lifestyle']['exercise']['std_log']
                self._ks_normal_test(log_vals, mu, sigma, "Exercise (log-space)")

        # Sleep: normal distribution
        if 'sleep_hours_per_night' in self.members.columns:
            sleep_cfg = self.distributions['lifestyle']['sleep']
            sleep = self.members['sleep_hours_per_night'].dropna()
            if len(sleep) >= 200:
                self._ks_normal_test(
                    sleep,
                    sleep_cfg['mean_hours'],
                    sleep_cfg['std_hours'],
                    "Sleep hours"
                )

    def _ks_normal_test(self, series: pd.Series, mean: float, std: float, label: str):
        if std <= 0 or series.empty:
            return
        standardized = (series - mean) / std
        stat, p_value = stats.kstest(standardized, 'norm')
        status = 'PASS' if p_value >= 0.05 else 'FAIL'
        self._log_result(
            label,
            status,
            observed=f"KS p={p_value:.3f}",
            target="Normal CDF",
            stat=stat,
            p_value=p_value
        )
        if p_value < 0.01:
            self.errors.append(f"{label}: KS test p-value {p_value:.4f} < 0.01")
        elif p_value < 0.05:
            self.warnings.append(f"{label}: KS test p-value {p_value:.4f} < 0.05")

    def _run_sex_chi_square(self):
        expected_dist = self.distributions.get('sex_distribution', {})
        observed_counts = self.members['Sex'].value_counts()
        labels = list(expected_dist.keys())
        observed = np.array([observed_counts.get(lbl, 0) for lbl in labels], dtype=float)
        n = observed.sum()
        if n == 0:
            return

        expected = np.array([expected_dist[lbl] * n for lbl in labels])
        stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        status = 'PASS' if p_value >= 0.05 else 'FAIL'
        self._log_result(
            "Sex distribution",
            status,
            observed=", ".join(f"{lbl}:{cnt:.0f}" for lbl, cnt in zip(labels, observed)),
            target=", ".join(f"{lbl}:{expected_dist[lbl]*100:.1f}%" for lbl in labels),
            stat=stat,
            p_value=p_value
        )
        if p_value < 0.05:
            self.errors.append(f"Sex distribution deviates from target (p={p_value:.4f}).")

    def _run_correlation_checks(self):
        members = self.members.dropna(subset=['height_cm', 'weight_kg'])
        anthro = self.distributions.get('anthropometrics', {})
        for sex_code, config_key in {'M': 'male', 'F': 'female'}.items():
            params = anthro.get(config_key)
            subset = members[members['Sex'] == sex_code]
            if params and len(subset) >= 200:
                actual = subset['height_cm'].corr(subset['weight_kg'])
                target = params.get('correlation', 0.6)
                delta = abs(actual - target)
                status = 'PASS' if delta <= 0.05 else 'FAIL'
                self._log_result(
                    f"Height-Weight corr ({sex_code})",
                    status,
                    observed=f"{actual:.3f}",
                    target=f"{target:.3f}"
                )
                if delta > 0.08:
                    self.errors.append(
                        f"Height-weight correlation for {sex_code} off by {delta:.3f} (actual {actual:.3f}, target {target:.3f})"
                    )

        self._run_lifestyle_correlation_checks()

    def _run_lifestyle_correlation_checks(self):
        corr_cfg = self.distributions.get('lifestyle_correlation')
        if not corr_cfg:
            return

        tolerance = corr_cfg.get('tolerance', 0.05)
        matrix = np.array(corr_cfg.get('correlation_matrix', []), dtype=float)
        order = corr_cfg.get('variable_order', [])
        pair_targets = {}
        for i, name_i in enumerate(order):
            for j in range(i + 1, len(order)):
                pair_targets[(name_i, order[j])] = matrix[i][j]

        column_map = {
            'smoker': 'smoker',
            'drinker': 'drinker',
            'exercise': 'exercise_minutes_per_week',
            'sleep': 'sleep_hours_per_night',
            'housing_insecurity': 'housing insecurity',
            'employment': 'employment status',
        }

        required_pairs = [
            ('smoker', 'drinker'),
            ('exercise', 'sleep'),
            ('housing_insecurity', 'employment'),
            ('smoker', 'exercise')
        ]

        for a, b in required_pairs:
            target = pair_targets.get((a, b)) or pair_targets.get((b, a))
            col_a = column_map.get(a)
            col_b = column_map.get(b)
            if target is None or col_a not in self.members.columns or col_b not in self.members.columns:
                continue

            pair_df = self.members[[col_a, col_b]].dropna()
            if len(pair_df) < 500:
                continue
            actual = pair_df[col_a].corr(pair_df[col_b])
            if pd.isna(actual):
                continue

            delta = abs(actual - target)
            status = 'PASS' if delta <= tolerance else 'FAIL'
            self._log_result(
                f"Corr({a},{b})",
                status,
                observed=f"{actual:.3f}",
                target=f"{target:.3f}"
            )
            if delta > tolerance + 0.02:
                self.errors.append(
                    f"Correlation {a}-{b} = {actual:.3f} (target {target:.3f}, Δ={delta:.3f})"
                )
            elif delta > tolerance:
                self.warnings.append(
                    f"Correlation {a}-{b} slightly outside tolerance (Δ={delta:.3f})"
                )

    def _run_condition_prevalence_checks(self):
        if self.member_conditions is None or self.condition_lookup is None:
            self.warnings.append("Condition prevalence checks skipped (missing MEMBER_CONDITION or CONDITION data).")
            return

        cond_df = self.member_conditions[['Member_ID', 'Condition_ID']].dropna()
        key_conditions = [
            'Hypertension',
            'Diabetes_Type_2',
            'COPD',
            'Depression',
            'Chronic_Kidney_Disease',
            'Breast_Cancer'
        ]
        prevalence_cfg = self.distributions.get('condition_prevalence', {}).get('conditions', {})

        member_group = self.members[['member_id', '_age_group']]
        member_group = member_group.dropna(subset=['_age_group'])

        for cond_name in key_conditions:
            expected_curve = prevalence_cfg.get(cond_name)
            cond_id = self._get_condition_id(cond_name)
            if not expected_curve or cond_id is None:
                self.warnings.append(f"Missing configuration or mapping for condition {cond_name}.")
                continue

            affected_members = set(
                cond_df[cond_df['Condition_ID'] == cond_id]['Member_ID'].unique()
            )

            for age_group, expected_rate in expected_curve.items():
                cohort = member_group[member_group['_age_group'] == age_group]
                if cohort.empty:
                    continue
                actual_rate = cohort['member_id'].isin(affected_members).mean()
                tolerance = max(0.02, expected_rate * 0.25)
                delta = abs(actual_rate - expected_rate)
                status = 'PASS' if delta <= tolerance else 'FAIL'
                self._log_result(
                    f"{cond_name} prevalence ({age_group})",
                    status,
                    observed=f"{actual_rate:.3f}",
                    target=f"{expected_rate:.3f}"
                )
                if delta > tolerance + 0.02:
                    self.errors.append(
                        f"{cond_name} age {age_group}: {actual_rate:.3f} vs target {expected_rate:.3f}"
                    )
                elif delta > tolerance:
                    self.warnings.append(
                        f"{cond_name} age {age_group}: slightly outside tolerance (Δ={delta:.3f})"
                    )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _get_condition_id(self, name: str) -> int | None:
        if not self.condition_name_map:
            return None

        normalized = ''.join(ch.lower() for ch in name if ch.isalnum())
        return self.condition_name_map.get(normalized)

    def _log_result(
        self,
        test: str,
        status: str,
        observed: str = "",
        target: str = "",
        stat: float | None = None,
        p_value: float | None = None
    ):
        self.report.append({
            'test': test,
            'status': status,
            'observed': observed,
            'target': target,
            'stat': stat,
            'p_value': p_value
        })

    def print_report(self):
        """Pretty-print the statistical validation report."""
        print("\n" + "=" * 80)
        print("STATISTICAL VALIDATION REPORT")
        print("=" * 80)

        if not self.report:
            print("No statistical tests were executed.")
        else:
            for entry in self.report:
                line = f"[{entry['status']}] {entry['test']}"
                if entry['observed']:
                    line += f" | observed={entry['observed']}"
                if entry['target']:
                    line += f" | target={entry['target']}"
                if entry['p_value'] is not None:
                    line += f" | p={entry['p_value']:.3f}"
                print("  - " + line)

        if self.errors:
            print(f"\n✗ Statistical Errors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  • {err}")

        if self.warnings:
            print(f"\n⚠ Statistical Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  • {warn}")

        if not self.errors and not self.warnings:
            print("\n✓ Statistical validation passed without findings.")

        print("=" * 80 + "\n")
