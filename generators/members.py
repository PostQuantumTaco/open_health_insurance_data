"""
Generate member data: demographics, biometrics, lifestyle factors, and health conditions.
This is the core of the synthetic data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from scipy import stats
import multiprocessing


class MemberGenerator:
    """Generates synthetic member data with realistic distributions."""

    def __init__(self, config: dict, distributions: dict, reference_date: str, random_seed: int = None):
        """
        Initialize the member generator.

        Args:
            config: Configuration dictionary
            distributions: Distribution parameters
            reference_date: Reference date for age calculations
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.distributions = distributions
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        self.rng = np.random.default_rng(random_seed)

    def generate_demographics(self, n_members: int, state_populations: pd.DataFrame) -> pd.DataFrame:
        """
        Generate demographic data (age, sex, state).

        Args:
            n_members: Number of members to generate
            state_populations: DataFrame with state proportions

        Returns:
            DataFrame with demographic data
        """
        # Generate age groups
        age_groups_config = self.distributions['age_groups']
        age_group_names = list(age_groups_config.keys())
        age_group_probs = [ag['proportion'] for ag in age_groups_config.values()]

        age_groups = self.rng.choice(age_group_names, size=n_members, p=age_group_probs)

        # Generate specific ages within groups (fully vectorized)
        # Create vectorized age sampling by unique age groups
        ages = np.zeros(n_members, dtype=int)

        for ag_name in age_group_names:
            mask = age_groups == ag_name
            n_in_group = mask.sum()
            if n_in_group > 0:
                min_age = age_groups_config[ag_name]['min_age']
                max_age = age_groups_config[ag_name]['max_age']
                ages[mask] = self.rng.integers(min_age, max_age + 1, size=n_in_group)

        # Vectorized DOB calculation using numpy datetime
        days_old = ages * 365 + self.rng.integers(0, 365, size=n_members)
        reference_timestamp = np.datetime64(self.reference_date, 'D')
        dob_timestamps = reference_timestamp - days_old.astype('timedelta64[D]')
        dobs = pd.to_datetime(dob_timestamps).strftime('%Y-%m-%d').tolist()

        # Generate sex
        sex_dist = self.distributions['sex_distribution']
        sexes = self.rng.choice(
            list(sex_dist.keys()),
            size=n_members,
            p=list(sex_dist.values())
        )

        # Generate states (normalize proportions to ensure they sum to 1.0)
        proportions = state_populations['proportion'].values
        proportions = proportions / proportions.sum()
        states = self.rng.choice(
            state_populations['state_code'].values,
            size=n_members,
            p=proportions
        )

        return pd.DataFrame({
            'member_id': range(1, n_members + 1),
            'DOB': dobs,
            'Sex': sexes,
            'state': states,
            '_age': ages,  # Keep for later use
            '_age_group': age_groups  # Keep for later use
        })

    def generate_biometrics(self, members_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate biometric data (height, weight, vitals).

        Args:
            members_df: DataFrame with demographic data

        Returns:
            DataFrame with added biometric columns
        """
        n = len(members_df)
        anthro_config = self.distributions['anthropometrics']
        vitals_config = self.distributions['vitals']

        # Generate height and weight (correlated) by sex
        heights = np.zeros(n)
        weights = np.zeros(n)

        for sex in ['M', 'F', 'O']:
            mask = members_df['Sex'] == sex
            n_sex = mask.sum()

            if n_sex == 0:
                continue

            sex_key = {'M': 'male', 'F': 'female', 'O': 'other'}[sex]
            params = anthro_config[sex_key]

            # Generate correlated height/weight using multivariate normal
            mean = [params['height_mean_cm'], params['weight_mean_kg']]
            cov = [
                [params['height_std_cm'] ** 2,
                 params['correlation'] * params['height_std_cm'] * params['weight_std_kg']],
                [params['correlation'] * params['height_std_cm'] * params['weight_std_kg'],
                 params['weight_std_kg'] ** 2]
            ]

            samples = self.rng.multivariate_normal(mean, cov, size=n_sex)
            heights[mask] = np.clip(samples[:, 0], 140, 210)  # Reasonable bounds
            weights[mask] = np.clip(samples[:, 1], 40, 200)

        members_df['height_cm'] = np.round(heights, 1)
        members_df['weight_kg'] = np.round(weights, 1)

        # Generate heart rate (age-adjusted)
        base_hr = self.rng.normal(
            vitals_config['heart_rate']['base_mean'],
            vitals_config['heart_rate']['base_std'],
            size=n
        )

        # Age adjustment
        age_adjustments = members_df['_age_group'].map({
            '0-17': 10,
            '18-24': 0, '25-34': 0, '35-44': 0, '45-54': 0, '55-64': 0,
            '65-74': -5, '75+': -5
        })

        heart_rates = base_hr + age_adjustments
        heart_rates = np.clip(heart_rates, 40, 120)
        members_df['heart_rate'] = np.round(heart_rates).astype(int)

        # Generate blood pressure (age-adjusted, correlated systolic/diastolic)
        bp_config = vitals_config['blood_pressure']
        age_decades = members_df['_age'] / 10

        systolic_base = self.rng.normal(bp_config['systolic_mean'], bp_config['systolic_std'], size=n)
        systolic = systolic_base + age_decades * bp_config['age_adjustment_per_decade']
        systolic = np.clip(systolic, bp_config['min_systolic'], bp_config['max_systolic'])

        # Diastolic correlated with systolic
        diastolic_base = self.rng.normal(bp_config['diastolic_mean'], bp_config['diastolic_std'], size=n)
        diastolic_adjustment = bp_config['correlation'] * (systolic - bp_config['systolic_mean']) / 2
        diastolic = diastolic_base + diastolic_adjustment
        diastolic = np.clip(diastolic, bp_config['min_diastolic'], bp_config['max_diastolic'])

        # Format as "XXX/YYY" (vectorized)
        systolic_str = systolic.astype(int).astype(str)
        diastolic_str = diastolic.astype(int).astype(str)
        members_df['blood_pressure'] = np.char.add(np.char.add(systolic_str, '/'), diastolic_str)

        # Generate blood oxygen
        bo_config = vitals_config['blood_oxygen']
        blood_oxygen = self.rng.normal(bo_config['mean'], bo_config['std'], size=n)
        blood_oxygen = np.clip(blood_oxygen, bo_config['min'], bo_config['max'])
        members_df['blood_oxygen'] = np.round(blood_oxygen).astype(int)

        return members_df

    def generate_lifestyle(self, members_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lifestyle factors (smoking, drinking, exercise, sleep).

        Args:
            members_df: DataFrame with demographic data

        Returns:
            DataFrame with added lifestyle columns
        """
        lifestyle_config = self.distributions['lifestyle']
        sdoh_config = self.distributions['social_determinants']
        corr_config = self.distributions.get('lifestyle_correlation')

        if corr_config:
            return self._generate_correlated_lifestyle(
                members_df.copy(),
                lifestyle_config,
                sdoh_config,
                corr_config
            )

        # Fallback to independent sampling (legacy behavior)
        return self._generate_independent_lifestyle(
            members_df.copy(),
            lifestyle_config,
            sdoh_config
        )

    def _generate_independent_lifestyle(self, members_df, lifestyle_config, sdoh_config):
        """Legacy independent sampling for lifestyle factors."""
        n = len(members_df)

        smoking_probs = members_df['_age_group'].map(lifestyle_config['smoking']['age_rates'])
        members_df['smoker'] = (self.rng.random(n) < smoking_probs).astype(int)

        alcohol_probs = members_df['_age_group'].map(lifestyle_config['alcohol']['age_rates'])
        members_df['drinker'] = (self.rng.random(n) < alcohol_probs).astype(int)

        exercise_config = lifestyle_config['exercise']
        mu = np.log(exercise_config['median_minutes_per_week'])
        exercise_minutes = self.rng.lognormal(mu, exercise_config['std_log'], size=n)
        exercise_minutes = np.clip(exercise_minutes, exercise_config['min'], exercise_config['max'])
        members_df['exercise_minutes_per_week'] = np.round(exercise_minutes).astype(int)

        sleep_config = lifestyle_config['sleep']
        sleep_hours = self.rng.normal(sleep_config['mean_hours'], sleep_config['std_hours'], size=n)
        sleep_hours = np.clip(sleep_hours, sleep_config['min'], sleep_config['max'])
        members_df['sleep_hours_per_night'] = np.round(sleep_hours, 1)

        housing_insecurity_prob = sdoh_config['housing_insecurity_rate']
        members_df['housing insecurity'] = (self.rng.random(n) < housing_insecurity_prob).astype(int)

        employment_probs = members_df['_age_group'].map(sdoh_config['employment_rates'])
        members_df['employment status'] = (self.rng.random(n) < employment_probs).astype(int)

        return members_df

    def _generate_correlated_lifestyle(self, members_df, lifestyle_config, sdoh_config, corr_config):
        """Generate lifestyle variables using a Gaussian copula to enforce correlations."""
        n = len(members_df)
        variable_order = corr_config.get('variable_order', [])
        matrix = np.array(corr_config.get('correlation_matrix', []), dtype=float)

        if not variable_order or matrix.size == 0 or matrix.shape[0] != len(variable_order):
            # Invalid configuration; fall back gracefully
            return self._generate_independent_lifestyle(members_df, lifestyle_config, sdoh_config)

        # Pre-compute Cholesky (with jitter if needed)
        chol = self._get_cholesky(matrix)
        latent = self.rng.standard_normal(size=(n, len(variable_order)))
        correlated = latent @ chol.T
        idx_lookup = {name: i for i, name in enumerate(variable_order)}
        required_vars = {'smoker', 'drinker', 'exercise', 'sleep', 'housing_insecurity', 'employment'}
        if not required_vars.issubset(idx_lookup.keys()):
            return self._generate_independent_lifestyle(members_df, lifestyle_config, sdoh_config)

        # Helper accessors
        def latent_values(name: str) -> np.ndarray:
            return correlated[:, idx_lookup[name]]

        def bernoulli_from_latent(z: np.ndarray, probs: np.ndarray) -> np.ndarray:
            probs = np.clip(probs, 1e-4, 1 - 1e-4)
            thresholds = stats.norm.ppf(probs)
            return (z <= thresholds).astype(int)

        # Probability schedules
        smoking_probs = members_df['_age_group'].map(
            lifestyle_config['smoking']['age_rates']
        ).fillna(lifestyle_config['smoking']['overall_rate']).values

        drinking_probs = members_df['_age_group'].map(
            lifestyle_config['alcohol']['age_rates']
        ).fillna(lifestyle_config['alcohol']['heavy_drinker_rate']).values

        employment_probs = members_df['_age_group'].map(
            sdoh_config['employment_rates']
        ).fillna(0).values

        housing_prob = sdoh_config.get('housing_insecurity_rate', 0.05)
        housing_probs = np.full(n, housing_prob)

        # Binary variables
        members_df['smoker'] = bernoulli_from_latent(latent_values('smoker'), smoking_probs)
        members_df['drinker'] = bernoulli_from_latent(latent_values('drinker'), drinking_probs)
        members_df['housing insecurity'] = bernoulli_from_latent(
            latent_values('housing_insecurity'),
            housing_probs
        )
        members_df['employment status'] = bernoulli_from_latent(
            latent_values('employment'),
            employment_probs
        )

        # Continuous/positive variables
        exercise_latent = latent_values('exercise')
        exercise_u = stats.norm.cdf(exercise_latent)
        exercise_config = lifestyle_config['exercise']
        mu = np.log(exercise_config['median_minutes_per_week'])
        lognorm = stats.lognorm(s=exercise_config['std_log'], scale=np.exp(mu))
        exercise_minutes = lognorm.ppf(np.clip(exercise_u, 1e-6, 1 - 1e-6))
        exercise_minutes = np.clip(exercise_minutes, exercise_config['min'], exercise_config['max'])
        members_df['exercise_minutes_per_week'] = np.round(exercise_minutes).astype(int)

        sleep_latent = latent_values('sleep')
        sleep_config = lifestyle_config['sleep']
        sleep_hours = sleep_config['mean_hours'] + sleep_config['std_hours'] * sleep_latent
        sleep_hours = np.clip(sleep_hours, sleep_config['min'], sleep_config['max'])
        members_df['sleep_hours_per_night'] = np.round(sleep_hours, 1)

        return members_df

    def _get_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        """Compute a stable Cholesky factor, adding jitter if necessary."""
        jitter = 0.0
        for _ in range(6):
            try:
                return np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * jitter)
            except np.linalg.LinAlgError:
                jitter = 1e-6 if jitter == 0.0 else jitter * 10
        raise ValueError("Lifestyle correlation matrix is not positive definite even after jitter adjustment.")

    def assign_conditions(
        self,
        members_df: pd.DataFrame,
        conditions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign health conditions to members (optimized vectorized version).

        Args:
            members_df: DataFrame with member data
            conditions_df: DataFrame with condition definitions

        Returns:
            Tuple of (updated members_df, member_condition_df)
        """
        condition_config = self.distributions['condition_prevalence']
        mean_conditions = condition_config['mean_conditions_by_age']
        condition_prevalence = condition_config['conditions']
        tracked_conditions = list(condition_prevalence.keys())

        n_members = len(members_df)

        # TIER 1 FEATURE 2: Calculate BMI and risk factors for all members (VECTORIZED)
        # This enables risk-adjusted prevalence rates
        bmi = (members_df['weight_kg'] / ((members_df['height_cm'] / 100) ** 2)).fillna(25.0)
        is_obese = (bmi >= 30).values
        is_smoker = (members_df['smoker'] == 1).fillna(False).values
        is_heavy_drinker = (members_df['drinker'] == 1).fillna(False).values
        is_housing_insecure = (members_df['housing insecurity'] == 1).fillna(False).values

        # Create risk profile tuple for each member (for grouping)
        # Convert to Python bools to make them hashable for dict keys
        risk_profiles = [
            (bool(obese), bool(smoker), bool(drinker), bool(housing))
            for obese, smoker, drinker, housing in zip(is_obese, is_smoker, is_heavy_drinker, is_housing_insecure)
        ]

        # Get risk multipliers configuration
        risk_multipliers_config = self.distributions.get('risk_multipliers', {})

        # PRE-COMPUTATION: Build condition name → ID mapping (done once)
        def _normalize_name(value: str) -> str:
            """Normalize condition names for fuzzy matching."""
            return ''.join(ch.lower() for ch in value if ch.isalnum())

        normalized_lookup = {
            _normalize_name(name): cond_id
            for cond_id, name in zip(conditions_df['Condition_ID'], conditions_df['Condition_name'])
        }

        condition_name_to_id = {}
        missing_condition_names = []
        for cond_name in tracked_conditions:
            normalized_key = _normalize_name(cond_name.replace('_', ' '))
            cond_id = normalized_lookup.get(normalized_key)
            if cond_id is not None:
                condition_name_to_id[cond_name] = cond_id
            else:
                missing_condition_names.append(cond_name)

        if missing_condition_names and self.config.get('verbose_logging', False):
            print(
                "[conditions] Missing Condition_IDs for: "
                + ', '.join(sorted(missing_condition_names))
            )

        # PRE-COMPUTATION: Build risk-adjusted prevalence matrices
        # Key: (age_group, risk_profile_tuple) → (condition_ids, prevalences)
        prevalence_groups = ['0-17', '18-34', '35-44', '45-54', '55-64', '65+']
        prevalence_matrix = {}

        # Helper function to apply risk multipliers to base prevalence
        def apply_risk_multipliers(base_prevalences, cond_names, risk_profile):
            """Apply risk multipliers to base prevalence rates."""
            adjusted = base_prevalences.copy()
            obese, smoker, drinker, housing = risk_profile

            # Apply obesity multipliers
            if obese and 'obesity' in risk_multipliers_config:
                for cond_name, multiplier in risk_multipliers_config['obesity'].items():
                    if cond_name in cond_names:
                        idx = cond_names.index(cond_name)
                        adjusted[idx] *= multiplier

            # Apply smoking multipliers
            if smoker and 'smoking' in risk_multipliers_config:
                for cond_name, multiplier in risk_multipliers_config['smoking'].items():
                    if cond_name in cond_names:
                        idx = cond_names.index(cond_name)
                        adjusted[idx] *= multiplier

            # Apply drinking multipliers
            if drinker and 'heavy_drinking' in risk_multipliers_config:
                for cond_name, multiplier in risk_multipliers_config['heavy_drinking'].items():
                    if cond_name in cond_names:
                        idx = cond_names.index(cond_name)
                        adjusted[idx] *= multiplier

            # Apply housing insecurity multipliers
            if housing and 'housing_insecurity' in risk_multipliers_config:
                for cond_name, multiplier in risk_multipliers_config['housing_insecurity'].items():
                    if cond_name in cond_names:
                        idx = cond_names.index(cond_name)
                        adjusted[idx] *= multiplier

            # Normalize to sum to 1.0
            return adjusted / adjusted.sum()

        # Build base prevalence for each age group and collect unique risk profiles
        base_prevalence_data = {}
        for prev_group in prevalence_groups:
            base_prevalences = []
            cond_ids = []
            cond_names = []
            for cond_name in tracked_conditions:
                if cond_name in condition_name_to_id:
                    base_prevalences.append(condition_prevalence[cond_name].get(prev_group, 0.01))
                    cond_ids.append(condition_name_to_id[cond_name])
                    cond_names.append(cond_name)

            if base_prevalences:
                base_prevalence_data[prev_group] = {
                    'cond_ids': cond_ids,
                    'cond_names': cond_names,
                    'base_prevalences': np.array(base_prevalences)
                }

        # Build prevalence matrices for (age_group, risk_profile) combinations that exist
        # We'll build them on-demand to avoid creating too many unused combinations

        # VECTORIZED: Map age groups to prevalence groups
        age_group_to_prev_group = {
            '0-17': '0-17',
            '18-24': '18-34', '25-34': '18-34',
            '35-44': '35-44',
            '45-54': '45-54',
            '55-64': '55-64',
            '65-74': '65+', '75+': '65+'
        }
        prev_groups = members_df['_age_group'].map(age_group_to_prev_group).values

        # VECTORIZED: Determine number of conditions for all members
        means = members_df['_age_group'].map(mean_conditions).fillna(1.0).values
        n_conditions_arr = self.rng.poisson(means)
        n_conditions_arr = np.maximum(n_conditions_arr, 0)

        # VECTORIZED: Parse DOBs and calculate days_range
        dobs = pd.to_datetime(members_df['DOB'])
        days_since_birth = (self.reference_date - dobs).dt.days
        days_ranges = days_since_birth.values
        member_ids = members_df['member_id'].values
        dob_series = pd.Series(dobs.values, index=member_ids)
        days_since_birth_series = pd.Series(days_ranges, index=member_ids)

        # RISK-ADJUSTED ASSIGNMENT: Sample conditions using risk-adjusted prevalences
        member_conditions = []

        # Process members with each (prev_group, n_conditions) combination
        for prev_group in prevalence_groups:
            if prev_group not in base_prevalence_data:
                continue

            base_data = base_prevalence_data[prev_group]
            cond_ids = base_data['cond_ids']
            cond_names = base_data['cond_names']
            base_prevalences = base_data['base_prevalences']
            max_conds = len(cond_ids)

            # Get members in this prevalence group
            group_mask = prev_groups == prev_group

            # Process different n_conditions values within this group
            for n_cond in range(1, min(max_conds + 1, n_conditions_arr.max() + 1)):
                # Get members with this specific n_conditions in this prev_group
                mask = group_mask & (n_conditions_arr == n_cond)
                n_in_group = mask.sum()

                if n_in_group == 0:
                    continue

                # Sample conditions for all members in this group
                actual_n = min(n_cond, max_conds)
                for member_idx in np.where(mask)[0]:
                    # Get this member's risk profile
                    risk_profile = risk_profiles[member_idx]

                    # Build or retrieve risk-adjusted prevalence for this profile
                    matrix_key = (prev_group, risk_profile)
                    if matrix_key not in prevalence_matrix:
                        # Build risk-adjusted prevalence on-demand
                        adjusted_prevalences = apply_risk_multipliers(
                            base_prevalences, cond_names, risk_profile
                        )
                        prevalence_matrix[matrix_key] = adjusted_prevalences

                    # Get risk-adjusted prevalences for this member
                    prevalences = prevalence_matrix[matrix_key]

                    # Sample conditions using risk-adjusted prevalences
                    selected_conds = self.rng.choice(
                        cond_ids,
                        size=actual_n,
                        p=prevalences,
                        replace=False
                    )

                    # Generate diagnostic dates for this member's conditions
                    days_range = days_ranges[member_idx]
                    dob_timestamp = np.datetime64(dobs.iloc[member_idx], 'D')

                    for cond_id in selected_conds:
                        days_offset = self.rng.integers(365, max(366, days_range))
                        diagnostic_date = dob_timestamp + np.timedelta64(int(days_offset), 'D')

                        member_conditions.append({
                            'Member_ID': member_ids[member_idx],
                            'Condition_ID': int(cond_id),
                            'Diagnostic_date': str(pd.to_datetime(diagnostic_date).date())
                        })

        member_condition_df = pd.DataFrame(member_conditions) if member_conditions else pd.DataFrame(
            columns=['Member_ID', 'Condition_ID', 'Diagnostic_date']
        )

        # Enforce targeted risk-factor relationships (e.g., obesity → diabetes)
        member_condition_df = self._enforce_risk_factor_targets(
            members_df,
            member_condition_df,
            condition_name_to_id,
            condition_prevalence,
            dob_series,
            days_since_birth_series
        )

        # TIER 1 FEATURE 1: Apply comorbidity rules to add related conditions
        member_condition_df = self._apply_comorbidity_rules(
            member_condition_df, members_df, condition_name_to_id
        )

        # TIER 1 FEATURE 3: Adjust vitals based on final condition set
        members_df = self._adjust_vitals_for_conditions(
            members_df, member_condition_df, condition_name_to_id
        )

        return members_df, member_condition_df

    def _enforce_risk_factor_targets(
        self,
        members_df: pd.DataFrame,
        member_condition_df: pd.DataFrame,
        condition_name_to_id: dict,
        condition_prevalence: dict,
        member_dob_map: pd.Series,
        member_days_since_birth: pd.Series
    ) -> pd.DataFrame:
        """
        Ensure key risk-factor multipliers hit their intended targets (e.g., obesity → diabetes).

        Args:
            members_df: Members with demographics/biometrics
            member_condition_df: Current condition assignments
            condition_name_to_id: Mapping of condition name → ID
            condition_prevalence: Base prevalence configuration
            member_dob_map: Series indexed by member_id with DOB timestamps
            member_days_since_birth: Series indexed by member_id with age in days

        Returns:
            Updated member_condition_df with any injected conditions
        """
        risk_multipliers = self.distributions.get('risk_multipliers', {})
        obesity_targets = risk_multipliers.get('obesity', {})
        diabetes_multiplier = obesity_targets.get('Diabetes_Type_2')
        diabetes_id = condition_name_to_id.get('Diabetes_Type_2')

        if not (diabetes_id and diabetes_multiplier):
            return member_condition_df

        bmi = members_df['weight_kg'] / ((members_df['height_cm'] / 100) ** 2)
        bmi = bmi.replace([np.inf, -np.inf], np.nan)
        obese_mask = bmi >= 30
        obese_mask = obese_mask.fillna(False)

        if not obese_mask.any():
            return member_condition_df

        age_group_to_prev_group = {
            '0-17': '0-17',
            '18-24': '18-34', '25-34': '18-34',
            '35-44': '35-44',
            '45-54': '45-54',
            '55-64': '55-64',
            '65-74': '65+', '75+': '65+'
        }
        prev_group_series = members_df['_age_group'].map(age_group_to_prev_group)

        member_df = pd.DataFrame({
            'member_id': members_df['member_id'].astype(int),
            'prev_group': prev_group_series,
            'is_obese': obese_mask.values
        }).dropna(subset=['prev_group'])

        if member_df.empty:
            return member_condition_df

        diabetes_members = set()
        if not member_condition_df.empty:
            diabetes_members = set(
                member_condition_df[
                    member_condition_df['Condition_ID'] == diabetes_id
                ]['Member_ID'].astype(int).tolist()
            )

        new_conditions = []
        members_to_remove: List[int] = []

        for prev_group, group_df in member_df.groupby('prev_group'):
            obese_ids = group_df[group_df['is_obese']]['member_id'].astype(int).tolist()
            non_obese_ids = group_df[~group_df['is_obese']]['member_id'].astype(int).tolist()

            if len(obese_ids) == 0 or len(non_obese_ids) == 0:
                continue

            current_obese = [mid for mid in obese_ids if mid in diabetes_members]
            current_non_obese = [mid for mid in non_obese_ids if mid in diabetes_members]

            if not current_obese and not current_non_obese:
                continue

            max_non_allowed = int(len(non_obese_ids) / diabetes_multiplier)
            if max_non_allowed < len(current_non_obese):
                removal_count = len(current_non_obese) - max_non_allowed
                removal_count = min(removal_count, len(current_non_obese))
                if removal_count > 0:
                    selected = self.rng.choice(current_non_obese, size=removal_count, replace=False)
                    for member_id in np.atleast_1d(selected):
                        members_to_remove.append(int(member_id))
                        diabetes_members.discard(int(member_id))
                    current_non_obese = [mid for mid in current_non_obese if mid not in set(selected)]

            if len(non_obese_ids) == 0:
                continue

            required_obese = int(np.ceil(
                len(current_non_obese) * diabetes_multiplier * len(obese_ids) / len(non_obese_ids)
            ))
            required_obese = min(required_obese, len(obese_ids))

            obese_shortfall = required_obese - len(current_obese)
            if obese_shortfall > 0:
                candidates = list(set(obese_ids) - set(current_obese))
                if candidates:
                    sample_size = min(obese_shortfall, len(candidates))
                    selected = self.rng.choice(candidates, size=sample_size, replace=False)
                    for member_id in np.atleast_1d(selected):
                        diagnostic_date = self._sample_random_condition_date(
                            int(member_id), member_dob_map, member_days_since_birth
                        )
                        new_conditions.append({
                            'Member_ID': int(member_id),
                            'Condition_ID': int(diabetes_id),
                            'Diagnostic_date': diagnostic_date
                        })
                        diabetes_members.add(int(member_id))

        if members_to_remove and not member_condition_df.empty:
            remove_set = set(members_to_remove)
            mask = ~(
                (member_condition_df['Condition_ID'] == diabetes_id) &
                (member_condition_df['Member_ID'].isin(remove_set))
            )
            member_condition_df = member_condition_df[mask].reset_index(drop=True)

        if new_conditions:
            new_df = pd.DataFrame(new_conditions)[['Member_ID', 'Condition_ID', 'Diagnostic_date']]
            if member_condition_df.empty:
                member_condition_df = new_df
            else:
                member_condition_df = pd.concat([member_condition_df, new_df], ignore_index=True)

        return member_condition_df

    def _sample_random_condition_date(
        self,
        member_id: int,
        member_dob_map: pd.Series,
        member_days_since_birth: pd.Series,
        min_age_days: int = 365
    ) -> str:
        """Sample a diagnostic date after the member's first birthday."""
        dob = member_dob_map.get(member_id)
        if pd.isna(dob):
            dob = pd.Timestamp(self.reference_date) - pd.Timedelta(days=365 * 30)

        available_days = int(member_days_since_birth.get(member_id, min_age_days + 1))
        upper_bound = max(min_age_days + 1, available_days)
        lower_bound = min(min_age_days, upper_bound - 1)
        days_offset = int(self.rng.integers(lower_bound, upper_bound))

        diagnostic_date = dob + pd.to_timedelta(days_offset, unit='D')
        if diagnostic_date > self.reference_date:
            diagnostic_date = self.reference_date

        return str(diagnostic_date.date())

    def _apply_comorbidity_rules(
        self,
        member_condition_df: pd.DataFrame,
        members_df: pd.DataFrame,
        condition_name_to_id: dict
    ) -> pd.DataFrame:
        """
        TIER 1 FEATURE 1: Apply comorbidity patterns to add related conditions (VECTORIZED).

        Args:
            member_condition_df: Initial conditions assigned
            members_df: Member demographics (for age filtering)
            condition_name_to_id: Condition name → ID mapping

        Returns:
            Updated member_condition_df with comorbid conditions added
        """
        comorbidity_rules = self.distributions.get('comorbidity_rules', [])

        if not comorbidity_rules or member_condition_df.empty:
            return member_condition_df

        # Create set of (Member_ID, Condition_ID) for fast lookups
        existing_conditions = set(
            zip(member_condition_df['Member_ID'].values, member_condition_df['Condition_ID'].values)
        )

        new_conditions = []

        # Calculate ages for age-based rules
        dobs = pd.to_datetime(members_df['DOB'])
        ages = ((self.reference_date - dobs).dt.days / 365).values
        member_id_to_age = dict(zip(members_df['member_id'].values, ages))

        # Apply each comorbidity rule
        for rule in comorbidity_rules:
            trigger = rule['trigger']
            adds = rule['adds']
            probability = rule['probability']
            age_min = rule.get('age_min', 0)
            delay_days_min = rule.get('delay_days_min', 0)
            delay_days_max = rule.get('delay_days_max', 365)

            # Get condition IDs
            if isinstance(trigger, list):
                # Multi-condition trigger (e.g., Hypertension + Hyperlipidemia)
                trigger_ids = [condition_name_to_id.get(t) for t in trigger if condition_name_to_id.get(t)]
                if len(trigger_ids) != len(trigger):
                    continue  # Some trigger conditions not in database

                # Find members with ALL trigger conditions
                members_with_trigger = member_condition_df[
                    member_condition_df['Condition_ID'].isin(trigger_ids)
                ]['Member_ID'].value_counts()
                members_with_all_triggers = members_with_trigger[members_with_trigger == len(trigger_ids)].index.values
            else:
                # Single condition trigger
                trigger_id = condition_name_to_id.get(trigger)
                if not trigger_id:
                    continue

                members_with_trigger = member_condition_df[
                    member_condition_df['Condition_ID'] == trigger_id
                ]['Member_ID'].unique()
                members_with_all_triggers = members_with_trigger

            # Get target condition ID
            target_id = condition_name_to_id.get(adds)
            if not target_id:
                continue

            # Filter by age if specified
            if age_min > 0:
                age_filtered = [m for m in members_with_all_triggers if member_id_to_age.get(m, 0) >= age_min]
                members_with_all_triggers = np.array(age_filtered)

            if len(members_with_all_triggers) == 0:
                continue

            members_to_add: List[int] = []
            target_rate = rule.get('target_rate')

            if target_rate is not None:
                members_with_target = member_condition_df[
                    (member_condition_df['Member_ID'].isin(members_with_all_triggers)) &
                    (member_condition_df['Condition_ID'] == target_id)
                ]['Member_ID'].unique()

                desired_total = int(round(target_rate * len(members_with_all_triggers)))
                desired_total = min(desired_total, len(members_with_all_triggers))
                n_to_add = max(0, desired_total - len(members_with_target))

                if n_to_add > 0:
                    eligible_members = np.setdiff1d(members_with_all_triggers, members_with_target, assume_unique=False)
                    if len(eligible_members) > 0:
                        sample_size = min(n_to_add, len(eligible_members))
                        selected = self.rng.choice(eligible_members, size=sample_size, replace=False)
                        members_to_add = selected.tolist()
            else:
                n_candidates = len(members_with_all_triggers)
                if n_candidates == 0:
                    continue

                add_condition_mask = self.rng.random(n_candidates) < probability

                # Filter out members who already have the target condition
                members_to_add = [
                    m for m, should_add in zip(members_with_all_triggers, add_condition_mask)
                    if should_add and (m, target_id) not in existing_conditions
                ]

            if not members_to_add:
                continue

            # Generate diagnostic dates (VECTORIZED)
            # Date should be AFTER the trigger condition's diagnosis
            for member_id in members_to_add:
                # Get the latest diagnostic date among trigger conditions
                if isinstance(trigger, list):
                    trigger_dates = member_condition_df[
                        (member_condition_df['Member_ID'] == member_id) &
                        (member_condition_df['Condition_ID'].isin(trigger_ids))
                    ]['Diagnostic_date']
                else:
                    trigger_dates = member_condition_df[
                        (member_condition_df['Member_ID'] == member_id) &
                        (member_condition_df['Condition_ID'] == trigger_id)
                    ]['Diagnostic_date']

                if trigger_dates.empty:
                    continue

                # Use latest trigger date as baseline
                latest_trigger_date = pd.to_datetime(trigger_dates.max())

                # Add delay
                days_offset = int(self.rng.integers(delay_days_min, delay_days_max + 1))
                new_diagnostic_date = latest_trigger_date + pd.Timedelta(days=days_offset)

                # Don't exceed reference date
                if new_diagnostic_date > pd.to_datetime(self.reference_date):
                    new_diagnostic_date = pd.to_datetime(self.reference_date)

                new_conditions.append({
                    'Member_ID': member_id,
                    'Condition_ID': int(target_id),
                    'Diagnostic_date': str(new_diagnostic_date.date())
                })

                # Add to existing set to prevent duplicates in subsequent rules
                existing_conditions.add((member_id, target_id))

        # Combine original + new conditions
        if new_conditions:
            new_df = pd.DataFrame(new_conditions)
            member_condition_df = pd.concat([member_condition_df, new_df], ignore_index=True)

        return member_condition_df

    def _adjust_vitals_for_conditions(
        self,
        members_df: pd.DataFrame,
        member_condition_df: pd.DataFrame,
        condition_name_to_id: dict
    ) -> pd.DataFrame:
        """
        TIER 1 FEATURE 3: Adjust vital signs based on member's conditions (VECTORIZED).

        Args:
            members_df: Members with baseline vitals
            member_condition_df: Assigned conditions
            condition_name_to_id: Condition name → ID mapping

        Returns:
            members_df with adjusted vitals
        """
        adjustments_config = self.distributions.get('condition_vital_adjustments', {})

        if not adjustments_config or member_condition_df.empty:
            return members_df

        member_ids = members_df['member_id'].values

        # Create boolean masks for each condition (VECTORIZED)
        condition_masks = {}
        for cond_name, cond_id in condition_name_to_id.items():
            members_with_condition = member_condition_df[
                member_condition_df['Condition_ID'] == cond_id
            ]['Member_ID'].unique()
            condition_masks[cond_name] = np.isin(member_ids, members_with_condition)

        # BLOOD PRESSURE ADJUSTMENTS
        if 'blood_pressure' in adjustments_config and 'blood_pressure' in members_df.columns:
            bp_config = adjustments_config['blood_pressure']

            # Parse blood pressure into systolic/diastolic
            bp_parts = members_df['blood_pressure'].fillna('120/80').str.split('/', expand=True)
            systolic = bp_parts[0].astype(float)
            diastolic = bp_parts[1].astype(float)

            # Apply adjustments for each condition
            for cond_name, adjustment in bp_config.items():
                if cond_name not in condition_masks:
                    continue

                mask = condition_masks[cond_name]
                n_affected = mask.sum()

                if n_affected == 0:
                    continue

                # Systolic adjustment
                if 'systolic_mean_increase' in adjustment:
                    mean_increase = adjustment['systolic_mean_increase']
                    std = adjustment.get('systolic_std', mean_increase * 0.5)
                    systolic[mask] += self.rng.normal(mean_increase, std, size=n_affected)

                # Diastolic adjustment
                if 'diastolic_mean_increase' in adjustment:
                    mean_increase = adjustment['diastolic_mean_increase']
                    std = adjustment.get('diastolic_std', mean_increase * 0.5)
                    diastolic[mask] += self.rng.normal(mean_increase, std, size=n_affected)

            # Clip and reformat
            systolic = np.clip(systolic, 90, 250).astype(int)
            diastolic = np.clip(diastolic, 60, 150).astype(int)
            members_df['blood_pressure'] = systolic.astype(str) + '/' + diastolic.astype(str)

        # HEART RATE ADJUSTMENTS
        if 'heart_rate' in adjustments_config and 'heart_rate' in members_df.columns:
            hr_config = adjustments_config['heart_rate']
            hr = members_df['heart_rate'].copy().astype(float)

            for cond_name, adjustment in hr_config.items():
                if cond_name not in condition_masks:
                    continue

                mask = condition_masks[cond_name]
                n_affected = mask.sum()

                if n_affected == 0:
                    continue

                mean_increase = adjustment.get('mean_increase', 0)
                std = adjustment.get('std', mean_increase * 0.5)
                hr[mask] += self.rng.normal(mean_increase, std, size=n_affected)

            members_df['heart_rate'] = np.clip(hr, 40, 120).astype(int)

        # BLOOD OXYGEN ADJUSTMENTS
        if 'blood_oxygen' in adjustments_config and 'blood_oxygen' in members_df.columns:
            spo2_config = adjustments_config['blood_oxygen']
            spo2 = members_df['blood_oxygen'].copy().astype(float)

            for cond_name, adjustment in spo2_config.items():
                if cond_name not in condition_masks:
                    continue

                mask = condition_masks[cond_name]
                n_affected = mask.sum()

                if n_affected == 0:
                    continue

                mean_decrease = adjustment.get('mean_decrease', 0)
                std = adjustment.get('std', mean_decrease * 0.5)
                spo2[mask] -= self.rng.normal(mean_decrease, std, size=n_affected)

            members_df['blood_oxygen'] = np.clip(spo2, 88, 100).astype(int)

        return members_df

    def apply_null_rates(self, members_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply NULL rates to optional fields.

        Args:
            members_df: DataFrame with complete member data

        Returns:
            DataFrame with NULL values applied
        """
        null_config = self.config.get('null_rates', {})

        optional_fields = {
            'heart_rate': null_config.get('heart_rate', 0.10),
            'blood_pressure': null_config.get('blood_pressure', 0.10),
            'blood_oxygen': null_config.get('blood_oxygen', 0.15),
            'housing insecurity': null_config.get('housing_insecurity', 0.05),
            'employment status': null_config.get('employment_status', 0.05)
        }

        for field, null_rate in optional_fields.items():
            if field in members_df.columns:
                null_mask = self.rng.random(len(members_df)) < null_rate
                members_df.loc[null_mask, field] = None

        return members_df


def _generate_member_chunk(args: Tuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Worker function to generate a chunk of members (for parallel processing).

    Args:
        args: Tuple of (start_id, end_id, config, distributions, state_populations_dict,
                        conditions_df_dict, facilities_ids, random_seed)

    Returns:
        Tuple of (members_df, member_condition_df) for this chunk
    """
    (start_id, end_id, config, distributions, state_populations_dict,
     conditions_df_dict, facility_ids_list, chunk_seed) = args

    # Reconstruct DataFrames from dictionaries
    state_populations = pd.DataFrame(state_populations_dict)
    conditions_df = pd.DataFrame(conditions_df_dict)

    # Create a mini facilities DataFrame for sampling
    facilities_df = pd.DataFrame({'Facility_ID': facility_ids_list})

    # Number of members in this chunk
    n_chunk = end_id - start_id

    # Create generator with chunk-specific seed
    generator = MemberGenerator(config, distributions, config.get('reference_date', '2024-01-01'), chunk_seed)

    # Generate demographics
    members_df = generator.generate_demographics(n_chunk, state_populations)

    # Adjust member IDs to match chunk range
    members_df['member_id'] = range(start_id, end_id)

    # Generate biometrics
    members_df = generator.generate_biometrics(members_df)

    # Generate lifestyle
    members_df = generator.generate_lifestyle(members_df)

    # Assign conditions
    members_df, member_condition_df = generator.assign_conditions(members_df, conditions_df)

    # Assign facilities (70% have PCP)
    pcp_rate = 0.70
    has_pcp = generator.rng.random(n_chunk) < pcp_rate
    facility_ids = generator.rng.choice(facility_ids_list, size=n_chunk)
    members_df['Facility_ID'] = facility_ids
    members_df.loc[~has_pcp, 'Facility_ID'] = None

    # Insurance and Enrollment IDs will be filled by enrollment generator
    members_df['Insurance ID'] = None
    members_df['Enrollment ID'] = None

    # Apply NULL rates
    members_df = generator.apply_null_rates(members_df)

    # Drop temporary columns
    members_df = members_df.drop(columns=['_age', '_age_group'])

    return members_df, member_condition_df


def generate_members_data(
    config: dict,
    distributions: dict,
    state_populations: pd.DataFrame,
    conditions_df: pd.DataFrame,
    facilities_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate all member-related data (supports parallel processing).

    Args:
        config: Configuration dictionary
        distributions: Distribution parameters
        state_populations: DataFrame with state proportions
        conditions_df: DataFrame with conditions
        facilities_df: DataFrame with facilities

    Returns:
        Tuple of (members_df, member_condition_df)
    """
    n_members = config.get('n_members', 100000)
    reference_date = config.get('reference_date', '2024-01-01')
    random_seed = config.get('random_seed')
    enable_parallel = config.get('enable_parallel', False)
    chunk_size = config.get('chunk_size', 50000)

    # Check if parallel processing is enabled and beneficial
    if enable_parallel and n_members >= chunk_size:
        # Parallel mode
        n_workers = config.get('n_workers')
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        # Calculate chunk boundaries
        n_chunks = (n_members + chunk_size - 1) // chunk_size
        chunks = []
        for i in range(n_chunks):
            start_id = i * chunk_size + 1
            end_id = min((i + 1) * chunk_size + 1, n_members + 1)
            # Generate chunk-specific seed for reproducibility
            chunk_seed = None if random_seed is None else random_seed + i
            chunks.append((
                start_id, end_id, config, distributions,
                state_populations.to_dict('list'),
                conditions_df.to_dict('list'),
                facilities_df['Facility_ID'].tolist(),
                chunk_seed
            ))

        # Process chunks in parallel
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(_generate_member_chunk, chunks)

        # Combine results
        members_dfs = [r[0] for r in results]
        member_condition_dfs = [r[1] for r in results]

        members_df = pd.concat(members_dfs, ignore_index=True)
        member_condition_df = pd.concat(member_condition_dfs, ignore_index=True) if any(
            len(df) > 0 for df in member_condition_dfs
        ) else pd.DataFrame(columns=['Member_ID', 'Condition_ID', 'Diagnostic_date'])

        return members_df, member_condition_df

    else:
        # Serial mode (original implementation)
        generator = MemberGenerator(config, distributions, reference_date, random_seed)

        # Generate demographics
        members_df = generator.generate_demographics(n_members, state_populations)

        # Generate biometrics
        members_df = generator.generate_biometrics(members_df)

        # Generate lifestyle
        members_df = generator.generate_lifestyle(members_df)

        # Assign conditions
        members_df, member_condition_df = generator.assign_conditions(members_df, conditions_df)

        # Assign facilities (70% have PCP)
        pcp_rate = 0.70
        has_pcp = generator.rng.random(n_members) < pcp_rate
        facility_ids = generator.rng.choice(facilities_df['Facility_ID'].values, size=n_members)
        members_df['Facility_ID'] = facility_ids
        members_df.loc[~has_pcp, 'Facility_ID'] = None

        # Insurance and Enrollment IDs will be filled by enrollment generator
        members_df['Insurance ID'] = None
        members_df['Enrollment ID'] = None

        # Apply NULL rates
        members_df = generator.apply_null_rates(members_df)

        # Drop temporary columns
        members_df = members_df.drop(columns=['_age', '_age_group'])

        return members_df, member_condition_df
