"""
Generate enrollment and insurance assignment data.
Assigns members to insurance plans and calculates premiums.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple


class EnrollmentGenerator:
    """Generates insurance enrollment data with ACA-compliant premiums."""

    def __init__(self, config: dict, distributions: dict, reference_date: str, random_seed: int = None):
        """
        Initialize the enrollment generator.

        Args:
            config: Configuration dictionary
            distributions: Distribution parameters
            reference_date: Reference date for enrollment
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.distributions = distributions
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        self.rng = np.random.default_rng(random_seed)

        # State cost factors (simplified - some states are more expensive)
        self.state_cost_factors = self._generate_state_cost_factors()

    def _generate_state_cost_factors(self) -> dict:
        """Generate random state cost adjustment factors (0.8 to 1.3)."""
        # In reality, these would be based on actual market data
        # For MVP, we'll use random factors with some high-cost states
        high_cost_states = ['AK', 'HI', 'NY', 'CA', 'MA', 'CT']
        low_cost_states = ['AR', 'MS', 'AL', 'TN', 'KY', 'WV']

        factors = {}
        all_states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]

        for state in all_states:
            if state in high_cost_states:
                factors[state] = self.rng.uniform(1.15, 1.30)
            elif state in low_cost_states:
                factors[state] = self.rng.uniform(0.80, 0.95)
            else:
                factors[state] = self.rng.uniform(0.95, 1.10)

        return factors

    def _calculate_age_from_dob(self, dob_str: str) -> int:
        """Calculate age from date of birth string."""
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        age = (self.reference_date - dob).days // 365
        return age

    def _get_age_group_for_plan(self, age: int) -> str:
        """Map age to age group for plan selection."""
        if age < 18:
            return '0-17'
        elif age < 35:
            return '18-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        elif age < 65:
            return '55-64'
        else:
            return '65+'

    def _select_plan_for_member(self, age: int, plans_by_insurance: dict) -> int:
        """
        Select an appropriate plan for a member based on age preferences.

        Args:
            age: Member's age
            plans_by_insurance: Dictionary mapping insurance_id to list of plans

        Returns:
            Selected plan_id
        """
        age_group = self._get_age_group_for_plan(age)
        insurance_config = self.distributions['insurance']

        # Get plan preferences for this age group
        # Preferences map to: [Catastrophic, Bronze, Silver, Gold, Platinum]
        plan_preferences = insurance_config['plan_preferences'][age_group]

        # Randomly select an insurance provider
        insurance_id = self.rng.choice(list(plans_by_insurance.keys()))
        plans = plans_by_insurance[insurance_id]

        # Select plan based on preferences
        # Assuming plans are ordered: Catastrophic, Bronze, Silver, Gold, Platinum
        if len(plans) >= len(plan_preferences):
            plan_idx = self.rng.choice(range(len(plan_preferences)), p=plan_preferences)
            return plans[plan_idx]['plan_id']
        else:
            # Fallback: random plan
            return self.rng.choice([p['plan_id'] for p in plans])

    def _calculate_premium(
        self,
        base_rate: float,
        age: int,
        state: str
    ) -> float:
        """
        Calculate ACA-compliant premium.

        Args:
            base_rate: Base premium rate from plan
            age: Member's age
            state: State code

        Returns:
            Calculated premium
        """
        aca_config = self.distributions['insurance']['aca_age_rating']
        reference_age = aca_config['reference_age']
        max_ratio = aca_config['max_ratio']

        # ACA age rating: 3:1 ratio max
        if age < 21:
            age_factor = 1.0
        elif age > 64:
            age_factor = max_ratio
        else:
            # Linear interpolation between 1.0 and 3.0
            age_factor = 1.0 + (age - reference_age) * (max_ratio - 1.0) / (64 - reference_age)
            age_factor = np.clip(age_factor, 1.0, max_ratio)

        # State cost adjustment
        state_factor = self.state_cost_factors.get(state, 1.0)

        # Calculate premium
        premium = base_rate * age_factor * state_factor

        return round(premium, 2)

    def generate_enrollments(
        self,
        members_df: pd.DataFrame,
        plans_df: pd.DataFrame,
        insurance_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate enrollment data and update members with insurance info (vectorized).

        Args:
            members_df: DataFrame with member data
            plans_df: DataFrame with insurance plans
            insurance_df: DataFrame with insurance providers

        Returns:
            Tuple of (updated members_df, enrollments_df)
        """
        insurance_config = self.distributions['insurance']
        coverage_rate = insurance_config['coverage_rate']
        active_rate = insurance_config['active_enrollment_rate']
        lookback_years = self.config.get('enrollment_lookback_years', 2)

        n_members = len(members_df)

        # Vectorized: Determine which members have insurance
        has_insurance = self.rng.random(n_members) < coverage_rate
        covered_members = members_df[has_insurance].copy()
        n_covered = len(covered_members)

        if n_covered == 0:
            return members_df, pd.DataFrame(columns=[
                'Enrollment_ID', 'Coverage_tier', 'start_date', 'end_date', 'state', 'plan_id', 'premium'
            ])

        # Vectorized: Calculate ages for all covered members
        dobs = pd.to_datetime(covered_members['DOB'])
        ages = ((self.reference_date - dobs).dt.days // 365).values

        # Vectorized: Map ages to age groups
        age_groups = np.select(
            [ages < 18, ages < 35, ages < 45, ages < 55, ages < 65],
            ['0-17', '18-34', '35-44', '45-54', '55-64'],
            default='65+'
        )

        # Vectorized: Select plans based on age preferences
        plan_preferences = insurance_config['plan_preferences']
        insurance_ids = insurance_df['insurance_id'].values
        n_insurers = len(insurance_ids)

        # For each member, randomly select an insurer
        selected_insurers = self.rng.choice(insurance_ids, size=n_covered)

        # For each member, select a plan based on age group preferences
        selected_plan_ids = np.zeros(n_covered, dtype=int)
        for i, (age_group, insurer_id) in enumerate(zip(age_groups, selected_insurers)):
            # Get plans for this insurer
            insurer_plans = plans_df[plans_df['insurance_id'] == insurer_id]
            if len(insurer_plans) == 0:
                # Fallback: select any plan
                selected_plan_ids[i] = self.rng.choice(plans_df['plan_id'].values)
            else:
                # Get plan preferences for this age group
                prefs = plan_preferences.get(age_group, [0.1, 0.2, 0.3, 0.25, 0.15])
                n_plans = min(len(insurer_plans), len(prefs))
                if n_plans < len(prefs):
                    # Normalize preferences if fewer plans available
                    prefs = prefs[:n_plans]
                    prefs = np.array(prefs) / np.sum(prefs)
                plan_idx = self.rng.choice(n_plans, p=prefs[:n_plans])
                selected_plan_ids[i] = insurer_plans.iloc[plan_idx]['plan_id']

        # Get plan details for selected plans
        plan_lookup = plans_df.set_index('plan_id')
        selected_plans = plan_lookup.loc[selected_plan_ids]
        base_rates = selected_plans['base_rate'].values
        insurance_ids_from_plans = selected_plans['insurance_id'].values

        # Vectorized: Calculate premiums
        # ACA age rating
        aca_config = insurance_config['aca_age_rating']
        reference_age = aca_config['reference_age']
        max_ratio = aca_config['max_ratio']

        age_factors = np.where(
            ages < 21, 1.0,
            np.where(
                ages > 64, max_ratio,
                np.clip(1.0 + (ages - reference_age) * (max_ratio - 1.0) / (64 - reference_age), 1.0, max_ratio)
            )
        )

        # State cost factors
        states = covered_members['state'].values
        state_factors = np.array([self.state_cost_factors.get(s, 1.0) for s in states])

        # Calculate base premiums
        base_premiums = base_rates * age_factors * state_factors

        # Vectorized: Select coverage tiers
        tiers = ['Individual', 'Individual + Spouse', 'Individual + Children', 'Family']
        tier_weights = [0.45, 0.20, 0.15, 0.20]
        coverage_tiers = self.rng.choice(tiers, size=n_covered, p=tier_weights)

        # Vectorized: Apply tier multipliers
        tier_multipliers = {
            'Individual': 1.0,
            'Individual + Spouse': 1.8,
            'Individual + Children': 1.6,
            'Family': 2.2
        }
        tier_mult_values = np.array([tier_multipliers[t] for t in coverage_tiers])
        premiums = np.round(base_premiums * tier_mult_values, 2)

        # Vectorized: Generate enrollment dates
        days_back_arr = self.rng.integers(30, lookback_years * 365, size=n_covered)
        reference_timestamp = np.datetime64(self.reference_date, 'D')
        start_dates = reference_timestamp - days_back_arr.astype('timedelta64[D]')

        # End dates: 70% active (NULL), 30% ended
        is_active = self.rng.random(n_covered) < active_rate
        end_dates = np.empty(n_covered, dtype='datetime64[D]')
        end_dates[is_active] = np.datetime64('NaT')

        # For inactive enrollments, calculate end date
        inactive_mask = ~is_active
        if inactive_mask.any():
            days_duration = self.rng.integers(1, np.maximum(days_back_arr[inactive_mask], 2))
            end_dates[inactive_mask] = start_dates[inactive_mask] + days_duration.astype('timedelta64[D]')

        # Create enrollments DataFrame
        enrollments_df = pd.DataFrame({
            'Enrollment_ID': np.arange(1, n_covered + 1),
            'Coverage_tier': coverage_tiers,
            'start_date': pd.to_datetime(start_dates).strftime('%Y-%m-%d'),
            'end_date': pd.Series(end_dates).apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else None),
            'state': states,
            'plan_id': selected_plan_ids,
            'premium': premiums
        })

        # Update members_df with insurance info
        covered_indices = covered_members.index
        members_df.loc[covered_indices, 'Insurance ID'] = insurance_ids_from_plans
        members_df.loc[covered_indices, 'Enrollment ID'] = enrollments_df['Enrollment_ID'].values

        return members_df, enrollments_df


def generate_enrollment_data(
    config: dict,
    distributions: dict,
    members_df: pd.DataFrame,
    plans_df: pd.DataFrame,
    insurance_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate enrollment data.

    Args:
        config: Configuration dictionary
        distributions: Distribution parameters
        members_df: DataFrame with member data
        plans_df: DataFrame with plans
        insurance_df: DataFrame with insurance providers

    Returns:
        Tuple of (updated members_df, enrollments_df)
    """
    reference_date = config.get('reference_date', '2024-01-01')
    random_seed = config.get('random_seed')

    generator = EnrollmentGenerator(config, distributions, reference_date, random_seed)

    return generator.generate_enrollments(members_df, plans_df, insurance_df)
