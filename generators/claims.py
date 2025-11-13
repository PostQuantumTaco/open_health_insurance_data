"""
Generate claims data for enrolled members.
Claims follow Poisson frequency and log-normal amounts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import multiprocessing

CLAIM_MINIMUMS = {
    'office_visit': 100,
    'specialist': 200,
    'emergency': 500,
    'inpatient': 5000,
    'outpatient_surgery': 2000
}


def _generate_claims_chunk(args: Tuple) -> pd.DataFrame:
    """
    Worker function for parallel claims generation.

    Args:
        args: Tuple containing (chunk_enrolled_dict, config, distributions, chunk_seed, chunk_offset)

    Returns:
        DataFrame with claims for this chunk
    """
    (chunk_enrolled_dict, config, distributions, chunk_seed, chunk_offset) = args

    # Reconstruct DataFrame from dict
    chunk_enrolled = pd.DataFrame(chunk_enrolled_dict)

    if chunk_enrolled.empty:
        return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

    # Extract configuration
    reference_date_str = config.get('reference_date', '2024-01-01')
    reference_date = datetime.strptime(reference_date_str, '%Y-%m-%d')
    lookback_years = config.get('claims_lookback_years', 1)

    # Create RNG with chunk-specific seed
    rng = np.random.default_rng(chunk_seed)

    # Vectorize age calculation
    dobs = pd.to_datetime(chunk_enrolled['DOB'])
    ages = (reference_date - dobs).dt.days // 365

    # Vectorize age group mapping
    age_groups = pd.cut(
        ages,
        bins=[-1, 18, 35, 45, 55, 65, 75, 150],
        labels=['0-17', '18-34', '35-44', '45-54', '55-64', '65-74', '75+']
    )

    # Map age groups to claim frequencies
    claims_config = distributions['claims']
    frequency_map = claims_config['frequency_by_age']
    lambda_per_year = age_groups.map(frequency_map).astype(float)

    # Vectorize coverage period calculation
    start_dates = pd.to_datetime(chunk_enrolled['start_date'])
    end_dates = pd.to_datetime(chunk_enrolled['end_date'].fillna(reference_date.strftime('%Y-%m-%d')))
    earliest_date = reference_date - timedelta(days=lookback_years * 365)

    # Apply lookback limit
    start_dates = start_dates.clip(lower=earliest_date)

    # Calculate coverage years
    coverage_days = (end_dates - start_dates).dt.days
    coverage_years = coverage_days / 365.0

    # Filter out invalid coverage periods
    valid_mask = coverage_days > 0
    chunk_enrolled = chunk_enrolled[valid_mask].copy()
    lambda_per_year = lambda_per_year[valid_mask]
    coverage_years = coverage_years[valid_mask]
    start_dates = start_dates[valid_mask]
    coverage_days = coverage_days[valid_mask]

    if chunk_enrolled.empty:
        return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

    # Vectorize Poisson sampling for n_claims
    lambda_total = lambda_per_year * coverage_years
    n_claims_arr = rng.poisson(lambda_total)

    # Filter out members with zero claims
    has_claims_mask = n_claims_arr > 0
    chunk_enrolled = chunk_enrolled[has_claims_mask].reset_index(drop=True)
    n_claims_arr = n_claims_arr[has_claims_mask]
    start_dates = start_dates[has_claims_mask].reset_index(drop=True)
    coverage_days = coverage_days[has_claims_mask].reset_index(drop=True)

    if len(chunk_enrolled) == 0:
        return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

    # Batch generate claims grouped by n_claims
    all_claims_list = []
    unique_n_claims = np.unique(n_claims_arr)

    claim_types_config = distributions['claims']['claim_type_distribution']
    claim_types = list(claim_types_config.keys())
    claim_type_probs = list(claim_types_config.values())

    amounts_config = distributions['claims']['amounts']
    maximums = {
        claim_type: params.get('max_amount', 1_000_000)
        for claim_type, params in amounts_config.items()
    }

    for n in unique_n_claims:
        mask = n_claims_arr == n
        n_members_in_group = mask.sum()

        # Get member info for this group
        group_members = chunk_enrolled[mask].reset_index(drop=True)
        group_start_dates = start_dates[mask].reset_index(drop=True)
        group_coverage_days = coverage_days[mask].reset_index(drop=True)

        # Generate all claim dates for this group
        all_dates = []
        all_member_ids = []
        all_insurance_ids = []

        for idx in range(n_members_in_group):
            member_start = group_start_dates.iloc[idx]
            member_coverage_days = int(group_coverage_days.iloc[idx])

            if member_coverage_days > 0:
                # Generate n dates for this member
                days_offsets = rng.integers(0, member_coverage_days, size=int(n))
                member_dates = [member_start + timedelta(days=int(offset)) for offset in days_offsets]
                member_dates.sort()

                all_dates.extend(member_dates)
                all_member_ids.extend([group_members.iloc[idx]['member_id']] * int(n))
                all_insurance_ids.extend([group_members.iloc[idx]['Insurance ID']] * int(n))

        # Generate claim types for all claims in this group
        claim_types_arr = rng.choice(claim_types, size=len(all_dates), p=claim_type_probs)

        # Generate amounts for all claims
        amounts = []
        for claim_type in claim_types_arr:
            params = amounts_config[claim_type]
            amount = rng.lognormal(params['mean_log'], params['std_log'])
            amount = max(amount, CLAIM_MINIMUMS.get(claim_type, 100))
            amount = min(amount, maximums.get(claim_type, 1_000_000))
            amounts.append(round(amount, 2))

        # Create claims for this group
        for i in range(len(all_dates)):
            all_claims_list.append({
                'member_id': all_member_ids[i],
                'amount': amounts[i],
                'date': all_dates[i].strftime('%Y-%m-%d'),
                'insurance_id': all_insurance_ids[i]
            })

    # Create DataFrame (claim IDs will be assigned after combining all chunks)
    if all_claims_list:
        claims_df = pd.DataFrame(all_claims_list)
    else:
        claims_df = pd.DataFrame(columns=['member_id', 'amount', 'date', 'insurance_id'])

    return claims_df


class ClaimsGenerator:
    """Generates synthetic medical claims data."""

    def __init__(self, config: dict, distributions: dict, reference_date: str, random_seed: int = None):
        """
        Initialize the claims generator.

        Args:
            config: Configuration dictionary
            distributions: Distribution parameters
            reference_date: Reference date for claims
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.distributions = distributions
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        self.rng = np.random.default_rng(random_seed)
        self.lookback_years = config.get('claims_lookback_years', 1)

    def _calculate_age_from_dob(self, dob_str: str) -> int:
        """Calculate age from date of birth."""
        dob = datetime.strptime(dob_str, '%Y-%m-%d')
        age = (self.reference_date - dob).days // 365
        return age

    def _get_age_group_for_claims(self, age: int) -> str:
        """Map age to age group for claims frequency."""
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
        elif age < 75:
            return '65-74'
        else:
            return '75+'

    def _get_claim_frequency(self, age: int) -> float:
        """
        Get expected claim frequency per year based on age.

        Args:
            age: Member's age

        Returns:
            Mean claims per year (Poisson lambda)
        """
        age_group = self._get_age_group_for_claims(age)
        claims_config = self.distributions['claims']
        return claims_config['frequency_by_age'][age_group]

    def _generate_claim_amount(self, claim_type: str) -> float:
        """
        Generate claim amount based on type (log-normal distribution).

        Args:
            claim_type: Type of claim

        Returns:
            Claim amount in dollars
        """
        amounts_config = self.distributions['claims']['amounts']
        params = amounts_config[claim_type]

        amount = self.rng.lognormal(params['mean_log'], params['std_log'])

        return round(self._apply_amount_bounds(claim_type, amount), 2)

    def _apply_amount_bounds(self, claim_type: str, amount: float) -> float:
        """Enforce min/max limits for claim amounts."""
        minimum = CLAIM_MINIMUMS.get(claim_type, 100)
        max_amount = self.distributions['claims']['amounts'].get(claim_type, {}).get('max_amount', 1_000_000)
        return float(min(max(amount, minimum), max_amount))

    def _select_claim_type(self) -> str:
        """Select a claim type based on distribution."""
        claim_types_config = self.distributions['claims']['claim_type_distribution']
        types = list(claim_types_config.keys())
        probs = list(claim_types_config.values())

        return self.rng.choice(types, p=probs)

    def _get_enrollment_coverage_period(self, enrollment_row: pd.Series) -> tuple:
        """
        Get the valid coverage period for an enrollment.

        Args:
            enrollment_row: Row from enrollments DataFrame

        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        start_date = datetime.strptime(enrollment_row['start_date'], '%Y-%m-%d')

        if pd.isna(enrollment_row['end_date']):
            # Active enrollment: coverage until reference date
            end_date = self.reference_date
        else:
            end_date = datetime.strptime(enrollment_row['end_date'], '%Y-%m-%d')

        # Limit to lookback period
        earliest_date = self.reference_date - timedelta(days=self.lookback_years * 365)
        start_date = max(start_date, earliest_date)

        return start_date, end_date

    def generate_claims_for_member(
        self,
        member_row: pd.Series,
        enrollment_row: pd.Series
    ) -> List[Dict]:
        """
        Generate claims for a single member during their enrollment period.

        Args:
            member_row: Row from members DataFrame
            enrollment_row: Row from enrollments DataFrame

        Returns:
            List of claim dictionaries
        """
        # Get coverage period
        start_date, end_date = self._get_enrollment_coverage_period(enrollment_row)

        # If no coverage in lookback period, no claims
        if start_date >= end_date:
            return []

        # Calculate coverage duration in years
        coverage_days = (end_date - start_date).days
        coverage_years = coverage_days / 365.0

        # Get expected claims frequency
        age = self._calculate_age_from_dob(member_row['DOB'])
        lambda_per_year = self._get_claim_frequency(age)

        # Calculate total expected claims
        lambda_total = lambda_per_year * coverage_years

        # Generate number of claims (Poisson)
        n_claims = self.rng.poisson(lambda_total)

        if n_claims == 0:
            return []

        # Generate claim dates uniformly across coverage period
        claim_timestamps = [
            start_date + timedelta(days=int(self.rng.integers(0, coverage_days)))
            for _ in range(n_claims)
        ]

        # Sort by date
        claim_timestamps.sort()

        # Generate claims
        claims = []
        for claim_date in claim_timestamps:
            claim_type = self._select_claim_type()
            amount = self._generate_claim_amount(claim_type)

            claims.append({
                'member_id': member_row['member_id'],
                'amount': amount,
                'date': claim_date.strftime('%Y-%m-%d'),
                'insurance_id': member_row['Insurance ID'],
                '_claim_type': claim_type  # For debugging, not in schema
            })

        return claims

    def generate_all_claims(
        self,
        members_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        member_condition_df: pd.DataFrame = None  # TIER 1 FEATURE 4
    ) -> pd.DataFrame:
        """
        Generate claims for all enrolled members with condition-specific patterns.

        Args:
            members_df: DataFrame with member data
            enrollments_df: DataFrame with enrollment data
            member_condition_df: DataFrame with member conditions

        Returns:
            DataFrame with claims data
        """
        # Pre-merge members + enrollments
        enrolled = members_df[members_df['Enrollment ID'].notna()].copy()
        enrolled = enrolled.merge(
            enrollments_df,
            left_on='Enrollment ID',
            right_on='Enrollment_ID',
            how='inner'
        )

        if enrolled.empty:
            return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

        # TIER 1 FEATURE 4: Merge condition data for condition-specific claims
        if member_condition_df is not None and not member_condition_df.empty:
            # Count conditions per member
            condition_counts = member_condition_df.groupby('Member_ID').size().reset_index(name='n_conditions')
            enrolled = enrolled.merge(condition_counts, left_on='member_id', right_on='Member_ID', how='left')
            enrolled['n_conditions'] = enrolled['n_conditions'].fillna(0).astype(int)

            # Create condition flags (boolean columns for each condition)
            # Get unique condition IDs
            unique_conditions = member_condition_df['Condition_ID'].unique()
            for cond_id in unique_conditions:
                members_with_cond = member_condition_df[
                    member_condition_df['Condition_ID'] == cond_id
                ]['Member_ID'].unique()
                enrolled[f'has_condition_{cond_id}'] = enrolled['member_id'].isin(members_with_cond)
        else:
            enrolled['n_conditions'] = 0

        # Check if parallel processing is enabled
        enable_parallel = self.config.get('enable_parallel', False)
        chunk_size = self.config.get('chunk_size', 100000)
        n_enrolled = len(enrolled)

        # Use parallel processing if enabled and dataset is large enough
        if enable_parallel and n_enrolled >= chunk_size:
            # Get number of workers
            n_workers = self.config.get('n_workers')
            if n_workers is None:
                n_workers = max(1, multiprocessing.cpu_count() - 1)

            # Calculate chunks
            n_chunks = (n_enrolled + chunk_size - 1) // chunk_size

            # Prepare chunks
            chunks = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_enrolled)
                chunk_enrolled = enrolled.iloc[start_idx:end_idx]

                # Convert to dict for serialization
                chunk_enrolled_dict = chunk_enrolled.to_dict('list')

                # Chunk-specific seed
                chunk_seed = None if self.config.get('random_seed') is None else self.config.get('random_seed') + 1000 + i

                chunks.append((chunk_enrolled_dict, self.config, self.distributions, chunk_seed, i))

            # Process chunks in parallel
            with multiprocessing.Pool(n_workers) as pool:
                chunk_results = pool.map(_generate_claims_chunk, chunks)

            # Combine results
            if chunk_results:
                claims_df = pd.concat(chunk_results, ignore_index=True)

                # Assign claim IDs
                if not claims_df.empty:
                    claims_df.insert(0, 'claim_id', range(1, len(claims_df) + 1))
            else:
                claims_df = pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

            return claims_df

        else:
            # Serial processing (vectorized)
            return self._generate_claims_serial(enrolled)

    def _generate_claims_serial(self, enrolled: pd.DataFrame) -> pd.DataFrame:
        """
        Generate claims for enrolled members (serial vectorized version).

        Args:
            enrolled: Pre-merged DataFrame of enrolled members

        Returns:
            DataFrame with claims data
        """
        # Vectorize age calculation
        dobs = pd.to_datetime(enrolled['DOB'])
        ages = (self.reference_date - dobs).dt.days // 365

        # Vectorize age group mapping
        age_groups = pd.cut(
            ages,
            bins=[-1, 18, 35, 45, 55, 65, 75, 150],
            labels=['0-17', '18-34', '35-44', '45-54', '55-64', '65-74', '75+']
        )

        # Map age groups to claim frequencies
        claims_config = self.distributions['claims']
        frequency_map = claims_config['frequency_by_age']
        lambda_per_year = age_groups.map(frequency_map).astype(float)

        # TIER 1 FEATURE 4: Apply condition multipliers to lambda (claim frequency)
        condition_multipliers = claims_config.get('condition_multipliers', {})
        if condition_multipliers and 'n_conditions' in enrolled.columns:
            # Get condition columns (format: has_condition_X)
            condition_cols = [col for col in enrolled.columns if col.startswith('has_condition_')]

            # Build mapping of condition_id to condition name (from distributions)
            # For simplicity, we'll use a direct approach with the condition IDs in the columns
            # The multipliers are keyed by condition name, but we have condition IDs

            # Add per-additional-condition multiplier
            per_additional = condition_multipliers.get('per_additional_condition', 0.5)
            lambda_per_year += enrolled['n_conditions'] * per_additional

            # Note: For now, we'll apply the per-condition multiplier only
            # Full condition-specific multipliers would require condition name → ID mapping
            # which we don't have readily available here. This is a simplified implementation.

        # Cap lambda at reasonable maximum
        lambda_per_year = np.minimum(lambda_per_year, 25)

        # Vectorize coverage period calculation
        start_dates = pd.to_datetime(enrolled['start_date'])
        end_dates = pd.to_datetime(enrolled['end_date'].fillna(self.reference_date.strftime('%Y-%m-%d')))
        earliest_date = self.reference_date - timedelta(days=self.lookback_years * 365)

        # Apply lookback limit
        start_dates = start_dates.clip(lower=earliest_date)

        # Calculate coverage years
        coverage_days = (end_dates - start_dates).dt.days
        coverage_years = coverage_days / 365.0

        # Filter out invalid coverage periods
        valid_mask = coverage_days > 0
        enrolled = enrolled[valid_mask].copy()
        lambda_per_year = lambda_per_year[valid_mask]
        coverage_years = coverage_years[valid_mask]
        start_dates = start_dates[valid_mask]
        coverage_days = coverage_days[valid_mask]

        if enrolled.empty:
            return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

        # Vectorize Poisson sampling for n_claims
        lambda_total = lambda_per_year * coverage_years
        n_claims_arr = self.rng.poisson(lambda_total)

        # Filter out members with zero claims
        has_claims_mask = n_claims_arr > 0
        enrolled = enrolled[has_claims_mask].reset_index(drop=True)
        n_claims_arr = n_claims_arr[has_claims_mask]
        start_dates = start_dates[has_claims_mask].reset_index(drop=True)
        coverage_days = coverage_days[has_claims_mask].reset_index(drop=True)

        if len(enrolled) == 0:
            return pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

        # Batch generate claims grouped by n_claims
        all_claims_list = []
        unique_n_claims = np.unique(n_claims_arr)

        claim_types_config = self.distributions['claims']['claim_type_distribution']
        claim_types = list(claim_types_config.keys())
        claim_type_probs = list(claim_types_config.values())

        amounts_config = self.distributions['claims']['amounts']

        for n in unique_n_claims:
            mask = n_claims_arr == n
            n_members_in_group = mask.sum()

            # Get member info for this group
            group_members = enrolled[mask].reset_index(drop=True)
            group_start_dates = start_dates[mask].reset_index(drop=True)
            group_coverage_days = coverage_days[mask].reset_index(drop=True)

            # Generate all claim dates for this group
            all_dates = []
            all_member_ids = []
            all_insurance_ids = []

            for idx in range(n_members_in_group):
                member_start = group_start_dates.iloc[idx]
                member_coverage_days = int(group_coverage_days.iloc[idx])

                if member_coverage_days > 0:
                    # Generate n dates for this member
                    days_offsets = self.rng.integers(0, member_coverage_days, size=int(n))
                    member_dates = [member_start + timedelta(days=int(offset)) for offset in days_offsets]
                    member_dates.sort()

                    all_dates.extend(member_dates)
                    all_member_ids.extend([group_members.iloc[idx]['member_id']] * int(n))
                    all_insurance_ids.extend([group_members.iloc[idx]['Insurance ID']] * int(n))

            # Generate claim types for all claims in this group
            claim_types_arr = self.rng.choice(claim_types, size=len(all_dates), p=claim_type_probs)

            # TIER 1 FEATURE 4: Prepare age multipliers for this group
            age_adjustments_config = claims_config.get('condition_amount_adjustments', {}).get('age_adjustments', {})
            pediatric_mult = age_adjustments_config.get('pediatric_multiplier', 0.7)
            elderly_mult = age_adjustments_config.get('elderly_multiplier', 1.4)

            # Build age multiplier array (one per claim)
            age_multipliers = []
            for member_id in all_member_ids:
                member_row = group_members[group_members['member_id'] == member_id]
                if not member_row.empty:
                    age = member_row.iloc[0]['_age'] if '_age' in member_row.columns else 50
                    if age < 18:
                        age_multipliers.append(pediatric_mult)
                    elif age >= 65:
                        age_multipliers.append(elderly_mult)
                    else:
                        age_multipliers.append(1.0)
                else:
                    age_multipliers.append(1.0)

            # Generate amounts for all claims
            amounts = []
            for idx, claim_type in enumerate(claim_types_arr):
                params = amounts_config[claim_type]
                base_amount = self.rng.lognormal(params['mean_log'], params['std_log'])

                # Apply age multiplier then clamp
                amount = base_amount * age_multipliers[idx]
                amount = self._apply_amount_bounds(claim_type, amount)
                amounts.append(round(amount, 2))

            # Create claims for this group
            for i in range(len(all_dates)):
                all_claims_list.append({
                    'member_id': all_member_ids[i],
                    'amount': amounts[i],
                    'date': all_dates[i].strftime('%Y-%m-%d'),
                    'insurance_id': all_insurance_ids[i]
                })

        # Create DataFrame and assign claim IDs
        if all_claims_list:
            claims_df = pd.DataFrame(all_claims_list)
            claims_df.insert(0, 'claim_id', range(1, len(claims_df) + 1))
        else:
            claims_df = pd.DataFrame(columns=['claim_id', 'member_id', 'amount', 'date', 'insurance_id'])

        return claims_df


def generate_claims_data(
    config: dict,
    distributions: dict,
    members_df: pd.DataFrame,
    enrollments_df: pd.DataFrame,
    member_condition_df: pd.DataFrame = None  # TIER 1 FEATURE 4: Add conditions
) -> pd.DataFrame:
    """
    Generate claims data with condition-specific patterns.

    Args:
        config: Configuration dictionary
        distributions: Distribution parameters
        members_df: DataFrame with member data
        enrollments_df: DataFrame with enrollment data
        member_condition_df: DataFrame with member conditions (for condition-specific claims)

    Returns:
        DataFrame with claims data
    """
    reference_date = config.get('reference_date', '2024-01-01')
    random_seed = config.get('random_seed')

    generator = ClaimsGenerator(config, distributions, reference_date, random_seed)

    return generator.generate_all_claims(members_df, enrollments_df, member_condition_df)
