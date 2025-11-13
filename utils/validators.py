"""
Basic validation checks for generated data.
MVP version: focuses on foreign key integrity and basic data quality.
"""

import pandas as pd
from typing import Dict, List, Tuple
import re


class DataValidator:
    """Performs basic validation checks on generated data."""

    def __init__(self):
        """Initialize the validator."""
        self.errors = []
        self.warnings = []

    def validate_all(self, tables: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.

        Args:
            tables: Dictionary of table name -> DataFrame

        Returns:
            Tuple of (is_valid, list_of_errors, list_of_warnings)
        """
        self.errors = []
        self.warnings = []

        # Check row counts
        self._check_row_counts(tables)

        # Check primary keys
        self._check_primary_keys(tables)

        # Check foreign keys
        self._check_foreign_keys(tables)

        # Check data quality
        self._check_data_quality(tables)

        # Check business rules
        self._check_business_rules(tables)

        # TIER 1 FEATURE VALIDATION: Check clinical realism
        self._check_clinical_realism(tables)

        is_valid = len(self.errors) == 0

        return is_valid, self.errors, self.warnings

    def _check_row_counts(self, tables: Dict[str, pd.DataFrame]):
        """Check that tables have reasonable row counts."""
        expected_ratios = {
            'INSURANCE': (8, 8),          # Exactly 8
            'PLAN': (40, 40),             # 8 insurers × 5 plans
            'FACILITY': (50, 500),        # Dynamically adjusted below
            'CONDITION': (50, 120),       # Spec requires 50+ lookup rows
            'MEMBERS': (1000, 10000000),  # 1K to 10M members
            'ENROLLMENT': (0, None),      # Variable
            'MEMBER_CONDITION': (0, None), # Variable
            'CLAIMS': (0, None)           # Variable
        }

        member_count = len(tables['MEMBERS']) if 'MEMBERS' in tables else None

        for table_name, (min_rows, max_rows) in expected_ratios.items():
            if table_name in tables:
                count = len(tables[table_name])

                if table_name == 'FACILITY' and member_count:
                    expected = max(50, int(member_count * 0.002))
                    tolerance = max(50, int(expected * 0.5))
                    min_rows = max(50, expected - tolerance)
                    max_rows = expected + tolerance

                if max_rows and count > max_rows:
                    self.warnings.append(f"{table_name}: Unexpectedly high row count ({count:,})")
                elif count < min_rows:
                    self.errors.append(f"{table_name}: Too few rows ({count:,}, expected at least {min_rows})")

    def _check_primary_keys(self, tables: Dict[str, pd.DataFrame]):
        """Check that primary keys are unique and non-null."""
        pk_columns = {
            'INSURANCE': 'insurance_id',
            'PLAN': 'plan_id',
            'FACILITY': 'Facility_ID',
            'CONDITION': 'Condition_ID',
            'MEMBERS': 'member_id',
            'ENROLLMENT': 'Enrollment_ID',
            'CLAIMS': 'claim_id'
        }

        for table_name, pk_col in pk_columns.items():
            if table_name in tables:
                df = tables[table_name]

                # Check for nulls
                if df[pk_col].isna().any():
                    self.errors.append(f"{table_name}: NULL values in primary key {pk_col}")

                # Check for duplicates
                if df[pk_col].duplicated().any():
                    n_dupes = df[pk_col].duplicated().sum()
                    self.errors.append(f"{table_name}: {n_dupes} duplicate values in primary key {pk_col}")

    def _check_foreign_keys(self, tables: Dict[str, pd.DataFrame]):
        """Check foreign key integrity."""
        # Define FK relationships
        fk_checks = [
            ('PLAN', 'insurance_id', 'INSURANCE', 'insurance_id'),
            ('ENROLLMENT', 'plan_id', 'PLAN', 'plan_id'),
            ('MEMBER_CONDITION', 'Member_ID', 'MEMBERS', 'member_id'),
            ('MEMBER_CONDITION', 'Condition_ID', 'CONDITION', 'Condition_ID'),
            ('CLAIMS', 'member_id', 'MEMBERS', 'member_id'),
            ('CLAIMS', 'insurance_id', 'INSURANCE', 'insurance_id'),
        ]

        # Optional FKs in MEMBERS table
        optional_fk_checks = [
            ('MEMBERS', 'Facility_ID', 'FACILITY', 'Facility_ID'),
            ('MEMBERS', 'Insurance ID', 'INSURANCE', 'insurance_id'),
            ('MEMBERS', 'Enrollment ID', 'ENROLLMENT', 'Enrollment_ID'),
        ]

        # Check required FKs
        for child_table, child_col, parent_table, parent_col in fk_checks:
            self._check_fk_relationship(
                tables, child_table, child_col, parent_table, parent_col, allow_null=False
            )

        # Check optional FKs
        for child_table, child_col, parent_table, parent_col in optional_fk_checks:
            self._check_fk_relationship(
                tables, child_table, child_col, parent_table, parent_col, allow_null=True
            )

    def _check_fk_relationship(
        self,
        tables: Dict[str, pd.DataFrame],
        child_table: str,
        child_col: str,
        parent_table: str,
        parent_col: str,
        allow_null: bool = False
    ):
        """Check a specific FK relationship."""
        if child_table not in tables or parent_table not in tables:
            return

        child_df = tables[child_table]
        parent_df = tables[parent_table]

        # Get non-null FK values
        if allow_null:
            fk_values = child_df[child_df[child_col].notna()][child_col]
        else:
            fk_values = child_df[child_col]

        if len(fk_values) == 0:
            return

        # Check if all FK values exist in parent
        parent_values = set(parent_df[parent_col].values)
        invalid_fks = fk_values[~fk_values.isin(parent_values)]

        if len(invalid_fks) > 0:
            self.errors.append(
                f"{child_table}.{child_col} -> {parent_table}.{parent_col}: "
                f"{len(invalid_fks)} invalid foreign key references"
            )

    def _check_data_quality(self, tables: Dict[str, pd.DataFrame]):
        """Check basic data quality constraints."""
        if 'MEMBERS' in tables:
            members = tables['MEMBERS']

            # Check blood pressure format
            if 'blood_pressure' in members.columns:
                bp_pattern = r'^\d{2,3}/\d{2,3}$'
                non_null_bp = members[members['blood_pressure'].notna()]['blood_pressure']
                invalid_bp = non_null_bp[~non_null_bp.str.match(bp_pattern, na=False)]

                if len(invalid_bp) > 0:
                    self.errors.append(f"MEMBERS: {len(invalid_bp)} invalid blood pressure formats")

            # Check state codes (2-letter uppercase)
            if 'state' in members.columns:
                invalid_states = members[~members['state'].str.match(r'^[A-Z]{2}$', na=False)]
                if len(invalid_states) > 0:
                    self.errors.append(f"MEMBERS: {len(invalid_states)} invalid state codes")

            # Check blood oxygen range
            if 'blood_oxygen' in members.columns:
                non_null_bo = members[members['blood_oxygen'].notna()]['blood_oxygen']
                invalid_bo = non_null_bo[(non_null_bo < 88) | (non_null_bo > 100)]
                if len(invalid_bo) > 0:
                    self.errors.append(f"MEMBERS: {len(invalid_bo)} blood oxygen values out of range (88-100)")

            # Check heart rate range
            if 'heart_rate' in members.columns:
                non_null_hr = members[members['heart_rate'].notna()]['heart_rate']
                invalid_hr = non_null_hr[(non_null_hr < 20) | (non_null_hr > 300)]
                if len(invalid_hr) > 0:
                    self.errors.append(f"MEMBERS: {len(invalid_hr)} heart rate values out of range (20-300)")

        # Check claims amounts
        if 'CLAIMS' in tables:
            claims = tables['CLAIMS']
            if 'amount' in claims.columns:
                invalid_amounts = claims[(claims['amount'] <= 0) | (claims['amount'] > 1000000)]
                if len(invalid_amounts) > 0:
                    self.warnings.append(f"CLAIMS: {len(invalid_amounts)} claims with unusual amounts")

    def _check_business_rules(self, tables: Dict[str, pd.DataFrame]):
        """Check business logic constraints."""
        # Check enrollment dates
        if 'ENROLLMENT' in tables:
            enrollments = tables['ENROLLMENT']

            # end_date should be after start_date (when not null)
            ended = enrollments[enrollments['end_date'].notna()].copy()
            if len(ended) > 0:
                ended['start_dt'] = pd.to_datetime(ended['start_date'])
                ended['end_dt'] = pd.to_datetime(ended['end_date'])
                invalid_dates = ended[ended['end_dt'] <= ended['start_dt']]

                if len(invalid_dates) > 0:
                    self.errors.append(f"ENROLLMENT: {len(invalid_dates)} records with end_date <= start_date")

        # Check that claims are during enrollment period (simplified check)
        if 'CLAIMS' in tables and 'MEMBERS' in tables and 'ENROLLMENT' in tables:
            # This is a more complex check - for MVP, just warn if dates seem off
            pass  # Skip for MVP

    def _check_clinical_realism(self, tables: Dict[str, pd.DataFrame]):
        """TIER 1 FEATURE VALIDATION: Check clinical relationships are realistic."""
        import numpy as np

        if 'MEMBERS' not in tables or 'MEMBER_CONDITION' not in tables:
            return

        members = tables['MEMBERS']
        conditions = tables['MEMBER_CONDITION']

        # Feature 2: Risk Factor Effects
        self._check_risk_factor_effects(members, conditions)

        # Feature 1: Comorbidity Patterns
        self._check_comorbidity_patterns(members, conditions)

        # Feature 3: Condition-Vital Adjustments
        self._check_condition_vital_effects(members, conditions)

        # Feature 4: Condition-Claims Effects
        if 'CLAIMS' in tables:
            self._check_condition_claim_effects(members, conditions, tables['CLAIMS'])

    def _check_risk_factor_effects(self, members: pd.DataFrame, conditions: pd.DataFrame):
        """Verify risk multipliers are working (Feature 2)."""
        import numpy as np

        # Calculate BMI
        bmi = members['weight_kg'] / ((members['height_cm'] / 100) ** 2)
        obese_members = members[bmi >= 30]['member_id'].values

        # Check obesity → diabetes relationship
        # Get condition IDs (diabetes is typically ID 2, but let's be flexible)
        diabetes_members = conditions[conditions['Condition_ID'] == 2]['Member_ID'].unique()

        if len(obese_members) > 100 and len(diabetes_members) > 10:
            obese_diabetes = len(set(diabetes_members) & set(obese_members))
            non_obese_diabetes = len(set(diabetes_members) - set(obese_members))

            obese_rate = obese_diabetes / len(obese_members) if len(obese_members) > 0 else 0
            non_obese_rate = non_obese_diabetes / (len(members) - len(obese_members)) if (len(members) - len(obese_members)) > 0 else 0

            if non_obese_rate > 0:
                ratio = obese_rate / non_obese_rate

                if not (4.0 < ratio < 10.0):  # Expected ~7×, allow 4-10× range
                    self.warnings.append(f"Obesity-diabetes multiplier seems off: {ratio:.2f}× (expected ~7×)")
                else:
                    # Success - no warning needed, but could log
                    pass

    def _check_comorbidity_patterns(self, members: pd.DataFrame, conditions: pd.DataFrame):
        """Verify comorbidity rules are working (Feature 1)."""
        # Check: Of diabetics, ~70% should have hypertension
        diabetes_id = 2  # Typically Diabetes Type 2
        hypertension_id = 1  # Typically Hypertension

        diabetics = conditions[conditions['Condition_ID'] == diabetes_id]['Member_ID'].unique()

        if len(diabetics) > 50:  # Need reasonable sample
            diabetics_with_htn = conditions[
                (conditions['Member_ID'].isin(diabetics)) &
                (conditions['Condition_ID'] == hypertension_id)
            ]['Member_ID'].nunique()

            rate = diabetics_with_htn / len(diabetics)

            if not (0.50 < rate < 0.85):  # Expected ~70%, allow 50-85% range
                self.warnings.append(f"Diabetes→Hypertension comorbidity rate {rate:.1%} (expected ~70%)")

    def _check_condition_vital_effects(self, members: pd.DataFrame, conditions: pd.DataFrame):
        """Verify conditions affect vitals as expected (Feature 3)."""
        import numpy as np

        if 'blood_pressure' not in members.columns:
            return

        # Parse blood pressure
        bp_parts = members['blood_pressure'].fillna('120/80').str.split('/', expand=True)
        systolic = pd.to_numeric(bp_parts[0], errors='coerce')

        # Get hypertensive vs non-hypertensive members
        hypertension_id = 1
        htn_members = conditions[conditions['Condition_ID'] == hypertension_id]['Member_ID'].unique()

        if len(htn_members) > 50:
            htn_bp = systolic[members['member_id'].isin(htn_members)].mean()
            non_htn_bp = systolic[~members['member_id'].isin(htn_members)].mean()

            difference = htn_bp - non_htn_bp

            if difference < 10:
                self.warnings.append(
                    f"Hypertensive BP only {difference:.1f} mmHg higher (expected ~20 mmHg)"
                )

    def _check_condition_claim_effects(self, members: pd.DataFrame, conditions: pd.DataFrame, claims: pd.DataFrame):
        """Verify conditions affect claims as expected (Feature 4)."""
        # Check: Members with conditions should have more claims on average
        members_with_conditions = conditions['Member_ID'].unique()

        if len(members_with_conditions) > 100:
            claims_with_cond = claims[claims['member_id'].isin(members_with_conditions)].groupby('member_id').size()
            claims_without_cond = claims[~claims['member_id'].isin(members_with_conditions)].groupby('member_id').size()

            if len(claims_with_cond) > 0 and len(claims_without_cond) > 0:
                avg_with = claims_with_cond.mean()
                avg_without = claims_without_cond.mean()

                ratio = avg_with / avg_without if avg_without > 0 else 0

                if ratio < 1.2:  # Should be at least 20% more
                    self.warnings.append(
                        f"Condition-claims effect seems weak: {ratio:.2f}× (expected >1.5×)"
                    )

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("\n✓ All validation checks passed!")
        else:
            if len(self.errors) > 0:
                print(f"\n✗ ERRORS ({len(self.errors)}):")
                for error in self.errors:
                    print(f"  • {error}")

            if len(self.warnings) > 0:
                print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"  • {warning}")

        print("=" * 80 + "\n")


def validate_generated_data(tables: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate generated data and print report.

    Args:
        tables: Dictionary of table DataFrames

    Returns:
        True if validation passed, False otherwise
    """
    validator = DataValidator()
    is_valid, errors, warnings = validator.validate_all(tables)
    validator.print_report()

    return is_valid
