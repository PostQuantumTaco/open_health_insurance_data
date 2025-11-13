#!/usr/bin/env python3
"""
Validate generated CSV files.

Usage:
    python validate_output.py [--data-dir DATA_DIR]

Example:
    python validate_output.py
    python validate_output.py --data-dir synthetic_data
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.validators import validate_generated_data
from utils.config_loader import ConfigLoader
from utils.statistical_validator import StatisticalValidator


def load_csv_files(data_dir: Path) -> dict:
    """
    Load all CSV files from the data directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Dictionary of table name -> DataFrame
    """
    tables = {}

    table_files = {
        'INSURANCE': 'insurance.csv',
        'PLAN': 'plan.csv',
        'FACILITY': 'facility.csv',
        'CONDITION': 'condition.csv',
        'MEMBERS': 'members.csv',
        'ENROLLMENT': 'enrollment.csv',
        'MEMBER_CONDITION': 'member_condition.csv',
        'CLAIMS': 'claims.csv'
    }

    print("\nLoading CSV files...")
    for table_name, filename in table_files.items():
        filepath = data_dir / filename

        if not filepath.exists():
            print(f"  ⚠ WARNING: {filename} not found, skipping")
            continue

        try:
            df = pd.read_csv(filepath, na_values=['\\N', 'NULL', '', 'nan', 'NaN', 'None'], keep_default_na=True)

            # Convert numeric columns to proper types
            if table_name == 'MEMBERS':
                numeric_cols = ['height_cm', 'weight_kg', 'heart_rate', 'blood_oxygen',
                               'exercise_minutes_per_week', 'sleep_hours_per_night']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Convert boolean columns
                bool_cols = ['smoker', 'drinker', 'housing insecurity', 'employment status']
                for col in bool_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            elif table_name == 'CLAIMS':
                if 'amount' in df.columns:
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

            elif table_name == 'ENROLLMENT':
                if 'premium' in df.columns:
                    df['premium'] = pd.to_numeric(df['premium'], errors='coerce')

            elif table_name == 'PLAN':
                numeric_cols = ['base_rate', 'deductable']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            tables[table_name] = df
            print(f"  ✓ Loaded {table_name}: {len(df):,} rows")
        except Exception as e:
            print(f"  ✗ ERROR loading {filename}: {e}")

    return tables


def print_data_summary(tables: dict):
    """
    Print summary statistics for the loaded data.

    Args:
        tables: Dictionary of table DataFrames
    """
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    if 'MEMBERS' in tables:
        members = tables['MEMBERS']
        print(f"\nMembers: {len(members):,}")

        if 'Sex' in members.columns:
            sex_dist = members['Sex'].value_counts()
            print("  Sex distribution:")
            for sex, count in sex_dist.items():
                pct = count / len(members) * 100
                print(f"    {sex}: {count:,} ({pct:.1f}%)")

        if 'Insurance ID' in members.columns:
            insured = members['Insurance ID'].notna().sum()
            pct = insured / len(members) * 100
            print(f"  Insured: {insured:,} ({pct:.1f}%)")

    if 'ENROLLMENT' in tables:
        enrollments = tables['ENROLLMENT']
        print(f"\nEnrollments: {len(enrollments):,}")

        if 'end_date' in enrollments.columns:
            active = enrollments['end_date'].isna().sum()
            pct = active / len(enrollments) * 100
            print(f"  Active: {active:,} ({pct:.1f}%)")

    if 'MEMBER_CONDITION' in tables:
        member_conditions = tables['MEMBER_CONDITION']
        print(f"\nMember-Condition Associations: {len(member_conditions):,}")

        if 'MEMBERS' in tables:
            unique_members_with_conditions = member_conditions['Member_ID'].nunique()
            avg_conditions = len(member_conditions) / unique_members_with_conditions
            print(f"  Avg conditions per affected member: {avg_conditions:.2f}")

    if 'CLAIMS' in tables:
        claims = tables['CLAIMS']
        print(f"\nClaims: {len(claims):,}")

        if 'amount' in claims.columns:
            total_amount = claims['amount'].sum()
            avg_amount = claims['amount'].mean()
            median_amount = claims['amount'].median()
            print(f"  Total amount: ${total_amount:,.2f}")
            print(f"  Average amount: ${avg_amount:,.2f}")
            print(f"  Median amount: ${median_amount:,.2f}")

    print("=" * 80)


def main():
    """Main validation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate generated CSV files')
    parser.add_argument('--data-dir', default='synthetic_data', help='Directory containing CSV files')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config.yaml')
    parser.add_argument('--stats', action='store_true', help='Run statistical validation suite')
    parser.add_argument('--stats-sample-size', type=int, default=200000,
                        help='Maximum number of members to sample for statistical tests')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"✗ ERROR: Data directory not found: {data_dir}")
        print(f"  Make sure you've run generate_data.py first")
        return 1

    print("=" * 80)
    print("VALIDATING GENERATED DATA")
    print("=" * 80)
    print(f"\nData Directory: {data_dir.absolute()}")

    # Load CSV files
    tables = load_csv_files(data_dir)

    if not tables:
        print("\n✗ ERROR: No CSV files found or all files failed to load")
        return 1

    # Print summary statistics
    print_data_summary(tables)

    # Run validation
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION CHECKS")
    print("=" * 80)

    structural_passed = validate_generated_data(tables)

    stats_passed = True
    if args.stats:
        try:
            config_loader = ConfigLoader(args.config)
        except FileNotFoundError as exc:
            print(f"\n✗ Statistical validation aborted: {exc}")
            return 1

        reference_date = config_loader.get('reference_date', '2024-01-01')
        stats_validator = StatisticalValidator(config_loader.distributions, reference_date=reference_date)
        stats_passed = stats_validator.run(tables, sample_size=args.stats_sample_size)
        stats_validator.print_report()

    overall_passed = structural_passed and stats_passed

    if overall_passed:
        if args.stats:
            print("\n✓ Structural and statistical validation checks passed!")
        else:
            print("\n✓ All validation checks passed!")
        print("\nData is ready to load into MySQL.")
        return 0

    print("\n✗ Validation failed. Please review errors above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
