#!/usr/bin/env python3
"""
Main script to generate synthetic health insurance data.

Usage:
    python generate_data.py [--config CONFIG_PATH]

Example:
    python generate_data.py
    python generate_data.py --config config/custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_loader import get_config
from utils.progress import ProgressTracker, print_generation_plan
from utils.csv_writer import CSVWriter, get_column_order, create_loader_sql
from utils.validators import validate_generated_data

from generators.reference_data import generate_all_reference_data
from generators.members import generate_members_data
from generators.enrollment import generate_enrollment_data
from generators.claims import generate_claims_data


def main():
    """Main data generation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic health insurance data')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation checks')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = get_config(args.config)

    # Print generation plan
    print_generation_plan(config.config, config.distributions)

    # Initialize progress tracker
    tracker = ProgressTracker(
        enabled=config.get('enable_progress_bars', True),
        show_eta=config.get('show_time_estimates', True)
    )
    tracker.start()

    # Initialize CSV writer
    output_dir = config.get_output_dir()
    csv_writer = CSVWriter(
        output_dir=output_dir,
        encoding=config.get('csv_encoding', 'utf-8'),
        date_format=config.get('csv_date_format', '%Y-%m-%d')
    )

    print(f"\nOutput directory: {output_dir}")

    # Dictionary to store all generated tables
    tables = {}

    # ========================================================================
    # STAGE 1: Generate Reference Data
    # ========================================================================
    with tracker.stage("Generate Reference Data (Insurance, Plans, Facilities, Conditions)"):
        state_pops_path = config.get_reference_data_path(
            config.get('state_populations_file', 'state_populations.csv')
        )
        conditions_path = config.get_reference_data_path(
            config.get('conditions_file', 'conditions.csv')
        )

        reference_data = generate_all_reference_data(
            config=config.config,
            state_populations_path=str(state_pops_path),
            conditions_path=str(conditions_path)
        )

        # Store tables
        tables['INSURANCE'] = reference_data['INSURANCE']
        tables['PLAN'] = reference_data['PLAN']
        tables['FACILITY'] = reference_data['FACILITY']
        tables['CONDITION'] = reference_data['CONDITION']
        state_populations = reference_data['_state_populations']

        # Print counts
        print(f"  ✓ Generated {len(tables['INSURANCE'])} insurance providers")
        print(f"  ✓ Generated {len(tables['PLAN'])} insurance plans")
        print(f"  ✓ Generated {len(tables['FACILITY'])} medical facilities")
        print(f"  ✓ Generated {len(tables['CONDITION'])} medical conditions")

    # ========================================================================
    # STAGE 2: Generate Members and Conditions
    # ========================================================================
    with tracker.stage("Generate Members (Demographics, Biometrics, Lifestyle)"):
        members_df, member_condition_df = generate_members_data(
            config=config.config,
            distributions=config.distributions,
            state_populations=state_populations,
            conditions_df=tables['CONDITION'],
            facilities_df=tables['FACILITY']
        )

        tables['MEMBERS'] = members_df
        tables['MEMBER_CONDITION'] = member_condition_df

        print(f"  ✓ Generated {len(members_df):,} members")
        print(f"  ✓ Generated {len(member_condition_df):,} member-condition associations")

    # ========================================================================
    # STAGE 3: Generate Enrollments
    # ========================================================================
    with tracker.stage("Generate Insurance Enrollments"):
        members_df, enrollments_df = generate_enrollment_data(
            config=config.config,
            distributions=config.distributions,
            members_df=members_df,
            plans_df=tables['PLAN'],
            insurance_df=tables['INSURANCE']
        )

        # Update tables with enrollment data
        tables['MEMBERS'] = members_df
        tables['ENROLLMENT'] = enrollments_df

        print(f"  ✓ Generated {len(enrollments_df):,} enrollment records")
        print(f"  ✓ {(members_df['Insurance ID'].notna().sum() / len(members_df) * 100):.1f}% of members have insurance")

    # ========================================================================
    # STAGE 4: Generate Claims (with condition-specific patterns)
    # ========================================================================
    with tracker.stage("Generate Medical Claims"):
        claims_df = generate_claims_data(
            config=config.config,
            distributions=config.distributions,
            members_df=members_df,
            enrollments_df=enrollments_df,
            member_condition_df=member_condition_df  # TIER 1 FEATURE 4
        )

        tables['CLAIMS'] = claims_df

        if len(claims_df) > 0:
            total_amount = claims_df['amount'].sum()
            avg_amount = claims_df['amount'].mean()
            print(f"  ✓ Generated {len(claims_df):,} claims")
            print(f"  ✓ Total claim amount: ${total_amount:,.2f}")
            print(f"  ✓ Average claim amount: ${avg_amount:,.2f}")
        else:
            print(f"  ⚠ No claims generated (check enrollment and claims config)")

    # ========================================================================
    # STAGE 5: Validate Data
    # ========================================================================
    if not args.skip_validation:
        with tracker.stage("Validate Generated Data"):
            is_valid = validate_generated_data(tables)

            if not is_valid:
                print("\n⚠ WARNING: Validation found issues. Review errors above.")
                response = input("Continue with CSV export anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Aborting. Fix validation errors and try again.")
                    return 1

    # ========================================================================
    # STAGE 6: Export to CSV
    # ========================================================================
    with tracker.stage("Export to CSV Files"):
        export_order = [
            'INSURANCE', 'PLAN', 'FACILITY', 'CONDITION',
            'MEMBERS', 'ENROLLMENT', 'MEMBER_CONDITION', 'CLAIMS'
        ]

        for table_name in export_order:
            if table_name in tables:
                df = tables[table_name]
                column_order = get_column_order(table_name)

                # Reorder columns if needed
                if column_order and all(col in df.columns for col in column_order):
                    df = df[column_order]

                output_path = csv_writer.write_dataframe(df, table_name.lower())
                print(f"  ✓ Exported {table_name}: {len(df):,} rows → {output_path.name}")

    # ========================================================================
    # STAGE 7: Create SQL Loader Script
    # ========================================================================
    with tracker.stage("Create SQL Loader Script"):
        sql_path = create_loader_sql(output_dir, database_name='health_insurance')
        print(f"  ✓ Created SQL loader script: {sql_path.name}")

    # ========================================================================
    # Print Summary
    # ========================================================================
    tracker.print_summary()

    print("\n" + "=" * 80)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput Location: {output_dir}")
    print(f"\nGenerated Files:")
    for table_name in export_order:
        if table_name in tables:
            print(f"  • {table_name.lower()}.csv ({len(tables[table_name]):,} rows)")
    print(f"  • load_data.sql (MySQL loader script)")

    print(f"\nNext Steps:")
    print(f"  1. Review the generated CSV files in: {output_dir}")
    print(f"  2. Load into MySQL:")
    print(f"     cd {output_dir}")
    print(f"     mysql -u <username> -p --local-infile=1 < load_data.sql")
    print(f"  3. Verify with: python validate_output.py")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
