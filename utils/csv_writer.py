"""
CSV export utilities for synthetic data.
Handles proper formatting and column name mappings for the database schema.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional


class CSVWriter:
    """
    Handles writing DataFrames to CSV files with proper formatting
    for MySQL import compatibility.
    """

    def __init__(self, output_dir: Path, encoding: str = 'utf-8', date_format: str = '%Y-%m-%d'):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory to write CSV files
            encoding: Character encoding for CSV files
            date_format: Date format string
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.encoding = encoding
        self.date_format = date_format

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        index: bool = False,
        na_rep: str = '\\N'
    ) -> Path:
        """
        Write a DataFrame to a CSV file.

        Args:
            df: DataFrame to write
            table_name: Name of the table (used for filename)
            index: Whether to include the index
            na_rep: String representation for NULL values (MySQL uses \\N)

        Returns:
            Path to the written file
        """
        output_path = self.output_dir / f"{table_name.lower()}.csv"

        df.to_csv(
            output_path,
            index=index,
            encoding=self.encoding,
            na_rep=na_rep,
            date_format=self.date_format,
            quoting=0,  # QUOTE_MINIMAL - only quote when necessary
        )

        return output_path

    def write_table(
        self,
        data: List[dict],
        table_name: str,
        column_order: Optional[List[str]] = None
    ) -> Path:
        """
        Write table data from a list of dictionaries.

        Args:
            data: List of row dictionaries
            table_name: Name of the table
            column_order: Optional list specifying column order

        Returns:
            Path to the written file
        """
        df = pd.DataFrame(data)

        if column_order:
            # Reorder columns if specified
            df = df[column_order]

        return self.write_dataframe(df, table_name)


# Column mappings for each table to match schema exactly
# Handles mixed case and spaces in column names

INSURANCE_COLUMNS = [
    'insurance_id',
    'insurance_name',
    'contact_info'
]

PLAN_COLUMNS = [
    'plan_id',
    'plan_name',
    'insurance_id',
    'base_rate',
    'deductable'  # Note: Intentional misspelling to match schema
]

FACILITY_COLUMNS = [
    'Facility_ID',  # PascalCase in schema
    'NPI',
    'Name',
    'State',
    'Contact_info'
]

CONDITION_COLUMNS = [
    'Condition_ID',
    'Condition_name'
]

ENROLLMENT_COLUMNS = [
    'Enrollment_ID',
    'Coverage_tier',
    'start_date',
    'end_date',
    'state',
    'plan_id',
    'premium'
]

MEMBERS_COLUMNS = [
    'member_id',
    'DOB',
    'Sex',
    'state',
    'height_cm',
    'weight_kg',
    'heart_rate',
    'blood_pressure',
    'blood_oxygen',
    'smoker',
    'drinker',
    'exercise_minutes_per_week',
    'sleep_hours_per_night',
    'housing insecurity',  # Space in column name
    'employment status',    # Space in column name
    'Facility_ID',
    'Insurance ID',         # Space in column name
    'Enrollment ID'         # Space in column name
]

MEMBER_CONDITION_COLUMNS = [
    'Member_ID',
    'Condition_ID',
    'Diagnostic_date'
]

CLAIMS_COLUMNS = [
    'claim_id',
    'member_id',
    'amount',
    'date',
    'insurance_id'
]


def get_column_order(table_name: str) -> List[str]:
    """
    Get the correct column order for a table.

    Args:
        table_name: Name of the table

    Returns:
        List of column names in correct order
    """
    column_map = {
        'INSURANCE': INSURANCE_COLUMNS,
        'PLAN': PLAN_COLUMNS,
        'FACILITY': FACILITY_COLUMNS,
        'CONDITION': CONDITION_COLUMNS,
        'ENROLLMENT': ENROLLMENT_COLUMNS,
        'MEMBERS': MEMBERS_COLUMNS,
        'MEMBER_CONDITION': MEMBER_CONDITION_COLUMNS,
        'CLAIMS': CLAIMS_COLUMNS
    }

    return column_map.get(table_name.upper(), [])


def validate_columns(df: pd.DataFrame, table_name: str) -> tuple[bool, List[str]]:
    """
    Validate that DataFrame has all required columns for a table.

    Args:
        df: DataFrame to validate
        table_name: Name of the table

    Returns:
        Tuple of (is_valid, list_of_missing_columns)
    """
    expected_columns = set(get_column_order(table_name))
    actual_columns = set(df.columns)

    missing = expected_columns - actual_columns

    return len(missing) == 0, list(missing)


def create_loader_sql(output_dir: Path, database_name: str = 'health_insurance') -> Path:
    """
    Create a SQL script to load all CSV files into MySQL.

    Args:
        output_dir: Directory containing CSV files
        database_name: Name of the database

    Returns:
        Path to the generated SQL script
    """
    sql_path = output_dir / 'load_data.sql'

    # Table loading order (respects foreign keys)
    table_order = [
        'insurance',
        'plan',
        'facility',
        'condition',
        'members',
        'enrollment',
        'member_condition',
        'claims'
    ]

    sql_lines = [
        f"-- Auto-generated SQL loader for synthetic data",
        f"-- Generated on: {pd.Timestamp.now()}",
        f"",
        f"USE {database_name};",
        f"",
        f"SET FOREIGN_KEY_CHECKS=0;",
        f"SET AUTOCOMMIT=0;",
        f"",
    ]

    for table in table_order:
        csv_file = f"{table}.csv"
        table_upper = table.upper()

        sql_lines.extend([
            f"-- Load {table_upper}",
            f"TRUNCATE TABLE {table_upper};",
            f"LOAD DATA LOCAL INFILE '{csv_file}'",
            f"INTO TABLE {table_upper}",
            f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"'",
            f"LINES TERMINATED BY '\\n'",
            f"IGNORE 1 ROWS;",
            f"",
        ])

    sql_lines.extend([
        f"COMMIT;",
        f"SET FOREIGN_KEY_CHECKS=1;",
        f"",
        f"-- Verify row counts",
    ])

    for table in table_order:
        sql_lines.append(f"SELECT '{table.upper()}' as 'Table', COUNT(*) as 'Rows' FROM {table.upper()};")

    with open(sql_path, 'w') as f:
        f.write('\n'.join(sql_lines))

    return sql_path
