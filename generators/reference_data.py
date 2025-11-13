"""
Generate reference data: insurance providers, plans, facilities, and conditions.
These are small lookup tables that members reference.
"""

import pandas as pd
import numpy as np
import json
from faker import Faker
from typing import List, Dict


class ReferenceDataGenerator:
    """Generates reference data for the health insurance system."""

    def __init__(self, random_seed: int = None):
        """
        Initialize the reference data generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        self.fake = Faker()
        if random_seed:
            Faker.seed(random_seed)

    def generate_insurance_providers(self, n_providers: int = 8) -> pd.DataFrame:
        """
        Generate insurance provider data.

        Args:
            n_providers: Number of insurance providers (default: 8)

        Returns:
            DataFrame with insurance provider data
        """
        # Real-world insurance companies
        provider_names = [
            "Blue Cross Blue Shield",
            "UnitedHealthcare",
            "Aetna",
            "Cigna",
            "Humana",
            "Kaiser Permanente",
            "Anthem",
            "Centene"
        ][:n_providers]

        providers = []
        for i, name in enumerate(provider_names, 1):
            contact_info = {
                "phone": self.fake.phone_number(),
                "email": f"customer.service@{name.lower().replace(' ', '')}.com",
                "website": f"www.{name.lower().replace(' ', '')}.com"
            }

            providers.append({
                'insurance_id': i,
                'insurance_name': name,
                'contact_info': json.dumps(contact_info)
            })

        return pd.DataFrame(providers)

    def generate_plans(self, insurance_df: pd.DataFrame, plans_per_provider: int = 5) -> pd.DataFrame:
        """
        Generate insurance plan data.

        Args:
            insurance_df: DataFrame of insurance providers
            plans_per_provider: Number of plans per provider (default: 5)

        Returns:
            DataFrame with insurance plan data
        """
        plan_tiers = [
            {'name': 'Catastrophic', 'base_rate': 250.00, 'deductable': 8000},
            {'name': 'Bronze', 'base_rate': 350.00, 'deductable': 6000},
            {'name': 'Silver', 'base_rate': 450.00, 'deductable': 4000},
            {'name': 'Gold', 'base_rate': 550.00, 'deductable': 2000},
            {'name': 'Platinum', 'base_rate': 650.00, 'deductable': 1000}
        ]

        plans = []
        plan_id = 1

        for _, insurance in insurance_df.iterrows():
            for tier in plan_tiers[:plans_per_provider]:
                # Add some variation to rates
                rate_variation = self.rng.uniform(0.9, 1.1)
                base_rate = round(tier['base_rate'] * rate_variation, 2)

                plans.append({
                    'plan_id': plan_id,
                    'plan_name': f"{tier['name']} Plan",
                    'insurance_id': insurance['insurance_id'],
                    'base_rate': base_rate,
                    'deductable': tier['deductable']
                })
                plan_id += 1

        return pd.DataFrame(plans)

    def generate_facilities(
        self,
        n_facilities: int,
        state_populations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate medical facility data proportional to state populations.

        Args:
            n_facilities: Total number of facilities to generate
            state_populations_df: DataFrame with state codes and proportions

        Returns:
            DataFrame with facility data
        """
        facilities = []

        # Normalize proportions to ensure they sum to 1.0
        proportions = state_populations_df['proportion'].values
        proportions = proportions / proportions.sum()

        # Sample states proportional to population
        states = self.rng.choice(
            state_populations_df['state_code'].values,
            size=n_facilities,
            p=proportions
        )

        for i in range(1, n_facilities + 1):
            state = states[i - 1]

            # Generate realistic NPI (10-digit number)
            npi = self.rng.integers(1000000000, 9999999999)

            # Generate facility name
            facility_types = [
                "Medical Center", "Clinic", "Hospital", "Health Center",
                "Family Practice", "Care Center", "Regional Hospital"
            ]
            facility_type = self.rng.choice(facility_types)
            city = self.fake.city()
            name = f"{city} {facility_type}"

            # Contact info
            contact_info = {
                "phone": self.fake.phone_number(),
                "address": self.fake.street_address(),
                "city": city,
                "state": state,
                "zip": self.fake.zipcode()
            }

            facilities.append({
                'Facility_ID': i,
                'NPI': str(npi),
                'Name': name,
                'State': state,
                'Contact_info': json.dumps(contact_info)
            })

        return pd.DataFrame(facilities)

    def load_conditions(self, conditions_csv_path: str) -> pd.DataFrame:
        """
        Load medical conditions from CSV file.

        Args:
            conditions_csv_path: Path to conditions.csv

        Returns:
            DataFrame with condition data
        """
        df = pd.read_csv(conditions_csv_path)

        # Keep only Condition_ID and Condition_name for database
        return df[['condition_id', 'condition_name']].rename(columns={
            'condition_id': 'Condition_ID',
            'condition_name': 'Condition_name'
        })


def generate_all_reference_data(
    config: dict,
    state_populations_path: str,
    conditions_path: str
) -> Dict[str, pd.DataFrame]:
    """
    Generate all reference data tables.

    Args:
        config: Configuration dictionary
        state_populations_path: Path to state populations CSV
        conditions_path: Path to conditions CSV

    Returns:
        Dictionary of DataFrames for each reference table
    """
    n_members = config.get('n_members', 100000)
    n_facilities = max(int(n_members * 0.002), 50)  # 0.2% of members, min 50

    generator = ReferenceDataGenerator(config.get('random_seed'))

    # Load state populations
    state_pops = pd.read_csv(state_populations_path)

    # Generate data
    insurance_df = generator.generate_insurance_providers(n_providers=8)
    plan_df = generator.generate_plans(insurance_df, plans_per_provider=5)
    facility_df = generator.generate_facilities(n_facilities, state_pops)
    condition_df = generator.load_conditions(conditions_path)

    return {
        'INSURANCE': insurance_df,
        'PLAN': plan_df,
        'FACILITY': facility_df,
        'CONDITION': condition_df,
        '_state_populations': state_pops  # Keep for member generation
    }
