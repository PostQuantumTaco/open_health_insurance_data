# Synthetic Health Insurance Data Generator

A Python-based system for generating realistic synthetic health insurance data at scale. Produces datasets with 100K-1M members complete with demographics, biometrics, health conditions, insurance enrollments, and medical claims.

## Features

- **Evidence-based distributions**: Demographics from US Census, biometrics from CDC NHANES, conditions from clinical literature
- **Clinical realism**: Age-stratified condition prevalence, correlated biometrics, ACA-compliant premiums
- **Data integrity**: All foreign keys validated, temporal consistency enforced
- **Progress tracking**: Real-time progress bars with time estimates
- **Configurable**: YAML-based configuration for easy customization
- **MySQL-ready**: Generates CSV files and SQL loader scripts

## Project Structure

```
open_health_insurance_data/
├── config/
│   ├── config.yaml                 # Runtime configuration
│   ├── distributions.yaml          # Statistical distributions
│   └── reference_data/
│       ├── state_populations.csv   # US state populations
│       └── conditions.csv          # Medical conditions with ICD-10 codes
├── generators/
│   ├── reference_data.py           # Insurance, plans, facilities, conditions
│   ├── members.py                  # Demographics, biometrics, lifestyle
│   ├── enrollment.py               # Insurance assignments and premiums
│   └── claims.py                   # Medical claims
├── utils/
│   ├── config_loader.py            # Configuration management
│   ├── progress.py                 # Progress tracking with ETAs
│   ├── csv_writer.py               # CSV export utilities
│   └── validators.py               # Data quality validation
├── generate_data.py                # Main generation script
├── validate_output.py              # Post-generation validation
├── health_insurance_final.sql      # Database schema
└── data_generation_spec.md         # Complete specification
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data

```bash
# Generate 100K members (default, ~10-15 min)
python generate_data.py

# Or with custom config
python generate_data.py --config config/my_config.yaml
```

### 3. Validate Output

```bash
# Run validation checks on generated CSVs
python validate_output.py

# Or specify custom directory
python validate_output.py --data-dir my_synthetic_data
```

### 4. Load into MySQL

```bash
# First, create the database schema
mysql -u <username> -p < health_insurance_final.sql

# Then load the data
cd synthetic_data
mysql -u <username> -p --local-infile=1 health_insurance < load_data.sql
```

## Configuration

### Runtime Parameters (config/config.yaml)

```yaml
# Scale
n_members: 100000           # Number of members to generate
batch_size: 25000           # Batch size for memory efficiency

# Temporal
reference_date: '2024-01-01'
claims_lookback_years: 1    # Generate 1 year of claims history

# Reproducibility
random_seed: 42             # Fixed seed for reproducible results

# Output
output_dir: 'synthetic_data'
```

### Distribution Parameters (config/distributions.yaml)

Controls statistical distributions for:
- **Demographics**: Age groups, sex ratio, state distribution
- **Biometrics**: Height/weight (correlated), vital signs
- **Lifestyle**: Smoking, alcohol, exercise, sleep patterns
- **Health Conditions**: Age-stratified prevalence rates
- **Insurance**: Plan preferences by age, premium calculation
- **Claims**: Frequency (Poisson) and amounts (log-normal)

See `data_generation_spec.md` for complete parameter documentation.

## Generated Data

### Output Files

The generator produces 8 CSV files:

| File | Rows (100K members) | Description |
|------|---------------------|-------------|
| `insurance.csv` | 8 | Insurance providers |
| `plan.csv` | 40 | Insurance plans (5 per provider) |
| `facility.csv` | ~200 | Medical facilities |
| `condition.csv` | 50 | Medical conditions with ICD-10 |
| `members.csv` | 100,000 | Member demographics and biometrics |
| `enrollment.csv` | ~90,000 | Insurance enrollments (90% covered) |
| `member_condition.csv` | ~150,000 | Member-condition associations |
| `claims.csv` | ~400,000 | Medical claims (1 year history) |

### Data Characteristics

**Demographics**
- Age distribution matches US Census 2023 (8 age groups)
- Sex: 50.5% F, 49.5% M, 0.1% Other
- State distribution proportional to US populations

**Biometrics**
- Height/weight: Correlated (r=0.60 male, r=0.58 female)
- Heart rate: Age-adjusted (40-120 bpm)
- Blood pressure: Age-adjusted, correlated systolic/diastolic
- Blood oxygen: 88-100%

**Health Conditions**
- Age-stratified prevalence (e.g., 70% elderly have hypertension)
- Average 0.3-4.5 conditions per member by age
- Top conditions: Hypertension, diabetes, hyperlipidemia, asthma

**Insurance**
- 90% coverage rate
- ACA-compliant 3:1 age rating
- State cost adjustments
- 70% active enrollments

**Claims**
- Poisson frequency: 2-9 claims/year by age
- Log-normal amounts: $100-$75,000 by type
- Temporal consistency: claims only during enrollment

## Customization

### Adjust Dataset Size

Edit `config/config.yaml`:

```yaml
n_members: 500000    # Generate 500K members instead
```

Dependent counts scale automatically:
- Facilities: 0.2% of members
- Claims: ~4 per member per year

### Modify Condition Prevalence

Edit `config/distributions.yaml`:

```yaml
condition_prevalence:
  conditions:
    Diabetes_Type_2:
      '18-34': 0.05    # Increase from 0.03 to 0.05
      '35-44': 0.12    # Increase from 0.08 to 0.12
      # ...
```

### Change Geographic Distribution

Override state weights in `config/distributions.yaml` or provide custom `state_populations.csv`.

## Progress Tracking

The generator shows real-time progress with ETAs:

```
================================================================================
Starting: Generate Members (Demographics, Biometrics, Lifestyle)
================================================================================
  ✓ Generated 100,000 members
  ✓ Generated 152,341 member-condition associations

✓ Completed: Generate Members (Demographics, Biometrics, Lifestyle) (2m 15s)
```

Time estimates:
- **100K members**: ~10-15 minutes
- **500K members**: ~30-60 minutes
- **1M members**: ~1-2 hours

## Validation

The generator includes automatic validation:

- **Foreign key integrity**: All references resolve correctly
- **Data quality**: Blood pressure format, state codes, vital sign ranges
- **Business rules**: End dates after start dates, claims during enrollment
- **Statistical checks**: Row counts, NULL rates, distributions

Run standalone validation:

```bash
python validate_output.py --data-dir synthetic_data
```

## Database Schema

The schema models:
- **INSURANCE** → **PLAN**: Insurance providers and their plan offerings
- **FACILITY**: Medical facilities with NPI identifiers
- **CONDITION**: Medical conditions with ICD-10 codes
- **MEMBERS**: Insured individuals with demographics, biometrics, and SDOH
- **ENROLLMENT**: Insurance coverage periods with premiums
- **MEMBER_CONDITION**: Many-to-many member-condition relationships
- **CLAIMS**: Medical claims linked to members and insurers

See `health_insurance_final.sql` for complete schema.

## Troubleshooting

### Out of Memory

Reduce batch size in `config/config.yaml`:

```yaml
batch_size: 10000  # Reduce from default 25000
```

### Foreign Key Violations

Run validation to identify issues:

```bash
python validate_output.py
```

Check that all reference data tables (insurance, plans, facilities, conditions) generated successfully.

### MySQL Load Errors

Enable local file loading:

```bash
mysql -u <username> -p --local-infile=1 < load_data.sql
```

If still failing, check MySQL `secure_file_priv` setting.

## Technical Details

### Dependencies

- **numpy, pandas**: Data manipulation
- **scipy**: Statistical distributions (multivariate normal, log-normal)
- **faker**: Realistic names and contact information
- **pyyaml**: Configuration file parsing
- **tqdm**: Progress bars with ETA

### Reproducibility

Set `random_seed` in `config/config.yaml` for deterministic generation:

```yaml
random_seed: 42  # Same seed = identical data every time
```

### Performance Optimization

For large-scale generation (1M+ members):

1. Increase batch size (if memory allows): `batch_size: 50000`
2. Reduce claims history: `claims_lookback_years: 1`
3. Use SSD storage for output directory

## Documentation

- **`data_generation_spec.md`**: Complete technical specification with all distributions and parameters
- **`CLAUDE.md`**: Development guide for extending the generator
- **`health_insurance_final.sql`**: Database schema documentation

## License

This is an educational project for synthetic data generation. Generated data is entirely fictional.

## Support

For issues or questions:
1. Check validation output: `python validate_output.py`
2. Review configuration: `config/config.yaml` and `config/distributions.yaml`
3. Consult specification: `data_generation_spec.md`
