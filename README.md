# Synthetic Health Insurance Data Generator

A Python-based system for generating realistic synthetic health insurance data at scale. Proven on 100K–5M member cohorts, it produces demographics, biometrics, lifestyle/SDOH factors, 60 medical conditions, insurance enrollments, and medical claims ready for analytics or database loading.

## Features

- **Evidence-based distributions**: Demographics from US Census, biometrics from CDC NHANES, 60 CDC/clinical conditions with age curves
- **Clinical realism**: Risk-based multimorbidity cascades, correlated biometrics, ACA-compliant premiums
- **Lifestyle correlations**: Gaussian copula enforces smoker/drinker/exercise/sleep/housing/employment relationships from spec §8.2
- **Statistical validation**: Kolmogorov–Smirnov, chi-square, correlation, and prevalence tests via `validate_output.py --stats`
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
│   ├── validators.py               # Structural/business validation
│   └── statistical_validator.py    # Statistical validation suite
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

# Or with custom config / larger scale
python generate_data.py --config config/my_config.yaml
# Example: 5M members completed in ~14 minutes on Apple Silicon Ultra
```

### 3. Validate Output

```bash
# Run structural/business validation on generated CSVs
python validate_output.py --data-dir synthetic_data

# Run structural + statistical validation suite
python validate_output.py --stats --config config/config.yaml \
    --data-dir synthetic_data --stats-sample-size 250000
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
- **Biometrics**: Sex-stratified multivariate normals and vital signs
- **Lifestyle & SDOH**: Age-specific rates plus 6×6 correlation matrix (smoker/drinker/exercise/sleep/housing/employment)
- **Health Conditions**: Age-stratified prevalence rates, risk multipliers, comorbidity rules
- **Insurance**: Plan preferences by age, premium calculation
- **Claims**: Frequency (Poisson) and amounts (log-normal) with condition multipliers

See `data_generation_spec.md` for complete parameter documentation.

## Generated Data

### Output Files

The generator produces 8 CSV files:

| File | Typical Rows (per 1M members) | Description |
|------|------------------------------|-------------|
| `insurance.csv` | 8 | Insurance providers |
| `plan.csv` | 40 | Insurance plans (5 per provider) |
| `facility.csv` | max(200, 0.2% of members) | Medical facilities sampled by state |
| `condition.csv` | 60 | Medical conditions with ICD-10 |
| `members.csv` | n_members | Member demographics, biometrics, lifestyle, SDOH |
| `enrollment.csv` | ~0.9 × n_members | Insurance enrollments (90% coverage target) |
| `member_condition.csv` | ~1.9 × n_members | Member-condition associations with diagnosis dates |
| `claims.csv` | ~4 × n_members × claims_lookback_years | Medical claims (lookback limited) |

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
- 60 chronic, acute, oncology, autoimmune, infectious, and mental health conditions
- Age-stratified prevalence (0.3–4.5 conditions per member depending on age)
- Risk multipliers for obesity, smoking, heavy drinking, housing insecurity
- Comorbidity cascades across metabolic, cardiovascular, renal, oncology, and mental health clusters

**Insurance**
- 90% coverage rate
- ACA-compliant 3:1 age rating
- State cost adjustments
- 70% active enrollments

**Claims**
- Poisson frequency: 2-9 claims/year by age (plus condition multipliers)
- Log-normal amounts: $100-$75,000 by type
- Temporal consistency: claims only during enrollment windows

**Lifestyle & SDOH**
- Smoker/drinker correlation r=0.30, smoker-exercise r=-0.15
- Exercise-sleep correlation r=0.25, housing insecurity-employment r=-0.50
- Gaussian copula enforces the §8.2 correlation matrix while preserving age-specific marginals

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

Time estimates (Apple Silicon Ultra, SSD output):
- **100K members**: ~10-15 minutes
- **500K members**: ~30-40 minutes
- **1M members**: ~60 minutes
- **5M members**: ~14 minutes (see sample run below)

## Validation

### Structural & Business Rules (default)

```
python validate_output.py --data-dir synthetic_data
```

- Foreign key integrity (members→plans, claims→members/insurers, etc.)
- Data quality (blood pressure format, vital ranges, state codes)
- Business logic (enrollment windows, coverage vs claims, null allowances)
- Clinical realism heuristics (risk multipliers, comorbidity patterns, vital deltas)

### Statistical Validation Suite (Tier 2)

```
python validate_output.py --stats --config config/config.yaml \
    --data-dir synthetic_data --stats-sample-size 250000
```

- Kolmogorov–Smirnov tests for biometrics, exercise (log-space), sleep
- Chi-square tests for categorical distributions (sex, etc.)
- Correlation checks (height-weight by sex, smoker/drinker, exercise-sleep, housing-employment, smoker-exercise)
- Age-stratified prevalence checks for key conditions (hypertension, diabetes, COPD, depression, CKD, breast cancer)
- Summaries printed in the “Statistical Validation Report” section with pass/fail and tolerances

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

## Sample Large-Scale Run (5M Members)

```
python generate_data.py --config config/config.yaml
...
✓ DATA GENERATION COMPLETE!

Members: 5,000,000
Enrollments: 4,499,910 (90.0% coverage / 70% active)
Member conditions: 9,710,863 (avg 3.06 per affected member)
Claims: 13,179,693 (avg $3,902.76, total $51.4B)
Total runtime: 14m 8s (Apple Silicon Ultra, SSD output)
```

Validated with:

```
python validate_output.py --data-dir synthetic_data
python validate_output.py --stats --config config/config.yaml \
    --data-dir synthetic_data --stats-sample-size 250000
```

Both structural and statistical checks passed without warnings.

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
