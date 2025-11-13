# Synthetic Health Insurance Data Generation Specification

**Version:** 1.0
**Target Scale:** 100K - 1M members
**Purpose:** Generate realistic synthetic health insurance data for analytics and testing

---

## 1. Reference Data Specifications

### 1.1 Insurance Providers

**Count:** 8-10 major providers

**Sample Providers:**
| Insurance ID | Insurance Name | Contact Info (JSON) |
|--------------|----------------|---------------------|
| 1 | Blue Cross Blue Shield | `{"phone": "1-800-xxx-xxxx", "email": "contact@bcbs.com"}` |
| 2 | UnitedHealthcare | `{"phone": "1-888-xxx-xxxx", "email": "service@uhc.com"}` |
| 3 | Aetna | `{"phone": "1-800-xxx-xxxx", "email": "help@aetna.com"}` |
| 4 | Cigna | `{"phone": "1-800-xxx-xxxx", "email": "support@cigna.com"}` |
| 5 | Humana | `{"phone": "1-800-xxx-xxxx", "email": "care@humana.com"}` |
| 6 | Kaiser Permanente | `{"phone": "1-800-xxx-xxxx", "email": "info@kp.org"}` |
| 7 | Anthem | `{"phone": "1-800-xxx-xxxx", "email": "help@anthem.com"}` |
| 8 | Centene | `{"phone": "1-800-xxx-xxxx", "email": "contact@centene.com"}` |

**Generation Rules:**
- Contact info: JSON with "phone", "email", "website" fields
- IDs: Sequential starting from 1

### 1.2 Insurance Plans

**Plans per Provider:** 4-6 plans (representing metal tiers + specialized plans)

**Plan Structure:**
| Tier | Base Rate (Annual) | Deductible | Premium Multiplier |
|------|-------------------|------------|-------------------|
| Catastrophic | $2,400 - $3,600 | $7,000 - $9,000 | 0.5 |
| Bronze | $4,800 - $6,000 | $6,000 - $7,500 | 0.75 |
| Silver | $6,000 - $7,200 | $4,000 - $5,000 | 1.0 |
| Gold | $7,200 - $9,600 | $2,000 - $3,000 | 1.3 |
| Platinum | $9,600 - $12,000 | $500 - $1,500 | 1.5 |
| Medicare Advantage | $0 - $2,400 | $0 - $3,000 | 0.3 |

**Plan Naming Convention:** `{InsuranceName} {Tier} {Variant}`
- Example: "Blue Cross Blue Shield Silver PPO", "Aetna Gold HMO"

**Generation Rules:**
- Plan IDs: Sequential starting from 1
- Unique constraint: (insurance_id, name) pairs must be unique
- Base rates and deductibles: Uniform random within tier ranges

### 1.3 Medical Facilities

**Total Count:** 500 - 2,000 facilities (scales with member count)

**Geographic Distribution:** Proportional to state population
- California: ~12% of facilities
- Texas: ~9%
- Florida: ~6.5%
- New York: ~6%
- etc.

**NPI Generation:**
- 10-digit numeric string
- Format: Valid NPI structure (real NPIs follow Luhn algorithm, but we'll use random for simplicity)
- 90% of facilities have NPI, 10% NULL

**Facility Names:**
- Pattern: `{City} {Type} {Specialty/Modifier}`
- Types: Medical Center, Hospital, Clinic, Health Center, Family Practice, Urgent Care
- Examples: "Springfield General Hospital", "Boston Family Clinic", "Austin Urgent Care"

**Contact Info (JSON):**
```json
{
  "phone": "xxx-xxx-xxxx",
  "fax": "xxx-xxx-xxxx",
  "address": "123 Main St, City, ST 12345"
}
```

### 1.4 Medical Conditions

**Total Count:** 50-100 most common conditions

**Top Conditions with ICD-10 Codes:**
| Condition ID | Condition Name | ICD-10 | Age-Stratified Prevalence | Category |
|--------------|----------------|--------|---------------------------|----------|
| 1 | Essential Hypertension | I10 | Age-based (see §5.1) | Cardiovascular |
| 2 | Type 2 Diabetes Mellitus | E11 | Age-based | Metabolic |
| 3 | Hyperlipidemia | E78.5 | Age-based | Metabolic |
| 4 | Asthma | J45 | 7.7% constant | Respiratory |
| 5 | Depression | F33 | 8.4% (higher 18-49) | Mental Health |
| 6 | Anxiety Disorder | F41 | 12% (higher 18-49) | Mental Health |
| 7 | COPD | J44 | Age-based (1% <45, 15% >65) | Respiratory |
| 8 | Osteoarthritis | M19 | Age-based (3% <45, 50% >65) | Musculoskeletal |
| 9 | Coronary Artery Disease | I25 | Age-based (rare <45, 15% >65) | Cardiovascular |
| 10 | Chronic Kidney Disease | N18 | Age-based | Renal |
| 11 | Obesity | E66 | 42% adults | Metabolic |
| 12 | Gastroesophageal Reflux | K21 | 20% adults | Gastrointestinal |
| 13 | Hypothyroidism | E03 | 5% (higher in women) | Endocrine |
| 14 | Sleep Apnea | G47.3 | 10% (correlated with obesity) | Respiratory |
| 15 | Migraine | G43 | 15% (higher in women) | Neurological |
| 16 | Atrial Fibrillation | I48 | Age-based (<1% <50, 10% >80) | Cardiovascular |
| 17 | Heart Failure | I50 | Age-based | Cardiovascular |
| 18 | Rheumatoid Arthritis | M06 | 1% | Autoimmune |
| 19 | Psoriasis | L40 | 3% | Dermatological |
| 20 | Allergic Rhinitis | J30 | 20% | Respiratory |

**Additional 30-80 conditions:** Include common acute and chronic conditions
- Acute: Upper respiratory infections, urinary tract infections, gastroenteritis
- Chronic: Various cancers, autoimmune diseases, neurological conditions

**Generation Rules:**
- IDs: Sequential starting from 1
- Names: Unique medical terms
- Include both common and rare conditions for realism

---

## 2. Demographic Distributions

### 2.1 Age Distribution

**Source:** US Census Bureau 2023 population estimates

**Age Group Distribution:**
| Age Group | Percentage | Mean Age | Std Dev |
|-----------|------------|----------|---------|
| 0-17 | 22.0% | 9 | 5.2 |
| 18-24 | 9.5% | 21 | 2.0 |
| 25-34 | 13.8% | 29.5 | 2.9 |
| 35-44 | 12.8% | 39.5 | 2.9 |
| 45-54 | 12.5% | 49.5 | 2.9 |
| 55-64 | 13.2% | 59.5 | 2.9 |
| 65-74 | 10.0% | 69.5 | 2.9 |
| 75+ | 6.2% | 82 | 5.0 |

**Generation Method:**
```python
# Sample age group proportionally
age_group = np.random.choice(age_groups, p=age_probs)
# Sample specific age within group using normal distribution (truncated)
age = np.random.normal(mean_for_group, std_for_group)
age = np.clip(age, min_age_for_group, max_age_for_group)
```

**Date of Birth Calculation:**
```python
reference_date = '2024-01-01'  # Adjust to desired reference
DOB = reference_date - age_in_years
# Add random day offset within year for variety
```

### 2.2 Sex Distribution

**Distribution:**
- Female (F): 50.5%
- Male (M): 49.5%
- Other (O): 0.1%

**Generation Method:**
```python
sex = np.random.choice(['F', 'M', 'O'], p=[0.505, 0.495, 0.001])
```

### 2.3 State Distribution

**Source:** US Census Bureau 2023 state populations

**Top 20 States (covers ~75% of US population):**
| State | Percentage | Approx. Members (for 1M total) |
|-------|------------|--------------------------------|
| CA | 11.87% | 118,700 |
| TX | 9.28% | 92,800 |
| FL | 6.94% | 69,400 |
| NY | 5.85% | 58,500 |
| PA | 3.84% | 38,400 |
| IL | 3.78% | 37,800 |
| OH | 3.51% | 35,100 |
| GA | 3.37% | 33,700 |
| NC | 3.34% | 33,400 |
| MI | 3.00% | 30,000 |
| NJ | 2.75% | 27,500 |
| VA | 2.59% | 25,900 |
| WA | 2.44% | 24,400 |
| AZ | 2.35% | 23,500 |
| MA | 2.11% | 21,100 |
| TN | 2.20% | 22,000 |
| IN | 2.05% | 20,500 |
| MD | 1.85% | 18,500 |
| MO | 1.84% | 18,400 |
| WI | 1.77% | 17,700 |
| (other) | ~24.27% | ~242,700 |

**Generation Method:**
```python
state = np.random.choice(state_codes, p=state_probabilities)
```

### 2.4 Primary Care Facility Assignment

**Assignment Rate:** 70% of members have an assigned PCP
- 30% have NULL (no assigned primary care)

**Assignment Logic:**
- Facility must be in the same state as member
- Random selection within state's facilities

---

## 3. Biometric Specifications

### 3.1 Height and Weight (Correlated)

**Method:** Multivariate normal distribution by sex

#### Male Biometrics
**Height:**
- Mean: 69.1 inches (175.5 cm)
- Std Dev: 2.9 inches (7.4 cm)
- Range: [60, 80] inches (truncate outliers)

**Weight:**
- Mean: 199.8 lbs (90.6 kg)
- Std Dev: 40.0 lbs (18.1 kg)
- Range: [100, 400] lbs (truncate outliers)

**Correlation Matrix:**
```
        Height  Weight
Height   1.00    0.60
Weight   0.60    1.00
```

**Covariance Matrix:**
```
        Height      Weight
Height  8.41        69.6
Weight  69.6        1600
```

#### Female Biometrics
**Height:**
- Mean: 63.7 inches (161.8 cm)
- Std Dev: 2.7 inches (6.9 cm)
- Range: [56, 76] inches (truncate outliers)

**Weight:**
- Mean: 170.8 lbs (77.5 kg)
- Std Dev: 38.0 lbs (17.2 kg)
- Range: [90, 350] lbs (truncate outliers)

**Correlation Matrix:**
```
        Height  Weight
Height   1.00    0.58
Weight   0.58    1.00
```

**Covariance Matrix:**
```
        Height      Weight
Height  7.29        59.5
Weight  59.5        1444
```

#### Other/Unspecified Sex
- Use weighted average of male/female distributions (50/50 mix)

#### Age Adjustments
**Weight increases with age (obesity trends):**
- Ages 18-34: baseline
- Ages 35-54: +5% weight
- Ages 55-64: +8% weight
- Ages 65+: +3% weight (some weight loss in elderly)

**Height decreases slightly with age (bone density, posture):**
- Ages <30: baseline
- Ages 30-60: -0.02 inches/year
- Ages 60+: -0.05 inches/year

**Null Handling:**
- 5% of members have NULL weight (missing data)
- 5% of members have NULL height (missing data)
- Use independent random sampling for nulls (not correlated)

### 3.2 Heart Rate

**Base Distribution:**
- Distribution: Normal
- Mean: 72 bpm
- Std Dev: 12 bpm
- Range: [20, 300] per CHECK constraint (practical range: 50-100 for healthy adults)

**Age Adjustments:**
- Ages 0-17: μ=80, σ=15 (children have faster heart rates)
- Ages 18-64: μ=72, σ=12
- Ages 65+: μ=75, σ=13 (slight increase)

**Condition Adjustments (additive):**
- Anxiety: +8 bpm (±4)
- Heart Failure: +10 bpm (±5)
- Hyperthyroidism: +15 bpm (±6)
- COPD: +8 bpm (±4)
- Obesity: +5 bpm (±3)

**Lifestyle Adjustments:**
- Regular exercise (>150 min/week): -5 bpm
- Smoker: +3 bpm

**Null Rate:** 10% (not all members have recent vitals)

### 3.3 Blood Pressure

**Format:** "systolic/diastolic" (e.g., "120/80")

**Systolic Blood Pressure:**
- Base Mean: 120 mmHg
- Base Std Dev: 15 mmHg
- Range: [90, 250]

**Diastolic Blood Pressure:**
- Correlated with systolic: DBP ≈ 0.6 × SBP + 20
- Additional noise: Normal(0, 5)
- Range: [60, 150]

**Age Adjustments:**
- Ages <30: μ_sys=115
- Ages 30-44: μ_sys=120
- Ages 45-54: μ_sys=125
- Ages 55-64: μ_sys=130
- Ages 65+: μ_sys=135

**Condition Adjustments:**
- Hypertension (diagnosed): +20 mmHg systolic (±10), +10 diastolic (±5)
- Diabetes: +8 mmHg systolic (±5)
- Obesity: +10 mmHg systolic (±5)
- Chronic Kidney Disease: +12 mmHg systolic (±6)

**Lifestyle Adjustments:**
- Smoker: +5 mmHg systolic
- Regular exercise: -5 mmHg systolic
- High alcohol consumption: +8 mmHg systolic

**Generation Logic:**
```python
systolic = base_systolic + age_effect + condition_effects + lifestyle_effects
systolic = np.clip(systolic, 90, 250)

diastolic = 0.6 * systolic + 20 + np.random.normal(0, 5)
diastolic = np.clip(diastolic, 60, 150)

blood_pressure = f"{int(systolic)}/{int(diastolic)}"
```

**Null Rate:** 10%

### 3.4 Blood Oxygen (SpO2)

**Base Distribution:**
- Distribution: Normal (left-truncated)
- Mean: 97%
- Std Dev: 2%
- Range: [0, 100] per CHECK constraint (practical healthy range: 95-100%)

**Age Adjustments:**
- Ages <65: μ=97%, σ=2%
- Ages 65+: μ=96%, σ=2.5% (slight decrease)

**Condition Adjustments:**
- COPD: -5% (±2)
- Asthma: -2% (±1.5)
- Sleep Apnea: -3% (±2)
- Heart Failure: -3% (±1.5)
- Pneumonia/Respiratory Infections: -4% (±2)

**Lifestyle Adjustments:**
- Smoker: -2%
- High altitude states (CO, WY, UT, NM): -1%

**Distribution Characteristics:**
- Left-skewed: Most people cluster at 97-100%
- Use Beta distribution transformed to [88, 100] range for more realism
  ```python
  # Beta(α=15, β=2) transformed to [88, 100]
  raw = np.random.beta(15, 2, size=n)
  spo2 = 88 + raw * 12
  ```

**Null Rate:** 15% (less commonly measured than BP/HR)

### 3.5 Derived Metric: BMI

**Not stored but used for validation and condition assignment**

**Calculation:**
```python
BMI = weight_lbs / (height_inches ** 2) * 703
```

**Categories:**
- Underweight: BMI < 18.5
- Normal: 18.5 ≤ BMI < 25
- Overweight: 25 ≤ BMI < 30
- Obese: BMI ≥ 30

**Target Distribution (US adult population):**
- Underweight: 1.5%
- Normal: 30.5%
- Overweight: 31.0%
- Obese: 42.0%

**Use:** BMI category influences condition assignment (obesity, diabetes, hypertension, sleep apnea)

---

## 4. Lifestyle & Social Determinants of Health

### 4.1 Smoking Status

**Overall Prevalence:** 12.5% of adults smoke (CDC 2022)

**Age-Stratified Prevalence:**
| Age Group | Smoking Rate |
|-----------|--------------|
| 0-17 | 0% |
| 18-24 | 11.0% |
| 25-44 | 16.0% |
| 45-64 | 15.5% |
| 65+ | 8.5% |

**Sex Differences:**
- Male: 1.3× female rate
- Female: baseline
- Other: average of M/F

**Correlations:**
- Negatively correlated with education/income proxies
- Positively correlated with alcohol use (r=0.3)

**Null Rate:** 5% (missing data)

### 4.2 Alcohol Consumption (Drinker)

**Definition:** Heavy/binge drinking (>7 drinks/week women, >14 men)

**Overall Prevalence:** 16% are heavy drinkers

**Age-Stratified Prevalence:**
| Age Group | Heavy Drinking Rate |
|-----------|---------------------|
| 0-17 | 0% |
| 18-24 | 28% |
| 25-44 | 20% |
| 45-64 | 14% |
| 65+ | 8% |

**Sex Differences:**
- Male: 1.6× female rate
- Female: baseline

**Correlations:**
- Positively correlated with smoking (r=0.3)
- Negatively correlated with certain health conditions

**Null Rate:** 5%

### 4.3 Exercise (Minutes per Week)

**Target Distribution:** Log-normal (most people exercise little, few exercise a lot)

**Parameters:**
- Median: 60 minutes/week
- Mean: 100 minutes/week
- 28% meet CDC guidelines (≥150 min/week)

**Log-Normal Parameters:**
```python
mu = 4.0  # log-space mean
sigma = 1.2  # log-space std dev
minutes = np.random.lognormal(mu, sigma)
minutes = np.clip(minutes, 0, 10080)  # max = minutes in a week
```

**Age Effects:**
- Ages 18-34: μ=4.2 (more active)
- Ages 35-54: μ=4.0
- Ages 55-64: μ=3.8
- Ages 65+: μ=3.5 (less active)
- Ages 0-17: μ=4.5 (school PE, sports)

**Condition Effects:**
- Heart disease, COPD, obesity → reduce by 30%
- Depression → reduce by 20%

**Null Rate:** 8%

### 4.4 Sleep (Hours per Day)

**Distribution:** Normal with slight left skew (chronic sleep deprivation common)

**Parameters:**
- Mean: 6.8 hours
- Std Dev: 1.2 hours
- Range: [0, 24] per CHECK constraint (practical: 4-11)

**Age Effects:**
| Age Group | Mean Sleep | Std Dev |
|-----------|------------|---------|
| 0-17 | 9.0 | 1.5 |
| 18-24 | 7.5 | 1.5 |
| 25-64 | 6.8 | 1.2 |
| 65+ | 7.2 | 1.3 |

**Condition Effects:**
- Sleep Apnea: -0.5 hours (disrupted sleep)
- Depression/Anxiety: -0.8 hours
- Chronic pain conditions: -0.6 hours

**Lifestyle Effects:**
- Employment status=1 (employed): -0.3 hours
- High exercise (>200 min/week): +0.4 hours

**Null Rate:** 8%

### 4.5 Housing Insecurity

**Definition:** Boolean indicating unstable housing situation

**Overall Prevalence:** 6.5%

**Correlations:**
- Strongly correlated with employment status (r=-0.5)
- Correlated with age (higher in young adults 18-29: 10%)
- Inversely correlated with insurance coverage

**Age-Stratified Prevalence:**
| Age Group | Rate |
|-----------|------|
| 0-17 | 8% (family-level) |
| 18-29 | 10% |
| 30-44 | 7% |
| 45-64 | 5% |
| 65+ | 3% (social safety nets) |

**Impact on Health:**
- Higher risk of: depression, anxiety, COPD, uncontrolled diabetes
- Lower insurance coverage rate (75% vs 92%)

**Null Rate:** 3%

### 4.6 Employment Status

**Definition:** Boolean (1=employed, 0=unemployed/not in workforce)

**Age-Stratified Employment:**
| Age Group | Employment Rate |
|-----------|-----------------|
| 0-17 | 0% (students) |
| 18-24 | 65% |
| 25-34 | 85% |
| 35-44 | 88% |
| 45-54 | 85% |
| 55-64 | 70% (early retirement) |
| 65+ | 20% (retirement) |

**Sex Differences:**
- Male: slightly higher (1.05×)
- Female: baseline (accounts for caregiving, etc.)

**Health Impacts:**
- Employed → higher insurance coverage (95% vs 75%)
- Unemployed → higher stress-related conditions
- Retired (age 65+) → Medicare eligibility

**Null Rate:** 2%

---

## 5. Health Condition Framework

### 5.1 Age-Stratified Condition Prevalence

**Prevalence Tables by Age Group:**

#### Cardiovascular Conditions
| Condition | 0-17 | 18-34 | 35-44 | 45-54 | 55-64 | 65-74 | 75+ |
|-----------|------|-------|-------|-------|-------|-------|-----|
| Hypertension | 0.5% | 5% | 15% | 30% | 50% | 65% | 75% |
| Hyperlipidemia | 0.1% | 8% | 20% | 35% | 45% | 55% | 60% |
| Coronary Artery Disease | 0% | 0.1% | 1% | 5% | 12% | 20% | 25% |
| Heart Failure | 0% | 0.1% | 0.5% | 2% | 5% | 10% | 15% |
| Atrial Fibrillation | 0% | 0.1% | 0.5% | 2% | 5% | 10% | 15% |

#### Metabolic Conditions
| Condition | 0-17 | 18-34 | 35-44 | 45-54 | 55-64 | 65-74 | 75+ |
|-----------|------|-------|-------|-------|-------|-------|-----|
| Type 2 Diabetes | 0.5% | 3% | 8% | 15% | 22% | 27% | 25% |
| Obesity (BMI ≥30) | 20% | 35% | 40% | 45% | 45% | 42% | 35% |
| Hypothyroidism | 0.2% | 2% | 4% | 6% | 8% | 10% | 12% |

#### Respiratory Conditions
| Condition | 0-17 | 18-34 | 35-44 | 45-54 | 55-64 | 65-74 | 75+ |
|-----------|------|-------|-------|-------|-------|-------|-----|
| Asthma | 8% | 8% | 8% | 8% | 7% | 7% | 6% |
| COPD | 0% | 0.5% | 2% | 6% | 10% | 15% | 18% |
| Sleep Apnea | 1% | 5% | 10% | 15% | 18% | 20% | 18% |

#### Mental Health Conditions
| Condition | 0-17 | 18-34 | 35-44 | 45-54 | 55-64 | 65-74 | 75+ |
|-----------|------|-------|-------|-------|-------|-------|-----|
| Depression | 2% | 12% | 10% | 8% | 7% | 5% | 6% |
| Anxiety Disorder | 3% | 15% | 12% | 10% | 8% | 6% | 5% |

#### Musculoskeletal Conditions
| Condition | 0-17 | 18-34 | 35-44 | 45-54 | 55-64 | 65-74 | 75+ |
|-----------|------|-------|-------|-------|-------|-------|-----|
| Osteoarthritis | 0% | 1% | 5% | 15% | 30% | 50% | 60% |
| Rheumatoid Arthritis | 0% | 0.5% | 1% | 1.5% | 1.5% | 1.5% | 1.5% |

#### Other Common Conditions
| Condition | Overall Prevalence | Notes |
|-----------|-------------------|-------|
| GERD | 20% adults | Increases with age/obesity |
| Allergic Rhinitis | 20% all ages | Constant across ages |
| Migraine | 15% adults | Peak ages 25-55, higher in women (2:1) |
| Chronic Kidney Disease | Age-dependent | 1% <45, 5% 45-64, 15% 65+ |
| Psoriasis | 3% | Constant across adults |

### 5.2 Risk Factor Multipliers

**Condition probability adjustments based on risk factors:**

#### Obesity Effects (BMI ≥ 30)
- Type 2 Diabetes: 7× risk
- Hypertension: 3× risk
- Sleep Apnea: 5× risk
- Osteoarthritis: 2× risk
- GERD: 2× risk
- Coronary Artery Disease: 2× risk

#### Smoking Effects
- COPD: 10× risk
- Coronary Artery Disease: 3× risk
- Hypertension: 1.5× risk
- Asthma exacerbation: 1.5× severity

#### Alcohol Effects (Heavy Drinking)
- Hypertension: 2× risk
- Liver disease: 5× risk
- Depression: 1.5× risk

#### Age Effects
- Most chronic conditions: exponential increase with age
- Use logistic regression curves for smooth transitions

#### Sex Effects
- Women: Higher rates of autoimmune, thyroid, migraine, osteoporosis (1.5-3× depending on condition)
- Men: Higher rates of cardiovascular disease <65 (1.3×)

### 5.3 Comorbidity Patterns

**Common Comorbidity Clusters:**

1. **Metabolic Syndrome Cluster:**
   - If Diabetes → 70% chance Hypertension
   - If Diabetes → 60% chance Hyperlipidemia
   - If Hypertension + Hyperlipidemia → 40% chance Diabetes
   - If Obesity → 50% chance at least one metabolic condition

2. **Cardiovascular Disease Cascade:**
   - If Hypertension + Hyperlipidemia → 25% chance CAD (ages 55+)
   - If CAD → 30% chance Heart Failure
   - If Diabetes + CAD → 50% chance Heart Failure

3. **Respiratory Cluster:**
   - If COPD → 40% chance Coronary Artery Disease
   - If Asthma + Smoking → 15% chance COPD

4. **Mental Health Comorbidity:**
   - If Depression → 60% chance Anxiety
   - If Chronic Pain → 40% chance Depression

5. **Obesity-Related Cluster:**
   - If Obesity + Age 50+ → 60% chance Sleep Apnea
   - If Obesity + Hypertension → 35% chance Diabetes

**Implementation:**
```python
# After initial condition assignment, run comorbidity rules
if has_condition('Diabetes'):
    if random() < 0.70:
        assign_condition('Hypertension')
    if random() < 0.60:
        assign_condition('Hyperlipidemia')
```

### 5.4 Number of Conditions per Member

**Distribution:**
- 0 conditions: 45% (healthy population)
- 1 condition: 25%
- 2 conditions: 15%
- 3-5 conditions: 12%
- 6+ conditions: 3% (complex patients)

**Age Adjustments:**
- Ages 0-17: mean=0.3 conditions
- Ages 18-34: mean=0.5
- Ages 35-44: mean=1.2
- Ages 45-54: mean=2.0
- Ages 55-64: mean=3.0
- Ages 65+: mean=4.5

**Generation Strategy:**
1. Sample number of conditions from Poisson(λ=age-adjusted mean)
2. Sample specific conditions weighted by age-stratified prevalence
3. Apply comorbidity rules to add related conditions
4. Apply risk factor multipliers

### 5.5 Diagnostic Date Generation

**Rules:**
- Diagnostic date must be after date of birth
- Diagnostic date should be before or equal to reference date (2024-01-01)

**Timing Patterns:**

**Chronic Conditions:**
- Diagnosed on average at:
  - Hypertension: 10 years after age 30, or when severity triggers diagnosis
  - Diabetes: 5 years after age of typical onset for age group
  - COPD: 15+ years after starting smoking (if smoker)

**Distribution:**
- Uniform random date between [age_of_typical_onset, current_age]

**Recent Diagnoses:**
- 20% of conditions diagnosed in last 2 years (recent events)
- 30% diagnosed 2-5 years ago
- 50% diagnosed 5+ years ago

**Implementation:**
```python
# For a member aged 60 with hypertension
onset_age = max(30, current_age - 20)  # Typically 10-20 years ago
diagnostic_date = DOB + random.uniform(onset_age_years, current_age_years)
```

---

## 6. Insurance & Enrollment Rules

### 6.1 Insurance Coverage Rate

**Overall:** 90% of members are insured, 10% uninsured

**Age-Stratified Coverage:**
| Age Group | Coverage Rate | Notes |
|-----------|--------------|-------|
| 0-17 | 95% | CHIP, parental plans |
| 18-25 | 85% | Young adults, parental plans (ACA) |
| 26-34 | 82% | Lowest coverage (employment transition) |
| 35-54 | 90% | Peak employment |
| 55-64 | 88% | Pre-Medicare |
| 65+ | 99% | Medicare eligibility |

**Socioeconomic Factors:**
- Employed: 95% coverage
- Unemployed: 70% coverage
- Housing Insecurity: 75% coverage

### 6.2 Plan Selection Logic

**Age-Based Preferences:**
| Age Group | Catastrophic | Bronze | Silver | Gold | Platinum | Medicare Adv. |
|-----------|--------------|--------|--------|------|----------|---------------|
| 18-25 | 30% | 40% | 25% | 5% | 0% | 0% |
| 26-34 | 15% | 35% | 35% | 12% | 3% | 0% |
| 35-44 | 5% | 25% | 40% | 22% | 8% | 0% |
| 45-54 | 2% | 20% | 35% | 30% | 13% | 0% |
| 55-64 | 1% | 15% | 30% | 35% | 19% | 0% |
| 65+ | 0% | 0% | 0% | 5% | 5% | 90% |

**Health-Based Adjustments:**
- 3+ chronic conditions: Shift toward Gold/Platinum (higher metal tier)
- No conditions: Shift toward Bronze/Catastrophic (lower premiums)

**Insurance Company Selection:**
- Random with slight preferences based on state (simulate market share)
- Some insurers more common in certain states (e.g., Kaiser in CA)

### 6.3 Enrollment Periods

**Start Date Distribution:**
- January 1: 60% (open enrollment)
- Other months: 40% (qualifying life events, job changes)

**Enrollment Duration:**
| Duration | Percentage | End Date Status |
|----------|------------|-----------------|
| Active (no end date) | 70% | NULL |
| 1-3 years ago | 20% | end_date set |
| 3-5 years ago | 7% | end_date set |
| 5+ years old | 3% | end_date set |

**Start Date Generation:**
```python
# Prefer Jan 1, but spread others throughout year
if random() < 0.6:
    month, day = 1, 1
else:
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Simplify

# Year: random in past 0-5 years
year = reference_year - random.choice([0, 1, 2, 3, 4, 5],
                                      p=[0.35, 0.25, 0.20, 0.10, 0.06, 0.04])
start_date = date(year, month, day)
```

**End Date Generation (if ended):**
```python
# Duration: 1-3 years typical
duration_years = random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
end_date = start_date + duration_years + random.randint(0, 365) days
# Ensure end_date < reference_date
```

### 6.4 Premium Calculation

**Base Formula:**
```python
premium = plan.base_rate * age_factor * state_factor * tier_multiplier / 12
# Monthly premium
```

**Age Rating Factor (ACA-compliant 3:1 ratio):**
| Age | Factor |
|-----|--------|
| 0-14 | 0.65 |
| 15-17 | 0.65 |
| 18-20 | 0.80 |
| 21-24 | 1.00 |
| 25-29 | 1.00 |
| 30-34 | 1.00 |
| 35-39 | 1.05 |
| 40-44 | 1.19 |
| 45-49 | 1.40 |
| 50-54 | 1.70 |
| 55-59 | 2.20 |
| 60-64 | 2.80 |
| 65+ | 0.30 (Medicare) |

**State Factor (cost variation):**
| State Group | Factor | Examples |
|-------------|--------|----------|
| High Cost | 1.3-1.5 | CA, NY, MA |
| Medium Cost | 1.0 | IL, TX, FL |
| Low Cost | 0.7-0.9 | AL, MS, AR |

**Family/Individual:**
- Individual: 100%
- Family: 250% (simplified, would be per-member in real system)

**Tier Multiplier:** (from §1.2 plan structure)

**Example Calculation:**
```python
# 45-year-old in CA with Silver plan
base_rate = $6,600  # Annual
age_factor = 1.40
state_factor = 1.35  # CA
tier_multiplier = 1.0  # Silver

annual_premium = 6600 * 1.40 * 1.35 * 1.0 = $12,474
monthly_premium = $1,039.50
```

### 6.5 Enrollment-Member Relationship

**Data Integrity:**
- If member has `Enrollment ID`, must also have `Insurance ID`
- `Insurance ID` in MEMBERS must match the insurance from ENROLLMENT → PLAN → INSURANCE chain
- Validation: `members.insurance_id == plan.insurance_id WHERE members.enrollment_id = enrollment.id`

**Uninsured Members:**
- `Enrollment ID` = NULL
- `Insurance ID` = NULL

**Insured but No Active Enrollment:**
- Edge case: 2% of insured members have `Insurance ID` but NULL `Enrollment ID`
- Represents: Lapsed enrollment, switching plans, data lag

---

## 7. Claims Generation Logic

### 7.1 Claim Frequency

**Distribution:** Poisson process per member per year

**Base Lambda (claims/year) by Age:**
| Age Group | Lambda (claims/year) |
|-----------|----------------------|
| 0-4 | 4.5 (well-child visits, vaccinations) |
| 5-17 | 2.0 |
| 18-34 | 2.5 |
| 35-44 | 3.0 |
| 45-54 | 4.0 |
| 55-64 | 5.5 |
| 65-74 | 7.0 |
| 75+ | 9.0 |

**Condition Multipliers (additive):**
| Condition Type | Lambda Increase |
|----------------|-----------------|
| Diabetes | +2.5 |
| Hypertension | +1.0 |
| Heart Failure | +4.0 |
| COPD | +3.5 |
| Cancer (active) | +8.0 |
| Mental Health | +1.5 |
| Chronic Kidney Disease | +3.0 |
| Each additional chronic condition | +0.5 |

**Calculation:**
```python
lambda_claims = base_lambda[age_group] + sum(condition_multipliers)
lambda_claims = min(lambda_claims, 25)  # Cap at 25/year (very high utilizers)

# Sample number of claims for enrollment period
enrollment_years = (end_date - start_date).days / 365.25
n_claims = np.random.poisson(lambda_claims * enrollment_years)
```

### 7.2 Claim Timing

**Distribution within enrollment period:**
- Uniform random dates between `start_date` and `end_date` (or reference_date if active)
- Constraint: Claims only during active enrollment
- No claims for uninsured members

**Seasonal Patterns (optional enhancement):**
- Winter months (Dec-Feb): +20% claims (flu, respiratory)
- Summer months (Jun-Aug): -10% claims

**Generation:**
```python
claim_dates = np.random.uniform(
    start_date.timestamp(),
    end_date.timestamp(),
    size=n_claims
)
claim_dates = sorted([date.fromtimestamp(ts) for ts in claim_dates])
```

### 7.3 Claim Amounts

**Distribution:** Log-normal (most claims small, some very large)

**Claim Type Categories:**
| Category | Percentage | Mean ($) | Std Dev (log) | Log Mean | Log Std |
|----------|------------|----------|---------------|----------|---------|
| Office Visit | 45% | $250 | | 5.52 | 0.3 |
| Diagnostic Test | 20% | $600 | | 6.40 | 0.5 |
| Procedure (Minor) | 15% | $2,000 | | 7.60 | 0.4 |
| Emergency Visit | 8% | $3,500 | | 8.16 | 0.6 |
| Hospitalization | 5% | $15,000 | | 9.62 | 0.8 |
| Surgery | 4% | $35,000 | | 10.46 | 0.7 |
| Specialty/High-Cost | 3% | $75,000 | | 11.23 | 1.0 |

**Generation:**
```python
# Sample claim type
claim_type = np.random.choice(types, p=type_probs)

# Generate amount from log-normal
log_mean = log_means[claim_type]
log_std = log_stds[claim_type]
amount = np.random.lognormal(log_mean, log_std)

# Round to cents
amount = round(amount, 2)
```

**Condition-Specific Adjustments:**
- Diabetes: 30% of claims are diabetes management (predictable amounts ~$500)
- Cancer: High probability of high-cost claims ($50K-$200K)
- Heart Failure: Mix of routine follow-ups ($300) and hospitalizations ($25K)
- Mental Health: Mostly therapy visits ($150-$250)

**Age Adjustments:**
- Ages 0-17: Lower per-claim amount (no high-cost procedures) - multiply by 0.7
- Ages 65+: Higher per-claim amount (more complex care) - multiply by 1.4

### 7.4 Claim-Insurance Consistency

**Foreign Key Integrity:**
- `claims.insurance_id` must match `members.insurance_id` for `claims.member_id`
- Rationale: Claims are filed with the member's insurer

**Generation Logic:**
```python
# For each member with insurance and claims:
claim = {
    'claim_id': next_id,
    'member_id': member.id,
    'insurance_id': member.insurance_id,  # Match member's insurer
    'date': claim_date,
    'amount': claim_amount
}
```

### 7.5 Claims Volume Estimates

**For 1M members over 3-year period:**
- Average: 4.5 claims/member/year
- Total: ~13.5M claims
- With variance: 10M - 18M claims

**Optimization:**
- Generate claims in batches by member
- Use vectorized operations for date/amount sampling

---

## 8. Correlation Matrices

### 8.1 Biometric Correlations

**Height-Weight Correlation (see §3.1):**
```
Males:   r = 0.60
Females: r = 0.58
```

### 8.2 Lifestyle Factor Correlations

**Correlation Matrix:**
```
                Smoker  Drinker  Exercise  Sleep  Housing  Employment
Smoker          1.00    0.30     -0.15    -0.10    0.08      -0.05
Drinker         0.30    1.00     -0.10    -0.15    0.05      -0.03
Exercise       -0.15   -0.10      1.00     0.25   -0.08       0.12
Sleep          -0.10   -0.15      0.25     1.00   -0.15       0.05
Housing Ins.    0.08    0.05     -0.08    -0.15    1.00      -0.50
Employment     -0.05   -0.03      0.12     0.05   -0.50       1.00
```

**Implementation:**
```python
# Use multivariate normal on latent variables, then threshold for binary
# Or use copula methods for mixed continuous/binary
```

### 8.3 Condition Clustering

**Metabolic Syndrome Correlation:**
- Diabetes-Hypertension: r = 0.55
- Diabetes-Hyperlipidemia: r = 0.48
- Hypertension-Hyperlipidemia: r = 0.42
- Obesity-Diabetes: r = 0.50

**Cardiovascular Cascade:**
- CAD-Heart Failure: r = 0.45
- Hypertension-CAD: r = 0.35

**Mental Health:**
- Depression-Anxiety: r = 0.60

**Implementation:** Use comorbidity rules (§5.3) rather than correlation matrix

### 8.4 Socioeconomic-Health Correlations

**Housing Insecurity Impact:**
- Correlated with poor diabetes control: r = 0.20
- Correlated with depression: r = 0.25
- Correlated with COPD: r = 0.15

**Employment-Health:**
- Employment correlated with better health outcomes: r = -0.10 to -0.20 for chronic conditions
- Implement via adjusted prevalence rates

---

## 9. Validation Criteria

### 9.1 Schema Compliance Checks

**Foreign Key Validation:**
```sql
-- All foreign keys must resolve
SELECT COUNT(*) FROM MEMBERS m
LEFT JOIN FACILITY f ON m.Primary_Care_Facility_ID = f.Facility_ID
WHERE m.Primary_Care_Facility_ID IS NOT NULL AND f.Facility_ID IS NULL;
-- Should be 0

-- Similar checks for all FK relationships
```

**Check Constraint Validation:**
- Weight > 0
- Height > 0
- Heart rate in [20, 300]
- Blood oxygen in [0, 100]
- Blood pressure format: `^\d{2,3}/\d{2,3}$`
- States: 2-letter uppercase codes
- Boolean fields: 0, 1, or NULL
- Sleep in [0, 24]
- Exercise ≤ 10080 minutes/week

### 9.2 Distribution Statistical Tests

**Kolmogorov-Smirnov Tests:**
```python
# Test if generated distributions match target distributions
from scipy import stats

# Age distribution
ks_stat, p_value = stats.kstest(generated_ages, target_age_distribution)
assert p_value > 0.05, "Age distribution doesn't match target"

# Height, weight, vitals
# Compare mean, std dev within tolerance (±5%)
assert abs(generated.mean() - target.mean()) / target.mean() < 0.05
```

**Categorical Distribution Tests:**
```python
# Chi-square test for sex distribution
chi2, p_value = stats.chisquare(
    observed=[count_F, count_M, count_O],
    expected=[n*0.505, n*0.495, n*0.001]
)
assert p_value > 0.05
```

### 9.3 Correlation Validation

**Verify key correlations:**
```python
# Height-Weight correlation
corr = np.corrcoef(height, weight)[0, 1]
assert 0.55 < corr < 0.65, "Height-weight correlation out of range"

# Smoker-Drinker correlation
# For binary, use point-biserial or tetrachoric correlation
```

### 9.4 Business Rule Validation

**Temporal Consistency:**
```sql
-- Diagnostic dates after DOB
SELECT COUNT(*) FROM MEMBER_CONDITION mc
JOIN MEMBERS m ON mc.Member_ID = m.member_id
WHERE mc.Diagnostic_date < m.DOB;
-- Should be 0

-- Claims during enrollment
SELECT COUNT(*) FROM CLAIMS c
JOIN MEMBERS m ON c.member_id = m.member_id
JOIN ENROLLMENT e ON m.`Enrollment ID` = e.Enrollment_ID
WHERE c.date < e.start_date
   OR (e.end_date IS NOT NULL AND c.date > e.end_date);
-- Should be 0
```

**Insurance Consistency:**
```sql
-- Members with enrollment must have insurance
SELECT COUNT(*) FROM MEMBERS
WHERE `Enrollment ID` IS NOT NULL AND `Insurance ID` IS NULL;
-- Should be 0

-- Claims insurance matches member insurance
SELECT COUNT(*) FROM CLAIMS c
JOIN MEMBERS m ON c.member_id = m.member_id
WHERE c.insurance_id != m.`Insurance ID`;
-- Should be 0
```

**Logical Consistency:**
```sql
-- Children (age < 18) should not be employed
SELECT COUNT(*) FROM MEMBERS
WHERE TIMESTAMPDIFF(YEAR, DOB, '2024-01-01') < 18
  AND `employment status` = 1;
-- Should be 0 or very low

-- Very few people exercise >20 hours/week
SELECT COUNT(*) FROM MEMBERS
WHERE minutes_exercise_per_week > 1200;
-- Should be < 1%
```

### 9.5 Clinical Plausibility Checks

**Condition Prevalence:**
```python
# Verify condition rates match targets (±10% tolerance)
diabetes_rate = (members.has_diabetes.sum() / len(members))
assert 0.095 < diabetes_rate < 0.115, f"Diabetes rate {diabetes_rate} out of range [9.5%, 11.5%]"

# Age-stratified checks
elderly = members[members.age >= 65]
elderly_hypertension = elderly.has_hypertension.mean()
assert 0.60 < elderly_hypertension < 0.75, "Elderly hypertension rate implausible"
```

**Vital Signs Plausibility:**
```python
# Mean BP for hypertensive patients should be elevated
hypertensive_bp = members[members.has_hypertension].bp_systolic.mean()
non_hypertensive_bp = members[~members.has_hypertension].bp_systolic.mean()
assert hypertensive_bp > non_hypertensive_bp + 10, "Hypertensive BP not elevated enough"

# COPD patients should have lower SpO2
copd_spo2 = members[members.has_copd].blood_oxygen.mean()
non_copd_spo2 = members[~members.has_copd].blood_oxygen.mean()
assert copd_spo2 < non_copd_spo2 - 1.5, "COPD SpO2 not reduced"
```

**Claims Reasonableness:**
```python
# Average claim amount should be in expected range [$1000 - $5000]
avg_claim = claims.amount.mean()
assert 1000 < avg_claim < 5000, f"Average claim ${avg_claim} out of range"

# High utilizers (10+ claims/year) should have chronic conditions
high_utilizers = members[members.n_claims_per_year > 10]
avg_conditions = high_utilizers.n_conditions.mean()
assert avg_conditions > 3, "High utilizers don't have enough conditions"
```

### 9.6 Summary Statistics Report

**Generate report with:**
- Total counts per table
- Mean, median, std dev for all numeric fields
- Distribution tables for categorical fields
- Correlation matrices
- Condition prevalence tables
- Insurance coverage rates
- Claims statistics (total amount, avg per member)

**Format:** Markdown or HTML report

---

## 10. Configuration Parameters

### 10.1 Scale Parameters

```yaml
n_members: 1000000           # Total members to generate
n_insurers: 8                # Number of insurance companies
n_plans_per_insurer: 5       # Plans per company (avg)
n_facilities: 2000           # Medical facilities
n_conditions: 75             # Number of conditions in database
```

### 10.2 Temporal Parameters

```yaml
reference_date: '2024-01-01'  # Current date reference
min_dob: '1924-01-01'         # Oldest member birth date (100 years)
max_dob: '2024-01-01'         # Youngest member birth date (newborns)
enrollment_lookback_years: 5  # Generate enrollments up to 5 years back
claims_lookback_years: 3      # Generate claims for last 3 years
```

### 10.3 Random Seed

```yaml
random_seed: 42  # For reproducibility
# Set numpy.random.seed() and random.seed()
```

### 10.4 Null Rate Specifications

```yaml
null_rates:
  weight: 0.05
  height: 0.05
  heart_rate: 0.10
  blood_pressure: 0.10
  blood_oxygen: 0.15
  smoker: 0.05
  drinker: 0.05
  housing_insecurity: 0.03
  employment_status: 0.02
  hours_sleep_per_day: 0.08
  minutes_exercise_per_week: 0.08
  primary_care_facility: 0.30
```

### 10.5 Performance Parameters

```yaml
batch_size: 50000             # Members per batch for generation
enable_parallel: true         # Use multiprocessing where possible
n_workers: 4                  # Number of parallel workers
csv_buffer_size: 10000        # Rows to buffer before writing to CSV
```

### 10.6 Output Parameters

```yaml
output_directory: './synthetic_data/'
output_files:
  insurance: 'insurance.csv'
  plan: 'plan.csv'
  facility: 'facility.csv'
  condition: 'condition.csv'
  enrollment: 'enrollment.csv'
  members: 'members.csv'
  member_condition: 'member_condition.csv'
  claims: 'claims.csv'

include_headers: true
encoding: 'utf-8'
delimiter: ','
date_format: '%Y-%m-%d'
```

### 10.7 Data Quality Flags

```yaml
run_validation: true          # Run all validation checks after generation
generate_report: true         # Generate summary statistics report
fail_on_validation_error: true  # Stop if validation fails
log_level: 'INFO'            # DEBUG, INFO, WARNING, ERROR
```

---

## 11. Data Sources & References

### 11.1 Demographic Data Sources

- **US Census Bureau:** Population estimates by age, sex, state (2023)
  - https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html

- **CDC Wonder:** Mortality and population databases
  - https://wonder.cdc.gov/

### 11.2 Health Statistics Sources

- **CDC National Health Interview Survey (NHIS):** Prevalence of chronic conditions
  - https://www.cdc.gov/nchs/nhis/index.htm

- **CDC NHANES:** Anthropometric and vital signs data
  - https://www.cdc.gov/nchs/nhanes/index.htm

- **CDC Behavioral Risk Factor Surveillance System (BRFSS):** Lifestyle factors
  - https://www.cdc.gov/brfss/index.html

- **American Heart Association:** Hypertension statistics
  - https://www.heart.org/en/health-topics/high-blood-pressure/the-facts-about-high-blood-pressure

- **American Diabetes Association:** Diabetes statistics
  - https://diabetes.org/about-us/statistics/about-diabetes

### 11.3 Healthcare Utilization Sources

- **Medical Expenditure Panel Survey (MEPS):** Claims frequency and costs
  - https://www.meps.ahrq.gov/mepsweb/

- **KFF Health Insurance Coverage Data:** Coverage rates by age
  - https://www.kff.org/

- **CMS Medicare Current Beneficiary Survey:** Medicare population patterns
  - https://www.cms.gov/Research-Statistics-Data-and-Systems/Research/MCBS

### 11.4 Insurance Premium Data

- **KFF Employer Health Benefits Survey:** Premium ranges
  - https://www.kff.org/health-costs/report/employer-health-benefits-survey/

- **Healthcare.gov:** Marketplace plan data
  - https://www.healthcare.gov/

### 11.5 Clinical Guidelines

- **ICD-10 Codes:** International Classification of Diseases
  - https://www.cdc.gov/nchs/icd/icd-10-cm.htm

- **Clinical Practice Guidelines:** For condition comorbidities
  - Various specialty societies (AHA, ADA, etc.)

---

## 12. Implementation Notes

### 12.1 Generation Order

**Must follow dependency order:**
```
1. INSURANCE (no dependencies)
2. PLAN (depends on INSURANCE)
3. FACILITY (no dependencies)
4. CONDITION (no dependencies)
5. ENROLLMENT (depends on PLAN)
6. MEMBERS (depends on FACILITY, INSURANCE, ENROLLMENT)
7. MEMBER_CONDITION (depends on MEMBERS, CONDITION)
8. CLAIMS (depends on MEMBERS, INSURANCE)
```

### 12.2 Memory Management

**For 1M members:**
- Estimated peak memory: ~10-15 GB
- Use batch processing (50K members at a time)
- Write CSVs incrementally using append mode
- Clear DataFrames after batch write

### 12.3 Suggested Libraries

```python
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy import stats
import json
import yaml
from faker import Faker
```

### 12.4 Parallelization Strategy

**Parallelizable tasks:**
- Member generation (by batches)
- Independent table generation (FACILITY, CONDITION in parallel)

**Sequential tasks:**
- Tables with dependencies (ENROLLMENT before MEMBERS)
- CSV writing (to maintain ID consistency)

### 12.5 Reproducibility

**Set all random seeds:**
```python
import random
import numpy as np

random.seed(config['random_seed'])
np.random.seed(config['random_seed'])
```

**Document version:**
- Record specification version
- Record generation timestamp
- Save configuration file with data

---

## Appendix A: Example Data Profiles

### A.1 Example Member Profiles

**Profile 1: Healthy Young Adult**
```yaml
member_id: 1001
DOB: 1998-05-15 (Age 25)
Sex: F
State: CA
Weight: 145 lbs
Height: 65 inches (BMI: 24.1 - Normal)
heart_rate: 68
blood_pressure: "115/75"
blood_oxygen: 98
smoker: 0
drinker: 0
housing_insecurity: 0
employment_status: 1
hours_sleep_per_day: 7.5
minutes_exercise_per_week: 180
Insurance: UnitedHealthcare
Plan: UnitedHealthcare Bronze PPO
Conditions: None
Claims (past year): 2 ($450 total)
```

**Profile 2: Middle-Aged with Metabolic Syndrome**
```yaml
member_id: 5032
DOB: 1974-11-22 (Age 49)
Sex: M
State: TX
Weight: 245 lbs
Height: 70 inches (BMI: 35.2 - Obese)
heart_rate: 82
blood_pressure: "145/92"
blood_oxygen: 96
smoker: 1
drinker: 1
housing_insecurity: 0
employment_status: 1
hours_sleep_per_day: 6.2
minutes_exercise_per_week: 45
Insurance: Blue Cross Blue Shield
Plan: BCBS Gold PPO
Conditions:
  - Type 2 Diabetes (diagnosed 2019)
  - Hypertension (diagnosed 2017)
  - Hyperlipidemia (diagnosed 2018)
  - Obesity (diagnosed 2016)
Claims (past year): 8 ($12,450 total)
```

**Profile 3: Elderly Medicare Beneficiary**
```yaml
member_id: 9845
DOB: 1944-03-08 (Age 79)
Sex: F
State: FL
Weight: 138 lbs
Height: 62 inches (BMI: 25.2 - Overweight)
heart_rate: 78
blood_pressure: "152/88"
blood_oxygen: 95
smoker: 0
drinker: 0
housing_insecurity: 0
employment_status: 0 (retired)
hours_sleep_per_day: 7.8
minutes_exercise_per_week: 90
Insurance: Humana
Plan: Humana Medicare Advantage
Conditions:
  - Hypertension (diagnosed 1998)
  - Osteoarthritis (diagnosed 2010)
  - Hyperlipidemia (diagnosed 2002)
  - Atrial Fibrillation (diagnosed 2020)
  - Hypothyroidism (diagnosed 2005)
Claims (past year): 15 ($28,600 total)
```

**Profile 4: Uninsured Low-Income Adult**
```yaml
member_id: 7234
DOB: 1990-09-14 (Age 33)
Sex: M
State: GA
Weight: 172 lbs
Height: 68 inches (BMI: 26.1 - Overweight)
heart_rate: 75
blood_pressure: "128/82"
blood_oxygen: 97
smoker: 1
drinker: 0
housing_insecurity: 1
employment_status: 0
hours_sleep_per_day: 6.0
minutes_exercise_per_week: 60
Insurance: None
Plan: None
Conditions:
  - Depression (diagnosed 2021)
  - Asthma (diagnosed 2008)
Claims (past year): 0 (uninsured, no claims filed)
```

---

## Appendix B: SQL Loader Script Notes

**CSV to SQL Loading:**

```python
# Generate SQL LOAD DATA statements
for table, csv_file in csv_files.items():
    sql = f"""
LOAD DATA LOCAL INFILE '{csv_file}'
INTO TABLE {table}
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS;
"""
    print(sql)
```

**Handling Special Cases:**
- Quote column names with spaces: `` `housing insecurity` ``
- JSON fields: Load as strings, MySQL will parse automatically
- Date format: Ensure CSV dates match MySQL format (YYYY-MM-DD)
- NULL values: Empty strings in CSV → NULL in MySQL

**Performance Tips:**
- Disable foreign key checks during bulk load:
  ```sql
  SET FOREIGN_KEY_CHECKS=0;
  -- Load all tables
  SET FOREIGN_KEY_CHECKS=1;
  ```
- Load in dependency order to avoid FK violations
- Use transactions for atomicity

---

## Version History

**v1.0 (2024-01-13):** Initial specification
- Complete distributions for all tables
- Age-stratified condition prevalence
- Biometric correlations
- Claims generation logic
- Validation criteria

---

**End of Specification**

This specification provides a complete blueprint for generating synthetic health insurance data. Implementation should follow this specification while maintaining flexibility for adjustments based on actual generation results and validation outcomes.
