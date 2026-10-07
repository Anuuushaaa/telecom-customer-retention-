# Telecom Customer Retention Intelligence

## Overview
A machine learning system that predicts telecom customer churn, explains *why*
each customer is likely to churn using SHAP, and recommends retention actions.
Built as an end-to-end, production-style Data Science portfolio project.

## Key Findings

- **Top churn drivers** (via statistical testing and SHAP): `Contract` type,
  `tenure` (and engineered `tenure_bucket`), and `InternetService` type are
  the dominant, consistently-validated churn signals — confirmed via
  Mann-Whitney U / chi-square tests (Phase 5) and SHAP (Phase 9), not just
  visual impression.
- **Threshold matters more than model choice at the default cutoff**:
  comparing models at the default 0.5 threshold was misleading — XGBoost's
  apparent recall deficit mostly disappeared once thresholds were tuned to
  match the business priority (recall).
- **Final model**: Logistic Regression @ threshold 0.35, chosen for
  interpretability over XGBoost's marginal edge in F1 at matched recall,
  given this project's emphasis on explainability.
- **Result** (after feature engineering): recall 0.9064 / precision 0.4508 /
  F1 0.6021 / ROC-AUC 0.8427 on held-out test data — catches ~91% of actual
  churners, at the cost of some false positives (an accepted tradeoff, since
  a missed churner costs more than an unnecessary retention offer).
- **Feature engineering validated**: 4 engineered features improved every
  metric on retraining; `tenure_bucket` in particular ranked #3 in global
  SHAP importance, confirming that tenure's relationship with churn is
  meaningfully non-linear.
- Full inference pipeline (`predict.py`) mirrors the training pipeline
  exactly (same cleaning, same feature engineering, fitted-not-refit
  scaler) to avoid training-serving skew, and is verified against a real
  test-set customer.

## Problem Statement
Telecom companies lose significant revenue when customers cancel their
subscriptions (churn). Since acquiring a new customer typically costs far more
than retaining an existing one, predicting churn *before* it happens allows
retention teams to intervene proactively.

## Dataset
IBM Telco Customer Churn dataset (7,043 customers, 21 features).
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Churn definition: binary flag indicating whether the customer left within the
last month.

**Note on limitations:** the dataset's churn label reflects account status,
not the underlying *reason* for churn (e.g., competitor porting, price
dissatisfaction). This is a known constraint of billing-system-derived churn
labels.

## Project Structure
telecom-customer-retention/
├── data/raw            # original dataset (not committed to git)
├── data/processed      # cleaned + feature-engineered data (not committed)
├── models/              # trained model artifacts (not committed)
├── reports/             # statistical test results, EDA + SHAP figures
├── src/                 # reusable pipeline code
│   ├── data_loader.py       # raw data loading + profiling
│   ├── cleaning.py           # TotalCharges fix, categorical collapsing
│   ├── eda.py                 # exploratory visualizations
│   ├── statistical_analysis.py # Mann-Whitney U + chi-square tests
│   ├── feature_engineering.py  # 4 engineered features
│   ├── preprocessing.py        # encoding, scaling, train/test split
│   ├── train.py                 # model training
│   ├── threshold_tuning.py      # precision/recall threshold sweep
│   ├── explain.py                # SHAP global + per-customer explanations
│   └── config.py                  # production model + threshold config
├── tests/                # unit tests
├── app.py                 # Streamlit app
├── predict.py              # inference pipeline
└── requirements.txt
## Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Getting Started (Full Pipeline)

Model artifacts (`models/*.pkl`) are not committed to this repo (standard
practice for binary files) and must be generated locally:

```powershell
# 1. Set up environment (see Setup above)
# 2. Download the dataset and place it at data/raw/telco_churn.csv
#    Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# 3. (Optional) Explore: EDA + statistical tests
python -m src.eda
python -m src.statistical_analysis

# 4. Train models (generates models/*.pkl)
python -m src.train

# 5. Tune the classification threshold
python -m src.threshold_tuning

# 6. Generate SHAP explainer (generates models/shap_explainer.pkl)
python -m src.explain

# 7. Run the app
streamlit run app.py
```

Optional: run the test suite to verify everything works correctly:
```powershell
pytest -v
```

## Data Dictionary

| Column | Type | Description |
|---|---|---|
| customerID | object | Unique customer identifier (not used as a model feature) |
| gender | object | Male / Female |
| SeniorCitizen | int64 | 1 if senior citizen, else 0 |
| Partner | object | Whether the customer has a partner (Yes/No) |
| Dependents | object | Whether the customer has dependents (Yes/No) |
| tenure | int64 | Number of months the customer has stayed with the company |
| PhoneService | object | Whether the customer has phone service |
| MultipleLines | object | Whether the customer has multiple lines |
| InternetService | object | DSL / Fiber optic / No |
| OnlineSecurity | object | Whether the customer has online security add-on |
| OnlineBackup | object | Whether the customer has online backup add-on |
| DeviceProtection | object | Whether the customer has device protection add-on |
| TechSupport | object | Whether the customer has tech support add-on |
| StreamingTV | object | Whether the customer has streaming TV |
| StreamingMovies | object | Whether the customer has streaming movies |
| Contract | object | Month-to-month / One year / Two year |
| PaperlessBilling | object | Whether billing is paperless |
| PaymentMethod | object | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | float64 | Current monthly charge amount |
| TotalCharges | object* | Cumulative amount charged to date (*stored as text due to blank strings for 11 zero-tenure customers — fixed in cleaning) |
| Churn | object | Target variable: Yes/No, whether the customer left in the last month |

## Data Quality Findings

- **Shape**: 7,043 rows × 21 columns, 0 duplicate rows.
- **TotalCharges** is misread as text due to 11 rows containing a blank
  string instead of a number. All 11 correspond to `tenure == 0` customers
  (no completed billing cycle yet) — filled with `0`, not treated as
  conventional missing data.
- **Class imbalance**: target variable is 73.46% No / 26.54% Yes — drives
  metric choice (recall-weighted evaluation over raw accuracy).
- **customerID** is a unique identifier, excluded from model features but
  retained for the Streamlit app's per-customer lookup.

## Data Cleaning

Cleaning logic lives in `src/cleaning.py` as a single reusable function,
`clean_data(df)`, used identically in training and inference (avoids
training-serving skew).

**Steps applied:**
- `TotalCharges`: converted from text to numeric, 11 zero-tenure blanks
  filled with `0`.
- **Categorical collapsing**: `"No internet service"` / `"No phone service"`
  collapsed into `"No"` — perfectly redundant with `InternetService`/
  `PhoneService`, so keeping them separate only added multicollinearity
  and diluted SHAP attribution, with zero informational gain.
- `customerID` retained through cleaning (for lookup), excluded from the
  feature matrix later, at preprocessing time.

Unit tests in `tests/test_cleaning.py`.

## Exploratory Data Analysis

Visualizations in `src/eda.py`, saved to `reports/figures/eda/`: target
distribution, numeric distributions, numeric-vs-Churn boxplots,
categorical-vs-Churn countplots, and a correlation heatmap.

**Key finding:** `tenure` and `TotalCharges` are strongly correlated
(r = 0.826) — expected, since cumulative billing grows with tenure. This
redundancy later explained a contradictory SHAP attribution (see
Explainability) and directly motivated the `avg_monthly_spend` and
`tenure_bucket` engineered features.

## Statistical Analysis

Numeric features tested with the **Mann-Whitney U test** (chosen over a
Student's t-test since `tenure` is visibly non-normal/bimodal — applied
uniformly across all three numeric columns for consistency). Categorical
features tested with a **chi-square test of independence**, reported
alongside **Cramér's V** effect size, since p-values alone can be
statistically significant but practically negligible at this sample size
(n=7,043).

**Numeric features (Mann-Whitney U) — all significant, p < 0.001:**

| Feature | Median (Churned) | Median (Stayed) | p-value |
|---|---|---|---|
| tenure | 10.0 months | 38.0 months | 2.4e-208 |
| MonthlyCharges | $79.65 | $64.43 | 3.3e-54 |
| TotalCharges | $703.55 | $1,679.52 | 5.7e-83 |

**Categorical features (Chi-square + Cramér's V), ranked by effect size:**

| Effect size | Features |
|---|---|
| Strong (V > 0.3) | Contract (0.410), InternetService (0.322), PaymentMethod (0.303) |
| Moderate (V 0.15–0.19) | PaperlessBilling, OnlineSecurity, TechSupport, Dependents, SeniorCitizen, Partner |
| Weak (V < 0.1) | OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies, MultipleLines |
| Not significant | PhoneService (p=0.34), **gender (p=0.49)** |

**Key finding:** gender shows no statistically significant relationship
with churn — an explicitly tested result, not an assumption.

Full results in `reports/mann_whitney_results.csv` and
`reports/chi_square_results.csv`.

## Feature Engineering

Four new features created in `src/feature_engineering.py`, each justified
by EDA/statistical findings above:

- **`num_services_subscribed`** — count of the 6 service add-ons
  subscribed. Tests whether bundling (a switching-cost proxy) matters more
  in aggregate than any single service does individually.
- **`avg_monthly_spend`** = `TotalCharges / tenure` — an attempt to
  recast cumulative spend as a rate, independent of tenure, to address the
  tenure/TotalCharges collinearity found in EDA.
- **`tenure_bucket`** (New/Short/Medium/Long, ordinal) — captures tenure's
  non-linear relationship with churn risk, which a single linear
  coefficient can't represent.
- **`is_electronic_check`** — isolates the strongest `PaymentMethod`
  category as its own flag.

**Validated impact:** retraining with these features improved Logistic
Regression across every metric (accuracy 0.7388→0.7473, precision
0.5052→0.5156, recall 0.7861→0.7968, F1 0.6151→0.6261, ROC-AUC
0.8419→0.8427), and similarly improved XGBoost.

**Honest limitations (documented, not corrected):**
- `avg_monthly_spend` ranked low in SHAP importance (0.030) and did not
  resolve the tenure/TotalCharges multicollinearity it targeted, since it
  was added alongside `TotalCharges` rather than replacing it.
- `is_electronic_check` and the existing one-hot `PaymentMethod_Electronic
  check` carry identical SHAP values (confirmed duplicate signal). Left
  in place — causes no correctness issue, only minor dilution of feature
  importance ranking.

## Preprocessing

`src/preprocessing.py`, split into `split_data()` and
`fit_transform_features()` — kept separate so the scaler is fit on the
training set only, never on test data (prevents leakage).

- **Train/test split**: 80/20, stratified on `Churn` (26.54% churn rate
  preserved in both sets).
- **`Contract`**: ordinal-encoded (natural order exists).
- **`gender`, `InternetService`, `PaymentMethod`**: one-hot encoded (no
  inherent order).
- **Binary Yes/No columns**: mapped directly to 1/0.
- **Numeric columns**: scaled with `StandardScaler`, fit on training data
  only.
- **`customerID`**: excluded from the feature matrix, returned separately
  for the Streamlit app's lookup.

Unit tests in `tests/test_preprocessing.py`.

## Model Training & Evaluation

Two models trained on identical preprocessed data (`src/train.py`), both
using loss-reweighting (`class_weight='balanced'` for Logistic Regression,
`scale_pos_weight` for XGBoost) to handle the 73/26 class imbalance — no
synthetic oversampling.

**Why accuracy alone isn't the metric:** a model predicting "No Churn" for
every customer would score ~73% accuracy while catching zero churners.
Recall on the churn class is prioritized, since the cost of missing a
churner exceeds the cost of an unnecessary retention offer (false
positive).

**Results after feature engineering, default threshold (0.5):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7473 | 0.5156 | 0.7968 | 0.6261 | 0.8427 |
| XGBoost | 0.7630 | 0.5444 | 0.6551 | 0.5947 | 0.8247 |

**Threshold tuning** (`src/threshold_tuning.py`): 0.5 is an arbitrary
cutoff, not derived from this problem's priorities. Both models were swept
across thresholds 0.20–0.50. At matched recall, XGBoost's F1 was
competitive with or better than Logistic Regression's — the "XGBoost is
worse" reading at the default threshold was an artifact of an unexamined
cutoff, not a real performance gap.

**Final choice: Logistic Regression @ threshold 0.35** (recall 0.9064,
precision 0.4508, F1 0.6021), locked in `src/config.py`. Chosen over
XGBoost despite XGBoost's marginally better F1 at matched recall, because
this project's core value proposition is explainability — Logistic
Regression's coefficients are directly interpretable, pairing better with
SHAP than a boosted tree ensemble.

## Explainability

SHAP (`shap.LinearExplainer`, the correct explainer for a linear model)
explains both global feature importance and individual predictions.

**Global feature importance** matches domain intuition: `Contract`,
`InternetService_Fiber optic`, `tenure_bucket`, and `tenure` are the
strongest churn drivers. `tenure_bucket` ranking #3 validates the
feature-engineering hypothesis that tenure's relationship with churn is
non-linear.

**Per-customer explanations** attribute each individual prediction to
specific feature contributions — e.g. "this customer is flagged because of
X and Y, despite Z" — the actionable output for a retention team.

**Known limitations (documented, not corrected):**
- `tenure` and `TotalCharges` occasionally receive opposing SHAP signs for
  the same customer, despite being positively correlated — a known
  consequence of multicollinearity in linear models. The `avg_monthly_spend`
  feature, engineered to address this, did not resolve it (see Feature
  Engineering).
- `is_electronic_check` duplicates `PaymentMethod_Electronic check`.

Global summary and per-customer waterfall plots saved to
`reports/figures/`. Explainer saved to `models/shap_explainer.pkl` for
reuse in `predict.py` and the Streamlit app.

## Inference Pipeline

`predict.py` takes a single raw customer record and returns a churn
prediction, probability, and SHAP explanation.

**Guarding against training-serving skew:** `predict.py` reuses
`clean_data()` and `engineer_features()` unchanged from training, then
applies the already-fitted scaler (`scaler.transform()`, never `.fit()`)
and reindexes against the exact saved `feature_columns` list, filling any
missing dummy columns with 0.

Verified end-to-end with a deliberately high-risk test profile
(month-to-month, tenure=2, fiber optic, electronic check) — correctly
predicted "Churn" at 91.4% probability, with `tenure_bucket`, `tenure`,
`InternetService_Fiber optic`, and `Contract` as the top contributing
factors, consistent with known churn drivers.

## Streamlit App

`app.py` is a thin UI wrapper around `predict.py` — no new modeling or
encoding logic, so the training-serving skew guarantees above still hold
for predictions made through the app.

**Two input modes:**
- **Lookup existing customer**: select any `customerID` via dropdown, view
  their raw data, get a live prediction.
- **Manual entry**: enter a hypothetical customer profile for "what if"
  scenario testing.

**Output:** churn prediction, probability, the decision threshold in use
(0.35), and a SHAP bar chart of the top 5 features driving that specific
prediction — red bars push toward churn, green bars push away from it.

Run locally with:
```powershell
streamlit run app.py
```

