# Telecom Customer Retention Intelligence

## Overview
A machine learning system that predicts telecom customer churn, explains *why*
each customer is likely to churn using SHAP, and recommends retention actions.
Built as an end-to-end, production-style Data Science portfolio project.

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
# Telecom Customer Retention Intelligence

## Overview
A machine learning system that predicts telecom customer churn, explains *why*
each customer is likely to churn using SHAP, and recommends retention actions.
Built as an end-to-end, production-style Data Science portfolio project.

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
# Telecom Customer Retention Intelligence

## Overview
A machine learning system that predicts telecom customer churn, explains *why*
each customer is likely to churn using SHAP, and recommends retention actions.
Built as an end-to-end, production-style Data Science portfolio project.

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
├── notebooks/          # exploratory analysis
├── models/             # trained model artifacts (not committed)
├── src/                # reusable pipeline code (cleaning, features, training, explainability)
├── tests/              # unit tests
├── app.py              # Streamlit app
├── train.py             # training pipeline
├── predict.py           # inference script
└── requirements.txt
## Setup
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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
| TotalCharges | object* | Cumulative amount charged to date (*stored as text due to blank strings for 11 zero-tenure customers — fixed in Phase 3) |
| Churn | object | Target variable: Yes/No, whether the customer left in the last month |

## Known Data Quality Findings (Phase 2)
- **Shape**: 7,043 rows × 21 columns, 0 duplicate rows.
- **TotalCharges** is misread as text due to 11 rows containing a blank string
  instead of a number. All 11 correspond to customers with `tenure == 0`
  (brand-new customers with no completed billing cycle yet). These will be
  set to `0` in Phase 3, since cumulative billing for a zero-tenure customer
  is genuinely zero — not missing data in the conventional sense.
- **Class imbalance**: target variable is 73.46% No / 26.54% Yes. This drives
  metric choice in later phases (recall-weighted evaluation over raw accuracy).
- **customerID** is a unique identifier, excluded from model features but
  retained for the Streamlit app (per-customer lookup).

## Status
## Data Cleaning (Phase 3)

Cleaning logic lives in `src/cleaning.py` as a single reusable function,
`clean_data(df)`, so training and inference use identical preprocessing
(avoids training-serving skew).

**Steps applied:**
- **TotalCharges**: converted from text to numeric; the 11 blank-string rows
  (all `tenure == 0`) are filled with `0`, since a zero-tenure customer has
  genuinely completed no billing cycle yet — not missing data in the
  conventional sense.
- **Categorical collapsing**: `"No internet service"` (in `OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
  `StreamingMovies`) and `"No phone service"` (in `MultipleLines`) are
  collapsed into `"No"`. These values are perfectly redundant with
  `InternetService == "No"` / `PhoneService == "No"` respectively, so keeping
  them as separate categories only added multicollinearity and diluted SHAP
  attribution across features, with zero informational gain.
- **customerID**: retained in the cleaned dataframe (needed for the Streamlit
  app's per-customer lookup); it will be excluded from the model's feature
  matrix at training time, not during cleaning.

Unit tests for this function live in `tests/test_cleaning.py`, covering the
TotalCharges fix, both category collapses, row-count preservation, and that
customerID survives cleaning.

## Status
## Feature Engineering & Preprocessing (Phase 4)

Preprocessing logic lives in `src/preprocessing.py`, split into two functions:
`split_data()` and `fit_transform_features()`. These are kept separate
deliberately — the split must happen **before** any encoding/scaling, so that
all learned transformations (the scaler) are fit on the training set only,
never on test data. This prevents data leakage.

**Steps applied:**
- **Train/test split**: 80/20, stratified on `Churn` to preserve the original
  26.54% churn rate in both sets (confirmed: 0.2654 in both train and test).
- **Contract**: ordinal-encoded (`Month-to-month`=0, `One year`=1,
  `Two year`=2) — a natural order exists, and this keeps feature count low.
- **Other categoricals** (`gender`, `InternetService`, `PaymentMethod`):
  one-hot encoded — no inherent order, so ordinal encoding would invent a
  false relationship between categories.
  - `gender` deliberately keeps both `gender_Female` and `gender_Male`
    columns (`drop_first=False`) rather than dropping one. This is mildly
    redundant but harmless for tree-based models and simpler to reason
    about; a stricter choice for pure Logistic Regression would be to drop
    one, but that's not required here.
- **Binary Yes/No columns** (Partner, Dependents, PhoneService, the 6
  internet-dependent columns, PaperlessBilling, MultipleLines): mapped
  directly to 1/0 rather than one-hot encoded, since they're genuinely binary.
- **Numeric columns** (tenure, MonthlyCharges, TotalCharges): scaled with
  `StandardScaler`, fit on the training set only. Scaling isn't required for
  tree-based models but is required for Logistic Regression, and since both
  are being evaluated in Phase 5, scaling is applied to keep both viable.
- **customerID**: excluded from the feature matrix, returned separately
  (`customer_ids_train` / `customer_ids_test`) for later use in the
  Streamlit app's per-customer lookup.
- **Churn**: mapped to 1/0 as the target variable.

Output: 25 features, 5634 training rows / 1409 test rows, zero NaNs.

Unit tests in `tests/test_preprocessing.py` cover stratification, correct
encoding of `Contract`, no data leakage in the scaler, exclusion of
`customerID` from features, and absence of NaNs.

## Status
## Model Training & Evaluation (Phase 5)

Two models trained on identical preprocessed data (`src/train.py`), both
using loss-reweighting (not oversampling) to handle the 73/26 class
imbalance — `class_weight='balanced'` for Logistic Regression,
`scale_pos_weight` for XGBoost — so no synthetic data was introduced.

**Why accuracy alone isn't the metric:** a model predicting "No Churn" for
every customer would score ~73% accuracy while catching zero churners.
Recall on the churn class is the primary metric, since the business cost of
missing a churner is higher than the cost of an unnecessary retention offer
(false positive).

**Initial default-threshold (0.5) results:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7388 | 0.5052 | 0.7861 | 0.6151 | 0.8419 |
| XGBoost | 0.7580 | 0.5358 | 0.6604 | 0.5916 | 0.8235 |

At the default threshold, XGBoost appeared weaker on recall — but this
turned out to be a threshold artifact, not a genuine model gap.

**Threshold tuning:** since 0.5 is an arbitrary cutoff not derived from this
problem's priorities, both models were swept across thresholds 0.20–0.50.
At matched recall (~0.80–0.83), XGBoost's precision was comparable to or
better than Logistic Regression's — the earlier "XGBoost is worse" reading
was an artifact of comparing both models at an unexamined default threshold,
not a real performance gap.

**Final choice: Logistic Regression @ threshold 0.35** (recall 0.9037,
precision 0.4471, F1 0.5982), locked in `src/config.py`. Chosen over XGBoost
despite XGBoost's slightly better precision at matched recall, because this
project's core value proposition is explainability — Logistic Regression's
coefficients are directly interpretable, which pairs better with the SHAP
analysis in Phase 6 than a boosted tree ensemble would.

Trained models and preprocessing artifacts are saved to `models/`
(not committed to git — see `.gitignore`).

## Status
🚧 In progress — Phase 5 (Model Training & Evaluation) complete.

