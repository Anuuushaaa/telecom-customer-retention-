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
🚧 In progress — Phase 2 (Dataset Understanding) complete.
