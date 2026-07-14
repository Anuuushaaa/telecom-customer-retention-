"""
src/cleaning.py

Reusable data cleaning for the Telecom Customer Retention project.
This function encodes every cleaning decision made in Phase 2/3:

1. TotalCharges: convert text -> numeric, fill 11 blank (zero-tenure) rows with 0
2. Collapse "No internet service" -> "No" across 6 internet-dependent columns
3. Collapse "No phone service" -> "No" in MultipleLines
4. customerID is NOT dropped here — it's kept for lookup purposes.
   Feature-selection (excluding it from the model matrix) happens later,
   at train/predict time, not in cleaning.
"""

import pandas as pd


INTERNET_DEPENDENT_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all Phase 3 cleaning steps to a raw Telco churn dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe, as loaded directly from the source CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe. Same number of rows as input; no rows dropped.
    """
    df = df.copy()

    # --- 1. TotalCharges: text -> numeric ---
    # Blank strings become NaN on coercion; those 11 rows are all
    # tenure == 0 customers (verified in Phase 2), so filling with 0
    # is a substantive claim ("no completed billing cycle yet"),
    # not a blind fillna.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- 2. Collapse "No internet service" -> "No" ---
    # These 6 columns are perfectly redundant with InternetService == "No".
    # Collapsing removes multicollinearity and prevents SHAP from diluting
    # the "no internet" effect across 6 features.
    for col in INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace("No internet service", "No")

    # --- 3. Collapse "No phone service" -> "No" in MultipleLines ---
    # Same logic, redundant with PhoneService == "No".
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df