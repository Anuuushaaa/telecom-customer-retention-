"""
src/feature_engineering.py

Phase 6: Feature Engineering for the Telecom Churn project.

Unlike src/preprocessing.py (which only re-represents existing columns via
encoding/scaling), this module creates genuinely NEW columns derived from
the raw features — each justified by findings from Phase 4 (EDA) and
Phase 5 (Statistical Analysis). See README "Feature Engineering" section
for the full rationale behind each one.

This must run AFTER src/cleaning.py (needs clean TotalCharges) and BEFORE
src/preprocessing.py (produces new columns that preprocessing then encodes/
scales like any other column).
"""

import numpy as np
import pandas as pd

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BUCKET_ORDER = {"New": 0, "Short": 1, "Medium": 2, "Long": 3}


def add_num_services_subscribed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of subscribed service add-ons (out of 6), per customer.

    Rationale (Phase 5): each service column individually showed weak
    Cramer's V against Churn, but bundling/switching-cost theory suggests
    the COUNT of services may be more predictive than any single one —
    more bundled customers face higher switching friction and tend to
    stay longer.
    """
    df = df.copy()
    df["num_services_subscribed"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly spend = TotalCharges / tenure.

    Rationale (Phase 4/5): tenure and TotalCharges are strongly correlated
    (r=0.826). This feature recasts cumulative spend as a RATE, independent
    of tenure length.

    Zero-tenure customers (division by zero) fall back to MonthlyCharges
    as the best available estimate of their in-progress spend rate.
    """
    df = df.copy()
    with_tenure = df["tenure"].replace(0, np.nan)
    avg_spend = df["TotalCharges"] / with_tenure
    df["avg_monthly_spend"] = avg_spend.fillna(df["MonthlyCharges"]).astype(float)
    return df


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal tenure bucket: New (0-6mo), Short (6-24mo), Medium (24-48mo),
    Long (48mo+).

    Rationale (Phase 4): tenure's distribution is visibly bimodal, not
    linear in its relationship to churn risk. Bucketing lets a linear
    model capture distinct risk regimes that a single continuous
    coefficient cannot represent.
    """
    df = df.copy()
    bins = [-1, 6, 24, 48, float("inf")]
    labels = ["New", "Short", "Medium", "Long"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    df["tenure_bucket"] = df["tenure_bucket"].map(TENURE_BUCKET_ORDER).astype("int64")
    return df


def add_is_electronic_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if PaymentMethod is "Electronic check", else 0.

    Rationale (Phase 5): PaymentMethod as a whole showed strong Cramer's V
    (0.303) against Churn. Electronic check is a well-documented specific
    driver within this dataset. Isolating this category tests whether it
    carries most of PaymentMethod's overall signal alone.
    """
    df = df.copy()
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all Phase 6 engineered features in sequence. Call this after
    src/cleaning.py's clean_data() and before src/preprocessing.py's
    fit_transform_features().
    """
    df = add_num_services_subscribed(df)
    df = add_avg_monthly_spend(df)
    df = add_tenure_bucket(df)
    df = add_is_electronic_check(df)
    return df


if __name__ == "__main__":
    from src.cleaning import clean_data

    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)
    df_engineered = engineer_features(df_clean)

    new_cols = [
        "num_services_subscribed", "avg_monthly_spend",
        "tenure_bucket", "is_electronic_check",
    ]
    print("New engineered columns (first 10 rows):")
    print(df_engineered[["customerID", "tenure", "TotalCharges"] + new_cols].head(10))
    print("\nSummary stats of new numeric features:")
    print(df_engineered[["num_services_subscribed", "avg_monthly_spend"]].describe())
    print("\ntenure_bucket value counts:")
    print(df_engineered["tenure_bucket"].value_counts().sort_index())