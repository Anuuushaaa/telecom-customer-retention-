
"""
Reusable data cleaning for Telecom Customer Retention.

Cleaning rules:
1. Convert TotalCharges to numeric.
2. Fill legitimate blank charges for zero-tenure customers.
3. Reject invalid, negative or non-finite charges.
4. Normalize internet-dependent service categories.
5. Normalize phone-dependent service categories.
6. Preserve customer IDs and input row count.
"""

import numpy as np
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
    """Clean raw telecom customer records without modifying the input."""

    df = df.copy()

    # Validate and convert TotalCharges.
    raw_charges = df["TotalCharges"]

    blank_charges = (
        raw_charges.isna()
        | raw_charges.astype(str).str.strip().eq("")
    )

    numeric_charges = pd.to_numeric(
        raw_charges,
        errors="coerce",
    )

    # Only blank charges for zero-tenure customers
    # are eligible for replacement with zero.
    valid_zero_tenure = (
        blank_charges & df["tenure"].eq(0)
    )

    invalid_charges = (
        numeric_charges.isna()
        & ~valid_zero_tenure
    )

    if invalid_charges.any():
        raise ValueError(
            f"Invalid TotalCharges in "
            f"{invalid_charges.sum()} record(s)."
        )

    numeric_charges = numeric_charges.mask(
        valid_zero_tenure,
        0.0,
    )

    if not np.isfinite(numeric_charges).all():
        raise ValueError(
            "TotalCharges contains non-finite values."
        )

    if (numeric_charges < 0).any():
        raise ValueError(
            "TotalCharges cannot be negative."
        )

    df["TotalCharges"] = numeric_charges.astype(float)

    # Normalize internet-dependent service categories.
    for col in INTERNET_DEPENDENT_COLS:
        df[col] = df[col].replace(
            "No internet service",
            "No",
        )

    # Normalize phone-dependent service categories.
    df["MultipleLines"] = df["MultipleLines"].replace(
        "No phone service",
        "No",
    )

    return df
