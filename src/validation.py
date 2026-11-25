
"""
Input schema validation for Telecom Customer Retention.

Supports training datasets and production inference records.
"""

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

for column in SERVICE_COLUMNS:
    CATEGORICAL_VALUES[column] = {
        "Yes",
        "No",
        "No internet service",
    }


NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

REQUIRED_FEATURES = (
    list(CATEGORICAL_VALUES.keys())
    + NUMERIC_COLUMNS
)


def validate_schema(
    df: pd.DataFrame,
    require_target: bool = True,
) -> None:
    """Validate required columns and dataset structure."""

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    required = set(REQUIRED_FEATURES)

    if require_target:
        required.update({"customerID", "Churn"})

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def validate_customer_ids(df: pd.DataFrame) -> None:
    """Ensure training customer IDs are valid and unique."""

    ids = df["customerID"]

    if ids.isna().any():
        raise ValueError("Missing customer IDs detected.")

    normalized_ids = ids.astype(str).str.strip()

    if normalized_ids.eq("").any():
        raise ValueError("Blank customer IDs detected.")

    if normalized_ids.duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")


def validate_categoricals(df: pd.DataFrame) -> None:
    """Reject missing or unsupported categorical values."""

    for column, allowed in CATEGORICAL_VALUES.items():

        invalid = ~df[column].isin(allowed)

        if invalid.any():
            values = (
                df.loc[invalid, column]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Invalid values in {column}: {values}"
            )


def validate_numerics(df: pd.DataFrame) -> None:
    """Validate numeric types, ranges and missing values."""

    for column in NUMERIC_COLUMNS:

        raw = df[column]

        numeric = pd.to_numeric(
            raw,
            errors="coerce",
        )

        if column == "TotalCharges":

            blank = (
                raw.isna()
                | raw.astype(str).str.strip().eq("")
            )

            tenure = pd.to_numeric(
                df["tenure"],
                errors="coerce",
            )

            valid_blank = blank & tenure.eq(0)

            numeric = numeric.mask(valid_blank, 0.0)

        if numeric.isna().any():
            raise ValueError(
                f"Missing or invalid numeric values in {column}."
            )

        if not np.isfinite(numeric).all():
            raise ValueError(
                f"Non-finite values in {column}."
            )

        if (numeric < 0).any():
            raise ValueError(
                f"Negative values in {column}."
            )

        if column == "SeniorCitizen":
            if not numeric.isin([0, 1]).all():
                raise ValueError(
                    "SeniorCitizen must be 0 or 1."
                )

        if column == "tenure":
            if not numeric.mod(1).eq(0).all():
                raise ValueError(
                    "Tenure must contain whole months."
                )


def validate_target(df: pd.DataFrame) -> None:
    """Validate the binary churn target."""

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError(
            "Churn must contain only 'Yes' or 'No'."
        )


def validate_raw_data(
    df: pd.DataFrame,
    require_target: bool = True,
) -> None:
    """Run all applicable input validation checks."""

    validate_schema(df, require_target)

    if require_target:
        validate_customer_ids(df)
        validate_target(df)

    validate_categoricals(df)
    validate_numerics(df)
