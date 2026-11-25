
"""
src/preprocessing.py

Feature engineering and preprocessing for Telecom Customer Retention.

Supports:
- Train/test splitting (legacy compatibility)
- Train/validation/test splitting
- Ordinal, binary and one-hot encoding
- Scaling fitted on training data only
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import engineer_features
from src.cleaning import clean_data
from src.validation import REQUIRED_FEATURES, validate_raw_data


CONTRACT_ORDER = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}

BINARY_YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

NOMINAL_ONEHOT_COLS = [
    "gender",
    "InternetService",
    "PaymentMethod",
]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_services_subscribed",
]


@dataclass
class PreprocessedData:
    """Container for processed features, labels and preprocessing metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    customer_ids_train: pd.Series
    customer_ids_test: pd.Series
    scaler: StandardScaler
    feature_columns: list

    # Optional validation fields for backward compatibility
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    customer_ids_val: pd.Series | None = None


def prepare_raw_data(
    df: pd.DataFrame,
    require_target: bool = True,
) -> pd.DataFrame:
    """Validate raw records, normalize numeric types and clean data."""
    validate_raw_data(df, require_target=require_target)
    allowed = set(REQUIRED_FEATURES) | {"customerID", "Churn"}
    df = df.loc[:, [col for col in df.columns if col in allowed]].copy()
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return clean_data(df)


def _prepare_for_split(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare cleaned data for stratified splitting."""
    df = df.copy()

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    if df["Churn"].isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No'.")

    df = engineer_features(df)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Original train/test split.

    Retained for compatibility with existing project code.
    """
    df = _prepare_for_split(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Churn"],
    )

    return train_df, test_df


def split_train_validation_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified train/validation/test split.

    Defaults:
        Training:   60%
        Validation: 20%
        Test:       20%

    No scaler or fitted encoder is applied before splitting.
    """
    if (
        not 0 < test_size < 1
        or not 0 < validation_size < 1
        or test_size + validation_size >= 1
    ):
        raise ValueError(
            "test_size and validation_size must each be between 0 and 1, "
            "and their sum must be less than 1."
        )

    df = _prepare_for_split(df)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Churn"],
    )

    relative_val_size = validation_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df["Churn"],
    )

    return train_df, val_df, test_df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic categorical encoding."""
    df = df.copy()

    # Ordinal encoding
    df["Contract"] = df["Contract"].map(CONTRACT_ORDER)

    # Binary encoding
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0,
        })

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=NOMINAL_ONEHOT_COLS,
        drop_first=False,
    )

    return df


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list,
) -> pd.DataFrame:
    """Apply fitted preprocessing to cleaned, engineered records."""
    encoded = _encode_categoricals(df)
    X = encoded.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    missing = set(feature_columns) - set(X.columns)
    unexpected_missing = {
        col for col in missing
        if not any(col.startswith(f"{nominal}_") for nominal in NOMINAL_ONEHOT_COLS)
    }
    if unexpected_missing:
        raise ValueError(f"Missing model features: {sorted(unexpected_missing)}")
    X = X.reindex(columns=feature_columns, fill_value=0)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    if X.isna().any().any():
        raise ValueError("Missing values detected after preprocessing.")
    return X


def fit_transform_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
) -> PreprocessedData:
    """Fit on training data only and share transformations across splits."""
    train_encoded = _encode_categoricals(train_df)
    feature_columns = [
        col for col in train_encoded.columns
        if col not in ["Churn", "customerID"]
    ]
    scaler = StandardScaler()
    scaler.fit(train_encoded[NUMERIC_COLS])

    X_train = transform_features(train_df, scaler, feature_columns)
    X_test = transform_features(test_df, scaler, feature_columns)
    X_val = None
    y_val = None
    customer_ids_val = None

    if val_df is not None:
        X_val = transform_features(val_df, scaler, feature_columns)
        y_val = val_df["Churn"]
        customer_ids_val = val_df["customerID"]

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=train_df["Churn"],
        y_test=test_df["Churn"],
        customer_ids_train=train_df["customerID"],
        customer_ids_test=test_df["customerID"],
        scaler=scaler,
        feature_columns=feature_columns,
        X_val=X_val,
        y_val=y_val,
        customer_ids_val=customer_ids_val,
    )
