"""
src/preprocessing.py

Phase 4: Feature engineering & preprocessing for the Telecom Churn project.

Design decisions (see README for full rationale):
- Contract: ordinal-encoded (0/1/2) — natural order exists.
- All other categoricals: one-hot encoded — no inherent order.
- Genuine binary Yes/No columns: mapped to 1/0.
- Numeric columns (tenure, MonthlyCharges, TotalCharges): StandardScaler.
- customerID: excluded from X, returned separately for lookup use (e.g. Streamlit app).
- Churn: mapped to 1/0 as the target.

IMPORTANT: split_data() must be called BEFORE fit_transform_features(),
and the scaler/encoder must be fit on the training set only, then used
to transform (not re-fit) the test set. This prevents data leakage.
"""

from dataclasses import dataclass
from src.feature_engineering import engineer_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}

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

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services_subscribed"]


@dataclass
class PreprocessedData:
    """Container for everything downstream training/prediction code needs."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    customer_ids_train: pd.Series
    customer_ids_test: pd.Series
    scaler: StandardScaler
    feature_columns: list


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split a CLEANED dataframe (post src/cleaning.py) into train/test sets,
    stratified on Churn to preserve class balance.

    Must be called before any encoding/scaling to avoid data leakage.
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = engineer_features(df)

    y = df["Churn"]
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_df, test_df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal + one-hot + binary encoding. No fitted state involved,
    so this is safe to call identically on train and test."""
    df = df.copy()

    # Ordinal: Contract
    df["Contract"] = df["Contract"].map(CONTRACT_ORDER)

    # Binary Yes/No -> 1/0
    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Nominal -> one-hot
    df = pd.get_dummies(df, columns=NOMINAL_ONEHOT_COLS, drop_first=False)

    return df


def fit_transform_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> PreprocessedData:
    """
    Fit encoders/scalers on train_df ONLY, then transform both train_df
    and test_df. Returns a PreprocessedData bundle ready for modeling.
    """
    train_df = _encode_categoricals(train_df)
    test_df = _encode_categoricals(test_df)

    # Align columns in case one-hot produced different columns in each split
    # (e.g. a category present in train but not test). Test gets zero-filled
    # for any missing dummy column; this is a transform, not a fit, so no leakage.
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Separate target, customerID, and features
    y_train = train_df["Churn"]
    y_test = test_df["Churn"]

    customer_ids_train = train_df["customerID"]
    customer_ids_test = test_df["customerID"]

    drop_cols = ["Churn", "customerID"]
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    # Scale numeric columns: fit on train, transform both
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        customer_ids_train=customer_ids_train,
        customer_ids_test=customer_ids_test,
        scaler=scaler,
        feature_columns=list(X_train.columns),
    )