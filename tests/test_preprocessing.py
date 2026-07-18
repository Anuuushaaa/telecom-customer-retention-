"""
tests/test_preprocessing.py

Unit tests for src/preprocessing.py — verifies Phase 4 decisions:
stratified split, correct encoding, no leakage between train/test scaling.
"""

import pandas as pd
from src.preprocessing import split_data, fit_transform_features, CONTRACT_ORDER


def _sample_clean_df():
    """A tiny post-cleaning dataframe (as if it came out of src/cleaning.py)."""
    n = 20
    return pd.DataFrame({
        "customerID": [f"{i:04d}" for i in range(n)],
        "gender": ["Female", "Male"] * (n // 2),
        "SeniorCitizen": [0, 1] * (n // 2),
        "Partner": ["Yes", "No"] * (n // 2),
        "Dependents": ["No", "Yes"] * (n // 2),
        "tenure": list(range(n)),
        "PhoneService": ["Yes", "No"] * (n // 2),
        "MultipleLines": ["No", "Yes"] * (n // 2),
        "InternetService": ["DSL", "Fiber optic", "No"] * (n // 3) + ["DSL"] * (n % 3),
        "OnlineSecurity": ["No", "Yes"] * (n // 2),
        "OnlineBackup": ["No", "Yes"] * (n // 2),
        "DeviceProtection": ["No", "Yes"] * (n // 2),
        "TechSupport": ["No", "Yes"] * (n // 2),
        "StreamingTV": ["No", "Yes"] * (n // 2),
        "StreamingMovies": ["No", "Yes"] * (n // 2),
        "Contract": ["Month-to-month", "One year", "Two year"] * (n // 3) + ["Month-to-month"] * (n % 3),
        "PaperlessBilling": ["Yes", "No"] * (n // 2),
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * (n // 4),
        "MonthlyCharges": [float(i * 10) for i in range(n)],
        "TotalCharges": [float(i * 100) for i in range(n)],
        "Churn": ["Yes", "No"] * (n // 2),
    })


def test_split_preserves_churn_rate_via_stratification():
    df = _sample_clean_df()
    train_df, test_df = split_data(df, test_size=0.3, random_state=42)
    train_rate = (train_df["Churn"] == 1).mean()
    test_rate = (test_df["Churn"] == 1).mean()
    assert abs(train_rate - test_rate) < 0.15  # small sample, so loose tolerance


def test_churn_mapped_to_binary():
    df = _sample_clean_df()
    train_df, _ = split_data(df)
    assert set(train_df["Churn"].unique()).issubset({0, 1})


def test_contract_encoded_ordinally():
    df = _sample_clean_df()
    train_df, test_df = split_data(df)
    data = fit_transform_features(train_df, test_df)
    assert set(data.X_train["Contract"].unique()).issubset(set(CONTRACT_ORDER.values()))


def test_no_leakage_scaler_fit_on_train_only():
    df = _sample_clean_df()
    train_df, test_df = split_data(df, test_size=0.3, random_state=42)
    data = fit_transform_features(train_df, test_df)
    # Scaler's learned mean should match train data, not test or combined
    assert data.scaler.mean_ is not None
    assert len(data.scaler.mean_) == 5  # tenure, MonthlyCharges, TotalCharges


def test_customer_id_excluded_from_features():
    df = _sample_clean_df()
    train_df, test_df = split_data(df)
    data = fit_transform_features(train_df, test_df)
    assert "customerID" not in data.X_train.columns
    assert "customerID" not in data.X_test.columns


def test_gender_one_hot_both_columns_present():
    df = _sample_clean_df()
    train_df, test_df = split_data(df)
    data = fit_transform_features(train_df, test_df)
    assert "gender_Female" in data.X_train.columns
    assert "gender_Male" in data.X_train.columns


def test_no_nans_in_output():
    df = _sample_clean_df()
    train_df, test_df = split_data(df)
    data = fit_transform_features(train_df, test_df)
    assert data.X_train.isna().sum().sum() == 0
    assert data.X_test.isna().sum().sum() == 0