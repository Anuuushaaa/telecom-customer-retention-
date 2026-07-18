"""
tests/test_feature_engineering.py

Unit tests for src/feature_engineering.py.
"""

import pandas as pd
import pytest

from src.feature_engineering import (
    add_avg_monthly_spend,
    add_is_electronic_check,
    add_num_services_subscribed,
    add_tenure_bucket,
    engineer_features,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["C1", "C2", "C3"],
        "tenure": [0, 24, 5],
        "MonthlyCharges": [50.0, 100.0, 30.0],
        "TotalCharges": [0.0, 2400.0, 150.0],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check"],
    })


def test_num_services_subscribed_counts_correctly():
    df = add_num_services_subscribed(_sample_df())
    assert df["num_services_subscribed"].tolist() == [0, 6, 0]


def test_avg_monthly_spend_normal_case():
    df = add_avg_monthly_spend(_sample_df())
    assert df.loc[1, "avg_monthly_spend"] == pytest.approx(100.0)
    assert df.loc[2, "avg_monthly_spend"] == pytest.approx(30.0)


def test_avg_monthly_spend_zero_tenure_uses_monthly_charges():
    df = add_avg_monthly_spend(_sample_df())
    assert df.loc[0, "avg_monthly_spend"] == pytest.approx(50.0)
    assert df["avg_monthly_spend"].isna().sum() == 0
    assert (df["avg_monthly_spend"] != float("inf")).all()


def test_tenure_bucket_assigns_correct_bucket_and_dtype():
    df = add_tenure_bucket(_sample_df())
    assert df.loc[0, "tenure_bucket"] == 0
    assert df.loc[2, "tenure_bucket"] == 0
    assert df["tenure_bucket"].dtype == "int64"


def test_is_electronic_check_flags_correctly():
    df = add_is_electronic_check(_sample_df())
    assert df["is_electronic_check"].tolist() == [1, 0, 1]


def test_engineer_features_adds_all_four_columns():
    df = engineer_features(_sample_df())
    expected_new_cols = {
        "num_services_subscribed", "avg_monthly_spend",
        "tenure_bucket", "is_electronic_check",
    }
    assert expected_new_cols.issubset(set(df.columns))


def test_engineer_features_preserves_row_count():
    original = _sample_df()
    engineered = engineer_features(original)
    assert len(engineered) == len(original)


def test_engineer_features_does_not_mutate_input():
    original = _sample_df()
    original_columns = set(original.columns)
    engineer_features(original)
    assert set(original.columns) == original_columns