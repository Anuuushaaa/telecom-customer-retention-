"""
tests/test_cleaning.py

Unit tests for src/cleaning.py — verifies every Phase 3 cleaning decision
holds on a small, hand-built dataframe (not the full dataset, so tests
stay fast and the expected values are easy to eyeball).
"""

import pandas as pd
from src.cleaning import clean_data


def _sample_raw_df():
    """A tiny raw dataframe covering every edge case cleaning must handle."""
    return pd.DataFrame({
        "customerID": ["0001", "0002", "0003"],
        "tenure": [0, 5, 12],
        "TotalCharges": [" ", "150.5", "600.0"],  # blank string = zero-tenure case
        "InternetService": ["No", "DSL", "Fiber optic"],
        "OnlineSecurity": ["No internet service", "Yes", "No"],
        "OnlineBackup": ["No internet service", "No", "Yes"],
        "DeviceProtection": ["No internet service", "Yes", "No"],
        "TechSupport": ["No internet service", "No", "No"],
        "StreamingTV": ["No internet service", "Yes", "Yes"],
        "StreamingMovies": ["No internet service", "No", "Yes"],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
    })


def test_total_charges_converted_to_numeric():
    df = clean_data(_sample_raw_df())
    assert df["TotalCharges"].dtype.kind == "f"  # float


def test_total_charges_blank_filled_with_zero_for_zero_tenure():
    df = clean_data(_sample_raw_df())
    # row 0 has tenure == 0 and blank TotalCharges -> should become 0.0
    assert df.loc[0, "TotalCharges"] == 0.0


def test_total_charges_no_nans_remain():
    df = clean_data(_sample_raw_df())
    assert df["TotalCharges"].isna().sum() == 0


def test_no_internet_service_collapsed_to_no():
    df = clean_data(_sample_raw_df())
    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in internet_cols:
        assert "No internet service" not in df[col].values
        # row 0 originally had "No internet service" -> should now be "No"
        assert df.loc[0, col] == "No"


def test_no_phone_service_collapsed_to_no():
    df = clean_data(_sample_raw_df())
    assert "No phone service" not in df["MultipleLines"].values
    assert df.loc[1, "MultipleLines"] == "No"


def test_row_count_unchanged():
    raw = _sample_raw_df()
    df = clean_data(raw)
    assert len(df) == len(raw)


def test_customer_id_not_dropped():
    df = clean_data(_sample_raw_df())
    assert "customerID" in df.columns