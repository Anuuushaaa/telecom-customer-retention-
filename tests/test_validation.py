
"""Schema validation tests using the original Telco dataset."""

import numpy as np
import pandas as pd
import pytest

from src.validation import validate_raw_data


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")


def test_original_dataset_is_valid(raw_data):
    validate_raw_data(raw_data)


def test_validation_does_not_modify_data(raw_data):
    original = raw_data.copy(deep=True)
    validate_raw_data(raw_data)
    pd.testing.assert_frame_equal(raw_data, original)


def test_empty_dataset_rejected(raw_data):
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(raw_data.iloc[:0])


def test_missing_feature_rejected(raw_data):
    df = raw_data.drop(columns=["tenure"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(df)


def test_duplicate_customer_id_rejected(raw_data):
    df = raw_data.iloc[:2].copy()

    df.iloc[1, df.columns.get_loc("customerID")] = (
        df.iloc[0]["customerID"]
    )

    with pytest.raises(ValueError, match="Duplicate customer IDs"):
        validate_raw_data(df)


def test_missing_customer_id_rejected(raw_data):
    df = raw_data.iloc[[0]].copy()
    df["customerID"] = None

    with pytest.raises(ValueError, match="Missing customer IDs"):
        validate_raw_data(df)


def test_invalid_category_rejected(raw_data):
    df = raw_data.iloc[[0]].copy()
    df["Contract"] = "Invalid contract"

    with pytest.raises(ValueError, match="Contract"):
        validate_raw_data(df)


def test_invalid_target_rejected(raw_data):
    df = raw_data.iloc[[0]].copy()
    df["Churn"] = "Unknown"

    with pytest.raises(ValueError, match="Churn"):
        validate_raw_data(df)


@pytest.mark.parametrize(
    "column,value",
    [
        ("tenure", -1),
        ("tenure", 2.5),
        ("MonthlyCharges", -100),
        ("MonthlyCharges", np.inf),
        ("SeniorCitizen", 2),
        ("TotalCharges", "invalid"),
    ],
)
def test_invalid_numeric_rejected(raw_data, column, value):
    df = raw_data.iloc[[0]].copy()
    df[column] = value

    with pytest.raises(ValueError, match=f"(?i){column}"):
        validate_raw_data(df)


def test_blank_total_charges_zero_tenure(raw_data):
    df = raw_data.loc[
        raw_data["tenure"].eq(0)
    ].iloc[[0]].copy()

    df["TotalCharges"] = " "

    validate_raw_data(df)


def test_blank_total_charges_positive_tenure_rejected(raw_data):
    df = raw_data.loc[
        raw_data["tenure"].gt(0)
    ].iloc[[0]].copy()

    df["TotalCharges"] = " "

    with pytest.raises(ValueError, match="TotalCharges"):
        validate_raw_data(df)


def test_inference_without_target(raw_data):
    df = raw_data.iloc[[0]].drop(columns=["Churn"])

    validate_raw_data(df, require_target=False)


def test_inference_without_customer_id(raw_data):
    df = raw_data.iloc[[0]].drop(
        columns=["Churn", "customerID"]
    )

    validate_raw_data(df, require_target=False)


def test_training_requires_target(raw_data):
    df = raw_data.drop(columns=["Churn"])

    with pytest.raises(ValueError, match="Churn"):
        validate_raw_data(df)
