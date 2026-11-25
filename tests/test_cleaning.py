
"""Data cleaning tests using the original Telco dataset."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    clean_data,
    INTERNET_DEPENDENT_COLS,
)


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")


@pytest.fixture(scope="module")
def cleaned_data(raw_data):
    return clean_data(raw_data)


def test_total_charges_converted_to_numeric(cleaned_data):
    assert cleaned_data["TotalCharges"].dtype.kind == "f"


def test_total_charges_blank_filled_with_zero_for_zero_tenure(
    raw_data, cleaned_data
):
    blank = (
        raw_data["TotalCharges"].isna()
        | raw_data["TotalCharges"].astype(str).str.strip().eq("")
    )

    mask = blank & raw_data["tenure"].eq(0)

    assert mask.any()
    assert cleaned_data.loc[mask, "TotalCharges"].eq(0).all()


def test_total_charges_no_nans_remain(cleaned_data):
    assert cleaned_data["TotalCharges"].isna().sum() == 0


def test_no_internet_service_collapsed_to_no(
    raw_data, cleaned_data
):
    for column in INTERNET_DEPENDENT_COLS:
        mask = raw_data[column].eq("No internet service")

        assert mask.any()
        assert cleaned_data.loc[mask, column].eq("No").all()

        assert "No internet service" not in (
            cleaned_data[column].values
        )


def test_no_phone_service_collapsed_to_no(
    raw_data, cleaned_data
):
    mask = raw_data["MultipleLines"].eq("No phone service")

    assert mask.any()

    assert cleaned_data.loc[
        mask, "MultipleLines"
    ].eq("No").all()

    assert "No phone service" not in (
        cleaned_data["MultipleLines"].values
    )


def test_row_count_unchanged(raw_data, cleaned_data):
    assert len(raw_data) == len(cleaned_data)


def test_customer_ids_preserved(raw_data, cleaned_data):
    pd.testing.assert_series_equal(
        raw_data["customerID"],
        cleaned_data["customerID"],
    )


def test_cleaning_does_not_modify_original(raw_data):
    original = raw_data.copy(deep=True)

    clean_data(raw_data)

    pd.testing.assert_frame_equal(
        raw_data,
        original,
    )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "invalid",
        -100.0,
        np.inf,
        " ",
    ],
)
def test_invalid_total_charges_rejected(
    raw_data, invalid_value
):
    df = raw_data.loc[
        raw_data["tenure"].gt(0)
    ].iloc[[0]].copy()

    df["TotalCharges"] = invalid_value

    with pytest.raises(ValueError, match="TotalCharges"):
        clean_data(df)
