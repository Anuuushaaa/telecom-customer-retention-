
"""
Production inference and training-serving consistency tests.

Uses original Telco customer records and the saved
production model artifacts.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from predict import predict_customer

from src.config import (
    CLASSIFICATION_THRESHOLD,
    PRODUCTION_MODEL_PATH,
)

from src.preprocessing import (
    prepare_raw_data,
    fit_transform_features,
    split_train_validation_test,
)


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")


@pytest.fixture(scope="module")
def pipeline_artifacts(raw_data):
    """Rebuild preprocessing without retraining the model."""

    df_clean = prepare_raw_data(raw_data)

    train_df, val_df, test_df = (
        split_train_validation_test(df_clean)
    )

    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    model = joblib.load(PRODUCTION_MODEL_PATH)

    return data, model


@pytest.fixture
def raw_customer(raw_data):
    """Return a real customer without the target label."""

    return raw_data.iloc[0].drop(
        labels=["Churn"]
    ).to_dict()


@pytest.mark.parametrize(
    "position",
    [0, 10, 100, 500, 1000],
)
def test_predict_matches_training_pipeline_probability(
    raw_data,
    pipeline_artifacts,
    position,
):
    data, model = pipeline_artifacts

    customer_id = data.customer_ids_test.iloc[position]

    X_sample = data.X_test.iloc[[position]]

    expected_probability = float(
        model.predict_proba(X_sample)[0, 1]
    )

    raw_customer = raw_data.loc[
        raw_data["customerID"].eq(customer_id)
    ].iloc[0].drop(labels=["Churn"]).to_dict()

    result = predict_customer(raw_customer)

    assert result["churn_probability"] == pytest.approx(
        round(expected_probability, 4),
        abs=1e-8,
        rel=0,
    )

    expected_prediction = (
        "Churn"
        if expected_probability >= CLASSIFICATION_THRESHOLD
        else "No Churn"
    )

    assert result["prediction"] == expected_prediction


def test_predict_output_has_expected_keys(raw_customer):
    result = predict_customer(raw_customer)

    expected_keys = {
        "customerID",
        "churn_probability",
        "prediction",
        "threshold_used",
        "top_contributors",
    }

    assert expected_keys.issubset(result.keys())

    assert result["customerID"] == raw_customer["customerID"]

    assert 0.0 <= result["churn_probability"] <= 1.0

    assert result["prediction"] in {
        "Churn",
        "No Churn",
    }

    assert result["threshold_used"] == CLASSIFICATION_THRESHOLD

    contributors = result["top_contributors"]

    assert isinstance(contributors, dict)
    assert len(contributors) == 5

    assert all(
        np.isfinite(value)
        for value in contributors.values()
    )


def test_numeric_string_inputs(raw_customer):
    expected = predict_customer(raw_customer)

    string_customer = raw_customer.copy()

    for column in [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]:
        string_customer[column] = str(
            string_customer[column]
        )

    actual = predict_customer(string_customer)

    assert actual == expected


def test_missing_required_feature_rejected(raw_customer):
    customer = raw_customer.copy()

    customer.pop("MonthlyCharges")

    with pytest.raises(
        ValueError,
        match="MonthlyCharges",
    ):
        predict_customer(customer)


def test_unknown_category_rejected(raw_customer):
    customer = raw_customer.copy()

    customer["Contract"] = "Invalid contract"

    with pytest.raises(ValueError, match="Contract"):
        predict_customer(customer)


def test_prediction_without_customer_id(raw_customer):
    customer = raw_customer.copy()

    customer.pop("customerID")

    result = predict_customer(customer)

    assert result["customerID"] == "unknown"

    assert 0.0 <= result["churn_probability"] <= 1.0


def test_zero_tenure_blank_total_charges(raw_data):
    customer = raw_data.loc[
        raw_data["tenure"].eq(0)
    ].iloc[0].drop(labels=["Churn"]).to_dict()

    customer["TotalCharges"] = " "

    result = predict_customer(customer)

    assert 0.0 <= result["churn_probability"] <= 1.0

    assert result["prediction"] in {
        "Churn",
        "No Churn",
    }
