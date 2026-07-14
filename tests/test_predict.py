"""
tests/test_predict.py

Skew check: confirms predict.py produces IDENTICAL probabilities to the
training pipeline for real test-set customers. This is the strongest
possible proof against training-serving skew — not just "predict.py runs
without crashing," but "predict.py agrees with the training pipeline on
the same data."
"""

import joblib
import pandas as pd
import pytest

from predict import predict_customer
from src.cleaning import clean_data
from src.preprocessing import fit_transform_features, split_data


@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Rebuild the training pipeline once for all tests in this module."""
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)
    train_df, test_df = split_data(df_clean)
    data = fit_transform_features(train_df, test_df)
    model = joblib.load("src/../models/logistic_regression.pkl")
    return df_raw, test_df, data, model


def test_predict_matches_training_pipeline_probability(pipeline_artifacts):
    df_raw, test_df, data, model = pipeline_artifacts

    # Pick a real customer from the test set
    sample_idx = 0
    customer_id = data.customer_ids_test.iloc[sample_idx]

    # Probability from the TRAINING pipeline
    X_sample = data.X_test.iloc[[sample_idx]]
    training_pipeline_proba = model.predict_proba(X_sample)[0, 1]

    # Same customer's RAW row, run through predict.py's inference pipeline
    raw_row = df_raw[df_raw["customerID"] == customer_id].iloc[0]
    raw_customer_dict = raw_row.drop(labels=["Churn"]).to_dict()

    result = predict_customer(raw_customer_dict)
    predict_py_proba = result["churn_probability"]

    # Should match to at least 4 decimal places — any drift here would
    # indicate training-serving skew (different encoding, scaling, or
    # column alignment between the two code paths).
    assert abs(training_pipeline_proba - predict_py_proba) < 1e-4, (
        f"Skew detected: training pipeline gave {training_pipeline_proba}, "
        f"predict.py gave {predict_py_proba}"
    )


def test_predict_output_has_expected_keys():
    sample_customer = {
        "customerID": "TEST-0002",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": "3600.0",
    }
    result = predict_customer(sample_customer)
    expected_keys = {
        "customerID", "churn_probability", "prediction",
        "threshold_used", "top_contributors",
    }
    assert expected_keys.issubset(result.keys())
    assert 0.0 <= result["churn_probability"] <= 1.0