
"""Automated tests for SHAP explanations and reporting."""

from copy import deepcopy
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.special import expit
from xgboost import XGBClassifier

from api.schemas import CustomerInput
from api.explanation_service import (
    ExplanationService,
    ExplanationUnavailableError,
    calculate_shap_values,
)
from api.explanation_routes import (
    router,
    get_explanation_service,
)
from api.model_service import (
    ModelNotReadyError,
    ModelPredictionError,
    BatchSizeError,
)
from frontend.api_client import (
    TelecomAPIClient,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
)
from src.shap_reporting import (
    aggregate_global_explanations,
)
from src.optimization import load_training_partitions
from src.preprocessing import fit_transform_features


# ============================================================
# ORIGINAL DATASET FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def original_data():

    return pd.read_csv(
        "data/raw/telco_churn.csv",
        keep_default_na=False,
    )


@pytest.fixture
def customer(original_data):

    record = original_data.drop(
        columns=["Churn"]
    ).iloc[0].to_dict()

    return CustomerInput.model_validate(
        record
    ).model_dump(mode="json")


@pytest.fixture(scope="module")
def trained_model():

    train, validation, _ = (
        load_training_partitions()
    )

    processed = fit_transform_features(
        train_df=train,
        test_df=validation,
    )

    model = XGBClassifier(
        n_estimators=20,
        max_depth=2,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )

    model.fit(
        processed.X_train,
        processed.y_train,
    )

    return model, processed.X_test.head(100)


@pytest.fixture(scope="module")
def shap_result(trained_model):

    model, features = trained_model

    return calculate_shap_values(
        model,
        features,
    )


# ============================================================
# SHAP MATHEMATICAL CORRECTNESS
# ============================================================

def test_shap_shape(trained_model, shap_result):

    _, features = trained_model

    assert shap_result["values"].shape == features.shape


def test_shap_feature_count(shap_result):

    assert shap_result["values"].shape[1] == 29


def test_shap_customer_count(shap_result):

    assert shap_result["values"].shape[0] == 100


@pytest.mark.parametrize(
    "key",
    [
        "values",
        "base_values",
        "raw_margins",
        "reconstructed_margins",
        "additivity_errors",
    ],
)
def test_shap_values_are_finite(shap_result, key):

    assert np.isfinite(
        shap_result[key]
    ).all()


def test_shap_additivity(shap_result):

    reconstructed = (
        shap_result["base_values"]
        + shap_result["values"].sum(axis=1)
    )

    np.testing.assert_allclose(
        reconstructed,
        shap_result["raw_margins"],
        atol=1e-4,
        rtol=1e-5,
    )


def test_shap_probability_reconstruction(
    trained_model,
    shap_result,
):

    model, features = trained_model

    probabilities = model.predict_proba(
        features
    )[:, 1]

    reconstructed = expit(
        shap_result["reconstructed_margins"]
    )

    np.testing.assert_allclose(
        reconstructed,
        probabilities,
        atol=1e-5,
        rtol=1e-5,
    )


def test_shap_errors_nonnegative(shap_result):

    assert (
        shap_result["additivity_errors"] >= 0
    ).all()


def test_shap_single_customer(trained_model):

    model, features = trained_model

    result = calculate_shap_values(
        model,
        features.iloc[[0]],
    )

    assert result["values"].shape == (1, 29)


def test_shap_rejects_empty_dataframe(trained_model):

    model, features = trained_model

    with pytest.raises(ValueError):

        calculate_shap_values(
            model,
            features.iloc[:0],
        )


def test_shap_rejects_non_dataframe(trained_model):

    model, features = trained_model

    with pytest.raises(TypeError):

        calculate_shap_values(
            model,
            features.to_numpy(),
        )


def test_shap_batch_consistency(
    trained_model,
    shap_result,
):

    model, features = trained_model

    single = calculate_shap_values(
        model,
        features.iloc[[0]],
    )

    np.testing.assert_allclose(
        single["values"][0],
        shap_result["values"][0],
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# GLOBAL AGGREGATION FIXTURES
# ============================================================

def make_global_result(
    values,
    feature_names,
    version="2",
    error=0.000001,
):

    return {
        "model_name": "xgboost",
        "model_version": version,
        "explanation_method": "TreeSHAP",
        "output_space": "raw_log_odds",
        "total_customers": len(values),
        "total_features": len(feature_names),
        "max_additivity_error": error,
        "feature_importance": [
            {
                "feature": str(feature),
                "mean_absolute_shap": float(
                    np.abs(values[:, index]).mean()
                ),
                "mean_signed_shap": float(
                    values[:, index].mean()
                ),
            }
            for index, feature in enumerate(
                feature_names
            )
        ],
    }


@pytest.fixture
def global_batches(
    trained_model,
    shap_result,
):

    _, features = trained_model

    values = shap_result["values"]

    return [
        make_global_result(
            values[:70],
            features.columns,
            error=0.000001,
        ),
        make_global_result(
            values[70:],
            features.columns,
            error=0.000002,
        ),
    ]


def aggregate(results):

    return aggregate_global_explanations(
        results=results,
        expected_customers=100,
        expected_model_version="2",
    )


# ============================================================
# WEIGHTED GLOBAL AGGREGATION
# ============================================================

def test_weighted_aggregation_matches_full_dataset(
    global_batches,
    trained_model,
    shap_result,
):

    _, features = trained_model

    importance, _ = aggregate(global_batches)

    importance = importance.set_index(
        "feature"
    ).loc[features.columns]

    expected = np.abs(
        shap_result["values"]
    ).mean(axis=0)

    np.testing.assert_allclose(
        importance["mean_absolute_shap"],
        expected,
        atol=1e-10,
    )


def test_weighted_signed_aggregation(
    global_batches,
    trained_model,
    shap_result,
):

    _, features = trained_model

    importance, _ = aggregate(global_batches)

    importance = importance.set_index(
        "feature"
    ).loc[features.columns]

    expected = shap_result["values"].mean(
        axis=0
    )

    np.testing.assert_allclose(
        importance["mean_signed_shap"],
        expected,
        atol=1e-10,
    )


def test_global_feature_count(global_batches):

    importance, _ = aggregate(global_batches)

    assert len(importance) == 29


def test_global_rank_order(global_batches):

    importance, _ = aggregate(global_batches)

    assert importance["rank"].tolist() == list(
        range(1, 30)
    )


def test_global_importance_sorted(global_batches):

    importance, _ = aggregate(global_batches)

    assert importance[
        "mean_absolute_shap"
    ].is_monotonic_decreasing


def test_global_maximum_error(global_batches):

    _, error = aggregate(global_batches)

    assert error == pytest.approx(0.000002)


def test_global_empty_results():

    with pytest.raises(ValueError):

        aggregate([])


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "logistic_regression"),
        ("model_version", "3"),
        ("explanation_method", "unknown"),
        ("output_space", "probability"),
        ("total_customers", 0),
        ("total_customers", -1),
        ("total_customers", True),
        ("total_customers", 31),
        ("total_features", 28),
        ("max_additivity_error", -1.0),
        ("max_additivity_error", float("nan")),
        ("max_additivity_error", float("inf")),
    ],
)
def test_global_rejects_invalid_metadata(
    global_batches,
    field,
    value,
):

    results = deepcopy(global_batches)

    results[1][field] = value

    with pytest.raises(ValueError):

        aggregate(results)


def test_global_rejects_duplicate_features(
    global_batches,
):

    results = deepcopy(global_batches)

    importance = results[1]["feature_importance"]

    importance[1]["feature"] = importance[0]["feature"]

    with pytest.raises(ValueError):

        aggregate(results)


def test_global_rejects_different_features(
    global_batches,
):

    results = deepcopy(global_batches)

    results[1]["feature_importance"][0][
        "feature"
    ] = "unexpected_feature"

    with pytest.raises(ValueError):

        aggregate(results)


@pytest.mark.parametrize(
    "value",
    [-1.0, float("nan"), float("inf")],
)
def test_global_rejects_invalid_importance(
    global_batches,
    value,
):

    results = deepcopy(global_batches)

    results[1]["feature_importance"][0][
        "mean_absolute_shap"
    ] = value

    with pytest.raises(ValueError):

        aggregate(results)


def test_global_rejects_inconsistent_signed_mean(
    global_batches,
):

    results = deepcopy(global_batches)

    feature = results[1]["feature_importance"][0]

    feature["mean_signed_shap"] = (
        feature["mean_absolute_shap"] + 1
    )

    with pytest.raises(ValueError):

        aggregate(results)


# ============================================================
# EXPLANATION SERVICE VALIDATION
# ============================================================

def test_explanation_model_not_ready():

    service = Mock()

    service.is_ready = False

    explainer = ExplanationService(service)

    with pytest.raises(ModelNotReadyError):

        explainer.get_python_model()


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "tensorflow_ann"],
)
def test_unsupported_explanation_model(model_name):

    service = Mock()

    service.is_ready = True

    service.get_model_info.return_value.model_name = (
        model_name
    )

    explainer = ExplanationService(service)

    with pytest.raises(ExplanationUnavailableError):

        explainer.get_python_model()


def test_explanation_model_metadata_mismatch():

    service = Mock()

    service.is_ready = True

    service.get_model_info.return_value.model_name = (
        "xgboost"
    )

    service._model.unwrap_python_model.return_value.model_name = (
        "logistic_regression"
    )

    explainer = ExplanationService(service)

    with pytest.raises(ModelPredictionError):

        explainer.get_python_model()


@pytest.mark.parametrize(
    "top_n",
    [0, -1, 101, True, 2.5, "10"],
)
def test_explanation_rejects_invalid_top_n(
    customer,
    top_n,
):

    explainer = ExplanationService(Mock())

    with pytest.raises(ValueError):

        explainer.explain_customer(
            CustomerInput.model_validate(customer),
            top_n=top_n,
        )


# ============================================================
# REST ENDPOINT VALIDATION
# ============================================================

@pytest.fixture
def route_client():

    service = Mock()

    application = FastAPI()

    application.include_router(router)

    application.dependency_overrides[
        get_explanation_service
    ] = lambda: service

    with TestClient(application) as client:

        yield client, service


def test_explain_missing_customer(route_client):

    client, service = route_client

    response = client.post(
        "/explain",
        json={},
    )

    assert response.status_code == 422

    service.explain_customer.assert_not_called()


@pytest.mark.parametrize(
    "top_n",
    [0, -1, 101, True, "10", 2.5],
)
def test_explain_invalid_top_n(
    route_client,
    customer,
    top_n,
):

    client, service = route_client

    response = client.post(
        "/explain",
        json={
            "customer": customer,
            "top_n": top_n,
        },
    )

    assert response.status_code == 422

    service.explain_customer.assert_not_called()


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, True, "0.5"],
)
def test_explain_invalid_threshold(
    route_client,
    customer,
    threshold,
):

    client, service = route_client

    response = client.post(
        "/explain",
        json={
            "customer": customer,
            "classification_threshold": threshold,
        },
    )

    assert response.status_code == 422

    service.explain_customer.assert_not_called()


def test_explain_rejects_target_column(
    route_client,
    customer,
):

    client, service = route_client

    customer["Churn"] = "Yes"

    response = client.post(
        "/explain",
        json={"customer": customer},
    )

    assert response.status_code == 422

    service.explain_customer.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (ModelNotReadyError, 503),
        (ExplanationUnavailableError, 503),
        (ModelPredictionError, 500),
        (ValueError, 400),
    ],
)
def test_local_explanation_error_mapping(
    route_client,
    customer,
    error,
    status,
):

    client, service = route_client

    service.explain_customer.side_effect = error(
        "Test failure"
    )

    response = client.post(
        "/explain",
        json={"customer": customer},
    )

    assert response.status_code == status


def test_global_rejects_empty_customers(route_client):

    client, service = route_client

    response = client.post(
        "/explain/global",
        json={"customers": []},
    )

    assert response.status_code == 422

    service.explain_global.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (BatchSizeError, 413),
        (ModelNotReadyError, 503),
        (ExplanationUnavailableError, 503),
        (ModelPredictionError, 500),
        (ValueError, 400),
    ],
)
def test_global_explanation_error_mapping(
    route_client,
    customer,
    error,
    status,
):

    client, service = route_client

    service.explain_global.side_effect = error(
        "Test failure"
    )

    response = client.post(
        "/explain/global",
        json={"customers": [customer]},
    )

    assert response.status_code == status


def test_global_successful_response(
    route_client,
    customer,
    global_batches,
):

    client, service = route_client

    result = deepcopy(global_batches[0])

    result["total_customers"] = 1

    service.explain_global.return_value = result

    response = client.post(
        "/explain/global",
        json={"customers": [customer]},
    )

    assert response.status_code == 200

    assert response.json()["total_features"] == 29


# ============================================================
# FRONTEND API CLIENT
# ============================================================

def test_frontend_local_explanation_payload(customer):

    client = TelecomAPIClient()

    client.request = Mock(return_value={})

    client.explain(
        customer,
        classification_threshold=0.35,
        top_n=5,
    )

    client.request.assert_called_once_with(
        method="POST",
        endpoint="/explain",
        payload={
            "customer": customer,
            "classification_threshold": 0.35,
            "top_n": 5,
        },
    )


def test_frontend_default_explanation_payload(customer):

    client = TelecomAPIClient()

    client.request = Mock(return_value={})

    client.explain(customer)

    client.request.assert_called_once_with(
        method="POST",
        endpoint="/explain",
        payload={
            "customer": customer,
            "top_n": 10,
        },
    )


def test_frontend_global_explanation_payload(customer):

    client = TelecomAPIClient()

    client.request = Mock(return_value={})

    client.explain_global([customer])

    client.request.assert_called_once_with(
        method="POST",
        endpoint="/explain/global",
        payload={
            "customers": [customer],
        },
    )


def test_frontend_explanation_timeout(
    monkeypatch,
    customer,
):

    def fail(**kwargs):

        raise httpx.ReadTimeout(
            "Request timed out"
        )

    monkeypatch.setattr(
        httpx,
        "request",
        fail,
    )

    with pytest.raises(APITimeoutError):

        TelecomAPIClient().explain(customer)


def test_frontend_explanation_connection_error(
    monkeypatch,
    customer,
):

    def fail(**kwargs):

        raise httpx.ConnectError(
            "Connection refused"
        )

    monkeypatch.setattr(
        httpx,
        "request",
        fail,
    )

    with pytest.raises(APIConnectionError):

        TelecomAPIClient().explain(customer)


def test_frontend_explanation_api_error(
    monkeypatch,
    customer,
):

    def fail(**kwargs):

        return httpx.Response(
            status_code=503,
            json={
                "error": "MODEL_UNAVAILABLE",
                "message": "Model unavailable.",
                "request_id": "test-request",
            },
        )

    monkeypatch.setattr(
        httpx,
        "request",
        fail,
    )

    with pytest.raises(APIClientError) as error:

        TelecomAPIClient().explain(customer)

    assert error.value.status_code == 503

    assert error.value.error_code == (
        "MODEL_UNAVAILABLE"
    )

    assert error.value.request_id == "test-request"
