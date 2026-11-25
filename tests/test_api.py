
"""
Automated tests for the Telecom Customer
Retention FastAPI application.
"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.config import get_settings
from api.main import create_app

from api.model_service import (
    ModelService,
    ModelLoadingError,
    ModelNotReadyError,
    ModelPredictionError,
    BatchSizeError,
)

from api.schemas import (
    CustomerInput,
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    ModelInfoResponse,
)


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "telco_churn.csv"
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def original_customers():

    dataframe = pd.read_csv(DATA_PATH)

    return dataframe.drop(
        columns=["Churn"]
    ).to_dict(orient="records")


@pytest.fixture
def customer(original_customers):

    return original_customers[0].copy()


@pytest.fixture
def settings():

    return replace(
        get_settings(),
        max_batch_size=100,
        classification_threshold=0.5,
    )


@pytest.fixture
def model_info(settings):

    return ModelInfoResponse(
        registered_model_name=(
            settings.registered_model_name
        ),
        model_alias=settings.model_alias,
        model_version="2",
        model_uri=(
            f"models:/{settings.registered_model_name}/2"
        ),
        model_name="xgboost",
        calibration_method="sigmoid",
        classification_threshold=0.5,
        run_id="test-run",
        max_batch_size=settings.max_batch_size,
    )


@pytest.fixture
def service(settings, model_info):

    instance = ModelService(settings)

    model = Mock()

    def predict(dataframe, params=None):

        threshold = params[
            "classification_threshold"
        ]

        probability = 0.7

        prediction = int(
            probability >= threshold
        )

        return pd.DataFrame({

            "customerID": (
                dataframe["customerID"].tolist()
            ),

            "raw_churn_probability": 0.8,

            "churn_probability": probability,

            "prediction": prediction,

            "prediction_label": (
                "Churn"
                if prediction
                else "No Churn"
            ),

            "classification_threshold": threshold,

            "model_name": "xgboost",

            "calibration_method": "sigmoid",
        })

    model.predict.side_effect = predict

    def load_model():

        instance._model = model

        instance._model_info = model_info

        return model_info

    instance.load_model = Mock(
        side_effect=load_model
    )

    return instance


@pytest.fixture
def client(settings, service):

    application = create_app(
        settings=settings,
        model_service=service,
    )

    with TestClient(application) as test_client:

        yield test_client


# ============================================================
# CUSTOMER SCHEMA TESTS
# ============================================================

def test_valid_customer(customer):

    result = CustomerInput.model_validate(
        customer
    )

    assert result.customerID == (
        customer["customerID"]
    )


def test_original_dataset_validation(
    original_customers,
):

    validated = [

        CustomerInput.model_validate(record)

        for record in original_customers
    ]

    assert len(validated) == 7043


@pytest.mark.parametrize(
    "field,value",
    [
        ("customerID", ""),
        ("customerID", "   "),
        ("gender", "Unknown"),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", -1),
        ("SeniorCitizen", True),
        ("SeniorCitizen", "1"),
        ("Partner", "Maybe"),
        ("Dependents", "Unknown"),
        ("tenure", -1),
        ("tenure", 1.5),
        ("tenure", True),
        ("PhoneService", "Invalid"),
        ("MultipleLines", "Invalid"),
        ("InternetService", "Invalid"),
        ("OnlineSecurity", "Invalid"),
        ("OnlineBackup", "Invalid"),
        ("DeviceProtection", "Invalid"),
        ("TechSupport", "Invalid"),
        ("StreamingTV", "Invalid"),
        ("StreamingMovies", "Invalid"),
        ("Contract", "Invalid"),
        ("PaperlessBilling", "Invalid"),
        ("PaymentMethod", "Invalid"),
        ("MonthlyCharges", -10.0),
        ("MonthlyCharges", True),
        ("MonthlyCharges", float("inf")),
        ("MonthlyCharges", float("nan")),
        ("TotalCharges", "-100"),
        ("TotalCharges", "invalid"),
        ("TotalCharges", True),
        ("TotalCharges", "nan"),
        ("TotalCharges", "inf"),
    ],
)
def test_invalid_customer_fields(
    customer,
    field,
    value,
):

    customer[field] = value

    with pytest.raises(ValidationError):

        CustomerInput.model_validate(
            customer
        )


@pytest.mark.parametrize(
    "field",
    list(CustomerInput.model_fields),
)
def test_missing_customer_field(
    customer,
    field,
):

    customer.pop(field)

    with pytest.raises(ValidationError):

        CustomerInput.model_validate(
            customer
        )


def test_unexpected_customer_field(customer):

    customer["unexpected"] = "invalid"

    with pytest.raises(ValidationError):

        CustomerInput.model_validate(
            customer
        )


def test_target_column_rejected(customer):

    customer["Churn"] = "Yes"

    with pytest.raises(ValidationError):

        CustomerInput.model_validate(
            customer
        )


def test_zero_tenure_blank_charges(customer):

    customer["tenure"] = 0

    customer["TotalCharges"] = " "

    result = CustomerInput.model_validate(
        customer
    )

    assert result.TotalCharges == ""


def test_nonzero_tenure_blank_charges(customer):

    customer["tenure"] = 10

    customer["TotalCharges"] = " "

    with pytest.raises(ValidationError):

        CustomerInput.model_validate(
            customer
        )


@pytest.mark.parametrize(
    "value",
    [100, 100.0, "100.0"],
)
def test_total_charges_normalization(
    customer,
    value,
):

    customer["TotalCharges"] = value

    result = CustomerInput.model_validate(
        customer
    )

    assert result.TotalCharges == "100.0"


# ============================================================
# REQUEST SCHEMA TESTS
# ============================================================

@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, True, "0.5", float("nan")],
)
def test_invalid_threshold(
    customer,
    threshold,
):

    with pytest.raises(ValidationError):

        PredictionRequest(

            customer=customer,

            classification_threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.25, 0.5, 1.0, None],
)
def test_valid_threshold(
    customer,
    threshold,
):

    request = PredictionRequest(

        customer=customer,

        classification_threshold=threshold,
    )

    assert (
        request.classification_threshold
        == threshold
    )


def test_empty_batch():

    with pytest.raises(ValidationError):

        BatchPredictionRequest(
            customers=[]
        )


def test_duplicate_batch_customers(customer):

    with pytest.raises(ValidationError):

        BatchPredictionRequest(
            customers=[customer, customer]
        )


# ============================================================
# HEALTH AND METADATA ENDPOINTS
# ============================================================

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert (
        response.json()["documentation"]
        == "/docs"
    )


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == (
        "healthy"
    )


def test_readiness_endpoint(client):

    response = client.get("/ready")

    assert response.status_code == 200

    assert response.json()["model_loaded"]

    assert response.json()["status"] == (
        "ready"
    )


def test_model_information(client):

    response = client.get("/model/info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_name"] == "xgboost"

    assert result["model_version"] == "2"

    assert result["calibration_method"] == (
        "sigmoid"
    )


def test_swagger_documentation(client):

    response = client.get("/docs")

    assert response.status_code == 200


def test_redoc_documentation(client):

    response = client.get("/redoc")

    assert response.status_code == 200


def test_openapi_endpoints(client):

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    expected = {

        "/health",

        "/ready",

        "/model/info",

        "/predict",

        "/predict/batch",
    }

    assert expected.issubset(paths)


# ============================================================
# SINGLE PREDICTION ENDPOINT TESTS
# ============================================================

def test_single_prediction(client, customer):

    response = client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["customerID"] == (
        customer["customerID"]
    )

    assert result["prediction"] == 1

    assert result["prediction_label"] == (
        "Churn"
    )

    assert result["churn_probability"] == 0.7


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0.0, 1),
        (0.5, 1),
        (0.7, 1),
        (0.8, 0),
        (1.0, 0),
    ],
)
def test_prediction_threshold_override(
    client,
    customer,
    threshold,
    expected,
):

    response = client.post(

        "/predict",

        json={

            "customer": customer,

            "classification_threshold": threshold,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == expected

    assert (
        result["classification_threshold"]
        == threshold
    )


def test_missing_prediction_customer(client):

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_invalid_prediction_customer(
    client,
    customer,
):

    customer["gender"] = "Invalid"

    response = client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 422


# ============================================================
# BATCH PREDICTION ENDPOINT TESTS
# ============================================================

@pytest.mark.parametrize(
    "batch_size",
    [1, 10, 50, 100],
)
def test_batch_prediction(
    client,
    original_customers,
    batch_size,
):

    customers = original_customers[:batch_size]

    response = client.post(

        "/predict/batch",

        json={"customers": customers},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_customers"] == (
        batch_size
    )

    assert len(result["predictions"]) == (
        batch_size
    )

    assert result["predicted_churn"] == (
        batch_size
    )


def test_batch_customer_alignment(
    client,
    original_customers,
):

    customers = original_customers[:10]

    response = client.post(

        "/predict/batch",

        json={"customers": customers},
    )

    assert response.status_code == 200

    actual_ids = [

        item["customerID"]

        for item in response.json()["predictions"]
    ]

    expected_ids = [

        item["customerID"]

        for item in customers
    ]

    assert actual_ids == expected_ids


def test_batch_limit_exceeded(
    client,
    original_customers,
):

    response = client.post(

        "/predict/batch",

        json={
            "customers": original_customers[:101]
        },
    )

    assert response.status_code == 413

    assert response.json()["error"] == (
        "BATCH_LIMIT_EXCEEDED"
    )


def test_empty_batch_endpoint(client):

    response = client.post(

        "/predict/batch",

        json={"customers": []},
    )

    assert response.status_code == 422


def test_duplicate_batch_endpoint(
    client,
    customer,
):

    response = client.post(

        "/predict/batch",

        json={
            "customers": [customer, customer]
        },
    )

    assert response.status_code == 422


def test_batch_threshold_override(
    client,
    original_customers,
):

    response = client.post(

        "/predict/batch",

        json={

            "customers": original_customers[:10],

            "classification_threshold": 1.0,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_churn"] == 0

    assert result["predicted_no_churn"] == 10


# ============================================================
# ERROR HANDLING AND MIDDLEWARE
# ============================================================

def test_request_id_header(client):

    response = client.get("/health")

    request_id = response.headers[
        "X-Request-ID"
    ]

    assert str(UUID(request_id)) == request_id


def test_unique_request_ids(client):

    first = client.get("/health")

    second = client.get("/health")

    assert (

        first.headers["X-Request-ID"]

        != second.headers["X-Request-ID"]
    )


def test_client_request_id_not_trusted(client):

    response = client.get(

        "/health",

        headers={
            "X-Request-ID": "client-supplied"
        },
    )

    assert (

        response.headers["X-Request-ID"]

        != "client-supplied"
    )


def test_response_time_header(client):

    response = client.get("/health")

    duration = float(
        response.headers["X-Response-Time-MS"]
    )

    assert duration >= 0


def test_validation_error_format(client):

    response = client.post(

        "/predict",

        json={"customer": {}},
    )

    assert response.status_code == 422

    result = response.json()

    assert set(result) == {

        "error",

        "message",

        "request_id",
    }

    assert result["error"] == (
        "VALIDATION_ERROR"
    )


def test_error_request_id_matches_header(client):

    response = client.post(

        "/predict",

        json={"customer": {}},
    )

    assert (

        response.json()["request_id"]

        == response.headers["X-Request-ID"]
    )


def test_validation_error_hides_input(
    client,
    customer,
):

    customer["gender"] = (
        "PRIVATE-INVALID-VALUE"
    )

    response = client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 422

    assert (
        "PRIVATE-INVALID-VALUE"
        not in response.text
    )


def test_unknown_endpoint(client):

    response = client.get(
        "/unknown-endpoint"
    )

    assert response.status_code == 404

    assert response.json()["error"] == (
        "NOT_FOUND"
    )


def test_method_not_allowed(client):

    response = client.get("/predict")

    assert response.status_code == 405

    assert response.json()["error"] == (
        "METHOD_NOT_ALLOWED"
    )


def test_malformed_json(client):

    response = client.post(

        "/predict",

        content="{invalid json",

        headers={
            "Content-Type": "application/json"
        },
    )

    assert response.status_code == 422


def test_prediction_failure(
    client,
    service,
    customer,
):

    service._model.predict.side_effect = (
        RuntimeError("private internal error")
    )

    response = client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 500

    assert response.json()["error"] == (
        "INTERNAL_SERVER_ERROR"
    )

    assert (
        "private internal error"
        not in response.text
    )


def test_unexpected_application_error(
    client,
    service,
):

    service.get_readiness = Mock(

        side_effect=RuntimeError(
            "private internal error"
        ),
    )

    response = client.get("/ready")

    assert response.status_code == 500

    assert response.json()["error"] == (
        "INTERNAL_SERVER_ERROR"
    )

    assert (
        "private internal error"
        not in response.text
    )


# ============================================================
# MODEL UNAVAILABILITY
# ============================================================

@pytest.fixture
def unavailable_client(settings):

    service = ModelService(settings)

    service.load_model = Mock(

        side_effect=ModelLoadingError(
            "Model unavailable"
        ),
    )

    application = create_app(

        settings=settings,

        model_service=service,
    )

    with TestClient(application) as client:

        yield client


def test_health_when_model_unavailable(
    unavailable_client,
):

    response = unavailable_client.get(
        "/health"
    )

    assert response.status_code == 200


def test_readiness_when_model_unavailable(
    unavailable_client,
):

    response = unavailable_client.get(
        "/ready"
    )

    assert response.status_code == 503

    assert response.json()["status"] == (
        "not_ready"
    )

    assert not response.json()["model_loaded"]


def test_metadata_when_model_unavailable(
    unavailable_client,
):

    response = unavailable_client.get(
        "/model/info"
    )

    assert response.status_code == 503


def test_prediction_when_model_unavailable(
    unavailable_client,
    customer,
):

    response = unavailable_client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 503


def test_batch_when_model_unavailable(
    unavailable_client,
    customer,
):

    response = unavailable_client.post(

        "/predict/batch",

        json={"customers": [customer]},
    )

    assert response.status_code == 503


# ============================================================
# MODEL SERVICE TESTS
# ============================================================

def test_service_initially_not_ready(settings):

    service = ModelService(settings)

    assert not service.is_ready


def test_service_prediction_before_loading(
    settings,
    customer,
):

    service = ModelService(settings)

    validated = CustomerInput.model_validate(
        customer
    )

    with pytest.raises(ModelNotReadyError):

        service.predict(validated)


def test_service_input_dtypes(
    service,
    customer,
):

    validated = CustomerInput.model_validate(
        customer
    )

    dataframe = service.prepare_input(
        [validated]
    )

    assert str(
        dataframe["SeniorCitizen"].dtype
    ) == "int64"

    assert str(
        dataframe["tenure"].dtype
    ) == "int64"

    assert str(
        dataframe["MonthlyCharges"].dtype
    ) == "float64"

    assert isinstance(
        dataframe["TotalCharges"].iloc[0],
        str,
    )


def test_service_empty_batch(service):

    with pytest.raises(BatchSizeError):

        service.prepare_input([])


def test_service_duplicate_customers(
    service,
    customer,
):

    validated = CustomerInput.model_validate(
        customer
    )

    with pytest.raises(ValueError):

        service.prepare_input(
            [validated, validated]
        )


def test_model_loaded_once(client, service):

    client.get("/health")

    client.get("/ready")

    client.get("/model/info")

    service.load_model.assert_called_once()


def test_service_shutdown(settings, service):

    application = create_app(
        settings=settings,
        model_service=service,
    )

    with TestClient(application):

        assert service.is_ready

    assert not service.is_ready


@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.5, 1.0],
)
def test_service_threshold_resolution(
    service,
    threshold,
):

    assert (
        service.resolve_threshold(threshold)
        == threshold
    )


def test_service_default_threshold(service):

    assert service.resolve_threshold() == 0.5


def test_invalid_model_output(
    client,
    service,
    customer,
):

    service._model.predict.side_effect = None

    service._model.predict.return_value = (
        pd.DataFrame()
    )

    response = client.post(

        "/predict",

        json={"customer": customer},
    )

    assert response.status_code == 500
