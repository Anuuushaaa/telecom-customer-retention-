
"""
Automated tests for MLflow tracking, inference,
model packaging, registry and promotion.

Registry operations are mocked to protect the
actual MLflow tracking database.

Customer input tests use the original dataset.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src import mlflow_config
from src import mlflow_tracking
from src import mlflow_registry
from src import mlflow_promotion

from src.mlflow_model import (
    TelecomChurnModel,
    validate_classification_threshold,
    validate_model_configuration,
)

from src.mlflow_packaging import (
    get_runtime_requirements,
)

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
    fit_transform_features,
)


# ============================================================
# FIXTURES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_churn.csv"
)


@pytest.fixture(scope="module")
def raw_data():

    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def raw_customers(raw_data):

    return (
        raw_data.head(20)
        .drop(columns=["Churn"])
        .copy()
    )


@pytest.fixture(scope="module")
def processed_data(raw_data):

    prepared = prepare_raw_data(
        raw_data,
        require_target=True,
    )

    train, validation, test = (
        split_train_validation_test(prepared)
    )

    return fit_transform_features(
        train_df=train,
        test_df=test,
        val_df=validation,
    )


@pytest.fixture
def model_configuration():

    return {
        "model_name": "logistic_regression",
        "calibration_method": "uncalibrated",
        "classification_threshold": 0.50,
    }


@pytest.fixture
def inference_wrapper(processed_data):

    wrapper = TelecomChurnModel()

    wrapper.model_name = "logistic_regression"

    wrapper.calibration_method = "uncalibrated"

    wrapper.classification_threshold = 0.50

    wrapper.scaler = processed_data.scaler

    wrapper.feature_columns = (
        processed_data.feature_columns
    )

    wrapper.calibrator = None

    return wrapper


@pytest.fixture
def candidate_metrics():

    # Recorded validation metrics from the
    # completed model comparison.

    return pd.DataFrame([
        {
            "model": "logistic_regression",
            "version": "1",
            "expected_net_profit": 18636.68387681424,
            "validation_roc_auc": 0.8388165542897,
            "brier_score": 0.137163489172382,
        },
        {
            "model": "xgboost",
            "version": "2",
            "expected_net_profit": 18750.64763205153,
            "validation_roc_auc": 0.8397065281975769,
            "brier_score": 0.137156213049172,
        },
        {
            "model": "tensorflow_ann",
            "version": "3",
            "expected_net_profit": 17766.756942376334,
            "validation_roc_auc": 0.8351352398666977,
            "brier_score": 0.1388005803065617,
        },
    ])


@pytest.fixture
def registry_client():

    """In-memory mock of MLflow alias operations."""

    client = MagicMock()

    aliases = {}

    def get_registered_model(name):

        return SimpleNamespace(
            aliases=aliases.copy()
        )

    def set_alias(name, alias, version):

        aliases[alias] = str(version)

    def delete_alias(name, alias):

        aliases.pop(alias, None)

    client.get_registered_model.side_effect = (
        get_registered_model
    )

    client.set_registered_model_alias.side_effect = (
        set_alias
    )

    client.delete_registered_model_alias.side_effect = (
        delete_alias
    )

    return client


# ============================================================
# CONFIGURATION TESTS
# ============================================================

def test_default_tracking_uri(monkeypatch):

    monkeypatch.delenv(
        "MLFLOW_TRACKING_URI",
        raising=False,
    )

    assert (
        mlflow_config.get_tracking_uri()
        == mlflow_config.DEFAULT_TRACKING_URI
    )


def test_custom_tracking_uri(monkeypatch):

    uri = "http://127.0.0.1:5999"

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        uri,
    )

    assert mlflow_config.get_tracking_uri() == uri


def test_registry_uri_defaults_to_tracking(
    monkeypatch,
):

    monkeypatch.delenv(
        "MLFLOW_REGISTRY_URI",
        raising=False,
    )

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5999",
    )

    assert (
        mlflow_config.get_registry_uri()
        == mlflow_config.get_tracking_uri()
    )


def test_custom_registry_uri(monkeypatch):

    uri = "http://127.0.0.1:5998"

    monkeypatch.setenv(
        "MLFLOW_REGISTRY_URI",
        uri,
    )

    assert mlflow_config.get_registry_uri() == uri


def test_supported_models():

    assert set(mlflow_config.SUPPORTED_MODELS) == {
        "logistic_regression",
        "xgboost",
        "tensorflow_ann",
    }


def test_supported_calibration_methods():

    assert set(
        mlflow_config.CALIBRATION_METHODS
    ) == {
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }


def test_distinct_registry_aliases():

    assert (
        mlflow_config.PRODUCTION_ALIAS
        != mlflow_config.CHALLENGER_ALIAS
    )


def test_storage_initialization(
    tmp_path,
    monkeypatch,
):

    store = tmp_path / "mlflow_store"

    artifacts = store / "artifacts"

    reports = tmp_path / "reports"

    monkeypatch.setattr(
        mlflow_config,
        "MLFLOW_STORE_DIR",
        store,
    )

    monkeypatch.setattr(
        mlflow_config,
        "ARTIFACT_ROOT",
        artifacts,
    )

    monkeypatch.setattr(
        mlflow_config,
        "REPORTS_DIR",
        reports,
    )

    mlflow_config.initialize_storage()

    assert store.is_dir()
    assert artifacts.is_dir()
    assert reports.is_dir()


# ============================================================
# TRACKING AND ARTIFACT TESTS
# ============================================================

def test_original_dataset_hash():

    expected = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    actual = mlflow_tracking.calculate_file_hash(
        DATA_PATH
    )

    assert actual == expected


def test_file_hash_is_deterministic():

    first = mlflow_tracking.calculate_file_hash(
        DATA_PATH
    )

    second = mlflow_tracking.calculate_file_hash(
        DATA_PATH
    )

    assert first == second


def test_file_hash_length():

    result = mlflow_tracking.calculate_file_hash(
        DATA_PATH
    )

    assert len(result) == 64


def test_file_hash_changes_with_content(tmp_path):

    path = tmp_path / "artifact.txt"

    path.write_text(
        "original",
        encoding="utf-8",
    )

    first = mlflow_tracking.calculate_file_hash(
        path
    )

    path.write_text(
        "modified",
        encoding="utf-8",
    )

    second = mlflow_tracking.calculate_file_hash(
        path
    )

    assert first != second


def test_require_existing_file():

    result = mlflow_tracking.require_file(
        DATA_PATH
    )

    assert Path(result).exists()


def test_require_missing_file(tmp_path):

    with pytest.raises(FileNotFoundError):

        mlflow_tracking.require_file(
            tmp_path / "missing.pkl"
        )


@pytest.mark.parametrize(
    "model",
    mlflow_config.SUPPORTED_MODELS,
)
@pytest.mark.parametrize(
    "variant",
    ["baseline", "optimized"],
)
def test_valid_model_variants(model, variant):

    mlflow_tracking.validate_model_variant(
        model,
        variant,
    )


def test_invalid_tracking_model():

    with pytest.raises(ValueError):

        mlflow_tracking.validate_model_variant(
            "unknown_model",
            "optimized",
        )


def test_invalid_tracking_variant():

    with pytest.raises(ValueError):

        mlflow_tracking.validate_model_variant(
            "xgboost",
            "unknown_variant",
        )


# ============================================================
# CLASSIFICATION THRESHOLD TESTS
# ============================================================

@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.25, 0.50, 0.75, 1.0],
)
def test_valid_thresholds(threshold):

    result = validate_classification_threshold(
        threshold
    )

    assert result == pytest.approx(threshold)


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        False,
        np.bool_(True),
        None,
        "invalid",
    ],
)
def test_invalid_thresholds(threshold):

    with pytest.raises(
        (ValueError, TypeError)
    ):

        validate_classification_threshold(
            threshold
        )


def test_default_inference_threshold(
    inference_wrapper,
):

    assert (
        inference_wrapper.resolve_threshold()
        == 0.50
    )


def test_override_inference_threshold(
    inference_wrapper,
):

    result = inference_wrapper.resolve_threshold({
        "classification_threshold": 0.35
    })

    assert result == pytest.approx(0.35)


def test_invalid_threshold_override(
    inference_wrapper,
):

    with pytest.raises(ValueError):

        inference_wrapper.resolve_threshold({
            "classification_threshold": 1.5
        })


def test_unsupported_inference_parameter(
    inference_wrapper,
):

    with pytest.raises(ValueError):

        inference_wrapper.resolve_threshold({
            "unsupported_parameter": 1
        })


# ============================================================
# MODEL CONFIGURATION TESTS
# ============================================================

def test_valid_model_configuration(
    model_configuration,
):

    result = validate_model_configuration(
        model_configuration
    )

    assert result["model_name"] == (
        "logistic_regression"
    )

    assert result["classification_threshold"] == 0.50


@pytest.mark.parametrize(
    "missing_key",
    [
        "model_name",
        "calibration_method",
        "classification_threshold",
    ],
)
def test_missing_configuration_fields(
    model_configuration,
    missing_key,
):

    model_configuration.pop(missing_key)

    with pytest.raises(ValueError):

        validate_model_configuration(
            model_configuration
        )


def test_invalid_configuration_model(
    model_configuration,
):

    model_configuration["model_name"] = "unknown"

    with pytest.raises(ValueError):

        validate_model_configuration(
            model_configuration
        )


def test_invalid_configuration_calibration(
    model_configuration,
):

    model_configuration[
        "calibration_method"
    ] = "unknown"

    with pytest.raises(ValueError):

        validate_model_configuration(
            model_configuration
        )


def test_invalid_configuration_threshold(
    model_configuration,
):

    model_configuration[
        "classification_threshold"
    ] = -0.5

    with pytest.raises(ValueError):

        validate_model_configuration(
            model_configuration
        )


# ============================================================
# RAW INPUT PREPROCESSING TESTS
# ============================================================

def test_prepare_original_customer_features(
    inference_wrapper,
    raw_customers,
):

    features = inference_wrapper.prepare_features(
        raw_customers
    )

    assert len(features) == len(raw_customers)

    assert list(features.columns) == (
        inference_wrapper.feature_columns
    )


def test_prepared_features_are_finite(
    inference_wrapper,
    raw_customers,
):

    features = inference_wrapper.prepare_features(
        raw_customers
    )

    assert np.isfinite(
        features.to_numpy(dtype=float)
    ).all()


def test_preprocessing_does_not_modify_input(
    inference_wrapper,
    raw_customers,
):

    original = raw_customers.copy(deep=True)

    inference_wrapper.prepare_features(
        raw_customers
    )

    pd.testing.assert_frame_equal(
        original,
        raw_customers,
    )


def test_empty_inference_input(
    inference_wrapper,
    raw_customers,
):

    with pytest.raises(ValueError):

        inference_wrapper.prepare_features(
            raw_customers.iloc[0:0]
        )


def test_invalid_inference_input_type(
    inference_wrapper,
):

    with pytest.raises(
        (ValueError, TypeError)
    ):

        inference_wrapper.prepare_features(
            ["invalid"]
        )


def test_inference_rejects_target_column(
    inference_wrapper,
    raw_data,
):

    with pytest.raises(ValueError):

        inference_wrapper.prepare_features(
            raw_data.head(5)
        )


def test_inference_missing_feature(
    inference_wrapper,
    raw_customers,
):

    invalid = raw_customers.drop(
        columns=["MonthlyCharges"]
    )

    with pytest.raises(ValueError):

        inference_wrapper.prepare_features(
            invalid
        )


def test_inference_invalid_category(
    inference_wrapper,
    raw_customers,
):

    invalid = raw_customers.copy()

    invalid.loc[
        invalid.index[0],
        "Contract",
    ] = "Invalid contract"

    with pytest.raises(ValueError):

        inference_wrapper.prepare_features(
            invalid
        )


def test_inference_negative_charges(
    inference_wrapper,
    raw_customers,
):

    invalid = raw_customers.copy()

    invalid.loc[
        invalid.index[0],
        "MonthlyCharges",
    ] = -100.0

    with pytest.raises(ValueError):

        inference_wrapper.prepare_features(
            invalid
        )


def test_single_customer_features(
    inference_wrapper,
    raw_customers,
):

    result = inference_wrapper.prepare_features(
        raw_customers.head(1)
    )

    assert len(result) == 1


# ============================================================
# INFERENCE OUTPUT TESTS
# ============================================================

def mock_probabilities(
    wrapper,
    monkeypatch,
    probability=0.70,
):

    def predict(features):

        return np.full(
            len(features),
            probability,
            dtype=float,
        )

    monkeypatch.setattr(
        wrapper,
        "predict_raw_probabilities",
        predict,
    )


def test_batch_prediction_count(
    inference_wrapper,
    raw_customers,
    monkeypatch,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
    )

    assert len(result) == len(raw_customers)


def test_prediction_output_columns(
    inference_wrapper,
    raw_customers,
    monkeypatch,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
    )

    expected = {
        "customerID",
        "raw_churn_probability",
        "churn_probability",
        "prediction",
        "prediction_label",
        "classification_threshold",
        "model_name",
        "calibration_method",
    }

    assert set(result.columns) == expected


def test_prediction_customer_alignment(
    inference_wrapper,
    raw_customers,
    monkeypatch,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
    )

    np.testing.assert_array_equal(
        result["customerID"].to_numpy(),
        raw_customers["customerID"].to_numpy(),
    )


def test_uncalibrated_probability_consistency(
    inference_wrapper,
    raw_customers,
    monkeypatch,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
    )

    np.testing.assert_allclose(
        result["raw_churn_probability"],
        result["churn_probability"],
    )


@pytest.mark.parametrize(
    "probability,threshold,expected",
    [
        (0.70, 0.50, 1),
        (0.30, 0.50, 0),
        (0.50, 0.50, 1),
        (0.20, 0.00, 1),
        (0.80, 1.00, 0),
        (1.00, 1.00, 1),
    ],
)
def test_prediction_threshold_behavior(
    inference_wrapper,
    raw_customers,
    monkeypatch,
    probability,
    threshold,
    expected,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
        probability=probability,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
        params={
            "classification_threshold": threshold
        },
    )

    assert result["prediction"].eq(
        expected
    ).all()


@pytest.mark.parametrize(
    "probability,expected_label",
    [
        (0.80, "Churn"),
        (0.20, "No Churn"),
    ],
)
def test_prediction_labels(
    inference_wrapper,
    raw_customers,
    monkeypatch,
    probability,
    expected_label,
):

    mock_probabilities(
        inference_wrapper,
        monkeypatch,
        probability=probability,
    )

    result = inference_wrapper.predict(
        None,
        raw_customers,
    )

    assert result["prediction_label"].eq(
        expected_label
    ).all()


# ============================================================
# PACKAGING TESTS
# ============================================================

def test_runtime_requirements_are_pinned():

    requirements = get_runtime_requirements()

    assert len(requirements) > 0

    assert all(
        "==" in requirement
        for requirement in requirements
    )


def test_runtime_requirements_are_unique():

    requirements = get_runtime_requirements()

    names = [
        requirement.split("==")[0]
        for requirement in requirements
    ]

    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "package",
    [
        "mlflow",
        "scikit-learn",
        "tensorflow",
        "xgboost",
        "pandas",
        "numpy",
        "joblib",
    ],
)
def test_required_runtime_packages(package):

    requirements = get_runtime_requirements()

    assert any(
        requirement.startswith(f"{package}==")
        for requirement in requirements
    )


# ============================================================
# MODEL REGISTRY TESTS
# ============================================================

def test_registry_reuses_registered_model():

    client = MagicMock()

    existing = SimpleNamespace(
        name=mlflow_config.REGISTERED_MODEL_NAME
    )

    client.get_registered_model.return_value = (
        existing
    )

    result = mlflow_registry.ensure_registered_model(
        client
    )

    assert result is existing

    client.create_registered_model.assert_not_called()


def test_no_existing_model_version():

    client = MagicMock()

    client.search_model_versions.return_value = []

    result = mlflow_registry.find_existing_version(
        client,
        "model-123",
        "run-123",
    )

    assert result is None


def test_existing_model_version_reused():

    client = MagicMock()

    existing = SimpleNamespace(
        version="1",
        run_id="run-123",
        source="models:/model-123",
        tags={
            "logged_model_id": "model-123"
        },
    )

    client.search_model_versions.return_value = [
        existing
    ]

    result = mlflow_registry.find_existing_version(
        client,
        "model-123",
        "run-123",
    )

    assert result is existing


def test_registry_rejects_run_mismatch():

    client = MagicMock()

    existing = SimpleNamespace(
        version="1",
        run_id="different-run",
        source="models:/model-123",
        tags={
            "logged_model_id": "model-123"
        },
    )

    client.search_model_versions.return_value = [
        existing
    ]

    with pytest.raises(ValueError):

        mlflow_registry.find_existing_version(
            client,
            "model-123",
            "run-123",
        )


def test_registry_rejects_duplicate_versions():

    client = MagicMock()

    versions = [
        SimpleNamespace(
            version=str(number),
            run_id="run-123",
            source="models:/model-123",
            tags={
                "logged_model_id": "model-123"
            },
        )
        for number in [1, 2]
    ]

    client.search_model_versions.return_value = (
        versions
    )

    with pytest.raises(ValueError):

        mlflow_registry.find_existing_version(
            client,
            "model-123",
            "run-123",
        )


# ============================================================
# MODEL PROMOTION RANKING TESTS
# ============================================================

def test_champion_selection(candidate_metrics):

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    assert ranked.iloc[0]["model"] == "xgboost"

    assert ranked.iloc[0]["assigned_role"] == (
        "champion"
    )


def test_challenger_selection(candidate_metrics):

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    assert ranked.iloc[1]["model"] == (
        "logistic_regression"
    )

    assert ranked.iloc[1]["assigned_role"] == (
        "challenger"
    )


def test_remaining_candidate(candidate_metrics):

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    assert ranked.iloc[2]["model"] == (
        "tensorflow_ann"
    )

    assert ranked.iloc[2]["assigned_role"] == (
        "candidate"
    )


def test_ranking_prioritizes_profit(candidate_metrics):

    modified = candidate_metrics.copy()

    modified.loc[
        modified["model"] == "tensorflow_ann",
        "expected_net_profit",
    ] = 25000.0

    ranked = mlflow_promotion.rank_candidates(
        modified
    )

    assert ranked.iloc[0]["model"] == (
        "tensorflow_ann"
    )


def test_ranking_auc_tie_breaker(candidate_metrics):

    modified = candidate_metrics.copy()

    modified["expected_net_profit"] = 10000.0

    ranked = mlflow_promotion.rank_candidates(
        modified
    )

    assert ranked.iloc[0]["model"] == "xgboost"


def test_ranking_brier_tie_breaker(candidate_metrics):

    modified = candidate_metrics.copy()

    modified["expected_net_profit"] = 10000.0

    modified["validation_roc_auc"] = 0.85

    ranked = mlflow_promotion.rank_candidates(
        modified
    )

    assert ranked.iloc[0]["model"] == "xgboost"


def test_ranking_is_deterministic(candidate_metrics):

    first = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    shuffled = candidate_metrics.sample(
        frac=1,
        random_state=42,
    )

    second = mlflow_promotion.rank_candidates(
        shuffled
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_ranking_does_not_modify_input(
    candidate_metrics,
):

    original = candidate_metrics.copy(deep=True)

    mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    pd.testing.assert_frame_equal(
        original,
        candidate_metrics,
    )


# ============================================================
# ALIAS MANAGEMENT AND ROLLBACK TESTS
# ============================================================

def test_missing_alias(registry_client):

    result = mlflow_promotion.get_alias_version(
        registry_client,
        "champion",
    )

    assert result is None


def test_existing_alias(registry_client):

    registry_client.set_registered_model_alias(
        name=mlflow_config.REGISTERED_MODEL_NAME,
        alias="champion",
        version="2",
    )

    result = mlflow_promotion.get_alias_version(
        registry_client,
        "champion",
    )

    assert result == "2"


def test_capture_empty_aliases(registry_client):

    aliases = (
        mlflow_promotion.capture_current_aliases(
            registry_client
        )
    )

    assert aliases == {
        "champion": None,
        "challenger": None,
    }


def test_assign_champion_and_challenger(
    registry_client,
    candidate_metrics,
):

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    assigned = mlflow_promotion.assign_aliases(
        registry_client,
        ranked,
    )

    assert assigned == {
        "champion": "2",
        "challenger": "1",
    }


def test_restore_previous_aliases(
    registry_client,
    candidate_metrics,
):

    previous = {
        "champion": "1",
        "challenger": "3",
    }

    for alias, version in previous.items():

        registry_client.set_registered_model_alias(
            name=mlflow_config.REGISTERED_MODEL_NAME,
            alias=alias,
            version=version,
        )

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    mlflow_promotion.assign_aliases(
        registry_client,
        ranked,
    )

    mlflow_promotion.restore_aliases(
        registry_client,
        previous,
    )

    actual = (
        mlflow_promotion.capture_current_aliases(
            registry_client
        )
    )

    assert actual == previous


def test_rollback_removes_new_aliases(
    registry_client,
    candidate_metrics,
):

    previous = {
        "champion": None,
        "challenger": None,
    }

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    mlflow_promotion.assign_aliases(
        registry_client,
        ranked,
    )

    mlflow_promotion.restore_aliases(
        registry_client,
        previous,
    )

    actual = (
        mlflow_promotion.capture_current_aliases(
            registry_client
        )
    )

    assert actual == previous


def test_repeated_alias_assignment(
    registry_client,
    candidate_metrics,
):

    ranked = mlflow_promotion.rank_candidates(
        candidate_metrics
    )

    first = mlflow_promotion.assign_aliases(
        registry_client,
        ranked,
    )

    second = mlflow_promotion.assign_aliases(
        registry_client,
        ranked,
    )

    assert first == second


def test_promotion_rolls_back_on_verification_failure(
    registry_client,
    candidate_metrics,
    monkeypatch,
):

    previous = {
        "champion": "1",
        "challenger": "3",
    }

    for alias, version in previous.items():

        registry_client.set_registered_model_alias(
            name=mlflow_config.REGISTERED_MODEL_NAME,
            alias=alias,
            version=version,
        )

    monkeypatch.setattr(
        mlflow_promotion,
        "initialize_experiment",
        lambda: "1",
    )

    monkeypatch.setattr(
        mlflow_promotion,
        "get_mlflow_client",
        lambda: registry_client,
    )

    monkeypatch.setattr(
        mlflow_promotion,
        "load_registered_candidates",
        lambda: candidate_metrics,
    )

    monkeypatch.setattr(
        mlflow_promotion,
        "verify_registry_candidates",
        lambda *args: None,
    )

    monkeypatch.setattr(
        mlflow_promotion,
        "load_raw_validation_data",
        lambda: (None, None, None),
    )

    def fail_verification(*args):

        raise RuntimeError(
            "Simulated verification failure"
        )

    monkeypatch.setattr(
        mlflow_promotion,
        "verify_promoted_aliases",
        fail_verification,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated verification failure",
    ):

        mlflow_promotion.promote_models()

    actual = (
        mlflow_promotion.capture_current_aliases(
            registry_client
        )
    )

    assert actual == previous
