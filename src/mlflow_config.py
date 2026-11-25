
"""
Centralized MLflow configuration for Telecom Customer Retention.

Provides local tracking storage, experiment initialization,
model registry configuration and artifact management.
"""

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_STORE_DIR = PROJECT_ROOT / "mlflow_store"

DATABASE_PATH = MLFLOW_STORE_DIR / "mlflow.db"

ARTIFACT_ROOT = MLFLOW_STORE_DIR / "artifacts"

REPORTS_DIR = PROJECT_ROOT / "reports" / "mlflow"

DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
)

CANDIDATE_MODEL_DIR = (
    PROJECT_ROOT / "models" / "candidates"
)

OPTIMIZED_MODEL_DIR = (
    CANDIDATE_MODEL_DIR / "optimized"
)

CALIBRATION_MODEL_DIR = (
    CANDIDATE_MODEL_DIR / "calibration"
)


# ============================================================
# MLFLOW SETTINGS
# ============================================================

EXPERIMENT_NAME = "Telecom Customer Retention"

REGISTERED_MODEL_NAME = "telecom-churn-classifier"

PRODUCTION_ALIAS = "champion"

CHALLENGER_ALIAS = "challenger"

RANDOM_STATE = 42

TRACKING_HOST = "127.0.0.1"

TRACKING_PORT = 5000

DEFAULT_TRACKING_URI = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

TRACKING_SERVER_URL = (
    f"http://{TRACKING_HOST}:{TRACKING_PORT}"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SUPPORTED_MODELS = (
    "logistic_regression",
    "xgboost",
    "tensorflow_ann",
)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "tensorflow_ann": "TensorFlow ANN",
}

MODEL_VARIANTS = (
    "baseline",
    "optimized",
)

CALIBRATION_METHODS = (
    "uncalibrated",
    "sigmoid",
    "isotonic",
)


# ============================================================
# EXPERIMENT TAGS
# ============================================================

EXPERIMENT_TAGS = {
    "project": "telecom-customer-retention",
    "project_version": "2.0",
    "domain": "telecommunications",
    "task": "binary_classification",
    "dataset": "IBM Telco Customer Churn",
    "framework": "mlflow",
    "environment": "local",
}


# ============================================================
# STORAGE INITIALIZATION
# ============================================================

def initialize_storage():
    """Create directories required for MLflow."""

    for directory in (
        MLFLOW_STORE_DIR,
        ARTIFACT_ROOT,
        REPORTS_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def get_tracking_uri():
    """Return configured tracking URI."""

    return (
        os.getenv("MLFLOW_TRACKING_URI")
        or DEFAULT_TRACKING_URI
    )


def get_registry_uri():
    """Return configured model registry URI."""

    return (
        os.getenv("MLFLOW_REGISTRY_URI")
        or get_tracking_uri()
    )


# ============================================================
# TRACKING CONFIGURATION
# ============================================================

def configure_mlflow():
    """Configure tracking and registry backends."""

    initialize_storage()

    tracking_uri = get_tracking_uri()

    registry_uri = get_registry_uri()

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_registry_uri(registry_uri)

    return tracking_uri


def get_mlflow_client():
    """Return a configured MLflow client."""

    configure_mlflow()

    return MlflowClient(
        tracking_uri=get_tracking_uri(),
        registry_uri=get_registry_uri(),
    )


# ============================================================
# EXPERIMENT INITIALIZATION
# ============================================================

def initialize_experiment(
    experiment_name=EXPERIMENT_NAME,
):
    """Create or retrieve the project experiment."""

    if (
        not isinstance(experiment_name, str)
        or not experiment_name.strip()
    ):
        raise ValueError(
            "Experiment name must be a non-empty string."
        )

    experiment_name = experiment_name.strip()

    client = get_mlflow_client()

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:

        create_options = {
            "name": experiment_name,
            "tags": EXPERIMENT_TAGS,
        }

        if get_tracking_uri().startswith("sqlite:"):

            create_options["artifact_location"] = (
                ARTIFACT_ROOT.resolve().as_uri()
            )

        experiment_id = client.create_experiment(
            **create_options
        )

    else:

        if experiment.lifecycle_stage != "active":

            raise RuntimeError(
                f"Experiment '{experiment_name}' "
                "is not active."
            )

        experiment_id = experiment.experiment_id

    mlflow.set_experiment(
        experiment_id=experiment_id
    )

    return experiment_id


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_configuration_summary():
    """Return MLflow configuration information."""

    return {
        "mlflow_version": mlflow.__version__,
        "project_root": str(PROJECT_ROOT),
        "tracking_uri": get_tracking_uri(),
        "registry_uri": get_registry_uri(),
        "database_path": str(DATABASE_PATH),
        "artifact_root": str(ARTIFACT_ROOT),
        "reports_directory": str(REPORTS_DIR),
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "production_alias": PRODUCTION_ALIAS,
        "challenger_alias": CHALLENGER_ALIAS,
        "tracking_server_url": TRACKING_SERVER_URL,
        "random_state": RANDOM_STATE,
        "supported_models": list(SUPPORTED_MODELS),
    }


def print_configuration_summary():
    """Display the current MLflow configuration."""

    print("\n" + "=" * 70)

    print("MLFLOW CONFIGURATION")

    print("=" * 70)

    for key, value in get_configuration_summary().items():

        label = key.replace("_", " ").title()

        print(f"{label}: {value}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    configure_mlflow()

    experiment_id = initialize_experiment()

    print_configuration_summary()

    print(f"\nExperiment ID: {experiment_id}")

    print("\nMLflow initialization successful.")

    print("Tracking backend configured.")

    print("Artifact storage configured.")

    print("Experiment ready.")

    print("Production models remain unchanged.")


if __name__ == "__main__":

    main()
