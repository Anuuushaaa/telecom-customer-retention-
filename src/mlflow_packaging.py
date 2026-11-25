
"""
MLflow model serialization and round-trip verification.

Packages the three optimized telecom churn models with
their preprocessing artifacts, calibration models,
inference configuration and runtime dependencies.

No retraining or production model changes.
"""

import argparse
import json
import tempfile

from pathlib import Path
from importlib.metadata import version
from types import SimpleNamespace

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from src.mlflow_config import (
    PROJECT_ROOT,
    DATA_PATH,
    REPORTS_DIR,
    SUPPORTED_MODELS,
    MODEL_DISPLAY_NAMES,
    get_mlflow_client,
    initialize_experiment,
)

from src.mlflow_tracking import (
    get_model_artifacts,
    calculate_file_hash,
    require_file,
)

from src.mlflow_model import TelecomChurnModel

from src.threshold_tuning import (
    load_calibrated_predictions,
)


# ============================================================
# CONFIGURATION
# ============================================================

RUN_SUMMARY_PATH = (
    REPORTS_DIR / "experiment_run_summary.csv"
)

PACKAGING_SUMMARY_PATH = (
    REPORTS_DIR / "packaged_model_summary.csv"
)

PACKAGING_METADATA_PATH = (
    REPORTS_DIR / "packaging_metadata.json"
)

SELECTED_METHODS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "calibration"
    / "selected_calibration_methods.csv"
)

DEFAULT_THRESHOLD = 0.50

PACKAGING_SCHEMA_VERSION = "1"


# ============================================================
# RUNTIME DEPENDENCIES
# ============================================================

RUNTIME_PACKAGES = (
    "mlflow",
    "cloudpickle",
    "joblib",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "tensorflow",
    "keras",
    "optuna",
    "matplotlib",
    "statsmodels",
    "protobuf",
    "pyarrow",
)


def get_runtime_requirements():
    """Return exact installed runtime package versions."""

    return [
        f"{package}=={version(package)}"
        for package in RUNTIME_PACKAGES
    ]


# ============================================================
# LOAD EXPERIMENT RUNS
# ============================================================

def load_optimized_runs():
    """Retrieve and validate optimized experiment runs."""

    summary = pd.read_csv(
        require_file(RUN_SUMMARY_PATH)
    )

    required_columns = {
        "model",
        "variant",
        "run_id",
        "fingerprint",
    }

    missing = (
        required_columns - set(summary.columns)
    )

    if missing:
        raise ValueError(
            f"Missing run summary columns: "
            f"{sorted(missing)}"
        )

    optimized = summary[
        summary["variant"] == "optimized"
    ].copy()

    if optimized["model"].duplicated().any():
        raise ValueError(
            "Duplicate optimized model runs."
        )

    if set(optimized["model"]) != set(SUPPORTED_MODELS):
        raise ValueError(
            "Optimized model runs are incomplete."
        )

    client = get_mlflow_client()

    for _, row in optimized.iterrows():

        run = client.get_run(row["run_id"])

        if run.info.status != "FINISHED":
            raise ValueError(
                f"Run is not finished: "
                f"{row['run_id']}"
            )

        tags = run.data.tags

        if tags.get("model_name") != row["model"]:
            raise ValueError(
                "Run model name mismatch."
            )

        if tags.get("model_variant") != "optimized":
            raise ValueError(
                "Run variant mismatch."
            )

        if (
            tags.get("run_fingerprint")
            != row["fingerprint"]
        ):
            raise ValueError(
                "Run fingerprint mismatch."
            )

    return optimized


# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

def load_selected_calibration_methods():
    """Load preferred calibration methods."""

    selected = pd.read_csv(
        require_file(SELECTED_METHODS_PATH)
    )

    if selected["model"].duplicated().any():
        raise ValueError(
            "Duplicate calibration selections."
        )

    methods = dict(
        zip(
            selected["model"],
            selected["calibration_method"],
        )
    )

    if set(methods) != set(SUPPORTED_MODELS):
        raise ValueError(
            "Incomplete calibration selections."
        )

    for method in methods.values():

        if method not in (
            "uncalibrated",
            "sigmoid",
            "isotonic",
        ):
            raise ValueError(
                f"Unsupported calibration method: "
                f"{method}"
            )

    return methods


# ============================================================
# VALIDATION INPUT
# ============================================================

def load_raw_validation_data():
    """Reconstruct original raw validation customers."""

    raw = pd.read_csv(
        require_file(DATA_PATH)
    )

    predictions, metadata = (
        load_calibrated_predictions()
    )

    if raw["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer identifiers."
        )

    validation = (
        raw.set_index("customerID")
        .loc[predictions["customerID"].tolist()]
        .reset_index()
        .drop(columns=["Churn"])
    )

    # Preserve TotalCharges as a string because the
    # original dataset includes blank values for
    # customers with zero tenure.

    validation["TotalCharges"] = (
        validation["TotalCharges"]
        .fillna("")
        .astype(str)
    )

    for column in (
        "SeniorCitizen",
        "tenure",
    ):

        validation[column] = (
            validation[column].astype("int64")
        )

    validation["MonthlyCharges"] = (
        validation["MonthlyCharges"]
        .astype("float64")
    )

    if len(validation) != len(predictions):
        raise ValueError(
            "Validation row count mismatch."
        )

    return validation, predictions, metadata


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

def create_inference_configuration(
    model_name,
    calibration_method,
    output_path,
):
    """Save model-specific inference configuration."""

    configuration = {
        "model_name": model_name,
        "model_variant": "optimized",
        "calibration_method": calibration_method,
        "classification_threshold": DEFAULT_THRESHOLD,
        "packaging_schema_version": (
            PACKAGING_SCHEMA_VERSION
        ),
        "dataset_hash": calculate_file_hash(DATA_PATH),
        "model_lifecycle": "candidate",
    }

    with Path(output_path).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )

    return configuration


# ============================================================
# MODEL ARTIFACTS
# ============================================================

def build_packaging_artifacts(
    model_name,
    calibration_method,
    configuration_path,
):
    """Collect artifacts required for inference."""

    source = get_model_artifacts(
        model_name,
        "optimized",
    )

    artifacts = {
        "model": str(source["model"]),
        "scaler": str(source["scaler"]),
        "feature_columns": str(
            source["feature_columns"]
        ),
        "configuration": str(
            configuration_path
        ),
    }

    if calibration_method != "uncalibrated":

        artifacts["calibrator"] = str(
            source[
                f"{calibration_method}_calibrator"
            ]
        )

    for path in artifacts.values():
        require_file(path)

    return artifacts


# ============================================================
# ARTIFACT INTEGRITY
# ============================================================

def verify_source_artifacts(
    run_id,
    artifacts,
):
    """Verify model files against the original run manifest."""

    client = get_mlflow_client()

    with tempfile.TemporaryDirectory() as directory:

        manifest_path = client.download_artifacts(
            run_id,
            "metadata/artifact_manifest.json",
            directory,
        )

        with Path(manifest_path).open(
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(file)

    if manifest["dataset_hash"] != calculate_file_hash(
        DATA_PATH
    ):
        raise ValueError(
            "Dataset fingerprint mismatch."
        )

    for name, path in artifacts.items():

        if name == "configuration":
            continue

        path = Path(path)

        relative = (
            path.relative_to(PROJECT_ROOT)
            .as_posix()
        )

        if relative not in manifest["files"]:
            raise ValueError(
                f"Artifact absent from run manifest: "
                f"{relative}"
            )

        expected_hash = manifest["files"][
            relative
        ]["sha256"]

        actual_hash = calculate_file_hash(path)

        if expected_hash != actual_hash:
            raise ValueError(
                f"Artifact fingerprint mismatch: "
                f"{relative}"
            )


# ============================================================
# MODEL SIGNATURE
# ============================================================

def create_model_signature(
    artifacts,
    validation,
):
    """Infer input, output and parameter schemas."""

    example = validation.head(5).copy()

    context = SimpleNamespace(
        artifacts=artifacts
    )

    wrapper = TelecomChurnModel()

    wrapper.load_context(context)

    example_output = wrapper.predict(
        context,
        example,
    )

    signature = infer_signature(
        model_input=example,
        model_output=example_output,
        params={
            "classification_threshold": (
                DEFAULT_THRESHOLD
            ),
        },
    )

    return signature, example


# ============================================================
# SERIALIZED MODEL VERIFICATION
# ============================================================

def verify_serialized_model(
    model_uri,
    model_name,
    calibration_method,
    validation,
    expected_predictions,
):
    """Reload an MLflow model and verify its predictions."""

    loaded_model = mlflow.pyfunc.load_model(
        model_uri
    )

    results = loaded_model.predict(
        validation
    )

    expected_raw = expected_predictions[
        f"{model_name}_uncalibrated_probability"
    ].to_numpy(dtype=float)

    expected_calibrated = expected_predictions[
        f"{model_name}_{calibration_method}_probability"
    ].to_numpy(dtype=float)

    if len(results) != len(validation):
        raise ValueError(
            "Serialized prediction count mismatch."
        )

    np.testing.assert_array_equal(
        results["customerID"].to_numpy(),
        expected_predictions[
            "customerID"
        ].to_numpy(),
    )

    np.testing.assert_allclose(
        results["raw_churn_probability"],
        expected_raw,
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        results["churn_probability"],
        expected_calibrated,
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_array_equal(
        results["prediction"],
        (
            expected_calibrated
            >= DEFAULT_THRESHOLD
        ).astype(int),
    )

    # Verify that MLflow passes threshold parameters
    # through to the underlying PythonModel.

    sample = validation.head(10).copy()

    low_threshold = loaded_model.predict(
        sample,
        params={
            "classification_threshold": 0.0
        },
    )

    high_threshold = loaded_model.predict(
        sample,
        params={
            "classification_threshold": 1.0
        },
    )

    np.testing.assert_array_equal(
        low_threshold["prediction"].to_numpy(),
        np.ones(len(sample), dtype=int),
    )

    np.testing.assert_array_equal(
        high_threshold["prediction"].to_numpy(),
        (
            high_threshold["churn_probability"]
            .to_numpy() >= 1.0
        ).astype(int),
    )

    np.testing.assert_allclose(
        low_threshold["classification_threshold"],
        0.0,
    )

    np.testing.assert_allclose(
        high_threshold["classification_threshold"],
        1.0,
    )

    maximum_raw_error = float(
        np.max(
            np.abs(
                results["raw_churn_probability"]
                .to_numpy()
                - expected_raw
            )
        )
    )

    maximum_calibrated_error = float(
        np.max(
            np.abs(
                results["churn_probability"]
                .to_numpy()
                - expected_calibrated
            )
        )
    )

    return {
        "verified_rows": len(results),
        "maximum_raw_probability_error": (
            maximum_raw_error
        ),
        "maximum_calibrated_probability_error": (
            maximum_calibrated_error
        ),
        "classification_verified": True,
        "threshold_override_verified": True,
        "serialization_verified": True,
    }


# ============================================================
# PACKAGE SINGLE MODEL
# ============================================================

def package_model(
    model_name,
    run_id,
    calibration_method,
    validation,
    expected_predictions,
):
    """Serialize and verify one optimized model."""

    print("\n" + "=" * 70)

    print(
        f"PACKAGING: "
        f"{MODEL_DISPLAY_NAMES[model_name]}"
    )

    print("=" * 70)

    with tempfile.TemporaryDirectory() as directory:

        configuration_path = (
            Path(directory)
            / "inference_configuration.json"
        )

        configuration = (
            create_inference_configuration(
                model_name,
                calibration_method,
                configuration_path,
            )
        )

        artifacts = build_packaging_artifacts(
            model_name,
            calibration_method,
            configuration_path,
        )

        verify_source_artifacts(
            run_id,
            artifacts,
        )

        signature, example = (
            create_model_signature(
                artifacts,
                validation,
            )
        )

        requirements = get_runtime_requirements()

        with mlflow.start_run(run_id=run_id):

            model_info = mlflow.pyfunc.log_model(
                name=f"{model_name}_inference",
                python_model=TelecomChurnModel(),
                artifacts=artifacts,
                code_paths=[
                    str(PROJECT_ROOT / "src")
                ],
                signature=signature,
                input_example=example,
                pip_requirements=requirements,
                metadata=configuration,
                model_type="classifier",
            )

            model_uri = model_info.model_uri

            verification = verify_serialized_model(
                model_uri=model_uri,
                model_name=model_name,
                calibration_method=calibration_method,
                validation=validation,
                expected_predictions=expected_predictions,
            )

            mlflow.log_metrics({
                "packaging_verified_rows": (
                    verification["verified_rows"]
                ),
                "packaging_maximum_raw_error": (
                    verification[
                        "maximum_raw_probability_error"
                    ]
                ),
                "packaging_maximum_calibrated_error": (
                    verification[
                        "maximum_calibrated_probability_error"
                    ]
                ),
            })

            mlflow.set_tags({
                "serialization_verified": "true",
                "threshold_override_verified": "true",
                "packaging_schema_version": (
                    PACKAGING_SCHEMA_VERSION
                ),
                "packaged_model_uri": model_uri,
                "model_registered": "false",
            })

            mlflow.log_dict(
                verification,
                "packaging/verification.json",
            )

            mlflow.log_dict(
                configuration,
                "packaging/inference_configuration.json",
            )

    print(f"Model URI: {model_uri}")

    print(
        f"Validation rows verified: "
        f"{verification['verified_rows']}"
    )

    print(
        f"Maximum calibrated probability error: "
        f"{verification['maximum_calibrated_probability_error']:.10f}"
    )

    print("Artifact integrity: PASSED")

    print("Model serialization: PASSED")

    print("Prediction consistency: PASSED")

    print("Threshold override: PASSED")

    return {
        "model": model_name,
        "variant": "optimized",
        "run_id": run_id,
        "model_uri": model_uri,
        "model_id": getattr(
            model_info,
            "model_id",
            None,
        ),
        "calibration_method": calibration_method,
        "classification_threshold": DEFAULT_THRESHOLD,
        **verification,
    }


# ============================================================
# SAVE PACKAGING REPORTS
# ============================================================

def save_packaging_reports(results):
    """Save packaged model identifiers and verification."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(results)

    summary.to_csv(
        PACKAGING_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "packaging_schema_version": (
            PACKAGING_SCHEMA_VERSION
        ),
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "packaged_models": (
            summary["model"].tolist()
        ),
        "total_models": len(summary),
        "total_verified_predictions": int(
            summary["verified_rows"].sum()
        ),
        "evaluation_partition": "validation",
        "serialization_verified": True,
        "threshold_override_verified": True,
        "training_repeated": False,
        "models_registered": False,
        "production_updated": False,
        "test_evaluated": False,
        "runtime_requirements": (
            get_runtime_requirements()
        ),
    }

    with PACKAGING_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Package optimized telecom churn models."
        )
    )

    parser.add_argument(
        "--model",
        choices=["all", *SUPPORTED_MODELS],
        default="all",
    )

    args = parser.parse_args()

    initialize_experiment()

    runs = load_optimized_runs()

    methods = (
        load_selected_calibration_methods()
    )

    validation, predictions, _ = (
        load_raw_validation_data()
    )

    selected_models = (
        SUPPORTED_MODELS
        if args.model == "all"
        else (args.model,)
    )

    results = []

    for model_name in selected_models:

        row = runs[
            runs["model"] == model_name
        ].iloc[0]

        result = package_model(
            model_name=model_name,
            run_id=row["run_id"],
            calibration_method=methods[model_name],
            validation=validation,
            expected_predictions=predictions,
        )

        results.append(result)

    if args.model == "all":

        summary = save_packaging_reports(
            results
        )

    else:

        summary = pd.DataFrame(results)

        print(
            "\nSingle-model verification completed. "
            "Full packaging reports not overwritten."
        )

    print("\n" + "=" * 70)

    print("MLFLOW MODEL PACKAGING SUMMARY")

    print("=" * 70)

    print(
        summary[
            [
                "model",
                "calibration_method",
                "verified_rows",
                "serialization_verified",
            ]
        ].to_string(index=False)
    )

    print(f"\nModels packaged: {len(summary)}")

    print(
        "Total predictions verified: "
        f"{summary['verified_rows'].sum()}"
    )

    print("\nMLflow model packaging completed.")

    print("No models were retrained.")

    print("No models registered yet.")

    print("Production models remain unchanged.")


if __name__ == "__main__":

    main()
