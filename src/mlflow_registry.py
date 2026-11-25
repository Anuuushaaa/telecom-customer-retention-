
"""
MLflow Model Registry integration.

Registers verified telecom churn inference models,
prevents duplicate model registrations and verifies
registered model prediction consistency.

Production aliases are not assigned in this step.
"""

import json

import mlflow
import numpy as np
import pandas as pd

from src.mlflow_config import (
    DATA_PATH,
    REPORTS_DIR,
    REGISTERED_MODEL_NAME,
    SUPPORTED_MODELS,
    MODEL_DISPLAY_NAMES,
    get_mlflow_client,
    initialize_experiment,
)

from src.mlflow_tracking import (
    require_file,
    calculate_file_hash,
)

from src.mlflow_packaging import (
    PACKAGING_SUMMARY_PATH,
    PACKAGING_METADATA_PATH,
    load_raw_validation_data,
    verify_serialized_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

REGISTRY_SUMMARY_PATH = (
    REPORTS_DIR / "registered_model_summary.csv"
)

REGISTRY_METADATA_PATH = (
    REPORTS_DIR / "registry_metadata.json"
)

REGISTRY_SCHEMA_VERSION = "1"


# ============================================================
# VALIDATE PACKAGED MODELS
# ============================================================

def load_packaged_models():
    """Load and validate verified model packages."""

    summary = pd.read_csv(
        require_file(PACKAGING_SUMMARY_PATH)
    )

    with require_file(
        PACKAGING_METADATA_PATH
    ).open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    required = {
        "model",
        "variant",
        "run_id",
        "model_uri",
        "model_id",
        "calibration_method",
        "classification_threshold",
        "verified_rows",
        "classification_verified",
        "threshold_override_verified",
        "serialization_verified",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing packaging columns: "
            f"{sorted(missing)}"
        )

    if summary["model"].duplicated().any():
        raise ValueError(
            "Duplicate packaged models."
        )

    if set(summary["model"]) != set(SUPPORTED_MODELS):
        raise ValueError(
            "Packaged models are incomplete."
        )

    if summary["model_id"].isna().any():
        raise ValueError(
            "Missing logged model identifiers."
        )

    if summary["model_id"].duplicated().any():
        raise ValueError(
            "Duplicate logged model identifiers."
        )

    if not summary["variant"].eq("optimized").all():
        raise ValueError(
            "Only optimized models may be registered."
        )

    for column in (
        "classification_verified",
        "threshold_override_verified",
        "serialization_verified",
    ):

        if not summary[column].astype(
            str
        ).str.lower().eq("true").all():

            raise ValueError(
                f"Verification failed: {column}"
            )

    if metadata["dataset_hash"] != calculate_file_hash(
        DATA_PATH
    ):
        raise ValueError(
            "Dataset fingerprint mismatch."
        )

    if metadata["serialization_verified"] is not True:
        raise ValueError(
            "Packaging verification is incomplete."
        )

    if metadata["test_evaluated"] is not False:
        raise ValueError(
            "Unexpected test evaluation metadata."
        )

    return summary, metadata


# ============================================================
# REGISTERED MODEL
# ============================================================

def ensure_registered_model(client):
    """Create the registered model if it does not exist."""

    try:

        registered_model = client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

    except mlflow.exceptions.MlflowException as error:

        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise

        registered_model = (
            client.create_registered_model(
                name=REGISTERED_MODEL_NAME,
                description=(
                    "Telecom customer churn classification "
                    "with unified preprocessing, probability "
                    "calibration and configurable thresholds."
                ),
                tags={
                    "project": "telecom-customer-retention",
                    "project_version": "2.0",
                    "task": "binary_classification",
                    "domain": "telecommunications",
                },
            )
        )

    return registered_model


# ============================================================
# DUPLICATE REGISTRATION PREVENTION
# ============================================================

def find_existing_version(
    client,
    model_id,
    run_id,
):
    """Find an existing version for a logged model."""

    versions = client.search_model_versions(
        filter_string=(
            f"name = '{REGISTERED_MODEL_NAME}'"
        )
    )

    matches = []

    for candidate in versions:

        tags = candidate.tags or {}

        same_model = (
            tags.get("logged_model_id") == model_id
            or getattr(candidate, "model_id", None)
            == model_id
            or candidate.source == f"models:/{model_id}"
        )

        if same_model:

            if candidate.run_id != run_id:
                raise ValueError(
                    "Registered model run ID mismatch."
                )

            matches.append(candidate)

    if len(matches) > 1:
        raise ValueError(
            f"Multiple registered versions found "
            f"for logged model {model_id}."
        )

    return matches[0] if matches else None


# ============================================================
# SOURCE RUN VERIFICATION
# ============================================================

def verify_source_run(client, row):
    """Verify the source experiment and logged model."""

    run = client.get_run(row["run_id"])

    if run.info.status != "FINISHED":
        raise ValueError(
            "Source experiment run is not finished."
        )

    tags = run.data.tags

    if tags.get("model_name") != row["model"]:
        raise ValueError(
            "Source run model mismatch."
        )

    if tags.get("model_variant") != "optimized":
        raise ValueError(
            "Source run variant mismatch."
        )

    if tags.get("serialization_verified") != "true":
        raise ValueError(
            "Source run serialization not verified."
        )

    if tags.get("packaged_model_uri") != row["model_uri"]:
        raise ValueError(
            "Packaged model URI mismatch."
        )

    logged_model = client.get_logged_model(
        row["model_id"]
    )

    if logged_model.source_run_id != row["run_id"]:
        raise ValueError(
            "Logged model source run mismatch."
        )

    return run


# ============================================================
# MODEL VERSION TAGS
# ============================================================

def build_version_tags(row, run):
    """Create model version metadata."""

    metrics = run.data.metrics

    return {
        "algorithm": row["model"],
        "model_variant": "optimized",
        "logged_model_id": row["model_id"],
        "source_run_id": row["run_id"],
        "calibration_method": row["calibration_method"],
        "classification_threshold": str(
            row["classification_threshold"]
        ),
        "dataset_hash": calculate_file_hash(DATA_PATH),
        "evaluation_partition": "validation",
        "model_lifecycle": "candidate",
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "validation_roc_auc": str(
            metrics["validation_roc_auc"]
        ),
        "validation_f1": str(
            metrics["validation_f1"]
        ),
        "selected_calibration_brier_score": str(
            metrics["selected_calibration_brier_score"]
        ),
        "business_expected_net_profit": str(
            metrics[
                "business_budget_constrained_expected_net_profit"
            ]
        ),
        "financial_results": "hypothetical_simulation",
        "production_updated": "false",
    }


# ============================================================
# REGISTER AND VERIFY MODEL VERSION
# ============================================================

def register_candidate(
    client,
    row,
    validation,
    expected_predictions,
):
    """Register and verify a single candidate model."""

    model_name = row["model"]

    print("\n" + "=" * 70)

    print(
        f"REGISTERING: "
        f"{MODEL_DISPLAY_NAMES[model_name]}"
    )

    print("=" * 70)

    run = verify_source_run(
        client,
        row,
    )

    existing = find_existing_version(
        client,
        row["model_id"],
        row["run_id"],
    )

    if existing is None:

        model_version = mlflow.register_model(
            model_uri=row["model_uri"],
            name=REGISTERED_MODEL_NAME,
            await_registration_for=300,
            tags=build_version_tags(row, run),
        )

        registration_status = "registered"

    else:

        model_version = existing

        registration_status = "existing"

        print(
            f"Reusing existing version: "
            f"{model_version.version}"
        )

    version_number = str(
        model_version.version
    )

    model_version = client.get_model_version(
        REGISTERED_MODEL_NAME,
        version_number,
    )

    if model_version.status != "READY":
        raise ValueError(
            f"Model version is not ready: "
            f"{model_version.status}"
        )

    registry_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/"
        f"{version_number}"
    )

    verification = verify_serialized_model(
        model_uri=registry_uri,
        model_name=model_name,
        calibration_method=row["calibration_method"],
        validation=validation,
        expected_predictions=expected_predictions,
    )

    tags = build_version_tags(
        row,
        run,
    )

    tags["registry_verification"] = "passed"

    tags["verified_rows"] = str(
        verification["verified_rows"]
    )

    for key, value in tags.items():

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=version_number,
            key=key,
            value=value,
        )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version_number,
        description=(
            f"{MODEL_DISPLAY_NAMES[model_name]} "
            "optimized using Optuna, with "
            f"{row['calibration_method']} probability "
            "calibration and verified raw-input inference. "
            "Candidate evaluated on validation data."
        ),
    )

    metrics = run.data.metrics

    print(f"Version: {version_number}")

    print(f"Registry URI: {registry_uri}")

    print(
        f"Validation rows verified: "
        f"{verification['verified_rows']}"
    )

    print("Registry loading: PASSED")

    print("Prediction consistency: PASSED")

    print("Threshold override: PASSED")

    return {
        "model": model_name,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": version_number,
        "registry_uri": registry_uri,
        "model_uri": row["model_uri"],
        "model_id": row["model_id"],
        "run_id": row["run_id"],
        "calibration_method": row["calibration_method"],
        "classification_threshold": float(
            row["classification_threshold"]
        ),
        "validation_roc_auc": metrics[
            "validation_roc_auc"
        ],
        "validation_f1": metrics[
            "validation_f1"
        ],
        "brier_score": metrics[
            "selected_calibration_brier_score"
        ],
        "expected_net_profit": metrics[
            "business_budget_constrained_expected_net_profit"
        ],
        "registration_status": registration_status,
        **verification,
    }


# ============================================================
# SAVE REGISTRY REPORTS
# ============================================================

def save_registry_reports(results):
    """Save model version identifiers and verification."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(results)

    persistent_summary = summary.drop(
        columns=["registration_status"]
    )

    persistent_summary.to_csv(
        REGISTRY_SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "registry_schema_version": (
            REGISTRY_SCHEMA_VERSION
        ),
        "registered_model_name": (
            REGISTERED_MODEL_NAME
        ),
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "total_registered_candidates": len(summary),
        "total_verified_predictions": int(
            summary["verified_rows"].sum()
        ),
        "model_versions": {
            row["model"]: str(row["version"])
            for _, row in summary.iterrows()
        },
        "evaluation_partition": "validation",
        "registry_verification": True,
        "training_repeated": False,
        "test_evaluated": False,
        "aliases_assigned_by_registry_step": False,
        "production_updated": False,
        "financial_results": "hypothetical_simulation",
    }

    with REGISTRY_METADATA_PATH.open(
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

    print("=" * 70)

    print("TELECOM CUSTOMER RETENTION")

    print("MLFLOW MODEL REGISTRY")

    print("=" * 70)

    initialize_experiment()

    client = get_mlflow_client()

    packaged_models, metadata = (
        load_packaged_models()
    )

    validation, predictions, _ = (
        load_raw_validation_data()
    )

    if not packaged_models["verified_rows"].eq(
        len(validation)
    ).all():

        raise ValueError(
            "Packaged validation row count mismatch."
        )

    ensure_registered_model(client)

    results = []

    for model_name in SUPPORTED_MODELS:

        row = packaged_models[
            packaged_models["model"] == model_name
        ].iloc[0]

        result = register_candidate(
            client,
            row,
            validation,
            predictions,
        )

        results.append(result)

    summary = save_registry_reports(results)

    print("\n" + "=" * 70)

    print("MODEL REGISTRY SUMMARY")

    print("=" * 70)

    print(
        summary[
            [
                "model",
                "version",
                "validation_roc_auc",
                "brier_score",
                "expected_net_profit",
            ]
        ].round(6).to_string(index=False)
    )

    print(
        f"\nRegistered candidates: {len(summary)}"
    )

    print(
        "New versions: "
        f"{summary['registration_status'].eq('registered').sum()}"
    )

    print(
        "Existing versions reused: "
        f"{summary['registration_status'].eq('existing').sum()}"
    )

    print(
        "Total predictions verified: "
        f"{summary['verified_rows'].sum()}"
    )

    print("\nReports generated:")

    print(" - registered_model_summary.csv")

    print(" - registry_metadata.json")

    print("\nModel registry integration completed.")

    print("No production aliases assigned.")

    print("Production models remain unchanged.")


if __name__ == "__main__":

    main()
