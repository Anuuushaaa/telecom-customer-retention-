
"""
MLflow champion and challenger model selection.

Features:
- Financially motivated model ranking
- Registry integrity verification
- Champion and challenger alias assignment
- Previous alias preservation
- Alias-based inference verification
- Promotion audit reports

Does not overwrite existing production model files.
"""

import json

from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd

from src.mlflow_config import (
    DATA_PATH,
    REPORTS_DIR,
    REGISTERED_MODEL_NAME,
    PRODUCTION_ALIAS,
    CHALLENGER_ALIAS,
    SUPPORTED_MODELS,
    get_mlflow_client,
    initialize_experiment,
)

from src.mlflow_tracking import (
    require_file,
    calculate_file_hash,
)

from src.mlflow_registry import (
    REGISTRY_SUMMARY_PATH,
    REGISTRY_METADATA_PATH,
)

from src.mlflow_packaging import (
    load_raw_validation_data,
    verify_serialized_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROMOTION_SUMMARY_PATH = (
    REPORTS_DIR / "model_promotion_summary.csv"
)

PROMOTION_METADATA_PATH = (
    REPORTS_DIR / "promotion_metadata.json"
)

PROMOTION_HISTORY_PATH = (
    REPORTS_DIR / "promotion_history.json"
)

SELECTION_OBJECTIVE = (
    "budget_constrained_expected_net_profit"
)

PROMOTION_SCHEMA_VERSION = "1"


# ============================================================
# LOAD REGISTERED CANDIDATES
# ============================================================

def load_registered_candidates():
    """Load and validate registered model candidates."""

    candidates = pd.read_csv(
        require_file(REGISTRY_SUMMARY_PATH),
        dtype={"version": str},
    )

    with require_file(
        REGISTRY_METADATA_PATH
    ).open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    required = {
        "model",
        "version",
        "run_id",
        "model_id",
        "registry_uri",
        "calibration_method",
        "classification_threshold",
        "validation_roc_auc",
        "validation_f1",
        "brier_score",
        "expected_net_profit",
        "serialization_verified",
        "classification_verified",
        "threshold_override_verified",
    }

    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing registry columns: {sorted(missing)}"
        )

    if candidates["model"].duplicated().any():
        raise ValueError("Duplicate candidate models.")

    if candidates["version"].duplicated().any():
        raise ValueError("Duplicate registry versions.")

    if set(candidates["model"]) != set(SUPPORTED_MODELS):
        raise ValueError("Incomplete candidate models.")

    if metadata["dataset_hash"] != calculate_file_hash(
        DATA_PATH
    ):
        raise ValueError("Dataset fingerprint mismatch.")

    if metadata["registry_verification"] is not True:
        raise ValueError("Registry verification incomplete.")

    for column in (
        "serialization_verified",
        "classification_verified",
        "threshold_override_verified",
    ):

        if not candidates[column].astype(
            str
        ).str.lower().eq("true").all():

            raise ValueError(
                f"Candidate verification failed: {column}"
            )

    numeric_columns = [
        "validation_roc_auc",
        "validation_f1",
        "brier_score",
        "expected_net_profit",
    ]

    for column in numeric_columns:

        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="raise",
        )

        if not np.isfinite(candidates[column]).all():
            raise ValueError(
                f"Non-finite values in {column}."
            )

    for column in (
        "validation_roc_auc",
        "validation_f1",
        "brier_score",
    ):

        if not candidates[column].between(0, 1).all():
            raise ValueError(
                f"Invalid metric range: {column}"
            )

    return candidates


# ============================================================
# REGISTRY INTEGRITY
# ============================================================

def verify_registry_candidates(client, candidates):
    """Verify candidate identity and readiness."""

    for _, row in candidates.iterrows():

        model_version = client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=row["version"],
        )

        if model_version.status != "READY":
            raise ValueError(
                f"Version {row['version']} is not ready."
            )

        if model_version.run_id != row["run_id"]:
            raise ValueError("Source run mismatch.")

        tags = model_version.tags or {}

        if tags.get("algorithm") != row["model"]:
            raise ValueError("Algorithm mismatch.")

        if tags.get("logged_model_id") != row["model_id"]:
            raise ValueError("Logged model ID mismatch.")

        if tags.get("registry_verification") != "passed":
            raise ValueError(
                "Registry verification missing."
            )

        if tags.get("dataset_hash") != calculate_file_hash(
            DATA_PATH
        ):
            raise ValueError("Registry dataset mismatch.")

        run = client.get_run(row["run_id"])

        metric_mapping = {
            "validation_roc_auc": "validation_roc_auc",
            "validation_f1": "validation_f1",
            "brier_score": (
                "selected_calibration_brier_score"
            ),
            "expected_net_profit": (
                "business_budget_constrained_expected_net_profit"
            ),
        }

        for column, metric in metric_mapping.items():

            if not np.isclose(
                float(row[column]),
                float(run.data.metrics[metric]),
                atol=1e-8,
                rtol=1e-8,
            ):
                raise ValueError(
                    f"Registry metric mismatch: {column}"
                )


# ============================================================
# MODEL RANKING
# ============================================================

def rank_candidates(candidates):
    """Rank candidates using the business objective."""

    ranked = candidates.sort_values(
        by=[
            "expected_net_profit",
            "validation_roc_auc",
            "brier_score",
            "model",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True).copy()

    ranked["rank"] = np.arange(
        1,
        len(ranked) + 1,
    )

    ranked["assigned_role"] = "candidate"

    ranked.loc[
        ranked["rank"] == 1,
        "assigned_role",
    ] = PRODUCTION_ALIAS

    ranked.loc[
        ranked["rank"] == 2,
        "assigned_role",
    ] = CHALLENGER_ALIAS

    return ranked


# ============================================================
# ALIAS MANAGEMENT
# ============================================================

def get_alias_version(client, alias):
    """Return the current alias version, if present."""

    registered_model = client.get_registered_model(
        REGISTERED_MODEL_NAME
    )

    version = (
        registered_model.aliases or {}
    ).get(alias)

    return str(version) if version is not None else None


def capture_current_aliases(client):
    """Capture existing aliases before promotion."""

    return {
        PRODUCTION_ALIAS: get_alias_version(
            client,
            PRODUCTION_ALIAS,
        ),
        CHALLENGER_ALIAS: get_alias_version(
            client,
            CHALLENGER_ALIAS,
        ),
    }


def restore_aliases(client, previous_aliases):
    """Restore aliases if promotion verification fails."""

    for alias, version in previous_aliases.items():

        if version is None:

            if get_alias_version(client, alias) is not None:

                client.delete_registered_model_alias(
                    name=REGISTERED_MODEL_NAME,
                    alias=alias,
                )

        else:

            client.set_registered_model_alias(
                name=REGISTERED_MODEL_NAME,
                alias=alias,
                version=version,
            )


def assign_aliases(client, ranked):
    """Assign champion and challenger aliases."""

    assigned = {}

    for alias in (
        PRODUCTION_ALIAS,
        CHALLENGER_ALIAS,
    ):

        selected = ranked[
            ranked["assigned_role"] == alias
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected one model for alias {alias}."
            )

        version = str(
            selected.iloc[0]["version"]
        )

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=alias,
            version=version,
        )

        assigned[alias] = version

    return assigned


# ============================================================
# ALIAS VERIFICATION
# ============================================================

def verify_promoted_aliases(
    client,
    ranked,
    validation,
    expected_predictions,
):
    """Verify inference through champion and challenger."""

    results = []

    for alias in (
        PRODUCTION_ALIAS,
        CHALLENGER_ALIAS,
    ):

        row = ranked[
            ranked["assigned_role"] == alias
        ].iloc[0]

        resolved = (
            client.get_model_version_by_alias(
                name=REGISTERED_MODEL_NAME,
                alias=alias,
            )
        )

        if str(resolved.version) != str(row["version"]):
            raise ValueError(
                f"Alias resolution failed: {alias}"
            )

        model_uri = (
            f"models:/{REGISTERED_MODEL_NAME}@{alias}"
        )

        verification = verify_serialized_model(
            model_uri=model_uri,
            model_name=row["model"],
            calibration_method=row["calibration_method"],
            validation=validation,
            expected_predictions=expected_predictions,
        )

        results.append({
            "alias": alias,
            "model": row["model"],
            "version": str(row["version"]),
            "model_uri": model_uri,
            **verification,
        })

        print(f"\nAlias: {alias}")
        print(f"Model: {row['model']}")
        print(f"Version: {row['version']}")
        print(f"URI: {model_uri}")
        print("Alias resolution: PASSED")
        print("Prediction verification: PASSED")

    return results


# ============================================================
# MODEL VERSION METADATA
# ============================================================

def update_lifecycle_tags(client, ranked):
    """Update lifecycle metadata for ranked candidates."""

    for _, row in ranked.iterrows():

        version = str(row["version"])

        tags = {
            "model_lifecycle": row["assigned_role"],
            "selection_rank": str(row["rank"]),
            "selection_objective": SELECTION_OBJECTIVE,
            "selection_partition": "validation",
            "financial_results": "hypothetical_simulation",
            "promotion_schema_version": (
                PROMOTION_SCHEMA_VERSION
            ),
        }

        for key, value in tags.items():

            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=version,
                key=key,
                value=value,
            )


# ============================================================
# PROMOTION REPORTS
# ============================================================

def save_promotion_reports(
    ranked,
    previous_aliases,
    assigned_aliases,
    verification,
):
    """Save selection results and promotion audit history."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = ranked[
        [
            "rank",
            "model",
            "version",
            "assigned_role",
            "calibration_method",
            "classification_threshold",
            "validation_roc_auc",
            "validation_f1",
            "brier_score",
            "expected_net_profit",
        ]
    ].copy()

    summary.to_csv(
        PROMOTION_SUMMARY_PATH,
        index=False,
    )

    champion = ranked.iloc[0]
    challenger = ranked.iloc[1]

    profit_difference = float(
        champion["expected_net_profit"]
        - challenger["expected_net_profit"]
    )

    metadata = {
        "promotion_schema_version": (
            PROMOTION_SCHEMA_VERSION
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "registered_model_name": (
            REGISTERED_MODEL_NAME
        ),
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "selection_objective": SELECTION_OBJECTIVE,
        "ranking_criteria": [
            "expected_net_profit_descending",
            "validation_roc_auc_descending",
            "brier_score_ascending",
            "model_name_ascending",
        ],
        "previous_aliases": previous_aliases,
        "assigned_aliases": assigned_aliases,
        "champion_model": champion["model"],
        "challenger_model": challenger["model"],
        "expected_profit_difference": profit_difference,
        "financial_results": "hypothetical_simulation",
        "evaluation_partition": "validation",
        "statistical_superiority_established": False,
        "alias_verification": verification,
        "registry_aliases_updated": True,
        "local_production_files_updated": False,
        "training_repeated": False,
        "test_evaluated": False,
    }

    with PROMOTION_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    history = []

    if PROMOTION_HISTORY_PATH.exists():

        with PROMOTION_HISTORY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(
            "Invalid promotion history format."
        )

    if previous_aliases != assigned_aliases:

        history.append(metadata)

    with PROMOTION_HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    return summary


# ============================================================
# PROMOTION WORKFLOW
# ============================================================

def promote_models():
    """Execute verified model alias promotion."""

    initialize_experiment()

    client = get_mlflow_client()

    candidates = load_registered_candidates()

    verify_registry_candidates(
        client,
        candidates,
    )

    ranked = rank_candidates(candidates)

    validation, predictions, _ = (
        load_raw_validation_data()
    )

    previous_aliases = capture_current_aliases(
        client
    )

    try:

        assigned_aliases = assign_aliases(
            client,
            ranked,
        )

        verification = verify_promoted_aliases(
            client,
            ranked,
            validation,
            predictions,
        )

    except Exception:

        restore_aliases(
            client,
            previous_aliases,
        )

        print(
            "Promotion failed. Previous aliases restored."
        )

        raise

    update_lifecycle_tags(
        client,
        ranked,
    )

    summary = save_promotion_reports(
        ranked,
        previous_aliases,
        assigned_aliases,
        verification,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TELECOM CUSTOMER RETENTION")
    print("MLFLOW MODEL PROMOTION")
    print("=" * 70)

    summary = promote_models()

    print("\n" + "=" * 70)
    print("MODEL PROMOTION SUMMARY")
    print("=" * 70)

    print(
        summary.round(6).to_string(
            index=False
        )
    )

    print("\nReports generated:")

    print(" - model_promotion_summary.csv")
    print(" - promotion_metadata.json")
    print(" - promotion_history.json")

    print("\nModel promotion completed.")

    print("Champion and challenger aliases verified.")

    print("No models were retrained.")

    print("Local production model files remain unchanged.")


if __name__ == "__main__":

    main()
