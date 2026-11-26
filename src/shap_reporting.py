
"""Generate global SHAP reports for the validation partition."""

import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api.schemas import CustomerInput
from frontend.api_client import TelecomAPIClient
from src.optimization import load_training_partitions


DATA_PATH = Path("data/raw/telco_churn.csv")

PREDICTIONS_PATH = Path(
    "models/candidates/optimized/validation_predictions.csv"
)

TRAINING_METADATA_PATH = Path(
    "models/candidates/optimized/training_metadata.json"
)

REPORTS_DIR = Path("reports/shap")

EXPECTED_VALIDATION_ROWS = 1409
EXPECTED_FEATURE_COUNT = 29


# ============================================================
# DATASET VERIFICATION
# ============================================================

def calculate_file_hash(path):

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_validation_customers():

    dataset_hash = calculate_file_hash(DATA_PATH)

    with TRAINING_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if metadata["dataset_hash"] != dataset_hash:

        raise ValueError(
            "Original dataset fingerprint mismatch."
        )

    _, validation, _ = load_training_partitions()

    saved_predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"customerID": str},
    )

    validation_ids = (
        validation["customerID"]
        .astype(str)
        .tolist()
    )

    if len(validation_ids) != EXPECTED_VALIDATION_ROWS:

        raise ValueError(
            "Unexpected validation partition size."
        )

    if len(set(validation_ids)) != len(validation_ids):

        raise ValueError(
            "Duplicate validation customer IDs."
        )

    if validation_ids != (
        saved_predictions["customerID"].tolist()
    ):

        raise ValueError(
            "Validation customer IDs do not match "
            "the saved optimized predictions."
        )

    if not np.array_equal(
        validation["Churn"].to_numpy(),
        saved_predictions["actual_churn"].to_numpy(),
    ):

        raise ValueError(
            "Validation target labels do not match."
        )

    raw = pd.read_csv(
        DATA_PATH,
        dtype={
            "customerID": str,
            "TotalCharges": str,
        },
        keep_default_na=False,
    )

    if raw["customerID"].duplicated().any():

        raise ValueError(
            "Duplicate IDs in original dataset."
        )

    raw_validation = (
        raw.set_index("customerID")
        .loc[validation_ids]
        .reset_index()
    )

    raw_validation = raw_validation[
        list(CustomerInput.model_fields)
    ]

    customers = [

        CustomerInput.model_validate(
            record
        ).model_dump(mode="json")

        for record in raw_validation.to_dict(
            orient="records"
        )
    ]

    return customers, dataset_hash


# ============================================================
# WEIGHTED SHAP AGGREGATION
# ============================================================

def aggregate_global_explanations(
    results,
    expected_customers,
    expected_model_version,
):

    if not results:

        raise ValueError(
            "No SHAP results were supplied."
        )

    total_customers = 0
    maximum_error = 0.0

    weighted_absolute = {}
    weighted_signed = {}

    expected_features = None

    for result in results:

        if result["model_name"] != "xgboost":

            raise ValueError(
                "Unexpected explanation model."
            )

        if str(result["model_version"]) != str(
            expected_model_version
        ):

            raise ValueError(
                "Model version changed between batches."
            )

        if result["explanation_method"] != "TreeSHAP":

            raise ValueError(
                "Unexpected explanation method."
            )

        if result["output_space"] != "raw_log_odds":

            raise ValueError(
                "Unexpected SHAP output space."
            )

        count = result["total_customers"]

        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):

            raise ValueError(
                "Invalid batch customer count."
            )

        importance = pd.DataFrame(
            result["feature_importance"]
        )

        if importance["feature"].duplicated().any():

            raise ValueError(
                "Duplicate SHAP feature names."
            )

        features = set(
            importance["feature"]
        )

        if (
            len(features) != EXPECTED_FEATURE_COUNT
            or result["total_features"] != len(features)
        ):

            raise ValueError(
                "Unexpected SHAP feature count."
            )

        if expected_features is None:

            expected_features = features

        elif features != expected_features:

            raise ValueError(
                "SHAP features differ between batches."
            )

        values = importance[
            [
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():

            raise ValueError(
                "Non-finite SHAP importance values."
            )

        if (values[:, 0] < 0).any():

            raise ValueError(
                "Negative mean absolute SHAP value."
            )

        if (
            np.abs(values[:, 1])
            > values[:, 0] + 1e-8
        ).any():

            raise ValueError(
                "Inconsistent signed SHAP averages."
            )

        for row in importance.itertuples(
            index=False
        ):

            feature = row.feature

            weighted_absolute[feature] = (
                weighted_absolute.get(feature, 0.0)
                + row.mean_absolute_shap * count
            )

            weighted_signed[feature] = (
                weighted_signed.get(feature, 0.0)
                + row.mean_signed_shap * count
            )

        batch_error = float(
            result["max_additivity_error"]
        )

        if (
            not np.isfinite(batch_error)
            or batch_error < 0
        ):

            raise ValueError(
                "Invalid SHAP additivity error."
            )

        maximum_error = max(
            maximum_error,
            batch_error,
        )

        total_customers += count

    if total_customers != expected_customers:

        raise ValueError(
            "Aggregated customer count mismatch."
        )

    importance = pd.DataFrame([

        {
            "feature": feature,

            "mean_absolute_shap": (
                weighted_absolute[feature]
                / total_customers
            ),

            "mean_signed_shap": (
                weighted_signed[feature]
                / total_customers
            ),
        }

        for feature in sorted(expected_features)
    ])

    importance = importance.sort_values(
        by=["mean_absolute_shap", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    importance.insert(
        0,
        "rank",
        np.arange(1, len(importance) + 1),
    )

    return importance, maximum_error


# ============================================================
# GLOBAL SHAP VISUALIZATION
# ============================================================

def plot_global_importance(
    importance,
    output_path,
    total_customers,
    model_version,
    top_n=15,
):

    display = (
        importance.head(top_n)
        .sort_values("mean_absolute_shap")
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.barh(
        display["feature"],
        display["mean_absolute_shap"],
        color="#3978A8",
    )

    ax.set_title(
        "Global TreeSHAP Feature Importance\n"
        f"XGBoost v{model_version} | "
        f"{total_customers:,} validation customers",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel(
        "Mean absolute SHAP value (raw log-odds)"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# REPORT GENERATION
# ============================================================

def main():

    customers, dataset_hash = (
        load_validation_customers()
    )

    client = TelecomAPIClient()

    readiness = client.get_readiness()

    if not readiness["model_loaded"]:

        raise RuntimeError(
            "Prediction model is not ready."
        )

    model_info = client.get_model_info()

    if model_info["model_name"] != "xgboost":

        raise RuntimeError(
            "Global TreeSHAP reporting requires XGBoost."
        )

    model_version = str(
        model_info["model_version"]
    )

    batch_size = int(
        model_info["max_batch_size"]
    )

    if batch_size <= 0:

        raise ValueError(
            "Invalid API batch size."
        )

    print("\nGLOBAL SHAP REPORT")
    print("Model:", model_info["model_name"])
    print("Version:", model_version)
    print("Validation customers:", len(customers))
    print("API batch size:", batch_size)

    results = []
    batch_sizes = []

    for start in range(
        0,
        len(customers),
        batch_size,
    ):

        batch = customers[
            start:start + batch_size
        ]

        print(
            f"\nExplaining batch {len(results) + 1}: "
            f"{len(batch)} customers"
        )

        result = client.explain_global(batch)

        if result["total_customers"] != len(batch):

            raise ValueError(
                "API returned an incorrect customer count."
            )

        results.append(result)
        batch_sizes.append(len(batch))

    final_model_info = client.get_model_info()

    if final_model_info != model_info:

        raise RuntimeError(
            "API model metadata changed during reporting."
        )

    importance, maximum_error = (
        aggregate_global_explanations(
            results=results,
            expected_customers=len(customers),
            expected_model_version=model_version,
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        REPORTS_DIR / "global_shap_importance.csv"
    )

    plot_path = (
        REPORTS_DIR / "global_shap_importance.png"
    )

    metadata_path = (
        REPORTS_DIR / "shap_metadata.json"
    )

    importance.to_csv(
        csv_path,
        index=False,
    )

    plot_global_importance(
        importance=importance,
        output_path=plot_path,
        total_customers=len(customers),
        model_version=model_version,
    )

    metadata = {
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),

        "dataset_hash": dataset_hash,

        "validation_predictions_hash": (
            calculate_file_hash(PREDICTIONS_PATH)
        ),

        "dataset_partition": "validation",

        "validation_customers": len(customers),

        "feature_count": len(importance),

        "registered_model_name": (
            model_info["registered_model_name"]
        ),

        "model_alias": model_info["model_alias"],

        "model_name": model_info["model_name"],

        "model_version": model_version,

        "run_id": model_info["run_id"],

        "calibration_method": (
            model_info["calibration_method"]
        ),

        "explanation_method": "TreeSHAP",

        "feature_perturbation": "tree_path_dependent",

        "output_space": "raw_log_odds",

        "aggregation_method": (
            "customer_count_weighted_mean"
        ),

        "batch_count": len(results),

        "batch_sizes": batch_sizes,

        "max_additivity_error": maximum_error,

        "additivity_verified_by_api": True,

        "test_partition_explained": False,

        "importance_report_hash": (
            calculate_file_hash(csv_path)
        ),

        "interpretation": (
            "SHAP explains model behavior, "
            "not causal effects or direct changes "
            "in calibrated probability."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nTOP 15 GLOBAL FEATURES")

    print(
        importance.head(15)
        .round(6)
        .to_string(index=False)
    )

    print("\nVERIFICATION SUMMARY")

    print("Customers:", len(customers))
    print("Features:", len(importance))
    print("Batches:", len(results))
    print("Batch sizes:", batch_sizes)
    print("Maximum additivity error:", maximum_error)

    print("\nREPORTS GENERATED")

    print(csv_path)
    print(plot_path)
    print(metadata_path)

    print("\nGLOBAL SHAP REPORT COMPLETED")


if __name__ == "__main__":

    main()
