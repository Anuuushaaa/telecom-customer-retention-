"""
src/threshold_tuning.py

Evaluate classification thresholds using
uncalibrated and calibrated validation probabilities.

The test partition must not be used for
threshold selection.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

from src.optimization import (
    SUPPORTED_MODELS,
)

from src.bootstrap_evaluation import (
    calculate_file_hash,
    load_validation_predictions,
)

from src.probability_calibration import (
    CALIBRATION_METHODS,
    validate_probabilities,
    validate_labels,
)


DATA_PATH = Path(
    "data/raw/telco_churn.csv"
)

PREDICTIONS_PATH = Path(
    "models/candidates/calibration/"
    "calibrated_validation_predictions.csv"
)

CALIBRATION_METADATA_PATH = Path(
    "reports/calibration/calibration_metadata.json"
)

SELECTED_METHODS_PATH = Path(
    "reports/calibration/selected_calibration_methods.csv"
)

REPORTS_DIR = Path(
    "reports/thresholds"
)

DEFAULT_THRESHOLDS = np.round(
    np.arange(0.01, 1.00, 0.01),
    2,
)

RECALL_TARGETS = (
    0.80,
    0.90,
)


def validate_thresholds(thresholds):
    """Validate candidate classification thresholds."""

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    if thresholds.ndim != 1:
        raise ValueError(
            "Thresholds must be one-dimensional."
        )

    if len(thresholds) == 0:
        raise ValueError(
            "Threshold list is empty."
        )

    if not np.isfinite(thresholds).all():
        raise ValueError(
            "Nonfinite threshold values."
        )

    if (
        (thresholds < 0).any()
        or (thresholds > 1).any()
    ):
        raise ValueError(
            "Thresholds must be between 0 and 1."
        )

    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError(
            "Duplicate threshold values."
        )

    return np.sort(thresholds)


def load_calibrated_predictions():
    """Load and verify calibrated validation predictions."""

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    with CALIBRATION_METADATA_PATH.open(
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if metadata["test_evaluated"] is not False:
        raise ValueError(
            "Unexpected test evaluation metadata."
        )

    if metadata["production_updated"] is not False:
        raise ValueError(
            "Unexpected production update."
        )

    if metadata["evaluation_partition"] != "validation":
        raise ValueError(
            "Expected validation predictions."
        )

    if metadata["dataset_hash"] != calculate_file_hash(
        DATA_PATH
    ):
        raise ValueError(
            "Dataset fingerprint mismatch."
        )

    if metadata["calibrated_predictions_hash"] != (
        calculate_file_hash(PREDICTIONS_PATH)
    ):
        raise ValueError(
            "Calibrated predictions fingerprint mismatch."
        )

    if len(predictions) != metadata["validation_rows"]:
        raise ValueError(
            "Validation row count mismatch."
        )

    original_predictions, original_metadata = (
        load_validation_predictions()
    )

    if metadata["dataset_hash"] != (
        original_metadata["dataset_hash"]
    ):
        raise ValueError(
            "Original validation dataset mismatch."
        )

    original_path = Path(
        "models/candidates/optimized/"
        "validation_predictions.csv"
    )

    if metadata["validation_predictions_hash"] != (
        calculate_file_hash(original_path)
    ):
        raise ValueError(
            "Original predictions fingerprint mismatch."
        )

    validate_calibrated_predictions(
        predictions,
        original_predictions,
    )

    return predictions, metadata


def validate_calibrated_predictions(
    predictions,
    original_predictions,
):
    """Verify prediction membership, labels and probabilities."""

    required_columns = {
        "customerID",
        "actual_churn",
    }

    required_columns.update(
        f"{model}_{method}_probability"
        for model in SUPPORTED_MODELS
        for method in CALIBRATION_METHODS
    )

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "Calibrated predictions are empty."
        )

    if predictions["customerID"].isna().any():
        raise ValueError(
            "Missing customer IDs."
        )

    if predictions["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs."
        )

    expected_ids = set(
        original_predictions["customerID"]
    )

    observed_ids = set(
        predictions["customerID"]
    )

    if expected_ids != observed_ids:
        raise ValueError(
            "Validation customer membership mismatch."
        )

    aligned_original = (
        original_predictions
        .set_index("customerID")
        .reindex(predictions["customerID"])
    )

    if not np.array_equal(
        aligned_original["actual_churn"].to_numpy(),
        predictions["actual_churn"].to_numpy(),
    ):
        raise ValueError(
            "Validation customer labels mismatch."
        )

    validate_labels(
        predictions["actual_churn"],
        len(predictions),
    )

    for model_name in SUPPORTED_MODELS:

        for method in CALIBRATION_METHODS:

            column = (
                f"{model_name}_{method}_probability"
            )

            validate_probabilities(
                predictions[column]
            )

        original_probabilities = aligned_original[
            f"{model_name}_probability"
        ].to_numpy()

        uncalibrated_probabilities = predictions[
            f"{model_name}_uncalibrated_probability"
        ].to_numpy()

        if not np.allclose(
            original_probabilities,
            uncalibrated_probabilities,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Uncalibrated probability mismatch: "
                f"{model_name}"
            )


def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate classification metrics at one threshold."""

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    if (
        not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Invalid classification threshold."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    false_positive_rate = (
        1 - specificity
    )

    false_negative_rate = (
        1 - sensitivity
    )

    balanced_accuracy = (
        sensitivity + specificity
    ) / 2

    youden_j = (
        sensitivity + specificity - 1
    )

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "f2": float(
            fbeta_score(
                labels,
                predictions,
                beta=2,
                zero_division=0,
            )
        ),

        "specificity": float(specificity),

        "balanced_accuracy": float(
            balanced_accuracy
        ),

        "youden_j": float(youden_j),

        "false_positive_rate": float(
            false_positive_rate
        ),

        "false_negative_rate": float(
            false_negative_rate
        ),

        "true_positives": int(tp),

        "true_negatives": int(tn),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "customers_targeted": int(
            predictions.sum()
        ),

        "targeting_rate": float(
            predictions.mean()
        ),
    }


def sweep_probability_thresholds(
    y_true,
    probabilities,
    thresholds=None,
):
    """Evaluate a sequence of probability thresholds."""

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    thresholds = validate_thresholds(
        thresholds
    )

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    records = []

    for threshold in thresholds:

        records.append(
            calculate_threshold_metrics(
                labels,
                probabilities,
                threshold,
            )
        )

    return pd.DataFrame(records)


def sweep_thresholds(
    model,
    X_val,
    y_val,
    thresholds=None,
):
    """Backward-compatible model-based threshold sweep."""

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    return sweep_probability_thresholds(
        y_val,
        probabilities,
        thresholds=thresholds,
    )


def evaluate_all_thresholds(
    predictions,
    thresholds=None,
):
    """Evaluate every model-calibration combination."""

    labels = predictions[
        "actual_churn"
    ].to_numpy()

    records = []

    for model_name in SUPPORTED_MODELS:

        for method in CALIBRATION_METHODS:

            probabilities = predictions[
                f"{model_name}_{method}_probability"
            ].to_numpy()

            results = sweep_probability_thresholds(
                labels,
                probabilities,
                thresholds=thresholds,
            )

            results.insert(
                0,
                "calibration_method",
                method,
            )

            results.insert(
                0,
                "model",
                model_name,
            )

            records.append(results)

            print(
                f"Threshold sweep completed: "
                f"{model_name} / {method}"
            )

    return pd.concat(
        records,
        ignore_index=True,
    )


def select_best_threshold(
    results,
    objective,
    minimum_recall=None,
):
    """Select a threshold using a specified objective."""

    supported_objectives = (
        "f1",
        "f2",
        "youden_j",
        "precision",
    )

    if objective not in supported_objectives:
        raise ValueError(
            f"Unsupported objective: {objective}"
        )

    candidates = results.copy()

    if minimum_recall is not None:

        if not 0 <= minimum_recall <= 1:
            raise ValueError(
                "Minimum recall must be between 0 and 1."
            )

        candidates = candidates[
            candidates["recall"] >= minimum_recall
        ]

    if candidates.empty:
        return None

    # Deterministic tie-breaking:
    # maximize objective, recall and precision,
    # then prefer the higher threshold.

    sort_columns = list(
        dict.fromkeys([
            objective,
            "recall",
            "precision",
            "threshold",
        ])
    )

    candidates = candidates.sort_values(
        sort_columns,
        ascending=False,
    )

    return candidates.iloc[0].to_dict()


def select_threshold_strategies(
    threshold_results,
):
    """Select five statistical threshold strategies."""

    records = []

    strategies = [
        ("maximum_f1", "f1", None),
        ("maximum_f2", "f2", None),
        ("maximum_youden_j", "youden_j", None),
    ]

    strategies.extend(
        (
            f"recall_at_least_{int(target * 100)}",
            "precision",
            target,
        )
        for target in RECALL_TARGETS
    )

    for model_name in SUPPORTED_MODELS:

        for method in CALIBRATION_METHODS:

            subset = threshold_results[
                (threshold_results["model"] == model_name)
                & (
                    threshold_results["calibration_method"]
                    == method
                )
            ]

            if subset.empty:
                raise ValueError(
                    f"Missing threshold results: "
                    f"{model_name}/{method}"
                )

            for strategy, objective, target in strategies:

                selected = select_best_threshold(
                    subset,
                    objective=objective,
                    minimum_recall=target,
                )

                if selected is None:

                    records.append({
                        "model": model_name,
                        "calibration_method": method,
                        "strategy": strategy,
                        "objective": objective,
                        "minimum_recall": target,
                        "feasible": False,
                    })

                    continue

                records.append({
                    **selected,

                    "strategy": strategy,

                    "objective": objective,

                    "minimum_recall": target,

                    "feasible": True,
                })

    return pd.DataFrame(records)


def select_preferred_calibration_thresholds(
    selected_thresholds,
    selected_methods,
):
    """Keep threshold strategies for preferred calibrators."""

    required = {
        "model",
        "calibration_method",
    }

    if not required.issubset(
        selected_methods.columns
    ):
        raise ValueError(
            "Invalid calibration selection columns."
        )

    if selected_methods["model"].duplicated().any():
        raise ValueError(
            "Duplicate selected calibration models."
        )

    if set(selected_methods["model"]) != set(
        SUPPORTED_MODELS
    ):
        raise ValueError(
            "Selected calibration model mismatch."
        )

    if not selected_methods[
        "calibration_method"
    ].isin(CALIBRATION_METHODS).all():
        raise ValueError(
            "Unsupported selected calibration method."
        )

    preferred = selected_thresholds.merge(
        selected_methods[
            ["model", "calibration_method"]
        ],
        on=[
            "model",
            "calibration_method",
        ],
        how="inner",
        validate="many_to_one",
    )

    expected_rows = (
        len(SUPPORTED_MODELS)
        * (3 + len(RECALL_TARGETS))
    )

    if len(preferred) != expected_rows:
        raise ValueError(
            "Incomplete preferred threshold results."
        )

    return preferred


def compare_default_thresholds(
    threshold_results,
    preferred_thresholds,
):
    """Compare selected strategies against threshold 0.50."""

    default_results = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            0.50,
        )
    ]

    records = []

    for _, selected in preferred_thresholds.iterrows():

        if not selected["feasible"]:
            continue

        baseline = default_results[
            (
                default_results["model"]
                == selected["model"]
            )
            & (
                default_results["calibration_method"]
                == selected["calibration_method"]
            )
        ]

        if len(baseline) != 1:
            raise ValueError(
                "Expected one default threshold result."
            )

        baseline = baseline.iloc[0]

        records.append({
            "model": selected["model"],

            "calibration_method": (
                selected["calibration_method"]
            ),

            "strategy": selected["strategy"],

            "default_threshold": 0.50,

            "selected_threshold": (
                selected["threshold"]
            ),

            "default_precision": (
                baseline["precision"]
            ),

            "selected_precision": (
                selected["precision"]
            ),

            "precision_change": (
                selected["precision"]
                - baseline["precision"]
            ),

            "default_recall": baseline["recall"],

            "selected_recall": selected["recall"],

            "recall_change": (
                selected["recall"]
                - baseline["recall"]
            ),

            "default_f1": baseline["f1"],

            "selected_f1": selected["f1"],

            "f1_change": (
                selected["f1"]
                - baseline["f1"]
            ),

            "default_f2": baseline["f2"],

            "selected_f2": selected["f2"],

            "f2_change": (
                selected["f2"]
                - baseline["f2"]
            ),

            "customers_targeted_change": int(
                selected["customers_targeted"]
                - baseline["customers_targeted"]
            ),

            "false_negatives_change": int(
                selected["false_negatives"]
                - baseline["false_negatives"]
            ),
        })

    return pd.DataFrame(records)


def save_threshold_results(
    threshold_results,
    selected_thresholds,
    preferred_thresholds,
    default_comparison,
    metadata,
    output_dir=REPORTS_DIR,
):
    """Export threshold analysis reports."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_csv(
        output_dir / "threshold_sweep.csv",
        index=False,
    )

    selected_thresholds.to_csv(
        output_dir / "selected_thresholds.csv",
        index=False,
    )

    preferred_thresholds.to_csv(
        output_dir / "preferred_calibration_thresholds.csv",
        index=False,
    )

    default_comparison.to_csv(
        output_dir / "default_threshold_comparison.csv",
        index=False,
    )

    with (
        output_dir / "threshold_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    print(
        f"\nThreshold reports saved: {output_dir}"
    )


def main():
    """Run validation-only threshold optimization."""

    print(
        "\nLoading verified calibrated predictions..."
    )

    predictions, calibration_metadata = (
        load_calibrated_predictions()
    )

    selected_methods = pd.read_csv(
        SELECTED_METHODS_PATH
    )

    print(
        "\nThreshold optimization configuration"
    )

    print(
        f"Validation customers: {len(predictions):,}"
    )

    print(
        f"Models: {len(SUPPORTED_MODELS)}"
    )

    print(
        f"Calibration methods: "
        f"{len(CALIBRATION_METHODS)}"
    )

    print(
        f"Thresholds per combination: "
        f"{len(DEFAULT_THRESHOLDS)}"
    )

    threshold_results = evaluate_all_thresholds(
        predictions,
        thresholds=DEFAULT_THRESHOLDS,
    )

    selected_thresholds = select_threshold_strategies(
        threshold_results
    )

    preferred_thresholds = (
        select_preferred_calibration_thresholds(
            selected_thresholds,
            selected_methods,
        )
    )

    default_comparison = compare_default_thresholds(
        threshold_results,
        preferred_thresholds,
    )

    metadata = {
        "dataset_hash": calibration_metadata[
            "dataset_hash"
        ],

        "calibrated_predictions_hash": (
            calculate_file_hash(
                PREDICTIONS_PATH
            )
        ),

        "selected_calibration_methods_hash": (
            calculate_file_hash(
                SELECTED_METHODS_PATH
            )
        ),

        "validation_rows": len(predictions),

        "models": list(SUPPORTED_MODELS),

        "calibration_methods": list(
            CALIBRATION_METHODS
        ),

        "threshold_min": float(
            DEFAULT_THRESHOLDS.min()
        ),

        "threshold_max": float(
            DEFAULT_THRESHOLDS.max()
        ),

        "threshold_step": 0.01,

        "threshold_count": len(
            DEFAULT_THRESHOLDS
        ),

        "recall_targets": list(
            RECALL_TARGETS
        ),

        "selection_strategies": [
            "maximum_f1",
            "maximum_f2",
            "maximum_youden_j",
            "recall_at_least_80",
            "recall_at_least_90",
        ],

        "evaluation_partition": "validation",

        "business_cost_optimization": False,

        "test_evaluated": False,

        "production_updated": False,
    }

    save_threshold_results(
        threshold_results,
        selected_thresholds,
        preferred_thresholds,
        default_comparison,
        metadata,
    )

    display_columns = [
        "model",
        "calibration_method",
        "strategy",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "specificity",
        "customers_targeted",
        "false_negatives",
    ]

    print(
        "\nPREFERRED CALIBRATION THRESHOLDS"
    )

    print(
        preferred_thresholds[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\nThreshold evaluations: "
        f"{len(threshold_results)}"
    )

    print(
        f"Selected strategies: "
        f"{len(selected_thresholds)}"
    )

    print(
        f"Preferred calibration strategies: "
        f"{len(preferred_thresholds)}"
    )

    print(
        "\nThreshold optimization completed."
    )

    print(
        "Production models and thresholds remain unchanged."
    )

    print(
        "The test partition has not been evaluated."
    )


if __name__ == "__main__":
    main()