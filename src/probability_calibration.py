"""
src/probability_calibration.py

Fit probability calibrators using out-of-fold
training predictions and evaluate them on
the separate validation partition.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

from src.optimization import (
    RANDOM_STATE,
    SUPPORTED_MODELS,
    load_training_partitions,
)

from src.bootstrap_evaluation import (
    calculate_file_hash,
    load_validation_predictions,
)

from src.calibration_oof import (
    create_calibration_folds,
)


DATA_PATH = Path(
    "data/raw/telco_churn.csv"
)

PARAMETERS_PATH = Path(
    "reports/optimization/best_hyperparameters.json"
)

OUTPUT_DIR = Path(
    "models/candidates/calibration"
)

REPORTS_DIR = Path(
    "reports/calibration"
)

OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"

OOF_METADATA_PATH = (
    REPORTS_DIR / "oof_metadata.json"
)

VALIDATION_PATH = Path(
    "models/candidates/optimized/validation_predictions.csv"
)

CALIBRATION_METHODS = (
    "uncalibrated",
    "sigmoid",
    "isotonic",
)

N_BINS = 10

EPSILON = 1e-7


def validate_probabilities(probabilities):
    """Validate a one-dimensional probability array."""

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if len(probabilities) == 0:
        raise ValueError(
            "Probability array is empty."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Nonfinite probability values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    return probabilities


def validate_labels(y_true, expected_rows):
    """Validate binary classification labels."""

    labels = np.asarray(y_true)

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be one-dimensional."
        )

    if len(labels) != expected_rows:
        raise ValueError(
            "Label and probability lengths differ."
        )

    if not np.array_equal(
        np.unique(labels),
        np.array([0, 1]),
    ):
        raise ValueError(
            "Both binary classes are required."
        )

    return labels.astype(int)


def probability_to_logit(probabilities):
    """Convert probabilities into stable log-odds."""

    probabilities = validate_probabilities(
        probabilities
    )

    clipped = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON,
    )

    logits = np.log(
        clipped / (1 - clipped)
    )

    return logits.reshape(-1, 1)


def validate_oof_predictions(
    predictions,
    train_df,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    """Verify OOF membership, labels and fold assignments."""

    required_columns = {
        "customerID",
        "actual_churn",
        "fold",
    }

    required_columns.update(
        f"{model_name}_probability"
        for model_name in SUPPORTED_MODELS
    )

    missing = (
        required_columns - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            f"Missing OOF columns: {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(
            "OOF predictions are empty."
        )

    if predictions["customerID"].isna().any():
        raise ValueError(
            "Missing OOF customer IDs."
        )

    if (
        predictions["customerID"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Blank OOF customer IDs."
        )

    if predictions["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate OOF customer IDs."
        )

    if train_df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate training customer IDs."
        )

    expected_ids = set(
        train_df["customerID"]
    )

    observed_ids = set(
        predictions["customerID"]
    )

    if expected_ids != observed_ids:
        raise ValueError(
            "OOF training membership mismatch."
        )

    expected_labels = (
        train_df.set_index("customerID")["Churn"]
    )

    observed_labels = (
        predictions
        .set_index("customerID")["actual_churn"]
    )

    aligned_labels = expected_labels.reindex(
        observed_labels.index
    )

    if not np.array_equal(
        aligned_labels.to_numpy(),
        observed_labels.to_numpy(),
    ):
        raise ValueError(
            "OOF customer labels mismatch."
        )

    validate_labels(
        predictions["actual_churn"],
        len(predictions),
    )

    # Recreate the original deterministic folds.

    expected_folds = np.full(
        len(train_df),
        -1,
        dtype=int,
    )

    folds = create_calibration_folds(
        train_df,
        n_splits=n_splits,
        random_state=random_state,
    )

    for fold_number, (_, score_indices) in enumerate(
        folds,
        start=1,
    ):
        expected_folds[
            score_indices
        ] = fold_number

    expected_fold_series = pd.Series(
        expected_folds,
        index=train_df["customerID"],
    )

    observed_fold_series = (
        predictions.set_index("customerID")["fold"]
    )

    aligned_folds = expected_fold_series.reindex(
        observed_fold_series.index
    )

    if not np.array_equal(
        aligned_folds.to_numpy(),
        observed_fold_series.to_numpy(),
    ):
        raise ValueError(
            "OOF fold assignments mismatch."
        )

    for model_name in SUPPORTED_MODELS:
        validate_probabilities(
            predictions[
                f"{model_name}_probability"
            ]
        )


def load_oof_predictions():
    """Load and verify the full OOF experiment."""

    predictions = pd.read_csv(
        OOF_PATH
    )

    with OOF_METADATA_PATH.open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if metadata["smoke_test"] is not False:
        raise ValueError(
            "Full OOF predictions are required."
        )

    if metadata["n_splits"] != 5:
        raise ValueError(
            "Expected five calibration folds."
        )

    if metadata["random_state"] != RANDOM_STATE:
        raise ValueError(
            "OOF random state mismatch."
        )

    if metadata["validation_evaluated"] is not False:
        raise ValueError(
            "Unexpected validation evaluation."
        )

    if metadata["test_evaluated"] is not False:
        raise ValueError(
            "Unexpected test evaluation."
        )

    if metadata["dataset_hash"] != calculate_file_hash(
        DATA_PATH
    ):
        raise ValueError(
            "Original dataset fingerprint mismatch."
        )

    if metadata["parameters_hash"] != calculate_file_hash(
        PARAMETERS_PATH
    ):
        raise ValueError(
            "Optimized parameters fingerprint mismatch."
        )

    if metadata["oof_predictions_hash"] != calculate_file_hash(
        OOF_PATH
    ):
        raise ValueError(
            "OOF predictions fingerprint mismatch."
        )

    if metadata["oof_rows"] != len(predictions):
        raise ValueError(
            "OOF row count mismatch."
        )

    if set(metadata["models"]) != set(
        SUPPORTED_MODELS
    ):
        raise ValueError(
            "OOF model membership mismatch."
        )

    train_df, val_df, test_df = (
        load_training_partitions(
            random_state=RANDOM_STATE
        )
    )

    partition_sizes = {
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
    }

    for key, expected_size in partition_sizes.items():
        if metadata[key] != expected_size:
            raise ValueError(
                f"Partition size mismatch: {key}"
            )

    validate_oof_predictions(
        predictions,
        train_df,
        n_splits=metadata["n_splits"],
        random_state=metadata["random_state"],
    )

    return predictions, metadata


def fit_sigmoid_calibrator(
    probabilities,
    y_true,
):
    """Fit logistic sigmoid calibration on log-odds."""

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    calibrator = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        probability_to_logit(probabilities),
        labels,
    )

    return calibrator


def fit_isotonic_calibrator(
    probabilities,
    y_true,
):
    """Fit a monotonic isotonic probability mapping."""

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    calibrator = IsotonicRegression(
        y_min=0,
        y_max=1,
        increasing=True,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probabilities,
        labels,
    )

    return calibrator


def apply_calibration(
    probabilities,
    method,
    calibrator=None,
):
    """Apply the selected calibration method."""

    probabilities = validate_probabilities(
        probabilities
    )

    if method == "uncalibrated":
        return probabilities.copy()

    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method: {method}"
        )

    if calibrator is None:
        raise ValueError(
            "A fitted calibrator is required."
        )

    if method == "sigmoid":
        calibrated = calibrator.predict_proba(
            probability_to_logit(probabilities)
        )[:, 1]

    else:
        calibrated = calibrator.predict(
            probabilities
        )

    return validate_probabilities(
        calibrated
    )


def calculate_calibration_bins(
    y_true,
    probabilities,
    n_bins=N_BINS,
):
    """Calculate equal-width reliability diagram bins."""

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, bool)
        or n_bins < 2
    ):
        raise ValueError(
            "At least two calibration bins are required."
        )

    edges = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    bin_indices = np.searchsorted(
        edges[1:-1],
        probabilities,
        side="right",
    )

    records = []

    for bin_index in range(n_bins):

        mask = (
            bin_indices == bin_index
        )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            labels[mask].mean()
        )

        calibration_gap = abs(
            mean_probability - observed_rate
        )

        records.append({
            "bin": bin_index + 1,
            "bin_lower": float(edges[bin_index]),
            "bin_upper": float(edges[bin_index + 1]),
            "count": count,
            "weight": count / len(labels),
            "mean_probability": mean_probability,
            "observed_churn_rate": observed_rate,
            "calibration_gap": calibration_gap,
        })

    return pd.DataFrame(records)


def calculate_calibration_metrics(
    y_true,
    probabilities,
    n_bins=N_BINS,
):
    """Calculate calibration and discrimination metrics."""

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        len(probabilities),
    )

    bins = calculate_calibration_bins(
        labels,
        probabilities,
        n_bins=n_bins,
    )

    ece = float(
        (
            bins["weight"]
            * bins["calibration_gap"]
        ).sum()
    )

    mce = float(
        bins["calibration_gap"].max()
    )

    return {
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),

        "ece": ece,

        "mce": mce,

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),

        "average_precision": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),

        "mean_predicted_probability": float(
            probabilities.mean()
        ),

        "observed_churn_rate": float(
            labels.mean()
        ),
    }


def fit_all_calibrators(oof_predictions):
    """Fit two calibrators for each optimized model."""

    labels = oof_predictions[
        "actual_churn"
    ].to_numpy()

    calibrators = {}

    for model_name in SUPPORTED_MODELS:

        probabilities = oof_predictions[
            f"{model_name}_probability"
        ].to_numpy()

        calibrators[model_name] = {
            "sigmoid": fit_sigmoid_calibrator(
                probabilities,
                labels,
            ),

            "isotonic": fit_isotonic_calibrator(
                probabilities,
                labels,
            ),
        }

        print(
            f"Calibrators fitted: {model_name}"
        )

    return calibrators


def evaluate_calibrators(
    validation_predictions,
    calibrators,
    n_bins=N_BINS,
):
    """Evaluate all nine combinations on validation data."""

    labels = validation_predictions[
        "actual_churn"
    ].to_numpy()

    results = []

    reliability_records = []

    calibrated_predictions = (
        validation_predictions[
            ["customerID", "actual_churn"]
        ].copy()
    )

    for model_name in SUPPORTED_MODELS:

        original_probabilities = (
            validation_predictions[
                f"{model_name}_probability"
            ].to_numpy()
        )

        for method in CALIBRATION_METHODS:

            calibrator = calibrators[
                model_name
            ].get(method)

            probabilities = apply_calibration(
                original_probabilities,
                method,
                calibrator,
            )

            metrics = calculate_calibration_metrics(
                labels,
                probabilities,
                n_bins=n_bins,
            )

            results.append({
                "model": model_name,
                "calibration_method": method,
                **metrics,
            })

            bins = calculate_calibration_bins(
                labels,
                probabilities,
                n_bins=n_bins,
            )

            bins.insert(
                0,
                "calibration_method",
                method,
            )

            bins.insert(
                0,
                "model",
                model_name,
            )

            reliability_records.append(bins)

            calibrated_predictions[
                f"{model_name}_{method}_probability"
            ] = probabilities

    metrics_df = pd.DataFrame(results)

    reliability_df = pd.concat(
        reliability_records,
        ignore_index=True,
    )

    return (
        metrics_df,
        reliability_df,
        calibrated_predictions,
    )


def compare_calibration_methods(metrics_df):
    """Compare each calibrator with its uncalibrated model."""

    records = []

    for model_name in SUPPORTED_MODELS:

        model_results = metrics_df[
            metrics_df["model"] == model_name
        ].set_index(
            "calibration_method"
        )

        baseline = model_results.loc[
            "uncalibrated"
        ]

        for method in (
            "sigmoid",
            "isotonic",
        ):

            calibrated = model_results.loc[
                method
            ]

            records.append({
                "model": model_name,
                "calibration_method": method,

                "brier_score_reduction": (
                    baseline["brier_score"]
                    - calibrated["brier_score"]
                ),

                "log_loss_reduction": (
                    baseline["log_loss"]
                    - calibrated["log_loss"]
                ),

                "ece_reduction": (
                    baseline["ece"]
                    - calibrated["ece"]
                ),

                "mce_reduction": (
                    baseline["mce"]
                    - calibrated["mce"]
                ),

                "roc_auc_change": (
                    calibrated["roc_auc"]
                    - baseline["roc_auc"]
                ),

                "average_precision_change": (
                    calibrated["average_precision"]
                    - baseline["average_precision"]
                ),
            })

    return pd.DataFrame(records)


def select_calibration_methods(metrics_df):
    """Select a candidate calibration method per model."""

    selections = []

    for model_name in SUPPORTED_MODELS:

        candidates = metrics_df[
            metrics_df["model"] == model_name
        ].copy()

        if len(candidates) != len(
            CALIBRATION_METHODS
        ):
            raise ValueError(
                f"Incomplete calibration results: {model_name}"
            )

        candidates = candidates.sort_values(
            [
                "brier_score",
                "log_loss",
                "ece",
                "calibration_method",
            ],
            ascending=True,
        )

        selected = candidates.iloc[0]

        selections.append(
            selected.to_dict()
        )

    return pd.DataFrame(selections)


def save_calibration_results(
    calibrators,
    metrics_df,
    reliability_df,
    comparisons_df,
    selected_methods,
    calibrated_predictions,
    metadata,
    output_dir=OUTPUT_DIR,
    reports_dir=REPORTS_DIR,
):
    """Save fitted calibrators and evaluation reports."""

    output_dir = Path(output_dir)

    reports_dir = Path(reports_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, methods in calibrators.items():

        for method, calibrator in methods.items():

            joblib.dump(
                calibrator,
                output_dir
                / f"{model_name}_{method}.pkl",
            )

    predictions_path = (
        output_dir
        / "calibrated_validation_predictions.csv"
    )

    calibrated_predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics_df.to_csv(
        reports_dir / "calibration_metrics.csv",
        index=False,
    )

    reliability_df.to_csv(
        reports_dir / "reliability_bins.csv",
        index=False,
    )

    comparisons_df.to_csv(
        reports_dir / "calibration_comparison.csv",
        index=False,
    )

    selected_methods.to_csv(
        reports_dir / "selected_calibration_methods.csv",
        index=False,
    )

    metadata = {
        **metadata,

        "calibrated_predictions_hash": (
            calculate_file_hash(
                predictions_path
            )
        ),
    }

    with (
        reports_dir / "calibration_metadata.json"
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
        f"\nCalibrators saved: {output_dir}"
    )

    print(
        f"Reports saved: {reports_dir}"
    )


def main():
    """Run the probability calibration experiment."""

    print(
        "\nLoading verified OOF predictions..."
    )

    oof_predictions, oof_metadata = (
        load_oof_predictions()
    )

    validation_predictions, validation_metadata = (
        load_validation_predictions()
    )

    if (
        oof_metadata["dataset_hash"]
        != validation_metadata["dataset_hash"]
    ):
        raise ValueError(
            "Dataset fingerprints do not match."
        )

    if (
        oof_metadata["random_state"]
        != validation_metadata["random_state"]
    ):
        raise ValueError(
            "Partition random states do not match."
        )

    training_ids = set(
        oof_predictions["customerID"]
    )

    validation_ids = set(
        validation_predictions["customerID"]
    )

    if training_ids & validation_ids:
        raise ValueError(
            "Calibration training and validation overlap."
        )

    print(
        "\nCalibration experiment configuration"
    )

    print(
        f"Calibration training rows: "
        f"{len(oof_predictions):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_predictions):,}"
    )

    print(
        f"Models: {len(SUPPORTED_MODELS)}"
    )

    print(
        "Calibration methods: "
        "uncalibrated, sigmoid, isotonic"
    )

    print(
        f"Calibration bins: {N_BINS}"
    )

    print(
        "\nFitting probability calibrators..."
    )

    calibrators = fit_all_calibrators(
        oof_predictions
    )

    print(
        "\nEvaluating calibration on validation data..."
    )

    (
        metrics_df,
        reliability_df,
        calibrated_predictions,
    ) = evaluate_calibrators(
        validation_predictions,
        calibrators,
        n_bins=N_BINS,
    )

    comparisons_df = compare_calibration_methods(
        metrics_df
    )

    selected_methods = select_calibration_methods(
        metrics_df
    )

    metadata = {
        "dataset_hash": oof_metadata[
            "dataset_hash"
        ],

        "parameters_hash": oof_metadata[
            "parameters_hash"
        ],

        "oof_predictions_hash": oof_metadata[
            "oof_predictions_hash"
        ],

        "validation_predictions_hash": (
            calculate_file_hash(
                VALIDATION_PATH
            )
        ),

        "random_state": RANDOM_STATE,

        "calibration_training_rows": len(
            oof_predictions
        ),

        "validation_rows": len(
            validation_predictions
        ),

        "oof_folds": oof_metadata[
            "n_splits"
        ],

        "models": list(SUPPORTED_MODELS),

        "calibration_methods": list(
            CALIBRATION_METHODS
        ),

        "n_bins": N_BINS,

        "binning_strategy": "equal_width",

        "sigmoid_input": "probability_log_odds",

        "selection_metric": "brier_score",

        "selection_tiebreakers": [
            "log_loss",
            "ece",
            "calibration_method",
        ],

        "calibrator_fit_partition": "training_oof",

        "evaluation_partition": "validation",

        "nested_hyperparameter_selection": False,

        "test_evaluated": False,

        "production_updated": False,
    }

    save_calibration_results(
        calibrators,
        metrics_df,
        reliability_df,
        comparisons_df,
        selected_methods,
        calibrated_predictions,
        metadata,
    )

    display_columns = [
        "model",
        "calibration_method",
        "brier_score",
        "log_loss",
        "ece",
        "mce",
        "roc_auc",
        "average_precision",
    ]

    print(
        "\nCALIBRATION VALIDATION METRICS"
    )

    print(
        metrics_df[display_columns]
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nCALIBRATION IMPROVEMENTS"
    )

    print(
        comparisons_df
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nSELECTED CALIBRATION METHOD PER MODEL"
    )

    print(
        selected_methods[display_columns]
        .round(6)
        .to_string(index=False)
    )

    print(
        f"\nModel-calibration combinations: "
        f"{len(metrics_df)}"
    )

    print(
        f"Fitted calibrators: "
        f"{sum(len(v) for v in calibrators.values())}"
    )

    print(
        "\nProduction models remain unchanged."
    )

    print(
        "The test partition has not been evaluated."
    )


if __name__ == "__main__":
    main()