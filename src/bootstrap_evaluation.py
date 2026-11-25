"""
src/bootstrap_evaluation.py

Stratified bootstrap confidence intervals and paired
statistical comparisons of optimized churn models.
"""

import hashlib
import json

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


RANDOM_STATE = 42

N_BOOTSTRAPS = 2000

CONFIDENCE_LEVEL = 0.95

CLASSIFICATION_THRESHOLD = 0.50

DATA_PATH = Path("data/raw/telco_churn.csv")

PREDICTIONS_PATH = Path(
    "models/candidates/optimized/validation_predictions.csv"
)

METADATA_PATH = Path(
    "models/candidates/optimized/training_metadata.json"
)

REPORTS_DIR = Path("reports/bootstrap")

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "tensorflow_ann": "TensorFlow ANN",
}

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)


def calculate_file_hash(file_path):
    """Calculate the SHA-256 fingerprint of a file."""

    hasher = hashlib.sha256()

    with Path(file_path).open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):

            hasher.update(chunk)

    return hasher.hexdigest()


def validate_predictions(
    predictions,
    threshold=CLASSIFICATION_THRESHOLD,
):
    """Validate saved customer predictions."""

    if not 0 <= threshold <= 1:

        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    required_columns = [
        "customerID",
        "actual_churn",
    ]

    for model_name in MODEL_NAMES:

        required_columns.extend([
            f"{model_name}_probability",
            f"{model_name}_prediction",
        ])

    missing_columns = (
        set(required_columns)
        - set(predictions.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:

        raise ValueError(
            "Prediction dataset is empty."
        )

    if predictions["customerID"].isna().any():

        raise ValueError(
            "Missing customer IDs."
        )

    customer_ids = (
        predictions["customerID"]
        .astype(str)
        .str.strip()
    )

    if customer_ids.eq("").any():

        raise ValueError(
            "Blank customer IDs."
        )

    if customer_ids.duplicated().any():

        raise ValueError(
            "Duplicate customer IDs."
        )

    labels = predictions[
        "actual_churn"
    ].to_numpy()

    if not np.array_equal(
        np.unique(labels),
        np.array([0, 1]),
    ):

        raise ValueError(
            "Actual churn labels must contain "
            "both binary classes."
        )

    for model_name in MODEL_NAMES:

        probabilities = predictions[
            f"{model_name}_probability"
        ].to_numpy(dtype=float)

        saved_predictions = predictions[
            f"{model_name}_prediction"
        ].to_numpy()

        if not np.isfinite(
            probabilities
        ).all():

            raise ValueError(
                f"Nonfinite probabilities: "
                f"{model_name}"
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):

            raise ValueError(
                f"Invalid probability range: "
                f"{model_name}"
            )

        expected_predictions = (
            probabilities >= threshold
        ).astype(int)

        if not np.array_equal(
            saved_predictions,
            expected_predictions,
        ):

            raise ValueError(
                f"Prediction threshold mismatch: "
                f"{model_name}"
            )


def load_validation_predictions(
    predictions_path=PREDICTIONS_PATH,
    metadata_path=METADATA_PATH,
    data_path=DATA_PATH,
):
    """Load and verify optimized validation predictions."""

    predictions = pd.read_csv(
        predictions_path
    )

    with Path(metadata_path).open(
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    current_hash = calculate_file_hash(
        data_path
    )

    if metadata["dataset_hash"] != current_hash:

        raise ValueError(
            "Original dataset fingerprint mismatch."
        )

    if len(predictions) != metadata[
        "validation_rows"
    ]:

        raise ValueError(
            "Validation row count mismatch."
        )

    if metadata["test_evaluated"] is not False:

        raise ValueError(
            "Unexpected test evaluation metadata."
        )

    threshold = metadata[
        "classification_threshold"
    ]

    validate_predictions(
        predictions,
        threshold=threshold,
    )

    # Verify customer IDs and labels against the
    # original validation partition.

    from src.optimization import (
        load_training_partitions,
    )

    _, validation_df, _ = (
        load_training_partitions(
            data_path=data_path,
            random_state=metadata["random_state"],
        )
    )

    expected_labels = validation_df.set_index(
        "customerID"
    )["Churn"]

    observed_labels = predictions.set_index(
        "customerID"
    )["actual_churn"]

    if set(expected_labels.index) != set(
        observed_labels.index
    ):

        raise ValueError(
            "Validation customer membership mismatch."
        )

    expected_labels = expected_labels.reindex(
        observed_labels.index
    )

    if not np.array_equal(
        expected_labels.to_numpy(),
        observed_labels.to_numpy(),
    ):

        raise ValueError(
            "Validation customer labels mismatch."
        )

    return predictions, metadata


def calculate_metrics(
    y_true,
    probabilities,
    threshold=CLASSIFICATION_THRESHOLD,
):
    """Calculate six binary classification metrics."""

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        y_true.ndim != 1
        or probabilities.ndim != 1
    ):

        raise ValueError(
            "Inputs must be one-dimensional."
        )

    if len(y_true) != len(probabilities):

        raise ValueError(
            "Input lengths do not match."
        )

    if not 0 <= threshold <= 1:

        raise ValueError(
            "Invalid classification threshold."
        )

    if not np.array_equal(
        np.unique(y_true),
        np.array([0, 1]),
    ):

        raise ValueError(
            "Both binary classes are required."
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Nonfinite probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):

        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predicted_labels,
        ),

        "precision": precision_score(
            y_true,
            predicted_labels,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predicted_labels,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            predicted_labels,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
    }


def generate_bootstrap_indices(
    y_true,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
):
    """Generate reproducible stratified bootstrap samples."""

    y_true = np.asarray(y_true)

    if y_true.ndim != 1:

        raise ValueError(
            "Labels must be one-dimensional."
        )

    if (
        not isinstance(
            n_bootstraps,
            (int, np.integer),
        )
        or isinstance(n_bootstraps, bool)
        or n_bootstraps < 2
    ):

        raise ValueError(
            "At least two bootstrap samples are required."
        )

    if not np.array_equal(
        np.unique(y_true),
        np.array([0, 1]),
    ):

        raise ValueError(
            "Both binary classes are required."
        )

    rng = np.random.default_rng(
        random_state
    )

    negative_indices = np.flatnonzero(
        y_true == 0
    )

    positive_indices = np.flatnonzero(
        y_true == 1
    )

    for _ in range(n_bootstraps):

        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_indices = np.concatenate([
            sampled_negative,
            sampled_positive,
        ])

        rng.shuffle(sampled_indices)

        yield sampled_indices


def calculate_confidence_interval(
    values,
    confidence_level=CONFIDENCE_LEVEL,
):
    """Calculate a percentile bootstrap confidence interval."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if (
        values.ndim != 1
        or len(values) < 2
    ):

        raise ValueError(
            "At least two one-dimensional "
            "observations are required."
        )

    if not np.isfinite(values).all():

        raise ValueError(
            "Confidence interval values must be finite."
        )

    if not 0 < confidence_level < 1:

        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    alpha = 1 - confidence_level

    lower, upper = np.quantile(
        values,
        [
            alpha / 2,
            1 - alpha / 2,
        ],
    )

    return float(lower), float(upper)


def run_bootstrap(
    predictions,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
    threshold=CLASSIFICATION_THRESHOLD,
    verbose=True,
):
    """Evaluate every model on identical bootstrap samples."""

    validate_predictions(
        predictions,
        threshold=threshold,
    )

    labels = predictions[
        "actual_churn"
    ].to_numpy(dtype=int)

    probabilities = {
        model_name: predictions[
            f"{model_name}_probability"
        ].to_numpy(dtype=float)

        for model_name in MODEL_NAMES
    }

    point_estimates = {
        model_name: calculate_metrics(
            labels,
            model_probabilities,
            threshold=threshold,
        )

        for model_name, model_probabilities
        in probabilities.items()
    }

    records = []

    bootstrap_indices = generate_bootstrap_indices(
        labels,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )

    for iteration, indices in enumerate(
        bootstrap_indices,
        start=1,
    ):

        sampled_labels = labels[indices]

        for model_name, model_probabilities in (
            probabilities.items()
        ):

            sampled_probabilities = (
                model_probabilities[indices]
            )

            metrics = calculate_metrics(
                sampled_labels,
                sampled_probabilities,
                threshold=threshold,
            )

            records.append({
                "iteration": iteration,
                "model": model_name,
                **metrics,
            })

        if verbose and (
            iteration % 250 == 0
            or iteration == n_bootstraps
        ):

            print(
                f"Bootstrap progress: "
                f"{iteration:,}/{n_bootstraps:,}"
            )

    bootstrap_results = pd.DataFrame(
        records
    )

    return point_estimates, bootstrap_results


def summarize_bootstrap(
    point_estimates,
    bootstrap_results,
    confidence_level=CONFIDENCE_LEVEL,
):
    """Summarize uncertainty for every model and metric."""

    records = []

    for model_name in MODEL_NAMES:

        model_results = bootstrap_results.loc[
            bootstrap_results["model"]
            == model_name
        ]

        for metric in METRIC_NAMES:

            values = model_results[
                metric
            ].to_numpy()

            lower, upper = (
                calculate_confidence_interval(
                    values,
                    confidence_level,
                )
            )

            records.append({
                "model": model_name,
                "metric": metric,
                "point_estimate": (
                    point_estimates[
                        model_name
                    ][metric]
                ),
                "bootstrap_mean": float(
                    np.mean(values)
                ),
                "standard_error": float(
                    np.std(values, ddof=1)
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_width": upper - lower,
                "confidence_level": confidence_level,
                "n_bootstraps": len(values),
            })

    return pd.DataFrame(records)


def compare_models(
    point_estimates,
    bootstrap_results,
    confidence_level=CONFIDENCE_LEVEL,
):
    """Calculate paired bootstrap intervals for differences."""

    records = []

    for model_a, model_b in combinations(
        MODEL_NAMES,
        2,
    ):

        results_a = bootstrap_results.loc[
            bootstrap_results["model"]
            == model_a
        ].set_index("iteration")

        results_b = bootstrap_results.loc[
            bootstrap_results["model"]
            == model_b
        ].set_index("iteration")

        if (
            not results_a.index.is_unique
            or not results_b.index.is_unique
        ):

            raise ValueError(
                "Duplicate bootstrap iteration IDs."
            )

        if set(results_a.index) != set(
            results_b.index
        ):

            raise ValueError(
                "Bootstrap iterations are not aligned."
            )

        results_b = results_b.reindex(
            results_a.index
        )

        for metric in METRIC_NAMES:

            differences = (
                results_a[metric].to_numpy()
                - results_b[metric].to_numpy()
            )

            lower, upper = (
                calculate_confidence_interval(
                    differences,
                    confidence_level,
                )
            )

            point_difference = (
                point_estimates[model_a][metric]
                - point_estimates[model_b][metric]
            )

            excludes_zero = (
                lower > 0
                or upper < 0
            )

            if lower > 0:

                interpretation = (
                    f"{model_a} higher"
                )

            elif upper < 0:

                interpretation = (
                    f"{model_b} higher"
                )

            else:

                interpretation = (
                    "Inconclusive difference"
                )

            records.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "point_difference": point_difference,
                "bootstrap_mean_difference": float(
                    np.mean(differences)
                ),
                "standard_error": float(
                    np.std(differences, ddof=1)
                ),
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "ci_excludes_zero": excludes_zero,
                "interpretation": interpretation,
                "n_bootstraps": len(differences),
            })

    return pd.DataFrame(records)


def save_bootstrap_reports(
    bootstrap_results,
    confidence_intervals,
    comparisons,
    metadata,
    output_dir=REPORTS_DIR,
):
    """Export bootstrap results and experiment metadata."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap_results.to_csv(
        output_dir / "bootstrap_distributions.csv",
        index=False,
    )

    confidence_intervals.to_csv(
        output_dir / "bootstrap_confidence_intervals.csv",
        index=False,
    )

    comparisons.to_csv(
        output_dir / "paired_model_comparisons.csv",
        index=False,
    )

    with (
        output_dir / "bootstrap_metadata.json"
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
        f"\nReports saved to: {output_dir}"
    )


def main():
    """Execute the complete bootstrap evaluation."""

    predictions, training_metadata = (
        load_validation_predictions()
    )

    threshold = training_metadata[
        "classification_threshold"
    ]

    print("\nBootstrap evaluation configuration")

    print(
        f"Validation customers: "
        f"{len(predictions):,}"
    )

    print(
        f"Models: {len(MODEL_NAMES)}"
    )

    print(
        f"Bootstrap samples: {N_BOOTSTRAPS:,}"
    )

    print(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL:.0%}"
    )

    print(
        f"Classification threshold: {threshold}"
    )

    print(
        "\nOriginal validation class distribution:"
    )

    print(
        predictions["actual_churn"]
        .value_counts()
        .sort_index()
    )

    point_estimates, bootstrap_results = (
        run_bootstrap(
            predictions,
            n_bootstraps=N_BOOTSTRAPS,
            random_state=RANDOM_STATE,
            threshold=threshold,
        )
    )

    confidence_intervals = summarize_bootstrap(
        point_estimates,
        bootstrap_results,
        confidence_level=CONFIDENCE_LEVEL,
    )

    comparisons = compare_models(
        point_estimates,
        bootstrap_results,
        confidence_level=CONFIDENCE_LEVEL,
    )

    metadata = {
        "dataset_hash": (
            training_metadata["dataset_hash"]
        ),
        "predictions_hash": calculate_file_hash(
            PREDICTIONS_PATH
        ),
        "validation_rows": len(predictions),
        "models": list(MODEL_NAMES),
        "metrics": list(METRIC_NAMES),
        "n_bootstraps": N_BOOTSTRAPS,
        "confidence_level": CONFIDENCE_LEVEL,
        "random_state": RANDOM_STATE,
        "classification_threshold": threshold,
        "sampling_method": (
            "paired_stratified_bootstrap"
        ),
        "interval_method": "percentile",
        "multiple_comparison_adjustment": None,
        "evaluation_partition": "validation",
        "test_evaluated": False,
    }

    save_bootstrap_reports(
        bootstrap_results,
        confidence_intervals,
        comparisons,
        metadata,
    )

    print("\nROC-AUC confidence intervals:")

    print(
        confidence_intervals.loc[
            confidence_intervals["metric"]
            == "roc_auc",
            [
                "model",
                "point_estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
            ],
        ].to_string(index=False)
    )

    print("\nPaired ROC-AUC comparisons:")

    print(
        comparisons.loc[
            comparisons["metric"]
            == "roc_auc",
            [
                "model_a",
                "model_b",
                "point_difference",
                "ci_lower",
                "ci_upper",
                "ci_excludes_zero",
            ],
        ].to_string(index=False)
    )

    print(
        f"\nBootstrap evaluation rows: "
        f"{len(bootstrap_results):,}"
    )

    print(
        f"Confidence intervals: "
        f"{len(confidence_intervals)}"
    )

    print(
        f"Paired comparisons: "
        f"{len(comparisons)}"
    )

    print(
        "\nProduction models remain unchanged."
    )


if __name__ == "__main__":
    main()