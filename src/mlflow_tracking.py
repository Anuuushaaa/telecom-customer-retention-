
"""
MLflow experiment tracking for Telecom Customer Retention 2.0.

Logs existing baseline and optimized models without retraining.

Features:
- Six model experiment runs
- Hyperparameter tracking
- Validation metric verification
- Bootstrap confidence intervals
- Probability calibration metrics
- Threshold optimization
- Business profitability metrics
- Financial sensitivity analysis
- Artifact fingerprinting
- Duplicate-run prevention
- Reproducibility metadata

Production models are not modified.
"""

import hashlib
import json
import math
import subprocess
import sys

from pathlib import Path
from numbers import Real

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import xgboost
import optuna

from src.mlflow_config import (
    PROJECT_ROOT,
    DATA_PATH,
    CANDIDATE_MODEL_DIR,
    OPTIMIZED_MODEL_DIR,
    CALIBRATION_MODEL_DIR,
    REPORTS_DIR,
    SUPPORTED_MODELS,
    MODEL_DISPLAY_NAMES,
    MODEL_VARIANTS,
    RANDOM_STATE,
    initialize_experiment,
    get_mlflow_client,
)

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
    transform_features,
)

from src.train_ann import predict_ann_proba

from src.optimization_evaluation import (
    calculate_metrics,
)

from src.probability_calibration import (
    apply_calibration,
)

from src.threshold_tuning import (
    load_calibrated_predictions,
)


# ============================================================
# REPORT PATHS
# ============================================================

SOURCE_REPORTS = PROJECT_ROOT / "reports"

BASELINE_METRICS_PATH = (
    SOURCE_REPORTS
    / "validation_model_comparison.csv"
)

OPTIMIZED_METRICS_PATH = (
    SOURCE_REPORTS
    / "optimization"
    / "optimized_validation_metrics.csv"
)

BEST_PARAMETERS_PATH = (
    SOURCE_REPORTS
    / "optimization"
    / "best_hyperparameters.json"
)

TRAINING_METADATA_PATH = (
    OPTIMIZED_MODEL_DIR
    / "training_metadata.json"
)

BOOTSTRAP_PATH = (
    SOURCE_REPORTS
    / "bootstrap"
    / "bootstrap_confidence_intervals.csv"
)

CALIBRATION_PATH = (
    SOURCE_REPORTS
    / "calibration"
    / "calibration_metrics.csv"
)

SELECTED_CALIBRATION_PATH = (
    SOURCE_REPORTS
    / "calibration"
    / "selected_calibration_methods.csv"
)

THRESHOLDS_PATH = (
    SOURCE_REPORTS
    / "thresholds"
    / "preferred_calibration_thresholds.csv"
)

BUSINESS_PATH = (
    SOURCE_REPORTS
    / "business"
    / "best_business_strategies.csv"
)

BUSINESS_ASSUMPTIONS_PATH = (
    SOURCE_REPORTS
    / "business"
    / "business_assumptions.json"
)

SENSITIVITY_PATH = (
    SOURCE_REPORTS
    / "business"
    / "sensitivity"
    / "model_stability_summary.csv"
)


# ============================================================
# TRACKING CONSTANTS
# ============================================================

TRACKING_SCHEMA_VERSION = "1"

CLASSIFICATION_THRESHOLD = 0.50

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

CALIBRATION_METRICS = (
    "brier_score",
    "log_loss",
    "ece",
    "mce",
    "roc_auc",
    "average_precision",
    "mean_predicted_probability",
    "observed_churn_rate",
)

MODEL_FILENAMES = {
    "logistic_regression": "logistic_regression.pkl",
    "xgboost": "xgboost.pkl",
    "tensorflow_ann": "tensorflow_ann.keras",
}


# ============================================================
# FILE UTILITIES
# ============================================================

def require_file(path):
    """Verify that a required file exists."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required file is empty: {path}"
        )

    return path


def calculate_file_hash(path):
    """Calculate the SHA-256 hash of a file."""

    path = require_file(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    """Load a JSON document."""

    with require_file(path).open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_csv(path, required_columns=None):
    """Load and validate a CSV report."""

    data = pd.read_csv(require_file(path))

    if data.empty:
        raise ValueError(
            f"Empty report: {path}"
        )

    if required_columns is not None:

        missing = (
            set(required_columns)
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns in {path}: "
                f"{sorted(missing)}"
            )

    return data


def select_single_row(data, **conditions):
    """Select exactly one matching report row."""

    selected = data.copy()

    for column, value in conditions.items():

        if column not in selected.columns:
            raise ValueError(
                f"Missing filter column: {column}"
            )

        selected = selected[
            selected[column] == value
        ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one matching row "
            f"for {conditions}; found {len(selected)}."
        )

    return selected.iloc[0]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def get_git_commit():
    """Return the current Git commit."""

    try:

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.SubprocessError,
        OSError,
    ):

        return "unknown"


def get_environment_versions():
    """Return relevant library versions."""

    return {
        "python": sys.version.split()[0],
        "mlflow": mlflow.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "xgboost": xgboost.__version__,
        "optuna": optuna.__version__,
        "joblib": joblib.__version__,
    }


def get_git_dirty_status():
    """Determine whether tracked or untracked changes exist."""

    try:

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return bool(result.stdout.strip())

    except (
        subprocess.SubprocessError,
        OSError,
    ):

        return None


# ============================================================
# MODEL ARTIFACTS
# ============================================================

def validate_model_variant(model_name, variant):
    """Validate supported model and variant names."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported variant: {variant}"
        )


def get_model_directory(variant):
    """Return the candidate artifact directory."""

    if variant == "baseline":
        return CANDIDATE_MODEL_DIR

    if variant == "optimized":
        return OPTIMIZED_MODEL_DIR

    raise ValueError(
        f"Unsupported variant: {variant}"
    )


def get_model_artifacts(model_name, variant):
    """Return required model and preprocessing files."""

    validate_model_variant(
        model_name,
        variant,
    )

    directory = get_model_directory(variant)

    artifacts = {
        "model": directory
        / MODEL_FILENAMES[model_name],

        "scaler": directory
        / "scaler.pkl",

        "feature_columns": directory
        / "feature_columns.pkl",
    }

    if variant == "optimized":

        artifacts["sigmoid_calibrator"] = (
            CALIBRATION_MODEL_DIR
            / f"{model_name}_sigmoid.pkl"
        )

        artifacts["isotonic_calibrator"] = (
            CALIBRATION_MODEL_DIR
            / f"{model_name}_isotonic.pkl"
        )

    for path in artifacts.values():
        require_file(path)

    return artifacts


def load_candidate_model(model_name, model_path):
    """Load a trusted local candidate model."""

    require_file(model_path)

    if model_name == "tensorflow_ann":

        return tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    return joblib.load(model_path)


def predict_probabilities(
    model_name,
    model,
    features,
):
    """Generate positive-class probabilities."""

    if model_name == "tensorflow_ann":

        probabilities = predict_ann_proba(
            model,
            features,
        )

    else:

        if list(model.classes_) != [0, 1]:
            raise ValueError(
                "Unexpected model class ordering."
            )

        probabilities = model.predict_proba(
            features
        )[:, 1]

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.shape != (len(features),):
        raise ValueError(
            "Unexpected prediction shape."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite model probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError(
            "Model probabilities outside [0, 1]."
        )

    return probabilities


# ============================================================
# LOAD EXISTING REPORTS
# ============================================================

def load_tracking_reports():
    """Load all reports required for experiment tracking."""

    reports = {
        "baseline": load_csv(
            BASELINE_METRICS_PATH,
            ["model", *METRIC_COLUMNS],
        ),

        "optimized": load_csv(
            OPTIMIZED_METRICS_PATH,
            ["model", *METRIC_COLUMNS],
        ),

        "bootstrap": load_csv(
            BOOTSTRAP_PATH,
            [
                "model",
                "metric",
                "ci_lower",
                "ci_upper",
                "standard_error",
            ],
        ),

        "calibration": load_csv(
            CALIBRATION_PATH,
            [
                "model",
                "calibration_method",
                *CALIBRATION_METRICS,
            ],
        ),

        "selected_calibration": load_csv(
            SELECTED_CALIBRATION_PATH,
            [
                "model",
                "calibration_method",
                *CALIBRATION_METRICS,
            ],
        ),

        "thresholds": load_csv(
            THRESHOLDS_PATH,
            [
                "model",
                "strategy",
                "threshold",
                "feasible",
            ],
        ),

        "business": load_csv(
            BUSINESS_PATH,
            [
                "model",
                "selection_type",
                "strategy",
                "expected_net_profit",
                "simulated_net_profit",
            ],
        ),

        "sensitivity": load_csv(
            SENSITIVITY_PATH,
            [
                "model",
                "scenario_family",
                "strategy",
                "win_rate",
            ],
        ),

        "best_parameters": load_json(
            BEST_PARAMETERS_PATH
        ),

        "training_metadata": load_json(
            TRAINING_METADATA_PATH
        ),

        "business_assumptions": load_json(
            BUSINESS_ASSUMPTIONS_PATH
        ),
    }

    dataset_hash = calculate_file_hash(
        DATA_PATH
    )

    if (
        reports["training_metadata"]["dataset_hash"]
        != dataset_hash
    ):
        raise ValueError(
            "Training dataset hash mismatch."
        )

    if (
        reports["training_metadata"]["test_evaluated"]
        is not False
    ):
        raise ValueError(
            "Unexpected test evaluation metadata."
        )

    for model_name in SUPPORTED_MODELS:

        study = reports["best_parameters"][
            model_name
        ]

        if (
            study["configuration"]["dataset_hash"]
            != dataset_hash
        ):
            raise ValueError(
                f"Optuna dataset hash mismatch: "
                f"{model_name}"
            )

        if study["configuration"]["smoke_test"]:
            raise ValueError(
                "Smoke-test studies cannot be logged "
                "as optimized production candidates."
            )

    return reports


# ============================================================
# VALIDATION DATA
# ============================================================

def load_validation_partition():
    """Recreate the original validation partition."""

    raw = pd.read_csv(
        require_file(DATA_PATH)
    )

    clean = prepare_raw_data(
        raw,
        require_target=True,
    )

    train_df, val_df, test_df = (
        split_train_validation_test(
            clean,
            random_state=RANDOM_STATE,
        )
    )

    return val_df, {
        "dataset_rows": len(clean),
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
    }


def verify_candidate_metrics(
    model_name,
    variant,
    model,
    artifacts,
    val_df,
    reports,
    calibrated_predictions=None,
):
    """
    Verify that saved models reproduce reported metrics.

    Uses validation data only.
    """

    scaler = joblib.load(
        artifacts["scaler"]
    )

    feature_columns = joblib.load(
        artifacts["feature_columns"]
    )

    features = transform_features(
        val_df,
        scaler,
        feature_columns,
    )

    probabilities = predict_probabilities(
        model_name,
        model,
        features,
    )

    actual_metrics = calculate_metrics(
        val_df["Churn"],
        probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    recorded = select_single_row(
        reports[variant],
        model=MODEL_DISPLAY_NAMES[model_name],
    )

    for metric in METRIC_COLUMNS:

        if not np.isclose(
            actual_metrics[metric],
            float(recorded[metric]),
            atol=1e-6,
            rtol=1e-6,
        ):

            raise ValueError(
                f"Metric verification failed: "
                f"{model_name}/{variant}/{metric}. "
                f"Expected {recorded[metric]}, "
                f"received {actual_metrics[metric]}."
            )

    if variant == "optimized":

        if calibrated_predictions is None:
            raise ValueError(
                "Missing calibrated validation predictions."
            )

        if not np.array_equal(
            val_df["customerID"].astype(str).to_numpy(),
            calibrated_predictions[
                "customerID"
            ].astype(str).to_numpy(),
        ):
            raise ValueError(
                "Validation customer alignment mismatch."
            )

        if not np.array_equal(
            val_df["Churn"].to_numpy(),
            calibrated_predictions[
                "actual_churn"
            ].to_numpy(),
        ):
            raise ValueError(
                "Validation label alignment mismatch."
            )

        for method in (
            "uncalibrated",
            "sigmoid",
            "isotonic",
        ):

            calibrator = None

            if method != "uncalibrated":

                calibrator = joblib.load(
                    artifacts[f"{method}_calibrator"]
                )

            calibrated = apply_calibration(
                probabilities,
                method,
                calibrator,
            )

            expected = calibrated_predictions[
                f"{model_name}_{method}_probability"
            ].to_numpy(dtype=float)

            if not np.allclose(
                calibrated,
                expected,
                atol=1e-5,
                rtol=1e-5,
            ):

                raise ValueError(
                    f"Calibration verification failed: "
                    f"{model_name}/{method}."
                )

    return actual_metrics


# ============================================================
# PARAMETER TRACKING
# ============================================================

def serialize_parameter(value):
    """Convert a parameter to an MLflow-compatible string."""

    if isinstance(
        value,
        (list, tuple, dict),
    ):

        return json.dumps(
            value,
            sort_keys=True,
            default=str,
        )

    return str(value)


def get_model_parameters(
    model_name,
    variant,
    model,
    reports,
):
    """Collect model-specific training parameters."""

    validate_model_variant(
        model_name,
        variant,
    )

    parameters = {
        "model_name": model_name,
        "variant": variant,
        "random_state": RANDOM_STATE,
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "class_weighting": "balanced",
    }

    if model_name != "tensorflow_ann":

        for key, value in model.get_params().items():

            parameters[
                f"model_{key}"
            ] = serialize_parameter(value)

    elif variant == "baseline":

        parameters.update({
            "hidden_units": [64, 32],
            "dropout_rates": [0.30, 0.20],
            "activation": "relu",
            "learning_rate": 0.001,
            "batch_size": 32,
            "maximum_epochs": 100,
            "early_stopping_patience": 10,
            "l2_strength": 0.0,
        })

        history = load_csv(
            SOURCE_REPORTS
            / "ann_training_history.csv",
            ["epoch", "val_loss"],
        )

        parameters["epochs_completed"] = (
            len(history)
        )

        parameters["best_epoch"] = int(
            history.loc[
                history["val_loss"].idxmin(),
                "epoch",
            ]
        )

    if variant == "optimized":

        study = reports["best_parameters"][
            model_name
        ]

        parameters.update({
            "optuna_study_name": study["study_name"],
            "optuna_best_trial": study["best_trial"],
            "cv_folds": study["configuration"]["n_splits"],
            "optimization_objective": "roc_auc",
        })

        for key, value in study[
            "best_params"
        ].items():

            parameters[
                f"optimized_{key}"
            ] = value

        for key, value in study[
            "trial_counts"
        ].items():

            parameters[
                f"optuna_trials_{key}"
            ] = value

        if model_name == "tensorflow_ann":

            parameters["selected_epochs"] = (
                reports["training_metadata"][
                    "ann_selected_epochs"
                ]
            )

            parameters["cv_best_epochs"] = (
                study["best_epochs"]
            )

        selected = select_single_row(
            reports["selected_calibration"],
            model=model_name,
        )

        parameters["selected_calibration_method"] = (
            selected["calibration_method"]
        )

        assumptions = reports[
            "business_assumptions"
        ]

        for key, value in assumptions.items():

            parameters[
                f"business_{key}"
            ] = value

    return {
        key: serialize_parameter(value)
        for key, value in parameters.items()
    }


# ============================================================
# METRIC TRACKING
# ============================================================

def add_numeric_metrics(
    destination,
    values,
    prefix,
    excluded=(),
):
    """Add finite numeric metrics using a namespace."""

    for key, value in values.items():

        if key in excluded:
            continue

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            continue

        if isinstance(value, Real):

            numeric_value = float(value)

            if math.isfinite(numeric_value):

                destination[
                    f"{prefix}_{key}"
                ] = numeric_value


def collect_validation_metrics(
    model_name,
    variant,
    reports,
):
    """Collect the original validation metrics."""

    row = select_single_row(
        reports[variant],
        model=MODEL_DISPLAY_NAMES[model_name],
    )

    return {
        f"validation_{metric}": float(row[metric])
        for metric in METRIC_COLUMNS
    }


def collect_bootstrap_metrics(
    model_name,
    reports,
):
    """Collect bootstrap uncertainty estimates."""

    metrics = {}

    for metric in METRIC_COLUMNS:

        row = select_single_row(
            reports["bootstrap"],
            model=model_name,
            metric=metric,
        )

        for field in (
            "point_estimate",
            "bootstrap_mean",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "ci_width",
            "confidence_level",
            "n_bootstraps",
        ):

            metrics[
                f"bootstrap_{metric}_{field}"
            ] = float(row[field])

    return metrics


def collect_calibration_metrics(
    model_name,
    reports,
):
    """Collect metrics for all calibration methods."""

    metrics = {}

    for method in (
        "uncalibrated",
        "sigmoid",
        "isotonic",
    ):

        row = select_single_row(
            reports["calibration"],
            model=model_name,
            calibration_method=method,
        )

        for metric in CALIBRATION_METRICS:

            metrics[
                f"calibration_{method}_{metric}"
            ] = float(row[metric])

    selected = select_single_row(
        reports["selected_calibration"],
        model=model_name,
    )

    for metric in CALIBRATION_METRICS:

        metrics[
            f"selected_calibration_{metric}"
        ] = float(selected[metric])

    return metrics


def collect_threshold_metrics(
    model_name,
    reports,
):
    """Collect preferred statistical threshold strategies."""

    metrics = {}

    rows = reports["thresholds"][
        reports["thresholds"]["model"]
        == model_name
    ]

    if rows.empty:
        raise ValueError(
            f"No threshold results for {model_name}."
        )

    if rows["strategy"].duplicated().any():
        raise ValueError(
            "Duplicate threshold strategies."
        )

    for _, row in rows.iterrows():

        strategy = row["strategy"]

        feasible = (
            str(row["feasible"]).lower()
            == "true"
        )

        metrics[
            f"threshold_{strategy}_feasible"
        ] = float(feasible)

        if not feasible:
            continue

        add_numeric_metrics(
            metrics,
            row.to_dict(),
            prefix=f"threshold_{strategy}",
            excluded=("minimum_recall",),
        )

    return metrics


def collect_business_metrics(
    model_name,
    reports,
):
    """Collect hypothetical campaign profitability metrics."""

    metrics = {}

    for selection_type in (
        "unconstrained",
        "budget_constrained",
    ):

        row = select_single_row(
            reports["business"],
            model=model_name,
            selection_type=selection_type,
        )

        add_numeric_metrics(
            metrics,
            row.to_dict(),
            prefix=f"business_{selection_type}",
        )

    return metrics


def collect_sensitivity_metrics(
    model_name,
    reports,
):
    """Collect financial scenario stability metrics."""

    metrics = {}

    rows = reports["sensitivity"][
        reports["sensitivity"]["model"]
        == model_name
    ]

    if rows.empty:
        raise ValueError(
            f"No sensitivity results for {model_name}."
        )

    if rows.duplicated(
        ["scenario_family", "strategy"]
    ).any():

        raise ValueError(
            "Duplicate sensitivity summary rows."
        )

    for _, row in rows.iterrows():

        prefix = (
            f"sensitivity_"
            f"{row['scenario_family']}_"
            f"{row['strategy']}"
        )

        add_numeric_metrics(
            metrics,
            row.to_dict(),
            prefix=prefix,
        )

    return metrics


def collect_all_metrics(
    model_name,
    variant,
    reports,
):
    """Combine all applicable experiment metrics."""

    metrics = collect_validation_metrics(
        model_name,
        variant,
        reports,
    )

    if variant == "optimized":

        study = reports["best_parameters"][
            model_name
        ]

        metrics.update({
            "optuna_best_cv_roc_auc": float(
                study["best_cv_roc_auc"]
            ),

            "optuna_cv_standard_deviation": float(
                study["cv_std"]
            ),
        })

        for collector in (
            collect_bootstrap_metrics,
            collect_calibration_metrics,
            collect_threshold_metrics,
            collect_business_metrics,
            collect_sensitivity_metrics,
        ):

            metrics.update(
                collector(
                    model_name,
                    reports,
                )
            )

    for key, value in metrics.items():

        if not math.isfinite(float(value)):
            raise ValueError(
                f"Non-finite metric: {key}"
            )

    return metrics


# ============================================================
# REPORT ARTIFACTS
# ============================================================

def get_report_artifacts(model_name, variant):
    """Collect evaluation reports relevant to a run."""

    validate_model_variant(
        model_name,
        variant,
    )

    if variant == "baseline":

        paths = [
            BASELINE_METRICS_PATH,
        ]

        if model_name == "tensorflow_ann":

            paths.extend([
                SOURCE_REPORTS
                / "ann_training_history.csv",

                SOURCE_REPORTS
                / "ann_learning_curves.png",

                SOURCE_REPORTS
                / "ann_validation_metrics.csv",
            ])

    else:

        paths = [
            OPTIMIZED_METRICS_PATH,
            BEST_PARAMETERS_PATH,
            TRAINING_METADATA_PATH,
        ]

        for directory in (
            "optimization",
            "bootstrap",
            "calibration",
            "thresholds",
            "business",
        ):

            report_directory = (
                SOURCE_REPORTS / directory
            )

            if not report_directory.is_dir():
                raise FileNotFoundError(
                    f"Missing report directory: "
                    f"{report_directory}"
                )

            for path in sorted(
                report_directory.rglob("*")
            ):

                if not path.is_file():
                    continue

                if "smoke" in path.relative_to(
                    report_directory
                ).parts:
                    continue

                if path.suffix.lower() in (
                    ".csv",
                    ".json",
                    ".png",
                ):
                    paths.append(path)

    return sorted(
        set(require_file(path) for path in paths)
    )


# ============================================================
# ARTIFACT FINGERPRINTING
# ============================================================

def build_artifact_manifest(
    model_name,
    variant,
    artifacts,
    report_paths,
):
    """Create a reproducible manifest of input files."""

    manifest = {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "model_name": model_name,
        "variant": variant,
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "files": {},
    }

    paths = list(
        artifacts.values()
    ) + list(report_paths)

    for path in sorted(set(paths)):

        relative_path = (
            path.relative_to(PROJECT_ROOT)
            .as_posix()
        )

        manifest["files"][relative_path] = {
            "sha256": calculate_file_hash(path),
            "size_bytes": path.stat().st_size,
        }

    return manifest


def calculate_run_fingerprint(
    manifest,
    parameters,
    metrics,
):
    """Create a deterministic fingerprint for a run."""

    payload = {
        "manifest": manifest,
        "parameters": parameters,
        "metrics": metrics,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def find_existing_run(
    client,
    experiment_id,
    fingerprint,
):
    """Find an already completed matching run."""

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.run_fingerprint = "
            f"'{fingerprint}'"
        ),
        max_results=100,
    )

    for run in runs:

        if run.info.status == "FINISHED":
            return run.info.run_id

    return None


# ============================================================
# EXPERIMENT TAGS
# ============================================================

def build_run_tags(
    model_name,
    variant,
    fingerprint,
):
    """Create descriptive and reproducibility tags."""

    return {
        "model_name": model_name,
        "model_variant": variant,
        "evaluation_partition": "validation",
        "tracking_schema_version": (
            TRACKING_SCHEMA_VERSION
        ),
        "run_fingerprint": fingerprint,
        "git_commit": get_git_commit(),
        "git_dirty": str(get_git_dirty_status()),
        "test_evaluated_in_tracking": "false",
        "production_updated": "false",
        "training_repeated": "false",
        "logging_mode": "existing_artifact_import",
        "financial_results": (
            "hypothetical_simulation"
            if variant == "optimized"
            else "not_applicable"
        ),
        "model_lifecycle": "candidate",
    }


# ============================================================
# ARTIFACT LOGGING
# ============================================================

def log_model_artifacts(artifacts):
    """Log original trusted model files."""

    for artifact_name, path in artifacts.items():

        mlflow.log_artifact(
            str(path),
            artifact_path=(
                f"candidate_artifacts/{artifact_name}"
            ),
        )


def log_report_artifacts(report_paths):
    """Log reports while preserving directory structure."""

    for path in report_paths:

        relative = path.relative_to(
            PROJECT_ROOT
        )

        destination = relative.parent.as_posix()

        mlflow.log_artifact(
            str(path),
            artifact_path=destination,
        )


def log_ann_training_history(
    model_name,
    variant,
):
    """Log baseline ANN epoch-level training metrics."""

    if (
        model_name != "tensorflow_ann"
        or variant != "baseline"
    ):
        return

    history = load_csv(
        SOURCE_REPORTS
        / "ann_training_history.csv",
        ["epoch"],
    )

    for _, row in history.iterrows():

        epoch_metrics = {}

        add_numeric_metrics(
            epoch_metrics,
            row.to_dict(),
            prefix="training",
            excluded=("epoch",),
        )

        if epoch_metrics:

            mlflow.log_metrics(
                epoch_metrics,
                step=int(row["epoch"]),
            )


# ============================================================
# SINGLE EXPERIMENT RUN
# ============================================================

def log_candidate_run(
    model_name,
    variant,
    reports,
    val_df,
    partition_metadata,
    experiment_id,
    calibrated_predictions,
    force=False,
):
    """Verify and log one candidate experiment."""

    validate_model_variant(
        model_name,
        variant,
    )

    artifacts = get_model_artifacts(
        model_name,
        variant,
    )

    report_paths = get_report_artifacts(
        model_name,
        variant,
    )

    model = load_candidate_model(
        model_name,
        artifacts["model"],
    )

    verify_candidate_metrics(
        model_name,
        variant,
        model,
        artifacts,
        val_df,
        reports,
        calibrated_predictions,
    )

    parameters = get_model_parameters(
        model_name,
        variant,
        model,
        reports,
    )

    parameters.update({
        key: str(value)
        for key, value in partition_metadata.items()
    })

    metrics = collect_all_metrics(
        model_name,
        variant,
        reports,
    )

    manifest = build_artifact_manifest(
        model_name,
        variant,
        artifacts,
        report_paths,
    )

    fingerprint = calculate_run_fingerprint(
        manifest,
        parameters,
        metrics,
    )

    client = get_mlflow_client()

    if not force:

        existing_run = find_existing_run(
            client,
            experiment_id,
            fingerprint,
        )

        if existing_run is not None:

            print(
                f"Skipped existing run: "
                f"{model_name}/{variant}"
            )

            return {
                "model": model_name,
                "variant": variant,
                "run_id": existing_run,
                "fingerprint": fingerprint,
                "status": "existing",
                **metrics,
            }

    run_name = (
        f"{MODEL_DISPLAY_NAMES[model_name]}"
        f" - {variant.title()}"
    )

    tags = build_run_tags(
        model_name,
        variant,
        fingerprint,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    ) as run:

        mlflow.log_params(parameters)

        mlflow.log_metrics(metrics)

        log_model_artifacts(artifacts)

        log_report_artifacts(report_paths)

        log_ann_training_history(
            model_name,
            variant,
        )

        mlflow.log_dict(
            manifest,
            "metadata/artifact_manifest.json",
        )

        mlflow.log_dict(
            get_environment_versions(),
            "metadata/environment_versions.json",
        )

        mlflow.log_dict(
            partition_metadata,
            "metadata/dataset_partitions.json",
        )

        mlflow.log_artifact(
            str(
                require_file(
                    PROJECT_ROOT
                    / "requirements.txt"
                )
            ),
            artifact_path="metadata",
        )

        run_id = run.info.run_id

    print(
        f"Logged: {run_name}"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Metrics logged: {len(metrics)}"
    )

    return {
        "model": model_name,
        "variant": variant,
        "run_id": run_id,
        "fingerprint": fingerprint,
        "status": "logged",
        **metrics,
    }


# ============================================================
# LOG ALL EXPERIMENTS
# ============================================================

def log_all_experiments(force=False):
    """Log all six candidate model experiments."""

    experiment_id = initialize_experiment()

    reports = load_tracking_reports()

    val_df, partition_metadata = (
        load_validation_partition()
    )

    calibrated_predictions, _ = (
        load_calibrated_predictions()
    )

    if (
        partition_metadata["validation_rows"]
        != reports["training_metadata"][
            "validation_rows"
        ]
    ):
        raise ValueError(
            "Validation partition size mismatch."
        )

    results = []

    for model_name in SUPPORTED_MODELS:

        for variant in MODEL_VARIANTS:

            print("\n" + "=" * 70)

            print(
                f"MODEL: {model_name}"
            )

            print(
                f"VARIANT: {variant}"
            )

            print("=" * 70)

            result = log_candidate_run(
                model_name=model_name,
                variant=variant,
                reports=reports,
                val_df=val_df,
                partition_metadata=partition_metadata,
                experiment_id=experiment_id,
                calibrated_predictions=calibrated_predictions,
                force=force,
            )

            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# TRACKING REPORTS
# ============================================================

def save_tracking_summary(results):
    """Save experiment run identifiers and metadata."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_columns = [
        "model",
        "variant",
        "run_id",
        "fingerprint",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
        "validation_average_precision",
    ]

    summary = results[
        summary_columns
    ].copy()

    summary.to_csv(
        REPORTS_DIR
        / "experiment_run_summary.csv",
        index=False,
    )

    metadata = {
        "tracking_schema_version": (
            TRACKING_SCHEMA_VERSION
        ),
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "models": list(SUPPORTED_MODELS),
        "variants": list(MODEL_VARIANTS),
        "total_candidate_runs": len(results),
        "evaluation_partition": "validation",
        "training_repeated": False,
        "test_evaluated_in_tracking": False,
        "production_updated": False,
        "financial_results": (
            "hypothetical_simulation"
        ),
        "environment": get_environment_versions(),
    }

    with (
        REPORTS_DIR
        / "tracking_metadata.json"
    ).open(
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

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Log existing telecom churn candidates "
            "to MLflow."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Create new runs even if matching "
            "completed runs already exist."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)

    print("TELECOM CUSTOMER RETENTION")

    print("MLFLOW EXPERIMENT TRACKING")

    print("=" * 70)

    results = log_all_experiments(
        force=args.force
    )

    summary = save_tracking_summary(
        results
    )

    print("\n" + "=" * 70)

    print("EXPERIMENT TRACKING SUMMARY")

    print("=" * 70)

    print(
        summary[
            [
                "model",
                "variant",
                "validation_roc_auc",
                "validation_f1",
            ]
        ].round(6).to_string(index=False)
    )

    print(
        f"\nCandidate runs: {len(results)}"
    )

    print(
        "New runs: "
        f"{(results['status'] == 'logged').sum()}"
    )

    print(
        "Existing runs reused: "
        f"{(results['status'] == 'existing').sum()}"
    )

    print("\nReports generated:")

    print(" - experiment_run_summary.csv")

    print(" - tracking_metadata.json")

    print("\nMLflow experiment tracking completed.")

    print("No models were retrained.")

    print("Production models remain unchanged.")


if __name__ == "__main__":

    main()
