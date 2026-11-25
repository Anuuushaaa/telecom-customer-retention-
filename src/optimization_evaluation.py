"""
src/optimization_evaluation.py

Retrain optimized models and evaluate their performance
against baseline models on the validation partition.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.optimization import (
    RANDOM_STATE,
    SUPPORTED_MODELS,
    load_training_partitions,
    build_logistic_model,
    build_xgboost_model,
)

from src.optimization_storage import (
    calculate_dataset_hash,
    calculate_code_hash,
    TRIAL_BUDGETS,
)

from src.preprocessing import (
    fit_transform_features,
)

from src.train_ann import (
    build_ann,
    prepare_ann_inputs,
    calculate_class_weights,
    predict_ann_proba,
    save_training_history,
    save_ann_model,
    plot_learning_curves,
)


REPORTS_DIR = Path("reports/optimization")

PARAMETERS_PATH = (
    REPORTS_DIR / "best_hyperparameters.json"
)

BASELINE_PATH = Path(
    "reports/validation_model_comparison.csv"
)

OPTIMIZED_MODEL_DIR = Path(
    "models/candidates/optimized"
)

CLASSIFICATION_THRESHOLD = 0.50

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "tensorflow_ann": "TensorFlow ANN",
}

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]


def load_best_parameters(
    parameters_path=PARAMETERS_PATH,
):
    """Load and validate the full optimization results."""

    parameters_path = Path(parameters_path)

    if not parameters_path.is_file():

        raise FileNotFoundError(
            f"Optimization results not found: "
            f"{parameters_path}"
        )

    with parameters_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    dataset_hash = calculate_dataset_hash()

    code_hash = calculate_code_hash()

    for model_name in SUPPORTED_MODELS:

        if model_name not in results:

            raise ValueError(
                f"Missing optimization results: "
                f"{model_name}"
            )

        result = results[model_name]

        configuration = result["configuration"]

        if configuration["smoke_test"]:

            raise ValueError(
                "Smoke-test parameters cannot be "
                "used for optimized retraining."
            )

        if configuration["dataset_hash"] != dataset_hash:

            raise ValueError(
                "Dataset fingerprint mismatch."
            )

        if configuration["code_hash"] != code_hash:

            raise ValueError(
                "Optimization code fingerprint mismatch."
            )

        counts = result["trial_counts"]

        finished_trials = (
            counts["completed"]
            + counts["pruned"]
        )

        if finished_trials < TRIAL_BUDGETS[model_name]:

            raise ValueError(
                f"Optimization budget not exhausted: "
                f"{model_name}"
            )

        if counts["completed"] == 0:

            raise ValueError(
                f"No completed trials: {model_name}"
            )

    return results


def calculate_metrics(
    y_true,
    probabilities,
    threshold=CLASSIFICATION_THRESHOLD,
):
    """Calculate classification and ranking metrics."""

    if not 0 <= threshold <= 1:

        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(probabilities):

        raise ValueError(
            "Labels and probabilities must "
            "have the same length."
        )

    if not np.array_equal(
        np.unique(y_true),
        [0, 1],
    ):

        raise ValueError(
            "Both target classes are required."
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):

        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
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


def retrain_logistic_regression(
    X_train,
    y_train,
    parameters,
):
    """Retrain optimized Logistic Regression."""

    model = build_logistic_model(
        parameters,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def retrain_xgboost(
    X_train,
    y_train,
    parameters,
):
    """Retrain optimized XGBoost."""

    model = build_xgboost_model(
        parameters,
        y_train,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def retrain_ann(
    X_train,
    y_train,
    parameters,
    best_epochs,
    verbose=1,
):
    """
    Retrain the optimized ANN on all training samples.

    The median best epoch from cross-validation is used
    as a fixed training duration. The external validation
    partition is not used for early stopping.
    """

    if not best_epochs:

        raise ValueError(
            "ANN best epoch information is missing."
        )

    epochs_array = np.asarray(
        best_epochs,
        dtype=float,
    )

    if (
        not np.isfinite(epochs_array).all()
        or (epochs_array <= 0).any()
        or (epochs_array % 1 != 0).any()
    ):

        raise ValueError(
            "Best epochs must be positive integers."
        )

    selected_epochs = int(
        np.median(epochs_array)
    )

    model_parameters = parameters.copy()

    batch_size = model_parameters.pop(
        "batch_size"
    )

    X_train, y_train = prepare_ann_inputs(
        X_train,
        y_train,
    )

    class_weights = calculate_class_weights(
        y_train
    )

    keras.backend.clear_session()

    keras.utils.set_random_seed(
        RANDOM_STATE
    )

    model = build_ann(
        input_dim=X_train.shape[1],
        random_state=RANDOM_STATE,
        **model_parameters,
    )

    print(
        f"\nCV best epochs: {best_epochs}"
    )

    print(
        f"Selected training epochs: "
        f"{selected_epochs}"
    )

    if verbose:

        model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=selected_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        shuffle=True,
        verbose=verbose,
    )

    return model, history, selected_epochs


def compare_with_baseline(
    optimized_results,
    baseline_path=BASELINE_PATH,
):
    """Compare optimized and baseline validation metrics."""

    baseline_path = Path(
        baseline_path
    )

    if not baseline_path.is_file():

        raise FileNotFoundError(
            f"Baseline report not found: "
            f"{baseline_path}"
        )

    baseline_df = pd.read_csv(
        baseline_path
    )

    required_columns = {
        "model",
        *METRIC_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(baseline_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing baseline columns: "
            f"{sorted(missing_columns)}"
        )

    comparison = baseline_df.merge(
        optimized_results,
        on="model",
        how="inner",
        suffixes=(
            "_baseline",
            "_optimized",
        ),
        validate="one_to_one",
    )

    if len(comparison) != len(SUPPORTED_MODELS):

        raise ValueError(
            "Baseline and optimized models "
            "could not be matched correctly."
        )

    for metric in METRIC_COLUMNS:

        comparison[f"{metric}_change"] = (
            comparison[f"{metric}_optimized"]
            - comparison[f"{metric}_baseline"]
        )

    comparison = comparison.sort_values(
        "roc_auc_optimized",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


def save_validation_predictions(
    customer_ids,
    y_true,
    probabilities,
    output_path,
):
    """Save aligned validation predictions for later analysis."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame({
        "customerID": np.asarray(customer_ids),
        "actual_churn": np.asarray(y_true),
    })

    for model_name, values in probabilities.items():

        predictions_df[
            f"{model_name}_probability"
        ] = np.asarray(values)

        predictions_df[
            f"{model_name}_prediction"
        ] = (
            np.asarray(values)
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

    if predictions_df["customerID"].duplicated().any():

        raise ValueError(
            "Duplicate customer IDs in predictions."
        )

    predictions_df.to_csv(
        output_path,
        index=False,
    )

    return predictions_df


def main():
    """Retrain and compare all optimized candidates."""

    optimization_results = load_best_parameters()

    train_df, val_df, test_df = (
        load_training_partitions()
    )

    print("\nDataset partitions:")

    print(f"Training: {len(train_df):,}")

    print(f"Validation: {len(val_df):,}")

    print(f"Test: {len(test_df):,}")

    print(
        "\nThe test partition will not be evaluated."
    )

    # Use the validation partition as the scoring
    # partition without passing it to model fitting.
    processed = fit_transform_features(
        train_df=train_df,
        test_df=val_df,
    )

    OPTIMIZED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    validation_probabilities = {}

    # Logistic Regression

    print("\n" + "=" * 60)

    print("Retraining optimized Logistic Regression")

    lr_model = retrain_logistic_regression(
        processed.X_train,
        processed.y_train,
        optimization_results[
            "logistic_regression"
        ]["best_params"],
    )

    lr_probabilities = lr_model.predict_proba(
        processed.X_test
    )[:, 1]

    validation_probabilities[
        "logistic_regression"
    ] = lr_probabilities

    results.append({
        "model": MODEL_NAMES[
            "logistic_regression"
        ],
        **calculate_metrics(
            processed.y_test,
            lr_probabilities,
        ),
    })

    joblib.dump(
        lr_model,
        OPTIMIZED_MODEL_DIR
        / "logistic_regression.pkl",
    )

    # XGBoost

    print("\n" + "=" * 60)

    print("Retraining optimized XGBoost")

    xgb_model = retrain_xgboost(
        processed.X_train,
        processed.y_train,
        optimization_results[
            "xgboost"
        ]["best_params"],
    )

    xgb_probabilities = xgb_model.predict_proba(
        processed.X_test
    )[:, 1]

    validation_probabilities[
        "xgboost"
    ] = xgb_probabilities

    results.append({
        "model": MODEL_NAMES[
            "xgboost"
        ],
        **calculate_metrics(
            processed.y_test,
            xgb_probabilities,
        ),
    })

    joblib.dump(
        xgb_model,
        OPTIMIZED_MODEL_DIR
        / "xgboost.pkl",
    )

    # TensorFlow ANN

    print("\n" + "=" * 60)

    print("Retraining optimized TensorFlow ANN")

    ann_result = optimization_results[
        "tensorflow_ann"
    ]

    ann_model, history, selected_epochs = retrain_ann(
        processed.X_train,
        processed.y_train,
        ann_result["best_params"],
        ann_result["best_epochs"],
    )

    ann_probabilities = predict_ann_proba(
        ann_model,
        processed.X_test,
    )

    validation_probabilities[
        "tensorflow_ann"
    ] = ann_probabilities

    results.append({
        "model": MODEL_NAMES[
            "tensorflow_ann"
        ],
        **calculate_metrics(
            processed.y_test,
            ann_probabilities,
        ),
    })

    save_ann_model(
        ann_model,
        OPTIMIZED_MODEL_DIR
        / "tensorflow_ann.keras",
    )

    save_training_history(
        history,
        REPORTS_DIR
        / "optimized_ann_training_history.csv",
    )

    # Save preprocessing artifacts

    joblib.dump(
        processed.scaler,
        OPTIMIZED_MODEL_DIR
        / "scaler.pkl",
    )

    joblib.dump(
        processed.feature_columns,
        OPTIMIZED_MODEL_DIR
        / "feature_columns.pkl",
    )

    # Export optimized validation metrics

    optimized_df = pd.DataFrame(
        results
    )

    optimized_df.to_csv(
        REPORTS_DIR
        / "optimized_validation_metrics.csv",
        index=False,
    )

    comparison_df = compare_with_baseline(
        optimized_df
    )

    comparison_df.to_csv(
        REPORTS_DIR
        / "optimized_model_comparison.csv",
        index=False,
    )

    # Save aligned probabilities for bootstrap analysis.
    # Keep customer-level predictions in the ignored
    # local candidate directory.

    save_validation_predictions(
        customer_ids=processed.customer_ids_test,
        y_true=processed.y_test,
        probabilities=validation_probabilities,
        output_path=(
            OPTIMIZED_MODEL_DIR
            / "validation_predictions.csv"
        ),
    )

    metadata = {
        "dataset_hash": calculate_dataset_hash(),
        "random_state": RANDOM_STATE,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "feature_count": len(
            processed.feature_columns
        ),
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "ann_selected_epochs": selected_epochs,
        "ann_epoch_selection": (
            "median_cv_best_epoch"
        ),
        "test_evaluated": False,
    }

    with (
        OPTIMIZED_MODEL_DIR
        / "training_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)

    print("Optimized validation performance")

    print(
        optimized_df.round(6).to_string(
            index=False
        )
    )

    print("\nBaseline versus optimized ROC-AUC:")

    print(
        comparison_df[
            [
                "model",
                "roc_auc_baseline",
                "roc_auc_optimized",
                "roc_auc_change",
            ]
        ].round(6).to_string(
            index=False
        )
    )

    print(
        f"\nOptimized models saved to: "
        f"{OPTIMIZED_MODEL_DIR}"
    )

    print(
        f"Reports saved to: {REPORTS_DIR}"
    )

    print(
        "\nProduction models remain unchanged."
    )


if __name__ == "__main__":
    main()