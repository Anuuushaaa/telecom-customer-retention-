"""
src/calibration_oof.py

Generate out-of-fold predictions for probability
calibration using the original training partition.
"""

import argparse
import gc
import json

from pathlib import Path

import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import StratifiedKFold

from src.optimization import (
    RANDOM_STATE,
    SUPPORTED_MODELS,
    load_training_partitions,
    build_logistic_model,
    build_xgboost_model,
)

from src.optimization_evaluation import (
    load_best_parameters,
    retrain_ann,
)

from src.preprocessing import (
    fit_transform_features,
)

from src.train_ann import (
    predict_ann_proba,
)

from src.bootstrap_evaluation import (
    calculate_file_hash,
)


N_SPLITS = 5

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


def create_calibration_folds(
    train_df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """Create stratified out-of-fold partitions."""

    if (
        not isinstance(n_splits, int)
        or isinstance(n_splits, bool)
        or n_splits < 2
    ):

        raise ValueError(
            "At least two folds are required."
        )

    labels = train_df["Churn"].to_numpy()

    if not np.array_equal(
        np.unique(labels),
        np.array([0, 1]),
    ):

        raise ValueError(
            "Both binary classes are required."
        )

    class_counts = np.bincount(
        labels.astype(int)
    )

    if class_counts.min() < n_splits:

        raise ValueError(
            "Insufficient observations per class."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return list(
        splitter.split(
            train_df,
            labels,
        )
    )


def validate_oof_probabilities(
    probabilities,
    expected_rows,
):
    """Validate generated probability predictions."""

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:

        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if len(probabilities) != expected_rows:

        raise ValueError(
            "Probability row count mismatch."
        )

    if not np.isfinite(probabilities).all():

        raise ValueError(
            "Nonfinite OOF probabilities."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):

        raise ValueError(
            "OOF probabilities outside [0, 1]."
        )

    return probabilities


def generate_oof_predictions(
    train_df,
    best_parameters,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    verbose=True,
):
    """Generate aligned predictions for all training rows."""

    if train_df.empty:

        raise ValueError(
            "Training partition is empty."
        )

    if train_df["customerID"].duplicated().any():

        raise ValueError(
            "Duplicate training customer IDs."
        )

    for model_name in SUPPORTED_MODELS:

        if model_name not in best_parameters:

            raise ValueError(
                f"Missing parameters: {model_name}"
            )

    train_df = train_df.reset_index(
        drop=True
    )

    n_rows = len(train_df)

    probabilities = {
        model_name: np.full(
            n_rows,
            np.nan,
            dtype=float,
        )

        for model_name in SUPPORTED_MODELS
    }

    fold_assignments = np.full(
        n_rows,
        -1,
        dtype=int,
    )

    assignment_counts = np.zeros(
        n_rows,
        dtype=int,
    )

    folds = create_calibration_folds(
        train_df,
        n_splits=n_splits,
        random_state=random_state,
    )

    fold_records = []

    for fold_number, (
        fit_indices,
        score_indices,
    ) in enumerate(
        folds,
        start=1,
    ):

        if verbose:

            print(
                f"\n{'=' * 60}"
            )

            print(
                f"Calibration fold "
                f"{fold_number}/{n_splits}"
            )

        if np.intersect1d(
            fit_indices,
            score_indices,
        ).size:

            raise ValueError(
                "Fold training and scoring overlap."
            )

        fold_train = train_df.iloc[
            fit_indices
        ].copy()

        fold_score = train_df.iloc[
            score_indices
        ].copy()

        if verbose:

            print(
                f"Training rows: {len(fold_train):,}"
            )

            print(
                f"Scoring rows: {len(fold_score):,}"
            )

        # Fit preprocessing exclusively on
        # the current fold's training rows.

        processed = fit_transform_features(
            train_df=fold_train,
            test_df=fold_score,
        )

        np.testing.assert_array_equal(
            np.asarray(
                processed.customer_ids_test
            ),
            fold_score[
                "customerID"
            ].to_numpy(),
        )

        fold_seed = (
            random_state + fold_number - 1
        )

        # Logistic Regression

        lr_model = build_logistic_model(
            best_parameters[
                "logistic_regression"
            ]["best_params"],
            random_state=fold_seed,
        )

        lr_model.fit(
            processed.X_train,
            processed.y_train,
        )

        lr_probabilities = (
            lr_model.predict_proba(
                processed.X_test
            )[:, 1]
        )

        probabilities[
            "logistic_regression"
        ][score_indices] = (
            validate_oof_probabilities(
                lr_probabilities,
                len(score_indices),
            )
        )

        if verbose:

            print(
                "Logistic Regression completed."
            )

        # XGBoost

        xgb_model = build_xgboost_model(
            best_parameters[
                "xgboost"
            ]["best_params"],
            processed.y_train,
            random_state=fold_seed,
        )

        xgb_model.fit(
            processed.X_train,
            processed.y_train,
        )

        xgb_probabilities = (
            xgb_model.predict_proba(
                processed.X_test
            )[:, 1]
        )

        probabilities[
            "xgboost"
        ][score_indices] = (
            validate_oof_probabilities(
                xgb_probabilities,
                len(score_indices),
            )
        )

        if verbose:

            print("XGBoost completed.")

        # TensorFlow ANN
        #
        # Use the median best epoch from
        # the completed Optuna study.
        #
        # The scoring fold is not used
        # for early stopping.

        ann_model = None

        history = None

        try:

            ann_model, history, selected_epochs = (
                retrain_ann(
                    processed.X_train,
                    processed.y_train,
                    best_parameters[
                        "tensorflow_ann"
                    ]["best_params"],
                    best_parameters[
                        "tensorflow_ann"
                    ]["best_epochs"],
                    verbose=0,
                )
            )

            ann_probabilities = (
                predict_ann_proba(
                    ann_model,
                    processed.X_test,
                )
            )

            probabilities[
                "tensorflow_ann"
            ][score_indices] = (
                validate_oof_probabilities(
                    ann_probabilities,
                    len(score_indices),
                )
            )

            if verbose:

                print(
                    "TensorFlow ANN completed "
                    f"({selected_epochs} epochs)."
                )

        finally:

            del ann_model

            del history

            keras.backend.clear_session()

            gc.collect()

        fold_assignments[
            score_indices
        ] = fold_number

        assignment_counts[
            score_indices
        ] += 1

        fold_records.append({
            "fold": fold_number,
            "training_rows": len(fit_indices),
            "scoring_rows": len(score_indices),
            "training_churn_rate": float(
                processed.y_train.mean()
            ),
            "scoring_churn_rate": float(
                processed.y_test.mean()
            ),
            "feature_count": (
                processed.X_train.shape[1]
            ),
            "ann_epochs": selected_epochs,
        })

        del lr_model

        del xgb_model

        gc.collect()

    if not np.all(
        assignment_counts == 1
    ):

        raise ValueError(
            "Each customer must receive "
            "exactly one OOF prediction."
        )

    if (fold_assignments < 1).any():

        raise ValueError(
            "Missing fold assignments."
        )

    results = pd.DataFrame({
        "customerID": train_df[
            "customerID"
        ].to_numpy(),

        "actual_churn": train_df[
            "Churn"
        ].to_numpy(dtype=int),

        "fold": fold_assignments,
    })

    for model_name in SUPPORTED_MODELS:

        results[
            f"{model_name}_probability"
        ] = validate_oof_probabilities(
            probabilities[model_name],
            n_rows,
        )

    fold_summary = pd.DataFrame(
        fold_records
    )

    return results, fold_summary


def save_oof_results(
    predictions,
    fold_summary,
    metadata,
    output_dir=OUTPUT_DIR,
    reports_dir=REPORTS_DIR,
):
    """Save predictions and calibration experiment metadata."""

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

    predictions_path = (
        output_dir / "oof_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metadata = {
        **metadata,
        "oof_predictions_hash": (
            calculate_file_hash(
                predictions_path
            )
        ),
    }

    fold_summary.to_csv(
        reports_dir / "oof_fold_summary.csv",
        index=False,
    )

    with (
        reports_dir / "oof_metadata.json"
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
        f"\nPredictions saved: {predictions_path}"
    )

    print(
        f"Reports saved: {reports_dir}"
    )


def main():
    """Execute out-of-fold prediction generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate calibration OOF predictions."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a reduced two-fold experiment.",
    )

    args = parser.parse_args()

    best_parameters = load_best_parameters()

    train_df, val_df, test_df = (
        load_training_partitions()
    )

    train_ids = set(
        train_df["customerID"]
    )

    validation_ids = set(
        val_df["customerID"]
    )

    test_ids = set(
        test_df["customerID"]
    )

    if (
        train_ids & validation_ids
        or train_ids & test_ids
        or validation_ids & test_ids
    ):

        raise ValueError(
            "Dataset partitions overlap."
        )

    n_splits = (
        2 if args.smoke_test else N_SPLITS
    )

    if args.smoke_test:

        output_dir = OUTPUT_DIR / "smoke"

        reports_dir = (
            OUTPUT_DIR / "smoke" / "reports"
        )

    else:

        output_dir = OUTPUT_DIR

        reports_dir = REPORTS_DIR

    print("\nCalibration OOF configuration")

    print(
        f"Training customers: {len(train_df):,}"
    )

    print(
        f"Validation customers: {len(val_df):,}"
    )

    print(
        f"Test customers: {len(test_df):,}"
    )

    print(f"Folds: {n_splits}")

    print(
        f"Smoke test: {args.smoke_test}"
    )

    print(
        "Only training customers will be "
        "used for OOF model fitting and scoring."
    )

    predictions, fold_summary = (
        generate_oof_predictions(
            train_df,
            best_parameters,
            n_splits=n_splits,
            random_state=RANDOM_STATE,
        )
    )

    metadata = {
        "dataset_hash": calculate_file_hash(
            DATA_PATH
        ),
        "parameters_hash": calculate_file_hash(
            PARAMETERS_PATH
        ),
        "random_state": RANDOM_STATE,
        "n_splits": n_splits,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "oof_rows": len(predictions),
        "models": list(SUPPORTED_MODELS),
        "sampling_method": (
            "stratified_kfold"
        ),
        "hyperparameter_source": (
            "completed_optuna_studies"
        ),
        "smoke_test": args.smoke_test,
        "validation_evaluated": False,
        "test_evaluated": False,
    }

    save_oof_results(
        predictions,
        fold_summary,
        metadata,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )

    print("\nFold summary:")

    print(
        fold_summary.to_string(index=False)
    )

    print(
        f"\nTotal OOF predictions: "
        f"{len(predictions):,}"
    )

    print("\nMissing probability values:")

    print(
        predictions.isna().sum().to_string()
    )

    print(
        "\nProduction models remain unchanged."
    )


if __name__ == "__main__":
    main()