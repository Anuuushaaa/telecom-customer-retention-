"""
src/optimization.py

Optuna hyperparameter optimization for telecom churn models.

Features:
- Logistic Regression optimization
- XGBoost optimization
- TensorFlow ANN optimization
- Stratified cross-validation
- Fold-specific preprocessing
- Independent ANN early stopping data
- TPE sampling and median pruning
- Reproducible optimization
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from xgboost import XGBClassifier

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
    fit_transform_features,
)

from src.train_ann import (
    train_ann,
    predict_ann_proba,
)


RANDOM_STATE = 42

DATA_PATH = Path("data/raw/telco_churn.csv")

N_SPLITS = 3

LR_TRIALS = 30
XGB_TRIALS = 40
ANN_TRIALS = 20

ANN_MAX_EPOCHS = 50
ANN_PATIENCE = 7

SUPPORTED_MODELS = (
    "logistic_regression",
    "xgboost",
    "tensorflow_ann",
)


def load_training_partitions(
    data_path=DATA_PATH,
    random_state=RANDOM_STATE,
):
    """Load and split the original telecom dataset."""

    raw_df = pd.read_csv(data_path)

    cleaned_df = prepare_raw_data(
        raw_df,
        require_target=True,
    )

    train_df, val_df, test_df = (
        split_train_validation_test(
            cleaned_df,
            random_state=random_state,
        )
    )

    return train_df, val_df, test_df


def create_cv_splits(
    train_df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """Generate stratified folds within the training partition."""

    if n_splits < 2:
        raise ValueError(
            "At least two cross-validation folds are required."
        )

    if train_df["Churn"].nunique() != 2:
        raise ValueError(
            "Training data must contain both target classes."
        )

    if train_df["Churn"].value_counts().min() < n_splits:
        raise ValueError(
            "Insufficient minority-class samples for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    return list(
        cv.split(
            train_df,
            train_df["Churn"],
        )
    )


def suggest_logistic_parameters(trial):
    """Define Logistic Regression search space."""

    solver = trial.suggest_categorical(
        "solver",
        ["lbfgs", "liblinear", "saga"],
    )

    if solver == "lbfgs":
        l1_ratio = 0.0

    elif solver == "liblinear":
        l1_ratio = trial.suggest_categorical(
            "liblinear_l1_ratio",
            [0.0, 1.0],
        )

    else:
        l1_ratio = trial.suggest_float(
            "saga_l1_ratio",
            0.0,
            1.0,
        )

    return {
        "C": trial.suggest_float(
            "C",
            0.0001,
            100.0,
            log=True,
        ),
        "solver": solver,
        "l1_ratio": l1_ratio,
    }


def suggest_xgboost_parameters(trial):
    """Define XGBoost search space."""

    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            600,
            step=50,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            6,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.005,
            0.2,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0.000001,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.000001,
            10.0,
            log=True,
        ),
    }


def suggest_ann_parameters(trial):
    """Define TensorFlow ANN search space."""

    n_layers = trial.suggest_int(
        "n_layers",
        1,
        3,
    )

    hidden_units = []

    dropout_rates = []

    for layer_index in range(n_layers):

        units = trial.suggest_categorical(
            f"units_{layer_index}",
            [16, 32, 64, 128],
        )

        dropout = trial.suggest_float(
            f"dropout_{layer_index}",
            0.1,
            0.5,
        )

        hidden_units.append(units)

        dropout_rates.append(dropout)

    return {
        "hidden_units": hidden_units,
        "dropout_rates": dropout_rates,
        "activation": trial.suggest_categorical(
            "activation",
            ["relu", "elu", "tanh"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.0001,
            0.01,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [16, 32, 64],
        ),
        "l2_strength": trial.suggest_float(
            "l2_strength",
            0.000001,
            0.01,
            log=True,
        ),
    }


def build_logistic_model(
    params,
    random_state=RANDOM_STATE,
):
    """Construct Logistic Regression from trial parameters."""

    return LogisticRegression(
        **params,
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )


def build_xgboost_model(
    params,
    y_train,
    random_state=RANDOM_STATE,
):
    """Construct XGBoost using fold-specific class imbalance."""

    labels = np.asarray(y_train)

    negatives = np.sum(labels == 0)

    positives = np.sum(labels == 1)

    if positives == 0 or negatives == 0:
        raise ValueError(
            "XGBoost training requires both target classes."
        )

    scale_pos_weight = float(
        negatives / positives
    )

    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=random_state,
        n_jobs=2,
        verbosity=0,
    )


def prepare_cv_folds(
    train_df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Fit preprocessing independently within each CV fold.

    Preprocessed folds can be reused across trials because
    preprocessing does not depend on model hyperparameters.
    """

    splits = create_cv_splits(
        train_df,
        n_splits=n_splits,
        random_state=random_state,
    )

    prepared_folds = []

    for train_indices, score_indices in splits:

        fold_train = train_df.iloc[
            train_indices
        ].copy()

        fold_score = train_df.iloc[
            score_indices
        ].copy()

        processed = fit_transform_features(
            train_df=fold_train,
            test_df=fold_score,
        )

        prepared_folds.append(processed)

    return prepared_folds


def prepare_ann_cv_folds(
    train_df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    """
    Prepare ANN folds with separate early stopping data.

    The outer CV scoring fold is never used for
    ANN fitting or early stopping.
    """

    splits = create_cv_splits(
        train_df,
        n_splits=n_splits,
        random_state=random_state,
    )

    prepared_folds = []

    for fold_index, (
        train_indices,
        score_indices,
    ) in enumerate(splits):

        outer_train = train_df.iloc[
            train_indices
        ].copy()

        outer_score = train_df.iloc[
            score_indices
        ].copy()

        inner_train, early_stop = train_test_split(
            outer_train,
            test_size=0.15,
            stratify=outer_train["Churn"],
            random_state=random_state + fold_index,
        )

        processed = fit_transform_features(
            train_df=inner_train,
            test_df=outer_score,
            val_df=early_stop,
        )

        prepared_folds.append(processed)

    return prepared_folds


def report_fold_score(
    trial,
    fold_scores,
    fold_index,
):
    """Report intermediate CV performance for pruning."""

    mean_score = float(
        np.mean(fold_scores)
    )

    trial.report(
        mean_score,
        step=fold_index,
    )

    trial.set_user_attr(
        "fold_scores",
        [float(score) for score in fold_scores],
    )

    if trial.should_prune():
        raise optuna.TrialPruned(
            f"Trial pruned after fold {fold_index + 1}."
        )


def logistic_objective(
    trial,
    prepared_folds,
    random_state=RANDOM_STATE,
):
    """Optimize Logistic Regression using mean CV ROC-AUC."""

    params = suggest_logistic_parameters(
        trial
    )

    trial.set_user_attr(
        "model_params",
        params,
    )

    fold_scores = []

    for fold_index, fold in enumerate(
        prepared_folds
    ):

        model = build_logistic_model(
            params,
            random_state=random_state + fold_index,
        )

        model.fit(
            fold.X_train,
            fold.y_train,
        )

        probabilities = model.predict_proba(
            fold.X_test
        )[:, 1]

        score = roc_auc_score(
            fold.y_test,
            probabilities,
        )

        fold_scores.append(float(score))

        report_fold_score(
            trial,
            fold_scores,
            fold_index,
        )

    trial.set_user_attr(
        "cv_std",
        float(np.std(fold_scores, ddof=1)),
    )

    return float(np.mean(fold_scores))


def xgboost_objective(
    trial,
    prepared_folds,
    random_state=RANDOM_STATE,
):
    """Optimize XGBoost using mean CV ROC-AUC."""

    params = suggest_xgboost_parameters(
        trial
    )

    trial.set_user_attr(
        "model_params",
        params,
    )

    fold_scores = []

    for fold_index, fold in enumerate(
        prepared_folds
    ):

        model = build_xgboost_model(
            params,
            fold.y_train,
            random_state=random_state + fold_index,
        )

        model.fit(
            fold.X_train,
            fold.y_train,
        )

        probabilities = model.predict_proba(
            fold.X_test
        )[:, 1]

        score = roc_auc_score(
            fold.y_test,
            probabilities,
        )

        fold_scores.append(float(score))

        report_fold_score(
            trial,
            fold_scores,
            fold_index,
        )

    trial.set_user_attr(
        "cv_std",
        float(np.std(fold_scores, ddof=1)),
    )

    return float(np.mean(fold_scores))


def ann_objective(
    trial,
    prepared_folds,
    random_state=RANDOM_STATE,
    max_epochs=ANN_MAX_EPOCHS,
):
    """Optimize ANN architecture using mean CV ROC-AUC."""

    params = suggest_ann_parameters(
        trial
    )

    trial.set_user_attr(
        "model_params",
        params,
    )

    fold_scores = []

    best_epochs = []

    for fold_index, fold in enumerate(
        prepared_folds
    ):

        model = None
        history = None

        try:

            model, history = train_ann(
                X_train=fold.X_train,
                y_train=fold.y_train,
                X_val=fold.X_val,
                y_val=fold.y_val,
                epochs=max_epochs,
                patience=ANN_PATIENCE,
                random_state=random_state + fold_index,
                verbose=0,
                **params,
            )

            probabilities = predict_ann_proba(
                model,
                fold.X_test,
            )

            score = roc_auc_score(
                fold.y_test,
                probabilities,
            )

            fold_scores.append(float(score))

            best_epoch = int(
                np.argmin(
                    history.history["val_loss"]
                ) + 1
            )

            best_epochs.append(best_epoch)

            trial.set_user_attr(
                "best_epochs",
                best_epochs.copy(),
            )

            report_fold_score(
                trial,
                fold_scores,
                fold_index,
            )

        finally:

            del model
            del history

            keras.backend.clear_session()

            gc.collect()

    trial.set_user_attr(
        "cv_std",
        float(np.std(fold_scores, ddof=1)),
    )

    return float(np.mean(fold_scores))


def create_study(
    model_name,
    random_state=RANDOM_STATE,
):
    """Create a reproducible Optuna study."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        n_startup_trials=5,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )

    return optuna.create_study(
        study_name=f"telecom_{model_name}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )


def optimize_model(
    model_name,
    train_df,
    n_trials,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    ann_max_epochs=ANN_MAX_EPOCHS,
    show_progress_bar=True,
):
    """Run hyperparameter optimization for one model."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if n_trials <= 0:
        raise ValueError(
            "n_trials must be positive."
        )

    if ann_max_epochs <= 0:
        raise ValueError(
            "ann_max_epochs must be positive."
        )

    study = create_study(
        model_name,
        random_state=random_state,
    )

    if model_name == "tensorflow_ann":

        prepared_folds = prepare_ann_cv_folds(
            train_df,
            n_splits=n_splits,
            random_state=random_state,
        )

        def objective(trial):

            return ann_objective(
                trial,
                prepared_folds,
                random_state=random_state,
                max_epochs=ann_max_epochs,
            )

    else:

        prepared_folds = prepare_cv_folds(
            train_df,
            n_splits=n_splits,
            random_state=random_state,
        )

        objective_function = (
            logistic_objective
            if model_name == "logistic_regression"
            else xgboost_objective
        )

        def objective(trial):

            return objective_function(
                trial,
                prepared_folds,
                random_state=random_state,
            )

    print("\n" + "=" * 60)

    print(f"Optimizing: {model_name}")

    print(f"Trials: {n_trials}")

    print(f"CV folds: {n_splits}")

    print("=" * 60)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=show_progress_bar,
    )

    return study


def summarize_study(study):
    """Print and return the best completed trial."""

    best_trial = study.best_trial

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    pruned = sum(
        trial.state == optuna.trial.TrialState.PRUNED
        for trial in study.trials
    )

    failed = sum(
        trial.state == optuna.trial.TrialState.FAIL
        for trial in study.trials
    )

    summary = {
        "study_name": study.study_name,
        "best_trial": best_trial.number,
        "best_cv_roc_auc": float(best_trial.value),
        "cv_std": best_trial.user_attrs.get("cv_std"),
        "fold_scores": best_trial.user_attrs.get(
            "fold_scores"
        ),
        "best_params": best_trial.user_attrs.get(
            "model_params",
            best_trial.params,
        ),
        "completed_trials": completed,
        "pruned_trials": pruned,
        "failed_trials": failed,
    }

    if "best_epochs" in best_trial.user_attrs:

        summary["best_epochs"] = (
            best_trial.user_attrs["best_epochs"]
        )

    print("\n" + "=" * 60)

    print(f"Study: {study.study_name}")

    print(f"Best trial: {best_trial.number}")

    print(
        f"Best CV ROC-AUC: {best_trial.value:.6f}"
    )

    print(f"CV standard deviation: {summary['cv_std']}")

    print(f"Fold scores: {summary['fold_scores']}")

    print(f"Completed trials: {completed}")

    print(f"Pruned trials: {pruned}")

    print(f"Failed trials: {failed}")

    print("\nBest model parameters:")

    for key, value in summary["best_params"].items():

        print(f"  {key}: {value}")

    print("=" * 60)

    return summary


def main():
    """Command-line entry point for optimization."""

    parser = argparse.ArgumentParser(
        description="Telecom churn hyperparameter optimization"
    )

    parser.add_argument(
        "--model",
        choices=[*SUPPORTED_MODELS, "all"],
        default="all",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one trial per model with reduced training.",
    )

    args = parser.parse_args()

    train_df, val_df, test_df = (
        load_training_partitions()
    )

    print("\nDataset partitions:")

    print(f"Training: {len(train_df):,}")

    print(f"Validation: {len(val_df):,}")

    print(f"Test: {len(test_df):,}")

    print(
        "\nOnly the training partition "
        "will be used for optimization."
    )

    trial_counts = {
        "logistic_regression": LR_TRIALS,
        "xgboost": XGB_TRIALS,
        "tensorflow_ann": ANN_TRIALS,
    }

    models = (
        SUPPORTED_MODELS
        if args.model == "all"
        else [args.model]
    )

    summaries = []

    for model_name in models:

        study = optimize_model(
            model_name=model_name,
            train_df=train_df,
            n_trials=(
                1
                if args.smoke_test
                else trial_counts[model_name]
            ),
            n_splits=(
                2
                if args.smoke_test
                else N_SPLITS
            ),
            ann_max_epochs=(
                3
                if args.smoke_test
                else ANN_MAX_EPOCHS
            ),
        )

        summaries.append(
            summarize_study(study)
        )

    comparison = pd.DataFrame(summaries)

    print("\nOptimization summary:")

    print(
        comparison[
            [
                "study_name",
                "best_cv_roc_auc",
                "cv_std",
                "completed_trials",
                "pruned_trials",
                "failed_trials",
            ]
        ].to_string(index=False)
    )

    print("\nProduction models remain unchanged.")


if __name__ == "__main__":
    main()