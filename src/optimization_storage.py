"""
src/optimization_storage.py

Persistent Optuna hyperparameter optimization.

Features:
- SQLite-backed study storage
- Interrupted optimization recovery
- Total trial budget enforcement
- Dataset fingerprinting
- Configuration compatibility checks
- Separate smoke-test studies
- Trial history export
- Best hyperparameter export
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from optuna.trial import TrialState

from src.optimization import (
    RANDOM_STATE,
    DATA_PATH,
    N_SPLITS,
    LR_TRIALS,
    XGB_TRIALS,
    ANN_TRIALS,
    ANN_MAX_EPOCHS,
    ANN_PATIENCE,
    SUPPORTED_MODELS,
    load_training_partitions,
    prepare_cv_folds,
    prepare_ann_cv_folds,
    logistic_objective,
    xgboost_objective,
    ann_objective,
    summarize_study,
)


REPORTS_DIR = Path("reports/optimization")

DATABASE_PATH = REPORTS_DIR / "optuna_studies.db"

SMOKE_REPORTS_DIR = REPORTS_DIR / "smoke"

SMOKE_DATABASE_PATH = (
    SMOKE_REPORTS_DIR / "optuna_smoke.db"
)

STUDY_VERSION = "v1"

TRIAL_BUDGETS = {
    "logistic_regression": LR_TRIALS,
    "xgboost": XGB_TRIALS,
    "tensorflow_ann": ANN_TRIALS,
}


def calculate_dataset_hash(
    data_path=DATA_PATH,
):
    """Calculate a SHA-256 fingerprint of the original dataset."""

    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    sha256 = hashlib.sha256()

    with data_path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_code_hash():
    """Fingerprint code affecting optimization results."""

    source_dir = Path(__file__).resolve().parent

    source_files = [
        "optimization.py",
        "optimization_storage.py",
        "train_ann.py",
        "preprocessing.py",
        "feature_engineering.py",
        "cleaning.py",
        "validation.py",
    ]

    sha256 = hashlib.sha256()

    for filename in source_files:

        path = source_dir / filename

        sha256.update(
            filename.encode("utf-8")
        )

        sha256.update(
            path.read_bytes()
        )

    return sha256.hexdigest()


def build_study_configuration(
    model_name,
    dataset_hash,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    ann_max_epochs=ANN_MAX_EPOCHS,
    smoke_test=False,
):
    """Build the configuration associated with a study."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return {
        "study_version": STUDY_VERSION,
        "model_name": model_name,
        "dataset_hash": dataset_hash,
        "code_hash": calculate_code_hash(),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "objective": "roc_auc",
        "direction": "maximize",
        "sampler": "TPE",
        "pruner": "MedianPruner",
        "ann_max_epochs": int(ann_max_epochs),
        "ann_patience": ANN_PATIENCE,
        "smoke_test": bool(smoke_test),
    }


def create_storage(
    database_path=DATABASE_PATH,
):
    """Create an SQLite-backed Optuna storage object."""

    database_path = Path(
        database_path
    ).resolve()

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_url = (
        f"sqlite:///{database_path.as_posix()}"
    )

    return optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {
                "timeout": 60,
            },
        },
    )


def validate_study_configuration(
    study,
    configuration,
):
    """Prevent incompatible experiments from being combined."""

    saved_configuration = study.user_attrs.get(
        "configuration"
    )

    if saved_configuration is None:

        if study.trials:
            raise ValueError(
                "Existing study has trials but no "
                "configuration metadata."
            )

        study.set_user_attr(
            "configuration",
            configuration,
        )

        return

    if saved_configuration != configuration:

        differences = {}

        all_keys = (
            set(saved_configuration)
            | set(configuration)
        )

        for key in sorted(all_keys):

            old_value = saved_configuration.get(key)

            new_value = configuration.get(key)

            if old_value != new_value:

                differences[key] = {
                    "saved": old_value,
                    "current": new_value,
                }

        raise ValueError(
            "Study configuration has changed.\n"
            + json.dumps(
                differences,
                indent=2,
            )
            + "\nUse a new study version or database."
        )


def recover_interrupted_trials(study):
    """
    Mark abandoned RUNNING trials as failed.

    This function assumes only one optimization process
    is using the database at a time.
    """

    running_trials = study.get_trials(
        deepcopy=False,
        states=(TrialState.RUNNING,),
    )

    recovered = 0

    for trial in running_trials:

        study.tell(
            trial.number,
            state=TrialState.FAIL,
            skip_if_finished=True,
        )

        recovered += 1

    if recovered:

        print(
            f"Recovered {recovered} interrupted trial(s)."
        )

    return recovered


def count_trial_states(study):
    """Count trials by their current state."""

    counts = {
        "completed": 0,
        "pruned": 0,
        "failed": 0,
        "running": 0,
        "waiting": 0,
    }

    state_mapping = {
        TrialState.COMPLETE: "completed",
        TrialState.PRUNED: "pruned",
        TrialState.FAIL: "failed",
        TrialState.RUNNING: "running",
        TrialState.WAITING: "waiting",
    }

    for trial in study.get_trials(
        deepcopy=False
    ):

        state_name = state_mapping.get(
            trial.state
        )

        if state_name is not None:

            counts[state_name] += 1

    return counts


def calculate_remaining_trials(
    study,
    total_budget,
):
    """Calculate the unused optimization trial budget."""

    if total_budget <= 0:
        raise ValueError(
            "total_budget must be positive."
        )

    counts = count_trial_states(study)

    consumed_budget = (
        counts["completed"]
        + counts["pruned"]
    )

    return max(
        0,
        total_budget - consumed_budget,
    )


def create_persistent_study(
    model_name,
    configuration,
    database_path=DATABASE_PATH,
):
    """Create or resume a persistent optimization study."""

    storage = create_storage(
        database_path
    )

    mode = (
        "smoke"
        if configuration["smoke_test"]
        else "full"
    )

    study_name = (
        f"telecom_{model_name}_"
        f"{mode}_{STUDY_VERSION}"
    )

    sampler = optuna.samplers.TPESampler(
        seed=configuration["random_state"],
        n_startup_trials=5,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    validate_study_configuration(
        study,
        configuration,
    )

    recover_interrupted_trials(
        study
    )

    # Use a different deterministic seed when resuming
    # to avoid restarting the same sampling sequence.
    previous_trials = len(study.trials)

    study.sampler = optuna.samplers.TPESampler(
        seed=(
            configuration["random_state"]
            + previous_trials
        ),
        n_startup_trials=5,
    )

    return study


def optimize_persistent_model(
    model_name,
    train_df,
    dataset_hash,
    total_budget,
    database_path=DATABASE_PATH,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    ann_max_epochs=ANN_MAX_EPOCHS,
    smoke_test=False,
    show_progress_bar=True,
):
    """Run or resume optimization for one model."""

    configuration = build_study_configuration(
        model_name=model_name,
        dataset_hash=dataset_hash,
        n_splits=n_splits,
        random_state=random_state,
        ann_max_epochs=ann_max_epochs,
        smoke_test=smoke_test,
    )

    study = create_persistent_study(
        model_name=model_name,
        configuration=configuration,
        database_path=database_path,
    )

    counts = count_trial_states(
        study
    )

    remaining_trials = calculate_remaining_trials(
        study,
        total_budget,
    )

    print("\n" + "=" * 60)

    print(f"Study: {study.study_name}")

    print(f"Total trial budget: {total_budget}")

    print(f"Completed: {counts['completed']}")

    print(f"Pruned: {counts['pruned']}")

    print(f"Failed: {counts['failed']}")

    print(f"Remaining: {remaining_trials}")

    print("=" * 60)

    if remaining_trials == 0:

        print(
            "Trial budget already exhausted. "
            "Skipping optimization."
        )

        return study

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

    study.optimize(
        objective,
        n_trials=remaining_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=show_progress_bar,
    )

    gc.collect()

    return study


def export_trial_history(
    studies,
    output_path,
):
    """Export trial histories from all available studies."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for model_name, study in studies.items():

        for trial in study.trials:

            duration = (
                trial.duration.total_seconds()
                if trial.duration is not None
                else None
            )

            record = {
                "model": model_name,
                "study_name": study.study_name,
                "trial_number": trial.number,
                "state": trial.state.name,
                "roc_auc": trial.value,
                "cv_std": trial.user_attrs.get(
                    "cv_std"
                ),
                "duration_seconds": duration,
                "fold_scores": json.dumps(
                    trial.user_attrs.get(
                        "fold_scores",
                        [],
                    )
                ),
                "best_epochs": json.dumps(
                    trial.user_attrs.get(
                        "best_epochs",
                        [],
                    )
                ),
                "model_parameters": json.dumps(
                    trial.user_attrs.get(
                        "model_params",
                        trial.params,
                    ),
                    sort_keys=True,
                ),
            }

            for key, value in trial.params.items():

                record[f"param_{key}"] = value

            records.append(record)

    history_df = pd.DataFrame(
        records
    )

    history_df.to_csv(
        output_path,
        index=False,
    )

    return history_df


def export_best_hyperparameters(
    studies,
    output_path,
):
    """Export the best completed trial for each model."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    for model_name, study in studies.items():

        counts = count_trial_states(
            study
        )

        if counts["completed"] == 0:

            continue

        best_trial = study.best_trial

        results[model_name] = {
            "study_name": study.study_name,
            "best_trial": best_trial.number,
            "best_cv_roc_auc": float(
                best_trial.value
            ),
            "cv_std": best_trial.user_attrs.get(
                "cv_std"
            ),
            "fold_scores": best_trial.user_attrs.get(
                "fold_scores",
                [],
            ),
            "best_epochs": best_trial.user_attrs.get(
                "best_epochs",
                [],
            ),
            "best_params": best_trial.user_attrs.get(
                "model_params",
                best_trial.params,
            ),
            "trial_counts": counts,
            "configuration": study.user_attrs.get(
                "configuration"
            ),
        }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            allow_nan=False,
        )

    return results


def load_saved_studies(
    database_path=DATABASE_PATH,
    smoke_test=False,
):
    """Load existing studies from the selected database."""

    database_path = Path(database_path)

    if not database_path.exists():

        return {}

    storage = create_storage(
        database_path
    )

    available_names = set(
        optuna.study.get_all_study_names(
            storage=storage
        )
    )

    mode = (
        "smoke"
        if smoke_test
        else "full"
    )

    studies = {}

    for model_name in SUPPORTED_MODELS:

        study_name = (
            f"telecom_{model_name}_"
            f"{mode}_{STUDY_VERSION}"
        )

        if study_name not in available_names:

            continue

        studies[model_name] = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

    return studies


def export_study_reports(
    studies,
    reports_dir=REPORTS_DIR,
):
    """Export optimization history and selected parameters."""

    reports_dir = Path(reports_dir)

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        reports_dir / "optuna_trials.csv"
    )

    parameters_path = (
        reports_dir / "best_hyperparameters.json"
    )

    history_df = export_trial_history(
        studies,
        history_path,
    )

    best_parameters = export_best_hyperparameters(
        studies,
        parameters_path,
    )

    print("\nOptimization reports:")

    print(
        f"Trial history: {history_path}"
    )

    print(
        f"Best parameters: {parameters_path}"
    )

    print(
        f"Total recorded trials: {len(history_df)}"
    )

    print(
        f"Models with completed trials: "
        f"{len(best_parameters)}"
    )

    return history_df, best_parameters


def main():
    """Command-line interface for persistent optimization."""

    parser = argparse.ArgumentParser(
        description=(
            "Persistent telecom churn "
            "hyperparameter optimization"
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            *SUPPORTED_MODELS,
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run reduced optimization using "
            "a separate SQLite database."
        ),
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export existing studies without training.",
    )

    args = parser.parse_args()

    database_path = (
        SMOKE_DATABASE_PATH
        if args.smoke_test
        else DATABASE_PATH
    )

    reports_dir = (
        SMOKE_REPORTS_DIR
        if args.smoke_test
        else REPORTS_DIR
    )

    if args.export_only:

        studies = load_saved_studies(
            database_path,
            smoke_test=args.smoke_test,
        )

        if not studies:

            print("No saved studies found.")

            return

        export_study_reports(
            studies,
            reports_dir,
        )

        return

    dataset_hash = calculate_dataset_hash()

    train_df, val_df, test_df = (
        load_training_partitions()
    )

    print("\nDataset partitions:")

    print(f"Training: {len(train_df):,}")

    print(f"Validation: {len(val_df):,}")

    print(f"Test: {len(test_df):,}")

    print(f"Dataset SHA-256: {dataset_hash}")

    print(f"Database: {database_path}")

    print(
        "\nOnly the training partition "
        "will be used for optimization."
    )

    models = (
        SUPPORTED_MODELS
        if args.model == "all"
        else [args.model]
    )

    try:

        for model_name in models:

            study = optimize_persistent_model(
                model_name=model_name,
                train_df=train_df,
                dataset_hash=dataset_hash,
                total_budget=(
                    1
                    if args.smoke_test
                    else TRIAL_BUDGETS[model_name]
                ),
                database_path=database_path,
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
                smoke_test=args.smoke_test,
            )

            if count_trial_states(
                study
            )["completed"] > 0:

                summarize_study(study)

            studies = load_saved_studies(
                database_path,
                smoke_test=args.smoke_test,
            )

            export_study_reports(
                studies,
                reports_dir,
            )

    except KeyboardInterrupt:

        print(
            "\nOptimization interrupted. "
            "Completed trials are saved in SQLite."
        )

        print(
            "Run the same command again to resume."
        )

    finally:

        studies = load_saved_studies(
            database_path,
            smoke_test=args.smoke_test,
        )

        if studies:

            export_study_reports(
                studies,
                reports_dir,
            )

    print("\nProduction models remain unchanged.")


if __name__ == "__main__":
    main()