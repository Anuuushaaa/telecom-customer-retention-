"""
Tests for Optuna optimization, persistent storage,
and optimized model evaluation.
"""

import json

import numpy as np
import pandas as pd
import optuna
import pytest

from optuna.trial import TrialState

from src.optimization import (
    load_training_partitions,
    create_cv_splits,
    prepare_cv_folds,
    prepare_ann_cv_folds,
    build_logistic_model,
    build_xgboost_model,
    optimize_model,
    summarize_study,
)

from src.optimization_storage import (
    calculate_dataset_hash,
    calculate_code_hash,
    build_study_configuration,
    create_persistent_study,
    validate_study_configuration,
    recover_interrupted_trials,
    count_trial_states,
    calculate_remaining_trials,
    export_trial_history,
    export_best_hyperparameters,
)

from src.optimization_evaluation import (
    calculate_metrics,
    compare_with_baseline,
    save_validation_predictions,
)


@pytest.fixture(scope="module")
def partitions():
    return load_training_partitions()


@pytest.fixture(scope="module")
def cv_folds(partitions):
    train_df, _, _ = partitions

    return prepare_cv_folds(
        train_df,
        n_splits=3,
    )


@pytest.fixture(scope="module")
def ann_cv_folds(partitions):
    train_df, _, _ = partitions

    return prepare_ann_cv_folds(
        train_df,
        n_splits=3,
    )


@pytest.fixture(scope="module")
def optimized_study(partitions):
    train_df, _, _ = partitions

    return optimize_model(
        model_name="logistic_regression",
        train_df=train_df,
        n_trials=2,
        n_splits=2,
        show_progress_bar=False,
    )


# Dataset and cross-validation tests


def test_partition_sizes(partitions):
    train_df, val_df, test_df = partitions

    assert len(train_df) == 4225
    assert len(val_df) == 1409
    assert len(test_df) == 1409


def test_partition_disjointness(partitions):
    train_df, val_df, test_df = partitions

    train_ids = set(train_df["customerID"])
    val_ids = set(val_df["customerID"])
    test_ids = set(test_df["customerID"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_cv_fold_count(partitions):
    train_df, _, _ = partitions

    splits = create_cv_splits(
        train_df,
        n_splits=3,
    )

    assert len(splits) == 3


def test_cv_indices_disjoint(partitions):
    train_df, _, _ = partitions

    splits = create_cv_splits(train_df)

    for train_indices, score_indices in splits:

        assert set(train_indices).isdisjoint(
            score_indices
        )


def test_cv_complete_coverage(partitions):
    train_df, _, _ = partitions

    splits = create_cv_splits(train_df)

    score_indices = np.concatenate([
        score
        for _, score in splits
    ])

    assert len(score_indices) == len(train_df)

    assert len(np.unique(score_indices)) == len(train_df)


def test_cv_invalid_fold_count(partitions):
    train_df, _, _ = partitions

    with pytest.raises(ValueError):

        create_cv_splits(
            train_df,
            n_splits=1,
        )


def test_cv_preprocessing_schema(cv_folds):
    for fold in cv_folds:

        assert list(fold.X_train.columns) == list(
            fold.X_test.columns
        )


def test_cv_scaler_training_only(cv_folds):
    for fold in cv_folds:

        assert int(
            fold.scaler.n_samples_seen_
        ) == len(fold.X_train)


def test_cv_customer_disjointness(cv_folds):
    for fold in cv_folds:

        assert set(
            fold.customer_ids_train
        ).isdisjoint(
            set(fold.customer_ids_test)
        )


def test_ann_cv_partition_disjointness(ann_cv_folds):
    for fold in ann_cv_folds:

        train_ids = set(fold.customer_ids_train)
        val_ids = set(fold.customer_ids_val)
        score_ids = set(fold.customer_ids_test)

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(score_ids)
        assert val_ids.isdisjoint(score_ids)


def test_ann_cv_scaler_training_only(ann_cv_folds):
    for fold in ann_cv_folds:

        assert int(
            fold.scaler.n_samples_seen_
        ) == len(fold.X_train)


# Model construction tests


def test_logistic_model_parameters():
    parameters = {
        "C": 2.0,
        "solver": "lbfgs",
        "l1_ratio": 0.0,
    }

    model = build_logistic_model(parameters)

    assert model.C == 2.0
    assert model.solver == "lbfgs"
    assert model.class_weight == "balanced"


def test_xgboost_class_weight(partitions):
    train_df, _, _ = partitions

    y_train = train_df["Churn"]

    parameters = {
        "n_estimators": 100,
        "max_depth": 2,
    }

    model = build_xgboost_model(
        parameters,
        y_train,
    )

    expected = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    assert model.get_params()[
        "scale_pos_weight"
    ] == pytest.approx(expected)


# Optimization tests


def test_optimization_trial_count(optimized_study):
    assert len(optimized_study.trials) == 2


def test_optimization_objective_range(optimized_study):
    assert 0 <= optimized_study.best_value <= 1


def test_optimization_fold_scores(optimized_study):
    scores = optimized_study.best_trial.user_attrs[
        "fold_scores"
    ]

    assert len(scores) == 2

    assert all(
        0 <= score <= 1
        for score in scores
    )

    assert optimized_study.best_value == pytest.approx(
        np.mean(scores)
    )


def test_optimization_cv_standard_deviation(
    optimized_study,
):
    attributes = optimized_study.best_trial.user_attrs

    assert attributes["cv_std"] == pytest.approx(
        np.std(
            attributes["fold_scores"],
            ddof=1,
        )
    )


def test_optimization_summary(optimized_study):
    summary = summarize_study(
        optimized_study
    )

    assert summary["completed_trials"] == 2

    assert summary["failed_trials"] == 0

    assert "best_params" in summary


# Fingerprint tests


def test_dataset_fingerprint():
    fingerprint = calculate_dataset_hash()

    assert len(fingerprint) == 64

    int(fingerprint, 16)


def test_dataset_fingerprint_reproducible():
    assert calculate_dataset_hash() == (
        calculate_dataset_hash()
    )


def test_code_fingerprint():
    fingerprint = calculate_code_hash()

    assert len(fingerprint) == 64

    int(fingerprint, 16)


# Persistent storage tests


@pytest.fixture
def persistent_configuration():
    return build_study_configuration(
        model_name="logistic_regression",
        dataset_hash=calculate_dataset_hash(),
        n_splits=2,
        smoke_test=True,
    )


@pytest.fixture
def persistent_study(
    tmp_path,
    persistent_configuration,
):
    return create_persistent_study(
        model_name="logistic_regression",
        configuration=persistent_configuration,
        database_path=tmp_path / "test_optuna.db",
    )


def test_persistent_study_creation(persistent_study):
    assert len(persistent_study.trials) == 0


def test_persistent_configuration(
    persistent_study,
    persistent_configuration,
):
    assert persistent_study.user_attrs[
        "configuration"
    ] == persistent_configuration


def test_configuration_mismatch(
    persistent_study,
    persistent_configuration,
):
    modified = persistent_configuration.copy()

    modified["random_state"] = 999

    with pytest.raises(ValueError):

        validate_study_configuration(
            persistent_study,
            modified,
        )


def test_trial_state_counts(persistent_study):
    trial = persistent_study.ask()

    persistent_study.tell(
        trial,
        0.84,
    )

    counts = count_trial_states(
        persistent_study
    )

    assert counts["completed"] == 1
    assert counts["failed"] == 0
    assert counts["running"] == 0


def test_remaining_trial_budget(persistent_study):
    trial = persistent_study.ask()

    persistent_study.tell(
        trial,
        0.84,
    )

    assert calculate_remaining_trials(
        persistent_study,
        total_budget=5,
    ) == 4


def test_exhausted_trial_budget(persistent_study):
    trial = persistent_study.ask()

    persistent_study.tell(
        trial,
        0.84,
    )

    assert calculate_remaining_trials(
        persistent_study,
        total_budget=1,
    ) == 0


def test_pruned_trial_consumes_budget(persistent_study):
    trial = persistent_study.ask()

    persistent_study.tell(
        trial,
        state=TrialState.PRUNED,
    )

    assert calculate_remaining_trials(
        persistent_study,
        total_budget=5,
    ) == 4


def test_failed_trial_does_not_consume_budget(
    persistent_study,
):
    trial = persistent_study.ask()

    persistent_study.tell(
        trial,
        state=TrialState.FAIL,
    )

    assert calculate_remaining_trials(
        persistent_study,
        total_budget=5,
    ) == 5


def test_interrupted_trial_recovery(persistent_study):
    persistent_study.ask()

    assert count_trial_states(
        persistent_study
    )["running"] == 1

    recover_interrupted_trials(
        persistent_study
    )

    counts = count_trial_states(
        persistent_study
    )

    assert counts["running"] == 0
    assert counts["failed"] == 1


def test_persistent_study_resume(
    tmp_path,
    persistent_configuration,
):
    database_path = tmp_path / "resume_test.db"

    first_study = create_persistent_study(
        model_name="logistic_regression",
        configuration=persistent_configuration,
        database_path=database_path,
    )

    trial = first_study.ask()

    first_study.tell(
        trial,
        0.84,
    )

    resumed_study = create_persistent_study(
        model_name="logistic_regression",
        configuration=persistent_configuration,
        database_path=database_path,
    )

    assert len(resumed_study.trials) == 1

    assert resumed_study.best_value == pytest.approx(
        0.84
    )


# Report export tests


def test_trial_history_export(
    optimized_study,
    tmp_path,
):
    output_path = tmp_path / "trials.csv"

    export_trial_history(
        {"logistic_regression": optimized_study},
        output_path,
    )

    assert output_path.is_file()

    results = pd.read_csv(output_path)

    assert len(results) == 2

    assert "roc_auc" in results.columns


def test_best_parameters_export(
    optimized_study,
    tmp_path,
):
    output_path = tmp_path / "parameters.json"

    export_best_hyperparameters(
        {"logistic_regression": optimized_study},
        output_path,
    )

    assert output_path.is_file()

    with output_path.open(
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    assert "logistic_regression" in results

    assert "best_params" in results[
        "logistic_regression"
    ]


# Evaluation tests


@pytest.fixture(scope="module")
def validation_data(partitions):
    _, val_df, _ = partitions

    labels = val_df["Churn"].to_numpy()

    # Controlled probabilities derived from real
    # validation labels for metric correctness tests.
    probabilities = np.where(
        labels == 1,
        0.8,
        0.2,
    )

    return val_df, labels, probabilities


def test_evaluation_metric_keys(validation_data):
    _, labels, probabilities = validation_data

    metrics = calculate_metrics(
        labels,
        probabilities,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }


def test_perfect_prediction_metrics(validation_data):
    _, labels, probabilities = validation_data

    metrics = calculate_metrics(
        labels,
        probabilities,
    )

    for value in metrics.values():

        assert value == pytest.approx(1.0)


def test_invalid_threshold(validation_data):
    _, labels, probabilities = validation_data

    with pytest.raises(ValueError):

        calculate_metrics(
            labels,
            probabilities,
            threshold=1.5,
        )


def test_mismatched_prediction_lengths(validation_data):
    _, labels, probabilities = validation_data

    with pytest.raises(ValueError):

        calculate_metrics(
            labels,
            probabilities[:-1],
        )


def test_nonfinite_probabilities(validation_data):
    _, labels, probabilities = validation_data

    invalid = probabilities.copy()

    invalid[0] = np.nan

    with pytest.raises(ValueError):

        calculate_metrics(
            labels,
            invalid,
        )


def test_out_of_range_probabilities(validation_data):
    _, labels, probabilities = validation_data

    invalid = probabilities.copy()

    invalid[0] = 1.5

    with pytest.raises(ValueError):

        calculate_metrics(
            labels,
            invalid,
        )


def test_validation_prediction_export(
    validation_data,
    tmp_path,
):
    val_df, labels, probabilities = validation_data

    output_path = tmp_path / "predictions.csv"

    result = save_validation_predictions(
        customer_ids=val_df["customerID"],
        y_true=labels,
        probabilities={
            "logistic_regression": probabilities,
        },
        output_path=output_path,
    )

    assert output_path.is_file()

    assert len(result) == len(val_df)

    assert result["customerID"].is_unique

    assert np.array_equal(
        result["logistic_regression_prediction"],
        labels,
    )


def test_duplicate_prediction_ids(
    validation_data,
    tmp_path,
):
    val_df, labels, probabilities = validation_data

    customer_ids = val_df[
        "customerID"
    ].to_numpy(copy=True)

    customer_ids[1] = customer_ids[0]

    with pytest.raises(ValueError):

        save_validation_predictions(
            customer_ids=customer_ids,
            y_true=labels,
            probabilities={
                "logistic_regression": probabilities,
            },
            output_path=tmp_path / "invalid.csv",
        )