"""
Tests for OOF prediction generation,
probability calibration and calibration metrics.
"""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.optimization import (
    SUPPORTED_MODELS,
    load_training_partitions,
)

from src.calibration_oof import (
    create_calibration_folds,
    validate_oof_probabilities,
)

from src.probability_calibration import (
    CALIBRATION_METHODS,
    validate_probabilities,
    validate_labels,
    probability_to_logit,
    validate_oof_predictions,
    fit_sigmoid_calibrator,
    fit_isotonic_calibrator,
    apply_calibration,
    calculate_calibration_bins,
    calculate_calibration_metrics,
    fit_all_calibrators,
    evaluate_calibrators,
    compare_calibration_methods,
    select_calibration_methods,
    save_calibration_results,
)


@pytest.fixture(scope="module")
def partitions():
    return load_training_partitions()


@pytest.fixture(scope="module")
def training_df(partitions):
    return partitions[0]


@pytest.fixture(scope="module")
def validation_df(partitions):
    return partitions[1]


@pytest.fixture(scope="module")
def oof_predictions(training_df):

    labels = training_df["Churn"].to_numpy()

    rng = np.random.default_rng(42)

    assignments = np.zeros(
        len(training_df),
        dtype=int,
    )

    for fold, (_, score_indices) in enumerate(
        create_calibration_folds(training_df),
        start=1,
    ):
        assignments[score_indices] = fold

    predictions = pd.DataFrame({
        "customerID": training_df[
            "customerID"
        ].to_numpy(),
        "actual_churn": labels,
        "fold": assignments,
    })

    for model in SUPPORTED_MODELS:

        predictions[
            f"{model}_probability"
        ] = np.clip(
            np.where(labels == 1, 0.7, 0.3)
            + rng.normal(0, 0.2, len(labels)),
            0,
            1,
        )

    return predictions


@pytest.fixture(scope="module")
def validation_predictions(validation_df):

    labels = validation_df["Churn"].to_numpy()

    rng = np.random.default_rng(43)

    predictions = pd.DataFrame({
        "customerID": validation_df[
            "customerID"
        ].to_numpy(),
        "actual_churn": labels,
    })

    for model in SUPPORTED_MODELS:

        predictions[
            f"{model}_probability"
        ] = np.clip(
            np.where(labels == 1, 0.7, 0.3)
            + rng.normal(0, 0.2, len(labels)),
            0,
            1,
        )

    return predictions


@pytest.fixture(scope="module")
def calibrators(oof_predictions):

    return fit_all_calibrators(
        oof_predictions
    )


@pytest.fixture(scope="module")
def evaluation_results(
    validation_predictions,
    calibrators,
):

    return evaluate_calibrators(
        validation_predictions,
        calibrators,
    )


# --------------------------------------------------
# OOF fold tests
# --------------------------------------------------


def test_fold_count(training_df):

    folds = create_calibration_folds(
        training_df
    )

    assert len(folds) == 5


def test_fold_sizes(training_df):

    folds = create_calibration_folds(
        training_df
    )

    for fit_indices, score_indices in folds:

        assert len(fit_indices) == 3380
        assert len(score_indices) == 845


def test_fold_no_overlap(training_df):

    for fit_indices, score_indices in (
        create_calibration_folds(training_df)
    ):

        assert not np.intersect1d(
            fit_indices,
            score_indices,
        ).size


def test_fold_complete_coverage(training_df):

    counts = np.zeros(
        len(training_df),
        dtype=int,
    )

    for _, score_indices in (
        create_calibration_folds(training_df)
    ):

        counts[score_indices] += 1

    assert np.all(counts == 1)


def test_fold_stratification(training_df):

    overall_rate = training_df[
        "Churn"
    ].mean()

    for _, score_indices in (
        create_calibration_folds(training_df)
    ):

        fold_rate = training_df.iloc[
            score_indices
        ]["Churn"].mean()

        assert abs(
            fold_rate - overall_rate
        ) < 0.01


def test_fold_reproducibility(training_df):

    first = create_calibration_folds(
        training_df
    )

    second = create_calibration_folds(
        training_df
    )

    for (a_train, a_score), (
        b_train,
        b_score,
    ) in zip(first, second):

        np.testing.assert_array_equal(
            a_train,
            b_train,
        )

        np.testing.assert_array_equal(
            a_score,
            b_score,
        )


@pytest.mark.parametrize(
    "n_splits",
    [0, 1, -1, True, 2.5],
)
def test_invalid_fold_count(
    training_df,
    n_splits,
):

    with pytest.raises(ValueError):

        create_calibration_folds(
            training_df,
            n_splits=n_splits,
        )


def test_fold_single_class(training_df):

    modified = training_df.copy()

    modified["Churn"] = 0

    with pytest.raises(ValueError):

        create_calibration_folds(modified)


def test_partition_disjointness(partitions):

    train, validation, test = partitions

    train_ids = set(train["customerID"])

    validation_ids = set(
        validation["customerID"]
    )

    test_ids = set(test["customerID"])

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


# --------------------------------------------------
# Probability validation tests
# --------------------------------------------------


def test_valid_probabilities():

    values = validate_probabilities(
        [0.0, 0.5, 1.0]
    )

    np.testing.assert_array_equal(
        values,
        [0.0, 0.5, 1.0],
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[0.2, 0.3]],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.5, np.nan],
        [0.5, np.inf],
    ],
)
def test_invalid_probabilities(values):

    with pytest.raises(ValueError):

        validate_probabilities(values)


def test_oof_probability_length():

    with pytest.raises(ValueError):

        validate_oof_probabilities(
            [0.1, 0.2],
            expected_rows=3,
        )


def test_valid_oof_probabilities():

    values = validate_oof_probabilities(
        [0.2, 0.8],
        expected_rows=2,
    )

    assert len(values) == 2


@pytest.mark.parametrize(
    "labels, expected_rows",
    [
        ([0, 1], 3),
        ([0, 0], 2),
        ([0, 2], 2),
        ([[0, 1]], 2),
    ],
)
def test_invalid_labels(
    labels,
    expected_rows,
):

    with pytest.raises(ValueError):

        validate_labels(
            labels,
            expected_rows,
        )


# --------------------------------------------------
# OOF artifact integrity
# --------------------------------------------------


def test_valid_oof_predictions(
    oof_predictions,
    training_df,
):

    validate_oof_predictions(
        oof_predictions,
        training_df,
    )


def test_oof_missing_column(
    oof_predictions,
    training_df,
):

    modified = oof_predictions.drop(
        columns=["fold"]
    )

    with pytest.raises(ValueError):

        validate_oof_predictions(
            modified,
            training_df,
        )


def test_oof_duplicate_customer(
    oof_predictions,
    training_df,
):

    modified = oof_predictions.copy()

    modified.loc[1, "customerID"] = (
        modified.loc[0, "customerID"]
    )

    with pytest.raises(ValueError):

        validate_oof_predictions(
            modified,
            training_df,
        )


def test_oof_membership_mismatch(
    oof_predictions,
    training_df,
):

    modified = oof_predictions.copy()

    modified.loc[0, "customerID"] = (
        "INVALID-CUSTOMER"
    )

    with pytest.raises(ValueError):

        validate_oof_predictions(
            modified,
            training_df,
        )


def test_oof_label_mismatch(
    oof_predictions,
    training_df,
):

    modified = oof_predictions.copy()

    modified.loc[0, "actual_churn"] = (
        1 - modified.loc[0, "actual_churn"]
    )

    with pytest.raises(ValueError):

        validate_oof_predictions(
            modified,
            training_df,
        )


def test_oof_fold_mismatch(
    oof_predictions,
    training_df,
):

    modified = oof_predictions.copy()

    modified.loc[0, "fold"] = 99

    with pytest.raises(ValueError):

        validate_oof_predictions(
            modified,
            training_df,
        )


def test_oof_order_independence(
    oof_predictions,
    training_df,
):

    shuffled = oof_predictions.sample(
        frac=1,
        random_state=42,
    )

    validate_oof_predictions(
        shuffled,
        training_df,
    )


# --------------------------------------------------
# Sigmoid and isotonic calibration
# --------------------------------------------------


def test_logit_conversion():

    result = probability_to_logit(
        [0.2, 0.5, 0.8]
    )

    assert result.shape == (3, 1)

    assert result[1, 0] == pytest.approx(0)

    assert result[0, 0] == pytest.approx(
        -result[2, 0]
    )


def test_logit_boundary_stability():

    result = probability_to_logit(
        [0.0, 1.0]
    )

    assert np.isfinite(result).all()


def test_sigmoid_calibration(
    oof_predictions,
):

    probabilities = oof_predictions[
        "logistic_regression_probability"
    ]

    labels = oof_predictions[
        "actual_churn"
    ]

    calibrator = fit_sigmoid_calibrator(
        probabilities,
        labels,
    )

    calibrated = apply_calibration(
        probabilities,
        "sigmoid",
        calibrator,
    )

    assert len(calibrated) == len(labels)

    assert np.isfinite(calibrated).all()

    assert np.all(
        (calibrated >= 0)
        & (calibrated <= 1)
    )


def test_sigmoid_monotonicity(calibrators):

    probabilities = np.linspace(
        0,
        1,
        101,
    )

    calibrated = apply_calibration(
        probabilities,
        "sigmoid",
        calibrators[
            "logistic_regression"
        ]["sigmoid"],
    )

    assert np.all(
        np.diff(calibrated) >= -1e-12
    )


def test_isotonic_calibration(
    oof_predictions,
):

    probabilities = oof_predictions[
        "logistic_regression_probability"
    ]

    labels = oof_predictions[
        "actual_churn"
    ]

    calibrator = fit_isotonic_calibrator(
        probabilities,
        labels,
    )

    calibrated = apply_calibration(
        probabilities,
        "isotonic",
        calibrator,
    )

    assert len(calibrated) == len(labels)

    assert np.all(
        (calibrated >= 0)
        & (calibrated <= 1)
    )


def test_isotonic_monotonicity(calibrators):

    probabilities = np.linspace(
        0,
        1,
        101,
    )

    calibrated = apply_calibration(
        probabilities,
        "isotonic",
        calibrators[
            "logistic_regression"
        ]["isotonic"],
    )

    assert np.all(
        np.diff(calibrated) >= -1e-12
    )


def test_uncalibrated_identity():

    probabilities = np.array(
        [0.1, 0.5, 0.9]
    )

    calibrated = apply_calibration(
        probabilities,
        "uncalibrated",
    )

    np.testing.assert_array_equal(
        probabilities,
        calibrated,
    )

    assert calibrated is not probabilities


def test_unsupported_calibration():

    with pytest.raises(ValueError):

        apply_calibration(
            [0.2, 0.8],
            "unsupported",
        )


@pytest.mark.parametrize(
    "method",
    ["sigmoid", "isotonic"],
)
def test_missing_calibrator(method):

    with pytest.raises(ValueError):

        apply_calibration(
            [0.2, 0.8],
            method,
        )


def test_calibrator_count(calibrators):

    assert set(calibrators) == set(
        SUPPORTED_MODELS
    )

    assert sum(
        len(methods)
        for methods in calibrators.values()
    ) == 6


# --------------------------------------------------
# Calibration metrics
# --------------------------------------------------


def test_perfect_calibration_metrics():

    metrics = calculate_calibration_metrics(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    )

    assert metrics["brier_score"] == 0

    assert metrics["ece"] == 0

    assert metrics["mce"] == 0

    assert metrics["roc_auc"] == 1

    assert metrics[
        "average_precision"
    ] == 1


def test_known_brier_score():

    metrics = calculate_calibration_metrics(
        [0, 1],
        [0.2, 0.8],
    )

    assert metrics[
        "brier_score"
    ] == pytest.approx(0.04)


def test_known_calibration_error():

    metrics = calculate_calibration_metrics(
        [0, 0, 1, 1],
        [0.25, 0.25, 0.75, 0.75],
    )

    assert metrics[
        "ece"
    ] == pytest.approx(0.25)

    assert metrics[
        "mce"
    ] == pytest.approx(0.25)


def test_reliability_bin_coverage():

    bins = calculate_calibration_bins(
        [0, 0, 1, 1],
        [0.0, 0.2, 0.8, 1.0],
    )

    assert bins["count"].sum() == 4

    assert bins["weight"].sum() == (
        pytest.approx(1.0)
    )


def test_reliability_boundary_bins():

    bins = calculate_calibration_bins(
        [0, 1],
        [0.0, 1.0],
        n_bins=10,
    )

    assert set(bins["bin"]) == {1, 10}


@pytest.mark.parametrize(
    "n_bins",
    [0, 1, -1, True, 2.5],
)
def test_invalid_bin_count(n_bins):

    with pytest.raises(ValueError):

        calculate_calibration_bins(
            [0, 1],
            [0.2, 0.8],
            n_bins=n_bins,
        )


# --------------------------------------------------
# Evaluation and model comparison
# --------------------------------------------------


def test_evaluation_combination_count(
    evaluation_results,
):

    metrics, _, _ = evaluation_results

    assert len(metrics) == 9


def test_evaluation_prediction_columns(
    evaluation_results,
):

    _, _, predictions = evaluation_results

    expected = {
        "customerID",
        "actual_churn",
    }

    expected.update(
        f"{model}_{method}_probability"
        for model in SUPPORTED_MODELS
        for method in CALIBRATION_METHODS
    )

    assert set(predictions.columns) == expected


def test_evaluation_customer_alignment(
    evaluation_results,
    validation_predictions,
):

    _, _, predictions = evaluation_results

    np.testing.assert_array_equal(
        predictions["customerID"],
        validation_predictions["customerID"],
    )

    np.testing.assert_array_equal(
        predictions["actual_churn"],
        validation_predictions["actual_churn"],
    )


def test_reliability_group_counts(
    evaluation_results,
):

    _, reliability, predictions = (
        evaluation_results
    )

    counts = reliability.groupby(
        ["model", "calibration_method"]
    )["count"].sum()

    assert len(counts) == 9

    assert (
        counts == len(predictions)
    ).all()


def test_calibration_comparison_count(
    evaluation_results,
):

    metrics, _, _ = evaluation_results

    comparison = compare_calibration_methods(
        metrics
    )

    assert len(comparison) == 6


def test_calibration_selection_count(
    evaluation_results,
):

    metrics, _, _ = evaluation_results

    selected = select_calibration_methods(
        metrics
    )

    assert len(selected) == 3

    assert selected["model"].is_unique


def test_calibration_selection_minimum_brier(
    evaluation_results,
):

    metrics, _, _ = evaluation_results

    selected = select_calibration_methods(
        metrics
    )

    for _, row in selected.iterrows():

        minimum = metrics.loc[
            metrics["model"] == row["model"],
            "brier_score",
        ].min()

        assert row["brier_score"] == (
            pytest.approx(minimum)
        )


def test_calibrator_serialization(
    calibrators,
    tmp_path,
):

    original = calibrators[
        "logistic_regression"
    ]["sigmoid"]

    path = tmp_path / "calibrator.pkl"

    joblib.dump(original, path)

    restored = joblib.load(path)

    probabilities = np.array(
        [0.1, 0.5, 0.9]
    )

    np.testing.assert_allclose(
        apply_calibration(
            probabilities,
            "sigmoid",
            original,
        ),
        apply_calibration(
            probabilities,
            "sigmoid",
            restored,
        ),
    )


def test_calibration_report_export(
    calibrators,
    evaluation_results,
    tmp_path,
):

    metrics, reliability, predictions = (
        evaluation_results
    )

    comparisons = compare_calibration_methods(
        metrics
    )

    selected = select_calibration_methods(
        metrics
    )

    output_dir = tmp_path / "models"

    reports_dir = tmp_path / "reports"

    save_calibration_results(
        calibrators,
        metrics,
        reliability,
        comparisons,
        selected,
        predictions,
        {"test_evaluated": False},
        output_dir=output_dir,
        reports_dir=reports_dir,
    )

    assert len(
        list(output_dir.glob("*.pkl"))
    ) == 6

    assert (
        output_dir
        / "calibrated_validation_predictions.csv"
    ).exists()

    assert len(
        list(reports_dir.glob("*.csv"))
    ) == 4

    assert (
        reports_dir
        / "calibration_metadata.json"
    ).exists()