"""
Tests for statistical threshold optimization,
classification metrics and strategy selection.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    SUPPORTED_MODELS,
    load_training_partitions,
)

from src.probability_calibration import (
    CALIBRATION_METHODS,
)

from src.threshold_tuning import (
    DEFAULT_THRESHOLDS,
    validate_thresholds,
    validate_calibrated_predictions,
    calculate_threshold_metrics,
    sweep_probability_thresholds,
    sweep_thresholds,
    evaluate_all_thresholds,
    select_best_threshold,
    select_threshold_strategies,
    select_preferred_calibration_thresholds,
    compare_default_thresholds,
    save_threshold_results,
)


@pytest.fixture(scope="module")
def prediction_data():

    _, validation_df, _ = (
        load_training_partitions()
    )

    labels = validation_df[
        "Churn"
    ].to_numpy()

    original = pd.DataFrame({
        "customerID": validation_df[
            "customerID"
        ].to_numpy(),
        "actual_churn": labels,
    })

    calibrated = original.copy()

    rng = np.random.default_rng(42)

    for model in SUPPORTED_MODELS:

        probabilities = np.clip(
            np.where(labels == 1, 0.7, 0.3)
            + rng.normal(0, 0.2, len(labels)),
            0,
            1,
        )

        original[
            f"{model}_probability"
        ] = probabilities

        original[
            f"{model}_prediction"
        ] = (
            probabilities >= 0.5
        ).astype(int)

        calibrated[
            f"{model}_uncalibrated_probability"
        ] = probabilities

        calibrated[
            f"{model}_sigmoid_probability"
        ] = probabilities ** 1.2

        calibrated[
            f"{model}_isotonic_probability"
        ] = np.round(probabilities, 2)

    return original, calibrated


@pytest.fixture(scope="module")
def threshold_results(prediction_data):

    _, calibrated = prediction_data

    return evaluate_all_thresholds(
        calibrated,
        thresholds=[
            0.01,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.99,
        ],
    )


@pytest.fixture(scope="module")
def selected_thresholds(threshold_results):

    return select_threshold_strategies(
        threshold_results
    )


@pytest.fixture(scope="module")
def selected_methods():

    return pd.DataFrame({
        "model": list(SUPPORTED_MODELS),
        "calibration_method": [
            "sigmoid"
        ] * len(SUPPORTED_MODELS),
    })


@pytest.fixture(scope="module")
def preferred_thresholds(
    selected_thresholds,
    selected_methods,
):

    return select_preferred_calibration_thresholds(
        selected_thresholds,
        selected_methods,
    )


# --------------------------------------------------
# Threshold validation
# --------------------------------------------------


def test_default_threshold_grid():

    assert len(DEFAULT_THRESHOLDS) == 99

    assert DEFAULT_THRESHOLDS.min() == 0.01

    assert DEFAULT_THRESHOLDS.max() == 0.99

    np.testing.assert_allclose(
        np.diff(DEFAULT_THRESHOLDS),
        0.01,
    )


def test_threshold_sorting():

    result = validate_thresholds(
        [0.8, 0.2, 0.5]
    )

    np.testing.assert_array_equal(
        result,
        [0.2, 0.5, 0.8],
    )


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        [[0.2, 0.5]],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.5, np.nan],
        [0.5, np.inf],
        [0.5, 0.5],
    ],
)
def test_invalid_thresholds(thresholds):

    with pytest.raises(ValueError):

        validate_thresholds(thresholds)


# --------------------------------------------------
# Calibrated prediction integrity
# --------------------------------------------------


def test_valid_calibrated_predictions(
    prediction_data,
):

    original, calibrated = prediction_data

    validate_calibrated_predictions(
        calibrated,
        original,
    )


def test_missing_calibrated_column(
    prediction_data,
):

    original, calibrated = prediction_data

    modified = calibrated.drop(
        columns=[
            "logistic_regression_sigmoid_probability"
        ]
    )

    with pytest.raises(ValueError):

        validate_calibrated_predictions(
            modified,
            original,
        )


def test_duplicate_customer(
    prediction_data,
):

    original, calibrated = prediction_data

    modified = calibrated.copy()

    modified.loc[1, "customerID"] = (
        modified.loc[0, "customerID"]
    )

    with pytest.raises(ValueError):

        validate_calibrated_predictions(
            modified,
            original,
        )


def test_customer_membership_mismatch(
    prediction_data,
):

    original, calibrated = prediction_data

    modified = calibrated.copy()

    modified.loc[0, "customerID"] = (
        "INVALID-CUSTOMER"
    )

    with pytest.raises(ValueError):

        validate_calibrated_predictions(
            modified,
            original,
        )


def test_customer_label_mismatch(
    prediction_data,
):

    original, calibrated = prediction_data

    modified = calibrated.copy()

    modified.loc[0, "actual_churn"] = (
        1 - modified.loc[0, "actual_churn"]
    )

    with pytest.raises(ValueError):

        validate_calibrated_predictions(
            modified,
            original,
        )


def test_uncalibrated_probability_mismatch(
    prediction_data,
):

    original, calibrated = prediction_data

    modified = calibrated.copy()

    column = (
        "logistic_regression_uncalibrated_probability"
    )

    modified.loc[0, column] = (
        1.0
        if modified.loc[0, column] < 0.5
        else 0.0
    )

    with pytest.raises(ValueError):

        validate_calibrated_predictions(
            modified,
            original,
        )


def test_prediction_order_independence(
    prediction_data,
):

    original, calibrated = prediction_data

    shuffled = calibrated.sample(
        frac=1,
        random_state=42,
    )

    validate_calibrated_predictions(
        shuffled,
        original,
    )


# --------------------------------------------------
# Mathematical metric correctness
# --------------------------------------------------


def test_perfect_classification():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "specificity",
        "balanced_accuracy",
        "youden_j",
    ]:

        assert metrics[metric] == 1


def test_known_confusion_matrix():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.2, 0.9],
        threshold=0.5,
    )

    assert metrics["true_positives"] == 1

    assert metrics["true_negatives"] == 1

    assert metrics["false_positives"] == 1

    assert metrics["false_negatives"] == 1


def test_known_precision_recall():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.2, 0.9],
        threshold=0.5,
    )

    assert metrics["precision"] == 0.5

    assert metrics["recall"] == 0.5

    assert metrics["f1"] == 0.5

    assert metrics["f2"] == 0.5


def test_known_f2_score():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.8, 0.9],
        threshold=0.5,
    )

    assert metrics["f2"] == pytest.approx(
        10 / 11
    )


def test_youden_j():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.8, 0.9],
        threshold=0.5,
    )

    assert metrics["youden_j"] == (
        pytest.approx(0.5)
    )


def test_threshold_inclusive_boundary():

    metrics = calculate_threshold_metrics(
        [0, 1],
        [0.4, 0.5],
        threshold=0.5,
    )

    assert metrics["true_positives"] == 1

    assert metrics["false_negatives"] == 0


def test_all_customers_targeted():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=0,
    )

    assert metrics["customers_targeted"] == 4

    assert metrics["recall"] == 1

    assert metrics["specificity"] == 0

    assert metrics["targeting_rate"] == 1


def test_no_customers_targeted():

    metrics = calculate_threshold_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        threshold=1,
    )

    assert metrics["customers_targeted"] == 0

    assert metrics["precision"] == 0

    assert metrics["recall"] == 0

    assert metrics["specificity"] == 1


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, np.nan, np.inf],
)
def test_invalid_single_threshold(threshold):

    with pytest.raises(ValueError):

        calculate_threshold_metrics(
            [0, 1],
            [0.2, 0.8],
            threshold=threshold,
        )


# --------------------------------------------------
# Threshold sweep tests
# --------------------------------------------------


def test_sweep_row_count():

    results = sweep_probability_thresholds(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        thresholds=[0.2, 0.5, 0.8],
    )

    assert len(results) == 3


def test_recall_monotonicity(
    threshold_results,
):

    for _, subset in threshold_results.groupby(
        ["model", "calibration_method"]
    ):

        ordered = subset.sort_values(
            "threshold"
        )

        assert np.all(
            np.diff(ordered["recall"])
            <= 1e-12
        )


def test_specificity_monotonicity(
    threshold_results,
):

    for _, subset in threshold_results.groupby(
        ["model", "calibration_method"]
    ):

        ordered = subset.sort_values(
            "threshold"
        )

        assert np.all(
            np.diff(ordered["specificity"])
            >= -1e-12
        )


def test_targeting_monotonicity(
    threshold_results,
):

    for _, subset in threshold_results.groupby(
        ["model", "calibration_method"]
    ):

        ordered = subset.sort_values(
            "threshold"
        )

        assert np.all(
            np.diff(
                ordered["customers_targeted"]
            ) <= 0
        )


def test_confusion_matrix_totals(
    threshold_results,
    prediction_data,
):

    _, calibrated = prediction_data

    totals = threshold_results[
        [
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ]
    ].sum(axis=1)

    assert (
        totals == len(calibrated)
    ).all()


def test_targeting_count_consistency(
    threshold_results,
):

    expected = (
        threshold_results["true_positives"]
        + threshold_results["false_positives"]
    )

    assert np.array_equal(
        expected,
        threshold_results["customers_targeted"],
    )


def test_all_combination_count(
    threshold_results,
):

    expected = (
        len(SUPPORTED_MODELS)
        * len(CALIBRATION_METHODS)
        * 11
    )

    assert len(threshold_results) == expected


def test_model_based_sweep_compatibility():

    class FixedProbabilityModel:

        def predict_proba(self, X):

            return np.array([
                [0.9, 0.1],
                [0.2, 0.8],
            ])

    results = sweep_thresholds(
        FixedProbabilityModel(),
        np.zeros((2, 1)),
        [0, 1],
        thresholds=[0.5],
    )

    assert results.iloc[0]["f1"] == 1


# --------------------------------------------------
# Threshold strategy selection
# --------------------------------------------------


@pytest.mark.parametrize(
    "objective",
    ["f1", "f2", "youden_j"],
)
def test_maximum_objective_selection(
    threshold_results,
    objective,
):

    subset = threshold_results[
        (
            threshold_results["model"]
            == "logistic_regression"
        )
        & (
            threshold_results["calibration_method"]
            == "sigmoid"
        )
    ]

    selected = select_best_threshold(
        subset,
        objective,
    )

    assert selected[objective] == (
        pytest.approx(
            subset[objective].max()
        )
    )


def test_recall_constrained_selection(
    threshold_results,
):

    subset = threshold_results[
        (
            threshold_results["model"]
            == "logistic_regression"
        )
        & (
            threshold_results["calibration_method"]
            == "sigmoid"
        )
    ]

    selected = select_best_threshold(
        subset,
        "precision",
        minimum_recall=0.80,
    )

    feasible = subset[
        subset["recall"] >= 0.80
    ]

    assert selected["recall"] >= 0.80

    assert selected["precision"] == (
        pytest.approx(
            feasible["precision"].max()
        )
    )


def test_infeasible_recall_constraint():

    results = pd.DataFrame({
        "threshold": [0.8, 0.9],
        "precision": [0.9, 1.0],
        "recall": [0.3, 0.2],
    })

    selected = select_best_threshold(
        results,
        "precision",
        minimum_recall=0.9,
    )

    assert selected is None


def test_unsupported_objective(
    threshold_results,
):

    with pytest.raises(ValueError):

        select_best_threshold(
            threshold_results,
            "unsupported",
        )


def test_invalid_recall_target(
    threshold_results,
):

    with pytest.raises(ValueError):

        select_best_threshold(
            threshold_results,
            "precision",
            minimum_recall=1.1,
        )


def test_deterministic_tie_breaking():

    results = pd.DataFrame({
        "threshold": [0.3, 0.4, 0.5],
        "precision": [0.6, 0.6, 0.6],
        "recall": [0.8, 0.8, 0.8],
        "f1": [0.7, 0.7, 0.7],
    })

    selected = select_best_threshold(
        results,
        "f1",
    )

    assert selected["threshold"] == 0.5


def test_strategy_count(
    selected_thresholds,
):

    assert len(selected_thresholds) == 45


def test_preferred_strategy_count(
    preferred_thresholds,
):

    assert len(preferred_thresholds) == 15

    assert set(
        preferred_thresholds[
            "calibration_method"
        ]
    ) == {"sigmoid"}


def test_recall_constraints_satisfied(
    selected_thresholds,
):

    constrained = selected_thresholds[
        selected_thresholds[
            "minimum_recall"
        ].notna()
        & selected_thresholds["feasible"]
    ]

    assert (
        constrained["recall"]
        >= constrained["minimum_recall"]
    ).all()


def test_duplicate_selected_methods(
    selected_thresholds,
    selected_methods,
):

    modified = pd.concat(
        [
            selected_methods,
            selected_methods.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):

        select_preferred_calibration_thresholds(
            selected_thresholds,
            modified,
        )


def test_default_threshold_comparison(
    threshold_results,
    preferred_thresholds,
):

    comparison = compare_default_thresholds(
        threshold_results,
        preferred_thresholds,
    )

    expected_rows = int(
        preferred_thresholds[
            "feasible"
        ].sum()
    )

    assert len(comparison) == expected_rows

    np.testing.assert_allclose(
        comparison["selected_f1"]
        - comparison["default_f1"],
        comparison["f1_change"],
    )

    assert (
        comparison["default_threshold"] == 0.5
    ).all()


def test_threshold_report_export(
    threshold_results,
    selected_thresholds,
    preferred_thresholds,
    tmp_path,
):

    comparison = compare_default_thresholds(
        threshold_results,
        preferred_thresholds,
    )

    save_threshold_results(
        threshold_results,
        selected_thresholds,
        preferred_thresholds,
        comparison,
        {"test_evaluated": False},
        output_dir=tmp_path,
    )

    expected_files = {
        "threshold_sweep.csv",
        "selected_thresholds.csv",
        "preferred_calibration_thresholds.csv",
        "default_threshold_comparison.csv",
        "threshold_metadata.json",
    }

    assert {
        path.name
        for path in tmp_path.iterdir()
    } == expected_files