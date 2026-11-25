"""
Tests for stratified bootstrap confidence intervals,
paired model comparisons, and statistical report exports.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    load_training_partitions,
)

from src.bootstrap_evaluation import (
    MODEL_NAMES,
    METRIC_NAMES,
    calculate_file_hash,
    validate_predictions,
    calculate_metrics,
    generate_bootstrap_indices,
    calculate_confidence_interval,
    run_bootstrap,
    summarize_bootstrap,
    compare_models,
    save_bootstrap_reports,
)


@pytest.fixture(scope="module")
def validation_df():
    _, validation, _ = load_training_partitions()

    return validation


@pytest.fixture(scope="module")
def predictions(validation_df):
    """
    Controlled predictions derived from the original
    validation customers for mathematical unit tests.

    These are not actual trained model predictions.
    """

    labels = validation_df["Churn"].to_numpy(
        dtype=int
    )

    rng = np.random.default_rng(42)

    result = pd.DataFrame({
        "customerID": validation_df[
            "customerID"
        ].to_numpy(),

        "actual_churn": labels,
    })

    for model_name in MODEL_NAMES:

        probabilities = np.clip(
            np.where(labels == 1, 0.7, 0.3)
            + rng.normal(0, 0.2, len(labels)),
            0,
            1,
        )

        result[
            f"{model_name}_probability"
        ] = probabilities

        result[
            f"{model_name}_prediction"
        ] = (probabilities >= 0.5).astype(int)

    return result


@pytest.fixture(scope="module")
def bootstrap_output(predictions):
    return run_bootstrap(
        predictions,
        n_bootstraps=30,
        random_state=42,
        verbose=False,
    )


@pytest.fixture(scope="module")
def confidence_intervals(bootstrap_output):
    estimates, distributions = bootstrap_output

    return summarize_bootstrap(
        estimates,
        distributions,
    )


@pytest.fixture(scope="module")
def comparisons(bootstrap_output):
    estimates, distributions = bootstrap_output

    return compare_models(
        estimates,
        distributions,
    )


# Prediction validation tests


def test_valid_predictions(predictions):
    validate_predictions(predictions)


def test_missing_prediction_column(predictions):
    invalid = predictions.drop(
        columns=["actual_churn"]
    )

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_empty_predictions(predictions):
    with pytest.raises(ValueError):

        validate_predictions(
            predictions.iloc[:0]
        )


def test_duplicate_customer_ids(predictions):
    invalid = predictions.copy()

    invalid.loc[1, "customerID"] = (
        invalid.loc[0, "customerID"]
    )

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_missing_customer_id(predictions):
    invalid = predictions.copy()

    invalid.loc[0, "customerID"] = None

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_blank_customer_id(predictions):
    invalid = predictions.copy()

    invalid.loc[0, "customerID"] = " "

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_invalid_actual_labels(predictions):
    invalid = predictions.copy()

    invalid.loc[0, "actual_churn"] = 2

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_single_class_labels(predictions):
    invalid = predictions.copy()

    invalid["actual_churn"] = 0

    with pytest.raises(ValueError):

        validate_predictions(invalid)


@pytest.mark.parametrize(
    "invalid_probability",
    [np.nan, np.inf, -0.1, 1.1],
)
def test_invalid_probabilities(
    predictions,
    invalid_probability,
):
    invalid = predictions.copy()

    invalid.loc[
        0,
        "logistic_regression_probability",
    ] = invalid_probability

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_prediction_threshold_mismatch(predictions):
    invalid = predictions.copy()

    column = "logistic_regression_prediction"

    invalid.loc[0, column] = (
        1 - invalid.loc[0, column]
    )

    with pytest.raises(ValueError):

        validate_predictions(invalid)


def test_invalid_validation_threshold(predictions):
    with pytest.raises(ValueError):

        validate_predictions(
            predictions,
            threshold=1.5,
        )


# Bootstrap sampling tests


def test_bootstrap_iteration_count(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    samples = list(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=10,
        )
    )

    assert len(samples) == 10


def test_bootstrap_sample_size(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    for indices in generate_bootstrap_indices(
        labels,
        n_bootstraps=10,
    ):

        assert len(indices) == len(labels)


def test_bootstrap_class_distribution(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    original_counts = np.bincount(labels)

    for indices in generate_bootstrap_indices(
        labels,
        n_bootstraps=10,
    ):

        sampled_counts = np.bincount(
            labels[indices]
        )

        np.testing.assert_array_equal(
            sampled_counts,
            original_counts,
        )


def test_bootstrap_sampling_with_replacement(
    predictions,
):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    indices = next(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=2,
        )
    )

    assert len(np.unique(indices)) < len(labels)


def test_bootstrap_reproducibility(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    first = list(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=5,
            random_state=42,
        )
    )

    second = list(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=5,
            random_state=42,
        )
    )

    for sample_a, sample_b in zip(first, second):

        np.testing.assert_array_equal(
            sample_a,
            sample_b,
        )


def test_bootstrap_different_random_seeds(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    first = next(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=2,
            random_state=42,
        )
    )

    second = next(
        generate_bootstrap_indices(
            labels,
            n_bootstraps=2,
            random_state=43,
        )
    )

    assert not np.array_equal(first, second)


def test_bootstrap_invalid_iteration_count(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    with pytest.raises(ValueError):

        list(
            generate_bootstrap_indices(
                labels,
                n_bootstraps=1,
            )
        )


# Metric correctness tests


def test_metric_names(predictions):
    metrics = calculate_metrics(
        predictions["actual_churn"],
        predictions[
            "logistic_regression_probability"
        ],
    )

    assert set(metrics) == set(METRIC_NAMES)


def test_perfect_classification(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    probabilities = np.where(
        labels == 1,
        0.9,
        0.1,
    )

    metrics = calculate_metrics(
        labels,
        probabilities,
    )

    for value in metrics.values():

        assert value == pytest.approx(1.0)


def test_metric_probability_length_mismatch(predictions):
    labels = predictions[
        "actual_churn"
    ].to_numpy()

    probabilities = predictions[
        "logistic_regression_probability"
    ].to_numpy()

    with pytest.raises(ValueError):

        calculate_metrics(
            labels,
            probabilities[:-1],
        )


# Confidence interval tests


def test_percentile_confidence_interval():
    values = np.arange(100, dtype=float)

    lower, upper = calculate_confidence_interval(
        values,
        confidence_level=0.95,
    )

    assert lower == pytest.approx(2.475)

    assert upper == pytest.approx(96.525)


def test_constant_confidence_interval():
    values = np.full(100, 0.84)

    lower, upper = calculate_confidence_interval(
        values
    )

    assert lower == pytest.approx(0.84)

    assert upper == pytest.approx(0.84)


def test_invalid_confidence_level():
    with pytest.raises(ValueError):

        calculate_confidence_interval(
            [0.8, 0.9],
            confidence_level=1.5,
        )


def test_nonfinite_confidence_interval_values():
    with pytest.raises(ValueError):

        calculate_confidence_interval(
            [0.8, np.nan, 0.9]
        )


def test_insufficient_confidence_interval_values():
    with pytest.raises(ValueError):

        calculate_confidence_interval(
            [0.8]
        )


# Bootstrap execution tests


def test_bootstrap_output_size(bootstrap_output):
    estimates, distributions = bootstrap_output

    assert len(estimates) == 3

    assert len(distributions) == 90


def test_bootstrap_model_iteration_alignment(
    bootstrap_output,
):
    _, distributions = bootstrap_output

    for _, group in distributions.groupby(
        "iteration"
    ):

        assert len(group) == 3

        assert set(group["model"]) == set(
            MODEL_NAMES
        )


def test_bootstrap_metric_ranges(bootstrap_output):
    _, distributions = bootstrap_output

    for metric in METRIC_NAMES:

        assert distributions[
            metric
        ].between(0, 1).all()


def test_bootstrap_point_estimates(
    predictions,
    bootstrap_output,
):
    estimates, _ = bootstrap_output

    for model_name in MODEL_NAMES:

        expected = calculate_metrics(
            predictions["actual_churn"],
            predictions[
                f"{model_name}_probability"
            ],
        )

        for metric in METRIC_NAMES:

            assert estimates[
                model_name
            ][metric] == pytest.approx(
                expected[metric]
            )


# Summary tests


def test_confidence_interval_count(
    confidence_intervals,
):
    assert len(confidence_intervals) == 18


def test_confidence_interval_bounds(
    confidence_intervals,
):
    assert (
        confidence_intervals["ci_lower"]
        <= confidence_intervals["ci_upper"]
    ).all()

    assert confidence_intervals[
        "ci_lower"
    ].between(0, 1).all()

    assert confidence_intervals[
        "ci_upper"
    ].between(0, 1).all()


def test_bootstrap_standard_errors(
    confidence_intervals,
):
    assert (
        confidence_intervals["standard_error"]
        >= 0
    ).all()


def test_confidence_interval_sample_counts(
    confidence_intervals,
):
    assert (
        confidence_intervals["n_bootstraps"] == 30
    ).all()


# Paired comparison tests


def test_paired_comparison_count(comparisons):
    assert len(comparisons) == 18


def test_paired_difference_calculation(
    bootstrap_output,
    comparisons,
):
    estimates, _ = bootstrap_output

    for _, row in comparisons.iterrows():

        expected = (
            estimates[row["model_a"]][row["metric"]]
            - estimates[row["model_b"]][row["metric"]]
        )

        assert row[
            "point_difference"
        ] == pytest.approx(expected)


def test_paired_zero_exclusion_flag(comparisons):
    expected = (
        (comparisons["ci_lower"] > 0)
        | (comparisons["ci_upper"] < 0)
    )

    np.testing.assert_array_equal(
        comparisons["ci_excludes_zero"],
        expected,
    )


def test_identical_models_have_zero_difference(
    predictions,
):
    identical = predictions.copy()

    probabilities = identical[
        "logistic_regression_probability"
    ].to_numpy()

    for model_name in MODEL_NAMES:

        identical[
            f"{model_name}_probability"
        ] = probabilities

        identical[
            f"{model_name}_prediction"
        ] = (probabilities >= 0.5).astype(int)

    estimates, distributions = run_bootstrap(
        identical,
        n_bootstraps=5,
        verbose=False,
    )

    results = compare_models(
        estimates,
        distributions,
    )

    assert np.allclose(
        results["point_difference"],
        0,
    )

    assert np.allclose(
        results["ci_lower"],
        0,
    )

    assert np.allclose(
        results["ci_upper"],
        0,
    )

    assert not results[
        "ci_excludes_zero"
    ].any()


def test_misaligned_bootstrap_iterations(
    bootstrap_output,
):
    estimates, distributions = bootstrap_output

    invalid = distributions.drop(
        distributions.index[0]
    )

    with pytest.raises(ValueError):

        compare_models(
            estimates,
            invalid,
        )


def test_duplicate_bootstrap_iterations(
    bootstrap_output,
):
    estimates, distributions = bootstrap_output

    invalid = pd.concat(
        [
            distributions,
            distributions.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):

        compare_models(
            estimates,
            invalid,
        )


def test_paired_comparison_order_independence(
    bootstrap_output,
    comparisons,
):
    estimates, distributions = bootstrap_output

    shuffled = distributions.sample(
        frac=1,
        random_state=42,
    )

    actual = compare_models(
        estimates,
        shuffled,
    )

    pd.testing.assert_frame_equal(
        actual,
        comparisons,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


# Report export tests


def test_bootstrap_report_exports(
    bootstrap_output,
    confidence_intervals,
    comparisons,
    tmp_path,
):
    _, distributions = bootstrap_output

    metadata = {
        "n_bootstraps": 30,
        "confidence_level": 0.95,
        "random_state": 42,
    }

    save_bootstrap_reports(
        distributions,
        confidence_intervals,
        comparisons,
        metadata,
        output_dir=tmp_path,
    )

    expected_files = [
        "bootstrap_distributions.csv",
        "bootstrap_confidence_intervals.csv",
        "paired_model_comparisons.csv",
        "bootstrap_metadata.json",
    ]

    for filename in expected_files:

        assert (
            tmp_path / filename
        ).is_file()

    assert len(
        pd.read_csv(
            tmp_path
            / "bootstrap_distributions.csv"
        )
    ) == 90

    assert len(
        pd.read_csv(
            tmp_path
            / "bootstrap_confidence_intervals.csv"
        )
    ) == 18

    assert len(
        pd.read_csv(
            tmp_path
            / "paired_model_comparisons.csv"
        )
    ) == 18

    with (
        tmp_path / "bootstrap_metadata.json"
    ).open(
        encoding="utf-8"
    ) as file:

        saved_metadata = json.load(file)

    assert saved_metadata == metadata


def test_file_hash_reproducibility(tmp_path):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "bootstrap evaluation",
        encoding="utf-8",
    )

    first = calculate_file_hash(file_path)

    second = calculate_file_hash(file_path)

    assert len(first) == 64

    assert first == second


def test_file_hash_detects_changes(tmp_path):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "original",
        encoding="utf-8",
    )

    original_hash = calculate_file_hash(
        file_path
    )

    file_path.write_text(
        "modified",
        encoding="utf-8",
    )

    modified_hash = calculate_file_hash(
        file_path
    )

    assert original_hash != modified_hash