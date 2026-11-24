
"""Tests for statistical inference calculations."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from src.statistical_inference import (
    fit_logistic_model,
    calculate_odds_ratios,
    perform_wald_test,
    perform_likelihood_ratio_test,
)


@pytest.fixture(scope="module")
def fitted_models():
    """Generate reproducible synthetic logistic regression data."""

    rng = np.random.default_rng(42)

    n = 1000

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    logits = -0.5 + 0.8 * x1 - 0.6 * x2

    probabilities = 1 / (1 + np.exp(-logits))

    y = pd.Series(rng.binomial(1, probabilities))

    X = pd.DataFrame({
        "const": np.ones(n),
        "x1": x1,
        "x2": x2,
    })

    full_model = fit_logistic_model(X, y)

    reduced_model = fit_logistic_model(
        X[["const", "x1"]],
        y,
    )

    return full_model, reduced_model


def test_logistic_model_converges(fitted_models):

    full_model, _ = fitted_models

    assert full_model.mle_retvals["converged"]


def test_odds_ratios_equal_exp_coefficients(fitted_models):

    full_model, _ = fitted_models

    results = calculate_odds_ratios(full_model)

    expected = np.exp(
        full_model.params.drop("const").values
    )

    np.testing.assert_allclose(
        results["odds_ratio"],
        expected,
    )


def test_odds_ratio_confidence_intervals(fitted_models):

    full_model, _ = fitted_models

    results = calculate_odds_ratios(full_model)

    expected = np.exp(
        full_model.conf_int().drop(index="const")
    )

    np.testing.assert_allclose(
        results["ci_lower_95"],
        expected.iloc[:, 0],
    )

    np.testing.assert_allclose(
        results["ci_upper_95"],
        expected.iloc[:, 1],
    )

    assert (
        results["ci_lower_95"] < results["odds_ratio"]
    ).all()

    assert (
        results["odds_ratio"] < results["ci_upper_95"]
    ).all()


def test_wald_statistics(fitted_models):

    full_model, _ = fitted_models

    results = perform_wald_test(full_model)

    expected = (
        full_model.params / full_model.bse
    ).drop("const") ** 2

    np.testing.assert_allclose(
        results["wald_statistic"],
        expected.values,
    )


def test_wald_p_values(fitted_models):

    full_model, _ = fitted_models

    results = perform_wald_test(full_model)

    expected = full_model.pvalues.drop("const")

    np.testing.assert_allclose(
        results["p_value"],
        expected.values,
    )


def test_likelihood_ratio_statistic(fitted_models):

    full_model, reduced_model = fitted_models

    result = perform_likelihood_ratio_test(
        full_model,
        reduced_model,
        comparison="Synthetic model comparison",
    )

    expected = 2 * (
        full_model.llf - reduced_model.llf
    )

    assert result["lr_statistic"] == pytest.approx(
        expected
    )

    assert result["lr_statistic"] >= 0


def test_likelihood_ratio_degrees_of_freedom(fitted_models):

    full_model, reduced_model = fitted_models

    result = perform_likelihood_ratio_test(
        full_model,
        reduced_model,
        comparison="Synthetic model comparison",
    )

    assert result["degrees_of_freedom"] == 1


def test_likelihood_ratio_p_value(fitted_models):

    full_model, reduced_model = fitted_models

    result = perform_likelihood_ratio_test(
        full_model,
        reduced_model,
        comparison="Synthetic model comparison",
    )

    expected = chi2.sf(
        result["lr_statistic"],
        result["degrees_of_freedom"],
    )

    assert result["p_value"] == pytest.approx(expected)


def test_likelihood_ratio_rejects_identical_models(fitted_models):

    full_model, _ = fitted_models

    with pytest.raises(ValueError):

        perform_likelihood_ratio_test(
            full_model,
            full_model,
            comparison="Invalid comparison",
        )


def test_rank_deficient_model_rejected():

    X = pd.DataFrame({
        "const": [1.0, 1.0, 1.0, 1.0],
        "x1": [0.0, 1.0, 2.0, 3.0],
        "x2": [0.0, 2.0, 4.0, 6.0],
    })

    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError, match="rank deficient"):

        fit_logistic_model(X, y)
