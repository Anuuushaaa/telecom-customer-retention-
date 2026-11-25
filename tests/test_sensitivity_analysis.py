"""
Tests for financial sensitivity analysis,
scenario evaluation and model stability.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.business_config import (
    BASELINE_CONFIG,
    DATA_PATH,
)

from src.optimization import SUPPORTED_MODELS

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
)

from src.sensitivity_analysis import (
    PARAMETER_SCENARIOS,
    evaluate_financial_scenario,
    run_one_way_sensitivity,
    run_two_way_sensitivity,
    create_named_scenarios,
    run_named_scenarios,
    summarize_parameter_impacts,
    calculate_scenario_winners,
    summarize_model_stability,
    summarize_budget_sensitivity,
    verify_baseline_consistency,
    save_sensitivity_reports,
)


# --------------------------------------------------
# Fixtures
# --------------------------------------------------


@pytest.fixture(scope="module")
def business_data():

    raw = pd.read_csv(DATA_PATH)

    cleaned = prepare_raw_data(
        raw,
        require_target=True,
    )

    _, validation, _ = (
        split_train_validation_test(cleaned)
    )

    data = validation[
        [
            "customerID",
            "MonthlyCharges",
            "Churn",
        ]
    ].copy()

    data = data.rename(
        columns={"Churn": "actual_churn"}
    ).reset_index(drop=True)

    rng = np.random.default_rng(42)

    labels = data["actual_churn"].to_numpy()

    for model in SUPPORTED_MODELS:

        data[f"{model}_sigmoid_probability"] = (
            np.clip(
                np.where(labels == 1, 0.7, 0.3)
                + rng.normal(0, 0.2, len(data)),
                0,
                1,
            )
        )

    return data


@pytest.fixture(scope="module")
def selected_methods():

    return {
        model: "sigmoid"
        for model in SUPPORTED_MODELS
    }


@pytest.fixture(scope="module")
def one_way_results(
    business_data,
    selected_methods,
):

    return run_one_way_sensitivity(
        business_data,
        selected_methods,
    )


@pytest.fixture(scope="module")
def two_way_results(
    business_data,
    selected_methods,
):

    return run_two_way_sensitivity(
        business_data,
        selected_methods,
    )


@pytest.fixture(scope="module")
def named_results(
    business_data,
    selected_methods,
):

    return run_named_scenarios(
        business_data,
        selected_methods,
    )


# --------------------------------------------------
# Scenario configuration
# --------------------------------------------------


def test_parameter_scenario_count():

    assert len(PARAMETER_SCENARIOS) == 6

    assert sum(
        len(values)
        for values in PARAMETER_SCENARIOS.values()
    ) == 29


def test_all_parameters_include_baseline():

    for parameter, values in (
        PARAMETER_SCENARIOS.items()
    ):

        baseline = getattr(
            BASELINE_CONFIG,
            parameter,
        )

        assert baseline in values


def test_parameter_scenarios_unique():

    for values in PARAMETER_SCENARIOS.values():

        assert len(values) == len(set(values))


def test_named_scenario_membership():

    scenarios = create_named_scenarios()

    assert set(scenarios) == {
        "pessimistic",
        "baseline",
        "optimistic",
    }

    assert scenarios["baseline"] == BASELINE_CONFIG


def test_named_scenarios_hold_budget_constant():

    scenarios = create_named_scenarios()

    assert all(
        config.campaign_budget
        == BASELINE_CONFIG.campaign_budget

        for config in scenarios.values()
    )


def test_named_scenario_cost_order():

    scenarios = create_named_scenarios()

    assert (
        scenarios["pessimistic"].intervention_cost
        > scenarios["baseline"].intervention_cost
        > scenarios["optimistic"].intervention_cost
    )


def test_named_scenario_success_order():

    scenarios = create_named_scenarios()

    assert (
        scenarios["pessimistic"].retention_success_rate
        < scenarios["baseline"].retention_success_rate
        < scenarios["optimistic"].retention_success_rate
    )


# --------------------------------------------------
# Financial scenario evaluation
# --------------------------------------------------


def test_scenario_evaluates_all_models(
    business_data,
    selected_methods,
):

    results = evaluate_financial_scenario(
        business_data,
        selected_methods,
    )

    assert len(results) == 6

    assert set(results["model"]) == set(
        SUPPORTED_MODELS
    )

    assert set(results["strategy"]) == {
        "customer_value_targeting",
        "budget_value_targeting",
    }


def test_scenario_budget_compliance(
    business_data,
    selected_methods,
):

    results = evaluate_financial_scenario(
        business_data,
        selected_methods,
    )

    constrained = results[
        results["strategy"]
        == "budget_value_targeting"
    ]

    assert constrained["within_budget"].all()

    assert constrained["customers_targeted"].le(
        400
    ).all()

    assert constrained["campaign_cost"].le(
        10000
    ).all()


def test_scenario_unconstrained_profit_upper_bound(
    business_data,
    selected_methods,
):

    results = evaluate_financial_scenario(
        business_data,
        selected_methods,
    )

    comparison = results.pivot(
        index="model",
        columns="strategy",
        values="expected_net_profit",
    )

    assert (
        comparison["customer_value_targeting"]
        >= comparison["budget_value_targeting"]
        - 1e-8
    ).all()


def test_zero_success_scenario(
    business_data,
    selected_methods,
):

    config = replace(
        BASELINE_CONFIG,
        retention_success_rate=0,
    )

    results = evaluate_financial_scenario(
        business_data,
        selected_methods,
        config,
    )

    assert results["customers_targeted"].eq(0).all()

    assert results["expected_net_profit"].eq(0).all()

    assert results["campaign_cost"].eq(0).all()

    assert results["expected_roi"].isna().all()


def test_zero_budget_scenario(
    business_data,
    selected_methods,
):

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=0,
    )

    results = evaluate_financial_scenario(
        business_data,
        selected_methods,
        config,
    )

    constrained = results[
        results["strategy"]
        == "budget_value_targeting"
    ]

    assert constrained["customers_targeted"].eq(
        0
    ).all()

    assert constrained["expected_net_profit"].eq(
        0
    ).all()


def test_scenario_expected_profit_label_independence(
    business_data,
    selected_methods,
):

    original = evaluate_financial_scenario(
        business_data,
        selected_methods,
    )

    modified = business_data.copy()

    modified["actual_churn"] = (
        1 - modified["actual_churn"]
    )

    changed = evaluate_financial_scenario(
        modified,
        selected_methods,
    )

    np.testing.assert_allclose(
        original["expected_net_profit"],
        changed["expected_net_profit"],
    )

    np.testing.assert_array_equal(
        original["customers_targeted"],
        changed["customers_targeted"],
    )


# --------------------------------------------------
# One-way sensitivity
# --------------------------------------------------


def test_one_way_evaluation_count(one_way_results):

    assert len(one_way_results) == 174

    assert set(one_way_results["parameter"]) == (
        set(PARAMETER_SCENARIOS)
    )


def test_one_way_baseline_count(one_way_results):

    baseline = one_way_results[
        one_way_results["is_baseline"]
    ]

    assert len(baseline) == 36


def test_one_way_parameter_isolation(one_way_results):

    fields = list(PARAMETER_SCENARIOS)

    for _, row in one_way_results.iterrows():

        changed_parameter = row["parameter"]

        for field in fields:

            if field == changed_parameter:

                assert row[field] == pytest.approx(
                    row["parameter_value"]
                )

            else:

                assert row[field] == pytest.approx(
                    getattr(BASELINE_CONFIG, field)
                )


@pytest.mark.parametrize(
    "scenarios",
    [
        {},
        {"unknown_parameter": [1, 2]},
        {"contact_cost": []},
        {"contact_cost": [5, 5]},
        {"contact_cost": [-1]},
    ],
)
def test_invalid_one_way_scenarios(
    business_data,
    selected_methods,
    scenarios,
):

    with pytest.raises(ValueError):

        run_one_way_sensitivity(
            business_data,
            selected_methods,
            parameter_scenarios=scenarios,
        )


def test_custom_one_way_scenarios(
    business_data,
    selected_methods,
):

    result = run_one_way_sensitivity(
        business_data,
        selected_methods,
        parameter_scenarios={
            "contact_cost": [5, 10],
        },
    )

    assert len(result) == 12

    assert result["is_baseline"].sum() == 6


# --------------------------------------------------
# Two-way sensitivity
# --------------------------------------------------


def test_two_way_evaluation_count(two_way_results):

    assert len(two_way_results) == 180

    assert two_way_results[
        "scenario_id"
    ].nunique() == 30


def test_two_way_scenario_membership(
    two_way_results,
):

    counts = two_way_results.groupby(
        "scenario_id"
    ).size()

    assert counts.eq(6).all()


def test_two_way_parameter_isolation(
    two_way_results,
):

    for field in [
        "contact_cost",
        "retention_months",
        "contribution_margin",
        "campaign_budget",
    ]:

        assert np.allclose(
            two_way_results[field],
            getattr(BASELINE_CONFIG, field),
        )


@pytest.mark.parametrize(
    "costs,rates",
    [
        ([], [0.3]),
        ([20], []),
        ([20, 20], [0.3]),
        ([20], [0.3, 0.3]),
        ([-1], [0.3]),
        ([20], [1.1]),
    ],
)
def test_invalid_two_way_scenarios(
    business_data,
    selected_methods,
    costs,
    rates,
):

    with pytest.raises(ValueError):

        run_two_way_sensitivity(
            business_data,
            selected_methods,
            incentive_costs=costs,
            success_rates=rates,
        )


# --------------------------------------------------
# Named financial scenarios
# --------------------------------------------------


def test_named_evaluation_count(named_results):

    assert len(named_results) == 18

    assert named_results.groupby(
        "scenario"
    ).size().eq(6).all()


def test_named_scenario_profit_order(named_results):

    comparison = named_results.pivot(
        index=["model", "strategy"],
        columns="scenario",
        values="expected_net_profit",
    )

    assert (
        comparison["optimistic"]
        >= comparison["baseline"] - 1e-8
    ).all()

    assert (
        comparison["baseline"]
        >= comparison["pessimistic"] - 1e-8
    ).all()


def test_pessimistic_no_intervention(named_results):

    pessimistic = named_results[
        named_results["scenario"]
        == "pessimistic"
    ]

    assert pessimistic["customers_targeted"].eq(
        0
    ).all()

    assert pessimistic["expected_net_profit"].eq(
        0
    ).all()


# --------------------------------------------------
# Baseline consistency
# --------------------------------------------------


def test_baseline_consistency(
    one_way_results,
    named_results,
):

    assert verify_baseline_consistency(
        one_way_results,
        named_results,
    ) is True


def test_baseline_inconsistency_detected(
    one_way_results,
    named_results,
):

    modified = named_results.copy()

    modified.loc[
        modified["scenario"] == "baseline",
        "expected_net_profit",
    ] += 100

    with pytest.raises(ValueError):

        verify_baseline_consistency(
            one_way_results,
            modified,
        )


def test_missing_named_baseline_rejected(
    one_way_results,
    named_results,
):

    incomplete = named_results[
        named_results["scenario"] != "baseline"
    ]

    with pytest.raises(ValueError):

        verify_baseline_consistency(
            one_way_results,
            incomplete,
        )


def test_missing_one_way_baseline_rejected(
    one_way_results,
    named_results,
):

    modified = one_way_results.copy()

    modified["is_baseline"] = False

    with pytest.raises(ValueError):

        verify_baseline_consistency(
            modified,
            named_results,
        )


# --------------------------------------------------
# Parameter impact summary
# --------------------------------------------------


def test_parameter_impact_summary_count(
    one_way_results,
):

    summary = summarize_parameter_impacts(
        one_way_results
    )

    assert len(summary) == 36


def test_parameter_impact_profit_ranges(
    one_way_results,
):

    summary = summarize_parameter_impacts(
        one_way_results
    )

    assert summary["profit_range"].ge(0).all()

    assert summary["downside_from_baseline"].le(
        1e-8
    ).all()

    assert summary["upside_from_baseline"].ge(
        -1e-8
    ).all()

    np.testing.assert_allclose(
        summary["profit_range"],
        summary["maximum_expected_profit"]
        - summary["minimum_expected_profit"],
    )


def test_parameter_impact_missing_baseline(
    one_way_results,
):

    modified = one_way_results.copy()

    modified["is_baseline"] = False

    with pytest.raises(ValueError):

        summarize_parameter_impacts(modified)


# --------------------------------------------------
# Model stability and scenario winners
# --------------------------------------------------


def test_one_way_winner_count(one_way_results):

    winners = calculate_scenario_winners(
        one_way_results,
        ["parameter", "parameter_value"],
    )

    assert len(winners) == 58


def test_two_way_winner_count(two_way_results):

    winners = calculate_scenario_winners(
        two_way_results,
        [
            "incentive_cost",
            "retention_success_rate",
        ],
    )

    assert len(winners) == 60


def test_named_winner_count(named_results):

    winners = calculate_scenario_winners(
        named_results,
        ["scenario"],
    )

    assert len(winners) == 6


def test_scenario_winner_uses_expected_profit():

    results = pd.DataFrame({
        "scenario": ["test"] * 3,
        "strategy": ["budget_value_targeting"] * 3,
        "model": list(SUPPORTED_MODELS),
        "expected_net_profit": [100, 200, 50],
        "simulated_net_profit": [500, 50, 300],
    })

    result = calculate_scenario_winners(
        results,
        ["scenario"],
    ).iloc[0]

    assert result["winning_model"] == "xgboost"

    assert result["profit_advantage"] == 100

    assert result["winning_model_count"] == 1


def test_scenario_tie_handling():

    results = pd.DataFrame({
        "scenario": ["test"] * 3,
        "strategy": ["budget_value_targeting"] * 3,
        "model": list(SUPPORTED_MODELS),
        "expected_net_profit": [100, 100, 50],
    })

    result = calculate_scenario_winners(
        results,
        ["scenario"],
    ).iloc[0]

    assert result["is_tie"]

    assert result["winning_model_count"] == 2

    assert set(
        result["winning_model"].split("|")
    ) == {
        "logistic_regression",
        "xgboost",
    }


def test_fractional_model_wins():

    results = pd.DataFrame({
        "scenario": ["test"] * 3,
        "strategy": ["budget_value_targeting"] * 3,
        "model": list(SUPPORTED_MODELS),
        "expected_net_profit": [100, 100, 50],
    })

    winners = calculate_scenario_winners(
        results,
        ["scenario"],
    )

    summary = summarize_model_stability(
        winners
    ).set_index("model")

    assert summary.loc[
        "logistic_regression",
        "fractional_wins",
    ] == 0.5

    assert summary.loc[
        "xgboost",
        "fractional_wins",
    ] == 0.5

    assert summary.loc[
        "tensorflow_ann",
        "fractional_wins",
    ] == 0


def test_no_intervention_scenario_winner():

    results = pd.DataFrame({
        "scenario": ["test"] * 3,
        "strategy": ["budget_value_targeting"] * 3,
        "model": list(SUPPORTED_MODELS),
        "expected_net_profit": [0, 0, 0],
    })

    winners = calculate_scenario_winners(
        results,
        ["scenario"],
    )

    assert winners.iloc[0]["winning_model"] == (
        "no_intervention"
    )

    assert winners.iloc[0][
        "no_intervention_preferred"
    ]

    summary = summarize_model_stability(
        winners
    )

    assert summary["win_rate"].eq(0).all()

    assert summary[
        "no_intervention_scenarios"
    ].eq(1).all()


def test_missing_model_rejected():

    results = pd.DataFrame({
        "scenario": ["test"] * 2,
        "strategy": ["budget_value_targeting"] * 2,
        "model": ["logistic_regression", "xgboost"],
        "expected_net_profit": [100, 200],
    })

    with pytest.raises(ValueError):

        calculate_scenario_winners(
            results,
            ["scenario"],
        )


def test_duplicate_model_rejected():

    results = pd.DataFrame({
        "scenario": ["test"] * 3,
        "strategy": ["budget_value_targeting"] * 3,
        "model": [
            "xgboost",
            "xgboost",
            "tensorflow_ann",
        ],
        "expected_net_profit": [100, 200, 50],
    })

    with pytest.raises(ValueError):

        calculate_scenario_winners(
            results,
            ["scenario"],
        )


def test_model_stability_rates_sum_correctly(
    one_way_results,
):

    winners = calculate_scenario_winners(
        one_way_results,
        ["parameter", "parameter_value"],
    )

    summary = summarize_model_stability(
        winners
    )

    for _, subset in summary.groupby("strategy"):

        no_intervention_rate = (
            subset["no_intervention_scenarios"].iloc[0]
            / subset["total_scenarios"].iloc[0]
        )

        assert (
            subset["win_rate"].sum()
            + no_intervention_rate
        ) == pytest.approx(1)


# --------------------------------------------------
# Budget sensitivity
# --------------------------------------------------


def test_budget_summary_count(one_way_results):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    assert len(result) == 15


def test_budget_summary_compliance(one_way_results):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    assert result["unused_budget"].ge(
        -1e-8
    ).all()

    assert (
        result["campaign_cost"]
        <= result["campaign_budget"] + 1e-8
    ).all()


def test_budget_profit_monotonicity(one_way_results):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    for _, subset in result.groupby("model"):

        subset = subset.sort_values(
            "campaign_budget"
        )

        assert np.all(
            np.diff(subset["expected_net_profit"])
            >= -1e-8
        )


def test_budget_target_count_monotonicity(
    one_way_results,
):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    for _, subset in result.groupby("model"):

        subset = subset.sort_values(
            "campaign_budget"
        )

        assert np.all(
            np.diff(subset["customers_targeted"])
            >= 0
        )


def test_marginal_return_calculation(
    one_way_results,
):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    positive_spend = result[
        result["additional_campaign_cost"] > 0
    ]

    assert not positive_spend.empty

    np.testing.assert_allclose(
        positive_spend["marginal_return_on_spend"],
        positive_spend["additional_expected_profit"]
        / positive_spend["additional_campaign_cost"],
    )


def test_marginal_return_undefined_without_spend(
    one_way_results,
):

    result = summarize_budget_sensitivity(
        one_way_results
    )

    undefined = result[
        result["additional_campaign_cost"].isna()
        | result["additional_campaign_cost"].eq(0)
    ]

    assert undefined[
        "marginal_return_on_spend"
    ].isna().all()


def test_missing_budget_results_rejected(
    one_way_results,
):

    incomplete = one_way_results[
        one_way_results["parameter"]
        != "campaign_budget"
    ]

    with pytest.raises(ValueError):

        summarize_budget_sensitivity(
            incomplete
        )


# --------------------------------------------------
# Financial sensitivity properties
# --------------------------------------------------


@pytest.mark.parametrize(
    "parameter",
    [
        "retention_success_rate",
        "retention_months",
        "contribution_margin",
    ],
)
def test_higher_retention_value_increases_profit(
    one_way_results,
    parameter,
):

    subset = one_way_results[
        one_way_results["parameter"] == parameter
    ]

    for _, group in subset.groupby(
        ["model", "strategy"]
    ):

        group = group.sort_values(
            "parameter_value"
        )

        assert np.all(
            np.diff(group["expected_net_profit"])
            >= -1e-8
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "contact_cost",
        "incentive_cost",
    ],
)
def test_higher_cost_reduces_optimal_profit(
    one_way_results,
    parameter,
):

    subset = one_way_results[
        one_way_results["parameter"] == parameter
    ]

    for _, group in subset.groupby(
        ["model", "strategy"]
    ):

        group = group.sort_values(
            "parameter_value"
        )

        assert np.all(
            np.diff(group["expected_net_profit"])
            <= 1e-8
        )


# --------------------------------------------------
# Report export
# --------------------------------------------------


def test_sensitivity_report_export(
    tmp_path,
    one_way_results,
    two_way_results,
    named_results,
):

    one_way_winners = calculate_scenario_winners(
        one_way_results,
        ["parameter", "parameter_value"],
    )

    two_way_winners = calculate_scenario_winners(
        two_way_results,
        [
            "incentive_cost",
            "retention_success_rate",
        ],
    )

    named_winners = calculate_scenario_winners(
        named_results,
        ["scenario"],
    )

    stability = pd.concat(
        [
            summarize_model_stability(
                one_way_winners
            ).assign(scenario_family="one_way"),

            summarize_model_stability(
                two_way_winners
            ).assign(scenario_family="two_way"),

            summarize_model_stability(
                named_winners
            ).assign(scenario_family="named"),
        ],
        ignore_index=True,
    )

    reports = {
        "one_way_sensitivity.csv":
            one_way_results,

        "two_way_sensitivity.csv":
            two_way_results,

        "named_scenario_analysis.csv":
            named_results,

        "parameter_impact_summary.csv":
            summarize_parameter_impacts(
                one_way_results
            ),

        "budget_sensitivity_summary.csv":
            summarize_budget_sensitivity(
                one_way_results
            ),

        "one_way_model_winners.csv":
            one_way_winners,

        "two_way_model_winners.csv":
            two_way_winners,

        "named_scenario_winners.csv":
            named_winners,

        "model_stability_summary.csv":
            stability,
    }

    metadata = {
        "total_evaluations": 372,
        "test_evaluated": False,
        "production_updated": False,
        "financial_assumptions":
            "hypothetical_simulation",
    }

    save_sensitivity_reports(
        reports,
        metadata,
        output_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*.csv"))) == 9

    assert len(list(tmp_path.glob("*.json"))) == 1

    saved = pd.read_csv(
        tmp_path / "one_way_sensitivity.csv"
    )

    assert len(saved) == 174

    with open(
        tmp_path / "sensitivity_metadata.json",
        encoding="utf-8",
    ) as file:

        assert json.load(file) == metadata