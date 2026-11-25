"""
Tests for business configuration, financial valuation,
campaign simulation and business-optimal targeting.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.business_config import (
    BASELINE_CONFIG,
    BusinessConfig,
    DATA_PATH,
)

from src.business_cost_analysis import (
    validate_monthly_charges,
    load_business_data,
    load_selected_methods,
    calculate_retained_values,
    calculate_expected_net_benefits,
    calculate_break_even_probabilities,
    select_threshold_targets,
    select_profitable_customers,
    select_budget_constrained_customers,
    evaluate_campaign,
    sweep_business_thresholds,
    select_profit_optimal_threshold,
    evaluate_model_strategies,
    summarize_customer_values,
    summarize_break_even_probabilities,
    select_best_business_strategies,
    save_business_reports,
)

from src.optimization import SUPPORTED_MODELS

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
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


@pytest.fixture
def financial_inputs():

    return {
        "labels": np.array([1, 0, 1, 0]),

        "probabilities": np.array(
            [0.8, 0.6, 0.3, 0.1]
        ),

        "charges": np.array(
            [100.0, 50.0, 80.0, 20.0]
        ),

        "customer_ids": np.array(
            ["A", "B", "C", "D"]
        ),

        "targets": np.array(
            [True, True, False, False]
        ),
    }


@pytest.fixture
def statistical_thresholds():

    strategies = {
        "maximum_f1": 0.40,
        "maximum_f2": 0.20,
        "maximum_youden_j": 0.30,
        "recall_at_least_80": 0.25,
        "recall_at_least_90": 0.15,
    }

    return pd.DataFrame([
        {
            "model": model,
            "calibration_method": "sigmoid",
            "strategy": strategy,
            "threshold": threshold,
            "feasible": True,
        }

        for model in SUPPORTED_MODELS

        for strategy, threshold
        in strategies.items()
    ])


@pytest.fixture(scope="module")
def model_results(business_data):

    campaigns = []
    thresholds = []

    statistical = pd.DataFrame([
        {
            "model": model,
            "calibration_method": "sigmoid",
            "strategy": strategy,
            "threshold": threshold,
            "feasible": True,
        }

        for model in SUPPORTED_MODELS

        for strategy, threshold in [
            ("maximum_f1", 0.40),
            ("maximum_f2", 0.20),
            ("maximum_youden_j", 0.30),
            ("recall_at_least_80", 0.25),
            ("recall_at_least_90", 0.15),
        ]
    ])

    for model in SUPPORTED_MODELS:

        campaign, threshold = (
            evaluate_model_strategies(
                business_data,
                model,
                "sigmoid",
                statistical,
            )
        )

        campaigns.append(campaign)

        thresholds.append(threshold)

    return (
        pd.concat(campaigns, ignore_index=True),
        pd.concat(thresholds, ignore_index=True),
    )


# --------------------------------------------------
# Business configuration
# --------------------------------------------------


def test_baseline_configuration():

    config = BASELINE_CONFIG

    assert config.intervention_cost == 25.0

    assert config.maximum_budget_targets == 400

    assert config.retained_value_multiplier == (
        pytest.approx(4.8)
    )

    assert config.currency == "USD"


@pytest.mark.parametrize(
    "field,value",
    [
        ("contact_cost", -1),
        ("incentive_cost", -1),
        ("contact_cost", np.nan),
        ("incentive_cost", np.inf),
        ("contact_cost", True),
        ("incentive_cost", "20"),
        ("retention_success_rate", -0.1),
        ("retention_success_rate", 1.1),
        ("retention_success_rate", np.nan),
        ("contribution_margin", -0.1),
        ("contribution_margin", 1.1),
        ("contribution_margin", np.inf),
        ("campaign_budget", -100),
        ("campaign_budget", np.inf),
        ("retention_months", 0),
        ("retention_months", -1),
        ("retention_months", 2.5),
        ("retention_months", True),
        ("currency", "EUR"),
    ],
)
def test_invalid_configuration(field, value):

    with pytest.raises(ValueError):

        replace(
            BASELINE_CONFIG,
            **{field: value},
        )


def test_zero_intervention_cost_rejected():

    with pytest.raises(ValueError):

        BusinessConfig(
            contact_cost=0,
            incentive_cost=0,
        )


def test_zero_budget_allowed():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=0,
    )

    assert config.maximum_budget_targets == 0


def test_budget_capacity_rounds_down():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=99,
    )

    assert config.maximum_budget_targets == 3


def test_configuration_serialization():

    result = BASELINE_CONFIG.to_dict()

    assert result["intervention_cost"] == 25

    assert result["maximum_budget_targets"] == 400

    assert result["assumption_type"] == (
        "hypothetical_simulation"
    )

    json.dumps(result, allow_nan=False)


# --------------------------------------------------
# Financial input validation
# --------------------------------------------------


@pytest.mark.parametrize(
    "charges",
    [
        [],
        [[10, 20]],
        [-10, 20],
        [np.nan, 20],
        [np.inf, 20],
        [-np.inf, 20],
        ["invalid", 20],
    ],
)
def test_invalid_monthly_charges(charges):

    with pytest.raises(ValueError):

        validate_monthly_charges(charges)


def test_valid_monthly_charges():

    result = validate_monthly_charges(
        [0, 25, 100]
    )

    np.testing.assert_allclose(
        result,
        [0, 25, 100],
    )


# --------------------------------------------------
# Customer financial valuation
# --------------------------------------------------


def test_retained_value_calculation():

    result = calculate_retained_values(
        [50, 100, 0]
    )

    np.testing.assert_allclose(
        result,
        [240, 480, 0],
    )


def test_zero_margin_retained_value():

    config = replace(
        BASELINE_CONFIG,
        contribution_margin=0,
    )

    result = calculate_retained_values(
        [50, 100],
        config,
    )

    np.testing.assert_array_equal(
        result,
        [0, 0],
    )


def test_expected_net_benefit():

    result = calculate_expected_net_benefits(
        [0.8, 0.2],
        [100, 50],
    )

    np.testing.assert_allclose(
        result,
        [90.2, -10.6],
    )


def test_expected_benefit_length_mismatch():

    with pytest.raises(ValueError):

        calculate_expected_net_benefits(
            [0.5, 0.8],
            [100],
        )


def test_zero_success_expected_benefit():

    config = replace(
        BASELINE_CONFIG,
        retention_success_rate=0,
    )

    result = calculate_expected_net_benefits(
        [0.9, 0.8],
        [100, 50],
        config,
    )

    np.testing.assert_allclose(
        result,
        [-25, -25],
    )


def test_break_even_probability():

    result = calculate_break_even_probabilities(
        [100, 50]
    )

    np.testing.assert_allclose(
        result,
        [
            25 / 144,
            25 / 72,
        ],
    )


def test_break_even_zero_customer_value():

    result = calculate_break_even_probabilities(
        [0]
    )

    assert np.isinf(result[0])


def test_break_even_zero_success_rate():

    config = replace(
        BASELINE_CONFIG,
        retention_success_rate=0,
    )

    result = calculate_break_even_probabilities(
        [50, 100],
        config,
    )

    assert np.isinf(result).all()


def test_unachievable_break_even_probability():

    result = calculate_break_even_probabilities(
        [1]
    )

    assert result[0] > 1


def test_break_even_benefit_is_zero():

    charges = np.array([25, 50, 100])

    probabilities = (
        calculate_break_even_probabilities(
            charges
        )
    )

    benefits = calculate_expected_net_benefits(
        probabilities,
        charges,
    )

    np.testing.assert_allclose(
        benefits,
        0,
        atol=1e-10,
    )


# --------------------------------------------------
# Customer targeting
# --------------------------------------------------


def test_threshold_targets_inclusive():

    result = select_threshold_targets(
        [0.2, 0.5, 0.8],
        0.5,
    )

    np.testing.assert_array_equal(
        result,
        [False, True, True],
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, np.nan, np.inf],
)
def test_invalid_targeting_threshold(threshold):

    with pytest.raises(ValueError):

        select_threshold_targets(
            [0.2, 0.8],
            threshold,
        )


def test_profitable_customer_selection():

    result = select_profitable_customers(
        [0.8, 0.2],
        [100, 50],
    )

    np.testing.assert_array_equal(
        result,
        [True, False],
    )


def test_no_profitable_customers():

    result = select_profitable_customers(
        [0, 0],
        [100, 50],
    )

    assert not result.any()


def test_budget_targets_highest_benefit():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=25,
    )

    result = select_budget_constrained_customers(
        [0.9, 0.7, 0.2],
        [30, 100, 20],
        ["A", "B", "C"],
        config,
    )

    np.testing.assert_array_equal(
        result,
        [False, True, False],
    )


def test_budget_tie_breaking():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=25,
    )

    result = select_budget_constrained_customers(
        [0.8, 0.8],
        [100, 100],
        ["B", "A"],
        config,
    )

    np.testing.assert_array_equal(
        result,
        [False, True],
    )


def test_budget_does_not_force_unprofitable_targets():

    result = select_budget_constrained_customers(
        [0.8, 0.01],
        [100, 50],
        ["A", "B"],
    )

    np.testing.assert_array_equal(
        result,
        [True, False],
    )


def test_zero_budget_selects_no_customers():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=0,
    )

    result = select_budget_constrained_customers(
        [0.9, 0.8],
        [100, 100],
        ["A", "B"],
        config,
    )

    assert not result.any()


def test_insufficient_budget_selects_no_customers():

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=24,
    )

    result = select_budget_constrained_customers(
        [0.9, 0.8],
        [100, 100],
        ["A", "B"],
        config,
    )

    assert not result.any()


def test_duplicate_customer_ids_rejected():

    with pytest.raises(ValueError):

        select_budget_constrained_customers(
            [0.8, 0.9],
            [100, 100],
            ["A", "A"],
        )


def test_customer_id_length_mismatch():

    with pytest.raises(ValueError):

        select_budget_constrained_customers(
            [0.8, 0.9],
            [100, 100],
            ["A"],
        )


# --------------------------------------------------
# Campaign financial evaluation
# --------------------------------------------------


def test_campaign_confusion_counts(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    assert result["true_positives"] == 1

    assert result["false_positives"] == 1

    assert result["false_negatives"] == 1

    assert result["true_negatives"] == 1

    assert result["precision"] == 0.5

    assert result["recall"] == 0.5


def test_campaign_financial_calculations(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    assert result["campaign_cost"] == 50

    assert result["expected_churners_targeted"] == (
        pytest.approx(1.4)
    )

    assert result["expected_retained_customers"] == (
        pytest.approx(0.42)
    )

    assert result["simulated_retained_customers"] == (
        pytest.approx(0.3)
    )

    assert result["expected_retained_contribution"] == (
        pytest.approx(158.4)
    )

    assert result["simulated_retained_contribution"] == (
        pytest.approx(144)
    )

    assert result["expected_net_profit"] == (
        pytest.approx(108.4)
    )

    assert result["simulated_net_profit"] == (
        pytest.approx(94)
    )

    assert result["expected_roi"] == (
        pytest.approx(2.168)
    )

    assert result["simulated_roi"] == (
        pytest.approx(1.88)
    )


def test_campaign_business_cost_identity(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    assert result["baseline_churn_loss"] == (
        pytest.approx(864)
    )

    assert result["simulated_remaining_churn_loss"] == (
        pytest.approx(720)
    )

    assert result["simulated_total_business_cost"] == (
        pytest.approx(770)
    )

    assert result["simulated_net_profit"] == (
        pytest.approx(
            result["baseline_churn_loss"]
            - result["simulated_total_business_cost"]
        )
    )


def test_campaign_break_even_success_rate(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    assert result["expected_break_even_success_rate"] == (
        pytest.approx(50 / 528)
    )

    assert result["simulated_break_even_success_rate"] == (
        pytest.approx(50 / 480)
    )


def test_no_intervention_campaign(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        np.zeros(4, dtype=bool),
    )

    assert result["campaign_cost"] == 0

    assert result["expected_net_profit"] == 0

    assert result["simulated_net_profit"] == 0

    assert result["recall"] == 0

    assert result["precision"] == 0

    assert np.isnan(result["expected_roi"])

    assert np.isnan(result["simulated_roi"])

    assert np.isnan(
        result["expected_break_even_success_rate"]
    )


def test_target_everyone_campaign(
    financial_inputs,
):

    data = financial_inputs

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        np.ones(4, dtype=bool),
    )

    assert result["customers_targeted"] == 4

    assert result["campaign_cost"] == 100

    assert result["recall"] == 1

    assert result["false_negatives"] == 0

    assert result["true_negatives"] == 0


def test_campaign_budget_violation(
    financial_inputs,
):

    data = financial_inputs

    config = replace(
        BASELINE_CONFIG,
        campaign_budget=25,
    )

    result = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
        config,
    )

    assert result["within_budget"] is False


@pytest.mark.parametrize(
    "targets",
    [
        [True],
        [1, 0, 2, 0],
        [[1, 0], [1, 0]],
        [1, 0, np.nan, 0],
    ],
)
def test_invalid_campaign_targets(
    financial_inputs,
    targets,
):

    data = financial_inputs

    with pytest.raises(ValueError):

        evaluate_campaign(
            data["labels"],
            data["probabilities"],
            data["charges"],
            targets,
        )


def test_expected_profit_independent_of_labels(
    financial_inputs,
):

    data = financial_inputs

    first = evaluate_campaign(
        data["labels"],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    second = evaluate_campaign(
        [0, 1, 0, 1],
        data["probabilities"],
        data["charges"],
        data["targets"],
    )

    assert first["expected_net_profit"] == (
        pytest.approx(second["expected_net_profit"])
    )

    assert first["simulated_net_profit"] != (
        second["simulated_net_profit"]
    )


# --------------------------------------------------
# Financial threshold optimization
# --------------------------------------------------


def test_business_threshold_sweep(
    financial_inputs,
):

    data = financial_inputs

    results = sweep_business_thresholds(
        data["labels"],
        data["probabilities"],
        data["charges"],
        thresholds=[0.2, 0.5, 0.9],
    )

    assert len(results) == 3

    assert results["threshold"].tolist() == (
        [0.2, 0.5, 0.9]
    )

    assert results["customers_targeted"].tolist() == (
        [3, 2, 0]
    )


def test_profit_threshold_uses_expected_profit():

    results = pd.DataFrame({
        "threshold": [0.2, 0.5],
        "expected_net_profit": [100, 200],
        "simulated_net_profit": [500, 50],
        "campaign_cost": [100, 50],
        "within_budget": [True, True],
    })

    selected = select_profit_optimal_threshold(
        results
    )

    assert selected["threshold"] == 0.5


def test_budget_constrained_threshold_selection():

    results = pd.DataFrame({
        "threshold": [0.2, 0.5],
        "expected_net_profit": [200, 100],
        "campaign_cost": [100, 50],
        "within_budget": [False, True],
    })

    selected = select_profit_optimal_threshold(
        results,
        budget_constrained=True,
    )

    assert selected["threshold"] == 0.5


def test_threshold_tie_breaking():

    results = pd.DataFrame({
        "threshold": [0.2, 0.4, 0.5],
        "expected_net_profit": [100, 100, 100],
        "campaign_cost": [100, 50, 50],
        "within_budget": [True, True, True],
    })

    selected = select_profit_optimal_threshold(
        results
    )

    assert selected["threshold"] == 0.5


def test_unprofitable_threshold_rejected():

    results = pd.DataFrame({
        "threshold": [0.2, 0.5],
        "expected_net_profit": [-100, 0],
        "campaign_cost": [100, 0],
        "within_budget": [True, True],
    })

    assert select_profit_optimal_threshold(
        results
    ) is None


def test_no_feasible_budget_threshold():

    results = pd.DataFrame({
        "threshold": [0.2],
        "expected_net_profit": [100],
        "campaign_cost": [100],
        "within_budget": [False],
    })

    assert select_profit_optimal_threshold(
        results,
        budget_constrained=True,
    ) is None


# --------------------------------------------------
# Original dataset integration
# --------------------------------------------------


def test_original_validation_data(business_data):

    assert len(business_data) == 1409

    assert business_data["customerID"].is_unique

    assert set(business_data["actual_churn"]) == (
        {0, 1}
    )

    assert business_data["MonthlyCharges"].ge(0).all()


def test_business_data_financial_alignment(
    business_data,
    monkeypatch,
):

    predictions = business_data.drop(
        columns=["MonthlyCharges"]
    )

    metadata = {
        "test_evaluated": False,
    }

    monkeypatch.setattr(
        "src.business_cost_analysis."
        "load_calibrated_predictions",
        lambda: (predictions, metadata),
    )

    loaded, returned_metadata = load_business_data()

    assert len(loaded) == len(business_data)

    assert returned_metadata == metadata

    np.testing.assert_allclose(
        loaded["MonthlyCharges"],
        business_data["MonthlyCharges"],
    )


def test_business_data_rejects_label_mismatch(
    business_data,
    monkeypatch,
):

    predictions = business_data.drop(
        columns=["MonthlyCharges"]
    ).copy()

    predictions.loc[0, "actual_churn"] = (
        1 - predictions.loc[0, "actual_churn"]
    )

    monkeypatch.setattr(
        "src.business_cost_analysis."
        "load_calibrated_predictions",
        lambda: (predictions, {}),
    )

    with pytest.raises(ValueError):

        load_business_data()


def test_business_data_rejects_unknown_customer(
    business_data,
    monkeypatch,
):

    predictions = business_data.drop(
        columns=["MonthlyCharges"]
    ).copy()

    predictions.loc[0, "customerID"] = (
        "UNKNOWN-TEST-CUSTOMER"
    )

    monkeypatch.setattr(
        "src.business_cost_analysis."
        "load_calibrated_predictions",
        lambda: (predictions, {}),
    )

    with pytest.raises(ValueError):

        load_business_data()


def test_selected_calibration_methods(
    tmp_path,
    monkeypatch,
):

    path = tmp_path / "methods.csv"

    pd.DataFrame({
        "model": list(SUPPORTED_MODELS),
        "calibration_method": ["sigmoid"] * 3,
    }).to_csv(path, index=False)

    monkeypatch.setattr(
        "src.business_cost_analysis."
        "SELECTED_METHODS_PATH",
        path,
    )

    result = load_selected_methods()

    assert result == {
        model: "sigmoid"
        for model in SUPPORTED_MODELS
    }


def test_original_data_budget_constraints(
    business_data,
):

    targets = select_budget_constrained_customers(
        business_data[
            "logistic_regression_sigmoid_probability"
        ],
        business_data["MonthlyCharges"],
        business_data["customerID"],
    )

    assert targets.sum() <= 400

    assert targets.sum() * 25 <= 10000


def test_budget_targets_are_profitable(
    business_data,
):

    probabilities = business_data[
        "logistic_regression_sigmoid_probability"
    ]

    charges = business_data["MonthlyCharges"]

    targets = select_budget_constrained_customers(
        probabilities,
        charges,
        business_data["customerID"],
    )

    benefits = calculate_expected_net_benefits(
        probabilities,
        charges,
    )

    assert np.all(benefits[targets] > 0)


def test_customer_value_summary(business_data):

    result = summarize_customer_values(
        business_data
    ).set_index("customer_group")

    assert len(result) == 3

    assert result.loc[
        "all_customers",
        "customer_count",
    ] == len(business_data)

    assert result.loc[
        "all_customers",
        "total_retained_contribution",
    ] == pytest.approx(
        business_data["MonthlyCharges"].sum()
        * 4.8
    )


def test_break_even_summary(business_data):

    result = summarize_break_even_probabilities(
        business_data
    ).iloc[0]

    assert result["customer_count"] == 1409

    assert (
        result["minimum_break_even_probability"]
        <= result["median_break_even_probability"]
        <= result["maximum_break_even_probability"]
    )


def test_break_even_summary_zero_success(
    business_data,
):

    config = replace(
        BASELINE_CONFIG,
        retention_success_rate=0,
    )

    result = summarize_break_even_probabilities(
        business_data,
        config,
    ).iloc[0]

    assert result[
        "unachievable_break_even_customers"
    ] == len(business_data)

    assert np.isnan(
        result["median_break_even_probability"]
    )


# --------------------------------------------------
# Model strategy evaluation
# --------------------------------------------------


def test_model_strategy_counts(model_results):

    campaigns, thresholds = model_results

    assert len(campaigns) == 36

    assert len(thresholds) == 297

    assert campaigns.groupby("model").size().eq(
        12
    ).all()


def test_customer_value_strategy_optimal(
    model_results,
):

    campaigns, _ = model_results

    for model in SUPPORTED_MODELS:

        subset = campaigns[
            campaigns["model"] == model
        ]

        value_profit = subset.loc[
            subset["strategy"]
            == "customer_value_targeting",
            "expected_net_profit",
        ].iloc[0]

        assert value_profit >= (
            subset["expected_net_profit"].max()
            - 1e-8
        )


def test_budget_value_strategy_optimal(
    model_results,
):

    campaigns, _ = model_results

    for model in SUPPORTED_MODELS:

        subset = campaigns[
            (campaigns["model"] == model)
            & campaigns["within_budget"]
        ]

        value_profit = subset.loc[
            subset["strategy"]
            == "budget_value_targeting",
            "expected_net_profit",
        ].iloc[0]

        assert value_profit >= (
            subset["expected_net_profit"].max()
            - 1e-8
        )


def test_best_business_strategies(model_results):

    campaigns, _ = model_results

    best = select_best_business_strategies(
        campaigns
    )

    assert len(best) == 6

    constrained = best[
        best["selection_type"]
        == "budget_constrained"
    ]

    assert len(constrained) == 3

    assert constrained["within_budget"].all()

    assert constrained["campaign_cost"].le(
        10000
    ).all()


def test_missing_model_results_rejected(
    model_results,
):

    campaigns, _ = model_results

    incomplete = campaigns[
        campaigns["model"] != "xgboost"
    ]

    with pytest.raises(ValueError):

        select_best_business_strategies(
            incomplete
        )


def test_unsupported_model_rejected(
    business_data,
    statistical_thresholds,
):

    with pytest.raises(ValueError):

        evaluate_model_strategies(
            business_data,
            "unknown_model",
            "sigmoid",
            statistical_thresholds,
        )


def test_unsupported_calibrator_rejected(
    business_data,
    statistical_thresholds,
):

    with pytest.raises(ValueError):

        evaluate_model_strategies(
            business_data,
            "xgboost",
            "unknown_calibrator",
            statistical_thresholds,
        )


# --------------------------------------------------
# Report export
# --------------------------------------------------


def test_business_report_export(
    tmp_path,
    business_data,
    model_results,
):

    campaigns, thresholds = model_results

    best = select_best_business_strategies(
        campaigns
    )

    customer_summary = summarize_customer_values(
        business_data
    )

    break_even = (
        summarize_break_even_probabilities(
            business_data
        )
    )

    metadata = {
        "test_evaluated": False,
        "production_updated": False,
    }

    save_business_reports(
        campaigns,
        thresholds,
        best,
        customer_summary,
        break_even,
        BASELINE_CONFIG,
        metadata,
        output_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*.csv"))) == 5

    assert len(list(tmp_path.glob("*.json"))) == 2

    saved = pd.read_csv(
        tmp_path / "campaign_strategy_comparison.csv"
    )

    assert len(saved) == len(campaigns)

    with open(
        tmp_path / "business_metadata.json",
        encoding="utf-8",
    ) as file:

        assert json.load(file) == metadata