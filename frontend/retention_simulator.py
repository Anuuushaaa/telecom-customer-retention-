
"""Expected-outcome customer retention campaign simulator."""

from dataclasses import replace

import numpy as np
import pandas as pd

from src.business_config import (
    BusinessConfig,
    BASELINE_CONFIG,
    INCENTIVE_COST_SCENARIOS,
    RETENTION_SUCCESS_SCENARIOS,
)

from src.business_cost_analysis import (
    calculate_retained_values,
    calculate_expected_net_benefits,
    calculate_break_even_probabilities,
    select_threshold_targets,
    select_profitable_customers,
    select_budget_constrained_customers,
)


STRATEGY_NAMES = {
    "threshold": "Classification Threshold",
    "profitable": "Expected Profit",
    "budget_constrained": "Budget-Constrained Profit",
}


# ============================================================
# PORTFOLIO VALIDATION
# ============================================================

def validate_portfolio(results):

    required = {
        "customerID",
        "churn_probability",
        "MonthlyCharges",
    }

    missing = required - set(results.columns)

    if missing:

        raise ValueError(
            "Missing portfolio columns: "
            + ", ".join(sorted(missing))
        )

    if results.empty:

        raise ValueError(
            "No customers available for simulation."
        )

    dataframe = results.copy().reset_index(
        drop=True
    )

    if dataframe["customerID"].isna().any():

        raise ValueError(
            "Missing customer IDs."
        )

    dataframe["customerID"] = (
        dataframe["customerID"]
        .astype(str)
        .str.strip()
    )

    if dataframe["customerID"].eq("").any():

        raise ValueError(
            "Empty customer IDs."
        )

    if dataframe["customerID"].duplicated().any():

        raise ValueError(
            "Duplicate customer IDs."
        )

    for column in [
        "churn_probability",
        "MonthlyCharges",
    ]:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

        if not np.isfinite(
            dataframe[column]
        ).all():

            raise ValueError(
                f"Non-finite values in {column}."
            )

    if not dataframe[
        "churn_probability"
    ].between(0, 1).all():

        raise ValueError(
            "Invalid churn probabilities."
        )

    if (
        dataframe["MonthlyCharges"] < 0
    ).any():

        raise ValueError(
            "Monthly charges cannot be negative."
        )

    return dataframe


# ============================================================
# CUSTOMER FINANCIAL ESTIMATES
# ============================================================

def calculate_customer_economics(
    results,
    config=BASELINE_CONFIG,
):

    dataframe = validate_portfolio(results)

    probabilities = dataframe[
        "churn_probability"
    ].to_numpy(dtype=float)

    charges = dataframe[
        "MonthlyCharges"
    ].to_numpy(dtype=float)

    dataframe["potential_retained_contribution"] = (
        calculate_retained_values(
            charges,
            config,
        )
    )

    dataframe["expected_retention_probability"] = (
        probabilities
        * config.retention_success_rate
    )

    dataframe["expected_retained_contribution"] = (
        dataframe["expected_retention_probability"]
        * dataframe["potential_retained_contribution"]
    )

    dataframe["intervention_cost"] = (
        config.intervention_cost
    )

    dataframe["expected_net_benefit"] = (
        calculate_expected_net_benefits(
            probabilities,
            charges,
            config,
        )
    )

    dataframe["break_even_probability"] = (
        calculate_break_even_probabilities(
            charges,
            config,
        )
    )

    dataframe["profitable_intervention"] = (
        dataframe["expected_net_benefit"] > 0
    )

    return dataframe


# ============================================================
# TARGET SELECTION
# ============================================================

def select_campaign_targets(
    dataframe,
    strategy,
    threshold,
    config,
):

    probabilities = dataframe[
        "churn_probability"
    ].to_numpy(dtype=float)

    charges = dataframe[
        "MonthlyCharges"
    ].to_numpy(dtype=float)

    customer_ids = dataframe[
        "customerID"
    ].to_numpy(dtype=str)

    if strategy == "threshold":

        return select_threshold_targets(
            probabilities,
            threshold,
        )

    if strategy == "profitable":

        return select_profitable_customers(
            probabilities,
            charges,
            config,
        )

    if strategy == "budget_constrained":

        return select_budget_constrained_customers(
            probabilities,
            charges,
            customer_ids,
            config,
        )

    raise ValueError(
        f"Unsupported strategy: {strategy}"
    )


# ============================================================
# EXPECTED CAMPAIGN EVALUATION
# ============================================================

def evaluate_expected_campaign(
    dataframe,
    targets,
    config,
):

    targets = np.asarray(targets)

    if (
        targets.ndim != 1
        or len(targets) != len(dataframe)
        or targets.dtype != np.dtype(bool)
    ):

        raise ValueError(
            "Invalid campaign target mask."
        )

    selected = dataframe.loc[targets]

    targeted_customers = len(selected)

    campaign_cost = (
        targeted_customers
        * config.intervention_cost
    )

    expected_churners_targeted = float(
        selected["churn_probability"].sum()
    )

    expected_retained_customers = float(
        selected[
            "expected_retention_probability"
        ].sum()
    )

    expected_retained_contribution = float(
        selected[
            "expected_retained_contribution"
        ].sum()
    )

    expected_net_profit = (
        expected_retained_contribution
        - campaign_cost
    )

    expected_roi = (
        expected_net_profit / campaign_cost
        if campaign_cost > 0
        else None
    )

    budget_utilization = (
        campaign_cost / config.campaign_budget
        if config.campaign_budget > 0
        else None
    )

    total_expected_churn = float(
        dataframe["churn_probability"].sum()
    )

    expected_churn_coverage = (
        expected_churners_targeted
        / total_expected_churn
        if total_expected_churn > 0
        else 0.0
    )

    return {
        "total_customers": len(dataframe),

        "targeted_customers": targeted_customers,

        "targeting_rate": (
            targeted_customers / len(dataframe)
        ),

        "campaign_cost": campaign_cost,

        "expected_churners_targeted":
            expected_churners_targeted,

        "expected_retained_customers":
            expected_retained_customers,

        "expected_retained_contribution":
            expected_retained_contribution,

        "expected_net_profit":
            expected_net_profit,

        "expected_roi": expected_roi,

        "budget_utilization":
            budget_utilization,

        "budget_remaining": (
            config.campaign_budget
            - campaign_cost
        ),

        "within_budget": bool(
            campaign_cost
            <= config.campaign_budget + 1e-8
        ),

        "expected_churn_coverage":
            expected_churn_coverage,

        "unprofitable_targets": int(
            (
                selected["expected_net_benefit"] <= 0
            ).sum()
        ),
    }


# ============================================================
# STRATEGY COMPARISON
# ============================================================

def compare_campaign_strategies(
    results,
    threshold,
    config=BASELINE_CONFIG,
):

    dataframe = calculate_customer_economics(
        results,
        config,
    )

    summaries = []
    target_tables = {}

    for strategy, label in STRATEGY_NAMES.items():

        targets = select_campaign_targets(
            dataframe=dataframe,
            strategy=strategy,
            threshold=threshold,
            config=config,
        )

        summary = evaluate_expected_campaign(
            dataframe,
            targets,
            config,
        )

        summaries.append({
            "strategy": strategy,
            "strategy_name": label,
            **summary,
        })

        selected = dataframe.loc[
            targets
        ].copy()

        selected = selected.sort_values(
            by=[
                "expected_net_benefit",
                "customerID",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

        selected.insert(
            0,
            "campaign_rank",
            np.arange(1, len(selected) + 1),
        )

        target_tables[strategy] = selected

    return (
        pd.DataFrame(summaries),
        target_tables,
        dataframe,
    )


# ============================================================
# TWO-WAY SENSITIVITY ANALYSIS
# ============================================================

def run_campaign_sensitivity(
    results,
    config=BASELINE_CONFIG,
    success_rates=RETENTION_SUCCESS_SCENARIOS,
    incentive_costs=INCENTIVE_COST_SCENARIOS,
):

    portfolio = validate_portfolio(results)

    scenarios = []

    for success_rate in success_rates:

        for incentive_cost in incentive_costs:

            scenario_config = replace(
                config,
                retention_success_rate=float(
                    success_rate
                ),
                incentive_cost=float(
                    incentive_cost
                ),
            )

            dataframe = calculate_customer_economics(
                portfolio,
                scenario_config,
            )

            targets = select_campaign_targets(
                dataframe=dataframe,
                strategy="budget_constrained",
                threshold=0.50,
                config=scenario_config,
            )

            summary = evaluate_expected_campaign(
                dataframe,
                targets,
                scenario_config,
            )

            scenarios.append({
                "retention_success_rate":
                    float(success_rate),

                "incentive_cost":
                    float(incentive_cost),

                "intervention_cost":
                    scenario_config.intervention_cost,

                **summary,
            })

    return pd.DataFrame(scenarios)
