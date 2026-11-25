"""
src/sensitivity_analysis.py

Financial sensitivity analysis for telecom
customer retention campaigns.

All financial outcomes are hypothetical.
"""

import json

from dataclasses import replace

import numpy as np
import pandas as pd

from src.business_config import (
    BASELINE_CONFIG,
    REPORTS_DIR,
    CONTACT_COST_SCENARIOS,
    INCENTIVE_COST_SCENARIOS,
    RETENTION_SUCCESS_SCENARIOS,
    RETENTION_MONTH_SCENARIOS,
    CONTRIBUTION_MARGIN_SCENARIOS,
    CAMPAIGN_BUDGET_SCENARIOS,
)

from src.business_cost_analysis import (
    load_business_data,
    load_selected_methods,
    select_profitable_customers,
    select_budget_constrained_customers,
    evaluate_campaign,
)

from src.optimization import SUPPORTED_MODELS


SENSITIVITY_DIR = (
    REPORTS_DIR / "sensitivity"
)


PARAMETER_SCENARIOS = {
    "contact_cost":
        CONTACT_COST_SCENARIOS,

    "incentive_cost":
        INCENTIVE_COST_SCENARIOS,

    "retention_success_rate":
        RETENTION_SUCCESS_SCENARIOS,

    "retention_months":
        RETENTION_MONTH_SCENARIOS,

    "contribution_margin":
        CONTRIBUTION_MARGIN_SCENARIOS,

    "campaign_budget":
        CAMPAIGN_BUDGET_SCENARIOS,
}


# --------------------------------------------------
# Financial scenario evaluation
# --------------------------------------------------


def evaluate_financial_scenario(
    business_data,
    selected_methods,
    config=BASELINE_CONFIG,
):
    """
    Evaluate all models under a financial scenario.

    Customer selection is based on expected profit,
    not actual churn labels.
    """

    labels = business_data[
        "actual_churn"
    ].to_numpy()

    charges = business_data[
        "MonthlyCharges"
    ].to_numpy()

    customer_ids = business_data[
        "customerID"
    ].to_numpy()

    rows = []

    for model_name in SUPPORTED_MODELS:

        method = selected_methods[
            model_name
        ]

        probability_column = (
            f"{model_name}_"
            f"{method}_probability"
        )

        probabilities = business_data[
            probability_column
        ].to_numpy()

        unconstrained_targets = (
            select_profitable_customers(
                probabilities,
                charges,
                config,
            )
        )

        constrained_targets = (
            select_budget_constrained_customers(
                probabilities,
                charges,
                customer_ids,
                config,
            )
        )

        for strategy, targets in [
            (
                "customer_value_targeting",
                unconstrained_targets,
            ),
            (
                "budget_value_targeting",
                constrained_targets,
            ),
        ]:

            metrics = evaluate_campaign(
                labels,
                probabilities,
                charges,
                targets,
                config,
            )

            rows.append({
                "model": model_name,

                "calibration_method":
                    method,

                "strategy":
                    strategy,

                "contact_cost":
                    config.contact_cost,

                "incentive_cost":
                    config.incentive_cost,

                "intervention_cost":
                    config.intervention_cost,

                "retention_success_rate":
                    config.retention_success_rate,

                "retention_months":
                    config.retention_months,

                "contribution_margin":
                    config.contribution_margin,

                "campaign_budget":
                    config.campaign_budget,

                "maximum_budget_targets":
                    config.maximum_budget_targets,

                **metrics,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# One-way sensitivity analysis
# --------------------------------------------------


def run_one_way_sensitivity(
    business_data,
    selected_methods,
    baseline_config=BASELINE_CONFIG,
    parameter_scenarios=None,
):
    """
    Change one financial parameter at a time.
    """

    if parameter_scenarios is None:

        parameter_scenarios = (
            PARAMETER_SCENARIOS
        )

    if not parameter_scenarios:

        raise ValueError(
            "No sensitivity scenarios provided."
        )

    all_results = []

    for parameter, values in (
        parameter_scenarios.items()
    ):

        if parameter not in PARAMETER_SCENARIOS:

            raise ValueError(
                f"Unsupported parameter: {parameter}"
            )

        values = tuple(values)

        if not values:

            raise ValueError(
                f"No values provided for {parameter}."
            )

        if len(set(values)) != len(values):

            raise ValueError(
                f"Duplicate values for {parameter}."
            )

        baseline_value = getattr(
            baseline_config,
            parameter,
        )

        for value in values:

            scenario_config = replace(
                baseline_config,
                **{parameter: value},
            )

            results = (
                evaluate_financial_scenario(
                    business_data,
                    selected_methods,
                    scenario_config,
                )
            )

            results.insert(
                0,
                "is_baseline",
                bool(
                    np.isclose(
                        value,
                        baseline_value,
                    )
                ),
            )

            results.insert(
                0,
                "baseline_value",
                baseline_value,
            )

            results.insert(
                0,
                "parameter_value",
                value,
            )

            results.insert(
                0,
                "parameter",
                parameter,
            )

            all_results.append(results)

        print(
            f"One-way sensitivity completed: "
            f"{parameter}"
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# --------------------------------------------------
# Two-way sensitivity analysis
# --------------------------------------------------


def run_two_way_sensitivity(
    business_data,
    selected_methods,
    baseline_config=BASELINE_CONFIG,
    incentive_costs=INCENTIVE_COST_SCENARIOS,
    success_rates=RETENTION_SUCCESS_SCENARIOS,
):
    """
    Evaluate interactions between incentive costs
    and retention success probabilities.
    """

    incentive_costs = tuple(
        incentive_costs
    )

    success_rates = tuple(
        success_rates
    )

    if not incentive_costs or not success_rates:

        raise ValueError(
            "Two-way scenarios cannot be empty."
        )

    if (
        len(set(incentive_costs))
        != len(incentive_costs)
        or len(set(success_rates))
        != len(success_rates)
    ):

        raise ValueError(
            "Duplicate two-way scenario values."
        )

    all_results = []

    for incentive_cost in incentive_costs:

        for success_rate in success_rates:

            scenario_config = replace(
                baseline_config,

                incentive_cost=incentive_cost,

                retention_success_rate=success_rate,
            )

            results = (
                evaluate_financial_scenario(
                    business_data,
                    selected_methods,
                    scenario_config,
                )
            )

            results.insert(
                0,
                "scenario_id",
                (
                    f"incentive_{incentive_cost:g}"
                    f"_success_{success_rate:g}"
                ),
            )

            all_results.append(results)

    print(
        "Two-way sensitivity completed."
    )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# --------------------------------------------------
# Named financial scenarios
# --------------------------------------------------


def create_named_scenarios(
    baseline_config=BASELINE_CONFIG,
):
    """
    Define hypothetical financial environments.

    The budget is held constant so scenarios
    remain financially comparable.
    """

    return {
        "pessimistic": replace(
            baseline_config,

            contact_cost=10.0,

            incentive_cost=30.0,

            retention_success_rate=0.15,

            retention_months=6,

            contribution_margin=0.25,
        ),

        "baseline":
            baseline_config,

        "optimistic": replace(
            baseline_config,

            contact_cost=3.0,

            incentive_cost=10.0,

            retention_success_rate=0.50,

            retention_months=18,

            contribution_margin=0.55,
        ),
    }


def run_named_scenarios(
    business_data,
    selected_methods,
    baseline_config=BASELINE_CONFIG,
):
    """
    Evaluate pessimistic, baseline and optimistic
    business environments.
    """

    scenarios = create_named_scenarios(
        baseline_config
    )

    all_results = []

    for scenario_name, config in (
        scenarios.items()
    ):

        results = (
            evaluate_financial_scenario(
                business_data,
                selected_methods,
                config,
            )
        )

        results.insert(
            0,
            "scenario",
            scenario_name,
        )

        all_results.append(results)

        print(
            f"Named scenario completed: "
            f"{scenario_name}"
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# --------------------------------------------------
# Sensitivity impact summary
# --------------------------------------------------


def summarize_parameter_impacts(
    one_way_results,
):
    """
    Quantify how strongly each financial parameter
    affects expected campaign profit.
    """

    rows = []

    grouped = one_way_results.groupby(
        [
            "model",
            "strategy",
            "parameter",
        ],
        sort=False,
    )

    for keys, subset in grouped:

        model, strategy, parameter = keys

        baseline = subset[
            subset["is_baseline"]
        ]

        if len(baseline) != 1:

            raise ValueError(
                f"Expected exactly one baseline "
                f"for {parameter}."
            )

        baseline_profit = float(
            baseline.iloc[0][
                "expected_net_profit"
            ]
        )

        minimum_row = subset.loc[
            subset[
                "expected_net_profit"
            ].idxmin()
        ]

        maximum_row = subset.loc[
            subset[
                "expected_net_profit"
            ].idxmax()
        ]

        minimum_profit = float(
            minimum_row[
                "expected_net_profit"
            ]
        )

        maximum_profit = float(
            maximum_row[
                "expected_net_profit"
            ]
        )

        rows.append({
            "model": model,

            "strategy": strategy,

            "parameter": parameter,

            "baseline_expected_profit":
                baseline_profit,

            "minimum_expected_profit":
                minimum_profit,

            "maximum_expected_profit":
                maximum_profit,

            "minimum_profit_parameter_value":
                float(
                    minimum_row[
                        "parameter_value"
                    ]
                ),

            "maximum_profit_parameter_value":
                float(
                    maximum_row[
                        "parameter_value"
                    ]
                ),

            "downside_from_baseline":
                minimum_profit
                - baseline_profit,

            "upside_from_baseline":
                maximum_profit
                - baseline_profit,

            "profit_range":
                maximum_profit
                - minimum_profit,

            "relative_profit_range":
                (
                    (
                        maximum_profit
                        - minimum_profit
                    )
                    / baseline_profit
                )
                if baseline_profit > 0
                else np.nan,
        })

    return pd.DataFrame(rows).sort_values(
        [
            "model",
            "strategy",
            "profit_range",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)


# --------------------------------------------------
# Model stability analysis
# --------------------------------------------------


def calculate_scenario_winners(
    results,
    scenario_columns,
):
    """
    Identify the highest expected-profit model
    for each scenario and targeting strategy.

    Financial ties are retained explicitly.
    """

    grouping_columns = [
        *scenario_columns,
        "strategy",
    ]

    rows = []

    grouped = results.groupby(
        grouping_columns,
        sort=False,
        dropna=False,
    )

    for keys, subset in grouped:

        if not isinstance(keys, tuple):

            keys = (keys,)

        scenario = dict(zip(
            grouping_columns,
            keys,
        ))

        if (
            subset["model"].duplicated().any()
            or set(subset["model"])
            != set(SUPPORTED_MODELS)
        ):

            raise ValueError(
                "Scenario model membership mismatch."
            )

        ranked = subset.sort_values(
            [
                "expected_net_profit",
                "model",
            ],
            ascending=[
                False,
                True,
            ],
        )

        best_profit = float(
            ranked.iloc[0][
                "expected_net_profit"
            ]
        )

        second_profit = float(
            ranked.iloc[1][
                "expected_net_profit"
            ]
        )

        winners = ranked[
            np.isclose(
                ranked["expected_net_profit"],
                best_profit,
                rtol=0,
                atol=1e-8,
            )
        ]

        no_intervention_preferred = (
            best_profit <= 1e-8
        )

        winner_names = (
            []
            if no_intervention_preferred
            else sorted(
                winners["model"].tolist()
            )
        )

        rows.append({
            **scenario,

            "winning_model":
                (
                    "no_intervention"
                    if no_intervention_preferred
                    else "|".join(winner_names)
                ),

            "winning_model_count":
                len(winner_names),

            "expected_net_profit":
                best_profit,

            "runner_up_expected_profit":
                second_profit,

            "profit_advantage":
                best_profit - second_profit,

            "relative_profit_advantage":
                (
                    (
                        best_profit
                        - second_profit
                    )
                    / second_profit
                )
                if second_profit > 0
                else np.nan,

            "is_tie":
                bool(
                    len(winner_names) > 1
                ),

            "no_intervention_preferred":
                no_intervention_preferred,
        })

    return pd.DataFrame(rows)


def summarize_model_stability(
    winners,
):
    """
    Summarize outright wins, shared wins and
    fractional wins across scenarios.

    A two-model tie contributes 0.5 wins
    to each tied model.
    """

    rows = []

    for strategy, subset in winners.groupby(
        "strategy",
        sort=False,
    ):

        total_scenarios = len(subset)

        no_intervention_count = int(
            subset[
                "no_intervention_preferred"
            ].sum()
        )

        for model_name in SUPPORTED_MODELS:

            outright_wins = 0

            shared_wins = 0

            fractional_wins = 0.0

            for _, row in subset.iterrows():

                if row[
                    "no_intervention_preferred"
                ]:

                    continue

                winning_models = row[
                    "winning_model"
                ].split("|")

                if model_name not in winning_models:

                    continue

                if len(winning_models) == 1:

                    outright_wins += 1

                else:

                    shared_wins += 1

                fractional_wins += (
                    1 / len(winning_models)
                )

            rows.append({
                "strategy": strategy,

                "model": model_name,

                "total_scenarios":
                    total_scenarios,

                "outright_wins":
                    outright_wins,

                "shared_wins":
                    shared_wins,

                "fractional_wins":
                    fractional_wins,

                "win_rate":
                    fractional_wins
                    / total_scenarios,

                "no_intervention_scenarios":
                    no_intervention_count,
            })

    return pd.DataFrame(rows).sort_values(
        [
            "strategy",
            "win_rate",
            "model",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


# --------------------------------------------------
# Budget sensitivity summary
# --------------------------------------------------


def summarize_budget_sensitivity(
    one_way_results,
):
    """
    Calculate incremental returns from expanding
    the retention campaign budget.
    """

    budget_results = one_way_results[
        (
            one_way_results["parameter"]
            == "campaign_budget"
        )
        & (
            one_way_results["strategy"]
            == "budget_value_targeting"
        )
    ].copy()

    if budget_results.empty:

        raise ValueError(
            "No budget sensitivity results."
        )

    budget_results = budget_results.sort_values(
        [
            "model",
            "campaign_budget",
        ]
    ).reset_index(drop=True)

    grouped = budget_results.groupby(
        "model",
        sort=False,
    )

    budget_results[
        "additional_budget"
    ] = grouped[
        "campaign_budget"
    ].diff()

    budget_results[
        "additional_campaign_cost"
    ] = grouped[
        "campaign_cost"
    ].diff()

    budget_results[
        "additional_expected_profit"
    ] = grouped[
        "expected_net_profit"
    ].diff()

    additional_cost = budget_results[
        "additional_campaign_cost"
    ].to_numpy(dtype=float)

    additional_profit = budget_results[
        "additional_expected_profit"
    ].to_numpy(dtype=float)

    budget_results[
        "marginal_return_on_spend"
    ] = np.divide(
        additional_profit,
        additional_cost,
        out=np.full(
            len(budget_results),
            np.nan,
        ),
        where=additional_cost > 0,
    )

    budget_results[
        "unused_budget"
    ] = (
        budget_results["campaign_budget"]
        - budget_results["campaign_cost"]
    )

    columns = [
        "model",
        "campaign_budget",
        "maximum_budget_targets",
        "customers_targeted",
        "campaign_cost",
        "unused_budget",
        "expected_net_profit",
        "simulated_net_profit",
        "expected_roi",
        "recall",
        "additional_budget",
        "additional_campaign_cost",
        "additional_expected_profit",
        "marginal_return_on_spend",
    ]

    return budget_results[columns]


# --------------------------------------------------
# Baseline consistency verification
# --------------------------------------------------


def verify_baseline_consistency(
    one_way_results,
    named_results,
):
    """
    Ensure repeated baseline configurations
    produce identical financial outcomes.
    """

    baseline_named = named_results[
        named_results["scenario"]
        == "baseline"
    ]

    baseline_one_way = one_way_results[
        one_way_results["is_baseline"]
    ]

    if baseline_named.empty:

        raise ValueError(
            "Missing named baseline scenario."
        )

    if baseline_one_way.empty:

        raise ValueError(
            "Missing one-way baseline scenarios."
        )

    comparison = baseline_one_way.merge(
        baseline_named[
            [
                "model",
                "strategy",
                "expected_net_profit",
                "campaign_cost",
                "customers_targeted",
            ]
        ],
        on=[
            "model",
            "strategy",
        ],
        suffixes=[
            "_one_way",
            "_named",
        ],
        how="left",
        validate="many_to_one",
    )

    for metric in [
        "expected_net_profit",
        "campaign_cost",
        "customers_targeted",
    ]:

        if not np.allclose(
            comparison[
                f"{metric}_one_way"
            ],

            comparison[
                f"{metric}_named"
            ],

            rtol=0,
            atol=1e-8,
        ):

            raise ValueError(
                f"Baseline inconsistency: {metric}"
            )

    print(
        "Baseline consistency verification passed."
    )

    return True


# --------------------------------------------------
# Report export
# --------------------------------------------------


def save_sensitivity_reports(
    reports,
    metadata,
    output_dir=SENSITIVITY_DIR,
):
    """Save aggregate sensitivity analysis reports."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, dataframe in reports.items():

        dataframe.to_csv(
            output_dir / filename,
            index=False,
        )

    with open(
        output_dir / "sensitivity_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    print(
        f"\nSensitivity reports saved: "
        f"{output_dir}"
    )


# --------------------------------------------------
# Main execution
# --------------------------------------------------


def main():

    print(
        "\nLoading verified business data..."
    )

    business_data, calibration_metadata = (
        load_business_data()
    )

    selected_methods = (
        load_selected_methods()
    )

    config = BASELINE_CONFIG

    print(
        f"Validation customers: "
        f"{len(business_data):,}"
    )

    print(
        "\nRunning one-way sensitivity analysis..."
    )

    one_way_results = (
        run_one_way_sensitivity(
            business_data,
            selected_methods,
            config,
        )
    )

    print(
        "\nRunning two-way sensitivity analysis..."
    )

    two_way_results = (
        run_two_way_sensitivity(
            business_data,
            selected_methods,
            config,
        )
    )

    print(
        "\nRunning named financial scenarios..."
    )

    named_results = run_named_scenarios(
        business_data,
        selected_methods,
        config,
    )

    verify_baseline_consistency(
        one_way_results,
        named_results,
    )

    parameter_impacts = (
        summarize_parameter_impacts(
            one_way_results
        )
    )

    budget_summary = (
        summarize_budget_sensitivity(
            one_way_results
        )
    )

    one_way_winners = (
        calculate_scenario_winners(
            one_way_results,
            [
                "parameter",
                "parameter_value",
            ],
        )
    )

    two_way_winners = (
        calculate_scenario_winners(
            two_way_results,
            [
                "incentive_cost",
                "retention_success_rate",
            ],
        )
    )

    named_winners = (
        calculate_scenario_winners(
            named_results,
            ["scenario"],
        )
    )

    # Report each scenario family separately.
    # These are designed scenarios, not random
    # draws from a real-world probability model.

    stability_results = []

    for family, winners in [
        (
            "one_way",
            one_way_winners,
        ),
        (
            "two_way",
            two_way_winners,
        ),
        (
            "named",
            named_winners,
        ),
    ]:

        stability = (
            summarize_model_stability(
                winners
            )
        )

        stability.insert(
            0,
            "scenario_family",
            family,
        )

        stability_results.append(
            stability
        )

    model_stability = pd.concat(
        stability_results,
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
            parameter_impacts,

        "budget_sensitivity_summary.csv":
            budget_summary,

        "one_way_model_winners.csv":
            one_way_winners,

        "two_way_model_winners.csv":
            two_way_winners,

        "named_scenario_winners.csv":
            named_winners,

        "model_stability_summary.csv":
            model_stability,
    }

    metadata = {
        "dataset_hash":
            calibration_metadata[
                "dataset_hash"
            ],

        "calibrated_predictions_hash":
            calibration_metadata[
                "calibrated_predictions_hash"
            ],

        "validation_rows":
            len(business_data),

        "models":
            list(SUPPORTED_MODELS),

        "selected_calibration_methods":
            selected_methods,

        "baseline_configuration":
            config.to_dict(),

        "one_way_parameters": {
            name: list(values)

            for name, values in
            PARAMETER_SCENARIOS.items()
        },

        "one_way_scenarios":
            sum(
                len(values)

                for values in
                PARAMETER_SCENARIOS.values()
            ),

        "two_way_scenarios":
            (
                len(INCENTIVE_COST_SCENARIOS)
                * len(RETENTION_SUCCESS_SCENARIOS)
            ),

        "named_scenarios": {
            name: scenario_config.to_dict()

            for name, scenario_config in
            create_named_scenarios(config).items()
        },

        "one_way_evaluations":
            len(one_way_results),

        "two_way_evaluations":
            len(two_way_results),

        "named_scenario_evaluations":
            len(named_results),

        "selection_objective":
            "expected_net_profit",

        "financial_assumptions":
            "hypothetical_simulation",

        "scenario_win_rates":
            "descriptive_not_probabilistic",

        "baseline_consistency_verified":
            True,

        "causal_intervention_effect_estimated":
            False,

        "evaluation_partition":
            "validation",

        "test_evaluated":
            False,

        "production_updated":
            False,
    }

    save_sensitivity_reports(
        reports,
        metadata,
    )

    print(
        "\nNAMED SCENARIO RESULTS"
    )

    display_columns = [
        "scenario",
        "model",
        "customers_targeted",
        "campaign_cost",
        "expected_net_profit",
        "simulated_net_profit",
        "expected_roi",
    ]

    print(
        named_results[
            named_results["strategy"]
            == "budget_value_targeting"
        ][display_columns].round(4).to_string(
            index=False
        )
    )

    print(
        "\nMODEL STABILITY SUMMARY"
    )

    print(
        model_stability.round(4).to_string(
            index=False
        )
    )

    print(
        "\nBUDGET SENSITIVITY SUMMARY"
    )

    budget_columns = [
        "model",
        "campaign_budget",
        "customers_targeted",
        "campaign_cost",
        "unused_budget",
        "expected_net_profit",
        "marginal_return_on_spend",
    ]

    print(
        budget_summary[
            budget_columns
        ].round(4).to_string(
            index=False
        )
    )

    total_evaluations = (
        len(one_way_results)
        + len(two_way_results)
        + len(named_results)
    )

    print(
        f"\nOne-way evaluations: "
        f"{len(one_way_results)}"
    )

    print(
        f"Two-way evaluations: "
        f"{len(two_way_results)}"
    )

    print(
        f"Named scenario evaluations: "
        f"{len(named_results)}"
    )

    print(
        f"Total sensitivity evaluations: "
        f"{total_evaluations}"
    )

    print(
        "\nSensitivity analysis completed."
    )

    print(
        "All financial outcomes are hypothetical."
    )

    print(
        "Scenario win rates are descriptive, "
        "not real-world probabilities."
    )

    print(
        "Production models remain unchanged."
    )


if __name__ == "__main__":
    main()