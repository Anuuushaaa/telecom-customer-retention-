"""
src/business_cost_analysis.py

Customer-level financial valuation, retention
campaign simulation and business-optimal targeting.

Financial outcomes are hypothetical simulations,
not observed intervention results.
"""

import json

import numpy as np
import pandas as pd

from src.business_config import (
    BASELINE_CONFIG,
    DATA_PATH,
    REPORTS_DIR,
    SELECTED_METHODS_PATH,
    SELECTED_THRESHOLDS_PATH,
)

from src.optimization import SUPPORTED_MODELS

from src.bootstrap_evaluation import (
    calculate_file_hash,
)

from src.probability_calibration import (
    CALIBRATION_METHODS,
    validate_probabilities,
    validate_labels,
)

from src.threshold_tuning import (
    DEFAULT_THRESHOLDS,
    load_calibrated_predictions,
    validate_thresholds,
)


# --------------------------------------------------
# Customer financial data
# --------------------------------------------------


def load_business_data():
    """
    Load verified validation predictions and align
    them with original customer financial data.
    """

    predictions, metadata = (
        load_calibrated_predictions()
    )

    raw_data = pd.read_csv(DATA_PATH)

    financial_data = raw_data[
        [
            "customerID",
            "MonthlyCharges",
            "Churn",
        ]
    ].copy()

    if financial_data[
        "customerID"
    ].duplicated().any():

        raise ValueError(
            "Duplicate customer IDs in raw data."
        )

    financial_data["raw_churn"] = (
        financial_data["Churn"].map({
            "Yes": 1,
            "No": 0,
        })
    )

    financial_data = financial_data.drop(
        columns=["Churn"]
    )

    business_data = predictions.merge(
        financial_data,
        on="customerID",
        how="left",
        validate="one_to_one",
    )

    if len(business_data) != len(predictions):

        raise ValueError(
            "Customer alignment changed row count."
        )

    if business_data[
        ["MonthlyCharges", "raw_churn"]
    ].isna().any().any():

        raise ValueError(
            "Missing customer financial information."
        )

    if not np.array_equal(
        business_data["actual_churn"],
        business_data["raw_churn"],
    ):

        raise ValueError(
            "Customer churn labels do not match."
        )

    business_data = business_data.drop(
        columns=["raw_churn"]
    )

    business_data["MonthlyCharges"] = (
        validate_monthly_charges(
            business_data["MonthlyCharges"]
        )
    )

    return business_data, metadata


def validate_monthly_charges(monthly_charges):
    """Validate customer monthly revenue."""

    charges = np.asarray(
        monthly_charges,
        dtype=float,
    )

    if charges.ndim != 1 or charges.size == 0:

        raise ValueError(
            "Monthly charges must be a "
            "non-empty one-dimensional array."
        )

    if not np.isfinite(charges).all():

        raise ValueError(
            "Monthly charges must be finite."
        )

    if np.any(charges < 0):

        raise ValueError(
            "Monthly charges cannot be negative."
        )

    return charges


def load_selected_methods():
    """Load preferred probability calibrators."""

    methods = pd.read_csv(
        SELECTED_METHODS_PATH
    )

    required = {
        "model",
        "calibration_method",
    }

    if not required.issubset(methods.columns):

        raise ValueError(
            "Missing calibration selection columns."
        )

    if methods["model"].duplicated().any():

        raise ValueError(
            "Duplicate preferred calibration methods."
        )

    if set(methods["model"]) != set(
        SUPPORTED_MODELS
    ):

        raise ValueError(
            "Preferred model membership mismatch."
        )

    if not methods[
        "calibration_method"
    ].isin(CALIBRATION_METHODS).all():

        raise ValueError(
            "Unsupported calibration method."
        )

    return dict(zip(
        methods["model"],
        methods["calibration_method"],
    ))


# --------------------------------------------------
# Customer-level financial valuation
# --------------------------------------------------


def calculate_retained_values(
    monthly_charges,
    config=BASELINE_CONFIG,
):
    """
    Potential retained contribution per customer.

    Monthly charges x retention months
    x contribution margin.
    """

    charges = validate_monthly_charges(
        monthly_charges
    )

    values = (
        charges
        * config.retained_value_multiplier
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Non-finite retained values."
        )

    return values


def calculate_expected_net_benefits(
    probabilities,
    monthly_charges,
    config=BASELINE_CONFIG,
):
    """
    Expected incremental net benefit of targeting
    each individual customer.

    p(churn) x retention success x retained value
    minus intervention cost.
    """

    probabilities = validate_probabilities(
        probabilities
    )

    values = calculate_retained_values(
        monthly_charges,
        config,
    )

    if len(probabilities) != len(values):

        raise ValueError(
            "Probability and financial data "
            "length mismatch."
        )

    return (
        probabilities
        * config.retention_success_rate
        * values
        - config.intervention_cost
    )


def calculate_break_even_probabilities(
    monthly_charges,
    config=BASELINE_CONFIG,
):
    """
    Customer-specific minimum churn probability
    required for profitable intervention.

    Values greater than 1 indicate that break-even
    cannot be achieved under the assumptions.

    Infinity indicates zero potential benefit.
    """

    values = calculate_retained_values(
        monthly_charges,
        config,
    )

    denominator = (
        config.retention_success_rate
        * values
    )

    return np.divide(
        config.intervention_cost,
        denominator,
        out=np.full(
            len(values),
            np.inf,
        ),
        where=denominator > 0,
    )


# --------------------------------------------------
# Campaign targeting strategies
# --------------------------------------------------


def select_threshold_targets(
    probabilities,
    threshold,
):
    """Target customers at or above a threshold."""

    probabilities = validate_probabilities(
        probabilities
    )

    threshold = validate_thresholds(
        [threshold]
    )[0]

    return probabilities >= threshold


def select_profitable_customers(
    probabilities,
    monthly_charges,
    config=BASELINE_CONFIG,
):
    """Target customers with positive expected benefit."""

    benefits = calculate_expected_net_benefits(
        probabilities,
        monthly_charges,
        config,
    )

    return benefits > 0


def select_budget_constrained_customers(
    probabilities,
    monthly_charges,
    customer_ids,
    config=BASELINE_CONFIG,
):
    """
    Select customers with the highest positive
    expected net benefit within the campaign budget.

    This ranking is optimal under constant
    per-customer intervention costs.
    """

    benefits = calculate_expected_net_benefits(
        probabilities,
        monthly_charges,
        config,
    )

    customer_ids = np.asarray(
        customer_ids,
        dtype=str,
    )

    if (
        customer_ids.ndim != 1
        or len(customer_ids) != len(benefits)
    ):

        raise ValueError(
            "Customer ID length mismatch."
        )

    if len(np.unique(customer_ids)) != len(
        customer_ids
    ):

        raise ValueError(
            "Duplicate customer IDs."
        )

    candidates = pd.DataFrame({
        "position": np.arange(len(benefits)),
        "customerID": customer_ids,
        "expected_net_benefit": benefits,
    })

    candidates = candidates[
        candidates["expected_net_benefit"] > 0
    ]

    candidates = candidates.sort_values(
        [
            "expected_net_benefit",
            "customerID",
        ],
        ascending=[False, True],
    )

    selected = candidates.head(
        config.maximum_budget_targets
    )

    targets = np.zeros(
        len(benefits),
        dtype=bool,
    )

    targets[
        selected["position"].to_numpy(dtype=int)
    ] = True

    return targets


# --------------------------------------------------
# Campaign financial evaluation
# --------------------------------------------------


def evaluate_campaign(
    y_true,
    probabilities,
    monthly_charges,
    targets,
    config=BASELINE_CONFIG,
):
    """
    Evaluate expected and label-based simulated
    financial outcomes for a targeting strategy.

    Actual churn labels are used only for evaluation,
    never for selecting individual campaign targets.
    """

    probabilities = validate_probabilities(
        probabilities
    )

    labels = validate_labels(
        y_true,
        expected_rows=len(probabilities),
    )

    values = calculate_retained_values(
        monthly_charges,
        config,
    )

    targets = np.asarray(targets)

    if (
        targets.ndim != 1
        or len(targets) != len(probabilities)
        or len(values) != len(probabilities)
    ):

        raise ValueError(
            "Campaign input length mismatch."
        )

    if not np.isin(
        targets,
        [False, True],
    ).all():

        raise ValueError(
            "Targets must contain binary values."
        )

    targets = targets.astype(bool)

    actual_churners = labels == 1

    tp = int(np.sum(
        targets & actual_churners
    ))

    fp = int(np.sum(
        targets & ~actual_churners
    ))

    fn = int(np.sum(
        ~targets & actual_churners
    ))

    tn = int(np.sum(
        ~targets & ~actual_churners
    ))

    customers_targeted = int(
        targets.sum()
    )

    campaign_cost = float(
        customers_targeted
        * config.intervention_cost
    )

    expected_churners_targeted = float(
        probabilities[targets].sum()
    )

    expected_retained_customers = (
        expected_churners_targeted
        * config.retention_success_rate
    )

    simulated_retained_customers = (
        tp * config.retention_success_rate
    )

    expected_retained_contribution = float(
        np.sum(
            probabilities[targets]
            * values[targets]
        )
        * config.retention_success_rate
    )

    simulated_retained_contribution = float(
        values[
            targets & actual_churners
        ].sum()
        * config.retention_success_rate
    )

    expected_net_profit = (
        expected_retained_contribution
        - campaign_cost
    )

    simulated_net_profit = (
        simulated_retained_contribution
        - campaign_cost
    )

    expected_roi = (
        expected_net_profit / campaign_cost
        if campaign_cost > 0
        else np.nan
    )

    simulated_roi = (
        simulated_net_profit / campaign_cost
        if campaign_cost > 0
        else np.nan
    )

    baseline_churn_loss = float(
        values[actual_churners].sum()
    )

    simulated_remaining_churn_loss = (
        baseline_churn_loss
        - simulated_retained_contribution
    )

    simulated_total_business_cost = (
        simulated_remaining_churn_loss
        + campaign_cost
    )

    expected_addressable_value = float(
        np.sum(
            probabilities[targets]
            * values[targets]
        )
    )

    simulated_addressable_value = float(
        values[
            targets & actual_churners
        ].sum()
    )

    expected_break_even_success_rate = (
        campaign_cost / expected_addressable_value
        if expected_addressable_value > 0
        else np.nan
    )

    simulated_break_even_success_rate = (
        campaign_cost / simulated_addressable_value
        if simulated_addressable_value > 0
        else np.nan
    )

    return {
        "total_customers": len(labels),

        "customers_targeted":
            customers_targeted,

        "targeting_rate":
            customers_targeted / len(labels),

        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,

        "precision":
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0,

        "recall":
            tp / (tp + fn),

        "campaign_cost":
            campaign_cost,

        "within_budget":
            bool(
                campaign_cost
                <= config.campaign_budget + 1e-9
            ),

        "expected_churners_targeted":
            expected_churners_targeted,

        "expected_retained_customers":
            expected_retained_customers,

        "simulated_retained_customers":
            simulated_retained_customers,

        "expected_retained_contribution":
            expected_retained_contribution,

        "simulated_retained_contribution":
            simulated_retained_contribution,

        "expected_net_profit":
            expected_net_profit,

        "simulated_net_profit":
            simulated_net_profit,

        "expected_roi":
            expected_roi,

        "simulated_roi":
            simulated_roi,

        "baseline_churn_loss":
            baseline_churn_loss,

        "simulated_remaining_churn_loss":
            simulated_remaining_churn_loss,

        "simulated_total_business_cost":
            simulated_total_business_cost,

        "expected_break_even_success_rate":
            expected_break_even_success_rate,

        "simulated_break_even_success_rate":
            simulated_break_even_success_rate,
    }


# --------------------------------------------------
# Profit-optimal threshold selection
# --------------------------------------------------


def sweep_business_thresholds(
    y_true,
    probabilities,
    monthly_charges,
    config=BASELINE_CONFIG,
    thresholds=None,
):
    """Evaluate financial performance across thresholds."""

    if thresholds is None:

        thresholds = DEFAULT_THRESHOLDS

    thresholds = validate_thresholds(
        thresholds
    )

    rows = []

    for threshold in thresholds:

        targets = select_threshold_targets(
            probabilities,
            threshold,
        )

        metrics = evaluate_campaign(
            y_true,
            probabilities,
            monthly_charges,
            targets,
            config,
        )

        rows.append({
            "threshold": float(threshold),
            **metrics,
        })

    return pd.DataFrame(rows)


def select_profit_optimal_threshold(
    results,
    budget_constrained=False,
):
    """
    Select a threshold maximizing expected profit.

    Label-based simulated profit is not used
    as a threshold selection criterion.
    """

    candidates = results.copy()

    if budget_constrained:

        candidates = candidates[
            candidates["within_budget"]
        ]

    if candidates.empty:

        return None

    candidates = candidates.sort_values(
        [
            "expected_net_profit",
            "campaign_cost",
            "threshold",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    )

    selected = candidates.iloc[0]

    # No intervention is preferable to a
    # negative or zero incremental return.

    if selected["expected_net_profit"] <= 0:

        return None

    return selected.to_dict()


# --------------------------------------------------
# Compare campaign strategies
# --------------------------------------------------


def evaluate_model_strategies(
    business_data,
    model_name,
    calibration_method,
    statistical_thresholds,
    config=BASELINE_CONFIG,
):
    """
    Compare statistical, financial and
    budget-constrained targeting strategies.
    """

    if model_name not in SUPPORTED_MODELS:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if calibration_method not in CALIBRATION_METHODS:

        raise ValueError(
            "Unsupported calibration method."
        )

    probability_column = (
        f"{model_name}_"
        f"{calibration_method}_probability"
    )

    probabilities = validate_probabilities(
        business_data[probability_column]
    )

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

    def add_strategy(
        strategy,
        targets,
        threshold=np.nan,
    ):

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
                calibration_method,

            "strategy": strategy,

            "threshold": threshold,

            **metrics,
        })

    # Baseline: no campaign.

    add_strategy(
        "no_intervention",
        np.zeros(len(labels), dtype=bool),
    )

    # Baseline: contact every customer.

    add_strategy(
        "target_everyone",
        np.ones(len(labels), dtype=bool),
    )

    # Conventional classification threshold.

    add_strategy(
        "default_threshold",
        select_threshold_targets(
            probabilities,
            0.50,
        ),
        threshold=0.50,
    )

    # Statistical threshold strategies from Phase 6.

    selected = statistical_thresholds[
        (
            statistical_thresholds["model"]
            == model_name
        )
        & (
            statistical_thresholds[
                "calibration_method"
            ] == calibration_method
        )
        & statistical_thresholds["feasible"]
    ]

    for _, row in selected.iterrows():

        threshold = float(
            row["threshold"]
        )

        add_strategy(
            row["strategy"],
            select_threshold_targets(
                probabilities,
                threshold,
            ),
            threshold=threshold,
        )

    # Financial threshold optimization.

    threshold_results = sweep_business_thresholds(
        labels,
        probabilities,
        charges,
        config,
    )

    threshold_results.insert(
        0,
        "calibration_method",
        calibration_method,
    )

    threshold_results.insert(
        0,
        "model",
        model_name,
    )

    for constrained, strategy in [
        (
            False,
            "profit_optimal_threshold",
        ),
        (
            True,
            "budget_optimal_threshold",
        ),
    ]:

        optimal = select_profit_optimal_threshold(
            threshold_results,
            budget_constrained=constrained,
        )

        if optimal is None:

            targets = np.zeros(
                len(labels),
                dtype=bool,
            )

            threshold = np.nan

        else:

            threshold = float(
                optimal["threshold"]
            )

            targets = select_threshold_targets(
                probabilities,
                threshold,
            )

        add_strategy(
            strategy,
            targets,
            threshold=threshold,
        )

    # Customer-specific expected-profit targeting.

    add_strategy(
        "customer_value_targeting",
        select_profitable_customers(
            probabilities,
            charges,
            config,
        ),
    )

    # Budget-constrained expected-profit targeting.

    add_strategy(
        "budget_value_targeting",
        select_budget_constrained_customers(
            probabilities,
            charges,
            customer_ids,
            config,
        ),
    )

    return (
        pd.DataFrame(rows),
        threshold_results,
    )


# --------------------------------------------------
# Customer financial summary
# --------------------------------------------------


def summarize_customer_values(
    business_data,
    config=BASELINE_CONFIG,
):
    """Summarize customer financial exposure."""

    data = business_data[
        [
            "MonthlyCharges",
            "actual_churn",
        ]
    ].copy()

    data["retained_contribution"] = (
        calculate_retained_values(
            data["MonthlyCharges"],
            config,
        )
    )

    rows = []

    for group_name, subset in [
        ("all_customers", data),

        (
            "churned_customers",
            data[data["actual_churn"] == 1],
        ),

        (
            "non_churned_customers",
            data[data["actual_churn"] == 0],
        ),
    ]:

        rows.append({
            "customer_group": group_name,

            "customer_count": len(subset),

            "mean_monthly_charges":
                subset["MonthlyCharges"].mean(),

            "median_monthly_charges":
                subset["MonthlyCharges"].median(),

            "mean_retained_contribution":
                subset[
                    "retained_contribution"
                ].mean(),

            "total_retained_contribution":
                subset[
                    "retained_contribution"
                ].sum(),
        })

    return pd.DataFrame(rows)


def summarize_break_even_probabilities(
    business_data,
    config=BASELINE_CONFIG,
):
    """Summarize customer-specific break-even cutoffs."""

    cutoffs = calculate_break_even_probabilities(
        business_data["MonthlyCharges"],
        config,
    )

    finite = cutoffs[
        np.isfinite(cutoffs)
    ]

    return pd.DataFrame([{
        "customer_count": len(cutoffs),

        "minimum_break_even_probability":
            float(finite.min())
            if len(finite)
            else np.nan,

        "median_break_even_probability":
            float(np.median(finite))
            if len(finite)
            else np.nan,

        "maximum_break_even_probability":
            float(finite.max())
            if len(finite)
            else np.nan,

        "mean_break_even_probability":
            float(finite.mean())
            if len(finite)
            else np.nan,

        "unachievable_break_even_customers":
            int(np.sum(cutoffs > 1)),
    }])


# --------------------------------------------------
# Model and strategy comparison
# --------------------------------------------------


def select_best_business_strategies(
    campaign_results,
):
    """
    Select best expected-profit strategies for
    each model, with and without a budget constraint.
    """

    rows = []

    for model_name in SUPPORTED_MODELS:

        subset = campaign_results[
            campaign_results["model"]
            == model_name
        ]

        if subset.empty:

            raise ValueError(
                f"Missing model results: {model_name}"
            )

        for selection, candidates in [
            (
                "unconstrained",
                subset,
            ),

            (
                "budget_constrained",
                subset[subset["within_budget"]],
            ),
        ]:

            if candidates.empty:

                raise ValueError(
                    "No feasible campaign strategies."
                )

            ranked = candidates.sort_values(
                [
                    "expected_net_profit",
                    "campaign_cost",
                    "strategy",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )

            best = ranked.iloc[0].to_dict()

            rows.append({
                "selection_type": selection,
                **best,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Report export
# --------------------------------------------------


def save_business_reports(
    campaign_results,
    threshold_results,
    best_strategies,
    customer_summary,
    break_even_summary,
    config,
    metadata,
    output_dir=REPORTS_DIR,
):
    """Export aggregate financial analysis reports."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {
        "campaign_strategy_comparison.csv":
            campaign_results,

        "business_threshold_sweep.csv":
            threshold_results,

        "best_business_strategies.csv":
            best_strategies,

        "customer_financial_summary.csv":
            customer_summary,

        "break_even_probability_summary.csv":
            break_even_summary,
    }

    for filename, report in reports.items():

        report.to_csv(
            output_dir / filename,
            index=False,
        )

    with open(
        output_dir / "business_assumptions.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config.to_dict(),
            file,
            indent=4,
            allow_nan=False,
        )

    with open(
        output_dir / "business_metadata.json",
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
        f"\nBusiness reports saved: {output_dir}"
    )


# --------------------------------------------------
# Main execution
# --------------------------------------------------


def main():

    config = BASELINE_CONFIG

    print(
        "\nLoading verified customer "
        "financial data..."
    )

    business_data, calibration_metadata = (
        load_business_data()
    )

    selected_methods = load_selected_methods()

    statistical_thresholds = pd.read_csv(
        SELECTED_THRESHOLDS_PATH
    )

    print("\nBUSINESS ANALYSIS CONFIGURATION")

    print(
        f"Validation customers: "
        f"{len(business_data):,}"
    )

    print(
        f"Intervention cost: "
        f"${config.intervention_cost:,.2f}"
    )

    print(
        f"Retention success rate: "
        f"{config.retention_success_rate:.0%}"
    )

    print(
        f"Retention duration: "
        f"{config.retention_months} months"
    )

    print(
        f"Contribution margin: "
        f"{config.contribution_margin:.0%}"
    )

    print(
        f"Campaign budget: "
        f"${config.campaign_budget:,.2f}"
    )

    print(
        f"Maximum budget targets: "
        f"{config.maximum_budget_targets:,}"
    )

    all_campaigns = []

    all_thresholds = []

    for model_name in SUPPORTED_MODELS:

        calibration_method = (
            selected_methods[model_name]
        )

        campaigns, thresholds = (
            evaluate_model_strategies(
                business_data,
                model_name,
                calibration_method,
                statistical_thresholds,
                config,
            )
        )

        all_campaigns.append(campaigns)

        all_thresholds.append(thresholds)

        print(
            f"Completed: {model_name} / "
            f"{calibration_method}"
        )

    campaign_results = pd.concat(
        all_campaigns,
        ignore_index=True,
    )

    threshold_results = pd.concat(
        all_thresholds,
        ignore_index=True,
    )

    best_strategies = (
        select_best_business_strategies(
            campaign_results
        )
    )

    customer_summary = summarize_customer_values(
        business_data,
        config,
    )

    break_even_summary = (
        summarize_break_even_probabilities(
            business_data,
            config,
        )
    )

    metadata = {
        "dataset_hash":
            calculate_file_hash(DATA_PATH),

        "calibrated_predictions_hash":
            calibration_metadata[
                "calibrated_predictions_hash"
            ],

        "selected_calibration_methods_hash":
            calculate_file_hash(
                SELECTED_METHODS_PATH
            ),

        "selected_thresholds_hash":
            calculate_file_hash(
                SELECTED_THRESHOLDS_PATH
            ),

        "validation_rows":
            len(business_data),

        "models":
            list(SUPPORTED_MODELS),

        "selected_calibration_methods":
            selected_methods,

        "threshold_count":
            len(DEFAULT_THRESHOLDS),

        "campaign_strategy_evaluations":
            len(campaign_results),

        "business_threshold_evaluations":
            len(threshold_results),

        "selection_objective":
            "expected_net_profit",

        "evaluation_partition":
            "validation",

        "financial_assumptions":
            "hypothetical_simulation",

        "customer_value_source":
            "original_monthly_charges",

        "intervention_cost_structure":
            "constant_cost_per_target",

        "causal_intervention_effect_estimated":
            False,

        "test_evaluated":
            False,

        "production_updated":
            False,
    }

    save_business_reports(
        campaign_results,
        threshold_results,
        best_strategies,
        customer_summary,
        break_even_summary,
        config,
        metadata,
    )

    display_columns = [
        "model",
        "selection_type",
        "strategy",
        "customers_targeted",
        "campaign_cost",
        "expected_net_profit",
        "simulated_net_profit",
        "expected_roi",
        "recall",
    ]

    print("\nBEST BUSINESS STRATEGIES")

    print(
        best_strategies[
            display_columns
        ].round(4).to_string(index=False)
    )

    print("\nCUSTOMER FINANCIAL SUMMARY")

    print(
        customer_summary.round(2).to_string(
            index=False
        )
    )

    print("\nBREAK-EVEN PROBABILITY SUMMARY")

    print(
        break_even_summary.round(4).to_string(
            index=False
        )
    )

    print(
        f"\nCampaign strategies evaluated: "
        f"{len(campaign_results)}"
    )

    print(
        f"Business thresholds evaluated: "
        f"{len(threshold_results)}"
    )

    print(
        "\nBusiness cost analysis completed."
    )

    print(
        "Financial outcomes are hypothetical."
    )

    print(
        "Production models remain unchanged."
    )

    print(
        "The test partition has not been "
        "evaluated during this analysis."
    )


if __name__ == "__main__":
    main()