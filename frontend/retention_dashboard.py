
"""Interactive Streamlit retention campaign simulator."""

import json

import numpy as np
import plotly.express as px
import streamlit as st

from src.business_config import (
    BusinessConfig,
    BASELINE_CONFIG,
)

from frontend.retention_simulator import (
    STRATEGY_NAMES,
    compare_campaign_strategies,
    run_campaign_sensitivity,
)


# ============================================================
# CAMPAIGN CONFIGURATION
# ============================================================

def render_campaign_configuration():

    st.subheader("Campaign Financial Assumptions")

    st.caption(
        "Adjust hypothetical intervention costs "
        "and expected retention effectiveness."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        budget = st.number_input(
            "Campaign budget ($)",
            min_value=0.0,
            value=BASELINE_CONFIG.campaign_budget,
            step=500.0,
        )

        contact_cost = st.number_input(
            "Contact cost per customer ($)",
            min_value=0.0,
            value=BASELINE_CONFIG.contact_cost,
            step=1.0,
        )

    with col2:

        incentive_cost = st.number_input(
            "Incentive cost per customer ($)",
            min_value=0.0,
            value=BASELINE_CONFIG.incentive_cost,
            step=1.0,
        )

        success_rate = st.slider(
            "Retention success rate (%)",
            min_value=0,
            max_value=100,
            value=int(
                BASELINE_CONFIG.retention_success_rate
                * 100
            ),
        )

    with col3:

        margin = st.slider(
            "Contribution margin (%)",
            min_value=0,
            max_value=100,
            value=int(
                BASELINE_CONFIG.contribution_margin
                * 100
            ),
        )

        months = st.slider(
            "Retention duration (months)",
            min_value=1,
            max_value=36,
            value=BASELINE_CONFIG.retention_months,
        )

    try:

        config = BusinessConfig(
            contact_cost=float(contact_cost),
            incentive_cost=float(incentive_cost),
            retention_success_rate=success_rate / 100,
            retention_months=int(months),
            contribution_margin=margin / 100,
            campaign_budget=float(budget),
        )

    except ValueError as error:

        st.error(str(error))

        return None

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Intervention cost",
        f"${config.intervention_cost:,.2f}",
    )

    col2.metric(
        "Maximum affordable targets",
        f"{config.maximum_budget_targets:,}",
    )

    col3.metric(
        "Retained value multiplier",
        f"{config.retained_value_multiplier:.2f}×",
    )

    return config


# ============================================================
# STRATEGY COMPARISON
# ============================================================

def render_strategy_comparison(comparison):

    st.subheader("Campaign Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:

        figure = px.bar(
            comparison,
            x="strategy_name",
            y="expected_net_profit",
            color="within_budget",
            color_discrete_map={
                True: "#198F76",
                False: "#DC2626",
            },
            title="Expected Net Profit",
            labels={
                "strategy_name": "Strategy",
                "expected_net_profit":
                    "Expected net profit ($)",
                "within_budget": "Within budget",
            },
        )

        figure.add_hline(
            y=0,
            line_color="gray",
        )

        figure.update_layout(
            height=430,
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    with col2:

        figure = px.bar(
            comparison,
            x="strategy_name",
            y=[
                "campaign_cost",
                "expected_retained_contribution",
            ],
            barmode="group",
            title="Campaign Cost vs Retained Contribution",
            labels={
                "strategy_name": "Strategy",
                "value": "Amount ($)",
                "variable": "Financial metric",
            },
        )

        figure.update_layout(
            height=430,
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    columns = [
        "strategy_name",
        "targeted_customers",
        "campaign_cost",
        "expected_retained_customers",
        "expected_retained_contribution",
        "expected_net_profit",
        "expected_roi",
        "within_budget",
    ]

    st.dataframe(
        comparison[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "campaign_cost":
                st.column_config.NumberColumn(
                    "Campaign cost",
                    format="dollar",
                ),

            "expected_retained_contribution":
                st.column_config.NumberColumn(
                    "Expected retained contribution",
                    format="dollar",
                ),

            "expected_net_profit":
                st.column_config.NumberColumn(
                    "Expected net profit",
                    format="dollar",
                ),

            "expected_roi":
                st.column_config.NumberColumn(
                    "Expected ROI",
                    format="percent",
                ),
        },
    )

    feasible = comparison.loc[
        comparison["within_budget"]
    ]

    if not feasible.empty:

        best = feasible.sort_values(
            "expected_net_profit",
            ascending=False,
        ).iloc[0]

        st.success(
            "Highest expected profit within budget: "
            f"{best['strategy_name']} — "
            f"${best['expected_net_profit']:,.2f}"
        )

    st.caption(
        "Threshold and unconstrained profit strategies "
        "may exceed the campaign budget. The "
        "budget-constrained strategy enforces it."
    )

    st.download_button(
        "Download strategy comparison",
        data=comparison.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="retention_strategy_comparison.csv",
        mime="text/csv",
    )


# ============================================================
# SELECTED CAMPAIGN
# ============================================================

def render_selected_campaign(
    comparison,
    target_tables,
    config,
):

    st.subheader("Campaign Targeting Plan")

    strategy = st.selectbox(
        "Select campaign strategy",
        list(STRATEGY_NAMES),
        index=2,
        format_func=lambda value: (
            STRATEGY_NAMES[value]
        ),
    )

    summary = comparison.loc[
        comparison["strategy"] == strategy
    ].iloc[0]

    targets = target_tables[strategy]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Targeted customers",
        f"{int(summary['targeted_customers']):,}",
    )

    col2.metric(
        "Expected retained customers",
        f"{summary['expected_retained_customers']:,.1f}",
    )

    col3.metric(
        "Campaign cost",
        f"${summary['campaign_cost']:,.2f}",
    )

    col4.metric(
        "Expected net profit",
        f"${summary['expected_net_profit']:,.2f}",
    )

    col1, col2, col3, col4 = st.columns(4)

    roi = summary["expected_roi"]

    col1.metric(
        "Expected ROI",
        f"{roi:.2%}"
        if roi is not None and np.isfinite(roi)
        else "N/A",
    )

    utilization = summary["budget_utilization"]

    col2.metric(
        "Budget utilization",
        f"{utilization:.2%}"
        if (
            utilization is not None
            and np.isfinite(utilization)
        )
        else "N/A",
    )

    col3.metric(
        "Remaining budget",
        f"${summary['budget_remaining']:,.2f}",
    )

    col4.metric(
        "Expected churn coverage",
        f"{summary['expected_churn_coverage']:.2%}",
    )

    if not summary["within_budget"]:

        st.warning(
            "This strategy exceeds the campaign budget."
        )

    if summary["unprofitable_targets"] > 0:

        st.warning(
            f"{int(summary['unprofitable_targets']):,} "
            "selected customers have non-positive "
            "expected intervention benefits."
        )

    if targets.empty:

        st.info(
            "No customers qualify for this campaign "
            "under the current assumptions."
        )

        return

    columns = [
        "campaign_rank",
        "customerID",
        "MonthlyCharges",
        "churn_probability",
        "expected_retention_probability",
        "expected_retained_contribution",
        "intervention_cost",
        "expected_net_benefit",
        "break_even_probability",
    ]

    display = targets[columns].copy()

    display["break_even_probability"] = (
        display["break_even_probability"]
        .replace([np.inf, -np.inf], np.nan)
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability":
                st.column_config.NumberColumn(
                    "Churn probability",
                    format="percent",
                ),

            "expected_retention_probability":
                st.column_config.NumberColumn(
                    "Expected retention probability",
                    format="percent",
                ),

            "expected_net_benefit":
                st.column_config.NumberColumn(
                    "Expected net benefit",
                    format="dollar",
                ),

            "break_even_probability":
                st.column_config.NumberColumn(
                    "Break-even probability",
                    format="percent",
                ),
        },
    )

    st.caption(
        "Break-even probabilities above 100% indicate "
        "that intervention cannot break even under "
        "the current assumptions. Missing break-even "
        "values indicate zero potential benefit."
    )

    st.download_button(
        "Download campaign targeting plan",
        data=targets.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=f"retention_targets_{strategy}.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download financial assumptions",
        data=json.dumps(
            config.to_dict(),
            indent=4,
        ),
        file_name="retention_campaign_config.json",
        mime="application/json",
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def render_sensitivity_analysis(
    results,
    config,
):

    st.subheader("Campaign Sensitivity Analysis")

    st.markdown(
        "Explore how retention effectiveness and "
        "incentive costs influence the expected "
        "profitability of budget-constrained targeting."
    )

    scenarios = run_campaign_sensitivity(
        results,
        config,
    )

    matrix = scenarios.pivot(
        index="retention_success_rate",
        columns="incentive_cost",
        values="expected_net_profit",
    ).sort_index()

    matrix.index = [
        f"{value:.0%}"
        for value in matrix.index
    ]

    matrix.columns = [
        f"${value:,.0f}"
        for value in matrix.columns
    ]

    figure = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels={
            "x": "Incentive cost",
            "y": "Retention success rate",
            "color": "Expected profit ($)",
        },
    )

    figure.update_layout(
        height=480,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    best = scenarios.loc[
        scenarios["expected_net_profit"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Highest scenario profit",
        f"${best['expected_net_profit']:,.2f}",
    )

    col2.metric(
        "Scenario success rate",
        f"{best['retention_success_rate']:.0%}",
    )

    col3.metric(
        "Scenario incentive cost",
        f"${best['incentive_cost']:,.2f}",
    )

    st.caption(
        "Each scenario independently reselects "
        "profitable customers within the budget. "
        "Higher assumed success rates are not "
        "evidence that such rates are achievable."
    )

    with st.expander(
        "View all sensitivity scenarios"
    ):

        st.dataframe(
            scenarios,
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Download sensitivity analysis",
        data=scenarios.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="campaign_sensitivity_analysis.csv",
        mime="text/csv",
    )


# ============================================================
# DASHBOARD ENTRY POINT
# ============================================================

def render_retention_dashboard(
    model_info,
    threshold,
):

    st.header("Business Retention Simulator")

    st.markdown(
        "Translate calibrated churn probabilities "
        "into financially informed retention campaigns."
    )

    saved = st.session_state.get(
        "batch_prediction_result"
    )

    if saved is None:

        st.info(
            "Generate batch predictions before "
            "running a retention simulation."
        )

        return

    if (
        saved["context"]["model_info"] != model_info
        or saved["context"]["threshold"] != threshold
    ):

        st.warning(
            "The active model or threshold has changed. "
            "Regenerate batch predictions."
        )

        return

    results = saved["results"]

    st.caption(
        f"Portfolio size: {len(results):,} customers"
    )

    st.warning(
        "All financial parameters are hypothetical "
        "simulation assumptions. Predicted churn "
        "probabilities do not establish the causal "
        "effectiveness of retention interventions."
    )

    config = render_campaign_configuration()

    if config is None:

        return

    try:

        comparison, target_tables, economics = (
            compare_campaign_strategies(
                results=results,
                threshold=threshold,
                config=config,
            )
        )

        st.divider()

        render_strategy_comparison(
            comparison
        )

        st.divider()

        render_selected_campaign(
            comparison,
            target_tables,
            config,
        )

        st.divider()

        render_sensitivity_analysis(
            results,
            config,
        )

        with st.expander(
            "Customer-level financial estimates"
        ):

            st.dataframe(
                economics[
                    [
                        "customerID",
                        "churn_probability",
                        "MonthlyCharges",
                        "potential_retained_contribution",
                        "expected_net_benefit",
                        "profitable_intervention",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

    except (
        ValueError,
        KeyError,
    ) as error:

        st.error(str(error))
