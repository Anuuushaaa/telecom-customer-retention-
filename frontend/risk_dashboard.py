
"""Streamlit portfolio risk and explainability dashboard."""

import plotly.express as px
import streamlit as st

from frontend.risk_analytics import (
    RISK_ORDER,
    SEGMENT_COLUMNS,
    prepare_analytics_data,
    calculate_portfolio_summary,
    calculate_risk_distribution,
    calculate_segment_summary,
    get_priority_customers,
    load_global_shap_report,
    shap_report_matches_model,
)


RISK_COLORS = {
    "Low": "#198F76",
    "Moderate": "#EAB308",
    "High": "#F97316",
    "Very High": "#DC2626",
}


# ============================================================
# PORTFOLIO METRICS
# ============================================================

def render_portfolio_metrics(summary):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total customers",
        f"{summary['total_customers']:,}",
    )

    col2.metric(
        "Predicted churn rate",
        f"{summary['predicted_churn_rate']:.2%}",
    )

    col3.metric(
        "Average churn probability",
        f"{summary['average_churn_probability']:.2%}",
    )

    col4.metric(
        "Expected churn count",
        f"{summary['expected_churn_count']:,.1f}",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Monthly portfolio revenue",
        f"${summary['total_monthly_revenue']:,.2f}",
    )

    col2.metric(
        "Revenue from flagged customers",
        f"${summary['flagged_monthly_revenue']:,.2f}",
    )

    col3.metric(
        "Probability-weighted revenue exposure",
        f"${summary['expected_monthly_revenue_loss']:,.2f}",
    )

    st.caption(
        "Revenue exposure is calculated as churn "
        "probability multiplied by monthly charges. "
        "It is a scenario-based estimate, not an "
        "observed or guaranteed financial loss."
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

def render_risk_distribution(
    dataframe,
    threshold,
):

    distribution = calculate_risk_distribution(
        dataframe
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Customer Risk Distribution"
        )

        figure = px.pie(
            distribution,
            names="risk_band",
            values="customers",
            hole=0.55,
            color="risk_band",
            color_discrete_map=RISK_COLORS,
            category_orders={
                "risk_band": RISK_ORDER
            },
        )

        figure.update_traces(
            textposition="inside",
            textinfo="percent",
        )

        figure.update_layout(
            height=420,
            legend_title_text="Risk band",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    with col2:

        st.subheader(
            "Churn Probability Distribution"
        )

        figure = px.histogram(
            dataframe,
            x="churn_probability",
            nbins=30,
            labels={
                "churn_probability":
                    "Calibrated churn probability"
            },
            color_discrete_sequence=[
                "#3978A8"
            ],
        )

        figure.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=(
                f"Threshold: {threshold:.2f}"
            ),
        )

        figure.update_xaxes(
            range=[0, 1],
            tickformat=".0%",
        )

        figure.update_layout(
            height=420,
            yaxis_title="Customers",
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    with st.expander(
        "Risk distribution details"
    ):

        st.dataframe(
            distribution,
            hide_index=True,
            width="stretch",
        )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

def render_segment_analysis(dataframe):

    st.subheader("Customer Segment Analysis")

    segment = st.selectbox(
        "Segment customers by",
        list(SEGMENT_COLUMNS),
        format_func=lambda value: (
            SEGMENT_COLUMNS[value]
        ),
    )

    summary = calculate_segment_summary(
        dataframe,
        segment,
    )

    display = summary.copy()

    display[segment] = (
        display[segment].astype(str)
    )

    col1, col2 = st.columns(2)

    with col1:

        figure = px.bar(
            display,
            x=segment,
            y="average_probability",
            color="average_probability",
            color_continuous_scale="OrRd",
            range_color=[0, 1],
            hover_data=[
                "customers",
                "predicted_churn",
            ],
            title="Average Churn Probability",
            labels={
                segment: SEGMENT_COLUMNS[segment],
                "average_probability":
                    "Average churn probability",
            },
        )

        figure.update_yaxes(
            range=[0, 1],
            tickformat=".0%",
        )

        figure.update_layout(
            height=430,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    with col2:

        figure = px.bar(
            display,
            x=segment,
            y="expected_revenue_loss",
            color_discrete_sequence=[
                "#3978A8"
            ],
            title="Revenue Exposure by Segment",
            labels={
                segment: SEGMENT_COLUMNS[segment],
                "expected_revenue_loss":
                    "Revenue exposure ($)",
            },
        )

        figure.update_layout(
            height=430,
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

    st.dataframe(
        summary,
        hide_index=True,
        width="stretch",
        column_config={
            "average_probability":
                st.column_config.NumberColumn(
                    "Average churn probability",
                    format="percent",
                ),

            "predicted_churn_rate":
                st.column_config.NumberColumn(
                    "Predicted churn rate",
                    format="percent",
                ),

            "customer_share":
                st.column_config.NumberColumn(
                    "Customer share",
                    format="percent",
                ),

            "expected_revenue_loss":
                st.column_config.NumberColumn(
                    "Revenue exposure",
                    format="dollar",
                ),
        },
    )

    st.download_button(
        "Download segment analysis",
        data=summary.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"segment_analysis_{segment}.csv"
        ),
        mime="text/csv",
    )

    st.caption(
        "Segment differences describe model predictions "
        "and do not establish causal churn drivers."
    )


# ============================================================
# PRIORITY CUSTOMER TABLE
# ============================================================

def render_priority_customers(dataframe):

    st.subheader("Customer Revenue Exposure Ranking")

    limit = st.selectbox(
        "Customers to display",
        [10, 25, 50, 100],
        index=1,
    )

    priority = get_priority_customers(
        dataframe,
        limit=limit,
    )

    columns = [
        "customerID",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "churn_probability",
        "risk_band",
        "expected_monthly_revenue_loss",
    ]

    st.dataframe(
        priority[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability":
                st.column_config.ProgressColumn(
                    "Churn probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),

            "expected_monthly_revenue_loss":
                st.column_config.NumberColumn(
                    "Revenue exposure",
                    format="dollar",
                ),
        },
    )

    st.download_button(
        "Download ranked customers",
        data=priority[columns].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="customer_revenue_exposure.csv",
        mime="text/csv",
    )

    st.caption(
        "This ranking measures revenue exposure, "
        "not expected retention campaign profitability. "
        "The retention simulator applies intervention "
        "costs, success rates and budget constraints."
    )


# ============================================================
# GLOBAL SHAP DASHBOARD
# ============================================================

def render_global_shap(model_info):

    st.header("Global Model Explainability")

    try:

        importance, metadata = (
            load_global_shap_report()
        )

    except (
        OSError,
        ValueError,
        KeyError,
    ) as error:

        st.warning(str(error))

        return

    matches = shap_report_matches_model(
        metadata,
        model_info,
    )

    if matches:

        st.success(
            "SHAP report matches the active model."
        )

    else:

        st.warning(
            "This report belongs to a previously "
            "explained model or the active model "
            "could not be verified."
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Explained model",
        metadata["model_name"].upper(),
    )

    col2.metric(
        "Model version",
        metadata["model_version"],
    )

    col3.metric(
        "Validation customers",
        f"{metadata['validation_customers']:,}",
    )

    col4.metric(
        "Features",
        metadata["feature_count"],
    )

    top_n = st.slider(
        "Number of global features",
        min_value=1,
        max_value=len(importance),
        value=min(15, len(importance)),
        key="global_shap_top_n",
    )

    display = importance.head(
        top_n
    ).sort_values(
        "mean_absolute_shap"
    )

    figure = px.bar(
        display,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        color="mean_absolute_shap",
        color_continuous_scale="Blues",
        hover_data={
            "mean_absolute_shap": ":.6f",
            "mean_signed_shap": ":.6f",
            "rank": True,
        },
        labels={
            "mean_absolute_shap":
                "Mean absolute SHAP (raw log-odds)",
            "feature": "Feature",
        },
    )

    figure.update_layout(
        height=max(
            450,
            top_n * 30 + 100,
        ),
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    st.caption(
        "Mean absolute SHAP measures the average "
        "magnitude of each feature's contribution "
        "to the underlying model score."
    )

    st.subheader(
        "Mean Signed Feature Contributions"
    )

    signed = importance.head(
        top_n
    ).copy()

    signed["Direction"] = signed[
        "mean_signed_shap"
    ].map(
        lambda value: (
            "Positive"
            if value > 0
            else "Negative"
            if value < 0
            else "Neutral"
        )
    )

    signed = signed.sort_values(
        "mean_signed_shap"
    )

    figure = px.bar(
        signed,
        x="mean_signed_shap",
        y="feature",
        color="Direction",
        orientation="h",
        color_discrete_map={
            "Positive": "#D9534F",
            "Negative": "#198F76",
            "Neutral": "#9CA3AF",
        },
        labels={
            "mean_signed_shap":
                "Mean signed SHAP (raw log-odds)",
            "feature": "Feature",
        },
    )

    figure.add_vline(
        x=0,
        line_color="gray",
    )

    figure.update_layout(
        height=max(
            450,
            top_n * 30 + 100,
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    st.caption(
        "Signed averages may cancel positive and "
        "negative individual contributions. They "
        "are not causal effects or universal "
        "directions of feature influence."
    )

    with st.expander(
        "SHAP report verification and metadata"
    ):

        st.metric(
            "Maximum additivity error",
            f"{metadata['max_additivity_error']:.2e}",
        )

        st.json(metadata)

    st.dataframe(
        importance,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download global SHAP importance",
        data=importance.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="global_shap_importance.csv",
        mime="text/csv",
    )

    st.caption(
        "This report explains the original validation "
        "cohort, not the currently uploaded customer "
        "portfolio."
    )


# ============================================================
# DASHBOARD ENTRY POINT
# ============================================================

def render_risk_dashboard(
    model_info,
    threshold,
):

    st.header("Customer Risk Analytics")

    portfolio_tab, shap_tab = st.tabs([
        "Portfolio Analytics",
        "Global SHAP",
    ])

    with portfolio_tab:

        saved = st.session_state.get(
            "batch_prediction_result"
        )

        if saved is None:

            st.info(
                "Generate batch predictions before "
                "opening portfolio analytics."
            )

        elif (
            saved["context"]["model_info"]
            != model_info
            or saved["context"]["threshold"]
            != threshold
        ):

            st.warning(
                "The model or threshold has changed. "
                "Regenerate batch predictions."
            )

        else:

            try:

                dataframe = prepare_analytics_data(
                    saved["results"]
                )

                summary = (
                    calculate_portfolio_summary(
                        dataframe
                    )
                )

                render_portfolio_metrics(
                    summary
                )

                st.divider()

                render_risk_distribution(
                    dataframe,
                    threshold,
                )

                st.divider()

                render_segment_analysis(
                    dataframe
                )

                st.divider()

                render_priority_customers(
                    dataframe
                )

            except (
                ValueError,
                KeyError,
            ) as error:

                st.error(str(error))

    with shap_tab:

        render_global_shap(model_info)
