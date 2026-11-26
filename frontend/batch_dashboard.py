
"""Streamlit batch prediction dashboard."""

import hashlib

import streamlit as st

from frontend.api_client import APIClientError

from frontend.batch_prediction import (
    load_uploaded_csv,
    validate_batch_dataframe,
    run_batch_prediction,
    summarize_batch,
    export_predictions,
)


def render_batch_dashboard(
    client,
    model_info,
    threshold,
):

    st.header("Batch Customer Prediction")

    st.markdown(
        "Upload customer records to generate calibrated "
        "churn predictions for an entire customer portfolio."
    )

    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        key="batch_customer_upload",
    )

    if uploaded is None:

        st.session_state.pop(
            "batch_prediction_result",
            None,
        )

        st.info(
            "Upload the original telecom dataset or "
            "another CSV with the same customer schema."
        )

        return

    file_content = uploaded.getvalue()

    file_hash = hashlib.sha256(
        file_content
    ).hexdigest()

    context = {
        "file_hash": file_hash,
        "model_info": model_info,
        "threshold": threshold,
    }

    previous = st.session_state.get(
        "batch_prediction_result"
    )

    if (
        previous is not None
        and previous["context"] != context
    ):

        st.session_state.pop(
            "batch_prediction_result",
            None,
        )

    try:

        dataframe = load_uploaded_csv(
            file_content
        )

        customers, ignored = (
            validate_batch_dataframe(
                dataframe
            )
        )

    except ValueError as error:

        st.error(str(error))

        return

    st.success(
        f"{len(customers):,} customer records "
        "validated successfully."
    )

    if ignored:

        st.info(
            "Excluded from model input: "
            + ", ".join(ignored)
        )

    with st.expander(
        "Preview uploaded dataset"
    ):

        st.dataframe(
            dataframe.head(20),
            hide_index=True,
            width="stretch",
        )

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )

    if st.button(
        "Generate Batch Predictions",
        type="primary",
        width="stretch",
        disabled=model_info is None,
    ):

        st.session_state.pop(
            "batch_prediction_result",
            None,
        )

        progress = st.progress(0.0)

        status = st.empty()

        def update_progress(completed, total):

            progress.progress(
                completed / total
            )

            status.info(
                f"Processed {completed:,} "
                f"of {total:,} customers."
            )

        try:

            results = run_batch_prediction(
                client=client,
                customers=customers,
                threshold=threshold,
                model_info=model_info,
                progress_callback=update_progress,
            )

            st.session_state[
                "batch_prediction_result"
            ] = {
                "context": context,
                "results": results,
            }

            progress.progress(1.0)

            status.success(
                "Batch prediction completed."
            )

        except (
            APIClientError,
            ValueError,
        ) as error:

            progress.empty()

            status.error(str(error))

            if getattr(
                error,
                "request_id",
                None,
            ):

                st.caption(
                    f"Request ID: {error.request_id}"
                )

    saved = st.session_state.get(
        "batch_prediction_result"
    )

    if saved is None:

        return

    results = saved["results"]

    summary = summarize_batch(results)

    st.divider()

    st.subheader("Batch Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customers",
        f"{summary['total_customers']:,}",
    )

    col2.metric(
        "Predicted churn",
        f"{summary['predicted_churn']:,}",
    )

    col3.metric(
        "Predicted no churn",
        f"{summary['predicted_no_churn']:,}",
    )

    col4.metric(
        "Predicted churn rate",
        f"{summary['predicted_churn_rate']:.2%}",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average churn probability",
        f"{summary['average_churn_probability']:.2%}",
    )

    col2.metric(
        "Expected churn count",
        f"{summary['expected_churn_count']:,.1f}",
    )

    col3.metric(
        "High / very high risk",
        f"{summary['high_risk_customers']:,}",
    )

    st.caption(
        "Expected churn count is the sum of calibrated "
        "probabilities, not the number of customers "
        "classified as churners."
    )

    st.subheader("Customer Predictions")

    risk_filter = st.multiselect(
        "Filter by risk band",
        [
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        default=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
    )

    display = results.loc[
        results["risk_band"].isin(risk_filter)
    ].sort_values(
        "churn_probability",
        ascending=False,
    )

    st.dataframe(
        display[
            [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "churn_probability",
                "prediction_label",
                "risk_band",
            ]
        ],
        column_config={
            "churn_probability":
                st.column_config.ProgressColumn(
                    "Churn probability",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
        },
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download all predictions (CSV)",
        data=export_predictions(results),
        file_name="telecom_churn_predictions.csv",
        mime="text/csv",
        width="stretch",
    )
