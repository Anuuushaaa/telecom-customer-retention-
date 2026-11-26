"""Telecom Customer Retention Intelligence dashboard."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from frontend.api_client import (
    TelecomAPIClient,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
)
from frontend.batch_dashboard import render_batch_dashboard
from frontend.risk_dashboard import render_risk_dashboard
from frontend.retention_dashboard import render_retention_dashboard

DATA_PATH = Path("data/raw/telco_churn.csv")
YES_NO = ["Yes", "No"]
INTERNET_ADDONS = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Telecom Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_raw_data(path, modified_time):
    return pd.read_csv(
        path,
        dtype={
            "customerID": str,
            "TotalCharges": str,
        },
        keep_default_na=False,
    )
# ============================================================
# ERROR HANDLING
# ============================================================
def display_api_error(error):
    if isinstance(error, APITimeoutError):
        st.error(
            "The API request timed out. Please try again."
        )
    elif isinstance(error, APIConnectionError):
        st.error(
            "Cannot connect to the prediction API. "
            "Check that FastAPI is running."
        )
    else:
        st.error(str(error))
    if error.request_id:
        st.caption(
            f"Request ID: {error.request_id}"
        )
# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar(client):
    model_info = None
    with st.sidebar:
        st.title("Retention Intelligence")
        st.caption(
            "Machine Learning & Explainable AI"
        )
        st.divider()
        st.subheader("System Status")
        st.caption(
            f"API: {client.settings.api_url}"
        )
        st.button(
            "Refresh service status",
            width="stretch",
        )
        try:
            health = client.get_health()
            readiness = client.get_readiness()
            st.success(
                "Prediction API online"
            )
            st.caption(
                f"API version: {health['version']}"
            )
            if readiness["model_loaded"]:
                model_info = client.get_model_info()
                st.success("Model ready")
            else:
                st.warning("Model not ready")
        except APIClientError as error:
            display_api_error(error)
        st.divider()
        st.subheader("Active Model")
        if model_info:
            model_names = {
                "xgboost": "XGBoost",
                "logistic_regression": "Logistic Regression",
                "tensorflow_ann": "TensorFlow ANN",
            }
            st.write(
                "**Algorithm:**",
                model_names.get(
                    model_info["model_name"],
                    model_info["model_name"],
                ),
            )
            st.write(
                "**Registry version:**",
                model_info["model_version"],
            )
            st.write(
                "**Alias:**",
                model_info["model_alias"],
            )
            st.write(
                "**Calibration:**",
                model_info["calibration_method"].title(),
            )
            st.write(
                "**Registry:** MLflow"
            )
        else:
            st.caption(
                "Model information unavailable."
            )
        st.divider()
        st.subheader("Prediction Settings")
        default_threshold = (
            float(
                model_info["classification_threshold"]
            )
            if model_info
            else 0.50
        )
        custom_threshold = st.checkbox(
            "Override model threshold",
            value=False,
        )
        if custom_threshold:
            threshold = st.slider(
                "Classification threshold",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.01,
            )
        else:
            threshold = default_threshold
            st.metric(
                "Default threshold",
                f"{threshold:.2f}",
            )
        st.caption(
            "Lower thresholds generally identify more "
            "potential churners but may increase false positives."
        )
        top_n = st.slider(
            "SHAP features to display",
            min_value=5,
            max_value=29,
            value=10,
        )
        st.divider()
        st.caption(
            "Telecom Customer Retention 2.0"
        )
    return model_info, threshold, top_n
# ============================================================
# CUSTOMER LOOKUP
# ============================================================
def render_customer_lookup():
    st.subheader("Customer Lookup")
    try:
        dataframe = load_raw_data(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime_ns,
        )
    except (OSError, ValueError):
        st.warning(
            "The original customer dataset is unavailable. "
            "Use manual customer entry instead."
        )
        return None
    if (
        dataframe.empty
        or "customerID" not in dataframe.columns
    ):
        st.error(
            "The customer dataset is invalid."
        )
        return None
    customer_id = st.selectbox(
        "Select customer ID",
        dataframe["customerID"].tolist(),
    )
    row = dataframe.loc[
        dataframe["customerID"] == customer_id
    ].iloc[0]
    customer = row.drop(
        labels=["Churn"],
        errors="ignore",
    ).to_dict()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Tenure",
        f"{customer['tenure']} months",
    )
    col2.metric(
        "Monthly charges",
        f"${float(customer['MonthlyCharges']):,.2f}",
    )
    col3.metric(
        "Contract",
        customer["Contract"],
    )
    col4.metric(
        "Internet service",
        customer["InternetService"],
    )
    return customer
# ============================================================
# MANUAL CUSTOMER ENTRY
# ============================================================
def render_manual_entry():
    st.subheader("Enter Customer Details")
    customer = {}
    customer["customerID"] = st.text_input(
        "Customer ID",
        value="MANUAL-ENTRY",
        max_chars=128,
    ).strip()
    st.markdown("#### Customer profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        customer["gender"] = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )
    with col2:
        customer["SeniorCitizen"] = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda value: (
                "Yes" if value else "No"
            ),
        )
    with col3:
        customer["Partner"] = st.selectbox(
            "Partner",
            YES_NO,
        )
    with col4:
        customer["Dependents"] = st.selectbox(
            "Dependents",
            YES_NO,
        )
    st.markdown("#### Subscription and services")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer["tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            value=12,
            step=1,
        )
    with col2:
        customer["PhoneService"] = st.selectbox(
            "Phone service",
            YES_NO,
        )
        if customer["PhoneService"] == "No":
            customer["MultipleLines"] = (
                "No phone service"
            )
            st.caption(
                "Multiple lines: Not applicable"
            )
        else:
            customer["MultipleLines"] = st.selectbox(
                "Multiple lines",
                YES_NO,
            )
    with col3:
        customer["InternetService"] = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )
    addon_columns = st.columns(3)
    for index, (field, label) in enumerate(
        INTERNET_ADDONS.items()
    ):
        with addon_columns[index % 3]:
            if customer["InternetService"] == "No":
                customer[field] = (
                    "No internet service"
                )
                st.caption(
                    f"{label}: Not applicable"
                )
            else:
                customer[field] = st.selectbox(
                    label,
                    YES_NO,
                    key=f"manual_{field}",
                )
    st.markdown("#### Contract and billing")
    col1, col2, col3 = st.columns(3)
    with col1:
        customer["Contract"] = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
        )
    with col2:
        customer["PaperlessBilling"] = st.selectbox(
            "Paperless billing",
            YES_NO,
        )
    with col3:
        customer["PaymentMethod"] = st.selectbox(
            "Payment method",
            PAYMENT_METHODS,
        )
    col1, col2 = st.columns(2)
    with col1:
        customer["MonthlyCharges"] = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )
    with col2:
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            value=840.0,
            step=1.0,
        )
        customer["TotalCharges"] = str(
            total_charges
        )
    if not customer["customerID"]:
        st.warning(
            "Customer ID cannot be empty."
        )
        return None
    if (
        customer["tenure"] == 0
        and total_charges > 0
    ):
        st.warning(
            "Please verify the total charges for "
            "this zero-tenure customer."
        )
    return customer
# ============================================================
# RISK CLASSIFICATION
# ============================================================
def get_risk_band(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    if probability < 0.75:
        return "High"
    return "Very High"
# ============================================================
# PREDICTION RESULTS
# ============================================================
def render_prediction(prediction):
    st.subheader("Prediction Results")
    probability = float(
        prediction["churn_probability"]
    )
    raw_probability = float(
        prediction["raw_churn_probability"]
    )
    threshold = float(
        prediction["classification_threshold"]
    )
    risk_band = get_risk_band(probability)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Calibrated churn probability",
        f"{probability:.2%}",
    )
    col2.metric(
        "Predicted outcome",
        prediction["prediction_label"],
    )
    col3.metric(
        "Risk band",
        risk_band,
    )
    col4.metric(
        "Classification threshold",
        f"{threshold:.2f}",
    )
    st.progress(probability)
    if prediction["prediction"] == 1:
        st.warning(
            "This customer is classified as a potential "
            "churner under the selected threshold."
        )
    else:
        st.success(
            "This customer is classified as a non-churner "
            "under the selected threshold."
        )
    with st.expander(
        "Probability calibration details"
    ):
        col1, col2 = st.columns(2)
        col1.metric(
            "Raw model probability",
            f"{raw_probability:.2%}",
        )
        col2.metric(
            "Calibrated probability",
            f"{probability:.2%}",
        )
        st.write(
            "**Calibration method:**",
            prediction["calibration_method"],
        )
        st.caption(
            "Calibration adjusts the raw model output "
            "to better estimate churn probability."
        )
    st.caption(
        "Risk bands are descriptive dashboard categories, "
        "not independently validated decision thresholds."
    )
# ============================================================
# LOCAL SHAP VISUALIZATION
# ============================================================
def render_shap_explanation(explanation):
    st.subheader("Why This Prediction?")
    st.caption(
        "TreeSHAP explains the underlying XGBoost "
        "model in raw log-odds units, not direct "
        "percentage-point changes in calibrated probability."
    )
    contributors = pd.DataFrame(
        explanation["top_contributors"]
    )
    if contributors.empty:
        st.info(
            "No SHAP contributions were returned."
        )
        return
    contributors["absolute_shap"] = (
        contributors["shap_value"].abs()
    )
    contributors = contributors.sort_values(
        "absolute_shap",
        ascending=True,
    )
    contributors["Impact"] = (
        contributors["direction"].map({
            "increases_risk": "Increases churn score",
            "decreases_risk": "Decreases churn score",
            "neutral": "Neutral",
        })
    )
    figure = px.bar(
        contributors,
        x="shap_value",
        y="feature",
        color="Impact",
        orientation="h",
        color_discrete_map={
            "Increases churn score": "#D9534F",
            "Decreases churn score": "#198F76",
            "Neutral": "#9CA3AF",
        },
        hover_data={
            "transformed_value": ":.4f",
            "shap_value": ":.6f",
            "absolute_shap": False,
        },
        labels={
            "shap_value": "SHAP contribution (raw log-odds)",
            "feature": "Feature",
        },
    )
    figure.update_layout(
        height=max(
            400,
            len(contributors) * 30 + 120,
        ),
        legend_title_text="Model impact",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=20,
        ),
    )
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=contributors["feature"].tolist(),
    )
    figure.add_vline(
        x=0,
        line_color="gray",
        line_width=1,
    )
    st.plotly_chart(
        figure,
        width="stretch",
    )
    positive = contributors.loc[
        contributors["shap_value"] > 0
    ].sort_values(
        "shap_value",
        ascending=False,
    )
    negative = contributors.loc[
        contributors["shap_value"] < 0
    ].sort_values(
        "shap_value"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top risk-increasing features**")
        if positive.empty:
            st.caption(
                "None among the displayed features."
            )
        else:
            st.dataframe(
                positive[
                    ["feature", "shap_value"]
                ].head(3),
                hide_index=True,
                width="stretch",
            )
    with col2:
        st.markdown("**Top risk-decreasing features**")
        if negative.empty:
            st.caption(
                "None among the displayed features."
            )
        else:
            st.dataframe(
                negative[
                    ["feature", "shap_value"]
                ].head(3),
                hide_index=True,
                width="stretch",
            )
    with st.expander(
        "Exact SHAP contributions and verification"
    ):
        st.dataframe(
            contributors[
                [
                    "feature",
                    "transformed_value",
                    "shap_value",
                    "direction",
                ]
            ].sort_values(
                "shap_value",
                key=lambda values: values.abs(),
                ascending=False,
            ),
            hide_index=True,
            width="stretch",
        )
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Base log-odds",
            f"{explanation['base_value']:.6f}",
        )
        col2.metric(
            "Model raw margin",
            f"{explanation['raw_margin']:.6f}",
        )
        col3.metric(
            "Additivity error",
            f"{explanation['additivity_error']:.2e}",
        )
        st.write(
            "Remaining feature contribution:",
            f"{explanation['remaining_contribution']:.6f}",
        )
        st.write(
            "Reconstructed raw probability:",
            f"{explanation['reconstructed_raw_probability']:.6%}",
        )
        st.caption(
            "Transformed values represent encoded or "
            "scaled model inputs, not necessarily the "
            "original customer measurements."
        )
    st.caption(
        "SHAP explains model behavior. It does not "
        "establish causal relationships."
    )
# ============================================================
# SINGLE CUSTOMER DASHBOARD
# ============================================================
def render_single_customer(
    client,
    model_info,
    threshold,
    top_n,
):
    mode = st.radio(
        "Customer input mode",
        [
            "Lookup existing customer",
            "Enter new customer manually",
        ],
        horizontal=True,
    )
    if mode == "Lookup existing customer":
        customer = render_customer_lookup()
    else:
        customer = render_manual_entry()
    if customer is not None:
        with st.expander(
            "View customer information"
        ):
            st.json(customer)
    st.divider()
    include_shap = st.checkbox(
        "Include SHAP explanation",
        value=True,
    )
    request_context = {
        "customer": customer,
        "threshold": threshold,
        "top_n": top_n,
        "include_shap": include_shap,
        "model_info": model_info,
    }
    previous = st.session_state.get(
        "single_customer_result"
    )
    if (
        previous is not None
        and previous["context"] != request_context
    ):
        st.session_state.pop(
            "single_customer_result",
            None,
        )
    if st.button(
        "Predict Churn Risk",
        type="primary",
        width="stretch",
        disabled=(
            customer is None
            or model_info is None
        ),
    ):
        st.session_state.pop(
            "single_customer_result",
            None,
        )
        try:
            with st.spinner(
                "Analyzing customer churn risk..."
            ):
                explanation = None
                if (
                    include_shap
                    and model_info["model_name"] == "xgboost"
                ):
                    explanation = client.explain(
                        customer=customer,
                        classification_threshold=threshold,
                        top_n=top_n,
                    )
                    prediction = explanation[
                        "prediction"
                    ]
                    if str(
                        explanation["model_version"]
                    ) != str(
                        model_info["model_version"]
                    ):
                        raise APIClientError(
                            "The active model changed. "
                            "Refresh and try again."
                        )
                else:
                    prediction = client.predict(
                        customer=customer,
                        classification_threshold=threshold,
                    )
                if client.get_model_info() != model_info:
                    raise APIClientError(
                        "Model metadata changed during "
                        "prediction. Please try again."
                    )
                st.session_state[
                    "single_customer_result"
                ] = {
                    "context": request_context,
                    "prediction": prediction,
                    "explanation": explanation,
                }
        except APIClientError as error:
            display_api_error(error)
    result = st.session_state.get(
        "single_customer_result"
    )
    if result is not None:
        render_prediction(
            result["prediction"]
        )
        if result["explanation"] is not None:
            st.divider()
            render_shap_explanation(
                result["explanation"]
            )
        elif include_shap:
            st.info(
                "TreeSHAP explanations are currently "
                "available for XGBoost only."
            )
        st.divider()
        st.subheader("Interpretation")
        if result["prediction"]["prediction"] == 1:
            st.write(
                "The retention team may prioritize this "
                "customer for further review."
            )
        else:
            st.write(
                "This customer does not meet the current "
                "churn classification threshold."
            )
        st.caption(
            "Predictions are statistical estimates, "
            "not guarantees of future customer behavior."
        )
# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
def main():
    client = TelecomAPIClient()
    st.title("Telecom Customer Retention Intelligence")
    st.markdown(
        "Calibrated churn prediction, explainable AI "
        "and customer retention decision support."
    )
    model_info, threshold, top_n = render_sidebar(client)
    for key in ("single_customer_result", "batch_prediction_result"):
        saved = st.session_state.get(key)
        if saved is not None:
            context = saved["context"]
            if (context["model_info"] != model_info
                    or context["threshold"] != threshold):
                st.session_state.pop(key, None)
    if model_info is None:
        st.warning("Prediction service unavailable. Check MLflow and FastAPI, then refresh service status.")
    st.divider()
    single, batch, risk, retention = st.tabs([
        "Single Customer", "Batch Prediction", "Risk Analytics", "Retention Simulator"
    ])
    with single:
        render_single_customer(client, model_info, threshold, top_n)
    with batch:
        render_batch_dashboard(client, model_info, threshold)
    with risk:
        render_risk_dashboard(model_info, threshold)
    with retention:
        render_retention_dashboard(model_info, threshold)


if __name__ == "__main__":
    main()
