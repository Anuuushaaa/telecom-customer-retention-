
"""
Telecom Customer Retention Intelligence
Streamlit application for predicting customer churn.

Uses predict.py for preprocessing, model inference, and SHAP explanations.
"""

import pandas as pd
import streamlit as st

from predict import predict_customer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📞",
    layout="centered",
)

st.title("📞 Telecom Customer Retention Intelligence")

st.markdown(
    "Predict customer churn risk and understand **why** using "
    "machine learning and SHAP explanations."
)

st.divider()


# --------------------------------------------------
# LOAD CUSTOMER DATA
# --------------------------------------------------

@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")


try:
    df_raw = load_raw_data()
except Exception as error:
    st.error(f"Unable to load customer dataset: {error}")
    st.stop()


# --------------------------------------------------
# INPUT MODE SELECTION
# --------------------------------------------------

mode = st.radio(
    "Choose input mode:",
    [
        "Lookup existing customer",
        "Enter new customer manually",
    ],
)

raw_customer = None


# --------------------------------------------------
# MODE 1: EXISTING CUSTOMER
# --------------------------------------------------

if mode == "Lookup existing customer":

    st.subheader("Customer Lookup")

    customer_id = st.selectbox(
        "Select customer ID:",
        df_raw["customerID"].tolist(),
    )

    raw_row = df_raw.loc[
        df_raw["customerID"] == customer_id
    ].iloc[0]

    raw_customer = raw_row.drop(
        labels=["Churn"],
        errors="ignore",
    ).to_dict()

    with st.expander("View customer information"):
        st.json(raw_customer)


# --------------------------------------------------
# MODE 2: MANUAL CUSTOMER ENTRY
# --------------------------------------------------

else:

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
        )

    # Ensure manually entered services are consistent.

    if phone_service == "No":
        multiple_lines = "No phone service"

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

    raw_customer = {
        "customerID": "MANUAL-ENTRY",
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": str(total_charges),
    }

    with st.expander("View entered customer information"):
        st.json(raw_customer)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True,
):

    if raw_customer is None:
        st.warning("Please provide customer information.")
        st.stop()

    try:

        with st.spinner("Analyzing customer churn risk..."):
            result = predict_customer(raw_customer)

        st.divider()

        st.subheader("Prediction Results")

        # ------------------------------------------
        # PREDICTION SUMMARY
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            if result["prediction"] == "Churn":

                st.error(
                    "⚠️ **High Churn Risk**"
                )

            else:

                st.success(
                    "✅ **Predicted to Stay**"
                )

        with col2:

            st.metric(
                "Churn Probability",
                f"{result['churn_probability']:.1%}",
            )

        st.caption(
            f"Classification threshold: "
            f"{result['threshold_used']:.2f} "
            "(selected using validation data to balance "
            "recall and precision)"
        )

        # ------------------------------------------
        # RISK INDICATOR
        # ------------------------------------------

        st.subheader("Customer Churn Risk")

        probability = result["churn_probability"]

        st.progress(
            min(max(float(probability), 0.0), 1.0)
        )

        st.write(
            f"Estimated churn probability: "
            f"**{probability:.1%}**"
        )

        # ------------------------------------------
        # SHAP EXPLANATIONS
        # ------------------------------------------

        st.divider()

        st.subheader("Why This Prediction?")

        st.markdown(
            "SHAP explains which customer characteristics "
            "influenced the model's prediction."
        )

        st.caption(
            "Positive SHAP values push the prediction "
            "toward churn. Negative values push it "
            "away from churn."
        )

        contributors = result["top_contributors"]

        if contributors:

            chart_df = pd.DataFrame(
                {
                    "Feature": list(contributors.keys()),
                    "SHAP Contribution": [
                        float(value)
                        for value in contributors.values()
                    ],
                }
            )

            chart_df = chart_df.set_index("Feature")

            # Native Streamlit chart avoids the
            # Matplotlib image-rendering issue.

            st.bar_chart(
                chart_df,
                horizontal=True,
                x_label="SHAP contribution (log-odds)",
                y_label="Feature",
            )

            with st.expander(
                "View exact SHAP contributions"
            ):
                st.dataframe(
                    chart_df,
                    use_container_width=True,
                )

        else:

            st.info(
                "No SHAP contributions were returned."
            )

        # ------------------------------------------
        # INTERPRETATION
        # ------------------------------------------

        st.divider()

        st.subheader("Interpretation")

        if result["prediction"] == "Churn":

            st.warning(
                "This customer is predicted to churn "
                "under the selected classification threshold. "
                "The retention team may prioritize this "
                "customer for further review."
            )

        else:

            st.success(
                "This customer is predicted not to churn "
                "under the selected classification threshold."
            )

        st.caption(
            "Predictions are statistical estimates, "
            "not guarantees. SHAP contributions explain "
            "the model's behavior rather than proving "
            "causal relationships."
        )

    except Exception as error:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(error)
