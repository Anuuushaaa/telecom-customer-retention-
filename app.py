"""
app.py

Phase 8: Streamlit app for the Telecom Customer Retention project.

This is a THIN WRAPPER around predict.py — it introduces no new modeling
or encoding logic, to avoid reopening the training-serving skew problem
solved in Phase 7. Two input modes:
1. Lookup an existing customer by customerID (realistic retention-team use)
2. Manually enter a hypothetical customer profile
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predict import predict_customer

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")

st.title("📞 Telecom Customer Retention Intelligence")
st.markdown(
    "Predicts customer churn risk and explains **why**, using SHAP — "
    "built to support proactive retention outreach."
)

# --- Load raw data once, for the lookup mode ---
@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")

df_raw = load_raw_data()

# --- Mode selector ---
mode = st.radio("Choose input mode:", ["Lookup existing customer", "Enter new customer manually"])

raw_customer = None

if mode == "Lookup existing customer":
    customer_id = st.selectbox("Select customerID:", df_raw["customerID"].tolist())
    raw_row = df_raw[df_raw["customerID"] == customer_id].iloc[0]
    raw_customer = raw_row.drop(labels=["Churn"]).to_dict()

    with st.expander("View raw customer data"):
        st.json(raw_customer)

else:
    st.subheader("Enter customer details")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 840.0)

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

# --- Predict button ---
if st.button("Predict Churn Risk", type="primary"):
    result = predict_customer(raw_customer)

    st.divider()

    # --- Prediction display ---
    col1, col2 = st.columns(2)
    with col1:
        if result["prediction"] == "Churn":
            st.error(f"⚠️ Prediction: **{result['prediction']}**")
        else:
            st.success(f"✅ Prediction: **{result['prediction']}**")
    with col2:
        st.metric("Churn Probability", f"{result['churn_probability']:.1%}")

    st.caption(f"Decision threshold: {result['threshold_used']} (tuned in Phase 5 to prioritize recall)")

    # --- SHAP explanation bar chart ---
    st.subheader("Why this prediction?")
    st.caption("Positive values push toward churn, negative values push away from churn.")

    contributors = result["top_contributors"]
    fig, ax = plt.subplots(figsize=(7, 4))
    features = list(contributors.keys())
    values = list(contributors.values())
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn prediction)")
    ax.invert_yaxis()
    st.pyplot(fig)