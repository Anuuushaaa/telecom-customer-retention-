"""
predict.py

Phase 7: Inference pipeline for a single new customer.

Reuses the exact cleaning logic from src/cleaning.py, and reuses the
FITTED scaler/feature columns from training (never re-fits anything) to
avoid training-serving skew. This is the file that will also back the
Streamlit app.
"""

import joblib
import pandas as pd
from src.feature_engineering import engineer_features
from src.config import CLASSIFICATION_THRESHOLD, PRODUCTION_MODEL_PATH
from src.preprocessing import prepare_raw_data, transform_features


def predict_customer(raw_customer: dict) -> dict:
    """
    Takes a single raw customer record (same shape as one row of the
    original CSV, minus Churn) and returns a prediction + explanation.
    """
    model = joblib.load(PRODUCTION_MODEL_PATH)
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")

    df_raw = pd.DataFrame([raw_customer])
    df_clean = prepare_raw_data(df_raw, require_target=False)
    df_engineered = engineer_features(df_clean)
    X = transform_features(df_engineered, scaler, feature_columns)

    proba = model.predict_proba(X)[0, 1]
    prediction = int(proba >= CLASSIFICATION_THRESHOLD)

    shap_values = explainer.shap_values(X)
    contributions = pd.Series(shap_values[0], index=X.columns).sort_values(
        key=abs, ascending=False
    )

    return {
        "customerID": raw_customer.get("customerID", "unknown"),
        "churn_probability": round(float(proba), 4),
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "threshold_used": CLASSIFICATION_THRESHOLD,
        "top_contributors": contributions.head(5).round(4).to_dict(),
    }


if __name__ == "__main__":
    # Example: a high-risk profile (month-to-month, low tenure, fiber optic)
    sample_customer = {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": "190.0",
    }

    result = predict_customer(sample_customer)
    print(result)