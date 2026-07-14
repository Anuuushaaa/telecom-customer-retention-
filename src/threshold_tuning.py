"""
src/threshold_tuning.py

Phase 5b: Threshold tuning.

Default classification threshold (0.5) is arbitrary — it doesn't reflect
that we value recall over precision for this business problem. This script
sweeps thresholds and reports metrics at each, so we can pick an operating
point deliberately instead of accepting sklearn's default.
"""

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.cleaning import clean_data
from src.preprocessing import fit_transform_features, split_data


def sweep_thresholds(model, X_test, y_test, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    y_proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        })
    return pd.DataFrame(rows)


def main():
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)
    train_df, test_df = split_data(df_clean)
    data = fit_transform_features(train_df, test_df)

    lr_model = joblib.load("models/logistic_regression.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")

    print("Logistic Regression — threshold sweep")
    print(sweep_thresholds(lr_model, data.X_test, data.y_test).round(4))

    print("\nXGBoost — threshold sweep")
    print(sweep_thresholds(xgb_model, data.X_test, data.y_test).round(4))


if __name__ == "__main__":
    main()