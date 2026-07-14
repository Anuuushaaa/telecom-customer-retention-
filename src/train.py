"""
src/train.py

Phase 5: Model training & evaluation.

Trains two models on the same preprocessed data:
- Logistic Regression (baseline, interpretable) with class_weight='balanced'
- XGBoost (stronger, industry-standard for tabular data) with scale_pos_weight

Both use loss-reweighting (not oversampling) to handle the 73/26 class
imbalance — no synthetic data, so every training example is a real customer.

Evaluation prioritizes recall/precision/F1/ROC-AUC over raw accuracy, since
accuracy is misleading on imbalanced data (see README Phase 5 section).
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.cleaning import clean_data
from src.preprocessing import fit_transform_features, split_data


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    # scale_pos_weight = (# negative) / (# positive) in the TRAINING set only,
    # consistent with fitting everything on train to avoid leakage.
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute the full metric set (not just accuracy) and return as a dict
    so results from multiple models can be compared side by side."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*50}\n{model_name}\n{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.4f}")
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def main():
    df_raw = pd.read_csv("data/raw/telco_churn.csv")  # adjust filename if needed
    df_clean = clean_data(df_raw)

    train_df, test_df = split_data(df_clean)
    data = fit_transform_features(train_df, test_df)

    # --- Logistic Regression ---
    lr_model = train_logistic_regression(data.X_train, data.y_train)
    lr_metrics = evaluate_model(lr_model, data.X_test, data.y_test, "Logistic Regression")

    # --- XGBoost ---
    xgb_model = train_xgboost(data.X_train, data.y_train)
    xgb_metrics = evaluate_model(xgb_model, data.X_test, data.y_test, "XGBoost")

    # --- Comparison summary ---
    results_df = pd.DataFrame([lr_metrics, xgb_metrics]).set_index("model")
    print(f"\n{'='*50}\nModel Comparison\n{'='*50}")
    print(results_df.round(4))

    # --- Save artifacts ---
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    joblib.dump(xgb_model, "models/xgboost.pkl")
    joblib.dump(data.scaler, "models/scaler.pkl")
    joblib.dump(data.feature_columns, "models/feature_columns.pkl")
    print("\nModels saved to models/")


if __name__ == "__main__":
    main()