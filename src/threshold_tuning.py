
"""
src/threshold_tuning.py

Compare classification thresholds using VALIDATION data only.

The held-out test set must not be used for threshold selection.
"""

import joblib
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.cleaning import clean_data
from src.preprocessing import (
    fit_transform_features,
    split_train_validation_test,
)


def sweep_thresholds(model, X_val, y_val, thresholds=None) -> pd.DataFrame:
    """Evaluate candidate thresholds on validation data."""

    if thresholds is None:
        thresholds = [
            0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60,
        ]

    y_proba = model.predict_proba(X_val)[:, 1]

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_val, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_val, y_pred, zero_division=0
            ),
        })

    return pd.DataFrame(rows)


def main():
    # Recreate the same deterministic splits used during training
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)

    train_df, val_df, test_df = split_train_validation_test(
        df_clean
    )

    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    # Load candidate models, not existing production models
    lr_model = joblib.load(
        "models/candidates/logistic_regression.pkl"
    )

    xgb_model = joblib.load(
        "models/candidates/xgboost.pkl"
    )

    print("\nLOGISTIC REGRESSION — VALIDATION THRESHOLDS")
    lr_results = sweep_thresholds(
        lr_model, data.X_val, data.y_val
    )
    print(lr_results.round(4).to_string(index=False))

    print("\nXGBOOST — VALIDATION THRESHOLDS")
    xgb_results = sweep_thresholds(
        xgb_model, data.X_val, data.y_val
    )
    print(xgb_results.round(4).to_string(index=False))

    print("\nThreshold tuning completed using validation data.")
    print("The test set has not been evaluated.")


if __name__ == "__main__":
    main()
