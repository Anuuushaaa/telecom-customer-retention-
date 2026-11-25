
"""
src/train.py

Train Logistic Regression and XGBoost models for telecom churn.

Workflow:
1. Split data into training, validation and test sets.
2. Fit preprocessing using training data only.
3. Train both models on training data.
4. Evaluate both models on validation data.
5. Save candidate models and preprocessing artifacts.

The test set is reserved for final evaluation.
"""

from pathlib import Path

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

from src.preprocessing import (
    fit_transform_features,
    prepare_raw_data,
    split_train_validation_test,
)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train class-weighted Logistic Regression."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train XGBoost with class weighting calculated from training data."""
    counts = y_train.value_counts()
    neg = counts[0]
    pos = counts[1]

    scale_pos_weight = neg / pos

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a model on the supplied dataset.

    X_test/y_test are parameter names retained for compatibility.
    The caller determines whether this is validation or test data.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'=' * 50}")
    print(f"{model_name} | Threshold: {threshold:.2f}")
    print("=" * 50)

    for key, value in metrics.items():
        if key != "model":
            print(f"{key:>12}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    return metrics


def main():
    # Load and clean the original dataset
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = prepare_raw_data(df_raw, require_target=True)
    print("Training data validation passed.")

    # Stratified 60/20/20 split
    train_df, val_df, test_df = split_train_validation_test(
        df_clean
    )

    # Fit preprocessing on training data only
    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    print("\nDataset sizes:")
    print(f"Training:   {len(data.X_train)}")
    print(f"Validation: {len(data.X_val)}")
    print(f"Test:       {len(data.X_test)}")

    # Train candidate models
    lr_model = train_logistic_regression(
        data.X_train, data.y_train
    )

    xgb_model = train_xgboost(
        data.X_train, data.y_train
    )

    # Validation evaluation at a common baseline threshold
    lr_metrics = evaluate_model(
        lr_model,
        data.X_val,
        data.y_val,
        "Logistic Regression",
    )

    xgb_metrics = evaluate_model(
        xgb_model,
        data.X_val,
        data.y_val,
        "XGBoost",
    )

    results_df = pd.DataFrame(
        [lr_metrics, xgb_metrics]
    ).set_index("model")

    print(f"\n{'=' * 50}")
    print("VALIDATION MODEL COMPARISON")
    print("=" * 50)
    print(results_df.round(4))

    # Preserve existing production artifacts.
    # These are candidate models until threshold tuning is complete.
    Path("models/candidates").mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        lr_model,
        "models/candidates/logistic_regression.pkl",
    )
    joblib.dump(
        xgb_model,
        "models/candidates/xgboost.pkl",
    )
    joblib.dump(
        data.scaler,
        "models/candidates/scaler.pkl",
    )
    joblib.dump(
        data.feature_columns,
        "models/candidates/feature_columns.pkl",
    )

    print("\nCandidate models and preprocessing artifacts saved.")
    print("Production models have not been overwritten.")
    print("Test set remains reserved for final evaluation.")


if __name__ == "__main__":
    main()
