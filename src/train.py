"""
src/train.py

Train and compare three telecom churn candidate models:
- Logistic Regression
- XGBoost
- TensorFlow ANN

Workflow:
1. Validate and clean the original dataset.
2. Create stratified training, validation and test partitions.
3. Fit preprocessing on training data only.
4. Train all three candidate models.
5. Evaluate models on validation data.
6. Save candidate models and preprocessing artifacts.
7. Generate training and evaluation reports.

The test set is reserved for final evaluation.
Production models remain unchanged.
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
    average_precision_score,
)

from xgboost import XGBClassifier

from src.preprocessing import (
    fit_transform_features,
    prepare_raw_data,
    split_train_validation_test,
)

from src.train_ann import (
    train_ann,
    evaluate_ann,
    save_ann_model,
    save_training_history,
    plot_learning_curves,
    save_validation_metrics,
)


RANDOM_STATE = 42
BASELINE_THRESHOLD = 0.50

DATA_PATH = Path("data/raw/telco_churn.csv")
CANDIDATE_DIR = Path("models/candidates")
REPORTS_DIR = Path("reports")


def train_logistic_regression(
    X_train,
    y_train,
) -> LogisticRegression:
    """Train class-weighted Logistic Regression."""

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(
    X_train,
    y_train,
) -> XGBClassifier:
    """Train XGBoost using training-only class weights."""

    counts = y_train.value_counts()

    neg = counts[0]
    pos = counts[1]

    scale_pos_weight = neg / pos

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    threshold: float = BASELINE_THRESHOLD,
) -> dict:
    """
    Evaluate a model on the supplied dataset.

    X_test and y_test parameter names are retained
    for backward compatibility.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    metrics = {
        "model": model_name,

        "accuracy": accuracy_score(
            y_test, y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_proba,
        ),

        "average_precision": average_precision_score(
            y_test,
            y_proba,
        ),
    }

    print(f"\n{'=' * 50}")
    print(f"{model_name} | Threshold: {threshold:.2f}")
    print("=" * 50)

    for key, value in metrics.items():

        if key != "model":
            print(f"{key:>20}: {value:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No Churn",
                "Churn",
            ],
            zero_division=0,
        )
    )

    return metrics


def main():

    print("=" * 60)
    print("TELECOM CUSTOMER RETENTION")
    print("CANDIDATE MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load and validate the original dataset
    # --------------------------------------------------

    df_raw = pd.read_csv(DATA_PATH)

    df_clean = prepare_raw_data(
        df_raw,
        require_target=True,
    )

    print("\nTraining data validation passed.")
    print(f"Total customer records: {len(df_clean)}")

    # --------------------------------------------------
    # 2. Stratified 60/20/20 split
    # --------------------------------------------------

    train_df, val_df, test_df = (
        split_train_validation_test(
            df_clean,
            random_state=RANDOM_STATE,
        )
    )

    # --------------------------------------------------
    # 3. Fit preprocessing on training data only
    # --------------------------------------------------

    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    print("\nDataset sizes:")

    print(f"Training:   {len(data.X_train)}")
    print(f"Validation: {len(data.X_val)}")
    print(f"Test:       {len(data.X_test)}")

    print(
        f"Number of features: "
        f"{len(data.feature_columns)}"
    )

    print("\nTraining class distribution:")

    print(
        data.y_train.value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # 4. Train Logistic Regression
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    lr_model = train_logistic_regression(
        data.X_train,
        data.y_train,
    )

    # --------------------------------------------------
    # 5. Train XGBoost
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    xgb_model = train_xgboost(
        data.X_train,
        data.y_train,
    )

    # --------------------------------------------------
    # 6. Train TensorFlow ANN
    # --------------------------------------------------

    print("\nTraining TensorFlow ANN...")

    ann_model, ann_history = train_ann(
        X_train=data.X_train,
        y_train=data.y_train,
        X_val=data.X_val,
        y_val=data.y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    # --------------------------------------------------
    # 7. Evaluate on validation data only
    # --------------------------------------------------

    lr_metrics = evaluate_model(
        lr_model,
        data.X_val,
        data.y_val,
        "Logistic Regression",
        threshold=BASELINE_THRESHOLD,
    )

    xgb_metrics = evaluate_model(
        xgb_model,
        data.X_val,
        data.y_val,
        "XGBoost",
        threshold=BASELINE_THRESHOLD,
    )

    ann_metrics = evaluate_ann(
        ann_model,
        data.X_val,
        data.y_val,
        threshold=BASELINE_THRESHOLD,
        model_name="TensorFlow ANN",
    )

    # --------------------------------------------------
    # 8. Compare validation performance
    # --------------------------------------------------

    results_df = pd.DataFrame(
        [
            lr_metrics,
            xgb_metrics,
            ann_metrics,
        ]
    ).set_index("model")

    print(f"\n{'=' * 70}")
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(results_df.round(4).to_string())

    # --------------------------------------------------
    # 9. Create output directories
    # --------------------------------------------------

    CANDIDATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 10. Save candidate models
    # --------------------------------------------------

    joblib.dump(
        lr_model,
        CANDIDATE_DIR / "logistic_regression.pkl",
    )

    joblib.dump(
        xgb_model,
        CANDIDATE_DIR / "xgboost.pkl",
    )

    save_ann_model(
        ann_model,
        CANDIDATE_DIR / "tensorflow_ann.keras",
    )

    # --------------------------------------------------
    # 11. Save shared preprocessing artifacts
    # --------------------------------------------------

    joblib.dump(
        data.scaler,
        CANDIDATE_DIR / "scaler.pkl",
    )

    joblib.dump(
        data.feature_columns,
        CANDIDATE_DIR / "feature_columns.pkl",
    )

    # --------------------------------------------------
    # 12. Generate ANN reports
    # --------------------------------------------------

    history_df = save_training_history(
        ann_history,
        REPORTS_DIR / "ann_training_history.csv",
    )

    plot_learning_curves(
        ann_history,
        REPORTS_DIR / "ann_learning_curves.png",
    )

    save_validation_metrics(
        ann_metrics,
        REPORTS_DIR / "ann_validation_metrics.csv",
    )

    # --------------------------------------------------
    # 13. Save combined comparison report
    # --------------------------------------------------

    results_df.to_csv(
        REPORTS_DIR / "validation_model_comparison.csv"
    )

    # --------------------------------------------------
    # 14. Training summary
    # --------------------------------------------------

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nANN epochs completed: "
        f"{len(history_df)}"
    )

    print("\nCandidate models saved:")
    print(" - Logistic Regression")
    print(" - XGBoost")
    print(" - TensorFlow ANN")

    print("\nPreprocessing artifacts saved.")

    print("\nReports generated:")
    print(" - ann_training_history.csv")
    print(" - ann_learning_curves.png")
    print(" - ann_validation_metrics.csv")
    print(" - validation_model_comparison.csv")

    print("\nProduction models have not been overwritten.")
    print("Test set remains reserved for final evaluation.")


if __name__ == "__main__":
    main()