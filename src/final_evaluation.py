
"""
src/final_evaluation.py

Final held-out test evaluation.

Selected model: Logistic Regression
Selected threshold: 0.50

Model and threshold were selected using validation data.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.cleaning import clean_data
from src.preprocessing import (
    fit_transform_features,
    split_train_validation_test,
)
from src.train import evaluate_model


def main():
    # Recreate the original deterministic data split
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

    # Load the candidate selected using validation data
    model = joblib.load(
        "models/candidates/logistic_regression.pkl"
    )

    threshold = 0.50

    print("\nFINAL HELD-OUT TEST EVALUATION")

    evaluate_model(
        model,
        data.X_test,
        data.y_test,
        "Logistic Regression - Final Test",
        threshold=threshold,
    )

    probabilities = model.predict_proba(data.X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        data.y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    print("\nCONFUSION MATRIX")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nFinal evaluation completed.")


if __name__ == "__main__":
    main()
