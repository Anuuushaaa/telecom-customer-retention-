"""
src/explain.py

Phase 6: SHAP explainability for the Logistic Regression production model.

Uses shap.LinearExplainer, the correct explainer for linear models (as
opposed to TreeExplainer for tree-based models, or the slower
model-agnostic KernelExplainer). Produces:
1. Global summary (text + beeswarm plot) — feature importance across all
   test customers, and a sanity check that the model's reasoning matches
   domain intuition.
2. Per-customer local explanation (text + waterfall plot) — the actionable
   output for the retention-team use case.

Plots are saved as PNG files under reports/figures/, since this script runs
headless (no notebook to render inline plots into).
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for headless PNG saving
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.cleaning import clean_data
from src.preprocessing import fit_transform_features, split_data

FIGURES_DIR = "reports/figures"


def build_explainer(model, X_train):
    """LinearExplainer needs the training data to establish a background
    distribution (baseline) against which SHAP values are computed."""
    return shap.LinearExplainer(model, X_train)


def global_summary(explainer, X_test):
    """Compute SHAP values for the whole test set, print mean absolute SHAP
    value per feature, and save a beeswarm summary plot."""
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print("\nGlobal feature importance (mean |SHAP value|):")
    print(mean_abs_shap.round(4))

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    summary_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {summary_path}")

    return shap_values, mean_abs_shap


def explain_customer(explainer, X_row: pd.DataFrame, customer_id: str):
    """Explain a single customer's prediction: print the text breakdown and
    save a waterfall plot. X_row must be a single-row dataframe with the
    same columns/order as training data."""
    shap_values = explainer.shap_values(X_row)
    contributions = pd.Series(
        shap_values[0], index=X_row.columns
    ).sort_values(key=abs, ascending=False)

    print(f"\nExplanation for customer {customer_id}:")
    print("(positive = pushes toward churn, negative = pushes away from churn)\n")
    print(contributions.round(4))

    # shap.plots.waterfall expects a shap.Explanation object, not a raw
    # array — we build one manually here since LinearExplainer.shap_values()
    # returns a plain numpy array rather than an Explanation object.
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_row.iloc[0].values,
        feature_names=list(X_row.columns),
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    safe_id = str(customer_id).replace("/", "_")
    waterfall_path = os.path.join(FIGURES_DIR, f"shap_waterfall_{safe_id}.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved waterfall plot to {waterfall_path}")

    return contributions


def main():
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)
    train_df, test_df = split_data(df_clean)
    data = fit_transform_features(train_df, test_df)

    model = joblib.load("models/logistic_regression.pkl")
    explainer = build_explainer(model, data.X_train)

    # 1. Global summary — text + beeswarm plot
    shap_values, mean_abs_shap = global_summary(explainer, data.X_test)

    # 2. Per-customer explanation — text + waterfall plot, first test customer as demo
    sample_idx = 0
    sample_row = data.X_test.iloc[[sample_idx]]
    sample_customer_id = data.customer_ids_test.iloc[sample_idx]
    explain_customer(explainer, sample_row, sample_customer_id)

    # Save explainer for reuse in predict.py / app.py
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("\nExplainer saved to models/shap_explainer.pkl")


if __name__ == "__main__":
    main()