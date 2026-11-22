
"""
src/explain.py

SHAP explainability for the selected Logistic Regression model.

Uses the same training split and preprocessing as src/train.py.

Outputs:
1. Global SHAP feature importance
2. SHAP beeswarm summary plot
3. Individual customer waterfall explanation
4. Saved SHAP explainer for inference

The explainer is saved alongside the candidate model.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.cleaning import clean_data
from src.preprocessing import (
    fit_transform_features,
    split_train_validation_test,
)


FIGURES_DIR = Path("reports/figures")
CANDIDATES_DIR = Path("models/candidates")


def build_explainer(model, X_train):
    """
    Build a SHAP LinearExplainer using training data
    as the background distribution.
    """
    return shap.LinearExplainer(
        model,
        X_train.astype(float),
    )


def global_summary(explainer, X_data):
    """Generate global feature importance and a beeswarm plot."""

    X_data = X_data.astype(float)

    shap_values = explainer.shap_values(X_data)

    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0),
        index=X_data.columns,
    ).sort_values(ascending=False)

    print("\nGlobal feature importance (mean absolute SHAP):")
    print(mean_abs_shap.round(4))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        X_data,
        show=False,
    )

    fig = plt.gcf()
    fig.tight_layout()

    output_path = FIGURES_DIR / "shap_summary_plot.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved global SHAP plot: {output_path}")

    return shap_values, mean_abs_shap


def explain_customer(explainer, X_row, customer_id):
    """Generate an individual customer SHAP explanation."""

    X_row = X_row.astype(float)

    shap_values = explainer.shap_values(X_row)

    contributions = pd.Series(
        shap_values[0],
        index=X_row.columns,
    ).sort_values(
        key=abs,
        ascending=False,
    )

    print(f"\nExplanation for customer {customer_id}:")
    print(contributions.round(4))

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_row.iloc[0].values,
        feature_names=list(X_row.columns),
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.waterfall(
        explanation,
        show=False,
    )

    fig = plt.gcf()
    fig.tight_layout()

    safe_id = str(customer_id).replace("/", "_").replace("\\", "_")

    output_path = FIGURES_DIR / f"shap_waterfall_{safe_id}.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved customer SHAP plot: {output_path}")

    return contributions


def main():
    # Load and clean data
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_clean = clean_data(df_raw)

    # Recreate the exact split used during training
    train_df, val_df, test_df = split_train_validation_test(
        df_clean
    )

    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    # Load the selected candidate model
    model = joblib.load(
        CANDIDATES_DIR / "logistic_regression.pkl"
    )

    # Build explainer using TRAINING data only
    explainer = build_explainer(
        model,
        data.X_train,
    )

    # Global explanation on validation data
    global_summary(
        explainer,
        data.X_val,
    )

    # Local explanation for one validation customer
    sample_idx = 0

    sample_row = data.X_val.iloc[[sample_idx]]
    sample_customer_id = data.customer_ids_val.iloc[sample_idx]

    explain_customer(
        explainer,
        sample_row,
        sample_customer_id,
    )

    # Save the new candidate explainer
    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = CANDIDATES_DIR / "shap_explainer.pkl"

    joblib.dump(
        explainer,
        output_path,
    )

    print(f"\nSHAP explainer saved: {output_path}")
    print("Existing production explainer has not been overwritten.")


if __name__ == "__main__":
    main()
