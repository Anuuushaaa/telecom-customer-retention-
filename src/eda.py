"""
src/eda.py

Phase 4: Exploratory Data Analysis for the Telecom Churn project.

Generates and saves the visualizations called for in the original project
brief: target distribution, numeric distributions, numeric-vs-target
boxplots, categorical-vs-target countplots, and a correlation heatmap.

Run standalone: python -m src.eda
All plots saved to reports/figures/eda/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.cleaning import clean_data

FIGURES_DIR = "reports/figures/eda"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_VS_TARGET_COLS = ["Contract", "InternetService", "PaymentMethod"]


def _ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame):
    """Pie chart + countplot of the Churn target — establishes the class
    imbalance visually, not just as printed numbers."""
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    df["Churn"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=axes[0], labels=["No", "Yes"], colors=["#4C72B0", "#DD8452"]
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Churn Distribution (Pie)")

    sns.countplot(data=df, x="Churn", hue="Churn", ax=axes[1],
              palette=["#4C72B0", "#DD8452"], legend=False)
    axes[1].set_title("Churn Distribution (Count)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_numeric_distributions(df: pd.DataFrame):
    """Histograms of the three numeric columns — checks skew, spread,
    and whether any values look like outliers or data errors."""
    _ensure_dir()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "numeric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_numeric_vs_target(df: pd.DataFrame):
    """Boxplots of each numeric column split by Churn — the core question:
    do churners and non-churners look meaningfully different on these
    values? This directly motivates the statistical tests in Phase 5."""
    _ensure_dir()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(data=df, x="Churn", y=col, hue="Churn", ax=ax,
            palette=["#4C72B0", "#DD8452"], legend=False)
        ax.set_title(f"{col} by Churn")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "numeric_vs_target_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_categorical_vs_target(df: pd.DataFrame):
    """Countplots of key categorical columns split by Churn — checks
    whether churn rate visibly differs across categories (e.g. does
    month-to-month contract show a much higher churn proportion?)."""
    _ensure_dir()
    fig, axes = plt.subplots(1, len(CATEGORICAL_VS_TARGET_COLS), figsize=(18, 5))
    for ax, col in zip(axes, CATEGORICAL_VS_TARGET_COLS):
        sns.countplot(data=df, x=col, hue="Churn", ax=ax, palette=["#4C72B0", "#DD8452"])
        ax.set_title(f"{col} by Churn")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "categorical_vs_target_countplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric columns plus the binary-encoded
    target — flags any suspiciously strong correlations (possible leakage)
    and highlights which numeric features are worth investigating further
    in Statistical Analysis (Phase 5)."""
    _ensure_dir()
    df = df.copy()
    df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    corr_cols = NUMERIC_COLS + ["Churn_binary"]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Numeric Features + Churn)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return corr


def main():
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df = clean_data(df_raw)

    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_numeric_vs_target(df)
    plot_categorical_vs_target(df)
    corr = plot_correlation_heatmap(df)

    print("\nCorrelation matrix (numeric features + Churn):")
    print(corr.round(3))


if __name__ == "__main__":
    main()