"""
src/calibration_visualization.py

Generate probability calibration visualizations
using the saved validation experiment results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports/calibration")

PREDICTIONS_PATH = Path(
    "models/candidates/calibration/"
    "calibrated_validation_predictions.csv"
)

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "tensorflow_ann": "TensorFlow ANN",
}

MODEL_COLORS = {
    "logistic_regression": "#2563EB",
    "xgboost": "#059669",
    "tensorflow_ann": "#9333EA",
}

METHOD_NAMES = {
    "uncalibrated": "Uncalibrated",
    "sigmoid": "Sigmoid",
    "isotonic": "Isotonic",
}

METHOD_COLORS = {
    "uncalibrated": "#94A3B8",
    "sigmoid": "#2563EB",
    "isotonic": "#059669",
}

METRIC_NAMES = {
    "brier_score": "Brier Score",
    "log_loss": "Log Loss",
    "ece": "Expected Calibration Error",
    "mce": "Maximum Calibration Error",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
}


def save_figure(fig, output_path):
    """Save a publication-quality figure."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

    return output_path


def plot_reliability_diagrams(
    reliability_df,
    output_path=REPORTS_DIR / "reliability_diagrams.png",
):
    """Plot reliability curves for all three models."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    for ax, model_name in zip(
        axes,
        MODEL_NAMES,
    ):

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            alpha=0.7,
            label="Perfect calibration",
        )

        for method in METHOD_NAMES:

            subset = reliability_df[
                (reliability_df["model"] == model_name)
                & (
                    reliability_df["calibration_method"]
                    == method
                )
            ].sort_values("mean_probability")

            ax.plot(
                subset["mean_probability"],
                subset["observed_churn_rate"],
                marker="o",
                markersize=5,
                linewidth=2,
                color=METHOD_COLORS[method],
                label=METHOD_NAMES[method],
            )

        ax.set_title(
            MODEL_NAMES[model_name],
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Mean predicted churn probability"
        )

        ax.set_ylabel(
            "Observed churn rate"
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            alpha=0.2,
        )

        ax.legend(
            fontsize=9,
            loc="upper left",
        )

    fig.suptitle(
        "Probability Calibration — Reliability Diagrams",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_calibration_metrics(
    metrics_df,
    output_path=REPORTS_DIR / "calibration_metric_comparison.png",
):
    """Compare calibration and discrimination metrics."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    x = np.arange(
        len(MODEL_NAMES)
    )

    width = 0.25

    for ax, metric in zip(
        axes.flat,
        METRIC_NAMES,
    ):

        for method_index, method in enumerate(
            METHOD_NAMES
        ):

            subset = metrics_df[
                metrics_df["calibration_method"]
                == method
            ].set_index("model")

            values = subset.loc[
                list(MODEL_NAMES),
                metric,
            ].to_numpy()

            positions = (
                x
                + (method_index - 1) * width
            )

            bars = ax.bar(
                positions,
                values,
                width=width,
                color=METHOD_COLORS[method],
                label=METHOD_NAMES[method],
            )

            ax.bar_label(
                bars,
                fmt="%.4f",
                fontsize=7,
                padding=3,
                rotation=90,
            )

        direction = (
            "Higher is better"
            if metric in (
                "roc_auc",
                "average_precision",
            )
            else "Lower is better"
        )

        ax.set_title(
            f"{METRIC_NAMES[metric]}\n{direction}",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            ["Logistic\nRegression", "XGBoost", "ANN"]
        )

        ax.set_ylim(
            0,
            metrics_df[metric].max() * 1.30,
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        "Validation Calibration Metric Comparison",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_probability_distributions(
    predictions,
    output_path=REPORTS_DIR / "calibrated_probability_distributions.png",
):
    """Compare probability distributions before and after calibration."""

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14),
        sharex=True,
        sharey=True,
    )

    bins = np.linspace(
        0,
        1,
        21,
    )

    labels = predictions[
        "actual_churn"
    ].to_numpy()

    for row_index, model_name in enumerate(
        MODEL_NAMES
    ):

        for col_index, method in enumerate(
            METHOD_NAMES
        ):

            ax = axes[
                row_index,
                col_index,
            ]

            probabilities = predictions[
                f"{model_name}_{method}_probability"
            ].to_numpy()

            ax.hist(
                probabilities[labels == 0],
                bins=bins,
                density=True,
                alpha=0.55,
                color="#2563EB",
                label="No Churn",
            )

            ax.hist(
                probabilities[labels == 1],
                bins=bins,
                density=True,
                alpha=0.55,
                color="#DC2626",
                label="Churn",
            )

            ax.axvline(
                probabilities.mean(),
                linestyle="--",
                color="black",
                linewidth=1.5,
                label="Mean probability",
            )

            ax.set_title(
                f"{MODEL_NAMES[model_name]}\n"
                f"{METHOD_NAMES[method]}",
                fontsize=11,
                fontweight="bold",
            )

            ax.set_xlim(0, 1)

            ax.set_xlabel(
                "Predicted churn probability"
            )

            ax.set_ylabel(
                "Density"
            )

            ax.grid(
                alpha=0.2,
            )

            ax.legend(
                fontsize=8,
            )

    fig.suptitle(
        "Churn Probability Distributions by Calibration Method",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_calibration_improvement(
    metrics_df,
    output_path=REPORTS_DIR / "sigmoid_calibration_improvement.png",
):
    """Visualize relative metric improvements from sigmoid calibration."""

    improvement_metrics = [
        "brier_score",
        "log_loss",
        "ece",
        "mce",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
    )

    for ax, metric in zip(
        axes.flat,
        improvement_metrics,
    ):

        improvements = []

        for model_name in MODEL_NAMES:

            subset = metrics_df[
                metrics_df["model"] == model_name
            ].set_index(
                "calibration_method"
            )

            before = float(
                subset.loc[
                    "uncalibrated",
                    metric,
                ]
            )

            after = float(
                subset.loc[
                    "sigmoid",
                    metric,
                ]
            )

            if before == 0:

                improvement = 0.0

            else:

                improvement = (
                    (before - after)
                    / before
                    * 100
                )

            improvements.append(
                improvement
            )

        bars = ax.barh(
            list(MODEL_NAMES.values()),
            improvements,
            color=[
                MODEL_COLORS[model_name]
                for model_name in MODEL_NAMES
            ],
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=5,
        )

        ax.axvline(
            0,
            color="black",
            linewidth=1,
        )

        ax.set_title(
            METRIC_NAMES[metric],
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Relative reduction (%)"
        )

        ax.invert_yaxis()

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        lower = min(
            0,
            min(improvements) * 1.2,
        )

        upper = max(
            10,
            max(improvements) * 1.25,
        )

        ax.set_xlim(
            lower,
            upper,
        )

    fig.suptitle(
        "Relative Improvement After Sigmoid Calibration",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_predicted_vs_observed(
    metrics_df,
    output_path=REPORTS_DIR / "predicted_vs_observed_churn.png",
):
    """Compare aggregate predicted risk with observed churn."""

    fig, ax = plt.subplots(
        figsize=(12, 7),
    )

    x = np.arange(
        len(MODEL_NAMES)
    )

    width = 0.25

    for method_index, method in enumerate(
        METHOD_NAMES
    ):

        subset = metrics_df[
            metrics_df["calibration_method"]
            == method
        ].set_index("model")

        values = (
            subset.loc[
                list(MODEL_NAMES),
                "mean_predicted_probability",
            ].to_numpy()
            * 100
        )

        bars = ax.bar(
            x + (method_index - 1) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_NAMES[method],
        )

        ax.bar_label(
            bars,
            fmt="%.2f%%",
            padding=4,
            fontsize=9,
        )

    observed_rate = (
        metrics_df["observed_churn_rate"]
        .iloc[0]
        * 100
    )

    ax.axhline(
        observed_rate,
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label=f"Observed churn: {observed_rate:.2f}%",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        list(MODEL_NAMES.values())
    )

    ax.set_ylabel(
        "Mean churn probability (%)"
    )

    ax.set_title(
        "Mean Predicted Probability vs Observed Churn Rate",
        fontsize=15,
        fontweight="bold",
    )

    maximum = max(
        metrics_df[
            "mean_predicted_probability"
        ].max() * 100,
        observed_rate,
    )

    ax.set_ylim(
        0,
        maximum * 1.25,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend()

    return save_figure(
        fig,
        output_path,
    )


def main():
    """Generate all calibration visualizations."""

    print(
        "\nGenerating calibration visualizations..."
    )

    metrics_df = pd.read_csv(
        REPORTS_DIR / "calibration_metrics.csv"
    )

    reliability_df = pd.read_csv(
        REPORTS_DIR / "reliability_bins.csv"
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    plot_reliability_diagrams(
        reliability_df
    )

    plot_calibration_metrics(
        metrics_df
    )

    plot_probability_distributions(
        predictions
    )

    plot_calibration_improvement(
        metrics_df
    )

    plot_predicted_vs_observed(
        metrics_df
    )

    print(
        "\nAll five calibration visualizations generated."
    )

    print(
        "Production models remain unchanged."
    )


if __name__ == "__main__":
    main()