"""
src/threshold_visualization.py

Visualize classification threshold optimization
using saved validation experiment reports.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports/thresholds")

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

METRIC_COLORS = {
    "precision": "#2563EB",
    "recall": "#DC2626",
    "f1": "#059669",
    "f2": "#9333EA",
    "specificity": "#D97706",
}

STRATEGY_NAMES = {
    "maximum_f1": "Maximum F1",
    "maximum_f2": "Maximum F2",
    "maximum_youden_j": "Youden's J",
    "recall_at_least_80": "Recall >= 80%",
    "recall_at_least_90": "Recall >= 90%",
}


def save_figure(fig, output_path):
    """Save a high-resolution visualization."""

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


def get_preferred_sweep(
    threshold_results,
    preferred_thresholds,
):
    """Keep threshold results for preferred calibrators."""

    selected_methods = (
        preferred_thresholds[
            ["model", "calibration_method"]
        ]
        .drop_duplicates()
    )

    if selected_methods["model"].duplicated().any():
        raise ValueError(
            "Multiple preferred methods per model."
        )

    if set(selected_methods["model"]) != set(
        MODEL_NAMES
    ):
        raise ValueError(
            "Preferred model membership mismatch."
        )

    preferred_sweep = threshold_results.merge(
        selected_methods,
        on=[
            "model",
            "calibration_method",
        ],
        how="inner",
        validate="many_to_one",
    )

    if preferred_sweep.empty:
        raise ValueError(
            "Preferred threshold results are empty."
        )

    return preferred_sweep


def plot_threshold_curves(
    preferred_sweep,
    preferred_thresholds,
    output_path=REPORTS_DIR / "threshold_performance_curves.png",
):
    """Plot classification metrics across thresholds."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6),
    )

    for ax, model_name in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = preferred_sweep[
            preferred_sweep["model"] == model_name
        ].sort_values("threshold")

        for metric, color in METRIC_COLORS.items():

            ax.plot(
                subset["threshold"],
                subset[metric],
                color=color,
                linewidth=2,
                label=metric.upper(),
            )

        best_f1 = preferred_thresholds[
            (
                preferred_thresholds["model"]
                == model_name
            )
            & (
                preferred_thresholds["strategy"]
                == "maximum_f1"
            )
            & preferred_thresholds["feasible"]
        ]

        if len(best_f1) != 1:
            raise ValueError(
                f"Invalid maximum F1 result: {model_name}"
            )

        best_threshold = float(
            best_f1.iloc[0]["threshold"]
        )

        ax.axvline(
            best_threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Best F1: {best_threshold:.2f}",
        )

        ax.axvline(
            0.50,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Default: 0.50",
        )

        method = subset[
            "calibration_method"
        ].iloc[0]

        ax.set_title(
            f"{MODEL_NAMES[model_name]}\n"
            f"{method.capitalize()} calibration",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Classification threshold"
        )

        ax.set_ylabel(
            "Metric value"
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        ax.grid(alpha=0.2)

        ax.legend(
            fontsize=8,
            loc="best",
        )

    fig.suptitle(
        "Validation Threshold Performance Curves",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_precision_recall_tradeoff(
    preferred_sweep,
    preferred_thresholds,
    output_path=REPORTS_DIR / "precision_recall_tradeoff.png",
):
    """Compare precision and recall across thresholds."""

    fig, ax = plt.subplots(
        figsize=(11, 8),
    )

    for model_name in MODEL_NAMES:

        subset = preferred_sweep[
            preferred_sweep["model"] == model_name
        ].sort_values(
            "threshold",
            ascending=False,
        )

        ax.plot(
            subset["recall"],
            subset["precision"],
            color=MODEL_COLORS[model_name],
            linewidth=2,
            label=MODEL_NAMES[model_name],
        )

        selected = preferred_thresholds[
            (
                preferred_thresholds["model"]
                == model_name
            )
            & (
                preferred_thresholds["strategy"]
                == "maximum_f1"
            )
            & preferred_thresholds["feasible"]
        ].iloc[0]

        ax.scatter(
            selected["recall"],
            selected["precision"],
            color=MODEL_COLORS[model_name],
            edgecolor="black",
            s=100,
            zorder=5,
        )

        ax.annotate(
            f"t={selected['threshold']:.2f}",
            (
                selected["recall"],
                selected["precision"],
            ),
            xytext={
                "logistic_regression": (8, -18),
                "xgboost": (8, 10),
                "tensorflow_ann": (-45, 10),
            }[model_name],
            textcoords="offset points",
            fontsize=9,
        )

    for target in (0.80, 0.90):

        ax.axvline(
            target,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"{target:.0%} recall",
        )

    ax.set_title(
        "Precision-Recall Trade-off",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)

    ax.grid(alpha=0.2)
    ax.legend()

    return save_figure(
        fig,
        output_path,
    )


def plot_confusion_matrices(
    preferred_thresholds,
    output_path=REPORTS_DIR / "optimized_confusion_matrices.png",
):
    """Plot confusion matrices at maximum F1 thresholds."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6),
    )

    selected = preferred_thresholds[
        (
            preferred_thresholds["strategy"]
            == "maximum_f1"
        )
        & preferred_thresholds["feasible"]
    ].set_index("model")

    count_columns = [
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    maximum_count = selected[
        count_columns
    ].to_numpy().max()

    for ax, model_name in zip(
        axes,
        MODEL_NAMES,
    ):

        row = selected.loc[model_name]

        matrix = np.array([
            [
                row["true_negatives"],
                row["false_positives"],
            ],
            [
                row["false_negatives"],
                row["true_positives"],
            ],
        ], dtype=int)

        ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0,
            vmax=maximum_count,
        )

        cell_names = [
            ["TN", "FP"],
            ["FN", "TP"],
        ]

        for i in range(2):

            for j in range(2):

                value = matrix[i, j]

                color = (
                    "white"
                    if value > maximum_count / 2
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{cell_names[i][j]}\n{value:,}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No Churn", "Churn"]
        )

        ax.set_yticklabels(
            ["No Churn", "Churn"]
        )

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")

        ax.set_title(
            f"{MODEL_NAMES[model_name]}\n"
            f"Threshold = {row['threshold']:.2f} | "
            f"F1 = {row['f1']:.4f}",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        "Confusion Matrices at Maximum F1 Thresholds",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_strategy_comparison(
    preferred_thresholds,
    output_path=REPORTS_DIR / "threshold_strategy_comparison.png",
):
    """Compare metrics across threshold selection strategies."""

    metrics = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "f2": "F2 Score",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 11),
    )

    x = np.arange(
        len(STRATEGY_NAMES)
    )

    width = 0.25

    feasible = preferred_thresholds[
        preferred_thresholds["feasible"]
    ]

    for ax, (metric, title) in zip(
        axes.flat,
        metrics.items(),
    ):

        for model_index, model_name in enumerate(
            MODEL_NAMES
        ):

            subset = feasible[
                feasible["model"] == model_name
            ].set_index(
                "strategy"
            ).reindex(
                list(STRATEGY_NAMES)
            )

            values = subset[metric].to_numpy()

            bars = ax.bar(
                x + (model_index - 1) * width,
                values,
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_NAMES[model_name],
            )

            ax.bar_label(
                bars,
                fmt="%.3f",
                fontsize=7,
                rotation=90,
                padding=3,
            )

        ax.set_xticks(x)

        ax.set_xticklabels(
            list(STRATEGY_NAMES.values()),
            rotation=15,
            ha="right",
        )

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylim(0, 1.15)

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        ax.legend(fontsize=8)

    fig.suptitle(
        "Statistical Threshold Strategy Comparison",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def plot_targeting_analysis(
    preferred_thresholds,
    output_path=REPORTS_DIR / "customer_targeting_analysis.png",
):
    """Visualize campaign size and missed churners."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 7),
        sharey=True,
    )

    y = np.arange(
        len(STRATEGY_NAMES)
    )

    feasible = preferred_thresholds[
        preferred_thresholds["feasible"]
    ]

    for ax, model_name in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = feasible[
            feasible["model"] == model_name
        ].set_index(
            "strategy"
        ).reindex(
            list(STRATEGY_NAMES)
        )

        tp = subset[
            "true_positives"
        ].to_numpy()

        fp = subset[
            "false_positives"
        ].to_numpy()

        fn = subset[
            "false_negatives"
        ].to_numpy()

        ax.barh(
            y,
            tp,
            color="#059669",
            label="Detected churners",
        )

        ax.barh(
            y,
            fp,
            left=tp,
            color="#94A3B8",
            label="False positives",
        )

        ax.barh(
            y,
            fn,
            left=tp + fp,
            color="#DC2626",
            label="Missed churners",
        )

        for index in range(len(y)):

            if not np.isfinite(tp[index]):
                continue

            targeted = int(
                tp[index] + fp[index]
            )

            missed = int(fn[index])

            total = (
                targeted + missed
            )

            ax.text(
                total + 8,
                y[index],
                f"{targeted} targeted\n"
                f"{missed} missed",
                va="center",
                fontsize=8,
            )

        ax.set_yticks(y)

        ax.set_yticklabels(
            list(STRATEGY_NAMES.values())
        )

        ax.set_title(
            MODEL_NAMES[model_name],
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Number of customers"
        )

        totals = tp + fp + fn

        if np.isfinite(totals).any():

            ax.set_xlim(
                0,
                np.nanmax(totals) * 1.25,
            )

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        ax.legend(fontsize=8)

    axes[0].invert_yaxis()

    fig.suptitle(
        "Customer Targeting and Missed Churn Analysis",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        output_path,
    )


def main():
    """Generate all threshold optimization visualizations."""

    print(
        "\nGenerating threshold visualizations..."
    )

    threshold_results = pd.read_csv(
        REPORTS_DIR / "threshold_sweep.csv"
    )

    preferred_thresholds = pd.read_csv(
        REPORTS_DIR
        / "preferred_calibration_thresholds.csv"
    )

    preferred_sweep = get_preferred_sweep(
        threshold_results,
        preferred_thresholds,
    )

    plot_threshold_curves(
        preferred_sweep,
        preferred_thresholds,
    )

    plot_precision_recall_tradeoff(
        preferred_sweep,
        preferred_thresholds,
    )

    plot_confusion_matrices(
        preferred_thresholds
    )

    plot_strategy_comparison(
        preferred_thresholds
    )

    plot_targeting_analysis(
        preferred_thresholds
    )

    print(
        "\nAll five threshold visualizations generated."
    )

    print(
        "Production models and thresholds remain unchanged."
    )


if __name__ == "__main__":
    main()