"""
src/bootstrap_visualization.py

Visualize bootstrap metric distributions,
confidence intervals, and paired model comparisons.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports/bootstrap")

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

METRIC_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
}


def save_figure(fig, output_path):
    """Save and close a figure."""

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


def plot_metric_distributions(
    distributions,
    output_path=REPORTS_DIR / "bootstrap_metric_distributions.png",
):
    """Plot bootstrap distributions for all six metrics."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
    )

    for ax, (metric, title) in zip(
        axes.ravel(),
        METRIC_NAMES.items(),
    ):

        bins = np.histogram_bin_edges(
            distributions[metric],
            bins=35,
        )

        for model_name, display_name in (
            MODEL_NAMES.items()
        ):

            values = distributions.loc[
                distributions["model"]
                == model_name,
                metric,
            ]

            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.35,
                color=MODEL_COLORS[model_name],
                label=display_name,
            )

            ax.axvline(
                values.mean(),
                color=MODEL_COLORS[model_name],
                linestyle="--",
                linewidth=1.5,
            )

        ax.set_title(title)

        ax.set_xlabel("Metric value")

        ax.set_ylabel("Density")

        ax.grid(alpha=0.2)

        ax.legend(fontsize=8)

    fig.suptitle(
        "Bootstrap Performance Distributions",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def plot_confidence_intervals(
    intervals,
    output_path=REPORTS_DIR / "bootstrap_confidence_intervals.png",
):
    """Plot model point estimates and confidence intervals."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
    )

    for ax, (metric, title) in zip(
        axes.ravel(),
        METRIC_NAMES.items(),
    ):

        metric_data = intervals.loc[
            intervals["metric"] == metric
        ].set_index("model")

        positions = np.arange(
            len(MODEL_NAMES)
        )

        for position, model_name in enumerate(
            MODEL_NAMES
        ):

            row = metric_data.loc[model_name]

            estimate = row["point_estimate"]

            lower = row["ci_lower"]

            upper = row["ci_upper"]

            color = MODEL_COLORS[model_name]

            ax.hlines(
                position,
                lower,
                upper,
                color=color,
                linewidth=3,
            )

            ax.scatter(
                estimate,
                position,
                color=color,
                s=65,
                zorder=5,
            )

            ax.vlines(
                [lower, upper],
                position - 0.08,
                position + 0.08,
                color=color,
                linewidth=2,
            )

        ax.set_yticks(
            positions,
            list(MODEL_NAMES.values()),
        )

        ax.invert_yaxis()

        ax.set_title(title)

        ax.set_xlabel("Metric value")

        ax.grid(
            axis="x",
            alpha=0.2,
        )

    fig.suptitle(
        "Model Performance with 95% Bootstrap Confidence Intervals",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def plot_paired_comparisons(
    comparisons,
    output_path=REPORTS_DIR / "paired_model_comparisons.png",
):
    """Plot confidence intervals for paired metric differences."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19, 10),
    )

    for ax, (metric, title) in zip(
        axes.ravel(),
        METRIC_NAMES.items(),
    ):

        metric_data = comparisons.loc[
            comparisons["metric"] == metric
        ].reset_index(drop=True)

        labels = []

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        for position, row in metric_data.iterrows():

            model_a = MODEL_NAMES[
                row["model_a"]
            ]

            model_b = MODEL_NAMES[
                row["model_b"]
            ]

            labels.append(
                f"{model_a}\n− {model_b}"
            )

            lower = row["ci_lower"]

            upper = row["ci_upper"]

            excludes_zero = (
                lower > 0 or upper < 0
            )

            color = (
                "#059669"
                if excludes_zero
                else "#64748B"
            )

            ax.hlines(
                position,
                lower,
                upper,
                color=color,
                linewidth=3,
            )

            ax.vlines(
                [lower, upper],
                position - 0.08,
                position + 0.08,
                color=color,
                linewidth=2,
            )

            ax.scatter(
                row["point_difference"],
                position,
                color=color,
                s=65,
                zorder=5,
            )

        ax.set_yticks(
            np.arange(len(labels)),
            labels,
        )

        ax.invert_yaxis()

        ax.set_title(title)

        ax.set_xlabel("Model A − Model B")

        ax.grid(
            axis="x",
            alpha=0.2,
        )

    fig.suptitle(
        "Paired Bootstrap Model Comparisons (95% CI)",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def plot_roc_auc_differences(
    distributions,
    comparisons,
    output_path=REPORTS_DIR / "paired_roc_auc_distributions.png",
):
    """Plot paired bootstrap ROC-AUC difference distributions."""

    auc_comparisons = comparisons.loc[
        comparisons["metric"] == "roc_auc"
    ].reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        len(auc_comparisons),
        figsize=(18, 5),
        squeeze=False,
    )

    for ax, (_, row) in zip(
        axes.ravel(),
        auc_comparisons.iterrows(),
    ):

        model_a = row["model_a"]

        model_b = row["model_b"]

        scores_a = distributions.loc[
            distributions["model"] == model_a
        ].set_index("iteration")["roc_auc"]

        scores_b = distributions.loc[
            distributions["model"] == model_b
        ].set_index("iteration")["roc_auc"]

        if (
            not scores_a.index.is_unique
            or not scores_b.index.is_unique
        ):

            raise ValueError(
                "Duplicate bootstrap iterations."
            )

        if set(scores_a.index) != set(
            scores_b.index
        ):

            raise ValueError(
                "Bootstrap iterations are not aligned."
            )

        differences = (
            scores_a
            - scores_b.reindex(scores_a.index)
        )

        ax.hist(
            differences,
            bins=35,
            density=True,
            color="#2563EB",
            alpha=0.65,
        )

        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=2,
            label="No difference",
        )

        ax.axvline(
            row["point_difference"],
            color="#DC2626",
            linewidth=2,
            label="Observed difference",
        )

        ax.axvspan(
            row["ci_lower"],
            row["ci_upper"],
            color="#94A3B8",
            alpha=0.2,
            label="95% percentile interval",
        )

        ax.set_title(
            f"{MODEL_NAMES[model_a]}\n"
            f"− {MODEL_NAMES[model_b]}"
        )

        ax.set_xlabel("ROC-AUC difference")

        ax.set_ylabel("Density")

        ax.grid(alpha=0.2)

        ax.legend(fontsize=8)

    fig.suptitle(
        "Paired Bootstrap ROC-AUC Difference Distributions",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def main():
    """Generate all bootstrap visualizations."""

    distributions = pd.read_csv(
        REPORTS_DIR / "bootstrap_distributions.csv"
    )

    intervals = pd.read_csv(
        REPORTS_DIR / "bootstrap_confidence_intervals.csv"
    )

    comparisons = pd.read_csv(
        REPORTS_DIR / "paired_model_comparisons.csv"
    )

    print(
        "\nGenerating bootstrap visualizations..."
    )

    plot_metric_distributions(distributions)

    plot_confidence_intervals(intervals)

    plot_paired_comparisons(comparisons)

    plot_roc_auc_differences(
        distributions,
        comparisons,
    )

    print(
        "\nAll bootstrap visualizations generated."
    )

    print(
        "\nComparisons whose nominal 95% intervals exclude zero:"
    )

    significant = comparisons.loc[
        (comparisons["ci_lower"] > 0)
        | (comparisons["ci_upper"] < 0)
    ]

    if significant.empty:

        print("None.")

    else:

        print(
            significant[
                [
                    "model_a",
                    "model_b",
                    "metric",
                    "point_difference",
                    "ci_lower",
                    "ci_upper",
                ]
            ].to_string(index=False)
        )

    print(
        "\nProduction models remain unchanged."
    )


if __name__ == "__main__":
    main()