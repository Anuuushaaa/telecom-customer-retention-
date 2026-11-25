"""
src/optimization_visualization.py

Generate visual reports from completed Optuna studies
and optimized model evaluation results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optuna.importance import get_param_importances
from optuna.trial import TrialState

from src.optimization_storage import (
    load_saved_studies,
    DATABASE_PATH,
)


REPORTS_DIR = Path("reports/optimization")

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


def save_figure(fig, output_path):
    """Save and close a Matplotlib figure."""

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


def get_completed_trials(study):
    """Return completed trials with finite objective values."""

    return [
        trial
        for trial in study.trials
        if (
            trial.state == TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        )
    ]


def plot_optimization_history(
    studies,
    output_path=REPORTS_DIR / "optimization_history.png",
):
    """Plot completed trials and cumulative best ROC-AUC."""

    fig, axes = plt.subplots(
        1,
        len(studies),
        figsize=(18, 5),
        squeeze=False,
    )

    for ax, (model_name, study) in zip(
        axes.ravel(),
        studies.items(),
    ):

        completed = get_completed_trials(study)

        if not completed:

            ax.set_title(
                MODEL_NAMES[model_name]
            )

            ax.text(
                0.5,
                0.5,
                "No completed trials",
                ha="center",
                transform=ax.transAxes,
            )

            continue

        trial_numbers = [
            trial.number + 1
            for trial in completed
        ]

        scores = np.asarray([
            trial.value
            for trial in completed
        ])

        cumulative_best = np.maximum.accumulate(
            scores
        )

        color = MODEL_COLORS[model_name]

        ax.scatter(
            trial_numbers,
            scores,
            color=color,
            alpha=0.65,
            label="Completed trial",
        )

        ax.step(
            trial_numbers,
            cumulative_best,
            where="post",
            color="black",
            linewidth=2,
            label="Best so far",
        )

        best_trial = study.best_trial

        ax.scatter(
            best_trial.number + 1,
            best_trial.value,
            color="red",
            marker="*",
            s=180,
            zorder=5,
            label="Best trial",
        )

        ax.set_title(
            MODEL_NAMES[model_name]
        )

        ax.set_xlabel("Trial number")

        ax.set_ylabel("Mean CV ROC-AUC")

        ax.grid(alpha=0.2)

        ax.legend(fontsize=8)

    fig.suptitle(
        "Optuna Hyperparameter Optimization History",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def plot_hyperparameter_importance(
    studies,
    output_path=REPORTS_DIR / "hyperparameter_importance.png",
):
    """Estimate and plot hyperparameter importance."""

    fig, axes = plt.subplots(
        1,
        len(studies),
        figsize=(19, 7),
        squeeze=False,
    )

    records = []

    for ax, (model_name, study) in zip(
        axes.ravel(),
        studies.items(),
    ):

        ax.set_title(
            MODEL_NAMES[model_name]
        )

        completed = get_completed_trials(study)

        if len(completed) < 2:

            ax.text(
                0.5,
                0.5,
                "Insufficient completed trials",
                ha="center",
                transform=ax.transAxes,
            )

            continue

        try:

            importances = get_param_importances(
                study
            )

        except (ValueError, RuntimeError) as error:

            print(
                f"Importance unavailable for "
                f"{model_name}: {error}"
            )

            ax.text(
                0.5,
                0.5,
                "Importance unavailable",
                ha="center",
                transform=ax.transAxes,
            )

            continue

        if not importances:

            ax.text(
                0.5,
                0.5,
                "No eligible parameters",
                ha="center",
                transform=ax.transAxes,
            )

            continue

        sorted_items = sorted(
            importances.items(),
            key=lambda item: item[1],
        )

        parameters = [
            item[0]
            for item in sorted_items
        ]

        values = [
            item[1]
            for item in sorted_items
        ]

        ax.barh(
            parameters,
            values,
            color=MODEL_COLORS[model_name],
        )

        ax.set_xlabel(
            "Estimated relative importance"
        )

        ax.set_xlim(0, 1)

        ax.grid(
            axis="x",
            alpha=0.2,
        )

        for parameter, importance in importances.items():

            records.append({
                "model": model_name,
                "parameter": parameter,
                "importance": importance,
            })

    fig.suptitle(
        "Optuna Hyperparameter Importance",
        fontsize=16,
        fontweight="bold",
    )

    output_path = save_figure(
        fig,
        output_path,
    )

    importance_df = pd.DataFrame(
        records,
        columns=[
            "model",
            "parameter",
            "importance",
        ],
    )

    importance_df.to_csv(
        output_path.parent
        / "hyperparameter_importance.csv",
        index=False,
    )

    return output_path


def plot_model_comparison(
    comparison_path=REPORTS_DIR / "optimized_model_comparison.csv",
    output_path=REPORTS_DIR / "baseline_vs_optimized.png",
):
    """Compare baseline and optimized validation metrics."""

    comparison = pd.read_csv(
        comparison_path
    )

    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    positions = np.arange(
        len(comparison)
    )

    width = 0.35

    labels = comparison["model"].replace({
        "Logistic Regression": "Logistic\nRegression",
        "TensorFlow ANN": "TensorFlow\nANN",
    })

    for ax, (metric, title) in zip(
        axes.ravel(),
        metrics,
    ):

        baseline_bars = ax.bar(
            positions - width / 2,
            comparison[f"{metric}_baseline"],
            width,
            label="Baseline",
            color="#94A3B8",
        )

        optimized_bars = ax.bar(
            positions + width / 2,
            comparison[f"{metric}_optimized"],
            width,
            label="Optimized",
            color="#059669",
        )

        ax.bar_label(
            baseline_bars,
            fmt="%.3f",
            fontsize=8,
            padding=3,
        )

        ax.bar_label(
            optimized_bars,
            fmt="%.3f",
            fontsize=8,
            padding=3,
        )

        ax.set_title(title)

        ax.set_xticks(
            positions,
            labels,
        )

        ax.set_ylim(0, 1)

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        ax.legend(fontsize=8)

    fig.suptitle(
        "Baseline vs Optimized Validation Performance",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def plot_cv_variability(
    studies,
    output_path=REPORTS_DIR / "cv_performance_variability.png",
):
    """Plot mean and standard deviation of best-trial CV scores."""

    names = []

    means = []

    standard_deviations = []

    fold_scores = []

    colors = []

    for model_name, study in studies.items():

        if not get_completed_trials(study):

            continue

        trial = study.best_trial

        scores = np.asarray(
            trial.user_attrs.get(
                "fold_scores",
                [],
            ),
            dtype=float,
        )

        if (
            len(scores) < 2
            or not np.isfinite(scores).all()
        ):

            continue

        names.append(
            MODEL_NAMES[model_name]
        )

        means.append(
            float(np.mean(scores))
        )

        standard_deviations.append(
            float(np.std(scores, ddof=1))
        )

        fold_scores.append(scores)

        colors.append(
            MODEL_COLORS[model_name]
        )

    if not names:

        raise ValueError(
            "No valid cross-validation scores available."
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    positions = np.arange(
        len(names)
    )

    ax.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=8,
        color=colors,
        alpha=0.75,
        label="Mean ± 1 SD",
    )

    for index, scores in enumerate(fold_scores):

        ax.scatter(
            np.full(len(scores), index),
            scores,
            color="black",
            marker="D",
            s=35,
            zorder=5,
            label=(
                "Individual fold score"
                if index == 0
                else None
            ),
        )

    ax.set_xticks(
        positions,
        names,
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("ROC-AUC")

    ax.set_title(
        "Best-Trial Cross-Validation Performance"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.legend()

    return save_figure(fig, output_path)


def plot_optimized_ann_history(
    history_path=REPORTS_DIR / "optimized_ann_training_history.csv",
    output_path=REPORTS_DIR / "optimized_ann_learning_curves.png",
):
    """Plot training-only metrics for the optimized ANN."""

    history = pd.read_csv(
        history_path
    )

    metrics = [
        ("loss", "Training Objective Loss"),
        ("accuracy", "Training Accuracy"),
        ("roc_auc", "Training ROC-AUC"),
        ("pr_auc", "Training PR-AUC"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    for ax, (metric, title) in zip(
        axes.ravel(),
        metrics,
    ):

        ax.plot(
            history["epoch"],
            history[metric],
            color="#9333EA",
            linewidth=2,
        )

        ax.set_title(title)

        ax.set_xlabel("Epoch")

        ax.set_ylabel(metric)

        ax.grid(alpha=0.2)

    fig.suptitle(
        "Optimized TensorFlow ANN - Training Curves",
        fontsize=16,
        fontweight="bold",
    )

    return save_figure(fig, output_path)


def main():
    """Generate all optimization visualizations."""

    studies = load_saved_studies(
        database_path=DATABASE_PATH,
        smoke_test=False,
    )

    missing_models = (
        set(MODEL_NAMES)
        - set(studies)
    )

    if missing_models:

        raise ValueError(
            f"Missing optimization studies: "
            f"{sorted(missing_models)}"
        )

    print(
        "\nGenerating optimization visualizations..."
    )

    plot_optimization_history(studies)

    plot_hyperparameter_importance(studies)

    plot_model_comparison()

    plot_cv_variability(studies)

    plot_optimized_ann_history()

    print(
        "\nAll optimization visualizations generated."
    )

    print(
        "Production models remain unchanged."
    )


if __name__ == "__main__":
    main()