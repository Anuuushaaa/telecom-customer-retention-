"""
src/business_visualization.py

Financial and sensitivity analysis visualizations
for telecom customer retention campaigns.

All financial outcomes are hypothetical simulations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import (
    FuncFormatter,
    PercentFormatter,
)

from src.business_config import (
    REPORTS_DIR,
    BASELINE_CONFIG,
)


SENSITIVITY_DIR = (
    REPORTS_DIR / "sensitivity"
)

MODEL_NAMES = {
    "logistic_regression":
        "Logistic Regression",

    "xgboost":
        "XGBoost",

    "tensorflow_ann":
        "TensorFlow ANN",
}

MODEL_COLORS = {
    "logistic_regression": "#2563EB",
    "xgboost": "#059669",
    "tensorflow_ann": "#9333EA",
}

STRATEGY_NAMES = {
    "no_intervention":
        "No Intervention",

    "target_everyone":
        "Target Everyone",

    "default_threshold":
        "Default Threshold",

    "maximum_f1":
        "Maximum F1",

    "maximum_f2":
        "Maximum F2",

    "maximum_youden_j":
        "Maximum Youden J",

    "recall_at_least_80":
        "Recall >= 80%",

    "recall_at_least_90":
        "Recall >= 90%",

    "profit_optimal_threshold":
        "Profit-optimal Threshold",

    "budget_optimal_threshold":
        "Budget-optimal Threshold",

    "customer_value_targeting":
        "Customer-value Targeting",

    "budget_value_targeting":
        "Budget-value Targeting",
}

PARAMETER_NAMES = {
    "contact_cost":
        "Contact Cost",

    "incentive_cost":
        "Incentive Cost",

    "retention_success_rate":
        "Retention Success Rate",

    "retention_months":
        "Retention Duration",

    "contribution_margin":
        "Contribution Margin",

    "campaign_budget":
        "Campaign Budget",
}


def currency_formatter(value, position=None):

    if value < 0:

        return f"-${abs(value):,.0f}"

    return f"${value:,.0f}"


def load_report(path, required_columns):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Missing report: {path}"
        )

    data = pd.read_csv(path)

    if data.empty:

        raise ValueError(
            f"Empty report: {path}"
        )

    missing = (
        set(required_columns)
        - set(data.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing)}"
        )

    return data


def format_axis(ax):

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        alpha=0.20,
        linestyle="--",
    )

    ax.set_axisbelow(True)


def save_figure(
    fig,
    filename,
    output_dir=REPORTS_DIR,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / filename
    )

    fig.text(
        0.5,
        0.015,
        (
            "Validation dataset | "
            "Hypothetical financial simulation"
        ),
        ha="center",
        fontsize=9,
        color="#64748B",
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.95]
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

    return output_path


# --------------------------------------------------
# 1. Campaign strategy profitability
# --------------------------------------------------


def plot_campaign_profitability(
    campaign_results,
    output_dir=REPORTS_DIR,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 9),
        sharex=True,
    )

    strategies = list(
        STRATEGY_NAMES.keys()
    )

    for ax, model in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = campaign_results[
            campaign_results["model"] == model
        ].set_index("strategy")

        subset = subset.reindex(
            strategies
        ).dropna(
            subset=["expected_net_profit"]
        )

        positions = np.arange(
            len(subset)
        )

        ax.barh(
            positions - 0.18,
            subset["expected_net_profit"],
            height=0.35,
            color=MODEL_COLORS[model],
            label="Expected profit",
        )

        ax.barh(
            positions + 0.18,
            subset["simulated_net_profit"],
            height=0.35,
            color="#94A3B8",
            label="Label-based simulation",
        )

        ax.set_yticks(positions)

        ax.set_yticklabels([
            STRATEGY_NAMES.get(
                strategy,
                strategy,
            )

            for strategy in subset.index
        ])

        ax.invert_yaxis()

        ax.axvline(
            0,
            color="#334155",
            linewidth=1,
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(currency_formatter)
        )

        ax.set_title(
            MODEL_NAMES[model],
            fontweight="bold",
        )

        ax.set_xlabel("Net Profit")

        ax.legend(
            fontsize=8,
            loc="best",
        )

        format_axis(ax)

    fig.suptitle(
        "Retention Campaign Profitability",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "campaign_profitability.png",
        output_dir,
    )


# --------------------------------------------------
# 2. Business-optimal threshold curves
# --------------------------------------------------


def plot_business_thresholds(
    threshold_results,
    output_dir=REPORTS_DIR,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        sharey=True,
    )

    for ax, model in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = threshold_results[
            threshold_results["model"] == model
        ].sort_values("threshold")

        ax.plot(
            subset["threshold"],
            subset["expected_net_profit"],
            color=MODEL_COLORS[model],
            linewidth=2.5,
            label="Expected profit",
        )

        ax.plot(
            subset["threshold"],
            subset["simulated_net_profit"],
            color="#64748B",
            linestyle="--",
            linewidth=1.5,
            label="Label-based simulation",
        )

        best = subset.loc[
            subset[
                "expected_net_profit"
            ].idxmax()
        ]

        if best["expected_net_profit"] > 0:

            ax.scatter(
                best["threshold"],
                best["expected_net_profit"],
                color="#DC2626",
                s=70,
                zorder=5,
            )

            ax.axvline(
                best["threshold"],
                color="#DC2626",
                linestyle=":",
                label=(
                    f"Optimal: "
                    f"{best['threshold']:.2f}"
                ),
            )

        ax.axhline(
            0,
            color="#334155",
            linewidth=1,
        )

        ax.set_title(
            MODEL_NAMES[model],
            fontweight="bold",
        )

        ax.set_xlabel(
            "Classification Threshold"
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(currency_formatter)
        )

        ax.legend(fontsize=8)

        format_axis(ax)

    axes[0].set_ylabel(
        "Net Campaign Profit"
    )

    fig.suptitle(
        "Financial Threshold Optimization",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "business_threshold_optimization.png",
        output_dir,
    )


# --------------------------------------------------
# 3. Budget-constrained model comparison
# --------------------------------------------------


def plot_budget_model_comparison(
    best_strategies,
    output_dir=REPORTS_DIR,
):

    subset = best_strategies[
        best_strategies["selection_type"]
        == "budget_constrained"
    ].set_index("model")

    subset = subset.reindex(
        list(MODEL_NAMES)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
    )

    metrics = [
        (
            "expected_net_profit",
            "Expected Net Profit",
            "currency",
        ),
        (
            "expected_roi",
            "Expected ROI",
            "percent",
        ),
        (
            "recall",
            "Churn Recall",
            "percent",
        ),
    ]

    colors = [
        MODEL_COLORS[model]

        for model in subset.index
    ]

    labels = [
        MODEL_NAMES[model]

        for model in subset.index
    ]

    for ax, (
        metric,
        title,
        display_type,
    ) in zip(axes, metrics):

        bars = ax.bar(
            labels,
            subset[metric],
            color=colors,
            width=0.60,
        )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        if display_type == "currency":

            ax.yaxis.set_major_formatter(
                FuncFormatter(currency_formatter)
            )

            bar_labels = [
                f"${value:,.0f}"

                for value in subset[metric]
            ]

        else:

            ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1)
            )

            bar_labels = [
                f"{value:.1%}"

                for value in subset[metric]
            ]

        ax.bar_label(
            bars,
            labels=bar_labels,
            padding=4,
            fontsize=9,
        )

        ax.margins(y=0.15)

        format_axis(ax)

    fig.suptitle(
        (
            "Model Comparison Under "
            f"${BASELINE_CONFIG.campaign_budget:,.0f}"
            " Campaign Budget"
        ),
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "budget_constrained_model_comparison.png",
        output_dir,
    )


# --------------------------------------------------
# 4. Budget sensitivity and marginal returns
# --------------------------------------------------


def plot_budget_sensitivity(
    budget_results,
    output_dir=REPORTS_DIR,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    metrics = [
        (
            "expected_net_profit",
            "Expected Net Profit",
        ),
        (
            "customers_targeted",
            "Customers Targeted",
        ),
        (
            "marginal_return_on_spend",
            "Marginal Return on Spend",
        ),
    ]

    for ax, (metric, title) in zip(
        axes,
        metrics,
    ):

        for model in MODEL_NAMES:

            subset = budget_results[
                budget_results["model"] == model
            ].sort_values("campaign_budget")

            ax.plot(
                subset["campaign_budget"],
                subset[metric],
                color=MODEL_COLORS[model],
                marker="o",
                linewidth=2,
                label=MODEL_NAMES[model],
            )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Available Campaign Budget"
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(currency_formatter)
        )

        if metric == "expected_net_profit":

            ax.yaxis.set_major_formatter(
                FuncFormatter(currency_formatter)
            )

        if metric == "marginal_return_on_spend":

            ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1)
            )

        ax.tick_params(
            axis="x",
            rotation=25,
        )

        ax.legend(fontsize=8)

        format_axis(ax)

    fig.suptitle(
        "Campaign Budget Sensitivity",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "budget_sensitivity_analysis.png",
        output_dir,
    )


# --------------------------------------------------
# 5. One-way sensitivity tornado charts
# --------------------------------------------------


def plot_parameter_impacts(
    parameter_impacts,
    output_dir=REPORTS_DIR,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6),
        sharex=True,
    )

    for ax, model in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = parameter_impacts[
            (
                parameter_impacts["model"]
                == model
            )
            & (
                parameter_impacts["strategy"]
                == "budget_value_targeting"
            )
        ].sort_values(
            "profit_range",
            ascending=True,
        )

        labels = [
            PARAMETER_NAMES.get(
                parameter,
                parameter,
            )

            for parameter in subset["parameter"]
        ]

        ax.barh(
            labels,
            subset["downside_from_baseline"],
            color="#DC2626",
            alpha=0.80,
            label="Downside",
        )

        ax.barh(
            labels,
            subset["upside_from_baseline"],
            color="#059669",
            alpha=0.80,
            label="Upside",
        )

        ax.axvline(
            0,
            color="#334155",
            linewidth=1,
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(currency_formatter)
        )

        ax.set_xlabel(
            "Change from Baseline Profit"
        )

        ax.set_title(
            MODEL_NAMES[model],
            fontweight="bold",
        )

        ax.legend(fontsize=8)

        format_axis(ax)

    fig.suptitle(
        "Financial Sensitivity Tornado Analysis",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "financial_sensitivity_tornado.png",
        output_dir,
    )


# --------------------------------------------------
# 6. Two-way sensitivity heatmaps
# --------------------------------------------------


def plot_two_way_sensitivity(
    two_way_results,
    output_dir=REPORTS_DIR,
):

    filtered = two_way_results[
        two_way_results["strategy"]
        == "budget_value_targeting"
    ]

    maximum_profit = max(
        1.0,
        float(
            filtered["expected_net_profit"].max()
        ),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6),
    )

    for ax, model in zip(
        axes,
        MODEL_NAMES,
    ):

        subset = filtered[
            filtered["model"] == model
        ]

        matrix = subset.pivot(
            index="retention_success_rate",
            columns="incentive_cost",
            values="expected_net_profit",
        ).sort_index(
            ascending=False
        ).sort_index(
            axis=1
        )

        values = matrix.to_numpy()

        ax.imshow(
            values,
            cmap="YlGn",
            aspect="auto",
            vmin=0,
            vmax=maximum_profit,
        )

        ax.set_xticks(
            np.arange(len(matrix.columns))
        )

        ax.set_xticklabels([
            f"${cost:g}"

            for cost in matrix.columns
        ])

        ax.set_yticks(
            np.arange(len(matrix.index))
        )

        ax.set_yticklabels([
            f"{rate:.0%}"

            for rate in matrix.index
        ])

        for row in range(values.shape[0]):

            for col in range(values.shape[1]):

                value = values[row, col]

                color = (
                    "white"
                    if value > maximum_profit * 0.60
                    else "#172554"
                )

                ax.text(
                    col,
                    row,
                    f"${value / 1000:.1f}k",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        ax.set_xlabel(
            "Retention Incentive Cost"
        )

        ax.set_ylabel(
            "Retention Success Rate"
        )

        ax.set_title(
            MODEL_NAMES[model],
            fontweight="bold",
        )

    fig.suptitle(
        "Two-way Sensitivity: Expected Net Profit",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "two_way_sensitivity_heatmaps.png",
        output_dir,
    )


# --------------------------------------------------
# 7. Named financial scenario comparison
# --------------------------------------------------


def plot_named_scenarios(
    named_results,
    output_dir=REPORTS_DIR,
):

    filtered = named_results[
        named_results["strategy"]
        == "budget_value_targeting"
    ]

    scenarios = [
        "pessimistic",
        "baseline",
        "optimistic",
    ]

    positions = np.arange(
        len(scenarios)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    metrics = [
        (
            "expected_net_profit",
            "Expected Net Profit",
        ),
        (
            "simulated_net_profit",
            "Label-based Simulated Net Profit",
        ),
    ]

    width = 0.24

    for ax, (metric, title) in zip(
        axes,
        metrics,
    ):

        for index, model in enumerate(
            MODEL_NAMES
        ):

            subset = filtered[
                filtered["model"] == model
            ].set_index(
                "scenario"
            ).reindex(scenarios)

            bars = ax.bar(
                positions
                + (index - 1) * width,
                subset[metric],
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_NAMES[model],
            )

            ax.bar_label(
                bars,
                labels=[
                    f"${value / 1000:.1f}k"

                    for value in subset[metric]
                ],
                fontsize=8,
                padding=3,
            )

        ax.set_xticks(positions)

        ax.set_xticklabels([
            scenario.title()

            for scenario in scenarios
        ])

        ax.yaxis.set_major_formatter(
            FuncFormatter(currency_formatter)
        )

        ax.set_title(
            title,
            fontweight="bold",
        )

        ax.margins(y=0.15)

        ax.legend(fontsize=8)

        format_axis(ax)

    fig.suptitle(
        "Pessimistic, Baseline and Optimistic Scenarios",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "named_financial_scenarios.png",
        output_dir,
    )


# --------------------------------------------------
# 8. Model stability across financial scenarios
# --------------------------------------------------


def plot_model_stability(
    stability_results,
    output_dir=REPORTS_DIR,
):

    families = [
        "one_way",
        "two_way",
        "named",
    ]

    family_labels = [
        "One-way",
        "Two-way",
        "Named",
    ]

    strategies = [
        "customer_value_targeting",
        "budget_value_targeting",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        sharey=True,
    )

    for ax, strategy in zip(
        axes,
        strategies,
    ):

        filtered = stability_results[
            stability_results["strategy"]
            == strategy
        ]

        bottom = np.zeros(
            len(families)
        )

        for model in MODEL_NAMES:

            subset = filtered[
                filtered["model"] == model
            ].set_index(
                "scenario_family"
            ).reindex(families)

            rates = subset[
                "win_rate"
            ].to_numpy()

            ax.bar(
                family_labels,
                rates,
                bottom=bottom,
                color=MODEL_COLORS[model],
                label=MODEL_NAMES[model],
                width=0.60,
            )

            for index, rate in enumerate(
                rates
            ):

                if rate >= 0.06:

                    ax.text(
                        index,
                        bottom[index] + rate / 2,
                        f"{rate:.1%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )

            bottom += rates

        no_intervention = (
            filtered.groupby(
                "scenario_family"
            ).first().reindex(families)
        )

        no_intervention_rates = (
            no_intervention[
                "no_intervention_scenarios"
            ]
            / no_intervention[
                "total_scenarios"
            ]
        ).to_numpy()

        ax.bar(
            family_labels,
            no_intervention_rates,
            bottom=bottom,
            color="#94A3B8",
            width=0.60,
            label="No Intervention",
        )

        ax.set_ylim(0, 1.05)

        ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1)
        )

        ax.set_title(
            STRATEGY_NAMES[strategy],
            fontweight="bold",
        )

        ax.legend(
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
        )

        format_axis(ax)

    axes[0].set_ylabel(
        "Fraction of Scenarios Won"
    )

    fig.suptitle(
        "Financial Model Stability",
        fontsize=17,
        fontweight="bold",
    )

    return save_figure(
        fig,
        "financial_model_stability.png",
        output_dir,
    )


# --------------------------------------------------
# Main execution
# --------------------------------------------------


def main():

    print(
        "\nLoading business analysis reports..."
    )

    campaign_results = load_report(
        REPORTS_DIR
        / "campaign_strategy_comparison.csv",
        [
            "model",
            "strategy",
            "expected_net_profit",
            "simulated_net_profit",
        ],
    )

    threshold_results = load_report(
        REPORTS_DIR
        / "business_threshold_sweep.csv",
        [
            "model",
            "threshold",
            "expected_net_profit",
            "simulated_net_profit",
        ],
    )

    best_strategies = load_report(
        REPORTS_DIR
        / "best_business_strategies.csv",
        [
            "model",
            "selection_type",
            "expected_net_profit",
            "expected_roi",
            "recall",
        ],
    )

    budget_results = load_report(
        SENSITIVITY_DIR
        / "budget_sensitivity_summary.csv",
        [
            "model",
            "campaign_budget",
            "customers_targeted",
            "expected_net_profit",
            "marginal_return_on_spend",
        ],
    )

    parameter_impacts = load_report(
        SENSITIVITY_DIR
        / "parameter_impact_summary.csv",
        [
            "model",
            "strategy",
            "parameter",
            "profit_range",
            "downside_from_baseline",
            "upside_from_baseline",
        ],
    )

    two_way_results = load_report(
        SENSITIVITY_DIR
        / "two_way_sensitivity.csv",
        [
            "model",
            "strategy",
            "incentive_cost",
            "retention_success_rate",
            "expected_net_profit",
        ],
    )

    named_results = load_report(
        SENSITIVITY_DIR
        / "named_scenario_analysis.csv",
        [
            "model",
            "strategy",
            "scenario",
            "expected_net_profit",
            "simulated_net_profit",
        ],
    )

    stability_results = load_report(
        SENSITIVITY_DIR
        / "model_stability_summary.csv",
        [
            "model",
            "strategy",
            "scenario_family",
            "win_rate",
            "total_scenarios",
            "no_intervention_scenarios",
        ],
    )

    print(
        "\nGenerating financial visualizations..."
    )

    generated = [
        plot_campaign_profitability(
            campaign_results
        ),

        plot_business_thresholds(
            threshold_results
        ),

        plot_budget_model_comparison(
            best_strategies
        ),

        plot_budget_sensitivity(
            budget_results
        ),

        plot_parameter_impacts(
            parameter_impacts
        ),

        plot_two_way_sensitivity(
            two_way_results
        ),

        plot_named_scenarios(
            named_results
        ),

        plot_model_stability(
            stability_results
        ),
    ]

    print(
        f"\nVisualizations generated: "
        f"{len(generated)}"
    )

    print(
        "Business visualization completed."
    )

    print(
        "All financial outcomes are hypothetical."
    )

    print(
        "Production models remain unchanged."
    )


if __name__ == "__main__":
    main()