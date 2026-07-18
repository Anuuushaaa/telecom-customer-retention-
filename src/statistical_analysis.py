"""
src/statistical_analysis.py

Phase 5: Statistical Analysis for the Telecom Churn project.

Two test types, matched to variable type:
- Mann-Whitney U test: numeric column (tenure, MonthlyCharges, TotalCharges)
  compared between the two Churn groups. Chosen over a classic Student's
  t-test because tenure's distribution is visibly non-normal (bimodal, per
  Phase 4 EDA) — Mann-Whitney makes no distributional assumption, so it's
  applied uniformly across all three numeric columns for consistency.
- Chi-square test of independence: categorical column vs Churn, testing
  whether the two variables are statistically associated or independent.

Run standalone: python -m src.statistical_analysis
"""

import os

import pandas as pd
from scipy import stats

from src.cleaning import clean_data

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

ALPHA = 0.05  # standard significance threshold


def run_mann_whitney_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric column, compare its distribution between Churn=Yes
    and Churn=No groups using the Mann-Whitney U test.

    Null hypothesis (H0): the two groups' distributions are identical
    (no tendency for one group's values to be systematically higher).
    A small p-value (< ALPHA) means we reject H0 — the groups differ.
    """
    results = []
    churned = df[df["Churn"] == "Yes"]
    stayed = df[df["Churn"] == "No"]

    for col in NUMERIC_COLS:
        stat, p_value = stats.mannwhitneyu(
            churned[col], stayed[col], alternative="two-sided"
        )
        results.append({
            "feature": col,
            "test": "Mann-Whitney U",
            "statistic": round(stat, 2),
            "p_value": p_value,
            "significant (p<0.05)": p_value < ALPHA,
            "median_churned": round(churned[col].median(), 2),
            "median_stayed": round(stayed[col].median(), 2),
        })

    return pd.DataFrame(results)


def run_chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each categorical column, test independence from Churn using a
    chi-square test on the contingency table (crosstab) of that column
    against Churn.

    Null hypothesis (H0): the column and Churn are independent (no
    association). A small p-value means we reject H0 — there's a
    statistically significant association.
    """
    results = []

    for col in CATEGORICAL_COLS:
        contingency = pd.crosstab(df[col], df["Churn"])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramer's V: effect size for chi-square — tells us not just
        # "is there an association" (p-value) but "how strong is it"
        # (0 = none, close to 1 = very strong). p-value alone can be
        # statistically significant but practically tiny with large n,
        # so reporting both is the more complete, honest analysis.
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 else float("nan")

        results.append({
            "feature": col,
            "test": "Chi-square",
            "chi2_statistic": round(chi2, 2),
            "p_value": p_value,
            "significant (p<0.05)": p_value < ALPHA,
            "cramers_v": round(cramers_v, 3),
        })

    return pd.DataFrame(results).sort_values("cramers_v", ascending=False)


def main():
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df = clean_data(df_raw)

    print("=" * 70)
    print("MANN-WHITNEY U TESTS (numeric features vs Churn)")
    print("=" * 70)
    mw_results = run_mann_whitney_tests(df)
    print(mw_results.to_string(index=False))

    print("\n" + "=" * 70)
    print("CHI-SQUARE TESTS (categorical features vs Churn)")
    print("Sorted by Cramer's V (effect size, strongest association first)")
    print("=" * 70)
    chi_results = run_chi_square_tests(df)
    print(chi_results.to_string(index=False))

    # Save both result tables for reference in the README / report
    os.makedirs("reports", exist_ok=True)
    mw_results.to_csv("reports/mann_whitney_results.csv", index=False)
    chi_results.to_csv("reports/chi_square_results.csv", index=False)
    print("\nSaved results to reports/mann_whitney_results.csv and reports/chi_square_results.csv")


if __name__ == "__main__":
    main()