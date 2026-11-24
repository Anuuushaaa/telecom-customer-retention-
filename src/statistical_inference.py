
"""
Statistical inference for telecom customer churn.

Implements:
- Unpenalized logistic regression
- Odds ratios and 95% confidence intervals
- Wald hypothesis tests
- Likelihood ratio tests

Run: python -m src.statistical_inference
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from src.cleaning import clean_data


ALPHA = 0.05

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "SeniorCitizen",
]

REFERENCE_CATEGORIES = {
    "Partner": "No",
    "Dependents": "No",
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "TechSupport": "No",
    "PaperlessBilling": "No",
    "PaymentMethod": "Electronic check",
}


def prepare_inference_data(df: pd.DataFrame):
    """Prepare numeric predictors and reference-coded categories."""

    df = df.copy()

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Invalid values in Churn column.")

    X = df[NUMERIC_FEATURES].astype(float).copy()

    # Interpret odds ratios per 10 currency units.
    X["MonthlyCharges"] = X["MonthlyCharges"] / 10
    X = X.rename(
        columns={"MonthlyCharges": "MonthlyCharges_per_10"}
    )

    for column, reference in REFERENCE_CATEGORIES.items():

        if df[column].isna().any():
            raise ValueError(f"Missing values in {column}.")

        categories = sorted(df[column].unique())

        if reference not in categories:
            raise ValueError(
                f"Reference category {reference} missing in {column}."
            )

        for category in categories:

            if category == reference:
                continue

            feature_name = f"{column}_{category}"

            X[feature_name] = (
                df[column] == category
            ).astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Predictors contain invalid numeric values.")

    X = sm.add_constant(X, has_constant="add")

    return X, y.astype(int)


def fit_logistic_model(X, y):
    """Fit an unpenalized maximum-likelihood logistic model."""

    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError("Design matrix is rank deficient.")

    model = sm.Logit(y, X)

    result = model.fit(
        maxiter=200,
        disp=False,
    )

    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("Logistic regression did not converge.")

    return result


def calculate_odds_ratios(result):
    """Calculate adjusted odds ratios and confidence intervals."""

    confidence_intervals = result.conf_int(alpha=ALPHA)

    odds_ratios = pd.DataFrame({
        "feature": result.params.index,
        "coefficient": result.params.values,
        "odds_ratio": np.exp(result.params.values),
        "ci_lower_95": np.exp(
            confidence_intervals.iloc[:, 0].values
        ),
        "ci_upper_95": np.exp(
            confidence_intervals.iloc[:, 1].values
        ),
    })

    return odds_ratios[
        odds_ratios["feature"] != "const"
    ].reset_index(drop=True)


def perform_wald_test(result):
    """Test whether individual regression coefficients equal zero."""

    wald_statistics = (
        result.params / result.bse
    ) ** 2

    p_values = chi2.sf(wald_statistics, df=1)

    results = pd.DataFrame({
        "feature": result.params.index,
        "coefficient": result.params.values,
        "standard_error": result.bse.values,
        "wald_statistic": wald_statistics.values,
        "degrees_of_freedom": 1,
        "p_value": p_values,
        "significant": p_values < ALPHA,
    })

    return results[
        results["feature"] != "const"
    ].reset_index(drop=True)


def perform_likelihood_ratio_test(
    full_model,
    reduced_model,
    comparison,
):
    """Compare two nested maximum-likelihood logistic models."""

    if full_model.nobs != reduced_model.nobs:
        raise ValueError("Models must use the same observations.")

    if not np.array_equal(
        full_model.model.endog,
        reduced_model.model.endog,
    ):
        raise ValueError("Model outcomes must match.")

    full_names = full_model.model.exog_names
    reduced_names = reduced_model.model.exog_names

    if not set(reduced_names).issubset(full_names):
        raise ValueError("Models must be nested.")

    indices = [
        full_names.index(name)
        for name in reduced_names
    ]

    if not np.array_equal(
        full_model.model.exog[:, indices],
        reduced_model.model.exog,
    ):
        raise ValueError("Shared predictors must match.")

    degrees_of_freedom = int(
        full_model.df_model - reduced_model.df_model
    )

    if degrees_of_freedom <= 0:
        raise ValueError("Full model must contain more parameters.")

    lr_statistic = 2 * (
        full_model.llf - reduced_model.llf
    )

    if lr_statistic < -1e-8:
        raise ValueError("Full model has a lower log-likelihood.")

    lr_statistic = max(0.0, lr_statistic)

    p_value = chi2.sf(
        lr_statistic,
        degrees_of_freedom,
    )

    return {
        "comparison": comparison,
        "full_log_likelihood": full_model.llf,
        "reduced_log_likelihood": reduced_model.llf,
        "lr_statistic": lr_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


def run_statistical_inference():

    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df = clean_data(df_raw)

    X, y = prepare_inference_data(df)

    full_model = fit_logistic_model(X, y)

    odds_ratios = calculate_odds_ratios(full_model)

    wald_results = perform_wald_test(full_model)

    lr_results = []

    feature_groups = {
        "Contract": "Contract_",
        "PaymentMethod": "PaymentMethod_",
    }

    for group_name, prefix in feature_groups.items():

        columns_to_remove = [
            column
            for column in X.columns
            if column.startswith(prefix)
        ]

        X_reduced = X.drop(columns=columns_to_remove)

        reduced_model = fit_logistic_model(X_reduced, y)

        lr_results.append(
            perform_likelihood_ratio_test(
                full_model,
                reduced_model,
                comparison=f"Full model vs without {group_name}",
            )
        )

    lr_results = pd.DataFrame(lr_results)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    odds_ratios.to_csv(
        reports_dir / "odds_ratios.csv",
        index=False,
    )

    wald_results.to_csv(
        reports_dir / "wald_test_results.csv",
        index=False,
    )

    lr_results.to_csv(
        reports_dir / "likelihood_ratio_results.csv",
        index=False,
    )

    print("\nLOGISTIC REGRESSION SUMMARY")
    print(full_model.summary())

    print("\nODDS RATIOS AND 95% CONFIDENCE INTERVALS")
    print(odds_ratios.to_string(index=False))

    print("\nWALD TEST RESULTS")
    print(wald_results.to_string(index=False))

    print("\nLIKELIHOOD RATIO TEST RESULTS")
    print(lr_results.to_string(index=False))

    print("\nStatistical inference reports saved successfully.")


if __name__ == "__main__":
    run_statistical_inference()
