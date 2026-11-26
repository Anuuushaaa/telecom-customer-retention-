
"""Portfolio-level customer churn risk analytics."""

import hashlib
import json

from pathlib import Path

import numpy as np
import pandas as pd


RISK_ORDER = [
    "Low",
    "Moderate",
    "High",
    "Very High",
]

SEGMENT_COLUMNS = {
    "Contract": "Contract type",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "tenure_group": "Tenure group",
    "SeniorCitizen": "Senior citizen",
    "PaperlessBilling": "Paperless billing",
}

SHAP_REPORTS_DIR = Path("reports/shap")


# ============================================================
# ANALYTICS DATA PREPARATION
# ============================================================

def prepare_analytics_data(results):

    required = {
        "customerID",
        "churn_probability",
        "prediction",
        "risk_band",
        "MonthlyCharges",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "SeniorCitizen",
        "PaperlessBilling",
    }

    missing = required - set(results.columns)

    if missing:

        raise ValueError(
            "Missing analytics columns: "
            + ", ".join(sorted(missing))
        )

    if results.empty:

        raise ValueError(
            "No customer predictions available."
        )

    dataframe = results.copy()

    if dataframe["customerID"].duplicated().any():

        raise ValueError(
            "Duplicate customer IDs."
        )

    for column in [
        "churn_probability",
        "MonthlyCharges",
        "tenure",
        "prediction",
    ]:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

        if not np.isfinite(
            dataframe[column]
        ).all():

            raise ValueError(
                f"Non-finite values in {column}."
            )

    if not dataframe[
        "churn_probability"
    ].between(0, 1).all():

        raise ValueError(
            "Invalid churn probabilities."
        )

    if not dataframe[
        "prediction"
    ].isin([0, 1]).all():

        raise ValueError(
            "Invalid prediction labels."
        )

    if not dataframe[
        "risk_band"
    ].isin(RISK_ORDER).all():

        raise ValueError(
            "Invalid customer risk bands."
        )

    if (
        dataframe["MonthlyCharges"] < 0
    ).any():

        raise ValueError(
            "Monthly charges cannot be negative."
        )

    if (
        dataframe["tenure"] < 0
    ).any():

        raise ValueError(
            "Tenure cannot be negative."
        )

    dataframe["tenure_group"] = pd.cut(
        dataframe["tenure"],
        bins=[-1, 6, 24, 48, np.inf],
        labels=[
            "0–6 months",
            "7–24 months",
            "25–48 months",
            "49+ months",
        ],
    )

    dataframe["expected_monthly_revenue_loss"] = (
        dataframe["churn_probability"]
        * dataframe["MonthlyCharges"]
    )

    return dataframe


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def calculate_portfolio_summary(dataframe):

    total_customers = len(dataframe)

    if total_customers == 0:

        raise ValueError(
            "No customers available."
        )

    monthly_revenue = float(
        dataframe["MonthlyCharges"].sum()
    )

    predicted_churn = int(
        dataframe["prediction"].sum()
    )

    flagged_revenue = float(
        dataframe.loc[
            dataframe["prediction"] == 1,
            "MonthlyCharges",
        ].sum()
    )

    expected_loss = float(
        dataframe[
            "expected_monthly_revenue_loss"
        ].sum()
    )

    return {
        "total_customers": total_customers,

        "predicted_churn": predicted_churn,

        "predicted_churn_rate": (
            predicted_churn / total_customers
        ),

        "average_churn_probability": float(
            dataframe["churn_probability"].mean()
        ),

        "expected_churn_count": float(
            dataframe["churn_probability"].sum()
        ),

        "total_monthly_revenue": monthly_revenue,

        "flagged_monthly_revenue": flagged_revenue,

        "expected_monthly_revenue_loss": expected_loss,

        "expected_revenue_loss_rate": (
            expected_loss / monthly_revenue
            if monthly_revenue > 0
            else 0.0
        ),
    }


# ============================================================
# RISK DISTRIBUTION
# ============================================================

def calculate_risk_distribution(dataframe):

    distribution = (
        dataframe.groupby(
            "risk_band",
            observed=True,
        )
        .agg(
            customers=("customerID", "count"),

            average_probability=(
                "churn_probability",
                "mean",
            ),

            predicted_churn=(
                "prediction",
                "sum",
            ),

            monthly_revenue=(
                "MonthlyCharges",
                "sum",
            ),

            expected_revenue_loss=(
                "expected_monthly_revenue_loss",
                "sum",
            ),
        )
        .reindex(RISK_ORDER)
    )

    distribution = distribution.fillna(0)

    distribution["customers"] = (
        distribution["customers"].astype(int)
    )

    distribution["predicted_churn"] = (
        distribution["predicted_churn"].astype(int)
    )

    distribution["customer_share"] = (
        distribution["customers"]
        / len(dataframe)
    )

    return distribution.reset_index()


# ============================================================
# CUSTOMER SEGMENT ANALYSIS
# ============================================================

def calculate_segment_summary(
    dataframe,
    segment,
):

    if segment not in SEGMENT_COLUMNS:

        raise ValueError(
            "Unsupported customer segment."
        )

    summary = (
        dataframe.groupby(
            segment,
            observed=True,
            dropna=False,
        )
        .agg(
            customers=(
                "customerID",
                "count",
            ),

            predicted_churn=(
                "prediction",
                "sum",
            ),

            average_probability=(
                "churn_probability",
                "mean",
            ),

            expected_churn_count=(
                "churn_probability",
                "sum",
            ),

            monthly_revenue=(
                "MonthlyCharges",
                "sum",
            ),

            expected_revenue_loss=(
                "expected_monthly_revenue_loss",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["predicted_churn_rate"] = (
        summary["predicted_churn"]
        / summary["customers"]
    )

    summary["customer_share"] = (
        summary["customers"]
        / len(dataframe)
    )

    summary = summary.sort_values(
        "average_probability",
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# CUSTOMER PRIORITIZATION
# ============================================================

def get_priority_customers(
    dataframe,
    limit=100,
):

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):

        raise ValueError(
            "Limit must be a positive integer."
        )

    return dataframe.sort_values(
        by=[
            "expected_monthly_revenue_loss",
            "churn_probability",
            "customerID",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).head(limit).reset_index(drop=True)


# ============================================================
# GLOBAL SHAP REPORT LOADING
# ============================================================

def calculate_file_hash(path):

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def load_global_shap_report(
    reports_dir=SHAP_REPORTS_DIR,
):

    reports_dir = Path(reports_dir)

    csv_path = (
        reports_dir
        / "global_shap_importance.csv"
    )

    metadata_path = (
        reports_dir
        / "shap_metadata.json"
    )

    if not (
        csv_path.is_file()
        and metadata_path.is_file()
    ):

        raise FileNotFoundError(
            "Global SHAP reports are unavailable."
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    if calculate_file_hash(csv_path) != (
        metadata["importance_report_hash"]
    ):

        raise ValueError(
            "Global SHAP report fingerprint mismatch."
        )

    importance = pd.read_csv(csv_path)

    required = {
        "rank",
        "feature",
        "mean_absolute_shap",
        "mean_signed_shap",
    }

    if not required.issubset(
        importance.columns
    ):

        raise ValueError(
            "Invalid SHAP report columns."
        )

    if (
        len(importance)
        != metadata["feature_count"]
    ):

        raise ValueError(
            "SHAP feature count mismatch."
        )

    if importance["feature"].duplicated().any():

        raise ValueError(
            "Duplicate SHAP features."
        )

    values = importance[
        [
            "mean_absolute_shap",
            "mean_signed_shap",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            "Non-finite SHAP importance values."
        )

    if (values[:, 0] < 0).any():

        raise ValueError(
            "Negative absolute SHAP importance."
        )

    if (
        np.abs(values[:, 1])
        > values[:, 0] + 1e-8
    ).any():

        raise ValueError(
            "Inconsistent SHAP averages."
        )

    if metadata["output_space"] != "raw_log_odds":

        raise ValueError(
            "Unexpected SHAP output space."
        )

    if metadata["explanation_method"] != "TreeSHAP":

        raise ValueError(
            "Unexpected explanation method."
        )

    if metadata["dataset_partition"] != "validation":

        raise ValueError(
            "Unexpected explanation dataset."
        )

    if not metadata["additivity_verified_by_api"]:

        raise ValueError(
            "SHAP additivity verification is missing."
        )

    importance = importance.sort_values(
        "rank"
    ).reset_index(drop=True)

    return importance, metadata


def shap_report_matches_model(
    metadata,
    model_info,
):

    if model_info is None:

        return False

    fields = [
        "registered_model_name",
        "model_name",
        "model_version",
        "run_id",
        "calibration_method",
    ]

    return all(
        str(metadata.get(field))
        == str(model_info.get(field))
        and metadata.get(field) is not None
        for field in fields
    )
