
"""Batch customer prediction and CSV validation."""

import io

import numpy as np
import pandas as pd

from api.schemas import CustomerInput
from frontend.api_client import APIClientError
from pydantic import ValidationError


MAX_UPLOAD_ROWS = 50000

REQUIRED_COLUMNS = list(
    CustomerInput.model_fields
)


# ============================================================
# CSV VALIDATION
# ============================================================

def load_uploaded_csv(file_content):

    if not file_content:

        raise ValueError(
            "The uploaded CSV is empty."
        )

    try:

        dataframe = pd.read_csv(
            io.BytesIO(file_content),
            dtype={
                "customerID": str,
                "TotalCharges": str,
            },
            keep_default_na=False,
        )

    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as error:

        raise ValueError(
            "Unable to read the uploaded CSV."
        ) from error

    return dataframe


def validate_batch_dataframe(dataframe):

    if dataframe.empty:

        raise ValueError(
            "The CSV contains no customer records."
        )

    if len(dataframe) > MAX_UPLOAD_ROWS:

        raise ValueError(
            f"Maximum upload size is "
            f"{MAX_UPLOAD_ROWS:,} customer records."
        )

    if dataframe.columns.duplicated().any():

        raise ValueError(
            "Duplicate column names detected."
        )

    missing = set(REQUIRED_COLUMNS) - set(
        dataframe.columns
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    ignored_columns = [
        column
        for column in dataframe.columns
        if column not in REQUIRED_COLUMNS
    ]

    dataframe = dataframe[
        REQUIRED_COLUMNS
    ].copy()

    customers = []
    errors = []

    for index, record in enumerate(
        dataframe.to_dict(orient="records"),
        start=2,
    ):

        try:

            customer = CustomerInput.model_validate(
                record
            )

            customers.append(
                customer.model_dump(mode="json")
            )

        except ValidationError as error:

            for issue in error.errors():

                field = ".".join(
                    str(part)
                    for part in issue["loc"]
                )

                errors.append(
                    f"CSV row {index}: "
                    f"{field} — {issue['msg']}"
                )

            if len(errors) >= 20:

                break

    if errors:

        raise ValueError(
            "Customer validation failed:\n"
            + "\n".join(errors[:20])
        )

    customer_ids = [
        customer["customerID"]
        for customer in customers
    ]

    if len(customer_ids) != len(set(customer_ids)):

        raise ValueError(
            "Duplicate customer IDs detected."
        )

    return customers, ignored_columns


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk_band(probability):

    if not np.isfinite(probability):

        raise ValueError(
            "Probability must be finite."
        )

    if not 0 <= probability <= 1:

        raise ValueError(
            "Probability must be between 0 and 1."
        )

    if probability < 0.25:

        return "Low"

    if probability < 0.50:

        return "Moderate"

    if probability < 0.75:

        return "High"

    return "Very High"


# ============================================================
# BATCH RESPONSE VERIFICATION
# ============================================================

def validate_batch_response(
    response,
    customers,
    model_info,
    threshold,
):

    expected_ids = [
        customer["customerID"]
        for customer in customers
    ]

    predictions = response["predictions"]

    if (
        response["total_customers"] != len(customers)
        or len(predictions) != len(customers)
    ):

        raise APIClientError(
            "API batch customer count mismatch."
        )

    returned_ids = [
        prediction["customerID"]
        for prediction in predictions
    ]

    if returned_ids != expected_ids:

        raise APIClientError(
            "API returned misaligned customer IDs."
        )

    churn_count = 0

    for prediction in predictions:

        if prediction["model_name"] != (
            model_info["model_name"]
        ):

            raise APIClientError(
                "Prediction model mismatch."
            )

        if prediction["calibration_method"] != (
            model_info["calibration_method"]
        ):

            raise APIClientError(
                "Calibration method mismatch."
            )

        probability = float(
            prediction["churn_probability"]
        )

        raw_probability = float(
            prediction["raw_churn_probability"]
        )

        for value in (
            probability,
            raw_probability,
        ):

            if (
                not np.isfinite(value)
                or not 0 <= value <= 1
            ):

                raise APIClientError(
                    "Invalid prediction probability."
                )

        if not np.isclose(
            prediction["classification_threshold"],
            threshold,
            atol=1e-12,
            rtol=0,
        ):

            raise APIClientError(
                "Classification threshold mismatch."
            )

        expected_prediction = int(
            probability >= threshold
        )

        if prediction["prediction"] != (
            expected_prediction
        ):

            raise APIClientError(
                "Inconsistent predicted outcome."
            )

        expected_label = (
            "Churn"
            if expected_prediction
            else "No Churn"
        )

        if prediction["prediction_label"] != (
            expected_label
        ):

            raise APIClientError(
                "Inconsistent prediction label."
            )

        churn_count += expected_prediction

    if (
        response["predicted_churn"] != churn_count
        or response["predicted_no_churn"]
        != len(customers) - churn_count
    ):

        raise APIClientError(
            "Batch prediction summary mismatch."
        )

    return predictions


# ============================================================
# CHUNKED BATCH INFERENCE
# ============================================================

def run_batch_prediction(
    client,
    customers,
    threshold,
    model_info,
    progress_callback=None,
):

    if not customers:

        raise ValueError(
            "No customers supplied."
        )

    if (
        isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):

        raise ValueError(
            "Invalid classification threshold."
        )

    if client.get_model_info() != model_info:

        raise APIClientError(
            "Active model changed before prediction."
        )

    batch_size = int(
        model_info["max_batch_size"]
    )

    if batch_size <= 0:

        raise ValueError(
            "Invalid API batch limit."
        )

    predictions = []

    total = len(customers)

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch = customers[
            start:start + batch_size
        ]

        response = client.predict_batch(
            customers=batch,
            classification_threshold=threshold,
        )

        validated = validate_batch_response(
            response=response,
            customers=batch,
            model_info=model_info,
            threshold=threshold,
        )

        predictions.extend(validated)

        if client.get_model_info() != model_info:

            raise APIClientError(
                "Active model changed during prediction. "
                "Batch results were discarded."
            )

        if progress_callback is not None:

            progress_callback(
                len(predictions),
                total,
            )

    results = pd.DataFrame(predictions)

    results["risk_band"] = results[
        "churn_probability"
    ].map(get_risk_band)

    results["model_version"] = str(
        model_info["model_version"]
    )

    results["model_run_id"] = model_info[
        "run_id"
    ]

    original = pd.DataFrame(customers)

    results = original.merge(
        results,
        on="customerID",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(results) != total:

        raise APIClientError(
            "Final batch result count mismatch."
        )

    return results


# ============================================================
# BATCH SUMMARY
# ============================================================

def summarize_batch(results):

    if results.empty:

        raise ValueError(
            "No prediction results available."
        )

    total = len(results)

    churn = int(
        results["prediction"].sum()
    )

    return {
        "total_customers": total,
        "predicted_churn": churn,
        "predicted_no_churn": total - churn,
        "predicted_churn_rate": churn / total,
        "average_churn_probability": float(
            results["churn_probability"].mean()
        ),
        "expected_churn_count": float(
            results["churn_probability"].sum()
        ),
        "high_risk_customers": int(
            results["risk_band"].isin(
                ["High", "Very High"]
            ).sum()
        ),
    }


def export_predictions(results):

    return results.to_csv(
        index=False
    ).encode("utf-8")
