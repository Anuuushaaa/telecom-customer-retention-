
"""SHAP explanations for the registered XGBoost model."""

import logging

import numpy as np
import pandas as pd
import shap

from scipy.special import expit

from api.model_service import (
    ModelNotReadyError,
    ModelPredictionError,
)


logger = logging.getLogger(__name__)

ADDITIVITY_ATOL = 1e-4
ADDITIVITY_RTOL = 1e-5


class ExplanationUnavailableError(RuntimeError):
    """Raised when the loaded model cannot be explained."""


def calculate_shap_values(model, features):

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "Features must be a pandas DataFrame."
        )

    if features.empty:
        raise ValueError(
            "Features cannot be empty."
        )

    explainer = shap.TreeExplainer(
        model,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )

    explanation = explainer(
        features,
        check_additivity=False,
    )

    values = np.asarray(
        explanation.values,
        dtype=float,
    )

    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    ).reshape(-1)

    if values.shape != features.shape:
        raise ValueError(
            "Unexpected SHAP contribution shape."
        )

    if base_values.size == 1:

        base_values = np.repeat(
            base_values,
            len(features),
        )

    if base_values.shape != (len(features),):
        raise ValueError(
            "Unexpected SHAP baseline shape."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "SHAP contributions contain non-finite values."
        )

    if not np.isfinite(base_values).all():
        raise ValueError(
            "SHAP baselines contain non-finite values."
        )

    raw_margins = np.asarray(
        model.predict(
            features,
            output_margin=True,
        ),
        dtype=float,
    ).reshape(-1)

    if raw_margins.shape != (len(features),):
        raise ValueError(
            "Unexpected raw margin shape."
        )

    if not np.isfinite(raw_margins).all():
        raise ValueError(
            "Raw margins contain non-finite values."
        )

    reconstructed_margins = (
        base_values + values.sum(axis=1)
    )

    errors = np.abs(
        reconstructed_margins - raw_margins
    )

    if not np.allclose(
        reconstructed_margins,
        raw_margins,
        atol=ADDITIVITY_ATOL,
        rtol=ADDITIVITY_RTOL,
    ):
        raise ValueError(
            "SHAP additivity verification failed."
        )

    return {
        "values": values,
        "base_values": base_values,
        "raw_margins": raw_margins,
        "reconstructed_margins": reconstructed_margins,
        "additivity_errors": errors,
    }


class ExplanationService:

    def __init__(self, model_service):

        self.model_service = model_service

    def get_python_model(self):

        service = self.model_service

        if not service.is_ready:

            raise ModelNotReadyError(
                "Prediction model is not loaded."
            )

        model_info = service.get_model_info()

        if model_info.model_name != "xgboost":

            raise ExplanationUnavailableError(
                "SHAP explanations currently support "
                "the XGBoost model only."
            )

        python_model = (
            service._model.unwrap_python_model()
        )

        if python_model.model_name != model_info.model_name:

            raise ModelPredictionError(
                "Explanation model does not match "
                "registered model metadata."
            )

        return python_model, model_info

    def explain_customer(
        self,
        customer,
        classification_threshold=None,
        top_n=10,
    ):

        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or not 1 <= top_n <= 100
        ):
            raise ValueError(
                "top_n must be an integer between 1 and 100."
            )

        service = self.model_service

        with service._lock:

            python_model, model_info = (
                self.get_python_model()
            )

            dataframe = service.prepare_input(
                [customer]
            )

            try:

                prediction = service.predict(
                    customer,
                    classification_threshold,
                )

                features = python_model.prepare_features(
                    dataframe
                )

                result = calculate_shap_values(
                    python_model.model,
                    features,
                )

                values = result["values"][0]

                base_value = float(
                    result["base_values"][0]
                )

                raw_margin = float(
                    result["raw_margins"][0]
                )

                reconstructed_margin = float(
                    result["reconstructed_margins"][0]
                )

                additivity_error = float(
                    result["additivity_errors"][0]
                )

                reconstructed_probability = float(
                    expit(reconstructed_margin)
                )

                if not np.isclose(
                    reconstructed_probability,
                    prediction.raw_churn_probability,
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise ValueError(
                        "SHAP explanation does not match "
                        "the model's raw probability."
                    )

                contributions = []

                for index, feature in enumerate(
                    features.columns
                ):

                    contribution = float(
                        values[index]
                    )

                    direction = (
                        "increases_risk"
                        if contribution > 0
                        else "decreases_risk"
                        if contribution < 0
                        else "neutral"
                    )

                    contributions.append({
                        "feature": str(feature),
                        "transformed_value": float(
                            features.iloc[0, index]
                        ),
                        "shap_value": contribution,
                        "direction": direction,
                    })

                contributions.sort(
                    key=lambda item: (
                        -abs(item["shap_value"]),
                        item["feature"],
                    )
                )

                top_contributions = contributions[:top_n]

                remaining_contribution = float(
                    sum(
                        item["shap_value"]
                        for item in contributions[top_n:]
                    )
                )

                return {
                    "customerID": customer.customerID,
                    "model_name": model_info.model_name,
                    "model_version": model_info.model_version,
                    "calibration_method": (
                        model_info.calibration_method
                    ),
                    "explanation_method": "TreeSHAP",
                    "output_space": "raw_log_odds",
                    "prediction": prediction.model_dump(),
                    "base_value": base_value,
                    "raw_margin": raw_margin,
                    "reconstructed_margin": (
                        reconstructed_margin
                    ),
                    "reconstructed_raw_probability": (
                        reconstructed_probability
                    ),
                    "additivity_error": additivity_error,
                    "total_features": len(contributions),
                    "remaining_contribution": (
                        remaining_contribution
                    ),
                    "top_contributors": top_contributions,
                }

            except ModelPredictionError:
                raise

            except Exception as error:

                logger.error(
                    "SHAP explanation failed: %s",
                    type(error).__name__,
                )

                raise ModelPredictionError(
                    "Unable to generate SHAP explanation."
                ) from error

    def explain_global(self, customers):

        service = self.model_service

        with service._lock:

            python_model, model_info = (
                self.get_python_model()
            )

            dataframe = service.prepare_input(
                customers
            )

            try:

                features = python_model.prepare_features(
                    dataframe
                )

                result = calculate_shap_values(
                    python_model.model,
                    features,
                )

                values = result["values"]

                mean_absolute = np.mean(
                    np.abs(values),
                    axis=0,
                )

                mean_signed = np.mean(
                    values,
                    axis=0,
                )

                importance = []

                for index, feature in enumerate(
                    features.columns
                ):

                    importance.append({
                        "feature": str(feature),
                        "mean_absolute_shap": float(
                            mean_absolute[index]
                        ),
                        "mean_signed_shap": float(
                            mean_signed[index]
                        ),
                    })

                importance.sort(
                    key=lambda item: (
                        -item["mean_absolute_shap"],
                        item["feature"],
                    )
                )

                return {
                    "model_name": model_info.model_name,
                    "model_version": model_info.model_version,
                    "explanation_method": "TreeSHAP",
                    "output_space": "raw_log_odds",
                    "total_customers": len(customers),
                    "total_features": len(importance),
                    "max_additivity_error": float(
                        result["additivity_errors"].max()
                    ),
                    "feature_importance": importance,
                }

            except Exception as error:

                logger.error(
                    "Global SHAP explanation failed: %s",
                    type(error).__name__,
                )

                raise ModelPredictionError(
                    "Unable to generate global SHAP explanations."
                ) from error
