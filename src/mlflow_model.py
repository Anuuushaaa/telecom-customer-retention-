
"""
Unified MLflow PyFunc inference model.

Supports:
- Logistic Regression
- XGBoost
- TensorFlow ANN
- Raw customer data validation
- Shared preprocessing
- Probability calibration
- Configurable classification thresholds
- Batch inference
"""

import json
from pathlib import Path

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.preprocessing import (
    prepare_raw_data,
    transform_features,
)

from src.feature_engineering import (
    engineer_features,
)

from src.probability_calibration import (
    apply_calibration,
)

from src.mlflow_config import (
    SUPPORTED_MODELS,
    CALIBRATION_METHODS,
)


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_classification_threshold(threshold):
    """Validate and return a classification threshold."""

    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError(
            "Threshold must be numeric."
        )

    try:
        threshold = float(threshold)

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Threshold must be numeric."
        ) from error

    if (
        not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return threshold


def validate_model_configuration(configuration):
    """Validate model inference configuration."""

    required = {
        "model_name",
        "calibration_method",
        "classification_threshold",
    }

    missing = required - set(configuration)

    if missing:
        raise ValueError(
            f"Missing configuration fields: "
            f"{sorted(missing)}"
        )

    if configuration["model_name"] not in SUPPORTED_MODELS:
        raise ValueError(
            "Unsupported model name."
        )

    if (
        configuration["calibration_method"]
        not in CALIBRATION_METHODS
    ):
        raise ValueError(
            "Unsupported calibration method."
        )

    validated = dict(configuration)

    validated["classification_threshold"] = (
        validate_classification_threshold(
            configuration["classification_threshold"]
        )
    )

    return validated


# ============================================================
# UNIFIED MLFLOW MODEL
# ============================================================

class TelecomChurnModel(mlflow.pyfunc.PythonModel):
    """Unified inference wrapper for telecom churn models."""

    def load_context(self, context):
        """Load model and preprocessing artifacts once."""

        required_artifacts = {
            "model",
            "scaler",
            "feature_columns",
            "configuration",
        }

        missing = (
            required_artifacts
            - set(context.artifacts)
        )

        if missing:
            raise ValueError(
                f"Missing model artifacts: "
                f"{sorted(missing)}"
            )

        with Path(
            context.artifacts["configuration"]
        ).open(
            "r",
            encoding="utf-8",
        ) as file:

            configuration = json.load(file)

        self.configuration = (
            validate_model_configuration(
                configuration
            )
        )

        self.model_name = (
            self.configuration["model_name"]
        )

        self.calibration_method = (
            self.configuration["calibration_method"]
        )

        self.classification_threshold = (
            self.configuration[
                "classification_threshold"
            ]
        )

        self.scaler = joblib.load(
            context.artifacts["scaler"]
        )

        self.feature_columns = list(
            joblib.load(
                context.artifacts["feature_columns"]
            )
        )

        if (
            not self.feature_columns
            or len(set(self.feature_columns))
            != len(self.feature_columns)
        ):
            raise ValueError(
                "Invalid feature columns artifact."
            )

        if self.model_name == "tensorflow_ann":

            import tensorflow as tf

            self.model = tf.keras.models.load_model(
                context.artifacts["model"],
                compile=False,
            )

            expected_features = int(
                self.model.input_shape[-1]
            )

        else:

            self.model = joblib.load(
                context.artifacts["model"]
            )

            if list(self.model.classes_) != [0, 1]:
                raise ValueError(
                    "Unexpected model class ordering."
                )

            expected_features = int(
                self.model.n_features_in_
            )

        if expected_features != len(
            self.feature_columns
        ):
            raise ValueError(
                "Model and preprocessing feature "
                "counts do not match."
            )

        self.calibrator = None

        if self.calibration_method != "uncalibrated":

            if "calibrator" not in context.artifacts:
                raise ValueError(
                    "Missing probability calibrator."
                )

            self.calibrator = joblib.load(
                context.artifacts["calibrator"]
            )

    # ========================================================
    # INPUT PREPARATION
    # ========================================================

    def prepare_features(self, model_input):
        """Validate raw input and transform features."""

        if not isinstance(
            model_input,
            pd.DataFrame,
        ):
            raise TypeError(
                "Model input must be a pandas DataFrame."
            )

        if model_input.empty:
            raise ValueError(
                "Model input cannot be empty."
            )

        if "Churn" in model_input.columns:
            raise ValueError(
                "Inference input must not contain "
                "the target column 'Churn'."
            )

        raw = model_input.copy()

        clean = prepare_raw_data(
            raw,
            require_target=False,
        )

        engineered = engineer_features(clean)

        features = transform_features(
            engineered,
            self.scaler,
            self.feature_columns,
        )

        if list(features.columns) != (
            self.feature_columns
        ):
            raise ValueError(
                "Unexpected feature ordering."
            )

        if not np.isfinite(
            features.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Non-finite transformed features."
            )

        return features

    # ========================================================
    # RAW MODEL PROBABILITIES
    # ========================================================

    def predict_raw_probabilities(self, features):
        """Generate uncalibrated churn probabilities."""

        if self.model_name == "tensorflow_ann":

            inputs = features.to_numpy(
                dtype=np.float32
            )

            probabilities = self.model.predict(
                inputs,
                verbose=0,
            ).reshape(-1)

        else:

            probabilities = (
                self.model.predict_proba(features)[:, 1]
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.shape != (len(features),):
            raise ValueError(
                "Unexpected probability shape."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Non-finite predicted probabilities."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Predicted probabilities outside [0, 1]."
            )

        return probabilities

    # ========================================================
    # CLASSIFICATION THRESHOLD
    # ========================================================

    def resolve_threshold(self, params=None):
        """Resolve default or user-specified threshold."""

        if params is None:
            return self.classification_threshold

        if not isinstance(params, dict):
            raise TypeError(
                "Prediction parameters must be a dictionary."
            )

        unsupported = (
            set(params)
            - {"classification_threshold"}
        )

        if unsupported:
            raise ValueError(
                f"Unsupported prediction parameters: "
                f"{sorted(unsupported)}"
            )

        return validate_classification_threshold(
            params.get(
                "classification_threshold",
                self.classification_threshold,
            )
        )

    # ========================================================
    # UNIFIED PREDICTION
    # ========================================================

    def predict(
        self,
        context,
        model_input,
        params=None,
    ):
        """Return calibrated churn predictions."""

        threshold = self.resolve_threshold(
            params
        )

        features = self.prepare_features(
            model_input
        )

        raw_probabilities = (
            self.predict_raw_probabilities(
                features
            )
        )

        calibrated_probabilities = (
            apply_calibration(
                raw_probabilities,
                self.calibration_method,
                self.calibrator,
            )
        )

        predictions = (
            calibrated_probabilities >= threshold
        ).astype(int)

        if "customerID" in model_input.columns:

            customer_ids = (
                model_input["customerID"]
                .astype(str)
                .to_numpy()
            )

        else:

            customer_ids = np.full(
                len(model_input),
                "unknown",
                dtype=object,
            )

        return pd.DataFrame({
            "customerID": customer_ids,
            "raw_churn_probability": raw_probabilities,
            "churn_probability": calibrated_probabilities,
            "prediction": predictions,
            "prediction_label": np.where(
                predictions == 1,
                "Churn",
                "No Churn",
            ),
            "classification_threshold": threshold,
            "model_name": self.model_name,
            "calibration_method": (
                self.calibration_method
            ),
        })
