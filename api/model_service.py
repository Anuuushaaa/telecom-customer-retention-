
"""
MLflow model-serving service for the Telecom
Customer Retention API.

Loads the registered model once and provides
validated single and batch predictions.
"""

import logging

from threading import RLock

import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow import MlflowClient

from api.config import APISettings, get_settings

from api.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    ReadinessResponse,
)

from src.mlflow_model import (
    validate_classification_threshold,
)


logger = logging.getLogger(__name__)


# ============================================================
# SERVICE EXCEPTIONS
# ============================================================

class ModelServiceError(RuntimeError):

    """Base exception for model-serving errors."""


class ModelNotReadyError(ModelServiceError):

    """Raised when the prediction model is unavailable."""


class ModelLoadingError(ModelServiceError):

    """Raised when model initialization fails."""


class ModelPredictionError(ModelServiceError):

    """Raised when model inference fails."""


class BatchSizeError(ValueError):

    """Raised when the configured batch limit is exceeded."""


# ============================================================
# MODEL SERVICE
# ============================================================

class ModelService:

    def __init__(
        self,
        settings: APISettings | None = None,
    ):

        self.settings = settings or get_settings()

        self._model = None

        self._model_info = None

        self._lock = RLock()


    # ========================================================
    # MODEL READINESS
    # ========================================================

    @property
    def is_ready(self) -> bool:

        with self._lock:

            return (
                self._model is not None
                and self._model_info is not None
            )


    # ========================================================
    # MLFLOW CONFIGURATION
    # ========================================================

    def configure_mlflow(self) -> None:

        mlflow.set_tracking_uri(
            self.settings.tracking_uri
        )

        mlflow.set_registry_uri(
            self.settings.registry_uri
        )


    def create_client(self) -> MlflowClient:

        return MlflowClient(
            tracking_uri=self.settings.tracking_uri,
            registry_uri=self.settings.registry_uri,
        )


    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def load_model(self) -> ModelInfoResponse:

        with self._lock:

            if self.is_ready:

                logger.info(
                    "Model already loaded. "
                    "Reusing cached model."
                )

                return self._model_info

            logger.info(
                "Initializing registered model: %s",
                self.settings.model_uri,
            )

            try:

                self.configure_mlflow()

                client = self.create_client()

                version = (
                    client.get_model_version_by_alias(
                        name=(
                            self.settings
                            .registered_model_name
                        ),
                        alias=self.settings.model_alias,
                    )
                )

                if str(version.status) != "READY":

                    raise ModelLoadingError(
                        "Registered model version "
                        "is not ready."
                    )

                model_version = str(version.version)

                tags = version.tags or {}

                model_name = tags.get("algorithm")

                calibration_method = tags.get(
                    "calibration_method"
                )

                if not model_name:

                    raise ModelLoadingError(
                        "Registered model is missing "
                        "algorithm metadata."
                    )

                if not calibration_method:

                    raise ModelLoadingError(
                        "Registered model is missing "
                        "calibration metadata."
                    )

                if not version.run_id:

                    raise ModelLoadingError(
                        "Registered model is missing "
                        "its source run ID."
                    )

                # Resolve the alias once and load the
                # exact registered model version.

                version_uri = (
                    f"models:/"
                    f"{self.settings.registered_model_name}"
                    f"/{model_version}"
                )

                model = mlflow.pyfunc.load_model(
                    version_uri
                )

                model_info = ModelInfoResponse(

                    registered_model_name=(
                        self.settings
                        .registered_model_name
                    ),

                    model_alias=(
                        self.settings.model_alias
                    ),

                    model_version=model_version,

                    model_uri=version_uri,

                    model_name=model_name,

                    calibration_method=(
                        calibration_method
                    ),

                    classification_threshold=(
                        self.settings
                        .classification_threshold
                    ),

                    run_id=str(version.run_id),

                    max_batch_size=(
                        self.settings.max_batch_size
                    ),
                )

                # Publish model state only after all
                # initialization steps succeed.

                self._model = model

                self._model_info = model_info

                logger.info(
                    "Model loaded successfully: "
                    "algorithm=%s version=%s "
                    "alias=%s calibration=%s",
                    model_name,
                    model_version,
                    self.settings.model_alias,
                    calibration_method,
                )

                return model_info

            except Exception as exc:

                self._model = None

                self._model_info = None

                logger.exception(
                    "Registered model loading failed."
                )

                raise ModelLoadingError(
                    "Unable to initialize the "
                    "registered prediction model."
                ) from exc


    # ========================================================
    # MODEL METADATA
    # ========================================================

    def get_model_info(self) -> ModelInfoResponse:

        with self._lock:

            if not self.is_ready:

                raise ModelNotReadyError(
                    "Prediction model is not loaded."
                )

            return self._model_info


    def get_readiness(self) -> ReadinessResponse:

        with self._lock:

            ready = self.is_ready

            return ReadinessResponse(

                status=(
                    "ready"
                    if ready
                    else "not_ready"
                ),

                model_loaded=ready,

                model_alias=self.settings.model_alias,

                model_version=(
                    self._model_info.model_version
                    if ready
                    else None
                ),
            )


    # ========================================================
    # THRESHOLD RESOLUTION
    # ========================================================

    def resolve_threshold(
        self,
        classification_threshold: float | None = None,
    ) -> float:

        if classification_threshold is None:

            classification_threshold = (
                self.settings
                .classification_threshold
            )

        return validate_classification_threshold(
            classification_threshold
        )


    # ========================================================
    # BATCH SIZE VALIDATION
    # ========================================================

    def validate_batch_size(
        self,
        customers: list[CustomerInput],
    ) -> None:

        if not customers:

            raise BatchSizeError(
                "At least one customer is required."
            )

        if len(customers) > (
            self.settings.max_batch_size
        ):

            raise BatchSizeError(
                f"Batch contains {len(customers)} "
                f"customers. Maximum permitted size "
                f"is {self.settings.max_batch_size}."
            )


    # ========================================================
    # INPUT PREPARATION
    # ========================================================

    def prepare_input(
        self,
        customers: list[CustomerInput],
    ) -> pd.DataFrame:

        self.validate_batch_size(customers)

        records = [

            customer.model_dump()

            for customer in customers
        ]

        dataframe = pd.DataFrame.from_records(
            records
        )

        if dataframe["customerID"].duplicated().any():

            raise ValueError(
                "Duplicate customer IDs are not "
                "permitted within a batch."
            )

        # Match the registered MLflow input signature.

        dataframe["customerID"] = (
            dataframe["customerID"].astype(str)
        )

        dataframe["SeniorCitizen"] = (
            dataframe["SeniorCitizen"].astype("int64")
        )

        dataframe["tenure"] = (
            dataframe["tenure"].astype("int64")
        )

        dataframe["MonthlyCharges"] = (
            dataframe["MonthlyCharges"].astype(
                "float64"
            )
        )

        dataframe["TotalCharges"] = (
            dataframe["TotalCharges"].astype(str)
        )

        return dataframe


    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    def validate_predictions(
        self,
        predictions: pd.DataFrame,
        expected_customer_ids: list[str],
        threshold: float,
    ) -> list[PredictionResponse]:

        if not isinstance(
            predictions,
            pd.DataFrame,
        ):

            raise ModelPredictionError(
                "Model returned an unsupported "
                "prediction format."
            )

        if len(predictions) != len(
            expected_customer_ids
        ):

            raise ModelPredictionError(
                "Prediction count does not match "
                "the input customer count."
            )

        required_columns = set(
            PredictionResponse.model_fields
        )

        missing_columns = (
            required_columns
            - set(predictions.columns)
        )

        if missing_columns:

            raise ModelPredictionError(
                "Model output is missing "
                "required prediction fields."
            )

        actual_customer_ids = (
            predictions["customerID"].tolist()
        )

        if actual_customer_ids != (
            expected_customer_ids
        ):

            raise ModelPredictionError(
                "Prediction customer IDs are "
                "not aligned with the request."
            )

        model_info = self.get_model_info()

        responses = []

        for record in predictions.to_dict(
            orient="records"
        ):

            response = (
                PredictionResponse.model_validate(
                    record
                )
            )

            if response.model_name != (
                model_info.model_name
            ):

                raise ModelPredictionError(
                    "Prediction algorithm does not "
                    "match the loaded model."
                )

            if response.calibration_method != (
                model_info.calibration_method
            ):

                raise ModelPredictionError(
                    "Prediction calibration method "
                    "does not match model metadata."
                )

            if response.classification_threshold != (
                threshold
            ):

                raise ModelPredictionError(
                    "Model did not apply the "
                    "requested classification threshold."
                )

            responses.append(response)

        return responses


    # ========================================================
    # CORE INFERENCE
    # ========================================================

    def predict_customers(
        self,
        customers: list[CustomerInput],
        classification_threshold: float | None = None,
    ) -> list[PredictionResponse]:

        with self._lock:

            if not self.is_ready:

                raise ModelNotReadyError(
                    "Prediction model is not loaded."
                )

            threshold = self.resolve_threshold(
                classification_threshold
            )

            dataframe = self.prepare_input(
                customers
            )

            expected_ids = (
                dataframe["customerID"].tolist()
            )

            try:

                predictions = self._model.predict(

                    dataframe,

                    params={
                        "classification_threshold": (
                            threshold
                        )
                    },
                )

                responses = (
                    self.validate_predictions(
                        predictions=predictions,
                        expected_customer_ids=(
                            expected_ids
                        ),
                        threshold=threshold,
                    )
                )

                logger.info(
                    "Inference completed: "
                    "customers=%s model=%s "
                    "version=%s threshold=%s",
                    len(responses),
                    self._model_info.model_name,
                    self._model_info.model_version,
                    threshold,
                )

                return responses

            except ModelPredictionError:

                logger.exception(
                    "Prediction validation failed."
                )

                raise

            except Exception as exc:

                logger.exception(
                    "Model inference failed."
                )

                raise ModelPredictionError(
                    "Unable to generate predictions."
                ) from exc


    # ========================================================
    # SINGLE CUSTOMER PREDICTION
    # ========================================================

    def predict(
        self,
        customer: CustomerInput,
        classification_threshold: float | None = None,
    ) -> PredictionResponse:

        responses = self.predict_customers(

            customers=[customer],

            classification_threshold=(
                classification_threshold
            ),
        )

        return responses[0]


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        customers: list[CustomerInput],
        classification_threshold: float | None = None,
    ) -> BatchPredictionResponse:

        predictions = self.predict_customers(

            customers=customers,

            classification_threshold=(
                classification_threshold
            ),
        )

        predicted_churn = sum(

            prediction.prediction

            for prediction in predictions
        )

        total_customers = len(predictions)

        return BatchPredictionResponse(

            total_customers=total_customers,

            predicted_churn=predicted_churn,

            predicted_no_churn=(
                total_customers - predicted_churn
            ),

            predictions=predictions,
        )


    # ========================================================
    # RESOURCE CLEANUP
    # ========================================================

    def unload_model(self) -> None:

        with self._lock:

            self._model = None

            self._model_info = None

            logger.info(
                "Prediction model released."
            )
