
"""
Pydantic request and response schemas for the
Telecom Customer Retention API.
"""

import math

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# CATEGORICAL TYPES
# ============================================================

YesNo = Literal["Yes", "No"]

Gender = Literal["Female", "Male"]

MultipleLinesType = Literal[
    "Yes",
    "No",
    "No phone service",
]

InternetServiceType = Literal[
    "DSL",
    "Fiber optic",
    "No",
]

InternetAddonType = Literal[
    "Yes",
    "No",
    "No internet service",
]

ContractType = Literal[
    "Month-to-month",
    "One year",
    "Two year",
]

PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

ModelName = Literal[
    "logistic_regression",
    "xgboost",
    "tensorflow_ann",
]

CalibrationMethod = Literal[
    "uncalibrated",
    "sigmoid",
    "isotonic",
]


# ============================================================
# BASE SCHEMA
# ============================================================

class StrictSchema(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


# ============================================================
# CUSTOMER INPUT SCHEMA
# ============================================================

class CustomerInput(StrictSchema):

    customerID: str = Field(
        min_length=1,
        max_length=128,
        description="Unique customer identifier.",
    )

    gender: Gender

    SeniorCitizen: int = Field(
        ge=0,
        le=1,
        description="0 = No, 1 = Yes.",
    )

    Partner: YesNo

    Dependents: YesNo

    tenure: int = Field(
        ge=0,
        description="Customer tenure in months.",
    )

    PhoneService: YesNo

    MultipleLines: MultipleLinesType

    InternetService: InternetServiceType

    OnlineSecurity: InternetAddonType

    OnlineBackup: InternetAddonType

    DeviceProtection: InternetAddonType

    TechSupport: InternetAddonType

    StreamingTV: InternetAddonType

    StreamingMovies: InternetAddonType

    Contract: ContractType

    PaperlessBilling: YesNo

    PaymentMethod: PaymentMethodType

    MonthlyCharges: float = Field(
        ge=0,
        description="Monthly customer charges.",
    )

    TotalCharges: str | float | int = Field(
        description=(
            "Total accumulated charges. Numeric strings "
            "are accepted. Blank values are permitted "
            "only for customers with zero tenure."
        ),
    )

    @field_validator("customerID")
    @classmethod
    def validate_customer_id(cls, value):

        value = value.strip()

        if not value:

            raise ValueError(
                "customerID cannot be blank."
            )

        return value

    @field_validator(
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        mode="before",
    )
    @classmethod
    def reject_boolean_numbers(cls, value):

        if isinstance(value, bool):

            raise ValueError(
                "Boolean values are not valid "
                "numeric inputs."
            )

        return value

    @model_validator(mode="after")
    def validate_total_charges(self):

        value = self.TotalCharges

        if isinstance(value, str):

            value = value.strip()

            if not value:

                if self.tenure != 0:

                    raise ValueError(
                        "Blank TotalCharges is permitted "
                        "only when tenure is zero."
                    )

                self.TotalCharges = ""

                return self

        try:

            numeric_value = float(value)

        except (ValueError, TypeError) as exc:

            raise ValueError(
                "TotalCharges must be numeric."
            ) from exc

        if not math.isfinite(numeric_value):

            raise ValueError(
                "TotalCharges must be finite."
            )

        if numeric_value < 0:

            raise ValueError(
                "TotalCharges cannot be negative."
            )

        # MLflow's registered input signature
        # expects TotalCharges as a string.

        self.TotalCharges = str(numeric_value)

        return self


# ============================================================
# SINGLE PREDICTION REQUEST
# ============================================================

class PredictionRequest(StrictSchema):

    customer: CustomerInput

    classification_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional classification threshold override."
        ),
    )

    @field_validator(
        "classification_threshold",
        mode="before",
    )
    @classmethod
    def validate_threshold_type(cls, value):

        if isinstance(value, bool):

            raise ValueError(
                "Threshold cannot be a boolean."
            )

        return value


# ============================================================
# BATCH PREDICTION REQUEST
# ============================================================

class BatchPredictionRequest(StrictSchema):

    customers: list[CustomerInput] = Field(
        min_length=1,
        max_length=10000,
        description="Customers to score.",
    )

    classification_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "classification_threshold",
        mode="before",
    )
    @classmethod
    def validate_threshold_type(cls, value):

        if isinstance(value, bool):

            raise ValueError(
                "Threshold cannot be a boolean."
            )

        return value

    @model_validator(mode="after")
    def validate_unique_customers(self):

        customer_ids = [
            customer.customerID
            for customer in self.customers
        ]

        if len(customer_ids) != len(set(customer_ids)):

            raise ValueError(
                "Duplicate customer IDs are not "
                "permitted within a batch."
            )

        return self


# ============================================================
# PREDICTION RESPONSE
# ============================================================

class PredictionResponse(StrictSchema):

    customerID: str

    raw_churn_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    churn_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    prediction: Literal[0, 1]

    prediction_label: Literal[
        "Churn",
        "No Churn",
    ]

    classification_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    model_name: ModelName

    calibration_method: CalibrationMethod

    @model_validator(mode="after")
    def validate_prediction_consistency(self):

        expected_prediction = int(
            self.churn_probability
            >= self.classification_threshold
        )

        if self.prediction != expected_prediction:

            raise ValueError(
                "Prediction does not match the "
                "probability and threshold."
            )

        expected_label = (
            "Churn"
            if self.prediction == 1
            else "No Churn"
        )

        if self.prediction_label != expected_label:

            raise ValueError(
                "Prediction label is inconsistent."
            )

        return self


# ============================================================
# BATCH PREDICTION RESPONSE
# ============================================================

class BatchPredictionResponse(StrictSchema):

    total_customers: int = Field(ge=1)

    predicted_churn: int = Field(ge=0)

    predicted_no_churn: int = Field(ge=0)

    predictions: list[PredictionResponse] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_batch_summary(self):

        if self.total_customers != len(self.predictions):

            raise ValueError(
                "Total customers does not match "
                "the prediction count."
            )

        actual_churn = sum(
            prediction.prediction
            for prediction in self.predictions
        )

        actual_no_churn = (
            self.total_customers - actual_churn
        )

        if self.predicted_churn != actual_churn:

            raise ValueError(
                "Incorrect predicted churn count."
            )

        if self.predicted_no_churn != actual_no_churn:

            raise ValueError(
                "Incorrect predicted no-churn count."
            )

        return self


# ============================================================
# HEALTH AND READINESS RESPONSES
# ============================================================

class HealthResponse(StrictSchema):

    status: Literal["healthy"]

    service: str

    version: str


class ReadinessResponse(StrictSchema):

    status: Literal[
        "ready",
        "not_ready",
    ]

    model_loaded: bool

    model_alias: str

    model_version: str | None = None


# ============================================================
# MODEL INFORMATION RESPONSE
# ============================================================

class ModelInfoResponse(StrictSchema):

    registered_model_name: str

    model_alias: str

    model_version: str

    model_uri: str

    model_name: ModelName

    calibration_method: CalibrationMethod

    classification_threshold: float = Field(
        ge=0.0,
        le=1.0,
    )

    run_id: str

    max_batch_size: int = Field(ge=1)


# ============================================================
# ERROR RESPONSE
# ============================================================

class ErrorResponse(StrictSchema):

    error: str

    message: str

    request_id: str | None = None
