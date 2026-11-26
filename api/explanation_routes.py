"""Validated REST endpoints for local and cohort TreeSHAP explanations."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from api.schemas import (
    StrictSchema, PredictionRequest, PredictionResponse,
    CustomerInput, CalibrationMethod, ErrorResponse,
)
from api.model_service import (
    ModelService, ModelNotReadyError, ModelPredictionError, BatchSizeError,
)
from api.explanation_service import ExplanationService, ExplanationUnavailableError


class ExplanationRequest(PredictionRequest):
    top_n: int = Field(default=10, ge=1, le=100)


class GlobalExplanationRequest(StrictSchema):
    customers: list[CustomerInput] = Field(min_length=1, max_length=10000)


class FeatureContribution(StrictSchema):
    feature: str
    transformed_value: float
    shap_value: float
    direction: Literal['increases_risk', 'decreases_risk', 'neutral']


class ExplanationResponse(StrictSchema):
    customerID: str
    model_name: Literal['xgboost']
    model_version: str
    calibration_method: CalibrationMethod
    explanation_method: Literal['TreeSHAP']
    output_space: Literal['raw_log_odds']
    prediction: PredictionResponse
    base_value: float
    raw_margin: float
    reconstructed_margin: float
    reconstructed_raw_probability: float = Field(ge=0, le=1)
    additivity_error: float = Field(ge=0)
    total_features: int = Field(ge=1)
    remaining_contribution: float
    top_contributors: list[FeatureContribution] = Field(min_length=1)


class FeatureImportance(StrictSchema):
    feature: str
    mean_absolute_shap: float = Field(ge=0)
    mean_signed_shap: float


class GlobalExplanationResponse(StrictSchema):
    model_name: Literal['xgboost']
    model_version: str
    explanation_method: Literal['TreeSHAP']
    output_space: Literal['raw_log_odds']
    total_customers: int = Field(ge=1)
    total_features: int = Field(ge=1)
    max_additivity_error: float = Field(ge=0)
    feature_importance: list[FeatureImportance] = Field(min_length=1)


router = APIRouter(tags=['Explainability'])
ERRORS = {code: {'model': ErrorResponse} for code in (400, 413, 422, 500, 503)}


def get_explanation_service(request: Request) -> ExplanationService:
    return ExplanationService(request.app.state.model_service)


def raise_http_error(exc):
    if isinstance(exc, BatchSizeError):
        code, detail = 413, 'Explanation batch exceeds configured limit.'
    elif isinstance(exc, ModelNotReadyError):
        code, detail = 503, 'Prediction model is unavailable.'
    elif isinstance(exc, ExplanationUnavailableError):
        code, detail = 503, 'Explanations are unavailable for the active model.'
    elif isinstance(exc, ValueError):
        code, detail = 400, 'Invalid explanation request.'
    else:
        code, detail = 500, 'Unable to generate explanation.'
    raise HTTPException(status_code=code, detail=detail) from exc


@router.post('/explain', response_model=ExplanationResponse, responses=ERRORS)
def explain(request: ExplanationRequest, service: ExplanationService = Depends(get_explanation_service)):
    try:
        return service.explain_customer(request.customer, request.classification_threshold, request.top_n)
    except (ModelNotReadyError, ExplanationUnavailableError, ModelPredictionError, ValueError) as exc:
        raise_http_error(exc)


@router.post('/explain/global', response_model=GlobalExplanationResponse, responses=ERRORS)
def explain_global(request: GlobalExplanationRequest, service: ExplanationService = Depends(get_explanation_service)):
    try:
        return service.explain_global(request.customers)
    except (ModelNotReadyError, ExplanationUnavailableError, ModelPredictionError, ValueError) as exc:
        raise_http_error(exc)
