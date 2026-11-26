"""
FastAPI application for Telecom Customer Retention.
Provides REST endpoints for calibrated churn
prediction using the MLflow model registry.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from api.explanation_routes import router as explanation_router

from api.config import (
    APISettings,
    get_settings,
)
from api.model_service import (
    ModelService,
    ModelLoadingError,
    ModelNotReadyError,
    ModelPredictionError,
    BatchSizeError,
)
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from api.exceptions import (
    register_exception_handlers,
)
from api.middleware import (
    register_middleware,
)
logger = logging.getLogger(__name__)
# ============================================================
# COMMON ERROR RESPONSES
# ============================================================
COMMON_ERROR_RESPONSES = {
    422: {
        "model": ErrorResponse,
        "description": "Request validation failed.",
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal server error.",
    },
}
PREDICTION_ERROR_RESPONSES = {
    **COMMON_ERROR_RESPONSES,
    400: {
        "model": ErrorResponse,
        "description": "Invalid prediction request.",
    },
    503: {
        "model": ErrorResponse,
        "description": "Model unavailable.",
    },
}
# ============================================================
# APPLICATION FACTORY
# ============================================================
def create_app(
    settings: APISettings | None = None,
    model_service: ModelService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    service = model_service or ModelService(
        settings=settings
    )
    # ========================================================
    # APPLICATION LIFECYCLE
    # ========================================================
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(
            level=getattr(
                logging,
                settings.log_level,
            ),
            format=(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            ),
        )
        application.state.model_service = service
        application.state.settings = settings
        logger.info(
            "Starting %s version %s",
            settings.title,
            settings.version,
        )
        try:
            service.load_model()
            logger.info(
                "Prediction service is ready."
            )
        except ModelLoadingError:
            logger.exception(
                "Model initialization failed. "
                "API remains available for "
                "health and readiness checks."
            )
        try:
            yield
        finally:
            service.unload_model()
            logger.info(
                "Prediction service shut down."
            )
    # ========================================================
    # APPLICATION INITIALIZATION
    # ========================================================
    application = FastAPI(
        title=settings.title,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "Health",
                "description": (
                    "Application health and "
                    "model readiness checks."
                ),
            },
            {
                "name": "Model",
                "description": (
                    "Registered model information."
                ),
            },
            {
                "name": "Prediction",
                "description": (
                    "Single and batch customer "
                    "churn predictions."
                ),
            },
        ],
    )
    register_exception_handlers(
        application
    )
    register_middleware(
        application
    )
    # ========================================================
    # DEPENDENCY INJECTION
    # ========================================================
    def get_model_service(
        request: Request,
    ) -> ModelService:
        return request.app.state.model_service
    # ========================================================
    # ROOT ENDPOINT
    # ========================================================
    @application.get(
        "/",
        include_in_schema=False,
    )
    def root():
        return {
            "service": settings.title,
            "version": settings.version,
            "documentation": "/docs",
            "health": "/health",
            "readiness": "/ready",
            "model_information": "/model/info",
        }
    # ========================================================
    # HEALTH ENDPOINT
    # ========================================================
    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Check application health",
    )
    def health():
        return HealthResponse(
            status="healthy",
            service=settings.title,
            version=settings.version,
        )
    # ========================================================
    # READINESS ENDPOINT
    # ========================================================
    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["Health"],
        summary="Check prediction model readiness",
        responses={
            503: {
                "model": ReadinessResponse,
                "description": "Model unavailable.",
            },
        },
    )
    def readiness(
        response: Response,
        model_service: ModelService = Depends(
            get_model_service
        ),
    ):
        readiness_status = (
            model_service.get_readiness()
        )
        if not readiness_status.model_loaded:
            response.status_code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
            )
        return readiness_status
    # ========================================================
    # MODEL INFORMATION ENDPOINT
    # ========================================================
    @application.get(
        "/model/info",
        response_model=ModelInfoResponse,
        tags=["Model"],
        summary="Retrieve active model metadata",
        responses={
            503: {
                "model": ErrorResponse,
                "description": "Model unavailable.",
            },
            500: {
                "model": ErrorResponse,
                "description": "Internal server error.",
            },
        },
    )
    def model_information(
        model_service: ModelService = Depends(
            get_model_service
        ),
    ):
        try:
            return model_service.get_model_info()
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable.",
            ) from exc
    # ========================================================
    # SINGLE PREDICTION ENDPOINT
    # ========================================================
    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Prediction"],
        summary="Predict churn for one customer",
        responses=PREDICTION_ERROR_RESPONSES,
    )
    def predict(
        request: PredictionRequest,
        model_service: ModelService = Depends(
            get_model_service
        ),
    ):
        try:
            return model_service.predict(
                customer=request.customer,
                classification_threshold=(
                    request.classification_threshold
                ),
            )
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid prediction request.",
            ) from exc
        except ModelPredictionError as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to generate prediction.",
            ) from exc
    # ========================================================
    # BATCH PREDICTION ENDPOINT
    # ========================================================
    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["Prediction"],
        summary="Predict churn for multiple customers",
        responses={
            **PREDICTION_ERROR_RESPONSES,
            413: {
                "model": ErrorResponse,
                "description": "Batch limit exceeded.",
            },
        },
    )
    def predict_batch(
        request: BatchPredictionRequest,
        model_service: ModelService = Depends(
            get_model_service
        ),
    ):
        try:
            return model_service.predict_batch(
                customers=request.customers,
                classification_threshold=(
                    request.classification_threshold
                ),
            )
        except BatchSizeError as exc:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Batch exceeds the configured "
                    f"limit of {settings.max_batch_size} "
                    "customers."
                ),
            ) from exc
        except ModelNotReadyError as exc:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid batch prediction request.",
            ) from exc
        except ModelPredictionError as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to generate batch predictions.",
            ) from exc
    application.include_router(explanation_router)
    return application
# ============================================================
# APPLICATION INSTANCE
# ============================================================
app = create_app()
# ============================================================
# DIRECT EXECUTION
# ============================================================
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
