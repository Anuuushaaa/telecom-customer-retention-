
"""
Centralized exception handling for the Telecom
Customer Retention API.
"""

import logging

from fastapi import FastAPI, Request

from fastapi.exceptions import RequestValidationError

from fastapi.responses import JSONResponse

from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from api.schemas import ErrorResponse


logger = logging.getLogger(__name__)


# ============================================================
# HTTP ERROR MAPPING
# ============================================================

ERROR_CODES = {

    400: "BAD_REQUEST",

    401: "UNAUTHORIZED",

    403: "FORBIDDEN",

    404: "NOT_FOUND",

    405: "METHOD_NOT_ALLOWED",

    413: "BATCH_LIMIT_EXCEEDED",

    422: "VALIDATION_ERROR",

    429: "TOO_MANY_REQUESTS",

    500: "INTERNAL_SERVER_ERROR",

    503: "MODEL_UNAVAILABLE",
}


# ============================================================
# ERROR RESPONSE FACTORY
# ============================================================

def create_error_response(
    request: Request,
    status_code: int,
    error: str,
    message: str,
    headers: dict | None = None,
) -> JSONResponse:

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    payload = ErrorResponse(

        error=error,

        message=message,

        request_id=request_id,
    )

    response_headers = dict(
        headers or {}
    )

    if request_id:

        response_headers["X-Request-ID"] = (
            request_id
        )

    return JSONResponse(

        status_code=status_code,

        content=payload.model_dump(
            mode="json"
        ),

        headers=response_headers,
    )


# ============================================================
# HTTP EXCEPTION HANDLER
# ============================================================

async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:

    error_code = ERROR_CODES.get(

        exc.status_code,

        "HTTP_ERROR",
    )

    message = (

        exc.detail

        if isinstance(exc.detail, str)

        else "The request could not be completed."
    )

    if exc.status_code >= 500:

        logger.error(
            "HTTP service error: "
            "request_id=%s status=%s",
            getattr(
                request.state,
                "request_id",
                None,
            ),
            exc.status_code,
        )

    return create_error_response(

        request=request,

        status_code=exc.status_code,

        error=error_code,

        message=message,

        headers=exc.headers,
    )


# ============================================================
# REQUEST VALIDATION HANDLER
# ============================================================

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:

    validation_errors = []

    for error in exc.errors():

        location = ".".join(

            str(part)

            for part in error.get("loc", [])
        )

        error_type = error.get(
            "type",
            "invalid_value",
        )

        # Do not expose raw input values or
        # customer records in error responses.

        validation_errors.append(
            f"{location}: {error_type}"
        )

    message = "; ".join(
        validation_errors[:20]
    )

    if len(validation_errors) > 20:

        message += (
            "; additional validation errors omitted"
        )

    logger.warning(
        "Request validation failed: "
        "request_id=%s error_count=%s",
        getattr(
            request.state,
            "request_id",
            None,
        ),
        len(validation_errors),
    )

    return create_error_response(

        request=request,

        status_code=422,

        error="VALIDATION_ERROR",

        message=message,
    )


# ============================================================
# UNEXPECTED EXCEPTION HANDLER
# ============================================================

async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:

    logger.error(

        "Unexpected application error: "
        "request_id=%s exception_type=%s",

        getattr(
            request.state,
            "request_id",
            None,
        ),

        type(exc).__name__,
    )

    return create_error_response(

        request=request,

        status_code=500,

        error="INTERNAL_SERVER_ERROR",

        message=(
            "An unexpected server error occurred."
        ),
    )


# ============================================================
# REGISTER EXCEPTION HANDLERS
# ============================================================

def register_exception_handlers(
    application: FastAPI,
) -> None:

    application.add_exception_handler(

        StarletteHTTPException,

        http_exception_handler,
    )

    application.add_exception_handler(

        RequestValidationError,

        validation_exception_handler,
    )

    application.add_exception_handler(

        Exception,

        unexpected_exception_handler,
    )
