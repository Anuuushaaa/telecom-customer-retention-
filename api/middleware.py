
"""
Request tracking and structured HTTP logging
for the Telecom Customer Retention API.
"""

import json
import logging

from datetime import datetime, timezone

from time import perf_counter

from uuid import uuid4

from fastapi import FastAPI, Request

from api.exceptions import (
    unexpected_exception_handler,
)


logger = logging.getLogger("api.requests")


# ============================================================
# STRUCTURED REQUEST LOGGING
# ============================================================

def log_request(
    request: Request,
    status_code: int,
    duration_ms: float,
) -> None:

    route = request.scope.get("route")

    route_path = getattr(
        route,
        "path",
        "unmatched",
    )

    event = {

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "event": "http_request",

        "request_id": getattr(
            request.state,
            "request_id",
            None,
        ),

        "method": request.method,

        "path": route_path,

        "status_code": status_code,

        "duration_ms": round(
            duration_ms,
            3,
        ),
    }

    message = json.dumps(event)

    if status_code >= 500:

        logger.error(message)

    elif status_code >= 400:

        logger.warning(message)

    else:

        logger.info(message)


# ============================================================
# REQUEST TRACKING MIDDLEWARE
# ============================================================

def register_middleware(
    application: FastAPI,
) -> None:

    @application.middleware("http")
    async def track_request(
        request: Request,
        call_next,
    ):

        request_id = str(uuid4())

        request.state.request_id = request_id

        start_time = perf_counter()

        try:

            response = await call_next(
                request
            )

        except Exception as exc:

            response = (
                await unexpected_exception_handler(
                    request,
                    exc,
                )
            )

        duration_ms = (

            perf_counter() - start_time

        ) * 1000

        response.headers["X-Request-ID"] = (
            request_id
        )

        response.headers["X-Response-Time-MS"] = (
            f"{duration_ms:.3f}"
        )

        log_request(

            request=request,

            status_code=response.status_code,

            duration_ms=duration_ms,
        )

        return response
