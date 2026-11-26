"""Configuration for the Streamlit frontend."""

import math
import os
from dataclasses import dataclass


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_HEALTH_TIMEOUT = 5.0


@dataclass(frozen=True)
class FrontendSettings:

    api_url: str
    request_timeout: float
    health_timeout: float


def get_timeout(name, default):

    value = float(os.getenv(name, str(default)))

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be finite and positive."
        )

    return value


def get_frontend_settings():

    api_url = os.getenv(
        "TELECOM_API_URL",
        DEFAULT_API_URL,
    ).strip().rstrip("/")

    if not api_url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "TELECOM_API_URL must be an HTTP or HTTPS URL."
        )

    return FrontendSettings(
        api_url=api_url,
        request_timeout=get_timeout(
            "TELECOM_API_TIMEOUT",
            DEFAULT_REQUEST_TIMEOUT,
        ),
        health_timeout=get_timeout(
            "TELECOM_HEALTH_TIMEOUT",
            DEFAULT_HEALTH_TIMEOUT,
        ),
    )