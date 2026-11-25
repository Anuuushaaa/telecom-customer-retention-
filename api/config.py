
"""
Centralized configuration for the Telecom Customer
Retention prediction API.
"""

import os

from dataclasses import dataclass

from src.mlflow_config import (
    REGISTERED_MODEL_NAME,
    PRODUCTION_ALIAS,
    get_tracking_uri,
    get_registry_uri,
)


# ============================================================
# API METADATA
# ============================================================

API_TITLE = "Telecom Customer Retention API"

API_VERSION = "2.0.0"

API_DESCRIPTION = (
    "REST API for calibrated telecom customer churn "
    "prediction using the MLflow model registry."
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8000

DEFAULT_THRESHOLD = 0.50

DEFAULT_MAX_BATCH_SIZE = 1000

DEFAULT_LOG_LEVEL = "INFO"

SUPPORTED_LOG_LEVELS = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


# ============================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================

def get_integer_setting(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:

    raw_value = os.getenv(name, str(default))

    try:

        value = int(raw_value)

    except ValueError as exc:

        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if not minimum <= value <= maximum:

        raise ValueError(
            f"{name} must be between "
            f"{minimum} and {maximum}."
        )

    return value


def get_threshold_setting() -> float:

    raw_value = os.getenv(
        "API_CLASSIFICATION_THRESHOLD",
        str(DEFAULT_THRESHOLD),
    )

    try:

        threshold = float(raw_value)

    except ValueError as exc:

        raise ValueError(
            "Classification threshold must be numeric."
        ) from exc

    if not 0.0 <= threshold <= 1.0:

        raise ValueError(
            "Classification threshold must be "
            "between 0 and 1."
        )

    return threshold


def get_log_level() -> str:

    level = os.getenv(
        "API_LOG_LEVEL",
        DEFAULT_LOG_LEVEL,
    ).strip().upper()

    if level not in SUPPORTED_LOG_LEVELS:

        raise ValueError(
            f"Unsupported log level: {level}"
        )

    return level


def get_nonempty_setting(
    name: str,
    default: str,
) -> str:

    value = os.getenv(
        name,
        default,
    ).strip()

    if not value:

        raise ValueError(
            f"{name} cannot be empty."
        )

    return value


# ============================================================
# API SETTINGS
# ============================================================

@dataclass(frozen=True)
class APISettings:

    title: str

    version: str

    description: str

    host: str

    port: int

    tracking_uri: str

    registry_uri: str

    registered_model_name: str

    model_alias: str

    classification_threshold: float

    max_batch_size: int

    log_level: str

    @property
    def model_uri(self) -> str:

        return (
            f"models:/{self.registered_model_name}"
            f"@{self.model_alias}"
        )


# ============================================================
# SETTINGS FACTORY
# ============================================================

def get_settings() -> APISettings:

    return APISettings(

        title=API_TITLE,

        version=API_VERSION,

        description=API_DESCRIPTION,

        host=get_nonempty_setting(
            "API_HOST",
            DEFAULT_HOST,
        ),

        port=get_integer_setting(
            name="API_PORT",
            default=DEFAULT_PORT,
            minimum=1,
            maximum=65535,
        ),

        tracking_uri=get_tracking_uri(),

        registry_uri=get_registry_uri(),

        registered_model_name=get_nonempty_setting(
            "API_REGISTERED_MODEL_NAME",
            REGISTERED_MODEL_NAME,
        ),

        model_alias=get_nonempty_setting(
            "API_MODEL_ALIAS",
            PRODUCTION_ALIAS,
        ),

        classification_threshold=(
            get_threshold_setting()
        ),

        max_batch_size=get_integer_setting(
            name="API_MAX_BATCH_SIZE",
            default=DEFAULT_MAX_BATCH_SIZE,
            minimum=1,
            maximum=10000,
        ),

        log_level=get_log_level(),
    )


# ============================================================
# CONFIGURATION VERIFICATION
# ============================================================

def print_configuration() -> None:

    settings = get_settings()

    print("=" * 65)

    print("TELECOM CUSTOMER RETENTION API")

    print("=" * 65)

    print(f"API title: {settings.title}")

    print(f"API version: {settings.version}")

    print(f"Host: {settings.host}")

    print(f"Port: {settings.port}")

    print(
        f"Tracking URI: {settings.tracking_uri}"
    )

    print(
        f"Registry URI: {settings.registry_uri}"
    )

    print(
        f"Registered model: "
        f"{settings.registered_model_name}"
    )

    print(f"Model alias: {settings.model_alias}")

    print(f"Model URI: {settings.model_uri}")

    print(
        f"Classification threshold: "
        f"{settings.classification_threshold}"
    )

    print(
        f"Maximum batch size: "
        f"{settings.max_batch_size}"
    )

    print(f"Log level: {settings.log_level}")

    print("=" * 65)


if __name__ == "__main__":

    print_configuration()
