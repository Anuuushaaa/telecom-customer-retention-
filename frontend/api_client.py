
"""HTTP client for the Telecom Churn API."""

import httpx

from frontend.config import get_frontend_settings


class APIClientError(RuntimeError):

    def __init__(
        self,
        message,
        status_code=None,
        error_code=None,
        request_id=None,
    ):

        super().__init__(message)

        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id


class APIConnectionError(APIClientError):
    pass


class APITimeoutError(APIClientError):
    pass


class TelecomAPIClient:

    def __init__(self, settings=None):

        self.settings = (
            settings or get_frontend_settings()
        )

    def request(
        self,
        method,
        endpoint,
        payload=None,
        timeout=None,
        allowed_status_codes=(200,),
    ):

        url = (
            f"{self.settings.api_url}/"
            f"{endpoint.lstrip('/')}"
        )

        if timeout is None:
            timeout = self.settings.request_timeout

        try:

            response = httpx.request(
                method=method,
                url=url,
                json=payload,
                timeout=timeout,
                headers={
                    "Accept": "application/json"
                },
            )

        except httpx.TimeoutException as error:

            raise APITimeoutError(
                "The API request timed out."
            ) from error

        except httpx.RequestError as error:

            raise APIConnectionError(
                "Unable to connect to the prediction API."
            ) from error

        request_id = response.headers.get(
            "X-Request-ID"
        )

        try:

            result = response.json()

        except ValueError as error:

            raise APIClientError(
                "The API returned an invalid JSON response.",
                status_code=response.status_code,
                request_id=request_id,
            ) from error

        if not isinstance(result, dict):

            raise APIClientError(
                "The API returned an unexpected response.",
                status_code=response.status_code,
                request_id=request_id,
            )

        if response.status_code not in allowed_status_codes:

            raise APIClientError(
                message=result.get(
                    "message",
                    "The API request failed.",
                ),
                status_code=response.status_code,
                error_code=result.get("error"),
                request_id=(
                    result.get("request_id")
                    or request_id
                ),
            )

        return result

    # ========================================================
    # HEALTH AND MODEL METADATA
    # ========================================================

    def get_health(self):

        return self.request(
            method="GET",
            endpoint="/health",
            timeout=self.settings.health_timeout,
        )

    def get_readiness(self):

        return self.request(
            method="GET",
            endpoint="/ready",
            timeout=self.settings.health_timeout,
            allowed_status_codes=(200, 503),
        )

    def get_model_info(self):

        return self.request(
            method="GET",
            endpoint="/model/info",
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        customer,
        classification_threshold=None,
    ):

        payload = {
            "customer": customer
        }

        if classification_threshold is not None:

            payload["classification_threshold"] = (
                classification_threshold
            )

        return self.request(
            method="POST",
            endpoint="/predict",
            payload=payload,
        )

    def predict_batch(
        self,
        customers,
        classification_threshold=None,
    ):

        payload = {
            "customers": customers
        }

        if classification_threshold is not None:

            payload["classification_threshold"] = (
                classification_threshold
            )

        return self.request(
            method="POST",
            endpoint="/predict/batch",
            payload=payload,
        )

    # ========================================================
    # LOCAL SHAP EXPLANATIONS
    # ========================================================

    def explain(
        self,
        customer,
        classification_threshold=None,
        top_n=10,
    ):

        payload = {
            "customer": customer,
            "top_n": top_n,
        }

        if classification_threshold is not None:

            payload["classification_threshold"] = (
                classification_threshold
            )

        return self.request(
            method="POST",
            endpoint="/explain",
            payload=payload,
        )

    # ========================================================
    # GLOBAL SHAP EXPLANATIONS
    # ========================================================

    def explain_global(self, customers):

        return self.request(
            method="POST",
            endpoint="/explain/global",
            payload={
                "customers": customers
            },
        )
