
"""
Verify unified MLflow inference against existing
calibrated validation predictions.
"""

import json
import tempfile

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.mlflow_config import (
    DATA_PATH,
    SUPPORTED_MODELS,
)

from src.mlflow_tracking import (
    get_model_artifacts,
)

from src.mlflow_model import (
    TelecomChurnModel,
)

from src.threshold_tuning import (
    load_calibrated_predictions,
)


def verify_model(
    model_name,
    raw_validation,
    expected_predictions,
    temporary_directory,
):
    """Verify one optimized model wrapper."""

    artifacts = get_model_artifacts(
        model_name,
        "optimized",
    )

    configuration = {
        "model_name": model_name,
        "calibration_method": "sigmoid",
        "classification_threshold": 0.50,
    }

    configuration_path = (
        Path(temporary_directory)
        / f"{model_name}_configuration.json"
    )

    with configuration_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4,
        )

    context = SimpleNamespace(
        artifacts={
            "model": str(artifacts["model"]),
            "scaler": str(artifacts["scaler"]),
            "feature_columns": str(
                artifacts["feature_columns"]
            ),
            "calibrator": str(
                artifacts["sigmoid_calibrator"]
            ),
            "configuration": str(
                configuration_path
            ),
        }
    )

    wrapper = TelecomChurnModel()

    wrapper.load_context(context)

    results = wrapper.predict(
        context,
        raw_validation,
    )

    expected_raw = expected_predictions[
        f"{model_name}_uncalibrated_probability"
    ].to_numpy(dtype=float)

    expected_calibrated = expected_predictions[
        f"{model_name}_sigmoid_probability"
    ].to_numpy(dtype=float)

    np.testing.assert_array_equal(
        results["customerID"].to_numpy(),
        expected_predictions[
            "customerID"
        ].to_numpy(),
    )

    np.testing.assert_allclose(
        results["raw_churn_probability"],
        expected_raw,
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        results["churn_probability"],
        expected_calibrated,
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_array_equal(
        results["prediction"],
        (expected_calibrated >= 0.50).astype(int),
    )

    # Verify configurable classification threshold.

    sample = raw_validation.head(10)

    low_threshold = wrapper.predict(
        context,
        sample,
        params={
            "classification_threshold": 0.0
        },
    )

    high_threshold = wrapper.predict(
        context,
        sample,
        params={
            "classification_threshold": 1.0
        },
    )

    assert low_threshold["prediction"].eq(1).all()

    assert (
        high_threshold["prediction"].to_numpy()
        == (
            high_threshold["churn_probability"]
            .to_numpy() >= 1.0
        ).astype(int)
    ).all()

    maximum_error = np.max(
        np.abs(
            results["churn_probability"].to_numpy()
            - expected_calibrated
        )
    )

    print(f"\nModel: {model_name}")

    print(f"Rows verified: {len(results)}")

    print(
        f"Maximum probability difference: "
        f"{maximum_error:.10f}"
    )

    print("Raw probabilities: PASSED")

    print("Calibrated probabilities: PASSED")

    print("Classification consistency: PASSED")

    print("Threshold override: PASSED")

    return results


def main():

    print("=" * 70)

    print("UNIFIED MLFLOW MODEL VERIFICATION")

    print("=" * 70)

    raw = pd.read_csv(DATA_PATH)

    expected_predictions, _ = (
        load_calibrated_predictions()
    )

    if raw["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer identifiers."
        )

    raw_validation = (
        raw.set_index("customerID")
        .loc[
            expected_predictions[
                "customerID"
            ].tolist()
        ]
        .reset_index()
        .drop(columns=["Churn"])
    )

    with tempfile.TemporaryDirectory() as directory:

        for model_name in SUPPORTED_MODELS:

            verify_model(
                model_name,
                raw_validation,
                expected_predictions,
                directory,
            )

    print("\n" + "=" * 70)

    print("ALL MODEL VERIFICATIONS PASSED")

    print("=" * 70)

    print("Models verified: 3")

    print(
        f"Total customer predictions verified: "
        f"{len(raw_validation) * len(SUPPORTED_MODELS)}"
    )

    print("No models retrained.")

    print("Production models remain unchanged.")


if __name__ == "__main__":

    main()
