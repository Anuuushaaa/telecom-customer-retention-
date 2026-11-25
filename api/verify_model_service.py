
"""
Live verification of the MLflow model-serving
service using the original customer dataset.
"""

import logging

from pathlib import Path

import numpy as np
import pandas as pd

from api.model_service import ModelService

from api.schemas import CustomerInput

from src.threshold_tuning import (
    load_calibrated_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_churn.csv"
)


def main():

    logging.basicConfig(

        level=logging.INFO,

        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    print("=" * 70)

    print("MLFLOW MODEL SERVICE VERIFICATION")

    print("=" * 70)

    service = ModelService()

    try:

        # ----------------------------------------------------
        # 1. Readiness before initialization
        # ----------------------------------------------------

        assert not service.is_ready

        print("\nInitial readiness:")

        print(
            service.get_readiness()
            .model_dump_json(indent=2)
        )

        # ----------------------------------------------------
        # 2. Load registered model
        # ----------------------------------------------------

        model_info = service.load_model()

        assert service.is_ready

        print("\nLoaded model information:")

        print(
            model_info.model_dump_json(indent=2)
        )

        # ----------------------------------------------------
        # 3. Verify cached model reuse
        # ----------------------------------------------------

        original_model = service._model

        service.load_model()

        assert service._model is original_model

        print("\nModel caching verified.")

        # ----------------------------------------------------
        # 4. Load original validation customers
        # ----------------------------------------------------

        raw_data = pd.read_csv(DATA_PATH)

        expected, _ = (
            load_calibrated_predictions()
        )

        validation = (

            raw_data.set_index("customerID")

            .loc[expected["customerID"].tolist()]

            .reset_index()

            .drop(columns=["Churn"])
        )

        records = validation.to_dict(
            orient="records"
        )

        customers = [

            CustomerInput.model_validate(record)

            for record in records
        ]

        print(
            f"\nValidation customers: "
            f"{len(customers)}"
        )

        # ----------------------------------------------------
        # 5. Single prediction
        # ----------------------------------------------------

        single = service.predict(
            customers[0]
        )

        print("\nSingle customer prediction:")

        print(
            single.model_dump_json(indent=2)
        )

        # ----------------------------------------------------
        # 6. Batch prediction
        # ----------------------------------------------------

        batch = service.predict_batch(
            customers[:100]
        )

        assert batch.total_customers == 100

        assert (

            batch.predicted_churn
            + batch.predicted_no_churn

            == batch.total_customers
        )

        print("\nBatch prediction summary:")

        print(
            f"Total customers: "
            f"{batch.total_customers}"
        )

        print(
            f"Predicted churn: "
            f"{batch.predicted_churn}"
        )

        print(
            f"Predicted no churn: "
            f"{batch.predicted_no_churn}"
        )

        # ----------------------------------------------------
        # 7. Verify all validation probabilities
        # ----------------------------------------------------

        batch_size = min(
            500,
            service.settings.max_batch_size,
        )

        all_predictions = []

        for start in range(
            0,
            len(customers),
            batch_size,
        ):

            response = service.predict_batch(
                customers[
                    start:start + batch_size
                ]
            )

            all_predictions.extend(
                response.predictions
            )

        actual_raw = np.array([

            prediction.raw_churn_probability

            for prediction in all_predictions
        ])

        actual_calibrated = np.array([

            prediction.churn_probability

            for prediction in all_predictions
        ])

        actual_ids = [

            prediction.customerID

            for prediction in all_predictions
        ]

        model_name = model_info.model_name

        calibration = (
            model_info.calibration_method
        )

        expected_raw = expected[
            f"{model_name}_uncalibrated_probability"
        ].to_numpy()

        expected_calibrated = expected[
            f"{model_name}_{calibration}_probability"
        ].to_numpy()

        assert actual_ids == (
            expected["customerID"].tolist()
        )

        np.testing.assert_allclose(

            actual_raw,

            expected_raw,

            atol=1e-5,

            rtol=1e-5,
        )

        np.testing.assert_allclose(

            actual_calibrated,

            expected_calibrated,

            atol=1e-5,

            rtol=1e-5,
        )

        maximum_raw_error = np.max(

            np.abs(
                actual_raw - expected_raw
            )
        )

        maximum_calibrated_error = np.max(

            np.abs(
                actual_calibrated
                - expected_calibrated
            )
        )

        print("\nFull validation verification:")

        print(
            f"Verified customers: "
            f"{len(all_predictions)}"
        )

        print(
            f"Maximum raw probability error: "
            f"{maximum_raw_error:.10e}"
        )

        print(
            f"Maximum calibrated probability error: "
            f"{maximum_calibrated_error:.10e}"
        )

        # ----------------------------------------------------
        # 8. Verify threshold overrides
        # ----------------------------------------------------

        zero_threshold = service.predict_batch(

            customers[:10],

            classification_threshold=0.0,
        )

        assert all(

            prediction.prediction == 1

            for prediction in
            zero_threshold.predictions
        )

        one_threshold = service.predict_batch(

            customers[:10],

            classification_threshold=1.0,
        )

        assert all(

            prediction.prediction
            == int(
                prediction.churn_probability >= 1.0
            )

            for prediction in
            one_threshold.predictions
        )

        print(
            "\nThreshold overrides verified."
        )

        print("\n" + "=" * 70)

        print("ALL MODEL SERVICE CHECKS PASSED")

        print("No model retraining performed.")

        print("No registry aliases modified.")

        print("=" * 70)

    finally:

        service.unload_model()

        assert not service.is_ready


if __name__ == "__main__":

    main()
