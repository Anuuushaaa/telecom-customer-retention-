
"""Live verification of registered model explanations."""

import json

import numpy as np
import pandas as pd

from api.model_service import ModelService
from api.schemas import CustomerInput
from api.explanation_service import ExplanationService


DATA_PATH = "data/raw/telco_churn.csv"


def main():

    dataframe = pd.read_csv(DATA_PATH)

    records = dataframe.drop(
        columns=["Churn"]
    ).head(100).to_dict(
        orient="records"
    )

    customers = [
        CustomerInput.model_validate(record)
        for record in records
    ]

    service = ModelService()

    try:

        model_info = service.load_model()

        print("\nREGISTERED MODEL")
        print(model_info.model_dump())

        explainer = ExplanationService(
            service
        )

        print("\nLOCAL SHAP EXPLANATION")

        result = explainer.explain_customer(
            customers[0],
            top_n=10,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        reconstructed_margin = (
            result["base_value"]
            + sum(
                item["shap_value"]
                for item in result["top_contributors"]
            )
            + result["remaining_contribution"]
        )

        assert np.isclose(
            reconstructed_margin,
            result["raw_margin"],
            atol=1e-4,
            rtol=1e-5,
        )

        assert np.isclose(
            result["reconstructed_raw_probability"],
            result["prediction"]["raw_churn_probability"],
            atol=1e-5,
            rtol=1e-5,
        )

        assert result["model_version"] == (
            model_info.model_version
        )

        print("\nGLOBAL SHAP EXPLANATION")

        global_result = explainer.explain_global(
            customers
        )

        print(
            "Customers explained:",
            global_result["total_customers"],
        )

        print(
            "Features explained:",
            global_result["total_features"],
        )

        print(
            "Maximum additivity error:",
            global_result["max_additivity_error"],
        )

        print("\nTop 10 global features:")

        print(
            pd.DataFrame(
                global_result["feature_importance"]
            ).head(10).to_string(
                index=False
            )
        )

        assert global_result["total_customers"] == 100

        assert global_result["total_features"] == (
            result["total_features"]
        )

        print("\nALL SHAP VERIFICATION CHECKS PASSED")

    finally:

        service.unload_model()


if __name__ == "__main__":

    main()
