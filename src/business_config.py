"""
src/business_config.py

Configurable financial assumptions for telecom
customer retention campaign simulations.

All financial parameters are hypothetical
simulation assumptions, not observed company costs.
"""

from dataclasses import dataclass, asdict
from numbers import Real, Integral
from pathlib import Path
import math


RANDOM_STATE = 42

DATA_PATH = Path("data/raw/telco_churn.csv")

REPORTS_DIR = Path("reports/business")

CALIBRATION_REPORTS_DIR = Path(
    "reports/calibration"
)

THRESHOLD_REPORTS_DIR = Path(
    "reports/thresholds"
)

CALIBRATED_PREDICTIONS_PATH = Path(
    "models/candidates/calibration/"
    "calibrated_validation_predictions.csv"
)

SELECTED_METHODS_PATH = (
    CALIBRATION_REPORTS_DIR
    / "selected_calibration_methods.csv"
)

SELECTED_THRESHOLDS_PATH = (
    THRESHOLD_REPORTS_DIR
    / "preferred_calibration_thresholds.csv"
)


def validate_numeric_parameter(
    name,
    value,
    minimum=None,
    maximum=None,
):
    """Validate a finite numerical parameter."""

    if isinstance(value, bool) or not isinstance(
        value, Real
    ):
        raise ValueError(
            f"{name} must be numeric."
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must not exceed {maximum}."
        )


@dataclass(frozen=True)
class BusinessConfig:
    """
    Financial assumptions for a retention campaign.

    Contact and incentive costs are incurred for
    every targeted customer.

    Retention success probability represents the
    probability of successfully retaining a customer
    who would otherwise churn.
    """

    contact_cost: float = 5.0

    incentive_cost: float = 20.0

    retention_success_rate: float = 0.30

    retention_months: int = 12

    contribution_margin: float = 0.40

    campaign_budget: float = 10000.0

    currency: str = "USD"

    def __post_init__(self):

        validate_numeric_parameter(
            "contact_cost",
            self.contact_cost,
            minimum=0,
        )

        validate_numeric_parameter(
            "incentive_cost",
            self.incentive_cost,
            minimum=0,
        )

        validate_numeric_parameter(
            "retention_success_rate",
            self.retention_success_rate,
            minimum=0,
            maximum=1,
        )

        validate_numeric_parameter(
            "contribution_margin",
            self.contribution_margin,
            minimum=0,
            maximum=1,
        )

        validate_numeric_parameter(
            "campaign_budget",
            self.campaign_budget,
            minimum=0,
        )

        if (
            isinstance(self.retention_months, bool)
            or not isinstance(
                self.retention_months,
                Integral,
            )
            or self.retention_months <= 0
        ):
            raise ValueError(
                "retention_months must be "
                "a positive integer."
            )

        if self.intervention_cost <= 0:
            raise ValueError(
                "Total intervention cost "
                "must be positive."
            )

        if self.currency != "USD":
            raise ValueError(
                "This simulation uses USD."
            )

    @property
    def intervention_cost(self):
        """Total cost per targeted customer."""

        return float(
            self.contact_cost
            + self.incentive_cost
        )

    @property
    def maximum_budget_targets(self):
        """Maximum affordable campaign size."""

        return int(
            self.campaign_budget
            // self.intervention_cost
        )

    @property
    def retained_value_multiplier(self):
        """
        Convert monthly charges into potential
        retained contribution.
        """

        return float(
            self.retention_months
            * self.contribution_margin
        )

    def to_dict(self):
        """Return serializable financial assumptions."""

        config = asdict(self)

        config.update({
            "intervention_cost":
                self.intervention_cost,

            "maximum_budget_targets":
                self.maximum_budget_targets,

            "retained_value_multiplier":
                self.retained_value_multiplier,

            "assumption_type":
                "hypothetical_simulation",
        })

        return config


BASELINE_CONFIG = BusinessConfig()


# --------------------------------------------------
# Sensitivity analysis configurations
# --------------------------------------------------

CONTACT_COST_SCENARIOS = (
    2.0,
    5.0,
    10.0,
)

INCENTIVE_COST_SCENARIOS = (
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
)

RETENTION_SUCCESS_SCENARIOS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
)

RETENTION_MONTH_SCENARIOS = (
    3,
    6,
    12,
    18,
    24,
)

CONTRIBUTION_MARGIN_SCENARIOS = (
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
)

CAMPAIGN_BUDGET_SCENARIOS = (
    2500.0,
    5000.0,
    10000.0,
    15000.0,
    20000.0,
)


def main():
    """Display baseline simulation assumptions."""

    print("\nBUSINESS SIMULATION CONFIGURATION")
    print("=" * 55)

    config = BASELINE_CONFIG.to_dict()

    for name, value in config.items():

        print(f"{name}: {value}")

    print("\nSensitivity analysis configurations")

    print(
        "Contact costs:",
        CONTACT_COST_SCENARIOS,
    )

    print(
        "Incentive costs:",
        INCENTIVE_COST_SCENARIOS,
    )

    print(
        "Retention success rates:",
        RETENTION_SUCCESS_SCENARIOS,
    )

    print(
        "Retention durations:",
        RETENTION_MONTH_SCENARIOS,
    )

    print(
        "Contribution margins:",
        CONTRIBUTION_MARGIN_SCENARIOS,
    )

    print(
        "Campaign budgets:",
        CAMPAIGN_BUDGET_SCENARIOS,
    )

    print(
        "\nAll financial parameters are "
        "hypothetical simulation assumptions."
    )

    print(
        "Production models remain unchanged."
    )


if __name__ == "__main__":
    main()