"""
src/config.py

Central place for decisions made during model selection, so predict.py
and app.py don't hardcode magic numbers separately.
"""

# Logistic Regression, trained on the full engineered feature set (Phase 6:
# num_services_subscribed, avg_monthly_spend, tenure_bucket,
# is_electronic_check added to the original columns). Chosen over XGBoost
# for interpretability, consistent with this project's explainability focus,
# despite XGBoost achieving a marginally better F1 at matched recall.
PRODUCTION_MODEL_PATH = "models/logistic_regression.pkl"

# Chosen via threshold sweep (src/threshold_tuning.py) on the retrained
# model: 0.35 gives recall 0.9064 / precision 0.4508 / F1 0.6021 —
# prioritizes catching churners (business cost of a missed churner >
# cost of an unnecessary retention offer) over the default 0.5 cutoff.
CLASSIFICATION_THRESHOLD = 0.35