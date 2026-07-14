"""
src/config.py

Central place for decisions made during Phase 5 model selection, so
predict.py and app.py don't hardcode magic numbers separately.
"""

# Chosen in Phase 5: Logistic Regression beats XGBoost on recall/F1/ROC-AUC
# at comparable operating points, and is simpler to explain — a deliberate
# choice given this project's emphasis on interpretability.
PRODUCTION_MODEL_PATH = "models/logistic_regression.pkl"

# Chosen via threshold sweep (src/threshold_tuning.py): 0.35 gives recall
# 0.9037 / precision 0.4471 / F1 0.5982 — prioritizes catching churners
# (business cost of a missed churner > cost of an unnecessary retention offer)
# over the default, arbitrary 0.5 cutoff.
CLASSIFICATION_THRESHOLD = 0.35