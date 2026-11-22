
"""
src/config.py

Central configuration for production inference.

Model: Logistic Regression
Threshold: 0.50

The model and threshold were selected using validation data
and evaluated on the held-out test dataset.
"""

PRODUCTION_MODEL_PATH = "models/logistic_regression.pkl"

CLASSIFICATION_THRESHOLD = 0.50
