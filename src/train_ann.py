"""
src/train_ann.py

Configurable TensorFlow Artificial Neural Network
for telecom customer churn.

Features:
- Configurable neural network architecture
- Class-weighted binary classification
- Dropout and L2 regularization
- Early stopping and learning rate scheduling
- Optuna-compatible callbacks
- Reproducible initialization
- Validation performance evaluation
- Training history export
- Learning curve visualization
- Model persistence
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


RANDOM_STATE = 42

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

DEFAULT_HIDDEN_UNITS = (64, 32)
DEFAULT_DROPOUT_RATES = (0.30, 0.20)
DEFAULT_ACTIVATION = "relu"
DEFAULT_L2_STRENGTH = 0.0
DEFAULT_PATIENCE = 10

SUPPORTED_ACTIVATIONS = (
    "relu",
    "elu",
    "tanh",
)

CANDIDATE_DIR = Path("models/candidates")
REPORTS_DIR = Path("reports")


def validate_ann_hyperparameters(
    input_dim,
    learning_rate,
    hidden_units,
    dropout_rates,
    activation,
    l2_strength,
):
    """Validate ANN architecture and optimization parameters."""

    if (
        isinstance(input_dim, bool)
        or not isinstance(input_dim, (int, np.integer))
        or input_dim <= 0
    ):
        raise ValueError("input_dim must be a positive integer.")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")

    if not isinstance(hidden_units, (list, tuple)):
        raise ValueError("hidden_units must be a list or tuple.")

    if len(hidden_units) == 0:
        raise ValueError("At least one hidden layer is required.")

    for units in hidden_units:
        if (
            isinstance(units, bool)
            or not isinstance(units, (int, np.integer))
            or units <= 0
        ):
            raise ValueError(
                "Hidden layer units must be positive integers."
            )

    if not isinstance(dropout_rates, (list, tuple)):
        raise ValueError("dropout_rates must be a list or tuple.")

    if len(dropout_rates) != len(hidden_units):
        raise ValueError(
            "Each hidden layer must have a corresponding dropout rate."
        )

    for rate in dropout_rates:
        if not np.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError(
                "Dropout rates must be between 0 and 1, excluding 1."
            )

    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation: {activation}. "
            f"Choose from {SUPPORTED_ACTIVATIONS}."
        )

    if not np.isfinite(l2_strength) or l2_strength < 0:
        raise ValueError(
            "l2_strength must be non-negative and finite."
        )


def build_ann(
    input_dim: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    hidden_units=DEFAULT_HIDDEN_UNITS,
    dropout_rates=DEFAULT_DROPOUT_RATES,
    activation: str = DEFAULT_ACTIVATION,
    l2_strength: float = DEFAULT_L2_STRENGTH,
) -> keras.Model:
    """Build and compile a configurable binary classification ANN."""

    validate_ann_hyperparameters(
        input_dim=input_dim,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        activation=activation,
        l2_strength=l2_strength,
    )

    keras.utils.set_random_seed(random_state)

    kernel_initializer = (
        "glorot_uniform"
        if activation == "tanh"
        else "he_normal"
    )

    kernel_regularizer = (
        regularizers.l2(l2_strength)
        if l2_strength > 0
        else None
    )

    model = keras.Sequential(
        name="telecom_churn_ann"
    )

    model.add(
        keras.Input(shape=(input_dim,))
    )

    for units, dropout_rate in zip(
        hidden_units,
        dropout_rates,
    ):
        model.add(
            layers.Dense(
                units=int(units),
                activation=activation,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
            )
        )

        model.add(
            layers.Dropout(float(dropout_rate))
        )

    model.add(
        layers.Dense(
            1,
            activation="sigmoid",
        )
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.Precision(
                name="precision"
            ),
            keras.metrics.Recall(
                name="recall"
            ),
            keras.metrics.AUC(
                name="roc_auc",
                curve="ROC",
            ),
            keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )

    return model


def calculate_class_weights(y_train) -> dict:
    """Calculate balanced class weights using training labels only."""

    labels = np.asarray(y_train).reshape(-1)
    classes = np.unique(labels)

    if not np.array_equal(classes, [0, 1]):
        raise ValueError(
            "Training labels must contain both classes: 0 and 1."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def prepare_ann_inputs(X, y=None):
    """Convert features and labels to TensorFlow-compatible arrays."""

    X_array = np.asarray(X, dtype=np.float32)

    if X_array.ndim != 2:
        raise ValueError(
            "Features must be a two-dimensional array."
        )

    if X_array.shape[0] == 0 or X_array.shape[1] == 0:
        raise ValueError("Feature array cannot be empty.")

    if not np.isfinite(X_array).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    if y is None:
        return X_array

    y_array = np.asarray(
        y,
        dtype=np.float32,
    ).reshape(-1)

    if len(X_array) != len(y_array):
        raise ValueError(
            "Features and labels must have the same number of rows."
        )

    if not np.isin(y_array, [0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    return X_array, y_array


def train_ann(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    random_state: int = RANDOM_STATE,
    verbose: int = 1,
    hidden_units=DEFAULT_HIDDEN_UNITS,
    dropout_rates=DEFAULT_DROPOUT_RATES,
    activation: str = DEFAULT_ACTIVATION,
    l2_strength: float = DEFAULT_L2_STRENGTH,
    patience: int = DEFAULT_PATIENCE,
    additional_callbacks=None,
):
    """Train an ANN with configurable hyperparameters."""

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if (
        isinstance(patience, bool)
        or not isinstance(patience, (int, np.integer))
        or patience <= 0
    ):
        raise ValueError("patience must be a positive integer.")

    X_train_array, y_train_array = prepare_ann_inputs(
        X_train,
        y_train,
    )

    X_val_array, y_val_array = prepare_ann_inputs(
        X_val,
        y_val,
    )

    if X_train_array.shape[1] != X_val_array.shape[1]:
        raise ValueError(
            "Training and validation feature counts must match."
        )

    class_weights = calculate_class_weights(
        y_train_array
    )

    keras.backend.clear_session()
    keras.utils.set_random_seed(random_state)

    model = build_ann(
        input_dim=X_train_array.shape[1],
        learning_rate=learning_rate,
        random_state=random_state,
        hidden_units=hidden_units,
        dropout_rates=dropout_rates,
        activation=activation,
        l2_strength=l2_strength,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=0.0001,
        restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=max(1, patience // 2),
        min_lr=0.000001,
    )

    callbacks = [
        early_stopping,
        reduce_lr,
    ]

    if additional_callbacks is not None:
        callbacks.extend(additional_callbacks)

    if verbose:
        print("\nANN hyperparameters:")

        print(
            f"Hidden units: {hidden_units}"
        )

        print(
            f"Dropout rates: {dropout_rates}"
        )

        print(
            f"Activation: {activation}"
        )

        print(
            f"Learning rate: {learning_rate}"
        )

        print(
            f"L2 strength: {l2_strength}"
        )

        print(
            f"Batch size: {batch_size}"
        )

        print(
            f"Early stopping patience: {patience}"
        )

        print("\nANN class weights:")
        print(class_weights)

        model.summary()

    history = model.fit(
        X_train_array,
        y_train_array,
        validation_data=(
            X_val_array,
            y_val_array,
        ),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )

    return model, history


def predict_ann_proba(
    model: keras.Model,
    X,
) -> np.ndarray:
    """Generate positive-class churn scores."""

    X_array = prepare_ann_inputs(X)

    if X_array.shape[1] != model.input_shape[-1]:
        raise ValueError(
            "Feature count does not match the ANN input dimension."
        )

    probabilities = model.predict(
        X_array,
        verbose=0,
    ).reshape(-1)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "ANN generated non-finite probabilities."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "ANN probabilities must be between 0 and 1."
        )

    return probabilities


def evaluate_ann(
    model: keras.Model,
    X,
    y,
    threshold: float = 0.50,
    model_name: str = "TensorFlow ANN",
    verbose: bool = True,
) -> dict:
    """Evaluate ANN classification and ranking performance."""

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    X_array, y_array = prepare_ann_inputs(X, y)

    if not np.array_equal(
        np.unique(y_array),
        [0, 1],
    ):
        raise ValueError(
            "Evaluation labels must contain both classes."
        )

    y_proba = predict_ann_proba(
        model,
        X_array,
    )

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(
            y_array,
            y_pred,
        ),
        "precision": precision_score(
            y_array,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_array,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_array,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_array,
            y_proba,
        ),
        "average_precision": average_precision_score(
            y_array,
            y_proba,
        ),
    }

    if verbose:
        print(f"\n{'=' * 50}")

        print(
            f"{model_name} | Threshold: {threshold:.2f}"
        )

        print("=" * 50)

        for key, value in metrics.items():
            if key != "model":
                print(f"{key:>20}: {value:.4f}")

        print("\nClassification report:")

        print(
            classification_report(
                y_array,
                y_pred,
                labels=[0, 1],
                target_names=[
                    "No Churn",
                    "Churn",
                ],
                zero_division=0,
            )
        )

    return metrics


def save_training_history(
    history,
    output_path=REPORTS_DIR / "ann_training_history.csv",
) -> pd.DataFrame:
    """Save epoch-level training and validation metrics."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(1, len(history_df) + 1),
    )

    history_df.to_csv(
        output_path,
        index=False,
    )

    return history_df


def plot_learning_curves(
    history,
    output_path=REPORTS_DIR / "ann_learning_curves.png",
):
    """Generate training and validation learning curves."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(
        history.history
    )

    epochs = np.arange(
        1,
        len(history_df) + 1,
    )

    metrics = [
        ("loss", "Training Objective Loss"),
        ("accuracy", "Accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    for ax, (metric, title) in zip(
        axes.flatten(),
        metrics,
    ):
        ax.plot(
            epochs,
            history_df[metric],
            label="Training",
        )

        ax.plot(
            epochs,
            history_df[f"val_{metric}"],
            label="Validation",
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(
        "TensorFlow ANN - Learning Curves",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_ann_model(
    model: keras.Model,
    output_path=CANDIDATE_DIR / "tensorflow_ann.keras",
) -> Path:
    """Save the ANN candidate in native Keras format."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(output_path)

    return output_path


def save_validation_metrics(
    metrics: dict,
    output_path=REPORTS_DIR / "ann_validation_metrics.csv",
) -> pd.DataFrame:
    """Export ANN validation performance."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        output_path,
        index=False,
    )

    return metrics_df