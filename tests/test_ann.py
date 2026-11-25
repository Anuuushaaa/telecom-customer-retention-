"""
tests/test_ann.py

Automated tests for TensorFlow ANN training,
evaluation, persistence and report generation.

Uses the original telecom customer churn dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensorflow import keras

from src.preprocessing import (
    prepare_raw_data,
    split_train_validation_test,
    fit_transform_features,
)

from src.train_ann import (
    build_ann,
    calculate_class_weights,
    prepare_ann_inputs,
    train_ann,
    predict_ann_proba,
    evaluate_ann,
    save_ann_model,
    save_training_history,
    plot_learning_curves,
    save_validation_metrics,
)


DATA_PATH = Path("data/raw/telco_churn.csv")


@pytest.fixture(scope="module")
def preprocessed_data():
    """Prepare the original dataset once."""

    df_raw = pd.read_csv(DATA_PATH)

    df_clean = prepare_raw_data(
        df_raw,
        require_target=True,
    )

    train_df, val_df, test_df = (
        split_train_validation_test(
            df_clean,
            random_state=42,
        )
    )

    return fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )


@pytest.fixture(scope="module")
def trained_ann(preprocessed_data):
    """Train a lightweight ANN once for automated testing."""

    data = preprocessed_data

    model, history = train_ann(
        X_train=data.X_train,
        y_train=data.y_train,
        X_val=data.X_val,
        y_val=data.y_val,
        epochs=3,
        batch_size=32,
        random_state=42,
        verbose=0,
    )

    return model, history


# --------------------------------------------------
# Architecture tests
# --------------------------------------------------


def test_ann_input_output_shape(preprocessed_data):

    input_dim = preprocessed_data.X_train.shape[1]

    model = build_ann(input_dim)

    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, 1)


def test_ann_dense_architecture(preprocessed_data):

    model = build_ann(
        preprocessed_data.X_train.shape[1]
    )

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.layers.Dense)
    ]

    assert len(dense_layers) == 3

    assert [
        layer.units for layer in dense_layers
    ] == [64, 32, 1]


def test_ann_activation_functions(preprocessed_data):

    model = build_ann(
        preprocessed_data.X_train.shape[1]
    )

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.layers.Dense)
    ]

    activations = [
        layer.activation.__name__
        for layer in dense_layers
    ]

    assert activations == [
        "relu",
        "relu",
        "sigmoid",
    ]


def test_ann_dropout_regularization(preprocessed_data):

    model = build_ann(
        preprocessed_data.X_train.shape[1]
    )

    dropout_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.layers.Dropout)
    ]

    assert len(dropout_layers) == 2

    assert dropout_layers[0].rate == pytest.approx(0.30)
    assert dropout_layers[1].rate == pytest.approx(0.20)


def test_ann_invalid_input_dimension():

    with pytest.raises(ValueError, match="input_dim"):
        build_ann(0)


# --------------------------------------------------
# Class weight tests
# --------------------------------------------------


def test_class_weights(preprocessed_data):

    y_train = preprocessed_data.y_train

    weights = calculate_class_weights(y_train)

    counts = y_train.value_counts()

    expected_zero = len(y_train) / (2 * counts[0])
    expected_one = len(y_train) / (2 * counts[1])

    assert weights[0] == pytest.approx(expected_zero)
    assert weights[1] == pytest.approx(expected_one)

    assert weights[1] > weights[0]


def test_class_weights_single_class(preprocessed_data):

    y_train = preprocessed_data.y_train

    single_class = y_train[y_train == 0]

    with pytest.raises(ValueError, match="both classes"):
        calculate_class_weights(single_class)


# --------------------------------------------------
# Input validation tests
# --------------------------------------------------


def test_prepare_ann_inputs_dtype(preprocessed_data):

    X, y = prepare_ann_inputs(
        preprocessed_data.X_train,
        preprocessed_data.y_train,
    )

    assert X.dtype == np.float32
    assert y.dtype == np.float32

    assert X.ndim == 2
    assert y.ndim == 1


def test_prepare_ann_inputs_finite(preprocessed_data):

    X, y = prepare_ann_inputs(
        preprocessed_data.X_train,
        preprocessed_data.y_train,
    )

    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_prepare_ann_inputs_invalid_features(
    preprocessed_data,
):

    X = preprocessed_data.X_train.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        prepare_ann_inputs(X)


def test_prepare_ann_inputs_invalid_labels(
    preprocessed_data,
):

    y = preprocessed_data.y_train.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y[0] = 2

    with pytest.raises(ValueError, match="0 and 1"):
        prepare_ann_inputs(
            preprocessed_data.X_train,
            y,
        )


def test_prepare_ann_inputs_mismatched_rows(
    preprocessed_data,
):

    y = preprocessed_data.y_train.iloc[:-1]

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        prepare_ann_inputs(
            preprocessed_data.X_train,
            y,
        )


# --------------------------------------------------
# Training tests
# --------------------------------------------------


def test_ann_training_history(trained_ann):

    _, history = trained_ann

    expected_metrics = {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_roc_auc",
        "val_pr_auc",
    }

    assert expected_metrics.issubset(
        history.history.keys()
    )

    assert len(history.history["loss"]) == 3


def test_ann_training_metrics_finite(trained_ann):

    _, history = trained_ann

    for values in history.history.values():

        assert np.isfinite(values).all()


# --------------------------------------------------
# Prediction tests
# --------------------------------------------------


def test_ann_prediction_shape(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    probabilities = predict_ann_proba(
        model,
        preprocessed_data.X_val,
    )

    assert probabilities.shape == (
        len(preprocessed_data.X_val),
    )


def test_ann_prediction_range(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    probabilities = predict_ann_proba(
        model,
        preprocessed_data.X_val,
    )

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_ann_single_customer_prediction(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    customer = preprocessed_data.X_val.iloc[[0]]

    probabilities = predict_ann_proba(
        model,
        customer,
    )

    assert probabilities.shape == (1,)
    assert 0 <= probabilities[0] <= 1


def test_ann_invalid_prediction_feature_count(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    invalid_features = (
        preprocessed_data.X_val.iloc[:, :-1]
    )

    with pytest.raises(
        ValueError,
        match="Feature count",
    ):
        predict_ann_proba(
            model,
            invalid_features,
        )


# --------------------------------------------------
# Evaluation tests
# --------------------------------------------------


def test_ann_evaluation_metrics(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    metrics = evaluate_ann(
        model,
        preprocessed_data.X_val,
        preprocessed_data.y_val,
        verbose=False,
    )

    expected_keys = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }

    assert set(metrics.keys()) == expected_keys

    assert metrics["model"] == "TensorFlow ANN"

    for key, value in metrics.items():

        if key != "model":
            assert np.isfinite(value)
            assert 0 <= value <= 1


def test_ann_invalid_threshold(
    trained_ann,
    preprocessed_data,
):

    model, _ = trained_ann

    with pytest.raises(ValueError, match="threshold"):

        evaluate_ann(
            model,
            preprocessed_data.X_val,
            preprocessed_data.y_val,
            threshold=1.5,
            verbose=False,
        )


# --------------------------------------------------
# Model persistence tests
# --------------------------------------------------


def test_ann_model_save_and_reload(
    trained_ann,
    preprocessed_data,
    tmp_path,
):

    model, _ = trained_ann

    X = preprocessed_data.X_val.iloc[:100]

    original_predictions = predict_ann_proba(
        model,
        X,
    )

    model_path = tmp_path / "test_ann.keras"

    saved_path = save_ann_model(
        model,
        model_path,
    )

    assert saved_path.exists()

    loaded_model = keras.models.load_model(
        saved_path
    )

    loaded_predictions = predict_ann_proba(
        loaded_model,
        X,
    )

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-5,
        atol=1e-6,
    )


# --------------------------------------------------
# Report generation tests
# --------------------------------------------------


def test_ann_training_history_export(
    trained_ann,
    tmp_path,
):

    _, history = trained_ann

    output_path = tmp_path / "history.csv"

    history_df = save_training_history(
        history,
        output_path,
    )

    assert output_path.exists()

    loaded_df = pd.read_csv(output_path)

    assert len(loaded_df) == len(history_df)

    assert loaded_df["epoch"].tolist() == [1, 2, 3]

    assert "loss" in loaded_df.columns
    assert "val_loss" in loaded_df.columns


def test_ann_learning_curve_generation(
    trained_ann,
    tmp_path,
):

    _, history = trained_ann

    output_path = tmp_path / "learning_curves.png"

    plot_learning_curves(
        history,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    with output_path.open("rb") as file:

        assert file.read(8) == b"\x89PNG\r\n\x1a\n"


def test_ann_validation_metrics_export(
    trained_ann,
    preprocessed_data,
    tmp_path,
):

    model, _ = trained_ann

    metrics = evaluate_ann(
        model,
        preprocessed_data.X_val,
        preprocessed_data.y_val,
        verbose=False,
    )

    output_path = tmp_path / "metrics.csv"

    save_validation_metrics(
        metrics,
        output_path,
    )

    assert output_path.exists()

    loaded_df = pd.read_csv(output_path)

    assert len(loaded_df) == 1

    assert loaded_df.loc[0, "model"] == "TensorFlow ANN"

    assert loaded_df.loc[0, "roc_auc"] == pytest.approx(
        metrics["roc_auc"]
    )