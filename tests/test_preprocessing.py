
"""Preprocessing and data leakage tests using the original dataset."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_features

from src.preprocessing import (
    CONTRACT_ORDER,
    NUMERIC_COLS,
    prepare_raw_data,
    split_data,
    split_train_validation_test,
    fit_transform_features,
    transform_features,
)


@pytest.fixture(scope="module")
def raw_data():
    return pd.read_csv("data/raw/telco_churn.csv")


@pytest.fixture(scope="module")
def pipeline(raw_data):
    cleaned = prepare_raw_data(raw_data)

    train_df, val_df, test_df = (
        split_train_validation_test(cleaned)
    )

    data = fit_transform_features(
        train_df,
        test_df,
        val_df=val_df,
    )

    return train_df, val_df, test_df, data


def test_all_customers_preserved(raw_data, pipeline):
    train_df, val_df, test_df, _ = pipeline

    assert (
        len(train_df) + len(val_df) + len(test_df)
        == len(raw_data)
    )


def test_split_sizes(raw_data, pipeline):
    train_df, val_df, test_df, _ = pipeline

    total = len(raw_data)

    assert abs(len(train_df) / total - 0.60) < 0.001
    assert abs(len(val_df) / total - 0.20) < 0.001
    assert abs(len(test_df) / total - 0.20) < 0.001


def test_no_customer_overlap(pipeline):
    train_df, val_df, test_df, _ = pipeline

    train_ids = set(train_df["customerID"])
    val_ids = set(val_df["customerID"])
    test_ids = set(test_df["customerID"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_preserves_churn_rate(raw_data, pipeline):
    train_df, val_df, test_df, _ = pipeline

    original_rate = raw_data["Churn"].eq("Yes").mean()

    for partition in [train_df, val_df, test_df]:
        rate = partition["Churn"].mean()
        assert abs(rate - original_rate) < 0.01


def test_churn_mapped_to_binary(pipeline):
    train_df, val_df, test_df, _ = pipeline

    for partition in [train_df, val_df, test_df]:
        assert set(partition["Churn"].unique()) == {0, 1}


def test_contract_encoded_ordinally(pipeline):
    _, _, _, data = pipeline

    assert set(data.X_train["Contract"].unique()) == (
        set(CONTRACT_ORDER.values())
    )


def test_scaler_fitted_on_training_only(pipeline):
    train_df, _, _, data = pipeline

    expected_means = train_df[NUMERIC_COLS].mean().to_numpy()

    np.testing.assert_allclose(
        data.scaler.mean_,
        expected_means,
        rtol=1e-10,
        atol=1e-10,
    )

    expected_variances = (
        train_df[NUMERIC_COLS]
        .var(ddof=0)
        .to_numpy()
    )

    np.testing.assert_allclose(
        data.scaler.var_,
        expected_variances,
        rtol=1e-10,
        atol=1e-10,
    )


def test_scaler_sample_count(pipeline):
    train_df, _, _, data = pipeline

    assert data.scaler.n_samples_seen_ == len(train_df)


def test_scaled_training_means(pipeline):
    _, _, _, data = pipeline

    np.testing.assert_allclose(
        data.X_train[NUMERIC_COLS].mean().to_numpy(),
        np.zeros(len(NUMERIC_COLS)),
        atol=1e-10,
    )


def test_customer_id_excluded_from_features(pipeline):
    _, _, _, data = pipeline

    for X in [data.X_train, data.X_val, data.X_test]:
        assert "customerID" not in X.columns


def test_target_excluded_from_features(pipeline):
    _, _, _, data = pipeline

    for X in [data.X_train, data.X_val, data.X_test]:
        assert "Churn" not in X.columns


def test_feature_columns_match_across_splits(pipeline):
    _, _, _, data = pipeline

    assert list(data.X_train.columns) == data.feature_columns
    assert list(data.X_val.columns) == data.feature_columns
    assert list(data.X_test.columns) == data.feature_columns


def test_gender_one_hot_columns_present(pipeline):
    _, _, _, data = pipeline

    assert "gender_Female" in data.X_train.columns
    assert "gender_Male" in data.X_train.columns


def test_no_missing_or_infinite_features(pipeline):
    _, _, _, data = pipeline

    for X in [data.X_train, data.X_val, data.X_test]:
        assert not X.isna().any().any()
        assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_inference_matches_batch_preprocessing(raw_data, pipeline):
    _, _, _, data = pipeline

    for position in [0, 10, 100]:
        customer_id = data.customer_ids_test.iloc[position]

        raw_customer = raw_data.loc[
            raw_data["customerID"].eq(customer_id)
        ].drop(columns=["Churn"])

        cleaned = prepare_raw_data(
            raw_customer,
            require_target=False,
        )

        engineered = engineer_features(cleaned)

        inference_features = transform_features(
            engineered,
            data.scaler,
            data.feature_columns,
        )

        expected = data.X_test.iloc[[position]]

        np.testing.assert_allclose(
            inference_features.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )


def test_transform_does_not_refit_scaler(pipeline):
    _, _, test_df, data = pipeline

    original_mean = data.scaler.mean_.copy()
    original_variance = data.scaler.var_.copy()
    original_count = data.scaler.n_samples_seen_

    transform_features(
        test_df,
        data.scaler,
        data.feature_columns,
    )

    np.testing.assert_array_equal(
        data.scaler.mean_,
        original_mean,
    )

    np.testing.assert_array_equal(
        data.scaler.var_,
        original_variance,
    )

    assert data.scaler.n_samples_seen_ == original_count


def test_split_is_reproducible(raw_data, pipeline):
    train_df, val_df, test_df, _ = pipeline

    cleaned = prepare_raw_data(raw_data)

    train_again, val_again, test_again = (
        split_train_validation_test(cleaned)
    )

    for original, repeated in [
        (train_df, train_again),
        (val_df, val_again),
        (test_df, test_again),
    ]:
        pd.testing.assert_series_equal(
            original["customerID"],
            repeated["customerID"],
        )


def test_legacy_split_remains_compatible(raw_data):
    cleaned = prepare_raw_data(raw_data)

    train_df, test_df = split_data(cleaned)

    data = fit_transform_features(train_df, test_df)

    assert len(data.X_train) + len(data.X_test) == len(raw_data)
    assert data.X_val is None
