"""Opt-in smoke tests against the running local Compose deployment."""
import os
import math
from pathlib import Path

import httpx
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv('RUN_DOCKER_INTEGRATION') != '1',
    reason='Set RUN_DOCKER_INTEGRATION=1 to test the live Docker stack',
)
ROOT = Path(__file__).resolve().parents[1]
API = 'http://127.0.0.1:8000'
MLFLOW = 'http://127.0.0.1:5000'
FRONTEND = 'http://127.0.0.1:8501'


@pytest.fixture(scope='module')
def client():
    with httpx.Client(timeout=120, trust_env=False) as session:
        yield session


@pytest.fixture(scope='module')
def customers():
    path = ROOT / 'data/raw/telco_churn.csv'
    assert path.is_file(), 'Original telecom dataset required for live tests'
    data = pd.read_csv(path, dtype={'customerID': str, 'TotalCharges': str}, keep_default_na=False)
    return data.drop(columns=['Churn']).head(100).to_dict('records')


@pytest.mark.parametrize('url', [API+'/health', API+'/ready', MLFLOW+'/health', FRONTEND+'/_stcore/health'])
def test_health(client, url):
    assert client.get(url).status_code == 200


def test_readiness(client):
    result = client.get(API+'/ready').json()
    assert result['status'] == 'ready'
    assert result['model_loaded'] is True
    assert result['model_alias'] == 'champion'
    assert str(result['model_version']) == '2'


def test_model_info(client):
    result = client.get(API+'/model/info').json()
    assert result['model_name'] == 'xgboost'
    assert result['calibration_method'] == 'sigmoid'
    assert result['classification_threshold'] == .5
    assert result['max_batch_size'] == 1000
    assert result['run_id'] == '1121b4c5e9ec41c2a3fd5d9f7c8d2ff1'


def post(client, endpoint, payload):
    response = client.post(API+endpoint, json=payload)
    response.raise_for_status()
    return response.json()


def test_single_prediction(client, customers):
    result = post(client, '/predict', {'customer':customers[0], 'classification_threshold':.5})
    assert result['customerID'] == '7590-VHVEG'
    assert result['raw_churn_probability'] == pytest.approx(.8740338683128357, abs=1e-6)
    assert result['churn_probability'] == pytest.approx(.7136371398760405, abs=1e-6)
    assert result['prediction'] == 1


def test_threshold_override(client, customers):
    result = post(client, '/predict', {'customer':customers[0], 'classification_threshold':.8})
    assert result['prediction'] == 0
    assert result['churn_probability'] == pytest.approx(.7136371398760405, abs=1e-6)


def test_batch(client, customers):
    result = post(client, '/predict/batch', {'customers':customers, 'classification_threshold':.5})
    predictions = result['predictions']
    assert len(predictions) == 100
    assert [p['customerID'] for p in predictions] == [c['customerID'] for c in customers]
    assert sum(p['prediction'] for p in predictions) == 27
    assert all(math.isfinite(p['churn_probability']) and 0 <= p['churn_probability'] <= 1 for p in predictions)


def test_individual_shap(client, customers):
    result = post(client, '/explain', {'customer':customers[0], 'top_n':10, 'classification_threshold':.5})
    assert result['explanation_method'] == 'TreeSHAP'
    assert result['output_space'] == 'raw_log_odds'
    assert result['total_features'] == 29
    assert len(result['top_contributors']) == 10
    assert abs(result['additivity_error']) < 1e-4


def test_global_shap(client, customers):
    result = post(client, '/explain/global', {'customers':customers})
    assert result['total_customers'] == 100
    assert result['total_features'] == 29
    assert len(result['feature_importance']) == 29
    assert abs(result['max_additivity_error']) < 1e-4


def test_empty_batch(client):
    assert client.post(API+'/predict/batch', json={'customers':[]}).status_code == 422


def test_invalid_threshold(client, customers):
    assert client.post(API+'/predict', json={'customer':customers[0], 'classification_threshold':1.5}).status_code == 422


def test_invalid_customer_id(client, customers):
    customer = dict(customers[0], customerID='')
    assert client.post(API+'/predict', json={'customer':customer}).status_code == 422


def test_registry():
    from mlflow import MlflowClient
    registry = MlflowClient(tracking_uri=MLFLOW, registry_uri=MLFLOW)
    experiment = registry.get_experiment_by_name('Telecom Customer Retention')
    assert experiment is not None
    assert len(registry.search_runs([experiment.experiment_id])) == 6
    name = 'telecom-churn-classifier'
    versions = registry.search_model_versions(f"name='{name}'")
    assert {str(v.version) for v in versions} == {'1','2','3'}
    assert all(v.status == 'READY' for v in versions)
    assert str(registry.get_model_version_by_alias(name, 'champion').version) == '2'
    assert str(registry.get_model_version_by_alias(name, 'challenger').version) == '1'
