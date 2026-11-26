"""Static checks for the local Docker deployment."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def compose():
    return yaml.safe_load((ROOT / 'docker-compose.yaml').read_text(encoding='utf-8'))


def test_services(compose):
    assert set(compose['services']) == {'api', 'frontend', 'mlflow'}


@pytest.mark.parametrize('service,port,endpoint', [
    ('api', 8000, '/ready'), ('frontend', 8501, '/_stcore/health'),
    ('mlflow', 5000, '/health'),
])
def test_service_configuration(compose, service, port, endpoint):
    config = compose['services'][service]
    assert config['ports'] == [f'127.0.0.1:{port}:{port}']
    assert config['build']['context'] == '.'
    assert config['build']['dockerfile'] == f'Dockerfile.{service}'
    assert config['networks'] == ['telecom-network']
    assert config['restart'] == 'unless-stopped'
    health = config['healthcheck']
    assert health['test'][:3] == ['CMD', 'python', '-c']
    assert f'http://127.0.0.1:{port}{endpoint}' in health['test'][3]
    assert health['retries'] > 0
    assert health['start_period']


def test_network(compose):
    assert compose['networks']['telecom-network']['driver'] == 'bridge'


@pytest.mark.parametrize('service,dependency', [('api','mlflow'), ('frontend','api')])
def test_startup_dependency(compose, service, dependency):
    assert compose['services'][service]['depends_on'][dependency]['condition'] == 'service_healthy'


def test_api_environment(compose):
    env = compose['services']['api']['environment']
    expected = {'API_HOST':'0.0.0.0', 'API_PORT':'8000',
                'MLFLOW_TRACKING_URI':'http://mlflow:5000',
                'MLFLOW_REGISTRY_URI':'http://mlflow:5000',
                'REGISTERED_MODEL_NAME':'telecom-churn-classifier',
                'MODEL_ALIAS':'champion', 'CLASSIFICATION_THRESHOLD':'0.5',
                'MAX_BATCH_SIZE':'1000'}
    for key, value in expected.items():
        assert env[key] == value


def test_frontend_environment(compose):
    assert compose['services']['frontend']['environment']['TELECOM_API_URL'] == 'http://api:8000'


def test_storage_mounts(compose):
    assert './docker_storage:/mlflow' in compose['services']['mlflow']['volumes']
    assert './data/raw:/app/data/raw:ro' in compose['services']['frontend']['volumes']


@pytest.mark.parametrize('service,port', [('api',8000), ('frontend',8501), ('mlflow',5000)])
def test_dockerfile(service, port):
    content = (ROOT / f'Dockerfile.{service}').read_text(encoding='utf-8')
    assert content.startswith('FROM python:3.13.12-slim')
    assert f'EXPOSE {port}' in content
    assert f'pip install -r requirements-{service}.txt' in content
    assert 'pip check' in content
    assert not re.search(r'^COPY\s+\.\s', content, re.MULTILINE)


def test_mlflow_artifact_serving():
    content = (ROOT / 'Dockerfile.mlflow').read_text(encoding='utf-8')
    for value in ['sqlite:////mlflow/mlflow.db', '--serve-artifacts', '/mlflow/artifacts', '--allowed-hosts']:
        assert value in content


@pytest.mark.parametrize('service', ['api','frontend','mlflow'])
def test_pinned_requirements(service):
    lines = (ROOT / f'requirements-{service}.txt').read_text(encoding='utf-8').splitlines()
    packages = [line.strip() for line in lines if line.strip() and not line.startswith(('#','-r '))]
    assert packages
    assert all(re.fullmatch(r'[\w.-]+==[\w.+-]+', line) for line in packages)


def test_frontend_shared_dependencies():
    assert '-r requirements-api.txt' in (ROOT / 'requirements-frontend.txt').read_text(encoding='utf-8').splitlines()


def test_mlflow_version_consistency():
    api = (ROOT / 'requirements-api.txt').read_text(encoding='utf-8').splitlines()
    server = (ROOT / 'requirements-mlflow.txt').read_text(encoding='utf-8').strip()
    assert server in api


@pytest.mark.parametrize('pattern', ['docker_storage/', '.docker_storage_staging/', 'mlflow_store', 'data/raw', 'models', '.env', '.streamlit/secrets.toml'])
def test_dockerignore(pattern):
    assert pattern in (ROOT / '.dockerignore').read_text(encoding='utf-8').splitlines()


@pytest.mark.parametrize('pattern', ['docker_storage/', '.docker_storage_staging/', 'mlflow_store/', 'data/raw/*', '.env'])
def test_gitignore(pattern):
    assert pattern in (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
