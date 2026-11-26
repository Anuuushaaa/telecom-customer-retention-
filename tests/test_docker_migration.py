"""Isolated tests for Docker-compatible MLflow storage migration."""
import hashlib
import sqlite3
from contextlib import closing

import pytest

from src import docker_migration as migration


ROOT = 'file:///C:/Project/mlflow_store/artifacts'
DEFAULT = 'file:///C:/Project/mlruns/0'


@pytest.mark.parametrize('uri,expected', [
    (None, None), ('', ''), ('models:/m-123', 'models:/m-123'),
    ('runs:/run/artifacts', 'runs:/run/artifacts'),
    ('mlflow-artifacts:/existing', 'mlflow-artifacts:/existing'),
    (ROOT, 'mlflow-artifacts:/'),
    (ROOT + '/', 'mlflow-artifacts:/'),
    (ROOT + '/models/m-123/artifacts', 'mlflow-artifacts:/models/m-123/artifacts'),
    ('file:c:/project/mlflow_store/artifacts/run', 'mlflow-artifacts:/run'),
    (ROOT + '/my%20model', 'mlflow-artifacts:/my model'),
    (DEFAULT, 'mlflow-artifacts:/default/0'),
])
def test_convert_uri(uri, expected):
    assert migration.convert_artifact_uri(uri, ROOT, DEFAULT) == expected


@pytest.mark.parametrize('uri', ['https://example.com/model', 's3://bucket/model', 'relative/path'])
def test_unsupported_uri(uri):
    with pytest.raises(ValueError, match='Unsupported artifact URI'):
        migration.convert_artifact_uri(uri, ROOT, DEFAULT)


@pytest.mark.parametrize('uri', [ROOT + '-other/run', 'file:///D:/other', DEFAULT + '/run'])
def test_unknown_uri(uri):
    with pytest.raises(ValueError, match='Unrecognized artifact location'):
        migration.convert_artifact_uri(uri, ROOT, DEFAULT)


def test_normalize_windows_uri():
    assert migration.normalize_file_uri('file:C:\\My%20Project\\artifacts\\') == 'C:/My Project/artifacts'


def test_sha256(tmp_path):
    file = tmp_path / 'model.bin'
    data = b'abc' * 400000
    file.write_bytes(data)
    assert migration.calculate_sha256(file) == hashlib.sha256(data).hexdigest()


def test_manifest(tmp_path):
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'nested' / 'model.bin').write_bytes(b'model')
    assert migration.build_manifest(tmp_path) == {
        'nested/model.bin': hashlib.sha256(b'model').hexdigest()
    }


def test_manifest_detects_change(tmp_path):
    file = tmp_path / 'model.bin'
    file.write_bytes(b'first')
    before = migration.build_manifest(tmp_path)
    file.write_bytes(b'second')
    assert migration.build_manifest(tmp_path) != before


def test_database_integrity():
    with closing(sqlite3.connect(':memory:')) as db:
        migration.verify_database(db)


def test_foreign_key_violation():
    with closing(sqlite3.connect(':memory:')) as db:
        db.executescript('CREATE TABLE parent(id INTEGER PRIMARY KEY); CREATE TABLE child(pid INTEGER REFERENCES parent(id)); INSERT INTO child VALUES (99);')
        with pytest.raises(RuntimeError, match='Foreign key violations'):
            migration.verify_database(db)


def test_missing_schema():
    with closing(sqlite3.connect(':memory:')) as db:
        with pytest.raises(RuntimeError, match='Missing column'):
            migration.validate_schema(db)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    source = tmp_path / 'mlflow_store'
    destination = tmp_path / 'docker_storage'
    staging = tmp_path / '.docker_storage_staging'
    for name, value in [('PROJECT_ROOT', tmp_path), ('SOURCE_DIR', source), ('DESTINATION_DIR', destination), ('STAGING_DIR', staging)]:
        monkeypatch.setattr(migration, name, value)
    artifacts = source / 'artifacts'
    (artifacts / 'run' / 'artifacts').mkdir(parents=True)
    (artifacts / 'models' / 'm-123' / 'artifacts').mkdir(parents=True)
    (artifacts / 'run' / 'artifacts' / 'metric.txt').write_bytes(b'metric')
    (artifacts / 'models' / 'm-123' / 'artifacts' / 'model.bin').write_bytes(b'model')
    root = artifacts.resolve().as_uri()
    default = (tmp_path / 'mlruns' / '0').resolve().as_uri()
    database = source / migration.DATABASE_NAME
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript('''
            CREATE TABLE experiments (artifact_location TEXT);
            CREATE TABLE runs (artifact_uri TEXT);
            CREATE TABLE logged_models (artifact_location TEXT);
            CREATE TABLE model_versions (name TEXT, version INTEGER, status TEXT, run_id TEXT, storage_location TEXT, source TEXT);
            CREATE TABLE registered_model_aliases (alias TEXT, version INTEGER);
        ''')
        db.executemany('INSERT INTO experiments VALUES (?)', [(root,), (default,)])
        db.execute('INSERT INTO runs VALUES (?)', (root + '/run/artifacts',))
        db.execute('INSERT INTO logged_models VALUES (?)', (root + '/models/m-123/artifacts',))
        db.execute('INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?)', ('telecom-churn-classifier', 2, 'READY', 'run', root + '/models/m-123/artifacts', 'models:/m-123'))
        db.execute('INSERT INTO registered_model_aliases VALUES (?, ?)', ('champion', 2))
        for table in migration.PRESERVED_TABLES:
            if table != 'registered_model_aliases':
                db.execute(f'CREATE TABLE "{table}" (value TEXT)')
                db.execute(f'INSERT INTO "{table}" VALUES (?)', (table + '-preserved',))
    return source, destination, staging


def test_migrate_database_counts_and_metadata(storage):
    source, _, _ = storage
    path = source / migration.DATABASE_NAME
    with closing(sqlite3.connect(path)) as db:
        before = migration.snapshot_preserved_tables(db)
    changes = migration.migrate_database(path)
    assert changes == {'experiments.artifact_location': 2, 'runs.artifact_uri': 1, 'logged_models.artifact_location': 1, 'model_versions.storage_location': 1, 'model_versions.source': 0}
    with closing(sqlite3.connect(path)) as db:
        assert migration.snapshot_preserved_tables(db) == before
        assert db.execute('SELECT source FROM model_versions').fetchone() == ('models:/m-123',)
        assert db.execute('SELECT artifact_uri FROM runs').fetchone() == ('mlflow-artifacts:/run/artifacts',)
        migration.verify_database(db)
    assert all(count == 0 for count in migration.migrate_database(path).values())


def test_database_rollback(storage):
    source, _, _ = storage
    path = source / migration.DATABASE_NAME
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE model_versions SET source='s3://unknown/model'")
        before = db.execute('SELECT * FROM experiments').fetchall()
    with pytest.raises(ValueError, match='Unsupported artifact URI'):
        migration.migrate_database(path)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('SELECT * FROM experiments').fetchall() == before


def test_successful_storage_migration(storage, capsys):
    source, destination, staging = storage
    original_hash = migration.calculate_sha256(source / migration.DATABASE_NAME)
    original_manifest = migration.build_manifest(source / 'artifacts')
    migration.migrate_storage()
    assert destination.is_dir()
    assert not staging.exists()
    assert migration.calculate_sha256(source / migration.DATABASE_NAME) == original_hash
    assert migration.build_manifest(source / 'artifacts') == original_manifest
    assert migration.build_manifest(destination / 'artifacts') == original_manifest
    assert (destination / 'artifacts/default/0').is_dir()
    migration.verify_artifact_locations(destination / migration.DATABASE_NAME, destination / 'artifacts')
    assert 'DOCKER STORAGE MIGRATION SUCCESSFUL' in capsys.readouterr().out


@pytest.mark.parametrize('which', [1, 2])
def test_existing_directory_protected(storage, which):
    directory = storage[which]
    directory.mkdir()
    sentinel = directory / 'keep.txt'
    sentinel.write_text('keep')
    with pytest.raises(FileExistsError):
        migration.migrate_storage()
    assert sentinel.read_text() == 'keep'


def test_missing_database(storage):
    source, destination, staging = storage
    (source / migration.DATABASE_NAME).unlink()
    with pytest.raises(FileNotFoundError):
        migration.migrate_storage()
    assert not destination.exists() and not staging.exists()


def test_missing_artifacts(storage):
    source, destination, staging = storage
    (source / 'artifacts').rename(source / 'other')
    with pytest.raises(FileNotFoundError):
        migration.migrate_storage()
    assert not destination.exists() and not staging.exists()


@pytest.mark.parametrize('uri,error,match', [
    ('mlflow-artifacts:/missing', FileNotFoundError, 'Missing artifact location'),
    ('mlflow-artifacts:/../outside', RuntimeError, 'Unsafe artifact path'),
])
def test_invalid_artifact_location(storage, uri, error, match):
    source, _, _ = storage
    path = source / migration.DATABASE_NAME
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('UPDATE runs SET artifact_uri=?', (uri,))
    with pytest.raises(error, match=match):
        migration.verify_artifact_locations(path, source / 'artifacts')


def test_failed_migration_cleanup(storage):
    source, destination, staging = storage
    path = source / migration.DATABASE_NAME
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE model_versions SET source='s3://unknown/model'")
    before = migration.calculate_sha256(path)
    manifest = migration.build_manifest(source / 'artifacts')
    with pytest.raises(ValueError):
        migration.migrate_storage()
    assert not staging.exists() and not destination.exists()
    assert migration.calculate_sha256(path) == before
    assert migration.build_manifest(source / 'artifacts') == manifest


def test_corrupted_copy_cleanup(storage, monkeypatch):
    source, destination, staging = storage
    copytree = migration.shutil.copytree
    def corrupt_copy(src, dst, *args, **kwargs):
        result = copytree(src, dst, *args, **kwargs)
        if dst == staging / 'artifacts':
            (dst / 'run/artifacts/metric.txt').write_bytes(b'corrupt')
        return result
    monkeypatch.setattr(migration.shutil, 'copytree', corrupt_copy)
    with pytest.raises(RuntimeError, match='Copied artifact integrity check failed'):
        migration.migrate_storage()
    assert not staging.exists() and not destination.exists()
    assert (source / 'artifacts/run/artifacts/metric.txt').read_bytes() == b'metric'
