"""
src/docker_migration.py

Migrate local MLflow storage to Docker-compatible
artifact URIs without retraining models or modifying
the original MLflow database.
"""

import hashlib
import shutil
import sqlite3

from contextlib import closing
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "mlflow_store"

DESTINATION_DIR = PROJECT_ROOT / "docker_storage"

STAGING_DIR = PROJECT_ROOT / ".docker_storage_staging"

DATABASE_NAME = "mlflow.db"

ARTIFACT_URI = "mlflow-artifacts:/"

URI_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("model_versions", "source"),
)

PRESERVED_TABLES = (
    "registered_models",
    "registered_model_aliases",
    "model_version_tags",
    "metrics",
    "latest_metrics",
    "params",
    "tags",
    "logged_model_metrics",
    "logged_model_params",
    "logged_model_tags",
)


def calculate_sha256(path):
    """Calculate the SHA-256 fingerprint of a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest(directory):
    """Create a relative-path artifact manifest."""

    directory = Path(directory)

    return {
        path.relative_to(directory).as_posix():
        calculate_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verify_database(connection):
    """Verify SQLite integrity and foreign keys."""

    integrity = connection.execute(
        "PRAGMA integrity_check"
    ).fetchall()

    if integrity != [("ok",)]:
        raise RuntimeError(
            f"Database integrity failure: {integrity}"
        )

    violations = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    if violations:
        raise RuntimeError(
            f"Foreign key violations: {violations}"
        )


def validate_schema(connection):
    """Check required migration columns."""

    for table, column in URI_COLUMNS:

        columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            )
        }

        if column not in columns:
            raise RuntimeError(
                f"Missing column: {table}.{column}"
            )


def snapshot_preserved_tables(connection):
    """Capture records that must remain unchanged."""

    return {
        table: sorted(
            connection.execute(
                f'SELECT * FROM "{table}"'
            ).fetchall(),
            key=repr,
        )
        for table in PRESERVED_TABLES
    }


def normalize_file_uri(uri):
    """Normalize Windows file URIs for comparison."""

    value = unquote(uri).replace("\\", "/")

    if value.startswith("file:"):
        value = value[5:]

    return value.lstrip("/").rstrip("/")


def convert_artifact_uri(
    uri,
    source_artifact_root,
    default_artifact_root,
):
    """Convert known local artifact paths."""

    if uri is None or uri == "":
        return uri

    if uri.startswith(
        ("models:/", "runs:/", "mlflow-artifacts:/")
    ):
        return uri

    if not uri.startswith("file:"):
        raise ValueError(
            f"Unsupported artifact URI: {uri}"
        )

    normalized = normalize_file_uri(uri)

    source_root = normalize_file_uri(
        source_artifact_root
    )

    default_root = normalize_file_uri(
        default_artifact_root
    )

    if normalized.casefold() == source_root.casefold():
        return ARTIFACT_URI

    if normalized.casefold().startswith(
        source_root.casefold() + "/"
    ):
        relative = normalized[len(source_root) + 1:]

        return ARTIFACT_URI + relative

    if normalized.casefold() == default_root.casefold():
        return ARTIFACT_URI + "default/0"

    raise ValueError(
        f"Unrecognized artifact location: {uri}"
    )


def migrate_database(database_path):
    """Rewrite artifact URIs in the deployment copy."""

    source_artifact_root = (
        SOURCE_DIR / "artifacts"
    ).resolve().as_uri()

    default_artifact_root = (
        PROJECT_ROOT / "mlruns" / "0"
    ).resolve().as_uri()

    changes = {}

    with closing(
        sqlite3.connect(database_path)
    ) as connection:

        verify_database(connection)
        validate_schema(connection)

        before = snapshot_preserved_tables(
            connection
        )

        with connection:

            for table, column in URI_COLUMNS:

                records = connection.execute(
                    f'SELECT DISTINCT "{column}" '
                    f'FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL'
                ).fetchall()

                updated = 0

                for (old_uri,) in records:

                    new_uri = convert_artifact_uri(
                        old_uri,
                        source_artifact_root,
                        default_artifact_root,
                    )

                    if new_uri == old_uri:
                        continue

                    cursor = connection.execute(
                        f'UPDATE "{table}" '
                        f'SET "{column}" = ? '
                        f'WHERE "{column}" = ?',
                        (new_uri, old_uri),
                    )

                    updated += cursor.rowcount

                changes[f"{table}.{column}"] = updated

            after = snapshot_preserved_tables(
                connection
            )

            if before != after:
                raise RuntimeError(
                    "Preserved registry or tracking "
                    "records changed unexpectedly."
                )

            verify_database(connection)

    return changes


def verify_artifact_locations(
    database_path,
    artifact_directory,
):
    """Verify migrated artifact paths exist."""

    artifact_directory = (
        Path(artifact_directory).resolve()
    )

    with closing(
        sqlite3.connect(database_path)
    ) as connection:

        for table, column in URI_COLUMNS:

            records = connection.execute(
                f'SELECT DISTINCT "{column}" '
                f'FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL'
            ).fetchall()

            for (uri,) in records:

                if not uri.startswith(ARTIFACT_URI):
                    continue

                relative = uri[len(ARTIFACT_URI):]

                path = (
                    artifact_directory / relative
                ).resolve()

                if not path.is_relative_to(
                    artifact_directory
                ):
                    raise RuntimeError(
                        f"Unsafe artifact path: {uri}"
                    )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing artifact location: {path}"
                    )


def print_registry_summary(database_path):
    """Display migrated model registry information."""

    with closing(
        sqlite3.connect(database_path)
    ) as connection:

        runs = connection.execute(
            "SELECT COUNT(*) FROM runs"
        ).fetchone()[0]

        models = connection.execute(
            "SELECT name, version, status, run_id "
            "FROM model_versions "
            "ORDER BY version"
        ).fetchall()

        aliases = connection.execute(
            "SELECT alias, version "
            "FROM registered_model_aliases "
            "ORDER BY alias"
        ).fetchall()

        print(f"\nPreserved experiment runs: {runs}")

        print("\nRegistered models:")

        for name, version, status, run_id in models:
            print(
                f"  {name} | Version {version} "
                f"| {status} | Run {run_id}"
            )

        print("\nModel aliases:")

        for alias, version in aliases:
            print(f"  {alias}: Version {version}")


def migrate_storage():
    """Create and verify Docker deployment storage."""

    source_database = SOURCE_DIR / DATABASE_NAME

    source_artifacts = SOURCE_DIR / "artifacts"

    if not source_database.is_file():
        raise FileNotFoundError(source_database)

    if not source_artifacts.is_dir():
        raise FileNotFoundError(source_artifacts)

    if DESTINATION_DIR.exists():
        raise FileExistsError(
            f"Deployment storage already exists: "
            f"{DESTINATION_DIR}"
        )

    if STAGING_DIR.exists():
        raise FileExistsError(
            f"Staging storage already exists: "
            f"{STAGING_DIR}"
        )

    original_database_hash = calculate_sha256(
        source_database
    )

    original_manifest = build_manifest(
        source_artifacts
    )

    STAGING_DIR.mkdir()

    try:

        staging_database = (
            STAGING_DIR / DATABASE_NAME
        )

        staging_artifacts = (
            STAGING_DIR / "artifacts"
        )

        print("Creating consistent SQLite backup...")

        source_uri = (
            source_database.resolve().as_uri()
            + "?mode=ro"
        )

        with closing(
            sqlite3.connect(source_uri, uri=True)
        ) as source:

            verify_database(source)

            with closing(
                sqlite3.connect(staging_database)
            ) as destination:

                source.backup(destination)

        print("Copying model artifacts...")

        shutil.copytree(
            source_artifacts,
            staging_artifacts,
        )

        copied_manifest = build_manifest(
            staging_artifacts
        )

        if copied_manifest != original_manifest:
            raise RuntimeError(
                "Copied artifact integrity check failed."
            )

        print(
            f"Verified artifact files: "
            f"{len(copied_manifest)}"
        )

        # The inspected Default experiment has no runs.
        # Reserve a valid artifact directory for it.

        default_directory = (
            staging_artifacts / "default" / "0"
        )

        default_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\nMigrating artifact URIs...")

        changes = migrate_database(
            staging_database
        )

        for field, count in changes.items():
            print(f"  {field}: {count} updated")

        verify_artifact_locations(
            staging_database,
            staging_artifacts,
        )

        if build_manifest(
            staging_artifacts
        ) != original_manifest:
            raise RuntimeError(
                "Model artifact contents changed."
            )

        if calculate_sha256(
            source_database
        ) != original_database_hash:
            raise RuntimeError(
                "Original database changed during migration."
            )

        if build_manifest(
            source_artifacts
        ) != original_manifest:
            raise RuntimeError(
                "Original artifacts changed during migration."
            )

        print_registry_summary(
            staging_database
        )

        STAGING_DIR.rename(
            DESTINATION_DIR
        )

    except Exception:

        shutil.rmtree(
            STAGING_DIR,
            ignore_errors=True,
        )

        raise

    print("\n" + "=" * 65)

    print("DOCKER STORAGE MIGRATION SUCCESSFUL")

    print("=" * 65)

    print("Database integrity: PASSED")
    print("Foreign key integrity: PASSED")
    print("Artifact integrity: PASSED")
    print("Registry preservation: PASSED")
    print("Original storage unchanged: PASSED")
    print("Model retraining: NOT PERFORMED")

    print(
        f"\nDeployment storage: {DESTINATION_DIR}"
    )


if __name__ == "__main__":
    migrate_storage()