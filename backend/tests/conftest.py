"""Shared pytest fixtures.

The environment is configured *before* any ``app.*`` import so that the cached
:class:`~app.core.config.Settings` singleton is built with test values.  Tests
run against a file-backed SQLite database created by the real Alembic
migrations -- not ``metadata.create_all`` -- so that a broken migration fails
the suite instead of silently diverging from production.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="securerag-tests-"))
_DB_PATH = _TMP_ROOT / "test.db"

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": f"sqlite:///{_DB_PATH.as_posix()}",
        "JWT_SECRET_KEY": "test-secret-not-for-production",
        "LOG_FORMAT": "console",
        "LOG_LEVEL": "WARNING",
        "LLM_PROVIDER": "echo",
        "EMBEDDING_PROVIDER": "hashing",
        "EMBEDDING_DIMENSIONS": "256",
        "RATE_LIMIT_ENABLED": "false",
        "ALLOW_REGISTRATION": "true",
    }
)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402
from app.core.config import BACKEND_DIR, settings  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base, User, UserRole  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _migrated_database() -> Iterator[None]:
    """Build the schema once per session using the real migrations."""
    cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(cfg, "head")
    yield
    engine.dispose()


@pytest.fixture
def db() -> Iterator[Session]:
    """A session that is rolled back to a clean slate after each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        # Truncate in FK-safe order rather than dropping the schema: much
        # faster than re-running migrations per test.
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    from app.db.session import get_db

    app = create_app()

    def _override_get_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# User / auth helpers
# ----------------------------------------------------------------------

DEFAULT_PASSWORD = "Str0ng-Test-Passw0rd!"


def make_user(
    db: Session,
    *,
    email: str | None = None,
    role: UserRole = UserRole.USER,
    password: str = DEFAULT_PASSWORD,
) -> User:
    from app.auth.password import hash_password

    user = User(
        email=email or f"user-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password(password),
        role=role.value,
        full_name="Test User",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def auth_headers(user: User) -> dict[str, str]:
    from app.auth.tokens import create_access_token

    return {"Authorization": f"Bearer {create_access_token(user)}"}


@pytest.fixture
def user(db: Session) -> User:
    return make_user(db, email="alice@example.com")


@pytest.fixture
def other_user(db: Session) -> User:
    return make_user(db, email="bob@example.com")


@pytest.fixture
def admin_user(db: Session) -> User:
    return make_user(db, email="admin@example.com", role=UserRole.ADMIN)


@pytest.fixture
def user_headers(user: User) -> dict[str, str]:
    return auth_headers(user)


@pytest.fixture
def other_user_headers(other_user: User) -> dict[str, str]:
    return auth_headers(other_user)


@pytest.fixture
def admin_headers(admin_user: User) -> dict[str, str]:
    return auth_headers(admin_user)


@pytest.fixture
def settings_fixture():
    return settings
