"""Fail-fast secrets validation for non-development environments."""

from __future__ import annotations

import os
from typing import List
from urllib.parse import unquote, urlparse

_DEFAULT_JWT = {
    "change-me-in-production",
    "aegis-dev-change-me",
    "change-me",
    "secret",
    "jwt-secret",
}

_DEFAULT_PASSWORDS = {
    "change_me",
    "changeme",
    "password",
    "postgres",
    "minioadmin",
    "admin",
    "root",
}


def _looks_default_password(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    if v in _DEFAULT_PASSWORDS:
        return True
    if "change" in v and "me" in v:
        return True
    return False


def _password_from_database_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return unquote(parsed.password or "")
    except Exception:
        return ""


def validate_production_secrets(
    *,
    app_env: str | None = None,
    jwt_secret: str | None = None,
    database_url: str | None = None,
    postgres_password: str | None = None,
    minio_user: str | None = None,
    minio_password: str | None = None,
) -> List[str]:
    """
    Return a list of human-readable problems. Empty list means OK.
    Only enforces when APP_ENV is not development/dev/test/local.
    """
    env = (app_env if app_env is not None else os.getenv("APP_ENV", "development")).lower()
    if env in {"development", "dev", "test", "local", "ci"}:
        return []

    problems: List[str] = []
    jwt = jwt_secret if jwt_secret is not None else os.getenv("JWT_SECRET", "aegis-dev-change-me")
    if not jwt or jwt.strip().lower() in _DEFAULT_JWT or "change-me" in jwt.lower():
        problems.append("JWT_SECRET is missing or still a default placeholder")

    db_url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
    db_pass = _password_from_database_url(db_url)
    pg_pass = (
        postgres_password
        if postgres_password is not None
        else os.getenv("POSTGRES_PASSWORD", "")
    )
    if _looks_default_password(db_pass) or _looks_default_password(pg_pass):
        problems.append("DATABASE_URL / POSTGRES_PASSWORD uses a default or empty password")

    m_user = minio_user if minio_user is not None else os.getenv("MINIO_ROOT_USER", "minioadmin")
    m_pass = (
        minio_password
        if minio_password is not None
        else os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    )
    if _looks_default_password(m_user) or _looks_default_password(m_pass):
        problems.append("MINIO_ROOT_USER / MINIO_ROOT_PASSWORD still use default credentials")

    return problems


def assert_safe_for_environment() -> None:
    problems = validate_production_secrets()
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(
            f"Refusing to start with APP_ENV=production (or non-dev): {joined}. "
            "Override secrets in .env before deploying."
        )
