"""Shared fixtures for unit and integration tests.

Two flavors of test live under tests/:

  * unit/        — fast, no network. Imports `app.app` and uses
                   `fastapi.testclient.TestClient`. Heavy ML deps are
                   stubbed by the root conftest.
  * integration/ — hits the live Railway deployment using `requests`.
                   Only runs when LIVE_API_BASE + LIVE_API_KEY are set
                   in the environment.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Iterator, List

import pytest
import requests


# ---------------------------------------------------------------------------
# Unit-test fixtures (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app_module():
    """Import the FastAPI app once per session.

    Relies on the root conftest having already stubbed
    sentence_transformers — without that this would download ~80MB.
    """
    import app as app_module  # noqa: WPS433 — late import is intentional
    return app_module


@pytest.fixture(scope="session")
def test_client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


@pytest.fixture(scope="session")
def app_api_key() -> str:
    """The API key the in-process app is configured with."""
    return os.environ.get("APP_API_KEY", "test-api-key")


# ---------------------------------------------------------------------------
# Live-API fixtures
# ---------------------------------------------------------------------------

LIVE_API_BASE_DEFAULT = "https://event-scout-production.up.railway.app"


@pytest.fixture(scope="session")
def live_api_base() -> str:
    return os.environ.get("LIVE_API_BASE", LIVE_API_BASE_DEFAULT).rstrip("/")


@pytest.fixture(scope="session")
def live_api_key() -> str:
    key = os.environ.get("LIVE_API_KEY", "")
    if not key:
        pytest.skip("LIVE_API_KEY not set — skipping live test")
    return key


@pytest.fixture(scope="session")
def live_session(live_api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-API-Key": live_api_key})
    return s


@pytest.fixture
def fresh_user(live_api_base: str, live_api_key: str) -> dict:
    """Register a brand-new user on the live API. Returned dict has
    user_id, email, password. The test is responsible for cleanup of
    contacts it creates; users themselves are left in place (no admin
    delete-user endpoint exists for self-service)."""
    from tests.helpers import register_user, make_email

    suffix = uuid.uuid4().hex[:10]
    email = make_email("pytest")
    password = "pytest-password-1!"
    name = f"Pytest User {suffix}"
    user_id = register_user(live_api_base, email, password, name)
    return {
        "user_id": user_id,
        "email": email,
        "password": password,
        "name": name,
    }


@pytest.fixture
def contact_cleanup(live_api_base: str, live_session: requests.Session):
    """Track contact IDs created during a test and delete them afterward."""
    created: List[tuple[str, str]] = []  # (user_id, contact_id)

    def _track(user_id: str, contact_id: str) -> None:
        created.append((user_id, contact_id))

    yield _track

    for user_id, contact_id in created:
        try:
            live_session.delete(
                f"{live_api_base}/contact/{contact_id}?user_id={user_id}",
                timeout=10,
            )
        except requests.RequestException:
            pass
