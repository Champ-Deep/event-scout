"""Session tokens replace the shared API key in the browser.

Run: python -m pytest tests/test_session_auth.py -v
"""
import os
import sys
import types
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

os.environ.setdefault("APP_API_KEY", "1234")

for name in ("faiss", "qrcode", "google", "google.generativeai",
             "sentence_transformers", "pyzbar", "pyzbar.pyzbar"):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()

st = types.ModuleType("sentence_transformers")


class _StubModel:
    def __init__(self, *a, **k):
        pass

    def encode(self, *a, **k):
        return [[0.0] * 3]


st.SentenceTransformer = _StubModel
sys.modules["sentence_transformers"] = st

sys.path.insert(0, __file__.rsplit("/", 2)[0])
import app as app_module  # noqa: E402


def _request(query: str = "") -> Request:
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": query.encode(),
        "headers": [],
        "client": ("127.0.0.1", 123),
        "server": ("127.0.0.1", 8000),
    })


def test_public_config_exposes_no_secrets():
    import asyncio
    data = asyncio.run(app_module.public_config())
    assert data == {"auth_mode": "session", "api_key_in_client": False}
    blob = str(data)
    if app_module.APP_API_KEY:
        assert app_module.APP_API_KEY not in blob
    assert "AIza" not in blob


def test_server_api_key_still_authorizes():
    got = app_module.verify_api_key(_request(), app_module.APP_API_KEY, None)
    assert got == app_module.APP_API_KEY


def test_missing_credentials_are_rejected():
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key(_request(), None, None)
    assert exc.value.status_code == 401


def test_session_token_authorizes_matching_user():
    user_id = str(uuid.uuid4())
    token = app_module.issue_session_token(user_id)
    got = app_module.verify_api_key(
        _request(f"user_id={user_id}"),
        None,
        f"Bearer {token}",
    )
    assert got == token


def test_session_token_rejects_a_different_user():
    token = app_module.issue_session_token(str(uuid.uuid4()))
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key(
            _request(f"user_id={uuid.uuid4()}"),
            None,
            f"Bearer {token}",
        )
    assert exc.value.status_code == 403


def test_expired_session_token_is_rejected():
    user_id = str(uuid.uuid4())
    original = app_module.SESSION_TTL_SECONDS
    app_module.SESSION_TTL_SECONDS = -10
    try:
        token = app_module.issue_session_token(user_id)
    finally:
        app_module.SESSION_TTL_SECONDS = original
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key(_request(f"user_id={user_id}"), None, f"Bearer {token}")
    assert exc.value.status_code == 401


def test_mobile_frontend_has_no_embedded_api_key():
    from pathlib import Path
    html = Path(__file__).resolve().parents[1].joinpath("mobile-frontend.html").read_text()
    assert "CONFIG.API_KEY" not in html
    assert "AIza" not in html
    assert "session_token" in html
    assert "/public-config" in html
