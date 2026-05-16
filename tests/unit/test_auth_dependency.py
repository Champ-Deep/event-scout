"""Unit tests for verify_api_key (app.py:209).

This is the single gate guarding ~70 protected endpoints. If it stops
raising 401, every protected route becomes unauthenticated.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_verify_api_key_accepts_correct_key(app_module, app_api_key):
    assert app_module.verify_api_key(app_api_key) == app_api_key


def test_verify_api_key_rejects_wrong_key(app_module):
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key("totally-wrong-key")
    assert exc.value.status_code == 401
    assert "Invalid API Key" in exc.value.detail


def test_verify_api_key_rejects_missing_header(app_module):
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key(None)
    assert exc.value.status_code == 401


def test_verify_api_key_rejects_empty_string(app_module):
    with pytest.raises(HTTPException) as exc:
        app_module.verify_api_key("")
    assert exc.value.status_code == 401
