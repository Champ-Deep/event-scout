"""Unit tests for hash_password / verify_password (app.py:229,236).

Bcrypt has a 72-byte input cap that app.py trims to. These tests pin
that behavior so a future refactor doesn't silently break login for
users with long passwords or unicode.
"""
from __future__ import annotations

import pytest


def test_hash_and_verify_round_trip(app_module):
    hashed = app_module.hash_password("hunter2")
    assert hashed != "hunter2"
    assert app_module.verify_password("hunter2", hashed) is True


def test_verify_rejects_wrong_password(app_module):
    hashed = app_module.hash_password("correct horse battery staple")
    assert app_module.verify_password("wrong", hashed) is False


def test_hash_is_salted_per_call(app_module):
    """Identical inputs must produce different hashes (random salt)."""
    a = app_module.hash_password("same-password")
    b = app_module.hash_password("same-password")
    assert a != b
    assert app_module.verify_password("same-password", a) is True
    assert app_module.verify_password("same-password", b) is True


def test_unicode_password_round_trip(app_module):
    pw = "пароль-密码-🔐"
    hashed = app_module.hash_password(pw)
    assert app_module.verify_password(pw, hashed) is True


def test_long_password_truncated_to_72_bytes(app_module):
    """Bcrypt only sees the first 72 bytes — passwords that agree on
    those bytes verify as equal. This is a known property, not a bug,
    and the test exists so anyone touching the truncation in app.py
    knows what they're changing."""
    base = "a" * 72
    hashed = app_module.hash_password(base)
    assert app_module.verify_password(base, hashed) is True
    assert app_module.verify_password(base + "ignored-suffix", hashed) is True


def test_verify_with_malformed_hash_raises_or_returns_false(app_module):
    """A garbage hash must not crash login silently — bcrypt raises
    ValueError, which the caller (login_user) wraps. We assert the
    contract: either False or a clear exception, never True."""
    with pytest.raises(Exception):
        app_module.verify_password("anything", "not-a-bcrypt-hash")
