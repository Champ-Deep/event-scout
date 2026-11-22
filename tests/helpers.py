"""Shared helpers for integration tests."""
from __future__ import annotations

import time
import uuid

import requests


def register_user(base: str, email: str, password: str, name: str) -> str:
    """Register a new user against the live API and return user_id."""
    r = requests.post(
        f"{base}/register/",
        json={"name": name, "email": email, "password": password},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    user_id = data.get("user_id")
    if not user_id:
        raise RuntimeError(f"Register did not return user_id: {data}")
    return user_id


def make_email(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}@example.com"
