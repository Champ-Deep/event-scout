"""ES-API-1: Idempotency-Key on POST /add_contact_from_image/. Needs a LOCAL Postgres.

    createdb es_idem_test && python -m pytest tests/test_scan_idempotency.py
    (override with TEST_DATABASE_URL; non-local hosts are refused)
"""
import asyncio
import os
import sys
import uuid
from io import BytesIO
from urllib.parse import urlparse

import httpx
import pytest
from PIL import Image
from sqlalchemy import select

URL = os.environ.get("TEST_DATABASE_URL", "postgresql://localhost:5432/es_idem_test")
assert urlparse(URL).hostname in ("localhost", "127.0.0.1"), "tests run against a local Postgres only"
os.environ["DATABASE_URL"] = URL
os.environ.pop("DATABASE_PUBLIC_URL", None)
os.environ.pop("BACKUP_DATABASE_URL", None)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module  # noqa: E402
import database  # noqa: E402
from database import ContactDB, UserDB  # noqa: E402

# One loop for the whole module: asyncpg pool connections are bound to the loop that made them.
LOOP = asyncio.new_event_loop()
KEY_HEADER = "Idempotency-Key"


def run(coro):
    return LOOP.run_until_complete(coro)


def _jpeg() -> bytes:
    buf = BytesIO()
    Image.effect_noise((200, 200), 80).convert("RGB").save(buf, "JPEG")
    assert len(buf.getvalue()) > 1000
    return buf.getvalue()


@pytest.fixture(scope="module", autouse=True)
def schema():
    assert run(database.init_db())
    yield


@pytest.fixture
def env(monkeypatch):
    calls = {"ocr": 0, "pipeline": 0, "webhook": 0}

    async def ocr(path):
        calls["ocr"] += 1
        await asyncio.sleep(0.05)  # lets two concurrent requests both pass the pre-OCR lookup
        return dict(app_module.OCR_EMPTY, name="Test Person", email="t@example.com", phone="+10000000000",
                    company_name="TestCo", _source="stub")

    async def pipeline(*a, **k):
        calls["pipeline"] += 1

    async def webhook(*a, **k):
        calls["webhook"] += 1

    monkeypatch.setattr(app_module, "extract_contact_with_openrouter", ocr)
    monkeypatch.setattr(app_module, "run_contact_pipeline", pipeline)
    monkeypatch.setattr(app_module, "fire_webhook", webhook)
    monkeypatch.setattr(app_module, "AUTO_PIPELINE_ENABLED", True)
    monkeypatch.setattr(app_module, "OPENROUTER_API_KEY", "stub")
    monkeypatch.setattr(app_module, "WEBHOOK_URL", "http://webhook.invalid")

    async def new_user() -> str:
        uid = uuid.uuid4()
        s = await app_module.get_db_session()
        try:
            s.add(UserDB(id=uid, name="T", email=f"{uid}@example.com", password_hash="x"))
            await s.commit()
        finally:
            await s.close()
        return str(uid)

    async def post(user_id, key=None):
        headers = {"x-api-key": app_module.APP_API_KEY}
        if key is not None:
            headers[KEY_HEADER] = key
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await c.post(f"/add_contact_from_image/?user_id={user_id}", headers=headers,
                                files={"file": ("business-card.jpg", _jpeg(), "image/jpeg")})

    async def rows(user_id, key=None):
        s = await app_module.get_db_session()
        try:
            q = select(ContactDB).where(ContactDB.user_id == uuid.UUID(user_id))
            if key:
                q = q.where(ContactDB.client_scan_id == uuid.UUID(key))
            return list((await s.execute(q)).scalars())
        finally:
            await s.close()

    async def settle():
        await asyncio.sleep(0.1)  # background tasks started with create_task

    return calls, new_user, post, rows, settle


def test_b1_same_key_twice_one_contact(env):
    calls, new_user, post, rows, settle = env
    uid, key = run(new_user()), str(uuid.uuid4())
    r1, r2 = run(post(uid, key)), run(post(uid, key))
    run(settle())
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()
    assert b1["status"] == b2["status"] == "success"
    assert b1["contact_id"] == b2["contact_id"]
    assert b1["idempotent_replay"] is False and b2["idempotent_replay"] is True
    assert b2["extracted_fields"] == {"name": "Test Person", "email": "t@example.com", "phone": "+10000000000",
                                      "company_name": "TestCo", "title": "", "linkedin": "N/A"}
    assert len(run(rows(uid, key))) == 1
    assert calls == {"ocr": 1, "pipeline": 1, "webhook": 1}


def test_b2_concurrent_same_key_one_contact(env):
    calls, new_user, post, rows, settle = env
    uid, key = run(new_user()), str(uuid.uuid4())

    async def both():
        return await asyncio.gather(post(uid, key), post(uid, key))

    r1, r2 = run(both())
    run(settle())
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["contact_id"] == r2.json()["contact_id"]
    assert len(run(rows(uid, key))) == 1
    assert len(run(rows(uid))) == 1
    assert calls["pipeline"] == 1 and calls["webhook"] == 1


def test_b3_no_header_unchanged_plus_flag(env):
    calls, new_user, post, rows, settle = env
    uid = run(new_user())
    r = run(post(uid))
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "success" and b["idempotent_replay"] is False
    assert {"message", "extracted_fields", "contact_id", "qr_base64", "photo_base64", "pipeline_started"} <= set(b)
    (row,) = run(rows(uid))
    assert str(row.id) == b["contact_id"] and row.client_scan_id is None


def test_b4_non_uuid_key_is_422(env):
    calls, new_user, post, rows, settle = env
    uid = run(new_user())
    r = run(post(uid, "abc"))
    assert r.status_code == 422
    assert r.json() == {"detail": "Idempotency-Key must be a UUID"}
    assert calls["ocr"] == 0 and run(rows(uid)) == []
