"""Admin API tests (docs/admin-dashboard-spec.md section 5). Hermetic: no LLM key, no model download.

Run against a throwaway Postgres (it is DROPPED and recreated, never point it at real data):
  TEST_DATABASE_URL=postgresql://user@127.0.0.1:5443/es_test pytest tests/test_admin_api.py
"""
import asyncio
import csv
import io
import os
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to a throwaway Postgres")

if TEST_DB:
    os.environ.update(DATABASE_URL=TEST_DB, APP_API_KEY="testkey", ADMIN_EMAILS="owner@test.dev")
    for k in ("DATABASE_PUBLIC_URL", "BACKUP_DATABASE_URL", "WEBHOOK_URL", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(k, None)

    class _FakeEmbedder:  # deterministic vectors so app.py imports without downloading a model
        def __init__(self, *a, **k): pass
        def encode(self, texts, **k):
            return np.array([np.random.default_rng(abs(hash(t)) % 2**32).random(8) for t in texts], dtype="float32")

    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_FakeEmbedder)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import admin_api  # noqa: E402
    import app as core  # noqa: E402
    from database import Base, ContactDB, ContactPipelineDB, UserDB, _create_engine, ASYNC_DATABASE_URL  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402
    from sqlalchemy import select  # noqa: E402

KEY = {"X-API-Key": "testkey"}
ID = {}  # name -> uuid string, filled by seed


def _run(coro):
    return asyncio.run(coro)


async def _with_db(fn):
    eng = _create_engine(ASYNC_DATABASE_URL)
    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker
        async with async_sessionmaker(eng, expire_on_commit=False)() as s:
            return await fn(s)
    finally:
        await eng.dispose()


async def _seed():
    eng = _create_engine(ASYNC_DATABASE_URL)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await eng.dispose()

    async def go(s):
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        users = {n: UserDB(id=uuid.uuid4(), name=n, email=f"{n.lower()}@test.dev", password_hash="x", is_admin=n in ("Owner", "Admin2"))
                 for n in ("Owner", "Admin2", "Lead", "R1", "R2", "R3")}
        s.add_all(users.values())
        await s.flush()
        users["R1"].team_lead_id = users["R2"].team_lead_id = users["Lead"].id
        ID.update({n: str(u.id) for n, u in users.items()})

        def c(key, owner, name, co, temp, score, event, created, **kw):
            row = ContactDB(id=uuid.uuid4(), user_id=users[owner].id, name=name, company_name=co, lead_temperature=temp,
                            lead_score=score, event_name=event, created_at=created, **kw)
            ID[key] = str(row.id)
            s.add(row)

        c("c1", "R1", "Alice Zeta", "Acme", "hot", 90, "WHX", today + timedelta(seconds=1), email="a@acme.com")
        c("c2", "R1", "Bob Yi", "Beta", "warm", 60, "WHX", now - timedelta(days=3))
        c("c3", "R2", "Carol Xu", "Acme", "cold", 20, "GITEX", now - timedelta(days=10))
        c("c4", "R3", "Dan 100%", "Delta", None, None, "", now - timedelta(days=40), phone="+1555")
        c("c5", "Lead", "Eve", "Echo", "hot", 80, "WHX", today - timedelta(hours=12))
        await s.flush()
        s.add(ContactPipelineDB(contact_id=uuid.UUID(ID["c1"]), user_id=users["R1"].id, status="complete"))
        await s.commit()

    await _with_db(go)


@pytest.fixture(scope="module")
def client():
    _run(_seed())
    with TestClient(core.app) as c:
        yield c


def call(client, method, path, who="Owner", **kw):
    params = {"user_id": ID[who], **kw.pop("params", {})}
    return client.request(method, path, headers=KEY, params=params, **kw)


def ids(resp):
    inv = {v: k for k, v in ID.items()}
    return {inv[c["id"]] for c in resp.json()["contacts"]}


def listing(client, **params):
    r = call(client, "GET", "/admin/leads", params=params)
    assert r.status_code == 200, r.text
    return r


def test_auth(client):
    assert client.get("/admin/leads", params={"user_id": ID["Owner"]}).status_code == 401
    assert call(client, "GET", "/admin/leads", who="R1").status_code == 403


@pytest.mark.parametrize("params,expected", [
    ({}, {"c1", "c2", "c3", "c4", "c5"}),
    ({"q": "acme"}, {"c1", "c3"}),
    ({"q": "100%"}, {"c4"}),            # % is literal
    ({"q": "a_me"}, set()),             # _ is literal, not a wildcard
    ({"team_lead": "Lead"}, {"c1", "c2", "c3", "c5"}),  # reps of the lead plus the lead's own
    ({"team_lead": "none"}, {"c4", "c5"}),
    ({"rep": "R1"}, {"c1", "c2"}),
    ({"temperature": "hot"}, {"c1", "c5"}),
    ({"temperature": "unscored"}, {"c4"}),
    ({"event": "WHX"}, {"c1", "c2", "c5"}),
    ({"date_filter": "today"}, {"c1"}),
    ({"date_filter": "yesterday"}, {"c5"}),
    ({"date_filter": "last_week"}, {"c1", "c2", "c5"}),
    ({"team_lead": "Lead", "temperature": "hot"}, {"c1", "c5"}),
    ({"q": "acme", "team_lead": "Lead", "date_filter": "last_week"}, {"c1"}),
    ({"q": "acme", "rep": "R2", "event": "GITEX", "temperature": "cold"}, {"c3"}),
])
def test_filter_composition(client, params, expected):
    params = {k: (ID[v] if v in ID else v) for k, v in params.items()}
    r = listing(client, **params)
    assert ids(r) == expected and r.json()["total"] == len(expected)


def test_sort_paging_and_validation(client):
    assert [c["name"] for c in listing(client, sort="name_asc").json()["contacts"]] == \
        ["Alice Zeta", "Bob Yi", "Carol Xu", "Dan 100%", "Eve"]
    scores = [c["lead_score"] for c in listing(client, sort="score_desc").json()["contacts"]]
    assert scores == [90, 80, 60, 20, None]  # unscored last
    p2 = listing(client, sort="name_asc", limit=2, page=2).json()
    assert [c["name"] for c in p2["contacts"]] == ["Carol Xu", "Dan 100%"] and p2["has_more"] and p2["total"] == 5
    for bad in ({"temperature": "lava"}, {"sort": "x"}, {"limit": 201}, {"team_lead": "nope"}, {"q": ""}):
        assert call(client, "GET", "/admin/leads", params=bad).status_code == 422
    row = listing(client, rep=ID["R1"], sort="score_desc").json()["contacts"][0]
    assert row["user_name"] == "R1" and row["team_lead_name"] == "Lead" and row["team_lead_id"] == ID["Lead"]


@pytest.mark.parametrize("params", [
    {}, {"q": "acme"}, {"team_lead": "Lead", "temperature": "hot"}, {"team_lead": "none"}, {"date_filter": "last_week", "sort": "name_asc"},
    {"q": "zzz-no-match"},
])
def test_export_count_equals_list_total(client, params):
    params = {k: (ID[v] if v in ID else v) for k, v in params.items()}
    lst = listing(client, limit=200, **params).json()
    r = call(client, "GET", "/admin/leads/export", params=params)
    assert r.status_code == 200
    assert r.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == admin_api.EXPORT_COLUMNS
    assert len(rows) - 1 == lst["total"] == int(r.headers["x-total-count"])
    assert [x[0] for x in rows[1:]] == [c["name"] for c in lst["contacts"]]  # same order as the screen
    assert "attachment" in r.headers["content-disposition"]


def test_export_guards(client, monkeypatch):
    rows = list(csv.reader(io.StringIO(call(client, "GET", "/admin/leads/export", params={"q": "Dan"}).content.decode("utf-8-sig"))))
    assert rows[1][3] == "'+1555"  # formula guard on phone
    assert rows[1][2] == ""        # legacy 'N/A' shown as empty
    monkeypatch.setattr(admin_api, "EXPORT_MAX_ROWS", 2)
    r = call(client, "GET", "/admin/leads/export")
    assert r.status_code == 422 and "Too many rows" in r.json()["detail"]


def test_lead_detail(client):
    d = call(client, "GET", f"/admin/leads/{ID['c1']}").json()["contact"]
    assert d["name"] == "Alice Zeta" and d["user_name"] == "R1" and d["team_lead_name"] == "Lead"
    assert {"lead_score_breakdown", "photo_base64", "linkedin_source", "admin_notes"} <= d.keys()
    assert call(client, "GET", f"/admin/leads/{uuid.uuid4()}").status_code == 404
    assert call(client, "GET", "/admin/leads/not-a-uuid").status_code == 422


def test_reassign(client):
    assert any(m["id"] == ID["c1"] for m in core.faiss_index.metadata[ID["R1"]])
    assert not any(m["id"] == ID["c1"] for m in core.faiss_index.metadata.get(ID["R2"], []))
    url = f"/admin/leads/{ID['c1']}/reassign"

    assert call(client, "POST", url, json={"to_user_id": ID["R1"]}).status_code == 400  # same owner
    assert call(client, "POST", url, json={"to_user_id": str(uuid.uuid4())}).status_code == 404
    assert call(client, "POST", f"/admin/leads/{uuid.uuid4()}/reassign", json={"to_user_id": ID["R2"]}).status_code == 404
    assert call(client, "POST", url, who="R3", json={"to_user_id": ID["R2"]}).status_code == 403

    r = call(client, "POST", url, json={"to_user_id": ID["R2"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["from_name"], body["to_name"], body["faiss_synced"]) == ("R1", "R2", True)

    # contact owner, pipeline owner
    async def owners(s):
        c = (await s.execute(select(ContactDB.user_id).where(ContactDB.id == uuid.UUID(ID["c1"])))).scalar()
        p = (await s.execute(select(ContactPipelineDB.user_id).where(ContactPipelineDB.contact_id == uuid.UUID(ID["c1"])))).scalar()
        return str(c), str(p)
    assert _run(_with_db(owners)) == (ID["R2"], ID["R2"])
    # both FAISS indexes
    assert not any(m["id"] == ID["c1"] for m in core.faiss_index.metadata[ID["R1"]])
    assert any(m["id"] == ID["c1"] for m in core.faiss_index.metadata[ID["R2"]])
    assert ids(listing(client, rep=ID["R2"])) == {"c1", "c3"}
    # moving it back so later tests keep the seed shape; second hop proves the endpoint is symmetric
    assert call(client, "POST", url, json={"to_user_id": ID["R1"]}).status_code == 200


def test_reassign_faiss_failure_still_200(client, monkeypatch):
    def boom(*a, **k): raise RuntimeError("index down")
    monkeypatch.setattr(core.faiss_index, "build_for_user", boom)
    r = call(client, "POST", f"/admin/leads/{ID['c2']}/reassign", json={"to_user_id": ID["R3"]})
    assert r.status_code == 200 and r.json()["faiss_synced"] is False
    monkeypatch.undo()
    assert call(client, "POST", f"/admin/leads/{ID['c2']}/reassign", json={"to_user_id": ID["R1"]}).status_code == 200


def test_team_lead_and_reps(client):
    reps = call(client, "GET", "/admin/reps").json()
    by = {r["name"]: r for r in reps["reps"]}
    assert by["R1"]["contact_count"] == 2 and by["R1"]["hot_count"] == 1 and by["R1"]["warm_count"] == 1
    assert by["R1"]["today_count"] == 1 and by["R1"]["last_scan_at"] and by["Admin2"]["last_scan_at"] is None
    assert reps["team_leads"] == [{"user_id": ID["Lead"], "name": "Lead", "rep_count": 2}]

    url = f"/admin/users/{ID['R3']}/team_lead"
    r = call(client, "PUT", url, json={"team_lead_id": ID["Lead"]})
    assert r.status_code == 200 and r.json()["team_lead_name"] == "Lead"
    assert call(client, "GET", "/admin/reps").json()["team_leads"][0]["rep_count"] == 3
    assert call(client, "PUT", url, json={"team_lead_id": ID["R3"]}).status_code == 400
    assert call(client, "PUT", url, json={"team_lead_id": str(uuid.uuid4())}).status_code == 404
    assert call(client, "PUT", f"/admin/users/{uuid.uuid4()}/team_lead", json={"team_lead_id": None}).status_code == 404
    assert call(client, "PUT", url, json={}).status_code == 422  # key is required, null clears
    r = call(client, "PUT", url, json={"team_lead_id": None})
    assert r.status_code == 200 and r.json()["team_lead_id"] is None


def test_rep_performance(client):
    r = call(client, "GET", f"/admin/reps/{ID['R1']}/performance", params={"days": 7}).json()
    assert r["rep"]["name"] == "R1" and r["avg_score"] == 75.0
    assert len(r["daily"]) == 7 and sum(d["count"] for d in r["daily"]) == 2
    assert r["daily"][-1]["count"] == 1 and r["daily"][-1]["date"] == datetime.now(timezone.utc).date().isoformat()
    assert r["by_event"] == [{"event_name": "WHX", "count": 2}]
    assert call(client, "GET", f"/admin/reps/{ID['Admin2']}/performance").json()["avg_score"] is None
    assert call(client, "GET", f"/admin/reps/{uuid.uuid4()}/performance").status_code == 404
    assert call(client, "GET", f"/admin/reps/{ID['R1']}/performance", params={"days": 91}).status_code == 422


def test_settings_permissions_and_validation(client):
    assert call(client, "GET", "/admin/settings").json()["can_write"] is True
    ro = call(client, "GET", "/admin/settings", who="Admin2").json()
    assert ro["can_write"] is False
    assert call(client, "PUT", "/admin/settings", who="Admin2", json={"auto_pipeline_enabled": True}).status_code == 403
    assert call(client, "PUT", "/admin/settings", who="Admin2", json={"webhook_url": "http://x"}).status_code == 403
    assert core.AUTO_PIPELINE_ENABLED is False  # the 403 changed nothing
    for bad in ({"webhook_url": "http://example.com/hook"}, {"webhook_url": "https://"}, {"webhook_url": "ftp://a.b"},
                {"webhook_url": "https://a.co/" + "x" * 500}, {"auto_pipeline_enabled": "yes"}):
        r = call(client, "PUT", "/admin/settings", json=bad)
        assert r.status_code == 422, bad
    assert core.WEBHOOK_URL == ""


def test_settings_persist_across_restart(client):
    r = call(client, "PUT", "/admin/settings", json={"webhook_url": " https://hooks.example.com/abc ", "auto_pipeline_enabled": True})
    assert r.status_code == 200 and r.json()["settings"] == {"webhook_url": "https://hooks.example.com/abc", "auto_pipeline_enabled": True}
    assert core.WEBHOOK_URL == "https://hooks.example.com/abc" and core.AUTO_PIPELINE_ENABLED is True
    # fresh process: reset the globals to env defaults, boot a second app lifecycle
    core.WEBHOOK_URL, core.AUTO_PIPELINE_ENABLED = "", False
    client.__exit__(None, None, None)
    with TestClient(core.app) as c2:
        s = call(c2, "GET", "/admin/settings").json()
        assert s["settings"] == {"webhook_url": "https://hooks.example.com/abc", "auto_pipeline_enabled": True}
        # empty string clears it, and that clear persists too
        assert call(c2, "PUT", "/admin/settings", json={"webhook_url": ""}).json()["settings"]["webhook_url"] == ""
    client.__enter__()
    assert call(client, "GET", "/admin/settings").json()["settings"]["webhook_url"] == ""
