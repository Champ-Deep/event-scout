"""Tests for the contact CSV export (Harbinger list-import compatibility).

Two layers:
  1. Pure unit tests for export_csv (no app import, no DB).
  2. Route tests for /export_contacts/ via TestClient, with the heavy AI/QR
     modules stubbed and the DB session faked, so we can assert per-user
     scoping without Postgres.

Run: python -m pytest tests/test_export_csv.py -v
"""
import csv
import io
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, __file__.rsplit("/", 2)[0])  # repo root

from export_csv import (  # noqa: E402
    ADMIN_CSV_HEADERS,
    CONTACT_CSV_HEADERS,
    admin_contact_rows,
    build_csv,
    contact_rows,
    safe_cell,
)

# --- Harbinger's contacts-shape aliases, mirrored from csv-mapping.ts -------
# (name, company, title, email, linkedin, domain, location alias lists).
HARBINGER_CONTACT_ALIASES = {
    "personname": ["name", "full name", "full_name", "contact", "contact name",
                   "person", "person name", "lead name"],
    "company": ["company", "company name", "organization", "organisation",
                "account", "employer"],
    "title": ["title", "job title", "role", "position", "designation"],
    "email": ["email", "work email", "email address", "e-mail", "mail",
              "business email"],
    "linkedin": ["linkedin", "linkedin url", "linkedin profile", "li url",
                 "profile", "profile url"],
    "domain": ["domain", "website", "site", "company domain"],
    "location": ["location", "city", "country", "region", "geography"],
}


def _header_field(header):
    """Which Harbinger field a lowercased header auto-maps to (first match)."""
    h = header.lower().strip()
    for field, aliases in HARBINGER_CONTACT_ALIASES.items():
        if h in aliases:
            return field
    return None


# ==================== Layer 1: pure export_csv ====================

class TestSafeCell:
    def test_none_becomes_empty(self):
        assert safe_cell(None) == ""

    def test_formula_prefixes_guarded(self):
        for raw in ("=1+1", "+9198765", "-55", "@mention"):
            assert safe_cell(raw) == "'" + raw, raw

    def test_plain_values_untouched(self):
        for raw in ("Acme", "a@b.com", "55", " plain "):
            assert safe_cell(raw) == raw


class TestBuildCsv:
    def test_starts_with_utf8_bom(self):
        out = build_csv(["Name"], [["Ravi"]])
        assert out.startswith("\ufeff")
        assert out.encode("utf-8-sig").decode("utf-8-sig") == out  # round-trips

    def test_header_then_rows(self):
        out = build_csv(["A", "B"], [["1", "2"]])
        rows = list(csv.reader(io.StringIO(out.lstrip("\ufeff"))))
        assert rows == [["A", "B"], ["1", "2"]]

    def test_quoting_of_commas_quotes_newlines(self):
        out = build_csv(["Name"], [['Doe, "Jane"'], ["line1\nline2"]])
        parsed = list(csv.reader(io.StringIO(out.lstrip("\ufeff"))))
        assert parsed[1] == ['Doe, "Jane"']
        assert parsed[2] == ["line1\nline2"]

    def test_formula_guard_applied_to_cells(self):
        out = build_csv(["Name"], [["=cmd|calc"]])
        assert "'=cmd|calc" in out


class TestContactRows:
    def _contact(self, **kw):
        base = dict(id=uuid.uuid4(), name="Ravi Kumar", email="ravi@acme.com",
                    phone="+91 98765", linkedin="https://linkedin.com/in/ravi",
                    company_name="Acme", event_name="WHX 2026", notes="met at booth",
                    source="scan", lead_score=82, lead_temperature="hot")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_missing_values_are_empty_not_na(self):
        c = self._contact(email=None, phone=None, linkedin=None, company_name=None)
        row = contact_rows([c], {})[0]
        # Name, Email, Phone, LinkedIn, Company at indexes 0..4
        assert row[1:5] == ["", "", "", ""]
        assert "N/A" not in [str(v) for v in row]

    def test_pipeline_columns_present_when_available(self):
        c = self._contact()
        pip = types.SimpleNamespace(research_summary="summary", pitch_angle="angle",
                                    pitch_email_subject="subject",
                                    gamma_deck_url="https://gamma.app/x")
        row = contact_rows([c], {str(c.id): pip})[0]
        assert row[10] == "summary"      # Research Summary
        assert row[11] == "subject"      # Pitch Email Subject
        assert row[12] == "https://gamma.app/x"

    def test_newlines_flattened(self):
        c = self._contact(notes="a\nb")
        row = contact_rows([c], {})[0]
        assert row[6] == "a b"

    def test_title_and_website_from_scan(self):
        c = self._contact(title="VP Sales", links=[{"url": "acme.com", "label": "Website", "added_by": "scan"}])
        row = contact_rows([c], {})[0]
        assert row[13] == "VP Sales" and row[14] == "acme.com"
        assert contact_rows([self._contact()], {})[0][13:] == ["", ""]


class TestHarbingerHeaderMapping:
    def test_identity_headers_auto_map(self):
        fields = [_header_field(h) for h in CONTACT_CSV_HEADERS]
        assert fields[0] == "personname"   # Name
        assert fields[1] == "email"        # Email
        assert fields[3] == "linkedin"     # LinkedIn
        assert fields[4] == "company"      # Company

    def test_no_header_maps_to_the_wrong_field(self):
        expected = {"Name": "personname", "Email": "email",
                    "LinkedIn": "linkedin", "Company": "company",
                    "Title": "title", "Website": "domain"}
        for h in CONTACT_CSV_HEADERS:
            got = _header_field(h)
            if h in expected:
                assert got == expected[h], h
            else:
                # Everything else must be unmapped so the importer ignores it
                # instead of stealing a canonical field.
                assert got is None, f"{h} unexpectedly maps to {got}"


# ==================== Layer 2: route tests (stubbed deps + fake DB) ====================

@pytest.fixture(scope="module")
def app_client():
    import os
    # Stub heavy optional deps before importing app.
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

    os.environ.setdefault("APP_API_KEY", "1234")
    import app as app_module
    from fastapi.testclient import TestClient
    return app_module, TestClient(app_module.app)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeSession:
    """Records executed statements; returns contacts/pipelines from fixtures."""

    def __init__(self, contacts, pipelines=()):
        self.contacts = contacts
        self.pipelines = pipelines
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        entity = stmt.column_descriptions[0]["entity"]
        name = getattr(entity, "__name__", "")
        if name == "ContactDB":
            return FakeResult(self.contacts)
        return FakeResult(list(self.pipelines))

    async def close(self):
        pass


def _contact(user_id, name, email):
    return types.SimpleNamespace(
        id=uuid.uuid4(), user_id=user_id, name=name, email=email,
        phone=None, linkedin=None, company_name="Acme", event_name="WHX 2026",
        notes="", source="manual", lead_score=None, lead_temperature=None,
        links=[], created_at=datetime.now(timezone.utc), admin_notes="")


class TestExportRoute:
    @pytest.fixture()
    def client(self, app_client):
        app_module, client = app_client
        client.app_module = app_module
        return client

    def _export(self, client, session, user_id):
        async def fake_get_db_session():
            return session
        client.app_module.get_db_session = fake_get_db_session
        return client.get(
            f"/export_contacts/?user_id={user_id}&format=csv",
            headers={"X-API-Key": "1234"},
        )

    def test_csv_response_shape(self, client):
        uid = uuid.uuid4()
        sess = FakeSession([_contact(uid, "Ravi", "ravi@acme.com")])
        r = self._export(client, sess, str(uid))
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "event_scout_contacts.csv" in r.headers["content-disposition"]

    def test_csv_body_bom_headers_and_rows(self, client):
        uid = uuid.uuid4()
        sess = FakeSession([_contact(uid, "Ravi", "ravi@acme.com")])
        r = self._export(client, sess, str(uid))
        assert r.content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM
        rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
        assert rows[0] == CONTACT_CSV_HEADERS
        assert rows[1][0] == "Ravi"
        assert rows[1][1] == "ravi@acme.com"
        assert rows[1][5] == "WHX 2026"  # event_name column

    def test_scoped_query_filters_by_user_id(self, client):
        uid_a, uid_b = uuid.uuid4(), uuid.uuid4()
        sess = FakeSession([_contact(uid_a, "Ravi", "ravi@acme.com"),
                            _contact(uid_b, "Meera", "meera@beta.io")])
        r = self._export(client, sess, str(uid_a))
        assert r.status_code == 200
        # The route's first query must carry a WHERE on contacts.user_id = uid_a
        stmt = sess.statements[0]
        compiled = str(stmt.compile())
        assert "contacts.user_id" in compiled
        params = {str(v) for v in stmt.compile().params.values()}
        assert str(uid_a) in params
        assert str(uid_b) not in params

    def test_formula_injection_guarded_in_route(self, client):
        uid = uuid.uuid4()
        c = _contact(uid, "=HYPERLINK(evil)", "ravi@acme.com")
        c.notes = "+9198765;=cmd"
        sess = FakeSession([c])
        r = self._export(client, sess, str(uid))
        body = r.content.decode("utf-8-sig")
        assert "'=HYPERLINK(evil)" in body
        assert "'+9198765;=cmd" in body

    def test_json_format_unchanged(self, client):
        uid = uuid.uuid4()
        sess = FakeSession([_contact(uid, "Ravi", "ravi@acme.com")])
        async def fake_get_db_session():
            return session_holder[0]
        session_holder = [sess]
        client.app_module.get_db_session = fake_get_db_session
        r = client.get(f"/export_contacts/?user_id={uid}&format=json",
                       headers={"X-API-Key": "1234"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["contacts"][0]["name"] == "Ravi"

    def test_admin_headers_still_align(self):
        # Admin CSV keeps its audit columns; identity headers in same slots.
        assert ADMIN_CSV_HEADERS[:5] == ["Name", "Email", "Phone", "LinkedIn", "Company"]
        uid = uuid.uuid4()
        c = _contact(uid, "Ravi", None)
        rows = admin_contact_rows([c], {}, {str(uid): "Deep"})
        assert rows[0][1] == ""          # empty email, not N/A
        assert rows[0][11] == "Deep"     # Scanned By (user name)
