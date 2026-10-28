"""Unit tests for the admin export column registry and presets (contact_view). Stdlib only:
    cd apps/backend && python3 -m pytest tests/test_export_columns.py -q
"""
import types
import uuid
from datetime import datetime, timezone

import pytest

import contact_view as cv


def _contact(**kw):
    base = dict(id=uuid.uuid4(), name="Ravi Kumar", job_title="VP Sales", company_name="Acme", email="ravi@acme.com",
                phone="+91 98765", linkedin="https://linkedin.com/in/ravi", event_name="WHX 2026", notes="met at booth",
                source="scan", lead_score=82, links=[{"url": "acme.com", "label": "Website", "added_by": "scan"}],
                created_at=datetime(2026, 10, 29, tzinfo=timezone.utc))
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_default_row_unchanged_by_registry():
    c = _contact()
    c.console_score = 75
    row = cv.admin_export_row(c, "Deep")
    assert row[:9] == [str(c.id), "Ravi Kumar", "VP Sales", "Acme", "ravi@acme.com", "+91 98765",
                       "https://linkedin.com/in/ravi", "WHX 2026", "Deep"]
    assert len(row) == len(cv.ADMIN_EXPORT_HEADER)
    assert set(cv.ADMIN_EXPORT_HEADER) <= set(cv.ADMIN_EXPORT_COLUMNS)


def test_columns_param_orders_and_rejects_unknown():
    assert cv.parse_export_columns(None) == cv.ADMIN_EXPORT_HEADER
    assert cv.parse_export_columns("Website, Name,Name") == ["Website", "Name"]
    with pytest.raises(ValueError):
        cv.parse_export_columns("Name,Shoe size")


def test_website_column_and_harbinger_preset():
    c = _contact()
    assert cv.ADMIN_EXPORT_COLUMNS["Website"](c, "") == "acme.com"
    assert cv.ADMIN_EXPORT_COLUMNS["Website"](_contact(links=[]), "") == ""
    for h in ("Name", "Title", "Company", "Email", "LinkedIn", "Website"):  # what Harbinger auto-maps
        assert h in cv.HARBINGER_PRESET["columns"]
    assert cv.parse_export_columns(",".join(cv.HARBINGER_PRESET["columns"])) == cv.HARBINGER_PRESET["columns"]


def test_validate_presets_round_trip_and_errors():
    ok = cv.validate_presets([{"name": " Hot leads ", "columns": ["Name", "Email"], "filters": {"minScore": 70, "bogus": 1, "event": ""}}])
    assert ok == [{"name": "Hot leads", "columns": ["Name", "Email"], "filters": {"minScore": 70}}]
    for bad in ([{"name": "", "columns": ["Name"]}], [{"name": "ChampHarbinger", "columns": ["Name"]}],
                [{"name": "a", "columns": ["Name"]}, {"name": "A", "columns": ["Name"]}], [{"name": "x", "columns": ["Nope"]}], "nope"):
        with pytest.raises(ValueError):
            cv.validate_presets(bad)
