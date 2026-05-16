"""Multi-user data-isolation tests (live API).

This is the #1 security invariant for Event Scout: user A must never
see, modify, or delete contacts owned by user B. Existing
test_multi_user_flow.py covered only the happy-path read; we
additionally verify:

  * Cross-user GET returns 404 / 403
  * Cross-user PUT returns 404 / 403 and contact is unchanged
  * Cross-user DELETE returns 404 / 403 and contact still exists
  * Search by user A does not surface user B's contacts

Tests are marked `live` because they hit the deployed API. They skip
automatically when LIVE_API_KEY isn't set, so unit-only CI stays fast.
"""
from __future__ import annotations

import uuid

import pytest
import requests

from tests.helpers import register_user, make_email


pytestmark = pytest.mark.live


def _add_contact(session, base, user_id, name, email, company):
    r = session.post(
        f"{base}/add_contact/",
        json={
            "contact": {
                "name": name,
                "email": email,
                "phone": "+1-555-000-0000",
                "linkedin": "N/A",
                "company_name": company,
            },
            "user_id": user_id,
        },
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    assert body.get("status") == "success", body
    return body["contact_id"]


def _register_pair(base: str) -> tuple[str, str]:
    """Register two fresh users and return their user_ids."""
    a = register_user(base, make_email("iso_a"), "pw-a-1!", "Iso A")
    b = register_user(base, make_email("iso_b"), "pw-b-1!", "Iso B")
    return a, b


def test_user_b_cannot_read_user_a_contact(
    live_api_base, live_session, contact_cleanup
):
    """Most critical isolation invariant."""
    user_a, user_b = _register_pair(live_api_base)

    contact_id = _add_contact(
        live_session, live_api_base, user_a,
        "Alice Private Contact", "alice.private@example.com", "AliceCorp",
    )
    contact_cleanup(user_a, contact_id)

    r = live_session.get(
        f"{live_api_base}/contact/{contact_id}?user_id={user_b}",
        timeout=15,
    )
    assert r.status_code in (403, 404), (
        f"User B read user A's contact: HTTP {r.status_code}, body={r.text[:300]}"
    )


def test_user_b_cannot_delete_user_a_contact(
    live_api_base, live_session, contact_cleanup
):
    user_a, user_b = _register_pair(live_api_base)

    contact_id = _add_contact(
        live_session, live_api_base, user_a,
        "Target Contact", "target@example.com", "TargetCorp",
    )
    contact_cleanup(user_a, contact_id)

    r = live_session.delete(
        f"{live_api_base}/contact/{contact_id}?user_id={user_b}",
        timeout=15,
    )
    assert r.status_code in (403, 404), (
        f"User B deleted user A's contact: HTTP {r.status_code}"
    )

    r = live_session.get(
        f"{live_api_base}/contact/{contact_id}?user_id={user_a}",
        timeout=15,
    )
    assert r.status_code == 200, (
        f"After failed cross-user delete, user A can't read own contact: "
        f"HTTP {r.status_code}"
    )


def test_user_b_cannot_update_user_a_contact(
    live_api_base, live_session, contact_cleanup
):
    user_a, user_b = _register_pair(live_api_base)

    contact_id = _add_contact(
        live_session, live_api_base, user_a,
        "Original Name", "original@example.com", "OriginalCorp",
    )
    contact_cleanup(user_a, contact_id)

    r = live_session.put(
        f"{live_api_base}/contact/{contact_id}?user_id={user_b}",
        json={"name": "Hijacked", "company_name": "Attacker Inc"},
        timeout=15,
    )
    assert r.status_code in (403, 404), (
        f"User B updated user A's contact: HTTP {r.status_code}"
    )

    r = live_session.get(
        f"{live_api_base}/contact/{contact_id}?user_id={user_a}",
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("name") == "Original Name", (
        f"Contact was mutated by cross-user update: {body}"
    )


def test_search_does_not_surface_other_users_contacts(
    live_api_base, live_session, contact_cleanup
):
    """User A's search must not return user B's contacts."""
    user_a, user_b = _register_pair(live_api_base)

    marker = f"ZZUNIQUE{uuid.uuid4().hex[:12]}"
    contact_b = _add_contact(
        live_session, live_api_base, user_b,
        f"{marker} Bob", f"{marker.lower()}@example.com", f"{marker}Corp",
    )
    contact_cleanup(user_b, contact_b)

    r = live_session.post(
        f"{live_api_base}/search/",
        json={"query": marker, "user_id": user_a},
        timeout=30,
    )
    assert r.status_code == 200, r.text[:200]
    results = r.json().get("results", [])
    leaked = [
        c for c in results
        if marker.lower() in (
            c.get("name", "") + c.get("email", "") + c.get("company_name", "")
        ).lower()
    ]
    assert not leaked, (
        f"User A search returned user B contacts containing '{marker}': {leaked}"
    )


def test_login_rejects_wrong_password(live_api_base, fresh_user):
    """Wrong-password login must not return a user_id."""
    r = requests.post(
        f"{live_api_base}/login/",
        json={"email": fresh_user["email"], "password": "not-the-password"},
        timeout=15,
    )
    if r.status_code == 200:
        body = r.json()
        assert body.get("status") != "success", body
        assert not body.get("user_id"), body
    else:
        assert r.status_code in (400, 401, 403), r.text[:200]


def test_login_succeeds_for_fresh_user(live_api_base, fresh_user):
    """Sanity: the user we just created can log back in."""
    r = requests.post(
        f"{live_api_base}/login/",
        json={"email": fresh_user["email"], "password": fresh_user["password"]},
        timeout=15,
    )
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body.get("status") == "success", body
    assert body.get("user_id") == fresh_user["user_id"], body
