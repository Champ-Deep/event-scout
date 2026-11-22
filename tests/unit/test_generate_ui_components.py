"""Unit tests for generate_ui_components (app.py:649).

The frontend's renderUIComponents() dispatches on component "type" —
contact_cards / exhibitor_cards / score_summary / action_buttons /
quick_replies. If the backend stops emitting one of these shapes, the
chat UI silently degrades to plain text.
"""
from __future__ import annotations


def _types(components):
    return [c["type"] for c in components]


def _by_type(components, t):
    return next(c for c in components if c["type"] == t)


def test_empty_state_returns_only_default_quick_replies(app_module):
    components = app_module.generate_ui_components(
        query="anything",
        response_text="some response",
        retrieved_contacts=[],
        all_contacts=[],
        exhibitors=[],
        user_profile={},
    )
    assert _types(components) == ["quick_replies"]
    qr = _by_type(components, "quick_replies")["data"]
    assert isinstance(qr, list) and 1 <= len(qr) <= 4


def test_retrieved_contacts_render_as_contact_cards(app_module):
    contacts = [
        {"id": "c1", "name": "Alice", "company_name": "Acme"},
        {"id": "c2", "name": "Bob", "company_name": "Beta"},
    ]
    components = app_module.generate_ui_components(
        query="who are my contacts",
        response_text="...",
        retrieved_contacts=contacts,
        all_contacts=contacts,
        exhibitors=[],
        user_profile={},
    )
    assert "contact_cards" in _types(components)
    cards = _by_type(components, "contact_cards")["data"]
    assert cards == contacts


def test_contact_cards_capped_at_four(app_module):
    contacts = [{"id": f"c{i}", "name": f"N{i}", "company_name": f"C{i}"} for i in range(10)]
    components = app_module.generate_ui_components(
        query="show contacts",
        response_text="...",
        retrieved_contacts=contacts,
        all_contacts=contacts,
        exhibitors=[],
        user_profile={},
    )
    cards = _by_type(components, "contact_cards")["data"]
    assert len(cards) == 4


def test_exhibitor_keyword_triggers_exhibitor_cards(app_module):
    exhibitors = [{"name": "WHX Booth A", "booth": "A1"}, {"name": "Demo Hall", "booth": "B2"}]
    components = app_module.generate_ui_components(
        query="which exhibitors should I visit?",
        response_text="...",
        retrieved_contacts=[],
        all_contacts=[],
        exhibitors=exhibitors,
        user_profile={},
    )
    assert "exhibitor_cards" in _types(components)


def test_exhibitor_cards_skipped_without_exhibitor_keyword(app_module):
    """Same exhibitors but query doesn't mention them — should not appear."""
    exhibitors = [{"name": "WHX Booth A", "booth": "A1"}]
    components = app_module.generate_ui_components(
        query="who are my contacts",
        response_text="...",
        retrieved_contacts=[],
        all_contacts=[],
        exhibitors=exhibitors,
        user_profile={},
    )
    assert "exhibitor_cards" not in _types(components)


def test_score_summary_buckets_correctly(app_module):
    contacts = [
        {"id": "1", "name": "A", "lead_score": 90, "lead_temperature": "hot"},
        {"id": "2", "name": "B", "lead_score": 85, "lead_temperature": "hot"},
        {"id": "3", "name": "C", "lead_score": 60, "lead_temperature": "warm"},
        {"id": "4", "name": "D", "lead_score": 30, "lead_temperature": "cold"},
        {"id": "5", "name": "E"},  # unscored
    ]
    components = app_module.generate_ui_components(
        query="show me my hot leads",
        response_text="...",
        retrieved_contacts=[],
        all_contacts=contacts,
        exhibitors=[],
        user_profile={},
    )
    assert "score_summary" in _types(components)
    summary = _by_type(components, "score_summary")["data"]
    assert summary["total"] == 5
    assert summary["scored"] == 4
    assert summary["hot"] == 2
    assert summary["warm"] == 1
    assert summary["cold"] == 1
    # Top contacts sorted descending by score, capped at 3
    assert [c["id"] for c in summary["top_contacts"]] == ["1", "2", "3"]


def test_score_summary_skipped_when_no_scored_contacts(app_module):
    """If nothing is scored, omit the summary entirely — don't emit
    an empty card."""
    components = app_module.generate_ui_components(
        query="show hot leads",
        response_text="...",
        retrieved_contacts=[],
        all_contacts=[{"id": "1", "name": "A"}],
        exhibitors=[],
        user_profile={},
    )
    assert "score_summary" not in _types(components)


def test_action_buttons_present_when_contacts_retrieved(app_module):
    contacts = [{"id": "c1", "name": "Alice", "company_name": "Acme"}]
    components = app_module.generate_ui_components(
        query="show alice",
        response_text="...",
        retrieved_contacts=contacts,
        all_contacts=contacts,
        exhibitors=[],
        user_profile={},
    )
    assert "action_buttons" in _types(components)
    actions = _by_type(components, "action_buttons")["data"]
    action_kinds = {a["action"] for a in actions}
    # Unscored contact -> "score" action included; "research" + "pitch" always present
    assert {"score", "research", "pitch"}.issubset(action_kinds)
    # All actions must reference the retrieved contact
    assert all(a["contact_id"] == "c1" for a in actions)


def test_score_action_omitted_for_already_scored_contact(app_module):
    contacts = [{"id": "c1", "name": "Alice", "company_name": "Acme", "lead_score": 75}]
    components = app_module.generate_ui_components(
        query="alice",
        response_text="...",
        retrieved_contacts=contacts,
        all_contacts=contacts,
        exhibitors=[],
        user_profile={},
    )
    actions = _by_type(components, "action_buttons")["data"]
    assert "score" not in {a["action"] for a in actions}


def test_component_schema_keys_stable(app_module):
    """Frontend renderUIComponents() relies on these exact keys.
    Any new component MUST have type+data; pin the contract."""
    contacts = [{"id": "c1", "name": "Alice", "company_name": "Acme", "lead_score": 80, "lead_temperature": "hot"}]
    components = app_module.generate_ui_components(
        query="show me hot leads exhibitors",
        response_text="...",
        retrieved_contacts=contacts,
        all_contacts=contacts,
        exhibitors=[{"name": "Booth1"}],
        user_profile={},
    )
    for c in components:
        assert set(c.keys()) == {"type", "data"}, c
        assert isinstance(c["type"], str)
