"""Unit tests for parse_vcard (app.py:819).

This regex-based parser is the deserialization half of the QR
contact-exchange flow. If it stops matching common vCard params (TYPE,
PREF, CHARSET) or different line endings, every QR scan silently
produces "N/A" contacts.
"""
from __future__ import annotations


def test_full_vcard_round_trip(app_module):
    """Pin the format that create_qr (app.py:786) emits."""
    vcard = (
        "BEGIN:VCARD\r\n"
        "VERSION:3.0\r\n"
        "N:Sarah Johnson\r\n"
        "FN:Sarah Johnson\r\n"
        "ORG:TechCorp Solutions\r\n"
        "TEL;TYPE=WORK,VOICE:+1-555-234-5678\r\n"
        "EMAIL;TYPE=PREF,INTERNET:sarah@techcorp.com\r\n"
        "URL:linkedin.com/in/sarahjohnson\r\n"
        "END:VCARD"
    )
    parsed = app_module.parse_vcard(vcard)
    assert parsed["name"] == "Sarah Johnson"
    assert parsed["company_name"] == "TechCorp Solutions"
    assert parsed["phone"] == "+1-555-234-5678"
    assert parsed["email"] == "sarah@techcorp.com"
    assert parsed["linkedin"] == "linkedin.com/in/sarahjohnson"


def test_missing_fields_default_to_na(app_module):
    parsed = app_module.parse_vcard("BEGIN:VCARD\r\nEND:VCARD")
    assert parsed == {
        "name": "N/A",
        "email": "N/A",
        "phone": "N/A",
        "linkedin": "N/A",
        "company_name": "N/A",
    }


def test_lf_only_line_endings(app_module):
    """vCards from some sources use \\n instead of \\r\\n."""
    vcard = (
        "BEGIN:VCARD\n"
        "FN:Alex Smith\n"
        "EMAIL:alex@example.com\n"
        "END:VCARD"
    )
    parsed = app_module.parse_vcard(vcard)
    assert parsed["name"] == "Alex Smith"
    assert parsed["email"] == "alex@example.com"


def test_n_field_used_when_fn_missing(app_module):
    """Older vCards may omit FN; the formatted-name fallback to N keeps
    parsing alive."""
    vcard = "BEGIN:VCARD\nN:Smith;Alex;;;\nEND:VCARD"
    parsed = app_module.parse_vcard(vcard)
    # Falls back to whatever N: contained
    assert parsed["name"] != "N/A"
    assert "Smith" in parsed["name"]


def test_email_with_multiple_type_params(app_module):
    """EMAIL;TYPE=PREF;TYPE=INTERNET:... must still be extracted."""
    vcard = (
        "BEGIN:VCARD\n"
        "FN:Contact\n"
        "EMAIL;TYPE=PREF;TYPE=INTERNET:user@example.com\n"
        "END:VCARD"
    )
    assert app_module.parse_vcard(vcard)["email"] == "user@example.com"


def test_url_with_https_scheme(app_module):
    vcard = (
        "BEGIN:VCARD\n"
        "FN:Contact\n"
        "URL:https://linkedin.com/in/someone\n"
        "END:VCARD"
    )
    assert (
        app_module.parse_vcard(vcard)["linkedin"]
        == "https://linkedin.com/in/someone"
    )


def test_values_with_leading_trailing_whitespace_stripped(app_module):
    vcard = "BEGIN:VCARD\nFN:   Padded Name   \nEND:VCARD"
    assert app_module.parse_vcard(vcard)["name"] == "Padded Name"


def test_returns_dict_with_exactly_five_keys(app_module):
    """Contract: callers (add_contact_from_qr) destructure these keys."""
    parsed = app_module.parse_vcard("BEGIN:VCARD\nEND:VCARD")
    assert set(parsed.keys()) == {"name", "email", "phone", "linkedin", "company_name"}
