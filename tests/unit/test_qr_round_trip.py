"""End-to-end QR round-trip: Contact -> create_qr -> scan_qr_image -> parse_vcard.

If any link in this chain breaks, QR contact exchange silently degrades
to "N/A" contacts in production. This test pins the whole chain.
"""
from __future__ import annotations

import os

import pytest


def _make_contact(app_module, **fields):
    """Build a Contact pydantic instance with sensible defaults."""
    defaults = dict(
        name="Round Trip",
        email="rt@example.com",
        phone="+1-555-111-2222",
        linkedin="linkedin.com/in/roundtrip",
        company_name="RoundTrip Inc",
    )
    defaults.update(fields)
    return app_module.Contact(**defaults)


@pytest.fixture
def qr_dir(tmp_path, app_module, monkeypatch):
    """Point QR_DIR at a tmp directory so we don't pollute saved_qr/."""
    monkeypatch.setattr(app_module, "QR_DIR", str(tmp_path))
    return tmp_path


def test_qr_round_trip_preserves_all_fields(app_module, qr_dir):
    pyzbar_available = True
    try:
        from pyzbar.pyzbar import decode as _decode
        if _decode.__module__ == "app":
            pyzbar_available = False
    except ImportError:
        pyzbar_available = False

    if not pyzbar_available:
        pytest.skip("pyzbar/libzbar0 not available — QR decode disabled")

    original = _make_contact(app_module)
    result = app_module.create_qr(original)
    assert os.path.exists(result["qr_path"])

    decoded_text = app_module.scan_qr_image(result["qr_path"])
    assert decoded_text is not None, "QR could not be decoded"

    parsed = app_module.parse_vcard(decoded_text)
    assert parsed["name"] == original.name
    assert parsed["email"] == original.email
    assert parsed["phone"] == original.phone
    assert parsed["linkedin"] == original.linkedin
    assert parsed["company_name"] == original.company_name


def test_scan_returns_none_for_image_without_qr(app_module, qr_dir, tmp_path):
    """No QR present -> None, not crash."""
    from PIL import Image

    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), "white").save(blank)
    assert app_module.scan_qr_image(str(blank)) is None


def test_create_qr_uses_provided_contact_id(app_module, qr_dir):
    fixed_id = "11111111-1111-1111-1111-111111111111"
    result = app_module.create_qr(_make_contact(app_module), contact_id=fixed_id)
    assert result["contact_id"] == fixed_id
    assert result["qr_path"].endswith(f"qr_{fixed_id}.png")


def test_create_qr_returns_base64_png(app_module, qr_dir):
    """The base64 payload must be a valid PNG so the frontend can render
    it via data:image/png;base64,..."""
    import base64

    result = app_module.create_qr(_make_contact(app_module))
    raw = base64.b64decode(result["qr_base64"])
    # PNG magic bytes
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
