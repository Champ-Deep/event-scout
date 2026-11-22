"""Unit tests for normalize_phone (app.py:4073).

Used by /admin/contacts/duplicates to detect duplicate contacts that
differ only in punctuation/formatting. False positives waste admin
time; false negatives let duplicates slip through.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("inp,expected", [
    ("", ""),
    (None, ""),
    ("N/A", ""),
    ("5552345678", "5552345678"),
    ("+15552345678", "+15552345678"),
    ("+1 (555) 234-5678", "+15552345678"),
    ("555-234-5678", "5552345678"),
    ("(555) 234-5678", "5552345678"),
    ("+1.555.234.5678", "+15552345678"),
    ("  555 234 5678  ", "5552345678"),
    ("+971 50 123 4567", "+971501234567"),
    ("+49-170-9876543", "+491709876543"),
    ("555 234 5678 x123", "5552345678123"),
])
def test_normalize_phone(app_module, inp, expected):
    assert app_module.normalize_phone(inp) == expected


def test_different_formats_normalize_to_same_value(app_module):
    """The whole point of normalize_phone — these must compare equal."""
    formats = [
        "+1 (555) 234-5678",
        "+1.555.234.5678",
        "+1-555-234-5678",
        "+15552345678",
    ]
    normalized = [app_module.normalize_phone(p) for p in formats]
    assert len(set(normalized)) == 1, f"Inconsistent normalization: {normalized}"
