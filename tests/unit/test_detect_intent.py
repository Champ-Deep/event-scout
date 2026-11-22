"""Unit tests for detect_intent (app.py:718).

Drives the routing in /converse/. If pitch_keywords stop firing before
research_keywords, a query like "generate a pitch researching X" would
incorrectly classify as 'research' and skip the slide-generation flow.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("query", [
    "generate a pitch deck for Sarah",
    "Create pitch for Acme Corp",
    "I need a presentation for the meeting",
    "Make some slides about the deal",
    "Draft a proposal",
    "GENERATE PITCH",  # case-insensitive
])
def test_pitch_intent(app_module, query):
    assert app_module.detect_intent(query) == "pitch"


@pytest.mark.parametrize("query", [
    "research Acme Corp",
    "Tell me about TechCo",
    "What do you know about Sarah's company?",
    "Investigate this lead",
    "Look up the company info for ABC Industries",
    "find out about their products",
])
def test_research_intent(app_module, query):
    assert app_module.detect_intent(query) == "research"


@pytest.mark.parametrize("query", [
    "hello",
    "who are my contacts",
    "show me hot leads",
    "what's my next meeting",
    "",
])
def test_chat_intent_default(app_module, query):
    assert app_module.detect_intent(query) == "chat"


def test_pitch_beats_research_when_both_present(app_module):
    """If both intents match, pitch wins (handles the
    'generate a pitch after researching X' edge case)."""
    assert app_module.detect_intent(
        "generate a pitch deck after you research the company"
    ) == "pitch"


def test_intent_is_case_and_whitespace_insensitive(app_module):
    assert app_module.detect_intent("   PITCH DECK   ") == "pitch"
    assert app_module.detect_intent("   Research   ") == "research"
