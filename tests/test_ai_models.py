"""Only low-cost models may be called, whatever env or a saved preference says.

Run: python -m pytest tests/test_ai_models.py -v
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load(monkeypatch, **env):
    for k in ("OCR_MODELS", "SCORING_MODELS", "CHAT_MODEL", "PIPELINE_MODEL", "GEMINI_TEXT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import ai_models
    return importlib.reload(ai_models)


def test_defaults_are_low_cost(monkeypatch):
    m = load(monkeypatch)
    used = set(m.OCR_MODELS) | set(m.SCORING_MODELS) | {m.CHAT_MODEL, m.PIPELINE_MODEL}
    assert used <= m.ALLOWED_OPENROUTER_MODELS
    assert not any(x in mid for mid in used for x in ("opus", "gpt", "pro", "sonnet"))


def test_env_cannot_select_a_frontier_model(monkeypatch):
    m = load(monkeypatch, CHAT_MODEL="anthropic/claude-opus-4", OCR_MODELS="openai/gpt-4.1,google/gemini-3.5-flash-lite")
    assert m.CHAT_MODEL == "anthropic/claude-haiku-5.5"
    assert m.OCR_MODELS == ["google/gemini-3.5-flash-lite"]


def test_old_saved_preferences_map_to_cheap_default(monkeypatch):
    m = load(monkeypatch)
    for key in ("claude-opus", "gpt-5", "gemini-pro", None, "nonsense"):
        assert m.resolve_chat_model(key)["id"] in m.ALLOWED_OPENROUTER_MODELS


def test_app_has_no_hardcoded_frontier_models():
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    for bad in ("claude-opus-4", "openai/gpt-4.1", "gemini-2.5-pro", "'gemini-2.0-flash'"):
        assert bad not in src, bad
