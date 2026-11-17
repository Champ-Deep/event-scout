"""Which AI models Event Scout may call, and how a user's choice maps to one.

Only low-cost models are allowed. Every model is set by env so it can be changed
without a deploy, but a value outside ALLOWED_OPENROUTER_MODELS is ignored, so a
typo or an old saved preference can never route traffic to a frontier model.

Prices are OpenRouter list prices per 1M tokens (input / output) at the time of the
bench in bench/ (2026-10-29), where both models read cards at ~$0.0003 each.
"""
import os

# Haiku-class and flash-lite-class only. Add here deliberately, never via env alone.
ALLOWED_OPENROUTER_MODELS = {
    "anthropic/claude-haiku-5.5",
    "google/gemini-3.5-flash-lite",
}
ALLOWED_GEMINI_MODELS = {
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
}

DEFAULT_PRIMARY = "anthropic/claude-haiku-5.5"
DEFAULT_FALLBACK = "google/gemini-3.5-flash-lite"


def _models_from_env(var: str, default: list, allowed: set) -> list:
    raw = os.environ.get(var, "")
    picked = [m.strip() for m in raw.split(",") if m.strip()] if raw else list(default)
    safe = [m for m in picked if m in allowed]
    if raw and len(safe) != len(picked):
        print(f"[MODELS] {var}: ignored models not on the low-cost list: "
              f"{sorted(set(picked) - set(safe))}")
    return safe or list(default)


# Card OCR (vision). Bench: haiku 100% of fields, flash-lite 97%.
OCR_MODELS = _models_from_env("OCR_MODELS", [DEFAULT_PRIMARY, DEFAULT_FALLBACK], ALLOWED_OPENROUTER_MODELS)
# Lead scoring (text, short JSON answer).
SCORING_MODELS = _models_from_env("SCORING_MODELS", [DEFAULT_FALLBACK, DEFAULT_PRIMARY], ALLOWED_OPENROUTER_MODELS)
# Chat assistant default and research/pitch pipeline.
CHAT_MODEL = _models_from_env("CHAT_MODEL", [DEFAULT_PRIMARY], ALLOWED_OPENROUTER_MODELS)[0]
PIPELINE_MODEL = _models_from_env("PIPELINE_MODEL", [DEFAULT_PRIMARY], ALLOWED_OPENROUTER_MODELS)[0]
# Direct Gemini key (fallback path only).
GEMINI_TEXT_MODEL = _models_from_env("GEMINI_TEXT_MODEL", ["gemini-2.5-flash-lite"], ALLOWED_GEMINI_MODELS)[0]
GEMINI_SCORING_MODELS = _models_from_env(
    "GEMINI_SCORING_MODELS", ["gemini-2.5-flash-lite", "gemini-flash-lite-latest"], ALLOWED_GEMINI_MODELS)

# Choices shown in the app's model picker.
AVAILABLE_MODELS = {
    "claude-haiku": {"id": "anthropic/claude-haiku-5.5", "name": "Claude Haiku 5.5", "provider": "Anthropic"},
    "gemini-flash-lite": {"id": "google/gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash Lite", "provider": "Google"},
}
DEFAULT_MODEL_KEY = next((k for k, v in AVAILABLE_MODELS.items() if v["id"] == CHAT_MODEL), "claude-haiku")

# Keys saved by older app versions; they now resolve to the low-cost default.
LEGACY_MODEL_KEYS = {"claude-opus", "gpt-5", "gemini-pro"}


def resolve_chat_model(model_key) -> dict:
    """Map a user's saved preference to an allowed model. Unknown or legacy keys get the default."""
    return AVAILABLE_MODELS.get(model_key or "", AVAILABLE_MODELS[DEFAULT_MODEL_KEY])
