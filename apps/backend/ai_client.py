"""
OpenRouter client — the one place Event Scout talks to an AI model.

Card reading (vision), lead scoring, the chat assistant (incl. attached images / PDFs), voice
transcription (audio input) and the research / pitch pipeline all go through here.

Environment:
  OPENROUTER_API_KEY           required (https://openrouter.ai/keys)
  OPENROUTER_MODEL             default model for everything          (default google/gemini-2.5-flash)
  OPENROUTER_FALLBACK_MODELS   comma-separated, tried in order        (default google/gemini-2.0-flash-001)
  OPENROUTER_VISION_MODEL      card reading                           (default = OPENROUTER_MODEL)
  OPENROUTER_AUDIO_MODEL       voice transcription (needs audio input) (default = OPENROUTER_MODEL)
  OPENROUTER_CHAT_MODEL        chat assistant                          (default = OPENROUTER_MODEL)
  OPENROUTER_REFERER           sent as HTTP-Referer (OpenRouter app attribution, optional)
"""
import base64
import json
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import httpx

# OPENROUTER_BASE_URL: only for a proxy / gateway in front of OpenRouter (or local tests).
OPENROUTER_URL = (os.environ.get("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "").strip() or "google/gemini-2.5-flash"
FALLBACK_MODELS = [m.strip() for m in os.environ.get("OPENROUTER_FALLBACK_MODELS", "google/gemini-2.0-flash-001").split(",") if m.strip()]
VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "").strip() or DEFAULT_MODEL
AUDIO_MODEL = os.environ.get("OPENROUTER_AUDIO_MODEL", "").strip() or DEFAULT_MODEL
CHAT_MODEL = os.environ.get("OPENROUTER_CHAT_MODEL", "").strip() or DEFAULT_MODEL
REFERER = os.environ.get("OPENROUTER_REFERER", "").strip() or "https://eventscout.app"

ai_configured = bool(OPENROUTER_API_KEY)


class AIError(Exception):
    """No model could answer. `reason` drives the user-facing message:
    not_configured | bad_key | no_credits | rate_limited | timeout | unavailable"""

    def __init__(self, message: str, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason


# Reasons where trying another model can't help (the key / account is the problem).
_FATAL = ("not_configured", "bad_key", "no_credits")


def _headers(title: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": REFERER,
        "X-Title": title,
    }


def _reason_for(status: int, message: str) -> str:
    m = (message or "").lower()
    if status in (401, 403) or "invalid api key" in m or "no auth credentials" in m or "user not found" in m:
        return "bad_key"
    if status == 402 or "credits" in m or "insufficient" in m:
        return "no_credits"
    if status == 429 or "rate limit" in m or "rate-limit" in m:
        return "rate_limited"
    if status == 408:
        return "timeout"
    return "unavailable"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _parse(resp: httpx.Response) -> Tuple[Optional[str], Optional[str], str]:
    """-> (text, error_message, reason)"""
    try:
        data = resp.json()
    except ValueError:
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}", _reason_for(resp.status_code, resp.text)
    err = data.get("error") if isinstance(data, dict) else None
    if err:
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else None
        status = code if isinstance(code, int) else resp.status_code
        return None, msg, _reason_for(status, msg)
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}", _reason_for(resp.status_code, "")
    choices = data.get("choices") or []
    if not choices:
        return None, "no choices in response", "unavailable"
    text = _content_text((choices[0].get("message") or {}).get("content")).strip()
    if not text:
        return None, "empty response", "unavailable"
    return text, None, "ok"


def _body(model: str, messages: list, max_tokens: int, temperature: float, json_mode: bool) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _models(model: Optional[str]) -> List[str]:
    seen, out = set(), []
    for m in [model or DEFAULT_MODEL, *FALLBACK_MODELS]:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _check_configured():
    if not ai_configured:
        raise AIError("OPENROUTER_API_KEY is not set", "not_configured")


def _log(title: str, model: str, msg: str):
    print(f"[AI] {title} · {model}: {msg}")


async def chat(messages: list, *, model: Optional[str] = None, max_tokens: int = 1500, temperature: float = 0.3,
               json_mode: bool = False, timeout: float = 60.0, title: str = "Event Scout") -> Tuple[str, str]:
    """Async chat completion. Returns (text, model_that_answered). Raises AIError."""
    _check_configured()
    errors: List[Tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for m in _models(model):
            try:
                resp = await client.post(OPENROUTER_URL, headers=_headers(title), json=_body(m, messages, max_tokens, temperature, json_mode))
            except httpx.TimeoutException:
                errors.append(("timeout", f"{m}: timed out")); _log(title, m, "timed out"); continue
            except httpx.HTTPError as e:
                errors.append(("unavailable", f"{m}: {e}")); _log(title, m, str(e)); continue
            text, err, reason = _parse(resp)
            if text is not None:
                return text, m
            errors.append((reason, f"{m}: {err}")); _log(title, m, f"{reason} — {err}")
            if reason in _FATAL:
                break
    raise _combined(errors)


def chat_sync(messages: list, *, model: Optional[str] = None, max_tokens: int = 1500, temperature: float = 0.3,
              json_mode: bool = False, timeout: float = 60.0, title: str = "Event Scout") -> Tuple[str, str]:
    """Blocking version (for code that already runs in a worker thread)."""
    _check_configured()
    errors: List[Tuple[str, str]] = []
    with httpx.Client(timeout=timeout) as client:
        for m in _models(model):
            try:
                resp = client.post(OPENROUTER_URL, headers=_headers(title), json=_body(m, messages, max_tokens, temperature, json_mode))
            except httpx.TimeoutException:
                errors.append(("timeout", f"{m}: timed out")); _log(title, m, "timed out"); continue
            except httpx.HTTPError as e:
                errors.append(("unavailable", f"{m}: {e}")); _log(title, m, str(e)); continue
            text, err, reason = _parse(resp)
            if text is not None:
                return text, m
            errors.append((reason, f"{m}: {err}")); _log(title, m, f"{reason} — {err}")
            if reason in _FATAL:
                break
    raise _combined(errors)


class _StreamError(Exception):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


async def chat_stream(messages: list, *, model: Optional[str] = None, max_tokens: int = 1500, temperature: float = 0.3,
                      timeout: float = 75.0, title: str = "Event Scout") -> AsyncIterator[Tuple[str, str]]:
    """Streaming chat completion. Yields ("model", model_id) once a model starts answering, then
    ("delta", text) pieces as they're written. Falls back to the next model only while nothing has
    been sent yet; a model that fails mid-answer raises. Raises AIError."""
    _check_configured()
    errors: List[Tuple[str, str]] = []
    # `timeout` is the gap allowed between chunks, not the whole answer.
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
        for m in _models(model):
            body = _body(m, messages, max_tokens, temperature, False)
            body["stream"] = True
            started = False
            try:
                async with client.stream("POST", OPENROUTER_URL, headers=_headers(title), json=body) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        _, err, reason = _parse(resp)
                        raise _StreamError(err or f"HTTP {resp.status_code}", reason)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue  # blank lines and ": OPENROUTER PROCESSING" keep-alives
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        err = chunk.get("error") if isinstance(chunk, dict) else None
                        if err:
                            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                            code = err.get("code") if isinstance(err, dict) else None
                            raise _StreamError(msg, _reason_for(code if isinstance(code, int) else 0, msg))
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        piece = _content_text((choices[0].get("delta") or {}).get("content"))
                        if piece:
                            if not started:
                                started = True
                                yield "model", m
                            yield "delta", piece
                if started:
                    return
                errors.append(("unavailable", f"{m}: empty response")); _log(title, m, "empty response")
                continue
            except httpx.TimeoutException:
                errors.append(("timeout", f"{m}: timed out")); _log(title, m, "timed out")
            except httpx.HTTPError as e:
                errors.append(("unavailable", f"{m}: {e}")); _log(title, m, str(e))
            except _StreamError as e:
                errors.append((e.reason, f"{m}: {e}")); _log(title, m, f"{e.reason} — {e}")
                if e.reason in _FATAL:
                    break
            if started:
                break  # part of the answer is already on screen: another model can't continue it
    raise _combined(errors)


def _citations(resp: httpx.Response) -> List[Dict[str, str]]:
    """The web pages OpenRouter's search plugin actually returned (url_citation annotations)."""
    try:
        msg = (resp.json().get("choices") or [{}])[0].get("message") or {}
    except (ValueError, AttributeError, IndexError):
        return []
    out = []
    for a in msg.get("annotations") or []:
        c = a.get("url_citation") if isinstance(a, dict) else None
        if isinstance(c, dict) and c.get("url"):
            out.append({"url": str(c["url"]), "title": str(c.get("title") or ""), "content": str(c.get("content") or "")})
    return out


async def web_search(prompt: str, *, model: Optional[str] = None, max_results: int = 8, max_tokens: int = 1500,
                     timeout: float = 75.0, title: str = "Event Scout Web Search") -> Tuple[str, List[Dict[str, str]]]:
    """Chat completion with OpenRouter's live web search plugin.

    Returns (model_text, citations). `citations` are the real result pages ({url, title, content});
    callers should only trust URLs that appear there, never ones that only appear in the text.
    Raises AIError like chat().
    """
    _check_configured()
    errors: List[Tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for m in _models(model):
            body = _body(m, [{"role": "user", "content": prompt}], max_tokens, 0.1, False)
            body["plugins"] = [{"id": "web", "max_results": max(1, min(max_results, 10))}]
            try:
                resp = await client.post(OPENROUTER_URL, headers=_headers(title), json=body)
            except httpx.TimeoutException:
                errors.append(("timeout", f"{m}: timed out")); _log(title, m, "timed out"); continue
            except httpx.HTTPError as e:
                errors.append(("unavailable", f"{m}: {e}")); _log(title, m, str(e)); continue
            text, err, reason = _parse(resp)
            if text is not None:
                return text, _citations(resp)
            errors.append((reason, f"{m}: {err}")); _log(title, m, f"{reason} — {err}")
            if reason in _FATAL:
                break
    raise _combined(errors)


def _combined(errors: Sequence[Tuple[str, str]]) -> AIError:
    reasons = [r for r, _ in errors]
    for r in ("bad_key", "no_credits", "rate_limited"):
        if r in reasons:
            reason = r
            break
    else:
        reason = "timeout" if reasons and all(r == "timeout" for r in reasons) else "unavailable"
    return AIError("; ".join(msg for _, msg in errors)[:600] or "no model answered", reason)


# ---- content parts ----

def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(data: bytes, mime: str = "image/jpeg") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def file_part(filename: str, data: bytes, mime: str = "application/pdf") -> dict:
    """PDFs (OpenRouter parses them for models without native PDF input)."""
    return {"type": "file", "file": {"filename": filename, "file_data": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def audio_part(data: bytes, fmt: str) -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}}


def parse_json(text: str) -> dict:
    """JSON object from a model reply (tolerates ```json fences and chatter around it)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def status() -> dict:
    """Safe to show in /health: which models are used, never the key."""
    return {
        "provider": "openrouter",
        "configured": ai_configured,
        "model": DEFAULT_MODEL,
        "fallbacks": FALLBACK_MODELS,
        "vision_model": VISION_MODEL,
        "audio_model": AUDIO_MODEL,
        "chat_model": CHAT_MODEL,
    }
