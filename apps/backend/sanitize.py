"""Input sanitization shared by every route (mobile app + admin console).

1) SanitizeJSONMiddleware — every JSON request body (POST / PUT / PATCH) is cleaned before a route sees it:
   - control characters are stripped from every string: NUL, C0 (except tab / newline / CR), DEL, C1, the
     invisible bidi embedding / override / isolate controls ("Trojan Source" text disguise), zero-width
     space and BOM. They break Postgres (NUL), CSVs, logs and AI prompts, and have no use in this app's data.
     Joiners (ZWJ / ZWNJ, used in emoji and Indic / Persian text) and LRM / RLM marks are kept.
   - URL fields (website, url, link, linkedin, href, homepage, webhook…) may not use a script scheme
     (javascript:, vbscript:, data:, file:, blob:) — such a value is refused with 400.
   Passwords, tokens and base64 payloads (photos, files) are passed through untouched.
   The body is only rewritten when something was actually removed.

2) csv_safe / safe_csv_writer — spreadsheet formula-injection guard for every CSV export: a cell that
   starts with = + - @ (or tab / CR) gets a leading apostrophe so Excel / Sheets show it as text instead of
   running it (e.g. a contact named =HYPERLINK(...)). Plain numbers and phone numbers ("+1 415 555 0142",
   "-12.5") are left as they are.
"""
from __future__ import annotations

import csv
import json
import re
from typing import Any, Tuple

from starlette.responses import JSONResponse

# --- strings -------------------------------------------------------------------------------------------
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​‪-‮⁦-⁩﻿]")
# keys whose values are secrets or binary payloads: never altered
_SKIP_KEY = re.compile(r"password|passcode|secret|token|base64|^photo|avatar|image|filedata|signature_?data", re.I)
# keys holding a web address
_URL_KEY = re.compile(r"(^|_)(url|href|link|links|website|homepage|linkedin|webhook)($|_)|url$|website$|Url$|Website$|Link$", re.I)
# keys that may legitimately hold an image data: URI
_IMAGE_KEY = re.compile(r"photo|avatar|image|logo|icon|thumb", re.I)
_BAD_SCHEME = re.compile(r"^(javascript|vbscript|data|file|blob):", re.I)


class UnsafeInput(ValueError):
    pass


def clean_text(value: str) -> str:
    """The string without control / invisible formatting characters (see module doc)."""
    return _CONTROL.sub("", value)


def _url_scheme_ok(value: str) -> bool:
    # browsers ignore whitespace / control characters inside the scheme ("java\tscript:"), so check without them
    squashed = re.sub(r"[\s\x00-\x1f]+", "", value)
    return not _BAD_SCHEME.match(squashed)


def _clean(value: Any, key: str | None) -> Tuple[Any, bool]:
    """(cleaned value, changed?) — raises UnsafeInput for a script link in a URL field."""
    if isinstance(value, str):
        if key and _SKIP_KEY.search(key):
            return value, False
        out = clean_text(value)
        if key and _URL_KEY.search(key) and not _IMAGE_KEY.search(key) and not _url_scheme_ok(out):
            label = {"linkedin": "LinkedIn link", "linkedin_url": "LinkedIn link", "linkedinurl": "LinkedIn link", "url": "link",
                     "links": "link", "href": "link"}.get(key.lower(), re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).replace("_", " ").lower())
            raise UnsafeInput(f"The {label} must be a web address (http:// or https://).")
        return out, out != value
    if isinstance(value, dict):
        changed = False
        out = {}
        for k, v in value.items():
            ck = clean_text(k) if isinstance(k, str) else k
            cv, ch = _clean(v, ck if isinstance(ck, str) else key)
            changed = changed or ch or ck != k
            out[ck] = cv
        return out, changed
    if isinstance(value, list):
        changed = False
        out = []
        for v in value:
            cv, ch = _clean(v, key)  # list items inherit their field's name (e.g. links[] → url checks)
            changed = changed or ch
            out.append(cv)
        return out, changed
    return value, False


def sanitize_payload(data: Any) -> Any:
    """Clean a decoded JSON payload (for code paths that don't go through the middleware)."""
    return _clean(data, None)[0]


class SanitizeJSONMiddleware:
    """Pure ASGI middleware (no buffering of responses, so streaming / SSE are unaffected)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        ctype = ""
        for k, v in scope.get("headers") or []:
            if k == b"content-type":
                ctype = v.decode("latin-1").lower()
                break
        if "json" not in ctype:  # multipart uploads / forms are handled by their routes
            return await self.app(scope, receive, send)

        chunks = []
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)

        new_body = body
        if body.strip():
            try:
                data = json.loads(body)
            except ValueError:
                data = None  # invalid JSON: the route answers 422 as before
            else:
                try:
                    cleaned, changed = _clean(data, None)
                except UnsafeInput as e:
                    return await JSONResponse({"detail": str(e)}, status_code=400)(scope, receive, send)
                if changed:
                    new_body = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")

        if new_body is not body:
            headers = [(k, v) for k, v in scope.get("headers") or [] if k != b"content-length"]
            headers.append((b"content-length", str(len(new_body)).encode("latin-1")))
            scope = dict(scope, headers=headers)

        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": new_body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


# --- CSV exports ---------------------------------------------------------------------------------------
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_PLAIN_NUMBER = re.compile(r"^[+-]?[\d\s().,/-]+$")  # numbers and phone numbers stay as they are


def csv_safe(value: Any) -> Any:
    """A CSV cell that Excel / Sheets will show as text, never run as a formula."""
    if not isinstance(value, str) or not value:
        return value
    value = clean_text(value)
    if value.startswith(_FORMULA_START) and not _PLAIN_NUMBER.match(value):
        return "'" + value
    return value


class _SafeWriter:
    def __init__(self, writer):
        self._w = writer

    def writerow(self, row):
        return self._w.writerow([csv_safe(v) for v in row])

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

    def __getattr__(self, name):
        return getattr(self._w, name)


def safe_csv_writer(stream, *args, **kwargs):
    """csv.writer(stream) whose rows are passed through csv_safe."""
    return _SafeWriter(csv.writer(stream, *args, **kwargs))
