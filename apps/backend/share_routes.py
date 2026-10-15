"""
Contact share links ("anyone with the link can view"), like sharing a notebook.

The owner ticks which fields to include; the link shows only those, live (edits appear,
deleting the contact or revoking the link ends it). Owner routes live under /api and need
the signed-in user; the public page /s/{token} and its vCard need nothing but the token.
"""

import os
import re
import html
import base64
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import ContactDB, ContactShareLinkDB
from api_routes import get_db, request_user_id, load_contact, _vcard_escape
from contact_view import present as _present

share_api_router = APIRouter(prefix="/api", tags=["Contact sharing"])
share_public_router = APIRouter(tags=["Shared contact pages"])

# Name is always shared; these are the optional parts the owner can tick.
SHAREABLE_FIELDS = ["title", "company", "email", "phone", "linkedin", "event", "score", "notes", "photo"]
MAX_ACTIVE_LINKS_PER_CONTACT = 20
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

NO_INDEX_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _base_url(request: Request) -> str:
    # Set PUBLIC_BASE_URL in production so links work outside the local network. Without it, use the
    # address the client called — honouring a proxy's X-Forwarded-* (the dev Metro proxy at
    # :8081/backend, a hosting provider's TLS proxy) so the link isn't 127.0.0.1 / http.
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    h = request.headers
    host = (h.get("x-forwarded-host") or "").split(",")[0].strip()
    if host and re.fullmatch(r"[A-Za-z0-9.\-\[\]:]+", host):
        proto = (h.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
        proto = proto if proto in ("http", "https") else "http"
        prefix = (h.get("x-forwarded-prefix") or "").strip().rstrip("/")
        prefix = prefix if re.fullmatch(r"(/[A-Za-z0-9._\-]+)*", prefix) else ""
        return f"{proto}://{host}{prefix}"
    return str(request.base_url).rstrip("/")


def _link_state(link: ContactShareLinkDB) -> str:
    if link.revoked_at:
        return "revoked"
    exp = _aware(link.expires_at)
    if exp and exp <= datetime.now(timezone.utc):
        return "expired"
    return "active"


def _format_link(link: ContactShareLinkDB, request: Request) -> dict:
    base = _base_url(request)
    return {
        "token": link.token,
        "url": f"{base}/s/{link.token}",
        "vcardUrl": f"{base}/s/{link.token}/contact.vcf",
        "fields": list(link.fields or []),
        "createdAt": _aware(link.created_at).isoformat() if link.created_at else None,
        "expiresAt": _aware(link.expires_at).isoformat() if link.expires_at else None,
        "viewCount": link.view_count or 0,
        "lastViewedAt": _aware(link.last_viewed_at).isoformat() if link.last_viewed_at else None,
        "state": _link_state(link),
    }


def _require_user(me: Optional[uuid.UUID]) -> uuid.UUID:
    if me is None:
        raise HTTPException(status_code=401, detail="Sign in to share contacts")
    return me


class CreateShareLinkPayload(BaseModel):
    fields: List[str] = []
    expiresInDays: Optional[int] = 30  # None = never


@share_api_router.post("/contacts/{contact_id}/share-links")
async def create_share_link(
    contact_id: str,
    payload: CreateShareLinkPayload,
    request: Request,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Create a view-only link showing the contact's name plus the ticked fields."""
    user = _require_user(me)
    contact = await load_contact(session, contact_id, user)

    unknown = [f for f in payload.fields if f not in SHAREABLE_FIELDS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Can't share: {', '.join(unknown)}")
    fields = [f for f in SHAREABLE_FIELDS if f in set(payload.fields)]  # canonical order, no dupes

    days = payload.expiresInDays
    if days is not None and not (1 <= days <= 365):
        raise HTTPException(status_code=400, detail="Links can last 1–365 days, or never expire")

    existing = (await session.execute(
        select(ContactShareLinkDB).where(ContactShareLinkDB.contact_id == contact.id, ContactShareLinkDB.user_id == user)
    )).scalars().all()
    if sum(1 for l in existing if _link_state(l) == "active") >= MAX_ACTIVE_LINKS_PER_CONTACT:
        raise HTTPException(status_code=429, detail="This contact already has 20 active links. Revoke some first.")

    now = datetime.now(timezone.utc)
    link = ContactShareLinkDB(
        token=secrets.token_urlsafe(18),
        contact_id=contact.id,
        user_id=user,
        fields=fields,
        created_at=now,
        expires_at=now + timedelta(days=days) if days else None,
    )
    session.add(link)
    await session.commit()
    return _format_link(link, request)


@share_api_router.get("/contacts/{contact_id}/share-links")
async def list_share_links(
    contact_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """This contact's links that still work (newest first)."""
    user = _require_user(me)
    contact = await load_contact(session, contact_id, user)
    links = (await session.execute(
        select(ContactShareLinkDB)
        .where(ContactShareLinkDB.contact_id == contact.id, ContactShareLinkDB.user_id == user)
        .order_by(ContactShareLinkDB.created_at.desc())
    )).scalars().all()
    return {"links": [_format_link(l, request) for l in links if _link_state(l) == "active"]}


@share_api_router.delete("/contacts/{contact_id}/share-links/{token}")
async def revoke_share_link(
    contact_id: str,
    token: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    me: Optional[uuid.UUID] = Depends(request_user_id),
):
    """Stop a link from working. Anyone opening it afterwards sees 'no longer shared'."""
    user = _require_user(me)
    contact = await load_contact(session, contact_id, user)
    link = (await session.execute(
        select(ContactShareLinkDB).where(
            ContactShareLinkDB.token == token,
            ContactShareLinkDB.contact_id == contact.id,
            ContactShareLinkDB.user_id == user,
        )
    )).scalar_one_or_none()
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    if not link.revoked_at:
        link.revoked_at = datetime.now(timezone.utc)
        await session.commit()
    return {"status": "revoked", **_format_link(link, request)}


# ==================== PUBLIC (token only) ====================

async def _resolve(session: AsyncSession, token: str):
    """(link, contact) for a working link, else raises 404 / 410 with a reason."""
    if not token or len(token) > 64:
        raise HTTPException(status_code=404, detail="not_found")
    link = (await session.execute(select(ContactShareLinkDB).where(ContactShareLinkDB.token == token))).scalar_one_or_none()
    if not link:
        raise HTTPException(status_code=404, detail="not_found")
    state = _link_state(link)
    if state != "active":
        raise HTTPException(status_code=410, detail=state)
    contact = (await session.execute(select(ContactDB).where(ContactDB.id == link.contact_id))).scalar_one_or_none()
    if not contact or contact.user_id != link.user_id:
        raise HTTPException(status_code=410, detail="deleted")
    return link, contact


def _shared_view(link: ContactShareLinkDB, c: ContactDB) -> dict:
    """Only the name + ticked fields — nothing else leaves the server."""
    f = set(link.fields or [])
    parts = (c.name or "").split()
    out = {
        "name": c.name or "Contact",
        "firstName": parts[0] if parts else "",
        "lastName": " ".join(parts[1:]) or None,
    }
    if "title" in f: out["title"] = c.job_title or None
    if "company" in f: out["company"] = _present(c.company_name)
    if "email" in f: out["email"] = _present(c.email)
    if "phone" in f: out["phone"] = _present(c.phone)
    if "linkedin" in f: out["linkedin"] = _present(c.linkedin)
    if "event" in f: out["event"] = c.event_name or None
    if "score" in f:
        out["score"] = c.lead_score
        out["temperature"] = c.lead_temperature
    if "notes" in f: out["notes"] = (c.notes or "").strip() or None
    if "photo" in f and c.photo_base64:
        out["photo"] = c.photo_base64 if c.photo_base64.startswith(("data:", "http")) else f"data:image/jpeg;base64,{c.photo_base64}"
    out["expiresAt"] = _aware(link.expires_at).isoformat() if link.expires_at else None
    return out


@share_public_router.get("/api/shared/{token}")
async def shared_contact_json(token: str, session: AsyncSession = Depends(get_db)):
    link, contact = await _resolve(session, token)
    return _shared_view(link, contact)


@share_public_router.get("/s/{token}/contact.vcf")
async def shared_contact_vcard(token: str, session: AsyncSession = Depends(get_db)):
    try:
        link, contact = await _resolve(session, token)
    except HTTPException as e:
        return Response(status_code=e.status_code, content="This contact is no longer shared.", media_type="text/plain")
    v = _shared_view(link, contact)
    lines = ["BEGIN:VCARD", "VERSION:3.0",
             f"N:{_vcard_escape(v.get('lastName') or '')};{_vcard_escape(v['firstName'])};;;",
             f"FN:{_vcard_escape(v['name'])}"]
    if v.get("company"): lines.append(f"ORG:{_vcard_escape(v['company'])}")
    if v.get("title"): lines.append(f"TITLE:{_vcard_escape(v['title'])}")
    if v.get("phone"): lines.append(f"TEL;TYPE=WORK,VOICE:{_vcard_escape(v['phone'])}")
    if v.get("email"): lines.append(f"EMAIL;TYPE=INTERNET:{_vcard_escape(v['email'])}")
    if v.get("linkedin"): lines.append(f"URL:{_vcard_escape(v['linkedin'])}")
    if v.get("notes"): lines.append(f"NOTE:{_vcard_escape(v['notes'])}")
    photo = v.get("photo") or ""
    if photo.startswith("data:image/jpeg;base64,"):
        lines.append(f"PHOTO;ENCODING=b;TYPE=JPEG:{photo.split(',', 1)[1]}")
    lines.append("END:VCARD")
    safe = "".join(ch for ch in v["name"] if ch.isalnum() or ch in " -_").strip().replace(" ", "_") or "contact"
    return Response(
        content="\r\n".join(lines) + "\r\n",
        media_type="text/vcard; charset=utf-8",
        headers={**NO_INDEX_HEADERS, "Content-Disposition": f'attachment; filename="{safe}.vcf"'},
    )


_PAGE_CSS = """
:root{--brand:#6d28d9;--brand50:#f5f3ff;--brand100:#ede9fe;--ink:#0f172a;--body:#475569;--muted:#94a3b8;--line:#e2e8f0;--bg:#fcfaff}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:460px;margin:0 auto;padding:28px 16px 40px}
.card{background:#fff;border:1px solid var(--line);border-radius:24px;padding:24px 20px;box-shadow:0 10px 30px rgba(76,29,149,.08);position:relative;overflow:hidden;animation:rise .45s cubic-bezier(.2,.8,.2,1)}
.glow{position:absolute;top:-90px;right:-60px;width:220px;height:220px;border-radius:50%;background:var(--brand50)}
.top{position:relative;display:flex;flex-direction:column;align-items:center;text-align:center}
.avatar{width:96px;height:96px;border-radius:50%;background:var(--brand100);color:var(--brand);display:flex;align-items:center;justify-content:center;font-size:34px;font-weight:800;overflow:hidden;border:3px solid #fff;box-shadow:0 6px 18px rgba(109,40,217,.18)}
.avatar img{width:100%;height:100%;object-fit:cover}
h1{font-size:24px;line-height:1.2;margin:14px 0 4px;letter-spacing:-.3px}
.sub{color:var(--body);font-size:15px;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:12px}
.chip{background:var(--brand50);color:var(--brand);font-size:12px;font-weight:700;padding:5px 10px;border-radius:999px}
.rows{position:relative;margin-top:20px;border-top:1px solid var(--line)}
.row{display:flex;align-items:center;gap:12px;padding:13px 2px;border-bottom:1px solid var(--line);color:inherit;text-decoration:none}
.row:last-child{border-bottom:0}
.ic{width:34px;height:34px;border-radius:10px;background:#f8fafc;display:flex;align-items:center;justify-content:center;flex:none}
.lbl{font-size:10px;font-weight:800;letter-spacing:1px;color:var(--muted)}
.val{font-size:15px;word-break:break-word}a.row .val{color:var(--brand);font-weight:600}
.notes{white-space:pre-wrap;font-size:14px;line-height:1.5;color:var(--body);background:#f8fafc;border-radius:14px;padding:12px 14px;margin-top:14px}
.actions{display:flex;gap:8px;margin-top:20px}
.btn{flex:1;display:flex;align-items:center;justify-content:center;gap:8px;padding:14px;border-radius:999px;font-weight:800;font-size:15px;text-decoration:none;transition:transform .12s}
.btn:active{transform:scale(.97)}
.primary{background:var(--brand);color:#fff;box-shadow:0 8px 20px rgba(109,40,217,.28)}
.ghost{background:#fff;color:var(--ink);border:1.5px solid var(--line)}
.foot{text-align:center;color:var(--muted);font-size:12px;margin-top:18px;line-height:1.6}
.state{text-align:center;padding:40px 20px}.state .ic{margin:0 auto 14px;width:64px;height:64px;border-radius:22px;font-size:28px}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
"""

_ICONS = {
    "email": "&#9993;", "phone": "&#9742;", "linkedin": "in", "company": "&#127970;", "title": "&#128188;",
}


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<meta name=robots content='noindex,nofollow'>"
        f"<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><div class=wrap>{body}<p class=foot>Shared with EventScout</p></div></body></html>"
    )
    return HTMLResponse(doc, status_code=status, headers=NO_INDEX_HEADERS)


_STATE_TEXT = {
    "not_found": ("Link not found", "Check the link — it may have been copied incompletely."),
    "revoked": ("No longer shared", "The person who shared this contact turned the link off."),
    "expired": ("This link has expired", "Ask the person who shared it for a new link."),
    "deleted": ("Contact unavailable", "This contact is no longer available."),
}


@share_public_router.get("/s/{token}", response_class=HTMLResponse)
async def shared_contact_page(token: str, session: AsyncSession = Depends(get_db)):
    """The page recipients open: the shared details, plus 'Save contact' (vCard)."""
    try:
        link, contact = await _resolve(session, token)
    except HTTPException as e:
        head, text = _STATE_TEXT.get(str(e.detail), _STATE_TEXT["not_found"])
        return _page(head, f"<div class='card state'><div class=ic>&#128279;</div><h1>{head}</h1><p class=sub>{text}</p></div>", e.status_code)

    link.view_count = (link.view_count or 0) + 1
    link.last_viewed_at = datetime.now(timezone.utc)
    await session.commit()

    v = _shared_view(link, contact)
    e = html.escape
    initials = "".join(p[0] for p in v["name"].split()[:2]).upper() or "?"
    avatar = f"<img alt='' src='{e(v['photo'])}'>" if v.get("photo") else e(initials)
    sub = " · ".join(x for x in (v.get("title"), v.get("company")) if x)
    chips = []
    if v.get("event"): chips.append(f"<span class=chip>Met at {e(v['event'])}</span>")
    if v.get("score") is not None:
        temp = f" · {e(str(v.get('temperature') or '').upper())}" if v.get("temperature") else ""
        chips.append(f"<span class=chip>Lead score {int(v['score'])}/100{temp}</span>")

    rows = []
    def row(key, label, value, href=None):
        inner = f"<span class=ic>{_ICONS.get(key, '&#8226;')}</span><span><div class=lbl>{label}</div><div class=val>{e(value)}</div></span>"
        rows.append(f"<a class=row href='{e(href)}'>{inner}</a>" if href else f"<div class=row>{inner}</div>")
    if v.get("email"): row("email", "EMAIL", v["email"], f"mailto:{v['email']}")
    if v.get("phone"): row("phone", "PHONE", v["phone"], "tel:" + "".join(ch for ch in v["phone"] if ch.isdigit() or ch == "+"))
    if v.get("linkedin"): row("linkedin", "LINKEDIN", v["linkedin"].replace("https://", "").replace("www.", ""), v["linkedin"] if v["linkedin"].startswith("https://") else None)

    notes = f"<div class=notes>{e(v['notes'])}</div>" if v.get("notes") else ""
    exp = ""
    if v.get("expiresAt"):
        exp = f"<br>Link expires {datetime.fromisoformat(v['expiresAt']).strftime('%d %b %Y')}"
    email_btn = f"<a class='btn ghost' href='mailto:{e(v['email'])}'>Email</a>" if v.get("email") else ""
    body = (
        "<div class=card><div class=glow></div><div class=top>"
        f"<div class=avatar>{avatar}</div><h1>{e(v['name'])}</h1>"
        f"{f'<p class=sub>{e(sub)}</p>' if sub else ''}"
        f"{f'<div class=chips>{chr(10).join(chips)}</div>' if chips else ''}"
        "</div>"
        f"{f'<div class=rows>{chr(10).join(rows)}</div>' if rows else ''}{notes}"
        f"<div class=actions><a class='btn primary' href='{e(token)}/contact.vcf'>&#43; Save contact</a>{email_btn}</div>"
        f"</div><p class=foot>Only the details chosen by the sender are shown.{exp}</p>"
    )
    return _page(f"{v['name']} · EventScout", body)
