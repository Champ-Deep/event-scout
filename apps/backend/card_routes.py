"""
"My card": the signed-in user's own contact QR code + a public card page.

The QR holds a vCard, so any phone camera can add you as a contact straight from it (no app,
no internet needed). A second QR / link points to a public page (/c/{token}) with a
"Save contact" button. The token lives in the user's profile and can be reset, which turns the
old link off.
"""

import io
import re
import asyncio
import html
import base64
import secrets
import uuid
from typing import Optional

import qrcode
from qrcode.image.pil import PilImage
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import select, cast, String
from sqlalchemy.ext.asyncio import AsyncSession

import event_identity
from database import UserDB, UserProfileDB
from api_routes import get_db, request_user_id, _vcard_escape
from share_routes import _page, _base_url, NO_INDEX_HEADERS
from business_card import render_business_card

card_api_router = APIRouter(prefix="/api", tags=["My card"])
card_public_router = APIRouter(tags=["Public card pages"])


def qr_data_url(text: str) -> str:
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    buf = io.BytesIO()
    qr.make_image(image_factory=PilImage, fill_color="black", back_color="white").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_vcard(name: str, company=None, title=None, phone=None, email=None, url=None, note=None) -> str:
    parts = (name or "").split()
    first, last = (parts[0] if parts else ""), " ".join(parts[1:])
    lines = ["BEGIN:VCARD", "VERSION:3.0", f"N:{_vcard_escape(last)};{_vcard_escape(first)};;;", f"FN:{_vcard_escape(name)}"]
    if company: lines.append(f"ORG:{_vcard_escape(company)}")
    if title: lines.append(f"TITLE:{_vcard_escape(title)}")
    if phone: lines.append(f"TEL;TYPE=CELL:{_vcard_escape(phone)}")
    if email: lines.append(f"EMAIL;TYPE=INTERNET:{_vcard_escape(email)}")
    if url: lines.append(f"URL:{_vcard_escape(url)}")
    if note: lines.append(f"NOTE:{_vcard_escape(note)}")
    lines.append("END:VCARD")
    return "\n".join(lines)


def _card_fields(user: UserDB, profile: dict) -> dict:
    def clean(v):
        v = (v or "").strip() if isinstance(v, str) else v
        return v or None
    linkedin = clean(profile.get("linkedin_url"))
    if linkedin and not linkedin.lower().startswith("http"):
        linkedin = "https://" + linkedin
    website = clean(profile.get("company_website"))
    if website and not website.lower().startswith("http"):
        website = "https://" + website
    return {
        "name": clean(profile.get("full_name")) or user.name,
        "title": clean(profile.get("job_title")),
        "company": clean(profile.get("company_name")),
        "email": user.email,
        "phone": clean(profile.get("phone")),
        "linkedin": linkedin,
        "website": website,
        "photo": profile.get("avatar") or None,
    }


async def _load(session: AsyncSession, user_id: uuid.UUID):
    user = (await session.execute(select(UserDB).where(UserDB.id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")
    row = (await session.execute(select(UserProfileDB).where(UserProfileDB.user_id == user_id))).scalar_one_or_none()
    return user, row


async def _ensure_token(session: AsyncSession, user_id: uuid.UUID, row: Optional[UserProfileDB], reset: bool = False) -> str:
    data = event_identity.read_profile(row)
    token = data.get("card_token")
    if token and not reset:
        return token
    token = secrets.token_urlsafe(12)
    data["card_token"] = token
    event_identity.write_profile(session, row, user_id, data)
    await session.commit()
    return token


async def _card_payload(session: AsyncSession, request: Request, me: uuid.UUID, reset: bool = False) -> dict:
    user, row = await _load(session, me)
    token = await _ensure_token(session, me, row, reset=reset)
    fields = _card_fields(user, event_identity.read_profile(row))
    url = f"{_base_url(request)}/c/{token}"
    vcard = build_vcard(fields["name"], fields["company"], fields["title"], fields["phone"], fields["email"],
                        fields["linkedin"] or fields["website"])
    return {
        "card": {k: v for k, v in fields.items() if k != "photo"},
        "hasPhoto": bool(fields["photo"]),
        "vcard": vcard,
        "qrDataUrl": qr_data_url(vcard),   # scan with any camera → "Add contact"
        "url": url,                        # public card page with "Save contact"
        "linkQrDataUrl": qr_data_url(url),
    }


@card_api_router.get("/me/card")
async def get_my_card(request: Request, session: AsyncSession = Depends(get_db), me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Your contact card: vCard QR, public page link and its QR (built from your profile)."""
    if me is None:
        raise HTTPException(status_code=401, detail="Sign in to see your card")
    return await _card_payload(session, request, me)


def _card_png(fields: dict) -> bytes:
    vcard = build_vcard(fields["name"], fields["company"], fields["title"], fields["phone"], fields["email"],
                        fields["linkedin"] or fields["website"])
    return render_business_card(fields["name"], fields["title"], fields["company"], fields["linkedin"],
                                fields["phone"], fields["email"], vcard)


def _card_filename(name: str) -> str:
    safe = "".join(ch for ch in (name or "") if ch.isalnum() or ch in " -_").strip().replace(" ", "-").lower()
    return f"{safe or 'my'}-business-card.png"


@card_api_router.get("/me/business-card")
async def get_my_business_card(session: AsyncSession = Depends(get_db), me: Optional[uuid.UUID] = Depends(request_user_id)):
    """Your premium business card as a PNG (2100×1200, 600 dpi): QR on the left, details on the right."""
    if me is None:
        raise HTTPException(status_code=401, detail="Sign in to get your business card")
    user, row = await _load(session, me)
    fields = _card_fields(user, event_identity.read_profile(row))
    png = _card_png(fields)
    return {
        "pngDataUrl": "data:image/png;base64," + base64.b64encode(png).decode(),
        "filename": _card_filename(fields["name"]),
        "width": 2100, "height": 1200,
        # what's printed on it, so the app can say what's missing
        "printed": {k: bool(fields.get(k)) for k in ("name", "title", "company", "linkedin", "phone", "email")},
    }


@card_api_router.post("/me/card/reset-link")
async def reset_my_card_link(request: Request, session: AsyncSession = Depends(get_db), me: Optional[uuid.UUID] = Depends(request_user_id)):
    """New card link; the old one stops working (the vCard QR itself never expires)."""
    if me is None:
        raise HTTPException(status_code=401, detail="Sign in to manage your card")
    return await _card_payload(session, request, me, reset=True)


async def _by_token(session: AsyncSession, token: str):
    if not token or len(token) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        return None
    # public, unauthenticated page: narrow it down in the database instead of loading every profile
    rows = (await session.execute(
        select(UserProfileDB).where(cast(UserProfileDB.profile_data, String).contains(token)).limit(5)
    )).scalars().all()
    row = next((r for r in rows if event_identity.read_profile(r).get("card_token") == token), None)
    if not row:
        return None
    user = (await session.execute(select(UserDB).where(UserDB.id == row.user_id))).scalar_one_or_none()
    return (user, row) if user else None


@card_public_router.get("/c/{token}/contact.vcf")
async def my_card_vcard(token: str, session: AsyncSession = Depends(get_db)):
    found = await _by_token(session, token)
    if not found:
        return Response(status_code=404, content="This card link isn't active.", media_type="text/plain")
    user, row = found
    f = _card_fields(user, event_identity.read_profile(row))
    vcard = build_vcard(f["name"], f["company"], f["title"], f["phone"], f["email"], f["linkedin"] or f["website"])
    photo = f["photo"] or ""
    if photo.startswith("data:image/jpeg;base64,"):
        vcard = vcard.replace("END:VCARD", f"PHOTO;ENCODING=b;TYPE=JPEG:{photo.split(',', 1)[1]}\nEND:VCARD")
    safe = "".join(ch for ch in f["name"] if ch.isalnum() or ch in " -_").strip().replace(" ", "_") or "contact"
    return Response(content=vcard.replace("\n", "\r\n") + "\r\n", media_type="text/vcard; charset=utf-8",
                    headers={**NO_INDEX_HEADERS, "Content-Disposition": f'attachment; filename="{safe}.vcf"'})


@card_public_router.get("/c/{token}/card.png")
async def my_card_png(token: str, session: AsyncSession = Depends(get_db)):
    """Download the business card image from the public card page."""
    found = await _by_token(session, token)
    if not found:
        return Response(status_code=404, content="This card link isn't active.", media_type="text/plain")
    user, row = found
    fields = _card_fields(user, event_identity.read_profile(row))
    png = await asyncio.to_thread(_card_png, fields)  # Pillow render: keep it off the event loop
    return Response(content=png, media_type="image/png",
                    headers={**NO_INDEX_HEADERS, "Content-Disposition": f'attachment; filename="{_card_filename(fields["name"])}"'})


@card_public_router.get("/c/{token}", response_class=HTMLResponse)
async def my_card_page(token: str, session: AsyncSession = Depends(get_db)):
    found = await _by_token(session, token)
    if not found:
        return _page("Card not found", "<div class='card state'><div class=ic>&#128279;</div><h1>Card not available</h1>"
                     "<p class=sub>This card link was turned off or never existed.</p></div>", 404)
    user, row = found
    f = _card_fields(user, event_identity.read_profile(row))
    e = html.escape
    initials = "".join(p[0] for p in f["name"].split()[:2]).upper() or "?"
    avatar = f"<img alt='' src='{e(f['photo'])}'>" if f["photo"] else e(initials)
    sub = " · ".join(x for x in (f["title"], f["company"]) if x)
    rows = []
    def row_html(label, value, href):
        rows.append(f"<a class=row href='{e(href)}'><span class=ic>&#8226;</span><span><div class=lbl>{label}</div><div class=val>{e(value)}</div></span></a>")
    if f["email"]: row_html("EMAIL", f["email"], f"mailto:{f['email']}")
    if f["phone"]: row_html("PHONE", f["phone"], "tel:" + "".join(ch for ch in f["phone"] if ch.isdigit() or ch == "+"))
    if f["linkedin"]: row_html("LINKEDIN", f["linkedin"].replace("https://", "").replace("www.", ""), f["linkedin"])
    if f["website"]: row_html("WEBSITE", f["website"].replace("https://", ""), f["website"])
    body = (
        "<div class=card><div class=glow></div><div class=top>"
        f"<div class=avatar>{avatar}</div><h1>{e(f['name'])}</h1>"
        f"{f'<p class=sub>{e(sub)}</p>' if sub else ''}</div>"
        f"{f'<div class=rows>{chr(10).join(rows)}</div>' if rows else ''}"
        f"<div class=actions><a class='btn primary' href='{e(token)}/contact.vcf'>&#43; Save contact</a>"
        f"<a class='btn ghost' href='{e(token)}/card.png'>Business card</a></div>"
        "</div>"
    )
    return _page(f"{f['name']} · EventScout", body)
