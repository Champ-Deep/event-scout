"""
The Contact read model: every shape a contact row is shown in, built in one place.

    format_contact(c)          the full API shape (camelCase) — the base of the app and console views
    app_view(view)             what the rep's app may see (the console's private adminNotes removed)
    await console_scores(...)  the admin console's view: score* = the ADMIN score, repScore* = the rep's,
                               level = admin level, repLevel = the rep's level
    lite(view)                 list rows without the photo / avatar / score write-up
    admin_score_out(row)       an admin score (admin_contact_scores) as the console reads it
    index_view(c)              the search-index (FAISS) entry — the only builder of it
    chat_card(meta) / chat_summary(...)    the assistant's contact cards / workspace summary rows
    export_view / export_csv_row           the rep's own export (/export_contacts/)
    admin_export_row(c, owner)             the console's CSV export row
    crm_view(c, owner)                     what crm.make_lead turns into a CRM lead
    dashboard_top(c)                       the rep dashboard's top-contacts entry

Old OCR / legacy rows store "N/A" for a missing field; every view reports those as missing (present()).
Score words come from score_levels only.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from database import AdminContactScoreDB, ContactDB
from score_levels import band_for, level_for, stage_for

PRESENT_BLANKS = ("N/A", "NA", "NONE", "NULL")


def present(value: Optional[str]) -> Optional[str]:
    """OCR/legacy rows store "N/A" for missing fields; report those as missing (None)."""
    if value is None:
        return None
    value = str(value).strip()
    return None if not value or value.upper() in PRESENT_BLANKS else value


def aware(dt: Optional[datetime]) -> Optional[datetime]:
    # SQLite returns naive datetimes even for timezone=True columns.
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def iso(dt) -> Optional[str]:
    if not dt:
        return None
    if isinstance(dt, str):
        return dt
    return aware(dt).isoformat()


def split_name(name: Optional[str]) -> tuple:
    """The stored one-column name → (first, last): the first word, then the rest."""
    parts = (name or "").strip().split()
    return (parts[0] if parts else ""), " ".join(parts[1:])


def generate_svg_avatar(name: str) -> str:
    colors = [
        ("%23059669", "%2334d399", "%23064e3b"),
        ("%232563eb", "%2360a5fa", "%231e3a8a"),
        ("%237c3aed", "%23a78bfa", "%234c1d95"),
        ("%23d97706", "%23fbbf24", "%2378350f"),
        ("%230284c7", "%2338bdf8", "%230c4a6e"),
        ("%23db2777", "%23f472b6", "%23831843"),
        ("%230d9488", "%232dd4bf", "%23134e4a"),
    ]
    clean = (name or "Lead").strip()
    idx = sum(ord(c) for c in clean) % len(colors)
    bg1, bg2, dark = colors[idx]
    parts = clean.split()
    initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()
    return f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80' width='80' height='80'><defs><linearGradient id='g_{idx}' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='{bg1}'/><stop offset='100%' stop-color='{bg2}'/></linearGradient></defs><rect width='80' height='80' rx='40' fill='url(%23g_{idx})'/><circle cx='40' cy='31' r='13' fill='white' fill-opacity='0.9'/><path d='M18 69 C18 52, 28 47, 40 47 C52 47, 62 52, 62 69 Z' fill='white' fill-opacity='0.9'/><text x='40' y='35' text-anchor='middle' font-family='sans-serif' font-size='9' font-weight='800' fill='{dark}'>{initials}</text></svg>"


def photo_url(c) -> Optional[str]:
    """A real picture (scanned card thumbnail or one the user set), else None."""
    photo = getattr(c, "photo_base64", None)
    if not photo:
        return None
    return photo if photo.startswith("http") or photo.startswith("data:") else f"data:image/jpeg;base64,{photo}"


def _score_meta(breakdown: Optional[dict]) -> Optional[dict]:
    m = (breakdown or {}).get("_meta")
    return {"runs": m.get("runs"), "spread": m.get("spread"), "model": m.get("model"), "scoredAt": m.get("scored_at")} if m else None


def format_contact(c: ContactDB) -> dict:
    """Format ContactDB model to standard EventScout frontend Contact type.

    Unscored contacts are reported as unscored (score/scoreBand/level = None) — never given a
    placeholder score — so the apps can show "Not scored" instead of a fake number.
    """
    score = c.lead_score  # None until the contact is actually scored
    full_name = (c.name or "").strip()
    first, last = split_name(full_name)

    # always with "+00:00": SQLite hands back naive UTC, which a plain `new Date()` (the mobile app)
    # would read as local time
    created_iso = aware(c.created_at).isoformat() if c.created_at else datetime.now(timezone.utc).isoformat()
    updated = getattr(c, "updated_at", None)
    updated_iso = aware(updated).isoformat() if updated else created_iso

    # photoUrl: a real picture, else None. avatarUrl: that picture, or initials from the contact's own name.
    photo = photo_url(c)
    breakdown = getattr(c, "lead_score_breakdown", None) or {}

    return {
        "id": str(c.id),
        "userId": str(c.user_id) if getattr(c, "user_id", None) else None,
        "eventId": getattr(c, "event_name", "") or "",  # event the contact was scanned at, if any
        "firstName": first or "Unknown",
        "lastName": last or None,
        "company": present(c.company_name),
        "title": c.job_title or None,
        "email": present(c.email),
        "phone": present(c.phone),
        "linkedin": present(getattr(c, "linkedin", None)),
        "notes": c.notes or None,
        "adminNotes": getattr(c, "admin_notes", None) or None,  # admin-console annotations
        "score": score,
        "scoreBand": band_for(score),
        "level": level_for(score),  # hot / warm / cool / cold (score_levels), None = unscored
        # Why it got that score (only when scored): 5 dimensions of 0-20, the AI's reasons, next steps.
        "scoreReasoning": (getattr(c, "lead_score_reasoning", None) or None) if score is not None else None,
        "scoreBreakdown": ({k: v for k, v in breakdown.items() if not str(k).startswith("_")} or None) if score is not None else None,
        # how it was scored: runs (answers combined), spread (how far they were apart), model, scoredAt
        "scoreMeta": _score_meta(breakdown) if score is not None else None,
        "scoreActions": (getattr(c, "lead_recommended_actions", None) or None) if score is not None else None,
        "stage": stage_for(score),
        "source": c.source or "scan",
        "scannedAt": created_iso,
        "createdAt": created_iso,
        "updatedAt": updated_iso,
        "avatarUrl": photo or generate_svg_avatar(full_name),
        "photoUrl": photo,
        "version": 1,
    }


def app_view(contact: dict) -> dict:
    """A formatted contact as the rep's app may see it: the admin console's notes are private to admins."""
    return {k: v for k, v in contact.items() if k != "adminNotes"}


# ---------------------------------------------------------------- the admin console's view

REP_SCORE_FIELDS = ("score", "scoreBand", "level", "scoreReasoning", "scoreBreakdown", "scoreMeta", "scoreActions")


def admin_score_out(row: Optional[AdminContactScoreDB]) -> Optional[dict]:
    """An admin score as the admin console reads it (same shape as the contact's score fields)."""
    if row is None:
        return None
    meta = (row.breakdown or {}).get("_meta") or {}
    return {
        "score": row.score,
        "scoreBand": band_for(row.score),
        "level": level_for(row.score),
        "temperature": row.temperature,
        "reasoning": row.reasoning or None,
        "breakdown": {k: v for k, v in (row.breakdown or {}).items() if not str(k).startswith("_")} or None,
        "actions": row.recommended_actions or None,
        "meta": {"runs": meta.get("runs"), "spread": meta.get("spread"), "model": meta.get("model"),
                 "scoredAt": meta.get("scored_at") or (row.scored_at.isoformat() if row.scored_at else None)},
    }


async def console_scores(session, me: Optional[uuid.UUID], contacts: List[dict]) -> List[dict]:
    """Admin console view of formatted contacts: `score*` / `level` become the ADMIN's own score
    (admin_contact_scores, made against the admin sales profile), and the rep's moves to `repScore*` /
    `repLevel` (read only). So the console ranks, sorts, filters and pitches by the admin's needs while each
    rep's score stays theirs. Signed-in app calls (me set) keep the rep's own score and never get the
    admin's private notes."""
    if me is not None:
        return [app_view(c) for c in contacts]
    if not contacts:
        return contacts
    ids = []
    for c in contacts:
        try:
            ids.append(uuid.UUID(str(c.get("id"))))
        except (ValueError, TypeError):
            pass
    rows = {str(r.contact_id): r for r in (await session.execute(
        select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id.in_(ids)))).scalars().all()} if ids else {}
    out = []
    for c in contacts:
        a = admin_score_out(rows.get(str(c.get("id"))))
        view = {**c, **{"rep" + k[0].upper() + k[1:]: c.get(k) for k in REP_SCORE_FIELDS}}
        view.update({
            "score": a["score"] if a else None, "scoreBand": a["scoreBand"] if a else None, "level": a["level"] if a else None,
            "scoreReasoning": a["reasoning"] if a else None, "scoreBreakdown": a["breakdown"] if a else None,
            "scoreMeta": a["meta"] if a else None, "scoreActions": a["actions"] if a else None,
            "stage": stage_for(a["score"] if a else None),
        })
        out.append(view)
    return out


async def console_contact(session, me: Optional[uuid.UUID], c: ContactDB) -> dict:
    """format_contact for one contact, with the admin console's score view (see console_scores)."""
    return (await console_scores(session, me, [format_contact(c)]))[0]


async def console_matches(session, me: Optional[uuid.UUID], matches: list) -> List[dict]:
    """[(contact, reasons)] → [{contact, reasons}] with the admin console's score view."""
    views = await console_scores(session, me, [format_contact(c) for c, _ in matches])
    return [{"contact": v, "reasons": r} for v, (_, r) in zip(views, matches)]


# Heavy per-row fields a list doesn't need (lite=1): the photo itself, the SVG avatar and the score write-up.
_LITE_DROP = ("scoreReasoning", "scoreBreakdown", "scoreActions", "scoreMeta",
              "repScoreReasoning", "repScoreBreakdown", "repScoreActions", "repScoreMeta")


def lite(view: dict) -> dict:
    out = {k: v for k, v in view.items() if k not in _LITE_DROP}
    out["hasPhoto"] = bool(view.get("photoUrl"))
    out["photoUrl"] = None
    out["avatarUrl"] = None
    return out


# ---------------------------------------------------------------- search index & the assistant

def index_view(c: ContactDB) -> dict:
    """The search-index (FAISS) entry for a contact — the ONE builder (the chat reads these back as
    retrieved contacts; missing fields are "N/A" there, as the assistant's prompts expect)."""
    return {
        "id": str(c.id),
        "name": c.name,
        "email": c.email or "N/A",
        "phone": c.phone or "N/A",
        "linkedin": c.linkedin or "N/A",
        "company_name": c.company_name or "N/A",
        "job_title": c.job_title or "",
        "notes": c.notes or "",
        "links": c.links or [],
        "source": c.source or "manual",
        "linkedin_source": c.linkedin_source or "card",
        "lead_score": c.lead_score,
        "lead_temperature": c.lead_temperature,
        "lead_score_reasoning": c.lead_score_reasoning or "",
        "lead_score_breakdown": c.lead_score_breakdown or {},
        "lead_recommended_actions": c.lead_recommended_actions or [],
        "created_at": c.created_at.isoformat() if c.created_at else "",
    }


def chat_card(meta: dict) -> dict:
    """A retrieved index entry as the assistant's answer shows it (contact chips / cards)."""
    return {
        "id": meta.get("id", ""),
        "name": meta.get("name", "N/A"),
        "email": meta.get("email", "N/A"),
        "phone": meta.get("phone", "N/A"),
        "linkedin": meta.get("linkedin", "N/A"),
        "company_name": meta.get("company_name", "N/A"),
        "lead_score": meta.get("lead_score"),
        "lead_temperature": meta.get("lead_temperature"),
        "lead_score_reasoning": meta.get("lead_score_reasoning", ""),
        "lead_recommended_actions": meta.get("lead_recommended_actions", []),
        "photo_base64": meta.get("photo_base64", ""),
        "source": meta.get("source", "manual"),
        "notes": (meta.get("notes", "") or "")[:200],
    }


def chat_summary(c: ContactDB, score: Optional[int], temperature: Optional[str]) -> dict:
    """One row of the "all contacts" overview the assistant gets (the rep's score, or the console's)."""
    return {"name": c.name, "company_name": c.company_name or "N/A", "lead_score": score, "lead_temperature": temperature}


def dashboard_top(c: ContactDB) -> dict:
    """A top-scored contact on the rep's app dashboard."""
    return {"id": str(c.id), "name": c.name, "lead_score": c.lead_score, "company_name": c.company_name}


# ---------------------------------------------------------------- exports & CRM

def _val(v):  # null instead of "N/A" / "" filler
    return None if (v is None or str(v).strip() in ("", "N/A")) else v


def _blank(v):  # "" instead of "N/A" filler cells
    return "" if (v is None or str(v).strip() in ("", "N/A")) else v


def export_view(c: ContactDB, pipeline=None) -> dict:
    """The rep's JSON export (eventscout.contacts.v1); `pipeline` = their own latest research run."""
    return {
        "id": str(c.id),
        "name": c.name,
        "title": _val(c.job_title),
        "company": _val(c.company_name),
        "email": _val(c.email),
        "phone": _val(c.phone),
        "linkedin": _val(c.linkedin),
        "event": _val(getattr(c, "event_name", None)),
        "notes": _val(c.notes),
        "links": c.links or [],
        "source": c.source or "manual",
        "lead_score": c.lead_score,
        "lead_temperature": _val(c.lead_temperature),
        "created_at": iso(c.created_at),
        "updated_at": iso(getattr(c, "updated_at", None)),
        "research_summary": _val(pipeline.research_summary) if pipeline else None,
        "pitch_angle": _val(pipeline.pitch_angle) if pipeline else None,
        "pitch_email_subject": _val(pipeline.pitch_email_subject) if pipeline else None,
        "gamma_deck_url": _val(getattr(pipeline, "gamma_deck_url", None)) if pipeline else None,
    }


EXPORT_CSV_HEADER = ["Name", "Title", "Email", "Phone", "LinkedIn", "Company", "Event",
                     "Notes", "Source", "Lead Score", "Lead Temperature", "Added On",
                     "Research Summary", "Pitch Email Subject", "Gamma Deck URL"]


def export_csv_row(c: ContactDB, pipeline=None) -> list:
    """The rep's CSV export row (EXPORT_CSV_HEADER)."""
    return [
        c.name, _blank(c.job_title), _blank(c.email), _blank(c.phone),
        _blank(c.linkedin), _blank(c.company_name),
        getattr(c, "event_name", "") or "",
        (c.notes or "").replace("\n", " "), c.source or "manual",
        "" if c.lead_score is None else c.lead_score, c.lead_temperature or "",
        iso(c.created_at) or "",
        (pipeline.research_summary or "").replace("\n", " ") if pipeline else "",
        (pipeline.pitch_email_subject or "") if pipeline else "",
        (getattr(pipeline, "gamma_deck_url", "") or "") if pipeline else "",
    ]


# Score / Level = the console's (admin) score, as in the contacts grid; Rep score = the rep's own, for reference
ADMIN_EXPORT_HEADER = ["ID", "Name", "Title", "Company", "Email", "Phone", "LinkedIn", "Event", "Owner", "Score", "Level", "Rep score",
                       "Source", "Notes", "Added On"]


def admin_export_row(c: ContactDB, owner: str) -> list:
    """The console's CSV export row (ADMIN_EXPORT_HEADER). `c.console_score` = the admin score
    (set by api_routes.admin_export_contacts)."""
    return [ADMIN_EXPORT_COLUMNS[h](c, owner) for h in ADMIN_EXPORT_HEADER]


def website_of(c: ContactDB) -> str:
    """First link the scanner labelled Website, else empty."""
    for link in c.links or []:
        if isinstance(link, dict) and link.get("label") == "Website" and link.get("url"):
            return link["url"]
    return ""


def _console_score(c):
    return getattr(c, "console_score", None)


# Every column an admin can put in a contact export, by its header. ADMIN_EXPORT_HEADER is the default
# order; `columns=` on GET /api/admin/export picks and orders from this registry.
ADMIN_EXPORT_COLUMNS = {
    "ID": lambda c, o: str(c.id),
    "Name": lambda c, o: c.name,
    "Title": lambda c, o: present(c.job_title) or "",
    "Company": lambda c, o: present(c.company_name) or "",
    "Email": lambda c, o: present(c.email) or "",
    "Phone": lambda c, o: present(c.phone) or "",
    "LinkedIn": lambda c, o: present(c.linkedin) or "",
    "Website": lambda c, o: website_of(c),
    "Event": lambda c, o: c.event_name or "",
    "Owner": lambda c, o: o or "",
    "Score": lambda c, o: "" if _console_score(c) is None else _console_score(c),
    "Level": lambda c, o: level_for(_console_score(c)) or "",
    "Rep score": lambda c, o: "" if c.lead_score is None else c.lead_score,
    "Source": lambda c, o: c.source or "",
    "Notes": lambda c, o: c.notes or "",
    "Added On": lambda c, o: iso(c.created_at) or "",
}

# Built-in preset: headers Harbinger's contacts import auto-maps (Name, Title, Company, Email, LinkedIn, Website -> domain).
HARBINGER_PRESET = {"name": "ChampHarbinger", "builtin": True, "filters": {},
                    "columns": ["Name", "Title", "Company", "Email", "Phone", "LinkedIn", "Website", "Event", "Notes", "Score", "Level", "Source"]}
PRESET_FILTER_KEYS = ("event", "userId", "minScore", "listId")


def parse_export_columns(raw: Optional[str]) -> List[str]:
    """`columns=` query -> ordered headers. None/empty -> the default header. Unknown header -> ValueError."""
    if not raw or not raw.strip():
        return list(ADMIN_EXPORT_HEADER)
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in cols if x not in ADMIN_EXPORT_COLUMNS]
    if bad or not cols:
        raise ValueError(f"Unknown export columns: {', '.join(bad) or '(none given)'}. Known: {', '.join(ADMIN_EXPORT_COLUMNS)}")
    return list(dict.fromkeys(cols))  # keep order, drop repeats


def validate_presets(presets) -> List[dict]:
    """Admin's saved presets as stored: [{name, columns, filters}]. Raises ValueError on a bad one."""
    if not isinstance(presets, list) or len(presets) > 50:
        raise ValueError("presets must be a list of at most 50")
    out, seen = [], set()
    for p in presets:
        name = str((p or {}).get("name", "")).strip()[:60]
        if not name or name.lower() in seen or name == HARBINGER_PRESET["name"]:
            raise ValueError(f"preset name missing, repeated or reserved: {name!r}")
        seen.add(name.lower())
        cols = parse_export_columns(",".join(str(c) for c in (p.get("columns") or [])))
        filters = {k: v for k, v in (p.get("filters") or {}).items() if k in PRESET_FILTER_KEYS and v not in (None, "")}
        out.append({"name": name, "columns": cols, "filters": filters})
    return out


def crm_view(c: ContactDB, owner: str) -> Dict[str, Any]:
    """A contact as the CRM push / CRM CSV needs it (crm.make_lead). The score is the admin console's own
    (`console_score`, set by api_routes.admin_export_contacts), else the rep's."""
    score = getattr(c, "console_score", c.lead_score)
    return {
        "id": str(c.id), "name": (c.name or "").strip(),
        "email": present(c.email) or "", "phone": present(c.phone) or "",
        "company": present(c.company_name) or "", "title": present(c.job_title) or "", "linkedin": present(c.linkedin) or "",
        "event": (c.event_name or "").strip(), "score": score, "level": level_for(score) or "",
        "notes": c.notes or "", "owner": owner or "", "source": c.source or "",
    }
