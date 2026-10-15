"""
The Contact write module: every contact create, update, delete, restore and merge goes through here.

    await create_contact(session, owner, fields, strict=...)      → Saved
    await update_contact(session, contact, fields, strict=...)    → Saved   (raises EditConflict when strict)
    await delete_contact(session, contact) / delete_many(...)     → Recently deleted (30 days)
    await restore_contact(session, deleted_row, owner)            → ContactDB
    await merge_contacts(session, primary, secondary)             → the archived duplicate
    await save(session, contact, event_type)                      → commit a trusted internal change (admin
                                                                    note, LinkedIn attach, pipeline output, scores)

`fields` use the API names the apps send: firstName / lastName (or name), company, title, email, phone,
notes, linkedin, eventId, score, photo, links, source.

It owns the field rules, the edit-conflict rules and the side effects that ALWAYS follow a write:
the search index (effects.index_update / index_delete) and the live event (effects.live, which also
kicks the industry classifier). The adapters are wired at startup (api_routes / app.py); tests pass fakes.

One rule each, where callers used to differ:
  strict (online: POST / PATCH /api/contacts)   a bad value is refused: 400 (photo, LinkedIn, score,
                                                 non-text notes), 413 for notes over NOTES_MAX_CHARS,
                                                 400 for a nameless new contact, 409 for edit conflicts
  tolerant (offline sync push, card / QR scans,  nothing is lost: a bad photo / LinkedIn / score is dropped
           the n8n enrichment webhook)           (the field keeps its old value), notes over the limit are
                                                 shortened AND reported (NOTES_SHORTENED), a nameless capture
                                                 is saved as "Unnamed contact" (an offline capture must never be
                                                 lost), a field changed elsewhere since the edit started keeps
                                                 the newer value (notes are merged) and is reported
  missing values are stored as NULL — never the old "N/A" placeholder (legacy "N/A" rows still read as missing)
  the live event: contact_scanned (create), score_updated (the score changed), contact_updated (any other
  edit), contact_deleted / contact_restored / contact_merged — with one payload shape (live_payload)
"""

import asyncio
import base64
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from sqlalchemy import DateTime, LargeBinary, delete, select, update
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

import event_identity
from contact_view import aware, format_contact, index_view, present, split_name
from database import (ContactDB, ContactFileDB, ContactListMemberDB, ContactPipelineDB, DeletedContactDB,
                      SharedContactDB)
from score_levels import band_for, temperature_for

NOTES_MAX_CHARS = 20000  # keep in sync with NOTES_MAX in the mobile NotesCard
NOTES_SHORTENED = "Notes were longer than 20,000 characters and were shortened"
UNNAMED = "Unnamed contact"
EVENT_NAME_MAX = 120
RECENTLY_DELETED_DAYS = 30

# Contact photos: accept a data URI or bare base64 image, store a square-ish JPEG thumbnail.
MAX_PHOTO_BYTES = 8 * 1024 * 1024
PHOTO_MAX_SIDE = 512


# ---------------------------------------------------------------- side effects (adapters)

class Effects:
    """What follows every write. index_update(user_id, index_view dict) / index_delete(user_id, contact_id)
    are blocking (run in a thread); live(event_type, data) and webhook(contact) are async."""

    def __init__(self):
        self.index_update = None
        self.index_delete = None
        self.live = None
        self.webhook = None


effects = Effects()


def live_payload(c: ContactDB) -> Dict[str, Any]:
    """The one shape every contact live event carries."""
    return {"contactId": str(c.id), "name": c.name, "company": present(c.company_name),
            "score": c.lead_score, "scoreBand": band_for(c.lead_score), "source": c.source}


async def sync_index(c: ContactDB, deleted: bool = False) -> None:
    """Best-effort search index (FAISS) update; a failure must never fail the request."""
    try:
        if deleted and effects.index_delete:
            await asyncio.to_thread(effects.index_delete, str(c.user_id), str(c.id))
        elif not deleted and effects.index_update:
            await asyncio.to_thread(effects.index_update, str(c.user_id), index_view(c))
    except Exception as e:
        print(f"[INDEX] Could not sync contact {c.id}: {e}")


async def announce(event_type: Optional[str], data: Dict[str, Any]) -> None:
    """Tell the admin console's live feed (never fails the request that changed the contact)."""
    if not event_type or not effects.live:
        return
    try:
        await effects.live(event_type, data)
    except Exception as e:
        print(f"[LIVE] couldn't send {event_type}: {e}")


async def save(session, contact: ContactDB, event_type: Optional[str] = "contact_updated", touch: bool = True,
               extra: Optional[Dict[str, Any]] = None) -> None:
    """Commit a change already made on `contact` (or a new one added to the session), then the side effects:
    search index, live event (event_type None = the caller announces it, e.g. one event for a batch)."""
    if touch:
        contact.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await sync_index(contact)
    await announce(event_type, {**live_payload(contact), **(extra or {})})


# ---------------------------------------------------------------- field rules

_SKIP = object()  # tolerant mode: drop the field, keep what's stored


def clean_text(value, limit: int = 255) -> Optional[str]:
    """One line, spaces collapsed, cut to `limit`; blank or an "N/A" placeholder → None."""
    if value is None:
        return None
    value = " ".join(str(value).split())[:limit]
    return present(value)


def join_name(first, last) -> str:
    return f"{(first or '').strip()} {(last or '').strip()}".strip()


def clean_event(value) -> str:
    """The event the contact was met at (stored by name, like scans tag it). "" = no event."""
    return " ".join(str(value or "").split())[:EVENT_NAME_MAX]


def normalize_linkedin_url(value) -> Optional[str]:
    """'' / None clears it. Accepts linkedin.com/in/… with or without scheme or 'www.'. Raises 400."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="LinkedIn must be a link")
    v = value.strip()
    if not v or not present(v):
        return None
    if not re.match(r"^https?://", v, re.I):
        v = "https://" + v
    parsed = urlparse(v)
    host = (parsed.hostname or "").lower()
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")) or len(parsed.path.strip("/")) < 3 or len(v) > 300:
        raise HTTPException(status_code=400, detail="That doesn't look like a LinkedIn profile link (linkedin.com/in/…)")
    return f"https://www.linkedin.com{parsed.path.rstrip('/')}"


def process_contact_photo(raw: str) -> str:
    """Validate an uploaded photo and return base64 JPEG (no data: prefix). Raises 400."""
    data = raw.split(",", 1)[1] if raw.startswith("data:") else raw
    try:
        blob = base64.b64decode(data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Photo must be a base64-encoded image")
    if len(blob) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=400, detail="Photo is too large (max 8 MB)")
    try:
        img = Image.open(BytesIO(blob))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file isn't a supported image")
    if img.mode not in ("RGB", "L"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
        img = bg
    img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE), Image.LANCZOS)
    out = BytesIO()
    img.convert("RGB").save(out, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(out.getvalue()).decode("utf-8")


def _tolerant(rule, label: str):
    """Strict → the rule's 400 stands; tolerant → a bad value drops the field (logged)."""
    def run(raw, strict: bool):
        if strict:
            return rule(raw)
        try:
            return rule(raw)
        except HTTPException as err:
            print(f"[CONTACTS] {label} skipped ({err.detail})")
            return _SKIP
    return run


def _photo_rule(raw):
    return process_contact_photo(raw) if raw else None


def _score_rule(raw):
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise HTTPException(status_code=400, detail="score must be a whole number from 0 to 100")
    try:
        score = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="score must be a whole number from 0 to 100")
    if not 0 <= score <= 100:
        raise HTTPException(status_code=400, detail="score must be a whole number from 0 to 100")
    return score


_photo = _tolerant(_photo_rule, "photo")
_linkedin = _tolerant(normalize_linkedin_url, "LinkedIn")
_score = _tolerant(_score_rule, "score")


def _notes(raw, strict: bool, report: "Saved"):
    if raw is not None and not isinstance(raw, str):
        if strict:
            raise HTTPException(status_code=400, detail="notes must be text")
        raw = str(raw)
    notes = (raw or "").rstrip()
    if len(notes) > NOTES_MAX_CHARS:
        if strict:
            raise HTTPException(status_code=413, detail=f"Notes are too long (max {NOTES_MAX_CHARS:,} characters)")
        notes = notes[:NOTES_MAX_CHARS]
        report.notes_shortened = True
    return notes


# (field, column, max length) for the plain one-line text fields
_TEXT_FIELDS = (("company", "company_name", 255), ("title", "job_title", 255), ("email", "email", 255), ("phone", "phone", 100))


def _columns(fields: Dict[str, Any], strict: bool, report: "Saved") -> Dict[str, Any]:
    """The API fields → column values (only those present; tolerant drops leave a field out)."""
    out: Dict[str, Any] = {}
    for key, col, limit in _TEXT_FIELDS:
        if key in fields:
            out[col] = clean_text(fields[key], limit)
    if "notes" in fields:
        out["notes"] = _notes(fields["notes"], strict, report)
    if "eventId" in fields:
        out["event_name"] = clean_event(fields["eventId"])
    if "links" in fields:
        out["links"] = [link for link in (fields["links"] or []) if isinstance(link, dict)]
    for key, col, rule in (("linkedin", "linkedin", _linkedin), ("photo", "photo_base64", _photo), ("score", "lead_score", _score)):
        if key in fields:
            v = rule(fields[key], strict)
            if v is not _SKIP:
                out[col] = v
    if "lead_score" in out:
        out["lead_temperature"] = temperature_for(out["lead_score"])
    return out


# ---------------------------------------------------------------- edit conflicts

CONFLICT_FIELDS = ("firstName", "lastName", "company", "title", "email", "phone", "notes", "linkedin", "eventId")
_FIELD_LABELS = {"firstName": "first name", "lastName": "last name", "eventId": "event", "title": "job title"}


def _norm_field(field_name: str, value) -> str:
    s = "" if value is None else str(value)
    if field_name == "notes":
        return s.strip()
    s = " ".join(s.split())
    if s.upper() == "N/A":
        return ""
    if field_name == "email":
        return s.lower()
    if field_name == "linkedin":
        s = re.sub(r"^https?://(www\.)?", "", s.lower()).rstrip("/")
    return s


def edit_conflicts(current: dict, base: Optional[dict], patch: dict) -> List[dict]:
    """Fields the caller changed that were ALSO changed elsewhere since the caller loaded them
    (3-way: base → server vs base → mine). Same new value on both sides isn't a conflict."""
    out = []
    for f in CONFLICT_FIELDS:
        if not base or f not in patch or f not in base:
            continue
        server, was, mine = (_norm_field(f, current.get(f)), _norm_field(f, base.get(f)), _norm_field(f, patch.get(f)))
        if server != was and server != mine:
            out.append({"field": f, "base": base.get(f), "server": current.get(f), "mine": patch.get(f)})
    return out


def merge_notes(server: Optional[str], base: Optional[str], mine: Optional[str]) -> str:
    """Both sides edited the notes: keep the server's text and add what this side added."""
    server, base, mine = (server or "").rstrip(), (base or "").rstrip(), (mine or "").rstrip()
    added = mine[len(base):].strip() if base and mine.startswith(base) else mine.strip()
    if not added or added in server:
        return server
    return f"{server}\n\n{added}".strip()


class EditConflict(Exception):
    """A strict edit of fields someone else changed since the edit started (the route answers 409)."""

    def __init__(self, conflicts: List[dict]):
        super().__init__("This contact was changed somewhere else")
        self.conflicts = conflicts


@dataclass
class Saved:
    """The outcome of a create / update: the contact, the live event sent, and what a tolerant write had
    to change (`clashes` = fields changed elsewhere that kept the newer value / merged notes)."""
    contact: Optional[ContactDB] = None
    event_type: Optional[str] = None
    clashes: List[dict] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    notes_shortened: bool = False

    @property
    def message(self) -> Optional[str]:
        """What to tell the phone about a tolerant write ("Kept the newer company changed elsewhere and merged
        the notes", "Notes were longer than … shortened"); None = applied as sent."""
        parts = [f"kept the newer {', '.join(_FIELD_LABELS.get(f, f) for f in self.kept)} changed elsewhere"] if self.kept else []
        if any(c["field"] == "notes" for c in self.clashes):
            parts.append("merged the notes")
        text = (" and ".join(parts) or ("merged" if self.clashes else "")).capitalize()
        if self.notes_shortened:
            text = f"{text}. {NOTES_SHORTENED}" if text else NOTES_SHORTENED
        return text or None


def _merge_clashes(contact: ContactDB, base: dict, fields: dict, report: Saved) -> dict:
    """Tolerant 3-way merge of an edit made from `base`: notes changed on both sides are merged, any other
    field changed on both sides keeps the newer (stored) value. The name is one column: kept whole."""
    clashes = edit_conflicts(format_contact(contact), base, fields)
    fields = dict(fields)
    for clash in clashes:
        f = clash["field"]
        if f == "notes":
            fields["notes"] = merge_notes(contact.notes, base.get("notes"), fields.get("notes"))
        else:
            report.kept.append(f)
            fields.pop(f, None)
            if f in ("firstName", "lastName"):
                fields.pop("firstName", None)
                fields.pop("lastName", None)
    report.clashes = clashes
    return fields


# ---------------------------------------------------------------- create / update

async def find_by_id(session, contact_id: uuid.UUID):
    """(the live contact, its Recently deleted row) with this id — either may be None. Phones choose
    contact ids, so a retried create finds the one that already went through."""
    live = (await session.execute(select(ContactDB).where(ContactDB.id == contact_id))).scalar_one_or_none()
    archived = None if live is not None else await session.get(DeletedContactDB, contact_id)
    return live, archived


async def create_contact(session, owner: uuid.UUID, fields: Dict[str, Any], *, strict: bool,
                         contact_id: Optional[uuid.UUID] = None, created_at: Optional[datetime] = None,
                         source: Optional[str] = None, photo_base64: Optional[str] = None,
                         event_from_profile: bool = False, shared_pool: bool = False,
                         webhook: bool = False, announce_it: bool = True) -> Saved:
    """A new contact for `owner`. Card / QR scans add three options: no event given → the rep's active event
    (event_from_profile), a copy in the shared_contacts pool (shared_pool), and the n8n webhook (webhook).
    `photo_base64` = a thumbnail the server made itself (a scanned card), stored as is."""
    report = Saved()
    name = join_name(fields.get("firstName"), fields.get("lastName")) if ("firstName" in fields or "lastName" in fields) \
        else " ".join(str(fields.get("name") or "").split())
    name = present(name) or ""
    if not name:
        if strict:
            raise HTTPException(status_code=400, detail="A contact needs a name")
        name = UNNAMED  # an offline capture is never lost for want of a name
    cols = _columns(fields, strict, report)
    cols.pop("lead_temperature", None)
    score = cols.pop("lead_score", None)
    if event_from_profile and not cols.get("event_name"):
        cols["event_name"] = clean_event(event_identity.active_event(await event_identity.load_profile(session, owner)))
    contact = ContactDB(
        id=contact_id or uuid.uuid4(),
        user_id=owner,
        name=name[:255],
        email=cols.get("email"),
        phone=cols.get("phone"),
        company_name=cols.get("company_name"),
        job_title=cols.get("job_title"),
        linkedin=cols.get("linkedin"),
        event_name=cols.get("event_name", ""),
        notes=cols.get("notes", ""),
        links=cols.get("links", []),
        source=source or fields.get("source") or "manual",
        lead_score=score,
        lead_temperature=temperature_for(score),
        photo_base64=cols.get("photo_base64", photo_base64),
        created_at=created_at or datetime.now(timezone.utc),
    )
    session.add(contact)
    if shared_pool:
        session.add(SharedContactDB(
            original_contact_id=contact.id, user_id=owner, name=contact.name, email=contact.email, phone=contact.phone,
            linkedin=contact.linkedin, company_name=contact.company_name, notes=contact.notes, links=contact.links,
            source=contact.source,
        ))
    await session.commit()
    await sync_index(contact)
    report.contact, report.event_type = contact, "contact_scanned"
    if announce_it:
        await announce("contact_scanned", live_payload(contact))
    if webhook and effects.webhook:
        await effects.webhook(contact)
    return report


async def update_contact(session, contact: ContactDB, fields: Dict[str, Any], *, strict: bool,
                         base: Optional[dict] = None, force: bool = False, console: bool = False,
                         announce_it: bool = True) -> Saved:
    """Apply an edit (API field names; only the fields sent change).

    `base` = the edited fields as they were when the edit started: a field also changed elsewhere since is
    a conflict — strict: EditConflict (unless force = "keep mine"); tolerant: merged (see _merge_clashes).
    `console` = the admin console: it can't set the rep's own score (it scores into admin_contact_scores)."""
    report = Saved(contact=contact)
    if console and "score" in fields:
        raise HTTPException(status_code=400, detail="The rep's score can't be set from the admin console. Use Score with AI for the admin score.")
    if base and not force:
        if strict:
            conflicts = edit_conflicts(format_contact(contact), base, fields)
            if conflicts:
                raise EditConflict(conflicts)
        else:
            fields = _merge_clashes(contact, base, fields, report)
    # The name is one `name` column, exposed as firstName / lastName: rebuild it from whichever parts
    # were sent, keeping the untouched part. A blank result leaves the name as it was.
    if "firstName" in fields or "lastName" in fields:
        first, last = split_name(contact.name)
        combined = join_name(fields.get("firstName", first), fields.get("lastName", last))
        if combined:
            contact.name = combined[:255]
    elif present(fields.get("name")):
        contact.name = " ".join(str(fields["name"]).split())[:255]
    old_score = contact.lead_score
    for col, value in _columns(fields, strict, report).items():
        setattr(contact, col, value)
    report.event_type = "score_updated" if contact.lead_score != old_score else "contact_updated"
    await save(session, contact, report.event_type if announce_it else None)
    return report


# ---------------------------------------------------------------- delete / restore / merge

def _json_safe(value):
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__b64__": base64.b64encode(bytes(value)).decode("ascii")}
    return value


def row_snapshot(row) -> dict:
    return {col.name: _json_safe(getattr(row, col.key)) for col in row.__table__.columns}


def from_snapshot(model, data: dict):
    """Rebuild a model instance from row_snapshot() output (types restored per column)."""
    kwargs = {}
    for col in model.__table__.columns:
        if col.name not in data:
            continue
        value = data[col.name]
        if value is not None:
            if isinstance(col.type, PG_UUID):
                value = uuid.UUID(str(value))
            elif isinstance(col.type, DateTime):
                value = datetime.fromisoformat(value)
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
            elif isinstance(col.type, LargeBinary):
                value = base64.b64decode(value["__b64__"])
        kwargs[col.key] = value
    return model(**kwargs)


async def _archive(session, contact: ContactDB, reason: Optional[str] = None) -> DeletedContactDB:
    """Snapshot a contact (and its attached files) into Recently deleted and remove it and the rows that
    reference it from the live tables. The caller commits."""
    now = datetime.now(timezone.utc)
    files = (await session.execute(select(ContactFileDB).where(ContactFileDB.contact_id == contact.id))).scalars().all()
    await session.execute(delete(DeletedContactDB).where(DeletedContactDB.id == contact.id))
    archived = DeletedContactDB(
        id=contact.id,
        user_id=contact.user_id,
        name=contact.name or UNNAMED,
        company_name=present(contact.company_name),
        snapshot=row_snapshot(contact),
        files=[row_snapshot(f) for f in files],
        reason=(reason or None) and reason[:255],
        deleted_at=now,
        purge_after=now + timedelta(days=RECENTLY_DELETED_DAYS),
    )
    # Foreign keys to contacts: Postgres refuses the delete otherwise.
    for model in (ContactFileDB, ContactPipelineDB, ContactListMemberDB):
        await session.execute(delete(model).where(model.contact_id == contact.id))
    await session.delete(contact)
    session.add(archived)
    return archived


async def delete_contact(session, contact: ContactDB, reason: Optional[str] = None) -> DeletedContactDB:
    """Move a contact to Recently deleted (restorable for 30 days): it disappears from every list, search
    and the AI index."""
    archived = await _archive(session, contact, reason)
    await session.commit()
    await sync_index(contact, deleted=True)
    await announce("contact_deleted", {"contactId": str(contact.id), "name": archived.name})
    return archived


async def delete_many(session, contacts: List[ContactDB], reason: Optional[str] = None) -> List[DeletedContactDB]:
    """Archive several contacts in one transaction (admin batch delete)."""
    archived = [await _archive(session, c, reason) for c in contacts]
    await session.commit()
    for c, a in zip(contacts, archived):
        await sync_index(c, deleted=True)
        await announce("contact_deleted", {"contactId": str(c.id), "name": a.name})
    return archived


async def restore_contact(session, deleted_row: DeletedContactDB, owner: uuid.UUID) -> ContactDB:
    """Put a Recently deleted contact back (same ID, fields, photo and attached files)."""
    if (await session.execute(select(ContactDB.id).where(ContactDB.id == deleted_row.id))).scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A contact with this ID already exists")
    contact = from_snapshot(ContactDB, deleted_row.snapshot or {})
    contact.user_id = owner
    contact.updated_at = datetime.now(timezone.utc)
    session.add(contact)
    await session.flush()
    for f in deleted_row.files or []:
        session.add(from_snapshot(ContactFileDB, f))
    await session.delete(deleted_row)
    await session.commit()
    await sync_index(contact)
    await announce("contact_restored", {"contactId": str(contact.id), "name": contact.name})
    return contact


def _pick(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Keep `a` unless it's empty / an OCR "N/A" placeholder, else take `b`."""
    return a if present(a) else (b if present(b) else a)


async def merge_contacts(session, primary: ContactDB, secondary: ContactDB) -> DeletedContactDB:
    """Absorb `secondary` into `primary`, then move `secondary` to Recently deleted ("Merged into …") so a
    wrong merge can be undone by restoring it.

    Fills primary's empty fields from secondary, combines notes/links/audio/actions, keeps the higher score,
    the photo and the earlier created date, and moves attached files, pipelines and list memberships over.
    """
    if primary.id == secondary.id:
        raise HTTPException(status_code=400, detail="Cannot merge a contact with itself")

    for attr in ("name", "email", "phone", "linkedin", "company_name", "job_title"):
        setattr(primary, attr, _pick(getattr(primary, attr), getattr(secondary, attr)))

    def join_text(a: Optional[str], b: Optional[str]) -> str:
        a, b = (a or "").strip(), (b or "").strip()
        return f"{a}\n---\n{b}" if a and b and b not in a else (a or b)

    primary.notes = join_text(primary.notes, secondary.notes)
    primary.admin_notes = join_text(primary.admin_notes, secondary.admin_notes)

    links = list(primary.links or [])
    urls = {link.get("url") for link in links if isinstance(link, dict)}
    links += [link for link in (secondary.links or []) if isinstance(link, dict) and link.get("url") not in urls]
    primary.links = links
    primary.audio_notes = list(primary.audio_notes or []) + list(secondary.audio_notes or [])
    actions = list(primary.lead_recommended_actions or [])
    primary.lead_recommended_actions = actions + [a for a in (secondary.lead_recommended_actions or []) if a not in actions]

    if (secondary.lead_score or -1) > (primary.lead_score or -1):
        primary.lead_score = secondary.lead_score
        primary.lead_temperature = temperature_for(secondary.lead_score)
        primary.lead_score_reasoning = secondary.lead_score_reasoning
        primary.lead_score_breakdown = dict(secondary.lead_score_breakdown or {})
    if not primary.photo_base64 and secondary.photo_base64:
        primary.photo_base64 = secondary.photo_base64
    if not primary.event_name and secondary.event_name:
        primary.event_name = secondary.event_name
    p_created, s_created = aware(primary.created_at), aware(secondary.created_at)
    if s_created and (not p_created or s_created < p_created):
        primary.created_at = secondary.created_at
    primary.updated_at = datetime.now(timezone.utc)

    # Move what's attached to the duplicate onto the kept contact.
    await session.execute(update(ContactFileDB).where(ContactFileDB.contact_id == secondary.id).values(contact_id=primary.id))
    await session.execute(update(ContactPipelineDB).where(ContactPipelineDB.contact_id == secondary.id).values(contact_id=primary.id))
    already = set((await session.execute(
        select(ContactListMemberDB.list_id).where(ContactListMemberDB.contact_id == primary.id)
    )).scalars().all())
    for member in (await session.execute(
        select(ContactListMemberDB).where(ContactListMemberDB.contact_id == secondary.id)
    )).scalars().all():
        if member.list_id in already:
            await session.delete(member)
        else:
            member.contact_id = primary.id
    await session.flush()

    archived = await _archive(session, secondary, reason=f"Merged into {primary.name or 'another contact'}")
    await session.commit()
    await sync_index(secondary, deleted=True)
    await sync_index(primary)
    await announce("contact_merged", {"primaryId": str(primary.id), "name": primary.name})
    return archived
