"""
"Same event?" — and a rep's own events (their profile).

There's no events table: an event is a NAME. It shows up in the booth directory (exhibitors), on contacts
("met at"), in reps' event lists (profile events / current_event_name), on event files, notes and the
planner. Two spellings are the same event when they match case- and spacing-insensitively:

    event_key(name)                        "  GITEX   2026 " → "gitex 2026"
    same_event(a, b)                       event_key(a) == event_key(b) (blank never matches)
    event_slug(name)                       "gitex-2026" (legacy slug ids: /api/events/{slug}/seats)
    await name_variants(session, col, n)   every stored spelling of n in that column
    await event_filter(session, col, n)    WHERE col is that event (col IN the stored spellings)

A rep's events live in user_profiles.profile_data (a JSON blob; some old rows hold it as a JSON string):

    profile_dict(row_or_blob)    a copy to edit; None = unreadable (leave it alone)
    read_profile(row_or_blob)    a dict to read ({} when unreadable)
    profile_events(blob)         (events listed, active event)
    event_goals(profile, event)  the goals for one event (per-event details, else the active event's)
    write_profile(...)           THE way profile_data is saved: a new dict + flag_modified
    set_profile_event(...)       add / remove / rename an event in a rep's list (keeps the active one valid)
    sync_event_context / normalize_profile_events: the profile-update rules for events
"""

import json
import uuid
from typing import List, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified

from database import UserProfileDB

MAX_PROFILE_EVENTS = 20
MAX_EVENT_NAME_LENGTH = 120


# ---------------------------------------------------------------- same event?

def event_key(name: Optional[str]) -> str:
    """Event / booth names compared the way people type them: case- and spacing-insensitive."""
    return " ".join(str(name or "").split()).lower()


def same_event(a: Optional[str], b: Optional[str]) -> bool:
    return bool(event_key(a)) and event_key(a) == event_key(b)


def event_slug(name: Optional[str]) -> str:
    """The id older clients used for an event ("GITEX 2026" → "gitex-2026")."""
    return "".join(ch if ch.isalnum() else "-" for ch in (name or "").lower()).strip("-")


async def name_variants(session, column, name: Optional[str]) -> List[str]:
    """Every stored spelling of an event name (any case / spacing) in `column`, for an exact IN filter
    that matches what event_key() groups. Nothing stored → [name] (matches nothing new)."""
    k = event_key(name)
    first_word = k.split(" ")[0] if k else ""
    values = (await session.execute(select(column).distinct().where(func.lower(column).contains(first_word)))).scalars().all()
    return [v for v in values if v is not None and event_key(v) == k] or [name or ""]


async def event_filter(session, column, name: Optional[str]):
    """WHERE condition: `column` holds the same event as `name` (any stored spelling)."""
    return column.in_(await name_variants(session, column, name))


# ---------------------------------------------------------------- a rep's profile (profile_data)

def _blob(row_or_blob):
    return row_or_blob.profile_data if isinstance(row_or_blob, UserProfileDB) else row_or_blob


def profile_dict(row_or_blob) -> Optional[dict]:
    """A copy of a profile's JSON to edit (old rows may hold it as a string). None = unreadable: leave it
    alone rather than overwrite the rep's whole profile with just an event list."""
    raw = _blob(row_or_blob)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def read_profile(row_or_blob) -> dict:
    """A profile to read: {} when there is none or it can't be read."""
    return profile_dict(row_or_blob) or {}


def profile_events(row_or_blob) -> Tuple[List[str], str]:
    """(events the rep lists, their current event) from a profile."""
    d = read_profile(row_or_blob)
    listed = [e for e in (d.get("events") or []) if isinstance(e, str) and e.strip()]
    current = d.get("current_event_name") if isinstance(d.get("current_event_name"), str) else ""
    return listed, (current or "").strip()


def active_event(row_or_blob) -> str:
    """The rep's current event ("" = none): new scans are tagged with it."""
    return profile_events(row_or_blob)[1]


def event_goals(profile: dict, event_name: Optional[str]) -> List[str]:
    """Goals for the event a contact was met at (per-event goals), else the current event's."""
    details = (profile or {}).get("event_details") or {}
    if event_name and isinstance(details, dict):
        for k, v in details.items():
            if same_event(k, event_name) and isinstance(v, dict):
                return list(v.get("goals") or [])
    return list((profile or {}).get("event_goals") or [])


async def load_profile(session, user_id: uuid.UUID) -> Optional[UserProfileDB]:
    return (await session.execute(select(UserProfileDB).where(UserProfileDB.user_id == user_id))).scalar_one_or_none()


def write_profile(session, row: Optional[UserProfileDB], user_id: uuid.UUID, data: dict) -> UserProfileDB:
    """Save a rep's profile: always a NEW dict + flag_modified (an in-place edit of the loaded JSON isn't
    seen by SQLAlchemy, so it was silently never written). Creates the row when there is none."""
    data = dict(data)
    if row is None:
        row = UserProfileDB(id=uuid.uuid4(), user_id=user_id, profile_data=data)
        session.add(row)
        return row
    row.profile_data = data
    flag_modified(row, "profile_data")
    return row


async def set_profile_event(session, user_id: uuid.UUID, name: str, add: bool, rename_to: Optional[str] = None) -> None:
    """Add / remove / rename an event in a rep's own event list (what their app shows under Events).
    Removing or renaming the rep's ACTIVE event updates it too, so the app never points at an event
    that no longer exists."""
    prof = await load_profile(session, user_id)
    data = profile_dict(prof)
    if data is None:
        print(f"[PLANNER] profile of {user_id} isn't readable JSON — event list left unchanged")
        return
    events = [e for e in (data.get("events") or []) if isinstance(e, str) and e.strip()]
    has = any(same_event(e, name) for e in events)
    active = same_event(data.get("current_event_name"), name)
    if rename_to is not None:
        if not has and not active:
            return
        events = [rename_to if same_event(e, name) else e for e in events]
        if active:
            data["current_event_name"] = rename_to
        details = data.get("event_details")
        if isinstance(details, dict):
            data["event_details"] = {(rename_to if same_event(k, name) else k): v for k, v in details.items()}
    elif add and not has:
        events.append(name)
    elif not add and (has or active):
        events = [e for e in events if not same_event(e, name)]
        if active:
            data["current_event_name"] = ""  # the app shows "no active event" instead of a dead one
    else:
        return
    data["events"] = events
    write_profile(session, prof, user_id, data)


def sync_event_context(merged: dict, update_data: dict, existing: dict) -> None:
    """Keep goals per event (event_details) and mirror the CURRENT event's goals into the
    legacy event_goals / current_event_description fields the AI prompts read.

    Goals used to be one global list, so every event showed the same goals.
    """
    cur = (merged.get("current_event_name") or "").strip()
    details = {k: dict(v) for k, v in (merged.get("event_details") or {}).items() if isinstance(v, dict)}
    find = lambda name: next((k for k in details if k.lower() == name.lower()), None)  # noqa: E731

    # Old profiles: the single global goal list belongs to the event that was current BEFORE
    # this update (switching events mustn't carry those goals over to the new one).
    old_cur = (existing.get("current_event_name") or cur).strip()
    if "event_details" not in existing and "event_details" not in update_data and old_cur and not find(old_cur):
        if existing.get("event_goals") or existing.get("current_event_description"):
            details[old_cur] = {"goals": list(existing.get("event_goals") or []),
                                "description": existing.get("current_event_description") or ""}
    # Older clients writing the flat fields: those are the current event's goals.
    if cur and "event_details" not in update_data and ("event_goals" in update_data or "current_event_description" in update_data):
        k = find(cur) or cur
        d = details.get(k, {"goals": [], "description": ""})
        if "event_goals" in update_data:
            d["goals"] = list(update_data.get("event_goals") or [])
        if "current_event_description" in update_data:
            d["description"] = update_data.get("current_event_description") or ""
        details[k] = d
    # Forget goals of events that were removed from the list.
    keep = {e.lower() for e in (merged.get("events") or [])} | ({cur.lower()} if cur else set())
    details = {k: v for k, v in details.items() if k.lower() in keep}

    merged["event_details"] = details
    active = details.get(find(cur)) if cur and find(cur) else None
    merged["event_goals"] = list((active or {}).get("goals") or [])
    merged["current_event_description"] = (active or {}).get("description") or ""


def normalize_profile_events(update_data: dict, existing: dict) -> None:
    """Clean `events` / `current_event_name` / `ended_events` in an incoming profile update, in place.

    Trims names, drops blanks and case-insensitive duplicates, enforces limits, and makes
    sure the active event is always part of the events list.
    """
    if "current_event_name" in update_data and update_data["current_event_name"] is not None:
        name = " ".join(str(update_data["current_event_name"]).split())
        if len(name) > MAX_EVENT_NAME_LENGTH:
            raise HTTPException(status_code=400, detail=f"Event names can be at most {MAX_EVENT_NAME_LENGTH} characters")
        update_data["current_event_name"] = name

    if update_data.get("ended_events") is not None:
        ended, seen_ended = [], set()
        for raw in update_data["ended_events"]:
            name = " ".join(str(raw).split())[:MAX_EVENT_NAME_LENGTH]
            if name and name.lower() not in seen_ended:
                seen_ended.add(name.lower())
                ended.append(name)
        update_data["ended_events"] = ended[-200:]

    if update_data.get("events") is None and "current_event_name" not in update_data:
        return

    events = update_data.get("events")
    if events is None:
        events = list(existing.get("events") or [])
    cleaned, seen = [], set()
    for raw in events:
        name = " ".join(str(raw).split())
        if not name:
            continue
        if len(name) > MAX_EVENT_NAME_LENGTH:
            raise HTTPException(status_code=400, detail=f"Event names can be at most {MAX_EVENT_NAME_LENGTH} characters")
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append(name)

    active = update_data.get("current_event_name")
    if active and active.lower() not in seen:
        cleaned.append(active)
    if len(cleaned) > MAX_PROFILE_EVENTS:
        raise HTTPException(status_code=400, detail=f"You can add up to {MAX_PROFILE_EVENTS} events")
    update_data["events"] = cleaned
