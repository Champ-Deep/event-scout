import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

import event_identity as ev
from conftest import make_user
from database import ContactDB, UserProfileDB


def test_same_event_rules():
    assert ev.event_key("  GITEX   2026 ") == "gitex 2026"
    assert ev.same_event("GITEX 2026", "gitex  2026") and not ev.same_event("GITEX", "GITEX 2026")
    assert not ev.same_event("", "") and not ev.same_event(None, None)
    assert ev.event_slug("WHX Dubai 2026!") == "whx-dubai-2026"


async def test_event_filter_matches_every_spelling(db):
    owner = await make_user(db)
    names = ["GITEX 2026", "gitex  2026", " Gitex 2026 ", "GITEX", "Gitex 2026 Expo", ""]
    async with db() as s:
        for n in names:
            s.add(ContactDB(id=uuid.uuid4(), user_id=owner, name=f"c{n}", event_name=n))
        await s.commit()
        got = (await s.execute(select(ContactDB.event_name).where(await ev.event_filter(s, ContactDB.event_name, "gitex 2026")))).scalars().all()
        assert sorted(got) == sorted(["GITEX 2026", "gitex  2026", " Gitex 2026 "])
        none = (await s.execute(select(ContactDB.id).where(await ev.event_filter(s, ContactDB.event_name, "Nowhere Expo")))).all()
        assert none == []


def test_profile_readers():
    assert ev.profile_dict(None) == {} and ev.read_profile("not json") == {} and ev.profile_dict("not json") is None
    blob = json.dumps({"events": ["A", "  ", 3, "B"], "current_event_name": " B ", "event_goals": ["g"],
                       "event_details": {"b": {"goals": ["meet CTOs"]}}})
    assert ev.profile_events(blob) == (["A", "B"], "B")
    assert ev.active_event(blob) == "B"
    profile = ev.read_profile(blob)
    assert ev.event_goals(profile, "B") == ["meet CTOs"] and ev.event_goals(profile, "Other") == ["g"]


async def profile_of(db, uid):
    async with db() as s:
        row = (await s.execute(select(UserProfileDB).where(UserProfileDB.user_id == uid))).scalar_one_or_none()
        return ev.profile_dict(row) if row else None


async def test_set_profile_event_add_rename_remove(db):
    uid = await make_user(db)
    async with db() as s:  # a string-encoded profile (old rows)
        s.add(UserProfileDB(id=uuid.uuid4(), user_id=uid, profile_data=json.dumps({
            "full_name": "Rep", "events": ["Old Expo"], "current_event_name": "old  expo",
            "event_details": {"Old Expo": {"goals": ["x"]}}})))
        await s.commit()
    async with db() as s:
        await ev.set_profile_event(s, uid, "OLD EXPO", add=False, rename_to="New Expo")
        await s.commit()
    p = await profile_of(db, uid)
    assert p["events"] == ["New Expo"] and p["current_event_name"] == "New Expo" and "New Expo" in p["event_details"]
    assert p["full_name"] == "Rep"  # the rest of the profile is kept
    async with db() as s:
        await ev.set_profile_event(s, uid, "Second", add=True)
        await ev.set_profile_event(s, uid, "second", add=True)  # already there (any spelling)
        await s.commit()
    assert (await profile_of(db, uid))["events"] == ["New Expo", "Second"]
    async with db() as s:
        await ev.set_profile_event(s, uid, "new expo", add=False)
        await s.commit()
    p = await profile_of(db, uid)
    assert p["events"] == ["Second"] and p["current_event_name"] == ""  # no dead active event


async def test_unreadable_profile_is_left_alone_and_writes_create_rows(db):
    uid, fresh = await make_user(db), await make_user(db)
    async with db() as s:
        s.add(UserProfileDB(id=uuid.uuid4(), user_id=uid, profile_data="{broken"))
        await s.commit()
    async with db() as s:
        await ev.set_profile_event(s, uid, "Expo", add=True)
        await ev.set_profile_event(s, fresh, "Expo", add=True)  # no profile yet → one is created
        await s.commit()
        raw = (await s.execute(select(UserProfileDB.profile_data).where(UserProfileDB.user_id == uid))).scalar_one()
    assert raw == "{broken"
    assert (await profile_of(db, fresh))["events"] == ["Expo"]
    async with db() as s:  # write_profile: a new dict, saved even though the JSON column is mutable
        row = await ev.load_profile(s, fresh)
        data = ev.read_profile(row)
        data["events"].append("Two")
        ev.write_profile(s, row, fresh, data)
        await s.commit()
    assert (await profile_of(db, fresh))["events"] == ["Expo", "Two"]


def test_normalize_profile_events():
    upd = {"events": [" A ", "a", "", "B"], "current_event_name": "  C  ", "ended_events": ["x", "X", " y "]}
    ev.normalize_profile_events(upd, {})
    assert upd["events"] == ["A", "B", "C"] and upd["current_event_name"] == "C" and upd["ended_events"] == ["x", "y"]
    with pytest.raises(HTTPException):
        ev.normalize_profile_events({"events": [f"E{i}" for i in range(ev.MAX_PROFILE_EVENTS + 1)]}, {})
    merged = {"current_event_name": "B", "events": ["A", "B"], "event_details": {"b": {"goals": ["g"], "description": "d"}}}
    ev.sync_event_context(merged, {"current_event_name": "B"}, {})
    assert merged["event_goals"] == ["g"] and merged["current_event_description"] == "d"
