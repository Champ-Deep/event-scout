import base64
import io
import json
import uuid

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import select

import contact_store as store
import contact_view
from conftest import make_user
from database import ContactDB, DeletedContactDB, SharedContactDB, UserProfileDB


def png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (200, 10, 10)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def new(db, owner, strict=True, **fields):
    async with db() as s:
        return (await store.create_contact(s, owner, fields, strict=strict)).contact


async def reload(db, cid):
    async with db() as s:
        return (await s.execute(select(ContactDB).where(ContactDB.id == cid))).scalar_one_or_none()


# ---------------------------------------------------------------- create

async def test_create_strict_cleans_and_runs_side_effects_once(db, effects):
    owner = await make_user(db)
    c = await new(db, owner, firstName=" Ada ", lastName="Lovelace", company="  Analytical   Engines ", email="N/A",
                  title="CTO", eventId="  GITEX   2026 ", score=60, notes="hello  \n", photo=png_b64())
    row = await reload(db, c.id)
    assert row.name == "Ada Lovelace" and row.company_name == "Analytical Engines" and row.job_title == "CTO"
    assert row.email is None and row.linkedin is None and row.phone is None  # no "N/A" placeholder is written
    assert row.event_name == "GITEX 2026" and row.notes == "hello"
    assert row.lead_score == 60 and row.lead_temperature == "cool"
    assert row.photo_base64 and base64.b64decode(row.photo_base64)[:2] == b"\xff\xd8"  # stored as a JPEG
    assert len(effects.indexed) == 1 and effects.indexed[0][1]["id"] == str(c.id)
    assert effects.events() == ["contact_scanned"]
    payload = effects.live[0][1]
    assert set(payload) == {"contactId", "name", "company", "score", "scoreBand", "source"} and payload["scoreBand"] == "C"


@pytest.mark.parametrize("fields,status", [
    ({"firstName": "  "}, 400),                                         # nameless
    ({"firstName": "A", "photo": "not-an-image"}, 400),
    ({"firstName": "A", "linkedin": "not a linkedin link"}, 400),
    ({"firstName": "A", "score": 101}, 400),
    ({"firstName": "A", "score": "high"}, 400),
    ({"firstName": "A", "notes": 5}, 400),
    ({"firstName": "A", "notes": "x" * (store.NOTES_MAX_CHARS + 1)}, 413),
])
async def test_create_strict_refuses_bad_values(db, effects, fields, status):
    owner = await make_user(db)
    with pytest.raises(HTTPException) as err:
        await new(db, owner, **fields)
    assert err.value.status_code == status
    assert effects.indexed == [] and effects.live == []  # nothing was written, so nothing announced


async def test_create_tolerant_keeps_the_capture(db, effects):
    owner = await make_user(db)
    async with db() as s:
        saved = await store.create_contact(s, owner, {"firstName": "", "linkedin": "John on LinkedIn", "photo": "garbage",
                                                      "score": "high", "notes": "y" * (store.NOTES_MAX_CHARS + 50)}, strict=False)
    row = await reload(db, saved.contact.id)
    assert row.name == store.UNNAMED  # an offline capture is never lost for want of a name
    assert row.linkedin is None and row.photo_base64 is None and row.lead_score is None
    assert len(row.notes) == store.NOTES_MAX_CHARS
    assert saved.notes_shortened and saved.message == store.NOTES_SHORTENED and saved.clashes == []
    assert len(effects.indexed) == 1 and effects.events() == ["contact_scanned"]


async def test_scan_options_event_from_profile_pool_and_webhook(db, effects):
    owner = await make_user(db)
    async with db() as s:  # an old row whose profile JSON is a string
        s.add(UserProfileDB(id=uuid.uuid4(), user_id=owner, profile_data=json.dumps({"current_event_name": "WHX 2026"})))
        await s.commit()
    async with db() as s:
        saved = await store.create_contact(s, owner, {"name": "Card Person", "email": "N/A", "eventId": ""}, strict=False,
                                           source="scan", event_from_profile=True, shared_pool=True, webhook=True)
        pool = (await s.execute(select(SharedContactDB).where(SharedContactDB.original_contact_id == saved.contact.id))).scalar_one()
    assert saved.contact.event_name == "WHX 2026" and saved.contact.source == "scan"
    assert pool.name == "Card Person"
    assert effects.webhooks == [str(saved.contact.id)]


# ---------------------------------------------------------------- update

async def test_update_strict_conflicts_force_and_events(db, effects):
    owner = await make_user(db)
    c = await new(db, owner, firstName="Conf", lastName="Test", email="a@x.com", notes="first")
    base = {"email": "a@x.com", "notes": "first"}
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        await store.update_contact(s, row, {"email": "admin@x.com"}, strict=True)  # someone else edits
    effects.live.clear(); effects.indexed.clear()
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        with pytest.raises(store.EditConflict) as clash:
            await store.update_contact(s, row, {"email": "mine@x.com"}, strict=True, base=base)
        assert clash.value.conflicts[0]["field"] == "email" and clash.value.conflicts[0]["server"] == "admin@x.com"
        assert effects.live == []
        # an untouched field still saves; the same value on both sides isn't a conflict; force keeps mine
        await store.update_contact(s, row, {"notes": "first\nmine"}, strict=True, base=base)
        await store.update_contact(s, row, {"email": "admin@x.com"}, strict=True, base=base)
        await store.update_contact(s, row, {"email": "mine@x.com"}, strict=True, base=base, force=True)
    row = await reload(db, c.id)
    assert row.email == "mine@x.com" and row.notes == "first\nmine"
    assert effects.events() == ["contact_updated"] * 3 and len(effects.indexed) == 3  # once per write


async def test_update_score_rules(db, effects):
    owner = await make_user(db)
    c = await new(db, owner, firstName="Score", lastName="Me")
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        with pytest.raises(HTTPException) as err:
            await store.update_contact(s, row, {"score": 90}, strict=True, console=True)  # the console can't set the rep's score
        assert err.value.status_code == 400
        saved = await store.update_contact(s, row, {"score": 90}, strict=True)
        assert saved.event_type == "score_updated" and row.lead_temperature == "hot"
        saved = await store.update_contact(s, row, {"score": 90, "title": "VP"}, strict=True)
        assert saved.event_type == "contact_updated"  # the score didn't change


async def test_update_name_keeps_the_untouched_part(db, effects):
    owner = await make_user(db)
    c = await new(db, owner, firstName="Maya", lastName="van Patel")
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        await store.update_contact(s, row, {"firstName": "Mia"}, strict=False)
        assert row.name == "Mia van Patel"
        await store.update_contact(s, row, {"firstName": "", "lastName": ""}, strict=True)
        assert row.name == "Mia van Patel"  # a blank name never replaces one


async def test_update_tolerant_merges_reports_and_drops(db, effects):
    owner = await make_user(db)
    c = await new(db, owner, firstName="Off", lastName="Line", notes="first line\nmine", photo=png_b64())
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        await store.update_contact(s, row, {"notes": "first line\nmine\nadmin added", "company": "AdminCo"}, strict=True)
        photo = row.photo_base64
    async with db() as s:
        row = await s.get(ContactDB, c.id)
        saved = await store.update_contact(s, row, {
            "notes": "first line\nmine\noffline added", "company": "PhoneCo", "phone": "+1 555 0100",
            "photo": "broken", "linkedin": "linkedin.com/in/someone",
        }, strict=False, base={"notes": "first line\nmine", "company": None, "phone": None})
    row = await reload(db, c.id)
    assert "admin added" in row.notes and "offline added" in row.notes          # notes merged
    assert row.company_name == "AdminCo" and row.phone == "+1 555 0100"         # newer company kept, rest applied
    assert row.photo_base64 == photo                                            # a bad photo is dropped, not cleared
    assert row.linkedin == "https://www.linkedin.com/in/someone"
    assert {c["field"] for c in saved.clashes} == {"notes", "company"} and saved.kept == ["company"]
    assert saved.message == "Kept the newer company changed elsewhere and merged the notes"


def test_conflict_and_merge_rules():
    base = {"email": "A@x.com ", "linkedin": "linkedin.com/in/x/"}
    # formatting differences aren't changes
    assert store.edit_conflicts({"email": "a@x.com", "linkedin": "https://www.linkedin.com/in/x"}, base,
                                {"email": "new@x.com", "linkedin": "linkedin.com/in/y"}) == []
    assert store.edit_conflicts({"company": "B"}, {"company": "A"}, {"company": "C"})[0]["field"] == "company"
    assert store.edit_conflicts({"company": "B"}, None, {"company": "C"}) == []
    assert store.merge_notes("server\nmore", "server", "server\nmine") == "server\nmore\n\nmine"
    assert store.merge_notes("server mine", "server", "server mine") == "server mine"  # already there


# ---------------------------------------------------------------- delete / restore / merge

async def test_delete_restore_merge(db, effects):
    owner = await make_user(db)
    a = await new(db, owner, firstName="Keep", lastName="Me", email="k@x.com", notes="a")
    b = await new(db, owner, firstName="Dup", lastName="Me", phone="+1 555", notes="b", score=90)
    effects.live.clear(); effects.indexed.clear()

    async with db() as s:
        archived = await store.delete_contact(s, await s.get(ContactDB, b.id))
    assert archived.name == "Dup Me" and await reload(db, b.id) is None
    assert effects.unindexed == [(str(owner), str(b.id))] and effects.events() == ["contact_deleted"]

    async with db() as s:
        restored = await store.restore_contact(s, await s.get(DeletedContactDB, b.id), owner)
        assert restored.name == "Dup Me" and await s.get(DeletedContactDB, b.id) is None
    assert effects.events()[-1] == "contact_restored" and effects.indexed[-1][1]["id"] == str(b.id)

    async with db() as s:
        primary, secondary = await s.get(ContactDB, a.id), await s.get(ContactDB, b.id)
        archived = await store.merge_contacts(s, primary, secondary)
        assert archived.reason == "Merged into Keep Me"
    row = await reload(db, a.id)
    assert row.phone == "+1 555" and row.lead_score == 90 and row.lead_temperature == "hot" and "a" in row.notes and "b" in row.notes
    assert effects.events()[-1] == "contact_merged"


def test_views_read_legacy_placeholders_as_missing():
    c = ContactDB(id=uuid.uuid4(), user_id=uuid.uuid4(), name="Legacy Row", email="N/A", linkedin="N/A", company_name="n/a",
                  lead_score=72, notes="")
    view = contact_view.format_contact(c)
    assert view["email"] is None and view["linkedin"] is None and view["company"] is None
    assert view["level"] == "warm" and view["scoreBand"] == "B" and view["stage"] == "new"
    assert "adminNotes" not in contact_view.app_view(view)
    lead = contact_view.crm_view(c, "Owner")
    assert lead["email"] == "" and lead["level"] == "warm" and lead["owner"] == "Owner"
