import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

import contact_store
import scoring
from conftest import make_user
from database import AdminContactScoreDB, AdminSalesProfileDB, ContactDB, UserProfileDB


class FakeAI:
    """Stands in for the OpenRouter scorer: a fixed score, and a record of what it was asked."""

    def __init__(self, score=88, fail=None):
        self.score, self.fail, self.calls = score, fail, []

    def __call__(self, facts, profile):
        self.calls.append((facts, profile))
        if self.fail:
            raise scoring.ScoringUnavailable("no credits", self.fail)
        return {"score": self.score, "temperature": "ignored", "reasoning": "fits", "recommended_actions": ["call"],
                "breakdown": {"role_fit": 18, "company_fit": 18, "need_fit": 18, "goal_fit": 17, "intent": 17},
                "model": "fake", "runs": 3, "spread": 2}


async def setup(db, effects):
    owner = await make_user(db)
    async with db() as s:
        s.add(UserProfileDB(id=uuid.uuid4(), user_id=owner, profile_data={"company_name": "RepCo", "target_roles": ["CTO"]}))
        s.add(AdminSalesProfileDB(id=scoring.ADMIN_SALES_ID, data={"company_name": "AdminCo", "products": [{"name": "Data"}]}))
        await s.commit()
        saved = await contact_store.create_contact(s, owner, {"firstName": "Ada", "lastName": "L", "title": "CTO"}, strict=True)
    effects.indexed.clear(); effects.live.clear()
    return owner, saved.contact.id


async def test_rep_score_fingerprint_skip_and_force(db, effects):
    owner, cid = await setup(db, effects)
    ai = FakeAI(72)
    first = await scoring.score(cid, "rep", owner=owner, ai=ai)
    assert first["score"] == 72 and first["temperature"] == "warm" and not first["unchanged"] and first["previous_score"] is None
    assert ai.calls[0][1]["company_name"] == "RepCo"  # the rep's own profile
    again = await scoring.score(cid, "rep", owner=owner, ai=ai)
    assert again["unchanged"] and len(ai.calls) == 1   # nothing changed → no AI call, same score
    await scoring.score(cid, "rep", owner=owner, ai=ai, force=True)
    assert len(ai.calls) == 2
    async with db() as s:  # an edit to what the score depends on → asks again
        await contact_store.update_contact(s, await s.get(ContactDB, cid), {"notes": "asked for a demo"}, strict=True)
    await scoring.score(cid, "rep", owner=owner, ai=ai)
    assert len(ai.calls) == 3
    # stored through the contact store: the search index follows; the caller (route) announces the score
    assert len(effects.indexed) == 4 and effects.events() == ["contact_updated"]
    async with db() as s:
        row = await s.get(ContactDB, cid)
        assert row.lead_score == 72 and row.lead_temperature == "warm" and row.lead_score_breakdown["_meta"]["fingerprint"]


async def test_admin_score_is_isolated_from_the_rep(db, effects):
    owner, cid = await setup(db, effects)
    ai = FakeAI(90)
    result = await scoring.score(cid, "admin", ai=ai, admin_id=owner)
    assert result["score"] == 90 and result["temperature"] == "hot"
    assert ai.calls[0][1] == {"company_name": "AdminCo", "products": [{"name": "Data"}]}  # ONLY the admin profile
    async with db() as s:
        rep_row = await s.get(ContactDB, cid)
        admin_row = (await s.execute(select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id == cid))).scalar_one()
    assert rep_row.lead_score is None and rep_row.lead_temperature is None  # the rep's score is never written
    assert admin_row.score == 90 and admin_row.temperature == "hot" and admin_row.scored_by == owner
    assert effects.indexed == [] and effects.live == []  # nothing on the rep's contact changed
    assert (await scoring.score(cid, "admin", ai=ai))["unchanged"] and len(ai.calls) == 1
    # the rep scoring the same contact is its own world: it asks the AI and keeps the admin row as is
    await scoring.score(cid, "rep", ai=FakeAI(40))
    async with db() as s:
        assert (await s.get(AdminContactScoreDB, cid)).score == 90 and (await s.get(ContactDB, cid)).lead_score == 40


async def test_errors_save_nothing(db, effects):
    owner, cid = await setup(db, effects)
    with pytest.raises(HTTPException) as err:
        await scoring.score(cid, "rep", ai=FakeAI(fail="no_credits"))
    assert err.value.status_code == 503 and "out of credits" in err.value.detail
    with pytest.raises(HTTPException) as err:
        await scoring.score(cid, "rep", owner=uuid.uuid4(), ai=FakeAI())
    assert err.value.status_code == 404  # not this owner's contact
    with pytest.raises(HTTPException) as err:
        await scoring.score("nope", "admin", ai=FakeAI())
    assert err.value.status_code == 400
    old, scoring.ai_scorer = scoring.ai_scorer, None
    try:
        with pytest.raises(HTTPException) as err:
            await scoring.score(cid, "rep")
        assert err.value.status_code == 503 and not scoring.configured()
    finally:
        scoring.ai_scorer = old
    async with db() as s:
        assert (await s.get(ContactDB, cid)).lead_score is None
