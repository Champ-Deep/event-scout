import uuid

import pytest
from sqlalchemy import func, select

import score_levels as sl
from conftest import make_user
from database import ContactDB

SAMPLES = [None, 0, 1, 49, 50, 51, 69, 70, 71, 84, 85, 86, 100]


@pytest.mark.parametrize("score,level,band", [
    (None, None, None), (0, "cold", "D"), (49, "cold", "D"), (50, "cool", "C"), (69, "cool", "C"),
    (70, "warm", "B"), (84, "warm", "B"), (85, "hot", "A"), (100, "hot", "A"),
])
def test_thresholds(score, level, band):
    assert sl.level_for(score) == level
    assert sl.band_for(score) == band
    assert sl.temperature_for(score) == level  # the stored temperature is the level (cool is never folded into cold)


def test_stage_and_predicates():
    assert sl.stage_for(85) == "qualified" and sl.stage_for(84) == "new" and sl.stage_for(None) == "new"
    assert sl.is_hot(85) and not sl.is_hot(84) and not sl.is_hot(None)
    assert sl.is_hot_or_warm(70) and not sl.is_hot_or_warm(69) and not sl.is_hot_or_warm(None)
    assert sl.band_condition(ContactDB.lead_score, "") is None
    with pytest.raises(ValueError):
        sl.level_condition(ContactDB.lead_score, "lukewarm")


async def test_sql_rule_agrees_with_python(db):
    owner = await make_user(db)
    async with db() as s:
        for score in SAMPLES:
            s.add(ContactDB(id=uuid.uuid4(), user_id=owner, name=f"S{score}", lead_score=score))
        await s.commit()

        # the CASE expression names every row like level_for does
        rows = (await s.execute(select(ContactDB.lead_score, sl.level_case(ContactDB.lead_score)))).all()
        assert {score: lvl for score, lvl in rows} == {score: sl.level_for(score) or sl.UNSCORED for score in SAMPLES}

        # band / level filters count exactly what the Python rule counts
        for band in ("A", "B", "C", "D", "U", "unscored"):
            n = (await s.execute(select(func.count(ContactDB.id)).where(sl.band_condition(ContactDB.lead_score, band)))).scalar()
            want = sum(1 for x in SAMPLES if (sl.band_for(x) or "U") == ("U" if band.upper() in ("U", "UNSCORED") else band))
            assert n == want, band
        for level in sl.LEVELS + (sl.UNSCORED,):
            n = (await s.execute(select(func.count(ContactDB.id)).where(sl.level_condition(ContactDB.lead_score, level)))).scalar()
            assert n == sum(1 for x in SAMPLES if (sl.level_for(x) or sl.UNSCORED) == level), level
