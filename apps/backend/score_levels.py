"""
The ONE score level rule, used everywhere a 0-100 lead score is turned into words or counted.

    hot   ≥ 85   (band A)
    warm  70–84  (band B)
    cool  50–69  (band C)
    cold  < 50   (band D)
    None  = unscored (no level, no band)

The stored temperature (contacts.lead_temperature, admin_contact_scores.temperature) is the level too —
the same four words, so "cool" is never folded into "cold". The same thresholds exist for SQL
(level_case / level_condition / band_condition) so a database count always agrees with the Python rule.
The apps mirror it (mobile src/features/contacts/scoreLevel.ts, admin ui/kit.tsx).
"""

from typing import Optional

from sqlalchemy import and_, case

HOT_MIN = 85
WARM_MIN = 70
COOL_MIN = 50

LEVELS = ("hot", "warm", "cool", "cold")
UNSCORED = "unscored"
BAND_OF_LEVEL = {"hot": "A", "warm": "B", "cool": "C", "cold": "D"}
LEVEL_OF_BAND = {b: lvl for lvl, b in BAND_OF_LEVEL.items()}


def level_for(score: Optional[int]) -> Optional[str]:
    """hot / warm / cool / cold, or None when unscored."""
    if score is None:
        return None
    return "hot" if score >= HOT_MIN else "warm" if score >= WARM_MIN else "cool" if score >= COOL_MIN else "cold"


# What is stored as a contact's (or an admin score's) temperature: the level itself.
temperature_for = level_for


def band_for(score: Optional[int]) -> Optional[str]:
    """A / B / C / D (the contacts grid's bands), or None when unscored."""
    level = level_for(score)
    return BAND_OF_LEVEL[level] if level else None


def stage_for(score: Optional[int]) -> str:
    """No pipeline stage is stored: a hot (A-band) score marks a lead as qualified. "contacted" etc. would
    claim outreach that never happened, so everything else is new."""
    return "qualified" if is_hot(score) else "new"


def is_hot(score: Optional[int]) -> bool:
    return score is not None and score >= HOT_MIN


def is_hot_or_warm(score: Optional[int]) -> bool:
    return score is not None and score >= WARM_MIN


# ---- SQL: the same rule for a score column (ContactDB.lead_score, AdminContactScoreDB.score) ----

def level_case(col):
    """CASE expression: the level name of `col`, "unscored" for NULL (GROUP BY it to count per level)."""
    return case(
        (col.is_(None), UNSCORED),
        (col >= HOT_MIN, "hot"),
        (col >= WARM_MIN, "warm"),
        (col >= COOL_MIN, "cool"),
        else_="cold",
    )


def level_condition(col, level: str):
    """WHERE condition: `col` is at that level ("hot", "warm", "cool", "cold" or "unscored")."""
    if level == "hot":
        return col >= HOT_MIN
    if level == "warm":
        return and_(col >= WARM_MIN, col < HOT_MIN)
    if level == "cool":
        return and_(col >= COOL_MIN, col < WARM_MIN)
    if level == "cold":
        return col < COOL_MIN
    if level == UNSCORED:
        return col.is_(None)
    raise ValueError(f"unknown level {level!r}")


def band_condition(col, band: Optional[str]):
    """WHERE condition for a contacts-grid band filter (A / B / C / D, U or UNSCORED); None = no filter."""
    b = (band or "").upper()
    if b in LEVEL_OF_BAND:
        return level_condition(col, LEVEL_OF_BAND[b])
    if b in ("U", "UNSCORED"):
        return col.is_(None)
    return None
