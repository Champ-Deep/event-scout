"""
The Scoring module: one way to score a contact, for the rep or for the admin console.

    await score(contact_id, audience="rep" | "admin", force=False, owner=None, admin_id=None, ai=None) → dict

Used by the rep's Score (POST /api/contact/{id}/score from the app), the console's Score with AI, Score
all (both), auto-score on scan and the pipeline's score step. It owns:

  the profile          rep: the contact owner's own profile · admin: ONLY the admin sales profile
                       (admin_ai_profile — never a rep's profile, so admin results don't depend on rep data)
  the facts            scoring_facts(contact) — the same facts for every caller
  the fingerprint      a hash of facts + profile; a saved score with the same fingerprint is kept without an
                       AI call (asking again only adds noise) — everywhere, unless force=True
  the AI call          `ai_scorer(facts, profile) -> result` (app.score_contact_with_ai, injected at startup;
                       blocking, run in a thread) with one timeout, SCORE_TIMEOUT_S
  storage, by audience rep: ContactDB.lead_* (through contact_store.save: index updated, no live event —
                       the caller announces) · admin: admin_contact_scores (the rep's contact is only read)

The result: {status, contact_id, name, score, temperature, reasoning, recommended_actions, breakdown, runs,
spread, model, scored_at, unchanged, previous_score (when not unchanged)}. Errors are HTTPExceptions:
400 bad id, 404 not found (or not the owner's), 503 AI unavailable / not configured, 504 timeout.
"""

import asyncio
import hashlib
import json
import traceback
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import HTTPException
from sqlalchemy import select

import contact_store
import event_identity
from database import AdminContactScoreDB, AdminSalesProfileDB, ContactDB, get_session_factory
from score_levels import temperature_for

SCORING_VERSION = "v2.1"  # bump when the prompt / rules change (forces a fresh score)
SCORE_TIMEOUT_S = 55      # one AI score (all its runs together) — the same for every caller

SCORING_UNAVAILABLE_MESSAGES = {
    "not_configured": "Lead scoring isn't set up on the server (no OpenRouter API key).",
    "bad_key": "Lead scoring is unavailable: the server's OpenRouter API key isn't working. Ask your admin to check it.",
    "no_credits": "Lead scoring is paused: the server's OpenRouter account is out of credits. Ask your admin to top it up.",
    "timeout": "Scoring took too long. Please try again.",
    "rate_limited": "Lead scoring is busy right now (AI rate limit). Please try again in a minute.",
    "unavailable": "Lead scoring couldn't reach the AI service. Please try again.",
}
NOT_AVAILABLE_DETAIL = "Lead scoring is not available on this server"

# The AI scorer: (facts, profile) -> {score, temperature, reasoning, recommended_actions, breakdown, model,
# runs, spread}. Blocking. Injected by app.py (score_contact_with_ai); tests pass a fake (`ai=` or here).
ai_scorer: Optional[Callable[[dict, dict], dict]] = None


class ScoringUnavailable(Exception):
    """The AI couldn't produce a score. Callers must NOT save a placeholder score:
    a made-up 50 looked like a real assessment in the app."""

    def __init__(self, message: str, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason  # "not_configured" | "bad_key" | "no_credits" | "rate_limited" | "timeout" | "unavailable"


def configured() -> bool:
    return ai_scorer is not None


# ---------------------------------------------------------------- what a score depends on

_GENERIC_MAIL = {"gmail", "yahoo", "hotmail", "outlook", "live", "icloud", "aol", "proton", "protonmail", "zoho",
                 "yandex", "mail", "rediffmail", "gmx", "msn"}


def scoring_facts(contact) -> dict:
    """Everything the scorer is told about a contact — ONE place for scan, Score and Score all.
    (The title used to be missing, so the AI kept saying "without a job title" and scored everyone alike.)"""
    def present(v):
        v = (v or "").strip() if isinstance(v, str) else v
        return None if (v in (None, "", "N/A")) else v
    email = present(contact.email) or ""
    domain = email.split("@", 1)[1].lower() if "@" in email else ""
    if domain and domain.split(".")[0] in _GENERIC_MAIL:
        domain = ""
    return {
        "name": contact.name,
        "title": present(getattr(contact, "job_title", None)),
        "company": present(contact.company_name),
        "company_domain": domain or None,
        "event": present(getattr(contact, "event_name", None)),
        "met_via": {"scan": "scanned their business card", "qr": "scanned their QR code / badge",
                    "manual": "added by hand"}.get(contact.source or "", contact.source or None),
        "has_linkedin": bool(present(contact.linkedin)),
        "has_email": bool(email),
        "has_phone": bool(present(contact.phone)),
        "notes": (contact.notes or "").strip()[:1500],
    }


def scoring_fingerprint(facts: dict, user_profile: dict) -> str:
    """Hash of everything a score depends on. Same fingerprint = same inputs, so a re-score
    keeps the saved result instead of asking the AI again (and getting a slightly different number)."""
    up = user_profile or {}
    basis = {
        "v": SCORING_VERSION,
        "contact": facts,
        "me": {k: up.get(k) for k in ("job_title", "company_name", "products", "value_propositions", "target_industries",
                                     "target_roles", "target_company_sizes", "target_geographies")},
        "event": facts.get("event") or up.get("current_event_name"),
        "goals": event_identity.event_goals(up, facts.get("event")),
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()[:32]


def breakdown_for_storage(result: dict, fingerprint: str) -> dict:
    """The five dimensions + how the score was made (stored with the score, incl. "_meta")."""
    out = {k: v for k, v in (result.get("breakdown") or {}).items() if not str(k).startswith("_")}
    out["_meta"] = {
        "fingerprint": fingerprint, "version": SCORING_VERSION, "runs": result.get("runs", 1),
        "spread": result.get("spread", 0), "model": result.get("model"),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    return out


# ---------------------------------------------------------------- the admin console's own profile

# Admin vs rep AI — two separate worlds:
#   rep AI   (the app)           → the rep's own user_profiles row; scores on ContactDB.lead_*
#   admin AI (the admin console) → ONLY admin_sales_profile (who's pitching, what we sell, how we sell,
#                                  who we target); scores in admin_contact_scores. No rep profile is read.
ADMIN_SALES_ID = "console"
ADMIN_PROFILE_KEYS = (
    "company_name", "job_title",                                                          # who is pitching
    "products",                                                                           # what we sell
    "pitch_style", "value_propositions", "common_objections", "case_studies",             # how we sell
    "target_industries", "target_roles", "target_company_sizes", "target_geographies",   # who we target
)


async def load_admin_sales_profile(session) -> dict:
    """The admin console's own profile ({} until an admin saves it)."""
    row = (await session.execute(select(AdminSalesProfileDB).where(AdminSalesProfileDB.id == ADMIN_SALES_ID))).scalar_one_or_none()
    return dict(row.data or {}) if row else {}


async def admin_ai_profile(session) -> dict:
    """The profile every admin-side AI call works from (scoring, research, pitch, assistant): the admin
    console's own fields only — never a rep's profile, so admin results don't depend on any rep's data."""
    data = await load_admin_sales_profile(session)
    return {k: data[k] for k in ADMIN_PROFILE_KEYS if data.get(k)}


# ---------------------------------------------------------------- storage (one adapter per audience)

def _result(contact, score, temperature, reasoning, actions, breakdown, scored_at=None) -> dict:
    meta = (breakdown or {}).get("_meta") or {}
    return {
        "status": "success", "contact_id": str(contact.id), "name": contact.name, "score": score, "temperature": temperature,
        "reasoning": reasoning or "", "recommended_actions": actions or [],
        "breakdown": {k: v for k, v in (breakdown or {}).items() if not str(k).startswith("_")},
        "runs": meta.get("runs"), "spread": meta.get("spread"), "model": meta.get("model"),
        "scored_at": meta.get("scored_at") or scored_at,
    }


class RepScores:
    """The rep's own score: ContactDB.lead_* against the contact owner's profile."""

    async def profile(self, session, contact) -> dict:
        return event_identity.read_profile(await event_identity.load_profile(session, contact.user_id))

    async def saved(self, session, contact):
        return contact if contact.lead_score is not None else None

    def fingerprint(self, saved) -> Optional[str]:
        return ((saved.lead_score_breakdown or {}).get("_meta") or {}).get("fingerprint")

    def result(self, contact, saved) -> dict:
        return _result(contact, saved.lead_score, saved.lead_temperature, saved.lead_score_reasoning,
                       saved.lead_recommended_actions, saved.lead_score_breakdown)

    def previous(self, saved) -> Optional[int]:
        return saved.lead_score if saved is not None else None

    async def store(self, session, contact, result: dict, fingerprint: str, admin_id=None):
        contact.lead_score = result["score"]
        contact.lead_temperature = temperature_for(result["score"])
        contact.lead_score_reasoning = result.get("reasoning", "")
        contact.lead_score_breakdown = breakdown_for_storage(result, fingerprint)
        contact.lead_recommended_actions = result.get("recommended_actions", [])
        await contact_store.save(session, contact, event_type=None)  # the caller announces it
        return contact


class AdminScores:
    """The admin console's own score: admin_contact_scores against the admin sales profile only. ContactDB
    (the rep's score, notes, updated_at), the rep's profile and the search index are never touched."""

    async def profile(self, session, contact) -> dict:
        return await admin_ai_profile(session)

    async def saved(self, session, contact):
        return (await session.execute(select(AdminContactScoreDB).where(AdminContactScoreDB.contact_id == contact.id))).scalar_one_or_none()

    def fingerprint(self, saved) -> Optional[str]:
        return ((saved.breakdown or {}).get("_meta") or {}).get("fingerprint")

    def result(self, contact, saved) -> dict:
        return _result(contact, saved.score, saved.temperature, saved.reasoning, saved.recommended_actions, saved.breakdown,
                       saved.scored_at.isoformat() if saved.scored_at else None)

    def previous(self, saved) -> Optional[int]:
        return saved.score if saved is not None else None

    async def store(self, session, contact, result: dict, fingerprint: str, admin_id=None):
        row = await self.saved(session, contact)
        if row is None:
            row = AdminContactScoreDB(contact_id=contact.id)
            session.add(row)
        row.owner_id = contact.user_id
        row.score = result["score"]
        row.temperature = temperature_for(result["score"])
        row.reasoning = result.get("reasoning", "")
        row.breakdown = breakdown_for_storage(result, fingerprint)
        row.recommended_actions = result.get("recommended_actions", [])
        row.scored_by = uuid.UUID(str(admin_id)) if admin_id else None
        row.scored_at = datetime.now(timezone.utc)
        session.expunge(contact)  # the contact row is only read; make sure nothing on it is flushed by accident
        await session.commit()
        return row


STORES = {"rep": RepScores(), "admin": AdminScores()}


# ---------------------------------------------------------------- score

async def score(contact_id, audience: str = "rep", force: bool = False, owner: Optional[uuid.UUID] = None,
                admin_id=None, ai: Optional[Callable[[dict, dict], dict]] = None) -> dict:
    """Score one contact for `audience` ("rep" = the rep's own score, "admin" = the console's) and save it.
    `owner`: the contact must belong to this account (404 otherwise). force=True asks the AI even when
    nothing the score depends on changed."""
    store = STORES[audience]
    scorer = ai or ai_scorer
    factory = get_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="Database not available. Check DATABASE_URL environment variable.")
    async with factory() as session:
        try:
            try:
                cid = uuid.UUID(str(contact_id))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid contact ID")
            query = select(ContactDB).where(ContactDB.id == cid)
            if owner is not None:
                query = query.where(ContactDB.user_id == owner)
            contact = (await session.execute(query)).scalar_one_or_none()
            if contact is None:
                raise HTTPException(status_code=404, detail="Contact not found")

            profile = await store.profile(session, contact)
            facts = scoring_facts(contact)
            fingerprint = scoring_fingerprint(facts, profile)
            saved = await store.saved(session, contact)
            if not force and saved is not None and store.fingerprint(saved) == fingerprint:
                # Nothing it depends on changed: keep the score (asking again would just add noise).
                return {**store.result(contact, saved), "unchanged": True}

            if scorer is None:
                raise HTTPException(status_code=503, detail=NOT_AVAILABLE_DETAIL)
            try:
                result = await asyncio.wait_for(asyncio.to_thread(scorer, facts, profile), timeout=SCORE_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Scoring timed out — the AI service didn't answer in time. Please try again.")
            except ScoringUnavailable as e:
                # Nothing is saved: the contact stays "not scored" instead of getting a fake 50.
                print(f"[SCORING] {audience} score unavailable for {contact_id} ({e.reason}): {e}")
                raise HTTPException(status_code=503, detail=SCORING_UNAVAILABLE_MESSAGES.get(e.reason, SCORING_UNAVAILABLE_MESSAGES["unavailable"]))

            previous = store.previous(saved)
            stored = await store.store(session, contact, result, fingerprint, admin_id)
            return {**store.result(contact, stored), "unchanged": False, "previous_score": previous}
        except HTTPException:
            raise
        except Exception as e:
            await session.rollback()
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
