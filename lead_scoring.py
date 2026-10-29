"""Deterministic lead scoring for Event Scout.

OLD LOGIC (score_contact_with_gemini in app.py, removed): one Gemini prompt asked for five 0-20
sub-scores. Failure mode: two of the five (engagement, timing) had no input data, so the model
invented them; the same contact scored differently run to run; any timeout, bad JSON or missing
API key returned a fake 50/warm that hid the failure and tied every affected lead; the returned
score was never range-checked.

NEW MODEL: pure function of (contact, profile), score 0-100. Points are earned per dimension and
a dimension is only counted when its input exists, so a missing profile field never lowers a
score and never gets guessed:

  contactability  max 25  email 10, LinkedIn 8, phone 5, full name 2          (always counted)
  engagement      max 20  notes >=40 chars 12 / any notes 6; source qr 8, scan 5  (always counted)
  role_match      max 25  a profile target role appears in name/notes          (needs target_roles)
  industry_match  max 20  a target industry appears in company/notes           (needs target_industries)
  geography_match max 10  a target geography appears in notes                  (needs target_geographies)

score = round(100 * earned / counted_max). Temperature: hot >= 70, warm >= 40, else cold.
Job title is not a stored contact field, so role matching reads the notes text.
shortcut: substring matching, not NLP. Upgrade when contacts store a structured job title.
"""
import re

HOT, WARM = 70, 40
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _has(value) -> bool:
    return bool(value) and str(value).strip().upper() != "N/A"


def _mentions(targets, text: str) -> bool:
    return any(str(t).strip().lower() in text for t in targets or [] if str(t).strip())


def score_contact(contact: dict, profile: dict) -> dict:
    contact, profile = contact or {}, profile or {}
    notes = str(contact.get("notes") or "").strip()
    notes_l = notes.lower()
    company = str(contact.get("company_name") or "").lower() if _has(contact.get("company_name")) else ""
    name = str(contact.get("name") or "").lower()
    email = str(contact.get("email") or "").strip()
    phone_digits = re.sub(r"\D", "", str(contact.get("phone") or ""))

    contactability = (
        (10 if _EMAIL.match(email) else 0)
        + (8 if _has(contact.get("linkedin")) else 0)
        + (5 if len(phone_digits) >= 7 else 0)
        + (2 if len(name.split()) >= 2 else 0)
    )
    engagement = (12 if len(notes) >= 40 else 6 if notes else 0) + {"qr": 8, "scan": 5}.get(contact.get("source"), 0)

    # (key, earned, max, counted)
    dims = [
        ("contactability", contactability, 25, True),
        ("engagement", engagement, 20, True),
        ("role_match", 25 if _mentions(profile.get("target_roles"), f"{name} {notes_l}") else 0, 25,
         bool(profile.get("target_roles"))),
        ("industry_match", 20 if _mentions(profile.get("target_industries"), f"{company} {notes_l}") else 0, 20,
         bool(profile.get("target_industries"))),
        ("geography_match", 10 if _mentions(profile.get("target_geographies"), notes_l) else 0, 10,
         bool(profile.get("target_geographies"))),
    ]
    counted = [d for d in dims if d[3]]
    score = round(100 * sum(d[1] for d in counted) / sum(d[2] for d in counted))
    temperature = "hot" if score >= HOT else "warm" if score >= WARM else "cold"

    breakdown = {k: earned for k, earned, _, ok in dims if ok}
    gaps = [k for k, earned, mx, ok in dims if ok and earned < mx]
    actions = []
    if not _EMAIL.match(email):
        actions.append("Find a valid email address")
    if not _has(contact.get("linkedin")):
        actions.append("Find the LinkedIn profile")
    if not notes:
        actions.append("Add notes from the conversation")
    actions.append({"hot": "Follow up within 24 hours", "warm": "Send a personalised follow-up this week",
                    "cold": "Add to the nurture list"}[temperature])
    return {
        "score": score,
        "temperature": temperature,
        "reasoning": f"{score}/100 from {', '.join(breakdown)}."
                     + (f" Below max on: {', '.join(gaps)}." if gaps else ""),
        "recommended_actions": actions[:3],
        "breakdown": breakdown,
    }
