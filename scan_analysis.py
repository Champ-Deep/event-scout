"""Scan-to-score analysis: deterministic score from the user's own profile, local-model reasoning.

Score and breakdown come from rules (a 1.5B model cannot be trusted with numbers).
The reasoning text is written by a local Ollama model when one is reachable, from the
user's profile plus the signals below; otherwise it is composed from the same signals.
No cloud key is needed. Stdlib only.
"""
import json
import os
import re
import urllib.request

LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:1.5b-instruct")
LOCAL_LLM_TIMEOUT = float(os.environ.get("LOCAL_LLM_TIMEOUT", "45"))

_STOP = {"and", "the", "for", "with", "inc", "ltd", "llc", "pvt", "corp", "company", "group", "services", "solutions"}


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if t not in _STOP}


def _match(targets, haystack: set) -> list:
    """Targets (phrases from the user's profile) that share a significant word with the contact text."""
    return [t for t in (targets or []) if isinstance(t, str) and _tokens(t) & haystack]


def _has(v) -> bool:
    return bool(v) and str(v).strip().upper() != "N/A"


def score_signals(contact: dict, profile: dict) -> dict:
    """Return score, temperature, breakdown (5 x 0-20, same keys as the cloud scorer) and matched signals."""
    notes = contact.get("notes", "") if _has(contact.get("notes")) else ""
    company = contact.get("company_name", "") if _has(contact.get("company_name")) else ""
    email = contact.get("email", "") if _has(contact.get("email")) else ""
    domain = email.split("@")[-1] if "@" in email else ""
    text = _tokens(f"{company} {notes} {domain}")
    notes_only = _tokens(notes)

    roles = _match(profile.get("target_roles"), notes_only)
    industries = _match(profile.get("target_industries"), text)
    offers = _match((profile.get("products") or []) + (profile.get("value_propositions") or []), text)
    goals = _match(profile.get("event_goals"), text)

    breakdown = {
        "profile_fit": 20 if offers else (10 if not profile.get("products") else 6),
        "role_relevance": 20 if roles else (8 if not notes else 4),
        "company_fit": 20 if industries else 6,
        "engagement": 8 * bool(email) + 6 * _has(contact.get("phone")) + 6 * _has(contact.get("linkedin")),
        "timing": min(20, (10 if profile.get("current_event_name") else 5) + (10 if goals else 0)),
    }
    score = sum(breakdown.values())
    return {
        "score": score,
        "temperature": "hot" if score >= 70 else "warm" if score >= 40 else "cold",
        "breakdown": breakdown,
        "signals": {"roles": roles, "industries": industries, "offers": offers, "goals": goals,
                    "notes_missing": not notes, "reachable": [k for k in ("email", "phone", "linkedin") if _has(contact.get(k))]},
    }


def compose_reasoning(contact: dict, profile: dict, r: dict) -> str:
    """Rules-composed reasoning. Every clause quotes something the user entered or something scanned."""
    s, who = r["signals"], contact.get("name") if _has(contact.get("name")) else "This contact"
    co = contact.get("company_name") if _has(contact.get("company_name")) else None
    me = profile.get("company_name") or "your company"
    parts = []
    if s["offers"]:
        parts.append(f"{who}{' at ' + co if co else ''} touches what {me} sells ({', '.join(s['offers'][:2])}).")
    elif profile.get("products"):
        parts.append(f"Nothing on the card links {who}{' at ' + co if co else ''} to {', '.join(profile['products'][:2])}.")
    else:
        parts.append(f"Your profile lists no products, so fit for {me} could not be judged.")
    if s["roles"]:
        parts.append(f"The role matches your target buyer ({', '.join(s['roles'][:2])}).")
    elif s["notes_missing"]:
        parts.append("No role was captured, so add a title in the notes to judge the buyer match.")
    else:
        parts.append(f"The role does not match your target roles ({', '.join((profile.get('target_roles') or ['none set'])[:2])}).")
    if s["industries"]:
        parts.append(f"The industry fits your focus on {', '.join(s['industries'][:2])}.")
    ev = profile.get("current_event_name")
    parts.append(f"Reachable by {', '.join(s['reachable']) or 'no channel yet'}" + (f", worth a follow-up from {ev}." if ev else "."))
    return " ".join(parts)


def local_reasoning(contact: dict, profile: dict, r: dict):
    """Ask the local Ollama model for the write-up. Returns None when it is unreachable or empty."""
    prompt = (
        "Write 2 or 3 plain sentences explaining this lead score to the salesperson. Use only the facts below, "
        "name their company and what they sell, and say what to do next. No lists, no dashes.\n"
        f"Salesperson company: {profile.get('company_name', 'unknown')}; sells: {profile.get('products', [])}; "
        f"targets roles {profile.get('target_roles', [])}, industries {profile.get('target_industries', [])}; "
        f"event: {profile.get('current_event_name', 'none')}.\n"
        f"Scanned contact: {contact.get('name')} at {contact.get('company_name')}, notes: {contact.get('notes') or 'none'}.\n"
        f"Score {r['score']}/100 ({r['temperature']}); breakdown {r['breakdown']}; matches {r['signals']}."
    )
    body = json.dumps({"model": LOCAL_LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 160}}).encode()
    try:
        req = urllib.request.Request(f"{LOCAL_LLM_URL}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=LOCAL_LLM_TIMEOUT) as resp:
            text = (json.loads(resp.read()).get("response") or "").strip()
    except Exception:
        return None
    return text.replace("\u2014", ",").replace("\u2013", ",") or None


def analyze(contact: dict, profile: dict) -> dict:
    r = score_signals(contact, profile or {})
    text = local_reasoning(contact, profile or {}, r)
    r["reasoning"] = text or compose_reasoning(contact, profile or {}, r)
    r["engine"] = f"ollama:{LOCAL_LLM_MODEL}" if text else "rules"
    return r
