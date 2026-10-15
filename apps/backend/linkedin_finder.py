"""
Find a contact's LinkedIn profile from what we know about them (name, company, job title, email
domain), using live web search through OpenRouter.

Why it's built this way: an older AI-only lookup (see the disabled block in app.py's card scan)
invented profile links. Here the model never supplies a URL we trust — a profile is only offered
if it appears in the real search results the search plugin returned, and the match score is
computed by us from those results, not by the model. Nothing is saved here: the user picks the
right person in the app, which then calls /api/contacts/{id}/linkedin/attach.

Photos: LinkedIn usually puts the profile picture in the public page's og:image. LinkedIn often
blocks automated page loads, so a photo is best-effort; the profile link always comes through.
"""
import asyncio
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import httpx
from PIL import Image, UnidentifiedImageError

import ai_client

# Words that say nothing about which company someone works at.
_COMPANY_NOISE = {
    "inc", "ltd", "llc", "llp", "plc", "corp", "corporation", "co", "company", "the", "and", "of", "group",
    "pvt", "private", "limited", "gmbh", "ag", "sa", "bv", "pte", "technologies", "technology", "solutions",
    "services", "global", "international", "india", "usa",
}
_TITLE_NOISE = {"of", "and", "the", "at", "for", "senior", "sr", "jr", "head", "lead", "team"}
_FREE_EMAIL = {
    "gmail", "googlemail", "yahoo", "outlook", "hotmail", "live", "msn", "icloud", "me", "aol", "proton",
    "protonmail", "rediffmail", "yandex", "zoho", "gmx", "mail", "hey",
}
_PHOTO_UA = "Mozilla/5.0 (compatible; EventScoutLinkPreview/1.0; +https://eventscout.app)"
MAX_PHOTO_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------- text helpers

def _norm(s: Optional[str]) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _tokens(s: Optional[str], noise=frozenset()) -> List[str]:
    return [t for t in _norm(s).split() if len(t) >= 2 and t not in noise]


def _present(v: Optional[str]) -> str:
    v = (v or "").strip()
    return "" if v.upper() == "N/A" else v


def canonical_profile_url(url: Optional[str]) -> Optional[str]:
    """https://www.linkedin.com/in/<slug> for any linkedin.com/in/… link (any country subdomain,
    query or trailing path); None for anything that isn't a personal profile."""
    if not url or not isinstance(url, str):
        return None
    v = url.strip()
    if not re.match(r"^https?://", v, re.I):
        v = "https://" + v
    try:
        p = urlparse(v)
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
        return None
    m = re.match(r"^/in/([^/?#]+)", p.path or "")
    if not m:
        return None
    slug = m.group(1).strip()
    if len(slug) < 2 or len(slug) > 150:
        return None
    return f"https://www.linkedin.com/in/{slug}"


def profile_from_post_url(url: Optional[str]) -> Optional[str]:
    """A LinkedIn post link names its author: linkedin.com/posts/<vanity>_<post-words>… ->
    their profile https://www.linkedin.com/in/<vanity>. For people whose profile page itself
    isn't in the search results but whose posts are."""
    if not url:
        return None
    try:
        p = urlparse(url if re.match(r"^https?://", url, re.I) else "https://" + url)
    except ValueError:
        return None
    host = (p.hostname or "").lower()
    if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
        return None
    m = re.match(r"^/posts/([A-Za-z0-9-]{3,100})_", p.path or "")
    return f"https://www.linkedin.com/in/{m.group(1)}" if m else None


def _post_author(title: str) -> str:
    """'Just wrapped our earnings call. | Satya Nadella' -> 'Satya Nadella'."""
    parts = [x.strip() for x in html.unescape(title or "").split("|") if x.strip()]
    return parts[-1] if len(parts) >= 2 and len(parts[-1]) <= 80 else ""


def _parse_result_title(title: str) -> Dict[str, str]:
    """'Jane Doe - VP Sales - Acme | LinkedIn' -> {name, headline}."""
    t = re.sub(r"\s*[|·]\s*LinkedIn.*$", "", html.unescape(title or ""), flags=re.I).strip()
    parts = [x.strip() for x in re.split(r"\s+[-–—]\s+", t) if x.strip()]
    if not parts:
        return {"name": "", "headline": ""}
    return {"name": parts[0], "headline": " · ".join(parts[1:])}


def _email_org(email: Optional[str]) -> str:
    """'jane@acme-corp.com' -> 'acme corp' (empty for gmail & co)."""
    e = _present(email)
    if "@" not in e:
        return ""
    domain = e.split("@", 1)[1].lower()
    root = domain.split(".")[0]
    return "" if root in _FREE_EMAIL else _norm(root)


# ---------------------------------------------------------------------------- matching

def score_candidate(person: Dict[str, str], name: str, headline: str, snippet: str) -> Dict[str, Any]:
    """How well a search result fits the contact. Returns {confidence 0-100, matched: [...]}."""
    first = _tokens(person.get("first"))
    last = _tokens(person.get("last"))
    cand_name = set(_tokens(name))
    hay = f" {_norm(name)} {_norm(headline)} {_norm(snippet)} "

    score, matched = 0, []
    first_hit = bool(first) and first[0] in cand_name
    last_hit = bool(last) and all(t in cand_name for t in last)
    if first_hit and (last_hit or not last):
        score += 50; matched.append("name")
    elif last_hit and first and any(t.startswith(first[0][0]) for t in cand_name):
        score += 35; matched.append("name")
    elif last_hit or first_hit:
        score += 15; matched.append("partial name")

    company = _tokens(person.get("company"), _COMPANY_NOISE)
    if company:
        hits = sum(1 for t in company if f" {t} " in hay)
        if hits and hits / len(company) >= 0.5:
            score += 30; matched.append("company")
    org = _email_org(person.get("email"))
    if "company" not in matched and org and (f" {org} " in hay or org.replace(" ", "") in hay.replace(" ", "")):
        score += 22; matched.append("email domain")

    title = _tokens(person.get("title"), _TITLE_NOISE)
    if title:
        hits = sum(1 for t in title if f" {t} " in hay)
        if hits and hits / len(title) >= 0.5:
            score += 15; matched.append("job title")
        elif hits:
            score += 6

    return {"confidence": min(score, 100), "matched": matched}


# ---------------------------------------------------------------------------- search

def _prompt(person: Dict[str, str]) -> str:
    lines = [f"Name: {person['first']} {person['last']}".strip()]
    if person.get("company"):
        lines.append(f"Company: {person['company']}")
    if person.get("title"):
        lines.append(f"Job title: {person['title']}")
    org = _email_org(person.get("email"))
    if org:
        lines.append(f"Work email domain: {person['email'].split('@', 1)[1]}")
    return (
        "Find this person's LinkedIn profile page (a linkedin.com/in/ link) with a web search.\n"
        + "\n".join(lines)
        + "\n\nReply with JSON only, no other text:\n"
        '{"candidates":[{"url":"https://www.linkedin.com/in/...","name":"...","headline":"their current role and company"}]}\n'
        "Only include linkedin.com/in/ profiles that appeared in your search results, best match first, at most 5. "
        'If you found none, reply {"candidates":[]}.'
    )


def people_search_url(person: Dict[str, str]) -> str:
    """LinkedIn's own people search for this person (the app's 'search on LinkedIn' fallback)."""
    q = " ".join(x for x in [person.get("first"), person.get("last"), person.get("company")] if x)
    return f"https://www.linkedin.com/search/results/people/?keywords={quote(q)}"


async def find_profiles(person: Dict[str, str], limit: int = 5) -> List[Dict[str, Any]]:
    """Candidates for `person` ({first, last, company, title, email}), best first.

    Each: {url, name, headline, confidence, matched, photoUrl|None}. Raises ai_client.AIError.
    """
    text, citations = await ai_client.web_search(_prompt(person), max_results=8)

    # What the model said about each profile (used for nicer labels only).
    model_info: Dict[str, Dict[str, str]] = {}
    try:
        for c in (ai_client.parse_json(text).get("candidates") or []):
            url = canonical_profile_url(c.get("url")) if isinstance(c, dict) else None
            if url:
                model_info.setdefault(url, {"name": str(c.get("name") or ""), "headline": str(c.get("headline") or "")})
    except (ValueError, json.JSONDecodeError, AttributeError):
        pass

    # Only profiles that are really in the search results.
    found: Dict[str, Dict[str, Any]] = {}
    for c in citations:
        url = canonical_profile_url(c.get("url"))
        if url:
            parsed = _parse_result_title(c.get("title", ""))
        else:  # a post by them still identifies their profile
            url = profile_from_post_url(c.get("url"))
            if not url:
                continue
            parsed = {"name": _post_author(c.get("title", "")), "headline": ""}
        info = model_info.get(url, {})
        # Labels only (the URL above came from the real results): the model's tidy version first.
        name = info.get("name") or parsed["name"]
        headline = info.get("headline") or parsed["headline"]
        snippet = c.get("content", "")[:1200]
        prev = found.get(url)
        if prev:  # the same profile cited twice (e.g. profile + a post): merge what each adds
            prev["_snippet"] += " " + snippet
            prev["name"] = prev["name"] or name
            prev["headline"] = prev["headline"] or headline
            continue
        found[url] = {"url": url, "name": name, "headline": headline, "_snippet": snippet}

    out = []
    for cand in found.values():
        fit = score_candidate(person, cand["name"], cand["headline"], cand["_snippet"])
        if not any(m in fit["matched"] for m in ("name", "partial name")):
            continue  # not this person
        out.append({
            "url": cand["url"],
            "name": cand["name"] or cand["url"].rsplit("/", 1)[-1],
            "headline": cand["headline"],
            "confidence": fit["confidence"],
            "matched": fit["matched"],
            "photoUrl": None,
        })
    out.sort(key=lambda c: c["confidence"], reverse=True)
    out = out[:limit]

    for c in out:
        c["photoSource"] = None
    if not out:
        return out

    # Photos, per candidate: 1) LinkedIn's own (remembered once found; skipped while LinkedIn is
    # refusing page loads), 2) Brave Image Search matched to the profile page, when configured.
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers={"User-Agent": _PHOTO_UA, "Accept-Language": "en"}) as client:
        # Top 3 only: every page load counts toward LinkedIn's limit.
        photos = await asyncio.gather(*(profile_photo_url(client, c["url"]) for c in out[:3]), return_exceptions=True)
    for c, p in zip(out[:3], photos):
        if isinstance(p, str):
            c["photoUrl"], c["photoSource"] = p, "linkedin"
    missing = [c["url"] for c in out if not c["photoUrl"]]
    if missing:
        found = await brave_profile_photos(person, missing)  # one image search for all of them
        for c in out:
            if not c["photoUrl"] and c["url"] in found:
                c["photoUrl"], c["photoSource"] = found[c["url"]], "brave"
    return out


# ---------------------------------------------------------------------------- photos

def _is_licdn(url: str) -> bool:
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = (p.hostname or "").lower()
    return p.scheme == "https" and (host == "licdn.com" or host.endswith(".licdn.com"))


# LinkedIn answers automated page loads with HTTP 999 once a network has made "too many"; keep
# hitting it and the block lasts longer. So: remember every photo found (LinkedIn photo URLs
# don't expire), and after a 999/429 stop asking for a while instead of retrying.
PHOTO_CACHE_TTL = 12 * 3600      # a found photo (or "this profile has none") is remembered this long
PHOTO_MISS_TTL = 3600
BLOCK_BACKOFF = 15 * 60          # pause page loads this long once LinkedIn refuses them all
BLOCK_AFTER = 4                  # …which we assume after this many refusals in a row
# LinkedIn refuses some profiles (often ones that aren't fully public) while serving others fine,
# so one refusal only skips that profile; only a run of them pauses everything.
_photo_cache: Dict[str, Tuple[float, Optional[str]]] = {}
_blocked_until = 0.0
_refusals_in_a_row = 0


def photos_limited() -> bool:
    """True while LinkedIn is refusing our page loads (photos can't be looked up right now)."""
    return time.time() < _blocked_until


async def profile_photo_url(client: httpx.AsyncClient, profile_url: str) -> Optional[str]:
    """The profile picture URL from the public page's og:image, or None (blocked / no photo)."""
    global _blocked_until, _refusals_in_a_row
    url = canonical_profile_url(profile_url)
    if not url:
        return None
    now = time.time()
    hit = _photo_cache.get(url)
    if hit and hit[0] > now:
        return hit[1]
    if photos_limited():
        return None
    try:
        r = await client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code in (999, 429):
        _refusals_in_a_row += 1
        _photo_cache[url] = (time.time() + PHOTO_MISS_TTL, None)  # don't ask for this profile again soon
        if _refusals_in_a_row >= BLOCK_AFTER:
            _blocked_until = time.time() + BLOCK_BACKOFF
            _refusals_in_a_row = 0
            print(f"[LINKEDIN] {BLOCK_AFTER} page loads refused in a row; pausing photo lookups for {BLOCK_BACKOFF // 60} min")
        return None
    _refusals_in_a_row = 0
    final_host = (r.url.host or "").lower()
    if r.status_code != 200 or not (final_host == "linkedin.com" or final_host.endswith(".linkedin.com")):
        return None
    photo = _photo_from_page(r.text)
    _photo_cache[url] = (time.time() + (PHOTO_CACHE_TTL if photo else PHOTO_MISS_TTL), photo)
    return photo


def _photo_from_page(text: str) -> Optional[str]:
    # The whole page: LinkedIn's <head> carries ~800 KB of inline styles/scripts before the
    # og:image tag (reading only the first 300 KB missed every photo).
    page = text[:4_000_000]
    m = None
    for prop in ("og:image", "twitter:image"):
        m = (re.search(rf'<meta[^>]+(?:property|name)=["\']{prop}["\'][^>]*content=["\']([^"\']+)', page, re.I)
             or re.search(rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']{prop}["\']', page, re.I))
        if m:
            break
    if not m:
        return None
    img = html.unescape(m.group(1)).strip()
    # LinkedIn's generic logo / grey silhouette isn't a photo of the person.
    if not _is_licdn(img) or "static.licdn.com" in img or "ghost" in img.lower():
        return None
    return img


# ---------------------------------------------------------------------------- other photo sources
# LinkedIn often refuses page loads (above), so photos also come from:
#  - Gravatar: the photo registered to the contact's email (official, free, no key).
#  - Brave Image Search (official API, BRAVE_SEARCH_API_KEY): image results whose source page is
#    the candidate's LinkedIn profile — the profile photo, without loading LinkedIn at all.

BRAVE_SEARCH_API_KEY = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
_gravatar_cache: Dict[str, Tuple[float, Optional[str]]] = {}


def brave_configured() -> bool:
    return bool(BRAVE_SEARCH_API_KEY)


async def gravatar_url(email: Optional[str]) -> Optional[str]:
    """The Gravatar photo for this email, or None when they haven't set one (d=404)."""
    e = _present(email).strip().lower()
    if "@" not in e or len(e) > 254:
        return None
    url = f"https://www.gravatar.com/avatar/{hashlib.sha256(e.encode()).hexdigest()}?s=400&d=404"
    hit = _gravatar_cache.get(url)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            r = await client.get(url)
    except httpx.HTTPError:
        return None
    found = url if r.status_code == 200 and (r.headers.get("content-type") or "").startswith("image/") else None
    if r.status_code in (200, 404):  # don't remember errors
        _gravatar_cache[url] = (time.time() + (PHOTO_CACHE_TTL if found else PHOTO_MISS_TTL), found)
    return found


async def brave_profile_photos(person: Dict[str, str], profile_urls: List[str]) -> Dict[str, str]:
    """{profile_url: image_url} from Brave Image Search, matched by each image's source page."""
    if not BRAVE_SEARCH_API_KEY or not profile_urls:
        return {}
    wanted = set(profile_urls)
    q = " ".join(x for x in [person.get("first"), person.get("last"), person.get("company"), "linkedin"] if x)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/images/search",
                params={"q": q, "count": 50, "safesearch": "strict"},
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_SEARCH_API_KEY},
            )
        if r.status_code != 200:
            print(f"[LINKEDIN] Brave image search HTTP {r.status_code}: {r.text[:200]}")
            return {}
        results = (r.json() or {}).get("results") or []
    except (httpx.HTTPError, ValueError) as e:
        print(f"[LINKEDIN] Brave image search failed: {e}")
        return {}
    out: Dict[str, str] = {}
    for res in results:
        page = canonical_profile_url((res or {}).get("url") or (res or {}).get("source"))
        if not page or page not in wanted or page in out:
            continue
        # Brave's own thumbnail proxy always loads on the phone; the original is on LinkedIn's CDN.
        img = ((res.get("thumbnail") or {}).get("src")) or ((res.get("properties") or {}).get("url"))
        if img and _allowed_image(img):
            out[page] = img
    return out


async def find_profile_photo(profile_url: str) -> Optional[str]:
    """profile_photo_url with its own client (for saving a profile whose photo wasn't looked up
    during the search, or refreshing the photo of an already-linked contact)."""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers={"User-Agent": _PHOTO_UA, "Accept-Language": "en"}) as client:
        return await profile_photo_url(client, profile_url)


def _allowed_image(url: str) -> bool:
    """Photos are only ever downloaded from these image hosts (never an arbitrary URL)."""
    try:
        p = urlparse(url)
    except ValueError:
        return False
    host = (p.hostname or "").lower()
    return p.scheme == "https" and (
        host == "licdn.com" or host.endswith(".licdn.com")            # LinkedIn
        or host in ("gravatar.com", "www.gravatar.com", "secure.gravatar.com")
        or host == "imgs.search.brave.com"                             # Brave image thumbnails
    )


async def download_photo(url: str) -> Optional[bytes]:
    """Image bytes from an allowed image host (LinkedIn, Gravatar, Brave), or None."""
    if not url or not _allowed_image(url):
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False, headers={"User-Agent": _PHOTO_UA}) as client:
            r = await client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or len(r.content) > MAX_PHOTO_BYTES:
        return None
    try:
        Image.open(BytesIO(r.content)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None
    return r.content
