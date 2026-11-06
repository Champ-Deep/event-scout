"""
Floor map link + LLM fallback for event-site scraping (design note: card t_95919932, section 3 and 4).

  extract_floor_map(pages)  heuristic: find the floor map link (pdf / image / viewer / page) in fetched HTML
  needs_llm(exhibitors, fm) coverage < 0.5 -> fire the LLM (mirrors LakeStream's auto mode threshold)
  llm_extract(pages, ...)   one OpenRouter pass over the page text; every result is post-guarded against the
                            fetched text so a hallucinated URL or exhibitor never reaches the database

Pure functions plus one async LLM call. No DB, no network except ai_client.chat. `pages` are
event_scraper.FetchedPage objects (only .url, .final_url and .html are read).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import urljoin, urlparse

import ai_client

if TYPE_CHECKING:  # event_scraper.py lives on a sibling card
    from event_scraper import FetchedPage

LLM_COVERAGE_THRESHOLD = 0.5  # LakeStream _AUTO_AI_THRESHOLD
LLM_CHUNK_CHARS = 30_000
LLM_MAX_CALLS = 3

_VIEWER_HOSTS = ("expocad", "a2zinc", "a2z.", "map-dynamics", "mapdynamics", "cadmapper", "mapsindoors", "expofp", "mapyourshow")
_NOT_A_FLOOR_MAP = re.compile(r"google\.[a-z.]+/maps|goo\.gl/maps|maps\.app\.goo|openstreetmap|sitemap|site-map|[?&](booth|exhibitor)\w*=", re.I)  # last: a link to one booth, not the map
_STRONG = re.compile(r"(floor\s*[-_]?\s*(plan|map)|(site|hall|venue|exhibit(ion)?|show)\s*[-_]?\s*(plan|map|layout)|expo\s*(map|plan))s?(?![a-z])", re.I)  # not "Show Planner"
_WEAK = re.compile(r"\b(map|layout)\b", re.I)
_UI_ASSET = re.compile(r"(?<![a-z])(icons?|logos?|sprites?|favicons?|avatars?|badges?|sponsors?)(?![a-z])", re.I)
_IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


@dataclass
class FloorMap:
    url: str
    kind: str  # "pdf" | "image" | "viewer" | "page"
    source: str  # "heuristic" | "llm"
    label: str = ""


class LLMExtraction(TypedDict):
    exhibitors: list  # dicts shaped like event_scraper.ScrapedExhibitor
    floor_map_url: Optional[str]
    model: str


class _Page(HTMLParser):
    """Collects (tag, url, label) for a/iframe/embed/img and the visible text with anchors inlined as 'text [url]'."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links: List[Tuple[str, str, str]] = []
        self.parts: List[str] = []
        self._skip = 0
        self._a: Optional[Tuple[str, str, List[str]]] = None  # (url, label attrs, text parts)

    def _abs(self, u: str) -> str:
        u = (u or "").strip()
        if not u or u.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            return ""
        full = urljoin(self.base, u)
        return full if urlparse(full).scheme in ("http", "https") else ""

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag == "a" and self._abs(a.get("href", "")):
            self._a = (self._abs(a["href"]), f"{a.get('title', '')} {a.get('aria-label', '')}", [])
        elif tag in ("iframe", "embed", "img"):
            src = self._abs(a.get("src", "") or a.get("data-src", ""))
            if src:
                self.links.append((tag, src, f"{a.get('alt', '')} {a.get('title', '')}"))

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._a:
            url, attrs, text = self._a
            label = " ".join("".join(text).split())
            self.links.append(("a", url, f"{label} {attrs}".strip()))
            self.parts.append(f" [{url}] ")
            self._a = None

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._a:
            self._a[2].append(data)


def _parse(page: "FetchedPage") -> _Page:
    p = _Page(getattr(page, "final_url", "") or page.url)
    p.feed(page.html or "")
    return p


def _kind(tag: str, url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith(_IMG_EXT):
        return "image"
    if tag == "iframe" or any(h in urlparse(url).netloc.lower() for h in _VIEWER_HOSTS):
        return "viewer"
    return "page"


def _score(tag: str, url: str, label: str) -> int:
    if _NOT_A_FLOOR_MAP.search(url) or _NOT_A_FLOOR_MAP.search(label):
        return 0
    text = f"{label} {urlparse(url).path}"
    kind = _kind(tag, url)
    strong, weak = bool(_STRONG.search(text)), bool(_WEAK.search(text))
    if not (strong or weak or kind == "viewer" and any(h in url.lower() for h in _VIEWER_HOSTS)):
        return 0
    if weak and not strong and kind == "page":  # a bare "map" link to a normal page is too noisy
        return 0
    if kind == "image" and _UI_ASSET.search(text) or tag == "img" and not strong:  # icons, sponsor logos, "map" in an alt text
        return 0
    # a bare <img> ranks below a genuine strong link (e.g. "View Floor Plans" to a viewer); linked image maps (DAX) are anchors
    return (4 if strong else 1) + {"viewer": 4, "pdf": 3, "image": 2, "page": 0}[kind] + (1 if tag == "iframe" else 0) - (3 if tag == "img" else 0)


def extract_floor_map(pages: Iterable["FetchedPage"]) -> Optional[FloorMap]:
    """Best floor-map candidate across pages (earlier page wins ties), or None."""
    best: Optional[Tuple[int, FloorMap]] = None
    for page in pages:
        for tag, url, label in _parse(page).links:
            s = _score(tag, url, label)
            if s and (best is None or s > best[0]):
                best = (s, FloorMap(url=url, kind=_kind(tag, url), source="heuristic", label=label[:120]))
    return best[1] if best else None


def _field(row, name: str) -> str:
    return str(row.get(name, "") if isinstance(row, dict) else getattr(row, name, "") or "").strip()


def coverage(exhibitors: list, floor_map: Optional[FloorMap]) -> float:
    """found / 3: >= 5 exhibitors, half the rows carry booth/hall/category, a floor map link."""
    rich = sum(1 for e in exhibitors if _field(e, "booth") or _field(e, "hall") or _field(e, "category"))
    found = [len(exhibitors) >= 5, bool(exhibitors) and rich / len(exhibitors) >= 0.5, floor_map is not None]
    return sum(found) / 3


def needs_llm(exhibitors: list, floor_map: Optional[FloorMap]) -> Tuple[bool, str]:
    """(fire, reason). Fires on zero exhibitors or coverage below LLM_COVERAGE_THRESHOLD."""
    if not exhibitors:
        return True, "no_exhibitors"
    cov = coverage(exhibitors, floor_map)
    if cov < LLM_COVERAGE_THRESHOLD:
        return True, f"coverage={cov:.2f} exhibitors={len(exhibitors)} floor_map={'yes' if floor_map else 'no'}"
    return False, f"coverage={cov:.2f}"


_PROMPT = (
    "You read the text of an event / trade show website (links appear as 'text [url]'). Reply with ONE JSON object:\n"
    '{"floor_map_url": "", "exhibitors": [{"name": "", "booth": "", "hall": "", "category": "", "country": "", "website": "", "description": ""}]}\n'
    "Rules: use ONLY what is literally in the text. Never invent a name, booth, or URL. Use an empty string when a "
    "value is absent. floor_map_url is the link to the floor plan / floor map / hall layout (PDF, image or viewer), "
    "copied exactly from a [url] in the text. {scope}"
)


def _page_texts(pages: List["FetchedPage"]) -> Tuple[str, set, str]:
    """-> (text for the model, every absolute URL seen, lowercase plain text for the guards)."""
    chunks, urls, plain = [], set(), []
    for page in pages:
        p = _parse(page)
        urls.update(u for _, u, _ in p.links)
        t = " ".join("".join(p.parts).split())
        chunks.append(f"=== {page.url} ===\n{t}")
        plain.append(re.sub(r"\s*\[[^\]]*\]\s*", " ", t).lower())
    return "\n\n".join(chunks), urls, " ".join(plain)


async def llm_extract(pages: List["FetchedPage"], *, want_exhibitors: bool, want_floor_map: bool,
                      event: str = "", reason: str = "") -> LLMExtraction:
    """One LLM pass (max LLM_MAX_CALLS chunk calls). Raises ai_client.AIError; the service downgrades it to a warning."""
    text, urls, plain = _page_texts(pages)
    wanted = [w for w, on in (("exhibitors", want_exhibitors), ("floor_map_url", want_floor_map)) if on]
    scope = f"Only fill: {', '.join(wanted)}; leave the rest empty." if wanted else ""
    system = _PROMPT.replace("{scope}", scope)
    exhibitors, floor_url, model, seen = [], None, "", set()
    for i in range(0, min(len(text), LLM_CHUNK_CHARS * LLM_MAX_CALLS), LLM_CHUNK_CHARS):
        reply, model = await ai_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": text[i:i + LLM_CHUNK_CHARS]}],
            json_mode=True, max_tokens=3000, temperature=0.0, timeout=60, title="Event Scout Event Scrape")
        try:
            data = ai_client.parse_json(reply)
        except ValueError:  # malformed reply for this chunk: keep what other chunks gave
            continue
        u = str(data.get("floor_map_url") or "").strip() if want_floor_map else ""
        # post-guard: the URL must literally exist in what we fetched
        if not floor_url and urlparse(u).scheme in ("http", "https") and (u in urls or u in text):
            floor_url = u
        for row in (data.get("exhibitors") or []) if want_exhibitors and isinstance(data.get("exhibitors"), list) else []:
            if not isinstance(row, dict):
                continue
            name = " ".join(str(row.get("name") or "").split())
            if len(name) < 2 or name.lower() in seen or name.lower() not in plain:
                continue  # invented or duplicate
            seen.add(name.lower())
            clean = {k: " ".join(str(row.get(k) or "").split()) for k in ("booth", "hall", "category", "country", "description")}
            site = str(row.get("website") or "").strip()
            clean["website"] = site if site and site.lower() in plain + " " + " ".join(urls).lower() else ""
            exhibitors.append({"name": name, **clean, "profile_url": ""})
    print(f"[SCRAPE] llm fallback event={event} reason={reason} pages={len(pages)} model={model} "
          f"exhibitors={len(exhibitors)} floor_map={'yes' if floor_url else 'no'}")
    return {"exhibitors": exhibitors, "floor_map_url": floor_url, "model": model}
