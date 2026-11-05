"""Exhibitor-list scraper for event websites (fetch + parse; no DB, no LLM).

    await fetch_page(url)            one SSRF-guarded GET → FetchedPage (or ScrapeError)
    find_candidate_links(page)       where the exhibitor list / floor map / next page / JSON API probably are
    await crawl(url)                 home page + at most 6 pages in all → list[FetchedPage]
    extract_exhibitors(pages)        embedded JSON, then tables, then repeated cards → list[ScrapedExhibitor]
    await scrape_exhibitors(url)     crawl + extract_exhibitors
    is_js_shell(page)                True when the page is an empty JS app (caller decides: "needs_js")

Mirrors LakeStream's conventions: a fetch result with status/tier, escalate-on-blocked (here only reported as
ScrapeError("blocked"); the optional LakeStream tier belongs to the service), 1 retry on 429/5xx, a per-host
rate limit, and field-coverage-driven fallbacks (the LLM fallback lives in event_floormap, not here).
Design: comment 446 on kanban t_95919932.
"""

import asyncio
import ipaddress
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlsplit, urldefrag

import httpx
from bs4 import BeautifulSoup

MAX_PAGES = 6
_TIMEOUT = 15.0
_MAX_BYTES = 2_000_000
_MAX_REDIRECTS = 5
_RETRY_STATUS = {429, 502, 503, 504}
_RETRY_DELAY = 2.0
_HOST_GAP = 1.0  # seconds between requests to one host
_last_hit: Dict[str, float] = {}

# ExhibitorDB column limits (design note section 3)
_LIMITS = {"name": 500, "booth": 100, "hall": 100, "category": 500, "country": 100, "website": 500,
           "description": 2000, "profile_url": 500}

_NAV_WORDS = {
    "home", "about", "about us", "contact", "contact us", "login", "log in", "sign in", "register", "menu",
    "search", "exhibitors", "exhibitor list", "sponsors", "view all", "load more", "read more", "next",
    "previous", "prev", "privacy policy", "terms", "faq", "click here", "learn more", "more info",
    "visit website", "all", "show all", "more", "back", "close", "cookie policy",
}


class ScrapeError(Exception):
    """reason ∈ bad_url, blocked_url, unreachable, blocked, needs_js, no_content, timeout"""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason


@dataclass
class ScrapedExhibitor:
    name: str
    booth: str = ""
    hall: str = ""
    category: str = ""
    country: str = ""
    website: str = ""
    description: str = ""
    profile_url: str = ""


@dataclass
class FetchedPage:
    url: str
    final_url: str
    status: int
    html: str
    content_type: str
    tier: str = "http"  # "http" | "lakestream"


# ------------------------------------------------------------------------------------------------ fetching


async def _resolve(host: str) -> List[str]:
    """Every address `host` resolves to. (A seam: tests replace it.)"""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as e:
        raise ScrapeError("unreachable", f"cannot resolve {host}") from e
    return [str(i[4][0]) for i in infos]


def _public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast


async def _check_url(url: str) -> tuple:
    """Raise unless `url` is a plain http(s) URL whose host is public. Returns (host, ip): the caller must connect
    to that validated `ip` (see _get_once), so a second DNS answer can never redirect the request (DNS rebinding)."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as e:
        raise ScrapeError("bad_url", str(e)) from e
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ScrapeError("bad_url", "only http(s) URLs with a host")
    if parts.username or parts.password:
        raise ScrapeError("bad_url", "credentials in URL")
    if port not in (None, 80, 443):
        raise ScrapeError("bad_url", f"port {port} not allowed")
    host = parts.hostname.lower()
    try:
        ipaddress.ip_address(host)
        addrs = [host]
    except ValueError:
        addrs = await _resolve(host)
    if not addrs or not all(_public(a) for a in addrs):
        raise ScrapeError("blocked_url", f"{host} is not a public address")
    return host, min(addrs, key=lambda a: ":" in a)  # IPv4 first: many hosts have no IPv6 route


async def _throttle(host: str) -> None:
    wait = _last_hit.get(host, 0.0) + _HOST_GAP - time.monotonic()
    _last_hit[host] = time.monotonic() + max(wait, 0.0)
    if wait > 0:
        await asyncio.sleep(wait)


def _user_agent() -> str:
    return f"EventScoutBot/1.0 (+{os.environ.get('PUBLIC_BASE_URL', '').rstrip('/')})"


async def _get_once(client: httpx.AsyncClient, url: str, ip: str):
    """(status, headers, text) for one GET, body capped at 2 MB. No redirect following. Connects to the already
    validated `ip` with the original Host header and TLS server name, so no second DNS lookup happens."""
    parts = urlsplit(url)
    pinned = parts._replace(netloc=(f"[{ip}]" if ":" in ip else ip) + (f":{parts.port}" if parts.port else "")).geturl()
    try:
        async with client.stream("GET", pinned, headers={"Host": parts.netloc}, extensions={"sni_hostname": parts.hostname},
                                 follow_redirects=False) as resp:
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) > _MAX_BYTES:
                    del body[_MAX_BYTES:]
                    break
            try:
                text = bytes(body).decode(resp.charset_encoding or "utf-8", errors="replace")
            except LookupError:
                text = bytes(body).decode("utf-8", errors="replace")
            return resp.status_code, resp.headers, text
    except httpx.TimeoutException as e:
        raise ScrapeError("timeout", url) from e
    except httpx.HTTPError as e:
        raise ScrapeError("unreachable", f"{type(e).__name__}: {url}") from e


async def _get(client: httpx.AsyncClient, url: str, ip: str):
    status, headers, text = await _get_once(client, url, ip)
    if status in _RETRY_STATUS:
        await asyncio.sleep(_RETRY_DELAY)
        status, headers, text = await _get_once(client, url, ip)
    return status, headers, text


_BLOCK_PAGE = re.compile(r"captcha|cf-chl|just a moment|attention required|access denied", re.I)


async def fetch_page(url: str, *, client: Optional[httpx.AsyncClient] = None) -> FetchedPage:
    """GET `url` (an admin-supplied address, fetched from our server) following up to 5 redirects by hand so
    every hop is checked against the SSRF rules. Raises ScrapeError."""
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=_TIMEOUT, headers={
            "User-Agent": _user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5"})
    try:
        cur = url.strip()
        for _ in range(_MAX_REDIRECTS + 1):
            host, ip = await _check_url(cur)
            await _throttle(host)
            status, headers, text = await _get(client, cur, ip)
            if status in (301, 302, 303, 307, 308) and headers.get("location"):
                cur = urljoin(cur, headers["location"])
                continue
            break
        else:
            raise ScrapeError("unreachable", "too many redirects")
    finally:
        if own:
            await client.aclose()
    ctype = headers.get("content-type", "").split(";")[0].strip().lower()
    if status in (401, 403, 429, 503):
        raise ScrapeError("blocked", f"HTTP {status}")
    if status >= 500:
        raise ScrapeError("unreachable", f"HTTP {status}")
    if status >= 400 or status in (301, 302, 303, 307, 308):
        raise ScrapeError("no_content", f"HTTP {status}")
    if ctype and not re.search(r"html|json|xml|text/plain", ctype):
        raise ScrapeError("no_content", f"content-type {ctype}")
    if len(text) < 20000 and _BLOCK_PAGE.search(text):
        raise ScrapeError("blocked", "challenge page")
    return FetchedPage(url=url, final_url=cur, status=status, html=text, content_type=ctype)


# ------------------------------------------------------------------------------------------------ links


def _host(url: str) -> str:
    h = (urlsplit(url).hostname or "").lower()
    return ".".join(h.split(".")[-2:])  # shortcut: breaks on co.uk-style suffixes (same-host test only), upgrade with tldextract


def _soup(page: FetchedPage) -> BeautifulSoup:
    return BeautifulSoup(page.html, "html.parser")


def _abs(base: str, href: Optional[str]) -> str:
    """Absolute http(s) URL without fragment, or ""."""
    if not href:
        return ""
    u = urldefrag(urljoin(base, href.strip()))[0]
    return u if urlsplit(u).scheme in ("http", "https") else ""


_EXH_STRONG = re.compile(r"exhibitor|a-z list|directory|participants|companies|company list", re.I)
_EXH_WEAK = re.compile(r"exhibit|sponsor|brands", re.I)
_EXH_NEG = re.compile(r"become|why |book|apply|sign.?up|log.?in|register|prospectus|manual|\bkit\b|portal|"
                      r"enquir|contact|how to|benefit|sell|pricing|\bsite ?map\b|\.pdf$", re.I)
_MAP_RE = re.compile(r"floor|plan|map|layout", re.I)
_NEXT_TEXT = re.compile(r"^\s*(next( page)?|[>›»]+|older)\s*$", re.I)
_API_RE = re.compile(r"""["']((?:https?:)?//[^"'\s<>]+|/[^"'\s<>]+)["']""")


def find_candidate_links(page: FetchedPage) -> dict:
    """{"exhibitors": [...], "floor_map": [...], "next": url|None, "api": [...]}: same-site URLs only, best first."""
    soup = _soup(page)
    site = _host(page.final_url)
    scored: Dict[str, int] = {}
    maps: List[str] = []
    for a in soup.find_all("a", href=True):
        url = _abs(page.final_url, a["href"])
        if not url or _host(url) != site or url == page.final_url:
            continue
        label = f"{a.get_text(' ', strip=True)} {a.get('title', '')} {a.get('aria-label', '')}"
        path = urlsplit(url).path
        if _MAP_RE.search(label + " " + path) and not re.search(r"site-?map", label + path, re.I):
            maps.append(url)
        if _EXH_NEG.search(label + " " + path):
            continue
        score = 2 * bool(_EXH_STRONG.search(label)) + bool(_EXH_STRONG.search(path)) + \
            bool(_EXH_WEAK.search(label + " " + path))
        if score:
            scored[url] = max(score, scored.get(url, 0))
    exhibitors = sorted(scored, key=lambda u: -scored[u])[:6]
    nxt = None
    for el in soup.find_all(["link", "a"], href=True):
        rel = el.get("rel") or []
        txt = el.get_text(" ", strip=True) + " " + el.get("aria-label", "")
        cls = " ".join(el.get("class", []))
        if "next" in rel or _NEXT_TEXT.match(txt.strip()) or re.search(r"(^|[-_ ])next($|[-_ ])", cls):
            cand = _abs(page.final_url, el["href"])
            if cand and _host(cand) == site and cand != page.final_url:
                nxt = cand
                break
    api = []
    for m in _API_RE.finditer(page.html):
        u = _abs(page.final_url, m.group(1) if not m.group(1).startswith("//") else "https:" + m.group(1))
        if u and _host(u) == site and re.search(r"exhibitor|compan|participant", u, re.I) and \
                re.search(r"api|json|ajax", u, re.I) and u not in api:
            api.append(u)
    return {"exhibitors": exhibitors, "floor_map": list(dict.fromkeys(maps))[:6], "next": nxt, "api": api[:3]}


async def crawl(url: str, *, fetch: Callable = fetch_page, max_pages: int = MAX_PAGES) -> List[FetchedPage]:
    """The home page (errors propagate) plus the likeliest exhibitor-list pages, their `next` pages, a JSON API
    the page references and the floor-map page; sub-page failures are skipped. At most `max_pages` fetches."""
    home = await fetch(url)
    pages, seen = [home], {url, home.final_url}
    links = find_candidate_links(home)
    queue = deque([(u, True) for u in links["exhibitors"][:2] + links["api"][:1]] +
                  [(u, False) for u in links["floor_map"][:1]])
    if links["next"] and extract_exhibitors([home]):
        queue.appendleft((links["next"], True))
    while queue and len(pages) < max_pages:
        nxt, paginate = queue.popleft()
        if nxt in seen:
            continue
        seen.add(nxt)
        try:
            page = await fetch(nxt)
        except ScrapeError:
            continue
        seen.add(page.final_url)
        pages.append(page)
        if paginate:
            more = find_candidate_links(page)["next"]
            if more and more not in seen:
                queue.appendleft((more, True))
    return pages


async def scrape_exhibitors(url: str, *, fetch: Callable = fetch_page, event: str = "") -> List[ScrapedExhibitor]:
    return extract_exhibitors(await crawl(url, fetch=fetch), event)


def is_js_shell(page: FetchedPage) -> bool:
    """An empty client-rendered app: almost no visible text and a mount point or a 'enable JavaScript' notice."""
    soup = _soup(page)
    for t in soup(["script", "style", "noscript", "head"]):
        t.decompose()
    return len(soup.get_text(" ", strip=True)) < 200 and \
        bool(re.search(r'id=["\'](root|app|__next|__nuxt)["\']|enable javascript', page.html, re.I))


# ------------------------------------------------------------------------------------------------ normalizing


def _clean(value, field: str) -> str:
    return " ".join(str(value or "").split())[:_LIMITS[field]]


def _booth(text: str) -> str:
    return re.sub(r"^\s*(booth|stand)\s*(no\.?|number|#)?\s*[:#-]?\s*", "", text.strip(), flags=re.I)


def _row(raw: dict, base: str) -> Optional[ScrapedExhibitor]:
    name = _clean(raw.get("name"), "name")
    if len(name) < 2 or name.lower() in _NAV_WORDS or not re.search(r"[^\W\d_]", name):
        return None
    out = {f: _clean(raw.get(f), f) for f in ("category", "country", "description")}
    if re.search(r"[\d|]", out["country"]):  # a date / time / venue string read from a "location" element, not a country
        out["country"] = ""
    out["booth"] = _clean(_booth(str(raw.get("booth") or "")), "booth")
    out["hall"] = _clean(raw.get("hall"), "hall")
    for f in ("website", "profile_url"):
        out[f] = _abs(base, raw.get(f))[: _LIMITS[f]]
    return ScrapedExhibitor(name=name, **out)


def _key(name: str) -> str:
    return " ".join(name.lower().split())


# ------------------------------------------------------------------------------------------------ embedded JSON

_K = lambda k: re.sub(r"[_\- ]", "", str(k).lower())  # noqa: E731
_JSON_FIELDS = {
    "name": ("name", "companyname", "company", "exhibitorname", "exhibitor", "displayname", "title",
             "organization", "organizationname", "legalname"),
    "booth": ("booth", "boothnumber", "boothno", "boothnum", "boothid", "stand", "standnumber", "standno"),
    "hall": ("hall", "hallname", "hallnumber", "pavilion"),
    "category": ("category", "categories", "sector", "industry", "productcategory"),
    "country": ("country", "countryname"),
    "website": ("website", "websiteurl", "web", "homepage", "companywebsite"),
    "profile_url": ("profileurl", "detailurl", "detailsurl", "link", "permalink", "exhibitorurl", "slugurl"),
    "description": ("description", "shortdescription", "summary", "about", "bio", "tagline"),
}
_LIST_HINT = re.compile(r"exhibitor|compan|sponsor|participant|vendor|brand|member|organi[sz]ation|itemlist", re.I)


def _s(v) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, list):
        return ", ".join(x for x in (_s(i) for i in v) if x)
    if isinstance(v, dict):
        return _s(v.get("name") or v.get("title") or v.get("label"))
    return ""


def _json_row(d: dict, page_host: str) -> dict:
    d = d["item"] if isinstance(d.get("item"), dict) else d
    flat = {_K(k): v for k, v in d.items()}
    raw = {f: next((_s(flat[k]) for k in keys if _s(flat.get(k))), "") for f, keys in _JSON_FIELDS.items()}
    if not raw["country"] and isinstance(d.get("address"), dict):
        raw["country"] = _s(d["address"].get("addressCountry"))
    url = _s(flat.get("url"))  # ambiguous: the company's own site (ld+json) or its page on the event site
    key = "profile_url" if _host(url) in ("", page_host) else "website"
    if url and not raw[key]:
        raw[key] = url
    return raw


def _lists(node, path, out):
    """Collect (path, [dict rows]) for every list of dicts that looks like an exhibitor list; don't descend into
    an accepted list (its rows' own sub-lists, e.g. products, are not exhibitors)."""
    if isinstance(node, dict):
        for k, v in node.items():
            _lists(v, path + (str(k),), out)
    elif isinstance(node, list):
        rows = [x for x in node if isinstance(x, dict)]
        # Bare schema.org ListItems (name + url, no Organization inside) are breadcrumbs / sitelinks, not exhibitors.
        bare = all(r.get("@type") == "ListItem" and not isinstance(r.get("item"), dict) for r in rows)
        page_graph = any(r.get("@type") in ("WebSite", "WebPage") for r in rows)  # the site's own @graph: it describes itself
        if len(rows) >= 2 and len(rows) >= 0.6 * len(node) and not bare and not page_graph:
            def named(r):
                r = r["item"] if isinstance(r.get("item"), dict) else r
                return any(_s(v) for k, v in r.items() if _K(k) in _JSON_FIELDS["name"])
            has_booth = any(_K(k) in _JSON_FIELDS["booth"] for r in rows for k in r)
            org = sum(1 for r in rows if re.search(r"Organization|Corporation|LocalBusiness",
                                                   _s(r.get("@type") or (r.get("item") or {}).get("@type"))))
            if sum(map(named, rows)) >= 0.6 * len(rows) and \
                    (any(_LIST_HINT.search(p) for p in path) or has_booth or org >= 0.5 * len(rows)):
                out.append(rows)
                return
        for x in node:
            _lists(x, path, out)


def _json_blobs(page: FetchedPage) -> Iterable:
    head = page.html.lstrip()[:1]
    if "json" in page.content_type or head in ("{", "["):
        try:
            yield json.loads(page.html)
        except ValueError:
            pass
        return
    if "<script" not in page.html:
        return
    for s in _soup(page).find_all("script"):
        body = s.string or s.get_text()
        if s.get("type") in ("application/ld+json", "application/json") or s.get("id") == "__NEXT_DATA__":
            try:
                yield json.loads(body)
            except ValueError:
                pass
        else:
            m = re.search(r"window\.__[A-Z_]*STATE__\s*=\s*", body)
            if m:
                try:
                    yield json.JSONDecoder().raw_decode(body, m.end())[0]
                except ValueError:
                    pass


def _from_json(page: FetchedPage) -> List[dict]:
    found, host = [], _host(page.final_url)
    for blob in _json_blobs(page):
        lists: list = []
        _lists(blob, (), lists)
        found += [_json_row(r, host) for rows in lists for r in rows]
    return found


# ------------------------------------------------------------------------------------------------ tables

_COLS = {
    "name": re.compile(r"company|exhibitor|organi[sz]ation|participant|brand|^name$", re.I),
    "booth": re.compile(r"booth|stand", re.I),
    "hall": re.compile(r"hall|pavilion", re.I),
    "category": re.compile(r"categor|sector|industry|product", re.I),
    "country": re.compile(r"country|location", re.I),
    "website": re.compile(r"web|url|site", re.I),
}


def _links(el, base: str, site: str) -> dict:
    """The (profile on this site, website elsewhere) links inside `el`, first of each."""
    out = {}
    for a in ([el] if el.name == "a" and el.get("href") else []) + el.find_all("a", href=True):
        u = _abs(base, a["href"])
        if u:
            out.setdefault("profile_url" if _host(u) == site else "website", u)
    return out


def _from_tables(page: FetchedPage) -> List[dict]:
    rows, site = [], _host(page.final_url)
    for table in _soup(page).find_all("table"):
        trs = table.find_all("tr")
        if len(trs) < 3:
            continue
        head = [c.get_text(" ", strip=True) for c in trs[0].find_all(["th", "td"])]
        cols: Dict[str, int] = {}
        for i, h in enumerate(head):
            for field, rx in _COLS.items():
                if field not in cols and rx.search(h):
                    cols[field] = i
                    break
        if "name" not in cols:
            continue
        for tr in trs[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= max(cols.values()):
                continue
            raw = {f: cells[i].get_text(" ", strip=True) for f, i in cols.items()}
            link_cell = {**_links(cells[cols["name"]], page.final_url, site)}
            if "website" in cols:
                link_cell.update({"website": _links(cells[cols["website"]], page.final_url, site).get("website")
                                  or raw.get("website", "")})
            rows.append({**raw, **{k: v for k, v in link_cell.items() if v}})
    return rows


# ------------------------------------------------------------------------------------------------ repeated cards

_BOOTH_TXT = re.compile(r"\b(?:booth|stand)\s*(?:no\.?|number|#|:)?\s*([A-Za-z]{0,3}[-\s]?\d[\w\-/]*)", re.I)
_HALL_TXT = re.compile(r"\bhall\s*([A-Za-z0-9]+)", re.I)
_MENU_ROLES = {"menu", "menubar", "navigation"}
_CARD_TAGS = {"div", "li", "article", "section", "a"}
_SIGNAL = ("booth", "category", "country")  # not hall: venue halls / floor-plan tiles ("West Hall") carry a hall but name no company


def _evidenced(raws: List[dict], base: str) -> List[ScrapedExhibitor]:
    """The valid rows of one candidate list, or [] unless a fifth of them carry booth/category/country (not just
    the name again). A real exhibitor directory has such fields; bare name lists are menus, logo walls, sessions."""
    rows = [r for r in (_row(raw, base) for raw in raws) if r]
    return rows if sum(1 for r in rows if any(getattr(r, f) and getattr(r, f) != r.name for f in _SIGNAL)) >= 0.2 * len(rows) else []


def _cls(rx: str):
    pat = re.compile(rx, re.I)
    return lambda c: bool(c) and bool(pat.search(c))


def _card(el, page: FetchedPage, site: str) -> dict:
    def text_of(rx):
        found = [t.get_text(" ", strip=True) for t in el.find_all(class_=_cls(rx))]
        return ", ".join(dict.fromkeys(t for t in found if t))
    head = el.find(re.compile(r"^h[1-6]$")) or el.find(class_=_cls(r"(^|[-_])(name|title|company)([-_]|$)"))
    name = head.get_text(" ", strip=True) if head else ""
    if not name:
        a = el.find("a")
        name = (a or el).get_text(" ", strip=True) if (a or not el.find(True)) else ""
    txt = el.get_text(" ", strip=True)
    booth = text_of(r"booth|stand") or (m.group(1) if (m := _BOOTH_TXT.search(txt)) else "")
    hall = text_of(r"hall") or (f"Hall {m.group(1)}" if (m := _HALL_TXT.search(txt)) else "")
    p = el.find("p")
    desc = text_of(r"desc|summary|excerpt|about") or (p.get_text(" ", strip=True) if p else "")
    raw = {"name": name, "booth": booth, "hall": hall, "category": text_of(r"categor|sector|industry|(^|[-_])tags?([-_]|$)"),
           "country": text_of(r"country|location"), "description": desc}
    raw.update(_links(el, page.final_url, site))
    return raw


def _from_cards(page: FetchedPage, min_cards: int = 5) -> List[dict]:
    """The largest group of `min_cards`+ look-alike sibling blocks (same tag and classes) outside nav/header/footer whose
    members read like cards: short, each with a name."""
    soup, site, best = _soup(page), _host(page.final_url), []
    for parent in soup.find_all(True):
        if parent.name in ("nav", "aside", "header", "footer", "select", "table", "thead", "tbody", "tr", "head", "script") or \
                parent.find_parent(["nav", "aside", "header", "footer"]) or \
                any(t.get("role") in _MENU_ROLES for t in (parent, *parent.parents)):  # div-built menus and sidebars
            continue
        groups: Dict[tuple, list] = {}
        for c in parent.find_all(True, recursive=False):
            if c.name in _CARD_TAGS:
                groups.setdefault((c.name, tuple(sorted(c.get("class", [])))), []).append(c)
        for els in groups.values():
            if len(els) < min_cards or len(els) <= len(best):
                continue
            if sum(len(e.get_text(" ", strip=True)) for e in els) / len(els) > 500:
                continue
            # Help-centre / FAQ sections: most blocks contain a question, which exhibitor cards rarely do.
            if sum("?" in e.get_text() for e in els) >= 0.5 * len(els):
                continue
            rows = [_card(e, page, site) for e in els]
            if len(_evidenced(rows, page.final_url)) >= 0.8 * len(els):
                best = rows
    return best


# ------------------------------------------------------------------------------------------------ public


def extract_exhibitors(pages: List[FetchedPage], event: str = "") -> List[ScrapedExhibitor]:
    """Per page the first method whose rows pass _evidenced (embedded JSON, tables with a company column, repeated
    cards); rows from all pages merged and de-duplicated by lower-cased single-spaced name (first seen wins). A row
    named like the event itself (`event`, years ignored) is the site's own title, not an exhibitor."""
    own = _key(re.sub(r"\b\d{4}\b", "", event))
    out, seen, cards = [], set(), False
    for page in pages:
        rows = _evidenced(_from_json(page), page.final_url) or _evidenced(_from_tables(page), page.final_url)
        if not rows and "json" not in page.content_type:
            # a stray group of 3-4 blocks is a teaser strip or tile row, not a directory; only a directory's short last page may be that small
            rows = _evidenced(_from_cards(page, 3 if cards else 5), page.final_url)
            cards = cards or bool(rows)
        for row in rows:
            if _key(row.name) not in seen and (not own or _key(re.sub(r"\b\d{4}\b", "", row.name)) != own):
                seen.add(_key(row.name))
                out.append(row)
    return out
