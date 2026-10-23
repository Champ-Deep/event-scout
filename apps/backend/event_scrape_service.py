"""
Server-side scrape of an event's website (design note: card t_95919932, sections 3 to 6).

  scrape_event_site(url)   pure: crawl, extract, LLM fallback; no DB. fetch / llm are injectable for tests.
  run_scrape(event_id)     loads the ManagedEventDB, scrapes, writes ExhibitorDB rows + floor map + status.
  kick_event_scrape(id)    fire-and-forget run_scrape for the planner routes; never raises into the request.

Scraped rows are ordinary ExhibitorDB rows (source="scrape", added_by NULL), so every exhibitor / booth
consumer (GET /api/exhibitors, event page, chat context, admin Exhibitors page) reads them unchanged.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Optional

from sqlalchemy import delete, select

import ai_client
import database
import event_floormap
import event_scraper
from event_identity import event_filter, event_key
from event_scraper import ScrapedExhibitor, ScrapeError

COOLDOWN = timedelta(seconds=60)  # after a finished scrape, automatic kicks wait; the manual button does not
ENABLED = os.getenv("EVENT_SCRAPE_ENABLED", "true").lower() not in ("0", "false", "no", "off")

# api_routes sets this to broadcast_live_event (it imports this module, so the reverse import would be circular)
notify: Optional[Callable[[str, dict], Awaitable[None]]] = None

_locks: dict = {}
_tasks: set = set()


@dataclass
class ScrapeResult:
    exhibitors: List[ScrapedExhibitor] = field(default_factory=list)
    floor_map: Optional[event_floormap.FloorMap] = None
    exhibitors_url: Optional[str] = None
    method: str = "none"  # heuristic | llm | mixed | none
    pages_fetched: int = 0
    coverage: float = 0.0
    llm_used: bool = False
    model: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None  # a ScrapeError reason code (+ message)


def _scraped(row: dict) -> ScrapedExhibitor:
    return ScrapedExhibitor(**{f.name: str(row.get(f.name) or "") for f in fields(ScrapedExhibitor)})


async def scrape_event_site(url: str, *, event: str = "", fetch=None, llm=None, use_llm: bool = True) -> ScrapeResult:
    fetch, llm = fetch or event_scraper.fetch_page, llm or event_floormap.llm_extract  # looked up late so tests can swap them
    res = ScrapeResult()
    try:
        pages = await event_scraper.crawl(url, fetch=fetch)
    except ScrapeError as e:
        res.error = str(e)
        return res
    res.pages_fetched = len(pages)
    res.exhibitors = event_scraper.extract_exhibitors(pages)
    res.floor_map = event_floormap.extract_floor_map(pages)
    res.exhibitors_url = next((p.final_url for p in pages if event_scraper.extract_exhibitors([p])), None)
    heuristic, llm_added = bool(res.exhibitors or res.floor_map), False

    fire, reason = event_floormap.needs_llm(res.exhibitors, res.floor_map)
    if not heuristic and event_scraper.is_js_shell(pages[0]):
        res.error = "needs_js: the site renders in the browser and no JS fetcher is configured"
    elif fire and use_llm:
        if not ai_client.ai_configured:
            res.warnings.append("ai_not_configured")
        else:
            try:
                got = await llm(pages, want_exhibitors=len(res.exhibitors) < 5, want_floor_map=res.floor_map is None,
                                event=event, reason=reason)
                res.llm_used, res.model = True, got["model"]
                have = {event_key(x.name) for x in res.exhibitors}
                added = [x for x in map(_scraped, got["exhibitors"]) if event_key(x.name) not in have]
                res.exhibitors += added  # heuristic rows win on a name clash
                if res.floor_map is None and got["floor_map_url"]:
                    u = got["floor_map_url"]
                    res.floor_map = event_floormap.FloorMap(u, event_floormap._kind("a", u), "llm")
                    llm_added = True
                llm_added = llm_added or bool(added)
            except ai_client.AIError as e:
                res.warnings.append(f"llm_failed: {e.reason}")
    res.method = ("mixed" if heuristic else "llm") if llm_added else "heuristic" if heuristic else "none"
    res.coverage = event_floormap.coverage(res.exhibitors, res.floor_map)
    if not (res.exhibitors or res.floor_map) and not res.error:
        res.error = "no_content: no exhibitor list or floor map found"
    return res


def _status(res: ScrapeResult) -> str:
    if res.error and not (res.exhibitors or res.floor_map):
        return "failed"
    return "ok" if res.exhibitors and (res.floor_map or len(res.exhibitors) >= 5) else "partial"


def _aware(dt):  # SQLite hands datetimes back naive
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def run_scrape(event_id: uuid.UUID, *, manual: bool = False, fetch=None, llm=None) -> dict:
    """Scrape one event and save the outcome. A failed scrape records the error and leaves old data alone."""
    lock = _locks.setdefault(event_id, asyncio.Lock())
    if lock.locked():
        return {"status": "running"}
    async with lock:
        factory = database.get_session_factory()
        async with factory() as s:
            e = (await s.execute(select(database.ManagedEventDB).where(database.ManagedEventDB.id == event_id))).scalar_one_or_none()
            if e is None or not e.website:
                return {"status": "skipped"}
            if not manual and e.scraped_at and datetime.now(timezone.utc) - _aware(e.scraped_at) < COOLDOWN:
                return {"status": "skipped"}
            name, url = e.name, e.website
            e.scrape_status = "running"
            await s.commit()

        t0 = time.monotonic()
        try:
            res = await scrape_event_site(url, event=name, fetch=fetch, llm=llm)
        except Exception as ex:  # a bug in a parser must not leave the event stuck on "running"
            res = ScrapeResult(error=f"unexpected: {type(ex).__name__}")

        async with factory() as s:
            e = (await s.execute(select(database.ManagedEventDB).where(database.ManagedEventDB.id == event_id))).scalar_one()
            status = _status(res)
            if status != "failed":
                await _save_rows(s, e.name, res.exhibitors)
                if res.floor_map and (not e.floor_map_url or e.floor_map_url == (e.scrape_stats or {}).get("floor_map_url_scraped")):
                    e.floor_map_url = res.floor_map.url[:500]
                e.exhibitors_url = (res.exhibitors_url or e.exhibitors_url or "")[:500]
            e.scrape_stats = {
                "pages": res.pages_fetched, "exhibitors": len(res.exhibitors),
                "with_booth": sum(1 for x in res.exhibitors if x.booth or x.hall), "coverage": round(res.coverage, 2),
                "llm_used": res.llm_used, "model": res.model, "duration_ms": int((time.monotonic() - t0) * 1000),
                "floor_map_kind": res.floor_map.kind if res.floor_map else None,
                "floor_map_url_scraped": res.floor_map.url[:500] if res.floor_map else (e.scrape_stats or {}).get("floor_map_url_scraped"),
                "warnings": res.warnings,
            }
            e.scrape_status, e.scrape_method = status, res.method if status != "failed" else ""
            e.scrape_error = (res.error or "")[:500]
            e.scraped_at = datetime.now(timezone.utc)
            await s.commit()
            out = {"id": str(e.id), "name": e.name, "status": status, "exhibitors": len(res.exhibitors),
                   "floorMap": e.floor_map_url or None}
    if notify:
        await notify("event_scraped", out)
    return out


async def _save_rows(s, event_name: str, rows: List[ScrapedExhibitor]) -> None:
    """Replace this event's earlier scraped rows. Imported and app-added rows (and visits, keyed by name) stay."""
    where = await event_filter(s, database.ExhibitorDB.event_name, event_name)
    await s.execute(delete(database.ExhibitorDB).where(where, database.ExhibitorDB.source == "scrape",
                                                       database.ExhibitorDB.added_by.is_(None)))
    taken = {event_key(n) for n in (await s.execute(select(database.ExhibitorDB.name).where(where))).scalars()}
    for x in rows:
        if event_key(x.name) in taken:
            continue
        taken.add(event_key(x.name))
        s.add(database.ExhibitorDB(id=uuid.uuid4(), event_name=event_name, name=x.name, booth=x.booth, hall=x.hall,
                                   category=x.category, country=x.country, website=x.website, description=x.description,
                                   profile_url=x.profile_url, products=[], tags=[], source="scrape",
                                   created_at=datetime.now(timezone.utc)))


def kick_event_scrape(event_id: uuid.UUID, *, manual: bool = False) -> None:
    if not ENABLED and not manual:
        return
    t = asyncio.create_task(run_scrape(event_id, manual=manual))
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


def is_running(event_id: uuid.UUID) -> bool:
    lock = _locks.get(event_id)
    return bool(lock and lock.locked())
