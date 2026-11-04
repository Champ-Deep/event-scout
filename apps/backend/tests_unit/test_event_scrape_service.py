"""Integration: admin adds an event website -> background scrape -> persisted rows -> the discovery consumers.

Real routes (api_routes), real service, SQLite; only the network (fetch) and the model (ai_client.chat) are faked.
"""
import asyncio
import pathlib
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

import ai_client
import api_routes
import database
import event_scrape_service as svc
import event_scraper
from conftest import make_user

FIX = pathlib.Path(__file__).parent / "fixtures"
SITE = "https://gulfexpo.example"
FILES = {SITE: "expo_home.html", SITE + "/": "expo_home.html", SITE + "/exhibitors": "expo_cards_p1.html",
         SITE + "/exhibitors?page=2": "expo_cards_p2.html", SITE + "/venue/floor-plan": "js_shell.html"}


def fake_site(files=FILES):
    async def fetch(url, **_):
        if url not in files:
            raise event_scraper.ScrapeError("unreachable", url)
        return event_scraper.FetchedPage(url=url, final_url=url, status=200, html=(FIX / files[url]).read_text(),
                                         content_type="text/html")
    return fetch


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(event_scraper, "fetch_page", fake_site())
    monkeypatch.setattr(svc, "notify", None)
    monkeypatch.setattr(ai_client, "ai_configured", False)
    svc._locks.clear()


async def settle():
    await asyncio.gather(*svc._tasks)


async def create(factory, admin, website=SITE, name="Gulf Expo 2026"):
    async with factory() as s:
        return await api_routes.planner_create_event(api_routes.PlannerEventPayload(name=name, website=website), s, admin)


async def planner_row(factory, event_id):
    async with factory() as s:
        return next(r for r in await api_routes._planner_rows(s) if r["id"] == event_id)


async def discovery(factory, event):
    async with factory() as s:  # the app's exhibitor / booth list, called as the route does
        return await api_routes.list_exhibitors(search=None, category=None, hall=None, event=event, limit=500, session=s, me=None)


async def test_add_event_website_scrapes_persists_and_feeds_discovery(db):
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()

    row = await planner_row(db, created["id"])
    assert row["scrape"]["status"] == "ok" and row["scrape"]["method"] == "heuristic" and row["scrape"]["error"] == ""
    assert row["floorMapUrl"] == SITE + "/venue/floor-plan"
    assert row["scrape"]["stats"]["exhibitors"] == 7 and row["scrape"]["stats"]["llm_used"] is False

    booths = await discovery(db, "gulf expo  2026")  # same event, other spelling
    assert len(booths) == 7
    acme = next(b for b in booths if b["name"] == "Acme Robotics")
    assert (acme["booth"], acme["hall"], acme["country"]) == ("A12", "Hall 3", "Germany") and acme["userAdded"] is False
    async with db() as s:
        assert {x.source for x in (await s.execute(select(database.ExhibitorDB))).scalars()} == {"scrape"}


async def test_no_website_means_no_scrape(db):
    admin = await make_user(db, admin=True)
    created = await create(db, admin, website="")
    await settle()
    assert (await planner_row(db, created["id"]))["scrape"]["status"] == "idle"
    assert await discovery(db, "Gulf Expo 2026") == []


async def test_changing_the_website_rescrapes_but_saving_other_fields_does_not(db, monkeypatch):
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    calls = []
    monkeypatch.setattr(svc, "kick_event_scrape", lambda eid, **kw: calls.append(eid))
    async with db() as s:
        await api_routes.planner_update_event(created["id"], api_routes.PlannerEventPayload(venue="Hall 1"), s)
        assert calls == []
        await api_routes.planner_update_event(created["id"], api_routes.PlannerEventPayload(website="https://other.example"), s)
    assert len(calls) == 1


async def test_rescrape_replaces_scraped_rows_only_and_keeps_visits_and_admin_floor_map(db):
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    eid = uuid.UUID(created["id"])
    async with db() as s:
        e = (await s.execute(select(database.ManagedEventDB))).scalar_one()
        e.floor_map_url = "https://admin.example/my-map.pdf"  # the admin's own link
        s.add(database.ExhibitorDB(id=uuid.uuid4(), event_name=e.name, name="Imported Co", source="import"))
        s.add(database.ExhibitorDB(id=uuid.uuid4(), event_name=e.name, name="Added In App", source="app", added_by=admin))
        s.add(database.ExhibitorDB(id=uuid.uuid4(), event_name=e.name, name="acme robotics", source="import"))  # clashes with a scraped name
        await s.commit()

    await svc.run_scrape(eid, manual=True)
    async with db() as s:
        rows = (await s.execute(select(database.ExhibitorDB))).scalars().all()
    by_source = {}
    for r in rows:
        by_source.setdefault(r.source, []).append(r.name)
    assert sorted(by_source["import"]) == ["Imported Co", "acme robotics"] and by_source["app"] == ["Added In App"]
    assert len(by_source["scrape"]) == 6 and "Acme Robotics" not in by_source["scrape"]  # import row wins on the name
    assert (await planner_row(db, created["id"]))["floorMapUrl"] == "https://admin.example/my-map.pdf"


async def test_failed_scrape_keeps_old_data_and_shows_the_reason(db, monkeypatch):
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    monkeypatch.setattr(event_scraper, "fetch_page", fake_site({}))  # site went down
    await svc.run_scrape(uuid.UUID(created["id"]), manual=True)

    row = await planner_row(db, created["id"])
    assert row["scrape"]["status"] == "failed" and row["scrape"]["error"].startswith("unreachable")
    assert row["floorMapUrl"] and len(await discovery(db, "Gulf Expo 2026")) == 7


async def test_automatic_kick_respects_the_cooldown_manual_does_not(db):
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    eid = uuid.UUID(created["id"])
    assert (await svc.run_scrape(eid))["status"] == "skipped"
    assert (await svc.run_scrape(eid, manual=True))["status"] == "ok"


async def test_js_only_site_reports_needs_js_without_failing_the_event_save(db, monkeypatch):
    monkeypatch.setattr(event_scraper, "fetch_page", fake_site({SITE: "js_shell.html"}))
    admin = await make_user(db, admin=True)
    created = await create(db, admin)  # the save itself succeeded
    await settle()
    row = await planner_row(db, created["id"])
    assert created["name"] == "Gulf Expo 2026" and row["scrape"]["status"] == "failed"
    assert row["scrape"]["error"].startswith("needs_js")


async def test_page_with_nothing_structured_falls_back_to_the_llm_and_says_so(db, monkeypatch):
    html = ("<html><body><h1>Gulf Expo</h1><p>Our exhibitors include Zenith Marine in hall 2 and Polar Cooling.</p>"
            '<a href="https://cdn.example/hall-layout.pdf">Hall layout</a></body></html>')
    page = event_scraper.FetchedPage(url=SITE, final_url=SITE, status=200, html=html, content_type="text/html")

    async def fetch(url, **_):
        return page

    async def chat(messages, **_):
        return ('{"floor_map_url": "https://invented.example/map.pdf", "exhibitors": ['
                '{"name": "Zenith Marine", "hall": "2"}, {"name": "Polar Cooling"}, {"name": "Totally Invented Ltd"}]}', "fake-model")

    monkeypatch.setattr(event_scraper, "fetch_page", fetch)
    monkeypatch.setattr(ai_client, "ai_configured", True)
    monkeypatch.setattr(ai_client, "chat", chat)
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()

    row = await planner_row(db, created["id"])
    assert row["scrape"]["status"] == "ok" and row["scrape"]["method"] == "mixed"  # floor map by heuristic, rows by LLM
    assert row["scrape"]["stats"]["llm_used"] is True and row["scrape"]["stats"]["model"] == "fake-model"
    assert row["floorMapUrl"] == "https://cdn.example/hall-layout.pdf"  # the invented URL never lands
    assert sorted(b["name"] for b in await discovery(db, "Gulf Expo 2026")) == ["Polar Cooling", "Zenith Marine"]


async def test_llm_unavailable_is_a_warning_not_an_error(db, monkeypatch):
    async def fetch(url, **_):
        return event_scraper.FetchedPage(url=SITE, final_url=SITE, status=200, html="<html><body><p>Welcome to the expo, "
                                         "a long paragraph with no list at all of anything.</p></body></html>", content_type="text/html")

    monkeypatch.setattr(event_scraper, "fetch_page", fetch)
    res = await svc.scrape_event_site(SITE)
    assert res.warnings == ["ai_not_configured"] and res.error.startswith("no_content") and res.method == "none"


async def test_manual_scrape_route_guards(db):
    admin = await make_user(db, admin=True)
    blank = await create(db, admin, website="", name="No Site Expo")
    site = await create(db, admin)
    await settle()
    async with db() as s:
        with pytest.raises(HTTPException) as no_site:
            await api_routes.planner_scrape_event(blank["id"], s)
        assert no_site.value.status_code == 400
        svc._locks[uuid.UUID(site["id"])] = lock = asyncio.Lock()
        await lock.acquire()
        with pytest.raises(HTTPException) as busy:
            await api_routes.planner_scrape_event(site["id"], s)
        assert busy.value.status_code == 409
        lock.release()
        assert await api_routes.planner_scrape_event(site["id"], s) == {"status": "running"}
        assert (await api_routes.planner_scrape_status(site["id"], s))["status"] in ("running", "ok")
    await settle()


# needs_js is the honest "JS tier unavailable" signal: only for an empty JS shell with nothing found, never a claim a JS render ran.
def _html_page(html):
    async def fetch(url, **_):
        return event_scraper.FetchedPage(url=url, final_url=url, status=200, html=html, content_type="text/html")
    return fetch


async def test_needs_js_only_for_an_empty_js_shell_and_says_no_js_render_ran(monkeypatch):
    monkeypatch.setenv("LAKESTREAM_URL", "https://lakestream.invalid")  # reserved, ignored: setting it must not change the outcome
    monkeypatch.setenv("LAKESTREAM_API_KEY", "x")
    res = await svc.scrape_event_site(SITE, fetch=fake_site({SITE: "js_shell.html"}))
    assert res.error.startswith("needs_js") and "not wired" in res.error and "no JS render was attempted" in res.error
    assert res.exhibitors == [] and res.floor_map is None and res.method == "none"


async def test_needs_js_not_reported_for_a_static_page_or_a_shell_that_yielded_content():
    static = await svc.scrape_event_site(SITE, fetch=_html_page("<html><body><p>Welcome to our event.</p></body></html>"))
    assert static.error.startswith("no_content")
    shell = '<html><body><div id="root"></div><a href="https://cdn.example/floor-plan.pdf">Floor plan</a></body></html>'
    found = await svc.scrape_event_site(SITE, fetch=_html_page(shell))
    assert found.floor_map is not None and not (found.error or "").startswith("needs_js")


async def test_route_made_floor_map_and_app_booth_survive_rescrapes(db):
    """Data entered through the real routes (planner PATCH, POST /exhibitors) is never touched by a scrape."""
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    async with db() as s:
        await api_routes.planner_update_event(created["id"], api_routes.PlannerEventPayload(floorMapUrl="https://admin.example/map.pdf"), s)
        await api_routes.add_booth(api_routes.BoothPayload(event="Gulf Expo 2026", name="Walk-up Co", booth="Z9"), s, admin, None)
    for _ in range(2):
        await svc.run_scrape(uuid.UUID(created["id"]), manual=True)
    assert (await planner_row(db, created["id"]))["floorMapUrl"] == "https://admin.example/map.pdf"
    booths = {b["name"]: b for b in await discovery(db, "Gulf Expo 2026")}
    assert booths["Walk-up Co"]["booth"] == "Z9" and booths["Walk-up Co"]["userAdded"] is True and len(booths) == 8


async def test_partial_rescrape_without_exhibitors_keeps_earlier_scraped_rows(db, monkeypatch):
    """Site still shows a floor plan but the directory page is down: old scraped exhibitors must not be wiped."""
    admin = await make_user(db, admin=True)
    created = await create(db, admin)
    await settle()
    assert len(await discovery(db, "Gulf Expo 2026")) == 7, "setup scrape"
    monkeypatch.setattr(event_scraper, "fetch_page", fake_site({SITE: "expo_home.html", SITE + "/": "expo_home.html", SITE + "/venue/floor-plan": "js_shell.html"}))
    out = await svc.run_scrape(uuid.UUID(created["id"]), manual=True)
    assert out["exhibitors"] == 0 and out["status"] == "partial", out
    assert len(await discovery(db, "Gulf Expo 2026")) == 7, "rescrape wiped rows"
