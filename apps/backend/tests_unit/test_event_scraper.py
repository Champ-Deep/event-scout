"""event_scraper through its interface: saved HTML fixtures (no network), httpx.MockTransport, a fake resolver."""

import pathlib

import httpx
import pytest

import event_scraper as es

FIX = pathlib.Path(__file__).parent / "fixtures"
SITE = "https://gulfexpo.example"


def page(name, url=SITE + "/exhibitors", ctype="text/html"):
    return es.FetchedPage(url=url, final_url=url, status=200, html=(FIX / name).read_text(), content_type=ctype)


def view(rows):
    return [(r.name, r.booth, r.hall) for r in rows]


# ------------------------------------------------------------------ extraction: layouts


def test_card_grid_layout():
    rows = es.extract_exhibitors([page("expo_cards_p1.html")])
    assert view(rows) == [("Acme Robotics", "A12", "Hall 3"), ("Bluewave Energy", "B-204", ""),
                          ("Cedar Foods", "C7", ""), ("Delta Logistics", "D1", ""), ("Echo Labs", "E5", "")]
    acme = rows[0]
    assert (acme.category, acme.country, acme.description) == ("Robotics", "Germany", "Industrial arms.")
    assert acme.profile_url == SITE + "/exhibitors/acme-robotics"
    assert acme.website == "https://www.acme-robotics.example"
    assert rows[3].profile_url == "" and rows[3].website == ""  # nav links never leak into cards


def test_table_layout():
    rows = es.extract_exhibitors([page("summit_table.html", "https://summit.example/exhibitors")])
    assert view(rows) == [("Medix Devices", "4.2-B", "Hall 4"), ("Novacare Pharma", "5.1-A", "Hall 5"),
                          ("Orthoworks", "", "Hall 4")]  # the repeated Medix row is dropped
    assert rows[0].country == "India" and rows[0].website == "https://medix.example"
    assert rows[0].profile_url == "https://summit.example/company/medix"
    assert rows[1].website == "" and rows[2].website == "http://orthoworks.example"


def test_next_data_json_ignores_menu_and_nested_products():
    rows = es.extract_exhibitors([page("nextdata.html")])
    assert view(rows) == [("Quark Analytics", "H-101", "Hall 2"), ("Rho Materials", "H-102", ""),
                          ("Sigma Power", "H-103", "")]
    assert rows[0].category == "Software, AI" and rows[0].website == "https://quark.example"
    assert rows[1].category == "Materials" and rows[1].country == "Japan"
    assert rows[1].profile_url == SITE + "/exhibitors/rho"


def test_ld_json_organizations():
    rows = es.extract_exhibitors([page("ldjson.html")])
    assert [(r.name, r.website, r.country) for r in rows] == [
        ("Tau Textiles", "https://tau.example", "Italy"), ("Upsilon Paper", "https://upsilon.example", "")]


def test_api_json_page():
    rows = es.extract_exhibitors([page("exhibitors_api.json", SITE + "/api/v1/exhibitors.json", "application/json")])
    assert view(rows) == [("Phi Optics", "P1", ""), ("Chi Steel", "P2", ""), ("Psi Labs", "P3", "")]


def test_page_with_no_list_gives_nothing_and_flags_js_shell():
    shell = page("js_shell.html")
    assert es.extract_exhibitors([shell, page("expo_home.html", SITE)]) == []
    assert es.is_js_shell(shell) and not es.is_js_shell(page("expo_cards_p1.html"))


def test_limits_and_nav_words():
    long = "x" * 700
    rows = es._row({"name": long, "booth": "Booth No. 12", "website": "javascript:void(0)"}, SITE)
    assert len(rows.name) == 500 and rows.booth == "12" and rows.website == ""
    assert es._row({"name": "Home"}, SITE) is None and es._row({"name": "2026"}, SITE) is None


# ------------------------------------------------------------------ links and crawl


def test_candidate_links():
    links = es.find_candidate_links(page("expo_home.html", SITE + "/"))
    assert links["exhibitors"] == [SITE + "/exhibitors"]  # not "Become an Exhibitor", sitemap or twitter
    assert links["floor_map"] == [SITE + "/venue/floor-plan"]
    assert links["api"] == [SITE + "/api/v1/exhibitors.json?limit=100"]
    assert es.find_candidate_links(page("expo_cards_p1.html"))["next"] == SITE + "/exhibitors?page=2"
    assert es.find_candidate_links(page("expo_cards_p2.html"))["next"] is None


async def test_crawl_follows_directory_pagination_and_dedupes():
    files = {SITE + "/": "expo_home.html", SITE + "/exhibitors": "expo_cards_p1.html",
             SITE + "/exhibitors?page=2": "expo_cards_p2.html", SITE + "/venue/floor-plan": "js_shell.html"}
    calls = []

    async def fetch(url):
        calls.append(url)
        if url.endswith(".json?limit=100"):
            raise es.ScrapeError("no_content", "gone")  # a dead API link must not sink the scrape
        return page(files[url], url)

    rows = await es.scrape_exhibitors(SITE + "/", fetch=fetch)
    assert [r.name for r in rows] == ["Acme Robotics", "Bluewave Energy", "Cedar Foods", "Delta Logistics",
                                      "Echo Labs", "Foxtrot Mobility", "Gamma Systems"]
    assert len(calls) == len(set(calls)) <= es.MAX_PAGES


async def test_crawl_page_cap():
    async def fetch(url):
        return page("expo_cards_p1.html", url)  # every page says "next" → endless

    pages = await es.crawl(SITE + "/exhibitors", fetch=fetch, max_pages=3)
    assert len(pages) == 3


async def test_crawl_home_failure_propagates():
    async def fetch(url):
        raise es.ScrapeError("blocked")

    with pytest.raises(es.ScrapeError) as e:
        await es.crawl(SITE, fetch=fetch)
    assert e.value.reason == "blocked"


# ------------------------------------------------------------------ fetching: SSRF, retry, errors


@pytest.fixture(autouse=True)
def fast_fetch(monkeypatch):
    monkeypatch.setattr(es, "_HOST_GAP", 0)
    monkeypatch.setattr(es, "_RETRY_DELAY", 0)
    table = {"gulfexpo.example": ["93.184.216.34"], "evil.example": ["10.0.0.5"], "rebind.example": ["8.8.8.8", "127.0.0.1"]}

    async def resolve(host):
        return table[host]

    monkeypatch.setattr(es, "_resolve", resolve)


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def reason(url, handler=lambda r: httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html"})):
    async with client(handler) as c:
        with pytest.raises(es.ScrapeError) as e:
            await es.fetch_page(url, client=c)
    return e.value.reason


@pytest.mark.parametrize("url,want", [
    ("ftp://gulfexpo.example/x", "bad_url"), ("file:///etc/passwd", "bad_url"), ("not a url", "bad_url"),
    ("http://user:pw@gulfexpo.example/", "bad_url"), ("http://gulfexpo.example:8080/", "bad_url"),
    ("http://127.0.0.1/", "blocked_url"), ("http://10.1.2.3/", "blocked_url"), ("http://[::1]/", "blocked_url"),
    ("http://169.254.169.254/latest/meta-data", "blocked_url"), ("http://evil.example/", "blocked_url"),
    ("http://rebind.example/", "blocked_url"),  # one private answer among public ones is enough
])
async def test_ssrf_and_url_rules(url, want):
    assert await reason(url) == want


async def test_redirect_to_private_address_is_rejected_and_never_requested():
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(302, headers={"location": "http://10.0.0.5/admin"})

    assert await reason("https://gulfexpo.example/", handler) == "blocked_url"
    assert seen == ["https://gulfexpo.example/"]


async def test_relative_redirect_followed_and_final_url_recorded():
    def handler(req):
        if req.url.path == "/old":
            return httpx.Response(301, headers={"location": "/exhibitors"})
        return httpx.Response(200, text="<p>hi</p>", headers={"content-type": "text/html; charset=utf-8"})

    async with client(handler) as c:
        p = await es.fetch_page("https://gulfexpo.example/old", client=c)
    assert (p.final_url, p.status, p.content_type, p.tier) == ("https://gulfexpo.example/exhibitors", 200, "text/html", "http")


async def test_redirect_loop_gives_up():
    h = lambda req: httpx.Response(302, headers={"location": "/again"})  # noqa: E731
    assert await reason("https://gulfexpo.example/", h) == "unreachable"


async def test_one_retry_on_503_then_success_and_then_blocked():
    n = {"c": 0}

    def flaky(req):
        n["c"] += 1
        return httpx.Response(503) if n["c"] == 1 else httpx.Response(200, text="<p>ok</p>", headers={"content-type": "text/html"})

    async with client(flaky) as c:
        assert (await es.fetch_page("https://gulfexpo.example/", client=c)).html == "<p>ok</p>" and n["c"] == 2
    always = lambda req: httpx.Response(503)  # noqa: E731
    assert await reason("https://gulfexpo.example/", always) == "blocked"


@pytest.mark.parametrize("resp,want", [
    (httpx.Response(403), "blocked"),
    (httpx.Response(404), "no_content"),
    (httpx.Response(500), "unreachable"),
    (httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}), "no_content"),
    (httpx.Response(200, text="<title>Just a moment...</title>", headers={"content-type": "text/html"}), "blocked"),
])
async def test_http_outcomes(resp, want):
    assert await reason("https://gulfexpo.example/", lambda req: resp) == want


async def test_timeout_and_connect_errors():
    def boom(exc):
        def handler(req):
            raise exc
        return handler

    assert await reason("https://gulfexpo.example/", boom(httpx.ReadTimeout("slow"))) == "timeout"
    assert await reason("https://gulfexpo.example/", boom(httpx.ConnectError("refused"))) == "unreachable"


async def test_body_capped_at_2mb():
    big = "a" * (es._MAX_BYTES + 5000)
    async with client(lambda r: httpx.Response(200, text=big, headers={"content-type": "text/html"})) as c:
        assert len((await es.fetch_page("https://gulfexpo.example/", client=c)).html) == es._MAX_BYTES
