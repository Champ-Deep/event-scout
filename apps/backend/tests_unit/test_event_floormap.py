"""event_floormap: heuristic floor map hits, needs_llm trigger, LLM fallback with a fake ai_client.chat + post-guards."""
import json
from types import SimpleNamespace

import pytest

import ai_client
import event_floormap as fm


def page(html, url="https://expo.example.com/"):
    return SimpleNamespace(url=url, final_url=url, html=html)


def test_pdf_anchor_is_found_and_made_absolute():
    r = fm.extract_floor_map([page('<nav><a href="/about">About</a></nav><a href="/files/Floor-Plan-2026.pdf">Download floor plan</a>')])
    assert (r.url, r.kind, r.source) == ("https://expo.example.com/files/Floor-Plan-2026.pdf", "pdf", "heuristic")


def test_iframe_viewer_beats_plain_page_link():
    html = '<a href="/floor-plan">Floor plan</a><iframe src="https://xyz.expocad.com/hall1"></iframe>'
    r = fm.extract_floor_map([page(html)])
    assert (r.kind, r.url) == ("viewer", "https://xyz.expocad.com/hall1")


def test_image_alt_match():
    r = fm.extract_floor_map([page('<img src="/img/hall.png" alt="Exhibition layout">')])
    assert (r.kind, r.url) == ("image", "https://expo.example.com/img/hall.png")


def test_google_maps_sitemap_and_bare_map_links_are_not_floor_maps():
    html = ('<a href="https://www.google.com/maps/place/x">Map</a><a href="/sitemap.xml">Site map</a>'
            '<a href="/travel-map">Map</a><a href="mailto:a@b.c">floor plan</a><a href="#floor-plan">floor plan</a>')
    assert fm.extract_floor_map([page(html)]) is None


def test_script_text_and_non_http_are_ignored():
    assert fm.extract_floor_map([page('<script>var a="<a href=\\"/floor-plan.pdf\\">floor plan</a>"</script>')]) is None


def rows(n, rich=True):
    return [SimpleNamespace(name=f"Co {i}", booth=f"A{i}" if rich else "", hall="", category="") for i in range(n)]


def test_needs_llm_conditions():
    fmap = fm.FloorMap("https://x/a.pdf", "pdf", "heuristic")
    assert fm.needs_llm([], fmap) == (True, "no_exhibitors")
    assert fm.needs_llm(rows(2), None)[0] is True  # only "rows have booth": 1/3
    assert fm.needs_llm(rows(2, rich=False), fmap)[0] is True  # only the floor map: 1/3
    assert fm.needs_llm(rows(2), fmap)[0] is False  # booth data + floor map: 2/3
    assert fm.needs_llm(rows(6), None)[0] is False  # >= 5 rows + booth data: 2/3
    assert fm.needs_llm(rows(6, rich=False), None)[0] is True  # only >= 5 rows: 1/3


def test_coverage_values():
    assert fm.coverage(rows(6), fm.FloorMap("u", "pdf", "heuristic")) == 1.0
    assert fm.coverage([], None) == 0.0


def fake_chat(reply):
    calls = []

    async def chat(messages, **kw):
        calls.append((messages, kw))
        return json.dumps(reply), "fake/model"

    chat.calls = calls
    return chat


HTML = ('<h1>Exhibitors</h1><p>Acme Robotics Booth 12</p><p>Globex Corp</p>'
        '<a href="/maps/hall.pdf">Hall plan</a><a href="https://acme.example/">Acme site</a>')


async def test_llm_fallback_returns_guarded_result_and_logs(monkeypatch, capsys):
    chat = fake_chat({
        "floor_map_url": "https://expo.example.com/maps/hall.pdf",
        "exhibitors": [
            {"name": "Acme Robotics", "booth": "12", "website": "https://acme.example/"},
            {"name": "Invented Ltd", "booth": "99"},  # not in page text: dropped
            {"name": "Globex Corp", "website": "https://made-up.example"},  # website not in page: blanked
        ],
    })
    monkeypatch.setattr(ai_client, "chat", chat)
    out = await fm.llm_extract([page(HTML)], want_exhibitors=True, want_floor_map=True, event="WHX", reason="no_exhibitors")
    assert out["model"] == "fake/model"
    assert out["floor_map_url"] == "https://expo.example.com/maps/hall.pdf"
    assert [(e["name"], e["booth"], e["website"]) for e in out["exhibitors"]] == [
        ("Acme Robotics", "12", "https://acme.example/"), ("Globex Corp", "", "")]
    assert chat.calls[0][1]["json_mode"] is True and chat.calls[0][1]["temperature"] == 0.0
    assert "[SCRAPE] llm fallback event=WHX reason=no_exhibitors pages=1 model=fake/model" in capsys.readouterr().out


async def test_llm_invented_floor_map_url_is_dropped(monkeypatch):
    monkeypatch.setattr(ai_client, "chat", fake_chat({"floor_map_url": "https://evil.example/plan.pdf", "exhibitors": []}))
    out = await fm.llm_extract([page(HTML)], want_exhibitors=False, want_floor_map=True)
    assert out["floor_map_url"] is None


async def test_llm_respects_scope_and_chunk_cap(monkeypatch):
    chat = fake_chat({"floor_map_url": "", "exhibitors": [{"name": "Acme Robotics"}]})
    monkeypatch.setattr(ai_client, "chat", chat)
    big = page("<p>Acme Robotics " + "x " * 100_000 + "</p>")
    out = await fm.llm_extract([big], want_exhibitors=False, want_floor_map=True)
    assert len(chat.calls) == fm.LLM_MAX_CALLS and out["exhibitors"] == []  # exhibitors not wanted


async def test_llm_without_key_raises_ai_error(monkeypatch):
    monkeypatch.setattr(ai_client, "ai_configured", False)
    with pytest.raises(ai_client.AIError) as e:
        await fm.llm_extract([page(HTML)], want_exhibitors=True, want_floor_map=True)
    assert e.value.reason == "not_configured"
