"""Real browser fetch adapter with intercepted API responses, no production traffic."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/scout-feed/proof'
SEED = json.loads((ROOT / 'tests/fixtures/scout-ranked.json').read_text())
USER = '00000000-0000-0000-0000-000000000001'
requests = []
errors = []
mode = 'normal'


def route_api(route):
    global mode
    url = urlparse(route.request.url)
    if url.hostname == '127.0.0.1' or url.hostname in ('cdn.tailwindcss.com', 'cdnjs.cloudflare.com'):
        route.continue_()
        return
    query = parse_qs(url.query, keep_blank_values=True)
    path = url.path
    data = {'status': 'success'}
    if path == '/exhibitors/ranked/':
        requests.append(query)
        if mode == 'network':
            route.abort('internetdisconnected')
            return
        if mode == 'server':
            route.fulfill(status=503, json={'detail': 'Test unavailable'})
            return
        items = SEED
        q = query.get('search', [''])[0].lower()
        categories = [c.lower() for c in query.get('category', [''])[0].split(',') if c]
        items = [e for e in items if q in (e['name'] + ' ' + e['category']).lower()
                 and (not categories or any(c in e['category'].lower() for c in categories))]
        offset = int(query['offset'][0])
        limit = int(query['limit'][0])
        data.update(items=items[offset:offset + limit], total=len(items), has_more=offset + limit < len(items))
    elif path == '/exhibitors/categories/':
        data['categories'] = ['Medical Imaging', 'Furniture']
    elif path == '/user/profile/':
        data['profile'] = {'current_event_name': 'Test Expo', 'target_industries': ['imaging']}
    elif path == '/user/validate':
        data['valid'] = True
    elif path == '/list_contacts/':
        data.update(contacts=[], total=0)
    elif path == '/broadcasts/active':
        data['broadcasts'] = []
    elif path == '/user/card/':
        data['card'] = None
    elif path == '/dashboard':
        data.update(total_contacts=0, hot_leads=0, warm_leads=0, cold_leads=0)
    route.fulfill(json=data)


with sync_playwright() as p:
    browser = p.chromium.launch(args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
    context = browser.new_context(viewport={'width': 390, 'height': 844})
    page = context.new_page()
    page.route('**/*', route_api)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.add_init_script("localStorage.setItem('eventScoutUser', JSON.stringify({user_id:'" + USER + "',name:'UI test',email:'ui@example.test',is_admin:false}))")
    page.goto('http://127.0.0.1:3118/mobile-frontend.html', wait_until='networkidle')
    page.locator('#scoutNavBtn').click()
    page.wait_for_function("scoutFeed.offset === 20 && !scoutFeed.busy")
    assert requests[-1]['event'] == ['Test Expo']
    assert requests[-1]['user_id'] == [USER]
    assert requests[-1]['limit'] == ['20']
    first_count = len(requests)
    page.locator('#feed-search').fill('Exhibitor 119')
    page.wait_for_timeout(100)
    assert len(requests) == first_count, 'Search was not debounced'
    page.wait_for_function("scoutFeed.total === 1 && !scoutFeed.busy")
    assert requests[-1]['offset'] == ['0']
    assert page.locator('.feed-row h2').inner_text() == 'Exhibitor 119'
    page.locator('#feed-search').fill('')
    page.wait_for_function("scoutFeed.offset === 20 && !scoutFeed.busy")

    mode = 'network'
    page.locator('#contentArea').evaluate('(e) => e.scrollTop = e.scrollHeight')
    page.wait_for_function("document.querySelector('#feed-state').dataset.state === 'offline'")
    assert page.locator('.feed-row').count() == 20
    page.locator('#feed-state').scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / 'offline-retained-390.png'))
    mode = 'normal'
    page.evaluate("window.dispatchEvent(new Event('online'))")
    page.wait_for_function("scoutFeed.offset >= 40 && !scoutFeed.busy")
    assert page.locator('.feed-row').count() >= 40

    mode = 'server'
    page.locator('#feed-search').fill('Exhibitor 119')
    page.wait_for_function("document.querySelector('#feed-state').dataset.state === 'error'")
    mode = 'normal'
    page.locator('[data-feed-action=retry]').click()
    page.wait_for_function("scoutFeed.total === 1 && !scoutFeed.busy")
    page.locator('#feed-search').fill('')
    page.wait_for_function("scoutFeed.offset === 20 && !scoutFeed.busy")
    context.set_offline(True)
    page.wait_for_function("document.querySelector('#feed-state').dataset.state === 'offline'")
    assert page.locator('.feed-row').count() == 20
    context.set_offline(False)
    page.wait_for_function("scoutFeed.offset >= 40 && !scoutFeed.busy")

    navigation = []
    for screen in ['contacts', 'scan', 'chat', 'profile', 'scout']:
        page.locator(f'.nav-btn[onclick="switchScreen(\'{screen}\')"]').click()
        page.wait_for_function("name => AppState.activeScreen === name", arg=screen)
        visible = '#cameraView' if screen == 'scan' else '#' + screen + 'Screen'
        assert page.locator(visible).is_visible(), screen
        assert page.locator(f'.nav-btn[onclick="switchScreen(\'{screen}\')"]').evaluate("e => e.classList.contains('active')")
        navigation.append(screen)
    assert not errors, errors
    page.wait_for_timeout(300)
    # Missing reason is deliberately logged and exposed, not hidden. Also check escaping.
    missing = page.evaluate("""() => {
      const calls = []; const original = console.error; console.error = (...args) => calls.push(args[0]);
      const html = renderExhibitorRow({name: '<img src=x onerror=alert(1)>', score: 20, reason: ''}, 1);
      console.error = original;
      const root = document.createElement('div'); root.innerHTML = html;
      return {text: root.querySelector('.feed-reason').textContent, danger: root.querySelector('.feed-reason').classList.contains('feed-reason-missing'), images: root.querySelectorAll('img').length, errors: calls.length};
    }""")
    assert missing == {'text': 'Reason unavailable', 'danger': True, 'images': 0, 'errors': 1}
    boundaries = page.evaluate("[0,39,40,69,70,100].map(n => {const e=document.createElement('div'); e.innerHTML=renderScoreBadge(n); return e.firstChild.dataset.temperature})")
    assert boundaries == ['cold', 'cold', 'warm', 'warm', 'hot', 'hot']
    browser.close()

result = {'request_count': len(requests), 'requests': requests, 'navigation': navigation,
          'network_failure_keeps_rows': True, 'online_auto_retry': True,
          'navigator_offline_auto_retry': True, 'http_error_retry': True,
          'debounce_and_offset_reset': True, 'missing_reason_and_xss': missing,
          'score_boundaries': boundaries, 'page_errors': errors,
          'limits': 'API responses intercepted. No live database, real camera, or external writes.'}
(OUT / 'network-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
