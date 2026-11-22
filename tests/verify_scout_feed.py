"""Browser proof for the shipped single-file Scout screen. Run with local server on 3118."""
import importlib.util
import json
from pathlib import Path
import re
from urllib.parse import urlparse, parse_qs

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/scout-feed/proof"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:3118/mobile-frontend.html"
SEED = json.loads((ROOT / "tests/fixtures/scout-ranked.json").read_text())
spec = importlib.util.spec_from_file_location("visual_audit", "/Users/deep/champ-skills/design/skills/visual-verify/scripts/audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
results = {"screenshots": [], "audits": [], "interactions": [], "console_errors": []}


def wait_state(page, state):
    page.wait_for_function("state => document.querySelector('#feed-state').dataset.state === state", arg=state)


def measure(page):
    return page.evaluate("""() => ({
      searchTop: document.querySelector('#feed-search-bar').getBoundingClientRect().top,
      chipsTop: document.querySelector('#feed-filters').getBoundingClientRect().top,
      rows: document.querySelectorAll('.feed-row').length,
      scores: [...document.querySelectorAll('.feed-row .score-badge')].map(e => Number(e.dataset.score)),
      reasons: [...document.querySelectorAll('.feed-reason')].every(e => e.textContent.trim()),
      overflow: [...document.querySelectorAll('#scoutScreen *')].filter(e => {
        const r = e.getBoundingClientRect();
        return r.width > 0 && (r.right > innerWidth + 1 || r.left < -1);
      }).map(e => e.id || e.className),
      disabled: [...document.querySelectorAll('#scoutScreen button, #feed-search')].filter(e => e.disabled).length,
      controls: [...document.querySelectorAll('#scoutScreen button, #feed-search')].filter(e => e.getBoundingClientRect().width).every(e => e.getBoundingClientRect().height >= 44)
    })""")


def audit_state(page, label):
    findings = page.evaluate(audit.PROBE)
    # The stock audit also inspects hidden forms in other screens. Retain those
    # findings as evidence, but do not misattribute them to the new Scout screen.
    hidden_legacy_controls = page.evaluate("""[...document.querySelectorAll('input, textarea, select')]
        .filter(e => e.closest('.hidden') && !e.closest('#scoutScreen'))
        .map(e => e.tagName.toLowerCase() + (e.id ? '#' + e.id : e.className ? '.' + e.className.trim().split(/\\s+/).slice(0, 2).join('.') : ''))""")
    scout_fails = [f for f in findings if f['severity'] == 'FAIL'
                   and f['where'].split(' ')[0] not in hidden_legacy_controls]
    results['audits'].append({'state': label, 'findings': findings, 'scout_failures': scout_fails})
    assert not scout_fails, scout_fails
    metrics = measure(page)
    assert not metrics['overflow'], metrics['overflow']
    assert not metrics['disabled']
    assert metrics['controls']


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 390, 'height': 844}, device_scale_factor=1)
    page.on('pageerror', lambda error: results['console_errors'].append(str(error)))
    page.on('console', lambda message: results['console_errors'].append(message.text) if message.type == 'error' else None)
    for state in ['empty', 'loading', 'populated', 'error', 'offline', 'noresults']:
        page.goto(URL + '?feedstate=' + state, wait_until='networkidle')
        wait_state(page, state)
        audit_state(page, state + '-390')
        shot = OUT / f'{state}-390.png'
        page.screenshot(path=str(shot))
        results['screenshots'].append(str(shot))
        if state in ('error', 'offline'):
            page.locator('[data-feed-action="retry"]').click()
            wait_state(page, state)
        if state == 'noresults':
            assert 'zzzz' in page.locator('#feed-state').inner_text()
            page.locator('[data-feed-action="clear"]').click()
            wait_state(page, 'populated')
            assert page.locator('#feed-search').input_value() == ''
            assert page.locator('.feed-row').count() == 20
            results['interactions'].append('No-results clear restores 20 rows and focuses search')

    for width in [360, 375, 768, 1280, 1440]:
        page.set_viewport_size({'width': width, 'height': 900 if width >= 768 else 844})
        page.goto(URL + '?feedstate=populated', wait_until='networkidle')
        wait_state(page, 'populated')
        audit_state(page, f'populated-{width}')
        shot = OUT / f'populated-{width}.png'
        page.screenshot(path=str(shot))
        results['screenshots'].append(str(shot))

    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(URL + '?feedstate=populated', wait_until='networkidle')
    wait_state(page, 'populated')
    before = measure(page)
    page.evaluate("document.querySelector('#contentArea').scrollTop = 3000")
    page.wait_for_timeout(300)
    after = measure(page)
    assert abs(before['searchTop'] - after['searchTop']) <= 1
    assert abs(before['chipsTop'] - after['chipsTop']) <= 1
    results['sticky'] = {'before': [before['searchTop'], before['chipsTop']],
                         'after_3000px': [after['searchTop'], after['chipsTop']],
                         'scrollTop': page.locator('#contentArea').evaluate('(e) => e.scrollTop')}
    assert results['sticky']['scrollTop'] == 3000, results['sticky']
    for _ in range(15):
        page.evaluate("document.querySelector('#contentArea').scrollTop = document.querySelector('#contentArea').scrollHeight")
        page.wait_for_timeout(400)
        if page.locator('.feed-row').count() == 120:
            break
    final = measure(page)
    assert final['rows'] == 120, final
    assert final['reasons']
    assert final['scores'] == sorted(final['scores'], reverse=True)
    assert 'You have seen all 120 exhibitors' in page.locator('#feed-state').inner_text()
    page.locator('#feed-state').scroll_into_view_if_needed()
    results['full_list'] = final
    audit_state(page, 'end-120-rows')
    page.screenshot(path=str(OUT / 'end-390.png'))
    results['screenshots'].append(str(OUT / 'end-390.png'))
    results['frame_times'] = page.evaluate("""() => new Promise(resolve => {
      const scroller = document.querySelector('#contentArea');
      scroller.scrollTop = 0;
      const frames = []; let last; let start;
      function step(now) {
        if (!start) start = now;
        if (last) frames.push(now - last);
        last = now;
        scroller.scrollTop = (scroller.scrollHeight - scroller.clientHeight) * Math.min(1, (now - start) / 3000);
        if (now - start < 3000) requestAnimationFrame(step);
        else {
          frames.sort((a, b) => a - b);
          resolve({samples: frames.length, p95_ms: frames[Math.ceil(frames.length * .95) - 1], max_ms: frames.at(-1)});
        }
      }
      requestAnimationFrame(step);
    })""")
    baseline = browser.new_page()
    results['blank_frame_baseline'] = baseline.evaluate("""() => new Promise(resolve => {
      const frames = []; let last; let start;
      function step(now) {
        if (!start) start = now;
        if (last) frames.push(now - last);
        last = now;
        if (now - start < 3000) requestAnimationFrame(step);
        else { frames.sort((a,b) => a-b); resolve({samples: frames.length, p95_ms: frames[Math.ceil(frames.length * .95) - 1], max_ms: frames.at(-1)}); }
      }
      requestAnimationFrame(step);
    })""")
    baseline.close()
    page.bring_to_front()
    page.evaluate("document.querySelector('#contentArea').scrollTop = 0")
    page.locator('.feed-more').first.click()
    assert page.locator('.feed-more').first.get_attribute('aria-expanded') == 'true'
    assert 'feed-reason-clamped' not in page.locator('.feed-reason').first.get_attribute('class')
    page.locator('.feed-more').first.click()
    assert page.locator('.feed-more').first.get_attribute('aria-expanded') == 'false'
    page.get_by_role('button', name='Medical Imaging', exact=True).click()
    wait_state(page, 'end')
    assert page.locator('.feed-row').count() == 18
    page.get_by_role('button', name='Furniture', exact=True).click()
    wait_state(page, 'populated')
    assert page.locator('.feed-chip[aria-pressed=true]').count() == 2
    page.get_by_role('button', name='All', exact=True).click()
    wait_state(page, 'populated')
    page.locator('#feed-search').fill('Exhibitor 119')
    assert page.locator('.feed-row').count() == 0
    wait_state(page, 'end')
    assert page.locator('.feed-row').count() == 1
    results['interactions'].append('More/Less, category multi-select, All, 300ms search and offset reset')
    audit_state(page, 'filtered-one-result')
    page.emulate_media(reduced_motion='reduce')
    page.locator('#feed-search').fill('zzzz')
    wait_state(page, 'noresults')
    page.locator('[data-feed-action=clear]').click()
    wait_state(page, 'populated')
    assert page.locator('.feed-row').count() == 20
    results['interactions'].append('Reduced motion clear/search flow')

    page.get_by_role('button', name='Medical Imaging', exact=True).focus()
    page.keyboard.press('Space')
    wait_state(page, 'end')
    assert page.locator('.feed-row').count() == 18
    assert page.get_by_role('button', name='Medical Imaging', exact=True).evaluate("e => e === document.activeElement && getComputedStyle(e).outlineStyle !== 'none'")
    page.keyboard.press('Space')
    wait_state(page, 'populated')
    page.locator('.feed-reason').first.evaluate("e => e.textContent = 'A long reason with profile evidence. '.repeat(30)")
    page.locator('.feed-row').first.scroll_into_view_if_needed()
    assert page.locator('.feed-reason').first.evaluate('e => e.scrollHeight > e.clientHeight && e.clientHeight <= 72')
    page.locator('.feed-more').first.click()
    assert page.locator('.feed-reason').first.evaluate('e => e.clientHeight > 72 && e.scrollHeight === e.clientHeight')
    results['interactions'].append('Keyboard category activation preserves visible focus; long reasons clamp to 3 lines and More reveals all')

    # Resolve a superseded request even after abort, proving the token rather than transport cancellation.
    race = page.evaluate("""async () => {
      const original = scoutFeedRequest;
      const pending = [];
      scoutFeedRequest = kind => kind === 'categories' ? Promise.resolve({categories: []}) : new Promise(resolve => pending.push(resolve));
      resetScoutFeed(); await new Promise(r => setTimeout(r, 20));
      loadScoutPage(); loadScoutPage();
      const duplicateGuard = pending.length === 1;
      resetScoutFeed(); await new Promise(r => setTimeout(r, 20));
      pending[1]({items: [{name: 'Current result', score: 90, reason: 'Current reason'}], total: 1, has_more: false});
      await new Promise(r => setTimeout(r, 0));
      pending[0]({items: [{name: 'Stale result', score: 99, reason: 'Stale reason'}], total: 1, has_more: false});
      await new Promise(r => setTimeout(r, 0));
      scoutFeedRequest = original;
      return {duplicateGuard, names: [...document.querySelectorAll('.feed-row h2')].map(e => e.textContent)};
    }""")
    assert race == {'duplicateGuard': True, 'names': ['Current result']}, race
    results['race_guard'] = race
    assert not results['console_errors'], results['console_errors']
    browser.close()

(OUT / 'browser-results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps({key: value for key, value in results.items() if key not in ['audits', 'full_list']}, indent=2))
print('PASS: all Scout state audits, geometry, ordering, pagination and interaction assertions')
