"""
EventScout API inventory — every backend API, who can call it, who does, and what's wrong with it.

WHAT IT REPORTS (printed to the console, and written to docs/API_INVENTORY.md + docs/api-inventory.json)

  1. All APIs, grouped by who may call them
       admin   needs an admin sign-in (require_admin, or an /admin path) — used by the admin console
       user    any signed-in user — used by the mobile app (reps)
       public  no sign-in: login / register / auth, health checks, public share & card pages
     and for each group: USED (by the mobile app, the admin console, or from outside) and UNUSED.

  2. Broken        a client calls a method + path that no backend route has (would 404 / 405).
  3. Fake / dummy  routes that answer with made-up data or claim to do something they don't:
       fake        the whole response is invented (nothing real happens)
       partly fake real data mixed with invented numbers / labels
       stub        always refuses (e.g. 501 "not connected yet")
       bug         real data, but wrong fields
     Curated from reading the handlers (FAKE below) + an automatic scan of handler code for
     mock / fake / dummy markers ("suspected", to review).
  4. New APIs      endpoints created for the mobile app (NEW_ENDPOINTS below).
  5. Newly used    APIs that existed but nothing called, now wired into the mobile app.
  6. Mobile client functions in apps/mobile/src/lib/api.ts that the app never calls, and why.

HOW IT KNOWS (reads the code; no server needed)
  backend  apps/backend/*.py            every @app / @<router> route (+ router prefix), handler, access level
  mobile   apps/mobile/src/lib/api.ts   each client function, the route(s) it calls, and the screens calling it
           apps/mobile/src/sync/engine.ts  the offline sync push (fetch)
  admin    apps/admin/src/**            apiFetch(...) / fetch(...) paths + method, and useEventStream('/api/live')
  outside  EXTERNAL below               webhooks, public links, the Docker health check

RUN     python scripts/api_inventory.py            (from anywhere; re-run after API changes)
        python scripts/api_inventory.py --quiet    (only write the files)
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACK = os.path.join(ROOT, 'apps', 'backend')
MOB = os.path.join(ROOT, 'apps', 'mobile')
ADMIN = os.path.join(ROOT, 'apps', 'admin', 'src')
OUT = os.path.join(ROOT, 'docs')
SKIP_DIRS = {'node_modules', 'dist', '.expo', 'android', 'ios', '__pycache__', 'build'}

# ----------------------------------------------------------------------------------------------
# Curated facts (things the code scan can't tell by itself). Keep these up to date.
# ----------------------------------------------------------------------------------------------

# Existed but nothing called them — wired into the mobile app (Sept 2026).
NEWLY_USED_IN_MOBILE = [
    ('GET', '/api/broadcasts/active', 'Feed → team announcements banner (only the ones sent to everyone / to you)'),
    ('GET', '/api/files/list', 'Me → Team files; Event page → Event files (demo placeholders hidden)'),
    ('GET', '/api/events/{event_id}/seats', 'Event page → Your team here (teammates + scan counts)'),
    ('GET', '/api/exhibitors/categories', "Event page → Add booth: category suggestions (this event's first)"),
    ('PATCH', '/api/exhibitors/{exhibitor_id}', 'Event page → a booth you added → Edit'),
    ('GET', '/api/contact/{contact_id}/files', 'Contact details → Files card (attach, open, save, remove)'),
    ('POST', '/auth/refresh', 'Background: renews the sign-in token when it expires within a day'),
]

# Created for the mobile app (Sept 2026).
NEW_ENDPOINTS = [
    ('POST', '/api/exhibitors', 'Add a booth to an event from the app (no duplicates; can mark it visited)'),
    ('PATCH', '/api/exhibitors/{exhibitor_id}', 'Edit a booth you added (admins: any)'),
    ('DELETE', '/api/exhibitors/{exhibitor_id}', 'Remove a booth you added (admins: any)'),
    ('PUT', '/api/exhibitors/{exhibitor_id}/visit', 'Mark a booth as visited by you (counts in "Booths met")'),
    ('DELETE', '/api/exhibitors/{exhibitor_id}/visit', 'Un-mark a booth as visited'),
    ('GET', '/api/files/{file_id}/download', 'Content of a team / event file (open or save it in the app)'),
    ('GET', '/api/contact/files/{file_id}/download', "Content of a file attached to one of your contacts"),
    ('POST', '/api/contact/{contact_id}/files', 'Attach a file to one of your contacts (Contact details → Files → Attach)'),
    ('DELETE', '/api/contact/files/{file_id}', 'Remove a file from one of your contacts'),
    # admin console (2026-10-01)
    ('PATCH', '/api/admin/users/{user_id}', 'Admin: rename an account or change its role'),
    ('DELETE', '/api/admin/users/{user_id}', "Admin: delete a rep's account and data"),
    ('POST', '/api/admin/contacts/duplicates/dismiss', 'Admin: mark two contacts as not the same person'),
    ('PATCH', '/api/admin/contact_lists/{list_id}', 'Admin: rename a list'),
    ('DELETE', '/api/admin/contact_lists/{list_id}/contacts/{contact_id}', 'Admin: remove a contact from a list'),
]

# Called from outside the apps (kept on purpose).
EXTERNAL = {
    ('GET', '/health/'): 'Docker health check (docker-compose.yml)',
    ('POST', '/contact/{contact_id}/enrich'): 'n8n enrichment webhook',
    ('POST', '/contact/{contact_id}/accepted'): 'n8n / ChampMail callback webhook',
    ('GET', '/c/{token}'): 'Public "my card" page (link people open)',
    ('GET', '/c/{token}/contact.vcf'): 'Linked from the public card page',
    ('GET', '/c/{token}/card.png'): 'Linked from the public card page',
    ('GET', '/s/{token}'): 'Public shared-contact page (link people open)',
    ('GET', '/s/{token}/contact.vcf'): 'Linked from the shared-contact page',
}

# Fake / dummy responses, found by reading the handlers.  kind: fake | partly fake | stub | bug
# (2026-10-01: the admin APIs that were fake — pipelines, backup, webhook test, models, dashboard,
#  activity, users, duplicates, list members, export — were rebuilt on real data; see FIXED below.)
FAKE = {
    ('GET', '/api/sync/state'): ('fake', 'Marked "Mock sync state": always 0 pending / 0 failed and "last synced" = now. '
                                          'Not used by the apps any more (the phone reads its own outbox).'),
    ('POST', '/api/contact/{contact_id}/enrich'): ('stub', 'Always 501 "not connected to a data provider" (no enrichment provider is set up).'),
}

# Were fake / dummy, made real on 2026-10-01 (admin console rebuild).
FIXED = [
    ('GET', '/api/admin/dashboard', 'Only real counts (no 1248 / "14 reps" / "100% sync"); levels, 7-day chart, events, pipelines, workspace totals.'),
    ('GET', '/api/admin/activity', 'Real timeline with real names: contacts added, AI scores, pipelines, announcements, files, deletions, new accounts.'),
    ('GET', '/api/admin/users', 'Real contact counts, added today and last activity (was 35×n and "now").'),
    ('POST', '/api/admin/users', 'Real hashed password the person signs in with (was a placeholder hash); duplicate email → 409.'),
    ('GET', '/api/admin/contacts/duplicates', 'Real email / phone / name+company matches per owner; no invented "dup-temp" pair.'),
    ('GET', '/api/admin/contact_lists/{list_id}/contacts', 'Empty list returns [] (was 3 random contacts).'),
    ('POST', '/api/admin/contact_lists/{list_id}/contacts', 'Skips unknown ids and contacts already in the list; reports counts.'),
    ('GET', '/api/admin/pipelines', 'Real pipeline runs (ContactPipelineDB) with their research, pitch, slides and deck link.'),
    ('POST', '/api/contact/{contact_id}/pipeline/run', 'Starts the real research → score → pitch → deck → attach pipeline.'),
    ('POST', '/api/contact/{contact_id}/pipeline/retry', 'Clears failed runs and starts the real pipeline again.'),
    ('GET', '/api/contact/{contact_id}/pipeline', "The contact's latest real run (status none if never run)."),
    ('POST', '/api/admin/backup', 'Real backup: to BACKUP_DATABASE_URL, else a consistent SQLite snapshot in backend/backups/.'),
    ('GET', '/api/admin/backup/status', 'Real target, last backup and snapshots on disk.'),
    ('POST', '/api/admin/webhook/test', 'Really POSTs a test event to WEBHOOK_URL and reports status / latency / response.'),
    ('GET', '/api/admin/export', 'Real CSV file download with the Title column fixed and the owner added.'),
    ('GET', '/api/models', 'The OpenRouter models this server really uses, and what for.'),
]

# Demo data in the database (not an API bug, but the APIs return it).
DEMO_DATA = [
    'Switched off on 2026-09-30 (is_active = 0, not deleted; backup: apps/backend/eventscout.db.bak-2026-09-30-before-demo-retire): '
    '3 seeded announcements and 4 seeded placeholder files. /api/broadcasts/active and /api/files/list now return only real rows.',
    'Still in the database from the old demo seeder (SEED_DEMO_DATA, now off): 20 demo contacts owned by the admin account '
    '(+1 (555) phone numbers, event "WHX 2026") and 3 demo contact lists. They show in the admin console and in that account\'s app.',
]

NOTES = {
    ('GET', '/debug'): 'Public diagnostic output — remove or lock down.',
    ('GET', '/card/{token}'): 'Old card page on a retired domain; the app uses /c/{token}.',
    ('GET', '/api/shared/{token}'): 'JSON version of a shared contact; the /s/{token} page is rendered on the server instead.',
    ('POST', '/api/admin/contact_lists/{list_id}/contacts'): 'The admin console has no "add contacts to list" action yet.',
}

# ----------------------------------------------------------------------------------------------


def read(p):
    with open(p, encoding='utf-8', errors='ignore') as f:
        return f.read()


def walk(base, exts):
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            if f.endswith(exts):
                yield os.path.join(d, f)


def norm(p):
    p = re.sub(r'\$\{[^}]*\}', '{}', p)
    p = re.sub(r'\{[^}/]*\}', '{}', p)
    return p.split('?')[0].rstrip('/') or '/'


def same_path(a, b):
    a, b = norm(a).strip('/').split('/'), norm(b).strip('/').split('/')
    return len(a) == len(b) and all(x == y or x == '{}' or y == '{}' for x, y in zip(a, b))


def rel(p):
    return os.path.relpath(p, ROOT).replace('\\', '/')


def in_list(method, path, items):
    return any(m == method and same_path(p, path) for m, p, *_ in items)


# ---------------- backend routes ----------------
routes = []
dec = re.compile(r'^@(\w+)\.(get|post|put|patch|delete)\(\s*["\']([^"\']*)["\']([^\n]*)', re.M)
MOCK_MARK = re.compile(r'\b(mock|fake|dummy|placeholder data|lorem)\b', re.I)
for fn in sorted(os.listdir(BACK)):
    if not fn.endswith('.py') or fn.startswith('test'):
        continue
    src = read(os.path.join(BACK, fn))
    prefix = {m.group(1): (re.search(r'prefix\s*=\s*["\']([^"\']*)', m.group(2)) or [None, ''])[1]
              for m in re.finditer(r'^(\w+)\s*=\s*APIRouter\(([^)]*)\)', src, re.M)}
    decs = list(dec.finditer(src))
    for i, m in enumerate(decs):
        obj, meth, path, rest = m.groups()
        full = (prefix.get(obj, '') if obj != 'app' else '') + path
        body = src[m.end():(decs[i + 1].start() if i + 1 < len(decs) else len(src))]
        sig = body[:900]
        handler = (re.search(r'def\s+(\w+)', sig) or [None, '?'])[1]
        if 'require_admin' in rest or 'require_admin' in sig or full.startswith(('/api/admin', '/admin')):
            access = 'admin'
        elif re.match(r'^/(c|s|card)/|^/api/shared/|^/(register|login)/?$|^/auth/|^/health|^/api/health|^/$', full):
            access = 'public'
        else:
            access = 'user'
        # automatic hint: mock / fake markers in the handler's code — docstrings and comments are
        # skipped (they often describe what the route used to fake), except an explicit "Mock …" docstring
        code = re.sub(r'#[^\n]*', '', re.sub(r'"""[\s\S]*?"""', '', body))
        suspect = [x.group(0) for x in MOCK_MARK.finditer(code)]
        doc_m = re.search(r'"""\s*(Mock|Fake|Dummy)\b', body[:600])
        if doc_m:
            suspect.append(doc_m.group(1))
        routes.append({'method': meth.upper(), 'path': full, 'file': f'apps/backend/{fn}',
                       'line': src.count('\n', 0, m.start()) + 1, 'handler': handler, 'access': access,
                       '_suspect': sorted(set(s.lower() for s in suspect))})


def find_routes(method, path):
    exact = [r for r in routes if r['method'] == method and norm(r['path']) == norm(path)]
    return exact or [r for r in routes if r['method'] == method and same_path(r['path'], path)]


# ---------------- mobile client (api.ts) ----------------
API = os.path.join(MOB, 'src', 'lib', 'api.ts')
api_src = read(API)
app_files = {p: read(p) for base in (os.path.join(MOB, 'app'), os.path.join(MOB, 'src')) for p in walk(base, ('.ts', '.tsx'))}
starts = [(m.start(), m.group(1)) for m in re.finditer(r'^  (?:async\s+)?([a-zA-Z_]\w*)\s*(?:<[^>]*>)?\(', api_src, re.M)
          if m.group(1) not in ('if', 'for', 'while', 'switch', 'catch', 'constructor', 'return')]
call_re = re.compile(r'this\.client\.(get|post|put|patch|delete)\s*(?:<[^()]*?>)?\(\s*([`\'"])([^`\'"]+)\2')
mobile_fns = []
for i, (pos, name) in enumerate(starts):
    body = api_src[pos:(starts[i + 1][0] if i + 1 < len(starts) else len(api_src))]
    calls = [(c.group(1).upper(), c.group(3)) for c in call_re.finditer(body)]
    # XMLHttpRequest calls (streaming): xhr.open('POST', ...) + the '/path/' literals in that function
    xo = re.search(r'xhr\.open\(\s*[\'"](\w+)[\'"]', body)
    if xo:
        calls += [(xo.group(1).upper(), p) for p in re.findall(r'[\'"](/[\w/-]+/)[\'"]', body)]
    used_in = sorted({rel(p) for p, t in app_files.items() if p != API and re.search(r'\bapi\s*\.\s*' + name + r'\b', t)})
    mobile_fns.append({'name': name, 'line': api_src.count('\n', 0, pos) + 1, 'calls': calls, 'usedIn': used_in, 'body': body})
for f in mobile_fns:  # used only through another api.ts function
    f['viaApi'] = [g['name'] for g in mobile_fns if g is not f and re.search(r'this\.' + f['name'] + r'\b', g['body'])]

mobile_calls = []  # (method, path, via)
for f in mobile_fns:
    if f['usedIn'] or f['viaApi']:
        mobile_calls += [(m, p, f"api.{f['name']}") for m, p in f['calls']]
eng = read(os.path.join(MOB, 'src', 'sync', 'engine.ts'))
for m in re.finditer(r'fetch\(\s*`\$\{\w+\}([^`]+)`\s*,\s*\{[^}]*?method:\s*[\'"](\w+)', eng, re.S):
    mobile_calls.append((m.group(2).upper(), m.group(1), 'sync/engine.ts'))

# ---------------- admin console ----------------
admin_calls = []
lit = re.compile(r'''(apiFetch|fetch)\(\s*([`'"])(/[^`'"]*)\2(\s*,\s*\{[^)]{0,400}?\})?''', re.S)
for p in walk(ADMIN, ('.ts', '.tsx')):
    t = read(p)
    for m in lit.finditer(t):
        mm = re.search(r"method:\s*['\"](\w+)", m.group(4) or '')
        path = m.group(3)
        if '${' in path and '}' not in path.split('${')[-1]:
            path = path.split('${')[0]  # `/api/contacts${qs ? `?${qs}` : ''}` → /api/contacts
        admin_calls.append(((mm.group(1) if mm else 'GET').upper(), path, rel(p)))
    for m in re.finditer(r"use\w*Stream\(\s*['\"](/[^'\"]+)", t):  # EventSource: useEventStream('/api/live')
        admin_calls.append(('GET', m.group(1), rel(p)))
    # typed client (src/lib/api.ts): get<T>(`/api/…`), post(…), patch(…), put(…), del(…)
    for m in re.finditer(r"\b(get|post|patch|put|del)\s*(?:<[^()]*?>)?\(\s*([`'\"])(/[^`'\"]*)\2", t):
        path = m.group(3).split("${qs(")[0]  # query string helper: `/api/x${qs({ a, ...b })}` → /api/x
        admin_calls.append(({'del': 'DELETE'}.get(m.group(1), m.group(1).upper()), path, rel(p)))

# ---------------- classify routes ----------------
for r in routes:
    key = (r['method'], r['path'])
    r['usedBy'] = []
    mob = sorted({v for m, p, v in mobile_calls if m == r['method'] and r in find_routes(m, p)})
    adm = sorted({v for m, p, v in admin_calls if m == r['method'] and same_path(r['path'], p)})
    if mob:
        r['usedBy'].append('mobile')
        r['mobileVia'] = mob
    if adm:
        r['usedBy'].append('admin')
        r['adminIn'] = adm
    if key in EXTERNAL:
        r['usedBy'].append('external')
        r['externalUse'] = EXTERNAL[key]
    r['status'] = 'used' if r['usedBy'] else 'unused'
    fk = next((v for (m, p), v in FAKE.items() if m == r['method'] and same_path(p, r['path']) and r['path'].startswith('/api') == p.startswith('/api')), None)
    if fk:
        r['fake'] = {'kind': fk[0], 'detail': fk[1]}
    elif r['_suspect']:
        r['fake'] = {'kind': 'suspected', 'detail': 'Handler code mentions: ' + ', '.join(r['_suspect']) + ' — review.'}
    if key in NOTES:
        r['note'] = NOTES[key]
    if in_list(r['method'], r['path'], NEWLY_USED_IN_MOBILE):
        r['newlyUsedInMobile'] = True
    if in_list(r['method'], r['path'], NEW_ENDPOINTS):
        r['newEndpoint'] = True
    # legacy = an old app.py route (pre-/api, from the retired web frontend) that nothing calls any more.
    # app.py routes still in use (sign-in, password, profile, dashboard, export…) are current, not legacy.
    r['legacy'] = r['file'].endswith('app.py') and not r['path'].startswith('/api') and r['status'] == 'unused'
    del r['_suspect']

# ---------------- mobile functions + broken ----------------
broken = []
for f in mobile_fns:
    f['status'] = 'used' if (f['usedIn'] or f['viaApi']) else 'unused'
    f['routes'] = []
    for m, p in f['calls']:
        hit = find_routes(m, p)
        f['routes'].append({'method': m, 'path': p, 'exists': bool(hit),
                            'access': hit[0]['access'] if hit else None,
                            'backend': f"{hit[0]['file']}:{hit[0]['line']}" if hit else None,
                            'fake': hit[0].get('fake') if hit else None})
        if not hit:
            broken.append({'client': 'mobile', 'caller': f"api.{f['name']} (api.ts:{f['line']})", 'method': m, 'path': p,
                           'calledByApp': f['status'] == 'used', 'problem': 'No backend route with this method and path (would 404 / 405).'})
for m, p, where in admin_calls:
    if not find_routes(m, p):
        broken.append({'client': 'admin', 'caller': where, 'method': m, 'path': p, 'calledByApp': True,
                       'problem': 'No backend route with this method and path (would 404 / 405).'})


def why_unused(f):
    if any(not r['exists'] for r in f['routes']):
        return 'Broken: the route does not exist.'
    if f['routes'] and all(r['access'] == 'admin' for r in f['routes']):
        return 'Admin-only route: a rep would be refused.'
    if any(norm(r['path']) in ('/add_contact', '/search', '/user/profile') for r in f['routes']):
        return 'Old duplicate of an API the app already uses.'
    fk = next((r['fake'] for r in f['routes'] if r.get('fake')), None)
    if fk:
        return f"{fk['kind'].capitalize()}: {fk['detail']}"
    return 'Not called by any screen.'


mobile_out = []
for f in mobile_fns:
    if not f['calls']:
        continue  # helpers without an HTTP call
    item = {'function': f"api.{f['name']}", 'line': f"apps/mobile/src/lib/api.ts:{f['line']}", 'status': f['status'],
            'routes': f['routes'], 'usedIn': f['usedIn']}
    if f['viaApi']:
        item['viaApi'] = f['viaApi']
    if f['status'] == 'unused':
        item['whyUnused'] = why_unused(f)
    mobile_out.append(item)

# ---------------- summary ----------------
ACCESS = ('admin', 'user', 'public')
by_access = {a: {'total': sum(r['access'] == a for r in routes),
                 'used': sum(r['access'] == a and r['status'] == 'used' for r in routes),
                 'unused': sum(r['access'] == a and r['status'] == 'unused' for r in routes)} for a in ACCESS}
fakes = [r for r in routes if r.get('fake')]
summary = {
    'backendRoutes': len(routes),
    'used': sum(r['status'] == 'used' for r in routes),
    'unused': sum(r['status'] == 'unused' for r in routes),
    'byAccess': by_access,
    'legacyRoutes': sum(r['legacy'] for r in routes),
    'legacyUnused': sum(r['legacy'] and r['status'] == 'unused' for r in routes),
    'usedByMobile': sum('mobile' in r['usedBy'] for r in routes),
    'usedByAdmin': sum('admin' in r['usedBy'] for r in routes),
    'usedExternally': sum('external' in r['usedBy'] for r in routes),
    'mobileClientFunctions': len(mobile_out),
    'mobileClientUsed': sum(f['status'] == 'used' for f in mobile_out),
    'mobileClientUnused': sum(f['status'] == 'unused' for f in mobile_out),
    'broken': len(broken),
    'fakeOrDummy': {k: sum(r['fake']['kind'] == k for r in fakes) for k in ('fake', 'partly fake', 'stub', 'bug', 'suspected')},
    'fakeStillCalled': sum(r['status'] == 'used' for r in fakes),
    'newEndpoints': len(NEW_ENDPOINTS),
    'newlyUsedInMobile': len(NEWLY_USED_IN_MOBILE),
}

doc = {
    'generatedAt': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'generator': 'scripts/api_inventory.py',
    'legend': {
        'access': {'admin': 'needs an admin sign-in', 'user': 'any signed-in user (mobile app)', 'public': 'no sign-in'},
        'usedBy': {'mobile': 'the mobile app calls it', 'admin': 'the admin console calls it', 'external': 'webhooks / public links / health check'},
        'fake.kind': {'fake': 'response invented, nothing real happens', 'partly fake': 'real data mixed with invented values',
                      'stub': 'always refuses', 'bug': 'real data, wrong fields', 'suspected': 'code mentions mock/fake — review'},
        'legacy': 'old route in app.py (pre-/api), mostly duplicated by /api routes',
    },
    'summary': summary,
    'newEndpoints': [{'method': m, 'path': p, 'purpose': w, 'usedBy': (find_routes(m, p) or [{}])[0].get('usedBy', [])} for m, p, w in NEW_ENDPOINTS],
    'newlyUsedInMobile': [{'method': m, 'path': p, 'where': w} for m, p, w in NEWLY_USED_IN_MOBILE],
    'broken': broken,
    'fakeOrDummy': [{'method': r['method'], 'path': r['path'], 'access': r['access'], 'status': r['status'], 'usedBy': r['usedBy'],
                     'kind': r['fake']['kind'], 'detail': r['fake']['detail'], 'at': f"{r['file']}:{r['line']}"} for r in fakes],
    'fixedFakes': [{'method': m, 'path': p, 'now': w} for m, p, w in FIXED],
    'demoData': DEMO_DATA,
    'mobileClient': mobile_out,
    'backendRoutes': routes,
}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'api-inventory.json'), 'w', encoding='utf-8') as fh:
    json.dump(doc, fh, indent=2, ensure_ascii=False)

# ---------------- markdown ----------------
L = []
w = L.append


def cell(s):
    return str(s or '').replace('|', '\\|').replace('\n', ' ')


def used_by(r):
    parts = []
    if 'mobile' in r['usedBy']:
        parts.append('mobile (' + ', '.join(r['mobileVia'][:2]) + ')')
    if 'admin' in r['usedBy']:
        parts.append('admin (' + ', '.join(x.split('/')[-1] for x in r['adminIn'][:2]) + ')')
    if 'external' in r['usedBy']:
        parts.append('external: ' + r['externalUse'])
    return '<br>'.join(parts)


def flags(r):
    out = []
    if r.get('newEndpoint'):
        out.append('🆕 new')
    if r.get('newlyUsedInMobile'):
        out.append('✅ newly used in app')
    if r.get('fake'):
        out.append(f"⚠️ {r['fake']['kind']}")
    if r['legacy']:
        out.append('legacy')
    return ' '.join(out)


s = summary
w('# EventScout API inventory')
w('')
w(f"Generated {doc['generatedAt']} by `scripts/api_inventory.py` from the code — re-run it after API changes. "
  'Machine-readable: `docs/api-inventory.json`.')
w('')
w('## Summary')
w('')
w('| | Total | Used | Unused |')
w('|---|---|---|---|')
w(f"| **All backend routes** | {s['backendRoutes']} | {s['used']} | {s['unused']} |")
for a in ACCESS:
    w(f"| {a.capitalize()} APIs | {by_access[a]['total']} | {by_access[a]['used']} | {by_access[a]['unused']} |")
w(f"| of the unused: legacy `app.py` routes (old web frontend) | {s['legacyRoutes']} | — | {s['legacyUnused']} |")
w('')
w('| | Count |')
w('|---|---|')
w(f"| Used by the mobile app | {s['usedByMobile']} |")
w(f"| Used by the admin console | {s['usedByAdmin']} |")
w(f"| Used from outside (webhooks, public links, health check) | {s['usedExternally']} |")
w(f"| Mobile client functions (`api.ts`) | {s['mobileClientFunctions']} — used {s['mobileClientUsed']}, unused {s['mobileClientUnused']} |")
w(f"| Broken calls | {s['broken']} |")
fd = s['fakeOrDummy']
w(f"| Fake / dummy APIs | {sum(fd.values())} — fake {fd['fake']}, partly fake {fd['partly fake']}, stub {fd['stub']}, bug {fd['bug']}, suspected {fd['suspected']} ({s['fakeStillCalled']} still called by a client) |")
w(f"| New APIs created | {s['newEndpoints']} |")
w(f"| Were unused, now used in the mobile app | {s['newlyUsedInMobile']} |")
w('')
w('Legend — ⚠️ fake / dummy · 🆕 new API · ✅ newly used in the app · legacy = old `app.py` route (pre-`/api`)')
w('')

w('## 1. Fake / dummy APIs')
w('')
w('| Kind | Method | Path | Access | Called by | What is fake | Where |')
w('|---|---|---|---|---|---|---|')
order = {'fake': 0, 'partly fake': 1, 'stub': 2, 'bug': 3, 'suspected': 4}
for r in sorted(fakes, key=lambda r: (order[r['fake']['kind']], r['path'])):
    w(f"| {r['fake']['kind']} | {r['method']} | `{r['path']}` | {r['access']} | {cell(used_by(r)) or 'nobody'} | {cell(r['fake']['detail'])} | `{r['file'].split('/')[-1]}:{r['line']}` |")
w('')
w('Were fake / dummy — now real:')
w('')
w('| Method | Path | Now |')
w('|---|---|---|')
for m, p, now in FIXED:
    w(f'| {m} | `{p}` | {cell(now)} |')
w('')
w('Demo data in the database (returned by real APIs):')
w('')
for d in DEMO_DATA:
    w(f'- {d}')
w('')

w('## 2. Broken')
w('')
if broken:
    w('| Client | Caller | Method | Path | Called by the app? | Problem |')
    w('|---|---|---|---|---|---|')
    for b in broken:
        w(f"| {b['client']} | `{b['caller']}` | {b['method']} | `{b['path']}` | {'yes' if b['calledByApp'] else 'no'} | {b['problem']} |")
else:
    w('None.')
w('')

w('## 3. New APIs created')
w('')
w('| Method | Path | Purpose | Used by |')
w('|---|---|---|---|')
for n in doc['newEndpoints']:
    w(f"| {n['method']} | `{n['path']}` | {cell(n['purpose'])} | {', '.join(n['usedBy']) or '—'} |")
w('')

w('## 4. Were unused — now used in the mobile app')
w('')
w('| Method | Path | Where in the app |')
w('|---|---|---|')
for m, p, where in NEWLY_USED_IN_MOBILE:
    w(f'| {m} | `{p}` | {cell(where)} |')
w('')

w('## 5. Mobile client functions the app never calls')
w('')
w('| Function | Calls | Why unused | Line |')
w('|---|---|---|---|')
for f in mobile_out:
    if f['status'] == 'unused':
        calls = '<br>'.join(f"{r['method']} `{r['path']}`" for r in f['routes'])
        w(f"| `{f['function']}` | {calls} | {cell(f['whyUnused'])} | `{f['line']}` |")
w('')

w('## 6. All APIs by access')
for i, a in enumerate(ACCESS):
    title = {'admin': 'Admin APIs (admin sign-in)', 'user': 'User APIs (any signed-in user — the mobile app)', 'public': 'Public APIs (no sign-in)'}[a]
    w('')
    w(f"### 6.{i + 1} {title} — {by_access[a]['total']} total, {by_access[a]['used']} used, {by_access[a]['unused']} unused")
    for status in ('used', 'unused'):
        rows = sorted([r for r in routes if r['access'] == a and r['status'] == status], key=lambda r: (r['legacy'], r['path'], r['method']))
        w('')
        w(f"**{status.capitalize()} ({len(rows)})**")
        w('')
        if not rows:
            w('None.')
            continue
        if status == 'used':
            w('| Method | Path | Used by | Flags | Notes | Handler |')
            w('|---|---|---|---|---|---|')
            for r in rows:
                note = r['fake']['detail'] if r.get('fake') else r.get('note', '')
                w(f"| {r['method']} | `{r['path']}` | {cell(used_by(r))} | {flags(r)} | {cell(note)} | `{r['file'].split('/')[-1]}:{r['line']}` {r['handler']} |")
        else:
            w('| Method | Path | Flags | Notes | Handler |')
            w('|---|---|---|---|---|')
            for r in rows:
                note = r['fake']['detail'] if r.get('fake') else r.get('note', '')
                w(f"| {r['method']} | `{r['path']}` | {flags(r)} | {cell(note)} | `{r['file'].split('/')[-1]}:{r['line']}` {r['handler']} |")
w('')
with open(os.path.join(OUT, 'API_INVENTORY.md'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(L))

# ---------------- console report ----------------
if '--quiet' not in sys.argv:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    line = '-' * 100
    P = print

    def rt(r):
        return f"  {r['method']:<7} {r['path']:<52} {r['file'].split('/')[-1]}:{r['line']}"

    P('=' * 100)
    P('EVENTSCOUT API INVENTORY'.center(100))
    P('=' * 100)
    P(f"Backend routes: {s['backendRoutes']}   used {s['used']}   unused {s['unused']}   (of the unused, {s['legacyUnused']} are legacy app.py routes)")
    for a in ACCESS:
        P(f"  {a:<7} {by_access[a]['total']:>3} total   {by_access[a]['used']:>3} used   {by_access[a]['unused']:>3} unused")
    P(f"Used by: mobile {s['usedByMobile']}   admin console {s['usedByAdmin']}   outside {s['usedExternally']}")
    P(f"Mobile api.ts functions: {s['mobileClientFunctions']} (used {s['mobileClientUsed']}, unused {s['mobileClientUnused']})")
    P(f"Broken: {s['broken']}   Fake/dummy: {sum(fd.values())} {fd}   New APIs: {s['newEndpoints']}   Newly used in app: {s['newlyUsedInMobile']}")

    P('\n' + line + '\n1. FAKE / DUMMY APIs\n' + line)
    for r in sorted(fakes, key=lambda r: (order[r['fake']['kind']], r['path'])):
        P(f"  [{r['fake']['kind'].upper()}] {r['method']} {r['path']}  ({r['access']}; called by: {', '.join(r['usedBy']) or 'nobody'}) — {r['file'].split('/')[-1]}:{r['line']}")
        P(f"      {r['fake']['detail']}")
    P('  Were fake — now real:')
    for m, p, now in FIXED:
        P(f"    {m:<7} {p:<48} {now}")
    P('  Demo data:')
    for d in DEMO_DATA:
        P(f'    - {d}')

    P('\n' + line + '\n2. BROKEN\n' + line)
    for b in broken or [None]:
        P('  none' if b is None else f"  [{b['client']}] {b['method']} {b['path']}  ← {b['caller']}  (called by the app: {'yes' if b['calledByApp'] else 'no'}) — {b['problem']}")

    P('\n' + line + '\n3. NEW APIs CREATED\n' + line)
    for n in doc['newEndpoints']:
        P(f"  {n['method']:<7} {n['path']:<45} {n['purpose']}  [used by: {', '.join(n['usedBy']) or '—'}]")

    P('\n' + line + '\n4. WERE UNUSED — NOW USED IN THE MOBILE APP\n' + line)
    for m, p, where in NEWLY_USED_IN_MOBILE:
        P(f"  {m:<7} {p:<45} {where}")

    P('\n' + line + '\n5. MOBILE api.ts FUNCTIONS THE APP NEVER CALLS\n' + line)
    for f in mobile_out:
        if f['status'] == 'unused':
            P(f"  {f['function']:<30} {', '.join(r['method'] + ' ' + r['path'] for r in f['routes'])}")
            P(f"      {f['whyUnused']}")

    for a in ACCESS:
        P('\n' + line + f"\n6. {a.upper()} APIs — {by_access[a]['total']} total, {by_access[a]['used']} used, {by_access[a]['unused']} unused\n" + line)
        for status in ('used', 'unused'):
            rows = sorted([r for r in routes if r['access'] == a and r['status'] == status], key=lambda r: (r['legacy'], r['path'], r['method']))
            P(f"  {status.upper()} ({len(rows)})")
            for r in rows:
                tag = flags(r)
                who = ', '.join(r['usedBy'])
                P(rt(r) + (f"  [{who}]" if who else '') + (f"  {tag}" if tag else ''))
    P('\n' + line)
    P('Wrote ' + rel(os.path.join(OUT, 'API_INVENTORY.md')) + ' and ' + rel(os.path.join(OUT, 'api-inventory.json')))
