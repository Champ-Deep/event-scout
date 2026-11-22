#!/usr/bin/env python3
"""Production smoke test for Event Scout card scan.

WARNING: This script creates one user per run and there is no user delete endpoint,
so run it only with Deep's go-ahead (the two leftover users are on t_18e0aa2f).
APP_API_KEY is read dynamically from Railway at runtime and is never printed or committed.
"""
import datetime, json, os, secrets, subprocess, sys, time, urllib.error, urllib.request, uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://event-scout-production.up.railway.app'
KEY = json.loads(subprocess.run(['railway', 'variables', '--json'], cwd=REPO, capture_output=True, text=True, check=True).stdout)['APP_API_KEY']


def call(method, path, body=None, headers=None, raw=None, ctype='application/json'):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {'X-API-Key': KEY, **(headers or {})}
    if data is not None:
        h['Content-Type'] = ctype
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=h)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out, code = r.read(), r.status
    except urllib.error.HTTPError as e:
        out, code = e.read(), e.code
    return code, out, round(time.time() - t, 2)


def scan(uid, name):
    img = subprocess.run(['git', 'show', f'origin/master:bench/cards/{name}'], cwd=REPO, capture_output=True, check=True).stdout
    b = uuid.uuid4().hex
    raw = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + img + f'\r\n--{b}--\r\n'.encode()
    return call('POST', f'/add_contact_from_image/?user_id={uid}', raw=raw, ctype=f'multipart/form-data; boundary={b}')


ok = True
email = f'ocr-verify-{datetime.datetime.now():%Y%m%d-%H%M}@championsmail.com'
pw = secrets.token_urlsafe(16)
code, out, _ = call('POST', '/register/', {'name': 'OCR Verify (test account)', 'email': email, 'password': pw})
print('register', code, out.decode()[:120] if code != 200 else 'ok')
if code != 200:  # an earlier run today left the account and its password is gone: report, do not guess
    sys.exit(1)
uid = json.loads(out)['user_id']
code, out, _ = call('POST', '/login/', {'email': email, 'password': pw})
print('login', code, json.loads(out).get('status'))
for card in ('c03_photo.jpg', 'c12_hard.jpg'):
    ref = json.loads(subprocess.run(['git', 'show', f'origin/master:bench/cards/{card[:3]}.json'], cwd=REPO, capture_output=True, text=True, check=True).stdout)
    code, out, secs = scan(uid, card)
    d = json.loads(out) if out[:1] == b'{' else {'raw': out.decode()[:300]}
    f = d.get('extracted_fields') or {}
    print(f'\nscan {card}: HTTP {code}, {secs}s, status={d.get("status")}')
    print(json.dumps(f, ensure_ascii=False) if f else json.dumps(d, ensure_ascii=False)[:600])
    norm = lambda s: ''.join(str(s or '').lower().replace('https://', '').replace('www.', '').split()).rstrip('/')
    web = f.get('website') or next((l.get('url') for l in (f.get('links') or d.get('links') or []) if isinstance(l, dict) and l.get('label') == 'Website'), None)
    miss = [k for k in ref if norm({**f, 'website': web}.get(k)) != norm(ref[k])]
    print('mismatch vs reference:', miss or 'none')
    ok &= code == 200 and not miss
code, out, _ = call('GET', f'/list_contacts/?user_id={uid}')
lst = json.loads(out)
contacts = lst.get('contacts', lst) if isinstance(lst, dict) else lst
print('\nlist_contacts', code, [(c.get('name'), c.get('title')) for c in contacts])
ok &= len(contacts) == 2 and all(c.get('title') for c in contacts)
code, out, _ = call('GET', f'/export_contacts/?user_id={uid}&format=csv')
lines = out.decode('utf-8-sig').splitlines()
print('csv', code, 'header:', lines[0] if lines else None)
for l in lines[1:3]:
    print('  row:', l[:260])
ok &= bool(lines) and lines[0].rstrip().endswith('Title,Website')
for c in contacts:
    cid = c.get('id') or c.get('contact_id')
    code, out, _ = call('DELETE', f'/contact/{cid}?user_id={uid}')
    print('delete contact', code)
code, out, _ = call('GET', f'/list_contacts/?user_id={uid}')
left = json.loads(out)
print('contacts left:', len(left.get('contacts', left) if isinstance(left, dict) else left), '| throwaway user kept (no user delete endpoint):', email)
print('\nRESULT:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
