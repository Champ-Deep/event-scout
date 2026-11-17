"""End-to-end smoke test for an Event Scout backend (written for the Cloudflare test deploy).

Checks: health + DB, Better Auth sign-up/sign-in, that the old shared key and the old /login/
no longer work, that one account cannot act as another, an existing user's sign-in with their
contact count compared to Railway, 5 card scans fired at the same time with the same
upload filename (the app's real behaviour), AI extraction against the bench answers,
lead scoring by a real model, and the chat assistant. Creates one throwaway user.
Never point this at production: it refuses the Railway production URL.

Usage: BASE_URL=https://event-scout-cf-test.<sub>.workers.dev python scripts/cf_smoke.py
Optional: EXISTING_EMAIL, EXISTING_PASSWORD, EXPECTED_CONTACTS (count on Railway) for the
existing-user check.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = os.environ["BASE_URL"].rstrip("/")
# The key hardcoded in the old frontend. On Cloudflare it must no longer open anything.
PUBLIC_KEY = "OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s"
if "event-scout-production" in BASE:
    sys.exit("Refusing to run against production.")
CARDS = Path(__file__).resolve().parent.parent / "bench" / "cards"
H = {}  # the Worker adds the backend key for signed-in sessions


def session():
    s = requests.Session()
    s.headers["Origin"] = BASE  # Better Auth only accepts same-origin POSTs
    return s


def sign_in(s, email, password, name=None):
    path = "sign-up/email" if name else "sign-in/email"
    body = {"email": email, "password": password, **({"name": name} if name else {})}
    r = s.post(f"{BASE}/api/auth/{path}", json=body, timeout=30)
    if not r.ok:
        return r, None
    me = s.get(f"{BASE}/api/me", timeout=30)
    return me, (me.json().get("user_id") if me.ok else None)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    t0 = time.time()
    r = requests.get(f"{BASE}/health/", timeout=300)  # first call may cold-start the container
    h = r.json() if r.ok else {}
    check("health", r.ok, f"{r.status_code} in {time.time() - t0:.0f}s")
    check("database connected", h.get("database_connected"), str(h.get("database")))
    check("openrouter configured", h.get("openrouter_configured"))

    r = requests.post(f"{BASE}/login/", json={"email": "x@example.com", "password": "x"}, timeout=30)
    check("old /login/ closed", r.status_code == 410, str(r.status_code))

    email = f"smoke-{int(time.time())}@eventscout-smoke.example.com"
    r, uid = sign_in(session(), email, "smoke-pass-123", name="CF Smoke")
    check("better auth sign-up", uid, str(r.status_code))
    S = session()
    r, uid = sign_in(S, email, "smoke-pass-123")
    if not check("better auth sign-in", uid, str(r.status_code)):
        return

    r = requests.get(f"{BASE}/list_contacts/", params={"user_id": uid}, headers={"X-API-Key": PUBLIC_KEY}, timeout=30)
    check("old public key alone is refused", r.status_code in (401, 403), str(r.status_code))
    r = S.get(f"{BASE}/list_contacts/", params={"user_id": "00000000-0000-0000-0000-000000000000"}, timeout=30)
    check("cannot act as another account", r.status_code == 403, str(r.status_code))

    ex_email, ex_pass = os.environ.get("EXISTING_EMAIL"), os.environ.get("EXISTING_PASSWORD")
    if ex_email and ex_pass:
        E = session()
        r, ex_uid = sign_in(E, ex_email, ex_pass)
        if check("existing user signs in with old password", ex_uid, str(r.status_code)):
            r = E.get(f"{BASE}/list_contacts/", params={"user_id": ex_uid}, timeout=60)
            n = len(r.json().get("contacts", [])) if r.ok else -1
            want = os.environ.get("EXPECTED_CONTACTS", "").strip()
            check("existing user's contacts match Railway", r.ok and want.isdigit() and n == int(want),
                  f"cloudflare={n} railway={want or '?'}")
    else:
        check("existing-user check configured", False, "set SMOKE_EXISTING_EMAIL and SMOKE_EXISTING_PASSWORD")

    names = ["c01", "c03", "c05", "c07", "c09"]

    def scan(n):
        img = CARDS / f"{n}.png"
        t = time.time()
        resp = S.post(f"{BASE}/add_contact_from_image/", params={"user_id": uid}, headers=H,
                             files={"file": ("business-card.jpg", img.read_bytes(), "image/jpeg")}, timeout=120)
        return n, resp, time.time() - t

    with ThreadPoolExecutor(len(names)) as pool:
        scans = list(pool.map(scan, names))
    ids = []
    for n, resp, secs in scans:
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        f = body.get("extracted_fields", {})
        want = json.loads((CARDS / f"{n}.json").read_text())
        ok = resp.ok and body.get("status") == "success" and f.get("name", "").lower() == want["name"].lower()
        check(f"concurrent scan {n}", ok,
              f"{resp.status_code} {secs:.1f}s via {f.get('_source', '?')} got={f.get('name')!r} want={want['name']!r}")
        if body.get("contact_id"):
            ids.append(body["contact_id"])
    check("5 distinct contacts saved", len(set(ids)) == len(names), f"{len(set(ids))} ids")

    r = S.get(f"{BASE}/list_contacts/", params={"user_id": uid}, headers=H, timeout=30)
    listed = r.json().get("contacts", []) if r.ok else []
    check("list contacts", r.ok and len(listed) >= len(names), f"{len(listed)} listed")

    if ids:
        r = S.post(f"{BASE}/contact/{ids[0]}/score", params={"user_id": uid}, headers=H, timeout=90)
        b = r.json() if r.ok else {}
        check("lead scoring by a model", r.ok and "temporarily unavailable" not in b.get("reasoning", "")
              and "not configured" not in b.get("reasoning", ""),
              f"score={b.get('score')} {b.get('temperature')}")

    r = S.post(f"{BASE}/converse/", headers=H, timeout=90,
                      json={"query": "Who did I meet from Medanta?", "user_id": uid})
    b = r.json() if r.ok else {}
    check("chat assistant", r.ok and len(b.get("response", "")) > 20, f"model={b.get('model')}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed. Test user: {email}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
