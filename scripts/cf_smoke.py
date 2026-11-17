"""End-to-end smoke test for an Event Scout backend (written for the Cloudflare test deploy).

Checks: health + DB, register/login, 5 card scans fired at the same time with the same
upload filename (the app's real behaviour), AI extraction against the bench answers,
lead scoring by a real model, and the chat assistant. Creates one throwaway user.
Never point this at production: it refuses the Railway production URL.

Usage: BASE_URL=https://event-scout-cf-test.<sub>.workers.dev APP_API_KEY=... python scripts/cf_smoke.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = os.environ["BASE_URL"].rstrip("/")
KEY = os.environ["APP_API_KEY"]
if "event-scout-production" in BASE:
    sys.exit("Refusing to run against production.")
CARDS = Path(__file__).resolve().parent.parent / "bench" / "cards"
H = {"X-API-Key": KEY}
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

    email = f"smoke-{int(time.time())}@eventscout-smoke.example.com"
    r = requests.post(f"{BASE}/register/", json={"name": "CF Smoke", "email": email, "password": "smoke-pass-123"}, timeout=30)
    check("register", r.ok, str(r.status_code))
    r = requests.post(f"{BASE}/login/", json={"email": email, "password": "smoke-pass-123"}, timeout=30)
    if not check("login", r.ok, str(r.status_code)):
        return
    uid = r.json()["user_id"]

    names = ["c01", "c03", "c05", "c07", "c09"]

    def scan(n):
        img = CARDS / f"{n}.png"
        t = time.time()
        resp = requests.post(f"{BASE}/add_contact_from_image/", params={"user_id": uid}, headers=H,
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

    r = requests.get(f"{BASE}/list_contacts/", params={"user_id": uid}, headers=H, timeout=30)
    listed = r.json().get("contacts", []) if r.ok else []
    check("list contacts", r.ok and len(listed) >= len(names), f"{len(listed)} listed")

    if ids:
        r = requests.post(f"{BASE}/contact/{ids[0]}/score", params={"user_id": uid}, headers=H, timeout=90)
        b = r.json() if r.ok else {}
        check("lead scoring by a model", r.ok and "temporarily unavailable" not in b.get("reasoning", "")
              and "not configured" not in b.get("reasoning", ""),
              f"score={b.get('score')} {b.get('temperature')}")

    r = requests.post(f"{BASE}/converse/", headers=H, timeout=90,
                      json={"query": "Who did I meet from Medanta?", "user_id": uid})
    b = r.json() if r.ok else {}
    check("chat assistant", r.ok and len(b.get("response", "")) > 20, f"model={b.get('model')}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed. Test user: {email}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
