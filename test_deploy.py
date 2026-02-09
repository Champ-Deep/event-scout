#!/usr/bin/env python3
"""Test deployed backend for generative UI components."""
import json
import urllib.request

BASE = "https://event-scout-production.up.railway.app"
API_KEY = "OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s"
USER_ID = "6fc69c2d-d8e0-40b8-8631-a3909c89233b"  # Sreedeep (6 contacts)

def test_converse(query, label=""):
    print(f"\n{'='*60}")
    print(f"TEST: {label or query}")
    print(f"{'='*60}")

    payload = json.dumps({
        "user_id": USER_ID,
        "query": query,
        "conversation_history": []
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/converse/",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            print(f"Status: {data.get('status')}")
            print(f"Intent: {data.get('intent')}")

            ui = data.get('ui_components', [])
            print(f"UI Components: {len(ui)}")
            for i, c in enumerate(ui):
                t = c.get('type', 'unknown') if isinstance(c, dict) else 'non-dict'
                d = c.get('data') if isinstance(c, dict) else c
                print(f"  [{i}] type={t}")
                # Just show data type and length/keys
                if isinstance(d, list):
                    print(f"      data: list of {len(d)} items")
                    if d and isinstance(d[0], dict):
                        print(f"      first item keys: {list(d[0].keys())[:8]}")
                    elif d and isinstance(d[0], str):
                        print(f"      values: {d[:4]}")
                elif isinstance(d, dict):
                    print(f"      data keys: {list(d.keys())}")
                else:
                    print(f"      data: {str(d)[:100]}")

            resp_text = data.get('response', '')
            print(f"\nResponse ({len(resp_text)} chars):")
            print(f"  {resp_text[:300]}...")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP ERROR {e.code}: {body[:500]}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_converse("Who are my contacts?", "Contact listing (expects contact_cards)")
    test_converse("Show my hot leads", "Hot leads (expects score_summary)")
    print("\n\nAll tests complete!")
