#!/usr/bin/env python3
"""
Event Scout Production Backend Test Suite
Tests all critical endpoints against the live Railway API.
Run: python test_production.py

Usage: export TEST_PASSWORD=yourpassword && python test_production.py
"""

import requests
import json
import sys
import os
import io
import time
from datetime import datetime

# --- CONFIG ---
API_BASE = "https://event-scout-production.up.railway.app"
API_KEY = "OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s"
TEST_EMAIL = "deep@lakeb2b.com"
TEST_PASSWORD = os.environ.get("TEST_PASSWORD", "")  # Set via: export TEST_PASSWORD=yourpass
TIMEOUT = 60  # seconds per request (scan can be slow)

# --- STATE ---
user_id = None
scanned_contact_ids = []
manual_contact_id = None
results = []


def log(msg):
    print(f"  {msg}")


def test_result(name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    icon = "[+]" if passed else "[X]"
    print(f"\n{icon} {status} - {name}")
    if message:
        log(message)
    results.append({"name": name, "passed": passed, "message": message})
    return passed


def generate_business_card(name, email, phone, company, title="Manager"):
    """Generate a synthetic business card image using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 450), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a decent font, fall back to default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large

    # Draw a simple business card layout
    # Blue header bar
    draw.rectangle([(0, 0), (800, 8)], fill="#2563EB")

    # Company name
    draw.text((50, 30), company, fill="#2563EB", font=font_medium)

    # Name (large)
    draw.text((50, 90), name, fill="#111827", font=font_large)

    # Title
    draw.text((50, 140), title, fill="#6B7280", font=font_medium)

    # Divider line
    draw.line([(50, 190), (750, 190)], fill="#E5E7EB", width=2)

    # Contact details
    draw.text((50, 220), f"Email: {email}", fill="#374151", font=font_small)
    draw.text((50, 260), f"Phone: {phone}", fill="#374151", font=font_small)
    draw.text((50, 300), f"Company: {company}", fill="#374151", font=font_small)

    # Bottom border
    draw.rectangle([(0, 442), (800, 450)], fill="#2563EB")

    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


# ============================================================
# TEST 1: Health Check
# ============================================================
def test_health():
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    try:
        r = requests.get(f"{API_BASE}/health/", timeout=TIMEOUT)
        data = r.json()

        gemini_ok = data.get("gemini_configured") is True
        openrouter_ok = data.get("openrouter_configured") is True
        db_ok = data.get("database") == "postgresql"
        healthy = data.get("status") == "healthy"

        test_result(
            "Health Check",
            healthy and gemini_ok and openrouter_ok and db_ok,
            f"Status: {data.get('status')}, DB: {data.get('database')}, "
            f"Gemini: {gemini_ok}, OpenRouter: {openrouter_ok}, "
            f"Users: {data.get('total_users')}, Version: {data.get('version')}"
        )
    except Exception as e:
        test_result("Health Check", False, f"Error: {e}")


# ============================================================
# TEST 2: Login
# ============================================================
def test_login():
    global user_id
    print("\n" + "=" * 60)
    print("TEST 2: Login")
    print("=" * 60)
    try:
        r = requests.post(
            f"{API_BASE}/login/",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            timeout=TIMEOUT,
        )
        data = r.json()
        if r.status_code == 200 and data.get("status") == "success":
            user_id = data["user_id"]
            test_result(
                "Login",
                True,
                f"user_id: {user_id}, name: {data.get('name')}, is_admin: {data.get('is_admin')}"
            )
        else:
            test_result("Login", False, f"Status {r.status_code}: {data}")
    except Exception as e:
        test_result("Login", False, f"Error: {e}")


# ============================================================
# TEST 3: Card Scan (CRITICAL)
# ============================================================
def test_card_scan():
    global scanned_contact_ids
    print("\n" + "=" * 60)
    print("TEST 3: Card Scan (CRITICAL)")
    print("=" * 60)

    if not user_id:
        test_result("Card Scan", False, "Skipped: no user_id from login")
        return

    # Generate a synthetic business card
    log("Generating synthetic business card image...")
    card_img = generate_business_card(
        name="Sarah Johnson",
        email="sarah.johnson@techcorp.com",
        phone="+1 (555) 234-5678",
        company="TechCorp Solutions",
        title="VP of Sales"
    )
    log(f"Card image size: {card_img.getbuffer().nbytes} bytes")

    try:
        log("Uploading to /add_contact_from_image/ ...")
        start = time.time()
        r = requests.post(
            f"{API_BASE}/add_contact_from_image/?user_id={user_id}",
            headers={"X-API-Key": API_KEY},
            files={"file": ("test_card.jpg", card_img, "image/jpeg")},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        if r.status_code == 200:
            data = r.json()
            fields = data.get("extracted_fields", {})
            contact_id = data.get("contact_id")
            if contact_id:
                scanned_contact_ids.append(contact_id)

            has_name = fields.get("name", "N/A") != "N/A"
            has_email = fields.get("email", "N/A") != "N/A"

            test_result(
                "Card Scan",
                has_name or has_email,
                f"Time: {elapsed:.1f}s | Name: {fields.get('name')} | "
                f"Email: {fields.get('email')} | Phone: {fields.get('phone')} | "
                f"Company: {fields.get('company_name')} | "
                f"Source: {fields.get('_source', 'unknown')} | "
                f"Contact ID: {contact_id}"
            )
        else:
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:500]}
            test_result(
                "Card Scan",
                False,
                f"Status {r.status_code} after {elapsed:.1f}s | Detail: {data.get('detail', data)}"
            )
    except requests.Timeout:
        test_result("Card Scan", False, f"TIMEOUT after {TIMEOUT}s")
    except Exception as e:
        test_result("Card Scan", False, f"Error: {e}")


# ============================================================
# TEST 4: Verify Scanned Contact
# ============================================================
def test_verify_contact():
    print("\n" + "=" * 60)
    print("TEST 4: Verify Scanned Contact in List")
    print("=" * 60)

    if not user_id:
        test_result("Verify Contact", False, "Skipped: no user_id")
        return

    try:
        r = requests.get(
            f"{API_BASE}/list_contacts/?user_id={user_id}",
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT,
        )
        data = r.json()
        total = data.get("total_contacts", 0)
        contacts = data.get("contacts", [])

        # Check if our scanned contact is in the list
        found = False
        for c in contacts:
            if c.get("id") in scanned_contact_ids:
                found = True
                log(f"Found scanned contact: {c.get('name')} ({c.get('email')})")
                break

        test_result(
            "Verify Contact in List",
            found or len(scanned_contact_ids) == 0,
            f"Total contacts: {total}, Scanned contact found: {found}"
        )
    except Exception as e:
        test_result("Verify Contact in List", False, f"Error: {e}")


# ============================================================
# TEST 5: Manual Contact Add
# ============================================================
def test_manual_add():
    global manual_contact_id
    print("\n" + "=" * 60)
    print("TEST 5: Manual Contact Add")
    print("=" * 60)

    if not user_id:
        test_result("Manual Add", False, "Skipped: no user_id")
        return

    try:
        r = requests.post(
            f"{API_BASE}/add_contact/",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={
                "contact": {
                    "name": "Test Backend User",
                    "email": "test.backend@example.com",
                    "phone": "+1-555-000-9999",
                    "linkedin": "N/A",
                    "company_name": "Test Corp"
                },
                "user_id": user_id,
            },
            timeout=TIMEOUT,
        )
        data = r.json()
        if r.status_code == 200 and data.get("status") == "success":
            manual_contact_id = data.get("contact_id")
            test_result(
                "Manual Contact Add",
                True,
                f"Contact ID: {manual_contact_id}"
            )
        else:
            test_result("Manual Contact Add", False, f"Status {r.status_code}: {data}")
    except Exception as e:
        test_result("Manual Contact Add", False, f"Error: {e}")


# ============================================================
# TEST 6: Search
# ============================================================
def test_search():
    print("\n" + "=" * 60)
    print("TEST 6: Semantic Search")
    print("=" * 60)

    if not user_id:
        test_result("Search", False, "Skipped: no user_id")
        return

    try:
        r = requests.post(
            f"{API_BASE}/search/",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"query": "test backend", "user_id": user_id},
            timeout=TIMEOUT,
        )
        data = r.json()
        num_results = len(data.get("results", []))
        test_result(
            "Semantic Search",
            r.status_code == 200 and data.get("status") == "success",
            f"Results: {num_results}"
        )
    except Exception as e:
        test_result("Semantic Search", False, f"Error: {e}")


# ============================================================
# TEST 7: Lead Scoring
# ============================================================
def test_lead_scoring():
    print("\n" + "=" * 60)
    print("TEST 7: Lead Scoring")
    print("=" * 60)

    contact_id = scanned_contact_ids[0] if scanned_contact_ids else manual_contact_id
    if not user_id or not contact_id:
        test_result("Lead Scoring", False, "Skipped: no contact to score")
        return

    try:
        log("Waiting 5s before scoring to avoid Gemini rate limiting...")
        time.sleep(5)
        log(f"Scoring contact {contact_id}...")
        start = time.time()
        r = requests.post(
            f"{API_BASE}/contact/{contact_id}/score?user_id={user_id}",
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start
        data = r.json()

        if r.status_code == 200:
            score = data.get("score")  # API returns "score" not "lead_score"
            temp = data.get("temperature")  # API returns "temperature" not "lead_temperature"
            reasoning = data.get("reasoning", "")
            real_score = reasoning != "Scoring temporarily unavailable."
            test_result(
                "Lead Scoring",
                score is not None and temp is not None and real_score,
                f"Time: {elapsed:.1f}s | Score: {score} | Temperature: {temp} | "
                f"Reasoning: {str(reasoning)[:100]}"
            )
        else:
            test_result("Lead Scoring", False, f"Status {r.status_code}: {data}")
    except Exception as e:
        test_result("Lead Scoring", False, f"Error: {e}")


# ============================================================
# TEST 8: Cleanup
# ============================================================
def test_cleanup():
    print("\n" + "=" * 60)
    print("TEST 8: Cleanup Test Contacts")
    print("=" * 60)

    if not user_id:
        test_result("Cleanup", False, "Skipped: no user_id")
        return

    deleted = 0
    total = len(scanned_contact_ids) + (1 if manual_contact_id else 0)

    all_ids = scanned_contact_ids.copy()
    if manual_contact_id:
        all_ids.append(manual_contact_id)

    for cid in all_ids:
        try:
            r = requests.delete(
                f"{API_BASE}/contact/{cid}?user_id={user_id}",
                headers={"X-API-Key": API_KEY},
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                deleted += 1
                log(f"Deleted: {cid}")
            else:
                log(f"Failed to delete {cid}: {r.status_code}")
        except Exception as e:
            log(f"Error deleting {cid}: {e}")

    test_result("Cleanup", deleted == total, f"Deleted {deleted}/{total} test contacts")


# ============================================================
# TEST 9: Multiple Card Formats
# ============================================================
def test_multiple_cards():
    print("\n" + "=" * 60)
    print("TEST 9: Multiple Card Scan Formats")
    print("=" * 60)

    if not user_id:
        test_result("Multi-Card Scan", False, "Skipped: no user_id")
        return

    cards = [
        {
            "name": "Ahmed Al-Rashid",
            "email": "ahmed@gulftech.ae",
            "phone": "+971 50 123 4567",
            "company": "Gulf Technology LLC",
            "title": "Chief Technology Officer",
        },
        {
            "name": "Maria Santos",
            "email": "maria.santos@eurohealth.eu",
            "phone": "+49 170 9876543",
            "company": "EuroHealth GmbH",
            "title": "Regional Director",
        },
    ]

    successes = 0
    cleanup_ids = []

    for i, card in enumerate(cards):
        if i > 0:
            log("Waiting 10s between scans to avoid rate limiting...")
            time.sleep(10)
        log(f"\nCard {i + 1}: {card['name']} @ {card['company']}")
        card_img = generate_business_card(**card)

        try:
            start = time.time()
            r = requests.post(
                f"{API_BASE}/add_contact_from_image/?user_id={user_id}",
                headers={"X-API-Key": API_KEY},
                files={"file": (f"card_{i + 1}.jpg", card_img, "image/jpeg")},
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start

            if r.status_code == 200:
                data = r.json()
                fields = data.get("extracted_fields", {})
                cid = data.get("contact_id")
                if cid:
                    cleanup_ids.append(cid)
                log(f"  OK ({elapsed:.1f}s): name={fields.get('name')}, email={fields.get('email')}, source={fields.get('_source', '?')}")
                successes += 1
            else:
                data = r.json() if "json" in r.headers.get("content-type", "") else {}
                log(f"  FAIL ({elapsed:.1f}s): {r.status_code} - {data.get('detail', r.text[:200])}")
        except Exception as e:
            log(f"  ERROR: {e}")

    test_result(
        "Multi-Card Scan",
        successes >= 1,
        f"Passed: {successes}/{len(cards)} cards extracted successfully"
    )

    # Cleanup
    for cid in cleanup_ids:
        try:
            requests.delete(
                f"{API_BASE}/contact/{cid}?user_id={user_id}",
                headers={"X-API-Key": API_KEY},
                timeout=10,
            )
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("EVENT SCOUT PRODUCTION BACKEND TEST SUITE")
    print("=" * 60)
    print(f"Target: {API_BASE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_health()
    test_login()
    test_card_scan()
    test_verify_contact()
    test_manual_add()
    test_search()
    test_lead_scoring()
    test_cleanup()
    test_multiple_cards()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    for r in results:
        icon = "[+]" if r["passed"] else "[X]"
        print(f"  {icon} {r['name']}")

    print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Success Rate: {(passed / total * 100):.0f}%")
    print("=" * 60)

    if failed == 0:
        print("\nAll backend tests passed! Frontend testing can proceed.")
    else:
        print(f"\n{failed} test(s) failed. Investigate before frontend testing.")
        for r in results:
            if not r["passed"]:
                print(f"  [X] {r['name']}: {r['message']}")

    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
