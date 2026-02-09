#!/usr/bin/env python3
"""
Event Scout — Tiered Test Suite
================================
Parameterized test suite that runs against any target URL.

Usage:
  python tests/test_suite.py                          # Tier 1 smoke against production
  python tests/test_suite.py --tier 2                 # Tier 2 functional tests
  python tests/test_suite.py --tier 3                 # Tier 3 integration tests
  python tests/test_suite.py --target http://localhost:8000 --tier 2
  python tests/test_suite.py --output json            # Machine-readable output

Environment:
  TEST_PASSWORD  — Password for test account (required for tier 2+)
  TEST_EMAIL     — Test account email (default: deep@lakeb2b.com)
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime

import requests

# Defaults
DEFAULT_TARGET = "https://event-scout-production.up.railway.app"
DEFAULT_FRONTEND = "https://event-scout-delta.vercel.app"
DEFAULT_API_KEY = "OGibuBdW6KP52UMTpv8g46Zs37g47d9SGv4w21W-o6s"
TIMEOUT = 60


class TestRunner:
    def __init__(self, target, api_key, frontend_url, tier, output_format):
        self.target = target.rstrip("/")
        self.api_key = api_key
        self.frontend_url = frontend_url
        self.tier = tier
        self.output_format = output_format
        self.results = []
        self.user_id = None
        self.is_admin = False
        self.test_contact_ids = []  # Track for cleanup

    def log(self, msg):
        if self.output_format != "json":
            print(f"  {msg}")

    def record(self, name, passed, message="", tier=1):
        self.results.append({
            "name": name,
            "passed": passed,
            "message": message,
            "tier": tier,
        })
        if self.output_format != "json":
            icon = "[+]" if passed else "[X]"
            status = "PASS" if passed else "FAIL"
            print(f"\n{icon} {status} — {name}")
            if message:
                self.log(message)
        return passed

    def headers(self):
        return {"X-API-Key": self.api_key}

    def json_headers(self):
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    # ==================================================================
    # TIER 1 — Smoke Tests (30 seconds)
    # ==================================================================

    def test_health(self):
        """GET /health/ — basic liveness check."""
        try:
            r = requests.get(f"{self.target}/health/", timeout=15)
            data = r.json()
            ok = (
                data.get("status") == "healthy"
                and data.get("database") == "postgresql"
                and data.get("gemini_configured") is True
            )
            self.record(
                "Health Check", ok,
                f"v{data.get('version')} | DB: {data.get('database')} | "
                f"Users: {data.get('total_users')} | Gemini: {data.get('gemini_configured')} | "
                f"OpenRouter: {data.get('openrouter_configured')}",
                tier=1,
            )
        except Exception as e:
            self.record("Health Check", False, str(e), tier=1)

    def test_login(self):
        """POST /login/ — authenticate and store user_id."""
        email = os.environ.get("TEST_EMAIL", "deep@lakeb2b.com")
        password = os.environ.get("TEST_PASSWORD", "")
        if not password:
            self.record("Login", False, "TEST_PASSWORD env var not set", tier=1)
            return
        try:
            r = requests.post(
                f"{self.target}/login/",
                json={"email": email, "password": password},
                timeout=15,
            )
            data = r.json()
            if r.status_code == 200 and data.get("status") == "success":
                self.user_id = data["user_id"]
                self.is_admin = data.get("is_admin", False)
                self.record(
                    "Login", True,
                    f"user_id: {self.user_id} | admin: {self.is_admin}",
                    tier=1,
                )
            else:
                self.record("Login", False, f"HTTP {r.status_code}: {data}", tier=1)
        except Exception as e:
            self.record("Login", False, str(e), tier=1)

    def test_list_contacts(self):
        """GET /list_contacts/ — basic CRUD read."""
        if not self.user_id:
            self.record("List Contacts", False, "Skipped: no user_id", tier=1)
            return
        try:
            r = requests.get(
                f"{self.target}/list_contacts/?user_id={self.user_id}",
                headers=self.headers(),
                timeout=15,
            )
            data = r.json()
            total = data.get("total_contacts", 0)
            self.record(
                "List Contacts", r.status_code == 200,
                f"{total} contacts", tier=1,
            )
        except Exception as e:
            self.record("List Contacts", False, str(e), tier=1)

    def test_new_endpoints_exist(self):
        """Verify new endpoints return non-404 (pipeline, converse, etc.)."""
        endpoints = [
            ("GET", f"/health/"),
            ("POST", f"/converse/"),
        ]
        # Pipeline endpoint needs a valid contact_id, just check it doesn't 404 on bad id
        try:
            r = requests.get(
                f"{self.target}/contact/00000000-0000-0000-0000-000000000000/pipeline?user_id={self.user_id or 'test'}",
                headers=self.headers(),
                timeout=10,
            )
            # 404 = endpoint not registered, 400/422/500 = endpoint exists but bad input
            pipeline_exists = r.status_code != 404
        except Exception:
            pipeline_exists = False

        try:
            r = requests.post(
                f"{self.target}/converse/",
                headers=self.json_headers(),
                json={"user_id": "test", "query": "test"},
                timeout=10,
            )
            converse_exists = r.status_code != 404
        except Exception:
            converse_exists = False

        ok = pipeline_exists and converse_exists
        self.record(
            "New Endpoints Exist", ok,
            f"pipeline: {'yes' if pipeline_exists else 'NO'} | "
            f"converse: {'yes' if converse_exists else 'NO'}",
            tier=1,
        )

    def test_frontend_accessible(self):
        """GET frontend URL — verify it loads."""
        if not self.frontend_url:
            self.record("Frontend Accessible", True, "Skipped: no frontend URL", tier=1)
            return
        try:
            r = requests.get(self.frontend_url, timeout=15)
            has_title = "Event Scout" in r.text
            has_resilient = "resilientFetch" in r.text
            ok = r.status_code == 200 and has_title
            self.record(
                "Frontend Accessible", ok,
                f"HTTP {r.status_code} | title: {has_title} | resilientFetch: {has_resilient}",
                tier=1,
            )
        except Exception as e:
            self.record("Frontend Accessible", False, str(e), tier=1)

    # ==================================================================
    # TIER 2 — Functional Tests (3 minutes)
    # ==================================================================

    def test_manual_add_and_delete(self):
        """POST /add_contact/ + DELETE — full CRUD cycle."""
        if not self.user_id:
            self.record("Manual Add + Delete", False, "Skipped: no user_id", tier=2)
            return
        try:
            r = requests.post(
                f"{self.target}/add_contact/",
                headers=self.json_headers(),
                json={
                    "contact": {
                        "name": "Test Suite Contact",
                        "email": "test.suite@example.com",
                        "phone": "+1-555-000-0000",
                        "company_name": "Test Suite Corp",
                    },
                    "user_id": self.user_id,
                },
                timeout=TIMEOUT,
            )
            data = r.json()
            if r.status_code != 200:
                self.record("Manual Add + Delete", False, f"Add failed: {data}", tier=2)
                return

            contact_id = data.get("contact_id")
            # Delete it
            rd = requests.delete(
                f"{self.target}/contact/{contact_id}?user_id={self.user_id}",
                headers=self.headers(),
                timeout=TIMEOUT,
            )
            deleted = rd.status_code == 200
            self.record(
                "Manual Add + Delete", deleted,
                f"Added {contact_id}, deleted: {deleted}",
                tier=2,
            )
        except Exception as e:
            self.record("Manual Add + Delete", False, str(e), tier=2)

    def test_card_scan(self):
        """POST /add_contact_from_image/ — OCR scan with synthetic card."""
        if not self.user_id:
            self.record("Card Scan", False, "Skipped: no user_id", tier=2)
            return
        try:
            card_img = self._generate_card(
                "Test Scanner", "test.scan@example.com",
                "+1-555-111-2222", "ScanTest Inc", "QA Engineer"
            )
            start = time.time()
            r = requests.post(
                f"{self.target}/add_contact_from_image/?user_id={self.user_id}",
                headers=self.headers(),
                files={"file": ("test_scan.jpg", card_img, "image/jpeg")},
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start
            if r.status_code == 200:
                data = r.json()
                cid = data.get("contact_id")
                fields = data.get("extracted_fields", {})
                if cid:
                    self.test_contact_ids.append(cid)
                has_name = fields.get("name", "N/A") != "N/A"
                self.record(
                    "Card Scan", has_name,
                    f"{elapsed:.1f}s | name={fields.get('name')} | email={fields.get('email')} | "
                    f"source={fields.get('_source', '?')}",
                    tier=2,
                )
            else:
                data = r.json() if "json" in r.headers.get("content-type", "") else {}
                self.record("Card Scan", False, f"HTTP {r.status_code}: {data.get('detail', '')}", tier=2)
        except requests.Timeout:
            self.record("Card Scan", False, f"TIMEOUT after {TIMEOUT}s", tier=2)
        except Exception as e:
            self.record("Card Scan", False, str(e), tier=2)

    def test_converse_with_ui(self):
        """POST /converse/ — chat returns ui_components."""
        if not self.user_id:
            self.record("Chat + Generative UI", False, "Skipped: no user_id", tier=2)
            return
        try:
            r = requests.post(
                f"{self.target}/converse/",
                headers=self.json_headers(),
                json={
                    "user_id": self.user_id,
                    "query": "Who are my contacts?",
                    "conversation_history": [],
                },
                timeout=TIMEOUT,
            )
            data = r.json()
            ui = data.get("ui_components", [])
            types = [c.get("type") for c in ui if isinstance(c, dict)]
            has_cards = "contact_cards" in types
            self.record(
                "Chat + Generative UI",
                r.status_code == 200 and len(ui) > 0,
                f"Components: {len(ui)} | Types: {types}",
                tier=2,
            )
        except Exception as e:
            self.record("Chat + Generative UI", False, str(e), tier=2)

    def test_search(self):
        """POST /search/ — semantic FAISS search."""
        if not self.user_id:
            self.record("Semantic Search", False, "Skipped: no user_id", tier=2)
            return
        try:
            r = requests.post(
                f"{self.target}/search/",
                headers=self.json_headers(),
                json={"query": "marketing director", "user_id": self.user_id},
                timeout=TIMEOUT,
            )
            data = r.json()
            num = len(data.get("results", []))
            self.record(
                "Semantic Search",
                r.status_code == 200 and data.get("status") == "success",
                f"{num} results",
                tier=2,
            )
        except Exception as e:
            self.record("Semantic Search", False, str(e), tier=2)

    def test_pipeline_status(self):
        """GET /contact/{id}/pipeline — pipeline status endpoint."""
        cid = self.test_contact_ids[0] if self.test_contact_ids else None
        if not self.user_id or not cid:
            self.record("Pipeline Status", False, "Skipped: no contact", tier=2)
            return
        try:
            r = requests.get(
                f"{self.target}/contact/{cid}/pipeline?user_id={self.user_id}",
                headers=self.headers(),
                timeout=15,
            )
            # May return 200 with null status (no pipeline run) or 404 — both are OK
            ok = r.status_code in (200, 404)
            self.record(
                "Pipeline Status", ok,
                f"HTTP {r.status_code}",
                tier=2,
            )
        except Exception as e:
            self.record("Pipeline Status", False, str(e), tier=2)

    # ==================================================================
    # TIER 3 — Integration Tests (10 minutes)
    # ==================================================================

    def test_lead_scoring(self):
        """POST /contact/{id}/score — AI scoring."""
        cid = self.test_contact_ids[0] if self.test_contact_ids else None
        if not self.user_id or not cid:
            self.record("Lead Scoring", False, "Skipped: no contact", tier=3)
            return
        try:
            self.log("Waiting 5s to avoid Gemini rate limiting...")
            time.sleep(5)
            start = time.time()
            r = requests.post(
                f"{self.target}/contact/{cid}/score?user_id={self.user_id}",
                headers=self.headers(),
                timeout=TIMEOUT,
            )
            elapsed = time.time() - start
            if r.status_code == 200:
                data = r.json()
                score = data.get("score")
                temp = data.get("temperature")
                reasoning = data.get("reasoning", "")
                real = reasoning != "Scoring temporarily unavailable."
                self.record(
                    "Lead Scoring", score is not None and real,
                    f"{elapsed:.1f}s | Score: {score} | Temp: {temp}",
                    tier=3,
                )
            else:
                self.record("Lead Scoring", False, f"HTTP {r.status_code}", tier=3)
        except Exception as e:
            self.record("Lead Scoring", False, str(e), tier=3)

    def test_export(self):
        """GET /export_contacts/ — CSV export."""
        if not self.user_id:
            self.record("CSV Export", False, "Skipped: no user_id", tier=3)
            return
        try:
            r = requests.get(
                f"{self.target}/export_contacts/?user_id={self.user_id}&format=csv",
                headers=self.headers(),
                timeout=TIMEOUT,
            )
            ok = r.status_code == 200 and len(r.content) > 0
            self.record("CSV Export", ok, f"HTTP {r.status_code} | {len(r.content)} bytes", tier=3)
        except Exception as e:
            self.record("CSV Export", False, str(e), tier=3)

    def test_frontend_features(self):
        """Check frontend has key features deployed."""
        if not self.frontend_url:
            self.record("Frontend Features", True, "Skipped: no frontend URL", tier=3)
            return
        try:
            r = requests.get(self.frontend_url, timeout=15)
            html = r.text
            checks = {
                "resilientFetch": "resilientFetch" in html,
                "renderContactCards": "renderContactCards" in html,
                "renderUIComponents": "renderUIComponents" in html,
                "loadPipelineStatus": "loadPipelineStatus" in html,
                "offlineBanner": "offlineBanner" in html,
                "TIMEOUT_MS: 60000": "TIMEOUT_MS: 60000" in html,
            }
            all_ok = all(checks.values())
            missing = [k for k, v in checks.items() if not v]
            self.record(
                "Frontend Features", all_ok,
                f"{'All present' if all_ok else 'MISSING: ' + ', '.join(missing)}",
                tier=3,
            )
        except Exception as e:
            self.record("Frontend Features", False, str(e), tier=3)

    def test_admin_pipelines(self):
        """GET /admin/pipelines — admin pipeline monitor."""
        if not self.user_id or not self.is_admin:
            self.record("Admin Pipeline Monitor", True, "Skipped: not admin", tier=3)
            return
        try:
            r = requests.get(
                f"{self.target}/admin/pipelines?user_id={self.user_id}",
                headers=self.headers(),
                timeout=15,
            )
            data = r.json()
            self.record(
                "Admin Pipeline Monitor",
                r.status_code == 200 and "pipelines" in data,
                f"HTTP {r.status_code} | auto_pipeline: {data.get('auto_pipeline_enabled')}",
                tier=3,
            )
        except Exception as e:
            self.record("Admin Pipeline Monitor", False, str(e), tier=3)

    def test_backup_status(self):
        """GET /admin/backup/status — backup system health."""
        if not self.user_id or not self.is_admin:
            self.record("Backup Status", True, "Skipped: not admin", tier=3)
            return
        try:
            r = requests.get(
                f"{self.target}/admin/backup/status?user_id={self.user_id}",
                headers=self.headers(),
                timeout=15,
            )
            self.record(
                "Backup Status",
                r.status_code == 200,
                f"HTTP {r.status_code}",
                tier=3,
            )
        except Exception as e:
            self.record("Backup Status", False, str(e), tier=3)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _generate_card(self, name, email, phone, company, title="Manager"):
        """Generate a synthetic business card image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # Return a minimal JPEG if Pillow not available
            self.log("Pillow not installed — using minimal test image")
            import struct
            # 1x1 white JPEG
            return io.BytesIO(
                b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
                b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
                b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
                b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
                b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
                b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00'
                b'\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
                b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
                b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
                b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
                b'\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
                b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99'
                b'\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
                b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5'
                b'\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1'
                b'\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa'
                b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a+\xff\xd9'
            )

        img = Image.new("RGB", (800, 450), "white")
        draw = ImageDraw.Draw(img)
        try:
            font_lg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            font_md = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            font_lg = font_md = font_sm = ImageFont.load_default()

        draw.rectangle([(0, 0), (800, 8)], fill="#2563EB")
        draw.text((50, 30), company, fill="#2563EB", font=font_md)
        draw.text((50, 90), name, fill="#111827", font=font_lg)
        draw.text((50, 140), title, fill="#6B7280", font=font_md)
        draw.line([(50, 190), (750, 190)], fill="#E5E7EB", width=2)
        draw.text((50, 220), f"Email: {email}", fill="#374151", font=font_sm)
        draw.text((50, 260), f"Phone: {phone}", fill="#374151", font=font_sm)
        draw.rectangle([(0, 442), (800, 450)], fill="#2563EB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf

    def cleanup(self):
        """Delete any test contacts created during the run."""
        if not self.user_id or not self.test_contact_ids:
            return
        for cid in self.test_contact_ids:
            try:
                requests.delete(
                    f"{self.target}/contact/{cid}?user_id={self.user_id}",
                    headers=self.headers(),
                    timeout=10,
                )
                self.log(f"Cleaned up: {cid}")
            except Exception:
                pass

    # ==================================================================
    # Runner
    # ==================================================================

    def run(self):
        start_time = time.time()

        if self.output_format != "json":
            print("=" * 60)
            print(f"EVENT SCOUT TEST SUITE — Tier {self.tier}")
            print("=" * 60)
            print(f"Target:   {self.target}")
            print(f"Frontend: {self.frontend_url or 'N/A'}")
            print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

        # Tier 1 — Smoke
        self.test_health()
        self.test_login()
        self.test_list_contacts()
        self.test_new_endpoints_exist()
        self.test_frontend_accessible()

        if self.tier >= 2:
            if self.output_format != "json":
                print(f"\n{'—' * 60}")
                print("TIER 2 — Functional Tests")
                print(f"{'—' * 60}")
            self.test_manual_add_and_delete()
            self.test_card_scan()
            self.test_converse_with_ui()
            self.test_search()
            self.test_pipeline_status()

        if self.tier >= 3:
            if self.output_format != "json":
                print(f"\n{'—' * 60}")
                print("TIER 3 — Integration Tests")
                print(f"{'—' * 60}")
            self.test_lead_scoring()
            self.test_export()
            self.test_frontend_features()
            self.test_admin_pipelines()
            self.test_backup_status()

        # Cleanup test data
        self.cleanup()

        elapsed = time.time() - start_time
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        if self.output_format == "json":
            print(json.dumps({
                "target": self.target,
                "tier": self.tier,
                "total": total,
                "passed": passed,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 1),
                "results": self.results,
            }, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print("SUMMARY")
            print(f"{'=' * 60}")
            for r in self.results:
                icon = "[+]" if r["passed"] else "[X]"
                print(f"  {icon} T{r['tier']} {r['name']}")

            print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
            print(f"Time: {elapsed:.1f}s")
            print(f"Success Rate: {(passed / total * 100):.0f}%" if total else "N/A")
            print("=" * 60)

            if failed == 0:
                print(f"\nAll Tier {self.tier} tests passed!")
            else:
                print(f"\n{failed} test(s) failed:")
                for r in self.results:
                    if not r["passed"]:
                        print(f"  [X] {r['name']}: {r['message']}")

        return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Event Scout Test Suite")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Backend API URL")
    parser.add_argument("--frontend", default=DEFAULT_FRONTEND, help="Frontend URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3], help="Test tier (1=smoke, 2=functional, 3=integration)")
    parser.add_argument("--output", default="text", choices=["text", "json"], help="Output format")
    args = parser.parse_args()

    runner = TestRunner(args.target, args.api_key, args.frontend, args.tier, args.output)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
