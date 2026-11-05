"""Runs tests/scan_flow_stubbed_run.js in real Chrome and screenshots every state.
   uv run --with playwright python tests/scan_flow_run.py   (serves the repo on :8766 itself)
   Exit code 1 if any check fails."""
import functools, http.server, pathlib, sys, threading
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "scan-to-score-screens"
STATES = ["permission_denied", "camera", "capturing", "extracting", "extract_failed", "data", "analyzing", "analysis_failed", "result"]

OUT.mkdir(parents=True, exist_ok=True)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT)))
threading.Thread(target=srv.serve_forever, daemon=True).start()

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    page = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2).new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8766/mobile-frontend.html", wait_until="load")
    page.wait_for_function("window.ScanFlow && window.ScoreCard")
    page.evaluate((ROOT / "tests" / "scan_flow_stubbed_run.js").read_text())
    for i, s in enumerate(STATES, 1):
        got = page.evaluate(f"scanFlowScenario('{s}')")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / f"{i:02d}-{s}.png"))
        print("shot", s, got["got"])
    results = page.evaluate("scanFlowRunAll()")
    b.close()

bad = [r for r in results if not r["ok"]]
for r in results:
    print("PASS" if r["ok"] else "FAIL", r["name"], "" if r["ok"] else r["got"])
print("page errors:", errors or "none")
sys.exit(1 if bad or errors else 0)
