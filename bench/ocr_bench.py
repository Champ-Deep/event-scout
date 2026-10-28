"""Benchmark business-card OCR candidates on bench/cards.

Run with the production env so the real keys and quotas apply:
    cd ~/repos/event-scout && railway run python3 bench/ocr_bench.py [--only gemini:gemini-2.5-flash,or:...] [--limit N]

Every candidate gets the same prompt and the same resized JPEG the app would send.
Scores: per-field normalized match, all-fields-correct rate, median/p95 latency, OpenRouter cost.
"""
import argparse
import asyncio
import base64
import glob
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

import httpx
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["name", "title", "company_name", "email", "phone", "linkedin", "website"]
LEGACY = ["name", "email", "phone", "linkedin", "company_name"]

PROMPT = """Read this business card photo and extract the contact fields.
Return ONLY a JSON object with exactly these keys:
{"name": "", "title": "", "company_name": "", "email": "", "phone": "", "linkedin": "", "website": ""}
Rules: copy text exactly as printed (keep +country codes, punctuation in emails and URLs). Use "" for a field that is not on the card. Do not guess or invent anything. No markdown, no commentary."""

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

CANDIDATES = [
    # production today: gemini-2.0-flash direct (NOT_FOUND since the Gemini API retired it), then openrouter google/gemini-2.5-flash
    "gemini:gemini-2.5-flash",
    "gemini:gemini-2.5-flash-lite",
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemini-3.8-flash",
    "gemini:gemini-flash-lite-latest",
    "or:google/gemini-2.5-flash",
    "or:google/gemini-3.5-flash-lite",
    "or:anthropic/claude-haiku-5.5",
    "or:openai/gpt-6-luna",
    "or:qwen/qwen3.8-flash",
    "or:z-ai/glm-5.3-flash",
    "or:mistralai/mistral-small-2603",
    "tesseract:local",
]


def prep_image(path):
    """Same preprocessing as app.extract_contact_from_image_with_gemini: RGB, max 1600px, JPEG."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


async def call_gemini(client, model, jpeg):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}]}],
        "generationConfig": {"temperature": 0, "response_mime_type": "application/json"},
    }
    r = await client.post(url, json=body)
    d = r.json()
    if "error" in d:
        raise RuntimeError(f"{d['error'].get('status')}: {d['error'].get('message', '')[:120]}")
    return d["candidates"][0]["content"]["parts"][0]["text"], 0.0


async def call_openrouter(client, model, jpeg):
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OR_KEY}", "X-Title": "Event Scout OCR bench"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
            ]}],
            "max_tokens": 400,
            "temperature": 0,
            "usage": {"include": True},
        },
    )
    d = r.json()
    if "error" in d:
        raise RuntimeError(str(d["error"])[:160])
    return d["choices"][0]["message"]["content"], float(d.get("usage", {}).get("cost") or 0)


def call_tesseract(jpeg):
    """Local baseline: tesseract text + regex. Name/title/company are the first non-contact lines."""
    p = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"], input=jpeg, capture_output=True, timeout=60)
    text = p.stdout.decode("utf-8", "ignore")
    out = {k: "" for k in FIELDS}
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    out["email"] = m.group(0) if m else ""
    m = re.search(r"\+?\d[\d\s().-]{8,}\d", text)
    out["phone"] = m.group(0).strip() if m else ""
    m = re.search(r"linkedin\.com/in/[\w-]+", text, re.I)
    out["linkedin"] = m.group(0) if m else ""
    m = re.search(r"(?<![\w@/])(?:www\.)?[\w-]+\.(?:com|ai|org|io|se|fr|jp|mx|ae)(?![\w@])", text, re.I)
    out["website"] = m.group(0) if m and "@" not in m.group(0) else ""
    rest = [l.strip() for l in text.splitlines() if l.strip() and not re.search(r"@|\+?\d{2,}|linkedin|www\.|\.com|\.ai", l) and len(l.strip()) > 2]
    for k, v in zip(["name", "title", "company_name"], rest):
        out[k] = v
    return json.dumps(out), 0.0


def norm(field, v):
    v = (v or "").strip()
    if v.upper() in ("N/A", "NA", "NONE", "NULL"):
        v = ""
    v = v.lower()
    if field == "phone":
        return re.sub(r"\D", "", v)
    if field in ("linkedin", "website"):
        v = re.sub(r"^https?://", "", v)
        v = re.sub(r"^www\.", "", v).rstrip("/")
    return re.sub(r"\s+", " ", v).strip()


async def run_candidate(cand, cases, limit_conc=4):
    kind, model = cand.split(":", 1)
    sem = asyncio.Semaphore(limit_conc)
    rows = []

    async def one(path, truth):
        jpeg = prep_image(path)
        err = None
        pred, cost = {}, 0.0
        try:
            async with sem:
                t0 = time.perf_counter()
                async with httpx.AsyncClient(timeout=45) as client:
                    if kind == "gemini":
                        text, cost = await call_gemini(client, model, jpeg)
                    elif kind == "or":
                        text, cost = await call_openrouter(client, model, jpeg)
                    else:
                        text, cost = await asyncio.to_thread(call_tesseract, jpeg)
            pred = parse_json(text)
        except Exception as e:  # keep going, an error is a data point
            err = str(e)[:160]
        lat = time.perf_counter() - t0
        correct = {f: norm(f, pred.get(f, "")) == norm(f, truth.get(f, "")) for f in FIELDS}
        rows.append(dict(image=os.path.basename(path), latency=lat, cost=cost, error=err, correct=correct, pred=pred))

    await asyncio.gather(*(one(p, t) for p, t in cases))
    return rows


def summarize(cand, rows):
    n = len(rows)
    acc = {f: sum(r["correct"][f] for r in rows) / n for f in FIELDS}
    legacy_all = sum(all(r["correct"][f] for f in LEGACY) for r in rows) / n
    full_all = sum(all(r["correct"].values()) for r in rows) / n
    lats = sorted(r["latency"] for r in rows)
    errors = sum(1 for r in rows if r["error"])
    return dict(candidate=cand, n=n, field_acc=acc, legacy_card_acc=legacy_all, full_card_acc=full_all,
                mean_field_acc=statistics.mean(acc.values()), lat_p50=statistics.median(lats),
                lat_p95=lats[int(0.95 * (n - 1))], cost_total=sum(r["cost"] for r in rows), errors=errors,
                first_error=next((r["error"] for r in rows if r["error"]), None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cands = [c for c in a.only.split(",") if c] or CANDIDATES
    cases = []
    for gt in sorted(glob.glob(os.path.join(HERE, "cards", "*.json"))):
        truth = json.load(open(gt))
        cid = os.path.basename(gt)[:-5]
        for img in sorted(glob.glob(os.path.join(HERE, "cards", cid + "*.??g"))):
            cases.append((img, truth))
    if a.limit:
        cases = cases[: a.limit]
    print(f"{len(cases)} images, {len(cands)} candidates", file=sys.stderr)
    results, detail = [], {}
    for cand in cands:
        if cand.startswith("gemini:") and not GEMINI_KEY or cand.startswith("or:") and not OR_KEY:
            print(f"skip {cand}: key missing", file=sys.stderr)
            continue
        rows = asyncio.run(run_candidate(cand, cases))
        s = summarize(cand, rows)
        results.append(s)
        detail[cand] = rows
        print(f"{cand:40s} field={s['mean_field_acc']:.2f} legacy5={s['legacy_card_acc']:.2f} full7={s['full_card_acc']:.2f} "
              f"p50={s['lat_p50']:.1f}s p95={s['lat_p95']:.1f}s cost=${s['cost_total']:.4f} err={s['errors']}", file=sys.stderr)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M")
    json.dump(dict(summary=results, detail=detail), open(os.path.join(HERE, "results", f"{ts}.json"), "w"), indent=1)
    results.sort(key=lambda s: (-s["mean_field_acc"], s["lat_p50"]))
    md = ["| candidate | mean field acc | 5-field card acc | 7-field card acc | " + " | ".join(FIELDS) + " | p50 s | p95 s | cost (n imgs) | errors |",
          "|" + "---|" * (10 + len(FIELDS))]
    for s in results:
        md.append(f"| {s['candidate']} | {s['mean_field_acc']:.2f} | {s['legacy_card_acc']:.2f} | {s['full_card_acc']:.2f} | "
                  + " | ".join(f"{s['field_acc'][f]:.2f}" for f in FIELDS)
                  + f" | {s['lat_p50']:.1f} | {s['lat_p95']:.1f} | ${s['cost_total']:.4f} ({s['n']}) | {s['errors']} |")
    open(os.path.join(HERE, "results", f"{ts}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
