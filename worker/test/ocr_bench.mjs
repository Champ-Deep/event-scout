// OCR model bench through the deployed Worker's admin probe (nothing is saved).
// BASE=... TOKEN=<admin session token> CARDS=~/repos/event-scout/bench/cards MODELS=@cf/a,@cf/b node test/ocr_bench.mjs
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const BASE = process.env.BASE.replace(/\/$/, "");
const dir = process.env.CARDS;
const models = (process.env.MODELS || "").split(",").filter(Boolean);
const variants = (process.env.VARIANTS || "photo,hard").split(",");
const norm = (s) => String(s || "").toLowerCase().replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/[\s/]+$/, "").replace(/\s+/g, " ");
const cards = readdirSync(dir).filter((f) => /^c\d+\.json$/.test(f)).map((f) => f.replace(".json", "")).sort();

for (const model of models) {
  const res = [];
  const jobs = cards.flatMap((c) => variants.map((v) => ({ c, v })));
  async function one({ c, v }) {
    const img = join(dir, `${c}_${v}.jpg`);
    const truth = JSON.parse(readFileSync(join(dir, `${c}.json`), "utf8"));
    const fd = new FormData();
    fd.append("file", new Blob([readFileSync(img)], { type: "image/jpeg" }), "card.jpg");
    const t0 = Date.now();
    const r = await fetch(`${BASE}/api/admin/ocr_probe?model=${encodeURIComponent(model)}`, { method: "POST", headers: { "X-API-Key": process.env.TOKEN }, body: fd });
    const d = await r.json();
    const keys = Object.keys(truth);
    const right = d.fields ? keys.filter((k) => norm(truth[k]) === norm(d.fields[k] === "N/A" ? "" : d.fields[k])).length : 0;
    res.push({ c, v, right, n: keys.length, ms: Date.now() - t0, err: d.source ? "" : (d.errors || [d.detail]).join(";").slice(0, 120) });
  }
  const q = [...jobs];
  await Promise.all(Array.from({ length: 4 }, async () => { while (q.length) await one(q.shift()); }));
  const ms = res.map((r) => r.ms).sort((a, b) => a - b);
  const acc = res.reduce((s, r) => s + r.right, 0) / res.reduce((s, r) => s + r.n, 0);
  const perfect = res.filter((r) => r.right === r.n).length;
  console.log(`${model}: fields ${(acc * 100).toFixed(1)}%  all-fields ${perfect}/${res.length}  p50 ${ms[Math.floor(ms.length / 2)]}ms  p95 ${ms[Math.floor(ms.length * 0.95)]}ms  errors ${res.filter((r) => r.err).length}`);
  for (const r of res.filter((r) => r.err)) console.log("   ", r.c, r.v, r.err);
}
