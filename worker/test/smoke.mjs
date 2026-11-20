// End-to-end smoke test against a deployed (or `wrangler dev`) Event Scout Worker.
// BASE=https://event-scout.deep-ae0.workers.dev CARD=path/to/card.jpg CARD_JSON=path/to/truth.json node test/smoke.mjs
// Creates one throwaway user, exercises the capture loop, deletes every contact it made.
// Prints SMOKE_USER_ID at the end so the caller can remove the user row.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const BASE = (process.env.BASE || "http://localhost:8791").replace(/\/$/, "");
const API = `${BASE}/api`;
const stamp = Date.now();
const email = `smoke+${stamp}@eventscout.dev`;
let token = "";
let uid = "";
const made = [];

async function call(method, path, { body, form, auth = true, raw = false } = {}) {
  const headers = {};
  if (auth && token) headers["X-API-Key"] = token;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const r = await fetch(API + path, { method, headers, body: form ?? (body !== undefined ? JSON.stringify(body) : undefined) });
  if (raw) return r;
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = text; }
  return { status: r.status, data };
}
const ok = (name, cond, extra = "") => { assert.ok(cond, `${name} ${extra}`); console.log(`PASS ${name} ${extra}`); };

let r = await call("GET", "/health/", { auth: false });
ok("health", r.status === 200 && r.data.database_connected, JSON.stringify({ users: r.data.total_users, ocr: r.data.ocr }));

r = await call("GET", "/list_contacts/?user_id=x", { auth: false });
ok("no session -> 401", r.status === 401);

r = await call("POST", "/register/", { body: { name: "Smoke Test", email, password: "smoke-pass-123" }, auth: false });
ok("register", r.status === 200 && r.data.session_token, r.status);
r = await call("POST", "/login/", { body: { email, password: "smoke-pass-123" }, auth: false });
ok("login", r.status === 200 && r.data.session_token);
token = r.data.session_token;
uid = r.data.user_id;
r = await call("POST", "/login/", { body: { email, password: "wrong" }, auth: false });
ok("bad password -> 401", r.status === 401);

r = await call("GET", `/user/validate?user_id=${uid}`);
ok("validate", r.data.valid === true);
r = await call("GET", `/list_contacts/?user_id=00000000-0000-0000-0000-000000000000`);
ok("other user_id -> 403", r.status === 403);

r = await call("PUT", `/user/profile/?user_id=${uid}`, { body: { current_event_name: "Smoke Expo" } });
ok("profile put", r.status === 200 && r.data.profile.current_event_name === "Smoke Expo");

r = await call("POST", "/add_contact/", {
  body: { user_id: uid, contact: { name: "Manual Person", email: "manual@example.com", phone: "+1 555 0100", linkedin: "N/A", company_name: "Acme" } },
});
ok("add_contact", r.status === 200 && r.data.contact_id);
made.push(r.data.contact_id);

if (process.env.CARD) {
  const fd = new FormData();
  fd.append("file", new Blob([readFileSync(process.env.CARD)], { type: "image/jpeg" }), "business-card.jpg");
  const t0 = Date.now();
  r = await call("POST", `/add_contact_from_image/?user_id=${uid}`, { form: fd });
  const ms = Date.now() - t0;
  ok("scan", r.status === 200 && r.data.contact_id, `${ms}ms via ${r.data.ocr_source} ${r.status} ${r.status !== 200 ? JSON.stringify(r.data) : ""}`);
  made.push(r.data.contact_id);
  if (process.env.CARD_JSON) {
    const truth = JSON.parse(readFileSync(process.env.CARD_JSON, "utf8"));
    const got = { ...r.data.extracted_fields, website: r.data.extracted_fields.links?.[0]?.url || "" };
    const norm = (s) => String(s || "").toLowerCase().replace(/^https?:\/\//, "").replace(/[\s/]+$/, "");
    const right = Object.keys(truth).filter((k) => norm(truth[k]) === norm(got[k]));
    console.log(`     fields ${right.length}/${Object.keys(truth).length}`, Object.keys(truth).filter((k) => !right.includes(k)).map((k) => `${k}: ${got[k]}`).join(" | "));
  }
  ok("scan under 30s client timeout", ms < 30000, `${ms}ms`);
}

r = await call("GET", `/list_contacts/?user_id=${uid}`);
ok("list_contacts", r.status === 200 && r.data.total_contacts === made.length, `${r.data.total_contacts}`);
ok("event tagged from profile", r.data.contacts.every((c) => c.event_name === "Smoke Expo"));

r = await call("PUT", `/contact/${made[0]}?user_id=${uid}`, { body: { notes: "met at booth" } });
ok("update contact", r.status === 200 && r.data.contact.notes === "met at booth");
r = await call("GET", `/contact/${made[0]}?user_id=${uid}`);
ok("get contact", r.status === 200 && r.data.contact.notes === "met at booth");

const csv = await call("GET", `/export_contacts/?user_id=${uid}&format=csv`, { raw: true });
const csvBytes = new Uint8Array(await csv.arrayBuffer());
const csvText = new TextDecoder("utf-8", { ignoreBOM: true }).decode(csvBytes);
ok("export csv", csv.status === 200 && csvBytes[0] === 0xef && csvText.startsWith("\ufeffName,Email") && csvText.includes("Manual Person"));

r = await call("POST", `/user/card/generate_token/?user_id=${uid}`);
ok("card token", r.status === 200 && r.data.shareable_url?.startsWith(BASE));
const pub = await call("GET", `/card/${r.data.shareable_token}`, { auth: false });
ok("public card", pub.status === 200 && pub.data.status === "success");
const page = await fetch(r.data.shareable_url);
ok("card page", page.status === 200 && (await page.text()).includes("<html"));

const home = await fetch(BASE + "/");
const html = await home.text();
ok("app page", home.status === 200 && html.includes("API_BASE_URL: '/api'") && !html.includes("railway.app"));

for (const id of made) {
  r = await call("DELETE", `/contact/${id}?user_id=${uid}`);
  ok("delete", r.status === 200, id);
}
r = await call("GET", `/list_contacts/?user_id=${uid}`);
ok("cleaned up", r.data.total_contacts === 0);
console.log(`SMOKE_USER_ID=${uid}`);
console.log(`SMOKE_TOKEN_SET=${token ? "yes" : "no"}`);
