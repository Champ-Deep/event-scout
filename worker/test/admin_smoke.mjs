// Admin + AI + files smoke against the deployed Worker, signed in as an admin test account.
// BASE=... TOKEN=<admin session token> node test/admin_smoke.mjs   (read-only on real data; its own test contact is deleted)
import assert from "node:assert/strict";

const BASE = process.env.BASE.replace(/\/$/, "");
const H = { "X-API-Key": process.env.TOKEN };
const me = await (await fetch(`${BASE}/api/user/validate`, { headers: H })).json();
const uid = me.user_id;
const ok = (name, cond, extra = "") => { assert.ok(cond, `${name} ${extra}`); console.log(`PASS ${name} ${extra}`); };
async function j(method, path, body) {
  const r = await fetch(BASE + "/api" + path, { method, headers: { ...H, ...(body ? { "Content-Type": "application/json" } : {}) }, body: body ? JSON.stringify(body) : undefined });
  const t = await r.text();
  try { return { s: r.status, d: JSON.parse(t) }; } catch { return { s: r.status, d: t }; }
}
const u = (p) => `${p}${p.includes("?") ? "&" : "?"}user_id=${uid}`;

ok("admin session", me.is_admin === true);
let r = await j("POST", u("/admin/reindex"));
ok("reindex embeddings", r.s === 200 && r.d.indexed >= 270, `${r.d.indexed}`);
r = await j("GET", u("/admin/users"));
ok("admin users", r.s === 200 && JSON.stringify(r.d).length > 100, `${r.s} ${(r.d.users || r.d.data || []).length} users`);
r = await j("GET", u("/admin/contacts?limit=500"));
const total = r.d.total ?? r.d.total_contacts ?? (r.d.contacts || []).length;
ok("admin contacts", r.s === 200 && total >= 270, `${total}`);
r = await j("GET", u("/admin/dashboard"));
ok("admin dashboard", r.s === 200, JSON.stringify(r.d).slice(0, 160));
r = await j("GET", u("/admin/activity"));
ok("admin activity", r.s === 200);
r = await j("GET", u("/admin/pipelines"));
ok("admin pipelines", r.s === 200, JSON.stringify(r.d).slice(0, 100));
r = await j("GET", u("/admin/contact_lists"));
ok("contact lists", r.s === 200 && JSON.stringify(r.d).includes("name"), JSON.stringify(r.d).slice(0, 100));
r = await j("GET", u("/admin/backup/status"));
ok("backup status", r.s === 200);
r = await j("GET", u("/admin/search?q=a"));
ok("admin search", r.s === 200, `${(r.d.results || r.d.contacts || []).length} hits`);
const csv = await fetch(`${BASE}/api` + u("/admin/export"), { headers: H });
const csvText = new TextDecoder("utf-8", { ignoreBOM: true }).decode(await csv.arrayBuffer());
ok("admin export csv", csv.status === 200 && csvText.startsWith("\ufeffName,") && csvText.split("\r\n").length >= 270, `${csvText.split("\r\n").length - 2} rows`);
r = await j("GET", u("/exhibitors/?limit=200"));
ok("exhibitors", r.s === 200, `${(r.d.exhibitors || []).length} rows, total ${r.d.total}`);
r = await j("GET", u("/exhibitors/categories/"));
ok("exhibitor categories", r.s === 200);
r = await j("GET", u("/dashboard"));
ok("user dashboard", r.s === 200);
r = await j("GET", u("/broadcasts/active"));
ok("broadcasts", r.s === 200);

r = await j("GET", u("/files/list"));
ok("event files list", r.s === 200 && r.d.files?.length >= 1, `${r.d.files?.length}`);
if (r.d.files?.length) {
  const f = r.d.files[0];
  const dl = await fetch(`${BASE}/api` + u(`/files/download/${f.id}`), { headers: H });
  const n = (await dl.arrayBuffer()).byteLength;
  ok("event file bytes from KV", dl.status === 200 && n === f.file_size, `${n} bytes`);
}
// One real contact file, found through the admin contacts list.
const withFiles = (await j("GET", u("/admin/contacts?limit=500"))).d.contacts || [];
let checked = false;
for (const c of withFiles) {
  const lf = await j("GET", u(`/contact/${c.id}/files`));
  if (lf.d.files?.length) {
    const f = lf.d.files[0];
    const dl = await fetch(`${BASE}/api` + u(`/contact/files/download/${f.id}`), { headers: H });
    const n = (await dl.arrayBuffer()).byteLength;
    ok("contact file bytes from KV", dl.status === 200 && n === f.file_size, `${n} bytes`);
    checked = true;
    break;
  }
  if (withFiles.indexOf(c) > 60) break;
}
if (!checked) console.log("SKIP contact file download (none found in first 60 contacts)");

// AI on our own throwaway contact.
r = await j("POST", "/add_contact/", { user_id: uid, contact: { name: "Bench Prospect", email: "cto@hospital.example", phone: "N/A", linkedin: "N/A", company_name: "City Hospital", title: "CTO", notes: "Wants data enrichment for 40 clinics, budget this quarter" } });
const cid = r.d.contact_id;
ok("add test contact", !!cid);
await new Promise((res) => setTimeout(res, 4000)); // embedding is written in waitUntil
r = await j("POST", "/search/", { user_id: uid, query: "hospital CTO" });
ok("semantic search", r.s === 200 && r.d.results?.some((x) => x.meta.id === cid), `${r.d.results?.length} results`);
r = await j("POST", `/contact/${cid}/score?user_id=${uid}`);
ok("lead score", r.s === 200 && typeof r.d.score === "number", `${r.s} ${r.d.score} ${r.d.temperature} ${r.d.detail || ""}`);
r = await j("POST", "/converse/", { user_id: uid, query: "Who should I follow up with first and why?", conversation_history: [] });
ok("converse", r.s === 200 && r.d.response?.length > 20, `${r.d.model} :: ${String(r.d.response).slice(0, 90).replace(/\s+/g, " ")}`);
r = await j("POST", `/contact/${cid}/pipeline/run?user_id=${uid}`);
ok("pipeline run is a clear 501", r.s === 501, r.d.detail);
r = await j("DELETE", `/contact/${cid}?user_id=${uid}`);
ok("delete test contact", r.s === 200);
