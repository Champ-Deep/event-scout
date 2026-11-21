// Local proof of the Harbinger sync: wrangler dev --local + a fake Harbinger that records what it receives.
// cd worker && node test/harbinger_sync.local.mjs
import { spawn, execFileSync } from "node:child_process";
import http from "node:http";
import { writeFileSync, rmSync } from "node:fs";
import assert from "node:assert/strict";

const PORT = 8793, FAKE = 8794, BASE = `http://127.0.0.1:${PORT}/api`;
const got = [];
const fake = http.createServer((req, res) => {
  let b = "";
  req.on("data", (d) => (b += d));
  req.on("end", () => {
    got.push({ path: req.url, auth: req.headers.authorization, body: JSON.parse(b || "{}") });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ status: "created", person_id: "p-" + got.length }));
  });
}).listen(FAKE);

const persist = ".wrangler/harbinger-test";
rmSync(persist, { recursive: true, force: true });
writeFileSync(".dev.vars.harbinger-test", `HARBINGER_API_KEY=chh_test_key_local_only\nHARBINGER_URL=http://127.0.0.1:${FAKE}\nSIGNUP_ALLOW_EMAILS=sync@example.com\n`);
const w = (args) => execFileSync("npx", ["wrangler", ...args, "--local", "--persist-to", persist], { stdio: "pipe" }).toString();
w(["d1", "migrations", "apply", "event-scout-db"]);
const dev = spawn("npx", ["wrangler", "dev", "--local", "--port", String(PORT), "--persist-to", persist, "--env-file", ".dev.vars.harbinger-test"], { stdio: "pipe" });
let up = false;
for (let i = 0; i < 60 && !up; i++) {
  await new Promise((r) => setTimeout(r, 1000));
  up = await fetch(`${BASE}/health/`).then((r) => r.ok, () => false);
}
const ok = (n, c, x = "") => { assert.ok(c, `${n} ${x}`); console.log(`PASS ${n} ${x}`); };
try {
  ok("dev server up", up);
  const call = async (m, p, body, t) => {
    const r = await fetch(BASE + p, { method: m, headers: { ...(t ? { "X-API-Key": t } : {}), ...(body ? { "content-type": "application/json" } : {}) }, body: body ? JSON.stringify(body) : undefined });
    return { s: r.status, d: await r.json() };
  };
  let r = await call("POST", "/register/", { name: "Sync Test", email: "sync@example.com", password: "sync-pass-123" });
  const uid = r.d.user_id;
  w(["d1", "execute", "event-scout-db", "--command", `update users set is_admin=1 where id='${uid}'`]);
  const t = (await call("POST", "/login/", { email: "sync@example.com", password: "sync-pass-123" })).d.session_token;
  await call("PUT", "/user/profile/", { current_event_name: "Bengaluru Tech Summit" }, t);
  r = await call("POST", "/add_contact/", { user_id: uid, contact: { name: "Asha Rao", email: "asha@acme.example", phone: "+91 90000 00000", linkedin: "N/A", company_name: "Acme Health", title: "VP Ops", notes: "Asked about hospital data for Q1" } }, t);
  const cid = r.d.contact_id;
  await new Promise((res) => setTimeout(res, 2000)); // immediate push runs in waitUntil
  ok("pushed on create", got.length === 1, `${got.length}`);
  const p = got[0];
  ok("route + bearer key", p.path === "/api/v1/event-scout/contacts" && p.auth === "Bearer chh_test_key_local_only");
  ok("person fields", p.body.person.name === "Asha Rao" && p.body.person.email === "asha@acme.example" && p.body.person.linkedin === "");
  ok("met at event, by whom", p.body.meeting.event_name === "Bengaluru Tech Summit" && p.body.meeting.met_by.email === "sync@example.com");
  ok("notes reach the agent", p.body.context.notes === "Asked about hospital data for Q1" && p.body.external_id === cid);
  r = await call("POST", "/admin/harbinger/sync", null, t);
  ok("unchanged contact is not re-sent", r.d.pushed === 0 && got.length === 1, JSON.stringify(r.d));
  await call("PUT", `/contact/${cid}`, { notes: "Asked about hospital data for Q1. Follow up after their board meeting." }, t);
  await new Promise((res) => setTimeout(res, 2000));
  ok("edited notes re-sent", got.length === 2 && got[1].body.context.notes.includes("board meeting"));
  w(["d1", "execute", "event-scout-db", "--command", `update contacts set admin_notes='Strategic account' where id='${cid}'`]);
  r = await call("POST", "/admin/harbinger/sync", null, t);
  ok("any write path is caught by the sweep", r.d.pushed === 1 && got[2]?.body.context.admin_notes === "Strategic account", JSON.stringify(r.d));
  r = await call("GET", "/admin/harbinger/status", null, t);
  ok("status", r.d.configured && r.d.by_status.sent === 1, JSON.stringify(r.d.by_status));
} finally {
  dev.kill("SIGTERM");
  fake.close();
  rmSync(".dev.vars.harbinger-test", { force: true });
}
