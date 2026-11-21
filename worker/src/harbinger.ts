// Event Scout -> ChampHarbinger: every contact, with where we met them and every note, goes to Harbinger's agent.
// Contract: docs/harbinger-contract.md. Off until HARBINGER_API_KEY (a chh_ key from Harbinger Settings) is set.
// Change detection is a payload hash, so any write path (scan, edit, voice note, admin note, score) is picked up
// by the sweep without a hook in each route.
import { Env, all, first, now, pj, run, sha256Hex } from "./lib";

const MAX_ATTEMPTS = 8;
const SWEEP_BATCH = 40;

export const harbingerOn = (env: Env) => !!(env.HARBINGER_API_KEY && env.HARBINGER_URL);

export async function buildPayload(env: Env, c: any, owners?: Map<string, any>) {
  const owner = owners ? owners.get(c.user_id) : await first<any>(env, "SELECT name, email FROM users WHERE id = ?", c.user_id);
  const na = (v: unknown) => (v && v !== "N/A" ? String(v) : "");
  const links = pj<any[]>(c.links, []);
  const website = links.find((l) => l?.label === "Website")?.url || "";
  return {
    source: "event_scout",
    external_id: c.id,
    updated_at: c.updated_at || c.created_at,
    person: {
      name: c.name, title: na(c.title), company_name: na(c.company_name), email: na(c.email), phone: na(c.phone),
      linkedin: na(c.linkedin), website, other_links: links.filter((l) => l?.label !== "Website").map((l) => l.url).filter(Boolean),
    },
    meeting: {
      event_name: c.event_name || "",
      met_at: c.created_at,
      captured_by: c.source || "manual", // scan | manual | qr
      met_by: { name: owner?.name || "", email: owner?.email || "" },
    },
    context: {
      notes: c.notes || "",
      admin_notes: c.admin_notes || "",
      voice_notes: pj<any[]>(c.audio_notes, []).map((n) => ({ transcript: n?.transcript || "", at: n?.timestamp || "" })).filter((n) => n.transcript),
      lead: c.lead_score == null ? null : {
        score: c.lead_score, temperature: c.lead_temperature, reasoning: c.lead_score_reasoning || "",
        recommended_actions: pj(c.lead_recommended_actions, []),
      },
    },
  };
}

async function push(env: Env, c: any, payload: any, hash: string) {
  const ts = now();
  try {
    const r = await fetch(`${env.HARBINGER_URL!.replace(/\/$/, "")}/api/v1/event-scout/contacts`, {
      method: "POST",
      headers: { Authorization: `Bearer ${env.HARBINGER_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(20000),
    });
    const text = await r.text();
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${text.slice(0, 200)}`);
    const d = (() => { try { return JSON.parse(text); } catch { return {}; } })();
    await run(env,
      `INSERT INTO harbinger_sync (contact_id, status, payload_hash, attempts, last_error, harbinger_person_id, synced_at, updated_at)
       VALUES (?, 'sent', ?, 0, NULL, ?, ?, ?)
       ON CONFLICT(contact_id) DO UPDATE SET status = 'sent', payload_hash = excluded.payload_hash, attempts = 0,
         last_error = NULL, harbinger_person_id = excluded.harbinger_person_id, synced_at = excluded.synced_at, updated_at = excluded.updated_at`,
      c.id, hash, d.person_id ?? null, ts, ts);
    return true;
  } catch (e) {
    await run(env,
      `INSERT INTO harbinger_sync (contact_id, status, attempts, last_error, updated_at) VALUES (?, 'failed', 1, ?, ?)
       ON CONFLICT(contact_id) DO UPDATE SET status = 'failed', attempts = attempts + 1, last_error = excluded.last_error, updated_at = excluded.updated_at`,
      c.id, String((e as Error).message || e).slice(0, 300), ts);
    return false;
  }
}

/** Push every contact whose payload changed since Harbinger last accepted it. Returns counts. */
export async function sweep(env: Env, limit = SWEEP_BATCH) {
  if (!harbingerOn(env)) return { configured: false, pushed: 0, failed: 0, pending: 0 };
  const rows = await all<any>(env,
    `SELECT c.*, s.payload_hash AS sync_hash, s.attempts AS sync_attempts FROM contacts c
       LEFT JOIN harbinger_sync s ON s.contact_id = c.id ORDER BY c.updated_at DESC`);
  const owners = new Map((await all<any>(env, "SELECT id, name, email FROM users")).map((u) => [u.id, u]));
  let pushed = 0, failed = 0, pending = 0;
  for (const c of rows) {
    if ((c.sync_attempts ?? 0) >= MAX_ATTEMPTS) continue; // parked: needs a look, see /admin/harbinger/status
    const payload = await buildPayload(env, c, owners);
    const hash = await sha256Hex(JSON.stringify(payload));
    if (hash === c.sync_hash) continue;
    if (pushed + failed >= limit) { pending++; continue; }
    (await push(env, c, payload, hash)) ? pushed++ : failed++;
  }
  return { configured: true, pushed, failed, pending };
}

/** Immediate push for one contact (after a scan or edit); the sweep is the safety net. */
export function syncSoon(env: Env, exec: ExecutionContext, contactId: string) {
  if (!harbingerOn(env)) return;
  exec.waitUntil((async () => {
    const c = await first<any>(env, "SELECT * FROM contacts WHERE id = ?", contactId);
    if (!c) return;
    const payload = await buildPayload(env, c);
    await push(env, c, payload, await sha256Hex(JSON.stringify(payload)));
  })().catch((e) => console.log("harbinger sync failed", String(e))));
}

export async function status(env: Env) {
  const counts = await all<any>(env, "SELECT status, count(*) n FROM harbinger_sync GROUP BY status");
  const total = (await first<any>(env, "SELECT count(*) n FROM contacts"))?.n ?? 0;
  const parked = await all<any>(env, `SELECT contact_id, attempts, last_error, updated_at FROM harbinger_sync WHERE attempts >= ${MAX_ATTEMPTS} LIMIT 20`);
  const lastErr = await first<any>(env, "SELECT last_error, updated_at FROM harbinger_sync WHERE status = 'failed' ORDER BY updated_at DESC LIMIT 1");
  return {
    status: "success", configured: harbingerOn(env), harbinger_url: env.HARBINGER_URL || null, total_contacts: total,
    by_status: Object.fromEntries(counts.map((r) => [r.status, r.n])), parked, last_error: lastErr,
  };
}
