// Admin, dashboard, exhibitor, backup and contact-list routes. Port of app.py's handlers
// (line numbers noted per group); response shapes match app.py so mobile-frontend.html parses them.
import type { Route } from "../index";
import {
  Ctx, HttpError, all, allIn, buildCsv, csvResponse, first, now, pipelineMap, pj, q, qInt, run, sj, uuid,
} from "../lib";
import { deleteContactRows } from "./core";
import { reindexLater } from "../embed";

const DEFAULT_EVENT = "WHX Dubai 2026";

function reqStr(b: any, field: string): string {
  const v = b?.[field];
  if (typeof v !== "string" || !v) throw new HttpError(422, `${field} is required`);
  return v;
}

function reqStrList(b: any, field: string): string[] {
  const v = b?.[field];
  if (!Array.isArray(v) || v.some((x) => typeof x !== "string")) throw new HttpError(422, `${field} must be a list of strings`);
  return v;
}

/** The same date windows as app.py's admin/user contact filters (UTC day boundaries). */
function dateConds(ctx: Ctx, col = "created_at"): { conds: string[]; args: unknown[] } {
  const conds: string[] = [], args: unknown[] = [];
  const df = q(ctx, "date_filter");
  const nowD = new Date();
  const todayStart = new Date(Date.UTC(nowD.getUTCFullYear(), nowD.getUTCMonth(), nowD.getUTCDate())).toISOString();
  if (df === "today") { conds.push(`${col} >= ?`); args.push(todayStart); }
  else if (df === "yesterday") { conds.push(`${col} >= ? AND ${col} < ?`); args.push(new Date(Date.parse(todayStart) - 864e5).toISOString(), todayStart); }
  else if (df === "last_week") { conds.push(`${col} >= ?`); args.push(new Date(Date.now() - 7 * 864e5).toISOString()); }
  else if (df === "last_month") { conds.push(`${col} >= ?`); args.push(new Date(Date.now() - 30 * 864e5).toISOString()); }
  return { conds, args };
}

// ---- dashboard / models (app.py ~3568, ~3666) ----
async function dashboard(ctx: Ctx) {
  const contacts = await all<any>(ctx.env, "SELECT * FROM contacts WHERE user_id = ?", ctx.user.id);
  const scored = contacts.filter((c) => c.lead_score !== null && c.lead_score !== undefined);
  const temps = { hot: 0, warm: 0, cold: 0 };
  for (const c of contacts) if (c.lead_temperature === "hot" || c.lead_temperature === "warm" || c.lead_temperature === "cold") temps[c.lead_temperature as "hot" | "warm" | "cold"]++;
  const top = [...scored].sort((a, b) => (b.lead_score || 0) - (a.lead_score || 0)).slice(0, 5).map((c) => ({
    id: c.id, name: c.name, company_name: c.company_name || "N/A",
    lead_score: c.lead_score, lead_temperature: c.lead_temperature,
  }));
  const p = await first<any>(ctx.env, "SELECT profile_data FROM user_profiles WHERE user_id = ?", ctx.user.id);
  const profile = pj<any>(p?.profile_data, {});
  return {
    status: "success",
    total_contacts: contacts.length,
    scored_contacts: scored.length,
    unscored_contacts: contacts.length - scored.length,
    by_temperature: temps,
    top_contacts: top,
    has_profile: !!p && Object.keys(profile).length > 0,
    current_event: (profile.current_event_name as string) || "",
  };
}

// app.py ships this list statically; only the OpenRouter flag is dynamic.
const AVAILABLE_MODELS: Record<string, { id: string; name: string; provider: string }> = {
  "claude-opus": { id: "anthropic/claude-opus-4", name: "Claude Opus 4.6", provider: "Anthropic" },
  "gpt-5": { id: "openai/gpt-4.1", name: "GPT-5.2", provider: "OpenAI" },
  "gemini-pro": { id: "google/gemini-2.5-pro-preview-06-05", name: "Gemini 2.5 Pro", provider: "Google" },
};

async function listModels(ctx: Ctx) {
  return {
    status: "success",
    models: Object.entries(AVAILABLE_MODELS).map(([key, info]) => ({ key, ...info })),
    openrouter_configured: !!ctx.env.OPENROUTER_API_KEY,
  };
}

// ---- exhibitors (app.py ~3685, ~3738, ~3786) ----
async function importExhibitors(ctx: Ctx) {
  const b = await ctx.body();
  const list = b?.exhibitors;
  if (!Array.isArray(list)) throw new HttpError(422, "exhibitors is required");
  let imported = 0, skipped = 0;
  const stmts: D1PreparedStatement[] = [];
  for (const ex of list) {
    if (!ex || typeof ex !== "object") throw new HttpError(422, "exhibitors must be a list of objects");
    const name = String(ex.name ?? "").trim();
    const event = typeof ex.event_name === "string" && ex.event_name ? ex.event_name : DEFAULT_EVENT;
    if (!name) { skipped++; continue; }
    if (await first(ctx.env, "SELECT id FROM exhibitors WHERE name = ? AND event_name = ?", name, event)) { skipped++; continue; }
    stmts.push(ctx.env.DB.prepare(
      `INSERT INTO exhibitors (id, event_name, name, booth, hall, category, subcategory, country, website, description, products, tags, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(
      uuid(), event, name, String(ex.booth ?? ""), String(ex.hall ?? ""), String(ex.category ?? ""),
      String(ex.subcategory ?? ""), String(ex.country ?? ""), String(ex.website ?? ""), String(ex.description ?? ""),
      sj(Array.isArray(ex.products) ? ex.products : []), sj(Array.isArray(ex.tags) ? ex.tags : []), now(),
    ));
    imported++;
  }
  if (stmts.length) await ctx.env.DB.batch(stmts);
  return { status: "success", imported, skipped };
}

async function listExhibitors(ctx: Ctx) {
  const event = q(ctx, "event") || DEFAULT_EVENT;
  const category = q(ctx, "category") || "";
  const search = q(ctx, "search") || "";
  const limit = qInt(ctx, "limit", 100, 0, 10000);
  const conds = ["event_name = ?"], args: unknown[] = [event];
  if (category) { conds.push("category LIKE ?"); args.push(`%${category}%`); }
  if (search) { conds.push("name LIKE ?"); args.push(`%${search}%`); }
  const rows = await all<any>(ctx.env, `SELECT * FROM exhibitors WHERE ${conds.join(" AND ")} ORDER BY name LIMIT ?`, ...args, limit);
  return {
    status: "success",
    total: rows.length,
    exhibitors: rows.map((e) => ({
      id: e.id, name: e.name, booth: e.booth || "", hall: e.hall || "", category: e.category || "",
      subcategory: e.subcategory || "", country: e.country || "", website: e.website || "",
      description: e.description || "", products: pj(e.products, []), tags: pj(e.tags, []),
    })),
  };
}

async function exhibitorCategories(ctx: Ctx) {
  const event = q(ctx, "event") || DEFAULT_EVENT;
  const rows = await all<{ category: string }>(ctx.env, "SELECT DISTINCT category FROM exhibitors WHERE event_name = ? AND category != '' ORDER BY category", event);
  return { status: "success", categories: rows.map((r) => r.category) };
}

// ---- admin users (app.py ~3811) ----
async function adminListUsers(ctx: Ctx) {
  const action = q(ctx, "action");
  const target = q(ctx, "target_user_id");
  if (action === "delete" && target) {
    if (target === ctx.user.id) throw new HttpError(400, "Cannot delete your own admin account");
    const user = await first<any>(ctx.env, "SELECT * FROM users WHERE id = ?", target);
    if (!user) throw new HttpError(404, "User not found");
    const ids = (await all<{ id: string }>(ctx.env, "SELECT id FROM contacts WHERE user_id = ?", target)).map((r) => r.id);
    await deleteContactRows(ctx.env, ids);
    await ctx.env.DB.batch([
      ctx.env.DB.prepare("DELETE FROM conversations WHERE user_id = ?").bind(target),
      ctx.env.DB.prepare("DELETE FROM user_profiles WHERE user_id = ?").bind(target),
      ctx.env.DB.prepare("DELETE FROM user_cards WHERE user_id = ?").bind(target),
      ctx.env.DB.prepare("DELETE FROM sessions WHERE user_id = ?").bind(target),
      ctx.env.DB.prepare("DELETE FROM contact_embeddings WHERE user_id = ?").bind(target),
      ctx.env.DB.prepare("DELETE FROM users WHERE id = ?").bind(target),
    ]);
    return { status: "success", message: `User '${user.name}' and all their data deleted`, user_id: target };
  }
  const rows = await all<any>(ctx.env,
    "SELECT u.*, (SELECT count(*) FROM contacts c WHERE c.user_id = u.id) AS contact_count FROM users u");
  const users = rows.map((u) => ({
    user_id: u.id, name: u.name, email: u.email, is_admin: !!u.is_admin,
    contact_count: u.contact_count || 0, created_at: u.created_at || "",
  }));
  return { status: "success", users, total: users.length };
}

// ---- admin contacts list / activity (app.py ~3883, ~3987) ----
async function adminListContacts(ctx: Ctx) {
  const conds: string[] = [], args: unknown[] = [];
  const filterUser = q(ctx, "filter_user");
  if (filterUser) { conds.push("user_id = ?"); args.push(filterUser); }
  const contactId = q(ctx, "contact_id");
  if (contactId) { conds.push("id = ?"); args.push(contactId); }
  const dc = dateConds(ctx);
  conds.push(...dc.conds); args.push(...dc.args);
  const eventFilter = q(ctx, "event_filter");
  if (eventFilter) { conds.push("event_name = ?"); args.push(eventFilter); }
  const limit = qInt(ctx, "limit", 50, 0, 1000);
  const page = qInt(ctx, "page", 1, 1, 100000);
  const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "";
  const total = ((await first<{ n: number }>(ctx.env, `SELECT count(*) AS n FROM contacts ${where}`, ...args))?.n) ?? 0;
  const rows = await all<any>(ctx.env, `SELECT * FROM contacts ${where} ORDER BY created_at DESC LIMIT ? OFFSET ?`, ...args, limit, (page - 1) * limit);
  const events = await all<{ event_name: string }>(ctx.env, "SELECT DISTINCT event_name FROM contacts WHERE event_name != '' ORDER BY event_name");
  const userIds = [...new Set(rows.map((c) => c.user_id))];
  const userMap = new Map<string, string>();
  if (userIds.length) for (const u of await allIn<any>(ctx.env, "SELECT id, name FROM users WHERE id IN (?IN)", userIds)) userMap.set(u.id, u.name);
  return {
    status: "success",
    total, page, limit,
    has_more: (page - 1) * limit + rows.length < total,
    distinct_events: events.map((e) => e.event_name),
    contacts: rows.map((c) => ({
      id: c.id, user_id: c.user_id, user_name: userMap.get(c.user_id) || "Unknown",
      name: c.name, email: c.email || "N/A", phone: c.phone || "N/A", company_name: c.company_name || "N/A",
      linkedin: c.linkedin || "N/A", event_name: c.event_name || "",
      lead_score: c.lead_score, lead_temperature: c.lead_temperature,
      lead_score_reasoning: c.lead_score_reasoning || "",
      lead_recommended_actions: pj(c.lead_recommended_actions, []),
      source: c.source || "manual", notes: c.notes || "", admin_notes: c.admin_notes || "",
      links: pj(c.links, []),
      audio_notes_count: pj(c.audio_notes, []).length,
      created_at: c.created_at || "", updated_at: c.updated_at || "",
    })),
  };
}

async function adminActivity(ctx: Ctx) {
  const limit = qInt(ctx, "limit", 50, 0, 500);
  const rows = await all<any>(ctx.env,
    "SELECT c.*, u.name AS user_name FROM contacts c JOIN users u ON u.id = c.user_id ORDER BY c.created_at DESC LIMIT ?", limit);
  return {
    status: "success",
    activities: rows.map((c) => ({
      type: "contact_added", contact_id: c.id, contact_name: c.name, company: c.company_name || "N/A",
      user_id: c.user_id, user_name: c.user_name, source: c.source || "manual",
      timestamp: c.created_at || "", lead_temperature: c.lead_temperature || "unscored",
      lead_score: c.lead_score,
    })),
  };
}

async function adminDeleteContact(ctx: Ctx) {
  const id = ctx.params.contact_id;
  const c = await first<any>(ctx.env, "SELECT id FROM contacts WHERE id = ?", id);
  if (!c) throw new HttpError(404, "Contact not found");
  await deleteContactRows(ctx.env, [id]);
  return { status: "success", message: "Contact deleted", contact_id: id };
}

// ---- merges (app.py ~4077, ~4175) ----
async function mergeUsers(ctx: Ctx) {
  const b = await ctx.body();
  const primaryId = reqStr(b, "primary_user_id");
  const secondaryId = reqStr(b, "secondary_user_id");
  if (primaryId === secondaryId) throw new HttpError(400, "Cannot merge a user with themselves");
  const primary = await first<any>(ctx.env, "SELECT * FROM users WHERE id = ?", primaryId);
  if (!primary) throw new HttpError(404, "Primary user not found");
  const secondary = await first<any>(ctx.env, "SELECT * FROM users WHERE id = ?", secondaryId);
  if (!secondary) throw new HttpError(404, "Secondary user not found");
  const transferred = ((await first<{ n: number }>(ctx.env, "SELECT count(*) AS n FROM contacts WHERE user_id = ?", secondaryId))?.n) ?? 0;
  await ctx.env.DB.batch([
    ctx.env.DB.prepare("UPDATE contacts SET user_id = ? WHERE user_id = ?").bind(primaryId, secondaryId),
    ctx.env.DB.prepare("UPDATE conversations SET user_id = ? WHERE user_id = ?").bind(primaryId, secondaryId),
    ctx.env.DB.prepare("UPDATE contact_pipelines SET user_id = ? WHERE user_id = ?").bind(primaryId, secondaryId),
    // FAISS rebuild in app.py; here the rows are re-keyed, search filters on user_id.
    ctx.env.DB.prepare("UPDATE contact_embeddings SET user_id = ? WHERE user_id = ?").bind(primaryId, secondaryId),
    ctx.env.DB.prepare("DELETE FROM user_profiles WHERE user_id = ?").bind(secondaryId),
    ctx.env.DB.prepare("DELETE FROM user_cards WHERE user_id = ?").bind(secondaryId),
    ctx.env.DB.prepare("DELETE FROM sessions WHERE user_id = ?").bind(secondaryId),
    ctx.env.DB.prepare("DELETE FROM users WHERE id = ?").bind(secondaryId),
  ]);
  return {
    status: "success",
    message: `User '${secondary.name}' merged into '${primary.name}'. ${transferred} contacts transferred.`,
    primary_user_id: primaryId, deleted_user_id: secondaryId, contacts_transferred: transferred,
  };
}

const pick = (a: any, b: any): any => {
  if (a && a !== "N/A" && String(a).trim()) return a;
  return b && b !== "N/A" && String(b).trim() ? b : a;
};

const concatNote = (a: any, b: any): string => {
  const existing = a || "";
  if (!b || !String(b).trim()) return existing;
  return existing.trim() ? `${existing}\n---\n${b}` : String(b);
};

async function mergeContacts(ctx: Ctx) {
  const b = await ctx.body();
  const primaryId = reqStr(b, "primary_id");
  const secondaryId = reqStr(b, "secondary_id");
  if (primaryId === secondaryId) throw new HttpError(400, "Cannot merge a contact with itself");
  const p = await first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ?", primaryId);
  if (!p) throw new HttpError(404, "Primary contact not found");
  const s = await first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ?", secondaryId);
  if (!s) throw new HttpError(404, "Secondary contact not found");

  const m: any = { ...p };
  for (const f of ["name", "email", "phone", "linkedin", "company_name"]) m[f] = pick(p[f], s[f]);
  m.notes = concatNote(p.notes, s.notes);
  m.admin_notes = concatNote(p.admin_notes, s.admin_notes);

  const links = pj<any[]>(p.links, []);
  const urls = new Set(links.map((l) => l && l.url));
  for (const l of pj<any[]>(s.links, [])) if (l && typeof l === "object" && !urls.has(l.url)) links.push(l);
  m.links = links;

  m.audio_notes = [...pj<any[]>(p.audio_notes, []), ...pj<any[]>(s.audio_notes, [])];
  const actions = pj<any[]>(p.lead_recommended_actions, []);
  for (const a of pj<any[]>(s.lead_recommended_actions, [])) if (!actions.includes(a)) actions.push(a);
  m.lead_recommended_actions = actions;

  if ((s.lead_score || 0) > (p.lead_score || 0)) {
    m.lead_score = s.lead_score;
    m.lead_temperature = s.lead_temperature;
    m.lead_score_reasoning = s.lead_score_reasoning;
    m.lead_score_breakdown = sj(pj(s.lead_score_breakdown, {}));
  }
  if (!p.photo_base64 && s.photo_base64) m.photo_base64 = s.photo_base64;
  if (s.created_at && (!p.created_at || s.created_at < p.created_at)) m.created_at = s.created_at;

  await ctx.env.DB.batch([
    ctx.env.DB.prepare("UPDATE contacts SET name = ?, email = ?, phone = ?, linkedin = ?, company_name = ?, notes = ?, admin_notes = ?, links = ?, audio_notes = ?, lead_recommended_actions = ?, lead_score = ?, lead_temperature = ?, lead_score_reasoning = ?, lead_score_breakdown = ?, photo_base64 = ?, created_at = ? WHERE id = ?")
      .bind(m.name, m.email, m.phone, m.linkedin, m.company_name, m.notes, m.admin_notes, sj(m.links),
        sj(m.audio_notes), sj(m.lead_recommended_actions), m.lead_score, m.lead_temperature,
        m.lead_score_reasoning, m.lead_score_breakdown, m.photo_base64, m.created_at, primaryId),
    ctx.env.DB.prepare("UPDATE contact_files SET contact_id = ? WHERE contact_id = ?").bind(primaryId, secondaryId),
    ctx.env.DB.prepare("UPDATE contact_pipelines SET contact_id = ? WHERE contact_id = ?").bind(primaryId, secondaryId),
    ctx.env.DB.prepare("DELETE FROM contact_list_members WHERE contact_id = ?").bind(secondaryId),
    ctx.env.DB.prepare("DELETE FROM contact_embeddings WHERE contact_id = ?").bind(secondaryId),
    ctx.env.DB.prepare("DELETE FROM contacts WHERE id = ?").bind(secondaryId),
  ]);
  reindexLater(ctx.env, ctx.exec, [{ ...m, id: primaryId }]);
  return {
    status: "success",
    message: `Contacts merged: '${m.name}' kept, duplicate deleted`,
    merged_contact_id: primaryId, deleted_contact_id: secondaryId,
  };
}

// ---- duplicates (app.py ~4297) ----
const normalizePhone = (phone: string) => (!phone || phone === "N/A" ? "" : phone.replace(/[^\d+]/g, ""));

async function adminDuplicates(ctx: Ctx) {
  const target = q(ctx, "target_user_id");
  if (!target) throw new HttpError(422, "target_user_id is required");
  const limit = qInt(ctx, "limit", 500, 0, 5000);
  const rows = await all<any>(ctx.env, "SELECT * FROM contacts WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", target, limit);
  if (!rows.length) return { status: "success", duplicate_groups: [], total_groups: 0, total_duplicate_contacts: 0 };

  const cl = rows.map((c) => ({
    id: c.id,
    name: c.name || "N/A",
    email: (c.email || "N/A").trim().toLowerCase(),
    phone: normalizePhone(c.phone || "N/A"),
    company_name: (c.company_name || "N/A").trim().toLowerCase(),
    lead_score: c.lead_score,
    lead_temperature: c.lead_temperature,
    source: c.source || "unknown",
    created_at: c.created_at || null,
    photo_base64: !!c.photo_base64,
  }));

  const parent: Record<string, string> = {};
  for (const c of cl) parent[c.id] = c.id;
  const find = (x: string): string => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const reasons: Record<string, Set<string>> = {};
  const union = (a: string, b: string, reason: string) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
    (reasons[a] ||= new Set()).add(reason);
    (reasons[b] ||= new Set()).add(reason);
  };
  const groupBy = (keyOf: (c: any) => string, mkReason: (key: string) => string, keep: (key: string) => boolean) => {
    const map = new Map<string, string[]>();
    for (const c of cl) {
      const k = keyOf(c);
      if (!keep(k)) continue;
      const arr = map.get(k) || [];
      arr.push(c.id);
      map.set(k, arr);
    }
    for (const [k, ids] of map) for (let i = 1; i < ids.length; i++) union(ids[0], ids[i], mkReason(k));
  };
  groupBy((c) => c.email, (k) => `email: ${k}`, (k) => !!k && k !== "n/a");
  groupBy((c) => c.phone, (k) => `phone: ${k}`, (k) => k.length >= 7);
  groupBy(
    (c) => `${c.name.trim().toLowerCase()}\n${c.company_name.trim().toLowerCase()}`,
    (k) => {
      const [n, co] = k.split("\n");
      return `name+company: ${n} @ ${co}`;
    },
    (k) => !k.startsWith("n/a\n") && !k.endsWith("\nn/a"),
  );

  const groups = new Map<string, string[]>();
  const byId = new Map(cl.map((c) => [c.id, c]));
  for (const c of cl) {
    const root = find(c.id);
    const arr = groups.get(root) || [];
    arr.push(c.id);
    groups.set(root, arr);
  }
  const out: any[] = [];
  let totalDupes = 0;
  let n = 0;
  for (const ids of groups.values()) {
    n++;
    if (ids.length < 2) continue;
    const rs = new Set<string>();
    for (const id of ids) for (const r of reasons[id] || []) rs.add(r);
    out.push({ group_id: n, match_reasons: [...rs].sort(), contacts: ids.map((id) => byId.get(id)) });
    totalDupes += ids.length;
  }
  return { status: "success", duplicate_groups: out, total_groups: out.length, total_duplicate_contacts: totalDupes };
}

// ---- batch delete (app.py ~4416) ----
async function adminBatchDelete(ctx: Ctx) {
  const b = await ctx.body();
  const ids = reqStrList(b, "contact_ids");
  if (!ids.length) throw new HttpError(400, "No contact IDs provided");
  const found = await allIn<{ id: string }>(ctx.env, "SELECT id FROM contacts WHERE id IN (?IN)", ids);
  if (!found.length) throw new HttpError(404, "No contacts found");
  const delIds = found.map((r) => r.id);
  await deleteContactRows(ctx.env, delIds);
  return { status: "success", deleted_count: delIds.length, deleted_ids: delIds, message: `${delIds.length} contacts deleted` };
}

// ---- webhooks (app.py ~4471, ~4624): WEBHOOK_URL is not configured on Cloudflare ----
async function webhookTest(_ctx: Ctx) {
  return { status: "error", message: "No webhook URL configured. Enter a URL above and test." };
}

async function testWebhook(_ctx: Ctx) {
  return {
    status: "error", webhook_sent: false,
    webhook_message: "WEBHOOK_URL not configured. Set it in Railway environment variables.",
  };
}

// ---- admin dashboard (app.py ~4530) ----
async function adminDashboard(ctx: Ctx) {
  const users = await all<any>(ctx.env, "SELECT * FROM users");
  const total = (await first<{ n: number }>(ctx.env, "SELECT count(*) AS n FROM contacts"))?.n ?? 0;
  const tempCounts = await all<{ t: string; n: number }>(ctx.env,
    "SELECT lead_temperature AS t, count(*) AS n FROM contacts WHERE lead_temperature IN ('hot','warm','cold') GROUP BY lead_temperature");
  const temps: Record<string, number> = { hot: 0, warm: 0, cold: 0 };
  for (const r of tempCounts) temps[r.t] = r.n;
  const todayStart = new Date().toISOString().slice(0, 10) + "T00:00:00.000Z";
  const stats = new Map<string, any>(
    (await all<any>(ctx.env,
      `SELECT user_id,
         count(*) AS cnt,
         sum(CASE WHEN lead_temperature = 'hot' THEN 1 ELSE 0 END) AS hot,
         sum(CASE WHEN lead_temperature = 'warm' THEN 1 ELSE 0 END) AS warm,
         sum(CASE WHEN lead_temperature = 'cold' THEN 1 ELSE 0 END) AS cold,
         sum(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS today_cnt,
         max(created_at) AS last_scan
       FROM contacts GROUP BY user_id`, todayStart)).map((r) => [r.user_id, r]),
  );
  const perUser = users.map((u) => {
    const s = stats.get(u.id);
    return {
      user_id: u.id, name: u.name, email: u.email, contact_count: s?.cnt || 0, is_admin: !!u.is_admin,
      hot_count: s?.hot || 0, warm_count: s?.warm || 0, cold_count: s?.cold || 0,
      today_count: s?.today_cnt || 0, last_scan_at: s?.last_scan ?? null,
    };
  });
  const userMap = new Map(users.map((u) => [u.id, u.name]));
  const recent = await all<any>(ctx.env, "SELECT * FROM contacts ORDER BY created_at DESC LIMIT 10");
  return {
    status: "success",
    total_users: users.length,
    total_contacts: total,
    by_temperature: temps,
    per_user: perUser,
    recent_contacts: recent.map((c) => ({
      id: c.id, name: c.name, company_name: c.company_name || "N/A",
      user_name: userMap.get(c.user_id) || "Unknown",
      lead_score: c.lead_score, lead_temperature: c.lead_temperature,
      source: c.source || "manual", created_at: c.created_at || "",
    })),
    webhook_configured: false,
    webhook_url_preview: null,
  };
}

// ---- admin search / note / export (app.py ~5245, ~5414, ~5449) ----
async function adminSearch(ctx: Ctx) {
  const term = q(ctx, "q");
  if (!term) throw new HttpError(422, "q is required");
  const limit = qInt(ctx, "limit", 50, 0, 500);
  const like = `%${term}%`;
  const rows = await all<any>(ctx.env,
    "SELECT * FROM contacts WHERE name LIKE ? OR email LIKE ? OR company_name LIKE ? OR notes LIKE ? ORDER BY created_at DESC LIMIT ?",
    like, like, like, like, limit);
  const userIds = [...new Set(rows.map((c) => c.user_id))];
  const userMap = new Map<string, string>();
  if (userIds.length) for (const u of await allIn<any>(ctx.env, "SELECT id, name FROM users WHERE id IN (?IN)", userIds)) userMap.set(u.id, u.name);
  return {
    status: "success",
    query: term,
    total: rows.length,
    contacts: rows.map((c) => ({
      id: c.id, user_id: c.user_id, user_name: userMap.get(c.user_id) || "Unknown",
      name: c.name, email: c.email || "N/A", phone: c.phone || "N/A", company_name: c.company_name || "N/A",
      linkedin: c.linkedin || "N/A", lead_score: c.lead_score, lead_temperature: c.lead_temperature,
      source: c.source || "manual", notes: c.notes || "", admin_notes: c.admin_notes || "",
      created_at: c.created_at || "",
    })),
  };
}

async function adminContactNote(ctx: Ctx) {
  const id = ctx.params.contact_id;
  const note = q(ctx, "note") ?? "";
  if (!await first(ctx.env, "SELECT id FROM contacts WHERE id = ?", id)) throw new HttpError(404, "Contact not found");
  await run(ctx.env, "UPDATE contacts SET admin_notes = ? WHERE id = ?", note, id);
  return { status: "success", contact_id: id, admin_notes: note, message: "Admin note updated" };
}

export const ADMIN_CSV_HEADERS = [
  "Name", "Email", "Phone", "LinkedIn", "Company", "Event",
  "Notes", "Admin Notes", "Lead Score", "Lead Temperature",
  "Source", "Scanned By", "Created At",
  "Research Summary", "Pitch Angle", "Pitch Email Subject", "Gamma Deck URL",
];

async function adminExport(ctx: Ctx) {
  const contacts = await all<any>(ctx.env, "SELECT * FROM contacts ORDER BY created_at DESC");
  const userMap = new Map<string, string>();
  for (const u of await all<any>(ctx.env, "SELECT id, name FROM users")) userMap.set(u.id, u.name);
  const pm = await pipelineMap(ctx.env, contacts.map((c) => c.id));
  const nl = (v: unknown) => String(v ?? "").replace(/\n/g, " ");
  const rows = contacts.map((c) => {
    const p = pm.get(c.id);
    return [
      c.name || "", c.email || "", c.phone || "", c.linkedin || "",
      c.company_name || "", c.event_name || "",
      nl(c.notes), nl(c.admin_notes),
      c.lead_score || "", c.lead_temperature || "",
      c.source || "manual", userMap.get(c.user_id) || "Unknown", c.created_at || "",
      nl(p?.research_summary), nl(p?.pitch_angle), p?.pitch_email_subject || "", p?.gamma_deck_url || "",
    ];
  });
  const filename = `event_scout_all_contacts_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}.csv`;
  return csvResponse(buildCsv(ADMIN_CSV_HEADERS, rows), filename);
}

// ---- backup (app.py ~5684, ~5696): Postgres backup is a no-op, D1 Time Travel covers restore ----
const TIME_TRAVEL = "No manual backup needed on Cloudflare: D1 Time Travel provides 30-day point-in-time restore.";

async function adminBackup(_ctx: Ctx) {
  return {
    status: "success", timestamp: now(), duration_seconds: 0, tables: {}, total_rows: 0, summary: TIME_TRAVEL,
  };
}

const BACKUP_TABLES = ["users", "contacts", "exhibitors", "conversations", "user_cards", "event_files", "contact_pipelines", "admin_broadcasts"];

async function adminBackupStatus(ctx: Ctx) {
  const results = await ctx.env.DB.batch(BACKUP_TABLES.map((t) => ctx.env.DB.prepare(`SELECT count(*) AS n FROM ${t}`)));
  const counts: Record<string, number> = {};
  BACKUP_TABLES.forEach((t, i) => (counts[t] = (results[i]?.results?.[0] as any)?.n ?? 0));
  return {
    backup_configured: true, status: "success", message: TIME_TRAVEL,
    last_backup: { status: "success", timestamp: now(), summary: TIME_TRAVEL },
    primary_counts: counts, backup_counts: { ...counts }, in_sync: true,
  };
}

// ---- pipelines (app.py ~5812, ~5872) ----
async function adminPipelines(ctx: Ctx) {
  const statusFilter = q(ctx, "status_filter");
  const limit = qInt(ctx, "limit", 50, 0, 500);
  const conds: string[] = [], args: unknown[] = [];
  if (statusFilter) { conds.push("status = ?"); args.push(statusFilter); }
  const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "";
  const rows = await all<any>(ctx.env, `SELECT * FROM contact_pipelines ${where} ORDER BY created_at DESC LIMIT ?`, ...args, limit);
  const contactIds = [...new Set(rows.map((p) => p.contact_id))];
  const userIds = [...new Set(rows.map((p) => p.user_id))];
  const contactMap = new Map<string, string>();
  if (contactIds.length) for (const c of await allIn<any>(ctx.env, "SELECT id, name FROM contacts WHERE id IN (?IN)", contactIds)) contactMap.set(c.id, c.name);
  const userMap = new Map<string, string>();
  if (userIds.length) for (const u of await allIn<any>(ctx.env, "SELECT id, name FROM users WHERE id IN (?IN)", userIds)) userMap.set(u.id, u.name);
  const counts: Record<string, number> = {};
  for (const r of await all<{ status: string; n: number }>(ctx.env, "SELECT status, count(*) AS n FROM contact_pipelines GROUP BY status")) counts[r.status] = r.n;
  return {
    pipelines: rows.map((p) => ({
      id: p.id, contact_id: p.contact_id, contact_name: contactMap.get(p.contact_id) || "Unknown",
      user_id: p.user_id, user_name: userMap.get(p.user_id) || "Unknown",
      status: p.status, current_step: p.current_step, error_message: p.error_message,
      started_at: p.started_at ?? null, completed_at: p.completed_at ?? null,
    })),
    status_counts: counts,
    auto_pipeline_enabled: false,
  };
}

async function pipelineSettings(ctx: Ctx) {
  const ae = q(ctx, "auto_enabled");
  if (ae !== null && !["true", "false", "1", "0"].includes(ae)) throw new HttpError(422, "auto_enabled must be a boolean");
  const result: Record<string, unknown> = {};
  // No runtime toggle on Cloudflare: the setting always reads back false.
  if (ae !== null) result.auto_pipeline_enabled = false;
  result.current_settings = {
    auto_pipeline_enabled: false,
    gamma_configured: false,
    openrouter_configured: !!ctx.env.OPENROUTER_API_KEY,
  };
  return result;
}

// ---- contact lists (app.py ~5895-6091) ----
async function createList(ctx: Ctx) {
  const b = await ctx.body();
  const name = reqStr(b, "name").trim();
  if (!name) throw new HttpError(422, "name is required");
  const id = uuid();
  await run(ctx.env, "INSERT INTO contact_lists (id, name, admin_id, created_at) VALUES (?, ?, ?, ?)", id, name, ctx.user.id, now());
  return { status: "success", list_id: id, name };
}

async function getLists(ctx: Ctx) {
  const rows = await all<any>(ctx.env,
    `SELECT l.id, l.name, l.created_at, (SELECT count(*) FROM contact_list_members m WHERE m.list_id = l.id) AS n
     FROM contact_lists l ORDER BY l.created_at DESC`);
  return {
    status: "success",
    lists: rows.map((l) => ({ id: l.id, name: l.name, contact_count: l.n || 0, created_at: l.created_at || "" })),
  };
}

async function addListContacts(ctx: Ctx) {
  const b = await ctx.body();
  const ids = reqStrList(b, "contact_ids");
  const listId = ctx.params.list_id;
  if (!await first(ctx.env, "SELECT id FROM contact_lists WHERE id = ?", listId)) throw new HttpError(404, "List not found");
  const existing = new Set((await all<{ contact_id: string }>(ctx.env, "SELECT contact_id FROM contact_list_members WHERE list_id = ?", listId)).map((r) => r.contact_id));
  const ts = now();
  const stmts: D1PreparedStatement[] = [];
  for (const cid of ids) {
    if (existing.has(cid)) continue;
    existing.add(cid);
    stmts.push(ctx.env.DB.prepare("INSERT INTO contact_list_members (id, list_id, contact_id, added_at) VALUES (?, ?, ?, ?)").bind(uuid(), listId, cid, ts));
  }
  if (stmts.length) await ctx.env.DB.batch(stmts);
  const added = stmts.length;
  return { status: "success", added, skipped_duplicates: ids.length - added };
}

async function getListContacts(ctx: Ctx) {
  const listId = ctx.params.list_id;
  const rows = await all<any>(ctx.env,
    `SELECT c.* FROM contacts c JOIN contact_list_members m ON m.contact_id = c.id WHERE m.list_id = ? ORDER BY m.added_at DESC`, listId);
  const userIds = [...new Set(rows.map((c) => c.user_id))];
  const userMap = new Map<string, string>();
  if (userIds.length) for (const u of await allIn<any>(ctx.env, "SELECT id, name FROM users WHERE id IN (?IN)", userIds)) userMap.set(u.id, u.name);
  return {
    status: "success", list_id: listId, total: rows.length,
    contacts: rows.map((c) => ({
      id: c.id, name: c.name, email: c.email || "N/A", company_name: c.company_name || "N/A",
      lead_score: c.lead_score, lead_temperature: c.lead_temperature, event_name: c.event_name || "",
      user_name: userMap.get(c.user_id) || "Unknown", created_at: c.created_at || "",
    })),
  };
}

async function deleteList(ctx: Ctx) {
  const listId = ctx.params.list_id;
  if (!await first(ctx.env, "SELECT id FROM contact_lists WHERE id = ?", listId)) throw new HttpError(404, "List not found");
  // Postgres cascaded these; D1 does not, so delete members explicitly.
  await ctx.env.DB.batch([
    ctx.env.DB.prepare("DELETE FROM contact_list_members WHERE list_id = ?").bind(listId),
    ctx.env.DB.prepare("DELETE FROM contact_lists WHERE id = ?").bind(listId),
  ]);
  return { status: "success", message: "List deleted" };
}

async function removeListContact(ctx: Ctx) {
  await run(ctx.env, "DELETE FROM contact_list_members WHERE list_id = ? AND contact_id = ?", ctx.params.list_id, ctx.params.contact_id);
  return { status: "success", message: "Contact removed from list" };
}

// ---- per-contact pipeline status (app.py ~6091) ----
async function contactPipeline(ctx: Ctx) {
  const id = ctx.params.contact_id;
  const c = await first<any>(ctx.env, "SELECT id, user_id FROM contacts WHERE id = ?", id);
  if (c && !ctx.user.is_admin && c.user_id !== ctx.user.id) throw new HttpError(404, "Contact not found");
  const p = await first<any>(ctx.env, "SELECT * FROM contact_pipelines WHERE contact_id = ? ORDER BY created_at DESC LIMIT 1", id);
  if (!p) return { status: "none", message: "No pipeline run for this contact" };
  return {
    id: p.id, status: p.status, current_step: p.current_step, error_message: p.error_message,
    research_summary: p.research_summary, research_data: pj(p.research_data, {}),
    score_completed: !!p.score_completed,
    pitch_angle: p.pitch_angle, pitch_email_subject: p.pitch_email_subject,
    pitch_email_body: p.pitch_email_body, pitch_slides_content: pj(p.pitch_slides_content, []),
    deck_file_id: p.deck_file_id || null, gamma_deck_url: p.gamma_deck_url ?? null,
    started_at: p.started_at ?? null, completed_at: p.completed_at ?? null,
  };
}

export const routes: Route[] = [
  ["GET", "/dashboard", dashboard],
  ["GET", "/models/", listModels, "public"],
  ["POST", "/admin/import_exhibitors/", importExhibitors, "admin"],
  ["GET", "/exhibitors/", listExhibitors],
  ["GET", "/exhibitors/categories/", exhibitorCategories],
  ["GET", "/admin/users", adminListUsers, "admin"],
  ["GET", "/admin/contacts", adminListContacts, "admin"],
  ["GET", "/admin/activity", adminActivity, "admin"],
  ["DELETE", "/admin/contact/:contact_id", adminDeleteContact, "admin"],
  ["POST", "/admin/users/merge", mergeUsers, "admin"],
  ["POST", "/admin/contacts/merge", mergeContacts, "admin"],
  ["GET", "/admin/contacts/duplicates", adminDuplicates, "admin"],
  ["POST", "/admin/contacts/batch_delete", adminBatchDelete, "admin"],
  ["POST", "/admin/webhook/test", webhookTest, "admin"],
  ["GET", "/admin/dashboard", adminDashboard, "admin"],
  ["POST", "/admin/test_webhook", testWebhook, "admin"],
  ["GET", "/admin/search", adminSearch, "admin"],
  ["PUT", "/admin/contact/:contact_id/note", adminContactNote, "admin"],
  ["GET", "/admin/export", adminExport, "admin"],
  ["POST", "/admin/backup", adminBackup, "admin"],
  ["GET", "/admin/backup/status", adminBackupStatus, "admin"],
  ["GET", "/admin/pipelines", adminPipelines, "admin"],
  ["POST", "/admin/pipeline/settings", pipelineSettings, "admin"],
  ["POST", "/admin/contact_lists", createList, "admin"],
  ["GET", "/admin/contact_lists", getLists, "admin"],
  ["POST", "/admin/contact_lists/:list_id/contacts", addListContacts, "admin"],
  ["GET", "/admin/contact_lists/:list_id/contacts", getListContacts, "admin"],
  ["DELETE", "/admin/contact_lists/:list_id", deleteList, "admin"],
  ["DELETE", "/admin/contact_lists/:list_id/contacts/:contact_id", removeListContact, "admin"],
  ["GET", "/contact/:contact_id/pipeline", contactPipeline],
];
