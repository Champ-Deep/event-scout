// Core user-facing routes: health, auth, contacts, card scan, export, profile, digital card.
import type { Route } from "../index";
import {
  Ctx, Env, HttpError, all, b64, buildCsv, contactOut, csvResponse, extractJson, first, isAdminEmail,
  now, pipelineMap, pj, q, run, sha256Hex, shareOrigin, sj, uuid, websiteOf,
} from "../lib";
import { dropEmbedding, indexContacts, reindexLater } from "../embed";
import { status as harbingerStatus, sweep as harbingerSweep, syncSoon } from "../harbinger";
import bcrypt from "bcryptjs";
import { renderSVG } from "uqr";

const SESSION_DAYS = 30;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

async function newSession(env: Env, userId: string): Promise<string> {
  const raw = new Uint8Array(32);
  crypto.getRandomValues(raw);
  const token = b64(raw).replace(/[+/=]/g, (c) => ({ "+": "-", "/": "_", "=": "" })[c]!);
  const exp = new Date(Date.now() + SESSION_DAYS * 864e5).toISOString();
  await run(env, "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)", await sha256Hex(token), userId, now(), exp);
  return token;
}

function str(v: unknown, field: string, required = true): string {
  if (v === undefined || v === null) {
    if (required) throw new HttpError(422, `${field} is required`);
    return "";
  }
  if (typeof v !== "string") throw new HttpError(422, `${field} must be a string`);
  return v;
}

// ---- auth ----
async function register(ctx: Ctx) {
  const b = await ctx.body();
  const name = str(b.name, "name").trim();
  const email = str(b.email, "email").trim();
  const password = str(b.password, "password");
  if (!EMAIL_RE.test(email)) throw new HttpError(422, "value is not a valid email address");
  if (!name) throw new HttpError(422, "name is required");
  if (password.length < 6) throw new HttpError(422, "Password must be at least 6 characters");
  if (await first(ctx.env, "SELECT id FROM users WHERE lower(email) = lower(?)", email)) {
    throw new HttpError(400, "Email already registered");
  }
  const id = uuid();
  const isAdmin = isAdminEmail(ctx.env, email) ? 1 : 0;
  await run(ctx.env, "INSERT INTO users (id, name, email, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?)",
    id, name, email, await bcrypt.hash(password.slice(0, 72), 10), isAdmin, now());
  return {
    status: "success", message: "User registered successfully",
    user_id: id, name, email, is_admin: !!isAdmin, session_token: await newSession(ctx.env, id),
  };
}

async function login(ctx: Ctx) {
  const b = await ctx.body();
  const email = str(b.email, "email").trim();
  const password = str(b.password, "password");
  const u = await first<any>(ctx.env, "SELECT * FROM users WHERE lower(email) = lower(?)", email);
  // Legacy hashes are Python bcrypt over the first 72 bytes; bcryptjs truncates the same way.
  if (!u || !(await bcrypt.compare(password, u.password_hash))) throw new HttpError(401, "Invalid credentials");
  let isAdmin = !!u.is_admin;
  if (!isAdmin && isAdminEmail(ctx.env, u.email)) {
    await run(ctx.env, "UPDATE users SET is_admin = 1 WHERE id = ?", u.id);
    isAdmin = true;
  }
  return {
    status: "success", message: "Login successful",
    user_id: u.id, name: u.name, email: u.email, is_admin: isAdmin, session_token: await newSession(ctx.env, u.id),
  };
}

async function logout(ctx: Ctx) {
  const token = ctx.req.headers.get("x-api-key") || ctx.req.headers.get("authorization")?.replace(/^Bearer\s+/i, "") || "";
  await run(ctx.env, "DELETE FROM sessions WHERE token_hash = ?", await sha256Hex(token));
  return { status: "success" };
}

async function validate(ctx: Ctx) {
  const u = ctx.user;
  return { status: "valid", valid: true, user_id: u.id, name: u.name, email: u.email, is_admin: !!u.is_admin };
}

async function health(ctx: Ctx) {
  let users = 0, ok = false;
  try {
    users = (await first<any>(ctx.env, "SELECT count(*) n FROM users"))?.n ?? 0;
    ok = true;
  } catch { /* reported below */ }
  return {
    status: ok ? "healthy" : "degraded", database: "d1", database_connected: ok, total_users: users,
    ocr: ctx.env.OPENROUTER_API_KEY ? "openrouter+workers-ai" : "workers-ai",
    openrouter_configured: !!ctx.env.OPENROUTER_API_KEY, webhook_configured: false,
    version: "4.0.0-cf", timestamp: now(),
  };
}

// ---- contacts ----
async function currentEvent(env: Env, userId: string): Promise<string> {
  const p = await first<any>(env, "SELECT profile_data FROM user_profiles WHERE user_id = ?", userId);
  return (pj<any>(p?.profile_data, {}).current_event_name as string) || "";
}

async function insertContact(ctx: Ctx, c: any, source: string, photo: string | null = null) {
  const id = uuid();
  const ts = now();
  const ev = c.event_name || (await currentEvent(ctx.env, ctx.user.id));
  const row = {
    id, user_id: ctx.user.id, name: c.name, email: c.email ?? "N/A", phone: c.phone ?? "N/A", linkedin: c.linkedin ?? "N/A",
    company_name: c.company_name || "N/A", title: c.title || "", notes: c.notes || "", links: c.links || [],
  };
  await ctx.env.DB.batch([
    ctx.env.DB.prepare(
      `INSERT INTO contacts (id, user_id, name, email, phone, linkedin, linkedin_source, company_name, title, notes, links,
        source, event_name, photo_base64, lead_score_breakdown, lead_recommended_actions, audio_notes, admin_notes, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, 'card', ?, ?, ?, ?, ?, ?, ?, '{}', '[]', '[]', '', ?, ?)`,
    ).bind(row.id, row.user_id, row.name, row.email, row.phone, row.linkedin, row.company_name, row.title, row.notes,
      sj(row.links), source, ev, photo, ts, ts),
    ctx.env.DB.prepare(
      `INSERT INTO shared_contacts (id, original_contact_id, user_id, name, email, phone, linkedin, company_name, notes, links, source, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(uuid(), id, row.user_id, row.name, row.email, row.phone, row.linkedin, row.company_name, row.notes, sj(row.links), source, ts),
  ]);
  reindexLater(ctx.env, ctx.exec, [row]);
  syncSoon(ctx.env, ctx.exec, id);
  return id;
}

function contactInput(c: any) {
  if (!c || typeof c !== "object") throw new HttpError(422, "contact is required");
  for (const f of ["name", "email", "phone", "linkedin"]) str(c[f], `contact.${f}`);
  if (c.links !== undefined && !Array.isArray(c.links)) throw new HttpError(422, "contact.links must be a list");
  return c;
}

async function addContact(ctx: Ctx) {
  const b = await ctx.body();
  const id = await insertContact(ctx, contactInput(b.contact), "manual");
  return { status: "success", message: "Contact added", contact_id: id, qr_base64: null };
}

async function duplicateOf(env: Env, userId: string, email: string, phone: string) {
  const conds: string[] = [], args: string[] = [];
  if (email && email !== "N/A") { conds.push("email = ?"); args.push(email); }
  if (phone && phone !== "N/A") { conds.push("phone = ?"); args.push(phone); }
  if (!conds.length) return null;
  const c = await first<any>(env, `SELECT id, name, email FROM contacts WHERE user_id = ? AND (${conds.join(" OR ")}) LIMIT 1`, userId, ...args);
  if (!c) return null;
  return { contact_id: c.id, name: c.name, matched_on: email && email !== "N/A" && c.email === email ? "email" : "phone" };
}

// ---- OCR ----
export const OCR_FIELDS = ["name", "email", "phone", "linkedin", "company_name", "title", "website"] as const;
const OCR_PROMPT = `Read this business card photo and extract the contact fields.
Return ONLY a JSON object with exactly these keys:
{"name": "", "title": "", "company_name": "", "email": "", "phone": "", "linkedin": "", "website": ""}
Rules: copy text exactly as printed (keep +country codes, punctuation in emails and URLs). Use "" for a field that is not on the card. Do not guess or invent anything. No markdown, no commentary.`;
// Workers AI bench 2026-11-21 (24 cards, photo + glare): mistral-small-3.1 99.4% fields, p50 3.6s; llama-4-scout 97.6%, p50 7.7s.
export const OCR_WAI_MODELS = ["@cf/mistralai/mistral-small-3.1-24b-instruct", "@cf/meta/llama-4-scout-17b-16e-instruct"];
const OCR_OR_MODELS = ["anthropic/claude-haiku-5.5", "google/gemini-3.5-flash-lite"];

function mergeOcr(raw: any) {
  const f: Record<string, string> = Object.fromEntries(OCR_FIELDS.map((k) => [k, "N/A"]));
  for (const k of OCR_FIELDS) {
    const v = raw?.[k];
    if (v && String(v).trim() && v !== "N/A") f[k] = String(v).trim();
  }
  return f;
}
const hasInfo = (f: Record<string, string>) => ["name", "email", "phone", "company_name"].some((k) => f[k] !== "N/A");

/** Card OCR: OpenRouter first when keyed (benchmarked best), then Workers AI vision. */
export async function ocrCard(env: Env, dataUrl: string, only?: string): Promise<{ fields: Record<string, string>; source: string; errors: string[] }> {
  const errors: string[] = [];
  const content = [{ type: "text", text: OCR_PROMPT }, { type: "image_url", image_url: { url: dataUrl } }];
  if (env.OPENROUTER_API_KEY && (!only || only.startsWith("openrouter:"))) {
    for (const m of only ? [only.slice(11)] : OCR_OR_MODELS) {
      try {
        const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST",
          headers: { Authorization: `Bearer ${env.OPENROUTER_API_KEY}`, "Content-Type": "application/json", "X-Title": "Event Scout OCR" },
          body: JSON.stringify({ model: m, messages: [{ role: "user", content }], max_tokens: 500, temperature: 0 }),
          signal: AbortSignal.timeout(20000),
        });
        const d: any = await r.json();
        const text = d?.choices?.[0]?.message?.content;
        if (!text) throw new Error(d?.error?.message || `HTTP ${r.status}`);
        const f = mergeOcr(extractJson(text));
        if (hasInfo(f)) return { fields: f, source: `openrouter:${m}`, errors };
        errors.push(`openrouter:${m}: no contact info`);
      } catch (e) {
        errors.push(`openrouter:${m}: ${String((e as Error).message || e).slice(0, 120)}`);
      }
    }
  }
  for (const m of only ? (only.startsWith("@cf/") ? [only] : []) : OCR_WAI_MODELS) {
    try {
      const out: any = await env.AI.run(m as any, { messages: [{ role: "user", content }], max_tokens: 500, temperature: 0 } as any);
      const text = typeof out?.response === "string" ? out.response : JSON.stringify(out?.response);
      const f = mergeOcr(extractJson(text));
      if (hasInfo(f)) return { fields: f, source: `workers-ai:${m}`, errors };
      errors.push(`${m}: no contact info`);
    } catch (e) {
      errors.push(`${m}: ${String((e as Error).message || e).slice(0, 120)}`);
    }
  }
  return { fields: mergeOcr({}), source: "", errors };
}

/** Resize through the Images binding when present; the raw upload otherwise. */
async function resized(env: any, bytes: ArrayBuffer, mime: string, px: number, quality: number): Promise<{ data: string; mime: string } | null> {
  if (!env.IMAGES) return null;
  try {
    const out = await env.IMAGES.input(new Blob([bytes]).stream())
      .transform({ width: px, height: px, fit: "scale-down" })
      .output({ format: "image/jpeg", quality });
    return { data: b64(await out.response().arrayBuffer()), mime: "image/jpeg" };
  } catch (e) {
    console.log("image resize failed", String(e));
    return null;
  }
}

async function addContactFromImage(ctx: Ctx) {
  const form = await ctx.req.formData().catch(() => null);
  const file = form?.get("file");
  if (!file || typeof file === "string") throw new HttpError(422, "file is required");
  const bytes = await (file as File).arrayBuffer();
  if (bytes.byteLength < 1000) throw new HttpError(400, "Image file is empty or corrupted. Please try again.");
  if (bytes.byteLength > 15 * 1024 * 1024) throw new HttpError(413, "Image is larger than 15 MB. Please retake the photo.");
  const mime = (file as File).type || "image/jpeg";

  const forOcr = (await resized(ctx.env, bytes, mime, 1600, 85)) || { data: b64(bytes), mime };
  const thumb = await resized(ctx.env, bytes, mime, 200, 60);
  const { fields, source, errors } = await ocrCard(ctx.env, `data:${forOcr.mime};base64,${forOcr.data}`);
  if (!hasInfo(fields)) {
    throw new HttpError(400, `No contact information found in image. Models tried: ${errors.join(", ")}`);
  }
  console.log("scan ok via", source);

  const out: any = { ...fields };
  const website = out.website;
  delete out.website;
  if (website !== "N/A") out.links = [{ url: website, label: "Website", added_by: "scan" }];
  if (out.title === "N/A") out.title = "";

  const dup = await duplicateOf(ctx.env, ctx.user.id, out.email, out.phone);
  if (dup) {
    return {
      status: "duplicate", message: `Contact already exists: ${dup.name}`, existing_contact_id: dup.contact_id,
      existing_name: dup.name, matched_on: dup.matched_on, extracted_fields: out,
    };
  }
  const photo = thumb?.data ?? null;
  const id = await insertContact(ctx, out, "scan", photo);
  return {
    status: "success", message: "Contact added from image", extracted_fields: out, contact_id: id,
    qr_base64: null, photo_base64: photo, pipeline_started: false, ocr_source: source,
  };
}

async function listContacts(ctx: Ctx) {
  const df = q(ctx, "date_filter");
  const d = new Date();
  const todayStart = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate())).toISOString();
  const conds = ["user_id = ?"], args: unknown[] = [ctx.user.id];
  if (df === "today") { conds.push("created_at >= ?"); args.push(todayStart); }
  else if (df === "yesterday") { conds.push("created_at >= ? AND created_at < ?"); args.push(new Date(Date.parse(todayStart) - 864e5).toISOString(), todayStart); }
  else if (df === "last_week") { conds.push("created_at >= ?"); args.push(new Date(Date.now() - 7 * 864e5).toISOString()); }
  else if (df === "last_month") { conds.push("created_at >= ?"); args.push(new Date(Date.now() - 30 * 864e5).toISOString()); }
  const rows = await all(ctx.env, `SELECT * FROM contacts WHERE ${conds.join(" AND ")} ORDER BY created_at DESC`, ...args);
  return { status: "success", total_contacts: rows.length, contacts: rows.map((c) => contactOut(c)) };
}

async function ownContact(ctx: Ctx) {
  const c = await first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ? AND user_id = ?", ctx.params.contact_id, ctx.user.id);
  if (!c) throw new HttpError(404, "Contact not found");
  return c;
}

async function getContact(ctx: Ctx) {
  return { status: "success", contact: contactOut(await ownContact(ctx), true) };
}

async function deleteContact(ctx: Ctx) {
  const c = await ownContact(ctx);
  await deleteContactRows(ctx.env, [c.id]);
  return { status: "success", message: `Contact '${c.name}' deleted successfully`, deleted_contact_id: c.id };
}

/** Delete contacts and everything hanging off them (Postgres FKs did this implicitly or not at all). */
export async function deleteContactRows(env: Env, ids: string[]) {
  const stmts: D1PreparedStatement[] = [];
  for (const id of ids) {
    stmts.push(env.DB.prepare("DELETE FROM contact_list_members WHERE contact_id = ?").bind(id));
    stmts.push(env.DB.prepare("DELETE FROM contact_pipelines WHERE contact_id = ?").bind(id));
    stmts.push(env.DB.prepare("DELETE FROM contact_embeddings WHERE contact_id = ?").bind(id));
    stmts.push(env.DB.prepare("DELETE FROM contacts WHERE id = ?").bind(id));
  }
  // Contact files: drop the KV bytes too.
  for (const id of ids) {
    const files = await all<any>(env, "SELECT id FROM contact_files WHERE contact_id = ?", id);
    for (const f of files) await env.FILES.delete(`file:${f.id}`);
    stmts.push(env.DB.prepare("DELETE FROM contact_files WHERE contact_id = ?").bind(id));
  }
  if (stmts.length) await env.DB.batch(stmts);
}

const UPDATABLE = ["name", "email", "phone", "linkedin", "company_name", "notes", "links", "title"] as const;

async function updateContact(ctx: Ctx) {
  const c = await ownContact(ctx);
  const b = await ctx.body();
  const sets: string[] = [], args: unknown[] = [];
  for (const k of UPDATABLE) {
    if (b[k] === undefined || b[k] === null) continue;
    if (k === "links") {
      if (!Array.isArray(b[k])) throw new HttpError(422, "links must be a list");
      sets.push("links = ?"); args.push(sj(b[k]));
    } else {
      sets.push(`${k} = ?`); args.push(str(b[k], k));
    }
    c[k] = b[k];
  }
  if (!sets.length) throw new HttpError(400, "No fields provided for update");
  await run(ctx.env, `UPDATE contacts SET ${sets.join(", ")}, updated_at = ? WHERE id = ?`, ...args, now(), c.id);
  reindexLater(ctx.env, ctx.exec, [c]);
  syncSoon(ctx.env, ctx.exec, c.id);
  const links = typeof c.links === "string" ? pj(c.links, []) : c.links;
  return {
    status: "success", message: "Contact updated successfully",
    contact: {
      id: c.id, name: c.name, email: c.email || "N/A", phone: c.phone || "N/A", linkedin: c.linkedin || "N/A",
      company_name: c.company_name || "N/A", notes: c.notes || "", links, qr_base64: null,
    },
  };
}

export const CONTACT_CSV_HEADERS = [
  "Name", "Email", "Phone", "LinkedIn", "Company", "Event", "Notes", "Source", "Lead Score", "Lead Temperature",
  "Research Summary", "Pitch Email Subject", "Gamma Deck URL", "Title", "Website",
];

async function exportContacts(ctx: Ctx) {
  const contacts = await all<any>(ctx.env, "SELECT * FROM contacts WHERE user_id = ?", ctx.user.id);
  const pm = await pipelineMap(ctx.env, contacts.map((c) => c.id));
  if (q(ctx, "format") === "json") {
    return {
      status: "success", total: contacts.length,
      contacts: contacts.map((c) => {
        const p = pm.get(c.id);
        return {
          name: c.name, email: c.email || "N/A", phone: c.phone || "N/A", linkedin: c.linkedin || "N/A",
          company_name: c.company_name || "N/A", event_name: c.event_name || "", notes: c.notes || "", links: pj(c.links, []),
          source: c.source || "manual", lead_score: c.lead_score, lead_temperature: c.lead_temperature,
          research_summary: p?.research_summary || "", pitch_angle: p?.pitch_angle || "",
          pitch_email_subject: p?.pitch_email_subject || "", gamma_deck_url: p?.gamma_deck_url || "",
        };
      }),
    };
  }
  // Empty instead of "N/A": Harbinger dedupes imports on email (see export_csv.py).
  const blank = (v: any) => (v && v !== "N/A" ? v : "");
  const rows = contacts.map((c) => {
    const p = pm.get(c.id);
    return [
      c.name, blank(c.email), blank(c.phone), blank(c.linkedin), blank(c.company_name), c.event_name || "",
      (c.notes || "").replace(/\n/g, " "), c.source || "manual", c.lead_score || "", c.lead_temperature || "",
      (p?.research_summary || "").replace(/\n/g, " "), p?.pitch_email_subject || "", p?.gamma_deck_url || "",
      c.title || "", websiteOf(c),
    ];
  });
  return csvResponse(buildCsv(CONTACT_CSV_HEADERS, rows), "event_scout_contacts.csv");
}

// ---- profile ----
const PROFILE_KEYS = [
  "full_name", "job_title", "company_name", "company_website", "linkedin_url", "products", "target_industries",
  "target_company_sizes", "target_roles", "target_geographies", "pitch_style", "value_propositions", "common_objections",
  "case_studies", "preferred_follow_up_delay_days", "auto_research_on_scan", "auto_score_on_add", "email_signature",
  "current_event_name", "current_event_description", "event_goals", "preferred_ai_model",
];

async function getProfile(ctx: Ctx) {
  const p = await first<any>(ctx.env, "SELECT profile_data FROM user_profiles WHERE user_id = ?", ctx.user.id);
  return { status: "success", profile: pj(p?.profile_data, {}) };
}

async function putProfile(ctx: Ctx) {
  const b = await ctx.body();
  const upd = Object.fromEntries(PROFILE_KEYS.filter((k) => b[k] !== undefined).map((k) => [k, b[k]]));
  if (!Object.keys(upd).length) throw new HttpError(400, "No fields provided for update");
  const p = await first<any>(ctx.env, "SELECT id, profile_data FROM user_profiles WHERE user_id = ?", ctx.user.id);
  const data = pj<any>(p?.profile_data, {});
  for (const [k, v] of Object.entries(upd)) if (v !== null) data[k] = v;
  if (p) await run(ctx.env, "UPDATE user_profiles SET profile_data = ?, updated_at = ? WHERE id = ?", sj(data), now(), p.id);
  else await run(ctx.env, "INSERT INTO user_profiles (id, user_id, profile_data, updated_at) VALUES (?, ?, ?, ?)", uuid(), ctx.user.id, sj(data), now());
  return { status: "success", profile: data };
}

// ---- digital card ----
const CARD_KEYS = ["full_name", "job_title", "company_name", "email", "phone", "linkedin_url", "zoom_number", "website", "photo_url", "photo_base64", "bio", "custom_fields"];
const cardUrl = (ctx: Ctx, token: string) => `${shareOrigin(ctx)}/card/${token}`;
// The card is shown on the public /card/:token page, same origin as the signed-in app: the photo must be an image data URL.
const CARD_PHOTO = /^data:image\/(png|jpe?g|webp|gif);base64,[A-Za-z0-9+/=\s]+$/;
function checkCardField(k: string, v: unknown) {
  if (k === "photo_base64" && typeof v === "string" && v.trim() && !CARD_PHOTO.test(v)) {
    throw new HttpError(422, "photo_base64 must be a data:image/... base64 URL");
  }
}

async function getCard(ctx: Ctx) {
  const c = await first<any>(ctx.env, "SELECT * FROM user_cards WHERE user_id = ?", ctx.user.id);
  if (!c) return { status: "success", card: {}, shareable_url: null };
  return {
    status: "success", card: pj(c.card_data, {}),
    shareable_url: c.shareable_token && c.is_active ? cardUrl(ctx, c.shareable_token) : null,
    is_active: !!c.is_active, updated_at: c.updated_at,
  };
}

async function putCard(ctx: Ctx) {
  const b = await ctx.body();
  const keep = (v: unknown) => v !== null && v !== undefined && (typeof v !== "string" || v.trim() !== "");
  const upd = Object.fromEntries(CARD_KEYS.filter((k) => b[k] !== undefined).map((k) => [k, b[k]]));
  if (!Object.keys(upd).length) throw new HttpError(400, "No fields provided for update");
  for (const [k, v] of Object.entries(upd)) checkCardField(k, v);
  const c = await first<any>(ctx.env, "SELECT * FROM user_cards WHERE user_id = ?", ctx.user.id);
  const data = pj<any>(c?.card_data, {});
  for (const [k, v] of Object.entries(upd)) if (keep(v)) data[k] = v;
  if (c) await run(ctx.env, "UPDATE user_cards SET card_data = ?, updated_at = ? WHERE id = ?", sj(data), now(), c.id);
  else await run(ctx.env, "INSERT INTO user_cards (id, user_id, card_data, is_active, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)", uuid(), ctx.user.id, sj(data), now(), now());
  return {
    status: "success", card: data,
    shareable_url: c?.shareable_token && c?.is_active ? cardUrl(ctx, c.shareable_token) : null,
  };
}

async function generateCardToken(ctx: Ctx) {
  let c = await first<any>(ctx.env, "SELECT * FROM user_cards WHERE user_id = ?", ctx.user.id);
  let token = c?.shareable_token as string | undefined;
  if (!c) {
    token = uuid();
    await run(ctx.env, "INSERT INTO user_cards (id, user_id, card_data, shareable_token, is_active, created_at, updated_at) VALUES (?, ?, '{}', ?, 1, ?, ?)", uuid(), ctx.user.id, token, now(), now());
  } else if (!token) {
    token = uuid();
    await run(ctx.env, "UPDATE user_cards SET shareable_token = ?, is_active = 1, updated_at = ? WHERE id = ?", token, now(), c.id);
  }
  const url = cardUrl(ctx, token!);
  return {
    status: "success", shareable_url: url, shareable_token: token,
    qr_code_base64: btoa(renderSVG(url, { border: 4 })), qr_code_mime: "image/svg+xml",
    message: "Share this QR code or URL with people you meet!",
  };
}

async function viewCard(ctx: Ctx) {
  const c = await first<any>(ctx.env, "SELECT * FROM user_cards WHERE shareable_token = ?", ctx.params.token);
  if (!c) throw new HttpError(404, "Card not found");
  if (!c.is_active) throw new HttpError(403, "This card is no longer active");
  return { status: "success", card: pj(c.card_data, {}), created_at: c.created_at };
}

async function sharedContacts(ctx: Ctx) {
  const rows = await all<any>(ctx.env, "SELECT * FROM shared_contacts WHERE user_id = ? ORDER BY created_at DESC", ctx.user.id);
  return {
    status: "success", total: rows.length,
    contacts: rows.map((r) => ({ ...r, links: pj(r.links, []), webhook_sent: !!r.webhook_sent, enriched: !!r.enriched })),
  };
}

const gone = (what: string) => async () => {
  throw new HttpError(501, `${what} is not available on the Cloudflare build yet.`);
};

/** Admin OCR probe for model benchmarks: extraction only, nothing is saved. */
async function ocrProbe(ctx: Ctx) {
  const form = await ctx.req.formData().catch(() => null);
  const file = form?.get("file");
  if (!file || typeof file === "string") throw new HttpError(422, "file is required");
  const bytes = await (file as File).arrayBuffer();
  const img = (await resized(ctx.env, bytes, (file as File).type, 1600, 85)) || { data: b64(bytes), mime: (file as File).type || "image/jpeg" };
  const t0 = Date.now();
  const r = await ocrCard(ctx.env, `data:${img.mime};base64,${img.data}`, q(ctx, "model") || undefined);
  return { ...r, ms: Date.now() - t0 };
}

export const routes: Route[] = [
  ["GET", "/", async () => ({ message: "Event Scout Intelligence API", version: "4.0.0-cf" }), "public"],
  ["GET", "/health/", health, "public"],
  ["POST", "/register/", register, "public"],
  ["POST", "/login/", login, "public"],
  ["POST", "/logout/", logout],
  ["GET", "/user/validate", validate],
  ["POST", "/add_contact/", addContact],
  ["POST", "/add_contact_from_image/", addContactFromImage],
  ["POST", "/scan_qr/", gone("QR image scanning")],
  ["GET", "/list_contacts/", listContacts],
  ["GET", "/contact/:contact_id", getContact],
  ["DELETE", "/contact/:contact_id", deleteContact],
  ["PUT", "/contact/:contact_id", updateContact],
  ["GET", "/export_contacts/", exportContacts],
  ["GET", "/user/profile/", getProfile],
  ["PUT", "/user/profile/", putProfile],
  ["GET", "/user/card/", getCard],
  ["PUT", "/user/card/", putCard],
  ["POST", "/user/card/generate_token/", generateCardToken],
  ["GET", "/card/:token", viewCard, "public"],
  ["GET", "/shared_contacts/", sharedContacts],
  ["POST", "/admin/ocr_probe", ocrProbe, "admin"],
  ["GET", "/admin/harbinger/status", (ctx) => harbingerStatus(ctx.env), "admin"],
  ["POST", "/admin/harbinger/sync", async (ctx) => {
    // ?reset=1 un-parks contacts that failed too often (after a fix on the Harbinger side).
    if (q(ctx, "reset") === "1") await run(ctx.env, "UPDATE harbinger_sync SET attempts = 0 WHERE status = 'failed'");
    return { status: "success", ...(await harbingerSweep(ctx.env, 200)) };
  }, "admin"],
  ["POST", "/admin/reindex", async (ctx) => {
    const rows = await all<any>(ctx.env, "SELECT * FROM contacts");
    for (let i = 0; i < rows.length; i += 50) await indexContacts(ctx.env, rows.slice(i, i + 50));
    return { status: "success", indexed: rows.length };
  }, "admin"],
];
