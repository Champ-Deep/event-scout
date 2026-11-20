// Shared helpers for the Event Scout Worker. Every route module imports from here only.

export interface Env {
  DB: D1Database;
  FILES: KVNamespace; // file bytes: key `file:<id>`
  AI: Ai;
  ASSETS: Fetcher;
  ADMIN_EMAILS?: string;
  OPENROUTER_API_KEY?: string; // optional: OCR/chat via OpenRouter first, Workers AI otherwise
  PUBLIC_ORIGIN?: string; // e.g. https://event-scout.deep-ae0.workers.dev (used in share links)
}

export interface User {
  id: string;
  name: string;
  email: string;
  is_admin: number;
  created_at: string | null;
}

export interface Ctx {
  req: Request;
  env: Env;
  exec: ExecutionContext;
  url: URL;
  params: Record<string, string>;
  user: User; // set for every non-public route; public routes get a placeholder
  body: () => Promise<any>; // parsed JSON body (cached), {} when empty
}

/** FastAPI-compatible error: `{detail}` with a status code. */
export class HttpError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

export const json = (body: unknown, status = 200, headers: HeadersInit = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });

export const now = () => new Date().toISOString();
export const uuid = () => crypto.randomUUID();

export function q(ctx: Ctx, name: string): string | null {
  return ctx.url.searchParams.get(name);
}

export function qInt(ctx: Ctx, name: string, dflt: number, min = 0, max = 100000): number {
  const v = q(ctx, name);
  const n = v === null ? dflt : parseInt(v, 10);
  if (Number.isNaN(n)) throw new HttpError(422, `${name} must be an integer`);
  return Math.min(max, Math.max(min, n));
}

/** Parse a JSON TEXT column; tolerate NULL and legacy junk. */
export function pj<T = any>(v: unknown, dflt: T): T {
  if (v === null || v === undefined || v === "") return dflt;
  if (typeof v !== "string") return v as T;
  try {
    return JSON.parse(v) as T;
  } catch {
    return dflt;
  }
}

export const sj = (v: unknown) => JSON.stringify(v ?? null);

export async function first<T = any>(env: Env, sql: string, ...args: unknown[]): Promise<T | null> {
  return (await env.DB.prepare(sql).bind(...args).first<T>()) ?? null;
}

export async function all<T = any>(env: Env, sql: string, ...args: unknown[]): Promise<T[]> {
  return (await env.DB.prepare(sql).bind(...args).all<T>()).results;
}

export async function run(env: Env, sql: string, ...args: unknown[]) {
  return env.DB.prepare(sql).bind(...args).run();
}

/** D1 caps bound parameters per statement (100); chunk IN-lists through this. */
export function chunks<T>(xs: T[], n = 90): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n));
  return out;
}

export async function allIn<T = any>(env: Env, sqlWithIn: string, ids: string[], ...pre: unknown[]): Promise<T[]> {
  // sqlWithIn contains the literal token `(?IN)` where the id list goes.
  const out: T[] = [];
  for (const part of chunks(ids)) {
    const sql = sqlWithIn.replace("(?IN)", `(${part.map(() => "?").join(",")})`);
    out.push(...(await all<T>(env, sql, ...pre, ...part)));
  }
  return out;
}

export const isAdminEmail = (env: Env, email: string) =>
  (env.ADMIN_EMAILS || "deep@lakeb2b.com")
    .split(",")
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean)
    .includes(email.trim().toLowerCase());

export function requireAdmin(ctx: Ctx) {
  if (!ctx.user.is_admin) throw new HttpError(403, "Admin access required");
}

/** Contact row -> the dict shape app.py returned from list_contacts/get_contact. */
export function contactOut(c: any, full = false) {
  const out: Record<string, unknown> = {
    id: c.id,
    name: c.name,
    email: c.email || "N/A",
    phone: c.phone || "N/A",
    linkedin: c.linkedin || "N/A",
    linkedin_source: c.linkedin_source || "card",
    company_name: c.company_name || "N/A",
    notes: c.notes || "",
    links: pj(c.links, []),
    source: c.source || "manual",
    created_at: c.created_at || "",
    lead_score: c.lead_score ?? null,
    lead_temperature: c.lead_temperature ?? null,
    lead_score_reasoning: c.lead_score_reasoning || "",
    event_name: c.event_name || "",
    title: c.title || "",
    admin_notes: c.admin_notes || "",
    photo_base64: c.photo_base64 ?? null,
    qr_base64: null, // QR images are rendered client-side now
  };
  if (full) {
    out.lead_score_breakdown = pj(c.lead_score_breakdown, {});
    out.lead_recommended_actions = pj(c.lead_recommended_actions, []);
  }
  return out;
}

/** Latest pipeline row per contact id (research/pitch/deck columns used by exports). */
export async function pipelineMap(env: Env, contactIds: string[]) {
  const rows = await allIn(env, "SELECT * FROM contact_pipelines WHERE contact_id IN (?IN) ORDER BY created_at DESC", contactIds);
  const map = new Map<string, any>();
  for (const p of rows) if (!map.has(p.contact_id)) map.set(p.contact_id, p);
  return map;
}

// ---- CSV (port of export_csv.py: UTF-8 BOM, formula-injection guard) ----
const FORMULA = ["=", "+", "-", "@"];
export function safeCell(v: unknown): string {
  const t = v === null || v === undefined ? "" : String(v);
  return t && FORMULA.includes(t[0]) ? "'" + t : t;
}
function csvField(s: string) {
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
export function buildCsv(header: string[], rows: unknown[][]): string {
  const lines = [header.map(csvField).join(",")];
  for (const r of rows) lines.push(r.map((v) => csvField(safeCell(v))).join(","));
  return "\ufeff" + lines.join("\r\n") + "\r\n";
}
export function websiteOf(c: any): string {
  for (const l of pj<any[]>(c.links, [])) if (l && l.label === "Website" && l.url) return l.url;
  return "";
}
export const csvResponse = (text: string, filename: string) =>
  new Response(text, {
    headers: { "Content-Type": "text/csv", "Content-Disposition": `attachment; filename=${filename}` },
  });

// ---- crypto ----
export async function sha256Hex(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function b64(bytes: ArrayBuffer | Uint8Array): string {
  const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
}

export function shareOrigin(ctx: Ctx): string {
  return ctx.env.PUBLIC_ORIGIN || ctx.url.origin;
}

// ---- LLM access: OpenRouter when a key is set, Workers AI otherwise ----
export const WAI_TEXT_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct";
export const OR_TEXT_MODEL = "anthropic/claude-haiku-5.5";

export async function chat(
  env: Env,
  messages: { role: string; content: any }[],
  opts: { maxTokens?: number; json?: boolean } = {},
): Promise<{ text: string; model: string }> {
  const maxTokens = opts.maxTokens ?? 1500;
  if (env.OPENROUTER_API_KEY) {
    try {
      const r = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENROUTER_API_KEY}`, "Content-Type": "application/json", "X-Title": "Event Scout" },
        body: JSON.stringify({ model: OR_TEXT_MODEL, messages, max_tokens: maxTokens, temperature: 0.3 }),
        signal: AbortSignal.timeout(45000),
      });
      const d: any = await r.json();
      const text = d?.choices?.[0]?.message?.content;
      if (r.ok && text) return { text: String(text).trim(), model: `openrouter:${OR_TEXT_MODEL}` };
      console.log("openrouter chat failed", r.status, JSON.stringify(d?.error ?? d).slice(0, 200));
    } catch (e) {
      console.log("openrouter chat error", String(e));
    }
  }
  const out: any = await env.AI.run(WAI_TEXT_MODEL as any, { messages, max_tokens: maxTokens } as any);
  const text = typeof out?.response === "string" ? out.response : JSON.stringify(out?.response ?? "");
  return { text: text.trim(), model: `workers-ai:${WAI_TEXT_MODEL}` };
}

/** Pull the first JSON object out of a model reply (handles ```json fences and chatter). */
export function extractJson(text: string): any {
  let t = text.trim();
  if (t.startsWith("```")) {
    const parts = t.split("```");
    t = (parts[1] || t).replace(/^json/i, "").trim();
  }
  try {
    return JSON.parse(t);
  } catch {
    const i = t.indexOf("{"), j = t.lastIndexOf("}");
    if (i !== -1 && j > i) return JSON.parse(t.slice(i, j + 1));
    throw new Error("No JSON object found in model reply");
  }
}
