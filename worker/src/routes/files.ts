// Event files, saved exports, voice notes, broadcasts and contact documents.
// Port of app.py's file/export/audio/broadcast endpoints; file bytes live in KV under `file:<id>`.
import type { Route } from "../index";
import {
  Ctx, HttpError, all, allIn, b64, buildCsv, first, now, pipelineMap, pj, q, run, sj, uuid, websiteOf,
} from "../lib";
import { CONTACT_CSV_HEADERS } from "./core";
import { reindexLater } from "../embed";

const p2 = (n: number) => String(n).padStart(2, "0");
const mb1 = (n: number) => (n / 1024 / 1024).toFixed(1);
const mb2 = (n: number) => Math.round((n / 1024 / 1024) * 100) / 100;
// Filenames ride in a response header: strip CR/LF and quotes, then quote the rest.
const disposition = (name: unknown) => `attachment; filename="${String(name ?? "").replace(/[\r\n"]/g, "")}"`;
const safeName = (s: string) => s.replace(/[^\w\s\-.]/g, "_");
const ownContact = (ctx: Ctx) => first<any>(ctx.env, "SELECT * FROM contacts WHERE id = ? AND user_id = ?", ctx.params.contact_id, ctx.user.id);

const EVENT_FILE_TYPES: Record<string, string> = {
  "application/pdf": "pdf",
  "application/vnd.ms-powerpoint": "ppt",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
};
const CONTACT_FILE_TYPES: Record<string, string> = {
  ...EVENT_FILE_TYPES,
  "application/msword": "doc",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "image/png": "png",
  "image/jpeg": "jpg",
};
const CONTACT_CATEGORIES = ["research", "pitch", "brief", "other"];

/** Multipart upload: FastAPI's `file: UploadFile = File(...)`. */
async function uploadedFile(ctx: Ctx) {
  const form = await ctx.req.formData().catch(() => null);
  const f = form?.get("file");
  if (!f || typeof f === "string") throw new HttpError(422, "file is required");
  return { bytes: await f.arrayBuffer(), name: f.name, mime: f.type };
}

// ---- exports ----

async function createExport(ctx: Ctx) {
  const b = await ctx.body();
  const ids: string[] = Array.isArray(b.contact_ids) ? b.contact_ids.filter((x: unknown) => typeof x === "string") : [];
  if (!ids.length) throw new HttpError(400, "contact_ids cannot be empty");
  const format = b.format === undefined || b.format === null ? "csv" : b.format;
  if (format !== "csv" && format !== "json") throw new HttpError(400, "format must be 'csv' or 'json'");
  const filterLabel: string = typeof b.filter_label === "string" ? b.filter_label : "All";

  const contacts = await allIn<any>(ctx.env, "SELECT * FROM contacts WHERE user_id = ? AND id IN (?IN)", ids, ctx.user.id);
  if (!contacts.length) throw new HttpError(404, "No matching contacts found");
  const pm = await pipelineMap(ctx.env, contacts.map((c) => c.id));

  let content: string;
  if (format === "json") {
    content = JSON.stringify(
      contacts.map((c) => {
        const p = pm.get(c.id);
        return {
          name: c.name, email: c.email || "N/A", phone: c.phone || "N/A", linkedin: c.linkedin || "N/A",
          company_name: c.company_name || "N/A", event_name: c.event_name || "", notes: c.notes || "",
          links: pj(c.links, []), source: c.source || "manual",
          lead_score: c.lead_score, lead_temperature: c.lead_temperature,
          research_summary: p?.research_summary || "", pitch_angle: p?.pitch_angle || "",
          pitch_email_subject: p?.pitch_email_subject || "", gamma_deck_url: p?.gamma_deck_url || "",
        };
      }),
      null, 2,
    );
  } else {
    // Rows follow export_csv.py: empty (not "N/A") so Harbinger's email dedupe works.
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
    content = buildCsv(CONTACT_CSV_HEADERS, rows);
  }

  const d = new Date();
  const ts = `${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}_${p2(d.getUTCHours())}${p2(d.getUTCMinutes())}${p2(d.getUTCSeconds())}`;
  const slug = filterLabel.toLowerCase().replace(/ /g, "_");
  const filename = (typeof b.filename === "string" && b.filename) || `contacts_${slug}_${ts}.${format}`;
  const id = uuid(), created = now();
  await run(
    ctx.env,
    "INSERT INTO user_exports (id, user_id, filename, format, filter_label, contact_count, file_content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    id, ctx.user.id, filename, format, filterLabel, contacts.length, content, created,
  );
  return {
    status: "success", id, filename, format, filter_label: filterLabel,
    contact_count: contacts.length, created_at: created,
  };
}

async function listExports(ctx: Ctx) {
  const rows = await all<any>(ctx.env, "SELECT * FROM user_exports WHERE user_id = ? ORDER BY created_at DESC", ctx.user.id);
  return {
    status: "success",
    exports: rows.map((e) => ({
      id: e.id, filename: e.filename, format: e.format, filter_label: e.filter_label,
      contact_count: e.contact_count, created_at: e.created_at,
    })),
  };
}

async function downloadExport(ctx: Ctx) {
  const e = await first<any>(ctx.env, "SELECT * FROM user_exports WHERE id = ? AND user_id = ?", ctx.params.export_id, ctx.user.id);
  if (!e) throw new HttpError(404, "Export not found");
  return new Response(e.file_content ?? "", {
    headers: {
      "Content-Type": e.format === "csv" ? "text/csv" : "application/json",
      "Content-Disposition": disposition(e.filename),
    },
  });
}

async function deleteExport(ctx: Ctx) {
  const e = await first<any>(ctx.env, "SELECT id FROM user_exports WHERE id = ? AND user_id = ?", ctx.params.export_id, ctx.user.id);
  if (!e) throw new HttpError(404, "Export not found");
  await run(ctx.env, "DELETE FROM user_exports WHERE id = ?", ctx.params.export_id);
  return { status: "success", deleted_id: ctx.params.export_id };
}

// ---- event files ----

async function uploadEventFile(ctx: Ctx) {
  const { bytes, name, mime } = await uploadedFile(ctx);
  const fileType = EVENT_FILE_TYPES[mime];
  if (!fileType) throw new HttpError(400, "File type not allowed. Supported: PDF, PPT, PPTX");
  if (bytes.byteLength === 0) throw new HttpError(400, "Empty file");
  if (bytes.byteLength > 20 * 1024 * 1024) throw new HttpError(400, `File too large. Max 20MB, got ${mb1(bytes.byteLength)}MB`);

  const original = name || "untitled";
  const id = uuid(), ts = now();
  await ctx.env.FILES.put(`file:${id}`, bytes);
  await run(
    ctx.env,
    `INSERT INTO event_files (id, filename, original_filename, file_type, mime_type, file_size, description, event_name, category, uploaded_by, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    id, safeName(original), original, fileType, mime, bytes.byteLength,
    q(ctx, "description") ?? "", q(ctx, "event_name") ?? "WHX Dubai 2026", q(ctx, "category") ?? "general",
    ctx.user.id, ts, ts,
  );
  return {
    status: "success", file_id: id, filename: safeName(original), file_size: bytes.byteLength,
    message: `File '${original}' uploaded successfully`,
  };
}

async function listEventFiles(ctx: Ctx) {
  const args: unknown[] = [q(ctx, "event_name") ?? "WHX Dubai 2026"];
  let sql = "SELECT * FROM event_files WHERE event_name = ? AND is_active = 1";
  const category = q(ctx, "category") ?? "";
  if (category) { sql += " AND category = ?"; args.push(category); }
  const files = await all<any>(ctx.env, `${sql} ORDER BY created_at DESC`, ...args);
  return {
    status: "success",
    total: files.length,
    files: files.map((f) => ({
      id: f.id, filename: f.original_filename, file_type: f.file_type, file_size: f.file_size,
      file_size_mb: mb2(f.file_size), description: f.description || "", category: f.category || "general",
      download_count: f.download_count || 0, created_at: f.created_at || null,
    })),
  };
}

// One download/delete pair for both file tables (same app.py behaviour, same shape).
type FileTable = "event_files" | "contact_files";

async function sendFile(ctx: Ctx, table: FileTable) {
  const id = ctx.params.file_id;
  const row = await first<any>(ctx.env, `SELECT * FROM ${table} WHERE id = ? AND is_active = 1`, id);
  if (!row) throw new HttpError(404, "File not found");
  if (table === "contact_files") await assertContactVisible(ctx, row.contact_id);
  const bytes = await ctx.env.FILES.get(`file:${id}`, "arrayBuffer");
  if (!bytes) throw new HttpError(404, "File not found");
  await run(ctx.env, `UPDATE ${table} SET download_count = download_count + 1 WHERE id = ?`, id);
  return new Response(bytes, {
    headers: {
      "Content-Type": row.mime_type,
      "Content-Disposition": disposition(row.original_filename),
      "Content-Length": String(row.file_size),
    },
  });
}

const deleteFile = (table: FileTable) => async (ctx: Ctx) => {
  const row = await first<any>(ctx.env, `SELECT id, original_filename FROM ${table} WHERE id = ?`, ctx.params.file_id);
  if (!row) throw new HttpError(404, "File not found");
  await run(ctx.env, `UPDATE ${table} SET is_active = 0 WHERE id = ?`, ctx.params.file_id);
  return { status: "success", message: `File '${row.original_filename}' deleted` };
};

// ---- voice notes ----

async function audioNote(ctx: Ctx) {
  const b = await ctx.body();
  const contact = await ownContact(ctx);
  if (!contact) throw new HttpError(404, "Contact not found");

  const d = new Date();
  const stamp = `${p2(d.getUTCMonth() + 1)}/${p2(d.getUTCDate())} ${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`;
  const transcript = typeof b.transcript === "string" ? b.transcript : "";
  const audio = typeof b.audio_base64 === "string" ? b.audio_base64 : "";
  const notes = pj<any[]>(contact.audio_notes, []);
  notes.push({ transcript, audio_base64: audio, timestamp: stamp });
  const text = transcript ? (contact.notes || "") + `\n[Voice Note ${stamp}] ` + transcript : contact.notes || "";

  await run(ctx.env, "UPDATE contacts SET audio_notes = ?, notes = ?, updated_at = ? WHERE id = ?", sj(notes), text, now(), contact.id);
  reindexLater(ctx.env, ctx.exec, [{ ...contact, notes: text }]); // app.py refreshed its FAISS summary here, best effort
  return { status: "success", message: "Audio note saved", total_audio_notes: notes.length };
}

async function transcribe(ctx: Ctx) {
  const { bytes } = await uploadedFile(ctx);
  if (bytes.byteLength > 25 * 1024 * 1024) throw new HttpError(413, "Audio file too large. Max 25MB.");
  try {
    // app.py used Gemini here; the Worker has no Gemini key, so whisper runs on Workers AI.
    const out: any = await ctx.env.AI.run("@cf/openai/whisper-large-v3-turbo" as any, { audio: b64(bytes) } as any);
    return { status: "success", transcript: String(out?.text ?? "").trim() };
  } catch (e) {
    throw new HttpError(500, `Transcription failed: ${String((e as Error)?.message || e)}`);
  }
}

// ---- card accepted webhook ----

async function contactAccepted(ctx: Ctx) {
  const contact = await ownContact(ctx);
  if (!contact) throw new HttpError(404, "Contact not found");
  // WEBHOOK_URL is empty in this deployment; app.py returned this same literal fallback.
  return {
    status: "success",
    message: "Contact acceptance recorded",
    webhook_sent: false,
    webhook_message: "WEBHOOK_URL not configured",
    contact: { id: contact.id, name: contact.name, email: contact.email },
  };
}

// ---- broadcasts ----

async function adminBroadcast(ctx: Ctx) {
  const message = q(ctx, "message");
  if (!message) throw new HttpError(422, "message must be at least 1 character");
  const priority = q(ctx, "priority") ?? "normal";
  const parsed = (q(ctx, "recipient_ids") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const recipients = parsed.length ? parsed : null;

  const id = uuid();
  await run(
    ctx.env,
    "INSERT INTO admin_broadcasts (id, admin_id, message, priority, recipient_ids, created_at) VALUES (?, ?, ?, ?, ?, ?)",
    id, ctx.user.id, message, priority === "urgent" ? "urgent" : "normal", recipients ? sj(recipients) : null, now(),
  );
  return {
    status: "success",
    broadcast_id: id,
    message: recipients ? `Broadcast sent to ${recipients.length} member(s)` : "Broadcast sent to team",
    recipient_count: recipients ? recipients.length : null,
  };
}

async function activeBroadcasts(ctx: Ctx) {
  const rows = await all<any>(
    ctx.env,
    `SELECT b.id, b.message, b.priority, b.recipient_ids, b.created_at, u.name AS admin_name
       FROM admin_broadcasts b JOIN users u ON u.id = b.admin_id
      WHERE b.is_active = 1 ORDER BY b.created_at DESC LIMIT 20`,
  );
  const dismissed = new Set(
    (await all<{ broadcast_id: string }>(ctx.env, "SELECT broadcast_id FROM broadcast_dismissals WHERE user_id = ?", ctx.user.id))
      .map((r) => r.broadcast_id),
  );
  const broadcasts: any[] = [];
  for (const r of rows) {
    const recipients = pj<string[] | null>(r.recipient_ids, null);
    if (recipients && recipients.length && !recipients.includes(ctx.user.id)) continue;
    if (dismissed.has(r.id)) continue;
    broadcasts.push({
      id: r.id, message: r.message, priority: r.priority, admin_name: r.admin_name,
      recipient_ids: recipients, created_at: r.created_at || "",
    });
  }
  return { status: "success", broadcasts };
}

async function dismissBroadcast(ctx: Ctx) {
  const found = await first<any>(ctx.env, "SELECT id FROM admin_broadcasts WHERE id = ?", ctx.params.broadcast_id);
  if (!found) throw new HttpError(404, "Broadcast not found");
  await run(ctx.env, "INSERT OR IGNORE INTO broadcast_dismissals (broadcast_id, user_id) VALUES (?, ?)", ctx.params.broadcast_id, ctx.user.id);
  return { status: "success", message: "Broadcast dismissed" };
}

// ---- contact documents ----

async function uploadContactFile(ctx: Ctx) {
  const { bytes, name, mime } = await uploadedFile(ctx);
  const fileType = CONTACT_FILE_TYPES[mime];
  if (!fileType) throw new HttpError(400, `File type not allowed: ${mime}. Allowed: PDF, PPT, PPTX, DOC, DOCX, PNG, JPG`);
  if (bytes.byteLength > 20 * 1024 * 1024) throw new HttpError(400, "File too large. Maximum 20MB.");

  const contact = await first<any>(ctx.env, "SELECT id, name FROM contacts WHERE id = ?", ctx.params.contact_id);
  if (!contact) throw new HttpError(404, "Contact not found");

  const original = name || "document";
  const category = q(ctx, "category") ?? "research";
  const id = uuid(), ts = now();
  await ctx.env.FILES.put(`file:${id}`, bytes);
  await run(
    ctx.env,
    `INSERT INTO contact_files (id, contact_id, filename, original_filename, file_type, mime_type, file_size, description, category, uploaded_by, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    id, ctx.params.contact_id, safeName(original), original, fileType, mime, bytes.byteLength,
    q(ctx, "description") ?? "", CONTACT_CATEGORIES.includes(category) ? category : "research",
    ctx.user.id, ts, ts,
  );
  return {
    status: "success", file_id: id, filename: original, file_size: bytes.byteLength, contact_name: contact.name,
  };
}

/** Contact files are visible to the contact's owner and to admins (app.py checked only the shared key). */
async function assertContactVisible(ctx: Ctx, contactId: string) {
  if (ctx.user.is_admin) return;
  const c = await first<any>(ctx.env, "SELECT id FROM contacts WHERE id = ? AND user_id = ?", contactId, ctx.user.id);
  if (!c) throw new HttpError(404, "Contact not found");
}

async function listContactFiles(ctx: Ctx) {
  await assertContactVisible(ctx, ctx.params.contact_id);
  const files = await all<any>(
    ctx.env,
    "SELECT * FROM contact_files WHERE contact_id = ? AND is_active = 1 ORDER BY created_at DESC",
    ctx.params.contact_id,
  );
  return {
    status: "success",
    total: files.length,
    files: files.map((f) => ({
      id: f.id, filename: f.original_filename, file_type: f.file_type, file_size: f.file_size,
      file_size_mb: mb2(f.file_size), description: f.description || "", category: f.category,
      download_count: f.download_count, created_at: f.created_at || "",
    })),
  };
}

export const routes: Route[] = [
  ["POST", "/exports/create", createExport],
  ["GET", "/exports/", listExports],
  ["GET", "/exports/:export_id/download", downloadExport],
  ["DELETE", "/exports/:export_id", deleteExport],
  ["POST", "/admin/files/upload", uploadEventFile, "admin"],
  ["GET", "/files/list", listEventFiles],
  ["GET", "/files/download/:file_id", (ctx) => sendFile(ctx, "event_files")],
  ["DELETE", "/admin/files/:file_id", deleteFile("event_files"), "admin"],
  ["POST", "/contact/:contact_id/audio_note", audioNote],
  ["POST", "/transcribe_audio/", transcribe],
  ["POST", "/contact/:contact_id/accepted", contactAccepted],
  ["POST", "/admin/broadcast", adminBroadcast, "admin"],
  ["GET", "/broadcasts/active", activeBroadcasts],
  ["POST", "/broadcasts/:broadcast_id/dismiss", dismissBroadcast],
  ["POST", "/admin/contact/:contact_id/files", uploadContactFile, "admin"],
  ["GET", "/contact/:contact_id/files", listContactFiles],
  ["GET", "/contact/files/download/:file_id", (ctx) => sendFile(ctx, "contact_files")],
  ["DELETE", "/admin/contact/files/:file_id", deleteFile("contact_files"), "admin"],
];
