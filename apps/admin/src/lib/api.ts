// Typed client for the admin console. Every call goes through apiFetch (admin token, refresh on
// 401, X-Admin-Console) and throws ApiError with the server's own message when it fails.
// Responses of the shapes the app shares (@es/types) are checked against the contract in development.
import { z } from 'zod';
import {
  addedExhibitorSchema, broadcastSchema, contactFileSchema, contactPageSchema, contactSchema, duplicateMatchesSchema, eventFileSchema,
  eventSummarySchema, exhibitorSchema, fileContentSchema, retrievedContactsSchema, workspaceDeletedContactsSchema, workspaceDuplicatePairSchema,
} from '@es/types';
import { apiFetch } from './auth';
import { checked, checkResponse } from './checkResponse';
import type {
  ActivityCalendar, ActivityItem, AdminUser, Broadcast, Contact, ContactFile, ContactStats, ContactPatch, Dashboard, WorkspaceDeletedContact, DeletedContactDetail, ListMember, Segments, SegmentDim, PlannerEvent, PlannerEventInput, BroadcastLog, DuplicateMatch, WorkspaceDuplicatePair,
  ConverseContact, ConverseReply, AdminEvent, EventDetail, EventFile, EventSummary, Exhibitor, BackupStatus, BackupResult, ContactList, Health, ModelInfo, PipelineList, PipelineRun, PipelineSettings,
  ScoreResult, WebhookResult, MyAccount, MyProfilePatch, AdminSalesProfile, AdminSalesPatch, CrmInfo, CrmJob, CrmKey, CrmTestResult,
} from './types';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
    if (Array.isArray(data?.detail) && data.detail[0]?.msg) return String(data.detail[0].msg);
  } catch {
    // not JSON
  }
  if (res.status === 403) return "Your account doesn't have admin access.";
  if (res.status >= 500) return `The server had a problem (${res.status}).`;
  return `Request failed (${res.status}).`;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await apiFetch(path, body !== undefined
      ? { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
      : { method });
  } catch (e) {
    throw new ApiError(e instanceof Error && e.message === 'Not signed in' ? e.message : "Can't reach the server.", 0);
  }
  if (!res.ok) throw new ApiError(await errorText(res), res.status);
  const type = res.headers.get('content-type') || '';
  return (type.includes('application/json') ? res.json() : res.text()) as Promise<T>;
}

const get = <T>(path: string) => request<T>('GET', path);
const post = <T>(path: string, body?: unknown) => request<T>('POST', path, body ?? {});
const patch = <T>(path: string, body: unknown) => request<T>('PATCH', path, body);
const put = <T>(path: string, body: unknown) => request<T>('PUT', path, body);
const del = <T>(path: string) => request<T>('DELETE', path);

function qs(params: Record<string, string | number | undefined | null>) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') q.append(k, String(v));
  });
  const s = q.toString();
  return s ? `?${s}` : '';
}

/** Save a file the browser downloaded (blob) under `filename`. */
export function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export function base64ToBlob(b64: string, mime: string) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(',', 2)[1] || '');
    r.onerror = () => reject(new Error('Could not read the file.'));
    r.readAsDataURL(file);
  });
}

export interface ContactQuery {
  search?: string | undefined;
  scoreBand?: 'A' | 'B' | 'C' | 'D' | 'U' | undefined;
  eventId?: string | undefined;
  userId?: string | undefined;
  sort?: 'recent' | 'score' | undefined;
  limit?: number | undefined;
  /** 1 = light rows for lists: no photo / avatar (hasPhoto says if there is one) and no score reasoning; GET /api/contacts/{id} has everything */
  lite?: 1 | undefined;
  /** only contacts created at / after this ISO time (e.g. the admin's local midnight for "today") */
  createdSince?: string | undefined;
  createdBefore?: string | undefined;
}

/** GET /api/contacts/stats: same filters as the list (the level is ignored). */
export type ContactStatsQuery = Pick<ContactQuery, 'search' | 'eventId' | 'userId' | 'createdSince' | 'createdBefore'>;

// Safety cap when loading the whole contacts list (the grid paginates client-side).
export const MAX_CONTACT_ROWS = 20000;
/** Pipeline runs per page on /pipelines. */
export const PIPELINE_PAGE = 50;
/** Most activity items the server returns for a date range (GET /api/admin/activity?start&end). */
export const MAX_ACTIVITY_ITEMS = 5000;

export const api = {
  // overview
  // the offset goes under both names (tzOffset is the documented one; tz_offset what older servers read)
  dashboard: (days = 7) => {
    const tz = new Date().getTimezoneOffset();
    return get<Dashboard>(`/api/admin/dashboard${qs({ days, tzOffset: tz, tz_offset: tz })}`);
  },
  activity: (limit = 60, range?: { start: string; end: string }) => get<ActivityItem[]>(`/api/admin/activity${qs({ limit, start: range?.start, end: range?.end })}`),
  /** how much happened on each local day of a range (≤ 45 days); tzOffset = Date#getTimezoneOffset() */
  activityCalendar: (start: string, end: string, tzOffset: number) =>
    get<ActivityCalendar>(`/api/admin/activity/calendar${qs({ start, end, tzOffset })}`),
  health: () => get<Health>('/api/health'),

  // contacts
  /** A limit of 1,000 (the server's page cap) means "the whole list": the next pages are followed via `cursor`. */
  contacts: async (q: ContactQuery = {}) => {
    const params = { limit: 500, ...q };
    const first = await get<{ contacts: Contact[]; cursor?: string | null }>(`/api/contacts${qs(params)}`).then(checked(contactPageSchema, 'GET /api/contacts'));
    let contacts = first.contacts;
    let cursor = (params.limit ?? 0) >= 1000 ? first.cursor : null;
    while (cursor && contacts.length < MAX_CONTACT_ROWS) {
      const page = await get<{ contacts: Contact[]; cursor?: string | null }>(`/api/contacts${qs({ ...params, cursor })}`).then(checked(contactPageSchema, 'GET /api/contacts'));
      contacts = contacts.concat(page.contacts);
      cursor = page.cursor;
    }
    return { contacts };
  },
  /** level counts (admin score: hot ≥85, warm 70–84, cool 50–69, cold <50, unscored) for the list's filters, counted on the server */
  contactStats: (f: ContactStatsQuery = {}) => get<ContactStats>(`/api/contacts/stats${qs(f)}`),
  contact: (id: string) => get<Contact>(`/api/contacts/${id}`).then(checked(contactSchema, 'GET /api/contacts/{id}')),
  createContact: (body: Record<string, unknown>) => post<Contact>('/api/contacts', body).then(checked(contactSchema, 'POST /api/contacts')),
  /** Edit any field (name, company, title, email, phone, LinkedIn, event, notes, photo). The server validates LinkedIn links. */
  updateContact: (id: string, fields: ContactPatch) =>
    patch<Contact>(`/api/contacts/${id}`, { baseVersion: 1, patch: fields }).then(checked(contactSchema, 'PATCH /api/contacts/{id}')),
  deleteContact: (id: string) => del<{ status: string }>(`/api/contacts/${id}`),
  batchDelete: (contactIds: string[]) => post<{ count: number; message: string }>('/api/admin/contacts/batch_delete', { contactIds }),
  scoreContact: (id: string, force = false) => post<ScoreResult>(`/api/contact/${id}/score${force ? '?force=true' : ''}`),
  /** the admin console's own score (never the rep's) */
  scoreAll: () => post<{ scoredCount: number; errors: number; pending: number }>('/api/contacts/score_all'),
  saveAdminNote: (id: string, notes: string) => put<{ adminNotes: string }>(`/api/admin/contact/${id}/note`, { notes }),
  contactQr: (id: string) => get<{ qrDataUrl: string; vcard: string; name: string }>(`/api/contact/${id}/qr`),
  /** look-alikes anywhere in the workspace: the owner's own contacts and other reps' */
  contactDuplicates: (id: string) =>
    get<{ matches: DuplicateMatch[] }>(`/api/admin/contacts/${id}/duplicates`).then(checked(duplicateMatchesSchema, 'GET /api/admin/contacts/{id}/duplicates')),
  contactFiles: (id: string) => get<ContactFile[]>(`/api/contact/${id}/files`).then(checked(z.array(contactFileSchema), 'GET /api/contact/{id}/files')),
  uploadContactFile: (id: string, body: { filename: string; contentBase64: string; description?: string | undefined }) =>
    post<ContactFile>(`/api/contact/${id}/files`, body).then(checked(contactFileSchema, 'POST /api/contact/{id}/files')),
  deleteContactFile: (fileId: string) => del<{ status: string }>(`/api/contact/files/${fileId}`),
  downloadContactFile: (fileId: string) =>
    get<{ filename: string; mimeType: string; contentBase64: string }>(`/api/contact/files/${fileId}/download`).then(checked(fileContentSchema, 'GET /api/contact/files/{id}/download')),
  /** contacts CSV built on the server (no row cap): everything, or narrowed by event / rep / min score / list */
  exportAll: async (f: ExportFilters & { columns?: string[] } = {}): Promise<{ blob: Blob; filename: string; rows: number | null }> => {
    if (f.columns && !f.columns.length) throw new ApiError('Select at least one column.', 400);
    const res = await apiFetch(`/api/admin/export${qs({ ...f, columns: f.columns?.join(',') })}`);
    if (!res.ok) throw new ApiError(await errorText(res), res.status);
    const cd = res.headers.get('content-disposition') || '';
    const filename = /filename="([^"]+)"/.exec(cd)?.[1] || 'eventscout-contacts.csv';
    const rows = res.headers.get('x-row-count');
    return { blob: await res.blob(), filename, rows: rows == null ? null : Number(rows) };
  },

  exportPresets: {
    get: () => get<ExportPresetConfig>('/api/admin/export-presets'),
    put: (presets: ExportPreset[]) => put<ExportPresetConfig>('/api/admin/export-presets', { presets }),
  },

  // recently deleted (every rep's; restorable for 30 days, restored to the owner)
  deletedContacts: () =>
    get<{ contacts: WorkspaceDeletedContact[]; retentionDays: number }>('/api/admin/contacts/deleted').then(checked(workspaceDeletedContactsSchema, 'GET /api/admin/contacts/deleted')),
  deletedContact: (id: string) => get<DeletedContactDetail>(`/api/admin/contacts/deleted/${id}`),
  restoreDeleted: (id: string, ownerId: string) =>
    post<Contact>(`/api/contacts/deleted/${id}/restore${qs({ userId: ownerId })}`).then(checked(contactSchema, 'POST /api/contacts/deleted/{id}/restore')),

  // duplicates
  duplicates: () => get<WorkspaceDuplicatePair[]>('/api/admin/contacts/duplicates').then(checked(z.array(workspaceDuplicatePairSchema), 'GET /api/admin/contacts/duplicates')),
  merge: (primaryId: string, duplicateId: string) => post<{ message: string }>('/api/admin/contacts/merge', { primaryId, duplicateId }),
  notDuplicate: (primaryId: string, duplicateId: string) => post<{ status: string }>('/api/admin/contacts/duplicates/dismiss', { primaryId, duplicateId }),

  // lists
  lists: () => get<ContactList[]>('/api/admin/contact_lists'),
  createList: (name: string) => post<ContactList>('/api/admin/contact_lists', { name }),
  renameList: (id: string, name: string) => patch<ContactList>(`/api/admin/contact_lists/${id}`, { name }),
  deleteList: (id: string) => del<{ status: string }>(`/api/admin/contact_lists/${id}`),
  listMembers: (id: string) => get<ListMember[]>(`/api/admin/contact_lists/${id}/contacts`),
  /** smart lists: every contact grouped by industry / seniority / role / company */
  segments: () => get<Segments>('/api/admin/segments'),
  segmentContacts: (dim: SegmentDim, key: string) => get<Contact[]>(`/api/admin/segments/${dim}/contacts${qs({ key })}`),
  addToList: (id: string, contactIds: string[]) => post<{ count: number; alreadyInList: number; notFound: number }>(`/api/admin/contact_lists/${id}/contacts`, { contactIds }),
  removeFromList: (id: string, contactId: string) => del<{ status: string }>(`/api/admin/contact_lists/${id}/contacts/${contactId}`),

  // pipelines
  /** one page of runs, newest first; `total` = runs matching the status filter */
  pipelines: (status?: string, offset = 0, limit = PIPELINE_PAGE) => get<PipelineList>(`/api/admin/pipelines${qs({ status, offset, limit })}`),
  pipelineFor: (contactId: string) => get<PipelineRun>(`/api/contact/${contactId}/pipeline`),
  runPipeline: (contactId: string) => post<{ status: string; message: string }>(`/api/contact/${contactId}/pipeline/run`),
  retryPipeline: (contactId: string) => post<{ status: string; message: string }>(`/api/contact/${contactId}/pipeline/retry`),
  /** the auto-pipeline switch (runs after every card / QR scan; in memory until the server restarts) */
  setPipelineSettings: (body: { autoPipelineEnabled: boolean }) => post<PipelineSettings>('/api/admin/pipelines/settings', body),

  // users
  users: () => get<AdminUser[]>('/api/admin/users'),
  createUser: (body: { name: string; email: string; password: string; isAdmin: boolean }) => post<AdminUser>('/api/admin/users', body),
  /** rename, change role, or set a new password (signs them out everywhere) */
  updateUser: (id: string, body: { name?: string; isAdmin?: boolean; password?: string }) => patch<AdminUser>(`/api/admin/users/${id}`, body),
  deleteUser: (id: string) => del<{ message: string }>(`/api/admin/users/${id}`),
  mergeUsers: (sourceUserId: string, targetUserId: string) =>
    post<{ message: string; movedContacts: number; linkedinMoved: boolean }>('/api/admin/users/merge', { sourceUserId, targetUserId }),

  // announcements
  broadcasts: () => get<Broadcast[]>('/api/broadcasts/active').then(checked(z.array(broadcastSchema), 'GET /api/broadcasts/active')),
  sendBroadcast: (message: string, priority: 'normal' | 'urgent', recipientIds?: string[], eventId?: string) =>
    post<{ id: string }>('/api/admin/broadcast', { message, priority, recipientIds: recipientIds?.length ? recipientIds : null, eventId: eventId || null }),
  /** every announcement sent (live and ended), newest first */
  broadcastHistory: (range?: { start: string; end: string }, limit?: number) =>
    get<BroadcastLog[]>(`/api/admin/broadcasts/history${qs({ start: range?.start, end: range?.end, limit })}`),
  endBroadcast: (id: string) => post<{ status: string }>(`/api/broadcasts/${id}/dismiss`),

  // event planner (add events, details, floor plans, team, announce)
  plannerEvents: () => get<PlannerEvent[]>('/api/admin/planner/events'),
  createPlannerEvent: (body: PlannerEventInput) => post<PlannerEvent>('/api/admin/planner/events', body),
  updatePlannerEvent: (id: string, body: PlannerEventInput) => patch<PlannerEvent>(`/api/admin/planner/events/${id}`, body),
  deletePlannerEvent: (id: string) => del<{ status: string }>(`/api/admin/planner/events/${id}`),
  setPlannerTeam: (id: string, members: { userId: string; role: 'lead' | 'member' }[]) => put<PlannerEvent>(`/api/admin/planner/events/${id}/team`, { members }),
  announcePlannerEvent: (id: string, body: { message: string; priority: 'normal' | 'urgent'; audience: 'team' | 'everyone' }) =>
    post<{ id: string; recipients: number | null }>(`/api/admin/planner/events/${id}/announce`, body),
  /** the AI reads an event's details out of text (nothing saved; unknown fields come back empty) */
  extractPlannerEvent: (text: string) => post<PlannerEventInput>('/api/admin/planner/extract', { text }),
  plannerDiscovery: () => get<{ connected: boolean; provider: string | null; message: string }>('/api/admin/planner/discovery'),

  // events & exhibitors
  events: () => get<EventSummary[]>('/api/events').then(checked(z.array(eventSummarySchema), 'GET /api/events')),
  exhibitors: (event?: string) => get<Exhibitor[]>(`/api/exhibitors${qs({ event, limit: 2000 })}`).then(checked(z.array(exhibitorSchema), 'GET /api/exhibitors')),
  addExhibitor: (body: { event: string; name: string; booth?: string; hall?: string; category?: string; website?: string; description?: string }) =>
    post<Exhibitor & { existed: boolean }>('/api/exhibitors', body).then(checked(addedExhibitorSchema, 'POST /api/exhibitors')),
  deleteExhibitor: (id: string) => del<{ status: string }>(`/api/exhibitors/${id}`),
  updateExhibitor: (id: string, body: { name?: string; booth?: string; hall?: string; category?: string; website?: string; description?: string }) =>
    patch<Exhibitor>(`/api/exhibitors/${id}`, body).then(checked(exhibitorSchema, 'PATCH /api/exhibitors/{id}')),
  /** every event: booth directory ∪ contacts' "met at" ∪ reps' event lists ∪ event files, with counts */
  adminEvents: () => get<AdminEvent[]>('/api/admin/events'),
  /** one event: booths (visits, contacts met), reps, captures per day, files (floor plans flagged) */
  eventDetail: (event: string) => get<EventDetail>(`/api/admin/events/detail${qs({ event })}`),
  importExhibitors: (exhibitors: unknown[], event: string | undefined, replace: boolean) =>
    post<{ count: number; skipped: number; replaced: number; events: Record<string, number> }>('/api/admin/import_exhibitors', { exhibitors, event, replace }),

  // files
  files: () => get<EventFile[]>('/api/files/list').then(checked(z.array(eventFileSchema), 'GET /api/files/list')),
  uploadFile: (body: { filename: string; contentBase64: string; mimeType?: string | undefined; category: string; description: string; event?: string | undefined }) =>
    post<{ id: string; filename: string }>('/api/admin/files/upload', body),
  downloadFile: (id: string) =>
    get<{ filename: string; mimeType: string; contentBase64: string }>(`/api/files/${id}/download`).then(checked(fileContentSchema, 'GET /api/files/{id}/download')),
  deleteFile: (id: string) => del<{ status: string }>(`/api/admin/files/${id}`),

  // AI
  models: () => get<ModelInfo[]>('/api/models'),

  // my account (the signed-in admin): the same profile the app's Me screen edits, and the password
  me: () => get<MyAccount>('/api/admin/me'),
  /** name, title, company, website, LinkedIn, phone, email signature; avatar: data URI to set, '' to remove */
  updateMe: (fields: MyProfilePatch) => patch<MyAccount>('/api/admin/me', fields),
  /** the console's own products & sales approach (its own table; no rep's profile is touched) */
  salesProfile: () => get<AdminSalesProfile>('/api/admin/sales-profile'),
  /** only the fields sent change */
  updateSalesProfile: (fields: AdminSalesPatch) => patch<AdminSalesProfile>('/api/admin/sales-profile', fields),
  /** needs the current password (unless the account has none yet); answers with fresh tokens for this console */
  changeMyPassword: (newPassword: string, currentPassword?: string) =>
    post<{ status: string; message: string; token: string; refreshToken: string }>('/api/admin/me/password', { newPassword, ...(currentPassword ? { currentPassword } : {}) }),
  converse: (query: string, conversationHistory: { role: string; content: string }[], uiFormat?: 'openui' | 'text') =>
    post<ConverseReply>('/api/converse', { query, conversationHistory, uiFormat }),
  /**
   * The copilot, streamed (POST /api/converse/stream, Server-Sent Events): `onDelta` gets the answer
   * as it's written; resolves with the full answer. Throws ApiError with the server's message.
   */
  /** The copilot, streamed. With `files` (images / PDF / text, 3 × 10 MB) it goes as multipart to /api/converse/stream/files. */
  converseStream: async (body: { query: string; conversationHistory: { role: string; content: string }[]; uiFormat?: 'openui' | 'text'; files?: File[] | undefined },
    on: { onMeta?: (m: { format: string; contacts: ConverseContact[] }) => void; onModel?: (model: string) => void; onDelta?: (text: string) => void },
    signal?: AbortSignal): Promise<ConverseReply> => {
    let res: Response;
    const { files, ...json } = body;
    let init: RequestInit;
    if (files?.length) {
      const form = new FormData();
      form.append('query', json.query);
      form.append('conversationHistory', JSON.stringify(json.conversationHistory));
      if (json.uiFormat) form.append('uiFormat', json.uiFormat);
      files.forEach((f) => form.append('files', f, f.name));
      init = { method: 'POST', body: form };
    } else {
      init = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(json) };
    }
    try {
      res = await apiFetch(files?.length ? '/api/converse/stream/files' : '/api/converse/stream', { ...init, ...(signal ? { signal } : {}) });
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') throw e;
      throw new ApiError(e instanceof Error && e.message === 'Not signed in' ? e.message : "Can't reach the server.", 0);
    }
    if (!res.ok || !res.body) throw new ApiError(await errorText(res), res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let text = '';
    let model = '';
    let format = 'text';
    let contacts: ConverseContact[] = [];
    const toContacts = (list: any[]): ConverseContact[] => (list || []).filter((c) => c?.id).map((c) => ({
      id: String(c.id), name: c.name && c.name !== 'N/A' ? c.name : '', company: c.company_name && c.company_name !== 'N/A' ? c.company_name : null,
      score: c.lead_score ?? null, temperature: c.lead_temperature ?? null,
    }));
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut: number;
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        let event = 'message';
        const data: string[] = [];
        block.split('\n').forEach((line) => {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
        });
        if (!data.length) continue; // padding comment
        const payload = JSON.parse(data.join('\n'));
        if (event === 'meta') {
          format = payload.format || 'text';
          contacts = toContacts(checkResponse(retrievedContactsSchema, payload.retrieved_contacts, 'SSE meta /api/converse/stream (retrieved_contacts)'));
          on.onMeta?.({ format, contacts });
        }
        else if (event === 'model') { model = payload.model || ''; on.onModel?.(model); }
        else if (event === 'delta') { text += payload.text || ''; on.onDelta?.(text); }
        else if (event === 'done') {
          const quick = (payload.ui_components || []).find((c: any) => c?.type === 'quick_replies')?.data;
          return { reply: payload.response ?? text, format: payload.format || format, modelUsed: payload.model || model,
            contacts: payload.retrieved_contacts
              ? toContacts(checkResponse(retrievedContactsSchema, payload.retrieved_contacts, 'SSE done /api/converse/stream (retrieved_contacts)'))
              : contacts,
            // drop suggestions built from a missing name / company ("Research N/A")
            quickReplies: Array.isArray(quick) ? quick.filter((x: unknown) => typeof x === 'string' && x.trim() && !/\bN\/A\b|\s$|for\s*$/.test(x)).slice(0, 4) : [] };
        } else if (event === 'error') throw new ApiError(payload.detail || "The assistant couldn't answer.", 503);
      }
    }
    if (!text) throw new ApiError('The answer broke off. Try again.', 0);
    return { reply: text, format, modelUsed: model, contacts };
  },
  /** A spoken question (browser recording) → its text (POST /api/transcribe, the app's audio model). */
  transcribe: async (audio: Blob, name: string): Promise<string> => {
    const form = new FormData();
    form.append('file', audio, name);
    let res: Response;
    try {
      res = await apiFetch('/api/transcribe', { method: 'POST', body: form });
    } catch (e) {
      throw new ApiError(e instanceof Error && e.message === 'Not signed in' ? e.message : "Can't reach the server.", 0);
    }
    if (!res.ok) throw new ApiError(await errorText(res), res.status);
    return String((await res.json()).transcript || '');
  },

  // system
  backupStatus: () => get<BackupStatus>('/api/admin/backup/status'),
  backupNow: () => post<BackupResult>('/api/admin/backup'),
  /** one SQLite snapshot from backend/backups/ as a file */
  downloadSnapshot: async (file: string): Promise<Blob> => {
    let res: Response;
    try {
      res = await apiFetch(`/api/admin/backup/snapshots/${encodeURIComponent(file)}`);
    } catch (e) {
      throw new ApiError(e instanceof Error && e.message === 'Not signed in' ? e.message : "Can't reach the server.", 0);
    }
    if (!res.ok) throw new ApiError(await errorText(res), res.status);
    return res.blob();
  },
  testWebhook: () => post<WebhookResult>('/api/admin/webhook/test'),

  // CRM (Exports → Send to CRM); same filters as exportAll
  crms: () => get<{ crms: CrmInfo[] }>('/api/admin/crm'),
  testCrm: (crm: CrmKey) => post<CrmTestResult>(`/api/admin/crm/${crm}/test`),
  pushToCrm: (crm: CrmKey, f: ExportFilters) => post<CrmJob>(`/api/admin/crm/${crm}/push`, f),
  crmJob: (id: string) => get<CrmJob>(`/api/admin/crm/jobs/${id}`),
  cancelCrmJob: (id: string) => post<CrmJob>(`/api/admin/crm/jobs/${id}/cancel`),
  /** CRM-ready CSV (that CRM's import column names); works without credentials */
  crmCsv: async (crm: CrmKey, f: ExportFilters = {}): Promise<{ blob: Blob; filename: string; rows: number | null }> => {
    const res = await apiFetch(`/api/admin/crm/${crm}/csv${qs(f)}`);
    if (!res.ok) throw new ApiError(await errorText(res), res.status);
    const cd = res.headers.get('content-disposition') || '';
    const filename = /filename="([^"]+)"/.exec(cd)?.[1] || `eventscout-${crm}.csv`;
    const rows = res.headers.get('x-row-count');
    return { blob: await res.blob(), filename, rows: rows == null ? null : Number(rows) };
  },
};

export type ExportFilters = { event?: string | undefined; userId?: string | undefined; minScore?: number | undefined; listId?: string | undefined };
export type ExportPreset = { name: string; columns: string[]; filters: ExportFilters; builtin?: boolean };
export type ExportPresetConfig = { columns: string[]; default: string[]; presets: ExportPreset[] };
