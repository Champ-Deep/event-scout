import type { Contact } from '@es/types';
import { asApiError, isNetworkError, type ApiError } from '../../lib/apiError';
import { CONFLICT_FIELDS, uuidv4, type ConflictField, type OutboxItem } from '../../sync/core';

/**
 * The contact repository: the ONE way screens read, list, write and delete contacts.
 *
 * It owns the online → offline fallback, every write to the phone's contact cache, the outbox
 * (changes saved offline, uploaded later by the sync engine), cursor paging, the cache-owner check
 * (offline data belongs to one account) and the conflict `base` sent with edits.
 *
 * Rule for every cache write of a server copy: a contact with a change still queued on this phone
 * keeps the phone's version (the server copy is older than what the user did here).
 *
 * Pure module: the backend and the local store are adapters (./index.ts wires the real ones).
 */

// ─── adapters ───────────────────────────────────────────────────────────────

export interface ContactListParams {
  search?: string | undefined;
  stage?: string | undefined;
  scoreBand?: string | undefined;
  eventId?: string | undefined;
  userId?: string | undefined;
  limit?: number | undefined;
  cursor?: string | undefined;
  /** 'score' = best lead score first (unscored last); default = newest first */
  sort?: 'recent' | 'score' | undefined;
}

export interface ContactRemote {
  list(params: ContactListParams): Promise<{ contacts: Contact[]; cursor: string | null } | null | undefined>;
  get(id: string): Promise<Contact>;
  create(input: NewContactInput & { id: string }): Promise<Contact>;
  update(id: string, patch: ContactPatch, opts: { base?: Record<string, unknown>; force?: boolean }): Promise<Contact>;
  /** moves it to Recently deleted */
  delete(id: string): Promise<{ daysLeft?: number } | null | undefined>;
  /** removes it from Recently deleted for good */
  purge(id: string): Promise<unknown>;
  restore(id: string): Promise<Contact>;
  duplicates(id: string): Promise<{ matches?: any[] } | null | undefined>;
  /** save a picked LinkedIn profile (and its photo) on the contact */
  attachLinkedIn(id: string, pick: LinkedInPick): Promise<LinkedInAttachResult>;
}

export interface LinkedInPick {
  url: string;
  photoUrl?: string | null;
  usePhoto?: boolean;
}

export interface LinkedInAttachResult {
  contact: Contact;
  photoAdded: boolean;
  photoStatus?: 'added' | 'limited' | 'none';
  photoSource?: 'search' | 'linkedin' | 'brave' | 'gravatar' | null;
}

export interface ContactLocal {
  /** clear another account's offline data first (if a different account signed in) */
  ensureOwner(): Promise<unknown>;
  signedInUser(): Promise<string | null>;
  get(id: string): Promise<Contact | null>;
  /** write a row; false when the local DB is unavailable */
  put(contact: any): Promise<boolean>;
  /** write a row unless the lists already saved it unchanged */
  putIfChanged(contact: any): Promise<boolean>;
  remove(id: string): Promise<void>;
  /** cached contacts for a list (offline fallback) */
  query(params: ContactListParams): Promise<Contact[]>;
  /** contact ids with a change still waiting to upload */
  pendingIds(): Promise<string[]>;
  queued(): Promise<OutboxItem[]>;
  hasPendingCreate(id: string): Promise<boolean>;
  enqueue(item: OutboxItem): Promise<void>;
  dropQueued(entityId: string): Promise<void>;
}

// ─── shapes ─────────────────────────────────────────────────────────────────

// Fields the edit screen can change. `photo` is a base64 / data-URI image, or null to remove it.
export interface ContactPatch {
  firstName?: string;
  lastName?: string | null;
  company?: string | null;
  title?: string | null;
  email?: string | null;
  phone?: string | null;
  linkedin?: string | null;
  eventId?: string | null;
  notes?: string | null;
  photo?: string | null;
}

export interface NewContactInput {
  /** the contact's id, made on the phone (default: a new uuid) */
  id?: string;
  firstName: string;
  lastName?: string | null;
  company?: string | null;
  title?: string | null;
  email?: string | null;
  phone?: string | null;
  notes?: string | null;
  eventId?: string | null;
  source?: string;
  photo?: string | null;
}

export type Pending<T> = T & { pendingSync?: boolean };

export interface ListQuery extends ContactListParams {
  /** follow the cursor until every page is loaded (`limit` = page size), up to `cap` contacts */
  allPages?: boolean | undefined;
  cap?: number | undefined;
}

export interface ListResult {
  contacts: Contact[];
  cursor: string | null;
  /** set when the server couldn't be reached and `contacts` is the phone's cached copy */
  error: ApiError | Error | null;
  fromCache: boolean;
}

// ─── contact change events ──────────────────────────────────────────────────
// Lets the list show "Moved to Recently deleted · Undo" after a delete made on another screen,
// and drop a discarded scan.
export type ContactEvent =
  | { type: 'deleted'; id: string; name: string; daysLeft: number }
  | { type: 'restored'; id: string; name: string }
  | { type: 'discarded'; id: string };

export function createEventBus<E>() {
  const listeners = new Set<(e: E) => void>();
  return {
    emit(e: E) { listeners.forEach((l) => l(e)); },
    subscribe(listener: (e: E) => void) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
  };
}

// ─── edit conflicts ─────────────────────────────────────────────────────────

export { CONFLICT_FIELDS, type ConflictField };

/** For `updateContact(..., { base })`: the patched fields' values as they were when the edit started. */
export function editBase(original: Partial<Contact> | null | undefined, patch: ContactPatch): Partial<Record<ConflictField, unknown>> {
  const base: Partial<Record<ConflictField, unknown>> = {};
  if (!original) return base;
  for (const f of CONFLICT_FIELDS) {
    if (f in patch) base[f] = (original as any)[f] ?? null;
  }
  return base;
}

export interface FieldConflict {
  field: ConflictField | string;
  base: unknown;
  server: unknown;
  mine: unknown;
}

export interface EditConflict {
  message: string;
  conflicts: FieldConflict[];
  /** the contact as it is on the server now */
  contact: Contact | null;
}

/** The 409 "changed somewhere else" answer of updateContact, or null for any other error. */
export function getEditConflict(e: unknown): EditConflict | null {
  const err = asApiError(e);
  if (err?.kind !== 'conflict') return null;
  const detail: any = err.detail;
  return {
    message: err.displayMessage,
    conflicts: err.conflicts,
    contact: detail?.contact ?? null,
  };
}

// ─── the repository ─────────────────────────────────────────────────────────

const DEFAULT_CAP = 5000;
const DEFAULT_DAYS_LEFT = 30;

export function createContactRepository(deps: {
  remote: ContactRemote;
  local: ContactLocal;
  /** told when something was queued offline (background sync starts) */
  onQueued?: () => void;
  now?: () => string;
}) {
  const { remote, local } = deps;
  const now = deps.now ?? (() => new Date().toISOString());
  const events = createEventBus<ContactEvent>();
  const background = new Set<Promise<unknown>>();

  /** Cache writes that don't hold up the screen (tests wait for them with flush()). */
  const inBackground = (p: Promise<unknown>) => {
    const tracked = p.catch(() => {}).finally(() => { background.delete(tracked); });
    background.add(tracked);
  };

  const pendingIds = async (): Promise<Set<string> | null> => {
    try { return new Set((await local.pendingIds()).map(String)); } catch { return null; }
  };

  const hasPendingCreate = async (id: string) => {
    try { return await local.hasPendingCreate(id); } catch { return false; }
  };

  /**
   * Save server copies to the phone — except contacts with a change still queued here (the
   * phone's version wins until it's uploaded), and nothing at all if the account changed since
   * the copies were fetched (`fetchedFor`).
   */
  async function saveServerCopies(list: any[], fetchedFor: string | null, opts: { memo?: boolean } = {}): Promise<number> {
    if (!Array.isArray(list) || !list.length) return 0;
    if (!fetchedFor || (await local.signedInUser()) !== fetchedFor) return 0;
    await local.ensureOwner();
    const pending = await pendingIds();
    if (!pending) return 0; // local DB unavailable (web)
    let saved = 0;
    for (const c of list) {
      if (!c?.id || pending.has(String(c.id))) continue;
      if (await (opts.memo ? local.putIfChanged(c) : local.put(c))) saved += 1;
    }
    return saved;
  }

  async function readCached(id: string): Promise<Contact | null> {
    try {
      await local.ensureOwner();
      return (await local.get(id)) ?? null;
    } catch {
      return null; // local cache unavailable
    }
  }

  async function enqueue(entityId: string, action: 'create' | 'update' | 'delete', payload: Record<string, unknown>) {
    await local.enqueue({
      id: uuidv4(),
      entityType: 'contact',
      entityId,
      action,
      baseVersion: 0,
      payload: JSON.stringify(payload),
      createdAt: now(),
      attempts: 0,
      lastError: null,
    });
    deps.onQueued?.();
  }

  /** Save a new contact on the phone and queue it for upload. */
  async function queueCreate(input: NewContactInput, id: string): Promise<Pending<Contact>> {
    const at = now();
    const contact: any = {
      id,
      userId: null,
      eventId: input.eventId || '',
      firstName: input.firstName || null,
      lastName: input.lastName ?? null,
      company: input.company ?? null,
      title: input.title ?? null,
      email: input.email ?? null,
      phone: input.phone ?? null,
      notes: input.notes ?? null,
      score: null,
      scoreBand: null,
      source: input.source === 'scan' ? 'scan' : 'manual',
      stage: 'new',
      version: 1,
      createdAt: at,
      updatedAt: at,
      scannedAt: at,
      photoUrl: input.photo || null,
      avatarUrl: input.photo || null,
      pendingSync: true,
    };
    await local.ensureOwner();
    await local.put(contact);
    const { id: _id, ...fields } = input;
    await enqueue(id, 'create', { ...fields, source: contact.source });
    return contact as Pending<Contact>;
  }

  /**
   * `_base` for a queued edit: what each changed field was before this phone changed it, so the server
   * can tell an edit made elsewhere in the meantime (it then keeps the newer value, or merges notes).
   * - A field no earlier queued edit touched: the value before the first offline edit — what the
   *   screen started from (`screenBase`), else the phone's copy.
   * - A field an earlier queued edit already changed: that edit's value (the phone's copy now). The
   *   server applies queued edits in order, so that is what it holds when this one arrives; using
   *   the older value would flag this phone's own earlier edit as a conflict (and repeat merged notes).
   * Nothing for a contact created offline that hasn't uploaded yet (nobody else can have changed it).
   */
  async function offlineBase(
    id: string, current: any, patch: Record<string, unknown>, screenBase?: Record<string, unknown>,
  ): Promise<Record<string, unknown> | null> {
    const fields = Object.keys(patch).filter((f) => (CONFLICT_FIELDS as readonly string[]).includes(f));
    if (!fields.length) return null;
    let earlier: Set<string> = new Set();
    try {
      const queued = (await local.queued()).filter((r) => String(r.entityId) === id);
      if (queued.some((r) => r.action === 'create')) return null;
      for (const r of queued) {
        if (r.action !== 'update') continue;
        try {
          const p = typeof r.payload === 'string' ? JSON.parse(r.payload) : r.payload;
          Object.keys(p || {}).forEach((k) => earlier.add(k));
        } catch { /* unreadable payload */ }
      }
    } catch {
      earlier = new Set();
    }
    const base: Record<string, unknown> = {};
    for (const f of fields) {
      const cached = current?.[f] ?? null;
      base[f] = !earlier.has(f) && screenBase && f in screenBase ? (screenBase[f] ?? null) : cached;
    }
    return base;
  }

  /** Apply an edit locally and queue it. null (nothing queued) when the phone has no copy. */
  async function queueUpdate(
    id: string, patch: Record<string, unknown>, opts: { base?: Record<string, unknown> | undefined; force?: boolean | undefined },
  ): Promise<Pending<Contact> | null> {
    const current: any = await readCached(id);
    if (!current) return null;
    // force ("Keep mine"): no _base, so the server saves it as is.
    const _base = opts.force ? null : await offlineBase(id, current, patch, opts.base);
    const next = { ...current, ...patch, updatedAt: now() };
    await local.put(next);
    await enqueue(id, 'update', _base ? { ...patch, _base } : patch);
    return { ...next, pendingSync: true } as Pending<Contact>;
  }

  /** Remove locally and queue the delete. A contact created offline and never uploaded just
   *  disappears: its queued changes are dropped. */
  async function queueDelete(id: string): Promise<void> {
    await local.ensureOwner();
    try { await local.remove(id); } catch { /* */ }
    if (await hasPendingCreate(id)) {
      try { await local.dropQueued(id); } catch { /* */ }
      deps.onQueued?.(); // the queue changed: screens refresh their counts
      return;
    }
    await enqueue(id, 'delete', {});
  }

  const repo = {
    events,

    /** Wait for cache writes still running in the background (tests). */
    async flush(): Promise<void> {
      while (background.size) await Promise.all([...background]);
    },

    /** null = the contact doesn't exist (or isn't yours). Throws when the server can't be
     *  reached and the device has no cached copy, so screens can say "offline", not "deleted". */
    async getContact(id: string): Promise<Contact | null> {
      const fetchedFor = await local.signedInUser().catch(() => null);
      try {
        // Try the server first (and keep the phone's copy fresh for offline use)
        const c = await remote.get(id);
        inBackground(saveServerCopies([c], fetchedFor));
        return c;
      } catch (error: any) {
        // A 404 means it doesn't exist (or isn't yours) — don't resurrect a stale cached copy…
        // …unless it was created offline and simply hasn't been uploaded yet.
        if (asApiError(error)?.kind === 'not_found') {
          if (await hasPendingCreate(id)) return readCached(id);
          return null;
        }
        // Offline / server down: fall back to the local cache.
        console.warn('API getContact failed, using local cache:', error);
        const cached = await readCached(id);
        if (cached) return cached;
        throw error;
      }
    },

    /**
     * A list of contacts from the server (every page with `allPages`, up to `cap`), saved to the
     * phone for offline use in the background. `fallbackToCache`: when the server can't be reached,
     * return the phone's cached copy with `error` set instead of throwing.
     */
    async listContacts(query: ListQuery = {}, opts: { fallbackToCache?: boolean } = {}): Promise<ListResult> {
      const { allPages, cap = DEFAULT_CAP, ...params } = query;
      try {
        const fetchedFor = await local.signedInUser().catch(() => null);
        const first = await remote.list(params);
        if (!first || !Array.isArray(first.contacts)) throw new Error('Unexpected response from the server.');
        let all: Contact[] = first.contacts;
        let next = allPages ? first.cursor : null;
        while (next && all.length < cap) {
          const page = await remote.list({ ...params, cursor: next });
          if (!page || !Array.isArray(page.contacts)) break;
          all = all.concat(page.contacts);
          next = page.cursor;
        }
        if (allPages && all.length > cap) all = all.slice(0, cap);
        // Keep what was just loaded for offline use (adds / updates only — removing contacts
        // deleted elsewhere is the full sync's job).
        inBackground(saveServerCopies(all, fetchedFor, { memo: true }));
        return { contacts: all, cursor: allPages ? null : first.cursor ?? null, error: null, fromCache: false };
      } catch (error: any) {
        if (!opts.fallbackToCache) throw error;
        console.warn('API load failed, falling back to local cache:', error);
        // The device cache when the server can't be reached (only this account's).
        let contacts: Contact[] = [];
        try {
          await local.ensureOwner();
          contacts = await local.query(params);
        } catch (cacheErr) {
          console.error('Failed to read the contact cache', cacheErr);
        }
        return { contacts: Array.isArray(contacts) ? contacts : [], cursor: null, error, fromCache: true };
      }
    },

    /** How many contacts match (every page, up to `cap`). */
    async countContacts(query: ContactListParams, cap = 10000): Promise<number> {
      const { contacts } = await repo.listContacts({ ...query, allPages: true, cap });
      return contacts.length;
    },

    /** Create a contact. Offline (no connection) it's saved on the phone and queued — the
     *  returned contact then has `pendingSync: true` and its id stays the same after upload. */
    async createContact(input: NewContactInput): Promise<Pending<Contact>> {
      // The id is made here and sent along: if the request reached the server but the answer
      // didn't come back (timeout, dropped connection), the queued retry carries the same id and
      // the server returns that contact instead of saving a second one.
      const id = input.id || uuidv4();
      const fetchedFor = await local.signedInUser().catch(() => null);
      try {
        const created = await remote.create({ ...input, id });
        inBackground(saveServerCopies([created], fetchedFor));
        return created;
      } catch (e: any) {
        if (!isNetworkError(e)) throw e;
        return queueCreate(input, id);
      }
    },

    /** Saves to the server and returns the updated contact. If the server can't be reached the
     *  edit is saved on the phone and queued (`pendingSync: true`); a real server error throws.
     *  `original`: the contact as the edit started (the conflict base is built from it), or an
     *  explicit `base`: if one of those fields changed elsewhere since, nothing is saved and it
     *  throws — getEditConflict(error) tells what changed. `force` saves anyway. */
    async updateContact(
      id: string, patch: ContactPatch,
      opts: { original?: Partial<Contact> | null; base?: Partial<Record<keyof ContactPatch, unknown>>; force?: boolean } = {},
    ): Promise<Pending<Contact>> {
      const base = opts.base ?? (opts.original ? editBase(opts.original, patch) : undefined);
      const fetchedFor = await local.signedInUser().catch(() => null);
      try {
        const updated = await remote.update(id, patch, {
          ...(base ? { base: base as Record<string, unknown> } : {}),
          ...(opts.force ? { force: true } : {}),
        });
        inBackground(saveServerCopies([updated], fetchedFor));
        return updated;
      } catch (e: any) {
        if (!isNetworkError(e)) throw e;
        const queued = await queueUpdate(id, patch as Record<string, unknown>, { base: base as Record<string, unknown> | undefined, force: opts.force });
        if (!queued) throw e; // nothing cached to show — let the screen say it couldn't save
        return queued;
      }
    },

    /** "Use theirs" after a conflict: keep the server's copy on the phone (unless a change is
     *  still queued here — that one wins until it's uploaded). */
    async acceptServerCopy(contact: Contact): Promise<void> {
      const fetchedFor = await local.signedInUser().catch(() => null);
      await saveServerCopies([contact], fetchedFor).catch(() => 0);
    },

    /** Moves the contact to Recently deleted on the server; the local cache is only cleared
     *  once that succeeded (offline: removed here, deleted on the server when back online).
     *  Emits a 'deleted' event for the Undo toast. */
    async deleteContact(id: string, name: string): Promise<{ daysLeft: number; pendingSync?: boolean }> {
      let res: { daysLeft?: number } | null | undefined;
      try {
        res = await remote.delete(id);
      } catch (e: any) {
        if (!isNetworkError(e)) throw e;
        await queueDelete(id);
        events.emit({ type: 'deleted', id, name, daysLeft: DEFAULT_DAYS_LEFT });
        return { daysLeft: DEFAULT_DAYS_LEFT, pendingSync: true };
      }
      await local.ensureOwner().catch(() => {});
      await local.remove(id).catch(() => { /* local cache is best-effort */ });
      const daysLeft = res?.daysLeft ?? DEFAULT_DAYS_LEFT;
      events.emit({ type: 'deleted', id, name, daysLeft });
      return { daysLeft };
    },

    /** A bad scan: remove it for good (not kept in Recently deleted). A contact that only exists on
     *  this phone (created offline, never uploaded) is dropped with its queued changes. */
    async discardContact(id: string): Promise<void> {
      try {
        await remote.delete(id);
      } catch (e: any) {
        const kind = asApiError(e)?.kind;
        const localOnly = (isNetworkError(e) || kind === 'not_found') && (await hasPendingCreate(id));
        if (!localOnly) throw e;
        await queueDelete(id);
        events.emit({ type: 'discarded', id });
        return;
      }
      await remote.purge(id).catch(() => { /* still in Recently deleted — harmless */ });
      await local.ensureOwner().catch(() => {});
      await local.remove(id).catch(() => {});
      events.emit({ type: 'discarded', id });
    },

    async restoreContact(id: string): Promise<Contact> {
      const fetchedFor = await local.signedInUser().catch(() => null);
      const restored = await remote.restore(id);
      inBackground(saveServerCopies([restored], fetchedFor));
      const name = [restored.firstName, restored.lastName].filter(Boolean).join(' ') || 'Contact';
      events.emit({ type: 'restored', id, name });
      return restored;
    },

    /** Save a LinkedIn profile picked for the contact; the updated contact is kept on the phone. */
    async attachLinkedIn(id: string, pick: LinkedInPick): Promise<LinkedInAttachResult> {
      const fetchedFor = await local.signedInUser().catch(() => null);
      const res = await remote.attachLinkedIn(id, pick);
      if (res?.contact) inBackground(saveServerCopies([res.contact], fetchedFor));
      return res;
    },

    async getDuplicates(id: string): Promise<any[]> {
      try {
        return (await remote.duplicates(id))?.matches ?? [];
      } catch {
        return []; // duplicate hints are advisory; never block viewing a contact
      }
    },
  };
  return repo;
}

export type ContactRepository = ReturnType<typeof createContactRepository>;
