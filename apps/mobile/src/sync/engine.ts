import { logger } from '../lib/log';
import { errorMessage } from '../lib/apiError';
import {
  isNetworkError, isRetryableReason, retryableReason, reasonText, syncEvents,
  type OutboxItem, type QueuedScan, type SyncEvent,
} from './core';

/**
 * The sync engine: push the outbox in order, upload offline scans, pull every contact page into the
 * cache. Its remote (the backend) and local store (outbox, contact cache, scan queue, sync meta,
 * cache owner) are adapters passed in — src/sync/state.ts wires the real ones; tests pass in-memory
 * ones. Pure module: no React Native / database imports.
 */

/** After this many real failures (the server said no — not offline / timeouts) an item is only
 *  retried from the Sync page, not automatically. */
export const MAX_AUTO_ATTEMPTS = 3;

export const PUSH_BATCH = 25;
const PULL_PAGE = 200;
const PULL_MAX_PAGES = 20;

// ---- adapters ----

export interface PushItem {
  id: string; entityType: string; entityId: string; action: string; payload: unknown; createdAt: string;
}

export interface SyncRemote {
  /** POST /api/sync/push → { acknowledgedIds, conflicts, failed } */
  push(batch: { items: PushItem[] }): Promise<any>;
  /** GET /api/sync/pull → { contacts, cursor, total? } */
  pull(params: { limit: number; cursor?: string }): Promise<any>;
  /** Upload one queued scan photo (its id goes up as client_id) → the scan endpoint's answer. */
  uploadScan(scan: QueuedScan, userId: string): Promise<any>;
  /** The server couldn't read a card (5xx): read it on the phone instead. true = saved. */
  readCardOnDevice?(scan: QueuedScan): Promise<boolean>;
}

export interface SyncStore {
  outbox: {
    all(): Promise<OutboxItem[]>;
    entityIds(): Promise<string[]>;
    count(): Promise<number>;
    acknowledge(ids: string[]): Promise<void>;
    recordFailure(id: string, reason: string, countsAsAttempt: boolean): Promise<void>;
  };
  contacts: {
    put(contact: any): Promise<boolean>;
    get(id: string): Promise<any | null>;
    allIds(): Promise<string[]>;
    /** removes rows (and forgets list-cache memos) */
    deleteMany(ids: string[]): Promise<void>;
  };
  scans: {
    all(): Promise<QueuedScan[]>;
    markFailed(id: string, reason: string, countsAsAttempt: boolean): Promise<void>;
    remove(id: string): Promise<void>;
  };
  meta: {
    get(key: string): Promise<string | null>;
    set(key: string, value: string): Promise<void>;
  };
  owner: {
    /** clear another account's offline data if a different account signed in */
    ensure(): Promise<unknown>;
    /** the signed-in account (from the saved token) */
    signedIn(): Promise<string | null>;
    /** the account the offline data belongs to (null = unclaimed) */
    cacheOwner(): Promise<string | null>;
  };
}

// ---- public shapes ----

export interface SyncQueueEntry {
  id: string;
  action: 'create' | 'update' | 'delete' | 'scan' | string;
  entity: string; // 'contact' | 'event' | 'scan-card' | 'scan-qr'
  details: string;
  status: 'PENDING' | 'FAILED';
  attempts: number;
  createdAt: string | null;
  errorHint?: string | undefined;
  /** background sync still retries it (false: failed too often, or waits on such a change — Retry on the Sync page) */
  autoRetry?: boolean;
}

export interface SyncState {
  pendingChanges: number;
  failedChanges: number;
  queue: SyncQueueEntry[];
  lastSyncAt: string | null;      // last run that fully succeeded
  lastSyncCount: number;
  lastAttemptAt: string | null;   // last run, successful or not
  lastError: string | null;       // why the last run didn't fully succeed
}

export type SyncPhase = 'push' | 'scans' | 'pull' | 'done';

/** A change the server acknowledged without applying it, and why. */
export interface SkippedChange {
  id: string;
  details: string;
  reason: string;
}

export interface SyncRunResult extends SyncState {
  pushed: number;   // changes the server applied
  scans: number;    // offline scans the server read into contacts
  failed: number;   // changes that failed this run (kept for retry)
  skipped: number;  // changes the server refused for good (e.g. not your contact)
  /** what was skipped — or applied only in part because it changed elsewhere — and why (removed from the queue) */
  skippedItems?: SkippedChange[];
  pulled: number;   // contacts downloaded
  cached: number;   // of those, saved to this device for offline use
  removed: number;  // contacts deleted elsewhere, removed from this device
  /** true when something failed only because the server couldn't be reached (worth retrying soon) */
  offline: boolean;
  error?: string;
}

export interface SyncEngine {
  runSync(onPhase?: (phase: SyncPhase) => void, opts?: { auto?: boolean }): Promise<SyncRunResult>;
  isSyncRunning(): boolean;
  getSyncState(): Promise<SyncState>;
  getLastPullAt(): Promise<string | null>;
  countPendingChanges(): Promise<number>;
  discardQueueItem(id: string): Promise<void>;
}

// ---- pure helpers ----

function syncErrorText(err: any, fallback: string): string {
  return errorMessage(err, fallback, { offline: "You're offline", unreachable: "Can't reach the server", unauthorized: 'Sign in again to sync' });
}

/** Not the item's fault — offline, timed out, or the sign-in needs renewing: keep it waiting
 *  without using up one of its automatic retries. */
function isRetryableError(err: any): boolean {
  return isNetworkError(err) || err?.response?.status === 401 || err?.status === 401;
}

function summarizePayload(payload: unknown): string {
  try {
    const data = typeof payload === 'string' ? JSON.parse(payload) : payload;
    const name = [data?.firstName, data?.lastName].filter(Boolean).join(' ');
    return name || data?.company || data?.email || 'Contact change';
  } catch {
    return 'Pending change';
  }
}

/** Failed often enough that only a manual sync retries it. */
function exhausted(item: { attempts?: number | null; lastError?: string | null }): boolean {
  return (item.attempts ?? 0) >= MAX_AUTO_ATTEMPTS && !isRetryableReason(item.lastError);
}

function isFailed(item: { attempts?: number | null; lastError?: string | null }): boolean {
  return (item.attempts ?? 0) > 0 && !!item.lastError && !isRetryableReason(item.lastError);
}

function toQueueEntry(item: OutboxItem, blocked: boolean): SyncQueueEntry {
  return {
    id: String(item.id),
    action: item.action ?? 'update',
    entity: item.entityType ?? 'contact',
    details: summarizePayload(item.payload),
    status: isFailed(item) ? 'FAILED' : 'PENDING',
    attempts: item.attempts ?? 0,
    createdAt: item.createdAt ?? null,
    errorHint: reasonText(item.lastError),
    autoRetry: !blocked && !exhausted(item),
  };
}

const SCAN_PREFIX = 'scan:';

function scanToQueueEntry(s: QueuedScan): SyncQueueEntry {
  return {
    id: SCAN_PREFIX + s.id,
    action: 'scan',
    entity: s.kind === 'qr' ? 'scan-qr' : 'scan-card',
    details: s.kind === 'qr' ? 'QR code photo — waiting to be read' : 'Business card photo — waiting to be read',
    status: isFailed(s) ? 'FAILED' : 'PENDING',
    attempts: s.attempts ?? 0,
    createdAt: s.createdAt ?? null,
    errorHint: reasonText(s.lastError),
    autoRetry: !exhausted(s),
  };
}

/** Outbox rows in upload order; a contact's later changes wait while an earlier one is stuck. */
function outboxEntries(items: OutboxItem[]): SyncQueueEntry[] {
  const stuck = new Set<string>();
  return items.map((item) => {
    const key = String(item.entityId);
    const blocked = stuck.has(key);
    if (blocked || exhausted(item)) stuck.add(key);
    return toQueueEntry(item, blocked);
  });
}

const FIELD_LABELS: Record<string, string> = {
  firstName: 'first name', lastName: 'last name', company: 'company', title: 'title', email: 'email',
  phone: 'phone', notes: 'notes', linkedin: 'LinkedIn', eventId: 'event',
};

function changedElsewhereText(fields: unknown): string {
  const names = Array.isArray(fields) ? fields.map((f) => FIELD_LABELS[String(f)] || String(f)) : [];
  return names.length ? `Kept the newer ${names.join(', ')} changed elsewhere` : 'Changed elsewhere — kept the newer version';
}

function conflictReason(reason: unknown): string {
  switch (reason) {
    case 'not_owner': return 'Not your contact';
    case 'not_found':
    case 'deleted': return 'Deleted elsewhere';
    default: return typeof reason === 'string' && reason ? reason : 'Skipped by the server';
  }
}

// ---- the engine ----

export function createSyncEngine(deps: {
  remote: SyncRemote;
  store: SyncStore;
  events?: { emit(e: SyncEvent): void };
  now?: () => string;
}): SyncEngine {
  const { remote, store } = deps;
  const events = deps.events ?? syncEvents;
  const now = deps.now ?? (() => new Date().toISOString());

  async function readOutbox(): Promise<OutboxItem[]> {
    try {
      return await store.outbox.all();
    } catch {
      // Local DB unavailable (e.g. web preview) — no persistent queue exists.
      logger.info('Outbox read unavailable; reporting empty queue');
      return [];
    }
  }

  async function outboxEntityIds(): Promise<Set<string>> {
    try {
      return new Set(await store.outbox.entityIds());
    } catch {
      return new Set();
    }
  }

  async function readScans(): Promise<QueuedScan[]> {
    try { return await store.scans.all(); } catch { return []; }
  }

  /** "Priya Shah" for a queued change: the payload's name, else the phone's copy of the contact. */
  async function describeChange(item: { entityId: unknown; payload: unknown }): Promise<string> {
    const fromPayload = summarizePayload(item.payload);
    if (fromPayload !== 'Contact change' && fromPayload !== 'Pending change') return fromPayload;
    try {
      const c: any = await store.contacts.get(String(item.entityId));
      const name = [c?.firstName, c?.lastName].filter(Boolean).join(' ');
      return name || c?.company || c?.email || fromPayload;
    } catch {
      return fromPayload;
    }
  }

  async function getSyncState(): Promise<SyncState> {
    await store.owner.ensure(); // never show / upload another account's queue
    const [items, scans] = await Promise.all([readOutbox(), readScans()]);
    const queue = [...outboxEntries(items), ...scans.map(scanToQueueEntry)]
      .sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')));
    const [lastSyncAt, lastCount, lastAttemptAt, lastError] = await Promise.all([
      store.meta.get('last_sync_at'), store.meta.get('last_sync_count'), store.meta.get('last_attempt_at'), store.meta.get('last_error'),
    ]);
    return {
      pendingChanges: queue.length,
      failedChanges: queue.filter((q) => q.status === 'FAILED').length,
      queue,
      lastSyncAt: lastSyncAt || null,
      lastSyncCount: Number(lastCount) || 0,
      lastAttemptAt,
      lastError: lastError || null,
    };
  }

  /** Upload photos scanned while offline; the server reads them into contacts as usual. Each goes
   *  up with its queue id as client_id, so a repeat (e.g. the answer got lost) can't save it twice. */
  async function pushQueuedScans(
    userId: string, auto: boolean, stillOwner: () => Promise<boolean>,
  ): Promise<{ done: number; failed: number; offline: boolean; aborted: boolean; error?: string | undefined }> {
    const scans = (await readScans())
      .filter((s) => !(auto && exhausted(s)))
      .sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')));
    if (!scans.length) return { done: 0, failed: 0, offline: false, aborted: false };
    let done = 0;
    let failed = 0;
    let error: string | undefined;
    let offline = false;
    let aborted = false;
    for (const s of scans) {
      if (!(await stillOwner())) { aborted = true; break; }
      try {
        const res: any = await remote.uploadScan(s, userId);
        if (res?.contact_id || res?.status === 'duplicate') {
          // Saved (or already in your contacts) — the photo is no longer needed.
          await store.scans.remove(s.id);
          done += 1;
        } else {
          failed += 1;
          await store.scans.markFailed(s.id, res?.message || 'No contact details could be read from this photo', true);
        }
      } catch (err: any) {
        // The server's AI couldn't read it (bad key, outage): read the card on this phone instead.
        if (s.kind === 'card' && (err?.response?.status ?? 0) >= 500 && remote.readCardOnDevice) {
          const local = await remote.readCardOnDevice(s).catch(() => false);
          if (local) { await store.scans.remove(s.id); done += 1; continue; }
        }
        failed += 1;
        const reason = syncErrorText(err, 'Upload failed');
        const retry = isRetryableError(err);
        await store.scans.markFailed(s.id, retry ? retryableReason(reason) : reason, !retry);
        if (isNetworkError(err)) { error = reason; offline = true; break; } // still offline — try the rest later
        // Rejected sign-in / API key: every other scan would get the same answer, so stop here.
        const status = err?.response?.status;
        if (status === 401 || status === 403) { error = reason; break; }
      }
    }
    if (failed && !error) error = `${failed} scan${failed === 1 ? '' : 's'} couldn't be read`;
    return { done, failed, offline, aborted, error };
  }

  async function doRunSync(onPhase: ((phase: SyncPhase) => void) | undefined, auto: boolean): Promise<SyncRunResult> {
    let pushed = 0;
    let scansDone = 0;
    let offline = false;
    let failed = 0;
    let skipped = 0;
    const skippedItems: SkippedChange[] = [];
    let pulled = 0;
    let cached = 0;
    let removed = 0;
    let pullComplete = false;
    let aborted = false;
    let error: string | undefined;

    onPhase?.('push');
    await store.owner.ensure(); // never upload another account's queued changes
    // Whose data this run handles. Signing out / into another account mid-run stops it before it
    // uploads or caches anything more (the queue / cache may already be the next account's).
    const owner = await store.owner.signedIn();
    const stillOwner = async () => {
      if (!owner || (await store.owner.signedIn()) !== owner) return false;
      const cacheOwner = await store.owner.cacheOwner();
      return !cacheOwner || cacheOwner === owner;
    };
    if (!owner) {
      aborted = true;
      error = 'Sign in again to sync';
    }

    // ---- push: oldest first, in batches, until the queue is empty or a batch fails ----
    if (!aborted) {
      const stuck = new Set<string>(); // contacts whose earlier change didn't go up: later ones wait (order matters)
      const queue: OutboxItem[] = [];
      for (const item of await readOutbox()) {
        const key = String(item.entityId);
        if (stuck.has(key)) continue;
        if (auto && exhausted(item)) { stuck.add(key); continue; }
        queue.push(item);
      }
      while (queue.length) {
        if (!(await stillOwner())) { aborted = true; break; }
        const batch: OutboxItem[] = [];
        while (queue.length && batch.length < PUSH_BATCH) {
          const item = queue.shift()!;
          if (!stuck.has(String(item.entityId))) batch.push(item);
        }
        if (!batch.length) break;
        try {
          const res: any = await remote.push({
            items: batch.map((i) => ({
              id: String(i.id), entityType: i.entityType, entityId: String(i.entityId),
              action: i.action, payload: i.payload, createdAt: i.createdAt,
            })),
          });
          const acked = new Set<string>((res?.acknowledgedIds || []).map(String));
          // Refused for good (not yours, deleted elsewhere, invalid…): acknowledged — drop it, say why.
          // "changed_elsewhere": the edit WAS applied, except fields changed on another device since
          // (the newer value was kept; notes were merged) — listed with the server's message.
          const conflicts: any[] = res?.conflicts || [];
          let refused = 0;
          for (const c of conflicts) {
            acked.add(String(c.id));
            const item = batch.find((i) => String(i.id) === String(c.id));
            // "notes_shortened" (applied: true): saved, but notes over 20,000 characters were cut.
            const partial = c.reason === 'changed_elsewhere' || c.reason === 'notes_shortened' || c.applied === true;
            if (!partial) refused += 1;
            skippedItems.push({
              id: String(c.id),
              details: item ? await describeChange(item) : 'Contact change',
              reason: partial
                ? (typeof c.message === 'string' && c.message ? c.message
                  : c.reason === 'notes_shortened' ? 'Saved — the notes were too long and were shortened'
                    : changedElsewhereText(c.fields))
                : conflictReason(c.reason),
            });
          }
          if (acked.size > 0) {
            try { await store.outbox.acknowledge([...acked]); } catch { /* local DB unavailable */ }
          }
          skipped += refused;
          pushed += acked.size - refused;
          const reported = new Set<string>();
          for (const f of res?.failed || []) {
            const id = String(f.id);
            reported.add(id);
            const item = batch.find((i) => String(i.id) === id);
            if (item) stuck.add(String(item.entityId));
            // Held back only because an earlier change to the same contact failed: not its own failure.
            const waiting = /waiting on an earlier change/i.test(String(f.reason || ''));
            if (!waiting) failed += 1;
            const reason = String(f.reason || 'Upload failed');
            try { await store.outbox.recordFailure(id, waiting ? retryableReason(reason) : reason, !waiting); } catch { /* */ }
          }
          // Neither acknowledged nor reported (shouldn't happen): kept, and its contact's later changes wait.
          for (const i of batch) {
            if (!acked.has(String(i.id)) && !reported.has(String(i.id))) stuck.add(String(i.entityId));
          }
        } catch (err: any) {
          // The whole batch failed: it gets an attempt + reason (offline / timeout don't count
          // against automatic retries), nothing is lost, and the rest waits for the next run.
          const reason = syncErrorText(err, 'Upload failed');
          const retry = isRetryableError(err);
          error = reason;
          if (isNetworkError(err)) offline = true;
          failed += batch.length;
          for (const item of batch) {
            try { await store.outbox.recordFailure(String(item.id), retry ? retryableReason(reason) : reason, !retry); } catch { /* */ }
          }
          logger.warn('Sync push failed', { error: reason });
          break;
        }
      }
      if (failed && !error) error = `${failed} change${failed === 1 ? '' : 's'} couldn't be uploaded`;
    }

    // ---- offline scans ----
    onPhase?.('scans');
    if (!aborted && owner) {
      const scanRes = await pushQueuedScans(owner, auto, stillOwner);
      scansDone = scanRes.done;
      failed += scanRes.failed;
      if (scanRes.offline) offline = true;
      if (scanRes.aborted) aborted = true;
      if (scanRes.error) error = error ? `${error}; ${scanRes.error.toLowerCase()}` : scanRes.error;
    }

    // ---- pull every page into the cache ----
    onPhase?.('pull');
    if (!aborted) {
      try {
        const seen = new Set<string>();
        let cursor: string | undefined;
        let complete = false;
        let firstTotal: number | undefined;
        let lastTotal: number | undefined;
        for (let page = 0; page < PULL_MAX_PAGES; page++) {
          const res: any = await remote.pull({ limit: PULL_PAGE, ...(cursor ? { cursor } : {}) });
          if (!(await stillOwner())) { aborted = true; break; } // signed out / switched: don't cache this page
          if (page === 0) firstTotal = typeof res?.total === 'number' ? res.total : undefined;
          lastTotal = typeof res?.total === 'number' ? res.total : undefined;
          // Contacts with changes still waiting to upload keep the phone's version: the server copy
          // is older (it used to overwrite the edit on the phone until the upload finally went through).
          const pendingIds = await outboxEntityIds();
          const list: any[] = res?.contacts || [];
          pulled += list.length;
          for (const c of list) {
            seen.add(String(c.id));
            if (pendingIds.has(String(c.id))) continue;
            if (await store.contacts.put(c)) cached += 1;
          }
          if (!res?.cursor) { complete = true; break; }
          cursor = res.cursor;
        }
        // Offset paging: a contact added / deleted on the server while paging shifts the pages, so one
        // can be skipped. Only a full list whose count matched throughout may remove contacts.
        const consistent = firstTotal === undefined || (firstTotal === lastTotal && seen.size === lastTotal);
        pullComplete = complete && !aborted;
        // Only with the full list: drop contacts deleted on the web / another phone, but never one
        // that's still waiting to upload (e.g. created offline — re-read: it may have been queued meanwhile).
        if (pullComplete && consistent) {
          try {
            const pendingIds = await outboxEntityIds();
            const stale = (await store.contacts.allIds()).filter((id) => !seen.has(id) && !pendingIds.has(id));
            if (stale.length) {
              await store.contacts.deleteMany(stale);
              removed = stale.length;
            }
          } catch { /* local DB unavailable */ }
        }
      } catch (err: any) {
        const reason = syncErrorText(err, 'Download failed');
        error = error ? `${error}; ${reason.toLowerCase()}` : reason;
        logger.warn('Sync pull failed', { error: reason });
      }
    }

    if (aborted) {
      // Signed out / another account signed in meanwhile: record nothing for the account that left.
      onPhase?.('done');
      events.emit({ type: 'synced', pushed, scans: scansDone, failed });
      const state = await getSyncState();
      return {
        ...state, pushed, scans: scansDone, failed, skipped, skippedItems, pulled, cached, removed, offline,
        error: error || 'Sync stopped: you signed out',
      };
    }

    const at = now();
    await store.meta.set('last_attempt_at', at);
    await store.meta.set('last_error', error || '');
    if (pullComplete) await store.meta.set('last_pull_at', at);
    if (!error) {
      await store.meta.set('last_sync_at', at);
      await store.meta.set('last_sync_count', String(pushed + scansDone + pulled));
    }
    onPhase?.('done');
    events.emit({ type: 'synced', pushed, scans: scansDone, failed });

    const state = await getSyncState();
    return { ...state, pushed, scans: scansDone, failed, skipped, skippedItems, pulled, cached, removed, offline, ...(error ? { error } : {}) };
  }

  /**
   * One sync at a time: a tap while another manual sync runs just waits for that run. `auto`
   * (background sync) leaves out items that failed MAX_AUTO_ATTEMPTS times; a manual sync asked
   * for while a background one runs waits for it, then runs once more with everything.
   */
  let inFlight: Promise<SyncRunResult> | null = null;
  let inFlightAuto = false;

  function runSync(onPhase?: (phase: SyncPhase) => void, opts: { auto?: boolean } = {}): Promise<SyncRunResult> {
    const auto = !!opts.auto;
    if (inFlight) {
      if (auto || !inFlightAuto) return inFlight;
      return inFlight.catch(() => null).then(() => runSync(onPhase, opts));
    }
    inFlightAuto = auto;
    inFlight = doRunSync(onPhase, auto).finally(() => { inFlight = null; });
    return inFlight;
  }

  return {
    runSync,
    isSyncRunning: () => !!inFlight,
    getSyncState,
    /** When the contact list was last fully downloaded to this phone (null = never). */
    getLastPullAt: async () => (await store.meta.get('last_pull_at')) || null,
    /** Changes and offline scans on this phone that haven't reached the server yet. */
    async countPendingChanges() {
      let n = 0;
      try { n += await store.outbox.count(); } catch { /* local DB unavailable */ }
      n += (await readScans()).length;
      return n;
    },
    /** Throw away one queued change the user doesn't want to upload. */
    async discardQueueItem(id: string) {
      if (id.startsWith(SCAN_PREFIX)) {
        await store.scans.remove(id.slice(SCAN_PREFIX.length));
        return;
      }
      await store.outbox.acknowledge([id]);
    },
  };
}
