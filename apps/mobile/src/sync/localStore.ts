import { eq } from 'drizzle-orm';
import { db } from '../db/client';
import { syncMeta } from '../db/schema';
import { outboxQueries } from '../db/queries/outbox';
import { fastStore } from '../lib/mmkv';
import { getCacheOwner } from './owner';
import { session } from '../features/session';
import { getQueuedScans, markScanFailed, removeQueuedScan } from './offline';
import {
  cacheContact, cachedContactIds, deleteCachedContacts, getCachedContact,
} from './contactCache';
import type { SyncStore } from './engine';
import type { OutboxItem } from './core';

/**
 * The phone's real local store for the sync engine and the contact repository: SQLite outbox,
 * contact cache, scan queue and sync meta, plus the cache owner. Tests use in-memory stand-ins.
 */

// ---- persisted meta (SQLite sync_meta; falls back to the fast store, e.g. on web) ----
async function getMeta(key: string): Promise<string | null> {
  try {
    const rows = await db.select().from(syncMeta).where(eq(syncMeta.key, key)).limit(1);
    if (rows[0]) return rows[0].value;
  } catch { /* local DB unavailable */ }
  const mirrored = (fastStore as any)[`_${key}`];
  if (mirrored != null) return mirrored;
  // "Last synced" was also kept in the fast store's own keys.
  if (key === 'last_sync_at') return fastStore.getLastSyncTime() || null;
  if (key === 'last_sync_count') return String(fastStore.getLastSyncCount());
  return null;
}

async function setMeta(key: string, value: string): Promise<void> {
  (fastStore as any)[`_${key}`] = value;
  if (key === 'last_sync_at') fastStore.setLastSyncTime(value);
  if (key === 'last_sync_count') fastStore.setLastSyncCount(Number(value) || 0);
  try {
    const now = new Date().toISOString();
    await db.insert(syncMeta).values({ key, value, updatedAt: now })
      .onConflictDoUpdate({ target: syncMeta.key, set: { value, updatedAt: now } });
  } catch { /* local DB unavailable */ }
}

export const outboxStore = {
  all: async () => (await outboxQueries.getAll()) as unknown as OutboxItem[],
  entityIds: () => outboxQueries.allEntityIds(),
  count: () => outboxQueries.count(),
  acknowledge: (ids: string[]) => outboxQueries.markAcknowledged(ids),
  recordFailure: async (id: string, reason: string, countsAsAttempt: boolean) => {
    await outboxQueries.recordAttemptFailure(id, reason, countsAsAttempt);
  },
  hasPendingCreate: (id: string) => outboxQueries.hasPendingCreate(id),
  enqueue: async (item: OutboxItem) => { await outboxQueries.enqueue(item as any); },
  dropForEntity: async (id: string) => { await outboxQueries.deleteForEntity(id); },
};

// Whose data this is comes from the session: it hands the offline cache over on sign-in; `ensure`
// only waits for / repeats that for work that may start before a sign-in finished.
export const ownerStore = {
  ensure: () => session.offlineCacheReady(),
  signedIn: async () => session.signedInUserId(),
  cacheOwner: () => getCacheOwner(),
};

export const localSyncStore: SyncStore = {
  outbox: outboxStore,
  contacts: {
    put: (c) => cacheContact(c),
    get: (id) => getCachedContact(id),
    allIds: () => cachedContactIds(),
    deleteMany: (ids) => deleteCachedContacts(ids),
  },
  scans: {
    all: () => getQueuedScans(),
    markFailed: (id, reason, countsAsAttempt) => markScanFailed(id, reason, countsAsAttempt),
    remove: (id) => removeQueuedScan(id),
  },
  meta: { get: getMeta, set: setMeta },
  owner: ownerStore,
};
