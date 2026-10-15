import { eq } from 'drizzle-orm';
import * as LegacyFS from 'expo-file-system/legacy';
import { db } from '../db/client';
import { contacts, outbox, scanQueue, syncMeta } from '../db/schema';
import { fastStore } from '../lib/mmkv';
import { resetListCacheMemo } from './contactCache';
import type { CacheOwnerStore } from '../features/session/cacheOwner';

/**
 * Storage for "whose offline data is this": the cache_owner mark in SQLite sync_meta, and wiping
 * the contact cache, outbox, offline scans (+ their photos) and sync meta. The rule — keep it for
 * the same account, wipe it for a different one — and when it runs (on sign-in) belong to the
 * session module (src/features/session).
 */
const KEY = 'cache_owner';

async function readOwner(): Promise<string | null> {
  const rows = await db.select().from(syncMeta).where(eq(syncMeta.key, KEY)).limit(1);
  return rows?.[0]?.value ?? null;
}

export const sqliteCacheOwnerStore: CacheOwnerStore = {
  owner: readOwner,
  async setOwner(userId) {
    const now = new Date().toISOString();
    await db.insert(syncMeta).values({ key: KEY, value: userId, updatedAt: now })
      .onConflictDoUpdate({ target: syncMeta.key, set: { value: userId, updatedAt: now } });
  },
  async wipe() {
    await db.delete(outbox);
    await db.delete(scanQueue);
    await db.delete(contacts);
    await db.delete(syncMeta); // last-sync times etc. were the other account's
    clearSyncTimeFallbacks(); // …and so are their copies in the fast store ("Last synced")
    fastStore.setActiveEventName(''); // the active event (tags offline scans) was theirs too
    resetListCacheMemo();
    await LegacyFS.deleteAsync(`${LegacyFS.documentDirectory}scan-queue/`, { idempotent: true }).catch(() => {});
  },
};

/** The sync times also live in the fast store (fallback for "Last synced"); forget them. */
export function clearSyncTimeFallbacks() {
  fastStore.setLastSyncTime('');
  fastStore.setLastSyncCount(0);
  // sync/localStore.ts mirrors its sync_meta values onto the fast store as `_<key>`
  for (const k of Object.keys(fastStore)) {
    if (k.startsWith('_')) delete (fastStore as any)[k];
  }
}

/** The account this phone's offline data belongs to (null = not claimed yet / DB unavailable). */
export async function getCacheOwner(): Promise<string | null> {
  try {
    return await readOwner();
  } catch {
    return null;
  }
}
