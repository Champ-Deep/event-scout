import type { OfflineCache } from './core';

/**
 * The phone's contact cache, upload queue and offline scans belong to ONE account. Signing out
 * keeps them (so your own unsynced changes survive a sign-out / sign-in), but when a DIFFERENT
 * account signs in they're cleared first — otherwise that person saw your cached contacts and
 * your queued offline changes were uploaded into their account.
 *
 * Pure rule; the storage is an adapter (src/sync/owner.ts: SQLite + scan photos; tests: memory).
 */
export interface CacheOwnerStore {
  /** The account the offline data belongs to (null = unclaimed). Throws when the local DB is unavailable. */
  owner(): Promise<string | null>;
  setOwner(userId: string): Promise<void>;
  /** Delete all offline data, the owner mark included. */
  wipe(): Promise<void>;
}

export type HandOver = 'same' | 'claimed' | 'cleared' | 'skipped';

export function createOfflineCache(store: CacheOwnerStore, log?: (message: string) => void): OfflineCache & {
  handOver(userId: string | null | undefined): Promise<HandOver>;
} {
  let checked: string | null = null;
  return {
    async handOver(userId) {
      if (!userId) return 'skipped';
      if (checked === userId) return 'same';
      try {
        const owner = await store.owner();
        if (owner && owner !== userId) {
          await store.wipe();
          await store.setOwner(userId);
          log?.('[sync] another account signed in — cleared the previous account\'s offline data');
          checked = userId;
          return 'cleared';
        }
        if (!owner) await store.setOwner(userId);
        checked = userId;
        return owner ? 'same' : 'claimed';
      } catch {
        return 'skipped'; // local DB unavailable (web)
      }
    },
    async wipe() {
      checked = null;
      await store.wipe();
    },
  };
}
