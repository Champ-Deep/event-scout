import type { Contact } from '@es/types';
import { contactsQueries } from '../db/queries/contacts';
import { feedQueries, type FeedFilters } from '../db/queries/feed';

/**
 * The phone's contact cache (SQLite `contacts`), as raw storage. Rules about WHEN a server copy may
 * be written (a queued local change wins, whose cache it is) live in the contact repository
 * (src/features/contacts) and the sync engine — this module only reads and writes rows.
 * Every call is best-effort: the local DB is a no-op on web.
 */

/** Write one contact (insert or update). false when the local DB is unavailable. */
export async function cacheContact(c: any): Promise<boolean> {
  try {
    const now = new Date().toISOString();
    // Keep everything the server sent; pendingSync is derived from the outbox, not stored.
    const { pendingSync: _p, raw: _r, ...full } = c || {};
    await contactsQueries.upsert({
      id: String(c.id),
      eventId: c.eventId || '',
      firstName: c.firstName ?? null,
      lastName: c.lastName ?? null,
      company: c.company ?? null,
      title: c.title ?? null,
      email: c.email ?? null,
      phone: c.phone ?? null,
      notes: c.notes ?? null,
      score: c.score ?? null,
      scoreBand: c.scoreBand ?? null,
      source: ['scan', 'manual', 'import'].includes(c.source) ? c.source : 'scan',
      stage: ['new', 'contacted', 'qualified', 'converted'].includes(c.stage) ? c.stage : 'new',
      version: c.version || 1,
      createdAt: c.createdAt || now,
      updatedAt: c.updatedAt || now,
      scannedAt: c.scannedAt ?? null,
      raw: JSON.stringify(full),
    } as any);
    return true;
  } catch {
    return false; // local DB unavailable (web)
  }
}

// What the lists last saved per contact (id → updatedAt|score), so reloading a list doesn't
// rewrite thousands of unchanged rows every time.
const listCached = new Map<string, string>();

/** Like cacheContact, but skips a row the lists already saved unchanged. */
export async function cacheContactIfChanged(c: any): Promise<boolean> {
  if (!c?.id) return false;
  const id = String(c.id);
  const stamp = String(c.updatedAt ?? '') + '|' + String(c.score ?? '');
  if (listCached.get(id) === stamp) return true;
  const ok = await cacheContact(c);
  if (ok) listCached.set(id, stamp);
  return ok;
}

/** Forget what the lists cached (rows were removed from the cache, e.g. another account signed in). */
export function resetListCacheMemo(id?: string) {
  if (id) listCached.delete(id);
  else listCached.clear();
}

export async function getCachedContact(id: string): Promise<Contact | null> {
  try {
    return ((await contactsQueries.getById(id)) as unknown as Contact) ?? null;
  } catch {
    return null; // local cache unavailable
  }
}

export async function deleteCachedContact(id: string): Promise<void> {
  try { await contactsQueries.delete(id); } catch { /* best-effort */ }
  resetListCacheMemo(id);
}

export async function deleteCachedContacts(ids: string[]): Promise<void> {
  await contactsQueries.deleteMany(ids);
  resetListCacheMemo();
}

export async function cachedContactIds(): Promise<string[]> {
  return contactsQueries.allIds();
}

/** Cached contacts matching the list filters (offline fallback). */
export async function queryCachedContacts(filters: FeedFilters): Promise<Contact[]> {
  const rows = await feedQueries.getRankedFeed(filters);
  return (Array.isArray(rows) ? rows : []) as unknown as Contact[];
}
