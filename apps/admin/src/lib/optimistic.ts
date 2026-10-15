// Optimistic writes with rollback: apply the change to every matching cached query right away,
// undo it exactly if the server rejects, and refetch once it settles.
import type { QueryClient, QueryKey } from '@tanstack/react-query';
import type { Contact } from './types';

export type Rollback = () => void;

/** Apply `update` to every cached query under `prefix`; returns a function that restores them. */
export async function patchQueries<T>(qc: QueryClient, prefix: QueryKey, update: (old: T) => T): Promise<Rollback> {
  await qc.cancelQueries({ queryKey: prefix });
  const before = qc.getQueriesData<T>({ queryKey: prefix });
  before.forEach(([key, data]) => {
    if (data !== undefined) qc.setQueryData<T>(key, update(data));
  });
  return () => before.forEach(([key, data]) => qc.setQueryData(key, data));
}

/** Run several patches; one rollback undoes all of them. */
export async function patchAll(...patches: Promise<Rollback>[]): Promise<Rollback> {
  const rollbacks = await Promise.all(patches);
  return () => rollbacks.reverse().forEach((r) => r());
}

type ContactPage = { contacts: Contact[] };
const isPage = (d: unknown): d is ContactPage => !!d && typeof d === 'object' && Array.isArray((d as ContactPage).contacts);

/** Contacts are cached as list pages ({contacts}) and single rows (Contact) under ['admin','contacts']. */
export function removeContacts(qc: QueryClient, ids: string[]) {
  const gone = new Set(ids);
  return patchQueries<unknown>(qc, ['admin', 'contacts', 'list'], (d) => (isPage(d) ? { ...d, contacts: d.contacts.filter((c) => !gone.has(c.id)) } : d));
}

export function patchContact(qc: QueryClient, id: string, patch: Partial<Contact>) {
  return patchAll(
    patchQueries<unknown>(qc, ['admin', 'contacts', 'list'], (d) => (isPage(d) ? { ...d, contacts: d.contacts.map((c) => (c.id === id ? { ...c, ...patch } : c)) } : d)),
    patchQueries<unknown>(qc, ['admin', 'contacts', 'one', id], (d) => (d && typeof d === 'object' ? { ...(d as Contact), ...patch } : d)),
  );
}

/** Remove items with these ids from every cached array under `prefix`. */
export function removeFromArrays<T extends { id: string }>(qc: QueryClient, prefix: QueryKey, ids: string[]) {
  const gone = new Set(ids);
  return patchQueries<T[] | undefined>(qc, prefix, (d) => (Array.isArray(d) ? d.filter((x) => !gone.has(x.id)) : d));
}

/** Patch one item (by id) in every cached array under `prefix`. */
export function patchInArrays<T extends { id: string }>(qc: QueryClient, prefix: QueryKey, id: string, patch: Partial<T>) {
  return patchQueries<T[] | undefined>(qc, prefix, (d) => (Array.isArray(d) ? d.map((x) => (x.id === id ? { ...x, ...patch } : x)) : d));
}
