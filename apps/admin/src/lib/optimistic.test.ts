import { describe, expect, it } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { patchAll, patchContact, patchInArrays, removeContacts, removeFromArrays } from './optimistic';
import type { Contact } from './types';

const contact = (id: string, over: Partial<Contact> = {}) => ({ id, firstName: id, score: 50, ...over } as Contact);

function seeded() {
  const qc = new QueryClient();
  qc.setQueryData(['admin', 'contacts', 'list', { sort: 'recent' }], { contacts: [contact('a'), contact('b'), contact('c')], total: 3 });
  qc.setQueryData(['admin', 'contacts', 'list', { scoreBand: 'A' }], { contacts: [contact('b')], total: 1 });
  qc.setQueryData(['admin', 'contacts', 'one', 'b'], contact('b'));
  qc.setQueryData(['admin', 'lists'], [{ id: 'l1', name: 'Old', memberCount: 2 }, { id: 'l2', name: 'Other', memberCount: 0 }]);
  return qc;
}
const ids = (qc: QueryClient, q: object) => (qc.getQueryData(['admin', 'contacts', 'list', q]) as { contacts: Contact[] }).contacts.map((c) => c.id);

describe('removeContacts', () => {
  it('removes the rows from every cached contact page and rolls back exactly', async () => {
    const qc = seeded();
    const rollback = await removeContacts(qc, ['b']);
    expect(ids(qc, { sort: 'recent' })).toEqual(['a', 'c']);
    expect(ids(qc, { scoreBand: 'A' })).toEqual([]);
    rollback();
    expect(ids(qc, { sort: 'recent' })).toEqual(['a', 'b', 'c']);
    expect(ids(qc, { scoreBand: 'A' })).toEqual(['b']);
  });
});

describe('patchContact', () => {
  it('patches the grid rows and the single-contact cache together', async () => {
    const qc = seeded();
    const rollback = await patchContact(qc, 'b', { adminNotes: 'call Monday' });
    expect((qc.getQueryData(['admin', 'contacts', 'one', 'b']) as Contact).adminNotes).toBe('call Monday');
    const page = qc.getQueryData(['admin', 'contacts', 'list', { sort: 'recent' }]) as { contacts: Contact[] };
    expect(page.contacts.find((c) => c.id === 'b')?.adminNotes).toBe('call Monday');
    expect(page.contacts.find((c) => c.id === 'a')?.adminNotes).toBeUndefined();
    rollback();
    expect((qc.getQueryData(['admin', 'contacts', 'one', 'b']) as Contact).adminNotes).toBeUndefined();
  });
});

describe('array helpers', () => {
  it('patchInArrays / removeFromArrays with one combined rollback', async () => {
    const qc = seeded();
    const before = qc.getQueryData(['admin', 'lists']);
    const rollback = await patchAll(
      patchInArrays<{ id: string; name: string }>(qc, ['admin', 'lists'], 'l1', { name: 'New' }),
      removeFromArrays(qc, ['admin', 'lists'], ['l2']),
    );
    expect(qc.getQueryData(['admin', 'lists'])).toEqual([{ id: 'l1', name: 'New', memberCount: 2 }]);
    rollback();
    expect(qc.getQueryData(['admin', 'lists'])).toEqual(before);
  });
  it('leaves non-array caches under the prefix alone', async () => {
    const qc = seeded();
    qc.setQueryData(['admin', 'lists', 'l1', 'members'], [contact('a')]);
    qc.setQueryData(['admin', 'lists', 'meta'], { note: 'not an array' });
    await removeFromArrays(qc, ['admin', 'lists'], ['a']);
    expect(qc.getQueryData(['admin', 'lists', 'l1', 'members'])).toEqual([]);
    expect(qc.getQueryData(['admin', 'lists', 'meta'])).toEqual({ note: 'not an array' });
  });
});
