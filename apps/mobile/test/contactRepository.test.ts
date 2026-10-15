import { describe, expect, it, vi } from 'vitest';
import { createContactRepository, editBase, getEditConflict, type ContactLocal, type ContactRemote } from '../src/features/contacts/repository';
import type { OutboxItem } from '../src/sync/core';

const offline = () => new Error('OFFLINE');
const http = (status: number, detail?: unknown) => Object.assign(new Error(`Request failed with status code ${status}`), {
  response: { status, data: detail === undefined ? {} : { detail } },
});

function memoryLocal(init: { contacts?: any[]; queued?: OutboxItem[]; user?: string | null } = {}) {
  const contacts = new Map<string, any>((init.contacts ?? []).map((c) => [String(c.id), c]));
  const queued: OutboxItem[] = [...(init.queued ?? [])];
  const who = { user: init.user === undefined ? 'u1' : init.user };
  const local: ContactLocal = {
    ensureOwner: vi.fn(async () => 'same'),
    signedInUser: async () => who.user,
    get: async (id) => contacts.get(id) ?? null,
    put: async (c) => { contacts.set(String(c.id), c); return true; },
    putIfChanged: async (c) => { contacts.set(String(c.id), c); return true; },
    remove: async (id) => { contacts.delete(id); },
    query: async (p) => [...contacts.values()].filter((c) => !p.eventId || c.eventId === p.eventId),
    pendingIds: async () => queued.map((q) => q.entityId),
    queued: async () => [...queued],
    hasPendingCreate: async (id) => queued.some((q) => q.entityId === id && q.action === 'create'),
    enqueue: async (q) => { queued.push(q); },
    dropQueued: async (id) => {
      for (let i = queued.length - 1; i >= 0; i--) if (queued[i]!.entityId === id) queued.splice(i, 1);
    },
  };
  return { local, contacts, queued, who };
}

function memoryRemote(over: Partial<ContactRemote> = {}): ContactRemote {
  return {
    list: async () => ({ contacts: [], cursor: null }),
    get: async (id) => ({ id } as any),
    create: async (input) => ({ ...input } as any),
    update: async (id, patch) => ({ id, ...patch } as any),
    delete: async () => ({ daysLeft: 30 }),
    purge: async () => ({}),
    restore: async (id) => ({ id, firstName: 'Back' } as any),
    duplicates: async () => ({ matches: [] }),
    attachLinkedIn: async (id, pick) => ({ contact: { id, linkedin: pick.url } as any, photoAdded: false }),
    ...over,
  };
}

const queuedUpdate = (entityId: string, payload: object = { firstName: 'Mine' }): OutboxItem => ({
  id: `q-${entityId}`, entityType: 'contact', entityId, action: 'update', payload: JSON.stringify(payload),
  createdAt: '2026-01-01T00:00:00.000Z', attempts: 0, lastError: null,
});

describe('contact repository — a queued local change beats the server copy', () => {
  it('getContact returns the server copy but does not overwrite a contact with a queued change', async () => {
    const { local, contacts } = memoryLocal({ contacts: [{ id: 'c1', firstName: 'Mine' }], queued: [queuedUpdate('c1')] });
    const repo = createContactRepository({ remote: memoryRemote({ get: async () => ({ id: 'c1', firstName: 'Server' } as any) }), local });

    const c = await repo.getContact('c1');
    await repo.flush();
    expect(c?.firstName).toBe('Server');
    expect(contacts.get('c1').firstName).toBe('Mine');
  });

  it('getContact refreshes the phone copy when nothing is queued', async () => {
    const { local, contacts } = memoryLocal({ contacts: [{ id: 'c1', firstName: 'Old' }] });
    const repo = createContactRepository({ remote: memoryRemote({ get: async () => ({ id: 'c1', firstName: 'Server' } as any) }), local });

    await repo.getContact('c1');
    await repo.flush();
    expect(contacts.get('c1').firstName).toBe('Server');
  });

  it('"Use theirs" after a conflict keeps a queued offline edit', async () => {
    const { local, contacts } = memoryLocal({ contacts: [{ id: 'c1', firstName: 'Mine' }], queued: [queuedUpdate('c1')] });
    const repo = createContactRepository({ remote: memoryRemote(), local });

    await repo.acceptServerCopy({ id: 'c1', firstName: 'Theirs' } as any);
    expect(contacts.get('c1').firstName).toBe('Mine');

    await repo.acceptServerCopy({ id: 'c2', firstName: 'Theirs' } as any);
    expect(contacts.get('c2').firstName).toBe('Theirs');
  });

  it('lists cache every contact except the ones with queued changes', async () => {
    const { local, contacts } = memoryLocal({ contacts: [{ id: 'c1', firstName: 'Mine' }], queued: [queuedUpdate('c1')] });
    const remote = memoryRemote({
      list: async () => ({ contacts: [{ id: 'c1', firstName: 'Server' }, { id: 'c2', firstName: 'Two' }] as any, cursor: null }),
    });
    const repo = createContactRepository({ remote, local });

    const res = await repo.listContacts({ limit: 500 });
    await repo.flush();
    expect(res.contacts).toHaveLength(2);
    expect(contacts.get('c1').firstName).toBe('Mine');
    expect(contacts.get('c2').firstName).toBe('Two');
  });

  it('caches nothing when another account signed in while the list was loading', async () => {
    const mem = memoryLocal();
    const remote = memoryRemote({
      list: async () => { mem.who.user = 'u2'; return { contacts: [{ id: 'c1' }] as any, cursor: null }; },
    });
    const repo = createContactRepository({ remote, local: mem.local });

    await repo.listContacts();
    await repo.flush();
    expect(mem.contacts.size).toBe(0);
  });
});

describe('contact repository — lists', () => {
  it('follows the cursor for every page, up to the cap', async () => {
    const pages: Record<string, { contacts: any[]; cursor: string | null }> = {
      start: { contacts: [{ id: '1' }, { id: '2' }], cursor: 'p2' },
      p2: { contacts: [{ id: '3' }, { id: '4' }], cursor: 'p3' },
      p3: { contacts: [{ id: '5' }, { id: '6' }], cursor: null },
    };
    const list = vi.fn(async (p: any) => pages[p.cursor ?? 'start']!);
    const repo = createContactRepository({ remote: memoryRemote({ list }), local: memoryLocal().local });

    expect((await repo.listContacts({ allPages: true })).contacts.map((c) => c.id)).toEqual(['1', '2', '3', '4', '5', '6']);
    expect((await repo.listContacts({ allPages: true, cap: 3 })).contacts.map((c) => c.id)).toEqual(['1', '2', '3']);
    expect(await repo.countContacts({ scoreBand: 'A' })).toBe(6);
    // one page only without allPages
    expect((await repo.listContacts({})).contacts).toHaveLength(2);
  });

  it("falls back to the phone's cache (with the error) when the server can't be reached", async () => {
    const { local } = memoryLocal({ contacts: [{ id: 'c1', eventId: 'Expo' }, { id: 'c2', eventId: 'Other' }] });
    const repo = createContactRepository({ remote: memoryRemote({ list: async () => { throw offline(); } }), local });

    const res = await repo.listContacts({ eventId: 'Expo' }, { fallbackToCache: true });
    expect(res.fromCache).toBe(true);
    expect(res.contacts.map((c) => c.id)).toEqual(['c1']);
    expect(res.error).toBeTruthy();
    expect(local.ensureOwner).toHaveBeenCalled();

    await expect(repo.listContacts({ eventId: 'Expo' })).rejects.toThrow('OFFLINE');
  });
});

describe('contact repository — writes', () => {
  it('an offline edit is saved on the phone and queued with the conflict base', async () => {
    const { local, contacts, queued } = memoryLocal({ contacts: [{ id: 'c1', firstName: 'Ann', email: 'old@x.io' }] });
    const onQueued = vi.fn();
    const repo = createContactRepository({ remote: memoryRemote({ update: async () => { throw offline(); } }), local, onQueued });

    const original = { id: 'c1', firstName: 'Ann', email: 'old@x.io' } as any;
    const saved = await repo.updateContact('c1', { email: 'new@x.io' }, { original });
    expect(saved.pendingSync).toBe(true);
    expect(contacts.get('c1').email).toBe('new@x.io');
    expect(queued).toHaveLength(1);
    expect(JSON.parse(queued[0]!.payload)).toEqual({ email: 'new@x.io', _base: { email: 'old@x.io' } });
    expect(onQueued).toHaveBeenCalled();
  });

  it('a server error is not queued', async () => {
    const { local, queued } = memoryLocal({ contacts: [{ id: 'c1' }] });
    const repo = createContactRepository({ remote: memoryRemote({ update: async () => { throw http(400, 'Bad email'); } }), local });
    await expect(repo.updateContact('c1', { email: 'x' })).rejects.toBeTruthy();
    expect(queued).toHaveLength(0);
  });

  it('discarding a scan removes it for good and tells the lists', async () => {
    const { local, contacts } = memoryLocal({ contacts: [{ id: 's1' }] });
    const purge = vi.fn(async () => ({}));
    const repo = createContactRepository({ remote: memoryRemote({ purge }), local });
    const seen: any[] = [];
    repo.events.subscribe((e) => seen.push(e));

    await repo.discardContact('s1');
    expect(purge).toHaveBeenCalledWith('s1');
    expect(contacts.has('s1')).toBe(false);
    expect(seen).toEqual([{ type: 'discarded', id: 's1' }]);
  });

  it('discarding a scan that only exists on this phone drops it and its queued upload', async () => {
    const create: OutboxItem = { ...queuedUpdate('s1'), action: 'create' };
    const { local, contacts, queued } = memoryLocal({ contacts: [{ id: 's1' }], queued: [create] });
    const repo = createContactRepository({ remote: memoryRemote({ delete: async () => { throw offline(); } }), local });

    await repo.discardContact('s1');
    expect(contacts.has('s1')).toBe(false);
    expect(queued).toHaveLength(0);
  });

  it('an offline delete removes the contact here and queues the delete', async () => {
    const { local, contacts, queued } = memoryLocal({ contacts: [{ id: 'c1' }] });
    const repo = createContactRepository({ remote: memoryRemote({ delete: async () => { throw offline(); } }), local });
    const seen: any[] = [];
    repo.events.subscribe((e) => seen.push(e));

    const res = await repo.deleteContact('c1', 'Ann');
    expect(res).toEqual({ daysLeft: 30, pendingSync: true });
    expect(contacts.has('c1')).toBe(false);
    expect(queued.map((q) => q.action)).toEqual(['delete']);
    expect(seen).toEqual([{ type: 'deleted', id: 'c1', name: 'Ann', daysLeft: 30 }]);
  });
});

describe('edit conflicts', () => {
  it('editBase takes only the patched conflict fields', () => {
    expect(editBase({ firstName: 'A', email: null, company: 'X' } as any, { email: 'e', photo: 'p' })).toEqual({ email: null });
  });

  it('getEditConflict reads the 409 detail', () => {
    const err = http(409, { message: 'Changed elsewhere', conflicts: [{ field: 'email', base: 'a', server: 'b', mine: 'c' }], contact: { id: 'c1' } });
    expect(getEditConflict(err)).toEqual({
      message: 'Changed elsewhere',
      conflicts: [{ field: 'email', base: 'a', server: 'b', mine: 'c' }],
      contact: { id: 'c1' },
    });
    expect(getEditConflict(http(409, 'Duplicate'))).toBeNull();
    expect(getEditConflict(http(500))).toBeNull();
  });
});
