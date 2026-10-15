import { describe, expect, it, vi } from 'vitest';
import { createSyncEngine, PUSH_BATCH, type SyncRemote, type SyncStore } from '../src/sync/engine';
import type { OutboxItem, QueuedScan } from '../src/sync/core';

// ─── in-memory adapters ─────────────────────────────────────────────────────

let clock = 0;
function item(entityId: string, over: Partial<OutboxItem> = {}): OutboxItem {
  clock += 1;
  return {
    id: over.id ?? `q${clock}`,
    entityType: 'contact',
    entityId,
    action: 'update',
    payload: JSON.stringify({ firstName: `Name ${entityId}` }),
    createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, clock)).toISOString(),
    attempts: 0,
    lastError: null,
    ...over,
  };
}

function memoryStore(init: { outbox?: OutboxItem[]; contacts?: any[]; scans?: QueuedScan[]; user?: string } = {}) {
  const outbox: OutboxItem[] = [...(init.outbox ?? [])];
  const contacts = new Map<string, any>((init.contacts ?? []).map((c) => [String(c.id), c]));
  const scans: QueuedScan[] = [...(init.scans ?? [])];
  const meta = new Map<string, string>();
  const who = { signedIn: (init.user ?? 'u1') as string | null, cacheOwner: (init.user ?? 'u1') as string | null };

  const store: SyncStore = {
    outbox: {
      all: async () => [...outbox].sort((a, b) => a.createdAt.localeCompare(b.createdAt)),
      entityIds: async () => outbox.map((i) => i.entityId),
      count: async () => outbox.length,
      acknowledge: async (ids) => {
        for (const id of ids) {
          const at = outbox.findIndex((i) => i.id === id);
          if (at >= 0) outbox.splice(at, 1);
        }
      },
      recordFailure: async (id, reason, countsAsAttempt) => {
        const it = outbox.find((i) => i.id === id);
        if (!it) return;
        it.attempts += countsAsAttempt ? 1 : 0;
        it.lastError = reason;
      },
    },
    contacts: {
      put: async (c) => { contacts.set(String(c.id), c); return true; },
      get: async (id) => contacts.get(id) ?? null,
      allIds: async () => [...contacts.keys()],
      deleteMany: async (ids) => { ids.forEach((id) => contacts.delete(id)); },
    },
    scans: {
      all: async () => [...scans],
      markFailed: async (id, reason, countsAsAttempt) => {
        const s = scans.find((x) => x.id === id);
        if (s) { s.attempts += countsAsAttempt ? 1 : 0; s.lastError = reason; }
      },
      remove: async (id) => {
        const at = scans.findIndex((x) => x.id === id);
        if (at >= 0) scans.splice(at, 1);
      },
    },
    meta: {
      get: async (k) => meta.get(k) ?? null,
      set: async (k, v) => { meta.set(k, v); },
    },
    owner: {
      ensure: async () => 'same',
      signedIn: async () => who.signedIn,
      cacheOwner: async () => who.cacheOwner,
    },
  };
  return { store, outbox, contacts, scans, meta, who };
}

/** Acknowledges everything it's sent unless told otherwise; one pull page with `pullContacts`. */
function memoryRemote(opts: {
  push?: (items: any[]) => any;
  pullPages?: Array<{ contacts: any[]; cursor: string | null; total?: number } | Error>;
} = {}) {
  const pushed: any[][] = [];
  const pulls: any[] = [];
  const remote: SyncRemote = {
    push: vi.fn(async (batch) => {
      pushed.push(batch.items);
      return opts.push ? opts.push(batch.items) : { acknowledgedIds: batch.items.map((i: any) => i.id) };
    }),
    pull: vi.fn(async (params) => {
      pulls.push(params);
      const pages = opts.pullPages ?? [{ contacts: [], cursor: null }];
      const page = pages[Math.min(pulls.length - 1, pages.length - 1)]!;
      if (page instanceof Error) throw page;
      return page;
    }),
    uploadScan: vi.fn(async () => ({ contact_id: 'new' })),
  };
  return { remote, pushed, pulls };
}

const offlineError = () => Object.assign(new Error('OFFLINE'), {});

// ─── tests ──────────────────────────────────────────────────────────────────

describe('sync engine — push', () => {
  it('sends the outbox in batches of 25 until it is empty', async () => {
    const items = Array.from({ length: 60 }, (_, i) => item(`c${i}`));
    const { store, outbox } = memoryStore({ outbox: items });
    const { remote, pushed } = memoryRemote();
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(PUSH_BATCH).toBe(25);
    expect(pushed.map((b) => b.length)).toEqual([25, 25, 10]);
    // oldest first, in order
    expect(pushed.flat().map((i) => i.id)).toEqual(items.map((i) => i.id));
    expect(outbox).toHaveLength(0);
    expect(res.pushed).toBe(60);
    expect(res.error).toBeUndefined();
  });

  it("holds back a contact's later changes while an earlier one failed (per-contact order)", async () => {
    const first = item('a', { id: 'a1' });
    const fillers = Array.from({ length: 24 }, (_, i) => item(`f${i}`));
    const later = item('a', { id: 'a2' });   // second batch, same contact as the failed one
    const other = item('b', { id: 'b1' });   // second batch, another contact
    const { store, outbox } = memoryStore({ outbox: [first, ...fillers, later, other] });
    const { remote, pushed } = memoryRemote({
      push: (batch) => ({
        acknowledgedIds: batch.filter((i: any) => i.id !== 'a1').map((i: any) => i.id),
        failed: batch.some((i: any) => i.id === 'a1') ? [{ id: 'a1', reason: 'Invalid email' }] : [],
      }),
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(pushed).toHaveLength(2);
    expect(pushed[1]!.map((i) => i.id)).toEqual(['b1']); // a2 waits for a1
    expect(outbox.map((i) => i.id).sort()).toEqual(['a1', 'a2']);
    const failed = outbox.find((i) => i.id === 'a1')!;
    expect(failed.attempts).toBe(1);
    expect(failed.lastError).toBe('Invalid email');
    expect(res.failed).toBe(1);
    expect(res.error).toBe("1 change couldn't be uploaded");
  });

  it('background sync skips an item that failed too often — and everything after it for that contact', async () => {
    const stuck = item('a', { id: 'a1', attempts: 3, lastError: 'Not allowed' });
    const later = item('a', { id: 'a2' });
    const other = item('b', { id: 'b1' });
    const { store, outbox } = memoryStore({ outbox: [stuck, later, other] });
    const { remote, pushed } = memoryRemote();
    await createSyncEngine({ remote, store, events: { emit() {} } }).runSync(undefined, { auto: true });

    expect(pushed.flat().map((i) => i.id)).toEqual(['b1']);
    expect(outbox.map((i) => i.id)).toEqual(['a1', 'a2']);

    const state = await createSyncEngine({ remote, store, events: { emit() {} } }).getSyncState();
    const entry = (id: string) => state.queue.find((q) => q.id === id)!;
    expect(entry('a1').status).toBe('FAILED');
    expect(entry('a1').autoRetry).toBe(false);
    expect(entry('a2').autoRetry).toBe(false); // waits on a1
  });

  it('drops acknowledged conflicts and reports why — including changes applied only in part', async () => {
    const ok = item('a', { id: 'i1' });
    const refused = item('b', { id: 'i2', payload: JSON.stringify({ firstName: 'Priya', lastName: 'Shah' }) });
    const partial = item('c', { id: 'i3', payload: JSON.stringify({ phone: '+1 555' }) });
    const partialWithMessage = item('d', { id: 'i4', payload: JSON.stringify({ notes: 'hi' }) });
    const shortened = item('e', { id: 'i5', payload: JSON.stringify({ firstName: 'Long', notes: 'x'.repeat(10) }) });
    const { store, outbox } = memoryStore({
      outbox: [ok, refused, partial, partialWithMessage, shortened],
      contacts: [{ id: 'c', firstName: 'Maya', lastName: 'Patel' }],
    });
    const { remote } = memoryRemote({
      push: () => ({
        acknowledgedIds: ['i1'],
        conflicts: [
          { id: 'i2', reason: 'not_owner' },
          { id: 'i3', reason: 'changed_elsewhere', fields: ['email', 'notes'] },
          { id: 'i4', reason: 'changed_elsewhere', fields: ['notes'], message: 'Notes merged with the newer text' },
          { id: 'i5', reason: 'notes_shortened', applied: true, message: 'Notes were cut to 20,000 characters' },
        ],
      }),
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(outbox).toHaveLength(0);
    expect(res.pushed).toBe(4);  // i1 + the three applied (in part / shortened) changes
    expect(res.skipped).toBe(1); // only the refused one
    expect(res.skippedItems).toEqual([
      { id: 'i2', details: 'Priya Shah', reason: 'Not your contact' },
      // no name in the change itself: the phone's copy of the contact names it
      { id: 'i3', details: 'Maya Patel', reason: 'Kept the newer email, notes changed elsewhere' },
      { id: 'i4', details: 'Contact change', reason: 'Notes merged with the newer text' },
      { id: 'i5', details: 'Long', reason: 'Notes were cut to 20,000 characters' },
    ]);
  });

  it('a batch that never reached the server keeps every item, without using up a retry', async () => {
    const items = [item('a'), item('b')];
    const { store, outbox } = memoryStore({ outbox: items });
    const { remote } = memoryRemote({ push: () => { throw offlineError(); } });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(outbox).toHaveLength(2);
    expect(outbox.every((i) => i.attempts === 0 && i.lastError === "retry:You're offline")).toBe(true);
    expect(res.offline).toBe(true);
    expect(res.error).toBe("You're offline");
  });
});

describe('sync engine — pull', () => {
  it('keeps the phone’s version of a contact whose change is still queued', async () => {
    const { store, contacts } = memoryStore({
      outbox: [item('c1')],
      contacts: [{ id: 'c1', firstName: 'Mine' }],
    });
    const { remote } = memoryRemote({
      push: () => { throw offlineError(); }, // the change stays queued
      pullPages: [{ contacts: [{ id: 'c1', firstName: 'Server' }, { id: 'c2', firstName: 'New' }], cursor: null, total: 2 }],
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(contacts.get('c1').firstName).toBe('Mine');
    expect(contacts.get('c2').firstName).toBe('New');
    expect(res.pulled).toBe(2);
    expect(res.cached).toBe(1);
  });

  it('removes contacts deleted elsewhere only after a complete, consistent pull', async () => {
    const { store, contacts } = memoryStore({ contacts: [{ id: 'gone' }, { id: 'c1' }] });
    const { remote } = memoryRemote({ pullPages: [{ contacts: [{ id: 'c1' }], cursor: null, total: 1 }] });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect([...contacts.keys()]).toEqual(['c1']);
    expect(res.removed).toBe(1);
  });

  it('removes nothing when the list changed while paging (counts disagree)', async () => {
    const { store, contacts } = memoryStore({ contacts: [{ id: 'gone' }] });
    const { remote } = memoryRemote({
      pullPages: [
        { contacts: [{ id: 'c1' }], cursor: 'p2', total: 3 },
        { contacts: [{ id: 'c2' }], cursor: null, total: 2 },
      ],
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(contacts.has('gone')).toBe(true);
    expect(res.removed).toBe(0);
  });

  it('removes nothing when the pull stopped part-way', async () => {
    const { store, contacts, meta } = memoryStore({ contacts: [{ id: 'gone' }] });
    const { remote } = memoryRemote({
      pullPages: [{ contacts: [{ id: 'c1' }], cursor: 'p2', total: 2 }, Object.assign(new Error('boom'), { response: { status: 500, data: {} } })],
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(contacts.has('gone')).toBe(true);
    expect(res.error).toBe('Download failed');
    expect(meta.get('last_pull_at')).toBeUndefined();
    expect(meta.get('last_sync_at')).toBeUndefined();
  });

  it('never removes a contact that is still waiting to upload', async () => {
    const { store, contacts } = memoryStore({
      outbox: [item('local', { action: 'create' })],
      contacts: [{ id: 'local' }],
    });
    const { remote } = memoryRemote({
      push: () => { throw offlineError(); },
      pullPages: [{ contacts: [], cursor: null, total: 0 }],
    });
    await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(contacts.has('local')).toBe(true);
  });
});

describe('sync engine — account switch', () => {
  it('stops when another account signs in mid-run: nothing more is uploaded, cached or recorded', async () => {
    const items = Array.from({ length: 30 }, (_, i) => item(`c${i}`));
    const { store, outbox, who, meta } = memoryStore({ outbox: items });
    const { remote, pushed, pulls } = memoryRemote({
      push: (batch) => {
        who.signedIn = 'u2'; // signed out / switched while the first batch was uploading
        return { acknowledgedIds: batch.map((i: any) => i.id) };
      },
    });
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(pushed).toHaveLength(1);
    expect(pulls).toHaveLength(0);
    expect(outbox).toHaveLength(5);
    expect(res.error).toBe('Sync stopped: you signed out');
    expect(meta.get('last_attempt_at')).toBeUndefined();
  });

  it('does not upload at all when nobody is signed in', async () => {
    const { store } = memoryStore({ outbox: [item('a')], user: 'u1' });
    const { remote, pushed } = memoryRemote();
    const s = store as any;
    s.owner.signedIn = async () => null;
    const res = await createSyncEngine({ remote, store, events: { emit() {} } }).runSync();

    expect(pushed).toHaveLength(0);
    expect(res.error).toBe('Sign in again to sync');
  });

  it('records a successful run', async () => {
    const { store, meta } = memoryStore({ outbox: [item('a')] });
    const { remote } = memoryRemote({ pullPages: [{ contacts: [{ id: 'a' }], cursor: null, total: 1 }] });
    const emitted: any[] = [];
    const res = await createSyncEngine({ remote, store, events: { emit: (e) => emitted.push(e) }, now: () => 'T' }).runSync();

    expect(res.error).toBeUndefined();
    expect(meta.get('last_sync_at')).toBe('T');
    expect(meta.get('last_sync_count')).toBe('2');
    expect(emitted).toEqual([{ type: 'synced', pushed: 1, scans: 0, failed: 0 }]);
  });
});
