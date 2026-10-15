import { describe, expect, it, vi } from 'vitest';
import {
  createSession, RENEW_WITHIN_MS,
  type AccountStore, type SessionBackend, type StoredAccount, type TokenStore,
} from '../src/features/session/core';
import { createOfflineCache, type CacheOwnerStore } from '../src/features/session/cacheOwner';

const jwt = (claims: Record<string, unknown>) =>
  `${btoa(JSON.stringify({ alg: 'HS256' }))}.${btoa(JSON.stringify(claims))}.sig`;

const hours = (h: number) => Math.floor((Date.now() + h * 3600 * 1000) / 1000);

const EMPTY: StoredAccount = { hasSession: false, userId: '', email: '', name: '', authMethod: '' };

function setup(opts: { stored?: Partial<StoredAccount>; tokens?: { access: string | null; refresh: string | null } } = {}) {
  // MMKV stand-in: the per-user session keys plus a "last synced" copy, to check sign-out clears them.
  const kv: StoredAccount & { lastSyncTime: string } = { ...EMPTY, ...opts.stored, lastSyncTime: '' };
  const account: AccountStore = {
    read: () => ({ hasSession: kv.hasSession, userId: kv.userId, email: kv.email, name: kv.name, authMethod: kv.authMethod }),
    write: (p) => { Object.assign(kv, p); },
    clear: () => { Object.assign(kv, EMPTY, { lastSyncTime: '' }); },
  };

  // SecureStore stand-in.
  const tok = { access: null as string | null, refresh: null as string | null, ...opts.tokens };
  const tokens: TokenStore = {
    access: async () => tok.access,
    refresh: async () => tok.refresh,
    save: async (a, r) => { tok.access = a; tok.refresh = r; },
    clear: async () => { tok.access = null; tok.refresh = null; },
  };

  // Backend: the pipeline's renewal saves the new tokens, like the real one.
  let revokedHandler: (() => void) | null = null;
  const backend = {
    renew: vi.fn(async (rt: string) => { tok.access = jwt({ sub: 'renewed', exp: hours(12) }); tok.refresh = `${rt}-next`; }),
    check: vi.fn(async (): Promise<{ name?: string }> => ({})),
    setRevokedHandler: (h: (() => void) | null) => { revokedHandler = h; },
    resetRevoked: vi.fn(),
  } satisfies SessionBackend;

  // SQLite stand-in: the offline outbox and its owner mark.
  const offline = { owner: null as string | null, outbox: [] as string[] };
  const ownerStore: CacheOwnerStore = {
    owner: async () => offline.owner,
    setOwner: async (id) => { offline.owner = id; },
    wipe: async () => { offline.owner = null; offline.outbox = []; },
  };

  const session = createSession({ account, tokens, backend, cache: createOfflineCache(ownerStore) });
  return { session, kv, tok, backend, offline, revoke: () => revokedHandler?.() };
}

const signInAs = (userId: string, extra: { name?: string; authMethod?: string } = {}) => ({
  token: jwt({ sub: userId, email: `${userId}@x.com`, exp: hours(12) }),
  refreshToken: `refresh-${userId}`,
  user: { name: extra.name ?? `Name ${userId}`, email: `${userId}@x.com` },
  authMethod: extra.authMethod ?? 'email',
});

describe('session', () => {
  it('signs in: stores tokens and account, publishes the session', async () => {
    const { session, kv, tok, backend } = setup();
    const seen = vi.fn();
    session.subscribe(seen);
    await session.signIn(signInAs('alice', { authMethod: 'linkedin' }));
    expect(session.current()).toEqual({
      isAuthenticated: true, userId: 'alice', userEmail: 'alice@x.com', userName: 'Name alice', authMethod: 'linkedin',
    });
    expect(kv).toMatchObject({ hasSession: true, userId: 'alice', name: 'Name alice', authMethod: 'linkedin' });
    expect(tok.refresh).toBe('refresh-alice');
    expect(session.signedInUserId()).toBe('alice');
    expect(seen).toHaveBeenCalled();
    expect(backend.resetRevoked).toHaveBeenCalledTimes(1);
  });

  it('starts from the stored account (before first paint)', () => {
    const { session } = setup({ stored: { hasSession: true, userId: 'bob', name: 'Bob', email: 'b@x.com', authMethod: 'email' } });
    expect(session.current()).toMatchObject({ isAuthenticated: true, userId: 'bob', userName: 'Bob' });
  });

  it('same account signs back in → its offline outbox is kept', async () => {
    const { session, offline } = setup();
    await session.signIn(signInAs('alice'));
    offline.outbox.push('unsynced-edit');
    await session.signOut({ keepOfflineData: true });
    expect(offline.outbox).toEqual(['unsynced-edit']);
    await session.signIn(signInAs('alice'));
    expect(offline.outbox).toEqual(['unsynced-edit']);
    expect(offline.owner).toBe('alice');
  });

  it('a different account signs in → the previous account\'s offline data is wiped first', async () => {
    const { session, offline } = setup();
    await session.signIn(signInAs('alice'));
    offline.outbox.push('alice-edit');
    await session.signOut();
    let outboxWhenPublished: string[] | null = null;
    session.subscribe(() => { if (session.current().isAuthenticated) outboxWhenPublished = [...offline.outbox]; });
    await session.signIn(signInAs('bob'));
    expect(offline.outbox).toEqual([]);
    expect(offline.owner).toBe('bob');
    expect(outboxWhenPublished).toEqual([]); // wiped before bob's session was visible
  });

  it('signOut({ keepOfflineData: false }) wipes the offline data', async () => {
    const { session, offline } = setup();
    await session.signIn(signInAs('alice'));
    offline.outbox.push('alice-edit');
    await session.signOut({ keepOfflineData: false });
    expect(offline).toEqual({ owner: null, outbox: [] });
  });

  it('signOut clears tokens and every session key, auth method and "last synced" included', async () => {
    const { session, kv, tok } = setup();
    await session.signIn(signInAs('alice', { authMethod: 'linkedin' }));
    kv.lastSyncTime = '2026-10-15T10:00:00Z';
    await session.signOut();
    expect(kv).toEqual({ ...EMPTY, lastSyncTime: '' });
    expect(tok).toEqual({ access: null, refresh: null });
    expect(session.current()).toEqual({
      isAuthenticated: false, userId: null, userEmail: null, userName: null, authMethod: null,
    });
    expect(session.signedInUserId()).toBeNull();
    expect(await session.offlineCacheReady()).toBeNull();
  });

  it('a session revoked on the server goes through signOut, then tells the app', async () => {
    const { session, kv, tok, revoke } = setup();
    await session.signIn(signInAs('alice', { authMethod: 'linkedin' }));
    const revoked = vi.fn(() => {
      // signed out already when the app is told (it navigates to the sign-in screen)
      expect(session.current().isAuthenticated).toBe(false);
    });
    session.onRevoked(revoked);
    revoke();
    await vi.waitFor(() => expect(revoked).toHaveBeenCalledTimes(1));
    expect(kv).toEqual({ ...EMPTY, lastSyncTime: '' });
    expect(tok).toEqual({ access: null, refresh: null });
  });

  it('renewIfDue renews once when the access token is about to expire', async () => {
    const { session, backend } = setup({
      stored: { hasSession: true, userId: 'alice' },
      tokens: { access: jwt({ sub: 'alice', exp: hours(1) }), refresh: 'r1' },
    });
    await Promise.all([session.renewIfDue(), session.renewIfDue()]);
    expect(backend.renew).toHaveBeenCalledTimes(1);
    expect(backend.renew).toHaveBeenCalledWith('r1');
  });

  it('renewIfDue does nothing while the access token is fresh (or signed out)', async () => {
    const fresh = setup({
      stored: { hasSession: true, userId: 'alice' },
      tokens: { access: jwt({ sub: 'alice', exp: hours(10) }), refresh: 'r1' },
    });
    await fresh.session.renewIfDue();
    expect(fresh.backend.renew).not.toHaveBeenCalled();
    expect(10 * 3600 * 1000).toBeGreaterThan(RENEW_WITHIN_MS);

    const out = setup({ tokens: { access: jwt({ sub: 'alice', exp: hours(1) }), refresh: 'r1' } });
    await out.session.renewIfDue();
    expect(out.backend.renew).not.toHaveBeenCalled();
  });

  it('a renewal that finishes after sign-out does not bring the tokens back', async () => {
    const { session, backend, tok } = setup({
      stored: { hasSession: true, userId: 'alice' },
      tokens: { access: jwt({ sub: 'alice', exp: hours(1) }), refresh: 'r1' },
    });
    let finish!: () => void;
    backend.renew.mockImplementationOnce(async () => {
      await new Promise<void>((r) => { finish = r; });
      tok.access = 'renewed'; tok.refresh = 'r2';
    });
    const renewal = session.renewIfDue();
    await vi.waitFor(() => expect(backend.renew).toHaveBeenCalled());
    await session.signOut();
    finish();
    await renewal;
    expect(tok).toEqual({ access: null, refresh: null });
  });

  it('check() keeps the stored name in step with the server (admin rename)', async () => {
    const { session, kv, backend } = setup();
    await session.signIn(signInAs('alice', { name: 'Alice' }));
    backend.check.mockResolvedValueOnce({ name: '  Alice Smith ' });
    await session.check();
    expect(session.current().userName).toBe('Alice Smith');
    expect(kv.name).toBe('Alice Smith');
  });

  it('check() answering after sign-out does not write the old name back', async () => {
    const { session, kv, backend } = setup();
    await session.signIn(signInAs('alice', { name: 'Alice' }));
    let answer!: (v: { name: string }) => void;
    backend.check.mockImplementationOnce(() => new Promise((r) => { answer = r; }));
    const checking = session.check();
    await session.signOut();
    answer({ name: 'Alice Smith' });
    await checking;
    expect(session.current().userName).toBeNull();
    expect(kv.name).toBe('');
  });

  it('check() failing (offline) changes nothing', async () => {
    const { session, backend } = setup();
    await session.signIn(signInAs('alice', { name: 'Alice' }));
    backend.check.mockRejectedValueOnce(new Error('Network Error'));
    await session.check();
    expect(session.current()).toMatchObject({ isAuthenticated: true, userName: 'Alice' });
  });

  it('offlineCacheReady hands an unclaimed cache to a restored session', async () => {
    const { session, offline } = setup({ stored: { hasSession: true, userId: 'alice' } });
    expect(await session.offlineCacheReady()).toBe('alice');
    expect(offline.owner).toBe('alice');
  });
});
