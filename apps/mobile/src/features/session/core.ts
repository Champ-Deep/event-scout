import { decodeJWT, extractUserIdFromToken } from '../../lib/jwt';

/**
 * The session module: who is signed in on this phone, and everything that changes it — sign-in
 * (tokens + account + handing the offline cache to that account), sign-out (one path, also taken
 * by a session revoked on the server), renewing the sign-in before the access token runs out, and
 * the server's session check (which also picks up an admin renaming the user).
 *
 * Pure: its storage and backend are adapters (./index.ts wires MMKV, SecureStore, the request
 * pipeline and the SQLite cache owner; tests pass in-memory ones). Token renewal itself stays in
 * the request pipeline (single-flight, shared with requests that hit an expired token).
 */

export interface SessionState {
  isAuthenticated: boolean;
  userId: string | null;
  userEmail: string | null;
  userName: string | null;
  /** 'linkedin' | 'email' | 'invite' */
  authMethod: string | null;
}

/** What a sign-in / sign-up / password reset / LinkedIn exchange returned. */
export interface SignInResult {
  token: string;
  refreshToken: string;
  user?: { name?: string | null | undefined; email?: string | null | undefined } | null | undefined;
  authMethod?: string | null | undefined;
}

// ---- adapters ----

/** The signed-in account as kept for a synchronous read before first paint (MMKV). */
export interface StoredAccount {
  hasSession: boolean;
  userId: string;
  email: string;
  name: string;
  authMethod: string;
}

export interface AccountStore {
  read(): StoredAccount;
  write(patch: Partial<StoredAccount>): void;
  /** Forget every per-user key (identity, auth method, sync-time copies…). */
  clear(): void;
}

/** Access + refresh token (SecureStore). */
export interface TokenStore {
  access(): Promise<string | null>;
  refresh(): Promise<string | null>;
  save(access: string, refresh: string): Promise<void>;
  clear(): Promise<void>;
}

export interface SessionBackend {
  /** Renew the sign-in with this refresh token (the request pipeline's single-flight renewal; it saves the new tokens). */
  renew(refreshToken: string): Promise<unknown>;
  /** GET /auth/session — rejects when it can't be checked; a revoked session is reported to the revoked handler. */
  check(): Promise<{ name?: string | null | undefined } | null | undefined>;
  setRevokedHandler(handler: (() => void) | null): void;
  /** After a sign-in, so a later revocation is reported again. */
  resetRevoked(): void;
}

/** The phone's offline data (contact cache, outbox, offline scans, sync meta) and whose it is. */
export interface OfflineCache {
  /** Hand it to this account: kept when it's already theirs, wiped first when it was another account's. */
  handOver(userId: string): Promise<unknown>;
  /** Throw all of it away. */
  wipe(): Promise<void>;
}

export interface SessionDeps {
  account: AccountStore;
  tokens: TokenStore;
  backend: SessionBackend;
  cache: OfflineCache;
  now?: () => number;
}

// ---- interface ----

export interface Session {
  /** The current session (a new object whenever it changes). */
  current(): SessionState;
  subscribe(listener: () => void): () => void;
  /** Store the tokens and account, hand the offline cache to that account, then publish the session. */
  signIn(result: SignInResult): Promise<void>;
  /** End the session: tokens and every per-user key are cleared. Offline data is kept unless asked. */
  signOut(opts?: { keepOfflineData?: boolean }): Promise<void>;
  /** Called after a session revoked on the server was signed out (same signOut path). */
  onRevoked(listener: () => void): () => void;
  /** Renew the sign-in when the access token has expired or expires within RENEW_WITHIN_MS. */
  renewIfDue(): Promise<void>;
  /** Ask the server whether this session is still valid; keeps the stored name in step with it. */
  check(): Promise<void>;
  /** The account signed in right now (null when signed out or while signing out / in). */
  signedInUserId(): string | null;
  /**
   * For code that may run before a sign-in finished (sync runs, cache reads): waits for the hand-over
   * in progress and makes sure the offline cache belongs to the signed-in account. Resolves with that
   * account (null when signed out).
   */
  offlineCacheReady(): Promise<string | null>;
}

/** Access tokens live 12 h by default; renew in their last 2 h (not on every resume). */
export const RENEW_WITHIN_MS = 2 * 3600 * 1000;

const SIGNED_OUT: SessionState = Object.freeze({
  isAuthenticated: false, userId: null, userEmail: null, userName: null, authMethod: null,
});

function fromStored(a: StoredAccount): SessionState {
  if (!a.hasSession) return SIGNED_OUT;
  return {
    isAuthenticated: true,
    userId: a.userId || null,
    userEmail: a.email || null,
    userName: a.name || null,
    authMethod: a.authMethod || null,
  };
}

export function createSession(deps: SessionDeps): Session {
  const { account, tokens, backend, cache } = deps;
  const now = deps.now ?? (() => Date.now());

  let state: SessionState = fromStored(account.read());
  const listeners = new Set<() => void>();
  const revokedListeners = new Set<() => void>();
  // Bumped by every sign-in / sign-out: work started before (a session check, a renewal) must not
  // write the previous session's data back afterwards.
  let epoch = 0;
  let changing = false;
  let handingOver: Promise<unknown> = Promise.resolve();
  let renewing: Promise<void> | null = null;

  const publish = (next: SessionState) => {
    state = next;
    listeners.forEach((l) => l());
  };

  async function signIn(result: SignInResult) {
    epoch += 1;
    changing = true;
    try {
      await tokens.save(result.token, result.refreshToken);
      const userId = extractUserIdFromToken(result.token);
      const tokenEmail = decodeJWT(result.token)?.email || '';
      const stored: StoredAccount = {
        hasSession: true,
        userId: userId || '',
        email: result.user?.email || tokenEmail,
        name: result.user?.name || '',
        authMethod: result.authMethod || '',
      };
      account.write(stored);
      if (userId) {
        handingOver = cache.handOver(userId).catch(() => undefined);
        await handingOver;
      }
      publish(fromStored(stored));
      backend.resetRevoked();
    } finally {
      changing = false;
    }
  }

  async function signOut(opts: { keepOfflineData?: boolean } = {}) {
    epoch += 1;
    changing = true;
    try {
      await tokens.clear().catch(() => {});
      account.clear();
      if (opts.keepOfflineData === false) await cache.wipe().catch(() => {});
    } finally {
      changing = false;
      publish(SIGNED_OUT);
    }
  }

  // Signed out on another device (e.g. password reset), or the refresh token was refused.
  backend.setRevokedHandler(() => {
    signOut({ keepOfflineData: true }).finally(() => {
      revokedListeners.forEach((l) => l());
    });
  });

  function renewIfDue(): Promise<void> {
    if (renewing) return renewing;
    if (!state.isAuthenticated) return Promise.resolve();
    const started = epoch;
    renewing = (async () => {
      try {
        const [access, refresh] = await Promise.all([tokens.access(), tokens.refresh()]);
        const exp = access ? decodeJWT(access)?.exp : undefined;
        if (!refresh || !exp || exp * 1000 - now() > RENEW_WITHIN_MS) return;
        await backend.renew(refresh);
        // Signed out while the renewal was out: don't leave its new tokens behind.
        if (epoch !== started && !state.isAuthenticated) await tokens.clear();
      } catch { /* keep the current tokens; a revoked session signs out through the handler */ }
    })().finally(() => { renewing = null; });
    return renewing;
  }

  async function check() {
    if (!state.isAuthenticated) return;
    const started = epoch;
    let res: Awaited<ReturnType<SessionBackend['check']>>;
    try { res = await backend.check(); } catch { return; /* offline etc.: try again next time */ }
    // Signed out / into another account meanwhile: this answer was about the previous session.
    if (epoch !== started || !state.isAuthenticated) return;
    // An admin may have renamed this user: keep the stored name in step with the server.
    const name = typeof res?.name === 'string' ? res.name.trim() : '';
    if (name && name !== (state.userName ?? '')) {
      account.write({ name });
      publish({ ...state, userName: name });
    }
  }

  const signedInUserId = () => (changing || !state.isAuthenticated ? null : state.userId);

  return {
    current: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    signIn,
    signOut,
    onRevoked(listener) {
      revokedListeners.add(listener);
      return () => { revokedListeners.delete(listener); };
    },
    renewIfDue,
    check,
    signedInUserId,
    async offlineCacheReady() {
      await handingOver;
      const userId = signedInUserId();
      if (!userId) return null;
      await cache.handOver(userId).catch(() => undefined);
      return userId;
    },
  };
}
