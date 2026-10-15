// Admin console session: the real JWT pair from POST /login/, sent on every backend call.
// The backend checks the token and that the account is an admin (403 otherwise).

export interface AdminSession {
  token: string;
  refreshToken: string;
  userId: string;
  name: string;
  email: string;
}

// Backend origin. Unset in dev (the Vite proxy forwards /api, /login/, /auth/*) and when the backend serves
// the built console itself; set VITE_API_URL at build time when the console is hosted on another origin.
const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '');

/** Absolute backend URL for a path like '/api/contacts' (unchanged when the console shares the backend's origin). */
export function apiUrl(path: string): string {
  return /^https?:\/\//.test(path) ? path : `${API_BASE}${path}`;
}

const STORAGE_KEY = 'es_admin_session';
// Keys written by the old placeholder login; cleared so a stale fake token can't linger.
const LEGACY_KEYS = ['admin_token', 'admin_user_id', 'admin_name', 'admin_email', 'admin_is_admin'];

let current: AdminSession | null = load();

function load(): AdminSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as AdminSession) : null;
    return parsed?.token && parsed.refreshToken ? parsed : null;
  } catch {
    return null;
  }
}

function save(session: AdminSession | null) {
  current = session;
  try {
    if (session) localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    else localStorage.removeItem(STORAGE_KEY);
    LEGACY_KEYS.forEach((k) => localStorage.removeItem(k));
  } catch {
    // Storage blocked: the session still lives in memory for this tab.
  }
}

export function getSession(): AdminSession | null {
  return current;
}

export function isSignedIn(): boolean {
  return current !== null;
}

// Components that show who's signed in (the sidebar account button) re-render when it changes.
const listeners = new Set<() => void>();
export function subscribeSession(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}

/** Change the signed-in session in place (new name after a profile edit, new tokens after a password change). */
export function updateSession(patch: Partial<AdminSession>) {
  if (!current) return;
  save({ ...current, ...patch });
  listeners.forEach((fn) => fn());
}

function goToLogin() {
  if (window.location.pathname !== '/login') {
    const next = window.location.pathname + window.location.search;
    window.location.assign(`/login?next=${encodeURIComponent(next)}`);
  }
}

export function signOut() {
  save(null);
  goToLogin();
}

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
    // FastAPI validation errors: [{ loc, msg }, …]
    if (Array.isArray(data?.detail) && data.detail.length) {
      const first = data.detail[0];
      const field = Array.isArray(first?.loc) ? first.loc[first.loc.length - 1] : '';
      if (field === 'email') return 'Enter a valid email address.';
      if (typeof first?.msg === 'string') return first.msg;
    }
  } catch {
    // not JSON
  }
  return fallback;
}

/** Sign in with email + password. Rejects accounts that aren't admins. */
export async function signIn(email: string, password: string): Promise<AdminSession> {
  const res = await fetch(apiUrl('/login/'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: email.trim(), password }),
  });
  if (!res.ok) throw new Error(await readError(res, 'Sign in failed'));
  const data = await res.json();
  if (!data.token || !data.refresh_token) throw new Error('The server did not return a login token');
  if (!data.is_admin) throw new Error('This account does not have admin access');
  const session: AdminSession = {
    token: data.token,
    refreshToken: data.refresh_token,
    userId: data.user_id,
    name: data.name || '',
    email: data.email || email.trim(),
  };
  save(session);
  return session;
}

/**
 * Forgot password, step 1: the server emails a 6-digit code (POST /auth/password/forgot — the app's flow).
 * Answers the same whether or not the email has an account.
 */
export async function requestPasswordReset(email: string): Promise<{ message: string; expiresIn: number; resendAfter: number }> {
  const res = await fetch(apiUrl('/auth/password/forgot'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: email.trim() }),
  });
  if (!res.ok) throw new Error(await readError(res, "Couldn't send a reset code"));
  const data = await res.json();
  return { message: data.message || '', expiresIn: data.expires_in ?? 900, resendAfter: data.resend_after ?? 60 };
}

/**
 * Forgot password, step 2: the code + a new password (POST /auth/password/reset). Every other device is
 * signed out; an admin account is signed in here with the fresh tokens it returns.
 */
export async function resetPasswordWithCode(email: string, code: string, newPassword: string): Promise<AdminSession> {
  const res = await fetch(apiUrl('/auth/password/reset'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: email.trim(), code: code.trim(), new_password: newPassword }),
  });
  if (!res.ok) throw new Error(await readError(res, "Couldn't reset the password"));
  const data = await res.json();
  if (!data.is_admin) throw new Error('Your password was changed, but this account does not have admin access.');
  if (!data.token || !data.refresh_token) throw new Error('The server did not return a login token');
  const session: AdminSession = { token: data.token, refreshToken: data.refresh_token, userId: data.user_id, name: data.name || '', email: data.email || email.trim() };
  save(session);
  return session;
}

let refreshing: Promise<boolean> | null = null;

/** Swap the refresh token for a new pair. One refresh at a time, shared by concurrent calls. */
function refresh(): Promise<boolean> {
  if (!current) return Promise.resolve(false);
  if (!refreshing) {
    const refreshToken = current.refreshToken;
    refreshing = fetch(apiUrl('/auth/refresh'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refreshToken }),
    })
      .then(async (res) => {
        if (!res.ok || !current) return false;
        const data = await res.json();
        if (!data.token || !data.refresh_token) return false;
        save({ ...current, token: data.token, refreshToken: data.refresh_token });
        return true;
      })
      .catch(() => false)
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

function withAuth(init: RequestInit | undefined, token: string): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${token}`);
  // Asks the /api routes for the cross-user admin view (only honoured for admin tokens).
  headers.set('X-Admin-Console', '1');
  return { ...init, headers };
}

/**
 * fetch() for backend calls: adds the admin token, refreshes it once on a 401, and sends the
 * user back to /login when the session can't be recovered. Resolves like fetch (check res.ok).
 */
export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  if (!current) {
    goToLogin();
    throw new Error('Not signed in');
  }
  let res = await fetch(apiUrl(input), withAuth(init, current.token));
  if (res.status === 401) {
    if (await refresh()) {
      res = await fetch(apiUrl(input), withAuth(init, current!.token));
    }
    if (res.status === 401) {
      save(null);
      goToLogin();
    }
  }
  return res;
}

/**
 * URL for EventSource, which can't send headers. The admin's token never goes in the URL (it would
 * land in server logs): a short-lived, single-use ticket from POST /api/admin/live/ticket does
 * (valid 60 s), so a fresh one is fetched before every (re)connect. null = not signed in;
 * throws when the ticket can't be had (the caller retries).
 */
export async function streamUrl(path: string): Promise<string | null> {
  if (!current) return null;
  const res = await apiFetch('/api/admin/live/ticket', { method: 'POST' });
  if (!res.ok) throw new Error(`Live ticket failed (${res.status})`);
  const data = (await res.json()) as { ticket?: string };
  if (!data.ticket) throw new Error('The server did not return a live ticket');
  const sep = path.includes('?') ? '&' : '?';
  return `${apiUrl(path)}${sep}ticket=${encodeURIComponent(data.ticket)}`;
}
