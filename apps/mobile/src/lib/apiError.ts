/**
 * The backend adapter's ONE error type. Every failure that leaves src/lib/api.ts (axios requests and
 * the two streamed endpoints) is an ApiError: classified once, with a ready display message and the
 * original response kept for code that needs more (e.g. the 409 conflict details).
 *
 * Pure module (no React Native imports) so it can be unit-tested.
 */

// Exact `detail` the backend sends (401) when a token was revoked, e.g. by a password reset
// on another device. Must match SESSION_REVOKED_DETAIL in apps/backend/app.py.
export const SESSION_REVOKED_DETAIL = 'Your session was signed out. Please sign in again.';

export type ApiErrorKind =
  | 'offline'      // the phone has no connection
  | 'unreachable'  // connected, but no answer came back (server down, DNS, dropped connection)
  | 'timeout'      // no answer in time
  | 'canceled'     // the app cancelled the request
  | 'revoked'      // signed out elsewhere / sign-in can't be renewed: the app signs out
  | 'unauthorized' // a 401 the token renewal couldn't fix (an expired token is renewed internally)
  | 'conflict'     // 409 "changed somewhere else", with the field conflicts
  | 'not_found'    // 404
  | 'validation'   // any other 4xx: `detail` says what's wrong
  | 'server';      // 5xx

export interface ApiErrorResponse {
  status: number;
  data: any;
  headers?: any;
}

export interface ApiFieldConflict {
  field: string;
  base: unknown;
  server: unknown;
  mine: unknown;
}

const BRAND = '__eventScoutApiError';

export class ApiError extends Error {
  readonly [BRAND] = true;
  readonly kind: ApiErrorKind;
  /** HTTP status, null when no answer came back */
  readonly status: number | null;
  /** the server's `detail` (string, list or object), if any */
  readonly detail: unknown;
  /** the original response (axios shape: status + data) */
  readonly response: ApiErrorResponse | undefined;
  /** axios error code, e.g. ECONNABORTED */
  readonly code: string | undefined;
  /** 409 field conflicts (kind 'conflict') */
  readonly conflicts: ApiFieldConflict[];
  /** a sentence that can be shown as is */
  readonly displayMessage: string;
  /** the error this was made from */
  readonly original: unknown;

  constructor(init: {
    kind: ApiErrorKind; message: string; status?: number | null; detail?: unknown;
    response?: ApiErrorResponse | undefined; code?: string | undefined; conflicts?: ApiFieldConflict[]; original?: unknown;
  }) {
    super(init.message);
    Object.setPrototypeOf(this, ApiError.prototype);
    this.name = 'ApiError';
    this.kind = init.kind;
    this.status = init.status ?? null;
    this.detail = init.detail;
    this.response = init.response;
    this.code = init.code;
    this.conflicts = init.conflicts ?? [];
    this.original = init.original;
    this.displayMessage = defaultMessage(init.kind, init.detail);
  }
}

export function isApiError(e: unknown): e is ApiError {
  return !!e && typeof e === 'object' && (e as any)[BRAND] === true;
}

const NETWORK_KINDS: ApiErrorKind[] = ['offline', 'unreachable', 'timeout'];

function defaultMessage(kind: ApiErrorKind, detail: unknown): string {
  const d = typeof detail === 'string' && detail ? detail : null;
  switch (kind) {
    case 'offline': return "You're offline.";
    case 'unreachable': return "Can't reach the server.";
    case 'timeout': return 'That took too long. Please try again.';
    case 'canceled': return 'Cancelled.';
    case 'revoked': return d || SESSION_REVOKED_DETAIL;
    case 'unauthorized': return d || 'Your session has ended. Please sign in again.';
    case 'conflict': {
      const m = detail && typeof detail === 'object' ? (detail as any).message : null;
      return typeof m === 'string' && m ? m : 'This contact was changed somewhere else';
    }
    default: return d || 'Something went wrong. Please try again.';
  }
}

export interface ClassifyContext {
  /** the phone reports no connection right now (only consulted when no answer came back) */
  deviceOffline?: boolean;
  /** a 401 on this request means the session can't be recovered (sign-in renewal / session check) */
  sessionRoute?: boolean;
}

/** Turn whatever a request threw (axios error, XHR failure, legacy shapes) into an ApiError. */
export function classifyError(raw: unknown, ctx: ClassifyContext = {}): ApiError {
  if (isApiError(raw)) return raw;
  const e: any = raw ?? {};
  const message = String(e?.message ?? 'Request failed');
  const code: string | undefined = typeof e?.code === 'string' ? e.code : undefined;
  const res = e?.response;
  const status: number | null = typeof res?.status === 'number' ? res.status : null;
  const response: ApiErrorResponse | undefined = res ? { status: res.status, data: res.data, headers: res.headers } : undefined;
  const detail = res?.data?.detail;
  const base = { status, detail, response, code, original: raw };

  if (!res || status == null) {
    if (message === 'OFFLINE' || ctx.deviceOffline) return new ApiError({ ...base, kind: 'offline', message: 'OFFLINE' });
    if (code === 'ERR_CANCELED' || e?.name === 'CanceledError') return new ApiError({ ...base, kind: 'canceled', message });
    if (code === 'ECONNABORTED' || code === 'ETIMEDOUT' || /timeout/i.test(message)) return new ApiError({ ...base, kind: 'timeout', message });
    return new ApiError({ ...base, kind: 'unreachable', message });
  }
  if (status === 401) {
    const revoked = detail === SESSION_REVOKED_DETAIL || !!ctx.sessionRoute;
    return new ApiError({ ...base, kind: revoked ? 'revoked' : 'unauthorized', message });
  }
  if (status === 404) return new ApiError({ ...base, kind: 'not_found', message });
  if (status === 409) {
    const conflicts = Array.isArray(detail?.conflicts) ? detail.conflicts : null;
    if (conflicts?.length) return new ApiError({ ...base, kind: 'conflict', message, conflicts });
  }
  if (status >= 500) return new ApiError({ ...base, kind: 'server', message });
  return new ApiError({ ...base, kind: 'validation', message });
}

/** Thrown by app code itself (not a request): its message is meant for the user. */
function isLocalError(e: any): boolean {
  if (!e || typeof e !== 'object' || isApiError(e)) return false;
  if (e.response || e.message === 'OFFLINE') return false;
  if (typeof e.code === 'string' && /^(ECONN|ERR_|ETIMEDOUT)/.test(e.code)) return false;
  return typeof e.message === 'string' && !/network|timeout|failed to fetch|connect/i.test(e.message);
}

/** The ApiError behind `e` (also for errors in the old axios shape); null for errors app code threw. */
export function asApiError(e: unknown): ApiError | null {
  if (isApiError(e)) return e;
  if (!e || typeof e !== 'object' || isLocalError(e)) return null;
  return classifyError(e);
}

/** No connection on the phone. */
export function isOffline(e: unknown): boolean {
  return asApiError(e)?.kind === 'offline';
}

/** The request never reached the server (offline, unreachable, timed out) — as opposed to the
 *  server answering with an error, which must be shown, not queued. */
export function isNetworkError(e: unknown): boolean {
  const err = asApiError(e);
  return !!err && NETWORK_KINDS.includes(err.kind);
}

export function errorKind(e: unknown): ApiErrorKind | null {
  return asApiError(e)?.kind ?? null;
}

export interface ErrorMessageOptions {
  /** shown when the phone is offline (default "You're offline.") */
  offline?: string;
  /** no answer from the server (default "Can't reach the server.") */
  unreachable?: string;
  /** timed out (default: the `unreachable` text) */
  timeout?: string;
  /** 401 the renewal couldn't fix; takes priority over the server's detail */
  unauthorized?: string;
  /** 413: the upload was too large */
  tooLarge?: string;
  /** 422 whose detail is FastAPI's list of field errors (not a sentence) */
  invalid?: string;
  /** 501: the server doesn't offer this */
  notAvailable?: string;
}

/**
 * The sentence to show for a failed request. Screens pass their own fallback (and, where they say
 * more, their own offline / unreachable wording); a string `detail` from the server wins otherwise.
 * An error the app threw itself shows its own message.
 */
export function errorMessage(e: unknown, fallback = 'Something went wrong. Please try again.', opts: ErrorMessageOptions = {}): string {
  const err = asApiError(e);
  if (!err) {
    const m = (e as any)?.message;
    return typeof m === 'string' && m ? m : fallback;
  }
  const unreachable = opts.unreachable ?? "Can't reach the server.";
  switch (err.kind) {
    case 'offline': return opts.offline ?? "You're offline.";
    case 'timeout': return opts.timeout ?? unreachable;
    case 'unreachable': return unreachable;
    case 'canceled': return fallback;
    case 'unauthorized':
    case 'revoked':
      if (opts.unauthorized) return opts.unauthorized;
      break;
    case 'conflict': return err.displayMessage;
    default: break;
  }
  if (err.status === 413 && opts.tooLarge) return opts.tooLarge;
  if (err.status === 501 && opts.notAvailable) return opts.notAvailable;
  if (Array.isArray(err.detail) && opts.invalid) return opts.invalid;
  if (err.detail === 'Invalid API Key') {
    return "The app couldn't authenticate with the server (API key mismatch). Please contact support.";
  }
  return typeof err.detail === 'string' && err.detail ? err.detail : fallback;
}
