import { SESSION_REVOKED_DETAIL } from './apiError';

/**
 * The one request pipeline every backend call goes through — axios requests AND the two streamed
 * (XMLHttpRequest) endpoints: which headers a request carries (API key + bearer token), renewing an
 * expired access token once and replaying, and noticing a session that was signed out elsewhere.
 *
 * Pure module: storage and the refresh call are passed in (src/lib/api.ts wires the real ones).
 */

// A 401 from these is never "access token expired, renew and replay": they are the sign-in /
// renewal calls themselves (the session check doesn't enforce expiry).
export const NO_RENEW_ROUTES = ['/auth/refresh', '/auth/session', '/login/', '/register/'];

// Routes that don't take the backend API key.
const NO_API_KEY_ROUTES = ['/register/', '/login/', '/auth/', '/api/health'];

export interface TokenPair {
  token: string;
  refresh_token: string;
  expires_in?: number;
}

export interface PipelineDeps {
  getToken(): Promise<string | null>;
  getRefreshToken(): Promise<string | null>;
  saveTokens(access: string, refresh: string): Promise<void>;
  /** POST /auth/refresh (through the normal client, so a refused refresh is noticed as revoked) */
  postRefresh(refreshToken: string): Promise<TokenPair>;
  isTokenExpired(token: string): boolean;
  apiKey(): string | undefined;
  warn?(message: string): void;
}

export interface RequestPipeline {
  /** Authorization + X-API-Key for a request to `url`. `fresh`: renew an expired token first (streams). */
  headers(url: string, opts?: { fresh?: boolean }): Promise<Record<string, string>>;
  /**
   * A request got 401: the token to replay it with (renewed once, shared by every request that failed
   * at the same time), or null when it can't be recovered. A revoked session is reported here.
   */
  recoverFrom401(req: { url: string; detail: unknown; sentToken: string | null }): Promise<string | null>;
  /** True when a failure means this session is over (and reports it once). */
  checkRevoked(req: { url: string; status: number | null | undefined; detail: unknown }): boolean;
  /** Is a 401 on this route the session itself being refused? */
  isSessionRoute(url: string): boolean;
  refreshToken(refreshToken: string): Promise<TokenPair>;
  setSessionRevokedHandler(handler: (() => void) | null): void;
  resetSessionRevoked(): void;
}

export function createRequestPipeline(deps: PipelineDeps): RequestPipeline {
  let refreshInFlight: Promise<TokenPair> | null = null;
  let revokedHandler: (() => void) | null = null;
  let revokeNotified = false;
  let missingKeyWarned = false;

  const getToken = async () => {
    try { return await deps.getToken(); } catch { return null; }
  };

  // One renewal at a time: the session's resume check (src/features/session) and requests that hit an expired
  // token can ask together, and a second call with the already-used refresh token is wasted.
  const refreshToken = (rt: string): Promise<TokenPair> => {
    if (refreshInFlight) return refreshInFlight;
    refreshInFlight = (async () => {
      const response = await deps.postRefresh(rt);
      if (!response?.token || !response?.refresh_token) throw new Error('Token refresh returned no tokens');
      await deps.saveTokens(response.token, response.refresh_token);
      return response;
    })().finally(() => { refreshInFlight = null; });
    return refreshInFlight;
  };

  const renew = async (): Promise<string | null> => {
    const rt = await deps.getRefreshToken().catch(() => null);
    if (!rt) return null;
    return (await refreshToken(rt)).token;
  };

  const notifyRevoked = () => {
    // Several requests can fail at once; only sign out once.
    if (revokeNotified) return;
    revokeNotified = true;
    revokedHandler?.();
  };

  const isSessionRoute = (url: string) => url.includes('/auth/refresh') || url.includes('/auth/session');

  return {
    async headers(url, opts = {}) {
      const headers: Record<string, string> = {};
      let token = await getToken();
      if (opts.fresh && token && deps.isTokenExpired(token)) {
        try { token = (await renew()) || token; } catch { /* keep the old one: the 401 path renews */ }
      }
      if (token) headers.Authorization = `Bearer ${token}`;
      const needsApiKey = !NO_API_KEY_ROUTES.some((r) => url.includes(r));
      const apiKey = deps.apiKey();
      if (needsApiKey && apiKey) {
        headers['X-API-Key'] = apiKey;
      } else if (needsApiKey && !missingKeyWarned) {
        // Checked here (not at startup): in dev, Expo fills env values in through a
        // virtual module that may not be ready when this file first loads.
        missingKeyWarned = true;
        deps.warn?.(`[API] Sending ${url} without X-API-Key — EXPO_PUBLIC_API_KEY is not set in apps/mobile/.env. Set it to the backend APP_API_KEY and restart Metro with "npm start -- -c".`);
      }
      return headers;
    },

    async recoverFrom401({ url, detail, sentToken }) {
      if (detail === SESSION_REVOKED_DETAIL) { notifyRevoked(); return null; }
      if (NO_RENEW_ROUTES.some((r) => url.includes(r))) return null;
      // Already renewed by another request while this one was out: just replay with that token.
      const current = await getToken();
      if (current && current !== sentToken && !deps.isTokenExpired(current)) return current;
      return renew().catch(() => null);
    },

    checkRevoked({ url, status, detail }) {
      // Signed out elsewhere (e.g. password reset), or the session check / token refresh itself
      // was refused (expired / unknown refresh token): this session can't be recovered.
      if (status === 401 && (detail === SESSION_REVOKED_DETAIL || isSessionRoute(url))) {
        notifyRevoked();
        return true;
      }
      return false;
    },

    isSessionRoute,
    refreshToken,

    setSessionRevokedHandler(handler) {
      revokedHandler = handler;
    },

    resetSessionRevoked() {
      revokeNotified = false;
    },
  };
}
