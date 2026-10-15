import { Platform } from 'react-native';
import * as WebBrowser from 'expo-web-browser';
import * as Linking from 'expo-linking';
import { api, LinkedInSession, LinkedInLinkResult } from '../../lib/api';
import { generateUUID } from '../../lib/utils/uuid';

// Ensures the auth popup/redirect can hand control back to the app (no-op on native).
WebBrowser.maybeCompleteAuthSession();

const REQUEST_STORAGE_KEY = 'linkedin_oauth_request';

/**
 * - `login`: sign in / sign up with LinkedIn (issues a new session).
 * - `link`:  connect LinkedIn to the account that is already signed in.
 */
export type LinkedInIntent = 'login' | 'link';

export interface WebOAuthRequest {
  state: string | null;
  intent: LinkedInIntent;
  // In-app path to return to after a web-redirect link (e.g. '/me'); null = default.
  returnTo: string | null;
}

function persistWebRequest(state: string, intent: LinkedInIntent, returnTo?: string) {
  if (Platform.OS !== 'web') return;
  try {
    window.sessionStorage.setItem(REQUEST_STORAGE_KEY, JSON.stringify({ state, intent, returnTo: returnTo ?? null }));
  } catch {
    /* sessionStorage unavailable — state verification is best-effort */
  }
}

/**
 * Read (and clear) the OAuth request persisted before a web full-page redirect, so the
 * /callback route can verify the state LinkedIn echoes back and knows whether to sign in
 * or to link. Defaults to a login when nothing was stored.
 */
export function consumeWebOAuthRequest(): WebOAuthRequest {
  const fallback: WebOAuthRequest = { state: null, intent: 'login', returnTo: null };
  if (Platform.OS !== 'web') return fallback;
  try {
    const raw = window.sessionStorage.getItem(REQUEST_STORAGE_KEY);
    window.sessionStorage.removeItem(REQUEST_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return {
      state: typeof parsed?.state === 'string' ? parsed.state : null,
      intent: parsed?.intent === 'link' ? 'link' : 'login',
      // Only accept in-app paths, never an external URL.
      returnTo: typeof parsed?.returnTo === 'string' && parsed.returnTo.startsWith('/') && !parsed.returnTo.startsWith('//')
        ? parsed.returnTo
        : null,
    };
  } catch {
    return fallback;
  }
}

/**
 * Run the LinkedIn OpenID Connect authorization step and return the authorization code.
 *
 * - Web: performs a full-page redirect to LinkedIn and never resolves (the page navigates
 *   away); the `/callback` route completes the flow using the persisted intent.
 * - Native: opens an in-app auth session and resolves with the code, or `null` if the user
 *   cancels/dismisses the browser.
 */
async function authorizeLinkedIn(intent: LinkedInIntent, returnTo?: string): Promise<string | null> {
  const state = generateUUID();
  // Native: the server sends LinkedIn back to its own https return page, which bounces to
  // eventscout://callback?code&state — redirectUri is that app link (what the auth session waits for).
  const { url, redirectUri } = await api.getLinkedInAuthUrl(state, Platform.OS === 'web' ? undefined : 'native');

  if (Platform.OS === 'web') {
    persistWebRequest(state, intent, returnTo);
    window.location.assign(url);
    // Navigation replaces the page; nothing after this runs.
    return null;
  }

  const result = await WebBrowser.openAuthSessionAsync(url, redirectUri || 'eventscout://callback');
  if (result.type !== 'success' || !result.url) {
    // 'cancel' | 'dismiss' | 'locked' — user aborted, not an error.
    return null;
  }

  const { queryParams } = Linking.parse(result.url);
  const code = typeof queryParams?.code === 'string' ? queryParams.code : undefined;
  const returnedState = typeof queryParams?.state === 'string' ? queryParams.state : undefined;
  const error = typeof queryParams?.error === 'string' ? queryParams.error : undefined;

  if (error) {
    throw new Error(`LinkedIn sign-in was denied (${error}).`);
  }
  if (!code) {
    throw new Error('LinkedIn did not return an authorization code.');
  }
  if (returnedState && returnedState !== state) {
    throw new Error('LinkedIn sign-in failed a security check (state mismatch).');
  }
  return code;
}

/**
 * Start the LinkedIn login. Native resolves with the new JWT session (or `null` if
 * cancelled); web redirects away and `/callback` finishes.
 */
export async function beginLinkedInLogin(): Promise<LinkedInSession | null> {
  const code = await authorizeLinkedIn('login');
  return code ? api.exchangeLinkedInCode(code, nativePlatform()) : null;
}

/**
 * Connect LinkedIn to the currently signed-in account without switching sessions.
 * Native resolves with the link result (or `null` if cancelled); web redirects away
 * and `/callback` finishes the link, then navigates to `returnTo` (default: onboarding).
 */
export async function beginLinkedInLink(options: { returnTo?: string } = {}): Promise<LinkedInLinkResult | null> {
  const code = await authorizeLinkedIn('link', options.returnTo);
  return code ? api.linkLinkedIn(code, nativePlatform()) : null;
}

// The code from a native sign-in was issued for the server's https return page, so the token
// exchange must say so (`platform: 'native'`). Web sends nothing, as before.
function nativePlatform(): 'native' | undefined {
  return Platform.OS === 'web' ? undefined : 'native';
}
