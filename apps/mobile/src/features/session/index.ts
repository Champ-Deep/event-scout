import { api } from '../../lib/api';
import { fastStore } from '../../lib/mmkv';
import { secureStorage } from '../../lib/secure';
import { clearSyncTimeFallbacks, sqliteCacheOwnerStore } from '../../sync/owner';
import { createOfflineCache } from './cacheOwner';
import { createSession, type AccountStore, type TokenStore } from './core';

export {
  createSession, RENEW_WITHIN_MS,
  type Session, type SessionState, type SignInResult, type SessionDeps,
  type AccountStore, type TokenStore, type SessionBackend, type OfflineCache, type StoredAccount,
} from './core';
export { createOfflineCache, type CacheOwnerStore, type HandOver } from './cacheOwner';

/** MMKV: read synchronously before first paint. */
const account: AccountStore = {
  read: () => ({
    hasSession: fastStore.hasSession(),
    userId: fastStore.getUserId() ?? '',
    email: fastStore.getUserEmail() ?? '',
    name: fastStore.getUserName() ?? '',
    authMethod: fastStore.getAuthMethod() ?? '',
  }),
  write: (p) => {
    if (p.userId !== undefined) fastStore.setUserId(p.userId);
    if (p.email !== undefined) fastStore.setUserEmail(p.email);
    if (p.name !== undefined) fastStore.setUserName(p.name);
    if (p.authMethod !== undefined) fastStore.setAuthMethod(p.authMethod);
    if (p.hasSession !== undefined) fastStore.setHasSession(p.hasSession);
  },
  clear: () => {
    fastStore.setHasSession(false);
    fastStore.setUserId('');
    fastStore.setUserEmail('');
    fastStore.setUserName('');
    fastStore.setAuthMethod('');
    fastStore.setSyncCursor('');
    // "Last synced" copies: the next account must not see the previous one's. (The SQLite sync
    // meta stays with the offline data and is wiped when a different account signs in.)
    clearSyncTimeFallbacks();
  },
};

/** SecureStore. */
const tokens: TokenStore = {
  access: () => secureStorage.getAuthToken(),
  refresh: () => secureStorage.getRefreshToken(),
  save: async (access, refresh) => {
    await secureStorage.setAuthToken(access);
    await secureStorage.setRefreshToken(refresh);
  },
  clear: () => secureStorage.clearTokens(),
};

/**
 * The session (see ./core): the only way the app signs in / out, renews the sign-in and checks it
 * with the server. React screens read it through useSession() (src/providers/SessionProvider).
 */
export const session = createSession({
  account,
  tokens,
  backend: {
    renew: (refreshToken) => api.refreshToken(refreshToken),
    check: () => api.checkSession(),
    setRevokedHandler: (handler) => api.setSessionRevokedHandler(handler),
    resetRevoked: () => api.resetSessionRevoked(),
  },
  cache: createOfflineCache(sqliteCacheOwnerStore, (m) => console.log(m)),
});
