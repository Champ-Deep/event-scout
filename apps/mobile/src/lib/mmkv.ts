// Expo Go and Web lack native MMKV JSI bindings.
// Gracefully fallback to in-memory store so Expo Go can preview without crashing.
let storageInstance: {
  getBoolean: (key: string) => boolean | undefined;
  set: (key: string, value: boolean | string | number) => void;
  getString: (key: string) => string | undefined;
};

try {
  const { MMKV } = require('react-native-mmkv');
  storageInstance = new MMKV({
    id: 'eventscout-fast-storage',
  });
} catch {
  const memStore = new Map<string, any>();
  storageInstance = {
    getBoolean: (key: string) => memStore.get(key) as boolean | undefined,
    getString: (key: string) => memStore.get(key) as string | undefined,
    set: (key: string, value: any) => { memStore.set(key, value); },
  };
}

export const storage = storageInstance;

export const mmkvKeys = {
  HAS_SESSION: 'has_session',
  USER_ID: 'user_id',
  USER_EMAIL: 'user_email',
  USER_NAME: 'user_name',
  AUTH_METHOD: 'auth_method', // 'linkedin' | 'email' | 'invite'
  ACTIVE_EVENT_ID: 'active_event_id',
  ACTIVE_EVENT_NAME: 'active_event_name', // profile.current_event_name, remembered for offline scans
  SYNC_CURSOR: 'sync_cursor',
  LAST_SYNC_TIME: 'last_sync_time',
  LAST_SYNC_COUNT: 'last_sync_count',
  FEED_COLLAPSED_PREFIX: 'feed_collapsed_', // + shortcut group key ('event' | 'quick')
} as const;

export const fastStore = {
  hasSession: (): boolean => storage.getBoolean(mmkvKeys.HAS_SESSION) ?? false,
  setHasSession: (val: boolean) => storage.set(mmkvKeys.HAS_SESSION, val),

  getUserId: (): string | undefined => storage.getString(mmkvKeys.USER_ID),
  setUserId: (id: string) => storage.set(mmkvKeys.USER_ID, id),

  getUserEmail: (): string | undefined => storage.getString(mmkvKeys.USER_EMAIL),
  setUserEmail: (email: string) => storage.set(mmkvKeys.USER_EMAIL, email),

  getUserName: (): string | undefined => storage.getString(mmkvKeys.USER_NAME),
  setUserName: (name: string) => storage.set(mmkvKeys.USER_NAME, name),

  getAuthMethod: (): string | undefined => storage.getString(mmkvKeys.AUTH_METHOD),
  setAuthMethod: (method: string) => storage.set(mmkvKeys.AUTH_METHOD, method),

  // The active event's name lives in the profile; this copy is kept by src/features/profile.
  getActiveEventName: (): string | undefined => storage.getString(mmkvKeys.ACTIVE_EVENT_NAME) || undefined,
  setActiveEventName: (name: string) => storage.set(mmkvKeys.ACTIVE_EVENT_NAME, name),

  getSyncCursor: (): string | undefined => storage.getString(mmkvKeys.SYNC_CURSOR),
  setSyncCursor: (cursor: string) => storage.set(mmkvKeys.SYNC_CURSOR, cursor),

  getLastSyncTime: (): string | undefined => storage.getString(mmkvKeys.LAST_SYNC_TIME),
  setLastSyncTime: (iso: string) => storage.set(mmkvKeys.LAST_SYNC_TIME, iso),

  getLastSyncCount: (): number => Number(storage.getString(mmkvKeys.LAST_SYNC_COUNT) ?? '0') || 0,
  setLastSyncCount: (n: number) => storage.set(mmkvKeys.LAST_SYNC_COUNT, String(n)),

  // Open until the user collapses a group (then their choice is kept).
  isFeedGroupCollapsed: (group: string): boolean => storage.getBoolean(mmkvKeys.FEED_COLLAPSED_PREFIX + group) ?? false,
  setFeedGroupCollapsed: (group: string, collapsed: boolean) => storage.set(mmkvKeys.FEED_COLLAPSED_PREFIX + group, collapsed),
};
