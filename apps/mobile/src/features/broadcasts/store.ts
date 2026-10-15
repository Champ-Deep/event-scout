import { useSyncExternalStore } from 'react';
import { AppState, Platform } from 'react-native';
import { api, type Broadcast } from '../../lib/api';

/**
 * Team announcements, shared by the Scout banner, the announcements dropdown and the pop-up that
 * shows on any tab. One live stream (GET /api/broadcasts/stream) tells the phone the moment an
 * admin sends or ends one; the list itself comes from GET /api/broadcasts/mine.
 */

// ---- small persisted values (this phone only) ----
function readKey<T>(key: string, fallback: T): T {
  try {
    const raw = Platform.OS === 'web' ? window.localStorage.getItem(key) : require('expo-secure-store').getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch { return fallback; }
}
function writeKey(key: string, value: unknown) {
  try {
    const raw = JSON.stringify(value);
    if (Platform.OS === 'web') window.localStorage.setItem(key, raw);
    else require('expo-secure-store').setItem(key, raw);
  } catch { /* not remembered: harmless */ }
}

const SEEN_KEY = 'eventscout.broadcasts.seenAt';

type State = {
  items: Broadcast[];
  loaded: boolean;
  live: boolean;              // the stream is connected
  seenAt: string | null;      // newest announcement the dropdown has shown
  menuOpen: boolean;
  menuSince: string | null;   // seenAt when the dropdown was opened: newer ones get a "New" tag
  popup: Broadcast | null;    // a just-arrived announcement shown over whatever tab is open
};

let state: State = { items: [], loaded: false, live: false, seenAt: readKey<string | null>(SEEN_KEY, null), menuOpen: false, menuSince: null, popup: null };
const listeners = new Set<() => void>();
function set(patch: Partial<State>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

export function useBroadcasts() {
  return useSyncExternalStore(
    (l) => { listeners.add(l); return () => { listeners.delete(l); }; },
    () => state,
  );
}

export const unreadCount = (s: State) =>
  s.items.filter((b) => !s.seenAt || new Date(b.createdAt ?? 0).getTime() > new Date(s.seenAt).getTime()).length;

let inflight: Promise<void> | null = null;
/** Reload the list. A live announcement that wasn't there before pops up (not on the first load). */
export function refreshBroadcasts(): Promise<void> {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const list = await api.getMyBroadcasts();
      const next = Array.isArray(list) ? list : [];
      const known = new Set(state.items.map((b) => b.id));
      const arrived = state.loaded ? next.find((b) => b.isActive && !known.has(b.id)) : undefined;
      // the pop-up goes away if that announcement was ended
      const popup = arrived ?? (state.popup && next.find((b) => b.id === state.popup!.id && b.isActive) ? state.popup : null);
      set({ items: next, loaded: true, popup });
    } catch { /* offline / server down: keep what's shown */ }
    finally { inflight = null; }
  })();
  return inflight;
}

export function openAnnouncements() {
  const newest = state.items[0]?.createdAt ?? state.seenAt;
  if (newest) writeKey(SEEN_KEY, newest);
  set({ menuOpen: true, menuSince: state.seenAt, popup: null, seenAt: newest });
}
export const closeAnnouncements = () => set({ menuOpen: false });

export function ago(iso: string | null) {
  if (!iso) return ''; // unknown time: say nothing rather than "1 Jan 1970"
  const ms = Date.now() - new Date(iso).getTime();
  if (!isFinite(ms)) return '';
  const m = Math.round(ms / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return new Date(iso).toLocaleDateString([], { day: 'numeric', month: 'short' });
}
export const dismissPopup = () => set({ popup: null });

// ---- live connection ----
const POLL_MS = 30_000;       // while the stream is down, check this often
const MAX_BACKOFF_MS = 30_000;

/** Connect while signed in; returns a function that disconnects. */
export function startLiveBroadcasts(): () => void {
  let alive = true;
  let closeStream: (() => void) | null = null;
  let retry: ReturnType<typeof setTimeout> | null = null;
  let poll: ReturnType<typeof setInterval> | null = null;
  let backoff = 2000;

  const stopPoll = () => { if (poll) clearInterval(poll); poll = null; };
  const startPoll = () => { if (!poll) poll = setInterval(refreshBroadcasts, POLL_MS); };

  const connect = async () => {
    if (!alive || closeStream) return;
    if (retry) { clearTimeout(retry); retry = null; }
    startPoll(); // until the stream says it's ready
    const close = await api.openBroadcastStream({
      onOpen: () => {
        backoff = 2000;
        stopPoll();
        set({ live: true });
        refreshBroadcasts(); // catch anything sent while disconnected
      },
      onChange: () => { refreshBroadcasts(); },
      onClose: async (status) => {
        closeStream = null;
        set({ live: false });
        if (!alive) return;
        startPoll();
        // expired sign-in: a normal request refreshes the token before we reconnect
        if (status === 401) await refreshBroadcasts();
        retry = setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      },
    });
    if (!alive) close(); else closeStream = close;
  };

  refreshBroadcasts();
  connect();

  // Android drops connections in the background: reconnect (and catch up) when the app is back.
  const appSub = AppState.addEventListener('change', (s) => {
    if (s === 'active') {
      refreshBroadcasts();
      backoff = 2000;
      connect();
    } else if (s === 'background') {
      closeStream?.();
      closeStream = null;
      if (retry) { clearTimeout(retry); retry = null; }
      stopPoll();
      set({ live: false });
    }
  });

  return () => {
    alive = false;
    appSub.remove();
    closeStream?.();
    closeStream = null;
    if (retry) clearTimeout(retry);
    stopPoll();
    set({ items: [], loaded: false, live: false, menuOpen: false, popup: null });
  };
}
