import { useEffect, useRef } from 'react';
import { AppState } from 'react-native';
import { getLastPullAt, getSyncState, runSync, type SyncQueueEntry } from './state';
import { isOnline, onConnectivityChange } from './network';
import { syncEvents } from './offline';
import { logger } from '../lib/log';
import { useSession } from '../providers/SessionProvider';

const RETRY_MS = 60_000;            // offline / waiting to upload: check again every minute
const FAILED_RETRY_MS = 10 * 60_000; // server said no (bad key, unreadable photo…): at most every 10 min
const REFRESH_MS = 15 * 60_000;      // nothing to upload: still download the latest contacts this often
const MIN_GAP_MS = 2 * 60_000;       // …but not again right after a run (e.g. the download keeps failing)

/** Items background sync still uploads by itself (not the ones that failed too often). */
function autoItems(queue: SyncQueueEntry[]) {
  return queue.filter((q) => q.autoRetry !== false);
}

function age(iso: string | null): number {
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isFinite(t) ? Date.now() - t : Infinity;
}

/**
 * Background sync: whenever the phone has changes or scans saved offline, upload them as soon
 * as the connection is back (reconnect, app returning to the foreground, or a periodic retry).
 * With nothing to upload it still downloads the contact list every ~15 min, so the offline copy
 * stays current. Renders nothing.
 */
export function AutoSync() {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFailedTry = useRef(0);
  const { userId } = useSession();

  useEffect(() => {
    if (!userId) return; // signed out: nothing to upload or download
    let alive = true;

    const schedule = (ms: number) => {
      if (!alive) return;
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => { timer.current = null; attempt('retry'); }, Math.max(5_000, ms));
    };

    const attempt = async (why: string) => {
      if (!alive) return;
      let next = REFRESH_MS;
      try {
        // (The session handed the offline cache to this account on sign-in; getSyncState / runSync
        // still wait for that through the session before reading or uploading anything.)
        const state = await getSyncState();
        const items = autoItems(state.queue);
        const fresh = items.some((q) => q.status === 'PENDING');
        const sincePull = age(await getLastPullAt());
        const pullDue = sincePull >= REFRESH_MS;
        if (!items.length && !pullDue) { next = REFRESH_MS - sincePull; return; }
        if (!(await isOnline())) { next = RETRY_MS; return; }
        if (!fresh) {
          // Only items the server already refused (or just a download)? Don't hammer it.
          if (age(state.lastAttemptAt) < MIN_GAP_MS) { next = items.length ? FAILED_RETRY_MS : MIN_GAP_MS; return; }
          if (items.length && !pullDue && Date.now() - lastFailedTry.current < FAILED_RETRY_MS) { next = FAILED_RETRY_MS; return; }
          if (items.length) lastFailedTry.current = Date.now();
        }
        logger.info(`[AutoSync] ${why}: ${items.length ? `uploading ${items.length} pending item(s)` : 'refreshing the offline copy'}`);
        const res = await runSync(undefined, { auto: true });
        // Connection problems retry soon; a server "no" waits longer (or a manual Retry); after
        // 3 such answers an item is left for the Sync page.
        const left = autoItems(res.queue);
        next = res.offline || left.some((q) => q.status === 'PENDING') ? RETRY_MS
          : left.length ? FAILED_RETRY_MS : REFRESH_MS;
      } catch (e: any) {
        logger.warn('[AutoSync] failed', { error: e?.message });
        next = RETRY_MS;
      } finally {
        schedule(next);
      }
    };

    attempt('startup');
    const offNet = onConnectivityChange((online) => { if (online) attempt('back online'); });
    const appSub = AppState.addEventListener('change', (s) => { if (s === 'active') attempt('foreground'); });
    // Something new was saved offline: try right away (the connection may be flaky, not gone).
    const offQueued = syncEvents.subscribe((e) => { if (e.type === 'queued') setTimeout(() => attempt('queued'), 1500); });

    return () => {
      alive = false;
      offNet();
      appSub.remove();
      offQueued();
      if (timer.current) clearTimeout(timer.current);
      timer.current = null;
    };
  }, [userId]);

  return null;
}
