import { api } from '../lib/api';
import { createSyncEngine, type SyncRemote } from './engine';
import { localSyncStore } from './localStore';

/**
 * Client-side sync state: the sync engine (./engine) wired to the real backend and this phone's
 * local store. Sync state is inherently local: the pending work lives in the device's outbox
 * (SQLite), so it is derived here rather than from the server. On web (where the local DB is a
 * no-op mock) the outbox reads are wrapped defensively and simply report an empty queue.
 */

export {
  MAX_AUTO_ATTEMPTS,
  type SyncQueueEntry, type SyncState, type SyncPhase, type SkippedChange, type SyncRunResult,
} from './engine';

const remote: SyncRemote = {
  // Batches of up to 25 changes (photos included).
  push: (batch) => api.syncPush(batch),
  pull: (params) => api.syncPull(params),
  uploadScan: (s, userId) => {
    const opts = { clientId: s.id, eventName: s.eventName, capturedAt: s.createdAt };
    return s.kind === 'qr' ? api.scanQRCode(s.imageUri, userId, opts) : api.addContactFromImage(s.imageUri, userId, opts);
  },
  readCardOnDevice: async (s) => {
    const { saveCardReadOnDevice } = require('../features/scan/ocr/saveOnDevice');
    return !!(await saveCardReadOnDevice(s.imageUri, { online: true, clientId: s.id }).catch(() => null));
  },
};

const engine = createSyncEngine({ remote, store: localSyncStore });

/**
 * Push pending outbox items (only acknowledged ones leave the queue; failures are recorded
 * per item so they show as FAILED and retry next time), then pull every contact page into
 * the local cache. "Last synced" only moves when the whole run succeeded.
 */
export const runSync = engine.runSync;
export const isSyncRunning = engine.isSyncRunning;
export const getSyncState = engine.getSyncState;
/** When the contact list was last fully downloaded to this phone (null = never). */
export const getLastPullAt = engine.getLastPullAt;
/** Changes and offline scans on this phone that haven't reached the server yet (e.g. to warn
 *  before signing out). 0 when the local DB is unavailable. */
export const countPendingChanges = engine.countPendingChanges;
/** Throw away one queued change the user doesn't want to upload. */
export const discardQueueItem = engine.discardQueueItem;
