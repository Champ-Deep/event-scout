import { eq } from 'drizzle-orm';
import * as LegacyFS from 'expo-file-system/legacy';
import { db } from '../db/client';
import { scanQueue } from '../db/schema';
import { syncEvents, uuidv4, type QueuedScan } from './core';

/**
 * Offline scans: when the server can't be reached, a scan keeps its photo in a queue and
 * `runSync` uploads it as soon as the connection is back (see sync/state.ts and AutoSync).
 * Offline contact changes (create / edit / delete) go through the contact repository
 * (src/features/contacts), which owns the contact cache and the outbox.
 */

export {
  isNetworkError, retryableReason, isRetryableReason, reasonText, uuidv4, syncEvents,
  type SyncEvent, type QueuedScan,
} from './core';

/** Keep the photo (camera files live in a temp cache) and queue it for the server to read.
 *  The row's id goes up as the scan's client_id (and becomes the contact's id), so uploading it
 *  twice — or after an online attempt with the same `id` that did reach the server — saves one
 *  contact. `eventName` / createdAt record the event and time of the capture. */
export async function queueScan(kind: 'card' | 'qr', imageUri: string, eventName?: string | null, id: string = uuidv4()): Promise<QueuedScan> {
  let stored = imageUri;
  try {
    const dir = `${LegacyFS.documentDirectory}scan-queue/`;
    await LegacyFS.makeDirectoryAsync(dir, { intermediates: true }).catch(() => {});
    const ext = /\.png($|\?)/i.test(imageUri) ? 'png' : 'jpg';
    stored = `${dir}${id}.${ext}`;
    await LegacyFS.copyAsync({ from: imageUri, to: stored });
  } catch (e: any) {
    console.warn('[offline] could not keep the scan photo, using the original file', e?.message);
    stored = imageUri;
  }
  const row: QueuedScan = { id, kind, imageUri: stored, eventName: eventName || null, createdAt: new Date().toISOString(), attempts: 0, lastError: null };
  await db.insert(scanQueue).values(row);
  syncEvents.emit({ type: 'queued' });
  return row;
}

export async function getQueuedScans(): Promise<QueuedScan[]> {
  try {
    const rows = await db.select().from(scanQueue);
    return (Array.isArray(rows) ? rows : []) as QueuedScan[];
  } catch {
    return [];
  }
}

/** `countsAsAttempt: false` = offline / timeout: shown, but doesn't use up an automatic retry. */
export async function markScanFailed(id: string, error: string, countsAsAttempt = true) {
  try {
    const rows = await db.select().from(scanQueue).where(eq(scanQueue.id, id)).limit(1);
    const attempts = ((rows?.[0] as any)?.attempts ?? 0) + (countsAsAttempt ? 1 : 0);
    await db.update(scanQueue).set({ attempts, lastError: error }).where(eq(scanQueue.id, id));
  } catch { /* */ }
}

export async function removeQueuedScan(id: string, deletePhoto = true) {
  try {
    const rows = await db.select().from(scanQueue).where(eq(scanQueue.id, id)).limit(1);
    const uri = (rows?.[0] as any)?.imageUri;
    await db.delete(scanQueue).where(eq(scanQueue.id, id));
    if (deletePhoto && uri && String(uri).includes('scan-queue/')) {
      await LegacyFS.deleteAsync(uri, { idempotent: true }).catch(() => {});
    }
  } catch { /* */ }
}
