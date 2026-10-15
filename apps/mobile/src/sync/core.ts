/**
 * Pure sync helpers shared by the sync engine, the contact repository and the offline scan queue
 * (no React Native / database imports, so they can be unit-tested).
 */
export { isNetworkError } from '../lib/apiError';

// A queued item's lastError starting with this didn't fail on its own (offline, timeout, waiting on
// an earlier change): it stays WAITING and keeps retrying automatically, unlike a real failure.
const RETRY_MARK = 'retry:';

export function retryableReason(reason: string): string {
  return RETRY_MARK + reason;
}

export function isRetryableReason(lastError: string | null | undefined): boolean {
  return !!lastError && lastError.startsWith(RETRY_MARK);
}

/** lastError as it should be shown (without the retry marker). */
export function reasonText(lastError: string | null | undefined): string | undefined {
  if (!lastError) return undefined;
  return isRetryableReason(lastError) ? lastError.slice(RETRY_MARK.length) : lastError;
}

// Hermes has no crypto.randomUUID; the server needs a real UUID to keep the same id.
export function uuidv4(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    return (ch === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

// ---- tiny event bus so screens refresh after a background sync ----
export type SyncEvent = { type: 'queued' } | { type: 'synced'; pushed: number; scans: number; failed: number };
const listeners = new Set<(e: SyncEvent) => void>();
export const syncEvents = {
  emit(e: SyncEvent) { listeners.forEach((l) => l(e)); },
  subscribe(l: (e: SyncEvent) => void) { listeners.add(l); return () => { listeners.delete(l); }; },
};

/** A queued contact change (outbox row). */
export interface OutboxItem {
  id: string;
  entityType: string;
  entityId: string;
  action: 'create' | 'update' | 'delete' | string;
  baseVersion?: number;
  payload: string;
  createdAt: string;
  attempts: number;
  lastError: string | null;
}

/** A scan photo waiting for the server to read it. */
export interface QueuedScan {
  id: string;
  kind: 'card' | 'qr';
  imageUri: string;
  eventName: string | null;
  createdAt: string;
  attempts: number;
  lastError: string | null;
}

/** Fields the server checks for "changed somewhere else" (API names). Defined once, here. */
export const CONFLICT_FIELDS = ['firstName', 'lastName', 'company', 'title', 'email', 'phone', 'notes', 'linkedin', 'eventId'] as const;
export type ConflictField = (typeof CONFLICT_FIELDS)[number];
