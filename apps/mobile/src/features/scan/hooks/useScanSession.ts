import { useState, useCallback } from 'react';
import type { Contact } from '@es/types';
import { api, asApiError, errorKind } from '../../../lib/api';
import { useSession } from '../../../providers/SessionProvider';
import { session } from '../../session';
import { isNetworkError, queueScan, uuidv4 } from '../../../sync/offline';
import { isOnline } from '../../../sync/network';
import { saveCardReadOnDevice, type OnDeviceSaved } from '../ocr/saveOnDevice';
import { profileStore } from '../../profile';

export interface ScanOutcome {
  // 'created' = a new contact was saved; 'duplicate' = it matched one you already have;
  // 'queued' = offline — the photo is kept on the phone and read when the connection is back.
  status: 'created' | 'duplicate' | 'queued';
  contactId: string;
  existingName: string | null;
  matchedOn: string | null;
  autoScoreStarted: boolean;
  /** read on this phone (PaddleOCR) instead of by the server */
  readOnDevice?: boolean;
  /** saved on the phone only — uploads when back online */
  pendingSync?: boolean;
}

function fromOnDevice(r: OnDeviceSaved): ScanOutcome {
  if (r.duplicateOf) {
    return { status: 'duplicate', contactId: r.duplicateOf.id, existingName: r.duplicateOf.name, matchedOn: r.duplicateOf.matchedOn, autoScoreStarted: false, readOnDevice: true };
  }
  return { status: 'created', contactId: r.contactId, existingName: null, matchedOn: null, autoScoreStarted: false, readOnDevice: true, pendingSync: r.pendingSync };
}

function scanErrorMessage(err: any, kind: 'card' | 'qr'): string {
  const e = asApiError(err);
  const failed = kind === 'card' ? 'Failed to read the card. Please try again.' : 'Failed to scan the code. Please try again.';
  if (!e) return err?.message || failed; // the app's own message (e.g. "Please log in to scan cards.")
  if (e.kind === 'offline') return "You're offline. Scanning needs a connection to read the card.";
  if (e.kind === 'timeout') return 'Reading the card took too long. Try again with a sharper, well-lit photo.';
  const detail = e.detail;
  // 503 = server setup / AI service problem (bad key, rate limit): show it as-is — retaking
  // the photo won't help.
  if (e.status === 503 && typeof detail === 'string') return detail;
  if (typeof detail === 'string') {
    if (/No QR code found/i.test(detail)) return "No QR code found. Hold the badge's QR code inside the frame.";
    if (/does not contain vCard/i.test(detail)) return "That QR code isn't a contact card (vCard).";
    if (/No contact information found/i.test(detail)) return "Couldn't find contact details on that card. Try a sharper, well-lit photo.";
    return detail;
  }
  if (e.kind === 'unreachable' || e.kind === 'canceled') return "Couldn't reach the server. Check your connection.";
  return failed;
}

export function useScanSession() {
  const [scanning, setScanning] = useState(false);
  const [lastScanned, setLastScanned] = useState<Contact | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated } = useSession();

  // Runs the real server-side scan and returns what happened. There is no on-device OCR,
  // so a failure surfaces the server's real error rather than fabricating placeholder data.
  const runScan = useCallback(async (kind: 'card' | 'qr', imageUri: string): Promise<ScanOutcome> => {
    setScanning(true);
    setError(null);
    try {
      if (!isAuthenticated) {
        throw new Error(kind === 'card' ? 'Please log in to scan cards.' : 'Please log in to scan codes.');
      }
      // the signed-in account (waits for a sign-in hand-over in progress, like sync does)
      const userId = await session.offlineCacheReady();
      if (!userId) {
        throw new Error('Your session has expired. Please log in again.');
      }

      // One id for this capture, whichever way it gets saved (server now, on the phone, or queued):
      // a request that reached the server before the connection dropped can't become a second contact.
      const clientId = uuidv4();
      const capturedAt = new Date().toISOString();

      // No connection: read the card on this phone and save it locally (uploads later). If that
      // isn't possible (Expo Go, QR codes, nothing readable) keep the photo for the server, with
      // the event that's active now (it uploads later, maybe after switching events).
      const queueIt = async (): Promise<ScanOutcome> => {
        if (kind === 'card') {
          const local = await saveCardReadOnDevice(imageUri, { online: false, clientId });
          if (local) return fromOnDevice(local);
        }
        const q = await queueScan(kind, imageUri, profileStore.activeEventName(), clientId);
        return { status: 'queued', contactId: q.id, existingName: null, matchedOn: null, autoScoreStarted: false };
      };
      if (!(await isOnline())) return await queueIt();

      let response;
      try {
        response = kind === 'card'
          ? await api.addContactFromImage(imageUri, userId, { clientId, capturedAt })
          : await api.scanQRCode(imageUri, userId, { clientId, capturedAt });
      } catch (e: any) {
        // Timed out while online: the server is probably still reading (and will save) the card, so
        // saving it again from the phone / the queue would create a duplicate.
        if (errorKind(e) === 'timeout' && (await isOnline())) {
          throw new Error('Reading this card is taking longer than usual. It may still appear in Contacts in a moment — check there before scanning it again.');
        }
        if (isNetworkError(e)) return await queueIt();
        // The server's AI couldn't read it (bad key, rate limit, outage): read it on the phone instead.
        if (kind === 'card' && (e?.response?.status ?? 0) >= 500) {
          const local = await saveCardReadOnDevice(imageUri, { online: true, clientId }).catch(() => null);
          if (local) return fromOnDevice(local);
        }
        throw e;
      }

      if (response.status === 'duplicate' && response.existing_contact_id) {
        return {
          status: 'duplicate',
          contactId: response.existing_contact_id,
          existingName: response.existing_name || 'this contact',
          matchedOn: response.matched_on || null,
          autoScoreStarted: false,
        };
      }
      if (!response.contact_id) {
        throw new Error(response.message || (kind === 'card'
          ? 'No contact details could be read from the card.'
          : 'No contact details could be read from the code.'));
      }
      return {
        status: 'created',
        contactId: response.contact_id,
        existingName: null,
        matchedOn: null,
        autoScoreStarted: !!response.auto_score_started,
      };
    } catch (err: any) {
      const message = scanErrorMessage(err, kind);
      setError(message);
      throw new Error(message);
    } finally {
      setScanning(false);
    }
  }, [isAuthenticated]);

  const processImageScan = useCallback((imageUri: string) => runScan('card', imageUri), [runScan]);
  const processQRScan = useCallback((imageUri: string) => runScan('qr', imageUri), [runScan]);

  const clearLastScanned = useCallback(() => {
    setLastScanned(null);
    setError(null);
  }, []);

  return {
    scanning,
    lastScanned,
    error,
    processImageScan,
    processQRScan,
    clearLastScanned,
  };
}
