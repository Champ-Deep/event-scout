import { api } from '../../../lib/api';
import { contactsFeature } from '../../contacts';
import { profileStore } from '../../profile';
import { cardThumbnail, isOnDeviceOcrAvailable, readCardOnDevice, warmUpOnDeviceOcr, type ParsedCard } from './index';

/** Refresh the active event (while online) so cards read offline are tagged with it too. The
 *  profile module also keeps this copy current whenever the profile is loaded or saved. */
export async function rememberActiveEvent(userId: string | null) {
  if (!userId) return;
  await profileStore.load(userId).catch(() => { /* offline: keep the last one */ });
}

/** Called when the scanner opens: load the OCR models and refresh the event name. */
export function prepareOnDeviceScanning(userId: string | null) {
  warmUpOnDeviceOcr();
  rememberActiveEvent(userId);
}

export interface OnDeviceSaved {
  contactId: string;
  pendingSync: boolean; // saved on the phone only, uploads when back online
  duplicateOf: { id: string; name: string; matchedOn: string | null } | null;
  confidence: number;
  card: ParsedCard;
}

/** Enough to be worth saving: a name, plus a way to reach them or where they work. */
function usable(c: ParsedCard) {
  return !!c.firstName && !!(c.email || c.phone || c.company);
}

/**
 * Read the card on this phone and save it as a contact (offline: saved locally and queued).
 * Returns null when on-device reading isn't in this build or didn't find a usable contact —
 * the caller then keeps the photo for the server to read.
 */
export async function saveCardReadOnDevice(imageUri: string, opts: { online: boolean; clientId?: string }): Promise<OnDeviceSaved | null> {
  if (!isOnDeviceOcrAvailable()) return null;
  let card: ParsedCard;
  try {
    ({ card } = await readCardOnDevice(imageUri));
  } catch (e: any) {
    console.warn('[ocr] on-device read failed', e?.message);
    return null;
  }
  if (!usable(card)) {
    // Dev builds only: it includes the name read from the card.
    if (__DEV__) console.log('[ocr] not enough read on the phone', JSON.stringify({ name: card.firstName?.value, conf: card.confidence }));
    return null;
  }

  const details = {
    firstName: card.firstName!.value,
    lastName: card.lastName?.value ?? null,
    company: card.company?.value ?? null,
    title: card.title?.value ?? null,
    email: card.email?.value ?? null,
    phone: card.phone?.value ?? null,
  };

  // Online: same duplicate rule as server scans.
  if (opts.online) {
    try {
      const { matches } = await api.checkDuplicates(details);
      const m = matches?.[0];
      if (m) {
        const name = [m.contact.firstName, m.contact.lastName].filter(Boolean).join(' ') || 'this contact';
        return { contactId: String(m.contact.id), pendingSync: false, duplicateOf: { id: String(m.contact.id), name, matchedOn: (m.reasons?.[0] as any) ?? null }, confidence: card.confidence, card };
      }
    } catch { /* duplicate hints are advisory */ }
  }

  const notes = [
    card.website ? `Website: ${card.website.value}` : null,
    ...card.otherLines.slice(0, 4),
  ].filter(Boolean).join('\n');

  // The scan's id (when given) becomes the contact's, so this capture is never saved twice.
  const created = await contactsFeature.createContact({
    ...(opts.clientId ? { id: opts.clientId } : {}),
    ...details,
    notes: notes ? `From the card:\n${notes}` : null,
    eventId: profileStore.activeEventName(),
    source: 'scan',
    photo: await cardThumbnail(imageUri),
  });
  // LinkedIn isn't a create field; set it straight after (queued too when offline).
  if (card.linkedin) {
    try { await contactsFeature.updateContact(String(created.id), { linkedin: card.linkedin.value } as any); } catch { /* optional */ }
  }
  return { contactId: String(created.id), pendingSync: !!(created as any).pendingSync, duplicateOf: null, confidence: card.confidence, card };
}
