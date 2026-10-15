// The Edit tab's form state and the PATCH it turns into (pure; tested in draft.test.ts).
import type { Contact, ContactPatch } from '../../lib/types';

export type Draft = {
  firstName: string; lastName: string; company: string; title: string; email: string; phone: string; linkedin: string; eventId: string; notes: string;
  /** undefined = unchanged; a data URI = new photo; null = remove it */
  photo: string | null | undefined;
};

/** A contact scanned without a name is stored as "Unknown": the form shows that as empty. */
const firstNameOf = (c: Contact) => (c.firstName === 'Unknown' && !c.lastName ? '' : c.firstName);

export const draftOf = (c: Contact): Draft => ({
  firstName: firstNameOf(c), lastName: c.lastName || '', company: c.company || '', title: c.title || '',
  email: c.email || '', phone: c.phone || '', linkedin: c.linkedin || '', eventId: c.eventId || '', notes: c.notes || '', photo: undefined,
});

const norm = (v: string | null | undefined) => (v ?? '').trim();

/** Only the fields that actually changed go to PATCH (trimmed; cleared optional fields become null). */
export function diffDraft(c: Contact, d: Draft): ContactPatch {
  const p: ContactPatch = {};
  if (norm(d.firstName) !== norm(firstNameOf(c))) p.firstName = d.firstName.trim();
  if (norm(d.lastName) !== norm(c.lastName)) p.lastName = d.lastName.trim() || null;
  if (norm(d.company) !== norm(c.company)) p.company = d.company.trim() || null;
  if (norm(d.title) !== norm(c.title)) p.title = d.title.trim() || null;
  if (norm(d.email) !== norm(c.email)) p.email = d.email.trim() || null;
  if (norm(d.phone) !== norm(c.phone)) p.phone = d.phone.trim() || null;
  if (norm(d.linkedin) !== norm(c.linkedin)) p.linkedin = d.linkedin.trim() || null;
  if (norm(d.eventId) !== norm(c.eventId)) p.eventId = d.eventId.trim() || null;
  if (d.notes.trimEnd() !== (c.notes || '').trimEnd()) p.notes = d.notes;
  if (d.photo !== undefined) p.photo = d.photo;
  return p;
}
