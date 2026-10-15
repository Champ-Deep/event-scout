import type { Contact } from '@es/types';
import { api } from '../../lib/api';
import { syncEvents } from '../../sync/core';
import {
  cacheContact, cacheContactIfChanged, deleteCachedContact, getCachedContact, queryCachedContacts,
} from '../../sync/contactCache';
import { outboxStore, ownerStore } from '../../sync/localStore';
import { createContactRepository, type ContactLocal, type ContactRemote } from './repository';

export {
  editBase, getEditConflict, CONFLICT_FIELDS,
  type ConflictField, type ContactPatch, type NewContactInput, type ContactEvent, type EditConflict,
  type FieldConflict, type ListQuery, type ListResult, type ContactRepository,
} from './repository';

const remote: ContactRemote = {
  list: (params) => api.listContacts(params),
  get: async (id) => (await api.getContact(id)) as unknown as Contact,
  create: (input) => api.createContact(input as any),
  update: (id, patch, opts) => api.updateContact(id, patch as any, opts),
  delete: (id) => api.deleteContact(id),
  purge: (id) => api.purgeDeletedContact(id),
  restore: (id) => api.restoreContact(id),
  duplicates: (id) => api.getContactDuplicates(id),
  attachLinkedIn: (id, pick) => api.attachLinkedIn(id, pick),
};

const local: ContactLocal = {
  ensureOwner: () => ownerStore.ensure(),
  signedInUser: () => ownerStore.signedIn(),
  get: (id) => getCachedContact(id),
  put: (c) => cacheContact(c),
  putIfChanged: (c) => cacheContactIfChanged(c),
  remove: (id) => deleteCachedContact(id),
  query: (params) => queryCachedContacts(params as any),
  pendingIds: () => outboxStore.entityIds(),
  queued: () => outboxStore.all(),
  hasPendingCreate: (id) => outboxStore.hasPendingCreate(id),
  enqueue: (item) => outboxStore.enqueue(item),
  dropQueued: (id) => outboxStore.dropForEntity(id),
};

/**
 * The contact repository (see ./repository): the only way screens read, list, write and delete
 * contacts. Online first, the phone's cache and outbox when the server can't be reached.
 */
export const contactsFeature = createContactRepository({
  remote,
  local,
  onQueued: () => syncEvents.emit({ type: 'queued' }),
});

/** Contact change events (deleted / restored / discarded) — e.g. the list's "· Undo" toast. */
export const contactEvents = contactsFeature.events;

export const duplicateReasonLabel: Record<string, string> = {
  email: 'same email',
  phone: 'same phone',
  name: 'same name & company',
};

export function describeReasons(reasons: string[]) {
  return reasons.map((r) => duplicateReasonLabel[r] || r).join(' · ');
}
