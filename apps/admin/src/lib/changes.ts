// What happened → which server data it moved. One map for every write in the console AND every
// live event from the server (features/live/useEventStream), so a merge, a re-score or a delete
// refreshes the same pages whether this tab made it, another admin did, or a rep did in the app.
//
// Keys follow the hierarchy in lib/queries: a prefix covers everything under it (['admin','contacts']
// covers the grid, the stats, one contact, its files, QR and duplicates), so a child key is never
// listed next to its parent.
import type { QueryClient, QueryKey } from '@tanstack/react-query';
import { keys } from './queries';

/** Event names the server sends on the live stream (broadcast_live_event in apps/backend). */
export const LIVE_CHANGES = [
  'contact_scanned', 'contact_updated', 'score_updated', 'admin_score_updated', 'contact_deleted', 'contact_restored',
  'contact_merged', 'contact_purged', 'batch_score_completed', 'industries_updated', 'pipeline_started', 'pipeline_finished',
  'broadcast_alert', 'event_planned', 'list_changed', 'user_created', 'user_updated', 'user_deleted', 'files_changed',
] as const;
export type LiveChange = (typeof LIVE_CHANGES)[number];

/** Changes only this console makes (the server sends no live event for them, or one too narrow). */
export const LOCAL_CHANGES = [
  'exhibitors_changed', 'duplicate_dismissed', 'file_downloaded', 'pipeline_settings_changed', 'backup_done', 'crm_pushed',
] as const;
export type LocalChange = (typeof LOCAL_CHANGES)[number];

export type Change = LiveChange | LocalChange;

const isIn = <T extends string>(list: readonly T[], v: string): v is T => (list as readonly string[]).includes(v);
export const isLiveChange = (name: string): name is LiveChange => isIn(LIVE_CHANGES, name);
export const isChange = (name: string): name is Change => isLiveChange(name) || isIn(LOCAL_CHANGES, name);

// one place per page of data (prefixes of lib/queries keys)
const CONTACTS = keys.contactsAll; // the grid, stats, one contact, its files / QR / duplicates tab
const DASHBOARD = ['admin', 'dashboard'] as const;
const ACTIVITY = ['admin', 'activity'] as const; // the feed, a day, the calendar
const DUPLICATES = keys.duplicates;
const EVENTS = keys.events; // event list, every event's detail
const SEGMENTS = keys.segments; // smart lists
const USERS = keys.users;
const PIPELINES = ['admin', 'pipelines'] as const;
const LISTS = keys.lists; // lists and their members
const DELETED = keys.deleted;
const BROADCASTS = keys.broadcasts; // live ones and the history
const PLANNER = keys.planner;
const FILES = keys.files;
const ME = keys.me;
const EXHIBITORS = ['admin', 'exhibitors'] as const;
const CONTACT_DUPLICATES = ['admin', 'contacts', 'duplicates'] as const;

// a contact appeared, left or came back: counts everywhere move
const CONTACT_SET = [CONTACTS, DASHBOARD, ACTIVITY, DUPLICATES, EVENTS, SEGMENTS, USERS, PIPELINES, LISTS, DELETED] as const;
// a contact's fields changed: an edited email / phone / name can make or break a duplicate pair
const CONTACT_FIELDS = [CONTACTS, DASHBOARD, ACTIVITY, DUPLICATES, EVENTS, SEGMENTS] as const;
// the admin console's own score — every console count uses it: the grid, dashboard levels, events,
// team hot counts, smart lists, lists, pipelines and the activity feed
const ADMIN_SCORE = [CONTACTS, DASHBOARD, ACTIVITY, EVENTS, SEGMENTS, USERS, PIPELINES, LISTS] as const;

export const AFFECTS: { readonly [C in Change]: readonly QueryKey[] } = {
  // (app card / QR scans and contacts added here) — moves capture counts, pipelines and smart-list membership
  contact_scanned: [CONTACTS, DASHBOARD, ACTIVITY, DUPLICATES, EVENTS, SEGMENTS, USERS, PIPELINES, LISTS],
  // edits in the app (offline sync push) and here, LinkedIn attach, admin notes, file attach / remove
  contact_updated: CONTACT_FIELDS,
  // the rep's own score (sent on every edit in the app, not just scores)
  score_updated: CONTACT_FIELDS,
  admin_score_updated: ADMIN_SCORE,
  batch_score_completed: ADMIN_SCORE,
  contact_deleted: CONTACT_SET,
  contact_restored: CONTACT_SET,
  // the duplicate leaves for Recently deleted, the kept one gains its fields, notes and files
  contact_merged: CONTACT_SET,
  // permanently deleted from Recently deleted (or the trash emptied)
  contact_purged: [DELETED, CONTACTS, DASHBOARD, DUPLICATES, ACTIVITY],
  // the AI finished sorting new companies into industries
  industries_updated: [SEGMENTS, DASHBOARD],
  pipeline_started: [PIPELINES, ACTIVITY],
  // a run can re-score the contact and attach a deck
  // the run's scoring step writes the admin score → every hot count, plus the run itself
  pipeline_finished: [...ADMIN_SCORE, PIPELINES],
  broadcast_alert: [BROADCASTS, DASHBOARD, ACTIVITY],
  // planner events, their team, status and floor plans (team files)
  event_planned: [PLANNER, EVENTS, FILES, DASHBOARD, ACTIVITY],
  list_changed: [LISTS],
  // team files: the Files page, each event's files / floor plans and the planner's
  files_changed: [FILES, EVENTS, PLANNER, DASHBOARD, ACTIVITY],
  user_created: [USERS, DASHBOARD, ACTIVITY],
  // a renamed rep (or a role / photo change): their name shows on contacts, duplicates, deleted rows, runs…
  user_updated: [...CONTACT_SET, PLANNER, ME],
  // a removed (or merged-away) rep's contacts are deleted or reassigned: every contact-derived view moves
  user_deleted: [...CONTACT_SET, PLANNER],

  exhibitors_changed: [EVENTS, EXHIBITORS],
  // "not a duplicate": the pair leaves the Duplicates page and both contacts' Duplicates tab
  duplicate_dismissed: [DUPLICATES, CONTACT_DUPLICATES],
  // the download count on Team files
  file_downloaded: [FILES],
  pipeline_settings_changed: [PIPELINES],
  backup_done: [keys.backup],
  crm_pushed: [keys.crms],
};

/** What an unknown live event refreshes (only the feed, as before). */
const FALLBACK: readonly QueryKey[] = [ACTIVITY];

const startsWith = (key: QueryKey, prefix: QueryKey) => prefix.length <= key.length && prefix.every((p, i) => p === key[i]);

/** The query keys these changes affect: deduplicated, and a key already covered by another's prefix dropped. */
export function affectedKeys(changes: Change | readonly Change[]): QueryKey[] {
  const all = (typeof changes === 'string' ? [changes] : changes).flatMap((c) => AFFECTS[c] ?? FALLBACK);
  return all.filter((k, i) => !all.some((p, j) => startsWith(k, p) && (p.length < k.length || j < i)));
}

/** Refresh everything these changes moved (named for what happened: applyChange(qc, 'contact_merged')). */
export function applyChange(qc: QueryClient, changes: Change | readonly Change[]): Promise<void> {
  return Promise.all(affectedKeys(changes).map((queryKey) => qc.invalidateQueries({ queryKey }))).then(() => undefined);
}

/** A live event by name: its change, or (an event this console doesn't know yet) the feed only. */
export function liveKeys(name: string): readonly QueryKey[] {
  if (isLiveChange(name)) return affectedKeys(name);
  if (import.meta.env?.DEV) console.warn(`[live] unknown event "${name}": refreshing the activity feed only`);
  return FALLBACK;
}
