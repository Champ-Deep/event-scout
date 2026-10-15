// URL search params for every page, validated with zod. Bad or missing values fall back to
// defaults (.catch) so a hand-edited or old link never crashes a page.
import { z } from 'zod';
import { dayKey, localDayRange } from './dates';

// The router JSON-parses search values, so ?q=123 or ?add=1 arrive as numbers: read them as text.
const text = (v: unknown) => (typeof v === 'number' ? String(v) : v);
const opt = z.preprocess(text, z.string().trim().min(1)).optional().catch(undefined);

/** Grid columns the URL can sort by (one at a time; the column header toggles it). */
export const CONTACT_SORTS = ['added', 'updated', 'score', 'name', 'company', 'title', 'event', 'owner', 'email'] as const;
export type ContactSort = (typeof CONTACT_SORTS)[number];
/** Tabs of the contact drawer (/contacts/$id?tab=…). */
export const CONTACT_TABS = ['overview', 'edit', 'score', 'duplicates', 'pipeline', 'files', 'qr'] as const;
export type ContactTab = (typeof CONTACT_TABS)[number];

// older links used ?sort=recent
const legacySort = (v: unknown) => (v === 'recent' ? 'added' : v);

export const contactsSearch = z.object({
  q: opt,
  band: z.enum(['A', 'B', 'C', 'D', 'U']).optional().catch(undefined),
  event: opt,
  owner: opt,
  /** ?since=today: only contacts added today (from the admin's local midnight) */
  since: z.enum(['today']).optional().catch(undefined),
  sort: z.preprocess(legacySort, z.enum(CONTACT_SORTS)).catch('added').default('added'),
  /** direction; left out = the column's natural order (newest / best first, text A→Z) */
  dir: z.enum(['asc', 'desc']).optional().catch(undefined),
  tab: z.enum(CONTACT_TABS).optional().catch(undefined),
  /** ?add=1 opens the "Add contact" dialog (also used by the ⌘K palette) */
  add: z.preprocess(text, z.enum(['1'])).optional().catch(undefined),
  /** ?full=1 shows the open contact as a full page instead of the side panel */
  full: z.preprocess(text, z.enum(['1'])).optional().catch(undefined),
});
export type ContactsSearch = z.infer<typeof contactsSearch>;

/** Natural direction of a column: dates and scores newest / best first, text A→Z. */
export const defaultDir = (sort: ContactSort): 'asc' | 'desc' => (sort === 'added' || sort === 'updated' || sort === 'score' ? 'desc' : 'asc');
export const sortDir = (s: Pick<ContactsSearch, 'sort' | 'dir'>): 'asc' | 'desc' => s.dir ?? defaultDir(s.sort);

/**
 * The /api/contacts query for the grid (what the loader ensures and the page reads). The server
 * orders by score or recency and caps at 1,000 rows, so the cap keeps the best / newest ones;
 * the grid sorts the loaded rows by whichever column is chosen.
 */
export const contactQuery = (s: ContactsSearch) => ({
  search: s.q, scoreBand: s.band, eventId: s.event, userId: s.owner, createdSince: sinceIso(s.since),
  sort: s.sort === 'score' ? ('score' as const) : ('recent' as const), limit: 1000,
  // light rows (no photos / score reasoning): the grid shows initials, the drawer loads the full contact
  lite: 1 as const,
});
/** Same filters without the level, for the level counts (GET /api/contacts/stats). */
export const contactCountsQuery = (s: Pick<ContactsSearch, 'q' | 'event' | 'owner' | 'since'>) =>
  ({ search: s.q, eventId: s.event, userId: s.owner, createdSince: sinceIso(s.since) });

/** ?since=… → the createdSince sent to the API: "today" = the admin's local midnight, as an ISO time. */
export function sinceIso(since: ContactsSearch['since'], now = new Date()): string | undefined {
  return since === 'today' ? localDayRange(dayKey(now)).start : undefined;
}

export const dashboardSearch = z.object({
  days: z.coerce.number().int().refine((n) => [7, 14, 30, 90].includes(n)).catch(7).default(7),
});

export const activitySearch = z.object({
  type: z.enum(['contact_added', 'contact_scored', 'pipeline', 'broadcast', 'file', 'contact_deleted', 'contact_merged', 'user_joined']).optional().catch(undefined),
  /** a local day picked on the calendar (YYYY-MM-DD) */
  date: z.preprocess((v) => (v == null ? v : String(v)), z.string().regex(/^\d{4}-\d{2}-\d{2}$/)).optional().catch(undefined),
});

export const pipelinesSearch = z.object({
  status: z.enum(['running', 'complete', 'failed']).optional().catch(undefined),
});

export const SEAT_SORTS = ['contacts', 'active', 'newest', 'name'] as const;
export const seatsSearch = z.object({
  q: opt,
  role: z.enum(['rep', 'admin']).optional().catch(undefined),
  /** from the summary tiles: added a contact in the last 24 h / never added one */
  show: z.enum(['active', 'idle']).optional().catch(undefined),
  sort: z.enum(SEAT_SORTS).optional().catch(undefined),
});

/** /lists: which tiles (smart lists by industry / seniority / role / company, or your own lists) and which one is open */
export const listsSearch = z.object({
  tab: z.enum(['mine', 'industry', 'seniority', 'function', 'company']).optional().catch(undefined),
  seg: opt,
  list: opt,
});

export const deletedSearch = z.object({ q: opt, owner: opt, view: opt });

/** /broadcasts: which tab (announcements | events) and the event open in the panel */
export const broadcastsSearch = z.object({ tab: z.enum(['announce', 'events']).optional().catch(undefined), event: opt });

export const EVENT_TABS = ['booths', 'map', 'team', 'files'] as const;
export const eventsSearch = z.object({
  event: opt, q: opt, category: opt, tab: z.enum(EVENT_TABS).optional().catch(undefined), hall: opt,
  /** booths filter from the summary tiles: visited by a rep / contacts met from that company */
  show: z.enum(['visited', 'met']).optional().catch(undefined),
});

/** Drop undefined / empty values so URLs stay short (?band=A, not ?band=&q=). */
export function clean<T extends Record<string, unknown>>(s: T): Partial<T> {
  return Object.fromEntries(Object.entries(s).filter(([, v]) => v !== undefined && v !== '' && v !== null)) as Partial<T>;
}
