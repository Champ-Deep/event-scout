// Pure helpers for the contacts grid's TanStack Table state: which columns show, how dense the
// rows are, column widths and rows per page (all remembered per browser), and the URL <-> sorting mapping.
import type { SortingState } from '@tanstack/react-table';
import { CONTACT_SORTS, defaultDir, type ContactSort, type ContactsSearch } from '../../lib/search';

/**
 * Columns a user can show / hide, in display order. The checkbox, the contact-icons column (right
 * after Name) and the actions column are always there.
 */
export const COLUMN_IDS = ['name', 'company', 'title', 'phone', 'email', 'linkedin', 'owner', 'event', 'source', 'added', 'score', 'updated'] as const;
export type ColumnId = (typeof COLUMN_IDS)[number];

/** Plain header labels (the product owner's wording). */
export const COLUMN_LABELS: Record<ColumnId, string> = {
  name: 'Name', company: 'Company', title: 'Designation', phone: 'Phone', email: 'Email', linkedin: 'LinkedIn',
  owner: 'Representative', event: 'Event', source: 'Source', added: 'Added', score: 'Admin score', updated: 'Last updated',
};

/**
 * What a new browser sees: every piece of contact data. The score sits next to the name, so its own
 * column starts hidden, like "Last updated".
 */
export const DEFAULT_VISIBILITY: Record<ColumnId, boolean> = {
  name: true, company: true, title: true, phone: true, email: true, linkedin: true, owner: true, event: true, source: true, added: true,
  score: false, updated: false,
};

export type Density = 'comfortable' | 'compact';
export const ROW_HEIGHT: Record<Density, number> = { comfortable: 64, compact: 44 };

export const PAGE_SIZES = [15, 25, 50, 100, 200] as const;
export const DEFAULT_PAGE_SIZE = 15;

/** How the contacts page lays rows out: mobile-style lead cards (default) or the full table. */
export type ContactsView = 'cards' | 'table';

export interface TablePrefs {
  visibility: Record<string, boolean>;
  density: Density;
  sizing: Record<string, number>;
  pageSize: number;
  view: ContactsView;
}

export const DEFAULT_PREFS: TablePrefs = { visibility: { ...DEFAULT_VISIBILITY }, density: 'comfortable', sizing: {}, pageSize: DEFAULT_PAGE_SIZE, view: 'table' };
export const PREFS_KEY = 'es.admin.contacts.table';
/** Bump when the default columns change, so an old saved layout doesn't hide the new ones. */
export const PREFS_VERSION = 4;

const isColumn = (k: string): k is ColumnId => (COLUMN_IDS as readonly string[]).includes(k);

/** Read saved preferences; anything missing, malformed or from an older layout falls back to the defaults. */
export function loadPrefs(storage: Pick<Storage, 'getItem'> | null = typeof localStorage === 'undefined' ? null : localStorage): TablePrefs {
  try {
    const raw = storage?.getItem(PREFS_KEY);
    if (!raw) return { ...DEFAULT_PREFS, visibility: { ...DEFAULT_VISIBILITY } };
    const p = JSON.parse(raw) as Partial<TablePrefs> & { v?: number };
    if (p.v !== PREFS_VERSION) return { ...DEFAULT_PREFS, visibility: { ...DEFAULT_VISIBILITY } };
    const visibility: Record<string, boolean> = { ...DEFAULT_VISIBILITY };
    if (p.visibility && typeof p.visibility === 'object') {
      Object.entries(p.visibility).forEach(([k, v]) => { if (isColumn(k) && typeof v === 'boolean') visibility[k] = v; });
    }
    visibility.name = true; // the name column can't be hidden
    const sizing: Record<string, number> = {};
    if (p.sizing && typeof p.sizing === 'object') {
      Object.entries(p.sizing).forEach(([k, v]) => { if (isColumn(k) && typeof v === 'number' && v >= 40 && v <= 1200) sizing[k] = Math.round(v); });
    }
    const pageSize = (PAGE_SIZES as readonly number[]).includes(p.pageSize as number) ? (p.pageSize as number) : DEFAULT_PAGE_SIZE;
    return { visibility, density: p.density === 'compact' ? 'compact' : 'comfortable', sizing, pageSize, view: p.view === 'cards' ? 'cards' : 'table' };
  } catch {
    return { ...DEFAULT_PREFS, visibility: { ...DEFAULT_VISIBILITY } };
  }
}

export function savePrefs(prefs: TablePrefs, storage: Pick<Storage, 'setItem'> | null = typeof localStorage === 'undefined' ? null : localStorage) {
  try { storage?.setItem(PREFS_KEY, JSON.stringify({ v: PREFS_VERSION, ...prefs })); } catch { /* storage blocked: prefs live for this tab only */ }
}

/** TanStack sorting state for the URL's sort + dir. */
export function sortingFor(s: Pick<ContactsSearch, 'sort' | 'dir'>): SortingState {
  return [{ id: s.sort, desc: (s.dir ?? defaultDir(s.sort)) === 'desc' }];
}

/**
 * URL patch for a new TanStack sorting state. Clearing the sort returns to the default (added,
 * newest first); a direction equal to the column's natural one is left out of the URL.
 */
export function searchForSorting(next: SortingState): { sort: ContactSort; dir: 'asc' | 'desc' | undefined } {
  const first = next[0];
  if (!first || !(CONTACT_SORTS as readonly string[]).includes(first.id)) return { sort: 'added', dir: undefined };
  const sort = first.id as ContactSort;
  const dir = first.desc ? 'desc' : 'asc';
  return { sort, dir: dir === defaultDir(sort) ? undefined : dir };
}
