import { describe, expect, it } from 'vitest';
import { DEFAULT_VISIBILITY, PREFS_KEY, loadPrefs, savePrefs, searchForSorting, sortingFor } from './table';
import { levelCounts } from '../../lib/levels';

function memory(initial: Record<string, string> = {}) {
  const m = new Map(Object.entries(initial));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => { m.set(k, v); }, dump: () => Object.fromEntries(m) };
}

describe('table prefs', () => {
  it('defaults when nothing is saved: table view, 15 rows a page', () => {
    expect(loadPrefs(memory())).toEqual({ visibility: DEFAULT_VISIBILITY, density: 'comfortable', sizing: {}, pageSize: 15, view: 'table' });
    expect(loadPrefs(null).density).toBe('comfortable');
  });
  it('round-trips through storage', () => {
    const s = memory();
    savePrefs({ visibility: { ...DEFAULT_VISIBILITY, email: true, event: false }, density: 'compact', sizing: { company: 240 }, pageSize: 100, view: 'cards' }, s);
    expect(loadPrefs(s)).toEqual({ visibility: { ...DEFAULT_VISIBILITY, email: true, event: false }, density: 'compact', sizing: { company: 240 }, pageSize: 100, view: 'cards' });
  });
  it('ignores junk and never hides the name column', () => {
    const s = memory({ [PREFS_KEY]: JSON.stringify({ v: 4, visibility: { name: false, bogus: true, score: 'yes' }, density: 'huge', sizing: { company: 5, name: 'wide', title: 300.4 } }) });
    expect(loadPrefs(s)).toEqual({ visibility: DEFAULT_VISIBILITY, density: 'comfortable', sizing: { title: 300 }, pageSize: 15, view: 'table' });
    expect(loadPrefs(memory({ [PREFS_KEY]: '{not json' }))).toEqual({ visibility: DEFAULT_VISIBILITY, density: 'comfortable', sizing: {}, pageSize: 15, view: 'table' });
  });
  it('drops a layout saved by an older version (so new default columns show up)', () => {
    const s = memory({ [PREFS_KEY]: JSON.stringify({ v: 3, visibility: { ...DEFAULT_VISIBILITY, phone: false }, density: 'compact', sizing: {}, pageSize: 50, view: 'cards' }) });
    expect(loadPrefs(s)).toEqual({ visibility: DEFAULT_VISIBILITY, density: 'comfortable', sizing: {}, pageSize: 15, view: 'table' });
  });
  it('opens the table view unless the cards were chosen', () => {
    expect(loadPrefs(memory({ [PREFS_KEY]: JSON.stringify({ v: 4, view: 'cards' }) })).view).toBe('cards');
    expect(loadPrefs(memory({ [PREFS_KEY]: JSON.stringify({ v: 4, view: 'grid' }) })).view).toBe('table');
  });
  it('only keeps a known page size', () => {
    expect(loadPrefs(memory({ [PREFS_KEY]: JSON.stringify({ v: 4, pageSize: 7 }) })).pageSize).toBe(15);
    expect(loadPrefs(memory({ [PREFS_KEY]: JSON.stringify({ v: 4, pageSize: 200 }) })).pageSize).toBe(200);
  });
  it('shows every piece of contact data by default (score sits next to the name)', () => {
    const shown = Object.entries(DEFAULT_VISIBILITY).filter(([, v]) => v).map(([k]) => k);
    expect(shown).toEqual(['name', 'company', 'title', 'phone', 'email', 'linkedin', 'owner', 'event', 'source', 'added']);
  });
});

describe('sorting <-> URL', () => {
  it('builds TanStack sorting from the URL, using the natural direction when none is given', () => {
    expect(sortingFor({ sort: 'added', dir: undefined })).toEqual([{ id: 'added', desc: true }]);
    expect(sortingFor({ sort: 'name', dir: undefined })).toEqual([{ id: 'name', desc: false }]);
    expect(sortingFor({ sort: 'name', dir: 'desc' })).toEqual([{ id: 'name', desc: true }]);
  });
  it('writes only what the URL needs', () => {
    expect(searchForSorting([{ id: 'score', desc: true }])).toEqual({ sort: 'score', dir: undefined });
    expect(searchForSorting([{ id: 'score', desc: false }])).toEqual({ sort: 'score', dir: 'asc' });
    expect(searchForSorting([{ id: 'company', desc: false }])).toEqual({ sort: 'company', dir: undefined });
    expect(searchForSorting([])).toEqual({ sort: 'added', dir: undefined });
    expect(searchForSorting([{ id: 'select', desc: true }])).toEqual({ sort: 'added', dir: undefined });
  });
});

describe('levelCounts', () => {
  it('buckets scores like the app (hot ≥85, warm ≥70, cool ≥50, else cold; null = unscored)', () => {
    expect(levelCounts([99, 85, 84, 70, 69, 50, 49, 0, null, undefined])).toEqual({ all: 10, A: 2, B: 2, C: 2, D: 2, U: 2 });
  });
});
