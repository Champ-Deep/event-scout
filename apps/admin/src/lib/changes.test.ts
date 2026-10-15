import { describe, expect, it, vi } from 'vitest';
import { QueryClient } from '@tanstack/react-query';
import { AFFECTS, LIVE_CHANGES, LOCAL_CHANGES, affectedKeys, applyChange, isLiveChange, liveKeys } from './changes';

const prefixOf = (key: readonly unknown[], prefix: readonly unknown[]) => prefix.length < key.length && prefix.every((p, i) => p === key[i]);

describe('AFFECTS', () => {
  it('has an entry for every live and local change', () => {
    for (const c of [...LIVE_CHANGES, ...LOCAL_CHANGES]) expect(AFFECTS[c]?.length, c).toBeGreaterThan(0);
  });
  it('every key is an admin key', () => {
    for (const ks of Object.values(AFFECTS)) for (const k of ks) expect(k[0]).toBe('admin');
  });
  it('never lists a child key next to a parent prefix', () => {
    for (const [c, ks] of Object.entries(AFFECTS)) {
      for (const k of ks) expect(ks.some((p) => prefixOf(k, p)), `${c}: ${k.join('/')}`).toBe(false);
    }
  });
  it('a merge refreshes the same data wherever it starts (contacts, duplicates, deleted, lists, counts)', () => {
    const ks = affectedKeys('contact_merged').map((k) => k.join('/'));
    expect(ks).toEqual(expect.arrayContaining(['admin/contacts', 'admin/duplicates', 'admin/deleted', 'admin/lists', 'admin/dashboard', 'admin/pipelines']));
  });
});

describe('affectedKeys', () => {
  it('merges several changes, deduplicated, dropping keys a prefix already covers', () => {
    const ks = affectedKeys(['contact_merged', 'duplicate_dismissed']).map((k) => k.join('/'));
    expect(new Set(ks).size).toBe(ks.length);
    expect(ks).toContain('admin/contacts');
    expect(ks).not.toContain('admin/contacts/duplicates');
    expect(affectedKeys('duplicate_dismissed').map((k) => k.join('/'))).toEqual(['admin/duplicates', 'admin/contacts/duplicates']);
  });
});

describe('live events', () => {
  it('knows the server event names', () => {
    expect(isLiveChange('contact_scanned')).toBe(true);
    expect(isLiveChange('exhibitors_changed')).toBe(false); // local only
  });
  it('an unknown event refreshes only the activity feed (and warns in dev)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(liveKeys('something_new')).toEqual([['admin', 'activity']]);
    warn.mockRestore();
  });
});

describe('applyChange', () => {
  it('invalidates each affected key', async () => {
    const qc = new QueryClient();
    const spy = vi.spyOn(qc, 'invalidateQueries');
    await applyChange(qc, 'list_changed');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith({ queryKey: ['admin', 'lists'] });
  });
});
