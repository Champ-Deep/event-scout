import { describe, expect, it } from 'vitest';
import { activitySearch, clean, contactCountsQuery, contactQuery, contactsSearch, dashboardSearch, defaultDir, eventsSearch, pipelinesSearch, seatsSearch, sortDir } from './search';

describe('contactsSearch', () => {
  it('defaults the sort and drops nothing valid', () => {
    expect(contactsSearch.parse({})).toEqual({ sort: 'added' });
    expect(contactsSearch.parse({ q: 'ana', band: 'A', event: 'HIMSS', owner: 'u1', sort: 'score', dir: 'asc', tab: 'edit', add: '1' }))
      .toEqual({ q: 'ana', band: 'A', event: 'HIMSS', owner: 'u1', sort: 'score', dir: 'asc', tab: 'edit', add: '1' });
  });
  it('falls back on bad or hand-edited values instead of throwing', () => {
    expect(contactsSearch.parse({ band: 'Z', sort: 'random', dir: 'up', tab: 'nope', q: '   ', add: 'yes' })).toEqual({ sort: 'added' });
    expect(contactsSearch.parse({ q: { nested: true } })).toEqual({ sort: 'added' });
  });
  it('reads values the router JSON-parsed into numbers (?add=1, ?q=123)', () => {
    expect(contactsSearch.parse({ add: 1, q: 123 })).toEqual({ add: '1', q: '123', sort: 'added' });
  });
  it('accepts U (not scored) as a band', () => {
    expect(contactsSearch.parse({ band: 'U' }).band).toBe('U');
  });
  it('still understands the old ?sort=recent links', () => {
    expect(contactsSearch.parse({ sort: 'recent' }).sort).toBe('added');
  });
  it('sorts every column the grid offers', () => {
    for (const sort of ['added', 'updated', 'score', 'name', 'company', 'title', 'event', 'owner', 'email']) {
      expect(contactsSearch.parse({ sort }).sort).toBe(sort);
    }
  });
});

describe('sort direction', () => {
  it('dates and scores start newest / best first, text A→Z', () => {
    expect(defaultDir('added')).toBe('desc');
    expect(defaultDir('updated')).toBe('desc');
    expect(defaultDir('score')).toBe('desc');
    expect(defaultDir('name')).toBe('asc');
    expect(defaultDir('company')).toBe('asc');
  });
  it('an explicit ?dir wins', () => {
    expect(sortDir({ sort: 'name', dir: 'desc' })).toBe('desc');
    expect(sortDir({ sort: 'score', dir: undefined })).toBe('desc');
  });
});

describe('contactQuery', () => {
  it('maps URL params to the /api/contacts query', () => {
    const s = contactsSearch.parse({ q: 'ana', band: 'B', event: 'E', owner: 'u1', sort: 'score' });
    expect(contactQuery(s)).toEqual({ search: 'ana', scoreBand: 'B', eventId: 'E', userId: 'u1', createdSince: undefined, sort: 'score', limit: 1000, lite: 1 });
  });
  it('?since=today asks for contacts created from local midnight', () => {
    const s = contactsSearch.parse({ since: 'today', band: 'A' });
    expect(s.since).toBe('today');
    const since = contactQuery(s).createdSince!;
    const d = new Date(since);
    expect([d.getHours(), d.getMinutes(), d.getSeconds()]).toEqual([0, 0, 0]);
    expect(d.toDateString()).toBe(new Date().toDateString());
    expect(contactsSearch.parse({ since: 'yesterday' }).since).toBeUndefined();
  });
  it('asks the server for the newest rows unless sorting by score (the 1,000 cap keeps the right ones)', () => {
    expect(contactQuery(contactsSearch.parse({ sort: 'company' })).sort).toBe('recent');
    expect(contactQuery(contactsSearch.parse({ sort: 'score', dir: 'asc' })).sort).toBe('score');
  });
  it('counts ignore the level (so every level pill keeps its number)', () => {
    expect(contactCountsQuery({ q: 'x', event: undefined, owner: 'u', since: undefined })).toEqual({ search: 'x', eventId: undefined, userId: 'u', createdSince: undefined });
  });
});

describe('dashboardSearch', () => {
  it('coerces ?days=30 and only allows the offered ranges', () => {
    expect(dashboardSearch.parse({ days: '30' })).toEqual({ days: 30 });
    expect(dashboardSearch.parse({ days: 90 })).toEqual({ days: 90 });
    expect(dashboardSearch.parse({ days: '5' })).toEqual({ days: 7 });
    expect(dashboardSearch.parse({})).toEqual({ days: 7 });
  });
});

describe('other page schemas', () => {
  it('validate their enums', () => {
    expect(activitySearch.parse({ type: 'pipeline' })).toEqual({ type: 'pipeline' });
    expect(activitySearch.parse({ type: 'nope' })).toEqual({});
    expect(pipelinesSearch.parse({ status: 'failed' })).toEqual({ status: 'failed' });
    expect(pipelinesSearch.parse({ status: 'x' })).toEqual({});
    expect(seatsSearch.parse({ q: 'ana', role: 'admin' })).toEqual({ q: 'ana', role: 'admin' });
    expect(seatsSearch.parse({ role: 'owner' })).toEqual({});
    expect(eventsSearch.parse({ event: 'HIMSS', category: '' })).toEqual({ event: 'HIMSS' });
  });
});

describe('clean', () => {
  it('drops empty values so URLs stay short', () => {
    expect(clean({ a: 'x', b: '', c: undefined, d: null, e: 0 })).toEqual({ a: 'x', e: 0 });
  });
});
