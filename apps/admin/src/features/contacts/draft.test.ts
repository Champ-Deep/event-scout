import { describe, expect, it } from 'vitest';
import { diffDraft, draftOf } from './draft';
import type { Contact } from '../../lib/types';

const contact = (over: Partial<Contact> = {}): Contact => ({
  id: 'c1', userId: 'u1', eventId: 'HIMSS', firstName: 'Ana', lastName: 'Ruiz', company: 'Mercy', title: 'CIO', email: 'ana@mercy.org',
  phone: '+1 555', linkedin: null, notes: 'met at booth', adminNotes: null, score: 80, scoreBand: 'B', scoreReasoning: null,
  scoreBreakdown: null, scoreActions: null, scoreMeta: null, createdAt: '2026-10-01 10:00:00',
  level: 'warm', stage: 'new', source: 'scan', scannedAt: '2026-10-01 10:00:00', updatedAt: '2026-10-01 10:00:00',
  avatarUrl: null, photoUrl: null, version: 1, ...over,
});

describe('draftOf', () => {
  it('shows a nameless scan ("Unknown") as an empty first name', () => {
    expect(draftOf(contact({ firstName: 'Unknown', lastName: null })).firstName).toBe('');
    expect(draftOf(contact({ firstName: 'Unknown', lastName: 'Ruiz' })).firstName).toBe('Unknown');
  });
  it('null fields become empty strings; the photo starts unchanged', () => {
    const d = draftOf(contact({ company: null, linkedin: null }));
    expect(d.company).toBe('');
    expect(d.linkedin).toBe('');
    expect(d.photo).toBeUndefined();
  });
});

describe('diffDraft', () => {
  it('an untouched draft sends nothing', () => {
    const c = contact();
    expect(diffDraft(c, draftOf(c))).toEqual({});
  });
  it('whitespace-only differences are not changes', () => {
    const c = contact();
    expect(diffDraft(c, { ...draftOf(c), company: '  Mercy ', notes: 'met at booth  ' })).toEqual({});
  });
  it('sends only the changed fields, trimmed', () => {
    const c = contact();
    expect(diffDraft(c, { ...draftOf(c), title: '  CTO ', email: 'ana@new.org' })).toEqual({ title: 'CTO', email: 'ana@new.org' });
  });
  it('a cleared optional field is sent as null; a cleared first name as ""', () => {
    const c = contact();
    expect(diffDraft(c, { ...draftOf(c), lastName: ' ', eventId: '' })).toEqual({ lastName: null, eventId: null });
    expect(diffDraft(c, { ...draftOf(c), firstName: '' })).toEqual({ firstName: '' });
  });
  it('naming an "Unknown" scan sends the new first name', () => {
    const c = contact({ firstName: 'Unknown', lastName: null });
    expect(diffDraft(c, { ...draftOf(c), firstName: 'Sam' })).toEqual({ firstName: 'Sam' });
  });
  it('notes keep their inner text as typed (only trailing space is ignored)', () => {
    const c = contact();
    expect(diffDraft(c, { ...draftOf(c), notes: '  met at booth' })).toEqual({ notes: '  met at booth' });
  });
  it('photo: a new data URI or null (remove) is sent; undefined is not', () => {
    const c = contact();
    expect(diffDraft(c, { ...draftOf(c), photo: 'data:image/png;base64,AA' })).toEqual({ photo: 'data:image/png;base64,AA' });
    expect(diffDraft(c, { ...draftOf(c), photo: null })).toEqual({ photo: null });
  });
});
