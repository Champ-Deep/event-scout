import { describe, expect, it } from 'vitest';
import { contactsToCsv } from './csv';
import type { Contact } from '../../lib/types';

const c = (over: Partial<Contact>): Contact => ({ id: 'c1', firstName: 'Ana', lastName: 'Ruiz', score: 90, createdAt: '2026-09-30 10:00:00', ...over } as Contact);

describe('contactsToCsv', () => {
  it('starts with a BOM and the server export header', () => {
    const csv = contactsToCsv([]);
    expect(csv.charCodeAt(0)).toBe(0xfeff);
    expect(csv.slice(1)).toBe('ID,Name,Title,Company,Email,Phone,LinkedIn,Event,Owner,Admin score,Admin level,Rep score,Source,Notes,Added On');
  });
  it('writes one CRLF row per contact with owner name and level', () => {
    const rows = contactsToCsv([c({ userId: 'u1', company: 'Mercy' }), c({ id: 'c2', score: null })], { u1: 'Sam Lee' }).split('\r\n');
    expect(rows).toHaveLength(3);
    expect(rows[1]).toContain('c1,Ana Ruiz,,Mercy,');
    expect(rows[1]).toContain(',Sam Lee,90,hot,');
    expect(rows[2]).toContain(',,,'); // unscored: empty score and level
  });
  it('quotes commas, quotes and newlines', () => {
    const row = contactsToCsv([c({ company: 'Acme, Inc.', notes: 'said "hi"\nthen left' })]).split('\r\n')[1]!;
    expect(row).toContain('"Acme, Inc."');
    expect(row).toContain('"said ""hi""\nthen left"');
  });
  it('maps scores to the app levels', () => {
    const lvl = (score: number) => contactsToCsv([c({ score })]).split('\r\n')[1]!.split(',')[10];
    expect([lvl(85), lvl(84), lvl(70), lvl(69), lvl(50), lvl(49)]).toEqual(['hot', 'warm', 'warm', 'cool', 'cool', 'cold']);
  });
});
