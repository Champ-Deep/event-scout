import type { Contact } from '../../lib/types';
import { contactLevel } from '../../lib/levels';

// Spreadsheet formula-injection guard (same rule as the server, sanitize.py csv_safe): a cell starting with
// = + - @ tab or CR gets a leading apostrophe so Excel / Sheets show it as text; numbers and phones stay as they are.
const neutralise = (s: string) => (/^[=+\-@\t\r]/.test(s) && !/^[+-]?[\d\s().,/-]+$/.test(s) ? `'${s}` : s);
const cell = (v: unknown) => {
  const s = neutralise(v == null ? '' : String(v));
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** CSV of the given contacts, with a BOM so Excel reads UTF-8. Score = the admin's own score; Rep score = the rep's. */
export function contactsToCsv(rows: Contact[], owners: Record<string, string> = {}) {
  const head = ['ID', 'Name', 'Title', 'Company', 'Email', 'Phone', 'LinkedIn', 'Event', 'Owner', 'Admin score', 'Admin level', 'Rep score', 'Source', 'Notes', 'Added On'];
  const lines = rows.map((c) => [
    c.id, [c.firstName, c.lastName].filter(Boolean).join(' '), c.title, c.company, c.email, c.phone, c.linkedin, c.eventId,
    owners[c.userId || ''] || '', c.score ?? '', contactLevel(c)?.key ?? '', c.repScore ?? '', c.source, c.notes, c.createdAt,
  ].map(cell).join(','));
  return '﻿' + [head.join(','), ...lines].join('\r\n');
}
