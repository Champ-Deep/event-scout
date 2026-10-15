// The wire contract (@es/types) against payloads shaped exactly as apps/backend builds them
// (contact_view.format_contact / app_view / console_scores / lite, api_routes handlers, app.py chat_card).
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import {
  addedExhibitorSchema, broadcastSchema, contactFileSchema, contactPageSchema, contactSchema, createResponseCheck, deletedContactsSchema,
  describeDrift, duplicateMatchesSchema, duplicatePairsSchema, eventFileSchema, eventSummarySchema, exhibitorSchema, fileContentSchema,
  retrievedContactsSchema, workspaceDeletedContactsSchema, workspaceDuplicatePairSchema,
  type Contact, type Exhibitor,
} from '@es/types';

const T0 = '2026-10-14T09:12:03.120000+00:00';
const T1 = '2026-10-14T11:40:00+00:00';
const SVG = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 80 80'></svg>";
const JPEG = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ';

/** format_contact → app_view: a scored contact as the rep's app gets it. */
const appContact = {
  id: '6d1f4a52-7c1e-4b8e-9b0a-0f5e2a9c1d11',
  userId: '0b8a3c55-1d2e-4f60-8a7b-9c0d1e2f3a4b',
  eventId: 'HIMSS 2026',
  firstName: 'Ana',
  lastName: 'Ruiz',
  company: 'Mercy Health',
  title: 'CIO',
  email: 'ana@mercy.org',
  phone: '+1 555 0100',
  linkedin: 'https://www.linkedin.com/in/anaruiz',
  notes: 'Met at the booth; wants a demo',
  score: 88,
  scoreBand: 'A',
  level: 'hot',
  scoreReasoning: 'CIO of a 40-hospital system actively replacing its data platform.',
  scoreBreakdown: { role_fit: 18, company_fit: 17, need_fit: 18, goal_fit: 17, intent: 18 },
  scoreMeta: { runs: 3, spread: 4, model: 'google/gemini-2.5-flash', scoredAt: T0 },
  scoreActions: ['Send the Mercy case study', 'Book a 30-minute demo'],
  stage: 'qualified',
  source: 'scan',
  scannedAt: T0,
  createdAt: T0,
  updatedAt: T1,
  avatarUrl: JPEG,
  photoUrl: JPEG,
  version: 1,
};

/** An unscored QR scan with a one-word name and no picture (score fields null, SVG avatar). */
const unscoredContact = {
  ...appContact,
  id: '9a0b1c2d-3e4f-4a5b-8c6d-7e8f9a0b1c2d',
  eventId: '',
  firstName: 'Unknown',
  lastName: null,
  company: null,
  title: null,
  email: null,
  phone: null,
  linkedin: null,
  notes: null,
  score: null,
  scoreBand: null,
  level: null,
  scoreReasoning: null,
  scoreBreakdown: null,
  scoreMeta: null,
  scoreActions: null,
  stage: 'new',
  source: 'qr',
  updatedAt: T0,
  avatarUrl: SVG,
  photoUrl: null,
};

/** console_scores: score* = the admin's own score (here: none yet), the rep's moved to repScore*. */
const consoleContact = {
  ...appContact,
  adminNotes: 'Budget approved for Q1',
  score: null,
  scoreBand: null,
  level: null,
  scoreReasoning: null,
  scoreBreakdown: null,
  scoreMeta: null,
  scoreActions: null,
  stage: 'new',
  repScore: 88,
  repScoreBand: 'A',
  repLevel: 'hot',
  repScoreReasoning: appContact.scoreReasoning,
  repScoreBreakdown: appContact.scoreBreakdown,
  repScoreMeta: appContact.scoreMeta,
  repScoreActions: appContact.scoreActions,
};

/** contact_view.lite: the score write-up dropped, photo / avatar null, hasPhoto set. */
const LITE_DROP = ['scoreReasoning', 'scoreBreakdown', 'scoreActions', 'scoreMeta',
  'repScoreReasoning', 'repScoreBreakdown', 'repScoreActions', 'repScoreMeta'];
const lite = (view: Record<string, unknown>) => ({
  ...Object.fromEntries(Object.entries(view).filter(([k]) => !LITE_DROP.includes(k))),
  hasPhoto: Boolean(view.photoUrl), photoUrl: null, avatarUrl: null,
});

const ok = (schema: z.ZodTypeAny, data: unknown) => expect(describeDrift(schema, data, 'test')).toBeNull();
const drift = (schema: z.ZodTypeAny, data: unknown) => {
  const report = describeDrift(schema, data, 'GET /test');
  expect(report).not.toBeNull();
  return report as string;
};

describe('contact', () => {
  it('accepts the app view, an unscored scan, the console view and lite rows', () => {
    ok(contactSchema, appContact);
    ok(contactSchema, unscoredContact);
    ok(contactSchema, consoleContact);
    ok(contactSchema, lite(consoleContact));
    ok(contactSchema, lite(appContact));
    ok(contactPageSchema, { contacts: [lite(consoleContact), lite(unscoredContact)], cursor: '1000' });
    ok(contactPageSchema, { contacts: [], cursor: null });
  });

  it('accepts score meta with keys the stored _meta never had (null)', () => {
    ok(contactSchema, { ...appContact, scoreMeta: { runs: null, spread: null, model: null, scoredAt: null } });
  });

  it('lets unknown extra fields through, at any depth', () => {
    const extra = { ...appContact, crmLinks: [{ crm: 'hubspot' }], scoreMeta: { ...appContact.scoreMeta, temperatureModel: 'x' } };
    ok(contactSchema, extra);
    expect(contactSchema.parse(extra)).toMatchObject({ crmLinks: [{ crm: 'hubspot' }] });
  });

  it('reports drift: a missing / wrong field, an unknown band', () => {
    const { firstName: _f, ...noName } = appContact;
    expect(drift(contactSchema, noName)).toContain('firstName: Required');
    expect(drift(contactSchema, { ...appContact, scoreBand: 'F' })).toContain('scoreBand');
    expect(drift(contactSchema, { ...appContact, score: '88' })).toContain('score: Expected number');
  });

  it('types: no index signature (typos stay compile errors) and Omit keeps the known fields', () => {
    const c = appContact as unknown as Contact;
    // @ts-expect-error — not a Contact field
    void c.notAField;
    const rest: Omit<Contact, 'adminNotes'> = c;
    expect(rest.firstName).toBe('Ana');
  });
});

describe('duplicates', () => {
  it('rep endpoints: {matches} and {pairs, count}', () => {
    ok(duplicateMatchesSchema, { matches: [{ contact: appContact, reasons: ['email', 'name'] }] });
    ok(duplicateMatchesSchema, { matches: [] });
    ok(duplicatePairsSchema, { pairs: [{ a: appContact, b: unscoredContact, reasons: ['phone'] }], count: 1 });
  });

  it('admin endpoints: matches with confidence / owner, and workspace pairs', () => {
    ok(duplicateMatchesSchema, { matches: [{ contact: consoleContact, reasons: ['email'], confidence: 98, ownerName: 'Maya Patel', sameRep: false }] });
    ok(z.array(workspaceDuplicatePairSchema), [{
      id: `${appContact.id}:${unscoredContact.id}`, confidence: 92, reasons: ['phone'], reason: 'Same phone number',
      owner: 'Maya Patel', ownerId: appContact.userId, duplicateOwner: 'Sam Lee', duplicateOwnerId: '1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f',
      crossRep: true, primary: consoleContact, duplicate: lite(consoleContact),
    }]);
  });

  it('reports an unknown reason', () => {
    expect(drift(duplicateMatchesSchema, { matches: [{ contact: appContact, reasons: ['website'] }] })).toContain('matches.*.reasons.*');
  });
});

describe('recently deleted', () => {
  const row = {
    id: '3c4d5e6f-7a8b-4c9d-8e0f-1a2b3c4d5e6f', name: 'Ana Ruiz', company: 'Mercy Health', title: null, photoUrl: null,
    reason: 'Merged into Ana Ruiz', deletedAt: T0, purgeAfter: '2026-11-13T09:12:03+00:00', daysLeft: 30,
  };
  it('the rep list and everyone\'s (with the owner)', () => {
    ok(deletedContactsSchema, { contacts: [row, { ...row, reason: null, deletedAt: null, purgeAfter: null, daysLeft: 0 }], retentionDays: 30 });
    ok(workspaceDeletedContactsSchema, { contacts: [{ ...row, ownerId: appContact.userId, owner: 'Unknown' }], retentionDays: 30 });
  });
  it('the console list needs the owner', () => {
    expect(drift(workspaceDeletedContactsSchema, { contacts: [row], retentionDays: 30 })).toContain('contacts.*.ownerId: Required');
  });
});

describe('files', () => {
  it('contact files (list and upload answer), team files, file content', () => {
    const file = { id: 'f1', filename: 'Proposal.pdf', fileType: 'pdf', category: 'general', fileSize: 48213, description: '', createdAt: T0 };
    ok(z.array(contactFileSchema), [file, { ...file, category: null, createdAt: null }]);
    ok(contactFileSchema, file);
    ok(z.array(eventFileSchema), [{
      id: 'e1', filename: 'Floor plan.pdf', fileType: 'pdf', mimeType: 'application/pdf', category: 'floorplan', fileSize: 1048576,
      description: 'Hall A', downloadCount: 3, eventName: null, placeholder: false, createdAt: T0,
    }]);
    ok(fileContentSchema, { filename: 'Proposal.pdf', mimeType: 'application/pdf', fileSize: 5, contentBase64: 'JVBERi0=' });
  });
});

describe('events & exhibitors', () => {
  const booth = {
    id: 'x1', event: 'HIMSS 2026', name: 'Epic Systems', booth: '1459', hall: 'Hall A', category: 'EHR', country: 'US',
    website: 'https://epic.com', description: '', subcategory: null, tags: ['ehr'], products: [], addedByMe: false, userAdded: false,
    visited: false, visitedAt: null, visitNote: '',
  };
  it('the directory, an added booth (existed) and a visited one', () => {
    ok(z.array(exhibitorSchema), [booth, { ...booth, id: 'x2', userAdded: true, addedByMe: true, visited: true, visitedAt: T1, visitNote: 'Ask for Dana' }]);
    ok(addedExhibitorSchema, { ...booth, existed: true });
    ok(z.array(eventSummarySchema), [{ id: 'himss-2026', name: 'HIMSS 2026', slug: 'himss-2026', exhibitorCount: 1200 }]);
  });
  it('an added booth must say whether it existed', () => {
    expect(drift(addedExhibitorSchema, booth)).toContain('existed: Required');
  });
  it('types: an exhibitor has the visit fields', () => {
    const e = booth as Exhibitor;
    expect(e.visitNote).toBe('');
  });
});

describe('broadcasts', () => {
  const b = { id: 'b1', message: 'Stand-up at 8:30 at the booth', priority: 'urgent', isActive: true, createdAt: T0 };
  it('active (console: sender + recipients; app: neither), mine (endedAt), unknown time', () => {
    ok(z.array(broadcastSchema), [{ ...b, sentBy: 'Maya Patel', recipients: [{ id: 'u2', name: 'Removed account' }] }, { ...b, sentBy: null, recipients: [] }]);
    ok(z.array(broadcastSchema), [b]);
    ok(z.array(broadcastSchema), [{ ...b, isActive: false, endedAt: T1 }, { ...b, createdAt: null, endedAt: null }]);
  });
});

describe('assistant retrieved contacts (chat_card)', () => {
  const card = {
    id: appContact.id, name: 'Ana Ruiz', email: 'ana@mercy.org', phone: 'N/A', linkedin: 'N/A', company_name: 'Mercy Health',
    lead_score: 88, lead_temperature: 'hot', lead_score_reasoning: 'CIO …', lead_recommended_actions: ['Book a demo'],
    photo_base64: '', source: 'scan', notes: 'Met at the booth',
  };
  it('the done / non-streamed answer, the meta event (no photo) and the console swap (admin score, no reasoning)', () => {
    ok(retrievedContactsSchema, [card]);
    const { photo_base64: _p, ...meta } = card;
    ok(retrievedContactsSchema, [meta]);
    ok(retrievedContactsSchema, [{ ...card, lead_score: null, lead_temperature: null, lead_score_reasoning: '', lead_recommended_actions: [] }]);
  });
});

describe('describeDrift', () => {
  it('folds array indices and counts repeats; caps the lines', () => {
    const rows = Array.from({ length: 40 }, (_, i) => ({ ...appContact, id: `c${i}`, scoreBand: 'E' }));
    const report = drift(contactPageSchema, { contacts: rows, cursor: null });
    expect(report).toContain('GET /test');
    expect(report).toMatch(/contacts\.\*\.scoreBand: .*\(x40\)/);
    expect(report.split('\n')).toHaveLength(2);
    const many = Object.fromEntries(Object.keys(appContact).map((k) => [k, Symbol('x')]));
    expect(drift(contactSchema, many)).toMatch(/…and \d+ more/);
  });
  it('the root itself', () => {
    expect(drift(contactSchema, null)).toContain('(root)');
  });
});

describe('createResponseCheck', () => {
  it('enabled: warns about drift once, returns the very same data', () => {
    const warn = vi.fn();
    const check = createResponseCheck({ enabled: true, warn });
    const data = { ...appContact, firstName: 7 };
    expect(check(contactSchema, data, 'GET /api/contacts/{id}')).toBe(data);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toContain('GET /api/contacts/{id}');
    expect(check(contactSchema, appContact, 'GET /api/contacts/{id}')).toBe(appContact);
    expect(warn).toHaveBeenCalledTimes(1);
  });
  it('disabled (production): parses nothing', () => {
    const safeParse = vi.fn();
    const check = createResponseCheck({ enabled: false, warn: () => { throw new Error('no'); } });
    const data = { any: 'thing' };
    expect(check({ safeParse } as unknown as z.ZodTypeAny, data, 'x')).toBe(data);
    expect(safeParse).not.toHaveBeenCalled();
  });
  it('never throws, even when the schema or the warning does', () => {
    const boom = { safeParse: () => { throw new Error('boom'); } } as unknown as z.ZodTypeAny;
    const check = createResponseCheck({ enabled: true, warn: () => { throw new Error('warn failed'); } });
    expect(() => check(boom, 1, 'x')).not.toThrow();
    expect(check(contactSchema, undefined, 'x')).toBeUndefined();
  });
});
