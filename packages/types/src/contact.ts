import { z } from 'zod';
import { loose, type WireOf } from './wire';

/**
 * The contact read model, as apps/backend/contact_view.py builds it.
 *
 *   app view      format_contact → app_view            (the rep's app: adminNotes removed)
 *   console view  format_contact → console_scores      (admin console: score* / level = the ADMIN's score,
 *                                                        the rep's own moves to repScore* / repLevel)
 *   lite row      console/app view → lite (?lite=1)     (no score write-up; photoUrl / avatarUrl null, hasPhoto set)
 */

/** Score band (score_levels.band_for): A ≥ 85, B 70–84, C 50–69, D < 50; null = unscored. */
export const scoreBandSchema = z.enum(['A', 'B', 'C', 'D']);
export type ScoreBand = z.infer<typeof scoreBandSchema>;

/** Score level (score_levels.level_for): same thresholds as the bands; null = unscored. */
export const scoreLevelSchema = z.enum(['hot', 'warm', 'cool', 'cold']);
export type ScoreLevel = z.infer<typeof scoreLevelSchema>;

/** How a score was made (contact_view._score_meta / admin_score_out.meta). */
export const scoreMetaSchema = loose({
  runs: z.number().nullish(),
  spread: z.number().nullish(),
  model: z.string().nullish(),
  scoredAt: z.string().nullish(),
});
export type ScoreMeta = WireOf<typeof scoreMetaSchema>;

/** Five 0–20 dimensions of a score (role_fit, company_fit, …). */
const scoreBreakdownSchema = z.record(z.number());

export const contactSchema = loose({
  id: z.string(),
  userId: z.string().nullable(),
  /** the event it was captured at; "" = none */
  eventId: z.string(),
  /** never empty: "Unknown" when the stored name is blank */
  firstName: z.string(),
  lastName: z.string().nullable(),
  company: z.string().nullable(),
  title: z.string().nullable(),
  email: z.string().nullable(),
  phone: z.string().nullable(),
  linkedin: z.string().nullable(),
  notes: z.string().nullable(),
  /** admin console only (the app view removes it) */
  adminNotes: z.string().nullable().optional(),
  /** app: the rep's score · console: the admin's own score */
  score: z.number().nullable(),
  scoreBand: scoreBandSchema.nullable(),
  level: scoreLevelSchema.nullable(),
  /** left out of lite rows */
  scoreReasoning: z.string().nullable().optional(),
  scoreBreakdown: scoreBreakdownSchema.nullable().optional(),
  scoreMeta: scoreMetaSchema.nullable().optional(),
  scoreActions: z.array(z.string()).nullable().optional(),
  /** score_levels.stage_for: "qualified" (hot) or "new" */
  stage: z.string(),
  /** "scan" · "qr" · "manual" · "import" … */
  source: z.string(),
  scannedAt: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  /** the photo, else an SVG initials avatar; null in lite rows */
  avatarUrl: z.string().nullable(),
  /** a real picture, else null; always null in lite rows (see hasPhoto) */
  photoUrl: z.string().nullable(),
  version: z.number(),
  /** lite rows only: whether the contact has a photo */
  hasPhoto: z.boolean().optional(),
  // ---- console view only: the rep's own score (read only)
  repScore: z.number().nullable().optional(),
  repScoreBand: scoreBandSchema.nullable().optional(),
  repLevel: scoreLevelSchema.nullable().optional(),
  repScoreReasoning: z.string().nullable().optional(),
  repScoreBreakdown: scoreBreakdownSchema.nullable().optional(),
  repScoreMeta: scoreMetaSchema.nullable().optional(),
  repScoreActions: z.array(z.string()).nullable().optional(),
});
export type Contact = WireOf<typeof contactSchema>;

/** GET /api/contacts — one page; `cursor` = the next page's, null on the last. */
export const contactPageSchema = loose({
  contacts: z.array(contactSchema),
  cursor: z.string().nullable(),
});
export type ContactPage = WireOf<typeof contactPageSchema>;

// ---------------------------------------------------------------- duplicates

/** Why two contacts look like the same person. */
export const duplicateReasonSchema = z.enum(['email', 'phone', 'name']);
export type DuplicateReason = z.infer<typeof duplicateReasonSchema>;

/**
 * A contact that looks like another one.
 * GET /api/contacts/{id}/duplicates, POST /api/contacts/check-duplicates: {contact, reasons};
 * GET /api/admin/contacts/{id}/duplicates adds confidence, ownerName and sameRep.
 */
export const duplicateMatchSchema = loose({
  contact: contactSchema,
  reasons: z.array(duplicateReasonSchema),
  confidence: z.number().optional(),
  /** the match's rep (admin endpoint) */
  ownerName: z.string().optional(),
  /** false = another rep captured the same person (admin endpoint) */
  sameRep: z.boolean().optional(),
});
export type DuplicateMatch = WireOf<typeof duplicateMatchSchema>;

export const duplicateMatchesSchema = loose({ matches: z.array(duplicateMatchSchema) });

/** One pair of the rep's own look-alike contacts (GET /api/contacts/duplicates → {pairs, count}). */
export const duplicatePairSchema = loose({
  a: contactSchema,
  b: contactSchema,
  reasons: z.array(duplicateReasonSchema),
});
export type DuplicatePair = WireOf<typeof duplicatePairSchema>;

export const duplicatePairsSchema = loose({ pairs: z.array(duplicatePairSchema), count: z.number() });

/** A look-alike pair anywhere in the workspace (GET /api/admin/contacts/duplicates, an array of these). */
export const workspaceDuplicatePairSchema = loose({
  /** "{lowId}:{highId}" */
  id: z.string(),
  confidence: z.number(),
  reasons: z.array(duplicateReasonSchema),
  reason: z.string(),
  /** rep of the contact that's kept (primary) */
  owner: z.string(),
  ownerId: z.string(),
  /** rep of the contact that would be merged in */
  duplicateOwner: z.string(),
  duplicateOwnerId: z.string(),
  /** two different reps captured the same person */
  crossRep: z.boolean(),
  primary: contactSchema,
  duplicate: contactSchema,
});
export type WorkspaceDuplicatePair = WireOf<typeof workspaceDuplicatePairSchema>;

// ---------------------------------------------------------------- recently deleted

/** A contact in Recently deleted (api_routes._format_deleted). */
export const deletedContactSchema = loose({
  id: z.string(),
  name: z.string(),
  company: z.string().nullable(),
  title: z.string().nullable(),
  photoUrl: z.string().nullable(),
  /** "Merged into …" when it was merged away; null = deleted */
  reason: z.string().nullable(),
  deletedAt: z.string().nullable(),
  purgeAfter: z.string().nullable(),
  daysLeft: z.number(),
});
export type DeletedContact = WireOf<typeof deletedContactSchema>;

/** Everyone's Recently deleted, as the admin console lists it: plus whose it is. */
export const workspaceDeletedContactSchema = deletedContactSchema.extend({
  ownerId: z.string(),
  owner: z.string(),
});
export type WorkspaceDeletedContact = WireOf<typeof workspaceDeletedContactSchema>;

/** GET /api/contacts/deleted (the rep's) and GET /api/admin/contacts/deleted (everyone's). */
export const deletedContactsSchema = loose({ contacts: z.array(deletedContactSchema), retentionDays: z.number() });
export const workspaceDeletedContactsSchema = loose({ contacts: z.array(workspaceDeletedContactSchema), retentionDays: z.number() });
