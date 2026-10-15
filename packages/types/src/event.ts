import { z } from 'zod';
import { loose, type WireOf } from './wire';

/** An event with an imported exhibitor directory (GET /api/events). */
export const eventSummarySchema = loose({
  /** the event's slug */
  id: z.string(),
  name: z.string(),
  slug: z.string(),
  exhibitorCount: z.number(),
});
export type EventSummary = WireOf<typeof eventSummarySchema>;

/**
 * A booth in an event's exhibitor directory (api_routes._exhibitor_out): GET /api/exhibitors, POST (plus
 * `existed`) / PATCH /api/exhibitors, PUT / DELETE /api/exhibitors/{id}/visit.
 * The text fields are "" when blank (every writer stores a string).
 */
export const exhibitorSchema = loose({
  id: z.string(),
  event: z.string(),
  name: z.string(),
  booth: z.string(),
  hall: z.string(),
  category: z.string(),
  country: z.string(),
  website: z.string(),
  description: z.string(),
  subcategory: z.string().nullable(),
  tags: z.array(z.string()),
  products: z.array(z.string()),
  /** the signed-in rep added it (they can edit / remove it); always false for the admin console */
  addedByMe: z.boolean(),
  /** a booth a rep added from the app (not the imported directory) */
  userAdded: z.boolean(),
  /** the signed-in rep marked it visited */
  visited: z.boolean(),
  visitedAt: z.string().nullable(),
  visitNote: z.string(),
});
export type Exhibitor = WireOf<typeof exhibitorSchema>;

/** POST /api/exhibitors: the booth, and whether the event already listed it. */
export const addedExhibitorSchema = exhibitorSchema.extend({ existed: z.boolean() });
