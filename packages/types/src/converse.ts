import { z } from 'zod';
import { loose, type WireOf } from './wire';

/**
 * A contact the assistant's answer drew on (contact_view.chat_card) — `retrieved_contacts` of the app's
 * POST /converse/ (and its streams) and of the admin console's /api/converse/stream `meta` / `done` events.
 * Snake case, as the assistant builds it; missing text fields are "N/A". The streams' `meta` event leaves
 * out photo_base64.
 */
export const retrievedContactSchema = loose({
  id: z.string(),
  name: z.string(),
  email: z.string(),
  phone: z.string(),
  linkedin: z.string(),
  company_name: z.string(),
  lead_score: z.number().nullable(),
  lead_temperature: z.string().nullable(),
  lead_score_reasoning: z.string(),
  lead_recommended_actions: z.array(z.string()),
  photo_base64: z.string().optional(),
  source: z.string(),
  notes: z.string(),
});
export type RetrievedContact = WireOf<typeof retrievedContactSchema>;

export const retrievedContactsSchema = z.array(retrievedContactSchema);
