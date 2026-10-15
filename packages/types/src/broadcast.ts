import { z } from 'zod';
import { loose, type WireOf } from './wire';

/**
 * A team announcement.
 *   GET /api/broadcasts/active  — the admin console's call adds sentBy + recipients
 *   GET /api/broadcasts/mine    — the app's: adds endedAt
 */
export const broadcastSchema = loose({
  id: z.string(),
  message: z.string(),
  /** "normal" | "urgent" */
  priority: z.string(),
  isActive: z.boolean(),
  /** null = not recorded */
  createdAt: z.string().nullable(),
  /** /broadcasts/mine only: when an admin ended it (null = still live, or not recorded) */
  endedAt: z.string().nullable().optional(),
  /** /broadcasts/active for the admin console only: who sent it */
  sentBy: z.string().nullable().optional(),
  /** /broadcasts/active for the admin console only: who it went to ([] = everyone) */
  recipients: z.array(loose({ id: z.string(), name: z.string() })).optional(),
});
export type Broadcast = WireOf<typeof broadcastSchema>;
