import { and, asc, eq, inArray, sql } from 'drizzle-orm';
import { db } from '../client';
import { outbox } from '../schema';

export const outboxQueries = {
  getPending: async (limit = 50) => {
    return db
      .select()
      .from(outbox)
      .orderBy(asc(outbox.createdAt))
      .limit(limit);
  },

  /** The whole queue, oldest first (no cap: a long offline day can queue hundreds of changes). */
  getAll: async () => {
    const rows = await db.select().from(outbox).orderBy(asc(outbox.createdAt));
    return Array.isArray(rows) ? rows : [];
  },

  /** Every contact id with a change still waiting to upload. */
  allEntityIds: async (): Promise<string[]> => {
    const rows = await db.select({ entityId: outbox.entityId }).from(outbox);
    return (Array.isArray(rows) ? rows : []).map((r: any) => String(r.entityId));
  },

  hasPendingCreate: async (entityId: string): Promise<boolean> => {
    const rows = await db.select({ id: outbox.id }).from(outbox)
      .where(and(eq(outbox.entityId, entityId), eq(outbox.action, 'create'))).limit(1);
    return Array.isArray(rows) && rows.length > 0;
  },

  /** Drop every queued change for one contact (e.g. created and deleted before it ever uploaded). */
  deleteForEntity: async (entityId: string) => {
    return db.delete(outbox).where(eq(outbox.entityId, entityId));
  },

  enqueue: async (item: typeof outbox.$inferInsert) => {
    return db.insert(outbox).values(item);
  },

  markAcknowledged: async (ids: string[]) => {
    if (ids.length === 0) return;
    for (let i = 0; i < ids.length; i += 200) {
      await db.delete(outbox).where(inArray(outbox.id, ids.slice(i, i + 200)));
    }
  },

  /** `countsAsAttempt: false` = not this item's fault (offline, timeout, waiting on another change):
   *  the reason is kept for display but the attempt count — which stops automatic retries — isn't raised. */
  recordAttemptFailure: async (id: string, error: string, countsAsAttempt = true) => {
    const existing = await db.select().from(outbox).where(eq(outbox.id, id)).limit(1);
    if (!existing[0]) return;

    return db
      .update(outbox)
      .set({
        attempts: existing[0].attempts + (countsAsAttempt ? 1 : 0),
        lastError: error,
      })
      .where(eq(outbox.id, id));
  },

  count: async () => {
    const rows = await db.select({ n: sql<number>`count(*)` }).from(outbox);
    return Number(rows?.[0]?.n ?? 0) || 0;
  },
};
