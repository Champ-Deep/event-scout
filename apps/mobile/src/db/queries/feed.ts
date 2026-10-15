import { desc, eq, and } from 'drizzle-orm';
import { db } from '../client';
import { contacts } from '../schema';
import { hydrateContact } from './contacts';

export interface FeedFilters {
  eventId?: string | undefined;
  userId?: string | undefined;
  search?: string | undefined;
  stage?: 'new' | 'contacted' | 'qualified' | 'converted' | undefined;
  scoreBand?: 'A' | 'B' | 'C' | 'D' | 'F' | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
  cursor?: string | undefined;
  sort?: 'recent' | 'score' | undefined;
  /** Server only: follow `cursor` until every page is loaded (`limit` = page size). */
  allPages?: boolean | undefined;
}

export const feedQueries = {
  getRankedFeed: async (filters: FeedFilters = {}) => {
    const conditions = [];

    if (filters.eventId) {
      conditions.push(eq(contacts.eventId, filters.eventId));
    }
    if (filters.stage) {
      conditions.push(eq(contacts.stage, filters.stage));
    }
    if (filters.scoreBand) {
      conditions.push(eq(contacts.scoreBand, filters.scoreBand));
    }

    const rows = await db
      .select()
      .from(contacts)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(contacts.score), desc(contacts.scannedAt), desc(contacts.createdAt));
    return (Array.isArray(rows) ? rows : []).map(hydrateContact);
  },
};
