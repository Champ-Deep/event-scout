import { eq, inArray } from 'drizzle-orm';
import { db } from '../client';
import { contacts } from '../schema';

/** Row → contact: the stored full JSON (raw) wins over the basic columns. */
export function hydrateContact(row: any): any {
  if (!row) return row;
  const { raw, ...cols } = row;
  if (!raw) return cols;
  try {
    return { ...cols, ...JSON.parse(raw) };
  } catch {
    return cols;
  }
}

export const contactsQueries = {
  getById: async (id: string) => {
    const results = await db.select().from(contacts).where(eq(contacts.id, id));
    return hydrateContact(results[0] ?? null);
  },

  allIds: async (): Promise<string[]> => {
    const rows = await db.select({ id: contacts.id }).from(contacts);
    return (Array.isArray(rows) ? rows : []).map((r: any) => String(r.id));
  },

  deleteMany: async (ids: string[]) => {
    for (let i = 0; i < ids.length; i += 200) {
      await db.delete(contacts).where(inArray(contacts.id, ids.slice(i, i + 200)));
    }
  },

  upsert: async (contactData: typeof contacts.$inferInsert) => {
    return db
      .insert(contacts)
      .values(contactData)
      .onConflictDoUpdate({
        target: contacts.id,
        set: {
          ...contactData,
          updatedAt: new Date().toISOString(),
        },
      });
  },

  delete: async (id: string) => {
    return db.delete(contacts).where(eq(contacts.id, id));
  },
};
