// Semantic contact search. Replaces the in-memory FAISS index: one Workers AI embedding per
// contact in D1, cosine similarity in the Worker.
// shortcut: brute-force scan of a user's vectors, fine to a few thousand contacts; move to Vectorize past that.
import { Env, all, run, now } from "./lib";

const EMBED_MODEL = "@cf/baai/bge-small-en-v1.5";

/** Same summary text app.py fed to FAISS. */
export function contactText(c: any): string {
  let s = `${c.name || ""}, ${c.email || ""}, ${c.phone || ""}, ${c.linkedin || ""}, ${c.company_name || ""}`;
  if (c.title) s += `, ${c.title}`;
  if (c.notes) s += `, ${String(c.notes).slice(0, 200)}`;
  return s;
}

export async function embed(env: Env, texts: string[]): Promise<number[][]> {
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += 50) {
    const r: any = await env.AI.run(EMBED_MODEL as any, { text: texts.slice(i, i + 50) } as any);
    out.push(...r.data);
  }
  return out;
}

export async function indexContacts(env: Env, contacts: any[]) {
  if (!contacts.length) return;
  const vecs = await embed(env, contacts.map(contactText));
  const ts = now();
  await env.DB.batch(
    contacts.map((c, i) =>
      env.DB.prepare(
        "INSERT INTO contact_embeddings (contact_id, user_id, vector, updated_at) VALUES (?, ?, ?, ?) " +
          "ON CONFLICT(contact_id) DO UPDATE SET user_id = excluded.user_id, vector = excluded.vector, updated_at = excluded.updated_at",
      ).bind(c.id, c.user_id, JSON.stringify(vecs[i].map((x) => +x.toFixed(5))), ts),
    ),
  );
}

/** Fire-and-forget index refresh; a failed embedding must never fail the contact write. */
export function reindexLater(env: Env, exec: ExecutionContext, contacts: any[]) {
  exec.waitUntil(indexContacts(env, contacts).catch((e) => console.log("embed failed", String(e))));
}

export async function dropEmbedding(env: Env, contactId: string) {
  await run(env, "DELETE FROM contact_embeddings WHERE contact_id = ?", contactId);
}

function cosine(a: number[], b: number[]) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na * nb) || 1);
}

/** Top-k contacts for a query. userId null = every user (admin mode). Returns full contact rows. */
export async function searchContacts(env: Env, query: string, userId: string | null, k = 4): Promise<any[]> {
  const [qv] = await embed(env, [query]);
  const rows = userId
    ? await all(env, "SELECT contact_id, vector FROM contact_embeddings WHERE user_id = ?", userId)
    : await all(env, "SELECT contact_id, vector FROM contact_embeddings");
  const top = rows
    .map((r: any) => ({ id: r.contact_id as string, s: cosine(qv, JSON.parse(r.vector)) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, k);
  if (!top.length) return [];
  const contacts = await all(env, `SELECT * FROM contacts WHERE id IN (${top.map(() => "?").join(",")})`, ...top.map((t) => t.id));
  const byId = new Map(contacts.map((c: any) => [c.id, c]));
  return top.map((t) => byId.get(t.id)).filter(Boolean);
}
