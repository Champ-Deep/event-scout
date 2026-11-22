import type { ExtractedFields } from "@/api/types";

export type QueuedCard = {
  id: string;
  userId: string;
  blob: Blob;
  created: number;
  attempts: number;
  fields?: ExtractedFields;
  status?: "queued" | "failed";
  error?: string;
  filename?: string;
  unreadable?: boolean;
};
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open("event-scout-scan-queue", 1);
    r.onupgradeneeded = () =>
      r.result.createObjectStore("cards", { keyPath: "id" });
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function transaction<T>(
  mode: IDBTransactionMode,
  act: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("cards", mode);
    const request = act(tx.objectStore("cards"));
    tx.oncomplete = () => {
      resolve(request.result);
      db.close();
    };
    tx.onerror = tx.onabort = () => {
      reject(tx.error || request.error);
      db.close();
    };
  });
}
export const queuedCards = async (userId: string) =>
  ((await transaction("readonly", (s) => s.getAll())) as QueuedCard[])
    .filter((c) => c.userId === userId)
    .sort((a, b) => a.created - b.created);
export const putCard = (card: QueuedCard) =>
  transaction("readwrite", (s) => s.put(card));
export const removeCard = (id: string) =>
  transaction("readwrite", (s) => s.delete(id));
// Serial within the one mounted scan surface, including StrictMode effect replay.
export const queueLock = { busy: false };
