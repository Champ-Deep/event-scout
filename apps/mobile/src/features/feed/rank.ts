import type { Contact } from '@es/types';

export function rankContacts(items: Contact[]): Contact[] {
  if (!items || !Array.isArray(items)) {
    return [];
  }
  return [...items].sort((a, b) => {
    // 1. Highest score first
    const scoreA = a.score ?? -1;
    const scoreB = b.score ?? -1;
    if (scoreB !== scoreA) return scoreB - scoreA;

    // 2. Most recently scanned first
    const timeA = a.scannedAt ? new Date(a.scannedAt).getTime() : 0;
    const timeB = b.scannedAt ? new Date(b.scannedAt).getTime() : 0;
    return timeB - timeA;
  });
}
