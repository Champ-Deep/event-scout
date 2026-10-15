// Local calendar days (the admin's own time zone) as "YYYY-MM-DD" keys and the UTC instants that
// bound them — what the activity calendar asks the server for.

/** "2026-10-08" for a Date, in local time. */
export function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** Local midnight of a "YYYY-MM-DD" key. */
export function fromDayKey(key: string): Date {
  const [y, m, d] = key.split('-').map(Number);
  return new Date(y!, (m ?? 1) - 1, d ?? 1);
}

/** Same day, `n` days later (handles DST and month ends). */
export function addDays(d: Date, n: number): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
}

/** From local midnight of `key` to local midnight `days` later, as ISO instants for the API. */
export function localDayRange(key: string, days = 1): { start: string; end: string } {
  const start = fromDayKey(key);
  return { start: start.toISOString(), end: addDays(start, days).toISOString() };
}

/** The Monday on or before the 1st of `month`'s month: the first cell of a 6-week grid. */
export function gridStart(month: Date): Date {
  const first = new Date(month.getFullYear(), month.getMonth(), 1);
  return addDays(first, -((first.getDay() + 6) % 7));
}
