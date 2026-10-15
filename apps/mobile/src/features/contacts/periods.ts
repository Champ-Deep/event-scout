/**
 * "Met today" / "This week" / "Scored" — the same rules the /dashboard tiles count with, so a
 * tile's number and the list it opens always agree. Today is the phone's local day (the app
 * sends Date.getTimezoneOffset() to /dashboard as tz_offset).
 */
export type QuickView = 'today' | 'week' | 'scored';

export const QUICK_VIEW_LABEL: Record<QuickView, string> = {
  today: 'Met today',
  week: 'Met this week',
  scored: 'Scored',
};

function created(c: { createdAt?: string | null }): Date | null {
  if (!c.createdAt) return null;
  // Server timestamps without a zone are UTC.
  const s = /[zZ]|[+-]\d\d:?\d\d$/.test(c.createdAt) ? c.createdAt : `${c.createdAt}Z`;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function metToday(c: { createdAt?: string | null }, now = new Date()): boolean {
  const d = created(c);
  return !!d && d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

/** Rolling last 7 days (same as the dashboard's added_this_week). */
export function metThisWeek(c: { createdAt?: string | null }, now = new Date()): boolean {
  const d = created(c);
  return !!d && now.getTime() - d.getTime() <= 7 * 24 * 3600 * 1000 && d.getTime() <= now.getTime() + 60_000;
}

export function inQuickView(view: QuickView, c: { createdAt?: string | null; score?: number | null }): boolean {
  if (view === 'today') return metToday(c);
  if (view === 'week') return metThisWeek(c);
  return c.score != null; // a real 0 is still a score
}

export function isQuickView(v: unknown): v is QuickView {
  return v === 'today' || v === 'week' || v === 'scored';
}
