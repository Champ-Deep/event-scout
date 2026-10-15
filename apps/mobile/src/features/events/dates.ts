/** Calendar-day helpers for planned events (dates are the event's own YYYY-MM-DD, no time zone). */

export const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
export const MONTHS_SHORT = MONTHS.map((m) => m.slice(0, 3));
export const WEEKDAYS_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const pad = (n: number) => String(n).padStart(2, '0');

/** Local Date → 'YYYY-MM-DD'. */
export const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;

/** 'YYYY-MM-DD' → local midnight Date (null if malformed). */
export function parseYmd(s?: string | null): Date | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s || '');
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : null;
}

export const todayYmd = () => ymd(new Date());

export function addDays(s: string, n: number): string {
  const d = parseYmd(s) || new Date();
  d.setDate(d.getDate() + n);
  return ymd(d);
}

const DAY_MS = 86400000;
/** Whole days from a to b (b − a). */
export function daysBetween(a: string, b: string): number {
  const x = parseYmd(a), y = parseYmd(b);
  return x && y ? Math.round((y.getTime() - x.getTime()) / DAY_MS) : 0;
}

/** "Tue, 3 Nov 2026" */
export function formatDay(s: string, withYear = true): string {
  const d = parseYmd(s);
  if (!d) return s;
  return `${WEEKDAYS_SHORT[d.getDay()]}, ${d.getDate()} ${MONTHS_SHORT[d.getMonth()]}${withYear ? ` ${d.getFullYear()}` : ''}`;
}

/** "3 – 5 Nov 2026" · "30 Oct – 2 Nov 2026" · "Tue, 3 Nov 2026" for one day. */
export function formatRange(start?: string | null, end?: string | null): string {
  const a = parseYmd(start), b = parseYmd(end);
  if (!a) return '';
  if (!b || ymd(a) === ymd(b)) return formatDay(ymd(a));
  if (a.getFullYear() !== b.getFullYear())
    return `${a.getDate()} ${MONTHS_SHORT[a.getMonth()]} ${a.getFullYear()} – ${b.getDate()} ${MONTHS_SHORT[b.getMonth()]} ${b.getFullYear()}`;
  if (a.getMonth() !== b.getMonth())
    return `${a.getDate()} ${MONTHS_SHORT[a.getMonth()]} – ${b.getDate()} ${MONTHS_SHORT[b.getMonth()]} ${b.getFullYear()}`;
  return `${a.getDate()} – ${b.getDate()} ${MONTHS_SHORT[a.getMonth()]} ${a.getFullYear()}`;
}

/** Whether a day falls within an event's dates (one-day event when there's no end date). */
export function coversDay(start: string | null | undefined, end: string | null | undefined, day: string): boolean {
  if (!start) return false;
  const last = end && end >= start ? end : start;
  return day >= start && day <= last;
}

/** "Happening now · day 2 of 3" · "Tomorrow" · "In 12 days" · "Ended" — relative to today. */
export function relativeToToday(start?: string | null, end?: string | null): { label: string; tone: 'live' | 'soon' | 'later' | 'past' } | null {
  if (!start) return null;
  const today = todayYmd();
  const last = end && end >= start ? end : start;
  if (today >= start && today <= last) {
    const total = daysBetween(start, last) + 1;
    return { label: total > 1 ? `Happening now · day ${daysBetween(start, today) + 1} of ${total}` : 'Today', tone: 'live' };
  }
  if (today > last) return { label: 'Ended', tone: 'past' };
  const n = daysBetween(today, start);
  return { label: n === 1 ? 'Tomorrow' : `In ${n} days`, tone: n <= 7 ? 'soon' : 'later' };
}

/** "9:00 AM" from a Date. */
export function formatTime(d: Date): string {
  const h = d.getHours(), m = d.getMinutes();
  return `${h % 12 || 12}:${pad(m)} ${h < 12 ? 'AM' : 'PM'}`;
}

/** "Today 9:00 AM" · "Tomorrow 6:00 PM" · "Tue, 3 Nov 9:00 AM" for a reminder time. */
export function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const day = ymd(d), today = todayYmd();
  const label = day === today ? 'Today' : day === addDays(today, 1) ? 'Tomorrow' : day === addDays(today, -1) ? 'Yesterday' : formatDay(day, d.getFullYear() !== new Date().getFullYear());
  return `${label} ${formatTime(d)}`;
}

/** 'HH:MM' (24h) → [h, m], or null. Also accepts '9', '9:5', '930'. */
export function parseTime(s: string): [number, number] | null {
  const t = s.trim().replace('.', ':');
  const m = /^(\d{1,2})(?::?(\d{2}))?$/.exec(t);
  if (!m) return null;
  const h = Number(m[1]), min = Number(m[2] || 0);
  return h <= 23 && min <= 59 ? [h, min] : null;
}

/** Local day + 'HH:MM' → Date. */
export function atTime(day: string, h: number, m: number): Date {
  const d = parseYmd(day) || new Date();
  d.setHours(h, m, 0, 0);
  return d;
}
