import { storage } from '../../lib/mmkv';
import type { EventNote, PlannedEvent } from '../../lib/api';
import { addDays, atTime, todayYmd } from './dates';

/*
 * Automatic reminders for events the admins scheduled: 6:00 PM (phone time) the evening before an
 * event starts, for events you're on the team for or have in your list (not ones you ended).
 *
 * They're worked out from the event's dates every time (never stored on the server), so an admin
 * moving the dates moves the reminder. Ticking one off / snoozing it is remembered on this phone.
 */

export const AUTO_PREFIX = 'auto:';
export const AUTO_HOUR = 18;
const STORE_KEY = 'auto_reminders_v1';

type State = Record<string, { done?: boolean; snoozeUntil?: string }>;
const read = (): State => { try { return JSON.parse(storage.getString(STORE_KEY) || '{}') || {}; } catch { return {}; } };
const write = (s: State) => { try { storage.set(STORE_KEY, JSON.stringify(s)); } catch { /* ignore */ } };
const k = (s?: string | null) => (s || '').split(/\s+/).join(' ').toLowerCase();

export const isAuto = (n: { id: string }) => n.id.startsWith(AUTO_PREFIX);

export function autoReminders(planned: PlannedEvent[], myEvents: string[], ended: string[]): EventNote[] {
  const mine = new Set(myEvents.map(k));
  const endedKeys = new Set(ended.map(k));
  const today = todayYmd();
  const st = read();
  return planned
    // not started yet (or starts today), and yours
    .filter((e) => !!e.startDate && e.startDate >= today && (!!e.myRole || mine.has(k(e.name))) && !endedKeys.has(k(e.name)))
    .map((e) => {
      const id = `${AUTO_PREFIX}${e.id}@${e.startDate}`;  // new dates → a new reminder
      const eve = addDays(e.startDate!, -1);
      const s = st[id] || {};
      const where = [e.venue, e.city].filter(Boolean).join(', ');
      return {
        id, event: e.name, date: eve,
        text: `Tomorrow: ${e.name}${where ? ` at ${where}` : ''}${e.details?.hours ? ` · ${e.details.hours}` : ''}${e.details?.booth ? ` · booth ${e.details.booth}` : ''}`,
        remindAt: s.snoozeUntil || atTime(eve, AUTO_HOUR, 0).toISOString(),
        done: !!s.done, createdAt: null, updatedAt: null,
      };
    });
}

export function setAutoDone(id: string, done: boolean) {
  const s = read();
  s[id] = { ...(s[id] || {}), done };
  write(s);
}

export function snoozeAuto(id: string, untilIso: string) {
  const s = read();
  s[id] = { ...(s[id] || {}), snoozeUntil: untilIso };
  write(s);
}
