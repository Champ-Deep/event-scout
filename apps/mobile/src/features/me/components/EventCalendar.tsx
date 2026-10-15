import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { View, Pressable, TextInput, ActivityIndicator } from 'react-native';
import { useFocusEffect } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeInLeft, FadeInRight, FadeInUp, FadeOut, LinearTransition, ZoomIn, ZoomOut,
  useSharedValue, useAnimatedStyle, useReducedMotion, withDelay, withRepeat, withSequence, withSpring, withTiming, Easing,
} from 'react-native-reanimated';
import {
  CalendarDays, ChevronLeft, ChevronRight, Plus, Bell, BellRing, StickyNote, Check, Trash2, MapPin, AlertCircle,
} from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { Sheet } from '../../../ui/Sheet';
import { api, errorMessage, type ErrorMessageOptions, type EventNote, type PlannedEvent } from '../../../lib/api';
import { PressableScale, SectionHeader } from '../../onboarding/components';
import {
  MONTHS, WEEKDAYS_SHORT, ymd, parseYmd, todayYmd, addDays, coversDay, formatDay, formatRange, relativeToToday,
  formatWhen, atTime,
} from '../../events/dates';
import { autoReminders, isAuto, setAutoDone } from '../../events/autoReminders';

/* Me → Calendar: the events your admins scheduled for you (by their dates), and your own notes &
   reminders per event and day. Reminders are shown in the app (due / overdue), not pushed. */

type Toast = { kind: 'ok' | 'error'; text: string };
type RemindMode = 'none' | 'morning' | 'eve' | 'custom';
type ListFilter = 'all' | 'reminders' | 'due' | 'done' | 'day';
/** `time` is 12-hour "h:mm" (e.g. "2:30"), with `ampm`. */
interface Draft { event: string; date: string; text: string; remind: RemindMode; time: string; ampm: 'AM' | 'PM' }

/** "2:30" + PM → [14, 30]; also accepts "230", "2", "02.30". Null when it isn't a 12-hour time. */
function parse12(time: string, ampm: 'AM' | 'PM'): [number, number] | null {
  const m = /^(\d{1,2})(?:[:.]?(\d{2}))?$/.exec(time.trim());
  if (!m) return null;
  const h = Number(m[1]), min = Number(m[2] || 0);
  if (h < 1 || h > 12 || min > 59) return null;
  return [(h % 12) + (ampm === 'PM' ? 12 : 0), min];
}

const EVENT_FG = ['#7c3aed', '#0891b2', '#d97706', '#db2777', '#059669', '#2563eb'];
const EVENT_BG = ['#ede9fe', '#cffafe', '#fef3c7', '#fce7f3', '#d1fae5', '#dbeafe'];
const k = (s?: string | null) => (s || '').split(/\s+/).join(' ').toLowerCase();
const pad = (n: number) => String(n).padStart(2, '0');

// Motion (same springs as PressableScale). Worklets only touch numbers, never theme colours.
const SPRING_IN = { damping: 24, stiffness: 380, overshootClamping: true };
const SPRING_OUT = { damping: 14, stiffness: 260 };
const POP = ZoomIn.springify().damping(14).stiffness(260);

/** Today's day: a soft ring that breathes (static when the phone asks for reduced motion). */
function TodayPulse() {
  const reduce = useReducedMotion();
  const t = useSharedValue(0);
  useEffect(() => {
    if (!reduce) t.value = withRepeat(withTiming(1, { duration: 1600, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, [reduce]);
  const ring = useAnimatedStyle(() => ({ opacity: 0.55 - t.value * 0.35, transform: [{ scale: 1 + t.value * 0.14 }] }));
  return <Animated.View pointerEvents="none" style={[styles.todayRing, ring]} />;
}

/** A due reminder's bell: a short wiggle every couple of seconds. */
function RingingBell({ size, color }: { size: number; color: string }) {
  const reduce = useReducedMotion();
  const r = useSharedValue(0);
  useEffect(() => {
    if (reduce) return;
    const step = (to: number) => withTiming(to, { duration: 70 });
    r.value = withRepeat(withSequence(step(-1), step(1), step(-0.7), step(0.7), step(0), withDelay(1800, step(0))), -1);
  }, [reduce]);
  const style = useAnimatedStyle(() => ({ transform: [{ rotate: `${r.value * 16}deg` }] }));
  return <Animated.View style={style}><BellRing size={size} color={color} /></Animated.View>;
}

/** Springs down while pressed. */
function useSqueeze(to = 0.86) {
  const s = useSharedValue(1);
  const style = useAnimatedStyle(() => ({ transform: [{ scale: s.value }] }));
  return {
    style,
    onPressIn: () => { s.value = withSpring(to, SPRING_IN); },
    onPressOut: () => { s.value = withSpring(1, SPRING_OUT); },
  };
}

// How this screen words a failed request (see errorMessage).
const CAL_ERRORS: ErrorMessageOptions = {
  offline: "You're offline. Try again when connected.",
  unreachable: "Can't reach the server. Check your connection.",
};

/** When the draft's reminder goes off, or null (no reminder / bad custom time). */
function remindDate(d: Draft): Date | null {
  if (d.remind === 'morning') return atTime(d.date, 9, 0);
  if (d.remind === 'eve') return atTime(addDays(d.date, -1), 18, 0);
  if (d.remind === 'custom') {
    const t = parse12(d.time, d.ampm);
    return t ? atTime(d.date, t[0], t[1]) : null;
  }
  return null;
}

export function useEventCalendar(myEvents: string[], activeEvent: string | null, onToast: (t: Toast) => void,
  opts: { ended?: string[]; onOpenEvent?: (name: string) => void } = {}) {
  const [planned, setPlanned] = useState<PlannedEvent[]>([]);
  const [serverNotes, setNotes] = useState<EventNote[]>([]);
  // Automatic "evening before" reminders for your scheduled events (worked out here, not stored).
  const [autoTick, setAutoTick] = useState(0);
  const endedKey = (opts.ended || []).join('|');
  const autoNotes = useMemo(() => autoReminders(planned, myEvents, opts.ended || []), [planned, myEvents.join('|'), endedKey, autoTick]);
  const notes = useMemo(() => [...serverNotes, ...autoNotes], [serverNotes, autoNotes]);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [month, setMonth] = useState(() => { const d = new Date(); return { y: d.getFullYear(), m: d.getMonth() }; });
  const [selected, setSelected] = useState(todayYmd());
  // Which way the month grid slides in: 1 = next month (from the right), -1 = previous, 0 = no slide.
  const [dir, setDir] = useState(0);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [editing, setEditing] = useState<EventNote | null>(null);
  const [draft, setDraft] = useState<Draft>({ event: '', date: todayYmd(), text: '', remind: 'none', time: '9:00', ampm: 'AM' });
  const [saving, setSaving] = useState<'save' | 'delete' | null>(null);
  const [sheetError, setSheetError] = useState('');
  const [busyId, setBusyId] = useState<string | null>(null);
  // Notes & reminders list (opened from the count tiles / the day's pill). Editing a note from it
  // returns to it afterwards.
  const [listOpen, setListOpen] = useState(false);
  const [listFilter, setListFilter] = useState<ListFilter>('all');
  const fromList = useRef(false);
  const openList = (f: ListFilter) => { setListFilter(f); setListOpen(true); };
  const closeList = () => setListOpen(false);
  const backToList = () => {
    if (!fromList.current) return;
    fromList.current = false;
    setTimeout(() => setListOpen(true), 230); // after the editor has slid away
  };

  const reload = useCallback(async () => {
    // Every event the admins scheduled goes on the calendar (yours get the evening-before reminder).
    const [pe, ns] = await Promise.allSettled([api.listMyPlannedEvents('all'), api.listEventNotes()]);
    if (pe.status === 'fulfilled') setPlanned(Array.isArray(pe.value) ? pe.value : []);
    if (ns.status === 'fulfilled') setNotes(Array.isArray(ns.value) ? ns.value : []);
    setFailed(pe.status === 'rejected' && ns.status === 'rejected');
    setLoaded(true);
  }, []);
  useFocusEffect(useCallback(() => { reload(); }, [reload]));

  const dated = useMemo(() => planned.filter((e) => parseYmd(e.startDate)), [planned]);
  // Stable colour per event (by name), so an event looks the same on every day it spans.
  const colorIndex = useMemo(() => {
    const names = Array.from(new Set(dated.map((e) => k(e.name)))).sort();
    return (name: string) => Math.max(0, names.indexOf(k(name))) % EVENT_FG.length;
  }, [dated]);
  const colorOf = (name: string) => toneFg(EVENT_FG[colorIndex(name)] || '#7c3aed');
  const tintOf = (name: string) => toneBg(EVENT_BG[colorIndex(name)] || '#ede9fe');

  const eventsOn = useCallback((day: string) => dated.filter((e) => coversDay(e.startDate, e.endDate, day)), [dated]);
  const notesOn = useCallback((day: string) => notes.filter((n) => (n.date || (n.remindAt ? ymd(new Date(n.remindAt)) : '')) === day), [notes]);

  const now = Date.now();
  const isDue = (n: EventNote) => !n.done && !!n.remindAt && new Date(n.remindAt).getTime() <= now;
  const reminders = notes.filter((n) => n.remindAt && !n.done).sort((a, b) => (a.remindAt! < b.remindAt! ? -1 : 1));
  const dueCount = reminders.filter(isDue).length;
  const today = todayYmd();
  const upcoming = dated.filter((e) => ((e.endDate && e.endDate >= (e.startDate || '')) ? e.endDate : e.startDate)! >= today)
    .sort((a, b) => (a.startDate! < b.startDate! ? -1 : 1));

  // Events a note can be for: the selected day's events first, then the active one, then the rest.
  const eventChoices = useMemo(() => {
    const out: string[] = [];
    for (const n of [...eventsOn(draft.date).map((e) => e.name), activeEvent || '', ...myEvents, ...planned.map((e) => e.name), draft.event]) {
      if (n && !out.some((o) => k(o) === k(n))) out.push(n);
    }
    return out;
  }, [eventsOn, draft.date, draft.event, activeEvent, myEvents, planned]);

  const select = (day: string) => {
    setSelected(day);
    const d = parseYmd(day);
    if (d && (d.getFullYear() !== month.y || d.getMonth() !== month.m)) {
      setDir(d.getFullYear() * 12 + d.getMonth() > month.y * 12 + month.m ? 1 : -1);
      setMonth({ y: d.getFullYear(), m: d.getMonth() });
    }
  };
  const shiftMonth = (n: number) => {
    setDir(n > 0 ? 1 : -1);
    setMonth(({ y, m }) => { const d = new Date(y, m + n, 1); return { y: d.getFullYear(), m: d.getMonth() }; });
  };
  const goToday = () => select(todayYmd());

  const openNew = (day: string, viaList = false) => {
    fromList.current = viaList;
    if (viaList) setListOpen(false);
    const first = eventsOn(day)[0]?.name || activeEvent || myEvents[0] || planned[0]?.name || '';
    setEditing(null);
    setDraft({ event: first, date: day, text: '', remind: 'none', time: '9:00', ampm: 'AM' });
    setSheetError('');
    setSheetOpen(true);
  };
  const openEdit = (n: EventNote, viaList = false) => {
    if (isAuto(n)) { setListOpen(false); opts.onOpenEvent?.(n.event); return; } // automatic: open the event
    fromList.current = viaList;
    if (viaList) setListOpen(false);
    const date = n.date || (n.remindAt ? ymd(new Date(n.remindAt)) : todayYmd());
    let remind: RemindMode = 'none', time = '9:00', ampm: 'AM' | 'PM' = 'AM';
    if (n.remindAt) {
      const r = new Date(n.remindAt);
      const hm = `${pad(r.getHours())}:${pad(r.getMinutes())}`;
      if (ymd(r) === date && hm === '09:00') remind = 'morning';
      else if (ymd(r) === addDays(date, -1) && hm === '18:00') remind = 'eve';
      else { remind = 'custom'; time = `${r.getHours() % 12 || 12}:${pad(r.getMinutes())}`; ampm = r.getHours() < 12 ? 'AM' : 'PM'; }
    }
    setEditing(n);
    setDraft({ event: n.event, date, text: n.text, remind, time, ampm });
    setSheetError('');
    setSheetOpen(true);
  };
  const closeSheet = () => { if (!saving) { setSheetOpen(false); backToList(); } };

  const save = async () => {
    const text = draft.text.trim();
    if (!text) { setSheetError('Write the note first.'); return; }
    if (!draft.event) { setSheetError('Pick the event this note is for.'); return; }
    let remindAt = '';
    if (draft.remind !== 'none') {
      const when = remindDate(draft);
      if (!when) { setSheetError('Enter the time like 2:30 (hour 1–12), then pick AM or PM.'); return; }
      remindAt = when.toISOString();
      const unchanged = editing?.remindAt && new Date(editing.remindAt).getTime() === when.getTime();
      if (!unchanged && when.getTime() < Date.now()) { setSheetError('That reminder time has already passed — pick a later one.'); return; }
    }
    setSaving('save');
    setSheetError('');
    try {
      const body = { event: draft.event, date: draft.date, text, remindAt };
      const saved = editing ? await api.updateEventNote(editing.id, body) : await api.addEventNote(body);
      setNotes((prev) => (editing ? prev.map((n) => (n.id === saved.id ? saved : n)) : [...prev, saved]));
      setSheetOpen(false);
      backToList();
      onToast({ kind: 'ok', text: remindAt ? `Reminder set · ${formatWhen(remindAt)}` : 'Note saved' });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save the note.", CAL_ERRORS));
    } finally {
      setSaving(null);
    }
  };

  const remove = async () => {
    if (!editing || saving) return;
    setSaving('delete');
    try {
      await api.deleteEventNote(editing.id);
      setNotes((prev) => prev.filter((n) => n.id !== editing.id));
      setSheetOpen(false);
      backToList();
      onToast({ kind: 'ok', text: 'Note deleted' });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't delete the note.", CAL_ERRORS));
    } finally {
      setSaving(null);
    }
  };

  const toggleDone = async (n: EventNote) => {
    if (isAuto(n)) { setAutoDone(n.id, !n.done); setAutoTick((t) => t + 1); return; }
    if (busyId) return;
    setBusyId(n.id);
    setNotes((prev) => prev.map((x) => (x.id === n.id ? { ...x, done: !n.done } : x))); // optimistic
    try {
      const saved = await api.updateEventNote(n.id, { done: !n.done });
      setNotes((prev) => prev.map((x) => (x.id === n.id ? saved : x)));
    } catch (e: any) {
      setNotes((prev) => prev.map((x) => (x.id === n.id ? n : x)));
      onToast({ kind: 'error', text: errorMessage(e, "Couldn't update the note.", CAL_ERRORS) });
    } finally {
      setBusyId(null);
    }
  };

  return {
    listOpen, listFilter, setListFilter, openList, closeList,
    planned, notes, loaded, failed, month, dir, selected, sheetOpen, editing, draft, setDraft, saving, sheetError, busyId,
    eventsOn, notesOn, colorOf, tintOf, isDue, reminders, dueCount, upcoming, eventChoices,
    reload, select, shiftMonth, goToday, openNew, openEdit, closeSheet, save, remove, toggleDone,
  };
}

export type EventCalendarState = ReturnType<typeof useEventCalendar>;

/* --------------------------------- card --------------------------------- */

function NoteRow({ cal, n, showDay, index = 0, viaList }: { cal: EventCalendarState; n: EventNote; showDay?: boolean; index?: number; viaList?: boolean }) {
  const due = cal.isDue(n);
  const row = useSqueeze(0.97);
  // Checkbox: squashes then springs back past full size when ticked.
  const pop = useSharedValue(1);
  const popStyle = useAnimatedStyle(() => ({ transform: [{ scale: pop.value }] }));
  const tick = () => {
    pop.value = withSequence(withTiming(0.7, { duration: 90 }), withSpring(1, { damping: 7, stiffness: 320 }));
    cal.toggleDone(n);
  };
  return (
    <Animated.View entering={FadeInDown.delay(index * 50).duration(240)} exiting={FadeOut.duration(140)} layout={LinearTransition.springify().damping(18)}>
      <Animated.View style={row.style}>
      <Pressable onPress={() => cal.openEdit(n, viaList)} onPressIn={row.onPressIn} onPressOut={row.onPressOut}
        style={[styles.noteRow, due && styles.noteRowDue]}
        accessibilityLabel={`Note: ${n.text}. Tap to edit`}>
        <Pressable onPress={tick} hitSlop={10} disabled={cal.busyId === n.id}
          accessibilityRole="checkbox" accessibilityState={{ checked: n.done }} accessibilityLabel={n.done ? 'Mark not done' : 'Mark done'}>
          <Animated.View style={[styles.check, n.done && styles.checkOn, popStyle]}>
            {n.done ? <Animated.View entering={POP} exiting={ZoomOut.duration(120)}><Check size={12} color="white" /></Animated.View> : null}
          </Animated.View>
        </Pressable>
        <View style={{ flex: 1 }}>
          <Animated.View key={n.done ? 'done' : 'open'} entering={FadeIn.duration(220)}>
            <Text style={[styles.noteText, n.done && styles.noteDone]} numberOfLines={3}>{n.text}</Text>
          </Animated.View>
          <View style={styles.noteMeta}>
            <View style={[styles.evDot, { backgroundColor: cal.colorOf(n.event) }]} />
            <Text style={styles.noteMetaText} numberOfLines={1}>
              {n.event}{isAuto(n) ? ' · automatic' : ''}{showDay && n.date ? ` · ${formatDay(n.date, false)}` : ''}
            </Text>
          </View>
          {n.remindAt ? (
            <Animated.View entering={FadeIn.delay(120).duration(220)} layout={LinearTransition.duration(200)}
              style={[styles.remindChip, due && styles.remindChipDue, n.done && { opacity: 0.6 }]}>
              {due ? <RingingBell size={11} color={theme.colors.semantic.danger} /> : <Bell size={11} color={theme.colors.brand[700]} />}
              <Text style={[styles.remindChipText, due && { color: theme.colors.semantic.danger }]}>
                {due ? 'Due · ' : ''}{formatWhen(n.remindAt)}
              </Text>
            </Animated.View>
          ) : null}
        </View>
      </Pressable>
      </Animated.View>
    </Animated.View>
  );
}

/** One day in the month grid: squeezes on press, the selection pops in, today breathes. */
function DayCell({ cal, day, n, inMonth, isToday, index }: {
  cal: EventCalendarState; day: string; n: number; inMonth: boolean; isToday: boolean; index: number;
}) {
  const press = useSqueeze(0.82);
  const evs = cal.eventsOn(day);
  const ns = cal.notesOn(day);
  const due = ns.some(cal.isDue);
  const reminder = ns.some((x) => !!x.remindAt && !x.done);   // a reminder still to go off (or due)
  const plain = ns.some((x) => !x.remindAt || x.done);        // a plain note (or a ticked-off reminder)
  const sel = day === cal.selected;
  const ev = evs[0];
  // Event days: a solid circle on the first / last day, a band in the event's colour across the days between.
  const last = ev ? (ev.endDate && ev.endDate >= (ev.startDate || '') ? ev.endDate : ev.startDate) : null;
  const isStart = !!ev && day === ev.startDate;
  const isEnd = !!ev && day === last;
  const edge = isStart || isEnd;
  const multi = !!ev && ev.startDate !== last;
  return (
    <Pressable onPress={() => cal.select(day)} onPressIn={press.onPressIn} onPressOut={press.onPressOut} style={styles.cell}
      accessibilityRole="button" accessibilityState={{ selected: sel }}
      accessibilityLabel={`${formatDay(day)}${evs.length ? `, ${evs.map((e) => e.name).join(', ')}` : ''}${ns.length ? `, ${ns.length} note${ns.length === 1 ? '' : 's'}` : ''}`}>
      {ev && multi ? (
        <Animated.View entering={FadeIn.delay(80 + index * 5).duration(240)} pointerEvents="none"
          style={[styles.band, { backgroundColor: cal.tintOf(ev.name) }, isStart && styles.bandStart, isEnd && styles.bandEnd]} />
      ) : null}
      <Animated.View style={[styles.bubble, ev && edge && { backgroundColor: cal.colorOf(ev.name) }, press.style]}>
        {isToday && !sel ? <TodayPulse /> : null}
        {isToday && !sel ? <View pointerEvents="none" style={styles.todayRingStatic} /> : null}
        {sel ? <Animated.View pointerEvents="none" entering={POP} exiting={ZoomOut.duration(140)} style={styles.selFill} /> : null}
        <Text style={[styles.dayNum, !inMonth && styles.dayOut,
          ev && { color: edge ? theme.colors.bg.surface : cal.colorOf(ev.name), fontWeight: '800' },
          sel && { color: 'white', fontWeight: '800' }]}>{n}</Text>
      </Animated.View>
      {/* What's on the day, by kind: event · note · reminder (red when due) */}
      <View style={styles.marks}>
        {ev ? (
          <Animated.View entering={ZoomIn.delay(140 + index * 6).duration(220)}>
            <CalendarDays size={9} strokeWidth={2.6} color={cal.colorOf(ev.name)} />
          </Animated.View>
        ) : null}
        {plain ? (
          <Animated.View entering={POP} exiting={ZoomOut.duration(120)}>
            <StickyNote size={9} strokeWidth={2.6} color={theme.colors.semantic.warning} />
          </Animated.View>
        ) : null}
        {reminder ? (
          <Animated.View entering={POP} exiting={ZoomOut.duration(120)}>
            {due ? <RingingBell size={9} color={theme.colors.semantic.danger} />
              : <Bell size={9} strokeWidth={2.6} color={theme.colors.semantic.info} />}
          </Animated.View>
        ) : null}
      </View>
    </Pressable>
  );
}

export function EventCalendarCard({ cal, entering, onOpenEvent }: {
  cal: EventCalendarState; entering?: any; onOpenEvent: (name: string) => void;
}) {
  const { y, m } = cal.month;
  const today = todayYmd();
  const cells = useMemo(() => {
    const startPad = new Date(y, m, 1).getDay();
    const days = new Date(y, m + 1, 0).getDate();
    return Array.from({ length: Math.ceil((startPad + days) / 7) * 7 }, (_, i) => {
      const d = new Date(y, m, 1 - startPad + i);
      return { day: ymd(d), n: d.getDate(), inMonth: d.getMonth() === m };
    });
  }, [y, m]);
  const viewingNow = y === new Date().getFullYear() && m === new Date().getMonth() && cal.selected === today;
  const dayEvents = cal.eventsOn(cal.selected);
  const dayNotes = cal.notesOn(cal.selected);
  const hasAnyEvent = cal.eventChoices.length > 0;

  return (
    <Animated.View entering={entering} layout={LinearTransition.duration(260)} style={styles.card}>
      <SectionHeader
        icon={<CalendarDays size={15} color={theme.colors.brand[600]} />}
        label="CALENDAR"
        hint={cal.dueCount ? `${cal.dueCount} reminder${cal.dueCount === 1 ? '' : 's'} due` : 'Upcoming events · tap a day for notes & reminders'}
        right={!viewingNow ? (
          <Animated.View entering={POP} exiting={ZoomOut.duration(140)}>
            <PressableScale onPress={cal.goToday} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Go to today">
              <Text style={styles.smallBtnText}>Today</Text>
            </PressableScale>
          </Animated.View>
        ) : null}
      />

      {/* Month: arrows nudge outward on press; the title and grid slide the way you're going */}
      <View style={styles.monthRow}>
        <PressableScale onPress={() => cal.shiftMonth(-1)} hitSlop={10} scaleTo={0.85} style={styles.navBtn} accessibilityLabel="Previous month"
          leadingIcon={<ChevronLeft size={18} color={theme.colors.text.body} />} />
        <View style={styles.monthTitle}>
          <Animated.View key={`${y}-${m}`} {...(cal.dir ? { entering: (cal.dir > 0 ? FadeInRight : FadeInLeft).duration(240) } : {})}>
            <Text style={styles.monthText}>{MONTHS[m]} {y}</Text>
          </Animated.View>
        </View>
        <PressableScale onPress={() => cal.shiftMonth(1)} hitSlop={10} scaleTo={0.85} style={styles.navBtn} accessibilityLabel="Next month"
          trailingIcon={<ChevronRight size={18} color={theme.colors.text.body} />} />
      </View>
      <View style={styles.grid}>
        {WEEKDAYS_SHORT.map((w) => <Text key={w} style={styles.weekday}>{w.slice(0, 2)}</Text>)}
      </View>
      <View style={styles.gridClip}>
        <Animated.View key={`${y}-${m}`} style={styles.grid}
          {...(cal.dir ? { entering: (cal.dir > 0 ? FadeInRight : FadeInLeft).duration(260).easing(Easing.out(Easing.cubic)) } : {})}>
          {cells.map((c, i) => (
            <DayCell key={c.day} cal={cal} day={c.day} n={c.n} inMonth={c.inMonth} isToday={c.day === today} index={i} />
          ))}
        </Animated.View>
      </View>
      <View style={styles.legend} accessibilityLabel="Legend: event, note, reminder, due">
        <View style={styles.legendItem}><CalendarDays size={11} strokeWidth={2.4} color={theme.colors.brand[600]} /><Text style={styles.legendText}>Event</Text></View>
        <View style={styles.legendItem}><StickyNote size={11} strokeWidth={2.4} color={theme.colors.semantic.warning} /><Text style={styles.legendText}>Note</Text></View>
        <View style={styles.legendItem}><Bell size={11} strokeWidth={2.4} color={theme.colors.semantic.info} /><Text style={styles.legendText}>Reminder</Text></View>
        <View style={styles.legendItem}><Bell size={11} strokeWidth={2.4} color={theme.colors.semantic.danger} /><Text style={styles.legendText}>Due</Text></View>
      </View>

      {/* Selected day: its content fades in fresh whenever the day changes */}
      <Animated.View layout={LinearTransition.springify().damping(20)} style={styles.dayPanel}>
        <View style={styles.dayHead}>
          <View style={{ flex: 1 }}>
            <Animated.View key={cal.selected} entering={FadeInUp.duration(220)}>
              <Text style={styles.dayTitle}>{formatDay(cal.selected)}{cal.selected === today ? ' · Today' : ''}</Text>
            </Animated.View>
          </View>
          {hasAnyEvent ? (
            <PressableScale onPress={() => cal.openNew(cal.selected)} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Add a note or reminder"
              leadingIcon={<Plus size={13} color={theme.colors.brand[700]} />}>
              <Text style={styles.smallBtnText}>Note</Text>
            </PressableScale>
          ) : null}
        </View>
        <View key={cal.selected}>
          {dayEvents.map((e, i) => {
            const pos = e.endDate && e.endDate > (e.startDate || '') ? ` · day ${Math.round(((parseYmd(cal.selected)?.getTime() || 0) - (parseYmd(e.startDate)?.getTime() || 0)) / 86400000) + 1}` : '';
            const where = [e.venue, e.city].filter(Boolean).join(', ');
            return (
              <Animated.View key={e.id} entering={FadeInDown.delay(i * 50).duration(240)} layout={LinearTransition.duration(240)}>
                <PressableScale onPress={() => onOpenEvent(e.name)} scaleTo={0.97} style={[styles.evRow, { borderLeftColor: cal.colorOf(e.name) }]}
                  accessibilityLabel={`Open ${e.name}`} trailingIcon={<ChevronRight size={16} color={theme.colors.text.muted} />}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.evName} numberOfLines={1}>{e.name}</Text>
                    <Text style={styles.evMeta} numberOfLines={1}>{formatRange(e.startDate, e.endDate)}{pos}</Text>
                    {where ? (
                      <View style={styles.whereRow}><MapPin size={11} color={theme.colors.text.muted} /><Text style={styles.evMeta} numberOfLines={1}>{where}</Text></View>
                    ) : null}
                  </View>
                </PressableScale>
              </Animated.View>
            );
          })}
          {dayNotes.length ? (() => {
            const rem = dayNotes.filter((n) => n.remindAt && !n.done).length;
            const due = dayNotes.some(cal.isDue);
            return (
              <Animated.View entering={FadeInDown.delay(dayEvents.length * 50).duration(240)}>
                <PressableScale onPress={() => cal.openList('day')} scaleTo={0.97} style={[styles.dayPill, due && styles.dayPillDue]}
                  accessibilityLabel={`${dayNotes.length} notes on this day. Show them`}
                  trailingIcon={<ChevronRight size={15} color={due ? theme.colors.semantic.danger : theme.colors.brand[600]} />}>
                  {due ? <RingingBell size={14} color={theme.colors.semantic.danger} /> : <StickyNote size={14} color={theme.colors.brand[600]} />}
                  <Text style={[styles.dayPillText, due && { color: theme.colors.semantic.danger }]}>
                    {dayNotes.length} note{dayNotes.length === 1 ? '' : 's'}{rem ? ` · ${rem} reminder${rem === 1 ? '' : 's'}` : ''}{due ? ' · due' : ''}
                  </Text>
                </PressableScale>
              </Animated.View>
            );
          })() : null}
          {!dayEvents.length && !dayNotes.length ? (
            <Animated.View entering={FadeIn.duration(220)}>
              <Text style={styles.muted}>
                {!cal.loaded ? 'Loading…' : cal.failed ? "Couldn't load your calendar. Pull down to retry."
                  : hasAnyEvent ? 'Nothing on this day. Tap Note to add a note or reminder.' : 'Add an event under Your events to start writing notes for it.'}
              </Text>
            </Animated.View>
          ) : null}
        </View>
      </Animated.View>

      {/* Notes & reminders: just the counts — tap a tile for the full list */}
      {cal.notes.length ? (
        <Animated.View entering={FadeInDown.duration(240)} layout={LinearTransition.springify().damping(20)} style={styles.tiles}>
          <CountTile index={0} icon={<StickyNote size={15} color={theme.colors.brand[600]} />} tint={theme.colors.brand[50]}
            value={cal.notes.length} label="Notes" onPress={() => cal.openList('all')} />
          <CountTile index={1} icon={<Bell size={15} color={theme.colors.semantic.warning} />} tint={theme.colors.semantic.warningBg}
            value={cal.reminders.length} label="Reminders" onPress={() => cal.openList('reminders')} />
          <CountTile index={2} tint={cal.dueCount ? theme.colors.semantic.dangerBg : toneBg('#f1f5f9')} alert={cal.dueCount > 0}
            icon={cal.dueCount ? <RingingBell size={15} color={theme.colors.semantic.danger} /> : <BellRing size={15} color={theme.colors.text.muted} />}
            value={cal.dueCount} label="Due" onPress={() => cal.openList('due')} />
        </Animated.View>
      ) : null}

      {/* Upcoming scheduled events */}
      {cal.upcoming.length ? (
        <Animated.View layout={LinearTransition.springify().damping(20)}>
          <Text style={styles.subLabel}>COMING UP</Text>
          {cal.upcoming.slice(0, 3).map((e, i) => {
            const rel = relativeToToday(e.startDate, e.endDate);
            return (
              <Animated.View key={e.id} entering={FadeInDown.delay(80 + i * 60).duration(260)}>
                <PressableScale onPress={() => cal.select(e.startDate!)} scaleTo={0.97} style={styles.upRow}
                  accessibilityLabel={`${e.name}, ${formatRange(e.startDate, e.endDate)}. Show on the calendar`}>
                  <Animated.View entering={ZoomIn.delay(160 + i * 60).springify().damping(14)} style={[styles.upDate, { backgroundColor: cal.tintOf(e.name) }]}>
                    <Text style={[styles.upDay, { color: cal.colorOf(e.name) }]}>{parseYmd(e.startDate)?.getDate()}</Text>
                    <Text style={[styles.upMon, { color: cal.colorOf(e.name) }]}>{(MONTHS[parseYmd(e.startDate)?.getMonth() ?? 0] || '').slice(0, 3).toUpperCase()}</Text>
                  </Animated.View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.evName} numberOfLines={1}>{e.name}</Text>
                    <Text style={styles.evMeta} numberOfLines={1}>{[formatRange(e.startDate, e.endDate), e.city].filter(Boolean).join(' · ')}</Text>
                  </View>
                  {rel ? <Text style={[styles.relText, rel.tone === 'live' && { color: theme.colors.semantic.successText }, rel.tone === 'soon' && { color: theme.colors.semantic.warning }]}>{rel.label}</Text> : null}
                </PressableScale>
              </Animated.View>
            );
          })}
        </Animated.View>
      ) : cal.loaded && !cal.failed && !cal.planned.length ? (
        <Animated.View entering={FadeIn.duration(260)}>
          <Text style={[styles.muted, { marginTop: 10 }]}>Events your admins schedule for you appear here with their dates.</Text>
        </Animated.View>
      ) : null}
    </Animated.View>
  );
}

/** A count tile (Notes / Reminders / Due): springs on press, the number pops when it changes. */
function CountTile({ icon, tint, value, label, onPress, index, alert }: {
  icon: React.ReactNode; tint: string; value: number; label: string; onPress: () => void; index: number; alert?: boolean;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(60 + index * 60).duration(260)} style={{ flex: 1 }}>
      <PressableScale onPress={onPress} scaleTo={0.94} containerStyle={{ flex: 1 }} style={[styles.tile, alert && styles.tileAlert]}
        accessibilityLabel={`${label}: ${value}. Show them`}>
        <View style={[styles.tileIcon, { backgroundColor: tint }]}>{icon}</View>
        <Animated.View key={value} entering={ZoomIn.springify().damping(14)}>
          <Text style={[styles.tileValue, alert && { color: theme.colors.semantic.danger }]}>{value}</Text>
        </Animated.View>
        <Text style={styles.tileLabel}>{label}</Text>
      </PressableScale>
    </Animated.View>
  );
}

const LIST_FILTERS: { key: ListFilter; label: string }[] = [
  { key: 'all', label: 'All' }, { key: 'reminders', label: 'Reminders' }, { key: 'due', label: 'Due' },
  { key: 'done', label: 'Done' }, { key: 'day', label: 'This day' },
];

/** Every note & reminder, filtered, grouped by day. Render it last in the screen (it's an overlay). */
export function EventNotesListSheet({ cal }: { cal: EventCalendarState }) {
  const match = (n: EventNote, f: ListFilter) =>
    f === 'all' ? true : f === 'reminders' ? !!n.remindAt && !n.done : f === 'due' ? cal.isDue(n) : f === 'done' ? n.done
      : (n.date || (n.remindAt ? ymd(new Date(n.remindAt)) : '')) === cal.selected;
  const list = cal.notes.filter((n) => match(n, cal.listFilter)).sort((a, b) => {
    const da = a.date || '9999', db = b.date || '9999';
    if (cal.listFilter === 'reminders' || cal.listFilter === 'due') return (a.remindAt || '') < (b.remindAt || '') ? -1 : 1;
    return da === db ? ((a.remindAt || a.createdAt || '') < (b.remindAt || b.createdAt || '') ? -1 : 1) : da < db ? -1 : 1;
  });
  // group by day (reminder lists stay one flat list, soonest first)
  const groups: { day: string; items: EventNote[] }[] = [];
  for (const n of list) {
    const day = cal.listFilter === 'reminders' || cal.listFilter === 'due' ? '' : n.date || '';
    const g = groups[groups.length - 1];
    if (g && g.day === day) g.items.push(n); else groups.push({ day, items: [n] });
  }
  const today = todayYmd();

  const header = (
    <View style={styles.listFilters}>
      {LIST_FILTERS.map((f) => {
        const on = cal.listFilter === f.key;
        const count = cal.notes.filter((n) => match(n, f.key)).length;
        return (
          <Animated.View key={f.key} layout={LinearTransition.springify().damping(18)}>
            <PressableScale onPress={() => cal.setListFilter(f.key)} scaleTo={0.92} style={[styles.listFilter, on && styles.listFilterOn]}
              accessibilityRole="tab" accessibilityState={{ selected: on }}>
              <Text style={[styles.listFilterText, on && { color: 'white' }]}>{f.key === 'day' ? formatDay(cal.selected, false) : f.label}</Text>
              <Text style={[styles.listFilterCount, on && { color: 'white' }]}>{count}</Text>
            </PressableScale>
          </Animated.View>
        );
      })}
    </View>
  );

  return (
    <Sheet visible={cal.listOpen} title="Notes & reminders" icon={<StickyNote size={15} color="white" />} onClose={cal.closeList} header={header} tall deferBody>
      {cal.eventChoices.length ? (
        <PressableScale onPress={() => cal.openNew(cal.listFilter === 'day' ? cal.selected : todayYmd(), true)} scaleTo={0.97}
          style={styles.newNoteBtn} leadingIcon={<Plus size={15} color={theme.colors.brand[700]} />}>
          <Text style={styles.newNoteText}>New note{cal.listFilter === 'day' ? ` · ${formatDay(cal.selected, false)}` : ''}</Text>
        </PressableScale>
      ) : null}
      {groups.map((g, gi) => (
        <Animated.View key={`${cal.listFilter}-${g.day || 'flat'}-${gi}`} layout={LinearTransition.springify().damping(20)}>
          {g.day ? (
            <Text style={styles.subLabel}>{formatDay(g.day).toUpperCase()}{g.day === today ? ' · TODAY' : ''}</Text>
          ) : gi === 0 ? <View style={{ height: 8 }} /> : null}
          {g.items.map((n, i) => <NoteRow key={n.id} cal={cal} n={n} showDay={!g.day} index={Math.min(gi + i, 8)} viaList />)}
        </Animated.View>
      ))}
      {!list.length ? (
        <Animated.View key={cal.listFilter} entering={FadeIn.duration(220)} style={styles.listEmpty}>
          <StickyNote size={22} color={theme.colors.text.muted} />
          <Text style={[styles.muted, { textAlign: 'center' }]}>
            {cal.listFilter === 'due' ? 'Nothing due. You’re all caught up.' : cal.listFilter === 'reminders' ? 'No reminders waiting.'
              : cal.listFilter === 'done' ? 'Nothing ticked off yet.' : cal.listFilter === 'day' ? 'No notes on this day.' : 'No notes yet.'}
          </Text>
        </Animated.View>
      ) : null}
      <View style={{ height: 12 }} />
    </Sheet>
  );
}

/* --------------------------------- sheet --------------------------------- */

const REMIND_OPTIONS: { value: RemindMode; label: string }[] = [
  { value: 'none', label: 'No reminder' },
  { value: 'morning', label: '9:00 AM that day' },
  { value: 'eve', label: '6:00 PM day before' },
  { value: 'custom', label: 'Pick a time' },
];

/** Write / edit a note or reminder. Render it last in the screen (it's an overlay). */
/** A choice chip that springs on press and pops a tick in when chosen. */
function Chip({ on, label, dot, onPress }: { on: boolean; label: string; dot?: string; onPress: () => void }) {
  return (
    <Animated.View layout={LinearTransition.springify().damping(18)}>
      <PressableScale onPress={onPress} scaleTo={0.92} style={[styles.chip, on && styles.chipOn]}
        accessibilityRole="radio" accessibilityState={{ selected: on }} accessibilityLabel={label}>
        {on ? (
          <Animated.View entering={POP} exiting={ZoomOut.duration(120)}><Check size={13} color={theme.colors.brand[700]} /></Animated.View>
        ) : dot ? <View style={[styles.evDot, { backgroundColor: dot }]} /> : null}
        <Text style={[styles.chipText, on && styles.chipTextOn]} numberOfLines={1}>{label}</Text>
      </PressableScale>
    </Animated.View>
  );
}

export function EventNoteSheet({ cal }: { cal: EventCalendarState }) {
  const { draft, setDraft } = cal;
  const set = (patch: Partial<Draft>) => setDraft((d) => ({ ...d, ...patch }));
  const when = remindDate(draft);
  // Day stepper slides the way you step; the error message shakes in.
  const [dayDir, setDayDir] = useState(0);
  const step = (n: number) => { setDayDir(n); set({ date: addDays(draft.date, n) }); };
  const shake = useSharedValue(0);
  const shakeStyle = useAnimatedStyle(() => ({ transform: [{ translateX: shake.value }] }));
  const reduce = useReducedMotion();
  const lastError = useRef('');
  useEffect(() => {
    if (cal.sheetError && cal.sheetError !== lastError.current && !reduce) {
      const t = (x: number) => withTiming(x, { duration: 55 });
      shake.value = withSequence(t(-8), t(8), t(-6), t(6), t(-3), t(0));
    }
    lastError.current = cal.sheetError;
  }, [cal.sheetError]);
  useEffect(() => { if (cal.sheetOpen) setDayDir(0); }, [cal.sheetOpen]);
  return (
    <Sheet visible={cal.sheetOpen} title={cal.editing ? 'Edit note' : 'New note'} icon={<StickyNote size={15} color="white" />} onClose={cal.closeSheet}>
      <Text style={[styles.subLabel, { marginTop: 0 }]}>EVENT</Text>
      {cal.eventChoices.length ? (
        <View style={styles.chips}>
          {cal.eventChoices.map((name, i) => (
            <Animated.View key={name} entering={FadeInDown.delay(i * 40).duration(220)}>
              <Chip on={k(name) === k(draft.event)} label={name} dot={cal.colorOf(name)} onPress={() => set({ event: name })} />
            </Animated.View>
          ))}
        </View>
      ) : <Text style={styles.muted}>Add an event under Your events first.</Text>}

      <Text style={styles.subLabel}>DAY</Text>
      <View style={styles.dayPick}>
        <PressableScale onPress={() => step(-1)} hitSlop={10} scaleTo={0.85} style={styles.navBtn} accessibilityLabel="Previous day"
          leadingIcon={<ChevronLeft size={18} color={theme.colors.text.body} />} />
        <View style={styles.monthTitle}>
          <Animated.View key={draft.date} {...(dayDir ? { entering: (dayDir > 0 ? FadeInRight : FadeInLeft).duration(200) } : {})}>
            <Text style={styles.dayPickText}>{formatDay(draft.date)}</Text>
          </Animated.View>
        </View>
        <PressableScale onPress={() => step(1)} hitSlop={10} scaleTo={0.85} style={styles.navBtn} accessibilityLabel="Next day"
          trailingIcon={<ChevronRight size={18} color={theme.colors.text.body} />} />
      </View>

      <Text style={styles.subLabel}>NOTE</Text>
      <TextInput
        value={draft.text} onChangeText={(text) => set({ text })} multiline maxLength={2000}
        placeholder="e.g. Print badges · meet Priya at booth B12 · bring 200 cards"
        placeholderTextColor={theme.colors.text.muted} style={styles.textArea} textAlignVertical="top"
      />

      <Text style={styles.subLabel}>REMINDER</Text>
      <View style={styles.chips}>
        {REMIND_OPTIONS.map((o) => (
          <Chip key={o.value} on={draft.remind === o.value} label={o.label} onPress={() => set({ remind: o.value })} />
        ))}
      </View>
      {draft.remind === 'custom' ? (
        <Animated.View entering={FadeInDown.duration(220)} exiting={FadeOut.duration(140)} style={styles.timeRow}>
          <TextInput value={draft.time} onChangeText={(time) => set({ time })} placeholder="2:30" maxLength={5}
            keyboardType="numbers-and-punctuation" placeholderTextColor={theme.colors.text.muted} style={styles.timeInput}
            accessibilityLabel="Reminder time, hour and minutes" />
          <View style={styles.ampm}>
            {(['AM', 'PM'] as const).map((v) => {
              const on = draft.ampm === v;
              return (
                <PressableScale key={v} onPress={() => set({ ampm: v })} scaleTo={0.9} style={[styles.ampmBtn, on && styles.ampmBtnOn]}
                  accessibilityRole="radio" accessibilityState={{ selected: on }}>
                  <Text style={[styles.ampmText, on && styles.ampmTextOn]}>{v}</Text>
                </PressableScale>
              );
            })}
          </View>
          <Text style={[styles.muted, { flex: 1 }]} numberOfLines={2}>on {formatDay(draft.date, false)}</Text>
        </Animated.View>
      ) : null}
      {draft.remind !== 'none' && when ? (
        <Animated.View key={when.toISOString()} entering={FadeIn.duration(220)} exiting={FadeOut.duration(120)} style={styles.whenRow}>
          <Animated.View entering={POP}><Bell size={12} color={theme.colors.brand[700]} /></Animated.View>
          <Text style={styles.whenText}>Reminds you {formatWhen(when.toISOString())} · shows here under Reminders</Text>
        </Animated.View>
      ) : null}

      {cal.sheetError ? (
        <Animated.View entering={FadeInDown.duration(200)} exiting={FadeOut.duration(140)}>
          <Animated.View style={[styles.errorRow, shakeStyle]}>
            <AlertCircle size={14} color={theme.colors.semantic.danger} />
            <Text style={styles.errorText}>{cal.sheetError}</Text>
          </Animated.View>
        </Animated.View>
      ) : null}

      <PressableScale onPress={cal.save} disabled={!!cal.saving || !draft.text.trim() || !draft.event} scaleTo={0.97}
        style={[styles.saveBtn, (!!cal.saving || !draft.text.trim() || !draft.event) && { opacity: 0.55 }]}
        leadingIcon={cal.saving === 'save' ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" />}>
        <Text style={styles.saveText}>{cal.saving === 'save' ? 'Saving…' : draft.remind !== 'none' ? 'Save reminder' : 'Save note'}</Text>
      </PressableScale>
      {cal.editing ? (
        <PressableScale onPress={cal.remove} disabled={!!cal.saving} scaleTo={0.97} style={styles.deleteBtn}
          leadingIcon={cal.saving === 'delete' ? <ActivityIndicator size="small" color={theme.colors.semantic.danger} /> : <Trash2 size={15} color={theme.colors.semantic.danger} />}>
          <Text style={styles.deleteText}>Delete note</Text>
        </PressableScale>
      ) : null}
    </Sheet>
  );
}

const styles = makeStyles(() => ({
  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 20, paddingHorizontal: 14, paddingTop: 10, paddingBottom: 12, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  smallBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  smallBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  muted: { fontSize: 13, lineHeight: 18, color: theme.colors.text.muted },
  subLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted, marginTop: 14, marginBottom: 8 },

  monthRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 4, marginBottom: 6 },
  navBtn: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  monthText: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  grid: { flexDirection: 'row', flexWrap: 'wrap' },
  weekday: { width: '14.2857%', textAlign: 'center', fontSize: 10, fontWeight: '800', letterSpacing: 0.5, color: theme.colors.text.muted, paddingVertical: 4 },
  cell: { width: '14.2857%', alignItems: 'center', paddingVertical: 2 },
  monthTitle: { flex: 1, alignItems: 'center', overflow: 'hidden' },
  gridClip: { overflow: 'hidden' },
  bubble: { width: 34, height: 34, borderRadius: 17, alignItems: 'center', justifyContent: 'center' },
  todayRing: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: 17, borderWidth: 2, borderColor: theme.colors.brand[400] },
  todayRingStatic: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: 17, borderWidth: 1.5, borderColor: theme.colors.brand[600] },
  selFill: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: 17, backgroundColor: theme.colors.brand[600] },
  dayNum: { fontSize: 13, fontWeight: '600', color: theme.colors.text.display },
  dayOut: { color: theme.colors.text.muted, opacity: 0.5 },
  dots: { flexDirection: 'row', gap: 3, height: 6, marginTop: 1 },
  marks: { flexDirection: 'row', alignItems: 'center', gap: 2, height: 11, marginTop: 1 },
  band: { position: 'absolute', top: 2, height: 34, left: 0, right: 0 },
  bandStart: { left: '50%' },
  bandEnd: { right: '50%' },
  legend: { flexDirection: 'row', justifyContent: 'center', flexWrap: 'wrap', gap: 14, marginTop: 6 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  legendText: { fontSize: 11, fontWeight: '600', color: theme.colors.text.muted },
  dot: { width: 5, height: 5, borderRadius: 3 },
  noteDot: { backgroundColor: theme.colors.semantic.warning },
  dueDot: { backgroundColor: theme.colors.semantic.danger },

  dayPanel: { marginTop: 8, paddingTop: 10, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  dayHead: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  dayTitle: { flex: 1, fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  evRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, borderLeftWidth: 4, marginBottom: 8 },
  evName: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  evMeta: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1, flexShrink: 1 },
  whereRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 1 },
  evDot: { width: 8, height: 8, borderRadius: 4 },

  noteRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, padding: 12, borderRadius: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#fffbeb'), marginBottom: 8 },
  noteRowDue: { borderColor: theme.colors.semantic.dangerBorder, backgroundColor: theme.colors.semantic.dangerBg },
  check: { width: 20, height: 20, borderRadius: 6, borderWidth: 2, borderColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  checkOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },
  noteText: { fontSize: 14, lineHeight: 19, color: theme.colors.text.display },
  noteDone: { textDecorationLine: 'line-through', color: theme.colors.text.muted },
  noteMeta: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 4 },
  noteMetaText: { fontSize: 12, color: theme.colors.text.muted, flexShrink: 1 },
  remindChip: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start', marginTop: 6, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  remindChipDue: { backgroundColor: toneBg('#ffffff') },
  remindChipText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700] },

  dayPill: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 12, borderWidth: 1, borderColor: theme.colors.semantic.warningBorder, backgroundColor: toneBg('#fffbeb'), marginBottom: 8 },
  dayPillDue: { borderColor: theme.colors.semantic.dangerBorder, backgroundColor: theme.colors.semantic.dangerBg },
  dayPillText: { flex: 1, fontSize: 13, fontWeight: '700', color: theme.colors.text.display },
  tiles: { flexDirection: 'row', gap: 8, marginTop: 12 },
  tile: { flex: 1, alignItems: 'center', gap: 2, paddingVertical: 10, borderRadius: 14, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  tileAlert: { borderColor: theme.colors.semantic.dangerBorder },
  tileIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', marginBottom: 2 },
  tileValue: { fontSize: 18, fontWeight: '900', color: theme.colors.text.display },
  tileLabel: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },
  listFilters: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, paddingTop: 2, paddingBottom: 4 },
  listFilter: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 11, paddingVertical: 6, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  listFilterOn: { borderColor: theme.colors.brand[600], backgroundColor: theme.colors.brand[600] },
  listFilterText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.body },
  listFilterCount: { fontSize: 11, fontWeight: '800', color: theme.colors.text.muted },
  newNoteBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 11, borderRadius: 999, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50], marginTop: 8 },
  newNoteText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  listEmpty: { alignItems: 'center', gap: 8, paddingVertical: 30 },
  upRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 8 },
  upDate: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  upDay: { fontSize: 16, lineHeight: 18, fontWeight: '900' },
  upMon: { fontSize: 9, fontWeight: '800', letterSpacing: 0.6 },
  relText: { fontSize: 11, fontWeight: '800', color: theme.colors.text.muted },

  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 6, maxWidth: '100%', paddingHorizontal: 11, paddingVertical: 7, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  chipOn: { borderColor: theme.colors.brand[600], backgroundColor: theme.colors.brand[50] },
  chipText: { fontSize: 13, fontWeight: '600', color: theme.colors.text.body, flexShrink: 1 },
  chipTextOn: { color: theme.colors.brand[800], fontWeight: '800' },
  dayPick: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  dayPickText: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display },
  textArea: { minHeight: 96, borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[300], padding: 12, fontSize: 15, lineHeight: 20, color: theme.colors.text.display, backgroundColor: toneBg('#ffffff') },
  timeRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 10 },
  timeInput: { width: 86, height: 44, borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[300], paddingHorizontal: 12, fontSize: 16, fontWeight: '700', textAlign: 'center', color: theme.colors.text.display, backgroundColor: toneBg('#ffffff') },
  ampm: { flexDirection: 'row', borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[300], overflow: 'hidden' },
  ampmBtn: { paddingHorizontal: 12, height: 41, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },
  ampmBtnOn: { backgroundColor: theme.colors.brand[600] },
  ampmText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  ampmTextOn: { color: 'white' },
  whenRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 10 },
  whenText: { flex: 1, fontSize: 12, fontWeight: '600', color: theme.colors.brand[700] },
  errorRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 12, padding: 10, borderRadius: 12, backgroundColor: theme.colors.semantic.dangerBg },
  errorText: { flex: 1, fontSize: 13, fontWeight: '600', color: theme.colors.semantic.danger },
  saveBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 15, borderRadius: 999, backgroundColor: theme.colors.brand[600], marginTop: 16, marginBottom: 6 },
  saveText: { fontSize: 15, fontWeight: '800', color: 'white' },
  deleteBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 12, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, backgroundColor: toneBg('#ffffff'), marginBottom: 6 },
  deleteText: { fontSize: 14, fontWeight: '800', color: theme.colors.semantic.danger },
}));
