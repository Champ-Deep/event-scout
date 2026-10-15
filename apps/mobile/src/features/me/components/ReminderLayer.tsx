import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, AppState, Vibration, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  SlideInUp, SlideOutUp, FadeIn, useSharedValue, useAnimatedStyle, useReducedMotion, withRepeat, withSequence, withTiming, withDelay,
} from 'react-native-reanimated';
import { BellRing, Check, Clock, ChevronRight } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';
import { api, type EventNote } from '../../../lib/api';
import { useSession } from '../../../providers/SessionProvider';
import { profileStore, type Profile } from '../../profile';
import { PressableScale } from '../../onboarding/components';
import { formatWhen } from '../../events/dates';
import { autoReminders, isAuto, setAutoDone, snoozeAuto } from '../../events/autoReminders';

/* Calendar reminders (Me → Calendar) pop up on any tab when they're due, while the app is open:
   Done · Snooze 10 min · Open. The phone itself isn't notified while the app is closed — that
   needs push notifications (expo-notifications + a new app build). */

const POLL_MS = 60_000;      // refresh the notes from the server
const TICK_MS = 10_000;      // check what's due
const SNOOZE_MIN = 10;
const RECENT_MS = 3 * 86400_000; // older overdue reminders stay in the calendar instead of popping up

function Wiggle({ children }: { children: React.ReactNode }) {
  const reduce = useReducedMotion();
  const r = useSharedValue(0);
  useEffect(() => {
    if (reduce) return;
    const s = (to: number) => withTiming(to, { duration: 70 });
    r.value = withRepeat(withSequence(s(-1), s(1), s(-0.7), s(0.7), s(0), withDelay(1400, s(0))), -1);
  }, [reduce]);
  const style = useAnimatedStyle(() => ({ transform: [{ rotate: `${r.value * 18}deg` }] }));
  return <Animated.View style={style}>{children}</Animated.View>;
}

export function ReminderLayer() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { isAuthenticated, userId } = useSession();
  const [serverNotes, setNotes] = useState<EventNote[]>([]);
  // + the automatic "evening before" reminders for your scheduled events
  const [auto, setAuto] = useState<EventNote[]>([]);
  const notes = [...serverNotes, ...auto];
  const [current, setCurrent] = useState<EventNote | null>(null);
  const [busy, setBusy] = useState<'done' | 'snooze' | null>(null);
  const shown = useRef(new Set<string>()); // id@remindAt already popped up this session
  const [, setTick] = useState(0);

  const refresh = useCallback(async () => {
    if (!isAuthenticated) return;
    const [ns, pe, pr] = await Promise.allSettled([
      api.listEventNotes(), api.listMyPlannedEvents(), userId ? profileStore.load(userId) : Promise.resolve(null),
    ]);
    if (ns.status === 'fulfilled') setNotes(Array.isArray(ns.value) ? ns.value : []);
    if (pe.status === 'fulfilled') {
      const p: Profile = (pr.status === 'fulfilled' ? pr.value : null) ?? {};
      setAuto(autoReminders(Array.isArray(pe.value) ? pe.value : [], p?.events || [], p?.ended_events || []));
    }
  }, [isAuthenticated, userId]);

  useEffect(() => {
    refresh();
    const poll = setInterval(refresh, POLL_MS);
    const tick = setInterval(() => setTick((t) => t + 1), TICK_MS);
    const sub = AppState.addEventListener('change', (s) => { if (s === 'active') refresh(); });
    return () => { clearInterval(poll); clearInterval(tick); sub.remove(); };
  }, [refresh]);

  // Pick the next due reminder (oldest first) that hasn't popped up yet.
  useEffect(() => {
    if (current) return;
    const now = Date.now();
    const due = notes
      .filter((n) => !n.done && n.remindAt)
      .filter((n) => { const t = new Date(n.remindAt!).getTime(); return t <= now && now - t < RECENT_MS; })
      .filter((n) => !shown.current.has(`${n.id}@${n.remindAt}`))
      .sort((a, b) => (a.remindAt! < b.remindAt! ? -1 : 1))[0];
    if (due) {
      shown.current.add(`${due.id}@${due.remindAt}`);
      setCurrent(due);
      try { Vibration.vibrate([0, 180, 120, 180]); } catch { /* no vibrator */ }
    }
  });

  const close = () => { setCurrent(null); setBusy(null); };

  const markDone = async () => {
    if (!current || busy) return;
    setBusy('done');
    if (isAuto(current)) {
      setAutoDone(current.id, true);
      setAuto((prev) => prev.map((n) => (n.id === current.id ? { ...n, done: true } : n)));
      close();
      return;
    }
    try {
      const saved = await api.updateEventNote(current.id, { done: true });
      setNotes((prev) => prev.map((n) => (n.id === saved.id ? saved : n)));
    } catch { /* stays due; it shows in the calendar */ }
    close();
  };

  const snooze = async () => {
    if (!current || busy) return;
    setBusy('snooze');
    const at = new Date(Date.now() + SNOOZE_MIN * 60_000).toISOString();
    if (isAuto(current)) {
      snoozeAuto(current.id, at);
      setAuto((prev) => prev.map((n) => (n.id === current.id ? { ...n, remindAt: at } : n)));
      close();
      return;
    }
    try {
      const saved = await api.updateEventNote(current.id, { remindAt: at });
      setNotes((prev) => prev.map((n) => (n.id === saved.id ? saved : n)));
    } catch { /* keep it as is */ }
    close();
  };

  // Me → Calendar, on the reminder's day (`t` makes a repeat tap scroll again).
  const open = () => {
    const day = current?.date || '';
    close();
    router.navigate({ pathname: '/me', params: { open: 'calendar', ...(day ? { day } : {}), t: String(Date.now()) } } as any);
  };

  if (!current) return null;
  return (
    <Animated.View entering={SlideInUp.springify().damping(18)} exiting={SlideOutUp.duration(200)}
      style={[styles.wrap, { top: insets.top + 8 }]} pointerEvents="box-none">
      <View style={styles.card} accessibilityRole="alert" accessibilityLiveRegion="assertive">
        <View style={styles.head}>
          <View style={styles.icon}><Wiggle><BellRing size={17} color="white" /></Wiggle></View>
          <View style={{ flex: 1 }}>
            <Text style={styles.kicker}>REMINDER · {current.event.toUpperCase()}</Text>
            <Text style={styles.when}>{current.remindAt ? formatWhen(current.remindAt) : ''}</Text>
          </View>
        </View>
        <Animated.View entering={FadeIn.delay(120).duration(220)}>
          <Text style={styles.text} numberOfLines={4}>{current.text}</Text>
        </Animated.View>
        <View style={styles.actions}>
          <PressableScale onPress={markDone} disabled={!!busy} scaleTo={0.94} containerStyle={{ flex: 1 }} style={[styles.btn, styles.btnPrimary]}
            leadingIcon={busy === 'done' ? <ActivityIndicator size="small" color="white" /> : <Check size={15} color="white" />}>
            <Text style={styles.btnPrimaryText}>Done</Text>
          </PressableScale>
          <PressableScale onPress={snooze} disabled={!!busy} scaleTo={0.94} containerStyle={{ flex: 1 }} style={styles.btn}
            leadingIcon={busy === 'snooze' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Clock size={15} color={theme.colors.brand[700]} />}>
            <Text style={styles.btnText}>{SNOOZE_MIN} min</Text>
          </PressableScale>
          <PressableScale onPress={open} disabled={!!busy} scaleTo={0.94} style={styles.btn} accessibilityLabel="Open the calendar"
            trailingIcon={<ChevronRight size={15} color={theme.colors.brand[700]} />}>
            <Text style={styles.btnText}>Open</Text>
          </PressableScale>
        </View>
      </View>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  wrap: { position: 'absolute', left: 12, right: 12, zIndex: 100, elevation: 100 },
  card: { borderRadius: 20, padding: 14, gap: 10, backgroundColor: theme.colors.bg.surface, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, ...theme.shadows.card, elevation: 12 },
  head: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  icon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.danger },
  kicker: { fontSize: 10, fontWeight: '800', letterSpacing: 0.8, color: theme.colors.semantic.danger },
  when: { fontSize: 12, fontWeight: '600', color: theme.colors.text.muted, marginTop: 1 },
  text: { fontSize: 15, lineHeight: 20, fontWeight: '600', color: theme.colors.text.display },
  actions: { flexDirection: 'row', gap: 8 },
  btn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 10, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.brand[200], backgroundColor: toneBg('#ffffff') },
  btnPrimary: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },
  btnPrimaryText: { fontSize: 14, fontWeight: '800', color: 'white' },
  btnText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
}));
