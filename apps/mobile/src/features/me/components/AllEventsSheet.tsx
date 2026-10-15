import React, { useEffect, useMemo, useState } from 'react';
import { View, TextInput, ActivityIndicator } from 'react-native';
import Animated, {
  FadeIn, FadeInDown, FadeOut, LinearTransition, ZoomIn,
  useSharedValue, useAnimatedStyle, useReducedMotion, withRepeat, withTiming, Easing,
} from 'react-native-reanimated';
import { CalendarDays, MapPin, Search, Users, AlertCircle, X } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { Sheet } from '../../../ui/Sheet';
import { api, isOffline, type PlannedEvent } from '../../../lib/api';
import { PressableScale } from '../../onboarding/components';
import { EndEventButton } from '../../events/EndEventButton';
import { MONTHS_SHORT, parseYmd, todayYmd, formatRange, relativeToToday } from '../../events/dates';

/* Me → Your events → All: every event — yours and every one the admins scheduled — with its dates,
   place, status and your role. Tap one for its full page (details, files, floor map, booths). */

type Filter = 'all' | 'upcoming' | 'live' | 'ended' | 'mine';
const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'All' }, { key: 'upcoming', label: 'Upcoming' }, { key: 'live', label: 'Happening now' },
  { key: 'ended', label: 'Past & ended' }, { key: 'mine', label: 'Yours' },
];
const k = (s?: string | null) => (s || '').split(/\s+/).join(' ').toLowerCase();

interface Row {
  name: string;
  plan: PlannedEvent | null;
  mine: boolean;
  active: boolean;
  ended: boolean;          // you ended it, its dates are over, or it's a previous event you met people at
  endedByYou: boolean;
  previous: boolean;       // not in your list any more, but you have contacts from it
  directory: boolean;      // has an exhibitor directory
  status: 'live' | 'upcoming' | 'ended' | 'undated';
}

type Section = 'live' | 'upcoming' | 'yours' | 'other' | 'past';
const SECTIONS: { key: Section; label: string }[] = [
  { key: 'live', label: 'LIVE NOW' }, { key: 'upcoming', label: 'UPCOMING' }, { key: 'yours', label: 'YOUR EVENTS' },
  { key: 'other', label: 'OTHER EVENTS' }, { key: 'past', label: 'PAST & ENDED' },
];
const sectionOf = (r: Row): Section => r.status === 'live' ? 'live' : r.status === 'upcoming' ? 'upcoming' : r.ended ? 'past' : r.mine ? 'yours' : 'other';

/** Red dot that pings outward — marks a live event. */
function LivePing() {
  const reduce = useReducedMotion();
  const t = useSharedValue(0);
  useEffect(() => { if (!reduce) t.value = withRepeat(withTiming(1, { duration: 1300, easing: Easing.out(Easing.quad) }), -1); }, [reduce]);
  const ring = useAnimatedStyle(() => ({ opacity: 0.7 * (1 - t.value), transform: [{ scale: 1 + t.value * 1.6 }] }));
  return (
    <View style={styles.liveWrap} pointerEvents="none">
      <Animated.View style={[styles.liveRing, ring]} />
      <View style={styles.liveDot} />
    </View>
  );
}

export function AllEventsSheet({ visible, onClose, myEvents, active, ended, byEvent, onOpen, onToggleEnded, busy }: {
  visible: boolean; onClose: () => void;
  myEvents: string[]; active: string | null; ended: string[]; byEvent: Record<string, number>;
  onOpen: (name: string) => void; onToggleEnded: (name: string) => void; busy: string | null;
}) {
  const [planned, setPlanned] = useState<PlannedEvent[] | null>(null);
  const [directory, setDirectory] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    setError('');
    api.listMyPlannedEvents('all')
      .then((r) => { if (!cancelled) setPlanned(Array.isArray(r) ? r : []); })
      .catch((e) => { if (!cancelled) { setPlanned((p) => p ?? []); setError(isOffline(e)
        ? "You're offline — showing your own events only."
        : "Couldn't load the scheduled events — showing your own events only."); } });
    // Events with an exhibitor directory (GET /api/events)
    api.listEvents()
      .then((r) => { if (!cancelled) setDirectory((Array.isArray(r) ? r : []).map((e: any) => e?.name).filter(Boolean)); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [visible]);

  const rows = useMemo<Row[]>(() => {
    const today = todayYmd();
    const byKey = new Map<string, Row>();
    const endedKeys = new Set(ended.map(k));
    const mineKeys = new Set(myEvents.map(k));
    const metKeys = new Set(Object.keys(byEvent).filter((n) => (byEvent[n] ?? 0) > 0).map(k));
    const dirKeys = new Set(directory.map(k));
    const add = (name: string, plan: PlannedEvent | null) => {
      const key = k(name);
      if (!key) return;
      const prev = byKey.get(key);
      const p = plan ?? prev?.plan ?? null;
      const mine = mineKeys.has(key) || !!p?.myRole;
      const isActive = k(active) === key;
      const last = p?.startDate ? (p.endDate && p.endDate >= p.startDate ? p.endDate : p.startDate) : null;
      const endedByYou = endedKeys.has(key);
      const datesOver = !!last && last < today;
      // A previous event: you met people there but it's no longer in your list (and isn't scheduled ahead).
      const previous = !mine && !isActive && metKeys.has(key) && !(p?.startDate && !datesOver);
      const status: Row['status'] = endedByYou || datesOver || previous ? 'ended'
        : p?.startDate ? (p.startDate <= today ? 'live' : 'upcoming')
        : isActive ? 'live' : 'undated';  // your active event is live for you even without dates
      byKey.set(key, {
        name: prev?.name ?? name, plan: p, mine, active: isActive,
        ended: status === 'ended', endedByYou, previous, directory: dirKeys.has(key), status,
      });
    };
    myEvents.forEach((n) => add(n, null));
    ended.forEach((n) => add(n, null));
    Object.keys(byEvent).forEach((n) => add(n, null));
    directory.forEach((n) => add(n, null));
    (planned || []).forEach((e) => add(e.name, e));
    const rank = { live: 0, upcoming: 1, undated: 2, ended: 3 } as const;
    return [...byKey.values()].sort((a, b) => {
      if (a.active !== b.active) return a.active ? -1 : 1;
      if (rank[a.status] !== rank[b.status]) return rank[a.status] - rank[b.status];
      const da = a.plan?.startDate || '', db = b.plan?.startDate || '';
      if (a.status === 'ended') return db.localeCompare(da) || a.name.localeCompare(b.name);
      return (da && db ? da.localeCompare(db) : da ? -1 : db ? 1 : 0) || a.name.localeCompare(b.name);
    });
  }, [myEvents, planned, ended, active, byEvent, directory]);

  const counts = useMemo(() => ({
    all: rows.length, upcoming: rows.filter((r) => r.status === 'upcoming').length, live: rows.filter((r) => r.status === 'live').length,
    ended: rows.filter((r) => r.ended).length, mine: rows.filter((r) => r.mine).length,
  }), [rows]);

  const q = query.trim().toLowerCase();
  const shown = rows.filter((r) => (filter === 'all' || (filter === 'mine' ? r.mine : filter === 'ended' ? r.ended : r.status === filter))
    && (!q || [r.name, r.plan?.city, r.plan?.venue, r.plan?.country, r.plan?.organizer].some((v) => (v || '').toLowerCase().includes(q))));

  const header = (
    <View style={{ gap: 8, paddingTop: 2 }}>
      <View style={styles.searchBox}>
        <Search size={15} color={theme.colors.text.muted} />
        <TextInput value={query} onChangeText={setQuery} placeholder="Search events, cities, venues" placeholderTextColor={theme.colors.text.muted}
          style={styles.searchInput} returnKeyType="search" />
        {query ? (
          <Animated.View entering={ZoomIn.duration(160)}>
            <PressableScale onPress={() => setQuery('')} hitSlop={10} scaleTo={0.8} accessibilityLabel="Clear search" leadingIcon={<X size={15} color={theme.colors.text.muted} />} />
          </Animated.View>
        ) : null}
      </View>
      <View style={styles.filters}>
        {FILTERS.map((f) => {
          const on = filter === f.key;
          return (
            <Animated.View key={f.key} layout={LinearTransition.springify().damping(18)}>
              <PressableScale onPress={() => setFilter(f.key)} scaleTo={0.92} style={[styles.filter, on && styles.filterOn]}
                accessibilityRole="tab" accessibilityState={{ selected: on }}>
                <Text style={[styles.filterText, on && styles.filterTextOn]}>{f.label}</Text>
                <Text style={[styles.filterCount, on && styles.filterTextOn]}>{counts[f.key]}</Text>
              </PressableScale>
            </Animated.View>
          );
        })}
      </View>
    </View>
  );

  return (
    <Sheet visible={visible} title="All events" icon={<CalendarDays size={15} color="white" />} onClose={onClose} header={header} tall deferBody>
      {error ? (
        <Animated.View entering={FadeInDown.duration(220)} style={styles.notice}>
          <AlertCircle size={14} color={theme.colors.semantic.warning} />
          <Text style={styles.noticeText}>{error}</Text>
        </Animated.View>
      ) : null}
      {planned === null && !myEvents.length ? (
        <View style={{ paddingVertical: 30 }}><ActivityIndicator color={theme.colors.brand[600]} /></View>
      ) : null}
      {(filter === 'all' ? SECTIONS : [null]).map((sec) => {
        const list = sec ? shown.filter((r) => sectionOf(r) === sec.key) : shown;
        if (!list.length) return null;
        return (
          <Animated.View key={sec?.key ?? filter} layout={LinearTransition.springify().damping(20)}>
            {sec ? (
              <Animated.View entering={FadeIn.duration(220)} style={styles.sectionHead}>
                {sec.key === 'live' ? <LivePing /> : null}
                <Text style={[styles.sectionLabel, sec.key === 'live' && { color: theme.colors.semantic.danger }]}>{sec.label}</Text>
                <Text style={styles.sectionCount}>{list.length}</Text>
              </Animated.View>
            ) : null}
            {list.map((r, i) => renderRow(r, i))}
          </Animated.View>
        );
      })}
      {planned !== null && !shown.length ? (
        <Animated.View entering={FadeIn.duration(220)} style={styles.empty}>
          <CalendarDays size={22} color={theme.colors.text.muted} />
          <Text style={styles.emptyText}>
            {q ? `No events match “${query.trim()}”.` : filter === 'all' ? 'No events yet. Add one, or your admins can schedule events for you.' : 'Nothing here right now.'}
          </Text>
        </Animated.View>
      ) : null}
      <View style={{ height: 12 }} />
    </Sheet>
  );

  function renderRow(r: Row, i: number) {
        const p = r.plan;
        const d = parseYmd(p?.startDate);
        const rel = r.endedByYou ? null : relativeToToday(p?.startDate, p?.endDate);
        const where = [p?.venue, p?.city].filter(Boolean).join(', ');
        const n = byEvent[r.name] ?? 0;
        return (
          <Animated.View key={k(r.name)} entering={FadeInDown.delay(Math.min(i, 10) * 45).duration(240)} exiting={FadeOut.duration(140)}
            layout={LinearTransition.springify().damping(20)}>
            <PressableScale onPress={() => onOpen(r.name)} scaleTo={0.97} style={[styles.row, r.active && styles.rowActive, r.ended && styles.rowEnded]}
              accessibilityLabel={`${r.name}${p?.startDate ? `, ${formatRange(p.startDate, p.endDate)}` : ''}. Open its details`}>
              <View style={[styles.dateTile, r.ended && styles.dateTileEnded, r.status === 'live' && styles.dateTileLive]}>
                {d ? (
                  <>
                    <Text style={[styles.dateDay, r.ended && styles.mutedText]}>{d.getDate()}</Text>
                    <Text style={[styles.dateMon, r.ended && styles.mutedText]}>{(MONTHS_SHORT[d.getMonth()] || '').toUpperCase()}</Text>
                  </>
                ) : <CalendarDays size={18} color={r.ended ? theme.colors.text.muted : theme.colors.brand[600]} />}
                {r.status === 'live' ? <View style={styles.tilePing}><LivePing /></View> : null}
              </View>
              <View style={{ flex: 1, gap: 2 }}>
                <Text style={[styles.name, r.ended && styles.mutedText]} numberOfLines={1}>{r.name}</Text>
                <Text style={styles.meta} numberOfLines={1}>{p?.startDate ? formatRange(p.startDate, p.endDate) : 'Dates not set'}</Text>
                {where ? (
                  <View style={styles.whereRow}><MapPin size={11} color={theme.colors.text.muted} /><Text style={styles.meta} numberOfLines={1}>{where}</Text></View>
                ) : null}
                <View style={styles.badges}>
                  {r.active ? <Badge text="Active" tone="live" /> : null}
                  {r.ended ? <Badge text={r.endedByYou ? 'Ended by you' : r.previous ? 'Previous' : 'Ended'} tone="past" />
                    : rel ? <Badge text={rel.label} tone={rel.tone === 'live' ? 'live' : rel.tone === 'soon' ? 'soon' : 'plain'} />
                    : r.status === 'live' ? <Badge text="Live" tone="live" /> : null}
                  {p?.myRole ? <Badge text={p.myRole === 'lead' ? 'You lead' : 'On the team'} tone="brand" />
                    : r.mine ? <Badge text="Your event" tone="brand" /> : p ? <Badge text="Scheduled by admin" tone="plain" /> : null}
                  {r.directory && !p ? <Badge text="Exhibitor list" tone="plain" /> : null}
                  {n ? (
                    <View style={styles.countPill}><Users size={10} color={theme.colors.text.muted} /><Text style={styles.countText}>{n}</Text></View>
                  ) : null}
                </View>
              </View>
              {r.mine ? (
                <EndEventButton compact ended={r.endedByYou} busy={busy === r.name} onPress={() => onToggleEnded(r.name)} />
              ) : null}
            </PressableScale>
          </Animated.View>
        );
  }
}

function Badge({ text, tone }: { text: string; tone: 'live' | 'soon' | 'past' | 'brand' | 'plain' }) {
  const t = {
    live: [theme.colors.semantic.successBg, theme.colors.semantic.successText],
    soon: [theme.colors.semantic.warningBg, theme.colors.semantic.warning],
    past: [toneBg('#f1f5f9'), toneFg('#64748b')],
    brand: [theme.colors.brand[50], theme.colors.brand[700]],
    plain: [toneBg('#f8fafc'), theme.colors.text.body],
  }[tone];
  return (
    <Animated.View entering={ZoomIn.duration(200)} style={[styles.badge, { backgroundColor: t[0] }]}>
      <Text style={[styles.badgeText, { color: t[1] }]}>{text}</Text>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  searchBox: { flexDirection: 'row', alignItems: 'center', gap: 8, height: 42, borderRadius: 12, paddingHorizontal: 12, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  searchInput: { flex: 1, fontSize: 14, color: theme.colors.text.display, paddingVertical: 0 },
  filters: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, paddingBottom: 4 },
  filter: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 11, paddingVertical: 6, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  filterOn: { borderColor: theme.colors.brand[600], backgroundColor: theme.colors.brand[600] },
  filterText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.body },
  filterCount: { fontSize: 11, fontWeight: '800', color: theme.colors.text.muted },
  filterTextOn: { color: 'white' },
  notice: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 12, backgroundColor: theme.colors.semantic.warningBg, marginBottom: 10 },
  noticeText: { flex: 1, fontSize: 12, fontWeight: '600', color: theme.colors.semantic.warning },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 16, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'), marginBottom: 8 },
  rowActive: { borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50] },
  rowEnded: { opacity: 0.85 },
  dateTile: { width: 46, height: 50, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  dateTileEnded: { backgroundColor: toneBg('#f1f5f9') },
  dateTileLive: { backgroundColor: theme.colors.semantic.successBg },
  tilePing: { position: 'absolute', top: -3, right: -3 },
  liveWrap: { width: 10, height: 10, alignItems: 'center', justifyContent: 'center' },
  liveRing: { position: 'absolute', width: 10, height: 10, borderRadius: 5, backgroundColor: theme.colors.semantic.danger },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.colors.semantic.danger, borderWidth: 1.5, borderColor: toneBg('#ffffff') },
  sectionHead: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 6, marginBottom: 8 },
  sectionLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  sectionCount: { fontSize: 10, fontWeight: '800', color: theme.colors.text.muted },
  dateDay: { fontSize: 18, lineHeight: 20, fontWeight: '900', color: theme.colors.brand[700] },
  dateMon: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.brand[700] },
  mutedText: { color: theme.colors.text.muted },
  name: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  meta: { fontSize: 12, color: theme.colors.text.muted, flexShrink: 1 },
  whereRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 5, marginTop: 4 },
  badge: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 999 },
  badgeText: { fontSize: 10, fontWeight: '800' },
  countPill: { flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 999, backgroundColor: toneBg('#f1f5f9') },
  countText: { fontSize: 10, fontWeight: '800', color: theme.colors.text.muted },
  empty: { alignItems: 'center', gap: 8, paddingVertical: 30, paddingHorizontal: 20 },
  emptyText: { fontSize: 13, lineHeight: 18, textAlign: 'center', color: theme.colors.text.muted },
}));
