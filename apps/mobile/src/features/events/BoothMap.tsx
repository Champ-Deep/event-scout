import React, { useEffect, useMemo, useState } from 'react';
import { View, Pressable, ScrollView, ActivityIndicator } from 'react-native';
import Animated, {
  FadeIn, FadeInDown, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay, Easing,
} from 'react-native-reanimated';
import { Map as MapIcon, Route, Check, Footprints, Flag, ChevronDown, Navigation } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import type { Exhibitor } from '../../lib/api';
import { FileList, type FileItem } from '../files/FileList';

/* ------------------------------------------------------------------ helpers */

const OTHER = 'Other';
const byCode = (a?: string | null, b?: string | null) => (a || '').localeCompare(b || '', undefined, { numeric: true, sensitivity: 'base' });
export const hallOf = (x: Exhibitor) => (x.hall || '').trim() || OTHER;
const hallLabel = (h: string) => (h === OTHER ? 'Unassigned' : /^hall\b/i.test(h) ? h : `Hall ${h}`);
const sortHalls = (hs: string[]) => hs.sort((a, b) => (a === OTHER ? 1 : b === OTHER ? -1 : byCode(a, b)));
/** "Hall 5 · Booth B12" — without doubling a prefix the imported value already has ("Hall B" stays "Hall B"). */
export const locationLabel = (x: { hall?: string | null; booth?: string | null }) => {
  const hall = (x.hall || '').trim();
  const booth = (x.booth || '').trim();
  return [
    hall && (/^hall\b/i.test(hall) ? hall : `Hall ${hall}`),
    booth && (/^(booth|stand|stall)\b/i.test(booth) ? booth : `Booth ${booth}`),
  ].filter(Boolean).join(' · ');
};

/** Has a hall or booth number, so it can be laid out on the floor map. */
export const located = (x: Exhibitor) => !!((x.hall || '').trim() || (x.booth || '').trim());

export interface Preferences {
  goals: string[];        // this event's goals (profile.event_details[event].goals)
  industries: string[];   // profile.target_industries
  geographies: string[];  // profile.target_geographies
}

// Words in a goal that say what you'll do, not who you're after ("Book 10 demos with hospital CIOs").
const STOP = new Set(['book', 'meet', 'meeting', 'meetings', 'with', 'from', 'find', 'demo', 'demos', 'leads', 'lead', 'talk',
  'visit', 'least', 'more', 'some', 'their', 'them', 'they', 'that', 'this', 'into', 'about', 'event', 'booth', 'booths',
  'partner', 'partners', 'collect', 'cards', 'people', 'contacts', 'connect', 'learn', 'other', 'every', 'each', 'make']);
const words = (s: string) => [...new Set(s.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length >= 4 && !STOP.has(w)))];
const MAX_PICKS = 20;

export interface Guide {
  /** Booths to visit, in walking order (hall by hall, booth by booth — no doubling back). */
  stops: Exhibitor[];
  /** Why each booth was picked, e.g. "Goal: hospital CIOs", "Target industry: Healthcare". */
  reasons: Map<string, string[]>;
  /** Stalls analysed (have a hall or booth number) and the halls they're in. */
  analysed: number;
  halls: number;
  /** Any preferences to go on? */
  hasPrefs: boolean;
}

/**
 * Reads every stall against your preferences — this event's goals first, then your target industries
 * and regions — plus what you've already done here (booths you added / met someone from), keeps the
 * best matches and orders them into a walking route.
 */
export function buildGuide(exhibitors: Exhibitor[], isMet: (x: Exhibitor) => boolean, prefs: Preferences): Guide {
  const goals = prefs.goals.map((g) => ({ label: g, words: words(g) })).filter((g) => g.words.length);
  const inds = prefs.industries.map((i) => ({ label: i, words: words(i) })).filter((i) => i.words.length);
  const geos = prefs.geographies.map((g) => g.trim().toLowerCase()).filter(Boolean);
  const short = (s: string) => (s.length > 34 ? `${s.slice(0, 32)}…` : s);

  const pool = exhibitors.filter(located);
  const reasons = new Map<string, string[]>();
  const scored = pool.map((x) => {
    const hay = ` ${[x.name, x.category, x.subcategory, x.description, x.products, ...(x.tags || [])].join(' ').toLowerCase()} `;
    const why: string[] = [];
    let score = 0;
    goals.forEach((g) => { if (g.words.some((w) => hay.includes(w))) { score += 3; why.push(`Goal: ${short(g.label)}`); } });
    inds.forEach((i) => { if (i.words.some((w) => hay.includes(w))) { score += 2; why.push(`Industry: ${short(i.label)}`); } });
    const country = (x.country || '').toLowerCase();
    if (country && geos.some((g) => country.includes(g) || g.includes(country))) { score += 1; why.push(`Region: ${x.country}`); }
    if (x.addedByMe) { score += 2; why.push('Added by you'); }
    if (isMet(x)) { score += 1; why.push(x.visited ? 'Visited' : 'You met someone here'); }
    if (why.length) reasons.set(x.id, why);
    return { x, score };
  });

  const stops = scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_PICKS)
    .map((s) => s.x)
    .sort((a, b) => {
      const ha = hallOf(a), hb = hallOf(b);
      if (ha !== hb) return sortHalls([ha, hb])[0] === ha ? -1 : 1;
      return byCode(a.booth, b.booth) || a.name.localeCompare(b.name);
    });
  return {
    stops,
    reasons,
    analysed: pool.length,
    halls: new Set(pool.map(hallOf)).size,
    hasPrefs: goals.length + inds.length + geos.length > 0,
  };
}

/* ------------------------------------------------------------- coming soon */

/** No floor plan for this event yet: a faded preview of the guide with a "coming soon" note. */
export function GuideComingSoon({ stalls }: { stalls: number }) {
  return (
    <Animated.View entering={FadeInDown.delay(80).duration(260)} style={styles.card}>
      <Head icon={<MapIcon size={14} color={theme.colors.brand[700]} />} label="FLOOR MAP & ROAD MAP"
        right={<View style={styles.soonPill}><Text style={styles.soonPillText}>COMING SOON</Text></View>} />
      <View style={styles.soonArt} pointerEvents="none">
        {[0, 1, 2].map((r) => (
          <View key={r} style={[styles.tileRow, { opacity: 0.55 - r * 0.15 }]}>
            {Array.from({ length: COLS }).map((_, c) => (
              <View key={c} style={[styles.tile, styles.soonTile, (r + c) % 4 === 1 && styles.soonTileOn]} />
            ))}
          </View>
        ))}
        <View style={styles.soonBadge}>
          <Route size={18} color={theme.colors.brand[600]} />
          <Text style={styles.soonTitle}>Your booth guide is coming soon</Text>
        </View>
      </View>
      <Text style={styles.muted}>
        Once the organiser's floor plan is in, we'll map {stalls ? `all ${stalls} stalls` : 'every stall'}, pick the booths that match your goals and
        targeting, and plan the shortest route through the halls.
      </Text>
    </Animated.View>
  );
}

function Progress({ value }: { value: number }) {
  const w = useSharedValue(0);
  useEffect(() => { w.value = withDelay(150, withTiming(value, { duration: 800, easing: Easing.out(Easing.cubic) })); }, [value]);
  const style = useAnimatedStyle(() => ({ width: `${w.value * 100}%` }));
  return <View style={styles.progTrack}><Animated.View style={[styles.progFill, style]} /></View>;
}

/** Soft ring that pulses around the next stop. */
function Pulse() {
  const t = useSharedValue(0);
  useEffect(() => { t.value = withRepeat(withTiming(1, { duration: 1500 }), -1); }, []);
  const style = useAnimatedStyle(() => ({ opacity: 0.55 * (1 - t.value), transform: [{ scale: 1 + t.value * 0.45 }] }));
  return <Animated.View pointerEvents="none" style={[StyleSheetAbsolute, styles.pulse, style]} />;
}
const StyleSheetAbsolute = { position: 'absolute' as const, top: 0, left: 0, right: 0, bottom: 0 };

/* ---------------------------------------------------------------- floor map */

const COLS = 5;
const MAX_TILES = 40;

export function FloorMap({ exhibitors, isMet, picks, nextId, focusId, onPick, planFiles, download }: {
  exhibitors: Exhibitor[];
  isMet: (x: Exhibitor) => boolean;
  /** Booths on your road map */
  picks: Set<string>;
  nextId: string | null;
  focusId: string | null;
  onPick: (x: Exhibitor) => void;
  planFiles: FileItem[];
  download: (id: string) => Promise<any>;
}) {
  const halls = useMemo(() => {
    const m = new Map<string, Exhibitor[]>();
    exhibitors.filter(located).forEach((x) => { const h = hallOf(x); m.set(h, [...(m.get(h) || []), x]); });
    m.forEach((list) => list.sort((a, b) => byCode(a.booth, b.booth) || a.name.localeCompare(b.name)));
    return sortHalls([...m.keys()]).map((h) => ({ hall: h, booths: m.get(h)! }));
  }, [exhibitors]);

  // Follow the focused booth (tapped in the route) or the next stop to its hall.
  const want = exhibitors.find((x) => x.id === focusId) ?? exhibitors.find((x) => x.id === nextId);
  const [hall, setHall] = useState<string | null>(null);
  useEffect(() => { if (want) setHall(hallOf(want)); }, [want?.id]);
  const current = halls.find((h) => h.hall === hall) ?? halls[0];

  if (!current) {
    return (
      <Animated.View entering={FadeInDown.duration(260)} style={styles.card}>
        <Head icon={<MapIcon size={14} color={theme.colors.brand[700]} />} label="FLOOR MAP" />
        <Text style={styles.muted}>Add a hall and booth number to booths and they'll be laid out here, hall by hall.</Text>
        {planFiles.length ? <FileList files={planFiles} download={download} /> : null}
      </Animated.View>
    );
  }

  const tiles = current.booths.slice(0, MAX_TILES);
  const rows: Exhibitor[][] = [];
  for (let i = 0; i < tiles.length; i += COLS) rows.push(tiles.slice(i, i + COLS));
  const done = current.booths.filter(isMet).length;

  return (
    <Animated.View entering={FadeInDown.delay(80).duration(260)} layout={LinearTransition.duration(260)} style={styles.card}>
      <Head icon={<MapIcon size={14} color={theme.colors.brand[700]} />} label="FLOOR MAP"
        right={<Text style={styles.count}>{halls.length} hall{halls.length === 1 ? '' : 's'}</Text>} />
      <View style={styles.analysis}>
        <Stat value={halls.reduce((n, h) => n + h.booths.length, 0)} label="stalls mapped" />
        <View style={styles.analysisDiv} />
        <Stat value={picks.size} label="for you" accent />
        <View style={styles.analysisDiv} />
        <Stat value={exhibitors.filter((x) => picks.has(x.id) && isMet(x)).length} label="done" />
      </View>

      {halls.length > 1 ? (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 6 }}>
          {halls.map((h) => {
            const on = h.hall === current.hall;
            const left = h.booths.filter((x) => picks.has(x.id) && !isMet(x)).length;
            return (
              <Pressable key={h.hall} onPress={() => setHall(h.hall)} style={[styles.hallChip, on && styles.hallChipOn]}>
                <Text style={[styles.hallChipText, on && { color: 'white' }]}>{hallLabel(h.hall)}</Text>
                <View style={[styles.hallBadge, on && { backgroundColor: 'rgba(255,255,255,0.22)' }]}>
                  <Text style={[styles.hallBadgeText, on && { color: 'white' }]}>{left || '✓'}</Text>
                </View>
              </Pressable>
            );
          })}
        </ScrollView>
      ) : null}

      <Animated.View key={current.hall} entering={FadeIn.duration(240)} style={styles.floor}>
        <View style={styles.floorHead}>
          <Text style={styles.floorTitle}>{hallLabel(current.hall)}</Text>
          <Text style={styles.floorSub}>{done}/{current.booths.length} visited</Text>
        </View>
        {rows.map((row, r) => (
          <React.Fragment key={r}>
            <View style={styles.tileRow}>
              {Array.from({ length: COLS }).map((_, c) => {
                const x = row[c];
                if (!x) return <View key={c} style={[styles.tile, styles.tileEmpty]} />;
                const met = isMet(x);
                const next = x.id === nextId;
                const focus = x.id === focusId;
                const pick = picks.has(x.id);
                return (
                  <Pressable key={x.id} onPress={() => onPick(x)} style={{ flex: 1 }}
                    accessibilityLabel={`${x.name}${x.booth ? `, booth ${x.booth}` : ''}${met ? ', visited' : next ? ', next stop' : pick ? ', recommended' : ''}`}>
                    <Animated.View entering={ZoomIn.delay(Math.min(r * COLS + c, 20) * 15).duration(220)}
                      style={[styles.tile, pick && styles.tilePick, met && styles.tileMet, next && styles.tileNext, focus && styles.tileFocus]}>
                      {next ? <Pulse /> : null}
                      {met ? <Check size={10} color={theme.colors.semantic.successText} strokeWidth={3} /> : null}
                      <Text numberOfLines={1} style={[styles.tileCode, met && { color: theme.colors.semantic.successText }, next && { color: 'white' }]}>
                        {x.booth || x.name.slice(0, 3).toUpperCase()}
                      </Text>
                      <Text numberOfLines={1} style={[styles.tileName, next && { color: 'rgba(255,255,255,0.85)' }]}>{x.name}</Text>
                    </Animated.View>
                  </Pressable>
                );
              })}
            </View>
          </React.Fragment>
        ))}
        {current.booths.length > MAX_TILES ? (
          <Text style={styles.more}>+{current.booths.length - MAX_TILES} more booths here — search the list below</Text>
        ) : null}
      </Animated.View>

      <View style={styles.legend}>
        <Legend swatch={[styles.tile, styles.legendSwatch, styles.tileNext]} label="Next stop" />
        <Legend swatch={[styles.tile, styles.legendSwatch, styles.tilePick]} label="For you" />
        <Legend swatch={[styles.tile, styles.legendSwatch, styles.tileMet]} label="Visited" />
        <Legend swatch={[styles.tile, styles.legendSwatch]} label="Other stalls" />
      </View>

      {planFiles.length ? (
        <View style={{ gap: 6 }}>
          <Text style={styles.subLabel}>OFFICIAL FLOOR PLAN</Text>
          <FileList files={planFiles} download={download} />
        </View>
      ) : null}
    </Animated.View>
  );
}

function Stat({ value, label, accent }: { value: number; label: string; accent?: boolean }) {
  return (
    <View style={{ flex: 1, alignItems: 'center' }}>
      <Text style={[styles.statValue, accent && { color: theme.colors.brand[600] }]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function Legend({ swatch, label }: { swatch: any[]; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={swatch} />
      <Text style={styles.legendText}>{label}</Text>
    </View>
  );
}

function Head({ icon, label, right }: { icon: React.ReactNode; label: string; right?: React.ReactNode }) {
  return (
    <View style={styles.head}>
      <View style={styles.headIcon}>{icon}</View>
      <Text style={[styles.label, { flex: 1 }]}>{label}</Text>
      {right}
    </View>
  );
}

/* ---------------------------------------------------------------- road map */

const FIRST = 5;

export function BoothRoute({ guide, isMet, nextId, focusId, busyId, onFocus, onToggle, onSetGoals }: {
  guide: Guide;
  isMet: (x: Exhibitor) => boolean;
  nextId: string | null;
  focusId: string | null;
  busyId: string | null;
  onFocus: (x: Exhibitor) => void;
  onToggle: (x: Exhibitor) => void;
  /** Jump to the goals card */
  onSetGoals: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const { stops, reasons } = guide;

  if (!stops.length) {
    return (
      <Animated.View entering={FadeInDown.delay(140).duration(260)} style={styles.card}>
        <Head icon={<Route size={14} color={theme.colors.brand[700]} />} label="YOUR ROAD MAP" />
        <Text style={styles.muted}>
          {guide.hasPrefs
            ? `None of the ${guide.analysed} stalls match your goals or targeting yet. Try broader goals — e.g. an industry or product you're after.`
            : `Tell us what you're after and we'll pick the booths to visit from ${guide.analysed} stalls and route you through them.`}
        </Text>
        <Pressable onPress={onSetGoals} style={styles.goalsBtn}>
          <Flag size={13} color={theme.colors.brand[700]} />
          <Text style={styles.goalsBtnText}>{guide.hasPrefs ? 'Edit your goals' : 'Set your goals'}</Text>
        </Pressable>
      </Animated.View>
    );
  }

  const done = stops.filter(isMet).length;
  const allDone = done === stops.length;
  // Hall journey: Start → Hall A → Hall B → Done
  const legs: { hall: string; total: number; left: number }[] = [];
  stops.forEach((x) => {
    const h = hallOf(x);
    const last = legs[legs.length - 1];
    if (last?.hall === h) { last.total += 1; if (!isMet(x)) last.left += 1; }
    else legs.push({ hall: h, total: 1, left: isMet(x) ? 0 : 1 });
  });
  // Pending stops lead; finished ones fold away unless the full route is open.
  const nextIdx = stops.findIndex((x) => x.id === nextId);
  const shown = showAll ? stops : stops.slice(Math.max(0, nextIdx), Math.max(0, nextIdx) + FIRST);

  return (
    <Animated.View entering={FadeInDown.delay(140).duration(260)} layout={LinearTransition.duration(260)} style={styles.card}>
      <Head icon={<Route size={14} color={theme.colors.brand[700]} />} label="YOUR ROAD MAP"
        right={<Text style={styles.count}>{done}/{stops.length}</Text>} />
      <Text style={styles.muted}>
        {stops.length} booth{stops.length === 1 ? '' : 's'} to visit, picked from {guide.analysed} stalls for your goals and targeting — in walking order.
      </Text>
      <Progress value={stops.length ? done / stops.length : 0} />

      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.journey}>
        <View style={styles.legStart}><Navigation size={11} color={theme.colors.brand[700]} /><Text style={styles.legText}>Start</Text></View>
        {legs.map((l, i) => (
          <React.Fragment key={`${l.hall}-${i}`}>
            <View style={styles.legLine} />
            <View style={[styles.leg, l.left === 0 && styles.legDone]}>
              {l.left === 0 ? <Check size={11} color={theme.colors.semantic.successText} strokeWidth={3} /> : null}
              <Text style={[styles.legText, l.left === 0 && { color: theme.colors.semantic.successText }]}>{hallLabel(l.hall)} · {l.total}</Text>
            </View>
          </React.Fragment>
        ))}
        <View style={styles.legLine} />
        <View style={[styles.legStart, allDone && styles.legDone]}><Flag size={11} color={allDone ? theme.colors.semantic.successText : theme.colors.brand[700]} /><Text style={styles.legText}>Done</Text></View>
      </ScrollView>

      {allDone ? (
        <Animated.View entering={ZoomIn.duration(260)} style={styles.allDone}>
          <Check size={16} color={theme.colors.semantic.successText} strokeWidth={3} />
          <Text style={styles.allDoneText}>Route complete — every stop visited.</Text>
        </Animated.View>
      ) : null}

      <View>
        {shown.map((x, i) => {
          const met = isMet(x);
          const next = x.id === nextId;
          const focus = x.id === focusId;
          const prev = shown[i - 1];
          const hallChange = prev && hallOf(prev) !== hallOf(x);
          const last = i === shown.length - 1;
          const n = stops.indexOf(x) + 1;
          return (
            <Animated.View key={x.id} entering={FadeInDown.delay(Math.min(i, 8) * 40).duration(240)} layout={LinearTransition.duration(240)}>
              {hallChange ? (
                <View style={styles.walk}>
                  <View style={styles.rail}><View style={[styles.railLine, styles.railDashed]} /></View>
                  <Footprints size={12} color={theme.colors.text.muted} />
                  <Text style={styles.walkText}>Walk to {hallLabel(hallOf(x))}</Text>
                </View>
              ) : null}
              <Pressable onPress={() => onFocus(x)} style={({ pressed }) => [styles.stop, pressed && { opacity: 0.85 }]}>
                <View style={styles.rail}>
                  <View style={[styles.railLine, i === 0 && { top: 18 }, last && { bottom: '50%' }, met && { backgroundColor: theme.colors.semantic.success }]} />
                  <View style={[styles.node, met && styles.nodeMet, next && styles.nodeNext]}>
                    {met ? <Check size={12} color="white" strokeWidth={3} /> : <Text style={[styles.nodeText, next && { color: 'white' }]}>{n}</Text>}
                  </View>
                </View>
                <View style={[styles.stopBody, next && styles.stopBodyNext, focus && styles.stopBodyFocus]}>
                  <View style={{ flex: 1 }}>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                      {next ? <View style={styles.nextTag}><Text style={styles.nextTagText}>NEXT</Text></View> : null}
                      <Text style={[styles.stopName, met && styles.stopNameDone]} numberOfLines={1}>{x.name}</Text>
                    </View>
                    <Text style={styles.stopSub} numberOfLines={1}>
                      {[x.hall && hallLabel(hallOf(x)), x.booth && `Booth ${x.booth}`, x.category].filter(Boolean).join(' · ')}
                    </Text>
                    {!met && reasons.get(x.id)?.length ? (
                      <View style={styles.reasons}>
                        {reasons.get(x.id)!.slice(0, 2).map((r) => (
                          <View key={r} style={styles.reason}><Text style={styles.reasonText} numberOfLines={1}>{r}</Text></View>
                        ))}
                      </View>
                    ) : null}
                  </View>
                  <Pressable onPress={() => onToggle(x)} disabled={busyId === x.id} hitSlop={8}
                    style={[styles.tick, x.visited && styles.tickOn]}
                    accessibilityLabel={x.visited ? `Un-mark ${x.name} as visited` : `Mark ${x.name} as visited`}>
                    {busyId === x.id ? <ActivityIndicator size="small" color={x.visited ? 'white' : theme.colors.brand[700]} />
                      : <Check size={14} color={x.visited ? 'white' : theme.colors.brand[600]} strokeWidth={3} />}
                  </Pressable>
                </View>
              </Pressable>
            </Animated.View>
          );
        })}
      </View>

      {stops.length > shown.length || showAll ? (
        <Pressable onPress={() => setShowAll((s) => !s)} style={styles.moreRow}>
          <Text style={styles.link}>{showAll ? 'Show next stops only' : `Show full route (${stops.length})`}</Text>
          <View style={{ transform: [{ rotate: showAll ? '180deg' : '0deg' }] }}><ChevronDown size={15} color={theme.colors.brand[600]} /></View>
        </Pressable>
      ) : null}
    </Animated.View>
  );
}

/* ------------------------------------------------------------------- styles */

const styles = makeStyles(() => ({
  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 18, padding: 14, gap: 10, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider },
  head: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  headIcon: { width: 26, height: 26, borderRadius: 8, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  label: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  subLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  count: { fontSize: 12, fontWeight: '800', color: theme.colors.text.muted },
  muted: { fontSize: 12, lineHeight: 17, color: theme.colors.text.muted },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  moreRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, paddingTop: 2 },

  hallChip: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingLeft: 11, paddingRight: 5, paddingVertical: 5, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.bg.divider },
  hallChipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },
  hallChipText: { fontSize: 12, fontWeight: '800', color: theme.colors.text.display },
  hallBadge: { minWidth: 20, height: 20, paddingHorizontal: 5, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  hallBadgeText: { fontSize: 10, fontWeight: '900', color: theme.colors.brand[700] },

  floor: { borderRadius: 14, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.bg.dashed, backgroundColor: theme.colors.bg.app, padding: 10, gap: 6 },
  floorHead: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 },
  floorTitle: { fontSize: 14, fontWeight: '900', color: theme.colors.text.display },
  floorSub: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },
  tileRow: { flexDirection: 'row', gap: 6 },
  tile: { flex: 1, height: 46, borderRadius: 9, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 3, backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider },
  tileEmpty: { backgroundColor: 'transparent', borderColor: 'transparent' },
  tileMet: { backgroundColor: theme.colors.semantic.successBg, borderColor: theme.colors.semantic.successBg },
  tileNext: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },
  tileFocus: { borderWidth: 2.5, borderColor: theme.colors.brand[800] },
  tileCode: { fontSize: 11, fontWeight: '900', color: theme.colors.text.display },
  tileName: { fontSize: 8, fontWeight: '600', color: theme.colors.text.muted, marginTop: 1 },
  pulse: { borderRadius: 11, borderWidth: 2, borderColor: theme.colors.brand[400] },
  more: { fontSize: 11, color: theme.colors.text.muted, textAlign: 'center', paddingTop: 2 },
  legend: { flexDirection: 'row', flexWrap: 'wrap', gap: 14 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendSwatch: { flex: 0, width: 14, height: 14, borderRadius: 4 },
  legendText: { fontSize: 11, fontWeight: '700', color: theme.colors.text.body },

  progTrack: { height: 8, borderRadius: 4, backgroundColor: theme.colors.bg.surfaceHover, overflow: 'hidden' },
  progFill: { height: 8, borderRadius: 4, backgroundColor: theme.colors.semantic.success },
  journey: { alignItems: 'center', paddingVertical: 2 },
  legStart: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 9, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  leg: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 9, paddingVertical: 5, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.brand[200], backgroundColor: toneBg('#ffffff') },
  legDone: { backgroundColor: theme.colors.semantic.successBg, borderColor: theme.colors.semantic.successBg },
  legText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  legLine: { width: 16, height: 2, backgroundColor: theme.colors.brand[200] },
  allDone: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 12, backgroundColor: theme.colors.semantic.successBg },
  allDoneText: { flex: 1, fontSize: 13, fontWeight: '700', color: theme.colors.semantic.successText },

  stop: { flexDirection: 'row', alignItems: 'stretch', gap: 8 },
  rail: { width: 28, alignItems: 'center', justifyContent: 'center' },
  railLine: { position: 'absolute', top: 0, bottom: 0, width: 2, backgroundColor: theme.colors.brand[100] },
  railDashed: { backgroundColor: 'transparent', borderLeftWidth: 2, borderStyle: 'dashed', borderColor: theme.colors.bg.dashed, width: 0 },
  node: { width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff'), borderWidth: 2, borderColor: theme.colors.brand[200] },
  nodeMet: { backgroundColor: theme.colors.semantic.success, borderColor: theme.colors.semantic.success },
  nodeNext: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },
  nodeText: { fontSize: 11, fontWeight: '900', color: theme.colors.brand[700] },
  stopBody: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 8, marginVertical: 4, padding: 10, borderRadius: 12, backgroundColor: theme.colors.bg.app, borderWidth: 1, borderColor: theme.colors.bg.divider },
  stopBodyNext: { borderColor: theme.colors.brand[200], backgroundColor: theme.colors.brand[50] },
  stopBodyFocus: { borderColor: theme.colors.brand[600], borderWidth: 1.5 },
  stopName: { flexShrink: 1, fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  stopNameDone: { color: theme.colors.text.muted, textDecorationLine: 'line-through' },
  stopSub: { fontSize: 11, color: theme.colors.text.muted, marginTop: 1 },
  nextTag: { paddingHorizontal: 6, paddingVertical: 1, borderRadius: 999, backgroundColor: theme.colors.brand[600] },
  nextTagText: { fontSize: 9, fontWeight: '900', letterSpacing: 0.5, color: 'white' },
  tick: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.brand[200] },
  tickOn: { backgroundColor: theme.colors.semantic.success, borderColor: theme.colors.semantic.success },
  reasons: { flexDirection: 'row', flexWrap: 'wrap', gap: 4, marginTop: 5 },
  reason: { maxWidth: '100%', paddingHorizontal: 7, paddingVertical: 2, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.brand[100] },
  reasonText: { fontSize: 10, fontWeight: '700', color: theme.colors.brand[700] },
  goalsBtn: { flexDirection: 'row', alignSelf: 'flex-start', alignItems: 'center', gap: 5, paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  goalsBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },

  tilePick: { backgroundColor: theme.colors.brand[50], borderColor: theme.colors.brand[300], borderWidth: 1.5 },
  analysis: { flexDirection: 'row', alignItems: 'center', paddingVertical: 8, borderRadius: 12, backgroundColor: theme.colors.bg.app },
  analysisDiv: { width: 1, height: 22, backgroundColor: theme.colors.bg.divider },
  statValue: { fontSize: 16, fontWeight: '900', color: theme.colors.text.display },
  statLabel: { fontSize: 10, fontWeight: '700', color: theme.colors.text.muted },

  soonPill: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: toneBg('#fef3c7') },
  soonPillText: { fontSize: 9, fontWeight: '900', letterSpacing: 1, color: toneFg('#b45309') },
  soonArt: { gap: 6, padding: 10, borderRadius: 14, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.bg.dashed, backgroundColor: theme.colors.bg.app, overflow: 'hidden' },
  soonTile: { height: 30 },
  soonTileOn: { backgroundColor: theme.colors.brand[50], borderColor: theme.colors.brand[200] },
  soonBadge: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', gap: 6 },
  soonTitle: { fontSize: 14, fontWeight: '900', color: theme.colors.text.display },

  walk: { flexDirection: 'row', alignItems: 'center', gap: 6, height: 26 },
  walkText: { fontSize: 11, fontWeight: '700', fontStyle: 'italic', color: theme.colors.text.muted },
}));
