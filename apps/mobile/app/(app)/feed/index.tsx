import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { View, StyleSheet, ScrollView, Pressable, RefreshControl, Linking, ActivityIndicator, useWindowDimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeInRight, FadeOut, ZoomIn, LinearTransition, Extrapolation, interpolate, type SharedValue,
  useSharedValue, useAnimatedStyle, useAnimatedScrollHandler, useAnimatedRef, useDerivedValue, useAnimatedReaction, withSpring,
  useAnimatedProps, withTiming, withDelay, withRepeat, withSequence, Easing,
} from 'react-native-reanimated';
import Svg, { Circle } from 'react-native-svg';
import {
  Radar, Flame, Sparkles, UserPlus, CalendarDays, Phone, Mail, ChevronRight, ScanLine, WifiOff, RotateCw,
  MapPin, Store, AlertCircle, CheckCircle2, Trophy, MessageCircleQuestion, ArrowUpRight,
  Map as MapIcon, Zap, Handshake, Trash2, CloudUpload, PenLine, UsersRound, FolderOpen, Flag, Target, QrCode,
} from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { useRankedFeed } from '../../../src/features/feed';
import { useSession } from '../../../src/providers/SessionProvider';
import { api, errorMessage, type Exhibitor, type ErrorMessageOptions } from '../../../src/lib/api';
import { profileStore } from '../../../src/features/profile';
import { locationLabel } from '../../../src/features/events/BoothMap';
import { ContactAvatar } from '../../../src/features/contacts/components/ContactAvatar';
import { AnimatedCount, SkeletonRows } from '../../../src/features/contacts/components/ListAnimations';
import { PressableScale } from '../../../src/features/onboarding/components';
import { scoreLevel as BAND, isWarmOrHotter } from '../../../src/features/contacts/scoreLevel';
import { AnnouncementsBanner } from '../../../src/features/broadcasts/AnnouncementsBanner';
import { AnnouncementsBell } from '../../../src/features/broadcasts/AnnouncementsMenu';
import { openContact, openContactsPage, useOpenContactsList } from '../../../src/features/contacts/open';

type Filter = 'all' | 'hot' | 'warm' | 'cool' | 'cold' | 'unscored';
const FEED_LIMIT = 500;

// How this screen words a failed request (see errorMessage).
const FEED_ERRORS: ErrorMessageOptions = { timeout: 'That took too long. Please try again.' };

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
}

/* ------------------------------ animated bits ------------------------------ */

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

/** Score ring that draws itself to the real score. */
function ScoreRing({ score, color, delay = 0 }: { score: number; color: string; delay?: number }) {
  const size = 48, stroke = 4, r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const p = useSharedValue(0);
  useEffect(() => {
    p.value = 0;
    p.value = withDelay(delay, withTiming(score / 100, { duration: 900, easing: Easing.out(Easing.cubic) }));
  }, [score]);
  const props = useAnimatedProps(() => ({ strokeDashoffset: c * (1 - p.value) }));
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={toneFg('#f1f5f9')} strokeWidth={stroke} fill="none" />
        <AnimatedCircle
          cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none" strokeLinecap="round"
          strokeDasharray={`${c} ${c}`} animatedProps={props} transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </Svg>
      <Text style={styles.ringText}>{score}</Text>
    </View>
  );
}

function RadarPulse() {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withTiming(1, { duration: 2200, easing: Easing.out(Easing.quad) }), -1);
  }, []);
  const ring = useAnimatedStyle(() => ({ opacity: 0.5 * (1 - t.value), transform: [{ scale: 1 + t.value * 0.9 }] }));
  return (
    <View style={styles.radarWrap}>
      <Animated.View style={[styles.radarRing, ring]} />
      <View style={styles.radarCore}><Radar size={18} color="white" /></View>
    </View>
  );
}

function StatTile({ icon, value, label, tint, index, active, onPress }: {
  icon: React.ReactNode; value: number | null; label: string; tint: string; index: number; active?: boolean; onPress?: () => void;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(80 + index * 60).duration(260)} style={{ flex: 1 }}>
      <PressableScale onPress={onPress} disabled={!onPress} scaleTo={0.94} style={[styles.statTile, active && styles.statTileOn]}>
        <View style={[styles.statIcon, { backgroundColor: tint }]}>{icon}</View>
        {value == null ? <Text style={styles.statValue}>—</Text> : <AnimatedCount value={value} style={styles.statValue} />}
        <Text style={styles.statLabel} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.8}>{label}</Text>
      </PressableScale>
    </Animated.View>
  );
}

/* ------------------------------ shortcut rails ------------------------------ */

const RAIL_TILE_W = 92;
const RAIL_GAP = 10;
const RAIL_STEP = RAIL_TILE_W + RAIL_GAP;
const RAIL_PANEL_PAD = 8;   // inner padding of each group's tinted panel
const RAIL_GROUP_GAP = 12;  // space between the two group panels

type Shortcut = { key: string; label: string; hint?: string | undefined; icon: React.ReactNode; tint: string; onPress: () => void };
type ShortcutGroupDef = {
  key: string; title: string; prefix?: string; tone: 'event' | 'quick'; items: Shortcut[];
  icon: React.ReactNode;                                              // small icon in the caption
  glyph: React.ComponentType<{ size?: number; color?: string }>;     // icon in the toggle
};

/**
 * One tile in the shortcut row. Springs in from the right on first show (staggered), and eases down
 * in size and opacity as it scrolls off the left edge so the row feels like it's moving in depth.
 * `left` is the tile's x inside the scroll content.
 */
function RailTile({ s, index, left: tileLeft, x, delay }: { s: Shortcut; index: number; left: number; x: SharedValue<number>; delay: number }) {
  const depth = useAnimatedStyle(() => {
    const left = tileLeft - x.value; // tile's left edge relative to the visible row
    return {
      opacity: interpolate(left, [-RAIL_TILE_W, 0], [0.35, 1], Extrapolation.CLAMP),
      transform: [{ scale: interpolate(left, [-RAIL_TILE_W, 0], [0.86, 1], Extrapolation.CLAMP) }],
    };
  });
  return (
    <Animated.View entering={FadeInRight.delay(delay + index * 55).springify().damping(16).stiffness(140)}>
      <Animated.View style={[{ width: RAIL_TILE_W }, depth]}>
        <PressableScale onPress={s.onPress} scaleTo={0.92} style={styles.shortcutTile} accessibilityRole="button"
          accessibilityLabel={s.hint ? `${s.label}. ${s.hint}` : s.label}>
          <View style={[styles.statIcon, { backgroundColor: s.tint }]}>{s.icon}</View>
          <Text style={styles.shortcutLabel} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.8}>{s.label}</Text>
          {s.hint ? <Text style={styles.shortcutHint} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.85}>{s.hint}</Text> : null}
        </PressableScale>
      </Animated.View>
    </Animated.View>
  );
}

const TOGGLE_SEG = 38;   // width of one icon segment in the toggle
const TOGGLE_PAD = 3;

/**
 * Caption for the group that's on screen. Each group's caption sits in the same spot and they
 * crossfade + slide as the toggle moves. Only this caption carries the event name, and it takes
 * whatever width the toggle leaves, so a long name just ends in "…" and can't push anything out.
 */
function RailCaption({ groups, progress }: { groups: ShortcutGroupDef[]; progress: SharedValue<number> }) {
  return (
    <View style={styles.railCaption}>
      {groups.map((g, gi) => <RailCaptionLine key={g.key} g={g} gi={gi} progress={progress} />)}
    </View>
  );
}

function RailCaptionLine({ g, gi, progress }: { g: ShortcutGroupDef; gi: number; progress: SharedValue<number> }) {
  const style = useAnimatedStyle(() => {
    const d = progress.value - gi; // 0 when this group is on screen
    return {
      opacity: interpolate(Math.abs(d), [0, 0.6], [1, 0], Extrapolation.CLAMP),
      transform: [{ translateY: interpolate(d, [-1, 0, 1], [10, 0, -10], Extrapolation.CLAMP) }],
    };
  });
  const event = g.tone === 'event';
  return (
    <Animated.View style={[styles.railCaptionLine, style]} pointerEvents="none">
      {g.icon}
      {g.prefix ? <Text style={styles.railCaptionPrefix}>{g.prefix}</Text> : null}
      <Text style={[styles.railCaptionText, event && styles.railCaptionTextEvent]} numberOfLines={1} ellipsizeMode="tail">{g.title}</Text>
      <Text style={styles.railCaptionCount}>· {g.items.length}</Text>
    </Animated.View>
  );
}

// Thumb edge springs: the edge moving toward the new icon leads (snappy), the other trails (soft).
const EDGE_LEAD = { damping: 20, stiffness: 340, mass: 0.6 };
const EDGE_TRAIL = { damping: 17, stiffness: 130, mass: 0.9 };

/**
 * Icon-only toggle between the shortcut groups — the same small size whatever the event is called.
 *  - "Worm" thumb: its leading edge shoots to the new icon and the trailing edge catches up, so it
 *    stretches toward where it's going and then snaps back to a circle.
 *  - A ring ripples out from where the thumb lands.
 *  - The new icon pops (overshoot) with a small wiggle; icons squish when pressed.
 * It follows the row as you swipe, and tapping an icon glides the row to that group.
 */
function RailToggle({ groups, active, progress, onSelect }: {
  groups: ShortcutGroupDef[]; active: SharedValue<number>; progress: SharedValue<number>; onSelect: (gi: number) => void;
}) {
  const left = useSharedValue(0);   // thumb's left edge, in icon slots
  const right = useSharedValue(0);  // thumb's right edge, in icon slots
  const ripple = useSharedValue(1); // 0 → 1 per change; 1 = idle (invisible)

  useAnimatedReaction(() => active.value, (now, prev) => {
    if (prev == null || prev === now) { left.value = now; right.value = now; return; }
    const forward = now > prev;
    right.value = withSpring(now, forward ? EDGE_LEAD : EDGE_TRAIL);
    left.value = withSpring(now, forward ? EDGE_TRAIL : EDGE_LEAD);
    ripple.value = 0;
    ripple.value = withDelay(90, withTiming(1, { duration: 560, easing: Easing.out(Easing.cubic) }));
  });

  const thumb = useAnimatedStyle(() => {
    const span = Math.max(right.value - left.value, 0);
    return {
      width: TOGGLE_SEG + span * TOGGLE_SEG,
      transform: [{ translateX: left.value * TOGGLE_SEG }, { scaleY: 1 - 0.14 * Math.min(span, 1) }],
    };
  });
  const ring = useAnimatedStyle(() => ({
    opacity: ripple.value >= 1 ? 0 : 0.5 * (1 - ripple.value),
    transform: [{ translateX: active.value * TOGGLE_SEG }, { scale: 1 + ripple.value * 0.75 }],
  }));

  return (
    <View style={[styles.railToggle, { width: groups.length * TOGGLE_SEG + TOGGLE_PAD * 2 }]} accessibilityRole="tablist">
      <Animated.View pointerEvents="none" style={[styles.railToggleRing, ring]} />
      <Animated.View pointerEvents="none" style={[styles.railToggleThumb, thumb]} />
      {groups.map((g, gi) => <RailToggleIcon key={g.key} g={g} gi={gi} active={active} progress={progress} onPress={() => onSelect(gi)} />)}
    </View>
  );
}

function RailToggleIcon({ g, gi, active, progress, onPress }: {
  g: ShortcutGroupDef; gi: number; active: SharedValue<number>; progress: SharedValue<number>; onPress: () => void;
}) {
  const Glyph = g.glyph;
  const pop = useSharedValue(1);
  const wiggle = useSharedValue(0);

  // Becoming the selected icon: overshoot-and-settle pop plus a quick wiggle.
  useAnimatedReaction(() => active.value === gi, (isOn, wasOn) => {
    if (!isOn || wasOn == null || wasOn) return;
    pop.value = withSequence(withTiming(1.3, { duration: 130, easing: Easing.out(Easing.quad) }), withSpring(1, { damping: 7, stiffness: 220 }));
    wiggle.value = withSequence(withTiming(-16, { duration: 90 }), withTiming(12, { duration: 110 }), withSpring(0, { damping: 6, stiffness: 240 }));
  });

  const on = useAnimatedStyle(() => {
    const t = interpolate(Math.abs(progress.value - gi), [0, 1], [1, 0], Extrapolation.CLAMP);
    return { opacity: t, transform: [{ scale: (0.6 + 0.4 * t) * pop.value }, { rotate: `${wiggle.value}deg` }] };
  });
  const off = useAnimatedStyle(() => {
    const t = interpolate(Math.abs(progress.value - gi), [0, 1], [0, 1], Extrapolation.CLAMP);
    return { opacity: t, transform: [{ scale: 0.85 + 0.15 * t }] };
  });
  const label = `${g.prefix ? `${g.prefix} ` : ''}${g.title}: show its ${g.items.length} shortcuts`;
  return (
    <PressableScale onPress={onPress} scaleTo={0.8} hitSlop={6} style={styles.railToggleSeg} accessibilityRole="tab" accessibilityLabel={label}>
      <Animated.View style={[styles.railToggleIcon, off]}><Glyph size={16} color={theme.colors.text.muted} /></Animated.View>
      <Animated.View style={[styles.railToggleIcon, on]}><Glyph size={16} color="#ffffff" /></Animated.View>
    </PressableScale>
  );
}

/**
 * Every shortcut in one horizontally scrolling row, with a fixed header above it.
 *  - Each group sits in its own tinted panel (event = brand colour, quick access = dashed grey).
 *  - The header follows the scroll: the group at the left of the row is highlighted, and a pill
 *    springs under it (sized to each button, so long and short labels both fit). Tapping a header
 *    button glides the row to that group's tiles.
 */
function ShortcutRail({ groups, delay = 0 }: { groups: ShortcutGroupDef[]; delay?: number }) {
  const ref = useAnimatedRef<Animated.ScrollView>();
  const x = useSharedValue(0);
  const maxX = useSharedValue(0);          // furthest the row can scroll
  const viewW = React.useRef(0);

  // Where each panel and tile starts inside the row (panels are a fixed size, so this is exact).
  const { groupStarts, panels, contentW } = useMemo(() => {
    const starts: number[] = [];
    let cursor = 0;
    const list = groups.map((g, gi) => {
      if (gi > 0) cursor += RAIL_GROUP_GAP;
      starts.push(cursor);
      const tiles = g.items.map((s, ti) => ({ s, left: cursor + RAIL_PANEL_PAD + ti * RAIL_STEP }));
      cursor += RAIL_PANEL_PAD * 2 + g.items.length * RAIL_STEP - RAIL_GAP;
      return { g, tiles };
    });
    return { groupStarts: starts, panels: list, contentW: cursor };
  }, [groups]);

  // Snap so a tile lines up with the screen padding; a group's first tile snaps to its panel's edge.
  const snaps = useMemo(() => panels.flatMap(({ tiles }) =>
    tiles.map((t, i) => (i === 0 ? t.left - RAIL_PANEL_PAD : t.left))), [panels]);

  // Which group is at the left of the row; at the very end, the last group (it may never reach
  // the left edge on a wide screen).
  const active = useDerivedValue(() => {
    if (maxX.value > 0 && x.value >= maxX.value - 2) return groupStarts.length - 1;
    let idx = 0;
    for (let i = 0; i < groupStarts.length; i++) if (x.value + 40 >= groupStarts[i]!) idx = i;
    return idx;
  });

  const onScroll = useAnimatedScrollHandler((e) => { x.value = e.contentOffset.x; });

  const updateMax = () => { maxX.value = Math.max(contentW + 32 - viewW.current, 0); };

  // Header tap: glide to that group's tiles (clamped to the end of the row).
  const goTo = (gi: number) => {
    const target = Math.min(groupStarts[gi]!, maxX.value || groupStarts[gi]!);
    (ref.current as any)?.scrollTo?.({ x: target, animated: true });
  };

  // One springy value (0 = first group, 1 = second…) drives the toggle thumb, icons and caption.
  const progress = useSharedValue(0);
  useAnimatedReaction(() => active.value, (now, prev) => {
    if (now !== prev) progress.value = withSpring(now, { damping: 15, stiffness: 170, mass: 0.9 });
  });

  return (
    <Animated.View entering={FadeInDown.delay(delay).duration(260)} style={styles.railWrap}>
      {/* Caption of the group on screen (takes the free width) + compact icon toggle */}
      <View style={styles.railHead}>
        <RailCaption groups={groups} progress={progress} />
        <RailToggle groups={groups} active={active} progress={progress} onSelect={goTo} />
      </View>

      <Animated.ScrollView ref={ref} horizontal showsHorizontalScrollIndicator={false} onScroll={onScroll} scrollEventThrottle={16}
        decelerationRate="fast" snapToOffsets={snaps}
        onLayout={(e) => { viewW.current = e.nativeEvent.layout.width; updateMax(); }}
        style={styles.railScroll} contentContainerStyle={styles.railContent}>
        {panels.map(({ g, tiles }, gi) => (
          <Animated.View key={g.key} entering={FadeInRight.delay(delay + gi * 120).duration(320)}
            style={[styles.railPanel, g.tone === 'event' ? styles.railPanelEvent : styles.railPanelQuick, gi > 0 && { marginLeft: RAIL_GROUP_GAP }]}>
            <View style={styles.railTiles}>
              {tiles.map((t, ti) => <RailTile key={t.s.key} s={t.s} index={ti} left={t.left} x={x} delay={delay + 60 + gi * 120} />)}
            </View>
          </Animated.View>
        ))}
      </Animated.ScrollView>
    </Animated.View>
  );
}

/* ---------------------------------- cards ---------------------------------- */

function LeadCard({ c, rank, index, onOpen, onScore, scoring }: {
  c: Contact; rank: number | null; index: number; onOpen: () => void; onScore: () => void; scoring: boolean;
}) {
  const band = BAND(c.score);
  const name = [c.firstName, c.lastName].filter(Boolean).join(' ') || 'Unnamed contact';
  const sub = [c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No title or company yet';
  const phone = (c.phone || '').replace(/[^\d+]/g, '');
  return (
    <Animated.View
      entering={FadeInDown.delay(Math.min(index, 10) * 55).duration(260)}
      exiting={FadeOut.duration(140)}
      layout={LinearTransition.duration(260)}
    >
      <PressableScale onPress={onOpen} scaleTo={0.98} style={[styles.card, rank === 1 && styles.cardTop]} accessibilityLabel={`Open ${name}`}>
        {band ? <View style={[styles.cardAccent, { backgroundColor: band.ring }]} /> : null}
        <View>
          <ContactAvatar firstName={c.firstName} lastName={c.lastName} photoUrl={c.photoUrl} size={48} />
          {rank && rank <= 3 ? (
            <Animated.View entering={ZoomIn.delay(250 + index * 55).duration(260)} style={[styles.rankBadge, rank === 1 && { backgroundColor: toneBg('#f59e0b') }]}>
              <Text style={styles.rankText}>{rank}</Text>
            </Animated.View>
          ) : null}
        </View>
        <View style={{ flex: 1, minWidth: 0 }}>
          <View style={styles.nameRow}>
            <Text style={styles.name} numberOfLines={1}>{name}</Text>
            {band ? (
              <View style={[styles.bandChip, { backgroundColor: band.bg }]}>
                <Text style={[styles.bandText, { color: band.fg }]}>{band.label}</Text>
              </View>
            ) : null}
          </View>
          <Text style={styles.sub} numberOfLines={1}>{sub}</Text>
          <View style={styles.cardActions}>
            {c.eventId ? (
              <View style={styles.metaChip}><CalendarDays size={11} color={theme.colors.text.muted} /><Text style={styles.metaText} numberOfLines={1}>{c.eventId}</Text></View>
            ) : null}
            {phone ? (
              <Pressable hitSlop={6} onPress={() => Linking.openURL(`tel:${phone}`)} style={styles.miniAction} accessibilityLabel={`Call ${name}`}>
                <Phone size={13} color={theme.colors.brand[700]} />
              </Pressable>
            ) : null}
            {c.email ? (
              <Pressable hitSlop={6} onPress={() => Linking.openURL(`mailto:${c.email}`)} style={styles.miniAction} accessibilityLabel={`Email ${name}`}>
                <Mail size={13} color={theme.colors.brand[700]} />
              </Pressable>
            ) : null}
          </View>
        </View>
        {c.score != null && band ? (
          <ScoreRing score={c.score} color={band.ring} delay={150 + Math.min(index, 10) * 55} />
        ) : (
          <PressableScale onPress={onScore} disabled={scoring} scaleTo={0.9} style={styles.scoreBtn} accessibilityLabel={`Score ${name}`}>
            {scoring ? <ActivityIndicator size="small" color={theme.colors.brand[600]} /> : <Sparkles size={15} color={theme.colors.brand[600]} />}
            <Text style={styles.scoreBtnText}>{scoring ? '' : 'Score'}</Text>
          </PressableScale>
        )}
      </PressableScale>
    </Animated.View>
  );
}

/* --------------------------------- screen --------------------------------- */

export default function FeedScreen() {
  const router = useRouter();
  // Narrow phones (most Androids are 360–393dp wide) get a tighter header so nothing runs off-screen.
  const { width } = useWindowDimensions();
  const compact = width < 380;
  const openList = useOpenContactsList(); // Contacts list on a view, closing any contact left open there
  const { userId, userName } = useSession();
  // The server sorts by score, so the ranking starts from your best leads across ALL contacts
  // (it used to rank only the newest 200 it had downloaded).
  const { contacts, loading, error, reload } = useRankedFeed({ limit: FEED_LIMIT, sort: 'score' });
  const [filter, setFilter] = useState<Filter>('all');
  const [dashboard, setDashboard] = useState<any>(null);
  const [activeEvent, setActiveEvent] = useState<string | null>(null);
  const [exhibitors, setExhibitors] = useState<Exhibitor[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [scoringId, setScoringId] = useState<string | null>(null);
  const [scoringAll, setScoringAll] = useState(false);
  const [notice, setNotice] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);

  const loadSide = useCallback(async () => {
    if (!userId) return;
    const [d, p] = await Promise.allSettled([api.getDashboard(userId), profileStore.load(userId)]);
    if (d.status === 'fulfilled') setDashboard(d.value);
    const ev = p.status === 'fulfilled' ? (p.value.current_event_name || null) : null;
    setActiveEvent(ev);
    if (ev) {
      try {
        const list: any = await api.listExhibitors({ event: ev, limit: 6 });
        setExhibitors(Array.isArray(list) ? list : []);
      } catch { setExhibitors([]); }
    } else {
      setExhibitors([]);
    }
  }, [userId]);

  // Fresh every time the tab is opened (new scans, scores, edits).
  const [first, setFirst] = useState(true);
  const [announceKey, setAnnounceKey] = useState(0); // reloads team announcements
  useFocusEffect(useCallback(() => {
    loadSide();
    setAnnounceKey((k) => k + 1);
    if (first) { setFirst(false); return; }
    reload();
  }, [loadSide, reload, first]));

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 3200);
    return () => clearTimeout(t);
  }, [notice]);

  const onRefresh = async () => {
    setRefreshing(true);
    reload();
    setAnnounceKey((k) => k + 1);
    await loadSide();
    setRefreshing(false);
  };

  const counts = useMemo(() => {
    const k = { all: contacts.length, hot: 0, warm: 0, cool: 0, cold: 0, unscored: 0 };
    contacts.forEach((c) => { const b = BAND(c.score); if (b) k[b.key] += 1; else k.unscored += 1; });
    return k;
  }, [contacts]);

  // Tiles use the same numbers as the filter pills (the dashboard's AI "hot" label could disagree
  // with score ≥ 85). Only when the list is capped at 200 do the server totals take over.
  const listComplete = contacts.length < FEED_LIMIT;
  const hotCount = loading && !contacts.length ? null : listComplete ? counts.hot : (dashboard?.by_temperature?.hot ?? counts.hot);
  const toScoreCount = loading && !contacts.length ? null : listComplete ? counts.unscored : (dashboard?.unscored_contacts ?? counts.unscored);

  // "Ask AI" quick questions, built from what's really in your leads (open the Ask tab with it).
  const askPrompts = useMemo(() => {
    const list: string[] = [];
    if (counts.hot > 0) list.push(`Who of my ${counts.hot} hot lead${counts.hot === 1 ? '' : 's'} should I follow up with first?`);
    else if (contacts.length) list.push('Who should I follow up with first, and why?');
    if (activeEvent) list.push(`Who should I meet next at ${activeEvent}?`);
    if (counts.unscored > 0 && list.length < 3) list.push(`Which of my ${counts.unscored} unscored leads look most promising?`);
    if (contacts.length && list.length < 3) list.push('Summarize the people I met today');
    return list.slice(0, 3);
  }, [counts.hot, counts.unscored, contacts.length, activeEvent]);

  const visible = useMemo(() => contacts.filter((c) => {
    if (filter === 'all') return true;
    const b = BAND(c.score);
    return filter === 'unscored' ? !b : b?.key === filter;
  }), [contacts, filter]);

  const scoreOne = async (c: Contact) => {
    if (scoringId || scoringAll) return;
    setScoringId(c.id);
    try {
      const res: any = await api.scoreContact(c.id);
      const s = res?.score ?? res?.lead_score;
      const lvl = BAND(typeof s === 'number' ? s : null);
      setNotice({
        kind: 'ok',
        text: res?.unchanged
          ? `Nothing changed — ${c.firstName || 'this contact'} stays at ${s}`
          : `${c.firstName || 'Contact'} scored ${s ?? ''}${lvl ? ` · ${lvl.name}` : ''}`.trim(),
      });
      reload();
      loadSide();
    } catch (e: any) {
      setNotice({ kind: 'error', text: errorMessage(e, "Couldn't score this contact.", FEED_ERRORS) });
    } finally {
      setScoringId(null);
    }
  };

  const scoreAll = async () => {
    if (scoringAll) return;
    setScoringAll(true);
    try {
      const res = await api.scoreAllContacts();
      setNotice({
        kind: res.errors ? 'error' : 'ok',
        text: `Scored ${res.scoredCount}${res.errors ? ` · ${res.errors} failed` : ''}${res.pending ? ` · ${res.pending} still to go` : ''}`,
      });
      reload();
      loadSide();
    } catch (e: any) {
      setNotice({ kind: 'error', text: errorMessage(e, "Couldn't score contacts right now.", FEED_ERRORS) });
    } finally {
      setScoringAll(false);
    }
  };

  const firstName = (userName || '').split(' ')[0];
  const FILTERS: Array<{ key: Filter; label: string }> = [
    { key: 'all', label: 'All' },
    { key: 'hot', label: 'Hot' },
    { key: 'warm', label: 'Warm' },
    { key: 'cool', label: 'Cool' },
    { key: 'cold', label: 'Cold' },
    { key: 'unscored', label: 'Not scored' },
  ];

  // Quick access to screens that aren't tabs. Booth screens live on the active event's page, so
  // without an active event they open Me, where you pick one.
  const openMe = (open: 'event' | 'targeting' | 'qr') =>
    router.navigate({ pathname: '/me', params: { open, t: String(Date.now()) } } as any);
  const openEvent = (section: 'booths' | 'map' | 'met' | 'seats' | 'files' | 'goals') => activeEvent
    ? router.push({ pathname: '/events/[eventId]', params: { eventId: activeEvent, section, t: String(Date.now()) } } as any)
    : openMe('event');
  const noEvent = activeEvent ? undefined : 'Pick an event';
  const EVENT_SHORTCUTS = [
    { key: 'booths', label: 'Booths', hint: noEvent, icon: <Store size={15} color={theme.colors.brand[700]} />, tint: theme.colors.brand[50], onPress: () => openEvent('booths') },
    { key: 'map', label: 'Floor map', hint: noEvent, icon: <MapIcon size={15} color={toneFg('#7c3aed')} />, tint: toneBg('#f5f3ff'), onPress: () => openEvent('map') },
    { key: 'met', label: 'Booths met', hint: noEvent, icon: <Handshake size={15} color={toneFg('#0369a1')} />, tint: toneBg('#f0f9ff'), onPress: () => openEvent('met') },
    { key: 'seats', label: 'Team seats', hint: noEvent, icon: <UsersRound size={15} color={toneFg('#0f766e')} />, tint: toneBg('#f0fdfa'), onPress: () => openEvent('seats') },
    { key: 'files', label: 'Event files', hint: noEvent, icon: <FolderOpen size={15} color={toneFg('#b45309')} />, tint: toneBg('#fffbeb'), onPress: () => openEvent('files') },
    { key: 'goals', label: 'Event goals', hint: noEvent, icon: <Flag size={15} color={toneFg('#dc2626')} />, tint: toneBg('#fef2f2'), onPress: () => openEvent('goals') },
  ];
  const QUICK_SHORTCUTS = [
    { key: 'new', label: 'Add contact', icon: <PenLine size={15} color={theme.colors.semantic.successText} />, tint: theme.colors.semantic.successBg, onPress: () => openContactsPage(router, 'new', '/feed') },
    { key: 'event', label: 'Switch event', icon: <CalendarDays size={15} color={theme.colors.brand[700]} />, tint: theme.colors.brand[50], onPress: () => openMe('event') },
    { key: 'targeting', label: 'Targeting', icon: <Target size={15} color={toneFg('#7c3aed')} />, tint: toneBg('#f5f3ff'), onPress: () => openMe('targeting') },
    { key: 'qr', label: 'My QR card', icon: <QrCode size={15} color={toneFg('#0369a1')} />, tint: toneBg('#f0f9ff'), onPress: () => openMe('qr') },
    { key: 'deleted', label: 'Deleted', icon: <Trash2 size={15} color={toneFg('#dc2626')} />, tint: toneBg('#fef2f2'), onPress: () => openContactsPage(router, 'deleted', '/feed') },
    { key: 'sync', label: 'Sync', icon: <CloudUpload size={15} color={toneFg('#b45309')} />, tint: toneBg('#fffbeb'), onPress: () => router.push('/me/sync' as any) },
  ];

  const empty = !loading && contacts.length === 0;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(260)} style={styles.header}>
        <RadarPulse />
        <View style={{ flex: 1, minWidth: 0 }}>
          <Text style={styles.hello} numberOfLines={1}>{greeting()}{firstName ? `, ${firstName}` : ''}</Text>
          <Text style={[styles.headerTitle, compact && styles.headerTitleCompact]} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.8}>Who to meet</Text>
        </View>
        {activeEvent ? (
          <Animated.View entering={ZoomIn.duration(260)} style={{ flexShrink: 1, maxWidth: Math.round(width * 0.38) }}>
            <PressableScale onPress={() => router.push(`/events/${encodeURIComponent(activeEvent)}` as any)} scaleTo={0.94} style={[styles.eventChip, { maxWidth: '100%' }]} accessibilityLabel={`Active event ${activeEvent}`}>
              <MapPin size={12} color={theme.colors.brand[700]} />
              <Text style={styles.eventChipText} numberOfLines={1}>{activeEvent}</Text>
            </PressableScale>
          </Animated.View>
        ) : null}
        <AnnouncementsBell />
      </Animated.View>

      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl progressBackgroundColor={theme.colors.bg.surface} refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.brand[600]} colors={[theme.colors.brand[600]]} />}
      >
        {/* Team announcements from the admin console (GET /api/broadcasts/active) */}
        <AnnouncementsBanner refreshKey={announceKey} />

        {/* Today at a glance (GET /dashboard) */}
        <View style={styles.statsRow}>
          <StatTile index={0} icon={<Flame size={15} color={toneFg('#dc2626')} />} tint={toneBg('#fef2f2')} value={hotCount}
            label="Hot leads" active={filter === 'hot'} onPress={() => setFilter(filter === 'hot' ? 'all' : 'hot')} />
          <StatTile index={1} icon={<UserPlus size={15} color={theme.colors.brand[700]} />} tint={theme.colors.brand[50]} value={dashboard?.added_today ?? null}
            label="Met today" onPress={() => openList('today')} />
          <StatTile index={2} icon={<Sparkles size={15} color={theme.colors.semantic.successText} />} tint={theme.colors.semantic.successBg}
            value={toScoreCount} label="To score"
            active={filter === 'unscored'} onPress={() => setFilter(filter === 'unscored' ? 'all' : 'unscored')} />
        </View>

        {/* Shortcuts to screens outside the tab bar */}
        {/* Quick access comes first, so the row (and the toggle) start on it. */}
        <ShortcutRail delay={100} groups={[
          { key: 'quick', tone: 'quick', title: 'Quick access',
            icon: <Zap size={13} color={theme.colors.text.muted} />, glyph: Zap, items: QUICK_SHORTCUTS },
          { key: 'event', tone: 'event', ...(activeEvent ? { prefix: 'At', title: activeEvent } : { title: 'Your event' }),
            icon: <MapPin size={13} color={theme.colors.brand[700]} />, glyph: MapPin, items: EVENT_SHORTCUTS },
        ]} />

        {error ? (
          <Animated.View entering={FadeInDown.duration(260)}>
            <PressableScale onPress={onRefresh} scaleTo={0.98} style={styles.offline}>
              <WifiOff size={16} color={theme.colors.semantic.danger} />
              <Text style={styles.offlineText}>{error} {contacts.length ? 'Showing contacts saved on this phone.' : ''} Tap to retry.</Text>
              <RotateCw size={15} color={theme.colors.semantic.danger} />
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Ask AI — quick questions about these leads; opens the Ask AI tab with it already asked.
            (Replaces a floating chat bar that duplicated the Ask tab, had a dead voice button and
            fought the keyboard.) */}
        {askPrompts.length ? (
          <Animated.View entering={FadeInDown.delay(160).duration(260)} style={styles.askCard}>
            <View style={styles.askHead}>
              <View style={styles.askIcon}><Sparkles size={15} color="white" /></View>
              <Text style={styles.askTitle}>Ask AI about your leads</Text>
              <Pressable onPress={() => router.push('/(app)/ask' as any)} hitSlop={8} accessibilityLabel="Open Ask AI">
                <ArrowUpRight size={18} color={theme.colors.brand[700]} />
              </Pressable>
            </View>
            {askPrompts.map((q) => (
              <PressableScale key={q} onPress={() => router.push({ pathname: '/ask', params: { q, n: String(Date.now()) } } as any)} scaleTo={0.98} style={styles.askPrompt}>
                <MessageCircleQuestion size={15} color={theme.colors.brand[600]} />
                <Text style={styles.askPromptText} numberOfLines={2}>{q}</Text>
                <ChevronRight size={15} color={theme.colors.text.muted} />
              </PressableScale>
            ))}
          </Animated.View>
        ) : null}

        {/* Score the unscored in one go (POST /api/contacts/score_all) */}
        {counts.unscored > 0 && !empty ? (
          <Animated.View entering={FadeInDown.delay(200).duration(260)}>
            <PressableScale onPress={scoreAll} disabled={scoringAll} scaleTo={0.98} style={styles.scoreAllCard}>
              <View style={styles.scoreAllIcon}>
                {scoringAll ? <ActivityIndicator size="small" color="white" /> : <Sparkles size={17} color="white" />}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.scoreAllTitle}>{scoringAll ? 'Scoring your leads…' : `${counts.unscored} lead${counts.unscored === 1 ? '' : 's'} not scored yet`}</Text>
                <Text style={styles.scoreAllSub}>{scoringAll ? 'The AI is checking each one against your targeting' : 'Score them with AI to see who to meet first'}</Text>
              </View>
              {!scoringAll ? <ChevronRight size={17} color={theme.colors.brand[700]} /> : null}
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Exhibitors at the active event (imported directory), above your ranked contacts */}
        {exhibitors.length > 0 ? (
          <Animated.View entering={FadeInDown.delay(200).duration(260)} style={{ marginBottom: 14 }}>
            <View style={styles.sectionHead}>
              <Store size={13} color={theme.colors.brand[700]} />
              <Text style={styles.sectionLabel} numberOfLines={1}>EXHIBITORS AT {activeEvent?.toUpperCase()}</Text>
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10, paddingRight: 8 }}>
              {exhibitors.map((x, i) => (
                <Animated.View key={x.id} entering={FadeInDown.delay(240 + i * 50).duration(260)} style={styles.exCard}>
                  <Text style={styles.exName} numberOfLines={2}>{x.name}</Text>
                  {x.category ? <Text style={styles.exCat} numberOfLines={1}>{x.category}</Text> : null}
                  {x.booth || x.hall ? (
                    <View style={styles.exBooth}>
                      <MapPin size={11} color={theme.colors.brand[700]} />
                      <Text style={styles.exBoothText}>{locationLabel(x)}</Text>
                    </View>
                  ) : null}
                </Animated.View>
              ))}
            </ScrollView>
          </Animated.View>
        ) : null}

        {/* Filters */}
        {!empty ? (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filters}>
            {FILTERS.map((f, i) => {
              const on = filter === f.key;
              const n = counts[f.key];
              return (
                <Animated.View key={f.key} entering={FadeInDown.delay(120 + i * 40).duration(260)}>
                  <PressableScale onPress={() => setFilter(f.key)} scaleTo={0.92} style={[styles.filterChip, on && styles.filterChipOn]}
                    accessibilityRole="button" accessibilityState={{ selected: on }}>
                    <Text style={[styles.filterText, on && { color: 'white' }]}>{f.label}</Text>
                    <View style={[styles.filterCount, on && { backgroundColor: 'rgba(255,255,255,0.25)' }]}>
                      <Text style={[styles.filterCountText, on && { color: 'white' }]}>{n}</Text>
                    </View>
                  </PressableScale>
                </Animated.View>
              );
            })}
          </ScrollView>
        ) : null}

        {/* Ranked leads: highest score first, then most recently met */}
        {loading && contacts.length === 0 ? (
          <SkeletonRows count={5} />
        ) : empty ? (
          <Animated.View entering={FadeInDown.duration(260)} style={styles.empty}>
            <View style={styles.emptyIcon}><ScanLine size={26} color={theme.colors.brand[600]} /></View>
            <Text style={styles.emptyTitle}>No leads yet</Text>
            <Text style={styles.emptyText}>Scan a badge or business card and it will show up here, ranked by how well it fits your targets.</Text>
            <PressableScale onPress={() => router.push('/(app)/scan' as any)} style={styles.emptyCta} leadingIcon={<ScanLine size={16} color="white" />}>
              <Text style={styles.emptyCtaText}>Scan a card</Text>
            </PressableScale>
          </Animated.View>
        ) : visible.length === 0 ? (
          <Animated.View key={filter} entering={FadeIn} style={styles.emptySmall}>
            <CheckCircle2 size={18} color={theme.colors.text.muted} />
            <Text style={styles.emptyText}>No {FILTERS.find((f) => f.key === filter)?.label.toLowerCase()} leads right now.</Text>
          </Animated.View>
        ) : (
          <View key={filter}>
            {filter === 'all' && visible[0]?.score != null ? (
              // Only call it "follow up first" when the best lead is actually warm or hot.
              <Animated.View entering={FadeIn} style={styles.sectionHead}>
                <Trophy size={13} color={toneFg('#b45309')} />
                <Text style={[styles.sectionLabel, { flexShrink: 1 }]} numberOfLines={1}>{isWarmOrHotter(visible[0]) ? 'FOLLOW UP FIRST' : 'RANKED BY SCORE · NO WARM LEADS YET'}</Text>
              </Animated.View>
            ) : null}
            {visible.slice(0, 60).map((c, i) => (
              <LeadCard
                key={c.id}
                c={c}
                index={i}
                rank={filter === 'all' && c.score != null ? i + 1 : null}
                scoring={scoringId === c.id}
                onOpen={() => openContact(router, c.id, '/feed')}
                onScore={() => scoreOne(c)}
              />
            ))}
            {visible.length > 60 ? (
              <Pressable onPress={() => openList(filter === 'hot' ? 'hot' : filter === 'unscored' ? 'unscored' : 'all')} style={styles.more}>
                <Text style={styles.moreText}>See all {visible.length} in Contacts</Text>
                <ChevronRight size={15} color={theme.colors.brand[600]} />
              </Pressable>
            ) : null}
          </View>
        )}
      </ScrollView>

      {notice ? (
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut} pointerEvents="none"
          style={[styles.toast, notice.kind === 'error' && styles.toastError]}>
          {notice.kind === 'ok' ? <CheckCircle2 size={16} color="white" /> : <AlertCircle size={16} color="white" />}
          <Text style={styles.toastText}>{notice.text}</Text>
        </Animated.View>
      ) : null}

    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  container: { flex: 1, backgroundColor: theme.colors.bg.app },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingTop: 8, paddingBottom: 10 },
  radarWrap: { width: 42, height: 42, alignItems: 'center', justifyContent: 'center' },
  radarRing: { position: 'absolute', width: 42, height: 42, borderRadius: 21, backgroundColor: theme.colors.brand[400] },
  radarCore: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  hello: { fontSize: 13, color: theme.colors.text.muted, fontWeight: '600' },
  headerTitle: { fontSize: 26, lineHeight: 32, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.5 },
  headerTitleCompact: { fontSize: 22, lineHeight: 28 },
  eventChip: { flexDirection: 'row', alignItems: 'center', gap: 4, maxWidth: 140, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  eventChipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700], flexShrink: 1 },
  listContent: { paddingHorizontal: 16, paddingTop: 4, paddingBottom: 32 },

  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  statTile: { alignItems: 'center', gap: 4, backgroundColor: toneBg('#ffffff'), borderRadius: 18, paddingVertical: 12, paddingHorizontal: 4, borderWidth: 1.5, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  statTileOn: { borderColor: theme.colors.brand[400] },
  statIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  statValue: { fontSize: 20, lineHeight: 25, fontWeight: '900', color: theme.colors.text.display },
  statLabel: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },

  railWrap: { marginBottom: 10 },
  // Bleeds to the screen edges so tiles scroll off-screen instead of being clipped at the padding.
  railScroll: { marginHorizontal: -16 },
  railContent: { paddingHorizontal: 16, paddingTop: 2, paddingBottom: 8 },
  // Each group is a tinted panel: the event's screens in brand colour, quick access in neutral grey.
  railPanel: { padding: RAIL_PANEL_PAD, borderRadius: 22, borderWidth: 1 },
  railPanelEvent: { backgroundColor: theme.colors.brand[50], borderColor: theme.colors.brand[100] },
  railPanelQuick: { backgroundColor: theme.colors.bg.surfaceHover, borderColor: theme.colors.bg.divider, borderStyle: 'dashed' },
  // Header above the rail: caption of the group on screen (flexible) + icon toggle (fixed size).
  railHead: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 8 },
  railCaption: { flex: 1, minWidth: 0, height: 22, justifyContent: 'center' },
  railCaptionLine: { position: 'absolute', left: 2, right: 0, flexDirection: 'row', alignItems: 'center', gap: 5 },
  railCaptionPrefix: { flexShrink: 0, fontSize: 13, fontWeight: '600', color: theme.colors.text.muted },
  railCaptionText: { flexShrink: 1, fontSize: 13, fontWeight: '800', color: theme.colors.text.display },
  railCaptionTextEvent: { color: theme.colors.brand[700] },
  railCaptionCount: { flexShrink: 0, fontSize: 12, fontWeight: '700', color: theme.colors.text.muted },
  railToggle: { flexShrink: 0, flexDirection: 'row', padding: TOGGLE_PAD, borderRadius: 999, backgroundColor: theme.colors.bg.surfaceHover, borderWidth: 1, borderColor: theme.colors.bg.divider },
  railToggleThumb: { position: 'absolute', top: TOGGLE_PAD, left: TOGGLE_PAD, width: TOGGLE_SEG, bottom: TOGGLE_PAD, borderRadius: 999, backgroundColor: theme.colors.brand[600], ...theme.shadows.card },
  railToggleSeg: { width: TOGGLE_SEG, height: 30, alignItems: 'center', justifyContent: 'center' },
  railToggleIcon: { position: 'absolute' },
  railToggleRing: { position: 'absolute', top: TOGGLE_PAD, left: TOGGLE_PAD, width: TOGGLE_SEG, bottom: TOGGLE_PAD, borderRadius: 999, borderWidth: 2, borderColor: theme.colors.brand[400] },
  railTiles: { flexDirection: 'row', gap: RAIL_GAP },
  shortcutTile: { alignItems: 'center', gap: 4, backgroundColor: toneBg('#ffffff'), borderRadius: 18, paddingVertical: 12, paddingHorizontal: 6, borderWidth: 1.5, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  shortcutLabel: { fontSize: 12, fontWeight: '800', color: theme.colors.text.display },
  shortcutHint: { fontSize: 10, fontWeight: '600', color: theme.colors.text.muted, marginTop: -2 },

  offline: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 12 },
  offlineText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  scoreAllCard: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 16, marginBottom: 12, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[200] },
  scoreAllIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  scoreAllTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[800] },
  scoreAllSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },

  filters: { gap: 8, paddingBottom: 12, paddingRight: 8 },
  filterChip: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingLeft: 12, paddingRight: 6, paddingVertical: 6, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.bg.divider },
  filterChipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  filterText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display },
  filterCount: { minWidth: 22, height: 20, borderRadius: 10, paddingHorizontal: 6, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  filterCountText: { fontSize: 11, fontWeight: '800', color: theme.colors.text.body },

  sectionHead: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8, paddingHorizontal: 2 },
  sectionLabel: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },

  card: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: toneBg('#ffffff'), padding: 12, borderRadius: 18, borderWidth: 1, borderColor: theme.colors.bg.divider, marginBottom: 10, overflow: 'hidden', ...theme.shadows.card },
  cardTop: { borderColor: toneBg('#fcd34d') },
  cardAccent: { position: 'absolute', left: 0, top: 12, bottom: 12, width: 3, borderTopRightRadius: 3, borderBottomRightRadius: 3 },
  rankBadge: { position: 'absolute', top: -4, left: -4, width: 20, height: 20, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600], borderWidth: 2, borderColor: toneBg('#ffffff') },
  rankText: { fontSize: 10, fontWeight: '900', color: 'white' },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  name: { fontSize: 16, fontWeight: '800', color: theme.colors.text.display, flexShrink: 1 },
  bandChip: { flexShrink: 0, paddingHorizontal: 6, paddingVertical: 1, borderRadius: 5 },
  bandText: { fontSize: 9, fontWeight: '900', letterSpacing: 0.6 },
  sub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2 },
  cardActions: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 6 },
  metaChip: { flexDirection: 'row', alignItems: 'center', gap: 3, maxWidth: 130, paddingHorizontal: 7, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.bg.surfaceHover },
  metaText: { fontSize: 11, color: theme.colors.text.muted, flexShrink: 1 },
  miniAction: { width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  ringText: { fontSize: 14, fontWeight: '900', color: theme.colors.text.display },
  scoreBtn: { alignItems: 'center', justifyContent: 'center', gap: 1, width: 52, height: 48, borderRadius: 14, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  scoreBtnText: { fontSize: 10, fontWeight: '800', color: theme.colors.brand[700] },

  more: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, paddingVertical: 12 },
  moreText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },

  empty: { alignItems: 'center', gap: 8, paddingTop: 30, paddingHorizontal: 16 },
  emptyIcon: { width: 60, height: 60, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  emptyTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  emptyText: { fontSize: 13, lineHeight: 19, color: theme.colors.text.muted, textAlign: 'center' },
  emptyCta: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8, backgroundColor: theme.colors.brand[600], paddingHorizontal: 18, paddingVertical: 11, borderRadius: 999 },
  emptyCtaText: { color: 'white', fontWeight: '800', fontSize: 14 },
  emptySmall: { alignItems: 'center', gap: 6, paddingVertical: 30 },

  exCard: { width: 170, backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, gap: 4 },
  exName: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  exCat: { fontSize: 12, color: theme.colors.text.muted },
  exBooth: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 4 },
  exBoothText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700] },

  toast: { position: 'absolute', left: 16, right: 16, bottom: 20, flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.successText, paddingHorizontal: 14, paddingVertical: 12, borderRadius: 14, ...theme.shadows.active },
  toastError: { backgroundColor: theme.colors.semantic.danger },
  toastText: { flex: 1, fontSize: 13, fontWeight: '700', color: 'white' },

  askCard: { backgroundColor: toneBg('#ffffff'), borderRadius: 18, padding: 12, marginBottom: 12, gap: 8, borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.card },
  askHead: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 2 },
  askIcon: { width: 28, height: 28, borderRadius: 9, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  askTitle: { flex: 1, fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  askPrompt: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 12, paddingVertical: 10, borderRadius: 12, backgroundColor: theme.colors.brand[50] },
  askPromptText: { flex: 1, fontSize: 13, fontWeight: '600', color: theme.colors.brand[800], lineHeight: 18 },
}));
