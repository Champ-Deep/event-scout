import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { View, SectionList, Pressable, ActivityIndicator, ScrollView, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeInUp, FadeOut, LinearTransition, ZoomIn,
  useSharedValue, useAnimatedStyle, interpolate, Extrapolation, withTiming, interpolateColor,
} from 'react-native-reanimated';
import {
  Search, X, Trash2, Copy, Flame, CalendarDays, Users, ChevronRight, Inbox, ScanLine, WifiOff, RotateCw, Sparkles,
  UserPlus, CalendarPlus, CheckCircle2,
} from 'lucide-react-native';
import { inQuickView, isQuickView, QUICK_VIEW_LABEL, type QuickView } from '../../../src/features/contacts/periods';
import type { Contact } from '@es/types';
import { useRankedFeed } from '../../../src/features/feed';
import { useSession } from '../../../src/providers/SessionProvider';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { api, errorMessage, type DuplicatePair } from '../../../src/lib/api';
import { profileStore } from '../../../src/features/profile';
import { isHot } from '../../../src/features/contacts/scoreLevel';
import { getSyncState, type SyncState } from '../../../src/sync/state';
import { contactEvents, contactsFeature } from '../../../src/features/contacts';
import { ContactCard } from '../../../src/features/contacts/components/ContactCard';
import type { SwipeableMethods } from 'react-native-gesture-handler/ReanimatedSwipeable';
import { DuplicateReviewSheet } from '../../../src/features/contacts/components/DuplicateReviewSheet';
import { UndoToast, type ToastState } from '../../../src/features/contacts/components/UndoToast';
import { PressableScale } from '../../../src/features/onboarding/components';
import { AnimatedCount, AnimatedSearch, BreathingGlow, ScrollTopButton, SkeletonRows } from '../../../src/features/contacts/components/ListAnimations';

type Tab = 'All' | 'Hot' | 'This event';

type Filter = 'all' | 'hot' | 'unscored' | 'event';

/** One filter pill: icon + label + live count. Selected = filled brand pill. */
function FilterPill({ icon: Icon, label, count, active, tint, onPress, index }: {
  icon: any; label: string; count: number | null; active: boolean; tint: string; onPress: () => void; index: number;
}) {
  const on = useSharedValue(active ? 1 : 0);
  useEffect(() => { on.value = withTiming(active ? 1 : 0, { duration: 180 }); }, [active]);
  const SURFACE = theme.colors.bg.surface; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const BRAND_600 = theme.colors.brand[600];
  const DIVIDER = theme.colors.bg.divider;
  const BRAND_700 = theme.colors.brand[700];
  const bg = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(on.value, [0, 1], [SURFACE, BRAND_600]),
    borderColor: interpolateColor(on.value, [0, 1], [DIVIDER, BRAND_700]),
    transform: [{ scale: 1 + on.value * 0.02 }],
  }));
  return (
    <Animated.View entering={FadeInDown.delay(80 + index * 45).duration(260)}>
      <Pressable onPress={onPress} accessibilityRole="tab" accessibilityState={{ selected: active }} accessibilityLabel={`${label}${count != null ? `, ${count}` : ''}`}>
        <Animated.View style={[styles.pill, bg]}>
          <Icon size={14} color={active ? 'white' : tint} />
          <Text style={[styles.pillLabel, active && { color: 'white' }]}>{label}</Text>
          <View style={[styles.pillCount, active && styles.pillCountOn]}>
            {count == null
              ? <Text style={[styles.pillCountText, active && { color: 'white' }]}>–</Text>
              : <AnimatedCount value={count} style={[styles.pillCountText, active && { color: 'white' }]} />}
          </View>
        </Animated.View>
      </Pressable>
    </Animated.View>
  );
}

export default function ContactsScreen() {
  const router = useRouter();
  const { userId } = useSession();
  // All = every contact you own (the API scopes to the signed-in user); Hot = A-band scores.
  const [activeTab, setActiveTab] = useState<Tab>('All');
  // "Not scored" filter (from the stat chip): all your contacts that have no score yet.
  const [unscoredOnly, setUnscoredOnly] = useState(false);
  const [scoringAll, setScoringAll] = useState(false);
  const [scoreMsg, setScoreMsg] = useState<{ ok: boolean; text: string } | null>(null);
  // Opened from a stat tile (Home "Met today", Me "Scored" / "This week"): a quick view on All.
  const [quick, setQuick] = useState<QuickView | null>(null);
  const selectTab = (tab: Tab) => { setActiveTab(tab); setUnscoredOnly(false); setQuick(null); };
  const [search, setSearch] = useState('');

  // ?show=today|week|scored|hot|unscored|all (+ t so tapping the same tile again re-applies it)
  const params = useLocalSearchParams<{ show?: string; t?: string }>();
  useEffect(() => {
    const show = params.show;
    if (!show) return;
    setSearch('');
    if (isQuickView(show)) { setActiveTab('All'); setUnscoredOnly(false); setQuick(show); }
    else if (show === 'hot') selectTab('Hot');
    else if (show === 'unscored') { setActiveTab('All'); setQuick(null); setUnscoredOnly(true); }
    else selectTab('All');
    router.setParams({ show: undefined, t: undefined } as any);
  }, [params.show, params.t]);

  // The active event chosen in onboarding / profile (profile.current_event_name).
  // Scans are tagged with it, so the event tab filters on the same name.
  const [activeEventName, setActiveEventName] = useState<string | null>(null);
  const [eventLoaded, setEventLoaded] = useState(false);
  // Re-read on every focus (below) too, so switching the active event on Me shows here right away.
  const profileReq = useRef(0);
  const eventLoadedRef = useRef(false);
  const loadActiveEvent = useCallback(() => {
    if (!userId) return;
    const req = ++profileReq.current;
    profileStore.load(userId)
      .then((p) => { if (req === profileReq.current) setActiveEventName(p.current_event_name || null); })
      .catch(() => { if (req === profileReq.current && !eventLoadedRef.current) setActiveEventName(null); })
      .finally(() => { if (req === profileReq.current) { eventLoadedRef.current = true; setEventLoaded(true); } });
  }, [userId]);
  useEffect(() => {
    loadActiveEvent();
    return () => { profileReq.current++; };
  }, [loadActiveEvent]);

  const activeFilters = useMemo(() => {
    // Page size; allPages loads the rest so search and the counts cover every contact.
    if (activeTab === 'Hot') return { scoreBand: 'A' as const, limit: 500, allPages: true };
    if (activeTab === 'This event' && activeEventName) return { eventId: activeEventName, limit: 500, allPages: true };
    return { limit: 500, allPages: true };
  }, [activeTab, activeEventName]);

  const { contacts: feedContacts, loading, error: loadError, reload } = useRankedFeed(activeFilters);
  const noActiveEvent = activeTab === 'This event' && eventLoaded && !activeEventName;
  const contacts = noActiveEvent ? [] : feedContacts;

  // Header stats come from the full (All) list, remembered while other tabs are showing.
  const [stats, setStats] = useState<{ total: number; hot: number; unscored: number; event: number } | null>(null);
  useEffect(() => {
    if (activeTab !== 'All' || loading || loadError) return;
    setStats({
      total: feedContacts.length,
      hot: feedContacts.filter((c) => isHot(c)).length,
      unscored: feedContacts.filter((c) => c.score == null).length,
      event: activeEventName ? feedContacts.filter((c) => c.eventId === activeEventName).length : 0,
    });
  }, [activeTab, loading, loadError, feedContacts, activeEventName]);

  // One filter row (it used to be stat chips AND tabs showing the same three things).
  const filter: Filter | null = quick ? null : unscoredOnly ? 'unscored' : activeTab === 'Hot' ? 'hot' : activeTab === 'This event' ? 'event' : 'all';
  const pickFilter = (f: Filter) => {
    if (f === 'unscored') { setActiveTab('All'); setQuick(null); setUnscoredOnly(true); return; }
    selectTab(f === 'hot' ? 'Hot' : f === 'event' ? 'This event' : 'All');
  };

  // Collapsing header: compact bar title + background fade in as the large title scrolls off.
  const scrollY = useSharedValue(0);
  const listRef = useRef<SectionList<Contact>>(null);
  const [showTop, setShowTop] = useState(false);
  const topBarBgStyle = useAnimatedStyle(() => ({
    opacity: interpolate(scrollY.value, [10, 50], [0, 1], Extrapolation.CLAMP),
  }));

  // Real side data: unsynced outbox changes, possible duplicates, Recently deleted count.
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const [dupPairs, setDupPairs] = useState<DuplicatePair[]>([]);
  const [deletedCount, setDeletedCount] = useState(0);
  const [reviewOpen, setReviewOpen] = useState(false);

  const loadSideData = useCallback(() => {
    getSyncState().then(setSyncState).catch(() => setSyncState(null));
    api.listDuplicatePairs().then((r) => setDupPairs(r?.pairs ?? [])).catch(() => setDupPairs([]));
    api.listDeletedContacts().then((r) => setDeletedCount(r?.contacts?.length ?? 0)).catch(() => {});
  }, []);

  // Refresh when coming back from a contact (edited, deleted or newly scanned).
  const firstFocus = useRef(true);
  useFocusEffect(
    useCallback(() => {
      loadSideData();
      if (firstFocus.current) {
        firstFocus.current = false; // the feed hook (and the profile effect) already load on mount
        return;
      }
      loadActiveEvent();
      reload();
    }, [reload, loadSideData, loadActiveEvent]),
  );

  // "Moved to Recently deleted · Undo" after a delete on the detail / edit screen.
  const [toast, setToast] = useState<ToastState | null>(null);
  const [undoing, setUndoing] = useState(false);
  useEffect(() => contactEvents.subscribe((e) => {
    if (e.type === 'deleted') {
      setToast({ kind: 'deleted', contactId: e.id, message: `${e.name} moved to Recently deleted` });
      setDeletedCount((n) => n + 1);
    }
  }), []);

  const undoDelete = async () => {
    if (!toast?.contactId || undoing) return;
    setUndoing(true);
    try {
      await contactsFeature.restoreContact(toast.contactId);
      setToast({ kind: 'restored', message: 'Contact restored', contactId: toast.contactId });
      setDeletedCount((n) => Math.max(0, n - 1));
      reload();
      loadSideData();
    } catch (e: any) {
      setToast({ kind: 'error', message: errorMessage(e, "Couldn't restore. Try Recently deleted.") });
    } finally {
      setUndoing(false);
    }
  };

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const base = unscoredOnly ? contacts.filter((c) => c.score == null)
      : quick ? contacts.filter((c) => inQuickView(quick, c as any))
      : contacts;
    if (!q) return base;
    return base.filter((c) => [c.firstName, c.lastName, c.company, c.title, c.email, c.phone, c.notes]
      .some((v) => (v || '').toLowerCase().includes(q)));
  }, [contacts, search, unscoredOnly, quick]);
  const quickCount = quick ? contacts.filter((c) => inQuickView(quick, c as any)).length : 0;

  // Score every unscored contact with AI (POST /api/contacts/score_all, up to 25 per run).
  const scoreAll = async () => {
    if (scoringAll) return;
    setScoringAll(true);
    setScoreMsg(null);
    try {
      const res = await api.scoreAllContacts();
      setScoreMsg({ ok: !res.errors, text: `Scored ${res.scoredCount}${res.errors ? ` · ${res.errors} failed` : ''}${res.pending ? ` · ${res.pending} still to go` : ''}` });
      reload();
    } catch (e: any) {
      setScoreMsg({ ok: false, text: errorMessage(e, "Couldn't score right now.") });
    } finally {
      setScoringAll(false);
    }
  };

  const sections = useMemo(() => {
    const map: Record<string, Contact[]> = {};
    filtered.forEach((c) => {
      if (!c) return;
      const initial = (c.firstName?.[0] || c.lastName?.[0] || '#').toUpperCase();
      (map[initial] ||= []).push(c);
    });
    return Object.keys(map).sort().map((key) => ({
      title: key,
      data: map[key]!.sort((a, b) =>
        [a.firstName, a.lastName].join(' ').toLowerCase().localeCompare([b.firstName, b.lastName].join(' ').toLowerCase())),
    }));
  }, [filtered]);

  // Row index across sections, for a quick stagger on first paint only.
  const rowIndex = useMemo(() => {
    const m = new Map<string, number>();
    let i = 0;
    sections.forEach((s) => s.data.forEach((c) => m.set(c.id, i++)));
    return m;
  }, [sections]);

  const dupIds = useMemo(() => {
    const s = new Set<string>();
    dupPairs.forEach((p) => { s.add(p.a.id); s.add(p.b.id); });
    return s;
  }, [dupPairs]);

  // One swiped-open card at a time: opening another closes the previous one.
  const openRow = useRef<SwipeableMethods | null>(null);
  const onRowOpen = (m: SwipeableMethods) => {
    if (openRow.current && openRow.current !== m) openRow.current.close();
    openRow.current = m;
  };
  // Swipe → Delete: moves it to Recently deleted; the 'deleted' event shows "… · Undo".
  const deleteFromList = async (item: Contact) => {
    const fullName = [item.firstName, item.lastName].filter(Boolean).join(' ') || 'Contact';
    try {
      await contactsFeature.deleteContact(item.id, fullName);
      reload();
    } catch (e: any) {
      setToast({ kind: 'error', message: errorMessage(e, `Couldn't delete ${fullName}. Try again.`) });
    }
  };

  const renderContact = (item: Contact) => {
    const idx = rowIndex.get(item.id) ?? 0;
    return (
      <Animated.View
        entering={idx < 14 ? FadeInDown.delay(idx * 35).duration(260) : FadeIn}
        exiting={FadeOut.duration(150)}
        layout={LinearTransition.duration(260)}
      >
        <ContactCard
          contact={item}
          index={idx}
          duplicate={dupIds.has(item.id)}
          onOpen={() => { openRow.current?.close(); router.push(`/contacts/${item.id}`); }}
          onDelete={() => deleteFromList(item)}
          onSwipeOpen={onRowOpen}
        />
      </Animated.View>
    );
  };

  const emptyState = () => {
    if (loading) return <SkeletonRows />;
    if (loadError) {
      // Never show "No contacts yet" when the list simply couldn't be loaded.
      return (
        <Animated.View entering={FadeInUp.duration(260)} style={styles.emptyContainer}>
          <View style={[styles.emptyIcon, { backgroundColor: theme.colors.semantic.dangerBg }]}>
            <WifiOff size={26} color={theme.colors.semantic.danger} />
          </View>
          <Text style={styles.emptyTitle}>Couldn't load contacts</Text>
          <Text style={styles.emptySubtitle}>{loadError} Pull down or tap to try again.</Text>
          <PressableScale
            style={styles.emptyCta}
            onPress={() => { reload(); loadSideData(); }}
            leadingIcon={<RotateCw size={16} color="white" />}
            accessibilityRole="button"
          >
            <Text style={styles.emptyCtaText}>Try again</Text>
          </PressableScale>
        </Animated.View>
      );
    }
    const [Icon, title, subtitle] = quick && !search.trim()
      ? quick === 'today' ? [UserPlus, 'Nobody new today', 'Contacts you scan or add today appear here.']
        : quick === 'week' ? [CalendarPlus, 'Nobody new this week', 'Contacts added in the last 7 days appear here.']
        : [CheckCircle2, 'No scored contacts yet', 'Score a lead from its contact page, or score them all from “Not scored”.']
      : unscoredOnly && !search.trim()
      ? [Sparkles, 'Every contact is scored', 'Nothing left to score — tap “contacts” to see everyone.']
      : search.trim()
      ? [Search, `No matches for “${search.trim()}”`, 'Try a name, company, email or phone number.']
      : activeTab === 'Hot'
      ? [Flame, 'No hot leads yet', 'Contacts scored 85 or higher appear here. Score leads from their contact page.']
      : noActiveEvent
      ? [CalendarDays, 'No active event', 'Pick the event you are at in onboarding, and scans will be grouped here.']
      : activeTab === 'This event'
      ? [CalendarDays, `No contacts for ${activeEventName || 'this event'} yet`, 'Contacts you scan while this event is active appear here.']
      : [Users, 'No contacts yet', 'Scan a badge or business card to add your first lead.'];
    return (
      <Animated.View entering={FadeInUp.duration(260)} style={styles.emptyContainer}>
        <View style={styles.emptyIcon}><Icon size={26} color={theme.colors.brand[600]} /></View>
        <Text style={styles.emptyTitle}>{title}</Text>
        <Text style={styles.emptySubtitle}>{subtitle}</Text>
        {activeTab === 'All' && !search.trim() && quick !== 'scored' ? (
          <View style={styles.emptyActions}>
            <PressableScale
              style={styles.emptyCta}
              onPress={() => router.push('/(app)/scan' as any)}
              leadingIcon={<ScanLine size={16} color="white" />}
              accessibilityRole="button"
            >
              <Text style={styles.emptyCtaText}>Scan a card</Text>
            </PressableScale>
            <PressableScale
              style={styles.emptyCtaSecondary}
              onPress={() => router.push('/contacts/new' as any)}
              leadingIcon={<UserPlus size={16} color={theme.colors.brand[700]} />}
              accessibilityRole="button"
            >
              <Text style={styles.emptyCtaSecondaryText}>Add manually</Text>
            </PressableScale>
          </View>
        ) : null}
      </Animated.View>
    );
  };

  // Everything above the rows scrolls with the list (it used to be pinned and ate ~half the
  // screen). Passed as an element, not a component, so the search input keeps focus.
  const listHeader = (
    <View>
      {/* Search (filters as you type) */}
      <Animated.View entering={FadeInDown.delay(60).duration(260)}>
        <AnimatedSearch
          value={search}
          onChangeText={setSearch}
          placeholder="Search name, company, email, notes…"
          returnKeyType="search"
          autoCorrect={false}
        />
      </Animated.View>

        {/* Filters: one row, each with its live count */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.pillsRow} keyboardShouldPersistTaps="handled">
          <FilterPill index={0} icon={Users} label="All" tint={theme.colors.brand[700]} count={stats ? stats.total : null}
            active={filter === 'all'} onPress={() => pickFilter('all')} />
          <FilterPill index={1} icon={Flame} label="Hot" tint={toneFg('#dc2626')} count={stats ? stats.hot : null}
            active={filter === 'hot'} onPress={() => pickFilter('hot')} />
          <FilterPill index={2} icon={Sparkles} label="Not scored" tint={theme.colors.text.body} count={stats ? stats.unscored : null}
            active={filter === 'unscored'} onPress={() => pickFilter('unscored')} />
          <FilterPill index={3} icon={CalendarDays} label={activeEventName ? activeEventName : 'This event'} tint={toneFg('#0369a1')}
            count={activeEventName ? (stats ? stats.event : null) : null}
            active={filter === 'event'} onPress={() => pickFilter('event')} />
        </ScrollView>

        {/* Server unreachable but the device had cached contacts: say they may be out of date. */}
        {loadError && contacts.length > 0 ? (
          <Animated.View entering={FadeInDown.duration(260)}>
            <PressableScale style={styles.offlineBanner} onPress={() => { reload(); loadSideData(); }} scaleTo={0.98} accessibilityRole="button">
              <WifiOff size={16} color={theme.colors.semantic.danger} />
              <Text style={styles.offlineText}>{loadError} Showing contacts saved on this device — tap to retry.</Text>
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Possible duplicates — asks the user to check them */}
        {dupPairs.length > 0 ? (
          <Animated.View entering={FadeInDown.duration(260)} layout={LinearTransition}>
            <PressableScale style={styles.dupBanner} onPress={() => setReviewOpen(true)} scaleTo={0.98} accessibilityRole="button">
              <View style={styles.dupIcon}><Copy size={16} color={theme.colors.semantic.warning} /></View>
              <View style={{ flex: 1 }}>
                <Text style={styles.dupTitle}>
                  {dupPairs.length} possible duplicate{dupPairs.length === 1 ? '' : 's'}
                </Text>
                <Text style={styles.dupSub}>Same email, phone or name — tap to check them</Text>
              </View>
              <ChevronRight size={18} color={theme.colors.semantic.warning} />
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Only when this device really has unsynced changes (from the local outbox). */}
        {syncState && syncState.pendingChanges > 0 ? (
          <Animated.View entering={FadeInDown.duration(260)}>
            <PressableScale style={styles.outboxWidget} onPress={() => router.push('/me/sync' as any)} scaleTo={0.98} accessibilityRole="button">
              <Inbox size={16} color={theme.colors.brand[700]} />
              <Text style={styles.outboxText}>
                {syncState.pendingChanges} change{syncState.pendingChanges === 1 ? '' : 's'} not synced yet
                {syncState.failedChanges > 0 ? ` · ${syncState.failedChanges} failed` : ''}
              </Text>
              <ChevronRight size={16} color={theme.colors.text.muted} />
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Quick view from a stat tile: what's shown + a way back to everyone */}
        {quick ? (
          <Animated.View key={quick} entering={FadeInDown.duration(260)} exiting={FadeOut.duration(150)} layout={LinearTransition}>
            <View style={styles.quickBanner}>
              <View style={styles.quickIcon}>
                {quick === 'today' ? <UserPlus size={15} color="white" /> : quick === 'week' ? <CalendarPlus size={15} color="white" /> : <CheckCircle2 size={15} color="white" />}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.quickTitle}>{QUICK_VIEW_LABEL[quick]}</Text>
                <Text style={styles.quickSub}>
                  <AnimatedCount value={quickCount} style={styles.quickSub} /> {quickCount === 1 ? 'contact' : 'contacts'}
                  {quick === 'week' ? ' in the last 7 days' : quick === 'scored' ? ' with an AI score' : ''}
                </Text>
              </View>
              <Pressable onPress={() => setQuick(null)} hitSlop={10} style={styles.quickClose} accessibilityLabel="Show all contacts">
                <X size={15} color={theme.colors.brand[700]} />
              </Pressable>
            </View>
          </Animated.View>
        ) : null}

        {/* Not-scored filter: say what's shown and offer to score them all */}
        {unscoredOnly ? (
          <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(150)} layout={LinearTransition}>
            <PressableScale style={styles.scoreBanner} onPress={scoreAll} disabled={scoringAll || filtered.length === 0} scaleTo={0.98} accessibilityRole="button">
              <View style={styles.scoreBannerIcon}>{scoringAll ? <ActivityIndicator size="small" color="white" /> : <Sparkles size={16} color="white" />}</View>
              <View style={{ flex: 1 }}>
                <Text style={styles.scoreBannerTitle}>
                  {scoringAll ? 'Scoring with AI…' : filtered.length === 0 ? 'Everything here is scored' : `Score ${filtered.length === 1 ? 'this lead' : `these ${filtered.length}`} with AI`}
                </Text>
                <Text style={[styles.scoreBannerSub, scoreMsg && !scoreMsg.ok ? { color: theme.colors.semantic.danger } : null]}>
                  {scoreMsg ? scoreMsg.text : 'Checks each one against your targeting'}
                </Text>
              </View>
              <Pressable onPress={() => setUnscoredOnly(false)} hitSlop={10} accessibilityLabel="Show all contacts"><X size={16} color={theme.colors.brand[700]} /></Pressable>
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Result count only while searching (the filter pills already show the counts) */}
        {contacts.length > 0 && search.trim() ? (
          <Text style={styles.resultCount}>
            {unscoredOnly
              ? `${filtered.length} not scored${search.trim() ? ` matching “${search.trim()}”` : ''}`
              : search.trim()
              ? `${filtered.length} of ${contacts.length} match “${search.trim()}”`
              : activeTab === 'This event' && activeEventName
              ? `${contacts.length} at ${activeEventName}`
              : `${contacts.length} ${activeTab === 'Hot' ? 'hot ' : ''}${contacts.length === 1 ? 'contact' : 'contacts'}`}
          </Text>
        ) : null}
    </View>
  );

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      {/* One header row: title + Recently deleted + Add + Scan. Gains a background on scroll. */}
      <View style={styles.topBar}>
        <Animated.View style={[styles.topBarBg, topBarBgStyle]} pointerEvents="none" />
        <Animated.Text entering={FadeInDown.duration(260)} style={styles.topBarTitle} numberOfLines={1}>Contacts</Animated.Text>
        <View style={styles.topBarActions}>
          <PressableScale
            style={styles.headerBtn}
            onPress={() => router.push('/contacts/deleted' as any)}
            scaleTo={0.9}
            accessibilityRole="button"
            accessibilityLabel={`Recently deleted${deletedCount ? `, ${deletedCount}` : ''}`}
          >
            <Trash2 size={18} color={theme.colors.text.display} />
            {deletedCount > 0 ? (
              <Animated.View entering={ZoomIn.duration(260)} style={styles.badge}>
                <Text style={styles.badgeText}>{deletedCount > 99 ? '99+' : deletedCount}</Text>
              </Animated.View>
            ) : null}
          </PressableScale>
          <PressableScale
            style={styles.headerBtn}
            onPress={() => router.push('/contacts/new' as any)}
            scaleTo={0.9}
            accessibilityRole="button"
            accessibilityLabel="Add a contact manually"
          >
            <UserPlus size={18} color={theme.colors.brand[700]} />
          </PressableScale>
          <View>
          <BreathingGlow />
          <PressableScale
            style={styles.scanBtn}
            onPress={() => router.push('/(app)/scan' as any)}
            scaleTo={0.92}
            leadingIcon={<ScanLine size={16} color="white" strokeWidth={2.5} />}
            accessibilityRole="button"
            accessibilityLabel="Scan a card"
          >
            <Text style={styles.scanBtnText}>Scan</Text>
          </PressableScale>
          </View>
        </View>
      </View>

      <SectionList
        ref={listRef}
        key={activeTab}
        style={styles.list}
        sections={sections}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => renderContact(item)}
        // Same pull-to-refresh as before; the spinner's circle follows light / dark (Android draws it white).
        refreshControl={<RefreshControl refreshing={loading && contacts.length > 0} onRefresh={() => { reload(); loadSideData(); }} progressBackgroundColor={theme.colors.bg.surface} />}
        onScrollBeginDrag={() => openRow.current?.close()}
        onScroll={(e) => {
          const y = e.nativeEvent.contentOffset.y;
          scrollY.value = y;
          if ((y > 700) !== showTop) setShowTop(y > 700);
        }}
        scrollEventThrottle={16}
        ListHeaderComponent={listHeader}
        renderSectionHeader={({ section: { title, data } }) => (
          <Animated.View entering={FadeIn.duration(260)} style={styles.sectionHeader}>
            <Animated.View entering={ZoomIn.duration(260)} style={styles.sectionLetter}><Text style={styles.sectionTitle}>{title}</Text></Animated.View>
            <View style={styles.sectionRule} />
            <Text style={styles.sectionCount}>{data.length}</Text>
          </Animated.View>
        )}
        stickySectionHeadersEnabled={false}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="on-drag"
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={emptyState}
        ListFooterComponent={<View style={{ height: 110 }} />}
      />

      <ScrollTopButton
        visible={showTop}
        onPress={() => (listRef.current as any)?.getScrollResponder?.()?.scrollTo({ y: 0, animated: true })}
      />

      <DuplicateReviewSheet
        visible={reviewOpen}
        pairs={dupPairs}
        onClose={() => setReviewOpen(false)}
        onOpenContact={(id) => router.push(`/contacts/${id}`)}
        onChanged={loadSideData}
      />

      <UndoToast toast={toast} onUndo={undoDelete} undoing={undoing} onDismiss={() => setToast(null)} />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  quickBanner: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10, padding: 12, borderRadius: 16, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  quickIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  quickTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[800] },
  quickSub: { fontSize: 12, fontWeight: '600', color: theme.colors.brand[700] },
  quickClose: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  list: { flex: 1 },

  topBar: {
    height: 60, flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, zIndex: 2,
  },
  topBarBg: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: theme.colors.bg.surface,
    borderBottomWidth: 1, borderBottomColor: theme.colors.bg.divider, ...theme.shadows.card,
  },
  topBarTitle: { flex: 1, fontSize: 26, lineHeight: 32, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.5 },
  topBarActions: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  scanBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.brand[600],
    paddingHorizontal: 14, height: 40, borderRadius: 20, ...theme.shadows.active,
  },
  scanBtnText: { color: 'white', fontSize: 14, fontWeight: '800' },

  largeTitleBlock: { marginBottom: 12, marginTop: 2 },
  pageTitle: { fontSize: 32, lineHeight: 40, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.8 },
  statRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  pillsRow: { gap: 8, paddingBottom: 12, paddingRight: 8 },
  pill: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingLeft: 12, paddingRight: 6, paddingVertical: 6, borderRadius: 999, borderWidth: 1.5, maxWidth: 220 },
  pillLabel: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display, flexShrink: 1 },
  pillCount: { minWidth: 24, height: 22, paddingHorizontal: 7, borderRadius: 11, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  pillCountOn: { backgroundColor: 'rgba(255,255,255,0.22)' },
  pillCountText: { fontSize: 12, fontWeight: '900', color: theme.colors.text.body },
  scoreBanner: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[200], borderRadius: 14, padding: 12, marginBottom: 10 },
  scoreBannerIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  scoreBannerTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[800] },
  scoreBannerSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  resultCount: { fontSize: 12, fontWeight: '700', color: theme.colors.text.muted, marginTop: 2, marginBottom: 2, paddingHorizontal: 2 },
  headerBtn: {
    width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.bg.surface,
    borderWidth: 1, borderColor: theme.colors.bg.divider, alignItems: 'center', justifyContent: 'center',
  },
  badge: {
    position: 'absolute', top: -4, right: -4, minWidth: 18, height: 18, borderRadius: 9, paddingHorizontal: 4,
    backgroundColor: theme.colors.semantic.danger, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: theme.colors.bg.app,
  },
  badgeText: { color: 'white', fontSize: 10, fontWeight: '800' },

  searchBox: {
    flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: theme.colors.bg.surface,
    borderWidth: 1, borderColor: theme.colors.bg.divider, borderRadius: 14, paddingHorizontal: 14, height: 46, marginBottom: 12,
  },
  searchInput: { flex: 1, fontSize: 15, color: theme.colors.text.display, paddingVertical: 8 },
  tabsWrap: { marginBottom: 12 },

  dupBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: theme.colors.semantic.warningBg,
    borderWidth: 1, borderColor: theme.colors.semantic.warningBorder, borderRadius: 14, padding: 12, marginBottom: 10,
  },
  dupIcon: { width: 32, height: 32, borderRadius: 10, backgroundColor: toneBg('#ffffff'), alignItems: 'center', justifyContent: 'center' },
  dupTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  dupSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  outboxWidget: {
    flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.brand[50],
    borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10, marginBottom: 10,
  },
  outboxText: { flex: 1, fontSize: 13, color: theme.colors.brand[800], fontWeight: '600' },
  offlineBanner: {
    flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg,
    borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10, marginBottom: 10,
  },
  offlineText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  listContent: { paddingHorizontal: 16, paddingTop: 4, paddingBottom: 20, flexGrow: 1 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingTop: 14, paddingBottom: 8 },
  sectionLetter: { minWidth: 26, height: 26, borderRadius: 8, paddingHorizontal: 6, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  sectionTitle: { fontSize: 13, fontWeight: '900', color: theme.colors.brand[700] },
  sectionRule: { flex: 1, height: 1, backgroundColor: theme.colors.bg.divider },
  sectionCount: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },


  emptyContainer: { alignItems: 'center', paddingTop: 48, paddingHorizontal: 24, gap: 8 },
  emptyIcon: { width: 60, height: 60, borderRadius: 20, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
  emptyTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display, textAlign: 'center' },
  emptySubtitle: { fontSize: 14, color: theme.colors.text.muted, textAlign: 'center', lineHeight: 20 },
  emptyCta: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 10, backgroundColor: theme.colors.brand[600], paddingHorizontal: 18, paddingVertical: 11, borderRadius: 999 },
  emptyCtaText: { color: 'white', fontWeight: '700', fontSize: 14 },
  emptyActions: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 10, marginTop: 2 },
  emptyCtaSecondary: {
    flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 10, backgroundColor: theme.colors.brand[50],
    borderWidth: 1.5, borderColor: theme.colors.brand[200], paddingHorizontal: 18, paddingVertical: 10, borderRadius: 999,
  },
  emptyCtaSecondaryText: { color: theme.colors.brand[700], fontWeight: '700', fontSize: 14 },
}));
