import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, ScrollView, RefreshControl, ActivityIndicator, Pressable } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, ZoomOut, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withSpring, Easing, cancelAnimation,
} from 'react-native-reanimated';
import { ArrowLeft, RefreshCw, CheckCircle2, CloudOff, UploadCloud, DownloadCloud, Check, X, AlertTriangle, Plus, Pencil, Trash2, Clock, RotateCcw, Smartphone, Inbox, ScanLine, QrCode } from 'lucide-react-native';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, makeStyles } from '../../../src/ui/theme';
import { PressableScale } from '../../../src/features/onboarding/components';
import { AnimatedCount } from '../../../src/features/contacts/components/ListAnimations';
import { OptionSheet } from '../../../src/features/contacts/components/OptionSheet';
import {
  getSyncState, runSync, discardQueueItem, type SyncState, type SyncRunResult, type SyncPhase, type SyncQueueEntry,
} from '../../../src/sync/state';

function ago(iso: string | null) {
  if (!iso) return null;
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString([], { day: 'numeric', month: 'short' });
}

/* ------------------------------ hero orb ------------------------------ */

function SyncOrb({ state }: { state: 'idle' | 'syncing' | 'ok' | 'error' | 'pending' }) {
  const spin = useSharedValue(0);
  const pulse = useSharedValue(0);
  const pop = useSharedValue(1);

  useEffect(() => {
    if (state === 'syncing') {
      spin.value = 0;
      spin.value = withRepeat(withTiming(360, { duration: 1100, easing: Easing.linear }), -1);
      pulse.value = withRepeat(withSequence(withTiming(1, { duration: 700 }), withTiming(0, { duration: 700 })), -1);
    } else {
      cancelAnimation(spin);
      spin.value = withTiming(0, { duration: 250 });
      pulse.value = withTiming(0, { duration: 250 });
      pop.value = withSequence(withSpring(1.05, { damping: 22, overshootClamping: true }), withSpring(1, { damping: 22, overshootClamping: true }));
    }
  }, [state]);

  const ring = useAnimatedStyle(() => ({ transform: [{ rotate: `${spin.value}deg` }] }));
  const halo = useAnimatedStyle(() => ({ opacity: 0.18 + pulse.value * 0.3, transform: [{ scale: 1 + pulse.value * 0.18 }] }));
  const core = useAnimatedStyle(() => ({ transform: [{ scale: pop.value }] }));

  const color = state === 'error' ? theme.colors.semantic.danger
    : state === 'pending' ? theme.colors.semantic.warning
    : state === 'ok' ? theme.colors.semantic.success
    : theme.colors.brand[600];

  return (
    <View style={styles.orbWrap}>
      <Animated.View style={[styles.orbHalo, { backgroundColor: color }, halo]} />
      <Animated.View style={[styles.orbRing, { borderColor: color, borderTopColor: 'transparent' }, ring]} />
      <Animated.View style={[styles.orbCore, { backgroundColor: color }, core]}>
        {state === 'syncing' ? <RefreshCw size={30} color="white" />
          : state === 'error' ? <CloudOff size={30} color="white" />
          : state === 'pending' ? <UploadCloud size={30} color="white" />
          : <Check size={32} color="white" strokeWidth={3} />}
      </Animated.View>
    </View>
  );
}

function PhaseRow({ label, icon, status }: { label: string; icon: React.ReactNode; status: 'waiting' | 'active' | 'done' | 'failed' }) {
  return (
    <Animated.View entering={FadeInDown.duration(260)} style={styles.phaseRow}>
      <View style={[styles.phaseIcon, status === 'active' && { backgroundColor: theme.colors.brand[100] }, status === 'done' && { backgroundColor: theme.colors.semantic.successBg }, status === 'failed' && { backgroundColor: theme.colors.semantic.dangerBg }]}>
        {icon}
      </View>
      <Text style={[styles.phaseLabel, status === 'waiting' && { color: theme.colors.text.muted }]}>{label}</Text>
      <View style={{ width: 22, alignItems: 'center' }}>
        {status === 'active' ? <ActivityIndicator size="small" color={theme.colors.brand[600]} />
          : status === 'done' ? <Animated.View entering={ZoomIn.duration(260)}><CheckCircle2 size={18} color={theme.colors.semantic.success} /></Animated.View>
          : status === 'failed' ? <Animated.View entering={ZoomIn.duration(260)}><AlertTriangle size={17} color={theme.colors.semantic.danger} /></Animated.View>
          : null}
      </View>
    </Animated.View>
  );
}

/* ------------------------------ queue item ------------------------------ */

function QueueItem({ item, index, busy, onRetry, onDiscard }: {
  item: SyncQueueEntry; index: number; busy: boolean; onRetry: () => void; onDiscard: () => void;
}) {
  const failed = item.status === 'FAILED';
  const isScan = item.action === 'scan';
  const Icon = isScan ? (item.entity === 'scan-qr' ? QrCode : ScanLine)
    : item.action === 'create' ? Plus : item.action === 'delete' ? Trash2 : Pencil;
  const verb = isScan ? (item.entity === 'scan-qr' ? 'Offline QR scan' : 'Offline card scan')
    : item.action === 'create' ? 'New contact' : item.action === 'delete' ? 'Delete' : 'Edit';
  return (
    <Animated.View
      entering={FadeInDown.delay(Math.min(index, 10) * 50).duration(260)}
      exiting={ZoomOut.duration(160)}
      layout={LinearTransition.duration(260)}
      style={[styles.item, failed && styles.itemFailed]}
    >
      <View style={[styles.itemIcon, failed && { backgroundColor: theme.colors.semantic.dangerBg }]}>
        <Icon size={15} color={failed ? theme.colors.semantic.danger : theme.colors.brand[700]} />
      </View>
      <View style={{ flex: 1 }}>
        <View style={styles.itemTop}>
          <Text style={styles.itemVerb}>{verb}</Text>
          <View style={[styles.statusChip, failed ? styles.statusChipFailed : null]}>
            <Text style={[styles.statusChipText, failed && { color: theme.colors.semantic.danger }]}>{failed ? 'FAILED' : 'WAITING'}</Text>
          </View>
        </View>
        <Text style={styles.itemDetails} numberOfLines={1}>{item.details}</Text>
        <View style={styles.itemMeta}>
          <Clock size={11} color={theme.colors.text.muted} />
          <Text style={styles.metaText}>
            {item.createdAt ? `Queued ${ago(item.createdAt)}` : 'Queued'}
            {item.attempts ? ` · ${item.attempts} attempt${item.attempts === 1 ? '' : 's'}` : ''}
          </Text>
        </View>
        {failed && item.errorHint ? (
          <Text style={styles.itemError} numberOfLines={2}>{item.errorHint}</Text>
        ) : null}
        {failed ? (
          <View style={styles.itemActions}>
            <PressableScale onPress={onRetry} disabled={busy} scaleTo={0.92} style={styles.itemBtn}
              leadingIcon={<RotateCcw size={13} color={theme.colors.brand[700]} />}>
              <Text style={styles.itemBtnText}>Retry</Text>
            </PressableScale>
            <PressableScale onPress={onDiscard} disabled={busy} scaleTo={0.92} style={[styles.itemBtn, styles.itemBtnDanger]}
              leadingIcon={<X size={13} color={theme.colors.semantic.danger} />}>
              <Text style={[styles.itemBtnText, { color: theme.colors.semantic.danger }]}>Discard</Text>
            </PressableScale>
          </View>
        ) : null}
      </View>
    </Animated.View>
  );
}

/* --------------------------------- screen --------------------------------- */

export default function SyncScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const [state, setState] = useState<SyncState | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [phase, setPhase] = useState<SyncPhase | null>(null);
  const [lastRun, setLastRun] = useState<SyncRunResult | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [discardId, setDiscardId] = useState<string | null>(null);
  const [, tick] = useState(0);
  const running = useRef(false);

  const refresh = useCallback(async () => {
    try { setState(await getSyncState()); } catch (e) { console.warn('Failed to load sync state', e); }
  }, []);

  useFocusEffect(useCallback(() => { refresh(); }, [refresh]));

  // Keep "x min ago" fresh while the page is open.
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 30000);
    return () => clearInterval(t);
  }, []);

  const doSync = useCallback(async () => {
    if (running.current) return;
    running.current = true;
    setSyncing(true);
    setLastRun(null);
    try {
      const res = await runSync(setPhase);
      setState(res);
      setLastRun(res);
    } catch (e: any) {
      setLastRun({
        ...(state || { pendingChanges: 0, failedChanges: 0, queue: [], lastSyncAt: null, lastSyncCount: 0, lastAttemptAt: null, lastError: null }),
        pushed: 0, scans: 0, failed: 0, skipped: 0, pulled: 0, cached: 0, removed: 0, offline: false, error: e?.message || 'Sync failed',
      });
    } finally {
      running.current = false;
      setSyncing(false);
      setPhase(null);
    }
  }, [state]);

  const discard = async (id: string) => {
    try {
      await discardQueueItem(id);
    } finally {
      setDiscardId(null);
      refresh();
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  const pending = state?.pendingChanges ?? 0;
  const failed = state?.failedChanges ?? 0;
  const orb: 'idle' | 'syncing' | 'ok' | 'error' | 'pending' =
    syncing ? 'syncing' : failed || lastRun?.error || (state?.lastError && !lastRun) ? 'error' : pending ? 'pending' : state?.lastSyncAt ? 'ok' : 'idle';
  const headline = syncing
    ? phase === 'pull' ? 'Downloading your contacts…' : phase === 'scans' ? 'Reading offline scans…' : 'Uploading your changes…'
    : orb === 'error' ? 'Some changes didn\'t sync'
    : orb === 'pending' ? `${pending} change${pending === 1 ? '' : 's'} waiting`
    : orb === 'ok' ? 'Everything is synced'
    : 'Not synced yet';
  const sub = syncing ? 'Keep the app open for a moment'
    : state?.lastSyncAt ? `Last synced ${ago(state.lastSyncAt)}`
    : 'Tap Sync now to back up and refresh this device';
  const pushStatus = !syncing ? null : phase === 'push' ? 'active' : 'done';
  const scanStatus = !syncing ? null : phase === 'push' ? 'waiting' : phase === 'scans' ? 'active' : 'done';
  const pullStatus = !syncing ? null : phase === 'pull' ? 'active' : phase === 'done' ? 'done' : 'waiting';
  const queue = state?.queue ?? [];
  const toDiscard = queue.find((q) => q.id === discardId);

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <Animated.View entering={FadeInDown.duration(260)} style={styles.header}>
        <PressableScale onPress={() => router.back()} scaleTo={0.9} style={styles.backBtn} accessibilityLabel="Back">
          <ArrowLeft size={20} color={theme.colors.text.display} />
        </PressableScale>
        <Text style={styles.title}>Sync</Text>
        <View style={{ flex: 1 }} />
        <Animated.View key={orb} entering={ZoomIn.duration(260)}
          style={[styles.pill, orb === 'error' && styles.pillBad, (orb === 'pending' || orb === 'syncing') && styles.pillWarn]}>
          <Text style={[styles.pillText, orb === 'error' && { color: theme.colors.semantic.danger }, (orb === 'pending' || orb === 'syncing') && { color: theme.colors.semantic.warning }]}>
            {orb === 'syncing' ? 'SYNCING' : orb === 'error' ? 'NEEDS ATTENTION' : orb === 'pending' ? 'PENDING' : orb === 'ok' ? 'UP TO DATE' : 'NEVER SYNCED'}
          </Text>
        </Animated.View>
      </Animated.View>

      <ScrollView
        contentContainerStyle={[styles.container, { paddingBottom: 120 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl progressBackgroundColor={theme.colors.bg.surface} refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.brand[600]} colors={[theme.colors.brand[600]]} />}
      >
        {/* Status hero */}
        <Animated.View entering={FadeInDown.delay(60).duration(260)} style={styles.hero}>
          <SyncOrb state={orb} />
          <Animated.Text key={headline} entering={FadeIn.duration(220)} style={styles.headline}>{headline}</Animated.Text>
          <Text style={styles.sub}>{sub}</Text>

          <View style={styles.statsRow}>
            <View style={styles.stat}>
              <AnimatedCount value={pending - failed} style={[styles.statNum, pending - failed > 0 && { color: theme.colors.semantic.warning }]} />
              <Text style={styles.statLabel}>Waiting</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.stat}>
              <AnimatedCount value={failed} style={[styles.statNum, failed > 0 && { color: theme.colors.semantic.danger }]} />
              <Text style={styles.statLabel}>Failed</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.stat}>
              <AnimatedCount value={state?.lastSyncCount ?? 0} style={styles.statNum} />
              <Text style={styles.statLabel}>Last sync</Text>
            </View>
          </View>
        </Animated.View>

        {/* Live progress while syncing (real phases from runSync) */}
        {syncing ? (
          <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(200)} style={styles.card}>
            <PhaseRow label="Upload changes from this phone" icon={<UploadCloud size={15} color={theme.colors.brand[700]} />} status={pushStatus as any} />
            <PhaseRow label="Read cards scanned offline" icon={<ScanLine size={15} color={theme.colors.brand[700]} />} status={scanStatus as any} />
            <PhaseRow label="Download your latest contacts" icon={<DownloadCloud size={15} color={theme.colors.brand[700]} />} status={pullStatus as any} />
          </Animated.View>
        ) : null}

        {/* Result of the run that just finished */}
        {lastRun && !syncing ? (
          <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut} layout={LinearTransition.duration(260)}
            style={[styles.result, lastRun.error ? styles.resultBad : null]}>
            {lastRun.error ? <AlertTriangle size={18} color={theme.colors.semantic.danger} /> : <CheckCircle2 size={18} color={theme.colors.semantic.successText} />}
            <View style={{ flex: 1 }}>
              <Text style={[styles.resultTitle, lastRun.error ? { color: theme.colors.semantic.danger } : null]}>
                {lastRun.error ? lastRun.error : 'Sync complete'}
              </Text>
              <Text style={styles.resultText}>
                Uploaded {lastRun.pushed}{lastRun.scans ? ` · ${lastRun.scans} offline scan${lastRun.scans === 1 ? '' : 's'} added` : ''} · downloaded {lastRun.pulled}
                {lastRun.cached && lastRun.cached !== lastRun.pulled ? ` (${lastRun.cached} saved offline)` : lastRun.cached ? ' · saved for offline' : ''}
                {lastRun.skipped ? ` · ${lastRun.skipped} skipped (deleted elsewhere or not yours)` : ''}
                {lastRun.removed ? ` · ${lastRun.removed} deleted elsewhere, removed from this phone` : ''}
              </Text>
              {lastRun.skippedItems?.slice(0, 6).map((s) => (
                <Text key={s.id} style={styles.resultText} numberOfLines={2}>• {s.details}: {s.reason}</Text>
              ))}
              {(lastRun.skippedItems?.length ?? 0) > 6 ? (
                <Text style={styles.resultText}>…and {lastRun.skippedItems!.length - 6} more</Text>
              ) : null}
            </View>
            <Pressable onPress={() => setLastRun(null)} hitSlop={8}><X size={16} color={theme.colors.text.muted} /></Pressable>
          </Animated.View>
        ) : null}

        {/* Previous run failed (shown when opening the page) */}
        {!lastRun && !syncing && state?.lastError ? (
          <Animated.View entering={FadeInDown.duration(260)} style={[styles.result, styles.resultBad]}>
            <AlertTriangle size={18} color={theme.colors.semantic.danger} />
            <View style={{ flex: 1 }}>
              <Text style={[styles.resultTitle, { color: theme.colors.semantic.danger }]}>Last attempt {ago(state.lastAttemptAt)}: {state.lastError}</Text>
              <Text style={styles.resultText}>Nothing was lost — tap Sync now to try again.</Text>
            </View>
          </Animated.View>
        ) : null}

        {/* Queue */}
        <View style={styles.sectionHead}>
          <Text style={styles.sectionLabel}>WAITING ON THIS PHONE</Text>
          <Text style={styles.sectionCount}>{queue.length}</Text>
        </View>
        {queue.length === 0 ? (
          <Animated.View entering={FadeIn.delay(120)} style={styles.empty}>
            <Animated.View entering={ZoomIn.delay(160).duration(260)} style={styles.emptyIcon}>
              <Inbox size={24} color={theme.colors.semantic.successText} />
            </Animated.View>
            <Text style={styles.emptyTitle}>Nothing waiting</Text>
            <Text style={styles.emptyText}>Changes made while offline appear here until they're uploaded.</Text>
          </Animated.View>
        ) : (
          queue.map((item, i) => (
            <QueueItem key={item.id} item={item} index={i} busy={syncing} onRetry={doSync} onDiscard={() => setDiscardId(item.id)} />
          ))
        )}

        <Animated.View entering={FadeInDown.delay(200).duration(260)} style={styles.info}>
          <Smartphone size={16} color={theme.colors.text.muted} />
          <Text style={styles.infoText}>
            Contacts are saved on the server as you add them. This phone keeps a copy for offline use and queues any
            changes made without a connection — failed ones stay here until they upload or you discard them.
          </Text>
        </Animated.View>
      </ScrollView>

      {/* Actions */}
      <View style={[styles.bottomBar, { paddingBottom: Math.max(insets.bottom, 14) }]}>
        {failed > 0 ? (
          <Animated.View entering={ZoomIn.duration(260)} exiting={ZoomOut} style={{ flex: 1 }}>
            <PressableScale onPress={doSync} disabled={syncing} scaleTo={0.95} style={styles.secondaryBtn}
              leadingIcon={<RotateCcw size={16} color={theme.colors.brand[700]} />}>
              <Text style={styles.secondaryText}>Retry {failed}</Text>
            </PressableScale>
          </Animated.View>
        ) : null}
        <View style={{ flex: 2 }}>
          <PressableScale onPress={doSync} disabled={syncing} scaleTo={0.96} style={[styles.primaryBtn, syncing && { opacity: 0.8 }]}
            leadingIcon={syncing ? <ActivityIndicator size="small" color="white" /> : <RefreshCw size={17} color="white" />}>
            <Text style={styles.primaryText}>{syncing ? (phase === 'pull' ? 'Downloading…' : phase === 'scans' ? 'Reading scans…' : 'Uploading…') : 'Sync now'}</Text>
          </PressableScale>
        </View>
      </View>

      <OptionSheet
        visible={!!discardId}
        title="Discard this change?"
        message={toDiscard ? `“${toDiscard.details}” won't be uploaded. This can't be undone.` : undefined}
        options={[{ label: 'Discard change', destructive: true, icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, onPress: () => discardId && discard(discardId) }]}
        onClose={() => setDiscardId(null)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingVertical: 10 },
  backBtn: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider },
  title: { fontSize: 24, lineHeight: 30, fontWeight: '900', color: theme.colors.text.display },
  pill: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.semantic.successBg },
  pillWarn: { backgroundColor: theme.colors.semantic.warningBg },
  pillBad: { backgroundColor: theme.colors.semantic.dangerBg },
  pillText: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.semantic.successText },
  container: { paddingHorizontal: 16, paddingTop: 4 },

  hero: { alignItems: 'center', backgroundColor: toneBg('#ffffff'), borderRadius: 24, padding: 20, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.active },
  orbWrap: { width: 110, height: 110, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  orbHalo: { position: 'absolute', width: 110, height: 110, borderRadius: 55 },
  orbRing: { position: 'absolute', width: 96, height: 96, borderRadius: 48, borderWidth: 3 },
  orbCore: { width: 72, height: 72, borderRadius: 36, alignItems: 'center', justifyContent: 'center', ...theme.shadows.active },
  headline: { fontSize: 19, lineHeight: 25, fontWeight: '900', color: theme.colors.text.display, textAlign: 'center' },
  sub: { fontSize: 13, color: theme.colors.text.muted, marginTop: 3, textAlign: 'center' },
  statsRow: { flexDirection: 'row', alignItems: 'center', alignSelf: 'stretch', marginTop: 18, paddingTop: 14, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  stat: { flex: 1, alignItems: 'center' },
  statNum: { fontSize: 22, lineHeight: 27, fontWeight: '900', color: theme.colors.text.display },
  statLabel: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },
  statDivider: { width: 1, height: 30, backgroundColor: theme.colors.bg.divider },

  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 18, padding: 12, gap: 8, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider },
  phaseRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  phaseIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  phaseLabel: { flex: 1, fontSize: 14, fontWeight: '700', color: theme.colors.text.display },

  result: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 12, borderRadius: 16, marginBottom: 12, backgroundColor: theme.colors.semantic.successBg, borderWidth: 1, borderColor: toneBg('#a7f3d0') },
  resultBad: { backgroundColor: theme.colors.semantic.dangerBg, borderColor: toneBg('#fecaca') },
  resultTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.semantic.successText },
  resultText: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },

  sectionHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 6, marginBottom: 8, paddingHorizontal: 2 },
  sectionLabel: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  sectionCount: { fontSize: 12, fontWeight: '800', color: theme.colors.text.muted },

  item: { flexDirection: 'row', gap: 12, backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  itemFailed: { borderColor: toneBg('#fecaca') },
  itemIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  itemTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  itemVerb: { fontSize: 12, fontWeight: '800', color: theme.colors.text.muted, letterSpacing: 0.3 },
  statusChip: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 6, backgroundColor: theme.colors.semantic.warningBg },
  statusChipFailed: { backgroundColor: theme.colors.semantic.dangerBg },
  statusChipText: { fontSize: 9, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.semantic.warning },
  itemDetails: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display, marginTop: 2 },
  itemMeta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 3 },
  metaText: { fontSize: 11, color: theme.colors.text.muted },
  itemError: { fontSize: 12, color: theme.colors.semantic.danger, marginTop: 6, lineHeight: 17 },
  itemActions: { flexDirection: 'row', gap: 8, marginTop: 10 },
  itemBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  itemBtnDanger: { backgroundColor: toneBg('#ffffff'), borderColor: theme.colors.semantic.dangerBorder },
  itemBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },

  empty: { alignItems: 'center', gap: 6, backgroundColor: toneBg('#ffffff'), borderRadius: 18, padding: 20, borderWidth: 1, borderColor: theme.colors.bg.divider },
  emptyIcon: { width: 52, height: 52, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.successBg, marginBottom: 4 },
  emptyTitle: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  emptyText: { fontSize: 13, color: theme.colors.text.muted, textAlign: 'center', lineHeight: 18 },

  info: { flexDirection: 'row', gap: 10, padding: 14, borderRadius: 16, borderWidth: 1, borderStyle: 'dashed', borderColor: theme.colors.bg.dashed, marginTop: 12 },
  infoText: { flex: 1, fontSize: 12, lineHeight: 18, color: theme.colors.text.muted },

  bottomBar: {
    position: 'absolute', left: 0, right: 0, bottom: 0, flexDirection: 'row', gap: 10, paddingHorizontal: 16, paddingTop: 12,
    backgroundColor: theme.colors.bg.app, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider,
  },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 15, borderRadius: 999, backgroundColor: theme.colors.brand[600], ...theme.shadows.active },
  primaryText: { fontSize: 15, fontWeight: '800', color: 'white' },
  secondaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 15, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.brand[200] },
  secondaryText: { fontSize: 15, fontWeight: '800', color: theme.colors.brand[700] },
}));
