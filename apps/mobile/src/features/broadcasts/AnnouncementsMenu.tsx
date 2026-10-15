import React, { useEffect, useState } from 'react';
import { View, Pressable, ScrollView, useWindowDimensions } from 'react-native';
import Animated, { FadeIn, FadeInUp, FadeOutUp, ZoomIn } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Bell, Megaphone, Siren, X } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../ui/SheetOverlay';
import type { Broadcast } from '../../lib/api';
import { useSession } from '../../providers/SessionProvider';
import {
  useBroadcasts, unreadCount, openAnnouncements, closeAnnouncements, dismissPopup, startLiveBroadcasts, ago,
} from './store';

const RED = '#b91c1c';

/** Header button: opens the announcements dropdown; the badge counts ones you haven't opened yet. */
export function AnnouncementsBell() {
  const s = useBroadcasts();
  const unread = unreadCount(s);
  return (
    <Pressable onPress={openAnnouncements} hitSlop={8} style={styles.bell}
      accessibilityRole="button" accessibilityLabel={unread ? `Announcements, ${unread} new` : 'Announcements'}>
      <Bell size={18} color={theme.colors.brand[700]} />
      {unread ? (
        <Animated.View key={unread} entering={ZoomIn.duration(220)} style={styles.badge}>
          <Text style={styles.badgeText}>{unread > 9 ? '9+' : unread}</Text>
        </Animated.View>
      ) : null}
    </Pressable>
  );
}

/**
 * Mounted once for the signed-in app (any tab): keeps the live connection, shows a pop-up the
 * moment an admin posts, and hosts the dropdown with every announcement.
 */
export function AnnouncementsLayer() {
  const { userId } = useSession();
  useEffect(() => (userId ? startLiveBroadcasts() : undefined), [userId]);
  return (
    <>
      <ArrivalPopup />
      <AnnouncementsDropdown />
    </>
  );
}

function ArrivalPopup() {
  const { popup } = useBroadcasts();
  const insets = useSafeAreaInsets();
  // a normal one tucks itself away; an urgent one stays until you look at it
  useEffect(() => {
    if (!popup || popup.priority === 'urgent') return;
    const t = setTimeout(dismissPopup, 9000);
    return () => clearTimeout(t);
  }, [popup]);
  if (!popup) return null;
  const urgent = popup.priority === 'urgent';
  return (
    <View pointerEvents="box-none" style={[styles.popupWrap, { top: insets.top + 8 }]}>
      <Animated.View key={popup.id} entering={FadeInUp.springify().damping(18)} exiting={FadeOutUp.duration(180)}
        style={[styles.popup, urgent && styles.popupUrgent]}>
        <Pressable onPress={openAnnouncements} style={styles.popupBody} accessibilityRole="alert" accessibilityHint="Opens all announcements">
          <View style={[styles.icon, urgent && styles.iconUrgent]}>
            {urgent ? <Siren size={15} color={toneFg(RED)} /> : <Megaphone size={15} color={theme.colors.brand[700]} />}
          </View>
          <View style={{ flex: 1 }}>
            <Text style={[styles.label, urgent && { color: toneFg(RED) }]}>{urgent ? 'URGENT · NEW ANNOUNCEMENT' : 'NEW ANNOUNCEMENT'}</Text>
            <Text style={styles.message} numberOfLines={3}>{popup.message}</Text>
          </View>
        </Pressable>
        <Pressable onPress={dismissPopup} hitSlop={10} style={styles.close} accessibilityLabel="Close">
          <X size={15} color={theme.colors.text.muted} />
        </Pressable>
      </Animated.View>
    </View>
  );
}

function AnnouncementsDropdown() {
  const s = useBroadcasts();
  const insets = useSafeAreaInsets();
  const { width, height } = useWindowDimensions();
  const live = s.items.filter((b) => b.isActive);
  const ended = s.items.filter((b) => !b.isActive);
  // In-app overlay (not an RN <Modal>, where taps were unreliable on Android). It's mounted at the
  // app layout above the tabs, so it covers the whole window; fades in with a small rise.
  const motion = useSheetMotion({ open: s.menuOpen, onClose: closeAnnouncements, kind: 'fade' });
  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.28)" zIndex={60} backdropLabel="Close announcements">
      <Animated.View
        style={[styles.panel, { top: insets.top + 58, width: Math.min(380, width - 24), maxHeight: height * 0.7 }, motion.sheetStyle]}>
        <View style={styles.panelHead}>
          <Text style={styles.panelTitle}>Announcements</Text>
          <View style={[styles.liveDot, { backgroundColor: s.live ? theme.colors.semantic.success : theme.colors.text.muted }]} />
          <Text style={styles.liveText}>{s.live ? 'Live' : 'Reconnecting…'}</Text>
          <View style={{ flex: 1 }} />
          <Pressable onPress={motion.dismiss} hitSlop={10} style={styles.close} accessibilityLabel="Close">
            <X size={16} color={theme.colors.text.muted} />
          </Pressable>
        </View>
        {!s.items.length ? (
          <View style={styles.empty}>
            <Megaphone size={22} color={theme.colors.text.muted} />
            <Text style={styles.emptyText}>{s.loaded ? 'No announcements yet. New ones show up here the moment they’re sent.' : 'Loading…'}</Text>
          </View>
        ) : (
          <ScrollView contentContainerStyle={{ padding: 12, gap: 8 }} showsVerticalScrollIndicator={false}>
            {live.map((b, i) => <Row key={b.id} b={b} index={i} isNew={isNewer(b, s.menuSince)} />)}
            {ended.length ? <Text style={styles.section}>ENDED</Text> : null}
            {ended.map((b, i) => <Row key={b.id} b={b} index={live.length + i} isNew={false} />)}
          </ScrollView>
        )}
      </Animated.View>
    </SheetOverlay>
  );
}

const isNewer = (b: Broadcast, since: string | null) => !since || new Date(b.createdAt ?? 0).getTime() > new Date(since).getTime();

function Row({ b, index, isNew }: { b: Broadcast; index: number; isNew: boolean }) {
  const [open, setOpen] = useState(false);
  const urgent = b.priority === 'urgent' && b.isActive;
  const long = b.message.length > 160;
  return (
    <Animated.View entering={FadeIn.delay(Math.min(index, 8) * 40).duration(200)}
      style={[styles.row, urgent && styles.cardUrgent, !b.isActive && styles.rowEnded]}>
      <View style={[styles.icon, urgent && styles.iconUrgent]}>
        {urgent ? <Siren size={14} color={toneFg(RED)} /> : <Megaphone size={14} color={b.isActive ? theme.colors.brand[700] : theme.colors.text.muted} />}
      </View>
      <Pressable style={{ flex: 1 }} onPress={long ? () => setOpen((o) => !o) : undefined} disabled={!long}>
        <View style={styles.rowMeta}>
          <Text style={[styles.label, urgent && { color: toneFg(RED) }, !b.isActive && { color: theme.colors.text.muted }]}>
            {urgent ? 'URGENT' : 'TEAM'} · {ago(b.createdAt)}{!b.isActive && b.endedAt ? ` · ended ${ago(b.endedAt)}` : ''}
          </Text>
          {isNew ? <View style={styles.newTag}><Text style={styles.newText}>NEW</Text></View> : null}
        </View>
        <Text style={[styles.message, !b.isActive && { color: theme.colors.text.body, fontWeight: '500' }]} numberOfLines={open ? undefined : 4}>{b.message}</Text>
        {long ? <Text style={[styles.more, urgent && { color: toneFg(RED) }]}>{open ? 'Show less' : 'Show more'}</Text> : null}
      </Pressable>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  bell: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  badge: { position: 'absolute', top: -3, right: -3, minWidth: 18, height: 18, paddingHorizontal: 4, borderRadius: 9, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.danger, borderWidth: 2, borderColor: theme.colors.bg.app },
  badgeText: { fontSize: 10, fontWeight: '800', color: '#ffffff' },

  popupWrap: { position: 'absolute', left: 12, right: 12, zIndex: 50, elevation: 50 },
  popup: { flexDirection: 'row', alignItems: 'flex-start', gap: 6, padding: 12, borderRadius: 16, backgroundColor: theme.colors.bg.surface, borderWidth: 1, borderColor: theme.colors.brand[100],
    shadowColor: '#000', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.16, shadowRadius: 18, elevation: 12 },
  popupUrgent: { borderColor: toneBg('#fecaca'), backgroundColor: toneBg('#fef2f2') },
  popupBody: { flex: 1, flexDirection: 'row', alignItems: 'flex-start', gap: 10 },

  panel: { position: 'absolute', right: 12, borderRadius: 20, backgroundColor: theme.colors.bg.surface, borderWidth: 1, borderColor: theme.colors.bg.divider, overflow: 'hidden',
    shadowColor: '#000', shadowOffset: { width: 0, height: 12 }, shadowOpacity: 0.18, shadowRadius: 24, elevation: 16 },
  panelHead: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 14, paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: theme.colors.bg.divider },
  panelTitle: { fontSize: 16, fontWeight: '800', color: theme.colors.text.display, marginRight: 4 },
  liveDot: { width: 7, height: 7, borderRadius: 4 },
  liveText: { fontSize: 11.5, fontWeight: '700', color: theme.colors.text.muted },
  section: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.text.muted, marginTop: 6, marginLeft: 2 },
  empty: { alignItems: 'center', gap: 8, padding: 28 },
  emptyText: { fontSize: 13, lineHeight: 19, textAlign: 'center', color: theme.colors.text.muted },

  row: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, padding: 12, borderRadius: 14, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  rowEnded: { backgroundColor: theme.colors.bg.surfaceHover, borderColor: theme.colors.bg.divider },
  rowMeta: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  cardUrgent: { backgroundColor: toneBg('#fef2f2'), borderColor: toneBg('#fecaca') },
  icon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },
  iconUrgent: { backgroundColor: toneBg('#fee2e2') },
  label: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.brand[700], flexShrink: 1 },
  newTag: { paddingHorizontal: 6, paddingVertical: 1, borderRadius: 6, backgroundColor: theme.colors.brand[600] },
  newText: { fontSize: 9, fontWeight: '800', letterSpacing: 0.5, color: '#ffffff' },
  message: { fontSize: 14, lineHeight: 20, fontWeight: '600', color: theme.colors.text.display, marginTop: 2 },
  more: { fontSize: 12.5, fontWeight: '800', color: theme.colors.brand[700], marginTop: 4 },
  close: { width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
}));
