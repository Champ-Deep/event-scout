import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Pressable, Linking } from 'react-native';
import ReanimatedSwipeable, { type SwipeableMethods } from 'react-native-gesture-handler/ReanimatedSwipeable';
import Animated, {
  useSharedValue, useAnimatedProps, useAnimatedStyle, withDelay, withTiming, interpolate, Extrapolation, Easing,
  type SharedValue,
} from 'react-native-reanimated';
import Svg, { Circle } from 'react-native-svg';
import { Phone, Mail, Trash2, CalendarDays, Copy } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../ui/Text';
import { theme, toneFg, makeStyles } from '../../../ui/theme';
import { PressableScale } from '../../onboarding/components';
import { scoreLevel } from '../scoreLevel';
import { ContactAvatar } from './ContactAvatar';

const ACTION_W = 76;
const AnimatedCircle = Animated.createAnimatedComponent(Circle);

/** "just now" · "5m ago" · "2h ago" · "yesterday" · "3d ago" · "12 Sep" */
export function metAgo(iso?: string | null): string | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const mins = Math.floor((Date.now() - t) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return 'yesterday';
  if (days < 7) return `${days}d ago`;
  return new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
}

/** Score ring that draws itself to the score, on a tinted disc in the level's colour; a faint
 *  empty ring with "—" when not scored. */
function ScoreRing({ score, color, fill, textColor, delay }: {
  score: number | null; color: string; fill?: string | undefined; textColor?: string | undefined; delay: number;
}) {
  const size = 46, stroke = 5, r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const p = useSharedValue(0);
  useEffect(() => {
    p.value = score == null ? 0 : withDelay(delay, withTiming(score / 100, { duration: 800, easing: Easing.out(Easing.cubic) }));
  }, [score]);
  const props = useAnimatedProps(() => ({ strokeDashoffset: c * (1 - p.value) }));
  return (
    <View style={styles.ring}>
      <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
        <Circle cx={size / 2} cy={size / 2} r={r} stroke={score != null ? `${color}33` : theme.colors.bg.divider} strokeWidth={stroke}
          fill={score != null && fill ? fill : 'none'} />
        {score != null ? (
          <AnimatedCircle cx={size / 2} cy={size / 2} r={r} stroke={color} strokeWidth={stroke} fill="none" strokeLinecap="round"
            strokeDasharray={`${c} ${c}`} animatedProps={props} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
        ) : null}
      </Svg>
      <Text style={[styles.ringText, score == null ? { color: theme.colors.text.muted } : textColor ? { color: textColor } : null]}>{score ?? '—'}</Text>
    </View>
  );
}

/** One swipe action: grows in as the card is dragged open. */
function SwipeAction({ icon, label, color, onPress, progress, accessibilityLabel }: {
  icon: React.ReactNode; label: string; color: string; onPress: () => void; progress: SharedValue<number>; accessibilityLabel: string;
}) {
  const style = useAnimatedStyle(() => ({
    opacity: interpolate(progress.value, [0, 0.6, 1], [0, 0.6, 1], Extrapolation.CLAMP),
    transform: [{ scale: interpolate(progress.value, [0, 1], [0.7, 1], Extrapolation.CLAMP) }],
  }));
  return (
    <Pressable onPress={onPress} style={[styles.action, { backgroundColor: color }]} accessibilityRole="button" accessibilityLabel={accessibilityLabel}>
      <Animated.View style={[styles.actionInner, style]}>
        {icon}
        <Text style={styles.actionText}>{label}</Text>
      </Animated.View>
    </Pressable>
  );
}

/**
 * Contact card for the Contacts list: avatar, name + level tag, title · company, the event and
 * when you met, and a score ring. Swipe right for Call / Email, left for Delete (moves it to
 * Recently deleted; the list shows the Undo toast). Actions are revealed then tapped, never fired
 * by the swipe alone, so a sloppy scroll can't delete anyone.
 */
export function ContactCard({ contact: c, index, duplicate, onOpen, onDelete, onSwipeOpen }: {
  contact: Contact;
  index: number;
  duplicate?: boolean;
  onOpen: () => void;
  onDelete: () => void;
  /** Called with this card's controls when it opens, so the list can close the previous one. */
  onSwipeOpen?: (methods: SwipeableMethods) => void;
}) {
  const ref = useRef<SwipeableMethods>(null);
  const name = [c.firstName, c.lastName].filter(Boolean).join(' ') || 'Unnamed';
  const sub = [c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No company yet';
  const level = scoreLevel(c.score);
  const phone = (c.phone || '').replace(/[^\d+]/g, '');
  const when = metAgo(c.createdAt);
  const close = () => ref.current?.close();

  const left = (progress: SharedValue<number>) => (phone || c.email) ? (
    <View style={styles.actionsLeft}>
      {phone ? (
        <SwipeAction progress={progress} color={theme.colors.semantic.successText} label="Call" accessibilityLabel={`Call ${name}`}
          icon={<Phone size={18} color="white" />} onPress={() => { close(); Linking.openURL(`tel:${phone}`); }} />
      ) : null}
      {c.email ? (
        <SwipeAction progress={progress} color={theme.colors.brand[600]} label="Email" accessibilityLabel={`Email ${name}`}
          icon={<Mail size={18} color="white" />} onPress={() => { close(); Linking.openURL(`mailto:${c.email}`); }} />
      ) : null}
    </View>
  ) : null;

  const right = (progress: SharedValue<number>) => (
    <View style={styles.actionsRight}>
      <SwipeAction progress={progress} color={theme.colors.semantic.danger} label="Delete" accessibilityLabel={`Delete ${name}`}
        icon={<Trash2 size={18} color="white" />} onPress={() => { close(); onDelete(); }} />
    </View>
  );

  return (
    <ReanimatedSwipeable
      ref={ref}
      friction={1.6}
      leftThreshold={ACTION_W / 2}
      rightThreshold={ACTION_W / 2}
      overshootLeft={false}
      overshootRight={false}
      renderLeftActions={(p) => left(p)}
      renderRightActions={(p) => right(p)}
      onSwipeableWillOpen={() => { if (ref.current) onSwipeOpen?.(ref.current); }}
      containerStyle={styles.swipe}
    >
      <PressableScale
        // Hot and warm leads get a border in their colour so they stand out in the list.
        style={[styles.card, level && (level.key === 'hot' || level.key === 'warm') && { borderColor: `${level.ring}80`, borderWidth: 1.5 }]}
        onPress={onOpen}
        scaleTo={0.98}
        accessibilityRole="button"
        accessibilityLabel={`${name}${level ? `, ${level.name} lead, score ${c.score}` : ''}`}
        accessibilityHint="Swipe right to call or email, left to delete"
        accessibilityActions={[
          ...(phone ? [{ name: 'call', label: 'Call' }] : []),
          ...(c.email ? [{ name: 'email', label: 'Email' }] : []),
          { name: 'delete', label: 'Delete' },
        ]}
        onAccessibilityAction={(e) => {
          if (e.nativeEvent.actionName === 'call') Linking.openURL(`tel:${phone}`);
          else if (e.nativeEvent.actionName === 'email') Linking.openURL(`mailto:${c.email}`);
          else if (e.nativeEvent.actionName === 'delete') onDelete();
        }}
      >
        {level ? <View style={[styles.accent, { backgroundColor: level.ring }]} /> : null}
        <ContactAvatar firstName={c.firstName} lastName={c.lastName} photoUrl={c.photoUrl} size={46} />
        <View style={styles.info}>
          <View style={styles.nameRow}>
            <Text style={styles.name} numberOfLines={1}>{name}</Text>
            {level && level.key !== 'cold' ? (
              <View style={[styles.tag, { backgroundColor: level.ring }]}>
                <Text style={styles.tagText}>{level.label}</Text>
              </View>
            ) : null}
            {duplicate ? (
              <View style={styles.dupDot} accessibilityLabel="Possible duplicate">
                <Copy size={10} color={theme.colors.semantic.warning} strokeWidth={2.5} />
              </View>
            ) : null}
          </View>
          <Text style={styles.sub} numberOfLines={1}>{sub}</Text>
          {c.eventId || when ? (
            <View style={styles.metaRow}>
              {c.eventId ? (
                <>
                  <CalendarDays size={11} color={theme.colors.brand[600]} />
                  <Text style={[styles.meta, styles.metaEvent]} numberOfLines={1}>{c.eventId}</Text>
                </>
              ) : null}
              {c.eventId && when ? <Text style={styles.meta}>·</Text> : null}
              {when ? <Text style={[styles.meta, styles.metaWhen]}>{when}</Text> : null}
            </View>
          ) : null}
        </View>
        <ScoreRing score={c.score ?? null} color={level?.ring ?? toneFg('#94a3b8')} fill={level?.bg} textColor={level?.fg}
          delay={120 + Math.min(index, 12) * 35} />
      </PressableScale>
    </ReanimatedSwipeable>
  );
}

const styles = makeStyles(() => ({
  swipe: { borderRadius: 18, marginBottom: 10, backgroundColor: 'transparent' },
  card: {
    flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: theme.colors.bg.surface,
    borderRadius: 18, paddingVertical: 12, paddingLeft: 14, paddingRight: 12,
    borderWidth: 1, borderColor: theme.colors.bg.divider, overflow: 'hidden', ...theme.shadows.card,
  },
  accent: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 4 },
  info: { flex: 1, gap: 2 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  name: { fontSize: 15.5, fontWeight: '800', color: theme.colors.text.display, flexShrink: 1 },
  tag: { paddingHorizontal: 7, paddingVertical: 2, borderRadius: 6 },
  tagText: { fontSize: 9.5, fontWeight: '900', letterSpacing: 0.7, color: 'white' },
  dupDot: { width: 18, height: 18, borderRadius: 9, backgroundColor: theme.colors.semantic.warningBg, alignItems: 'center', justifyContent: 'center' },
  sub: { fontSize: 13, color: theme.colors.text.body },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 3 },
  meta: { fontSize: 11.5, color: theme.colors.text.muted, flexShrink: 1 },
  metaEvent: { color: theme.colors.brand[700], fontWeight: '700' },
  metaWhen: { flexShrink: 0 },
  ring: { width: 46, height: 46, alignItems: 'center', justifyContent: 'center' },
  ringText: { fontSize: 14, fontWeight: '900', color: theme.colors.text.display },

  // Flush with the card (a margin here isn't counted in the open distance, so the card would stop short).
  actionsLeft: { flexDirection: 'row', borderTopLeftRadius: 18, borderBottomLeftRadius: 18, overflow: 'hidden' },
  actionsRight: { flexDirection: 'row', borderTopRightRadius: 18, borderBottomRightRadius: 18, overflow: 'hidden' },
  action: { width: ACTION_W, alignItems: 'center', justifyContent: 'center' },
  actionInner: { alignItems: 'center', gap: 4 },
  actionText: { fontSize: 11, fontWeight: '800', color: 'white' },
}));
