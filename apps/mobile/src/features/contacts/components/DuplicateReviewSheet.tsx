import React, { useState } from 'react';
import { View, Pressable, ScrollView, ActivityIndicator } from 'react-native';
import Animated, { FadeInDown, FadeOut, LinearTransition } from 'react-native-reanimated';
import { Copy, Check, X } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../ui/Text';
import { theme, makeStyles } from '../../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';
import { api, errorMessage, type DuplicatePair } from '../../../lib/api';
import { describeReasons } from '..';
import { ContactAvatar } from './ContactAvatar';
import { PressableScale } from '../../onboarding/components';

interface Props {
  visible: boolean;
  pairs: DuplicatePair[];
  onClose: () => void;
  onOpenContact: (id: string) => void;
  onChanged: () => void;
}

function MiniContact({ c, onPress }: { c: Contact; onPress: () => void }) {
  const name = [c.firstName, c.lastName].filter(Boolean).join(' ') || 'Unnamed';
  return (
    <PressableScale style={styles.mini} containerStyle={{ flex: 1 }} onPress={onPress} scaleTo={0.97} accessibilityRole="button">
      <ContactAvatar firstName={c.firstName} lastName={c.lastName} photoUrl={c.photoUrl} size={36} />
      <View style={{ flex: 1 }}>
        <Text style={styles.miniName} numberOfLines={1}>{name}</Text>
        <Text style={styles.miniSub} numberOfLines={1}>{c.email || c.phone || c.company || 'No details'}</Text>
      </View>
    </PressableScale>
  );
}

/**
 * Lists pairs of contacts that look like the same person so the user can check them.
 * Drawn in the app's window layer (covers the tab bar), not an RN <Modal> — taps in a Modal were unreliable on Android.
 */
export function DuplicateReviewSheet({ visible, pairs, onClose, onOpenContact, onChanged }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [error, setError] = useState('');
  const motion = useSheetMotion({ open: visible, onClose });

  const keyOf = (p: DuplicatePair) => `${p.a.id}:${p.b.id}`;
  const shown = pairs.filter((p) => !hidden.has(keyOf(p)));

  const notDuplicates = async (p: DuplicatePair) => {
    setBusy(keyOf(p));
    setError('');
    try {
      await api.dismissDuplicate(p.a.id, p.b.id);
      setHidden((prev) => new Set(prev).add(keyOf(p)));
      onChanged();
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't save that. Please try again."));
    } finally {
      setBusy(null);
    }
  };

  const open = (id: string) => {
    onClose();
    setTimeout(() => onOpenContact(id), 250);
  };

  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.45)" zIndex={60} layer="window">
        <Animated.View style={[styles.sheet, motion.sheetStyle]}>
          <View style={styles.header}>
            <View style={styles.headerIcon}><Copy size={16} color={theme.colors.semantic.warning} /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.title}>Possible duplicates</Text>
              <Text style={styles.sub}>Check each pair. Open a contact to edit or delete the extra one.</Text>
            </View>
            <Pressable onPress={motion.dismiss} hitSlop={10} accessibilityLabel="Close"><X size={20} color={theme.colors.text.muted} /></Pressable>
          </View>

          {error ? <Text style={styles.error}>{error}</Text> : null}

          <ScrollView style={{ maxHeight: 460 }} contentContainerStyle={{ paddingBottom: 8 }}>
            {shown.length === 0 ? (
              <Animated.View entering={FadeInDown} style={styles.done}>
                <Check size={22} color={theme.colors.semantic.successText} />
                <Text style={styles.doneText}>All reviewed — no duplicates left.</Text>
              </Animated.View>
            ) : (
              shown.map((p, i) => (
                <Animated.View
                  key={keyOf(p)}
                  entering={FadeInDown.delay(i * 60).duration(260)}
                  exiting={FadeOut.duration(180)}
                  layout={LinearTransition.duration(260)}
                  style={styles.pair}
                >
                  <Text style={styles.reason}>{describeReasons(p.reasons)}</Text>
                  <View style={styles.row}>
                    <MiniContact c={p.a} onPress={() => open(p.a.id)} />
                    <MiniContact c={p.b} onPress={() => open(p.b.id)} />
                  </View>
                  <PressableScale
                    style={styles.dismiss}
                    onPress={() => notDuplicates(p)}
                    disabled={busy === keyOf(p)}
                    leadingIcon={busy === keyOf(p) ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Check size={14} color={theme.colors.brand[700]} strokeWidth={2.5} />}
                    accessibilityRole="button"
                  >
                    <Text style={styles.dismissText}>Different people</Text>
                  </PressableScale>
                </Animated.View>
              ))
            )}
          </ScrollView>
        </Animated.View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  sheet: { backgroundColor: theme.colors.bg.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: theme.spacing.lg, paddingBottom: 32 },
  header: { flexDirection: 'row', alignItems: 'flex-start', gap: 12, marginBottom: 12 },
  headerIcon: { width: 32, height: 32, borderRadius: 10, backgroundColor: theme.colors.semantic.warningBg, alignItems: 'center', justifyContent: 'center' },
  title: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  sub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2, lineHeight: 18 },
  error: { fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', marginBottom: 8 },
  pair: { borderWidth: 1, borderColor: theme.colors.bg.divider, borderRadius: 16, padding: 12, marginBottom: 10, backgroundColor: theme.colors.bg.surfaceHover },
  reason: { fontSize: 11, fontWeight: '800', color: theme.colors.semantic.warning, letterSpacing: 0.5, textTransform: 'uppercase', marginBottom: 8 },
  row: { flexDirection: 'row', gap: 8 },
  mini: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.bg.surface, borderRadius: 12, padding: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  miniName: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display },
  miniSub: { fontSize: 11, color: theme.colors.text.muted },
  dismiss: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', marginTop: 10, backgroundColor: theme.colors.brand[50], paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999 },
  dismissText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  done: { alignItems: 'center', gap: 8, paddingVertical: 24 },
  doneText: { fontSize: 14, color: theme.colors.semantic.successText, fontWeight: '600' },
}));
