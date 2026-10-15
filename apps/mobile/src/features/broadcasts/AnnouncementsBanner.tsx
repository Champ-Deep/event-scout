import React, { useEffect, useState } from 'react';
import { View, Pressable, Platform } from 'react-native';
import Animated, { FadeInDown, FadeOut, LinearTransition } from 'react-native-reanimated';
import { Megaphone, Siren, X } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../ui/theme';
import { useBroadcasts, refreshBroadcasts, ago } from './store';

// Hidden announcements are remembered on this phone only (only admins can end one for everybody).
const KEY = 'eventscout.broadcasts.hidden';
function readHidden(): string[] {
  try {
    const raw = Platform.OS === 'web' ? window.localStorage.getItem(KEY) : require('expo-secure-store').getItem(KEY);
    const v = raw ? JSON.parse(raw) : [];
    return Array.isArray(v) ? v : [];
  } catch { return []; }
}
function writeHidden(ids: string[]) {
  const raw = JSON.stringify(ids.slice(-100));
  try {
    if (Platform.OS === 'web') window.localStorage.setItem(KEY, raw);
    else require('expo-secure-store').setItem(KEY, raw);
  } catch { /* not remembered: it shows again next time */ }
}

/** Live team announcements from the admin console (shared store, updated the moment one is sent or
 *  ended). `refreshKey` changing reloads them too (pull to refresh). */
export function AnnouncementsBanner({ refreshKey }: { refreshKey?: number }) {
  const items = useBroadcasts().items.filter((b) => b.isActive);
  const [hidden, setHidden] = useState<string[]>(() => readHidden());
  // announcements can be long (up to 2,000 characters): show 4 lines, "Show more" for the rest
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [long, setLong] = useState<Record<string, boolean>>({});

  useEffect(() => { if (refreshKey) refreshBroadcasts(); }, [refreshKey]);

  const hide = (id: string) => {
    const next = [...hidden, id];
    setHidden(next);
    writeHidden(next);
  };

  const shown = items.filter((b) => !hidden.includes(b.id)).slice(0, 3);
  if (!shown.length) return null;
  return (
    <View style={styles.wrap}>
      {shown.map((b, i) => {
        const urgent = b.priority === 'urgent';
        return (
          <Animated.View key={b.id} entering={FadeInDown.delay(i * 60).duration(280)} exiting={FadeOut.duration(180)} layout={LinearTransition.duration(240)}
            style={[styles.card, urgent && styles.cardUrgent]} accessibilityRole="alert">
            <View style={[styles.icon, urgent && styles.iconUrgent]}>
              {urgent ? <Siren size={15} color={toneFg('#b91c1c')} /> : <Megaphone size={15} color={theme.colors.brand[700]} />}
            </View>
            <View style={{ flex: 1 }}>
              <Text style={[styles.label, urgent && { color: toneFg('#b91c1c') }]}>{urgent ? 'URGENT · TEAM' : 'TEAM ANNOUNCEMENT'} · {ago(b.createdAt)}</Text>
              <Text style={styles.message} numberOfLines={open[b.id] ? undefined : 4}
                onTextLayout={(e) => { if (!long[b.id] && e.nativeEvent.lines.length > 4) setLong((m) => ({ ...m, [b.id]: true })); }}>
                {b.message}
              </Text>
              {long[b.id] || b.message.length > 220 ? (
                <Pressable onPress={() => setOpen((m) => ({ ...m, [b.id]: !m[b.id] }))} hitSlop={8} accessibilityRole="button">
                  <Text style={[styles.more, urgent && { color: toneFg('#b91c1c') }]}>{open[b.id] ? 'Show less' : 'Show more'}</Text>
                </Pressable>
              ) : null}
            </View>
            <Pressable onPress={() => hide(b.id)} hitSlop={10} style={styles.close} accessibilityLabel="Hide this announcement">
              <X size={15} color={theme.colors.text.muted} />
            </Pressable>
          </Animated.View>
        );
      })}
    </View>
  );
}

const styles = makeStyles(() => ({
  wrap: { gap: 8, marginBottom: 12 },
  card: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, padding: 12, borderRadius: 16, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  cardUrgent: { backgroundColor: toneBg('#fef2f2'), borderColor: toneBg('#fecaca') },
  icon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },
  iconUrgent: { backgroundColor: toneBg('#fee2e2') },
  label: { fontSize: 10, fontWeight: '800', letterSpacing: 0.6, color: theme.colors.brand[700] },
  message: { fontSize: 14, lineHeight: 20, fontWeight: '600', color: theme.colors.text.display, marginTop: 2 },
  more: { fontSize: 12.5, fontWeight: '800', color: theme.colors.brand[700], marginTop: 4 },
  close: { width: 26, height: 26, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
}));
