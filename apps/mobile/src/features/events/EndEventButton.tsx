import React from 'react';
import { ActivityIndicator } from 'react-native';
import Animated, { FadeIn, ZoomIn, ZoomOut } from 'react-native-reanimated';
import { Flag, RotateCcw } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, makeStyles } from '../../ui/theme';
import { PressableScale } from '../onboarding/components';

/** "End event" / "Reopen event" — ends the event for you (it stays in your list, marked ended). */
export function EndEventButton({ ended, busy, onPress, compact }: { ended: boolean; busy?: boolean; onPress: () => void; compact?: boolean }) {
  const color = ended ? theme.colors.brand[700] : theme.colors.semantic.danger;
  return (
    <Animated.View key={ended ? 'reopen' : 'end'} entering={FadeIn.duration(200)}>
      <PressableScale onPress={onPress} disabled={busy} scaleTo={0.94} style={[styles.btn, compact && styles.compact, ended ? styles.reopen : styles.end]}
        accessibilityLabel={ended ? 'Reopen this event' : 'End this event'}
        leadingIcon={busy ? <ActivityIndicator size="small" color={color} />
          : <Animated.View entering={ZoomIn.springify().damping(14)} exiting={ZoomOut.duration(120)}>
            {ended ? <RotateCcw size={compact ? 12 : 14} color={color} /> : <Flag size={compact ? 12 : 14} color={color} />}
          </Animated.View>}>
        <Text style={[styles.text, compact && styles.textCompact, { color }]}>{ended ? (compact ? 'Reopen' : 'Reopen event') : (compact ? 'End' : 'End event')}</Text>
      </PressableScale>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  btn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999, borderWidth: 1.5 },
  compact: { paddingHorizontal: 10, paddingVertical: 5, gap: 4 },
  end: { borderColor: theme.colors.semantic.dangerBorder, backgroundColor: toneBg('#ffffff') },
  reopen: { borderColor: theme.colors.brand[200], backgroundColor: theme.colors.brand[50] },
  text: { fontSize: 13, fontWeight: '800' },
  textCompact: { fontSize: 12 },
}));
