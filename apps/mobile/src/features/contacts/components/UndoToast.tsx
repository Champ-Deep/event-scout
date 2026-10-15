import React, { useEffect } from 'react';
import { View, Pressable, ActivityIndicator } from 'react-native';
import Animated, { FadeInDown, FadeOutDown, useSharedValue, useAnimatedStyle, withTiming, Easing } from 'react-native-reanimated';
import { Trash2, RotateCcw, Check } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, makeStyles } from '../../../ui/theme';

export interface ToastState {
  kind: 'deleted' | 'restored' | 'error';
  message: string;
  contactId?: string;
}

interface Props {
  toast: ToastState | null;
  onUndo?: (() => void) | undefined;
  undoing?: boolean;
  onDismiss: () => void;
  durationMs?: number;
}

/** Bottom toast with an optional Undo action and a shrinking countdown bar. */
export function UndoToast({ toast, onUndo, undoing, onDismiss, durationMs = 6000 }: Props) {
  const progress = useSharedValue(1);

  useEffect(() => {
    if (!toast) return;
    progress.value = 1;
    progress.value = withTiming(0, { duration: durationMs, easing: Easing.linear });
    const t = setTimeout(onDismiss, durationMs);
    return () => clearTimeout(t);
  }, [toast]);

  const barStyle = useAnimatedStyle(() => ({ width: `${progress.value * 100}%` }));

  if (!toast) return null;
  const Icon = toast.kind === 'deleted' ? Trash2 : toast.kind === 'restored' ? Check : Trash2;

  return (
    <Animated.View
      key={`${toast.kind}-${toast.contactId}-${toast.message}`}
      entering={FadeInDown.duration(260)}
      exiting={FadeOutDown.duration(180)}
      style={styles.wrap}
      pointerEvents="box-none"
    >
      <View style={[styles.toast, toast.kind === 'error' && styles.toastError]}>
        <Icon size={16} color="white" />
        <Text style={styles.text} numberOfLines={2}>{toast.message}</Text>
        {onUndo && toast.kind === 'deleted' ? (
          <Pressable onPress={onUndo} disabled={undoing} hitSlop={8} style={styles.undo} accessibilityRole="button">
            {undoing ? <ActivityIndicator size="small" color={theme.colors.fg.brand300} /> : <RotateCcw size={14} color={theme.colors.fg.brand300} strokeWidth={2.5} />}
            <Text style={styles.undoText}>Undo</Text>
          </Pressable>
        ) : null}
        <Animated.View style={[styles.bar, barStyle]} />
      </View>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  wrap: { position: 'absolute', left: 16, right: 16, bottom: 100 },
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: theme.colors.fill.ink,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
    overflow: 'hidden',
    ...theme.shadows.active,
  },
  toastError: { backgroundColor: theme.colors.semantic.danger },
  text: { flex: 1, color: 'white', fontSize: 14, fontWeight: '600' },
  undo: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 6 },
  undoText: { color: theme.colors.fg.brand300, fontSize: 14, fontWeight: '800' },
  bar: { position: 'absolute', left: 0, bottom: 0, height: 3, backgroundColor: theme.colors.brand[400] },
}));
