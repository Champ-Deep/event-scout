import React from 'react';
import { View, Pressable, Platform } from 'react-native';
import Animated from 'react-native-reanimated';
import { Text } from '../../../ui/Text';
import { theme, makeStyles } from '../../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';

export interface SheetOption {
  label: string;
  icon?: React.ReactNode;
  destructive?: boolean;
  onPress: () => void;
}

interface OptionSheetProps {
  visible: boolean;
  title: string;
  message?: string | undefined;
  options: SheetOption[];
  onClose: () => void;
}

/**
 * Cross-platform action sheet (Alert with several buttons doesn't work on web). Drawn in the app's
 * window layer over the whole screen and tab bar, not an RN <Modal> — taps in a Modal were unreliable on Android.
 * Fades in with a small rise; back button / backdrop / Cancel close it.
 */
export function OptionSheet({ visible, title, message, options, onClose }: OptionSheetProps) {
  const motion = useSheetMotion({ open: visible, onClose, kind: 'fade' });
  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.45)" zIndex={70} layer="window">
      <Animated.View style={[styles.sheet, motion.sheetStyle]}>
        <Text style={styles.title}>{title}</Text>
        {message ? <Text style={styles.message}>{message}</Text> : null}
        {options.map((opt) => (
          <Pressable
            key={opt.label}
            style={({ pressed }) => [styles.option, pressed && styles.optionPressed]}
            onPress={() => {
              onClose();
              // iOS can't present the image picker while this sheet is still dismissing.
              setTimeout(opt.onPress, Platform.OS === 'ios' ? 450 : 0);
            }}
            accessibilityRole="button"
          >
            {opt.icon ? <View style={styles.optionIcon}>{opt.icon}</View> : null}
            <Text style={[styles.optionText, opt.destructive && styles.destructive]}>{opt.label}</Text>
          </Pressable>
        ))}
        <Pressable style={({ pressed }) => [styles.cancel, pressed && styles.optionPressed]} onPress={motion.dismiss} accessibilityRole="button">
          <Text style={styles.cancelText}>Cancel</Text>
        </Pressable>
      </Animated.View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  sheet: { backgroundColor: theme.colors.bg.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: theme.spacing.lg, paddingBottom: 36 },
  title: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display, marginBottom: 4 },
  message: { fontSize: 14, color: theme.colors.text.body, lineHeight: 20, marginBottom: 8 },
  option: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 14, paddingHorizontal: 4, borderRadius: 12 },
  optionPressed: { backgroundColor: theme.colors.bg.surfaceHover },
  optionIcon: { width: 24, alignItems: 'center' },
  optionText: { fontSize: 16, fontWeight: '600', color: theme.colors.text.display },
  destructive: { color: theme.colors.semantic.danger },
  cancel: { marginTop: 8, paddingVertical: 14, alignItems: 'center', borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.bg.divider },
  cancelText: { fontSize: 15, fontWeight: '700', color: theme.colors.text.body },
}));
