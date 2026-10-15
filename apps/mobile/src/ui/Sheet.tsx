import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, ScrollView, Pressable, ActivityIndicator } from 'react-native';
import { GestureDetector } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated from 'react-native-reanimated';
import { X } from 'lucide-react-native';
import { Text } from './Text';
import { theme, toneBg, makeStyles } from './theme';
import { useSheetMotion, SheetOverlay } from './SheetOverlay';

const TOP_GAP = 12; // space kept above the sheet, below the status bar

/**
 * Bottom sheet drawn inside the screen (not an RN <Modal>): inside a Modal with entering
 * animations the ✕ / Save taps didn't register on Android (same issue the share dialog had).
 * Swipe the handle/title down, tap the backdrop, ✕ or the Android back button to close.
 * Render it last in the screen so it sits above everything else. Motion: useSheetMotion.
 *
 * It sizes itself to the space it's actually drawn in (flex, not the window height): on a screen
 * above a tab bar a window-sized sheet overflowed its container and pushed the ✕ out of reach.
 * Closing always slides out — whether from ✕, a swipe, back, or the parent setting visible=false.
 *
 * `header` stays put above the scrolling body (search boxes, filters); `tall` opens it at full
 * height so it doesn't jump while a list filters; `deferBody` renders the body once the sheet has
 * slid in, so a long list doesn't stutter the opening animation.
 */
export function Sheet({ visible, title, icon, onClose, header, tall, deferBody, children }: {
  visible: boolean; title: string; icon: React.ReactNode; onClose: () => void;
  header?: React.ReactNode; tall?: boolean; deferBody?: boolean; children: React.ReactNode;
}) {
  const insets = useSafeAreaInsets();
  const motion = useSheetMotion({ open: visible, onClose });
  const [bodyReady, setBodyReady] = useState(!deferBody);
  // Where the overlay sits in the window: only the part under the status bar needs extra room.
  const probe = useRef<View>(null);
  const [topInset, setTopInset] = useState(insets.top);

  useEffect(() => {
    if (!visible || !deferBody) return undefined;
    setBodyReady(false);
    const t = setTimeout(() => setBodyReady(true), 240);
    return () => clearTimeout(t);
  }, [visible]);

  return (
    <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.45)" zIndex={50} keyboardAware>
      <View ref={probe} collapsable={false} pointerEvents="none" style={StyleSheet.absoluteFill}
        onLayout={() => probe.current?.measureInWindow((_x, y) => setTopInset(Math.max(0, insets.top - y)))} />
      <Animated.View style={[styles.sheet, { marginTop: topInset + TOP_GAP, paddingBottom: Math.max(insets.bottom, 16) }, tall && { flex: 1 }, motion.sheetStyle]}>
        <View style={styles.topRow}>
          {/* Drag area; the ✕ sits outside it so the swipe gesture can't swallow its tap. */}
          <GestureDetector gesture={motion.pan}>
            <View collapsable={false} style={{ flex: 1 }}>
              <View style={styles.handleArea}><View style={styles.handle} /></View>
              <View style={styles.header}>
                <View style={styles.icon}>{icon}</View>
                <Text style={styles.title} numberOfLines={1}>{title}</Text>
              </View>
            </View>
          </GestureDetector>
          <Pressable onPress={motion.dismiss} hitSlop={12} style={({ pressed }) => [styles.closeBtn, pressed && { opacity: 0.6, transform: [{ scale: 0.92 }] }]}
            accessibilityRole="button" accessibilityLabel="Close">
            <X size={18} color={theme.colors.text.body} />
          </Pressable>
        </View>
        {header ? <View style={styles.fixedHeader}>{header}</View> : null}
        {bodyReady ? (
          <ScrollView style={tall ? { flex: 1 } : { flexShrink: 1 }} keyboardShouldPersistTaps="handled" keyboardDismissMode="on-drag"
            contentContainerStyle={{ paddingHorizontal: 16, paddingTop: header ? 4 : 0 }}>
            {children}
          </ScrollView>
        ) : (
          <View style={styles.loading}><ActivityIndicator color={theme.colors.brand[600]} /></View>
        )}
      </Animated.View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  topRow: { flexDirection: 'row', alignItems: 'flex-end', paddingRight: 16 },
  sheet: { backgroundColor: theme.colors.bg.app, borderTopLeftRadius: 26, borderTopRightRadius: 26, flexShrink: 1, overflow: 'hidden' },
  handleArea: { alignItems: 'center', paddingTop: 10, paddingBottom: 4 },
  handle: { width: 42, height: 5, borderRadius: 3, backgroundColor: toneBg('#cbd5e1') },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingVertical: 12 },
  icon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  title: { flex: 1, fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  closeBtn: { width: 34, height: 34, borderRadius: 17, marginBottom: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider },
  fixedHeader: { paddingHorizontal: 16, paddingBottom: 6, borderBottomWidth: 1, borderBottomColor: theme.colors.bg.divider },
  loading: { paddingVertical: 32, alignItems: 'center' },
}));
