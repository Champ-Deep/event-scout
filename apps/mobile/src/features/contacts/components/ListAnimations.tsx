import React, { useEffect, useRef, useState } from 'react';
import { View, TextInput, Pressable, type TextInputProps } from 'react-native';
import Animated, { FadeIn, FadeOut, ZoomIn, ZoomOut, useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withSpring, interpolateColor, Easing } from 'react-native-reanimated';
import { Search, X, ArrowUp } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';

/** Counts up/down to `value` (e.g. "12 contacts" ticking up when the list loads). */
// Counts on the JS thread with requestAnimationFrame. (The worklet version relied on
// runOnJS from an animated reaction, which never fired here — the number stayed blank/0.)
export function AnimatedCount({ value, style }: { value: number; style?: any }) {
  const [shown, setShown] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    const start = from.current;
    const end = Number.isFinite(value) ? value : 0;
    if (start === end) { setShown(end); return; }
    const t0 = Date.now();
    let raf = 0;
    const step = () => {
      const p = Math.min(1, (Date.now() - t0) / 650);
      const eased = 1 - Math.pow(1 - p, 3);
      setShown(Math.round(start + (end - start) * eased));
      if (p < 1) raf = requestAnimationFrame(step);
      else from.current = end;
    };
    from.current = start;
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); from.current = end; };
  }, [value]);
  return <Text style={style}>{shown}</Text>;
}

/** Placeholder rows that shimmer while contacts load (instead of a lone spinner). */
export function SkeletonRows({ count = 6 }: { count?: number }) {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withSequence(withTiming(1, { duration: 700 }), withTiming(0, { duration: 700 })), -1);
  }, []);
  const pulse = useAnimatedStyle(() => ({ opacity: 0.45 + t.value * 0.4 }));
  return (
    <View>
      {Array.from({ length: count }).map((_, i) => (
        // Entering animation on the wrapper, pulsing opacity on the inner view: putting both on
        // one view made Reanimated warn "opacity may be overwritten by a layout animation".
        <Animated.View key={i} entering={FadeIn.delay(i * 60)}>
          <Animated.View style={[styles.skeletonCard, pulse]}>
            <View style={styles.skeletonAvatar} />
            <View style={{ flex: 1, gap: 8 }}>
              <View style={[styles.skeletonLine, { width: `${55 + ((i * 17) % 30)}%` }]} />
              <View style={[styles.skeletonLine, { width: `${35 + ((i * 23) % 35)}%`, height: 8 }]} />
            </View>
            <View style={styles.skeletonChip} />
          </Animated.View>
        </Animated.View>
      ))}
    </View>
  );
}

/** Search field with an animated focus ring, a bouncing icon and a pop-in clear button. */
export function AnimatedSearch({ value, onChangeText, ...rest }: TextInputProps & { value: string; onChangeText: (t: string) => void }) {
  const focus = useSharedValue(0);
  const DIVIDER = theme.colors.bg.divider; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const BRAND_400 = theme.colors.brand[400];
  const box = useAnimatedStyle(() => ({
    borderColor: interpolateColor(focus.value, [0, 1], [DIVIDER, BRAND_400]),
    transform: [{ scale: 1 + focus.value * 0.01 }],
    shadowOpacity: 0.04 + focus.value * 0.1,
  }));
  const icon = useAnimatedStyle(() => ({ transform: [{ scale: 1 + focus.value * 0.12 }, { rotate: `${focus.value * -8}deg` }] }));
  return (
    <Animated.View style={[styles.searchBox, box]}>
      <Animated.View style={icon}>
        <Search size={18} color={value ? theme.colors.brand[600] : theme.colors.text.muted} />
      </Animated.View>
      <TextInput
        {...rest}
        value={value}
        onChangeText={onChangeText}
        style={styles.searchInput}
        placeholderTextColor={theme.colors.text.muted}
        onFocus={(e) => { focus.value = withSpring(1, { damping: 22, overshootClamping: true }); rest.onFocus?.(e); }}
        onBlur={(e) => { focus.value = withTiming(0, { duration: 180 }); rest.onBlur?.(e); }}
      />
      {value ? (
        <Animated.View entering={ZoomIn.duration(260)} exiting={ZoomOut.duration(120)}>
          <Pressable onPress={() => onChangeText('')} hitSlop={10} style={styles.clearBtn} accessibilityLabel="Clear search">
            <X size={13} color="white" />
          </Pressable>
        </Animated.View>
      ) : null}
    </Animated.View>
  );
}

/** A soft glow that breathes behind the Scan button so the main action stands out. */
export function BreathingGlow() {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, []);
  const style = useAnimatedStyle(() => ({ opacity: 0.18 + t.value * 0.22, transform: [{ scaleX: 1 + t.value * 0.12 }, { scaleY: 1 + t.value * 0.3 }] }));
  return <Animated.View pointerEvents="none" style={[styles.glow, style]} />;
}

/** Floating "back to top" button that pops in once you've scrolled down. */
export function ScrollTopButton({ visible, onPress }: { visible: boolean; onPress: () => void }) {
  if (!visible) return null;
  return (
    <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(150)} style={styles.topBtnWrap}>
      <Pressable onPress={onPress} style={styles.topBtn} accessibilityLabel="Back to top">
        <ArrowUp size={18} color="white" />
      </Pressable>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  skeletonCard: {
    flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: theme.colors.bg.surface,
    borderRadius: 16, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
  skeletonAvatar: { width: 46, height: 46, borderRadius: 23, backgroundColor: theme.colors.brand[100] },
  skeletonLine: { height: 11, borderRadius: 6, backgroundColor: toneBg('#e9e5f5') },
  skeletonChip: { width: 36, height: 24, borderRadius: 12, backgroundColor: toneBg('#eef2f7') },
  searchBox: {
    flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5, borderRadius: 14, paddingHorizontal: 14, height: 46, marginBottom: 12,
    shadowColor: theme.colors.brand[700], shadowRadius: 10, shadowOffset: { width: 0, height: 3 }, elevation: 1,
  },
  searchInput: { flex: 1, fontSize: 15, color: theme.colors.text.display, paddingVertical: 8 },
  clearBtn: { width: 20, height: 20, borderRadius: 10, backgroundColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center' },
  glow: { position: 'absolute', left: -4, right: -4, top: -3, bottom: -3, borderRadius: 24, backgroundColor: theme.colors.brand[400] },
  topBtnWrap: { position: 'absolute', right: 18, bottom: 22 },
  topBtn: {
    width: 44, height: 44, borderRadius: 22, backgroundColor: theme.colors.brand[600], alignItems: 'center', justifyContent: 'center',
    ...theme.shadows.active,
  },
}));
