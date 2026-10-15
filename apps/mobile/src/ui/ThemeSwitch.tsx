import React, { useEffect, useRef, useState } from 'react';
import { Pressable, StyleSheet } from 'react-native';
import Animated, {
  Easing, runOnJS, useAnimatedStyle, useSharedValue, withSequence, withSpring, withTiming,
} from 'react-native-reanimated';
import { Moon, Sun } from 'lucide-react-native';
import { theme, getColorScheme, applyColorScheme, makeStyles, useAppearance } from './theme';
import { APP_BG, resolveColorScheme, saveAppearancePref, type AppearancePref } from './appearance';

/**
 * Light / dark switching in place — no app reload:
 *   switch → the screen fades into the new mode's background → the palette and every style
 *   sheet update and the screens redraw underneath → the overlay fades away in the new mode.
 * Works the same in Expo Go, the dev build and store builds.
 */
type Runner = (color: string, midway: () => void) => Promise<void>;
let runTransition: Runner | null = null;

/** Mount once at the root (above everything). */
export function ThemeTransitionHost() {
  const [color, setColor] = useState<string>(APP_BG[getColorScheme()]);
  const [visible, setVisible] = useState(false);
  const opacity = useSharedValue(0);

  useEffect(() => {
    runTransition = (c, midway) => new Promise<void>((resolve) => {
      setColor(c);
      setVisible(true);
      opacity.value = 0;
      const fadeOut = () => {
        // two frames for the redraw to paint under the cover, then reveal it
        requestAnimationFrame(() => requestAnimationFrame(() => {
          opacity.value = withTiming(0, { duration: 360, easing: Easing.out(Easing.cubic) }, (fin) => {
            if (fin) { runOnJS(setVisible)(false); runOnJS(resolve)(); }
          });
        }));
      };
      const covered = () => { midway(); setTimeout(fadeOut, 60); };
      opacity.value = withTiming(1, { duration: 220, easing: Easing.inOut(Easing.quad) }, (fin) => { if (fin) runOnJS(covered)(); });
    });
    return () => { runTransition = null; };
  }, []);

  const style = useAnimatedStyle(() => ({ opacity: opacity.value }));
  if (!visible) return null;
  return <Animated.View pointerEvents="auto" style={[StyleSheet.absoluteFill, styles.overlay, { backgroundColor: color }, style]} />;
}

export type SwitchResult = 'switched' | 'unchanged';

/** Save the choice and, if the mode changes, fade into it (no reload). */
export async function switchAppearance(pref: AppearancePref): Promise<SwitchResult> {
  saveAppearancePref(pref);
  const target = resolveColorScheme(pref);
  if (target === getColorScheme()) {
    applyColorScheme(target, pref); // records the choice (e.g. System) → button + card update, no redraw
    return 'unchanged';
  }
  const apply = () => applyColorScheme(target, pref);
  if (runTransition) await runTransition(APP_BG[target], apply);
  else apply();
  return 'switched';
}

/** Round sun / moon button: tap flips Light ↔ Dark (from System it flips the current mode). */
export function ThemeToggleButton({ onResult, size = 38 }: { onResult?: (r: SwitchResult, target: 'light' | 'dark') => void; size?: number }) {
  // Follows the shared state, so the icon always matches the Appearance card (and vice versa).
  const { isDark } = useAppearance();
  const [preview, setPreview] = useState<boolean | null>(null); // the icon swaps a moment before the fade
  const showDark = preview ?? isDark;
  useEffect(() => { setPreview(null); }, [isDark]);
  const spin = useSharedValue(0);
  const pop = useSharedValue(1);
  const busy = useRef(false);

  const onPress = () => {
    if (busy.current) return;
    busy.current = true;
    const target = getColorScheme() === 'dark' ? 'light' : 'dark';
    // icon turns and swaps first, then the screen fades into the new mode
    spin.value = withTiming(spin.value + 180, { duration: 380, easing: Easing.out(Easing.cubic) });
    pop.value = withSequence(withTiming(0.82, { duration: 120 }), withSpring(1, { damping: 22, overshootClamping: true }));
    setTimeout(() => setPreview(target === 'dark'), 150);
    setTimeout(() => {
      switchAppearance(target).then((r) => {
        busy.current = false;
        setPreview(null);
        onResult?.(r, target);
      }).catch(() => { busy.current = false; setPreview(null); });
    }, 260);
  };

  const iconStyle = useAnimatedStyle(() => ({ transform: [{ rotate: `${spin.value}deg` }, { scale: pop.value }] }));
  return (
    <Pressable onPress={onPress} hitSlop={8} accessibilityRole="button"
      accessibilityLabel={showDark ? 'Switch to light mode' : 'Switch to dark mode'}
      style={({ pressed }) => [styles.btn, { width: size, height: size, borderRadius: size / 2 }, pressed && { opacity: 0.85 }]}>
      <Animated.View style={iconStyle}>
        {showDark ? <Moon size={17} color={theme.colors.brand[700]} /> : <Sun size={17} color={SUN} />}
      </Animated.View>
    </Pressable>
  );
}

const SUN = '#f59e0b';

const styles = makeStyles(() => ({
  overlay: { zIndex: 9999, elevation: 9999 },
  btn: {
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.bg.surface, borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
}));
