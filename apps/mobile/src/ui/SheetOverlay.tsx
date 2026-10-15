import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Pressable, BackHandler, useWindowDimensions, type StyleProp, type ViewStyle } from 'react-native';
import { Gesture, type PanGesture } from 'react-native-gesture-handler';
import Animated, {
  useSharedValue, useAnimatedStyle, withTiming, withSpring, Easing, interpolate, Extrapolation, runOnJS,
  type SharedValue, type AnimatedStyle,
} from 'react-native-reanimated';
import { KeyboardAwareView } from './keyboard';
import { WindowPortal } from './OverlayHost';

/**
 * The motion every in-screen overlay shares: sheets, dialogs, action sheets, dropdowns.
 *
 * Overlays are drawn inside the screen, not in an RN <Modal> — inside a Modal (with entering
 * animations) taps on ✕ / Save didn't register on Android. ONE shared value (`offset`) drives the
 * slide/fade, the drag and the backdrop, so nothing can stop halfway (Reanimated *entering*
 * animations sometimes did, leaving a sheet half off-screen).
 *
 *   const motion = useSheetMotion({ open, onClose });
 *   <SheetOverlay motion={motion} backdropColor="rgba(15,23,42,0.45)">
 *     <Animated.View style={[myChrome, motion.sheetStyle]}>
 *       <GestureDetector gesture={motion.pan}>…handle / title…</GestureDetector>
 *       <Pressable onPress={motion.dismiss}>✕</Pressable>   // ✕ OUTSIDE the drag area
 *     </Animated.View>
 *   </SheetOverlay>
 *
 * Render it last in the screen (it fills the view it's drawn in). The chrome stays with the caller.
 */

const OPEN_SPRING = { damping: 24, stiffness: 210, mass: 0.9, overshootClamping: true }; // clamped: no bounce
const CLOSE_TIMING = { duration: 210, easing: Easing.in(Easing.quad) };
const FADE_IN = { duration: 260, easing: Easing.out(Easing.cubic) };
const FADE_OUT = { duration: 180, easing: Easing.in(Easing.quad) };
const FADE_RISE = 25; // px the content rises while fading in (like FadeInDown)
const SWIPE_CLOSE_DISTANCE = 110;
const SWIPE_CLOSE_VELOCITY = 900;

export interface SheetMotionOptions {
  /** Parent-controlled visibility. Omit for a panel that's open while mounted: it slides out, then calls onClose. */
  open?: boolean;
  onClose: () => void;
  /** false: back button, backdrop and swipe don't close it (e.g. while saving). Default true. */
  dismissible?: boolean;
  /** 'slide' (default): up from the bottom, swipe down to close. 'fade': fades in place with a small rise (menus). */
  kind?: 'slide' | 'fade';
}

export interface SheetMotion {
  /** false once it has finished closing: render nothing. */
  mounted: boolean;
  /** false while closing: taps fall through instead of hitting a leaving overlay. */
  interactive: boolean;
  /** Close the way the user would (✕, backdrop, back button, swipe). Respects `dismissible`. */
  dismiss: () => void;
  /** Swipe-down-to-close: attach to the handle / title area only (never around the ✕). */
  pan: PanGesture;
  /** The one value behind it all: 0 = shown; window height (slide) or 1 (fade) = hidden. Read-only for callers. */
  offset: SharedValue<number>;
  sheetStyle: AnimatedStyle<ViewStyle>;
  backdropStyle: AnimatedStyle<ViewStyle>;
}

export function useSheetMotion({ open, onClose, dismissible = true, kind = 'slide' }: SheetMotionOptions): SheetMotion {
  const { height: winH } = useWindowDimensions();
  const controlled = open !== undefined;
  const isOpen = open ?? true;
  const hidden = kind === 'fade' ? 1 : winH;
  const offset = useSharedValue(hidden);
  const [mounted, setMounted] = useState(isOpen);
  const [leaving, setLeaving] = useState(false);
  const leavingRef = useRef(false);
  const latest = useRef({ onClose, dismissible, controlled, hidden });
  latest.current = { onClose, dismissible, controlled, hidden };

  const finishLeaving = () => latest.current.onClose();

  const dismiss = () => {
    const cur = latest.current;
    if (!cur.dismissible || leavingRef.current) return;
    // Controlled: the parent hides it and the effect below animates it out (and a parent that
    // refuses — busy — simply keeps it open).
    if (cur.controlled) { cur.onClose(); return; }
    leavingRef.current = true;
    setLeaving(true);
    offset.value = withTiming(cur.hidden, kind === 'fade' ? FADE_OUT : CLOSE_TIMING, () => { runOnJS(finishLeaving)(); });
  };
  const dismissRef = useRef(dismiss);
  dismissRef.current = dismiss;

  useEffect(() => {
    if (isOpen) {
      leavingRef.current = false;
      setLeaving(false);
      setMounted(true);
      offset.value = hidden;
      offset.value = kind === 'fade' ? withTiming(0, FADE_IN) : withSpring(0, OPEN_SPRING);
      const sub = BackHandler.addEventListener('hardwareBackPress', () => { dismissRef.current(); return true; });
      return () => sub.remove();
    }
    // Closed by anyone: leave from wherever it is (mid-drag included), then unmount.
    offset.value = withTiming(hidden, kind === 'fade' ? FADE_OUT : CLOSE_TIMING, (done) => { if (done) runOnJS(setMounted)(false); });
    return undefined;
  }, [isOpen]);

  const pan = Gesture.Pan()
    .runOnJS(true)
    .activeOffsetY(6)
    .failOffsetX([-24, 24])
    .onUpdate((e) => { offset.value = Math.max(0, e.translationY); })
    .onEnd((e) => {
      if ((e.translationY > SWIPE_CLOSE_DISTANCE || e.velocityY > SWIPE_CLOSE_VELOCITY) && latest.current.dismissible) dismissRef.current();
      else offset.value = withSpring(0, OPEN_SPRING);
    });

  const fade = kind === 'fade';
  const sheetStyle = useAnimatedStyle(() => (fade
    ? { opacity: 1 - offset.value, transform: [{ translateY: offset.value * FADE_RISE }] }
    : { transform: [{ translateY: offset.value }] }));
  const backdropStyle = useAnimatedStyle(() => ({
    opacity: fade ? 1 - offset.value : interpolate(offset.value, [0, winH * 0.6], [1, 0], Extrapolation.CLAMP),
  }));

  return { mounted, interactive: controlled ? isOpen : !leaving, dismiss, pan, offset, sheetStyle, backdropStyle };
}

/**
 * The full-size layer an overlay lives in: dims with a tappable backdrop that follows the motion,
 * stacks children at the bottom. `keyboardAware` pads it by the keyboard so inputs stay visible.
 *
 * layer 'screen' (default): fills the view it's rendered in — render it last in the screen; on a
 * tab screen it sits above the tab bar. layer 'window': drawn in the app's OverlayProvider host,
 * covering the whole window including the tab bar (like the RN <Modal> it replaces).
 */
export function SheetOverlay({ motion, backdropColor, zIndex = 50, keyboardAware, backdropLabel = 'Close', layer = 'screen', style, children }: {
  motion: SheetMotion;
  backdropColor: string;
  zIndex?: number;
  keyboardAware?: boolean;
  backdropLabel?: string;
  layer?: 'screen' | 'window';
  style?: StyleProp<ViewStyle>;
  children: React.ReactNode;
}) {
  if (!motion.mounted) return null;
  const Frame = keyboardAware ? KeyboardAwareView : View;
  const frame = (
    <Frame style={[styles.overlay, { zIndex, elevation: zIndex }, style]} pointerEvents={motion.interactive ? 'auto' : 'none'}>
      <Animated.View style={[StyleSheet.absoluteFill, motion.backdropStyle]}>
        <Pressable style={[styles.backdrop, { backgroundColor: backdropColor }]} onPress={motion.dismiss} accessibilityLabel={backdropLabel} />
      </Animated.View>
      {children}
    </Frame>
  );
  return layer === 'window' ? <WindowPortal>{frame}</WindowPortal> : frame;
}

const styles = StyleSheet.create({
  overlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, justifyContent: 'flex-end' },
  backdrop: { flex: 1 },
});
