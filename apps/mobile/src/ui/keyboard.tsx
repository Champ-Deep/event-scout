import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Dimensions, Keyboard, Platform, StyleSheet, Text as RNText, TextInput, View, type ViewProps, type KeyboardEvent } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, useAnimatedKeyboard, withTiming, Easing } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { toneFg, makeStyles } from './theme';
/**
 * Keyboard handling that works under Android edge-to-edge (Expo SDK 5x / RN 0.8x), where the
 * window is NOT resized when the keyboard opens — so `KeyboardAvoidingView` with no Android
 * behavior did nothing and inputs/buttons ended up under the keyboard.
 */

const SHOW = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
const HIDE = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';

/** Keyboard top edge in window coordinates (null when hidden). */
export function useKeyboardTop() {
  const [state, setState] = useState<{ top: number | null; duration: number }>({ top: null, duration: 0 });
  useEffect(() => {
    const s = Keyboard.addListener(SHOW, (e: KeyboardEvent) => setState({ top: e.endCoordinates.screenY, duration: e.duration || 220 }));
    const h = Keyboard.addListener(HIDE, (e: KeyboardEvent) => setState({ top: null, duration: e?.duration || 200 }));
    return () => { s.remove(); h.remove(); };
  }, []);
  return state;
}

/**
 * A flex container that pads its bottom by exactly how much the keyboard covers it
 * (measured in the window, so tab bars, safe areas and sheets are accounted for).
 * Use it for screens with a bottom composer / buttons, and around sheets in modals.
 */
// Same options everywhere (Reanimated warns if hooks disagree). The app is edge-to-edge on
// Android, so the keyboard height is measured from the very bottom of the screen.
const KEYBOARD_OPTIONS = { isStatusBarTranslucentAndroid: true, isNavigationBarTranslucentAndroid: true };

export function KeyboardAwareView({ style, children, onLayout, debug, ...rest }: ViewProps & { debug?: boolean }) {
  // Measured through a plain, non-collapsable View that fills us. Measuring the Animated.View
  // itself (or a View Android flattened away) can silently do nothing.
  const probe = useRef<View>(null);
  const [kbEvent, setKbEvent] = useState<{ top: number | null; height: number; duration: number }>({ top: null, height: 0, duration: 0 });
  useEffect(() => {
    const s = Keyboard.addListener(SHOW, (e: KeyboardEvent) => setKbEvent({ top: e.endCoordinates.screenY, height: e.endCoordinates.height, duration: e.duration || 220 }));
    const h = Keyboard.addListener(HIDE, () => setKbEvent({ top: null, height: 0, duration: 200 }));
    return () => { s.remove(); h.remove(); };
  }, []);

  // Reanimated reads the keyboard height straight from the window insets, frame by frame.
  const keyboard = useAnimatedKeyboard(KEYBOARD_OPTIONS);
  // With a translucent navigation bar that height can exclude the gesture / button bar.
  const insets = useSafeAreaInsets();
  const navBar = Platform.OS === 'android' ? insets.bottom : 0;
  // Space between our bottom edge and the bottom of the screen (e.g. a visible tab bar).
  // Measured against both the screen and window heights; the smaller one is used, because
  // over-estimating it is what leaves the input under the keyboard.
  const gapBelow = useSharedValue(0);
  const eventPad = useSharedValue(0);
  const [dbg, setDbg] = useState('');
  const frame = useRef({ y: 0, h: 0 });

  const measure = useCallback(() => {
    const node = probe.current;
    if (!node?.measureInWindow) return;
    node.measureInWindow((_x, y, _w, h) => {
      if (!h) return;
      frame.current = { y, h };
      const bottom = y + h;
      const screenGap = Dimensions.get('screen').height - bottom;
      const windowGap = Dimensions.get('window').height - bottom;
      gapBelow.value = Math.min(400, Math.max(0, Math.min(screenGap, windowGap)));
      const fromEventTop = kbEvent.top == null ? 0 : bottom - kbEvent.top;
      const fromEventHeight = kbEvent.top == null ? 0 : kbEvent.height + navBar - gapBelow.value;
      eventPad.value = withTiming(Math.max(0, fromEventTop, fromEventHeight), {
        duration: Math.min(kbEvent.duration || 220, 250), easing: Easing.out(Easing.quad),
      });
    });
  }, [kbEvent, navBar]);

  useEffect(() => {
    measure();
    // Android lays out again after the keyboard / tab bar settle: re-check.
    const a = setTimeout(measure, 150);
    const b = setTimeout(measure, 450);
    return () => { clearTimeout(a); clearTimeout(b); };
  }, [measure]);

  // Temporary on-screen readout to see what this phone actually reports.
  useEffect(() => {
    if (!debug) return;
    const t = setInterval(() => {
      const f = frame.current;
      setDbg(
        `kb(reanimated)=${Math.round(keyboard.height.value)} ev.h=${Math.round(kbEvent.height)} ev.top=${kbEvent.top == null ? '-' : Math.round(kbEvent.top)} ` +
        `nav=${Math.round(navBar)} gap=${Math.round(gapBelow.value)} y=${Math.round(f.y)} h=${Math.round(f.h)} ` +
        `screen=${Math.round(Dimensions.get('screen').height)} window=${Math.round(Dimensions.get('window').height)} evPad=${Math.round(eventPad.value)}`,
      );
    }, 400);
    return () => clearInterval(t);
  }, [debug, kbEvent, navBar]);

  const padStyle = useAnimatedStyle(() => {
    const kb = keyboard.height.value;
    const fromInsets = kb > 0 ? Math.max(0, kb + navBar - gapBelow.value) : 0;
    return { paddingBottom: Math.max(fromInsets, eventPad.value) };
  });
  return (
    <Animated.View
      {...rest}
      style={[{ flex: 1 }, style, padStyle]}
      onLayout={(e) => { onLayout?.(e); measure(); }} // e.g. the tab bar hiding changes our frame
    >
      <View ref={probe} collapsable={false} pointerEvents="none" style={StyleSheet.absoluteFill} />
      {children}
      {debug && dbg ? (
        <View pointerEvents="none" style={debugStyles.box}><RNText style={debugStyles.text}>{dbg}</RNText></View>
      ) : null}
    </Animated.View>
  );
}

const debugStyles = makeStyles(() => ({
  box: { position: 'absolute', top: 4, left: 4, right: 4, backgroundColor: 'rgba(15,23,42,0.85)', borderRadius: 8, padding: 6 },
  text: { color: toneFg('#a7f3d0'), fontSize: 11, fontFamily: Platform.OS === 'android' ? 'monospace' : 'Menlo' },
}));

type Measurable = { measureInWindow: (cb: (x: number, y: number, w: number, h: number) => void) => void };

/**
 * Scrolls a ScrollView so the focused input — or a given view such as a Save row — sits above
 * the keyboard. `getScrollY` returns the current offset (e.g. from a Reanimated scroll handler).
 */
export function useRevealAboveKeyboard(scrollRef: React.RefObject<any>, getScrollY: () => number, margin = 20) {
  const top = useRef<number | null>(null);
  const target = useRef<Measurable | null>(null);

  const reveal = useCallback((view?: Measurable | null) => {
    if (view !== undefined) target.current = view;
    const kbTop = top.current;
    const node: any = target.current || ((TextInput.State as any).currentlyFocusedInput?.() ?? null);
    if (kbTop == null || !node?.measureInWindow || !scrollRef.current) return;
    node.measureInWindow((_x: number, y: number, _w: number, h: number) => {
      const overlap = y + h + margin - kbTop;
      if (overlap > 0) scrollRef.current?.scrollTo?.({ y: getScrollY() + overlap, animated: true });
    });
  }, [scrollRef, getScrollY, margin]);

  useEffect(() => {
    const s = Keyboard.addListener(SHOW, (e: KeyboardEvent) => {
      top.current = e.endCoordinates.screenY;
      // Wait for KeyboardAwareView's padding to lay out, or the scroll gets clamped.
      setTimeout(() => reveal(), Platform.OS === 'ios' ? 260 : 120);
    });
    const h = Keyboard.addListener(HIDE, () => { top.current = null; target.current = null; });
    // Moving to another field while the keyboard stays open fires no keyboard event:
    // watch the focused input and reveal it when it changes (no onFocus wiring needed).
    let lastFocused: any = null;
    const poll = setInterval(() => {
      if (top.current == null) { lastFocused = null; return; }
      const focused = (TextInput.State as any).currentlyFocusedInput?.() ?? null;
      if (focused && focused !== lastFocused) {
        if (lastFocused) reveal(target.current ? undefined : null);
        lastFocused = focused;
      }
    }, 250);
    return () => { s.remove(); h.remove(); clearInterval(poll); };
  }, [reveal]);

  /** Call from onFocus; pass a view to reveal instead of the focused input (e.g. input + its Save row). */
  const request = useCallback((view?: Measurable | null) => {
    target.current = view ?? null;
    setTimeout(() => reveal(), 90); // keyboard already open (switching fields) → reveal now
  }, [reveal]);

  return { reveal: request };
}
