import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, TextInput, TextInputProps, Pressable, PressableProps, StyleProp, ViewStyle, Keyboard, Platform, ScrollView, LayoutChangeEvent } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withSequence,
  interpolateColor,
  ZoomIn,
  ZoomOut,
  FadeIn,
  FadeInDown,
  FadeOut,
  LinearTransition,
} from 'react-native-reanimated';
import { X, Pencil, Plus, Check } from 'lucide-react-native';
import { Text } from '../../ui/Text';
import { theme, toneBg, toneFg, makeStyles, live } from '../../ui/theme';

const SPRING_IN = { damping: 24, stiffness: 380, overshootClamping: true };
const SPRING_OUT = { damping: 24, stiffness: 260, overshootClamping: true };

// ─── PressableScale ─────────────────────────────────────────────────────────
// Springs down on press; nudges a trailing icon right and a leading icon left.

interface PressableScaleProps extends Omit<PressableProps, 'style' | 'children'> {
  style?: StyleProp<ViewStyle>;
  containerStyle?: StyleProp<ViewStyle>;
  leadingIcon?: React.ReactNode | undefined;
  trailingIcon?: React.ReactNode | undefined;
  scaleTo?: number;
  children?: React.ReactNode;
}

export function PressableScale({
  style,
  containerStyle,
  leadingIcon,
  trailingIcon,
  scaleTo = 0.96,
  disabled,
  children,
  onPressIn,
  onPressOut,
  ...rest
}: PressableScaleProps) {
  const scale = useSharedValue(1);
  const nudge = useSharedValue(0);

  const scaleStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  const trailingStyle = useAnimatedStyle(() => ({ transform: [{ translateX: nudge.value * 4 }] }));
  const leadingStyle = useAnimatedStyle(() => ({ transform: [{ translateX: nudge.value * -3 }] }));

  return (
    <Animated.View style={[containerStyle, scaleStyle]}>
      <Pressable
        {...rest}
        disabled={disabled}
        style={[style, disabled && styles.disabled]}
        onPressIn={(e) => {
          scale.value = withSpring(scaleTo, SPRING_IN);
          nudge.value = withSpring(1, SPRING_IN);
          onPressIn?.(e);
        }}
        onPressOut={(e) => {
          scale.value = withSpring(1, SPRING_OUT);
          nudge.value = withSpring(0, SPRING_OUT);
          onPressOut?.(e);
        }}
      >
        {leadingIcon ? <Animated.View style={leadingStyle}>{leadingIcon}</Animated.View> : null}
        {children}
        {trailingIcon ? <Animated.View style={trailingStyle}>{trailingIcon}</Animated.View> : null}
      </Pressable>
    </Animated.View>
  );
}

// ─── SectionHeader ──────────────────────────────────────────────────────────

interface SectionHeaderProps {
  icon: React.ReactNode;
  label: string;
  hint?: string | undefined;
  right?: React.ReactNode | undefined;
}

export function SectionHeader({ icon, label, hint, right }: SectionHeaderProps) {
  return (
    <View style={styles.sectionHeader}>
      <View style={styles.sectionIcon}>{icon}</View>
      <View style={{ flex: 1 }}>
        <Text style={styles.sectionLabel}>{label}</Text>
        {hint ? <Text style={styles.sectionHint}>{hint}</Text> : null}
      </View>
      {right}
    </View>
  );
}

// ─── EditableField ──────────────────────────────────────────────────────────
// Input with a leading icon, a pencil affordance and an animated focus border.

interface EditableFieldProps extends Omit<TextInputProps, 'style'> {
  icon: React.ReactNode;
  label: string;
  inputRef?: React.Ref<TextInput> | undefined;
  error?: string | undefined;
}

export function EditableField({ icon, label, inputRef, error, multiline, onFocus, onBlur, ...inputProps }: EditableFieldProps) {
  const focus = useSharedValue(0);

  const DANGER = theme.colors.semantic.danger; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const DIVIDER = theme.colors.bg.divider;
  const BRAND_500 = theme.colors.brand[500];
  const HOVER = theme.colors.bg.surfaceHover;
  const SURFACE = theme.colors.bg.surface;
  const borderStyle = useAnimatedStyle(() => ({
    borderColor: error
      ? DANGER
      : interpolateColor(focus.value, [0, 1], [DIVIDER, BRAND_500]),
    backgroundColor: interpolateColor(focus.value, [0, 1], [HOVER, SURFACE]),
  }));
  const pencilStyle = useAnimatedStyle(() => ({
    opacity: 1 - focus.value * 0.6,
    transform: [{ rotate: `${focus.value * -12}deg` }, { scale: 1 + focus.value * 0.1 }],
  }));

  return (
    <View style={styles.fieldWrap}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <Animated.View style={[styles.field, multiline && styles.fieldMultiline, borderStyle]}>
        <View style={[styles.fieldIcon, multiline && { marginTop: 2 }]}>{icon}</View>
        <TextInput
          ref={inputRef}
          {...inputProps}
          multiline={multiline}
          style={[styles.fieldInput, multiline && styles.fieldInputMultiline]}
          placeholderTextColor={theme.colors.text.muted}
          onFocus={(e) => {
            focus.value = withTiming(1, { duration: 180 });
            onFocus?.(e);
          }}
          onBlur={(e) => {
            focus.value = withTiming(0, { duration: 180 });
            onBlur?.(e);
          }}
        />
        <Animated.View style={[styles.fieldPencil, multiline && { marginTop: 2 }, pencilStyle]}>
          <Pencil size={14} color={theme.colors.brand[500]} />
        </Animated.View>
      </Animated.View>
      {error ? <Text style={styles.fieldError}>{error}</Text> : null}
    </View>
  );
}

// ─── AnimatedTag ────────────────────────────────────────────────────────────
// Pops in when added, shrinks out when removed, neighbours reflow smoothly.

export type TagTone = 'industry' | 'role';

const TAG_TONES: Record<TagTone, { bg: string; border: string; text: string; icon: string; iconBg: string }> = live(() => ({
  industry: {
    bg: theme.colors.brand[600],
    border: theme.colors.brand[700],
    text: theme.colors.text.inverse,
    icon: theme.colors.brand[700],
    iconBg: theme.colors.brand[100],
  },
  role: {
    bg: toneBg('#0d9488'),
    border: toneFg('#0f766e'),
    text: theme.colors.text.inverse,
    icon: toneFg('#0f766e'),
    iconBg: toneBg('#ccfbf1'),
  },
}));

interface AnimatedTagProps {
  label: string;
  tone: TagTone;
  onRemove: () => void;
}

export function AnimatedTag({ label, tone, onRemove }: AnimatedTagProps) {
  const t = TAG_TONES[tone];
  return (
    <Animated.View
      entering={ZoomIn.duration(260)}
      exiting={ZoomOut.duration(160)}
      layout={LinearTransition.duration(260)}
    >
      <PressableScale
        style={[styles.tag, { backgroundColor: t.bg, borderColor: t.border }]}
        onPress={onRemove}
        scaleTo={0.92}
        accessibilityRole="button"
        accessibilityLabel={`Remove ${label}`}
      >
        <Text style={[styles.tagText, { color: t.text }]}>{label}</Text>
        <View style={[styles.tagX, { backgroundColor: t.iconBg }]}>
          <X size={11} color={t.icon} strokeWidth={3} />
        </View>
      </PressableScale>
    </Animated.View>
  );
}

// ─── TagEditor ──────────────────────────────────────────────────────────────
// Chips + inline add input. Enter adds and keeps the input open for the next one.

interface TagEditorProps {
  tags: string[];
  setTags: React.Dispatch<React.SetStateAction<string[]>>;
  tone: TagTone;
  addLabel: string;
  placeholder: string;
  onInputFocus?: () => void;
}

export function TagEditor({ tags, setTags, tone, addLabel, placeholder, onInputFocus }: TagEditorProps) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState('');
  const [notice, setNotice] = useState('');
  // Read through a ref so blur-after-confirm can't commit the same draft twice.
  const draftRef = useRef('');
  const t = TAG_TONES[tone];

  const updateDraft = (text: string) => {
    draftRef.current = text;
    setDraft(text);
    if (notice) setNotice('');
  };

  const commit = (close: boolean) => {
    const value = draftRef.current.trim().replace(/\s+/g, ' ');
    draftRef.current = '';
    setDraft('');
    if (value) {
      if (tags.some((tag) => tag.toLowerCase() === value.toLowerCase())) {
        setNotice(`"${value}" is already added`);
      } else {
        setTags((prev) => (prev.some((tag) => tag.toLowerCase() === value.toLowerCase()) ? prev : [...prev, value]));
      }
    }
    if (close) setAdding(false);
  };

  return (
    <View>
      <View style={styles.tagsContainer}>
        {tags.map((tag) => (
          <AnimatedTag key={tag} label={tag} tone={tone} onRemove={() => setTags((prev) => prev.filter((x) => x !== tag))} />
        ))}
        {!adding && (
          <Animated.View layout={LinearTransition.duration(260)} entering={ZoomIn.duration(180)}>
            <PressableScale
              style={[styles.tagAdd, { borderColor: t.border }]}
              onPress={() => setAdding(true)}
              leadingIcon={<Plus size={14} color={t.border} strokeWidth={2.5} />}
              accessibilityRole="button"
            >
              <Text style={[styles.tagAddText, { color: t.border }]}>{addLabel}</Text>
            </PressableScale>
          </Animated.View>
        )}
      </View>

      {adding && (
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={[styles.tagInputRow, { borderColor: t.border }]}>
          <TextInput
            autoFocus
            value={draft}
            onChangeText={updateDraft}
            onSubmitEditing={() => commit(false)}
            submitBehavior="submit"
            onBlur={() => commit(true)}
            onFocus={onInputFocus}
            returnKeyType="done"
            autoCapitalize="words"
            maxLength={40}
            placeholder={placeholder}
            placeholderTextColor={theme.colors.text.muted}
            style={styles.tagInput}
          />
          <PressableScale
            style={[styles.tagConfirm, { backgroundColor: t.bg }]}
            onPress={() => commit(true)}
            scaleTo={0.9}
            accessibilityRole="button"
            accessibilityLabel="Add"
          >
            <Check size={16} color={theme.colors.text.inverse} strokeWidth={3} />
          </PressableScale>
        </Animated.View>
      )}

      {notice ? <Text style={styles.tagNotice}>{notice}</Text> : null}
      {!adding && tags.length === 0 ? (
        <Animated.Text entering={FadeIn} style={styles.tagEmpty}>Nothing added yet.</Animated.Text>
      ) : null}
    </View>
  );
}

// ─── SegmentedSelect ────────────────────────────────────────────────────────
// Indicator slides between options with a spring.

interface SegmentedOption {
  value: string;
  icon?: (active: boolean) => React.ReactNode;
}

interface SegmentedSelectProps {
  options: SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
}

export function SegmentedSelect({ options, value, onChange }: SegmentedSelectProps) {
  const [trackWidth, setTrackWidth] = useState(0);
  const index = options.findIndex((o) => o.value === value);
  const segmentWidth = trackWidth > 0 ? (trackWidth - 8) / options.length : 0;

  const x = useSharedValue(0);
  const visible = useSharedValue(index >= 0 ? 1 : 0);
  const initialised = useRef(false);

  useEffect(() => {
    if (!segmentWidth) return;
    if (index >= 0) {
      // First placement jumps; later changes slide.
      x.value = initialised.current ? withSpring(index * segmentWidth, { damping: 22, stiffness: 200, overshootClamping: true }) : index * segmentWidth;
      visible.value = withTiming(1, { duration: 180 });
    } else {
      visible.value = withTiming(0, { duration: 180 });
    }
    initialised.current = true;
  }, [index, segmentWidth]);

  const indicatorStyle = useAnimatedStyle(() => ({
    width: segmentWidth,
    opacity: visible.value,
    transform: [{ translateX: x.value }, { scale: 0.9 + visible.value * 0.1 }],
  }));

  return (
    <View
      style={styles.segmented}
      onLayout={(e: LayoutChangeEvent) => setTrackWidth(e.nativeEvent.layout.width)}
      accessibilityRole="radiogroup"
    >
      <Animated.View style={[styles.segmentIndicator, indicatorStyle]} pointerEvents="none" />
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <PressableScale
            key={opt.value}
            containerStyle={{ flex: 1 }}
            style={styles.segmentBtn}
            onPress={() => onChange(opt.value)}
            scaleTo={0.94}
            accessibilityRole="radio"
            accessibilityState={{ selected: active }}
          >
            {opt.icon?.(active)}
            <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{opt.value}</Text>
          </PressableScale>
        );
      })}
    </View>
  );
}

// ─── AnimatedToggle ─────────────────────────────────────────────────────────

interface AnimatedToggleProps {
  value: boolean;
  onValueChange: (value: boolean) => void;
  accessibilityLabel?: string;
}

const TRACK_W = 50;
const TRACK_H = 30;
const THUMB = 24;

export function AnimatedToggle({ value, onValueChange, accessibilityLabel }: AnimatedToggleProps) {
  const progress = useSharedValue(value ? 1 : 0);
  const squash = useSharedValue(0);
  const pop = useSharedValue(1);

  useEffect(() => {
    progress.value = withSpring(value ? 1 : 0, { damping: 22, stiffness: 220, overshootClamping: true });
    // Small overshoot "pop" of the thumb each time it settles on a new side.
    pop.value = withSequence(withTiming(0.82, { duration: 90 }), withSpring(1, { damping: 22, stiffness: 260, overshootClamping: true }));
  }, [value]);

  const TRACK_OFF = theme.colors.bg.divider; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const TRACK_ON = theme.colors.brand[600];
  const trackStyle = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(progress.value, [0, 1], [TRACK_OFF, TRACK_ON]),
  }));
  const glowStyle = useAnimatedStyle(() => ({
    opacity: progress.value * 0.35,
    transform: [{ scale: 1 + progress.value * 0.12 }],
  }));
  const thumbStyle = useAnimatedStyle(() => ({
    // Stretches toward the direction of travel while pressed.
    width: THUMB + squash.value * 6,
    transform: [
      { translateX: progress.value * (TRACK_W - THUMB - 6) - squash.value * 6 * progress.value },
      { scale: pop.value },
    ],
  }));
  const onIconStyle = useAnimatedStyle(() => ({
    opacity: progress.value,
    transform: [{ scale: 0.4 + progress.value * 0.6 }, { rotate: `${(1 - progress.value) * -90}deg` }],
  }));
  const offIconStyle = useAnimatedStyle(() => ({
    opacity: 1 - progress.value,
    transform: [{ scale: 1 - progress.value * 0.6 }, { rotate: `${progress.value * 90}deg` }],
  }));

  return (
    <Pressable
      onPress={() => onValueChange(!value)}
      onPressIn={() => { squash.value = withSpring(1, SPRING_IN); }}
      onPressOut={() => { squash.value = withSpring(0, SPRING_OUT); }}
      accessibilityRole="switch"
      accessibilityState={{ checked: value }}
      accessibilityLabel={accessibilityLabel}
      hitSlop={8}
    >
      <Animated.View style={[styles.toggleGlow, glowStyle]} pointerEvents="none" />
      <Animated.View style={[styles.toggleTrack, trackStyle]}>
        <Animated.View style={[styles.toggleThumb, thumbStyle]}>
          <Animated.View style={[styles.toggleIcon, onIconStyle]}>
            <Check size={13} color={theme.colors.brand[600]} strokeWidth={3.5} />
          </Animated.View>
          <Animated.View style={[styles.toggleIcon, offIconStyle]}>
            <X size={12} color={theme.colors.text.muted} strokeWidth={3} />
          </Animated.View>
        </Animated.View>
      </Animated.View>
    </Pressable>
  );
}

// ─── useKeyboardAwareScroll ─────────────────────────────────────────────────
// Adds bottom room for the keyboard and scrolls the focused input above it.
// Works under Android edge-to-edge, where the window is not resized.

export function useKeyboardAwareScroll() {
  const scrollRef = useRef<ScrollView>(null);
  const scrollY = useRef(0);
  const keyboardTop = useRef<number | null>(null);
  const [keyboardHeight, setKeyboardHeight] = useState(0);

  const revealFocusedInput = useCallback(() => {
    const top = keyboardTop.current;
    const focused = TextInput.State.currentlyFocusedInput?.() as any;
    if (top == null || !focused?.measureInWindow || !scrollRef.current) return;
    focused.measureInWindow((_x: number, y: number, _w: number, h: number) => {
      const overlap = y + h + 24 - top;
      if (overlap > 0) {
        scrollRef.current?.scrollTo({ y: scrollY.current + overlap, animated: true });
      }
    });
  }, []);

  useEffect(() => {
    const showEvent = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
    const hideEvent = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';
    const show = Keyboard.addListener(showEvent, (e) => {
      keyboardTop.current = e.endCoordinates.screenY;
      setKeyboardHeight(e.endCoordinates.height);
      setTimeout(revealFocusedInput, Platform.OS === 'ios' ? 250 : 50);
    });
    const hide = Keyboard.addListener(hideEvent, () => {
      keyboardTop.current = null;
      setKeyboardHeight(0);
    });
    return () => {
      show.remove();
      hide.remove();
    };
  }, [revealFocusedInput]);

  // Call from an input's onFocus so switching fields while the keyboard is open also scrolls.
  const onInputFocus = useCallback(() => {
    setTimeout(revealFocusedInput, 80);
  }, [revealFocusedInput]);

  const scrollProps = {
    ref: scrollRef,
    scrollEventThrottle: 16,
    onScroll: (e: any) => {
      scrollY.current = e.nativeEvent.contentOffset.y;
    },
  };

  return { scrollProps, keyboardHeight, onInputFocus };
}

const styles = makeStyles(() => ({
  disabled: { opacity: 0.5 },

  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 12, marginTop: 8 },
  sectionIcon: {
    width: 30,
    height: 30,
    borderRadius: 9,
    backgroundColor: theme.colors.brand[50],
    borderWidth: 1,
    borderColor: theme.colors.brand[100],
    alignItems: 'center',
    justifyContent: 'center',
  },
  sectionLabel: { fontSize: 11, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  sectionHint: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },

  fieldWrap: { marginBottom: 12 },
  fieldLabel: { fontSize: 12, fontWeight: '700', color: theme.colors.text.body, marginBottom: 6 },
  field: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderRadius: 12,
    paddingHorizontal: 12,
    minHeight: 48,
  },
  fieldMultiline: { alignItems: 'flex-start', paddingVertical: 12 },
  fieldIcon: { marginRight: 10 },
  fieldInput: { flex: 1, fontSize: 15, color: theme.colors.text.display, paddingVertical: 10 },
  fieldInputMultiline: { minHeight: 76, textAlignVertical: 'top', paddingVertical: 0, lineHeight: 21 },
  fieldPencil: { marginLeft: 8 },
  fieldError: { fontSize: 12, color: theme.colors.semantic.danger, marginTop: 6, fontWeight: '600' },

  tag: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingLeft: 14,
    paddingRight: 6,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
    gap: 8,
  },
  tagText: { fontSize: 14, fontWeight: '600' },
  tagX: { width: 20, height: 20, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  tagsContainer: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  tagAdd: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderStyle: 'dashed',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  tagAddText: { fontSize: 14, fontWeight: '600' },
  tagInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5,
    borderRadius: 999,
    paddingLeft: 16,
    padding: 4,
  },
  tagInput: { flex: 1, fontSize: 14, color: theme.colors.text.display, paddingVertical: 8 },
  tagConfirm: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center' },
  tagNotice: { fontSize: 12, color: theme.colors.semantic.warning, marginTop: 8, fontWeight: '600' },
  tagEmpty: { fontSize: 12, color: theme.colors.text.muted, marginTop: 8 },

  segmented: {
    flexDirection: 'row',
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
    borderRadius: 16,
    padding: 4,
  },
  segmentIndicator: {
    position: 'absolute',
    top: 4,
    bottom: 4,
    left: 4,
    borderRadius: 12,
    backgroundColor: theme.colors.brand[600],
    ...theme.shadows.active,
  },
  segmentBtn: { paddingVertical: 10, alignItems: 'center', justifyContent: 'center', borderRadius: 12, gap: 4 },
  segmentText: { fontSize: 13, fontWeight: '600', color: theme.colors.text.body },
  segmentTextActive: { color: theme.colors.text.inverse },

  toggleTrack: { width: TRACK_W, height: TRACK_H, borderRadius: TRACK_H / 2, padding: 3, justifyContent: 'center' },
  toggleGlow: {
    position: 'absolute',
    top: -4,
    left: -4,
    width: TRACK_W + 8,
    height: TRACK_H + 8,
    borderRadius: (TRACK_H + 8) / 2,
    backgroundColor: theme.colors.brand[400],
  },
  toggleIcon: { position: 'absolute', top: 0, bottom: 0, left: 0, right: 0, alignItems: 'center', justifyContent: 'center' },
  toggleThumb: {
    height: THUMB,
    borderRadius: THUMB / 2,
    backgroundColor: theme.colors.text.inverse,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
}));
