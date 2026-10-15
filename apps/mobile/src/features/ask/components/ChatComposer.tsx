import React, { useEffect, useRef, useState } from 'react';
import { View, TextInput, Image, ActivityIndicator, Pressable, Keyboard, Platform } from 'react-native';
import Animated, { FadeIn, FadeOut, ZoomIn, ZoomOut, LinearTransition, FadeInDown, useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, interpolateColor, Easing } from 'react-native-reanimated';
import { Paperclip, Mic, ArrowUp, X, FileText, Square, Loader2 } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';
import type { ChatAttachment } from '../../../lib/api';
import { PressableScale } from '../../onboarding/components';
import { isImage, formatBytes, MAX_ATTACHMENTS } from '../attachments';


const BARS = 28;

interface VoiceProps {
  isRecording: boolean;
  seconds: number;
  level: number; // 0 … 1
  maxSeconds: number;
  onCancel: () => void;
  onStop: () => void;
}

interface ChatComposerProps {
  value: string;
  onChangeText: (t: string) => void;
  attachments: ChatAttachment[];
  onRemoveAttachment: (uri: string) => void;
  onAttach: () => void;
  onSend: () => void;
  onMic: () => void;
  voice: VoiceProps;
  transcribing: boolean;
  sending: boolean;
  placeholder?: string | undefined;
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** Live waveform from the recorder's metering: newest sample on the right. */
function Waveform({ level }: { level: number }) {
  const [samples, setSamples] = useState<number[]>(() => Array(BARS).fill(0));
  useEffect(() => {
    setSamples((prev) => [...prev.slice(1), level]);
  }, [level]);
  return (
    <View style={styles.wave}>
      {samples.map((v, i) => (
        <View
          key={i}
          style={[
            styles.waveBar,
            {
              height: 4 + v * 26,
              opacity: 0.35 + (i / BARS) * 0.65,
              backgroundColor: v > 0.55 ? theme.colors.brand[700] : theme.colors.brand[400],
            },
          ]}
        />
      ))}
    </View>
  );
}

/** Pulsing ring behind the stop button while recording. */
function RecordPulse() {
  const p = useSharedValue(0);
  useEffect(() => {
    p.value = withRepeat(withTiming(1, { duration: 1300, easing: Easing.out(Easing.quad) }), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ opacity: 0.5 * (1 - p.value), transform: [{ scale: 1 + p.value * 0.6 }] }));
  return <Animated.View pointerEvents="none" style={[styles.pulse, style]} />;
}

function RecDot() {
  const o = useSharedValue(1);
  useEffect(() => {
    o.value = withRepeat(withSequence(withTiming(0.2, { duration: 600 }), withTiming(1, { duration: 600 })), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ opacity: o.value }));
  return <Animated.View style={[styles.recDot, style]} />;
}

function Spinner({ color }: { color: string }) {
  const r = useSharedValue(0);
  useEffect(() => {
    r.value = withRepeat(withTiming(360, { duration: 900, easing: Easing.linear }), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ transform: [{ rotate: `${r.value}deg` }] }));
  return <Animated.View style={style}><Loader2 size={18} color={color} /></Animated.View>;
}

function AttachmentChip({ a, onRemove, disabled }: { a: ChatAttachment; onRemove: () => void; disabled: boolean }) {
  return (
    <Animated.View entering={ZoomIn.duration(260)} exiting={ZoomOut.duration(150)} layout={LinearTransition.duration(260)} style={styles.chip}>
      {isImage(a) ? (
        <Image source={{ uri: a.uri }} style={styles.chipThumb} />
      ) : (
        <View style={[styles.chipThumb, styles.chipFileIcon]}><FileText size={16} color={theme.colors.brand[700]} /></View>
      )}
      <View style={{ maxWidth: 110 }}>
        <Text style={styles.chipName} numberOfLines={1}>{a.name}</Text>
        {a.size ? <Text style={styles.chipSize}>{formatBytes(a.size)}</Text> : null}
      </View>
      {!disabled && (
        <Pressable hitSlop={8} onPress={onRemove} style={styles.chipRemove} accessibilityLabel={`Remove ${a.name}`}>
          <X size={12} color="white" />
        </Pressable>
      )}
    </Animated.View>
  );
}

export function ChatComposer({
  value, onChangeText, attachments, onRemoveAttachment, onAttach, onSend, onMic, voice, transcribing, sending, placeholder,
}: ChatComposerProps) {
  const inputRef = useRef<TextInput>(null);
  // Bottom padding is for the tab bar / screen edge; with the keyboard up it just left a gap
  // between the input and the keyboard (Pixel 7), so it tightens while typing.
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  useEffect(() => {
    const s = Keyboard.addListener(Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow', () => setKeyboardOpen(true));
    const h = Keyboard.addListener(Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide', () => setKeyboardOpen(false));
    return () => { s.remove(); h.remove(); };
  }, []);
  const tight = keyboardOpen ? styles.containerTight : null;
  const focus = useSharedValue(0);
  const canSend = (value.trim().length > 0 || attachments.length > 0) && !sending && !transcribing;
  const showSend = value.trim().length > 0 || attachments.length > 0;

  const BG_F8FAFC = toneBg('#f8fafc'); // current mode's colour, read on the JS thread (worklets can't call toneBg)
  const BG_FFFFFF = toneBg('#ffffff'); // current mode's colour, read on the JS thread (worklets can't call toneBg)
  const DIVIDER = theme.colors.bg.divider; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const BRAND_500 = theme.colors.brand[500];
  const boxStyle = useAnimatedStyle(() => ({
    borderColor: interpolateColor(focus.value, [0, 1], [DIVIDER, BRAND_500]),
    backgroundColor: interpolateColor(focus.value, [0, 1], [BG_F8FAFC, BG_FFFFFF]),
  }));

  if (voice.isRecording) {
    const left = Math.max(0, voice.maxSeconds - voice.seconds);
    return (
      <Animated.View entering={FadeInDown.duration(260)} style={[styles.container, tight]}>
        <View style={styles.voiceRow}>
          <PressableScale onPress={voice.onCancel} style={styles.voiceCancel} accessibilityLabel="Cancel recording">
            <X size={20} color={theme.colors.text.body} />
          </PressableScale>
          <View style={styles.voiceMid}>
            <View style={styles.voiceTop}>
              <RecDot />
              <Text style={styles.voiceTime}>{fmt(voice.seconds)}</Text>
              <Text style={styles.voiceHint}>{left <= 15 ? `${Math.ceil(left)}s left` : 'Listening… tap ■ to ask'}</Text>
            </View>
            <Waveform level={voice.level} />
          </View>
          <View style={styles.stopWrap}>
            <RecordPulse />
            <PressableScale onPress={voice.onStop} style={styles.stopBtn} accessibilityLabel="Stop and ask">
              <Square size={16} color="white" fill="white" />
            </PressableScale>
          </View>
        </View>
      </Animated.View>
    );
  }

  return (
    <View style={[styles.container, tight]}>
      {attachments.length > 0 && (
        <Animated.View entering={FadeIn} exiting={FadeOut} layout={LinearTransition} style={styles.chipsRow}>
          {attachments.map((a) => (
            <AttachmentChip key={a.uri} a={a} disabled={sending} onRemove={() => onRemoveAttachment(a.uri)} />
          ))}
          <Text style={styles.chipCount}>{attachments.length}/{MAX_ATTACHMENTS}</Text>
        </Animated.View>
      )}

      {transcribing && (
        <Animated.View entering={FadeInDown} exiting={FadeOut} style={styles.transcribing}>
          <Spinner color={theme.colors.brand[600]} />
          <Text style={styles.transcribingText}>Transcribing your question…</Text>
        </Animated.View>
      )}

      <View style={styles.row}>
        <PressableScale
          onPress={onAttach}
          disabled={sending || transcribing || attachments.length >= MAX_ATTACHMENTS}
          style={[styles.sideBtn, attachments.length >= MAX_ATTACHMENTS && { opacity: 0.4 }]}
          accessibilityLabel="Attach a file or photo"
        >
          <Paperclip size={20} color={theme.colors.brand[700]} />
        </PressableScale>

        <Animated.View style={[styles.inputBox, boxStyle]}>
          <TextInput
            ref={inputRef}
            style={styles.input}
            value={value}
            onChangeText={onChangeText}
            placeholder={transcribing ? 'Transcribing…' : placeholder || 'Ask about your contacts, leads, event…'}
            placeholderTextColor={theme.colors.text.muted}
            multiline
            editable={!transcribing}
            onFocus={() => { focus.value = withTiming(1, { duration: 180 }); }}
            onBlur={() => { focus.value = withTiming(0, { duration: 180 }); }}
            maxLength={2000}
          />
        </Animated.View>

        {showSend ? (
          <Animated.View key="send" entering={ZoomIn.duration(260)} exiting={ZoomOut.duration(120)}>
            <PressableScale
              onPress={onSend}
              disabled={!canSend}
              style={[styles.actionBtn, styles.sendBtn, !canSend && { opacity: 0.55 }]}
              accessibilityLabel="Send question"
            >
              {sending ? <ActivityIndicator size="small" color="white" /> : <ArrowUp size={20} color="white" />}
            </PressableScale>
          </Animated.View>
        ) : (
          <Animated.View key="mic" entering={ZoomIn.duration(260)} exiting={ZoomOut.duration(120)}>
            <PressableScale
              onPress={onMic}
              disabled={sending || transcribing}
              style={[styles.actionBtn, styles.micBtn, (sending || transcribing) && { opacity: 0.55 }]}
              accessibilityLabel="Ask by voice"
            >
              <Mic size={20} color={theme.colors.brand[700]} />
            </PressableScale>
          </Animated.View>
        )}
      </View>
    </View>
  );
}

const styles = makeStyles(() => ({
  container: {
    paddingHorizontal: 12,
    paddingTop: 10,
    paddingBottom: 12,
    backgroundColor: toneBg('#ffffff'),
    borderTopWidth: 1,
    borderTopColor: theme.colors.bg.divider,
    gap: 8,
  },
  containerTight: { paddingBottom: 4, paddingTop: 8 },
  row: { flexDirection: 'row', alignItems: 'flex-end', gap: 8 },
  sideBtn: {
    width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100],
  },
  inputBox: {
    flex: 1, minHeight: 44, maxHeight: 132, borderRadius: 22, borderWidth: 1.5,
    paddingHorizontal: 14, justifyContent: 'center',
  },
  input: {
    fontSize: 15, lineHeight: 20, color: theme.colors.text.display, paddingTop: 11, paddingBottom: 11,
    fontFamily: theme.typography.fontFamily.sans, maxHeight: 128,
  },
  actionBtn: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  sendBtn: {
    backgroundColor: theme.colors.fill.brand700,
    shadowColor: theme.colors.brand[700], shadowOpacity: 0.35, shadowRadius: 8, shadowOffset: { width: 0, height: 3 }, elevation: 4,
  },
  micBtn: { backgroundColor: theme.colors.brand[100] },
  chipsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, alignItems: 'center' },
  chip: {
    flexDirection: 'row', alignItems: 'center', gap: 8, paddingLeft: 4, paddingRight: 10, paddingVertical: 4,
    borderRadius: 12, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100],
  },
  chipThumb: { width: 32, height: 32, borderRadius: 8, backgroundColor: theme.colors.brand[100] },
  chipFileIcon: { alignItems: 'center', justifyContent: 'center' },
  chipName: { fontSize: 12, fontWeight: '600', color: theme.colors.text.display },
  chipSize: { fontSize: 10, color: theme.colors.text.muted },
  chipRemove: {
    position: 'absolute', top: -6, right: -6, width: 18, height: 18, borderRadius: 9,
    backgroundColor: theme.colors.fill.ink, alignItems: 'center', justifyContent: 'center',
  },
  chipCount: { fontSize: 11, color: theme.colors.text.muted, marginLeft: 2 },
  transcribing: {
    flexDirection: 'row', alignItems: 'center', gap: 8, alignSelf: 'flex-start',
    backgroundColor: theme.colors.brand[50], paddingHorizontal: 12, paddingVertical: 6, borderRadius: 14,
  },
  transcribingText: { fontSize: 12, fontWeight: '600', color: theme.colors.brand[700] },
  voiceRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 4 },
  voiceCancel: {
    width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center',
    backgroundColor: toneBg('#f1f5f9'),
  },
  voiceMid: { flex: 1, gap: 4 },
  voiceTop: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.colors.semantic.danger },
  voiceTime: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display, fontVariant: ['tabular-nums'] },
  voiceHint: { fontSize: 12, color: theme.colors.text.muted, marginLeft: 4 },
  wave: { flexDirection: 'row', alignItems: 'center', height: 32, gap: 3 },
  waveBar: { width: 3, borderRadius: 2 },
  stopWrap: { width: 52, height: 52, alignItems: 'center', justifyContent: 'center' },
  pulse: { position: 'absolute', width: 52, height: 52, borderRadius: 26, backgroundColor: theme.colors.semantic.danger },
  stopBtn: {
    width: 52, height: 52, borderRadius: 26, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.semantic.danger,
  },
}));
