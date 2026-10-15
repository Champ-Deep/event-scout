import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, ScrollView, Image, Linking, Pressable, Keyboard } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeInUp, FadeOut, SlideInRight, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withDelay, Easing,
} from 'react-native-reanimated';
import {
  Sparkles, SquarePen, Flame, Send, Users, Mail, MapPin, Paperclip, Mic, AlertCircle, RotateCcw,
  FileText, Camera, ImageIcon, FolderOpen, ChevronRight, Settings, X, Cpu,
} from 'lucide-react-native';
import { Text } from '../../../src/ui/Text';
import { openContact } from '../../../src/features/contacts/open';
import { KeyboardAwareView } from '../../../src/ui/keyboard';
import { theme, toneBg, toneFg, makeStyles, live } from '../../../src/ui/theme';
import { api, errorMessage, type ErrorMessageOptions, type ChatAttachment, type ChatTurn, type RetrievedContact } from '../../../src/lib/api';
import { contactsFeature } from '../../../src/features/contacts';
import { profileStore } from '../../../src/features/profile';
import { useSession } from '../../../src/providers/SessionProvider';
import { PressableScale } from '../../../src/features/onboarding/components';
import { OptionSheet } from '../../../src/features/contacts/components/OptionSheet';
import { ChatComposer } from '../../../src/features/ask/components/ChatComposer';
import { Markdown } from '../../../src/features/ask/Markdown';
import { RichAnswer } from '../../../src/features/ask/openui/RichAnswer';
import { useVoiceRecorder } from '../../../src/features/ask/useVoiceRecorder';
import { pickAttachments, isImage, AttachmentError, MAX_ATTACHMENTS, type AttachSource } from '../../../src/features/ask/attachments';
import { askHandoff } from '../../../src/features/ask/handoff';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  prompt?: string | undefined; // what was actually sent, when it differs from what's shown (carried over from a contact's sheet)
  via?: 'voice' | 'text';
  attachments?: ChatAttachment[];
  contacts?: RetrievedContact[] | undefined;
  quickReplies?: string[];
  model?: string;
  format?: 'openui' | 'text' | undefined; // 'openui': content may include OpenUI blocks (RichAnswer renders them)
  streaming?: boolean; // the answer is still arriving
  at: number;
  retry?: { text: string; attachments: ChatAttachment[]; via: 'voice' | 'text' };
}

const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
const STREAM_FLUSH_MS = 60; // how often a streaming answer re-renders
const timeOf = (t: number) => new Date(t).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

// How this screen words a failed request (see errorMessage).
const ASK_ERRORS: ErrorMessageOptions = {
  offline: "You're offline. Connect to the internet and try again.",
  timeout: 'The assistant took too long to answer. Please try again.',
  unreachable: "Can't reach the EventScout server. Check your connection and try again.",
};

const TEMP_COLORS: Record<string, { bg: string; fg: string }> = live(() => ({
  hot: { bg: toneBg('#fee2e2'), fg: toneFg('#b91c1c') },
  warm: { bg: toneBg('#fef3c7'), fg: toneFg('#b45309') },
  cool: { bg: toneBg('#e0f2fe'), fg: toneFg('#0369a1') }, // = scoreLevel.ts cool
  cold: { bg: toneBg('#e0f2fe'), fg: toneFg('#0369a1') },
}));

/* ------------------------------ small animated bits ------------------------------ */

function GlowOrb() {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withTiming(1, { duration: 2600, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, []);
  const halo = useAnimatedStyle(() => ({ opacity: 0.25 + t.value * 0.35, transform: [{ scale: 1 + t.value * 0.18 }] }));
  const core = useAnimatedStyle(() => ({ transform: [{ rotate: `${-8 + t.value * 16}deg` }, { scale: 1 + t.value * 0.04 }] }));
  return (
    <View style={styles.orbWrap}>
      <Animated.View style={[styles.orbHalo, halo]} />
      <Animated.View style={[styles.orbCore, core]}>
        <Sparkles size={34} color="white" />
      </Animated.View>
    </View>
  );
}

function Dot({ delay }: { delay: number }) {
  const y = useSharedValue(0);
  useEffect(() => {
    y.value = withDelay(delay, withRepeat(withSequence(withTiming(-5, { duration: 280 }), withTiming(0, { duration: 280 })), -1));
  }, []);
  const style = useAnimatedStyle(() => ({ transform: [{ translateY: y.value }] }));
  return <Animated.View style={[styles.typingDot, style]} />;
}

function Thinking({ withFiles }: { withFiles: boolean }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    const a = setTimeout(() => setPhase(1), 2500);
    const b = setTimeout(() => setPhase(2), 9000);
    return () => { clearTimeout(a); clearTimeout(b); };
  }, []);
  const label = [
    withFiles ? 'Reading your files…' : 'Searching your contacts…',
    'Writing an answer…',
    'Still working — longer answers take a moment…',
  ][phase];
  return (
    <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(150)} style={styles.aiRow}>
      <View style={styles.aiAvatar}><Sparkles size={14} color="white" /></View>
      <View style={styles.thinkingBubble}>
        <View style={styles.typingDots}><Dot delay={0} /><Dot delay={140} /><Dot delay={280} /></View>
        <Animated.Text key={phase} entering={FadeIn} style={styles.thinkingText}>{label}</Animated.Text>
      </View>
    </Animated.View>
  );
}

function StatusPill({ mode }: { mode: 'ready' | 'thinking' | 'listening' | 'transcribing' }) {
  const pulse = useSharedValue(1);
  useEffect(() => {
    pulse.value = mode === 'ready'
      ? withTiming(1)
      : withRepeat(withSequence(withTiming(0.3, { duration: 500 }), withTiming(1, { duration: 500 })), -1);
  }, [mode]);
  const dot = useAnimatedStyle(() => ({ opacity: pulse.value }));
  const cfg = {
    ready: { bg: theme.colors.semantic.successBg, fg: theme.colors.semantic.successText, label: 'Ready' },
    thinking: { bg: theme.colors.brand[50], fg: theme.colors.brand[700], label: 'Thinking' },
    listening: { bg: theme.colors.semantic.dangerBg, fg: toneFg('#b91c1c'), label: 'Listening' },
    transcribing: { bg: theme.colors.brand[50], fg: theme.colors.brand[700], label: 'Transcribing' },
  }[mode];
  return (
    <Animated.View layout={LinearTransition.duration(260)} style={[styles.pill, { backgroundColor: cfg.bg }]}>
      <Animated.View style={[styles.pillDot, { backgroundColor: cfg.fg }, dot]} />
      <Animated.Text key={mode} entering={FadeIn.duration(200)} style={[styles.pillText, { color: cfg.fg }]}>{cfg.label}</Animated.Text>
    </Animated.View>
  );
}

/* ---------------------------------- messages ---------------------------------- */

function ContactChip({ c, index, onPress }: { c: RetrievedContact; index: number; onPress: () => void }) {
  const temp = (c.lead_temperature || '').toLowerCase();
  const colors = TEMP_COLORS[temp];
  const initials = (c.name || '?').split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  return (
    <Animated.View entering={FadeInDown.delay(120 + index * 70).duration(260)}>
      <PressableScale onPress={onPress} style={styles.contactChip} accessibilityLabel={`Open ${c.name}`}>
        <View style={styles.contactInitials}><Text style={styles.contactInitialsText}>{initials}</Text></View>
        <View style={{ flex: 1 }}>
          <Text style={styles.contactName} numberOfLines={1}>{c.name || 'Unnamed contact'}</Text>
          <Text style={styles.contactCompany} numberOfLines={1}>
            {c.company_name && c.company_name !== 'N/A' ? c.company_name : 'No company'}
          </Text>
        </View>
        {c.lead_score != null ? (
          <View style={[styles.scoreBadge, colors && { backgroundColor: colors.bg }]}>
            <Text style={[styles.scoreText, colors && { color: colors.fg }]}>{c.lead_score}</Text>
          </View>
        ) : (
          <Text style={styles.notScored}>Not scored</Text>
        )}
        <ChevronRight size={16} color={theme.colors.text.muted} />
      </PressableScale>
    </Animated.View>
  );
}

function UserBubble({ m }: { m: Message }) {
  return (
    <Animated.View entering={SlideInRight.duration(260)} layout={LinearTransition} style={styles.userRow}>
      {!!m.attachments?.length && (
        <View style={styles.userAttachments}>
          {m.attachments.map((a) => isImage(a) ? (
            <Image key={a.uri} source={{ uri: a.uri }} style={styles.userThumb} />
          ) : (
            <View key={a.uri} style={styles.userFile}>
              <FileText size={14} color={theme.colors.brand[700]} />
              <Text style={styles.userFileName} numberOfLines={1}>{a.name}</Text>
            </View>
          ))}
        </View>
      )}
      <View style={styles.userBubble}>
        {m.via === 'voice' && (
          <View style={styles.voiceTag}><Mic size={11} color={theme.colors.fg.brand200} /><Text style={styles.voiceTagText}>Voice</Text></View>
        )}
        <Text style={styles.userText}>{m.content}</Text>
      </View>
      <Text style={styles.meta}>{timeOf(m.at)}</Text>
    </Animated.View>
  );
}

function AssistantBubble({ m, onContact, onQuickReply, disabled }: {
  m: Message; onContact: (id: string) => void; onQuickReply: (q: string) => void; disabled: boolean;
}) {
  return (
    // No layout animation while streaming: the bubble grows many times a second.
    <Animated.View entering={FadeInUp.duration(260)} {...(m.streaming ? {} : { layout: LinearTransition })} style={styles.aiRow}>
      <Animated.View entering={ZoomIn.duration(260)} style={styles.aiAvatar}><Sparkles size={14} color="white" /></Animated.View>
      <View style={{ flex: 1 }}>
        <View style={styles.aiCard}>
          {m.format === 'openui'
            ? <RichAnswer text={m.content} contacts={m.contacts || []} onContact={onContact} streaming={!!m.streaming} />
            : <Markdown text={m.content} />}
          {/* The rich answer already shows contacts as cards: don't list them twice
              (and wait for the whole answer to know whether it does). */}
          {!m.streaming && !!m.contacts?.length && !(m.format === 'openui' && m.content.includes('ContactCard(')) && (
            <View style={styles.contactsBlock}>
              <View style={styles.blockHeader}>
                <Users size={13} color={theme.colors.brand[700]} />
                <Text style={styles.blockHeaderText}>From your contacts</Text>
              </View>
              {m.contacts.slice(0, 4).map((c, i) => (
                <ContactChip key={c.id || i} c={c} index={i} onPress={() => c.id && onContact(c.id)} />
              ))}
            </View>
          )}
        </View>
        {!!m.quickReplies?.length && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.quickRow}>
            {m.quickReplies.map((q, i) => (
              <Animated.View key={q} entering={FadeInDown.delay(200 + i * 60).duration(260)}>
                <PressableScale onPress={() => onQuickReply(q)} disabled={disabled} style={[styles.quick, disabled && { opacity: 0.5 }]}>
                  <Text style={styles.quickText}>{q}</Text>
                </PressableScale>
              </Animated.View>
            ))}
          </ScrollView>
        )}
        <View style={styles.aiMeta}>
          {m.model ? <><Cpu size={11} color={theme.colors.text.muted} /><Text style={styles.meta}>{m.model} · </Text></> : null}
          <Text style={styles.meta}>{m.streaming ? 'Writing…' : timeOf(m.at)}</Text>
        </View>
      </View>
    </Animated.View>
  );
}

function ErrorBubble({ m, onRetry, disabled }: { m: Message; onRetry: () => void; disabled: boolean }) {
  return (
    <Animated.View entering={FadeInDown.duration(260)} layout={LinearTransition} style={styles.errorCard}>
      <AlertCircle size={18} color={theme.colors.semantic.danger} />
      <View style={{ flex: 1, gap: 8 }}>
        <Text style={styles.errorText}>{m.content}</Text>
        {m.retry && (
          <PressableScale onPress={onRetry} disabled={disabled} style={styles.retryBtn}>
            <RotateCcw size={14} color={theme.colors.brand[700]} />
            <Text style={styles.retryText}>Try again</Text>
          </PressableScale>
        )}
      </View>
    </Animated.View>
  );
}

/* ----------------------------------- screen ----------------------------------- */

export default function AskScreen() {
  const router = useRouter();
  const { q: incomingQ, n: incomingN } = useLocalSearchParams<{ q?: string; n?: string }>();
  const { userId } = useSession();
  const voice = useVoiceRecorder();
  const scrollRef = useRef<ScrollView>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [sending, setSending] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [attachSheet, setAttachSheet] = useState(false);
  const [notice, setNotice] = useState<{ text: string; settings?: boolean } | null>(null);
  const [eventName, setEventName] = useState<string | null>(null);
  const [contactCount, setContactCount] = useState<{ n: number; more: boolean } | null>(null);
  const pendingFiles = useRef(false);
  const [streamingId, setStreamingId] = useState<string | null>(null); // the answer that's arriving
  const streamRef = useRef<AbortController | null>(null);

  // Leaving the screen ends the answer's request.
  useEffect(() => () => streamRef.current?.abort(), []);

  // Real context for the header / suggestions: the user's active event and how many contacts the AI can see.
  useFocusEffect(useCallback(() => {
    if (!userId) return;
    let alive = true;
    profileStore.load(userId)
      .then((p) => alive && setEventName(p.current_event_name || null))
      .catch(() => {});
    contactsFeature.listContacts({ limit: 100 })
      .then((res) => alive && setContactCount({ n: res.contacts.length, more: !!res.cursor }))
      .catch(() => alive && setContactCount(null));
    return () => { alive = false; };
  }, [userId]));

  useEffect(() => {
    if (voice.error) setNotice({ text: voice.error, settings: voice.status === 'denied' });
  }, [voice.error, voice.status]);

  const scrollToEnd = () => requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: true }));

  // When the keyboard opens, keep the latest message visible above the composer.
  useEffect(() => {
    const sub = Keyboard.addListener('keyboardDidShow', () => setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 120));
    return () => sub.remove();
  }, []);

  const history = (): ChatTurn[] =>
    messages
      .filter((m) => m.role !== 'error')
      .slice(-12)
      .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.prompt || m.content }));

  const ask = async (text: string, files: ChatAttachment[], via: 'voice' | 'text', replaceId?: string) => {
    const q = text.trim();
    if ((!q && !files.length) || sending) return;
    if (!userId) {
      setNotice({ text: 'Please sign in to use the AI assistant.' });
      return;
    }
    setNotice(null);
    const prior = history();
    const userMsg: Message = {
      id: uid(), role: 'user', via, at: Date.now(), attachments: files,
      content: q || (files.length === 1 ? `About ${files[0]!.name}` : `About these ${files.length} files`),
    };
    setMessages((prev) => [...prev.filter((m) => m.id !== replaceId), userMsg]);
    setInput('');
    setAttachments([]);
    setSending(true);
    pendingFiles.current = files.length > 0;
    scrollToEnd();

    // The answer streams into one bubble (replyId), created when its first words arrive.
    // Updates are batched (STREAM_FLUSH_MS) so a fast model doesn't re-render per token.
    const replyId = uid();
    const controller = new AbortController();
    streamRef.current = controller;
    let live: Pick<Message, 'contacts' | 'format' | 'model'> = { contacts: [], format: 'text' };
    let latest = '';
    let shown = false;
    let flushTimer: ReturnType<typeof setTimeout> | null = null;
    const patchReply = (patch: Partial<Message>) =>
      setMessages((prev) => prev.map((m) => (m.id === replyId ? { ...m, ...patch } : m)));
    const flush = () => {
      flushTimer = null;
      if (shown) return patchReply({ content: latest, ...live });
      shown = true;
      setStreamingId(replyId);
      setMessages((prev) => [...prev, { id: replyId, role: 'assistant', at: Date.now(), content: latest, streaming: true, ...live }]);
    };

    try {
      const res = await api.converseStream(q, userId, {
        history: prior,
        files,
        ui: 'openui',
        signal: controller.signal,
        onMeta: (meta) => { live = { ...live, contacts: meta.retrieved_contacts, format: meta.format === 'openui' ? 'openui' : 'text' }; },
        onModel: (model) => { live = { ...live, model }; },
        onDelta: (answer) => {
          latest = answer;
          if (!flushTimer) flushTimer = setTimeout(flush, shown ? STREAM_FLUSH_MS : 0);
        },
      });
      if (flushTimer) clearTimeout(flushTimer);
      if (__DEV__) {
        // Shows in the Metro terminal: whether this answer can render rich UI at all.
        console.log(`[Ask] format=${res.format} openui_block=${/```openui/.test(res.response || '')} model=${res.model}`);
      }
      const quick = (res.ui_components || []).find((c) => c.type === 'quick_replies')?.data;
      const final: Message = {
        id: replyId,
        role: 'assistant',
        at: Date.now(),
        content: res.response,
        contacts: res.retrieved_contacts || [],
        quickReplies: Array.isArray(quick) ? quick.filter((x: any) => typeof x === 'string' && x.trim()).slice(0, 4) : [],
        model: res.model,
        format: res.format === 'openui' ? 'openui' : 'text',
        streaming: false,
      };
      setMessages((prev) => (prev.some((m) => m.id === replyId) ? prev.map((m) => (m.id === replyId ? final : m)) : [...prev, final]));
    } catch (e: any) {
      if (flushTimer) clearTimeout(flushTimer);
      if (controller.signal.aborted) return; // left the screen
      // Drop the question (and any half answer); the error bubble offers to send it again.
      setMessages((prev) => [...prev.filter((m) => m.id !== userMsg.id && m.id !== replyId), {
        id: uid(), role: 'error', at: Date.now(),
        content: errorMessage(e, "The assistant couldn't answer that. Please try again.", ASK_ERRORS),
        retry: { text: q, attachments: files, via },
      }]);
    } finally {
      if (streamRef.current === controller) streamRef.current = null;
      if (!controller.signal.aborted) {
        setStreamingId(null);
        setSending(false);
        scrollToEnd();
      }
    }
  };

  // A question typed elsewhere (e.g. the Scout feed composer) arrives as ?q= — ask it once.
  // Senders add a nonce (?n=) so tapping the same suggestion again asks again.
  const handledQ = useRef<string | null>(null);
  useEffect(() => {
    const q = typeof incomingQ === 'string' ? incomingQ.trim() : '';
    const key = `${q}|${typeof incomingN === 'string' ? incomingN : ''}`;
    if (!q || !userId || handledQ.current === key) return;
    handledQ.current = key;
    ask(q, [], 'text');
    router.setParams({ q: undefined, n: undefined } as any);
  }, [incomingQ, incomingN, userId]);

  // "Open in Ask" from a contact's sheet: continue that conversation here.
  useFocusEffect(useCallback(() => {
    const carried = askHandoff.take();
    if (!carried) return;
    streamRef.current?.abort();
    const base = Date.now();
    setMessages(carried.map((t, i) => ({
      id: `${base}-h${i}`,
      role: t.role,
      content: t.content,
      prompt: t.prompt,
      format: t.format,
      contacts: t.contacts,
      at: base,
    })));
    setStreamingId(null);
    setSending(false);
    setNotice(null);
    scrollToEnd();
  }, []));

  const sendVoice = async (uri: string | null) => {
    if (!uri || !userId) return;
    setTranscribing(true);
    try {
      const { transcript } = await api.transcribeAudio(uri, userId);
      setTranscribing(false);
      await ask(transcript, attachments, 'voice');
    } catch (e: any) {
      setTranscribing(false);
      setNotice({ text: errorMessage(e, "Couldn't transcribe that. Please try again, or type your question.", ASK_ERRORS) });
    }
  };

  // The recorder stops itself at the time limit: send what was recorded.
  useEffect(() => {
    if (voice.autoUri) {
      const uri = voice.autoUri;
      voice.clearAutoUri();
      sendVoice(uri);
    }
  }, [voice.autoUri]);

  const startVoice = async () => {
    setNotice(null);
    voice.clearError();
    await voice.start();
  };

  const attach = async (source: AttachSource) => {
    try {
      const picked = await pickAttachments(source, MAX_ATTACHMENTS - attachments.length);
      if (picked.length) setAttachments((prev) => [...prev, ...picked].slice(0, MAX_ATTACHMENTS));
    } catch (e: any) {
      setNotice({ text: e instanceof AttachmentError ? e.message : "Couldn't open that file. Please try another one.", settings: e instanceof AttachmentError && /Settings/.test(e.message) });
    }
  };

  const newChat = () => {
    if (sending) return;
    setMessages([]);
    setNotice(null);
  };

  const mode = voice.isRecording ? 'listening' : transcribing ? 'transcribing' : sending ? 'thinking' : 'ready';
  const busy = sending || transcribing || voice.isRecording;
  const eventLabel = eventName ? ` at ${eventName}` : '';

  const suggestions = [
    { icon: <Flame size={18} color={toneFg('#dc2626')} />, tint: toneBg('#fef2f2'), text: 'Who are my hottest leads right now?' },
    { icon: <Send size={18} color={theme.colors.brand[700]} />, tint: theme.colors.brand[50], text: 'Who should I follow up with first, and why?' },
    { icon: <Users size={18} color={toneFg('#0369a1')} />, tint: toneBg('#f0f9ff'), text: `Summarize everyone I've met${eventLabel}` },
    { icon: <Mail size={18} color={toneFg('#059669')} />, tint: toneBg('#ecfdf5'), text: 'Draft a follow-up email to my top lead' },
    ...(eventName ? [{ icon: <MapPin size={18} color={toneFg('#b45309')} />, tint: toneBg('#fffbeb'), text: `Which exhibitors at ${eventName} should I visit?` }] : []),
  ];

  const contextLine = contactCount == null
    ? 'Your AI event partner'
    : contactCount.n === 0
      ? 'No contacts yet — scan a card to give me something to work with'
      : `Knows your ${contactCount.n}${contactCount.more ? '+' : ''} contact${contactCount.n === 1 ? '' : 's'}${eventLabel}`;

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <KeyboardAwareView>
        {/* Header */}
        <Animated.View entering={FadeInDown.duration(260)} style={styles.header}>
          <View style={styles.headerIcon}><Sparkles size={18} color="white" /></View>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>Ask AI</Text>
            <Text style={styles.subtitle} numberOfLines={1}>{contextLine}</Text>
          </View>
          <StatusPill mode={mode} />
          {messages.length > 0 && (
            <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut}>
              <PressableScale onPress={newChat} disabled={sending} style={styles.newChat} accessibilityLabel="New chat">
                <SquarePen size={18} color={theme.colors.text.display} />
              </PressableScale>
            </Animated.View>
          )}
        </Animated.View>

        <ScrollView
          ref={scrollRef}
          style={{ flex: 1 }}
          contentContainerStyle={[styles.feed, messages.length === 0 && { flexGrow: 1 }]}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
          onContentSizeChange={() => messages.length && scrollToEnd()}
        >
          {messages.length === 0 ? (
            <View style={styles.empty}>
              <Animated.View entering={ZoomIn.duration(260)}><GlowOrb /></Animated.View>
              <Animated.Text entering={FadeInDown.delay(120).duration(260)} style={styles.emptyTitle}>What do you want to know?</Animated.Text>
              <Animated.Text entering={FadeInDown.delay(180).duration(260)} style={styles.emptyText}>
                Ask about your contacts, leads and event — by typing, by voice, or with a photo or file.
              </Animated.Text>
              <View style={styles.suggestions}>
                {suggestions.map((s, i) => (
                  <Animated.View key={s.text} entering={FadeInDown.delay(240 + i * 70).duration(260)}>
                    <PressableScale onPress={() => ask(s.text, [], 'text')} disabled={busy} style={styles.suggestion}>
                      <View style={[styles.suggestionIcon, { backgroundColor: s.tint }]}>{s.icon}</View>
                      <Text style={styles.suggestionText}>{s.text}</Text>
                      <ChevronRight size={16} color={theme.colors.text.muted} />
                    </PressableScale>
                  </Animated.View>
                ))}
              </View>
              <Animated.View entering={FadeIn.delay(650)} style={styles.tips}>
                <View style={styles.tip}><Mic size={13} color={theme.colors.brand[600]} /><Text style={styles.tipText}>Tap the mic and speak</Text></View>
                <View style={styles.tip}><Paperclip size={13} color={theme.colors.brand[600]} /><Text style={styles.tipText}>Attach slides, badges, PDFs, CSVs</Text></View>
              </Animated.View>
            </View>
          ) : (
            messages.map((m) =>
              m.role === 'user' ? <UserBubble key={m.id} m={m} />
                : m.role === 'assistant' ? (
                  <AssistantBubble
                    key={m.id} m={m} disabled={busy}
                    onContact={(id) => openContact(router, id, '/ask')}
                    onQuickReply={(q) => ask(q, [], 'text')}
                  />
                ) : (
                  <ErrorBubble key={m.id} m={m} disabled={busy} onRetry={() => m.retry && ask(m.retry.text, m.retry.attachments, m.retry.via, m.id)} />
                ),
            )
          )}
          {sending && !streamingId && <Thinking withFiles={pendingFiles.current} />}
        </ScrollView>

        {notice && (
          <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(150)} style={styles.notice}>
            <AlertCircle size={16} color={toneFg('#b45309')} />
            <Text style={styles.noticeText}>{notice.text}</Text>
            {notice.settings && (
              <Pressable onPress={() => Linking.openSettings()} hitSlop={6} style={styles.noticeAction}>
                <Settings size={14} color={theme.colors.brand[700]} />
                <Text style={styles.noticeActionText}>Settings</Text>
              </Pressable>
            )}
            <Pressable onPress={() => { setNotice(null); voice.clearError(); }} hitSlop={8} accessibilityLabel="Dismiss">
              <X size={16} color={theme.colors.text.muted} />
            </Pressable>
          </Animated.View>
        )}

        <ChatComposer
          value={input}
          onChangeText={setInput}
          attachments={attachments}
          onRemoveAttachment={(uri) => setAttachments((prev) => prev.filter((a) => a.uri !== uri))}
          onAttach={() => setAttachSheet(true)}
          onSend={() => ask(input, attachments, 'text')}
          onMic={startVoice}
          transcribing={transcribing}
          sending={sending}
          placeholder={attachments.length ? 'Ask about the attached file…' : undefined}
          voice={{
            isRecording: voice.isRecording,
            seconds: voice.seconds,
            level: voice.level,
            maxSeconds: voice.maxSeconds,
            onCancel: () => { voice.cancel(); },
            onStop: async () => sendVoice(await voice.stop()),
          }}
        />
      </KeyboardAwareView>

      <OptionSheet
        visible={attachSheet}
        title="Attach to your question"
        message={`Images, PDFs or text/CSV files · up to ${MAX_ATTACHMENTS}, 10 MB each`}
        onClose={() => setAttachSheet(false)}
        options={[
          { label: 'Take a photo', icon: <Camera size={20} color={theme.colors.brand[600]} />, onPress: () => attach('camera') },
          { label: 'Photo library', icon: <ImageIcon size={20} color={theme.colors.brand[600]} />, onPress: () => attach('library') },
          { label: 'Browse files (PDF, CSV, text)', icon: <FolderOpen size={20} color={theme.colors.brand[600]} />, onPress: () => attach('document') },
        ]}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  container: { flex: 1, backgroundColor: theme.colors.bg.app },
  header: {
    flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingVertical: 10,
    backgroundColor: toneBg('#ffffff'), borderBottomWidth: 1, borderBottomColor: theme.colors.bg.divider,
  },
  headerIcon: {
    width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.brand[600],
  },
  title: { fontSize: 22, lineHeight: 28, fontWeight: '800', color: theme.colors.text.display },
  subtitle: { fontSize: 12, lineHeight: 16, color: theme.colors.text.muted },
  pill: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12 },
  pillDot: { width: 6, height: 6, borderRadius: 3 },
  pillText: { fontSize: 11, fontWeight: '700', letterSpacing: 0.3 },
  newChat: {
    width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'),
  },
  feed: { paddingHorizontal: 14, paddingTop: 16, paddingBottom: 24, gap: 14 },

  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 12 },
  orbWrap: { width: 110, height: 110, alignItems: 'center', justifyContent: 'center', marginBottom: 16 },
  orbHalo: { position: 'absolute', width: 110, height: 110, borderRadius: 55, backgroundColor: theme.colors.brand[300] },
  orbCore: {
    width: 76, height: 76, borderRadius: 26, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.brand[600],
    shadowColor: theme.colors.brand[700], shadowOpacity: 0.4, shadowRadius: 14, shadowOffset: { width: 0, height: 6 }, elevation: 8,
  },
  emptyTitle: { fontSize: 22, lineHeight: 28, fontWeight: '800', color: theme.colors.text.display, textAlign: 'center' },
  emptyText: { fontSize: 14, lineHeight: 20, color: theme.colors.text.body, textAlign: 'center', marginTop: 6, maxWidth: 300 },
  suggestions: { alignSelf: 'stretch', gap: 10, marginTop: 22 },
  suggestion: {
    flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 16,
    backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
  suggestionIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  suggestionText: { flex: 1, fontSize: 14, lineHeight: 19, fontWeight: '600', color: theme.colors.text.display },
  tips: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 14, marginTop: 18 },
  tip: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  tipText: { fontSize: 12, color: theme.colors.text.muted },

  userRow: { alignItems: 'flex-end', gap: 4 },
  userAttachments: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'flex-end', gap: 6, maxWidth: '85%' },
  userThumb: { width: 84, height: 84, borderRadius: 14, backgroundColor: theme.colors.brand[100] },
  userFile: {
    flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 10, paddingVertical: 8, borderRadius: 12,
    backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100], maxWidth: 200,
  },
  userFileName: { fontSize: 12, fontWeight: '600', color: theme.colors.text.display, flexShrink: 1 },
  userBubble: {
    backgroundColor: theme.colors.fill.brand700, paddingHorizontal: 14, paddingVertical: 10,
    borderRadius: 20, borderBottomRightRadius: 6, maxWidth: '85%',
  },
  voiceTag: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 2 },
  voiceTagText: { fontSize: 10, fontWeight: '700', color: theme.colors.fg.brand200, letterSpacing: 0.4 },
  userText: { fontSize: 15, lineHeight: 21, color: 'white' },
  meta: { fontSize: 11, color: theme.colors.text.muted },

  aiRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  aiAvatar: {
    width: 28, height: 28, borderRadius: 10, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.brand[600], marginTop: 2,
  },
  aiCard: {
    backgroundColor: toneBg('#ffffff'), borderRadius: 18, borderTopLeftRadius: 6, padding: 14,
    borderWidth: 1, borderColor: theme.colors.bg.divider, gap: 12,
  },
  aiMeta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 6, marginLeft: 4 },
  contactsBlock: { gap: 8, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider, paddingTop: 12 },
  blockHeader: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  blockHeaderText: { fontSize: 11, fontWeight: '700', letterSpacing: 0.5, color: theme.colors.brand[700], textTransform: 'uppercase' },
  contactChip: {
    flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10, borderRadius: 14,
    backgroundColor: toneBg('#f8fafc'), borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
  contactInitials: {
    width: 34, height: 34, borderRadius: 17, alignItems: 'center', justifyContent: 'center',
    backgroundColor: theme.colors.brand[100],
  },
  contactInitialsText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[800] },
  contactName: { fontSize: 14, fontWeight: '600', color: theme.colors.text.display },
  contactCompany: { fontSize: 12, color: theme.colors.text.muted },
  scoreBadge: { minWidth: 34, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 10, backgroundColor: toneBg('#f1f5f9'), alignItems: 'center' },
  scoreText: { fontSize: 13, fontWeight: '800', color: theme.colors.text.display },
  notScored: { fontSize: 11, color: theme.colors.text.muted },
  quickRow: { gap: 8, paddingTop: 10, paddingRight: 8 },
  quick: {
    paddingHorizontal: 12, paddingVertical: 8, borderRadius: 16,
    backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.brand[200],
  },
  quickText: { fontSize: 13, fontWeight: '600', color: theme.colors.brand[700] },

  thinkingBubble: {
    flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 14, paddingVertical: 12,
    backgroundColor: toneBg('#ffffff'), borderRadius: 18, borderTopLeftRadius: 6, borderWidth: 1, borderColor: theme.colors.bg.divider,
  },
  typingDots: { flexDirection: 'row', gap: 4, alignItems: 'center', height: 12 },
  typingDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: theme.colors.brand[500] },
  thinkingText: { fontSize: 13, color: theme.colors.text.body },

  errorCard: {
    flexDirection: 'row', gap: 10, padding: 14, borderRadius: 16,
    backgroundColor: theme.colors.semantic.dangerBg, borderWidth: 1, borderColor: toneBg('#fecaca'),
  },
  errorText: { fontSize: 14, lineHeight: 20, color: toneFg('#7f1d1d') },
  retryBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start',
    paddingHorizontal: 12, paddingVertical: 7, borderRadius: 14, backgroundColor: toneBg('#ffffff'),
  },
  retryText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },

  notice: {
    flexDirection: 'row', alignItems: 'center', gap: 8, marginHorizontal: 12, marginBottom: 8,
    paddingHorizontal: 12, paddingVertical: 10, borderRadius: 14,
    backgroundColor: theme.colors.semantic.warningBg, borderWidth: 1, borderColor: theme.colors.semantic.warningBorder,
  },
  noticeText: { flex: 1, fontSize: 13, lineHeight: 18, color: toneFg('#78350f') },
  noticeAction: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  noticeActionText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },
}));
