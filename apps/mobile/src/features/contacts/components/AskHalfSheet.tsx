import React, { useEffect, useRef, useState } from 'react';
import { View, Pressable, ActivityIndicator, ScrollView } from 'react-native';
import { GestureDetector } from 'react-native-gesture-handler';
import Animated from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import { Text } from '../../../ui/Text';
import { Icon } from '../../../ui/Icon';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { AskComposer } from '../../ask/components/AskComposer';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';
import { api, errorMessage, type RetrievedContact } from '../../../lib/api';
import { useSession } from '../../../providers/SessionProvider';
import { Markdown } from '../../ask/Markdown';
import { RichAnswer } from '../../ask/openui/RichAnswer';
import { askHandoff } from '../../ask/handoff';

interface AskHalfSheetProps {
  onClose: () => void;
  contactName: string;
  contactId?: string;
  contactCompany?: string | null;
  contactEmail?: string | null;
}

interface Turn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  prompt?: string; // what was sent for a user turn (the question plus who it's about)
  format?: 'openui' | 'text'; // 'openui': content may include OpenUI blocks (RichAnswer renders them)
  contacts?: RetrievedContact[];
  streaming?: boolean; // the answer is still arriving
}

export function AskHalfSheet({ onClose, contactName, contactId, contactCompany, contactEmail }: AskHalfSheetProps) {
  const router = useRouter();
  const { userId } = useSession();
  const firstName = contactName.split(' ')[0] || contactName;
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatRef = useRef<ScrollView>(null);
  const streamRef = useRef<AbortController | null>(null);
  // Slide-in, swipe the handle down, back button, backdrop: the shared sheet motion. It's open
  // while mounted; dismissing slides it out, then calls onClose (the screen unmounts it).
  const motion = useSheetMotion({ onClose });

  // Closing the sheet ends the answer's request.
  useEffect(() => () => streamRef.current?.abort(), []);

  const streaming = turns.some(t => t.streaming);

  const handleSend = async (text: string) => {
    if (!text.trim() || isLoading) return;

    if (!userId) {
      setTurns(prev => [
        ...prev,
        { id: `${Date.now()}-e`, role: 'assistant', content: 'Please log in to use the assistant.' },
      ]);
      return;
    }

    // Scope the question to this contact. /converse takes no contact id, so the question names
    // them with company + email so retrieval finds this exact person, not a namesake.
    const who = [
      contactName,
      contactCompany && contactCompany !== contactName ? `at ${contactCompany}` : '',
      contactEmail ? `(${contactEmail})` : '',
    ].filter(Boolean).join(' ');
    const scopedQuery = `Regarding my contact ${who}: ${text}`;

    const history = turns.map(t => ({ role: t.role, content: t.prompt || t.content }));
    setTurns(prev => [...prev, { id: `${Date.now()}`, role: 'user', content: text, prompt: scopedQuery }]);
    setIsLoading(true);

    // The answer streams into one turn (replyId), created when its first words arrive.
    const replyId = `${Date.now()}-a`;
    const controller = new AbortController();
    streamRef.current = controller;
    let live: Pick<Turn, 'format' | 'contacts'> = { format: 'text', contacts: [] };
    let latest = '';
    let shown = false;
    let flushTimer: ReturnType<typeof setTimeout> | null = null;
    const flush = () => {
      flushTimer = null;
      const turn: Turn = { id: replyId, role: 'assistant', content: latest, streaming: true, ...live };
      if (shown) return setTurns(prev => prev.map(t => (t.id === replyId ? turn : t)));
      shown = true;
      setTurns(prev => [...prev, turn]);
    };
    const settle = (turn: Turn) =>
      setTurns(prev => (prev.some(t => t.id === replyId) ? prev.map(t => (t.id === replyId ? turn : t)) : [...prev, turn]));

    try {
      const res = await api.converseStream(scopedQuery, userId, {
        history,
        ui: 'openui',
        signal: controller.signal,
        onMeta: (meta) => { live = { format: meta.format === 'openui' ? 'openui' : 'text', contacts: meta.retrieved_contacts }; },
        onDelta: (answer) => {
          latest = answer;
          if (!flushTimer) flushTimer = setTimeout(flush, shown ? 60 : 0);
        },
      });
      if (flushTimer) clearTimeout(flushTimer);
      settle({
        id: replyId, role: 'assistant', content: res.response || 'No response',
        format: res.format === 'openui' ? 'openui' : 'text',
        contacts: res.retrieved_contacts || [],
      });
    } catch (err: any) {
      if (flushTimer) clearTimeout(flushTimer);
      if (controller.signal.aborted) return; // sheet closed
      const message = errorMessage(err, 'Failed to get a response. Please try again.', {
        offline: "You're offline. Connect and try again.",
        unreachable: "Can't reach the EventScout server.",
      });
      settle({ id: replyId, role: 'assistant', content: message });
    } finally {
      if (streamRef.current === controller) streamRef.current = null;
      if (!controller.signal.aborted) setIsLoading(false);
    }
  };

  const openInAsk = () => {
    // Carry the conversation over (finished turns only — a half-written answer is dropped).
    askHandoff.put(
      turns
        .filter(t => !t.streaming)
        .map(t => ({ role: t.role, content: t.content, prompt: t.prompt, format: t.format, contacts: t.contacts })),
    );
    onClose();
    router.push('/(app)/ask' as any);
  };

  return (
    // keyboardAware: pads by the keyboard overlap so the composer stays visible (Android doesn't resize the window).
    <SheetOverlay motion={motion} backdropColor="rgba(0,0,0,0.4)" zIndex={999} keyboardAware>
      <Animated.View style={[styles.modalSheet, motion.sheetStyle]}>
        {/* Drag Handle (swipe down to close) */}
        <GestureDetector gesture={motion.pan}>
          <View collapsable={false} style={styles.handleContainer}>
            <View style={styles.handle} />
          </View>
        </GestureDetector>

        {/* Header */}
        <View style={styles.headerRow}>
          <View style={styles.headerLeft}>
             <View style={styles.iconBox}>
               <Icon name="Sparkles" size="xs" color={theme.colors.brand[700]} />
             </View>
             <Text style={styles.headerTitle}>About {contactName}</Text>
          </View>
          <Pressable style={styles.openBtn} onPress={openInAsk}>
             <Text style={styles.openBtnTextPurple}>OPEN IN ASK</Text>
             <Icon name="ChevronRight" size="xs" color={theme.colors.brand[700]} />
          </Pressable>
        </View>

        {/* Chat Content — scrolls, so answers never push the composer off the sheet */}
        <ScrollView
          ref={chatRef}
          style={styles.chatScroll}
          contentContainerStyle={styles.chatContent}
          keyboardShouldPersistTaps="handled"
          onContentSizeChange={() => chatRef.current?.scrollToEnd({ animated: true })}
        >
           <View style={styles.aiBadgeRow}>
             <View style={styles.iconBoxSmall}>
               <Icon name="Sparkles" size="xs" color={theme.colors.brand[700]} />
             </View>
             <Text style={styles.aiBadgeText}>FIELD COPILOT · SCOPED</Text>
           </View>

           {turns.length === 0 && !isLoading && (
             <Text style={styles.emptyText}>Ask anything about {firstName} — notes, talking points, or next steps.</Text>
           )}

           {turns.map((turn) => (
             turn.role === 'user' ? (
               <View key={turn.id} style={styles.userBubbleWrapper}>
                 <View style={styles.userBubble}>
                   <Text style={styles.userBubbleText}>{turn.content}</Text>
                 </View>
               </View>
             ) : (
               <View key={turn.id} style={styles.aiCard}>
                 {turn.format === 'openui' ? (
                   <RichAnswer
                     text={turn.content}
                     streaming={!!turn.streaming}
                     contacts={turn.contacts || []}
                     onContact={(id) => { onClose(); router.push(`/contacts/${id}` as any); }}
                   />
                 ) : (
                   <Markdown text={turn.content} />
                 )}
               </View>
             )
           ))}

           {isLoading && !streaming && (
             <View style={[styles.aiCard, styles.loadingCard]}>
               <ActivityIndicator size="small" color={theme.colors.brand[600]} />
             </View>
           )}
        </ScrollView>

        {/* Composer */}
        <AskComposer
          variant="half-sheet"
          placeholder={`Ask about ${firstName}`}
          onSend={handleSend}
        />

      </Animated.View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  modalSheet: {
    backgroundColor: toneBg('#ffffff'),
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    paddingTop: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
    elevation: 20,
    // ensure composer respects safe area padding inside its own layout
  },
  handleContainer: {
    alignItems: 'center',
    paddingVertical: 12,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: toneBg('#cbd5e1'),
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.bg.divider,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  iconBox: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: theme.colors.brand[50],
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: toneFg('#0f172a'),
  },
  openBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  openBtnTextPurple: {
    fontSize: 11,
    fontWeight: '800',
    color: theme.colors.brand[700],
    letterSpacing: 1,
  },
  openBtnTextOrange: {
    fontSize: 11,
    fontWeight: '800',
    color: toneFg('#ea580c'),
    letterSpacing: 1,
  },
  chatScroll: {
    maxHeight: 380,
    flexGrow: 0,
  },
  chatContent: {
    paddingHorizontal: 24,
    paddingTop: 24,
    paddingBottom: 24,
  },
  userBubbleWrapper: {
    alignItems: 'flex-end',
    marginBottom: 20,
  },
  userBubble: {
    backgroundColor: theme.colors.fill.ink,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
    borderBottomRightRadius: 4,
  },
  userBubbleText: {
    fontSize: 15,
    color: 'white',
  },
  aiBadgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
    marginLeft: 4,
  },
  iconBoxSmall: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: theme.colors.brand[50],
    justifyContent: 'center',
    alignItems: 'center',
  },
  aiBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: toneFg('#475569'),
    letterSpacing: 1,
  },
  aiCard: {
    backgroundColor: toneBg('#ffffff'),
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
    marginBottom: 16,
  },
  loadingCard: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  emptyText: {
    fontSize: 14,
    color: theme.colors.text.muted,
    lineHeight: 20,
    marginBottom: 16,
  },
  aiBodyText: {
    fontSize: 15,
    color: theme.colors.text.display,
    lineHeight: 24,
    marginBottom: 20,
  },
  boldText: {
    fontWeight: '700',
  },
  purpleText: {
    fontWeight: '700',
    color: theme.colors.brand[700],
  },
  detailsTable: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: toneBg('#f1f5f9'),
    paddingTop: 16,
  },
  detailColLeft: {
    flex: 1,
  },
  detailColRight: {
    flex: 1,
    alignItems: 'flex-end',
  },
  detailLabel: {
    fontSize: 11,
    color: toneFg('#94a3b8'),
    marginBottom: 4,
  },
  detailLabelRight: {
    fontSize: 11,
    color: toneFg('#94a3b8'),
    marginBottom: 4,
    textAlign: 'right',
  },
  detailValue: {
    fontSize: 13,
    fontWeight: '500',
    color: theme.colors.text.display,
  },
  detailValueRight: {
    fontSize: 13,
    fontWeight: '500',
    color: theme.colors.text.display,
    textAlign: 'right',
  },
  pillsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  pillActive: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.fill.ink,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    gap: 8,
  },
  pillTextActive: {
    fontSize: 13,
    fontWeight: '600',
    color: 'white',
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: toneBg('#ffffff'),
    borderWidth: 1,
    borderColor: toneBg('#cbd5e1'),
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    gap: 8,
  },
  pillText: {
    fontSize: 13,
    fontWeight: '600',
    color: theme.colors.text.display,
  },
}));
