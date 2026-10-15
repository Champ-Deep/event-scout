// React Native renderers for the Ask AI OpenUI components (definitions live in ./spec.ts).
import React, { createContext, useContext, useState } from 'react';
import { Pressable, View } from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { createLibrary, defineComponent } from '@openuidev/react-lang';
import { Check, ChevronRight, Copy, Lightbulb, AlertTriangle, Sparkles, MapPin, Mail, Linkedin, MessageSquare } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, makeStyles } from '../../../ui/theme';
import { scoreLevel } from '../../contacts/scoreLevel';
import { Markdown } from '../Markdown';
import { defineAskComponents, OPENUI_ROOT } from './spec';

/** What a rendered answer can link to: the contacts the backend retrieved for this turn. */
export interface AskUiContextValue {
  contacts: Array<{ id?: string; name?: string }>;
  onContact?: ((id: string) => void) | undefined;
}
export const AskUiContext = createContext<AskUiContextValue>({ contacts: [] });

const norm = (s?: string) => (s || '').trim().toLowerCase().replace(/\s+/g, ' ');

function useContactId(name?: string): string | undefined {
  const { contacts } = useContext(AskUiContext);
  const n = norm(name);
  if (!n) return undefined;
  return (contacts.find((c) => norm(c.name) === n) || contacts.find((c) => norm(c.name).startsWith(n) || n.startsWith(norm(c.name))))?.id || undefined;
}

const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]!.toUpperCase()).join('') || '?';

function TextR({ props }: any) {
  const { content = '', variant = 'body' } = props;
  if (variant === 'heading') return <Text style={styles.heading}>{content}</Text>;
  if (variant === 'caption') return <Text style={styles.caption}>{content}</Text>;
  return <Markdown text={String(content)} />;
}

function ContactCardR({ props }: any) {
  const { name = '', company, title, score, note } = props;
  const { onContact } = useContext(AskUiContext);
  const id = useContactId(name);
  const level = typeof score === 'number' ? scoreLevel(score) : null;
  const sub = [title, company].filter(Boolean).join(' · ');
  const body = (
    <>
      <View style={styles.avatar}><Text style={styles.avatarText}>{initials(name)}</Text></View>
      <View style={{ flex: 1, minWidth: 0 }}>
        <Text style={styles.contactName} numberOfLines={1}>{name}</Text>
        {!!sub && <Text style={styles.contactSub} numberOfLines={1}>{sub}</Text>}
        {!!note && <Text style={styles.contactNote} numberOfLines={2}>{note}</Text>}
      </View>
      {level && (
        <View style={[styles.scorePill, { backgroundColor: level.bg }]}>
          <Text style={[styles.scoreText, { color: level.fg }]}>{Math.round(score)}</Text>
        </View>
      )}
      {id && onContact ? <ChevronRight size={16} color={theme.colors.text.muted} /> : null}
    </>
  );
  if (id && onContact) {
    return (
      <Pressable style={({ pressed }) => [styles.contactCard, pressed && styles.pressed]} onPress={() => onContact(id)}
        accessibilityRole="button" accessibilityLabel={`Open ${name}`}>
        {body}
      </Pressable>
    );
  }
  return <View style={styles.contactCard}>{body}</View>;
}

function ContactListR({ props, renderNode }: any) {
  const { title, contacts } = props;
  const items = Array.isArray(contacts) ? contacts : [];
  if (!items.length) return null;
  return (
    <View style={styles.block}>
      {!!title && <Text style={styles.blockTitle}>{title}</Text>}
      <View style={{ gap: 8 }}>{items.map((c: any, i: number) => <View key={i}>{renderNode(c)}</View>)}</View>
    </View>
  );
}

function StatsR({ props }: any) {
  const items = (Array.isArray(props.items) ? props.items : []).filter((s: any) => s && s.label != null).slice(0, 4);
  if (!items.length) return null;
  return (
    <View style={styles.statsRow}>
      {items.map((s: any, i: number) => (
        <View key={i} style={styles.stat}>
          <Text style={styles.statValue} numberOfLines={1}>{String(s.value ?? '–')}</Text>
          <Text style={styles.statLabel} numberOfLines={2}>{String(s.label)}</Text>
        </View>
      ))}
    </View>
  );
}

function BarChartR({ props }: any) {
  const data = (Array.isArray(props.data) ? props.data : [])
    .filter((d: any) => d && d.label != null && Number.isFinite(Number(d.value)))
    .slice(0, 8);
  if (!data.length) return null;
  const max = Math.max(...data.map((d: any) => Number(d.value)), 0) || 1;
  return (
    <View style={styles.block}>
      {!!props.title && <Text style={styles.blockTitle}>{props.title}</Text>}
      <View style={{ gap: 8 }}>
        {data.map((d: any, i: number) => (
          <View key={i} style={styles.barRow}>
            <Text style={styles.barLabel} numberOfLines={1}>{String(d.label)}</Text>
            <View style={styles.barTrack}>
              <View style={[styles.barFill, { width: `${Math.max(4, (Number(d.value) / max) * 100)}%` }]} />
            </View>
            <Text style={styles.barValue}>{Number(d.value).toLocaleString()}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

function StepsR({ props }: any) {
  const steps = (Array.isArray(props.steps) ? props.steps : []).filter((s: any) => typeof s === 'string' && s.trim());
  if (!steps.length) return null;
  return (
    <View style={styles.block}>
      {!!props.title && <Text style={styles.blockTitle}>{props.title}</Text>}
      <View style={{ gap: 10 }}>
        {steps.map((s: string, i: number) => (
          <View key={i} style={styles.stepRow}>
            <View style={styles.stepNum}><Text style={styles.stepNumText}>{i + 1}</Text></View>
            <Text style={styles.stepText}>{s}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

function CalloutR({ props }: any) {
  const tone = props.tone === 'warning' ? 'warning' : props.tone === 'tip' ? 'tip' : 'insight';
  if (!props.text) return null;
  const Icon = tone === 'warning' ? AlertTriangle : tone === 'tip' ? Lightbulb : Sparkles;
  const color = tone === 'warning' ? theme.colors.semantic.warning : theme.colors.brand[700];
  return (
    <View style={[styles.callout, tone === 'warning' ? styles.calloutWarning : styles.calloutInsight]}>
      <Icon size={16} color={color} />
      <Text style={styles.calloutText}>{props.text}</Text>
    </View>
  );
}

function DraftR({ props }: any) {
  const { channel = 'email', to, subject, body = '' } = props;
  const [copied, setCopied] = useState(false);
  const Icon = channel === 'linkedin' ? Linkedin : channel === 'text' ? MessageSquare : Mail;
  const label = channel === 'linkedin' ? 'LinkedIn message' : channel === 'text' ? 'Text message' : 'Email';
  const copy = async () => {
    await Clipboard.setStringAsync(subject ? `Subject: ${subject}\n\n${body}` : body);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  if (!body) return null;
  return (
    <View style={styles.draft}>
      <View style={styles.draftHeader}>
        <Icon size={14} color={theme.colors.brand[700]} />
        <Text style={styles.draftKind}>{label}{to ? ` · to ${to}` : ''}</Text>
        <Pressable onPress={copy} style={({ pressed }) => [styles.copyBtn, pressed && styles.pressed]} hitSlop={8}
          accessibilityRole="button" accessibilityLabel="Copy message">
          {copied ? <Check size={13} color={theme.colors.semantic.success} /> : <Copy size={13} color={theme.colors.brand[700]} />}
          <Text style={styles.copyText}>{copied ? 'Copied' : 'Copy'}</Text>
        </Pressable>
      </View>
      {!!subject && <Text style={styles.draftSubject}>{subject}</Text>}
      <Text style={styles.draftBody} selectable>{body}</Text>
    </View>
  );
}

function ExhibitorR({ props }: any) {
  const { name, booth, hall, reason } = props;
  if (!name) return null;
  const where = [hall, booth && `Booth ${booth}`].filter(Boolean).join(' · ');
  return (
    <View style={styles.exhibitor}>
      <View style={styles.exhibitorIcon}><MapPin size={15} color={theme.colors.brand[700]} /></View>
      <View style={{ flex: 1, minWidth: 0 }}>
        <Text style={styles.contactName} numberOfLines={1}>{name}</Text>
        {!!where && <Text style={styles.contactSub}>{where}</Text>}
        {!!reason && <Text style={styles.contactNote}>{reason}</Text>}
      </View>
    </View>
  );
}

function CardR({ props, renderNode }: any) {
  const children = Array.isArray(props.children) ? props.children : [];
  return <View style={styles.card}>{children.map((c: any, i: number) => <View key={i}>{renderNode(c)}</View>)}</View>;
}

export const askLibrary = createLibrary({
  components: defineAskComponents(defineComponent as any, {
    Text: TextR, ContactCard: ContactCardR, ContactList: ContactListR, Stats: StatsR, BarChart: BarChartR,
    Steps: StepsR, Callout: CalloutR, Draft: DraftR, Exhibitor: ExhibitorR, Card: CardR,
  }),
  root: OPENUI_ROOT,
});

const styles = makeStyles(() => ({
  card: { gap: 12 },
  heading: { fontSize: 16, fontWeight: '800', color: theme.colors.text.display, lineHeight: 22 },
  caption: { fontSize: 12, color: theme.colors.text.muted, lineHeight: 17 },
  block: { gap: 8 },
  blockTitle: { fontSize: 12, fontWeight: '800', color: theme.colors.brand[700], letterSpacing: 0.4, textTransform: 'uppercase' },
  pressed: { opacity: 0.7 },
  contactCard: {
    flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10,
    borderRadius: theme.radii.md, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: theme.colors.bg.surface,
  },
  avatar: { width: 34, height: 34, borderRadius: 17, backgroundColor: theme.colors.brand[100], alignItems: 'center', justifyContent: 'center' },
  avatarText: { fontSize: 12, fontWeight: '800', color: theme.colors.brand[800] },
  contactName: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  contactSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  contactNote: { fontSize: 12, color: theme.colors.text.muted, marginTop: 3, lineHeight: 16 },
  scorePill: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: theme.radii.full },
  scoreText: { fontSize: 12, fontWeight: '800' },
  statsRow: { flexDirection: 'row', gap: 8 },
  stat: {
    flex: 1, paddingVertical: 10, paddingHorizontal: 8, borderRadius: theme.radii.md,
    backgroundColor: theme.colors.brand[50], alignItems: 'center',
  },
  statValue: { fontSize: 20, fontWeight: '900', color: theme.colors.brand[800] },
  statLabel: { fontSize: 11, color: theme.colors.text.body, textAlign: 'center', marginTop: 2 },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  barLabel: { width: 92, fontSize: 12, color: theme.colors.text.body },
  barTrack: { flex: 1, height: 10, borderRadius: 5, backgroundColor: theme.colors.bg.divider, overflow: 'hidden' },
  barFill: { height: 10, borderRadius: 5, backgroundColor: theme.colors.brand[500] },
  barValue: { minWidth: 28, textAlign: 'right', fontSize: 12, fontWeight: '700', color: theme.colors.text.display },
  stepRow: { flexDirection: 'row', gap: 10, alignItems: 'flex-start' },
  stepNum: { width: 22, height: 22, borderRadius: 11, backgroundColor: theme.colors.fill.brand900, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  stepNumText: { fontSize: 11, fontWeight: '800', color: '#ffffff' },
  stepText: { flex: 1, fontSize: 14, color: theme.colors.text.display, lineHeight: 20 },
  callout: { flexDirection: 'row', gap: 8, padding: 10, borderRadius: theme.radii.md, alignItems: 'flex-start' },
  calloutInsight: { backgroundColor: theme.colors.brand[50] },
  calloutWarning: { backgroundColor: theme.colors.semantic.warningBg },
  calloutText: { flex: 1, fontSize: 13, color: theme.colors.text.display, lineHeight: 19 },
  draft: { borderRadius: theme.radii.md, borderWidth: 1, borderColor: theme.colors.brand[200], backgroundColor: theme.colors.bg.surface, padding: 12, gap: 6 },
  draftHeader: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  draftKind: { flex: 1, fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  copyBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 4, borderRadius: theme.radii.full, backgroundColor: theme.colors.brand[50] },
  copyText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  draftSubject: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  draftBody: { fontSize: 14, color: theme.colors.text.body, lineHeight: 20 },
  exhibitor: { flexDirection: 'row', gap: 10, alignItems: 'flex-start', padding: 10, borderRadius: theme.radii.md, backgroundColor: theme.colors.bg.surfaceHover },
  exhibitorIcon: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.brand[100], alignItems: 'center', justifyContent: 'center' },
}));
