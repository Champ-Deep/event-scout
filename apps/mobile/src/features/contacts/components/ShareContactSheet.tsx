import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Pressable, ScrollView, Share, Platform, ActivityIndicator, Linking } from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { GestureDetector } from 'react-native-gesture-handler';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, useAnimatedReaction, withTiming, withSequence, withSpring, interpolateColor, type SharedValue,
} from 'react-native-reanimated';
import {
  X, Check, Share2, Link2, Globe, Briefcase, Building2, Mail, Smartphone, Linkedin, CalendarDays, TrendingUp,
  StickyNote, Camera, Eye, Trash2, Clock, AlertCircle, ChevronDown, Copy, Lock, FileText, UserRound, ExternalLink, ListChecks,
} from 'lucide-react-native';
import type { Contact } from '@es/types';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { api, errorMessage, type ErrorMessageOptions, type ContactShareLink, type ShareField } from '../../../lib/api';
import { PressableScale } from '../../onboarding/components';
import { ContactAvatar } from './ContactAvatar';
import { useSheetMotion } from '../../../ui/SheetOverlay';


const EXPIRY: Array<{ label: string; days: number | null }> = [
  { label: '1 day', days: 1 },
  { label: '7 days', days: 7 },
  { label: '30 days', days: 30 },
  { label: 'Never', days: null },
];

// Private by default: notes and the score are the user's own intel, so they start unticked.
const DEFAULT_ON: ShareField[] = ['title', 'company', 'email', 'phone', 'linkedin', 'photo'];

// How a failed request is worded here (see errorMessage).
const SHARE_ERRORS: ErrorMessageOptions = {
  offline: "You're offline — connect to create a share link.",
  unreachable: "Can't reach the server — the link wasn't created.",
};

const fmtDate = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString([], { day: 'numeric', month: 'short' }) : '');

/** Animated tick box: fills brand purple and pops the check in. */
function CheckBox({ on }: { on: boolean }) {
  const t = useSharedValue(on ? 1 : 0);
  useEffect(() => { t.value = withTiming(on ? 1 : 0, { duration: 160 }); }, [on]);
  const BG_FFFFFF = toneBg('#ffffff'); // current mode's colour, read on the JS thread (worklets can't call toneBg)
  const BRAND_600 = theme.colors.brand[600]; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const MUTED = theme.colors.text.muted;
  const style = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(t.value, [0, 1], [BG_FFFFFF, BRAND_600]),
    borderColor: interpolateColor(t.value, [0, 1], [MUTED, BRAND_600]),
    transform: [{ scale: 0.9 + t.value * 0.1 }],
  }));
  return (
    <Animated.View style={[styles.checkbox, style]}>
      {on ? <Animated.View entering={ZoomIn.duration(260)}><Check size={14} color="white" strokeWidth={3} /></Animated.View> : null}
    </Animated.View>
  );
}

/**
 * The share flow, shown as its own screen (/contacts/[id]/share). It used to live in an
 * Android Modal with entering animations, where taps on Share / Copy link never reached the
 * API (no link was ever created from the phone) and the sheet could go blank.
 */
export function ShareContactPanel({ contact, onClose, dragY: dragYProp }: { contact: Contact; onClose: () => void; dragY?: SharedValue<number> }) {
  const visible = true;
  const fullName = [contact.firstName, contact.lastName].filter(Boolean).join(' ') || 'Unnamed contact';
  const insets = useSafeAreaInsets(); // keep the buttons clear of Android's gesture / nav bar

  // Only offer what this contact actually has.
  const options = useMemo(() => {
    const list: Array<{ key: ShareField; label: string; value: string; icon: React.ReactNode; private?: boolean }> = [];
    const ic = (I: any) => <I size={16} color={theme.colors.brand[700]} />;
    if (contact.title) list.push({ key: 'title', label: 'Title', value: contact.title, icon: ic(Briefcase) });
    if (contact.company) list.push({ key: 'company', label: 'Company', value: contact.company, icon: ic(Building2) });
    if (contact.email) list.push({ key: 'email', label: 'Email', value: contact.email, icon: ic(Mail) });
    if (contact.phone) list.push({ key: 'phone', label: 'Phone', value: contact.phone, icon: ic(Smartphone) });
    if (contact.linkedin) list.push({ key: 'linkedin', label: 'LinkedIn', value: contact.linkedin.replace(/^https?:\/\/(www\.)?/, ''), icon: ic(Linkedin) });
    if (contact.photoUrl) list.push({ key: 'photo', label: 'Photo', value: 'Profile picture', icon: ic(Camera) });
    if (contact.eventId) list.push({ key: 'event', label: 'Where you met', value: contact.eventId, icon: ic(CalendarDays) });
    if (contact.score != null) list.push({ key: 'score', label: 'Lead score', value: `${contact.score}/100`, icon: ic(TrendingUp), private: true });
    if (contact.notes) list.push({ key: 'notes', label: 'Your notes', value: contact.notes.replace(/\s+/g, ' '), icon: ic(StickyNote), private: true });
    return list;
  }, [contact]);

  const [selected, setSelected] = useState<Set<ShareField>>(new Set());
  const [expiry, setExpiry] = useState<number | null>(30);
  const [busy, setBusy] = useState<'copy' | 'share' | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const [links, setLinks] = useState<ContactShareLink[]>([]);
  const [linksOpen, setLinksOpen] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [confirmRevoke, setConfirmRevoke] = useState<string | null>(null);
  const made = useRef(new Map<string, ContactShareLink>()); // same selection → reuse its link
  // Link access: a view-only link anyone can open, or just the ticked details as text (no link).
  const [access, setAccess] = useState<'link' | 'text'>('link');
  const [lastLink, setLastLink] = useState<ContactShareLink | null>(null);
  const [copiedWhat, setCopiedWhat] = useState<'link' | 'details'>('link');
  const [status, setStatus] = useState('');
  // Which dropdown is open (NotebookLM-style "General access" / expiry pickers).
  const [menu, setMenu] = useState<'access' | 'expiry' | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(true);

  useEffect(() => {
    if (!visible) return;
    setSelected(new Set(options.filter((o) => DEFAULT_ON.includes(o.key)).map((o) => o.key)));
    setError('');
    setCopied(false);
    setLastLink(null);
    api.listShareLinks(contact.id).then((r) => setLinks(r.links || [])).catch(() => setLinks([]));
  }, [visible, contact.id]);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 2200);
    return () => clearTimeout(t);
  }, [copied]);

  const toggle = (k: ShareField) => setSelected((prev) => {
    const next = new Set(prev);
    next.has(k) ? next.delete(k) : next.add(k);
    return next;
  });
  const allOn = options.length > 0 && options.every((o) => selected.has(o.key));
  const toggleAll = () => setSelected(allOn ? new Set() : new Set(options.map((o) => o.key)));

  const chosen = options.filter((o) => selected.has(o.key));

  const selectionKey = `${chosen.map((o) => o.key).sort().join(',')}|${expiry}`;
  const linkReady = lastLink && made.current.get(selectionKey)?.token === lastLink.token ? lastLink : null;

  const ensureLink = async (): Promise<ContactShareLink> => {
    const fields = chosen.map((o) => o.key);
    const key = selectionKey;
    const cached = made.current.get(key);
    if (cached) { setLastLink(cached); return cached; }
    setStatus('Creating a view-only link…');
    const link = await api.createShareLink(contact.id, fields, expiry);
    if (__DEV__) console.log('[share] link created', link?.url); // the URL carries the share token — dev builds only
    if (!link?.url) throw new Error("The server didn't return a link.");
    setStatus('');
    made.current.set(key, link);
    setLastLink(link);
    setLinks((prev) => [link, ...prev.filter((l) => l.token !== link.token)]);
    return link;
  };

  const shareText = (url?: string) => {
    const lines = [fullName];
    const sub = [selected.has('title') ? contact.title : null, selected.has('company') ? contact.company : null].filter(Boolean).join(' · ');
    if (sub) lines.push(sub);
    if (selected.has('email') && contact.email) lines.push(`Email: ${contact.email}`);
    if (selected.has('phone') && contact.phone) lines.push(`Phone: ${contact.phone}`);
    if (selected.has('linkedin') && contact.linkedin) lines.push(`LinkedIn: ${contact.linkedin}`);
    if (selected.has('event') && contact.eventId) lines.push(`Met at: ${contact.eventId}`);
    if (selected.has('score') && contact.score != null) lines.push(`Lead score: ${contact.score}/100`);
    if (selected.has('notes') && contact.notes) lines.push('', `Notes: ${contact.notes}`);
    if (url) lines.push('', `View & save contact: ${url}`);
    return lines.join('\n');
  };

  /** Put text on the clipboard; if the clipboard isn't available, hand it to the system sheet
   *  (Android's share sheet has a "Copy" action). Returns how it was copied. */
  const putOnClipboard = async (text: string): Promise<'clipboard' | 'sheet'> => {
    try {
      await Clipboard.setStringAsync(text);
      return 'clipboard';
    } catch (err: any) {
      console.warn('[share] clipboard failed, falling back to the share sheet', err?.message);
      await Share.share({ message: text });
      return 'sheet';
    }
  };

  const copyLink = async () => {
    if (busy) return;
    setBusy('copy');
    setError('');
    let link: ContactShareLink | null = null;
    try {
      if (access === 'link') link = await ensureLink();
    } catch (e: any) {
      if (__DEV__) console.warn('[share] link create failed', e?.response?.status, e?.response?.data || e?.message);
      setStatus('');
      setError(errorMessage(e, "Couldn't create the link. Please try again.", SHARE_ERRORS));
      setBusy(null);
      return;
    }
    try {
      const how = await putOnClipboard(link ? link.url : shareText());
      setCopiedWhat(link ? 'link' : 'details');
      setCopied(true);
      setStatus(how === 'clipboard'
        ? (link ? 'Link copied — paste it anywhere' : 'Details copied — paste them anywhere')
        : 'Use “Copy” in the menu that opened');
    } catch (e: any) {
      setError(link ? 'The link was created but couldn’t be copied. Long-press the link above to copy it.' : 'Couldn’t copy. Please try again.');
    } finally {
      setBusy(null);
    }
  };

  const share = async () => {
    if (busy) return;
    setBusy('share');
    setError('');
    setCopied(false);
    let link: ContactShareLink | null = null;
    try {
      if (access === 'link') link = await ensureLink();
    } catch (e: any) {
      if (__DEV__) console.warn('[share] link create failed', e?.response?.status, e?.response?.data || e?.message);
      setStatus('');
      setError(errorMessage(e, "Couldn't create the link. Please try again.", SHARE_ERRORS));
      setBusy(null);
      return;
    }
    try {
      const message = shareText(link?.url);
      if (Platform.OS === 'web' && !(navigator as any).share) {
        await Clipboard.setStringAsync(message);
        setCopied(true);
        return;
      }
      // Opens the system share sheet: every app (WhatsApp, Mail, Slack…) and recent people.
      const result = await Share.share(
        Platform.OS === 'ios' && link ? { message, url: link.url, title: fullName } : { message, title: `Contact: ${fullName}` },
        { dialogTitle: `Share ${fullName}`, subject: `Contact: ${fullName}` },
      );
      if (Platform.OS === 'android') {
        // Android can't tell us whether an app was picked — only that the menu opened.
        setStatus(link ? 'Pick an app in the share menu to send the contact' : 'Pick an app in the share menu to send the details');
      } else {
        setStatus(result.action === Share.sharedAction ? (link ? 'Shared ✓ — the link is active' : 'Shared ✓') : (link ? 'Share closed — your link is still active' : ''));
      }
    } catch (e: any) {
      console.warn('[share] share sheet failed', e?.message);
      setStatus('');
      setError(link ? 'The link was created, but the share menu didn’t open. Tap “Copy link” instead.' : 'The share menu didn’t open. Please try again.');
    } finally {
      setBusy(null);
    }
  };

  const revoke = async (token: string) => {
    setRevoking(token);
    setError('');
    try {
      await api.revokeShareLink(contact.id, token);
      setLinks((prev) => prev.filter((l) => l.token !== token));
      made.current.forEach((v, k) => { if (v.token === token) made.current.delete(k); });
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't turn off that link. Please try again.", SHARE_ERRORS));
    } finally {
      setRevoking(null);
      setConfirmRevoke(null);
    }
  };

  const copyExisting = async (url: string) => {
    setError('');
    try {
      const how = await putOnClipboard(url);
      setStatus(how === 'clipboard' ? 'Link copied — paste it anywhere' : 'Use “Copy” in the menu that opened');
    } catch {
      setError('Couldn’t copy the link. Long-press it to copy.');
      return;
    }
    setCopiedWhat('link');
    setCopied(true);
  };

  // Share button: a little bounce whenever the selection changes.
  const bounce = useSharedValue(1);
  useEffect(() => { bounce.value = withSequence(withSpring(1.04, { damping: 22, overshootClamping: true }), withSpring(1, { damping: 22, overshootClamping: true })); }, [selected.size]);
  const bounceStyle = useAnimatedStyle(() => ({ transform: [{ scale: bounce.value }] }));

  const fieldLabel = (f: ShareField) => options.find((o) => o.key === f)?.label || f;

  // Slide-in, swipe down on the handle / title, back button and slide-out: the shared sheet
  // motion. Open while mounted (it's its own route): closing slides out, then calls onClose.
  const motion = useSheetMotion({ onClose });
  const dismiss = motion.dismiss;
  // The route's backdrop lightens as the sheet moves down (dragY is its read-only mirror).
  useAnimatedReaction(() => motion.offset.value, (y) => { if (dragYProp) dragYProp.value = y; }, [dragYProp]);

  const accessOptions = [
    { key: 'link' as const, icon: Globe, title: 'Anyone with the link', sub: "Can view the ticked details and save the contact. You can turn the link off any time." },
    { key: 'text' as const, icon: Lock, title: 'No link — details only', sub: 'Sends the ticked details as text. Nothing is published.' },
  ];
  const currentAccess = accessOptions.find((a) => a.key === access)!;
  const AccessIcon = currentAccess.icon;
  const expiryLabel = EXPIRY.find((x) => x.days === expiry)?.label || '30 days';

  return (
    // One shared value carries the slide-in, the drag and the slide-out.
    <View style={styles.dialogWrap}>
    <Animated.View style={[styles.dialog, { paddingBottom: Math.max(insets.bottom, 12) }, motion.sheetStyle]}>
      <GestureDetector gesture={motion.pan}>
      <View collapsable={false}>
      <View style={styles.handleArea}><View style={styles.handle} /></View>

      {/* Title — like NotebookLM: Share "<name>" */}
      <Animated.View entering={FadeInDown.delay(60).duration(260)} style={styles.header}>
        <ContactAvatar firstName={contact.firstName} lastName={contact.lastName} photoUrl={contact.photoUrl} size={40} />
        <View style={{ flex: 1 }}>
          <Text style={styles.title} numberOfLines={1}>Share “{fullName}”</Text>
          <Text style={styles.subtitle} numberOfLines={1}>
            {[contact.title, contact.company].filter(Boolean).join(' · ') || 'Contact'}
          </Text>
        </View>
      </Animated.View>
      </View>
      </GestureDetector>
      {/* ✕ outside the drag area so the swipe gesture can't swallow its tap */}
      <Pressable onPress={dismiss} hitSlop={12} style={({ pressed }) => [styles.closeBtn, styles.closeFloating, pressed && { opacity: 0.6 }]}
        accessibilityRole="button" accessibilityLabel="Close">
        <X size={18} color={theme.colors.text.body} />
      </Pressable>

      {/* flexShrink keeps the footer buttons inside the dialog (Android ignores taps outside it). */}
      <ScrollView style={{ flexGrow: 0, flexShrink: 1 }} contentContainerStyle={styles.body} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
        {/* What's shared */}
        <Pressable onPress={() => setDetailsOpen((x) => !x)} style={styles.sectionHead} accessibilityRole="button">
          <ListChecks size={15} color={theme.colors.text.body} />
          <Text style={styles.sectionLabel}>What’s shared</Text>
          <Animated.View key={chosen.length} entering={ZoomIn.duration(260)} style={styles.countPill}>
            <Text style={styles.countText}>{chosen.length + 1}</Text>
          </Animated.View>
          <View style={{ flex: 1 }} />
          {detailsOpen && options.length > 1 ? (
            <Pressable onPress={toggleAll} hitSlop={8}><Text style={styles.link}>{allOn ? 'Clear all' : 'Select all'}</Text></Pressable>
          ) : null}
          <Animated.View style={{ transform: [{ rotate: detailsOpen ? '180deg' : '0deg' }], marginLeft: 8 }}>
            <ChevronDown size={16} color={theme.colors.text.muted} />
          </Animated.View>
        </Pressable>

        {detailsOpen ? (
          <Animated.View entering={FadeInDown.duration(260)} layout={LinearTransition.duration(260)} style={styles.card}>
            <View style={[styles.row, styles.rowLocked]}>
              <View style={styles.rowIcon}><UserRound size={15} color={theme.colors.brand[700]} /></View>
              <View style={{ flex: 1 }}>
                <Text style={styles.rowLabel}>Name</Text>
                <Text style={styles.rowValue} numberOfLines={1}>{fullName}</Text>
              </View>
              <Text style={styles.always}>Always</Text>
            </View>
            {options.map((o, i) => {
              const on = selected.has(o.key);
              return (
                <Animated.View key={o.key} entering={FadeInDown.delay(40 + i * 35).duration(260)}>
                  <Pressable
                    onPress={() => toggle(o.key)}
                    style={({ pressed }) => [styles.row, i === options.length - 1 && { borderBottomWidth: 0 }, pressed && { backgroundColor: theme.colors.bg.surfaceHover }]}
                    accessibilityRole="checkbox"
                    accessibilityState={{ checked: on }}
                    accessibilityLabel={o.label}
                  >
                    <View style={[styles.rowIcon, !on && { opacity: 0.45 }]}>{o.icon}</View>
                    <View style={{ flex: 1 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Text style={[styles.rowLabel, !on && { color: theme.colors.text.muted }]}>{o.label}</Text>
                        {o.private ? <Text style={styles.privateTag}>PRIVATE</Text> : null}
                      </View>
                      <Text style={[styles.rowValue, !on && styles.rowValueOff]} numberOfLines={1}>{o.value}</Text>
                    </View>
                    <CheckBox on={on} />
                  </Pressable>
                </Animated.View>
              );
            })}
          </Animated.View>
        ) : (
          <Animated.View entering={FadeIn} style={styles.summaryChips}>
            {['Name', ...chosen.map((o) => o.label)].map((l) => (
              <View key={l} style={styles.summaryChip}><Text style={styles.summaryChipText}>{l}</Text></View>
            ))}
          </Animated.View>
        )}
        {options.length === 0 ? <Text style={styles.hint}>Only the name will be shared — add details to this contact to share more.</Text> : null}

        {/* General access — NotebookLM-style dropdown */}
        <Text style={[styles.sectionLabel, styles.sectionLabelBlock]}>General access</Text>
        <View style={styles.card}>
          <Pressable onPress={() => setMenu(menu === 'access' ? null : 'access')} style={({ pressed }) => [styles.accessRow, pressed && { backgroundColor: theme.colors.bg.surfaceHover }]}
            accessibilityRole="button" accessibilityLabel={`General access: ${currentAccess.title}`}>
            <Animated.View key={access} entering={ZoomIn.duration(260)} style={[styles.accessIcon, access === 'link' ? styles.accessIconOn : null]}>
              <AccessIcon size={17} color={access === 'link' ? theme.colors.semantic.successText : theme.colors.text.body} />
            </Animated.View>
            <View style={{ flex: 1 }}>
              <View style={styles.dropdownTitleRow}>
                <Text style={styles.dropdownTitle}>{currentAccess.title}</Text>
                <Animated.View style={{ transform: [{ rotate: menu === 'access' ? '180deg' : '0deg' }] }}>
                  <ChevronDown size={16} color={theme.colors.text.display} />
                </Animated.View>
              </View>
              <Text style={styles.hintSmall}>{currentAccess.sub}</Text>
            </View>
          </Pressable>
          {menu === 'access' ? (
            <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={styles.menu}>
              {accessOptions.map((o) => {
                const on = o.key === access;
                const I = o.icon;
                return (
                  <Pressable key={o.key} onPress={() => { setAccess(o.key); setMenu(null); }}
                    style={({ pressed }) => [styles.menuItem, on && styles.menuItemOn, pressed && { opacity: 0.8 }]}
                    accessibilityRole="radio" accessibilityState={{ selected: on }}>
                    <I size={16} color={on ? theme.colors.brand[700] : theme.colors.text.body} />
                    <Text style={[styles.menuText, on && { color: theme.colors.brand[800], fontWeight: '800' }]}>{o.title}</Text>
                    {on ? <Animated.View entering={ZoomIn.duration(260)}><Check size={16} color={theme.colors.brand[700]} /></Animated.View> : null}
                  </Pressable>
                );
              })}
            </Animated.View>
          ) : null}

          {access === 'link' ? (
            <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)}>
              <View style={styles.divider} />
              <Pressable onPress={() => setMenu(menu === 'expiry' ? null : 'expiry')} style={({ pressed }) => [styles.accessRow, pressed && { backgroundColor: theme.colors.bg.surfaceHover }]}
                accessibilityRole="button" accessibilityLabel={`Link expires: ${expiryLabel}`}>
                <View style={styles.accessIcon}><Clock size={16} color={theme.colors.text.body} /></View>
                <Text style={[styles.hintSmall, { flex: 1, fontSize: 13 }]}>Link expires</Text>
                <View style={styles.expiryPill}>
                  <Text style={styles.expiryPillText}>{expiryLabel}</Text>
                  <ChevronDown size={14} color={theme.colors.brand[700]} />
                </View>
              </Pressable>
              {menu === 'expiry' ? (
                <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={styles.expiryMenu}>
                  {EXPIRY.map((x) => {
                    const on = expiry === x.days;
                    return (
                      <PressableScale key={x.label} onPress={() => { setExpiry(x.days); setMenu(null); }} scaleTo={0.92}
                        style={[styles.expiryChip, on && styles.expiryChipOn]} accessibilityRole="radio" accessibilityState={{ selected: on }}>
                        <Text style={[styles.expiryText, on && { color: 'white' }]}>{x.label}</Text>
                      </PressableScale>
                    );
                  })}
                </Animated.View>
              ) : null}
            </Animated.View>
          ) : null}
        </View>

        {/* The link that was just made, with Open + Copy */}
        {access === 'link' && linkReady ? (
          <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(150)} style={styles.readyCard}>
            <View style={styles.readyIcon}><Check size={14} color="white" strokeWidth={3} /></View>
            <View style={{ flex: 1 }}>
              <Text style={styles.readyTitle}>Link ready · {linkReady.expiresAt ? `expires ${fmtDate(linkReady.expiresAt)}` : 'never expires'}</Text>
              <Text style={styles.readyUrl} numberOfLines={2} selectable>{linkReady.url}</Text>
            </View>
            <PressableScale onPress={() => Linking.openURL(linkReady.url).catch(() => setError("Couldn't open the link on this phone."))}
              scaleTo={0.9} style={styles.iconSmall} accessibilityLabel="Open the shared page">
              <ExternalLink size={14} color={theme.colors.brand[700]} />
            </PressableScale>
            <PressableScale onPress={() => copyExisting(linkReady.url)} scaleTo={0.9} style={styles.iconSmall} accessibilityLabel="Copy link">
              <Copy size={14} color={theme.colors.brand[700]} />
            </PressableScale>
          </Animated.View>
        ) : null}

        {/* Links already made for this contact */}
        {links.length > 0 ? (
          <Animated.View entering={FadeIn} layout={LinearTransition.duration(260)} style={{ marginTop: 14 }}>
            <Pressable onPress={() => setLinksOpen((x) => !x)} style={styles.linksToggle} hitSlop={6}>
              <Link2 size={14} color={theme.colors.brand[700]} />
              <Text style={styles.linksToggleText}>Active links ({links.length})</Text>
              <Animated.View style={{ transform: [{ rotate: linksOpen ? '180deg' : '0deg' }] }}>
                <ChevronDown size={16} color={theme.colors.brand[700]} />
              </Animated.View>
            </Pressable>
            {linksOpen ? links.map((l, i) => (
              <Animated.View key={l.token} entering={FadeInDown.delay(i * 50).duration(260)} exiting={FadeOut} layout={LinearTransition} style={styles.linkRow}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.linkFields} numberOfLines={1}>
                    Name{l.fields.length ? `, ${l.fields.map(fieldLabel).join(', ')}` : ''}
                  </Text>
                  <View style={styles.linkMeta}>
                    <Eye size={11} color={theme.colors.text.muted} />
                    <Text style={styles.hintSmall}>
                      {l.viewCount} view{l.viewCount === 1 ? '' : 's'} · made {fmtDate(l.createdAt)} · {l.expiresAt ? `expires ${fmtDate(l.expiresAt)}` : 'never expires'}
                    </Text>
                  </View>
                </View>
                {confirmRevoke === l.token ? (
                  <Animated.View entering={ZoomIn.duration(260)} style={{ flexDirection: 'row', gap: 6 }}>
                    <PressableScale onPress={() => revoke(l.token)} disabled={revoking === l.token} scaleTo={0.92} style={styles.revokeConfirm}>
                      {revoking === l.token ? <ActivityIndicator size="small" color="white" /> : <Text style={styles.revokeConfirmText}>Turn off</Text>}
                    </PressableScale>
                    <PressableScale onPress={() => setConfirmRevoke(null)} scaleTo={0.92} style={styles.iconSmall}><X size={14} color={theme.colors.text.body} /></PressableScale>
                  </Animated.View>
                ) : (
                  <>
                    <PressableScale onPress={() => copyExisting(l.url)} scaleTo={0.9} style={styles.iconSmall} accessibilityLabel="Copy this link">
                      <Copy size={14} color={theme.colors.brand[700]} />
                    </PressableScale>
                    <PressableScale onPress={() => setConfirmRevoke(l.token)} scaleTo={0.9} style={styles.iconSmall} accessibilityLabel="Turn off this link">
                      <Trash2 size={14} color={theme.colors.semantic.danger} />
                    </PressableScale>
                  </>
                )}
              </Animated.View>
            )) : null}
          </Animated.View>
        ) : null}

        {error ? (
          <Animated.View entering={FadeInDown.duration(260)} style={styles.error}>
            <AlertCircle size={15} color={theme.colors.semantic.danger} />
            <Text style={styles.errorText}>{error}</Text>
          </Animated.View>
        ) : null}
      </ScrollView>

      {status ? (
        <Animated.Text key={status} entering={FadeIn.duration(200)} style={styles.status}>{status}</Animated.Text>
      ) : null}

      {/* Footer — NotebookLM: "Copy link" on the left, primary action on the right */}
      <View style={styles.actions}>
        <PressableScale
          onPress={copyLink}
          disabled={!!busy}
          style={[styles.copyBtn, copied && styles.copyBtnDone]}
          leadingIcon={busy === 'copy' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} />
            : copied ? <Animated.View entering={ZoomIn.duration(260)}><Check size={17} color={theme.colors.semantic.successText} strokeWidth={3} /></Animated.View>
            : access === 'link' ? <Link2 size={17} color={theme.colors.brand[700]} /> : <FileText size={17} color={theme.colors.brand[700]} />}
          accessibilityLabel={access === 'link' ? 'Copy link' : 'Copy details'}
        >
          <Text style={[styles.copyText, copied && { color: theme.colors.semantic.successText }]}>
            {copied ? (copiedWhat === 'link' ? 'Link copied!' : 'Copied!') : access === 'link' ? 'Copy link' : 'Copy details'}
          </Text>
        </PressableScale>
        <View style={{ flex: 1 }} />
        <Animated.View style={bounceStyle}>
          <PressableScale
            onPress={share}
            disabled={!!busy}
            style={styles.shareBtn}
            leadingIcon={busy === 'share' ? <ActivityIndicator size="small" color="white" /> : <Share2 size={17} color="white" />}
            accessibilityLabel="Share"
          >
            <Text style={styles.shareText}>Share</Text>
          </PressableScale>
        </Animated.View>
      </View>
    </Animated.View>
    </View>
  );
}

const styles = makeStyles(() => ({
  dialogWrap: { maxHeight: '88%' },
  handleArea: { alignItems: 'center', paddingTop: 10, paddingBottom: 6 },
  dialog: {
    backgroundColor: toneBg('#ffffff'), borderTopLeftRadius: 28, borderTopRightRadius: 28, flexShrink: 1,
    shadowColor: '#0f172a', shadowOpacity: 0.18, shadowRadius: 24, shadowOffset: { width: 0, height: -6 }, elevation: 24,
  },
  handle: { width: 42, height: 5, borderRadius: 3, backgroundColor: toneBg('#cbd5e1') },
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingLeft: 20, paddingRight: 64, paddingTop: 8, paddingBottom: 12 },
  title: { fontSize: 19, lineHeight: 25, fontWeight: '800', color: theme.colors.text.display },
  subtitle: { fontSize: 13, color: theme.colors.text.muted },
  closeBtn: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  closeFloating: { position: 'absolute', top: 26, right: 16, zIndex: 5 },
  body: { paddingHorizontal: 16, paddingBottom: 8 },

  sectionHead: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 4, paddingVertical: 8 },
  sectionLabel: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  sectionLabelBlock: { paddingHorizontal: 4, marginTop: 14, marginBottom: 8 },
  countPill: { minWidth: 22, height: 20, paddingHorizontal: 6, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  countText: { fontSize: 11, fontWeight: '900', color: theme.colors.brand[700] },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },

  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 18, borderWidth: 1, borderColor: theme.colors.bg.divider, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 14, paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: toneBg('#f1f5f9') },
  rowLocked: { backgroundColor: theme.colors.brand[50] },
  rowIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  rowLabel: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  rowValue: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  rowValueOff: { color: theme.colors.text.muted, textDecorationLine: 'line-through' },
  always: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  privateTag: { fontSize: 9, fontWeight: '800', letterSpacing: 0.6, color: toneFg('#b45309'), backgroundColor: toneBg('#fef3c7'), paddingHorizontal: 5, paddingVertical: 1, borderRadius: 4, overflow: 'hidden' },
  checkbox: { width: 22, height: 22, borderRadius: 7, borderWidth: 2, alignItems: 'center', justifyContent: 'center' },
  summaryChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, paddingHorizontal: 4 },
  summaryChip: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  summaryChipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  hint: { fontSize: 13, color: theme.colors.text.muted, marginTop: 8, paddingHorizontal: 4 },
  hintSmall: { fontSize: 12, color: theme.colors.text.muted, lineHeight: 16 },

  accessRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 14, paddingVertical: 12 },
  accessIcon: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  accessIconOn: { backgroundColor: theme.colors.semantic.successBg },
  dropdownTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start' },
  dropdownTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  menu: { marginHorizontal: 12, marginBottom: 12, borderRadius: 14, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'), overflow: 'hidden', ...theme.shadows.card },
  menuItem: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 14, paddingVertical: 12 },
  menuItemOn: { backgroundColor: theme.colors.brand[50] },
  menuText: { flex: 1, fontSize: 14, fontWeight: '600', color: theme.colors.text.display },
  divider: { height: 1, backgroundColor: toneBg('#f1f5f9'), marginHorizontal: 14 },
  expiryPill: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  expiryPillText: { fontSize: 12, fontWeight: '800', color: theme.colors.brand[700] },
  expiryMenu: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, paddingHorizontal: 14, paddingBottom: 12 },
  expiryChip: { paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, backgroundColor: theme.colors.bg.surfaceHover, borderWidth: 1, borderColor: theme.colors.bg.divider },
  expiryChipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  expiryText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.display },

  readyCard: {
    flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 12, padding: 12, borderRadius: 14,
    backgroundColor: theme.colors.semantic.successBg, borderWidth: 1, borderColor: theme.colors.semantic.success,
  },
  readyIcon: { width: 24, height: 24, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.success },
  readyTitle: { fontSize: 12, fontWeight: '800', color: theme.colors.semantic.successText },
  readyUrl: { fontSize: 12, color: theme.colors.text.body, fontFamily: theme.typography.fontFamily.mono, marginTop: 2 },
  iconSmall: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },

  linksToggle: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8, paddingHorizontal: 4 },
  linksToggleText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },
  linkRow: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.bg.surfaceHover, borderRadius: 14, padding: 10, marginBottom: 8 },
  linkFields: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display },
  linkMeta: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  revokeConfirm: { height: 32, paddingHorizontal: 12, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.danger },
  revokeConfirmText: { fontSize: 12, fontWeight: '800', color: 'white' },
  error: { flexDirection: 'row', gap: 8, alignItems: 'flex-start', backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 10, marginTop: 12 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  status: { textAlign: 'center', fontSize: 12, color: theme.colors.text.muted, paddingTop: 6, paddingHorizontal: 16 },
  actions: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingTop: 12, borderTopWidth: 1, borderTopColor: toneBg('#f1f5f9'), marginTop: 4 },
  copyBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 18, height: 46, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.bg.divider },
  copyBtnDone: { borderColor: theme.colors.semantic.success, backgroundColor: theme.colors.semantic.successBg },
  copyText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  shareBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 26, height: 46, borderRadius: 999, backgroundColor: theme.colors.brand[600], ...theme.shadows.active },
  shareText: { fontSize: 15, fontWeight: '800', color: 'white' },
}));
