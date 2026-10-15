import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, ScrollView, Pressable, TextInput, ActivityIndicator, Image } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeInUp, FadeOut, ZoomIn, SlideInDown, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withDelay, Easing,
} from 'react-native-reanimated';
import {
  User, Mail, Building2, Briefcase, Smartphone, Linkedin, StickyNote, Check, Trash2, Sparkles, Copy, AlertCircle,
  ScanLine, QrCode, ChevronRight, X, CalendarDays, Repeat, Flame, CheckCircle2,
} from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../src/ui/Text';
import { openContact, replaceWithContact } from '../../../src/features/contacts/open';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { errorMessage, type DuplicateMatch, type ErrorMessageOptions } from '../../../src/lib/api';
import { contactsFeature, describeReasons, type ContactPatch } from '../../../src/features/contacts';
import { isHot, type Scored } from '../../../src/features/contacts/scoreLevel';
import { useEditConflict } from '../../../src/features/contacts/components/ConflictSheet';
import { ContactAvatar } from '../../../src/features/contacts/components/ContactAvatar';
import { OptionSheet } from '../../../src/features/contacts/components/OptionSheet';
import { PressableScale, EditableField, useKeyboardAwareScroll } from '../../../src/features/onboarding/components';
import { KeyboardAwareView } from '../../../src/ui/keyboard';
import { VALIDATION } from '../../../src/config/constants';

type Key = 'name' | 'title' | 'company' | 'email' | 'phone' | 'linkedin' | 'notes';
type Values = Record<Key, string>;
const READ_KEYS: Key[] = ['name', 'title', 'company', 'email', 'phone', 'linkedin'];

const valuesOf = (c: Contact): Values => ({
  name: [c.firstName, c.lastName].filter(Boolean).join(' '),
  title: c.title || '', company: c.company || '', email: c.email || '', phone: c.phone || '',
  linkedin: c.linkedin || '', notes: c.notes || '',
});

// How this screen words a failed save (see errorMessage).
const SAVE_ERRORS: ErrorMessageOptions = {
  offline: "You're offline — changes weren't saved.",
  unreachable: "Can't reach the server — changes weren't saved.",
};

/** Fills to how many fields the scan actually read. */
function ReadMeter({ read, total }: { read: number; total: number }) {
  const w = useSharedValue(0);
  useEffect(() => { w.value = withDelay(300, withTiming(read / total, { duration: 900, easing: Easing.out(Easing.cubic) })); }, [read, total]);
  const style = useAnimatedStyle(() => ({ width: `${w.value * 100}%` }));
  const tone = read >= 4 ? theme.colors.semantic.success : read >= 2 ? theme.colors.semantic.warning : theme.colors.semantic.danger;
  return (
    <View style={styles.meterRow}>
      <View style={styles.meterTrack}><Animated.View style={[styles.meterFill, { backgroundColor: tone }, style]} /></View>
      <Text style={styles.meterText}>{read}/{total} read</Text>
    </View>
  );
}

/** Background auto-score: polls the contact until the real score lands (or gives up). */
function useAutoScore(id: string | undefined, enabled: boolean) {
  const [state, setState] = useState<{ status: 'off' | 'waiting' | 'done' | 'timeout'; score?: number | null; level?: string | null }>({ status: enabled ? 'waiting' : 'off' });
  useEffect(() => {
    if (!enabled || !id) return;
    let alive = true;
    let tries = 0;
    const tick = async () => {
      tries += 1;
      try {
        const c = await contactsFeature.getContact(id);
        if (!alive) return;
        if (c?.score != null) { setState({ status: 'done', score: c.score, level: (c as Scored).level ?? null }); return; }
      } catch { /* keep trying */ }
      if (tries >= 20) { if (alive) setState({ status: 'timeout' }); return; } // ~60 s
      setTimeout(tick, 3000);
    };
    const t = setTimeout(tick, 2500);
    return () => { alive = false; clearTimeout(t); };
  }, [id, enabled]);
  return state;
}

function Shimmer() {
  const t = useSharedValue(0);
  useEffect(() => { t.value = withRepeat(withSequence(withTiming(1, { duration: 700 }), withTiming(0.3, { duration: 700 })), -1); }, []);
  const style = useAnimatedStyle(() => ({ opacity: t.value }));
  return <Animated.View style={style}><Sparkles size={12} color={theme.colors.brand[700]} /></Animated.View>;
}

export default function ScanReviewScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { id, autoScore, source, onDevice } = useLocalSearchParams<{ id: string; autoScore?: string; source?: string; onDevice?: string }>();
  const { scrollProps, onInputFocus } = useKeyboardAwareScroll();

  const [contact, setContact] = useState<Contact | null>(null);
  const [initial, setInitial] = useState<Values | null>(null);
  const [values, setValues] = useState<Values>({ name: '', title: '', company: '', email: '', phone: '', linkedin: '', notes: '' });
  const [loadError, setLoadError] = useState('');
  const [saving, setSaving] = useState<'open' | 'next' | null>(null);
  const [saved, setSaved] = useState<'open' | 'next' | null>(null);
  const [discarding, setDiscarding] = useState(false);
  const [error, setError] = useState('');
  const [touched, setTouched] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [dups, setDups] = useState<DuplicateMatch[]>([]);
  const [photoOpen, setPhotoOpen] = useState(false);
  const refs = useRef<Partial<Record<Key, TextInput | null>>>({});
  const scoring = useAutoScore(id, autoScore === '1');

  useEffect(() => {
    if (!id) return;
    contactsFeature.getContact(id).then((c) => {
      if (!c) { setLoadError('This contact no longer exists.'); return; }
      const v = valuesOf(c);
      setContact(c);
      setInitial(v);
      setValues(v);
    }).catch(() => setLoadError("Couldn't load the scanned contact. Check your connection."));
    // Name/company matches the scan's email/phone check can't catch.
    contactsFeature.getDuplicates(id).then(setDups);
  }, [id]);

  const set = (k: Key) => (t: string) => { setValues((p) => ({ ...p, [k]: t })); if (error) setError(''); };
  const v = useMemo(() => {
    const o = {} as Values;
    (Object.keys(values) as Key[]).forEach((k) => { o[k] = values[k].trim(); });
    return o;
  }, [values]);
  const nameMissing = !v.name;
  const emailInvalid = !!v.email && !VALIDATION.EMAIL_REGEX.test(v.email);
  const linkedinInvalid = !!v.linkedin && !/linkedin\.com\/.{2,}/i.test(v.linkedin);
  const readCount = initial ? READ_KEYS.filter((k) => initial[k].trim()).length : 0;
  const notRead = (k: Key) => !!initial && !initial[k].trim();
  const edited = initial ? (Object.keys(values) as Key[]).filter((k) => values[k].trim() !== initial[k].trim()) : [];

  const finish = (then: 'open' | 'next') => {
    if (!contact) return;
    setSaved(then);
    setTimeout(() => {
      if (then === 'next') router.back(); // back to the scanner, ready for the next card
      // Back from the saved contact returns to the scanner (src/features/contacts/open.ts).
      else replaceWithContact(router, contact.id, '/scan');
    }, 650);
  };

  // Changed elsewhere while this was being reviewed: keep mine / use theirs (/ keep both notes).
  const pendingThen = useRef<'open' | 'next'>('open');
  const conflict = useEditConflict({
    onSaved: () => finish(pendingThen.current),
    // "Use theirs": take the newer values for those fields; other corrections stay to save again.
    onUseTheirs: (server, fields) => {
      if (!server || !initial) return;
      const fresh = valuesOf(server);
      const keys = new Set<Key>();
      fields.forEach((f) => {
        if (f === 'firstName' || f === 'lastName') keys.add('name');
        else if (f in fresh) keys.add(f as Key);
      });
      setContact(server);
      setInitial(fresh);
      setValues((prev) => {
        const next = { ...prev };
        (Object.keys(fresh) as Key[]).forEach((k) => {
          if (keys.has(k) || prev[k].trim() === initial[k].trim()) next[k] = fresh[k];
        });
        return next;
      });
      setError('');
    },
  });

  const save = async (then: 'open' | 'next') => {
    if (!contact || saving || saved) return;
    setTouched(true);
    const bad: Key | null = nameMissing ? 'name' : emailInvalid ? 'email' : linkedinInvalid ? 'linkedin' : null;
    if (bad) {
      setError(bad === 'name' ? 'Add a name before saving.' : bad === 'email' ? 'Please fix the email address.' : 'LinkedIn should be a linkedin.com/in/… link.');
      refs.current[bad]?.focus();
      return;
    }
    setSaving(then);
    setError('');
    pendingThen.current = then;
    let patch: ContactPatch = {};
    try {
      // Only send what was corrected (the scan already created the contact).
      if (edited.length && initial) {
        patch = {};
        if (edited.includes('name')) {
          const parts = v.name.split(/\s+/).filter(Boolean);
          patch.firstName = parts[0] || '';
          patch.lastName = parts.slice(1).join(' ') || null;
        }
        if (edited.includes('title')) patch.title = v.title || null;
        if (edited.includes('company')) patch.company = v.company || null;
        if (edited.includes('email')) patch.email = v.email || null;
        if (edited.includes('phone')) patch.phone = v.phone || null;
        if (edited.includes('linkedin')) patch.linkedin = v.linkedin || null;
        if (edited.includes('notes')) patch.notes = values.notes.trimEnd() || null;
        // base = what was read, so a change made elsewhere meanwhile isn't silently overwritten
        await contactsFeature.updateContact(contact.id, patch, { original: contact });
      }
      finish(then);
    } catch (e: any) {
      if (conflict.handle(e, contact.id, patch)) return;
      setError(errorMessage(e, 'Could not save the contact. Please try again.', SAVE_ERRORS));
    } finally {
      setSaving(null);
    }
  };

  // A bad capture shouldn't leave a junk contact behind: remove it for good.
  const handleDiscard = async () => {
    if (!contact || discarding) return;
    setDiscarding(true);
    setError('');
    try {
      // Through the repository: removed for good, dropped from the phone's cache and the lists.
      await contactsFeature.discardContact(contact.id);
      router.back();
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't discard the scan. Please try again."));
    } finally {
      setDiscarding(false);
    }
  };

  if (!contact || !initial) {
    return (
      <View style={[styles.container, styles.center]}>
        {loadError ? (
          <Animated.View entering={FadeIn} style={styles.center}>
            <AlertCircle size={28} color="white" />
            <Text style={styles.loadText}>{loadError}</Text>
            <PressableScale style={styles.whiteBtn} onPress={() => router.back()} accessibilityRole="button">
              <Text style={styles.whiteBtnText}>Back to scanner</Text>
            </PressableScale>
          </Animated.View>
        ) : (
          <>
            <ActivityIndicator color="white" />
            <Text style={styles.loadText}>Loading what was read…</Text>
          </>
        )}
      </View>
    );
  }

  const parts = v.name.split(/\s+/).filter(Boolean);
  const fromQr = source === 'qr' || (contact.source as string) === 'qr';
  const hot = scoring.status === 'done' && isHot(scoring);
  const field = (k: Key, i: number, el: React.ReactNode) => (
    <Animated.View key={k} entering={FadeInDown.delay(240 + i * 50).duration(260)} layout={LinearTransition.duration(260)}>
      {el}
      {notRead(k) && !values[k].trim() && k !== 'notes' ? (
        <Animated.View entering={FadeIn} exiting={FadeOut} style={styles.hintRow}>
          <View style={styles.hintDot} />
          <Text style={styles.hintText}>Not found on the {fromQr ? 'QR code' : 'card'} — add it if you know it</Text>
        </Animated.View>
      ) : !notRead(k) && edited.includes(k) ? (
        <Animated.View entering={FadeIn} exiting={FadeOut} style={styles.hintRow}>
          <View style={[styles.hintDot, { backgroundColor: theme.colors.brand[500] }]} />
          <Text style={styles.hintText}>Corrected · read as “{initial[k]}”</Text>
          <Pressable onPress={() => set(k)(initial[k])} hitSlop={8}><Text style={styles.undo}>Undo</Text></Pressable>
        </Animated.View>
      ) : null}
    </Animated.View>
  );

  return (
    <KeyboardAwareView style={styles.container}>
      <Pressable style={styles.backdrop} onPress={() => router.back()} accessibilityLabel="Back to scanner" />

      <Animated.View entering={SlideInDown.duration(260)} style={[styles.sheet, { paddingBottom: Math.max(insets.bottom, 8) }]}>
        <View style={styles.handleContainer}><View style={styles.handle} /></View>

        <ScrollView
          {...scrollProps}
          style={styles.scrollArea}
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <Animated.View entering={FadeInDown.delay(80).duration(260)} style={styles.headerRow}>
            <View style={{ flex: 1 }}>
              <View style={styles.sourceChip}>
                {fromQr ? <QrCode size={13} color={theme.colors.brand[700]} /> : <ScanLine size={13} color={theme.colors.brand[700]} />}
                <Text style={styles.sourceChipText}>{fromQr ? 'Read from QR code' : 'Read from card'}</Text>
              </View>
              <Text style={styles.title}>Check the details</Text>
              <Text style={styles.subtitle}>Fix anything that was read wrong, then save.</Text>
            </View>
            <PressableScale onPress={() => router.back()} scaleTo={0.9} style={styles.closeBtn} accessibilityLabel="Back to scanner">
              <X size={18} color={theme.colors.text.body} />
            </PressableScale>
          </Animated.View>

          {/* What was captured */}
          <Animated.View entering={FadeInDown.delay(140).duration(260)} style={styles.capturedCard}>
            <View style={styles.capturedTop}>
              <Animated.View entering={ZoomIn.delay(200).duration(260)}>
                <ContactAvatar firstName={parts[0] || null} lastName={parts.slice(1).join(' ') || null} photoUrl={contact.photoUrl} size={60} />
              </Animated.View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.capturedName, !v.name && { color: theme.colors.text.muted }]} numberOfLines={1}>{v.name || 'No name read'}</Text>
                <Text style={styles.capturedSub} numberOfLines={1}>{[v.title, v.company].filter(Boolean).join(' · ') || 'No title or company read'}</Text>
                <View style={styles.chips}>
                  {onDevice ? (
                    <Animated.View entering={ZoomIn.delay(220).duration(260)} style={[styles.chip, { backgroundColor: theme.colors.semantic.infoBg }]}>
                      <Smartphone size={11} color={theme.colors.brand[700]} />
                      <Text style={styles.chipText} numberOfLines={1}>{onDevice === 'offline' ? 'Read offline · uploads later' : 'Read on this phone'}</Text>
                    </Animated.View>
                  ) : null}
                  {contact.eventId ? (
                    <Animated.View entering={ZoomIn.delay(260)} style={styles.chip}>
                      <CalendarDays size={11} color={theme.colors.brand[700]} />
                      <Text style={styles.chipText} numberOfLines={1}>{contact.eventId}</Text>
                    </Animated.View>
                  ) : null}
                  {scoring.status === 'waiting' ? (
                    <Animated.View entering={ZoomIn.delay(300)} style={styles.chip}>
                      <Shimmer />
                      <Text style={styles.chipText}>Scoring…</Text>
                    </Animated.View>
                  ) : scoring.status === 'done' ? (
                    <Animated.View entering={ZoomIn.duration(260)} style={[styles.chip, hot ? { backgroundColor: toneBg('#fee2e2') } : { backgroundColor: theme.colors.semantic.successBg }]}>
                      {hot ? <Flame size={11} color={toneFg('#b91c1c')} /> : <CheckCircle2 size={11} color={theme.colors.semantic.successText} />}
                      <Text style={[styles.chipText, { color: hot ? toneFg('#b91c1c') : theme.colors.semantic.successText }]}>Scored {scoring.score}</Text>
                    </Animated.View>
                  ) : scoring.status === 'timeout' ? (
                    <Animated.View entering={FadeIn} style={[styles.chip, { backgroundColor: theme.colors.bg.surfaceHover }]}>
                      <Text style={[styles.chipText, { color: theme.colors.text.muted }]}>Not scored yet</Text>
                    </Animated.View>
                  ) : null}
                </View>
              </View>
            </View>
            {!fromQr ? <ReadMeter read={readCount} total={READ_KEYS.length} /> : null}
            {contact.photoUrl && !fromQr ? (
              <Pressable onPress={() => setPhotoOpen((x) => !x)} style={styles.photoToggle}>
                <Text style={styles.link}>{photoOpen ? 'Hide the card photo' : 'Compare with the card photo'}</Text>
                <Animated.View style={{ transform: [{ rotate: photoOpen ? '90deg' : '0deg' }] }}><ChevronRight size={14} color={theme.colors.brand[600]} /></Animated.View>
              </Pressable>
            ) : null}
            {photoOpen && contact.photoUrl ? (
              <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)}>
                <Image source={{ uri: contact.photoUrl }} style={styles.cardPhoto} resizeMode="contain" />
              </Animated.View>
            ) : null}
          </Animated.View>

          {dups.map((d, i) => (
            <Animated.View key={d.contact.id} entering={FadeInDown.delay(180 + i * 60).duration(260)}>
              <PressableScale style={styles.dupCard} onPress={() => openContact(router, d.contact.id, '/scan')} scaleTo={0.98} accessibilityRole="button">
                <Copy size={16} color={theme.colors.semantic.warning} />
                <Text style={styles.dupText}>
                  You may already have <Text style={styles.bold}>{[d.contact.firstName, d.contact.lastName].filter(Boolean).join(' ') || 'this person'}</Text> ({describeReasons(d.reasons)}). Tap to check — discard this scan if it's the same person.
                </Text>
                <ChevronRight size={16} color={theme.colors.semantic.warning} />
              </PressableScale>
            </Animated.View>
          ))}

          {field('name', 0, <EditableField inputRef={(r) => { refs.current.name = r; }} icon={<User size={16} color={nameMissing ? theme.colors.semantic.danger : theme.colors.text.muted} />}
            label="Name" value={values.name} onChangeText={set('name')} placeholder="Full name" autoCapitalize="words"
            onFocus={onInputFocus} error={nameMissing && touched ? 'A name is required' : undefined} />)}
          {field('title', 1, <EditableField icon={<Briefcase size={16} color={theme.colors.text.muted} />} label="Role / title" value={values.title}
            onChangeText={set('title')} placeholder="Role or title" autoCapitalize="words" onFocus={onInputFocus} />)}
          {field('company', 2, <EditableField icon={<Building2 size={16} color={theme.colors.text.muted} />} label="Company" value={values.company}
            onChangeText={set('company')} placeholder="Company" autoCapitalize="words" onFocus={onInputFocus} />)}
          {field('email', 3, <EditableField inputRef={(r) => { refs.current.email = r; }} icon={<Mail size={16} color={emailInvalid ? theme.colors.semantic.danger : theme.colors.text.muted} />}
            label="Email" value={values.email} onChangeText={set('email')} placeholder="name@company.com"
            keyboardType="email-address" autoCapitalize="none" autoCorrect={false} onFocus={onInputFocus}
            error={emailInvalid ? "That doesn't look like an email address" : undefined} />)}
          {field('phone', 4, <EditableField icon={<Smartphone size={16} color={theme.colors.text.muted} />} label="Phone" value={values.phone}
            onChangeText={set('phone')} placeholder="Phone number" keyboardType="phone-pad" onFocus={onInputFocus} />)}
          {field('linkedin', 5, <EditableField inputRef={(r) => { refs.current.linkedin = r; }} icon={<Linkedin size={16} color={linkedinInvalid && touched ? theme.colors.semantic.danger : theme.colors.text.muted} />}
            label="LinkedIn" value={values.linkedin} onChangeText={set('linkedin')} placeholder="linkedin.com/in/…"
            keyboardType="url" autoCapitalize="none" autoCorrect={false} onFocus={onInputFocus}
            error={linkedinInvalid && touched ? 'Use a linkedin.com/in/… link' : undefined} />)}
          {field('notes', 6, <EditableField icon={<StickyNote size={16} color={theme.colors.text.muted} />} label="Quick note" value={values.notes}
            onChangeText={set('notes')} placeholder="What did you talk about? Next step?" multiline maxLength={20000} onFocus={onInputFocus} />)}

          {error ? (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.errorBanner}>
              <AlertCircle size={16} color={theme.colors.semantic.danger} />
              <Text style={styles.errorText}>{error}</Text>
            </Animated.View>
          ) : null}
          <View style={{ height: 12 }} />
        </ScrollView>

        <Animated.View entering={FadeInUp.delay(260).duration(260)} style={styles.bottomBar}>
          <PressableScale style={styles.iconBtn} onPress={() => setConfirmDiscard(true)} disabled={discarding || !!saving} scaleTo={0.9}
            accessibilityRole="button" accessibilityLabel="Discard this scan">
            {discarding ? <ActivityIndicator size="small" color={theme.colors.semantic.danger} /> : <Trash2 size={18} color={theme.colors.semantic.danger} />}
          </PressableScale>
          <PressableScale containerStyle={{ flex: 1 }} style={styles.btnOutline} onPress={() => save('next')} disabled={!!saving || discarding} scaleTo={0.95}
            leadingIcon={saving === 'next' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Repeat size={16} color={theme.colors.brand[700]} />}
            accessibilityRole="button">
            <Text style={styles.btnOutlineText}>Save & scan next</Text>
          </PressableScale>
          <PressableScale containerStyle={{ flex: 1 }} style={styles.btnSolid} onPress={() => save('open')} disabled={!!saving || discarding} scaleTo={0.95}
            leadingIcon={saving === 'open' ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" strokeWidth={2.5} />}
            accessibilityRole="button">
            <Text style={styles.btnSolidText}>Save</Text>
          </PressableScale>
        </Animated.View>
      </Animated.View>

      {saved ? (
        <Animated.View entering={FadeIn.duration(150)} style={styles.successOverlay} pointerEvents="auto">
          <Animated.View entering={ZoomIn.duration(260)} style={styles.successCircle}>
            <Check size={40} color="white" strokeWidth={3} />
          </Animated.View>
          <Animated.Text entering={FadeInDown.delay(120).duration(260)} style={styles.successText}>
            {saved === 'next' ? 'Saved — ready for the next card' : 'Contact saved'}
          </Animated.Text>
        </Animated.View>
      ) : null}

      <OptionSheet
        visible={confirmDiscard}
        title="Discard this scan?"
        message="The contact created from this scan will be removed permanently."
        options={[{ label: 'Discard scan', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: handleDiscard }]}
        onClose={() => setConfirmDiscard(false)}
      />

      {conflict.sheet}
    </KeyboardAwareView>
  );
}

const styles = makeStyles(() => ({
  container: { flex: 1, backgroundColor: 'rgba(9,10,16,0.92)', justifyContent: 'flex-end' },
  center: { alignItems: 'center', justifyContent: 'center', gap: 14, padding: 24, flex: 1 },
  loadText: { color: 'white', textAlign: 'center', fontSize: 15 },
  whiteBtn: { backgroundColor: toneBg('#ffffff'), paddingHorizontal: 18, paddingVertical: 11, borderRadius: 999 },
  whiteBtnText: { fontWeight: '800', color: theme.colors.text.display },
  backdrop: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 },
  sheet: { backgroundColor: toneBg('#ffffff'), maxHeight: '92%', flexShrink: 1, borderTopLeftRadius: 32, borderTopRightRadius: 32, overflow: 'hidden' },
  handleContainer: { alignItems: 'center', paddingVertical: 12 },
  handle: { width: 40, height: 4, borderRadius: 2, backgroundColor: toneBg('#cbd5e1') },
  scrollArea: { flexGrow: 0, flexShrink: 1 },
  scrollContent: { paddingHorizontal: 20, paddingTop: 4 },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },

  headerRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10, marginBottom: 16 },
  closeBtn: { width: 36, height: 36, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.bg.surfaceHover },
  sourceChip: { flexDirection: 'row', alignItems: 'center', gap: 5, alignSelf: 'flex-start', backgroundColor: theme.colors.brand[50], paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999, marginBottom: 10 },
  sourceChipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  title: { fontSize: 26, lineHeight: 32, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.4 },
  subtitle: { fontSize: 14, color: theme.colors.text.body, marginTop: 2 },

  capturedCard: { backgroundColor: theme.colors.bg.surfaceHover, borderRadius: 18, padding: 14, marginBottom: 14, borderWidth: 1, borderColor: theme.colors.bg.divider, gap: 12 },
  capturedTop: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  capturedName: { fontSize: 17, lineHeight: 22, fontWeight: '800', color: theme.colors.text.display },
  capturedSub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 8 },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 4, maxWidth: 170, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  chipText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700], flexShrink: 1 },
  meterRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  meterTrack: { flex: 1, height: 6, borderRadius: 3, backgroundColor: theme.colors.bg.divider, overflow: 'hidden' },
  meterFill: { height: 6, borderRadius: 3 },
  meterText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.muted },
  photoToggle: { flexDirection: 'row', alignItems: 'center', gap: 2, alignSelf: 'flex-start' },
  cardPhoto: { width: '100%', height: 180, borderRadius: 12, backgroundColor: toneBg('#0f172a') },

  dupCard: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: theme.colors.semantic.warningBg, borderWidth: 1, borderColor: theme.colors.semantic.warningBorder, borderRadius: 14, padding: 12, marginBottom: 14 },
  dupText: { flex: 1, fontSize: 13, color: theme.colors.text.body, lineHeight: 18 },
  bold: { fontWeight: '800', color: theme.colors.text.display },

  hintRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: -8, marginBottom: 12, paddingHorizontal: 2 },
  hintDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: theme.colors.semantic.warning },
  hintText: { flex: 1, fontSize: 12, color: theme.colors.text.muted },
  undo: { fontSize: 12, fontWeight: '800', color: theme.colors.brand[600] },

  errorBanner: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginTop: 4 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  bottomBar: { flexDirection: 'row', gap: 8, paddingHorizontal: 16, paddingTop: 12, paddingBottom: 8, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  iconBtn: { width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center', borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder },
  btnOutline: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, borderWidth: 1.5, borderColor: theme.colors.brand[200], borderRadius: 999, height: 48 },
  btnOutlineText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  btnSolid: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: theme.colors.brand[600], borderRadius: 999, height: 48, ...theme.shadows.active },
  btnSolidText: { fontSize: 15, fontWeight: '800', color: 'white' },

  successOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(9,10,16,0.85)', alignItems: 'center', justifyContent: 'center', gap: 14 },
  successCircle: { width: 88, height: 88, borderRadius: 44, backgroundColor: theme.colors.semantic.success, alignItems: 'center', justifyContent: 'center' },
  successText: { fontSize: 17, fontWeight: '800', color: 'white', textAlign: 'center', paddingHorizontal: 24 },
}));
