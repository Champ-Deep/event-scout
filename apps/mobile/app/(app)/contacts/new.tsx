import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, ScrollView, Pressable, TextInput, ActivityIndicator, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { contactHref, useReturnTo } from '../../../src/features/contacts/open';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withSpring, Easing,
} from 'react-native-reanimated';
import {
  X, Check, UserPlus, Camera, ImageIcon, Trash2, Copy, CalendarDays, User, Building2, Briefcase, Mail,
  Smartphone, StickyNote, AlertCircle, Phone, IdCard, MessageSquare,
} from 'lucide-react-native';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, getIsDark, makeStyles } from '../../../src/ui/theme';
import { api, errorMessage, type DuplicateMatch, type ErrorMessageOptions } from '../../../src/lib/api';
import { profileStore } from '../../../src/features/profile';
import { VALIDATION } from '../../../src/config/constants';
import { useSession } from '../../../src/providers/SessionProvider';
import { describeReasons, contactsFeature } from '../../../src/features/contacts';
import { pickContactPhoto, type PhotoSource } from '../../../src/features/contacts/photo';
import { ContactAvatar } from '../../../src/features/contacts/components/ContactAvatar';
import { OptionSheet, type SheetOption } from '../../../src/features/contacts/components/OptionSheet';
import { PressableScale, EditableField, SectionHeader, useKeyboardAwareScroll } from '../../../src/features/onboarding/components';

// How this screen words a failed save (see errorMessage).
const SAVE_ERRORS: ErrorMessageOptions = { offline: "You're offline — the contact wasn't saved. Try again when connected." };

type FieldKey = 'name' | 'company' | 'title' | 'email' | 'phone' | 'notes';
const COMPLETENESS_TOTAL = 7; // six fields + photo

const enter = (i: number) => FadeInDown.delay(80 + i * 70).duration(260);

/** Soft pulsing ring behind the avatar while there's no photo yet. */
function PulseRing({ active, size }: { active: boolean; size: number }) {
  const pulse = useSharedValue(0);
  useEffect(() => {
    pulse.value = active
      ? withRepeat(withSequence(withTiming(1, { duration: 1400, easing: Easing.out(Easing.quad) }), withTiming(0, { duration: 0 })), -1)
      : withTiming(0, { duration: 200 });
  }, [active]);
  const style = useAnimatedStyle(() => ({
    opacity: active ? 0.45 * (1 - pulse.value) : 0,
    transform: [{ scale: 1 + pulse.value * 0.35 }],
  }));
  return <Animated.View pointerEvents="none" style={[styles.pulseRing, { width: size, height: size, borderRadius: size / 2 }, style]} />;
}

/** Animated completeness bar for the preview card. */
function CompletenessBar({ filled }: { filled: number }) {
  const width = useSharedValue(0);
  useEffect(() => {
    width.value = withSpring(filled / COMPLETENESS_TOTAL, { damping: 22, stiffness: 140, overshootClamping: true });
  }, [filled]);
  const style = useAnimatedStyle(() => ({ width: `${width.value * 100}%` }));
  return (
    <View style={styles.progressTrack}>
      <Animated.View style={[styles.progressFill, style]} />
    </View>
  );
}

export default function NewContactScreen() {
  const router = useRouter();
  // Opened from Scout (`from`): Back returns there, and the new contact keeps that origin.
  const { from } = useLocalSearchParams<{ from?: string }>();
  const { userId } = useSession();
  const { scrollProps, keyboardHeight, onInputFocus } = useKeyboardAwareScroll();

  const [values, setValues] = useState<Record<FieldKey, string>>({ name: '', company: '', title: '', email: '', phone: '', notes: '' });
  const [photo, setPhoto] = useState<string | null>(null);
  const [photoSheet, setPhotoSheet] = useState(false);
  const [activeEvent, setActiveEvent] = useState<string | null>(null);
  const [tagEvent, setTagEvent] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [savedOffline, setSavedOffline] = useState(false);
  const [error, setError] = useState('');
  const [touched, setTouched] = useState(false);
  const [dupMatches, setDupMatches] = useState<DuplicateMatch[]>([]);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const refs = useRef<Partial<Record<FieldKey, TextInput | null>>>({});

  // Offer to tag the contact with the active event chosen in onboarding / profile.
  useEffect(() => {
    if (!userId) return;
    profileStore.load(userId)
      .then((p) => setActiveEvent(p.current_event_name || null))
      .catch(() => setActiveEvent(null));
  }, [userId]);

  const set = (key: FieldKey) => (v: string) => {
    setValues((prev) => ({ ...prev, [key]: v }));
    if (error) setError('');
  };

  const v = {
    name: values.name.trim(), company: values.company.trim(), title: values.title.trim(),
    email: values.email.trim(), phone: values.phone.trim(), notes: values.notes.trim(),
  };
  const nameMissing = v.name.length === 0;
  const emailInvalid = v.email.length > 0 && !VALIDATION.EMAIL_REGEX.test(v.email);
  const dirty = useMemo(() => !!photo || Object.values(values).some((x) => x.trim().length > 0), [values, photo]);
  const filled = Object.values(v).filter(Boolean).length + (photo ? 1 : 0);
  const nameParts = v.name.split(/\s+/).filter(Boolean);

  const exit = useReturnTo(from, () => leave());
  const leave = () => (dirty && !saved ? setConfirmDiscard(true) : exit());

  const payload = () => ({
    firstName: nameParts[0] || '',
    lastName: nameParts.slice(1).join(' ') || null,
    company: v.company || null,
    title: v.title || null,
    email: v.email || null,
    phone: v.phone || null,
    notes: v.notes || null,
  });

  const create = async () => {
    setSaving(true);
    setError('');
    try {
      // Offline, this saves on the phone and uploads automatically when the connection is back.
      const created = await contactsFeature.createContact({
        ...payload(),
        eventId: tagEvent && activeEvent ? activeEvent : null,
        source: 'manual',
        photo,
      });
      setSavedOffline(!!(created as any).pendingSync);
      setSaved(true); // brief success check, then open the new contact
      setTimeout(() => router.replace(contactHref(created.id, from) as any), 750);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't add the contact. Please try again.", SAVE_ERRORS));
    } finally {
      setSaving(false);
    }
  };

  const handleSave = async () => {
    if (saving || saved) return;
    setTouched(true);
    if (nameMissing) {
      setError('Add a name to save this contact.');
      refs.current.name?.focus();
      return;
    }
    if (emailInvalid) {
      setError('Please enter a valid email address.');
      refs.current.email?.focus();
      return;
    }
    // You may already have this person: ask before creating a duplicate.
    setSaving(true);
    try {
      const res = await api.checkDuplicates(payload());
      if (res?.matches?.length) {
        setDupMatches(res.matches);
        setSaving(false);
        return;
      }
    } catch {
      /* the check is advisory — never block adding a contact on it */
    }
    setSaving(false);
    await create();
  };

  const choosePhoto = async (source: PhotoSource) => {
    try {
      const picked = await pickContactPhoto(source);
      if (picked) setPhoto(picked);
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't use that photo.", SAVE_ERRORS));
    }
  };

  const photoOptions: SheetOption[] = [
    { label: 'Choose from library', icon: <ImageIcon size={20} color={theme.colors.brand[600]} />, onPress: () => choosePhoto('library') },
    ...(Platform.OS !== 'web'
      ? [{ label: 'Take a photo', icon: <Camera size={20} color={theme.colors.brand[600]} />, onPress: () => choosePhoto('camera') }]
      : []),
    ...(photo
      ? [{ label: 'Remove photo', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: () => setPhoto(null) }]
      : []),
  ];

  const muted = theme.colors.text.muted;
  const brand = theme.colors.brand[600];
  const subtitle = [v.title, v.company].filter(Boolean).join(' · ');

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.formHeader}>
        <PressableScale style={styles.cancelBtn} onPress={leave} scaleTo={0.94}
          leadingIcon={<X size={16} color={theme.colors.text.body} />} accessibilityRole="button">
          <Text style={styles.cancelText}>Cancel</Text>
        </PressableScale>
        <Text style={styles.formTitle}>New contact</Text>
        <PressableScale
          style={[styles.saveBtn, (saving || nameMissing) && styles.saveBtnDisabled]}
          onPress={handleSave}
          disabled={saving || saved}
          scaleTo={0.94}
          leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" strokeWidth={2.5} />}
          accessibilityRole="button"
        >
          <Text style={styles.saveBtnText}>{saving ? 'Saving…' : 'Save'}</Text>
        </PressableScale>
      </View>

      <ScrollView
        {...scrollProps}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: 60 + keyboardHeight }]}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="interactive"
        showsVerticalScrollIndicator={false}
      >
        {/* Live preview card — fills in as you type */}
        <Animated.View entering={FadeInDown.duration(260)} style={styles.previewCard}>
          <View style={styles.previewGlowA} />
          <View style={styles.previewGlowB} />
          <View style={styles.previewTop}>
            <View style={styles.avatarWrap}>
              <PulseRing active={!photo} size={84} />
              <Animated.View entering={ZoomIn.delay(80).duration(260)}>
                <ContactAvatar
                  firstName={nameParts[0] || null}
                  lastName={nameParts.slice(1).join(' ') || null}
                  photoUrl={photo}
                  size={84}
                  editable
                  onPress={() => setPhotoSheet(true)}
                />
              </Animated.View>
            </View>
            <View style={{ flex: 1 }}>
              <Animated.Text key={v.name ? 'named' : 'empty'} entering={FadeIn.duration(180)}
                style={[styles.previewName, !v.name && styles.previewNamePlaceholder]} numberOfLines={2}>
                {v.name || 'New contact'}
              </Animated.Text>
              <Text style={[styles.previewSub, !subtitle && { color: muted }]} numberOfLines={1}>
                {subtitle || 'Title · Company'}
              </Text>
              <Pressable onPress={() => setPhotoSheet(true)} hitSlop={6} style={styles.photoLinkRow}>
                <Camera size={13} color={brand} />
                <Text style={styles.photoLink}>{photo ? 'Change photo' : 'Add photo'}</Text>
              </Pressable>
            </View>
          </View>

          {v.email || v.phone || (tagEvent && activeEvent) ? (
            <Animated.View layout={LinearTransition.duration(260)} style={styles.previewChips}>
              {v.email ? (
                <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(120)} style={styles.previewChip}>
                  <Mail size={12} color={theme.colors.brand[700]} />
                  <Text style={styles.previewChipText} numberOfLines={1}>{v.email}</Text>
                </Animated.View>
              ) : null}
              {v.phone ? (
                <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(120)} style={styles.previewChip}>
                  <Phone size={12} color={theme.colors.brand[700]} />
                  <Text style={styles.previewChipText} numberOfLines={1}>{v.phone}</Text>
                </Animated.View>
              ) : null}
              {tagEvent && activeEvent ? (
                <Animated.View entering={ZoomIn.duration(260)} exiting={FadeOut.duration(120)} style={styles.previewChip}>
                  <CalendarDays size={12} color={theme.colors.brand[700]} />
                  <Text style={styles.previewChipText} numberOfLines={1}>{activeEvent}</Text>
                </Animated.View>
              ) : null}
            </Animated.View>
          ) : null}

          <View style={styles.progressRow}>
            <CompletenessBar filled={filled} />
            <Text style={styles.progressText}>{filled}/{COMPLETENESS_TOTAL} added</Text>
          </View>
        </Animated.View>

        {/* Who they are */}
        <Animated.View entering={enter(1)} style={styles.sectionCard}>
          <SectionHeader icon={<IdCard size={15} color={brand} />} label="WHO THEY ARE" hint="Only the name is required" />
          <EditableField
            inputRef={(r) => { refs.current.name = r; }}
            icon={<User size={16} color={muted} />}
            label="Full name"
            value={values.name}
            onChangeText={set('name')}
            placeholder="e.g. Maya Patel"
            autoCapitalize="words"
            autoFocus
            returnKeyType="next"
            submitBehavior="submit"
            onSubmitEditing={() => refs.current.title?.focus()}
            onFocus={onInputFocus}
            error={touched && nameMissing ? 'A name is required' : undefined}
          />
          <EditableField
            inputRef={(r) => { refs.current.title = r; }}
            icon={<Briefcase size={16} color={muted} />}
            label="Role / title"
            value={values.title}
            onChangeText={set('title')}
            placeholder="e.g. Head of Data"
            autoCapitalize="words"
            returnKeyType="next"
            submitBehavior="submit"
            onSubmitEditing={() => refs.current.company?.focus()}
            onFocus={onInputFocus}
          />
          <EditableField
            inputRef={(r) => { refs.current.company = r; }}
            icon={<Building2 size={16} color={muted} />}
            label="Company"
            value={values.company}
            onChangeText={set('company')}
            placeholder="e.g. Northwind Health"
            autoCapitalize="words"
            returnKeyType="next"
            submitBehavior="submit"
            onSubmitEditing={() => refs.current.email?.focus()}
            onFocus={onInputFocus}
          />
        </Animated.View>

        {/* How to reach them */}
        <Animated.View entering={enter(2)} style={styles.sectionCard}>
          <SectionHeader icon={<MessageSquare size={15} color={brand} />} label="HOW TO REACH THEM" />
          <EditableField
            inputRef={(r) => { refs.current.email = r; }}
            icon={<Mail size={16} color={emailInvalid ? theme.colors.semantic.danger : muted} />}
            label="Email"
            value={values.email}
            onChangeText={set('email')}
            placeholder="name@company.com"
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="next"
            submitBehavior="submit"
            onSubmitEditing={() => refs.current.phone?.focus()}
            onFocus={onInputFocus}
            error={emailInvalid ? "That doesn't look like an email address" : undefined}
          />
          <EditableField
            inputRef={(r) => { refs.current.phone = r; }}
            icon={<Smartphone size={16} color={muted} />}
            label="Phone"
            value={values.phone}
            onChangeText={set('phone')}
            placeholder="+1 415 555 0142"
            keyboardType="phone-pad"
            onFocus={onInputFocus}
          />
        </Animated.View>

        {/* Notes */}
        <Animated.View entering={enter(3)} style={styles.sectionCard}>
          <SectionHeader icon={<StickyNote size={15} color={brand} />} label="NOTES" hint="What you talked about, follow-ups" />
          <EditableField
            inputRef={(r) => { refs.current.notes = r; }}
            icon={<StickyNote size={16} color={muted} />}
            label="Notes"
            value={values.notes}
            onChangeText={set('notes')}
            placeholder="Met at the booth — wants a demo next week"
            multiline
            maxLength={4000}
            onFocus={onInputFocus}
          />
        </Animated.View>

        {/* Tag with the active event (the same tag scans get) */}
        {activeEvent ? (
          <Animated.View entering={enter(4)}>
            <PressableScale
              style={[styles.eventRow, tagEvent && styles.eventRowOn]}
              onPress={() => setTagEvent((x) => !x)}
              scaleTo={0.98}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: tagEvent }}
            >
              <View style={[styles.eventIcon, tagEvent && { backgroundColor: brand }]}>
                <CalendarDays size={16} color={tagEvent ? 'white' : brand} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.eventTitle}>Met at {activeEvent}</Text>
                <Text style={styles.eventSub}>{tagEvent ? 'Shows under This event' : 'Not tagged with an event'}</Text>
              </View>
              <View style={[styles.checkbox, tagEvent && styles.checkboxOn]}>
                {tagEvent ? <Animated.View entering={ZoomIn.duration(260)}><Check size={14} color="white" strokeWidth={3} /></Animated.View> : null}
              </View>
            </PressableScale>
          </Animated.View>
        ) : null}

        {error ? (
          <Animated.View entering={FadeInDown.duration(260)} style={styles.errorBanner}>
            <AlertCircle size={16} color={theme.colors.semantic.danger} />
            <Text style={styles.errorBannerText}>{error}</Text>
          </Animated.View>
        ) : null}

        <Animated.View entering={enter(5)}>
          <PressableScale
            style={[styles.primaryBtn, (saving || nameMissing) && { opacity: 0.75 }]}
            onPress={handleSave}
            disabled={saving || saved}
            leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : <UserPlus size={18} color="white" />}
            accessibilityRole="button"
          >
            <Text style={styles.primaryBtnText}>{saving ? 'Adding contact…' : 'Add contact'}</Text>
          </PressableScale>
        </Animated.View>
      </ScrollView>

      {/* Success moment before opening the new contact */}
      {saved ? (
        <Animated.View entering={FadeIn.duration(150)} style={styles.successOverlay} pointerEvents="auto">
          <Animated.View entering={ZoomIn.duration(260)} style={styles.successCircle}>
            <Check size={40} color="white" strokeWidth={3} />
          </Animated.View>
          <Animated.Text entering={FadeInDown.delay(120).duration(260)} style={styles.successText}>
            {savedOffline ? 'Saved on this phone — it will upload when you’re back online' : 'Contact added'}
          </Animated.Text>
        </Animated.View>
      ) : null}

      <OptionSheet
        visible={photoSheet}
        title={photo ? 'Contact photo' : 'Add a photo'}
        message="Shown on this contact's card and in your contacts list."
        options={photoOptions}
        onClose={() => setPhotoSheet(false)}
      />
      <OptionSheet
        visible={dupMatches.length > 0}
        title="You may already have this contact"
        message={dupMatches.map((m) =>
          `${[m.contact.firstName, m.contact.lastName].filter(Boolean).join(' ') || 'A contact'} (${describeReasons(m.reasons)})`,
        ).join('\n')}
        options={[
          ...dupMatches.slice(0, 2).map((m) => ({
            label: `View ${m.contact.firstName || 'existing contact'}`,
            icon: <Copy size={20} color={brand} />,
            onPress: () => router.replace(contactHref(m.contact.id, from) as any),
          })),
          { label: 'Add anyway', icon: <UserPlus size={20} color={brand} />, onPress: () => { setDupMatches([]); create(); } },
        ]}
        onClose={() => setDupMatches([])}
      />
      <OptionSheet
        visible={confirmDiscard}
        title="Discard this contact?"
        message="What you've entered won't be saved."
        options={[{ label: 'Discard', destructive: true, onPress: () => exit() }]}
        onClose={() => setConfirmDiscard(false)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  formHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12 },
  cancelBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'), paddingHorizontal: 12, paddingVertical: 7, borderRadius: 20 },
  cancelText: { fontSize: 14, color: theme.colors.text.body, fontWeight: '700' },
  formTitle: { fontSize: 18, fontWeight: '800', color: theme.colors.text.display },
  saveBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: theme.colors.fill.brand700, paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20 },
  saveBtnDisabled: { opacity: 0.5 },
  saveBtnText: { fontSize: 14, fontWeight: '700', color: 'white' },
  scrollContent: { paddingHorizontal: 16, paddingTop: 4 },

  previewCard: {
    backgroundColor: theme.colors.bg.surface, borderRadius: 24, padding: 18, marginBottom: 14,
    borderWidth: 1, borderColor: theme.colors.brand[100], overflow: 'hidden', ...theme.shadows.active,
  },
  previewGlowA: { position: 'absolute', top: -70, right: -50, width: 190, height: 190, borderRadius: 95, backgroundColor: theme.colors.brand[50] },
  previewGlowB: { position: 'absolute', bottom: -60, left: -40, width: 140, height: 140, borderRadius: 70, backgroundColor: toneBg('#f0fdfa') },
  previewTop: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  avatarWrap: { width: 84, height: 84, alignItems: 'center', justifyContent: 'center' },
  pulseRing: { position: 'absolute', backgroundColor: theme.colors.brand[300] },
  previewName: { fontSize: 22, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.3, lineHeight: 28 },
  previewNamePlaceholder: { color: theme.colors.text.muted },
  previewSub: { fontSize: 14, color: theme.colors.text.body, marginTop: 2 },
  photoLinkRow: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 8, alignSelf: 'flex-start' },
  photoLink: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  previewChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 14 },
  previewChip: { flexDirection: 'row', alignItems: 'center', gap: 5, maxWidth: '100%', backgroundColor: theme.colors.brand[50], paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999 },
  previewChipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[800], flexShrink: 1 },
  progressRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 16 },
  progressTrack: { flex: 1, height: 6, borderRadius: 3, backgroundColor: theme.colors.bg.surfaceHover, overflow: 'hidden' },
  progressFill: { height: 6, borderRadius: 3, backgroundColor: theme.colors.brand[500] },
  progressText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.muted },

  sectionCard: {
    backgroundColor: theme.colors.bg.surface, borderRadius: 20, paddingHorizontal: 14, paddingTop: 12, paddingBottom: 4,
    marginBottom: 14, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card,
  },

  eventRow: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 14, borderWidth: 1.5, borderColor: theme.colors.bg.divider, marginBottom: 14 },
  eventRowOn: { borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50] },
  eventIcon: { width: 36, height: 36, borderRadius: 11, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  eventTitle: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  eventSub: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },
  checkbox: { width: 24, height: 24, borderRadius: 8, borderWidth: 2, borderColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center' },
  checkboxOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.brand[600] },

  errorBanner: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 14 },
  errorBannerText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: theme.colors.brand[600], borderRadius: 999, paddingVertical: 16, ...theme.shadows.active },
  primaryBtnText: { color: 'white', fontSize: 16, fontWeight: '800' },

  successOverlay: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: getIsDark() ? 'rgba(13,12,20,0.94)' : 'rgba(252,250,255,0.94)',
    alignItems: 'center', justifyContent: 'center', gap: 14,
  },
  successCircle: {
    width: 88, height: 88, borderRadius: 44, backgroundColor: theme.colors.semantic.success,
    alignItems: 'center', justifyContent: 'center', ...theme.shadows.active,
  },
  successText: { fontSize: 18, fontWeight: '800', color: theme.colors.text.display },
}));
