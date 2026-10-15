import React, { useState, useEffect, useMemo, useRef } from 'react';
import { View, ScrollView, Pressable, TextInput, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useNavigation, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withSpring,
} from 'react-native-reanimated';
import {
  X, Check, Trash2, Copy, User, Briefcase, Building2, Mail, Smartphone, Linkedin, StickyNote, CalendarDays,
  IdCard, MessageSquare, Camera, AlertCircle, Undo2, WifiOff, RotateCw, UserX, CircleSlash,
} from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../../src/ui/Text';
import { theme, toneBg, getIsDark, makeStyles } from '../../../../src/ui/theme';
import { contactsFeature, describeReasons, type ContactPatch } from '../../../../src/features/contacts';
import { profileStore } from '../../../../src/features/profile';
import { useEditConflict } from '../../../../src/features/contacts/components/ConflictSheet';
import { api, errorMessage, type DuplicateMatch, type ErrorMessageOptions } from '../../../../src/lib/api';
import { useSession } from '../../../../src/providers/SessionProvider';
import { ContactAvatar } from '../../../../src/features/contacts/components/ContactAvatar';
import { OptionSheet } from '../../../../src/features/contacts/components/OptionSheet';
import { usePhotoEditor } from '../../../../src/features/contacts/components/usePhotoEditor';
import { NOTES_MAX } from '../../../../src/features/contacts/components/NotesCard';
import { useKeyboardAwareScroll, PressableScale, EditableField, SectionHeader } from '../../../../src/features/onboarding/components';
import { VALIDATION } from '../../../../src/config/constants';

type FieldKey = 'name' | 'title' | 'company' | 'email' | 'phone' | 'linkedin' | 'notes';
type Values = Record<FieldKey, string>;

const FIELD_LABELS: Record<FieldKey, string> = {
  name: 'Name', title: 'Title', company: 'Company', email: 'Email', phone: 'Phone', linkedin: 'LinkedIn', notes: 'Notes',
};

const enter = (i: number) => FadeInDown.delay(80 + i * 70).duration(260);

// How this screen words a failed save (see errorMessage).
const SAVE_ERRORS: ErrorMessageOptions = {
  offline: "You're offline — changes weren't saved. Try again when connected.",
  timeout: 'Saving took too long. Please try again.',
  unreachable: "Can't reach the server — changes weren't saved.",
};

function valuesOf(c: Contact): Values {
  return {
    name: [c.firstName, c.lastName].filter(Boolean).join(' '),
    title: c.title || '',
    company: c.company || '',
    email: c.email || '',
    phone: c.phone || '',
    linkedin: c.linkedin || '',
    notes: c.notes || '',
  };
}

const same = (k: FieldKey, a: string, b: string) =>
  k === 'notes' ? a.trimEnd() === b.trimEnd() : a.trim() === b.trim();

/** "Changed · was X · Undo" under an edited field. */
function ChangeHint({ was, onUndo }: { was: string; onUndo: () => void }) {
  return (
    <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={styles.changeRow}>
      <View style={styles.changeDot} />
      <Text style={styles.changeText} numberOfLines={1}>
        {was ? <>Was <Text style={styles.changeWas}>{was}</Text></> : 'Was empty'}
      </Text>
      <Pressable onPress={onUndo} hitSlop={8} style={styles.undoBtn} accessibilityLabel="Undo this change">
        <Undo2 size={12} color={theme.colors.brand[700]} />
        <Text style={styles.undoText}>Undo</Text>
      </Pressable>
    </Animated.View>
  );
}

function PulseDot() {
  const o = useSharedValue(1);
  useEffect(() => {
    o.value = withRepeat(withSequence(withTiming(0.25, { duration: 650 }), withTiming(1, { duration: 650 })), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ opacity: o.value }));
  return <Animated.View style={[styles.pulseDot, style]} />;
}

export default function ContactEditScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const navigation = useNavigation();
  const { userId } = useSession();
  const { scrollProps, keyboardHeight, onInputFocus } = useKeyboardAwareScroll();

  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading');
  const [reloadKey, setReloadKey] = useState(0);
  const [contact, setContact] = useState<Contact | null>(null);
  const [initial, setInitial] = useState<Values | null>(null);
  const [values, setValues] = useState<Values>({ name: '', title: '', company: '', email: '', phone: '', linkedin: '', notes: '' });
  const [initialEvent, setInitialEvent] = useState('');
  const [event, setEvent] = useState('');
  const [events, setEvents] = useState<string[]>([]);
  const [touched, setTouched] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [dupMatches, setDupMatches] = useState<DuplicateMatch[]>([]);
  const refs = useRef<Partial<Record<FieldKey, TextInput | null>>>({});
  const allowLeave = useRef(false);
  const pendingAction = useRef<any>(null);
  // The contact as this screen loaded it: the "base" of the edit (the photo editor replaces `contact`).
  const loaded = useRef<Contact | null>(null);
  // The photo is saved immediately (not with the Save button), straight to the server.
  const photo = usePhotoEditor(contact, setContact);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    setLoadState('loading');
    contactsFeature.getContact(id)
      .then((c) => {
        if (!alive) return;
        if (!c) { setLoadState('missing'); return; }
        const v = valuesOf(c);
        loaded.current = c;
        setContact(c);
        setInitial(v);
        setValues(v);
        setInitialEvent(c.eventId || '');
        setEvent(c.eventId || '');
        setLoadState('ready');
      })
      .catch(() => alive && setLoadState('error')); // server unreachable — not the same as "not found"
    return () => { alive = false; };
  }, [id, reloadKey]);

  // The user's events (onboarding / profile) for the "Met at" picker.
  useEffect(() => {
    if (!userId) return;
    profileStore.load(userId)
      .then((p) => {
        const list: string[] = [...(p.events || []), p.current_event_name].filter((x): x is string => typeof x === 'string' && !!x.trim());
        setEvents(Array.from(new Set(list)));
      })
      .catch(() => setEvents([]));
  }, [userId]);

  const eventOptions = useMemo(() => {
    const list = [...events];
    if (initialEvent && !list.includes(initialEvent)) list.unshift(initialEvent);
    return list;
  }, [events, initialEvent]);

  const v = useMemo(() => {
    const t = {} as Values;
    (Object.keys(values) as FieldKey[]).forEach((k) => { t[k] = k === 'notes' ? values[k].trimEnd() : values[k].trim(); });
    return t;
  }, [values]);

  const changed = useMemo(() => {
    if (!initial) return [] as string[];
    const list = (Object.keys(values) as FieldKey[]).filter((k) => !same(k, values[k], initial[k])).map((k) => FIELD_LABELS[k]);
    if (event !== initialEvent) list.push('Event');
    return list;
  }, [values, initial, event, initialEvent]);
  const dirty = changed.length > 0;

  const nameMissing = v.name.length === 0;
  const emailInvalid = v.email.length > 0 && !VALIDATION.EMAIL_REGEX.test(v.email);
  const phoneInvalid = v.phone.length > 0 && v.phone.replace(/\D/g, '').length < 6;
  const linkedinInvalid = v.linkedin.length > 0 && !/linkedin\.com\/.{2,}/i.test(v.linkedin);
  const firstInvalid: FieldKey | null = nameMissing ? 'name' : emailInvalid ? 'email' : phoneInvalid ? 'phone' : linkedinInvalid ? 'linkedin' : null;

  // Guard every way out (header, Android back, swipe) while there are unsaved edits.
  useEffect(() => {
    const unsub = navigation.addListener('beforeRemove' as any, (e: any) => {
      if (!dirty || allowLeave.current) return;
      e.preventDefault();
      pendingAction.current = e.data?.action;
      setConfirmDiscard(true);
    });
    return unsub;
  }, [navigation, dirty]);

  const leaveNow = () => {
    allowLeave.current = true;
    if (pendingAction.current) navigation.dispatch(pendingAction.current);
    else router.back();
  };

  // Save button pops when edits start.
  const savePop = useSharedValue(1);
  useEffect(() => {
    if (dirty) savePop.value = withSequence(withSpring(1.04, { damping: 22, overshootClamping: true }), withSpring(1, { damping: 22, overshootClamping: true }));
  }, [dirty]);
  const savePopStyle = useAnimatedStyle(() => ({ transform: [{ scale: savePop.value }] }));

  const shake = useSharedValue(0);
  const shakeStyle = useAnimatedStyle(() => ({ transform: [{ translateX: shake.value }] }));
  const doShake = () => {
    shake.value = withSequence(withTiming(-8, { duration: 50 }), withTiming(8, { duration: 50 }), withTiming(-5, { duration: 50 }), withTiming(0, { duration: 50 }));
  };

  const set = (key: FieldKey) => (text: string) => {
    setValues((prev) => ({ ...prev, [key]: text }));
    if (saveError) setSaveError('');
  };
  const undo = (key: FieldKey) => initial && setValues((prev) => ({ ...prev, [key]: initial[key] }));
  const revertAll = () => {
    if (!initial) return;
    setValues(initial);
    setEvent(initialEvent);
    setSaveError('');
    setTouched(false);
  };

  const buildPatch = (): ContactPatch => {
    if (!initial) return {};
    const patch: ContactPatch = {};
    if (!same('name', values.name, initial.name)) {
      // The API represents names as firstName/lastName; split the single input on save.
      const parts = v.name.split(/\s+/).filter(Boolean);
      patch.firstName = parts[0] || '';
      patch.lastName = parts.slice(1).join(' ') || null;
    }
    if (!same('title', values.title, initial.title)) patch.title = v.title || null;
    if (!same('company', values.company, initial.company)) patch.company = v.company || null;
    if (!same('email', values.email, initial.email)) patch.email = v.email || null;
    if (!same('phone', values.phone, initial.phone)) patch.phone = v.phone || null;
    if (!same('linkedin', values.linkedin, initial.linkedin)) patch.linkedin = v.linkedin || null;
    if (!same('notes', values.notes, initial.notes)) patch.notes = v.notes || null;
    if (event !== initialEvent) patch.eventId = event || null;
    return patch;
  };

  const finishSaved = () => {
    allowLeave.current = true;
    setSaved(true); // brief success check, then back to the contact
    setTimeout(() => router.back(), 700);
  };

  // Saved while this screen was open, somewhere else: pick what to keep.
  const conflict = useEditConflict({
    onSaved: finishSaved,
    // "Use theirs": load the newer contact; edits to the conflicting fields are dropped, other
    // unsaved edits stay (Save again to keep them).
    onUseTheirs: (server, fields) => {
      if (!server || !initial) { setReloadKey((k) => k + 1); return; }
      const fresh = valuesOf(server);
      const keys = new Set<FieldKey>();
      fields.forEach((f) => {
        if (f === 'firstName' || f === 'lastName') keys.add('name');
        else if (f in fresh) keys.add(f as FieldKey);
      });
      loaded.current = server;
      setContact(server);
      setInitial(fresh);
      setValues((prev) => {
        const next = { ...prev };
        (Object.keys(fresh) as FieldKey[]).forEach((k) => {
          // dropped fields, and fields the user hadn't touched, follow the server
          if (keys.has(k) || same(k, prev[k], initial[k])) next[k] = fresh[k];
        });
        return next;
      });
      const serverEvent = server.eventId || '';
      if (fields.includes('eventId') || event === initialEvent) setEvent(serverEvent);
      setInitialEvent(serverEvent);
      setSaveError('');
    },
  });

  const save = async () => {
    if (!contact) return;
    setSaving(true);
    setSaveError('');
    const patch = buildPatch(); // only the fields that changed
    try {
      // base = those fields as this screen loaded them, so an edit made elsewhere meanwhile isn't overwritten
      await contactsFeature.updateContact(contact.id, patch, { original: loaded.current ?? contact });
      finishSaved();
    } catch (e: any) {
      if (conflict.handle(e, contact.id, patch)) return;
      setSaveError(errorMessage(e, 'Could not save changes. Please try again.', SAVE_ERRORS));
      doShake();
    } finally {
      setSaving(false);
    }
  };

  const handleSave = async () => {
    if (!contact || saving || saved || !dirty) return;
    setTouched(true);
    if (firstInvalid) {
      setSaveError(
        firstInvalid === 'name' ? 'A contact needs a name.'
          : firstInvalid === 'email' ? 'Please enter a valid email address before saving.'
          : firstInvalid === 'phone' ? 'That phone number looks too short.'
          : 'LinkedIn should be a linkedin.com/in/… link.',
      );
      doShake();
      refs.current[firstInvalid]?.focus();
      return;
    }

    // If the details now match another contact, ask the user to check before saving.
    const identityChanged = (['name', 'company', 'email', 'phone'] as FieldKey[]).some((k) => initial && !same(k, values[k], initial[k]));
    if (identityChanged) {
      setSaving(true);
      try {
        const res = await api.checkDuplicates({
          firstName: v.name.split(/\s+/)[0] || '', lastName: v.name.split(/\s+/).slice(1).join(' ') || null,
          company: v.company || null, email: v.email || null, phone: v.phone || null, excludeId: contact.id,
        } as any);
        if (res?.matches?.length) {
          setDupMatches(res.matches);
          setSaving(false);
          return;
        }
      } catch {
        /* the check is advisory — never block a save on it */
      }
      setSaving(false);
    }
    await save();
  };

  const handleDelete = async () => {
    if (!contact || deleting) return;
    setDeleting(true);
    setSaveError('');
    try {
      const nm = [contact.firstName, contact.lastName].filter(Boolean).join(' ') || 'Contact';
      await contactsFeature.deleteContact(contact.id, nm);
      allowLeave.current = true;
      // Leave both the edit and the (now deleted) detail screen; the list offers Undo.
      router.dismissTo('/(app)/contacts' as any);
    } catch (e: any) {
      setSaveError(errorMessage(e, "Couldn't delete this contact. Please try again.", SAVE_ERRORS));
    } finally {
      setDeleting(false);
    }
  };

  if (loadState !== 'ready' || !contact || !initial) {
    return (
      <SafeAreaView style={[styles.safeArea, styles.centered]}>
        {loadState === 'loading' ? (
          <>
            <ActivityIndicator color={theme.colors.brand[600]} />
            <Text style={styles.muted}>Loading contact…</Text>
          </>
        ) : (
          <Animated.View entering={FadeIn} style={styles.centered}>
            <View style={styles.stateIcon}>
              {loadState === 'error' ? <WifiOff size={28} color={theme.colors.text.muted} /> : <UserX size={28} color={theme.colors.text.muted} />}
            </View>
            <Text style={styles.missingTitle}>{loadState === 'error' ? "Couldn't load this contact" : 'Contact not found'}</Text>
            <Text style={styles.muted}>
              {loadState === 'error' ? "The server can't be reached. Check your connection." : 'It may have been deleted.'}
            </Text>
            {loadState === 'error' && (
              <PressableScale style={styles.primaryPill} onPress={() => setReloadKey((k) => k + 1)} leadingIcon={<RotateCw size={15} color="white" />}>
                <Text style={styles.primaryPillText}>Try again</Text>
              </PressableScale>
            )}
            <Pressable onPress={() => router.back()} hitSlop={8}><Text style={styles.link}>Go back</Text></Pressable>
          </Animated.View>
        )}
      </SafeAreaView>
    );
  }

  const muted = theme.colors.text.muted;
  const brand = theme.colors.brand[600];
  const nameParts = v.name.split(/\s+/).filter(Boolean);
  const subtitle = [v.title, v.company].filter(Boolean).join(' · ');
  const hint = (k: FieldKey) => (!same(k, values[k], initial[k]) ? <ChangeHint was={initial[k]} onUndo={() => undo(k)} /> : null);
  const words = v.notes ? v.notes.split(/\s+/).length : 0;

  return (
    <SafeAreaView style={styles.safeArea}>
      <Animated.View entering={FadeInDown.duration(260)} style={styles.formHeader}>
        <PressableScale
          onPress={() => router.back()}
          style={styles.cancelBtn}
          scaleTo={0.94}
          leadingIcon={<X size={16} color={theme.colors.text.body} />}
          accessibilityRole="button"
          accessibilityLabel={dirty ? 'Cancel and discard changes' : 'Cancel'}
        >
          <Text style={styles.cancelText}>Cancel</Text>
        </PressableScale>
        <View style={styles.headerCenter}>
          <Text style={styles.formTitle}>Edit contact</Text>
          {dirty ? (
            <Animated.Text key={changed.length} entering={FadeIn.duration(160)} style={styles.headerSub}>
              {changed.length} unsaved change{changed.length === 1 ? '' : 's'}
            </Animated.Text>
          ) : (
            <Text style={styles.headerSubMuted}>No changes</Text>
          )}
        </View>
        <Animated.View style={savePopStyle}>
          <PressableScale
            onPress={handleSave}
            style={[styles.saveBtn, (saving || !dirty) && styles.saveBtnDisabled]}
            disabled={saving || !dirty || saved}
            scaleTo={0.94}
            leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" strokeWidth={2.5} />}
            accessibilityRole="button"
          >
            <Text style={styles.saveBtnText}>{saving ? 'Saving…' : 'Save'}</Text>
          </PressableScale>
        </Animated.View>
      </Animated.View>

      <ScrollView
        {...scrollProps}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: 60 + keyboardHeight }]}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="interactive"
        showsVerticalScrollIndicator={false}
      >
        {/* Live preview — updates as you type */}
        <Animated.View entering={FadeInDown.duration(260)} layout={LinearTransition.duration(260)} style={[styles.previewCard, dirty && styles.previewCardDirty]}>
          <View style={styles.previewGlowA} />
          <View style={styles.previewGlowB} />
          <View style={styles.previewTop}>
            <Animated.View entering={ZoomIn.delay(80).duration(260)}>
              <ContactAvatar
                firstName={nameParts[0] || null}
                lastName={nameParts.slice(1).join(' ') || null}
                photoUrl={contact.photoUrl}
                size={80}
                editable
                busy={photo.busy}
                onPress={photo.open}
              />
            </Animated.View>
            <View style={{ flex: 1 }}>
              <Text style={[styles.previewName, !v.name && { color: muted }]} numberOfLines={2}>{v.name || 'Name required'}</Text>
              <Text style={[styles.previewSub, !subtitle && { color: muted }]} numberOfLines={1}>{subtitle || 'Title · Company'}</Text>
              <Pressable onPress={photo.open} disabled={photo.busy} hitSlop={6} style={styles.photoLinkRow}>
                <Camera size={13} color={brand} />
                <Text style={styles.photoLink}>{contact.photoUrl ? 'Change photo' : 'Add photo'}</Text>
                <Text style={styles.photoHint}>· saves right away</Text>
              </Pressable>
            </View>
          </View>
          {dirty ? (
            <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(150)} style={styles.dirtyRow}>
              <PulseDot />
              <Text style={styles.dirtyText} numberOfLines={2}>Changed: {changed.join(', ')}</Text>
              <PressableScale onPress={revertAll} scaleTo={0.92} style={styles.revertBtn} leadingIcon={<Undo2 size={13} color={theme.colors.brand[700]} />}>
                <Text style={styles.revertText}>Revert all</Text>
              </PressableScale>
            </Animated.View>
          ) : null}
        </Animated.View>
        {photo.error ? <Animated.Text entering={FadeIn} style={styles.inlineError}>{photo.error}</Animated.Text> : null}

        <Animated.View style={shakeStyle}>
          {/* Who they are */}
          <Animated.View entering={enter(1)} layout={LinearTransition.duration(260)} style={styles.sectionCard}>
            <SectionHeader icon={<IdCard size={15} color={brand} />} label="WHO THEY ARE" hint="Only the name is required" />
            <EditableField
              inputRef={(r) => { refs.current.name = r; }}
              icon={<User size={16} color={nameMissing ? theme.colors.semantic.danger : muted} />}
              label="Full name" value={values.name} onChangeText={set('name')} placeholder="e.g. Maya Patel"
              autoCapitalize="words" returnKeyType="next" submitBehavior="submit"
              onSubmitEditing={() => refs.current.title?.focus()} onFocus={onInputFocus}
              error={nameMissing && (touched || initial.name) ? 'A name is required' : undefined}
            />
            {hint('name')}
            <EditableField
              inputRef={(r) => { refs.current.title = r; }}
              icon={<Briefcase size={16} color={muted} />}
              label="Role / title" value={values.title} onChangeText={set('title')} placeholder="e.g. Head of Data"
              autoCapitalize="words" returnKeyType="next" submitBehavior="submit"
              onSubmitEditing={() => refs.current.company?.focus()} onFocus={onInputFocus}
            />
            {hint('title')}
            <EditableField
              inputRef={(r) => { refs.current.company = r; }}
              icon={<Building2 size={16} color={muted} />}
              label="Company" value={values.company} onChangeText={set('company')} placeholder="e.g. Northwind Health"
              autoCapitalize="words" returnKeyType="next" submitBehavior="submit"
              onSubmitEditing={() => refs.current.email?.focus()} onFocus={onInputFocus}
            />
            {hint('company')}
          </Animated.View>

          {/* How to reach them */}
          <Animated.View entering={enter(2)} layout={LinearTransition.duration(260)} style={styles.sectionCard}>
            <SectionHeader icon={<MessageSquare size={15} color={brand} />} label="HOW TO REACH THEM" />
            <EditableField
              inputRef={(r) => { refs.current.email = r; }}
              icon={<Mail size={16} color={emailInvalid ? theme.colors.semantic.danger : muted} />}
              label="Email" value={values.email} onChangeText={set('email')} placeholder="name@company.com"
              keyboardType="email-address" autoCapitalize="none" autoCorrect={false} returnKeyType="next" submitBehavior="submit"
              onSubmitEditing={() => refs.current.phone?.focus()} onFocus={onInputFocus}
              error={emailInvalid ? "That doesn't look like an email address" : undefined}
            />
            {hint('email')}
            <EditableField
              inputRef={(r) => { refs.current.phone = r; }}
              icon={<Smartphone size={16} color={phoneInvalid ? theme.colors.semantic.danger : muted} />}
              label="Phone" value={values.phone} onChangeText={set('phone')} placeholder="+1 415 555 0142"
              keyboardType="phone-pad" onFocus={onInputFocus}
              error={phoneInvalid && touched ? 'That phone number looks too short' : undefined}
            />
            {hint('phone')}
            <EditableField
              inputRef={(r) => { refs.current.linkedin = r; }}
              icon={<Linkedin size={16} color={linkedinInvalid && touched ? theme.colors.semantic.danger : muted} />}
              label="LinkedIn" value={values.linkedin} onChangeText={set('linkedin')} placeholder="linkedin.com/in/maya-patel"
              keyboardType="url" autoCapitalize="none" autoCorrect={false} onFocus={onInputFocus}
              error={linkedinInvalid && touched ? 'Use a linkedin.com/in/… link' : undefined}
            />
            {hint('linkedin')}
          </Animated.View>

          {/* Met at */}
          {eventOptions.length > 0 ? (
            <Animated.View entering={enter(3)} layout={LinearTransition.duration(260)} style={styles.sectionCard}>
              <SectionHeader icon={<CalendarDays size={15} color={brand} />} label="MET AT" hint="Shows under This event in your contacts" />
              <View style={styles.eventChips}>
                {[...eventOptions, ''].map((name) => {
                  const on = event === name;
                  return (
                    <PressableScale
                      key={name || '__none'}
                      onPress={() => setEvent(name)}
                      scaleTo={0.94}
                      style={[styles.eventChip, on && styles.eventChipOn]}
                      accessibilityRole="radio"
                      accessibilityState={{ selected: on }}
                      leadingIcon={on
                        ? <Animated.View entering={ZoomIn.duration(260)}><Check size={13} color="white" strokeWidth={3} /></Animated.View>
                        : name ? <CalendarDays size={13} color={brand} /> : <CircleSlash size={13} color={muted} />}
                    >
                      <Text style={[styles.eventChipText, on && { color: 'white' }]} numberOfLines={1}>{name || 'No event'}</Text>
                    </PressableScale>
                  );
                })}
              </View>
              {event !== initialEvent ? (
                <ChangeHint was={initialEvent || ''} onUndo={() => setEvent(initialEvent)} />
              ) : <View style={{ height: 10 }} />}
            </Animated.View>
          ) : null}

          {/* Notes */}
          <Animated.View entering={enter(4)} layout={LinearTransition.duration(260)} style={styles.sectionCard}>
            <SectionHeader
              icon={<StickyNote size={15} color={brand} />}
              label="NOTES"
              hint="What you talked about, follow-ups"
              right={<Text style={styles.counter}>{words} word{words === 1 ? '' : 's'}</Text>}
            />
            <EditableField
              inputRef={(r) => { refs.current.notes = r; }}
              icon={<StickyNote size={16} color={muted} />}
              label="Notes" value={values.notes} onChangeText={set('notes')}
              placeholder="Met at the booth — wants a demo next week"
              multiline maxLength={NOTES_MAX} onFocus={onInputFocus}
            />
            {!same('notes', values.notes, initial.notes) ? (
              <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={styles.changeRow}>
                <View style={styles.changeDot} />
                <Text style={styles.changeText}>Notes edited</Text>
                <Pressable onPress={() => undo('notes')} hitSlop={8} style={styles.undoBtn}>
                  <Undo2 size={12} color={theme.colors.brand[700]} />
                  <Text style={styles.undoText}>Undo</Text>
                </Pressable>
              </Animated.View>
            ) : null}
          </Animated.View>
        </Animated.View>

        {saveError ? (
          <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut} style={styles.errorBanner}>
            <AlertCircle size={16} color={theme.colors.semantic.danger} />
            <Text style={styles.errorBannerText}>{saveError}</Text>
          </Animated.View>
        ) : null}

        <Animated.View entering={enter(5)}>
          <PressableScale
            style={[styles.primaryBtn, (!dirty || saving) && { opacity: 0.55 }]}
            onPress={handleSave}
            disabled={!dirty || saving || saved}
            leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : <Check size={18} color="white" strokeWidth={2.5} />}
            accessibilityRole="button"
          >
            <Text style={styles.primaryBtnText}>{saving ? 'Saving…' : dirty ? `Save ${changed.length} change${changed.length === 1 ? '' : 's'}` : 'No changes to save'}</Text>
          </PressableScale>
          <PressableScale
            style={styles.deleteBtn}
            onPress={() => setConfirmDelete(true)}
            disabled={deleting}
            scaleTo={0.97}
            leadingIcon={deleting ? <ActivityIndicator size="small" color={theme.colors.semantic.danger} /> : <Trash2 size={16} color={theme.colors.semantic.danger} />}
            accessibilityRole="button"
          >
            <Text style={styles.deleteText}>{deleting ? 'Deleting…' : 'Delete contact'}</Text>
          </PressableScale>
        </Animated.View>
      </ScrollView>

      {/* Success moment before returning to the contact */}
      {saved ? (
        <Animated.View entering={FadeIn.duration(150)} style={styles.successOverlay} pointerEvents="auto">
          <Animated.View entering={ZoomIn.duration(260)} style={styles.successCircle}>
            <Check size={40} color="white" strokeWidth={3} />
          </Animated.View>
          <Animated.Text entering={FadeInDown.delay(120).duration(260)} style={styles.successText}>Changes saved</Animated.Text>
        </Animated.View>
      ) : null}

      {photo.sheet}

      {conflict.sheet}

      <OptionSheet
        visible={confirmDiscard}
        title="Discard changes?"
        message={`You changed ${changed.join(', ').toLowerCase() || 'this contact'} — those edits won't be saved.`}
        options={[
          { label: 'Discard changes', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: leaveNow },
          { label: 'Keep editing', icon: <Check size={20} color={brand} />, onPress: () => { pendingAction.current = null; } },
        ]}
        onClose={() => setConfirmDiscard(false)}
      />
      <OptionSheet
        visible={dupMatches.length > 0}
        title="This may be a duplicate"
        message={dupMatches.map((m) =>
          `${[m.contact.firstName, m.contact.lastName].filter(Boolean).join(' ') || 'Another contact'} (${describeReasons(m.reasons)})`,
        ).join('\n') + '\n\nCheck it before saving — you may already have this person.'}
        options={[
          ...dupMatches.slice(0, 2).map((m) => ({
            label: `View ${m.contact.firstName || 'existing contact'}`,
            icon: <Copy size={20} color={brand} />,
            onPress: () => router.push(`/contacts/${m.contact.id}`),
          })),
          { label: 'Save anyway', icon: <Check size={20} color={brand} />, onPress: () => { setDupMatches([]); save(); } },
        ]}
        onClose={() => setDupMatches([])}
      />
      <OptionSheet
        visible={confirmDelete}
        title="Delete this contact?"
        message="It moves to Recently deleted, where you can restore it for 30 days."
        options={[{ label: 'Delete contact', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: handleDelete }]}
        onClose={() => setConfirmDelete(false)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 10, padding: 24 },
  muted: { color: theme.colors.text.muted, textAlign: 'center' },
  link: { fontSize: 14, fontWeight: '700', color: theme.colors.brand[600] },
  stateIcon: { width: 64, height: 64, borderRadius: 22, backgroundColor: theme.colors.bg.surfaceHover, alignItems: 'center', justifyContent: 'center' },
  missingTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  primaryPill: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.brand[600], paddingHorizontal: 18, paddingVertical: 10, borderRadius: 999 },
  primaryPillText: { color: 'white', fontWeight: '700' },

  formHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8, paddingHorizontal: 16, paddingVertical: 12 },
  headerCenter: { flex: 1, alignItems: 'center' },
  cancelBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff'), paddingHorizontal: 12, paddingVertical: 7, borderRadius: 20 },
  cancelText: { fontSize: 14, color: theme.colors.text.body, fontWeight: '700' },
  formTitle: { fontSize: 18, lineHeight: 23, fontWeight: '800', color: theme.colors.text.display },
  headerSub: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[600] },
  headerSubMuted: { fontSize: 11, color: theme.colors.text.muted },
  saveBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: theme.colors.fill.brand700, paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20 },
  saveBtnDisabled: { opacity: 0.5 },
  saveBtnText: { fontSize: 14, fontWeight: '700', color: 'white' },
  scrollContent: { paddingHorizontal: 16, paddingTop: 4 },

  previewCard: {
    backgroundColor: theme.colors.bg.surface, borderRadius: 24, padding: 18, marginBottom: 14,
    borderWidth: 1, borderColor: theme.colors.brand[100], overflow: 'hidden', ...theme.shadows.active,
  },
  previewCardDirty: { borderColor: theme.colors.brand[300] },
  previewGlowA: { position: 'absolute', top: -70, right: -50, width: 190, height: 190, borderRadius: 95, backgroundColor: theme.colors.brand[50] },
  previewGlowB: { position: 'absolute', bottom: -60, left: -40, width: 140, height: 140, borderRadius: 70, backgroundColor: toneBg('#f0fdfa') },
  previewTop: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  previewName: { fontSize: 22, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.3, lineHeight: 28 },
  previewSub: { fontSize: 14, color: theme.colors.text.body, marginTop: 2 },
  photoLinkRow: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 8, alignSelf: 'flex-start' },
  photoLink: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  photoHint: { fontSize: 12, color: theme.colors.text.muted },
  dirtyRow: {
    flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 14, padding: 10, borderRadius: 14,
    backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100],
  },
  pulseDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: theme.colors.brand[600] },
  dirtyText: { flex: 1, fontSize: 12, fontWeight: '600', color: theme.colors.brand[800] },
  revertBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999, backgroundColor: toneBg('#ffffff') },
  revertText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  inlineError: { fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', marginTop: -6, marginBottom: 12 },

  sectionCard: {
    backgroundColor: theme.colors.bg.surface, borderRadius: 20, paddingHorizontal: 14, paddingTop: 12, paddingBottom: 4,
    marginBottom: 14, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card,
  },
  changeRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: -6, marginBottom: 12, paddingHorizontal: 2 },
  changeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: theme.colors.brand[500] },
  changeText: { flex: 1, fontSize: 12, color: theme.colors.text.muted },
  changeWas: { fontSize: 12, color: theme.colors.text.body, textDecorationLine: 'line-through' },
  undoBtn: { flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  undoText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  counter: { fontSize: 12, color: theme.colors.text.muted },

  eventChips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4, marginBottom: 10 },
  eventChip: {
    flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 999,
    backgroundColor: theme.colors.bg.surfaceHover, borderWidth: 1.5, borderColor: theme.colors.bg.divider, maxWidth: 260,
  },
  eventChipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  eventChipText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.display, flexShrink: 1 },

  errorBanner: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 14 },
  errorBannerText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, backgroundColor: theme.colors.brand[600], borderRadius: 999, paddingVertical: 16, ...theme.shadows.active },
  primaryBtnText: { color: 'white', fontSize: 16, fontWeight: '800' },
  deleteBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, borderRadius: 999, paddingVertical: 14, marginTop: 12, backgroundColor: toneBg('#ffffff') },
  deleteText: { fontSize: 15, fontWeight: '700', color: theme.colors.semantic.danger },

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
