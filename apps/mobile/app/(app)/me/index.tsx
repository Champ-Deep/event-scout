import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, ScrollView, Pressable, TextInput, RefreshControl, ActivityIndicator, Image, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import Animated, { FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition, useSharedValue, useAnimatedStyle, withRepeat, withTiming, Easing } from 'react-native-reanimated';
import {
  Users, Flame, Sparkles, CalendarPlus, CalendarDays, Pencil, Target, Briefcase, Building2, Globe, Linkedin,
  RefreshCw, CloudOff, CheckCircle2, Download, Trash2, LogOut, ChevronRight, Plus, X, Check, AlertCircle, Zap,
  Camera, ImageIcon, QrCode, Phone, Sun, Moon, Smartphone, Palette, FileSpreadsheet, FileJson, Share2, FolderDown, ScanLine, ArrowLeftRight, Package, MessageSquareQuote, MapPin, Trophy, Flag, Lightbulb, ShieldQuestion, UserRound, FolderOpen,
} from 'lucide-react-native';
import { QrDialog, type QrDialogData } from '../../../src/features/contacts/components/QrDialog';
import { pickContactPhoto, type PhotoSource } from '../../../src/features/contacts/photo';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { Sheet } from '../../../src/ui/Sheet';
import { openContactsPage, useOpenContactsList } from '../../../src/features/contacts/open';
import { useSession } from '../../../src/features/auth';
import { api, errorMessage, type EventFile, type ErrorMessageOptions } from '../../../src/lib/api';
import { contactsFeature } from '../../../src/features/contacts';
import { eventRules, profileStore, type Profile, type ProfilePatch } from '../../../src/features/profile';
import { FileList } from '../../../src/features/files/FileList';
import { getSyncState, runSync, countPendingChanges, type SyncState } from '../../../src/sync/state';
import { LinkedInConnectionCard } from '../../../src/features/me/components/LinkedInConnectionCard';
import { exportContacts, saveContactsToDownloads, resetDownloadsFolder, downloadsFolderLabel, type ExportFormat } from '../../../src/features/me/export';
import { AnimatedCount } from '../../../src/features/contacts/components/ListAnimations';
import { OptionSheet } from '../../../src/features/contacts/components/OptionSheet';
import { ChoiceChips, ListEditor, PairListEditor, type Pair } from '../../../src/features/me/components/ProfileEditors';
import { eventDetailFor, withEventDetail } from '../../../src/features/me/eventGoals';
import { useEventCalendar, EventCalendarCard, EventNoteSheet, EventNotesListSheet } from '../../../src/features/me/components/EventCalendar';
import { AllEventsSheet } from '../../../src/features/me/components/AllEventsSheet';
import { coversDay, todayYmd } from '../../../src/features/events/dates';
import {
  PressableScale, EditableField, SectionHeader, AnimatedToggle, TagEditor, SegmentedSelect,
} from '../../../src/features/onboarding/components';
import { useAppearance } from '../../../src/ui/theme';
import { type AppearancePref } from '../../../src/ui/appearance';
import { switchAppearance, ThemeToggleButton, type SwitchResult } from '../../../src/ui/ThemeSwitch';

const enter = (i: number) => FadeInDown.delay(60 + i * 70).duration(260);

// Same values the backend prompts understand (app.py PITCH_STYLES) and onboarding offers.
const PITCH_STYLES = [
  { value: 'consultative', label: 'Consultative', hint: 'Ask questions first, act as a partner' },
  { value: 'direct', label: 'Direct', hint: 'Lead with the solution and ROI' },
  { value: 'challenger', label: 'Challenger', hint: 'Reframe their thinking with insights' },
  { value: 'relationship', label: 'Relationship', hint: 'Build trust first, long-term play' },
];
const COMPANY_SIZES = ['Startup', 'Mid-market', 'Enterprise'];

interface Product { name: string; description?: string; ideal_customer?: string }

// How this screen words a failed request (see errorMessage).
const ME_ERRORS: ErrorMessageOptions = { offline: "You're offline. Try again when connected.", unreachable: "Can't reach the server. Check your connection." };

type Toast = { kind: 'ok' | 'error'; text: string } | null;

/* ---------------------------------- bits ---------------------------------- */

function AvatarRing({ initials, photo, onPress, busy }: { initials: string; photo?: string | null; onPress?: () => void; busy?: boolean }) {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.inOut(Easing.quad) }), -1, true);
  }, []);
  const ring = useAnimatedStyle(() => ({ opacity: 0.25 + t.value * 0.35, transform: [{ scale: 1 + t.value * 0.1 }] }));
  return (
    <View style={styles.avatarWrap}>
      <Animated.View style={[styles.avatarRing, ring]} />
      <Animated.View entering={ZoomIn.duration(260)}>
        <Pressable onPress={onPress} disabled={!onPress} style={styles.avatar} accessibilityLabel={photo ? 'Change your photo' : 'Add your photo'}>
          {photo ? <Image source={{ uri: photo }} style={styles.avatarImg} /> : <Text style={styles.avatarText}>{initials}</Text>}
          {busy ? <View style={styles.avatarBusy}><ActivityIndicator size="small" color="white" /></View> : null}
        </Pressable>
      </Animated.View>
      {onPress ? (
        <View pointerEvents="none" style={styles.avatarBadge}><Camera size={11} color="white" /></View>
      ) : null}
    </View>
  );
}

function StatTile({ icon, value, label, tint, index, onPress }: {
  icon: React.ReactNode; value: number | null; label: string; tint: string; index: number; onPress?: () => void;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(120 + index * 60).duration(260)} style={{ flex: 1 }}>
      <PressableScale onPress={onPress} disabled={!onPress} scaleTo={0.95} style={styles.statTile} accessibilityLabel={`${value ?? 0} ${label}`}>
        <View style={[styles.statIcon, { backgroundColor: tint }]}>{icon}</View>
        {value == null ? <Text style={styles.statValue}>—</Text> : <AnimatedCount value={value} style={styles.statValue} />}
        <Text style={styles.statLabel} numberOfLines={1}>{label}</Text>
      </PressableScale>
    </Animated.View>
  );
}

/** Soft pulsing "live" dot for the current event. */
function LiveDot() {
  const t = useSharedValue(0);
  useEffect(() => {
    t.value = withRepeat(withTiming(1, { duration: 1400, easing: Easing.out(Easing.quad) }), -1, false);
  }, []);
  const ring = useAnimatedStyle(() => ({ opacity: 0.55 * (1 - t.value), transform: [{ scale: 1 + t.value * 1.6 }] }));
  return (
    <View style={styles.liveWrap} accessibilityLabel="Live">
      <Animated.View style={[styles.liveRing, ring]} />
      <View style={styles.liveDot} />
    </View>
  );
}

function Spin({ on, children }: { on: boolean; children: React.ReactNode }) {
  const r = useSharedValue(0);
  useEffect(() => {
    r.value = on ? withRepeat(withTiming(360, { duration: 900, easing: Easing.linear }), -1) : withTiming(0, { duration: 200 });
  }, [on]);
  const style = useAnimatedStyle(() => ({ transform: [{ rotate: `${r.value}deg` }] }));
  return <Animated.View style={style}>{children}</Animated.View>;
}

/* --------------------------------- screen --------------------------------- */

export default function MeScreen() {
  const router = useRouter();
  const openList = useOpenContactsList(); // Contacts list on a view, closing any contact left open there
  // Set by /callback when a web-redirect LinkedIn connect started here failed.
  // `open` (event | targeting | qr) + `t` come from the Scout shortcuts; `open=calendar` (+ `day`)
  // from a reminder's Open button.
  const { linkError, open, t: openT, day: openDay } = useLocalSearchParams<{ linkError?: string; open?: string; t?: string; day?: string }>();
  const scrollRef = useRef<ScrollView>(null);
  const calendarY = useRef(0);
  const { logout, userName, userEmail, userId } = useSession();

  const [profile, setProfile] = useState<Profile | null>(null);
  const [dashboard, setDashboard] = useState<any>(null);
  const [deletedCount, setDeletedCount] = useState<number | null>(null);
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [toast, setToast] = useState<Toast>(null);

  const [syncing, setSyncing] = useState(false);
  // 'csv:save' | 'csv:share' | 'json:save' | 'json:share' while one is running
  const [exporting, setExporting] = useState<string | null>(null);
  // Appearance (System / Light / Dark): saved, then the app reloads so every screen redraws
  // One shared state for the header button and this card — they can't disagree.
  const { pref: appearance, scheme: currentScheme } = useAppearance();
  const [appearanceNote, setAppearanceNote] = useState<string | null>(null);
  // Live switch: no message needed when it changes; System just says what it follows.
  const noteFor = (r: SwitchResult, pref: AppearancePref) =>
    r === 'unchanged' && pref === 'system' ? 'Following your phone’s setting' : null;
  // Appearance card and the header toggle both go through switchAppearance (fade → reload).
  const chooseAppearance = async (label: string) => {
    const pref = label.toLowerCase() as AppearancePref;
    if (pref === appearance) return;
    setAppearanceNote(noteFor(await switchAppearance(pref), pref));
  };
  const [exportFormat, setExportFormat] = useState<ExportFormat>('csv');
  const [saveFolder, setSaveFolder] = useState<string | null>(null); // e.g. "Downloads/EventScout"
  useEffect(() => { downloadsFolderLabel().then(setSaveFolder).catch(() => {}); }, []);
  const [savingAuto, setSavingAuto] = useState(false);
  const [eventBusy, setEventBusy] = useState<string | null>(null);
  const [newEvent, setNewEvent] = useState('');
  const [addingEvent, setAddingEvent] = useState(false);
  const [eventPicker, setEventPicker] = useState(false);
  // Events in the imported exhibitor directory (GET /api/events) — suggestions when adding one.
  const [directoryEvents, setDirectoryEvents] = useState<string[]>([]);

  const [profileSheet, setProfileSheet] = useState(false);
  const [form, setForm] = useState({ full_name: '', job_title: '', company_name: '', company_website: '', linkedin_url: '', phone: '' });
  const [targetSheet, setTargetSheet] = useState(false);
  const [industries, setIndustries] = useState<string[]>([]);
  const [roles, setRoles] = useState<string[]>([]);
  const [sizes, setSizes] = useState<string[]>([]);
  const [geos, setGeos] = useState<string[]>([]);
  // Products & services (profile.products → AI chat, scoring, research, pitches)
  const [productSheet, setProductSheet] = useState<{ index: number | null } | null>(null);
  const [productForm, setProductForm] = useState({ name: '', description: '', ideal_customer: '' });
  const [confirmDeleteProduct, setConfirmDeleteProduct] = useState(false);
  // Sales approach
  const [salesSheet, setSalesSheet] = useState(false);
  const [pitchStyle, setPitchStyle] = useState('');
  const [valueProps, setValueProps] = useState<string[]>([]);
  const [objections, setObjections] = useState<Pair[]>([]);
  const [caseStudies, setCaseStudies] = useState<Pair[]>([]);
  // Goals per event (profile.event_details[event]) — each event has its own
  const [goalsSheet, setGoalsSheet] = useState(false);
  const [goalsEvent, setGoalsEvent] = useState('');
  const [eventDesc, setEventDesc] = useState('');
  const [goals, setGoals] = useState<string[]>([]);
  const [sheetSaving, setSheetSaving] = useState(false);
  const [sheetError, setSheetError] = useState('');
  const [confirmSignOut, setConfirmSignOut] = useState(false);
  const [photoSheet, setPhotoSheet] = useState(false);
  // My QR card (GET /api/me/card): vCard QR + public card link
  const [qrOpen, setQrOpen] = useState(false);
  const [qrData, setQrData] = useState<QrDialogData | null>(null);
  const [qrLoading, setQrLoading] = useState(false);
  const [qrError, setQrError] = useState<string | null>(null);
  const [photoBusy, setPhotoBusy] = useState(false);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(t);
  }, [toast]);

  // Team files: documents admins uploaded (Admin console → Files). Demo placeholders are left out.
  const [teamFiles, setTeamFiles] = useState<EventFile[] | null>(null);

  // "Hot leads" = the same rule as the list the tile opens (Contacts → Hot: score ≥ 85, band A),
  // counted from that list itself so the number always matches what you see there.
  const [hotCount, setHotCount] = useState<number | null>(null);
  const countHot = () => contactsFeature.countContacts({ scoreBand: 'A', limit: 500 }, 10000);

  const load = useCallback(async () => {
    if (!userId) { setLoading(false); return; }
    const [p, d, del, s, ev, tf, hot] = await Promise.allSettled([
      profileStore.load(userId),
      api.getDashboard(userId),
      api.listDeletedContacts(),
      getSyncState(),
      api.listEvents(),
      api.listEventFiles(),
      countHot(),
    ]);
    if (hot.status === 'fulfilled') setHotCount(hot.value);
    if (tf.status === 'fulfilled') setTeamFiles((Array.isArray(tf.value) ? tf.value : []).filter((f) => !f.placeholder));
    if (ev.status === 'fulfilled') setDirectoryEvents((Array.isArray(ev.value) ? ev.value : []).map((e: any) => e?.name).filter(Boolean));
    if (p.status === 'fulfilled') setProfile(p.value);
    if (d.status === 'fulfilled') setDashboard(d.value);
    if (del.status === 'fulfilled') setDeletedCount((del.value as any)?.contacts?.length ?? 0);
    if (s.status === 'fulfilled') setSyncState(s.value);
    setLoadError(p.status === 'rejected' && d.status === 'rejected' ? errorMessage((p as any).reason, "Couldn't load your profile.", ME_ERRORS) : '');
    setLoading(false);
  }, [userId]);

  // Refresh whenever the tab is opened (contacts / events change elsewhere).
  useFocusEffect(useCallback(() => { load(); }, [load]));

  // Calendar: scheduled events (GET /api/me/events) + your notes & reminders (GET /api/me/event-notes).
  const cal = useEventCalendar(profile?.events || [], profile?.current_event_name || null, setToast, {
    ended: profile?.ended_events || [],
    onOpenEvent: (name) => router.push(`/events/${encodeURIComponent(name)}` as any),
  });

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([load(), cal.reload()]);
    setRefreshing(false);
  };

  /** PUT /user/profile/ and merge the saved profile back in. */
  const saveProfile = async (patch: ProfilePatch) => {
    if (!userId) throw new Error('Sign in again to save.');
    const saved = await profileStore.save(userId, patch);
    setProfile((prev) => ({ ...(prev || {}), ...saved }));
  };

  /* ---- actions ---- */
  const toggleAutoScore = async (on: boolean) => {
    if (savingAuto) return;
    const before = !!profile?.auto_score_on_add;
    setProfile((prev: any) => ({ ...(prev || {}), auto_score_on_add: on })); // optimistic
    setSavingAuto(true);
    try {
      await saveProfile({ auto_score_on_add: on });
      setToast({ kind: 'ok', text: on ? 'New scans will be scored automatically' : 'Auto-scoring turned off' });
    } catch (e: any) {
      setProfile((prev: any) => ({ ...(prev || {}), auto_score_on_add: before }));
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't change auto-scoring.", ME_ERRORS) });
    } finally {
      setSavingAuto(false);
    }
  };

  const setActiveEvent = async (name: string) => {
    if (eventBusy || profile?.current_event_name === name) return;
    setEventBusy(name);
    try {
      // Same rules as the event page: it joins your events, and an ended event is reopened.
      await saveProfile(eventRules.activate(profile, name));
      setToast({ kind: 'ok', text: `New scans will be tagged “${name}”` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't switch the event.", ME_ERRORS) });
    } finally {
      setEventBusy(null);
    }
  };

  // "None": no active event, so new scans aren't tagged. Each event's goals stay saved (event_details).
  const clearActiveEvent = async () => {
    if (eventBusy || !profile?.current_event_name) return;
    setEventBusy('__none');
    try {
      await saveProfile(eventRules.clearActive());
      setToast({ kind: 'ok', text: 'No active event · new scans won’t be tagged' });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't clear the active event.", ME_ERRORS) });
    } finally {
      setEventBusy(null);
    }
  };

  // All events sheet + End / Reopen (profile.ended_events; ending the active event clears it).
  const [allEventsOpen, setAllEventsOpen] = useState(false);
  const [endBusy, setEndBusy] = useState<string | null>(null);
  const toggleEnded = async (name: string) => {
    if (endBusy) return;
    const was = eventRules.isEnded(profile, name);
    const wasActive = !was && eventRules.isActive(profile, name);
    setEndBusy(name);
    try {
      await saveProfile(eventRules.toggleEnded(profile, name));
      setToast({ kind: 'ok', text: was ? `Reopened “${name}”` : wasActive ? `Ended “${name}” · new scans aren't tagged now` : `Ended “${name}”` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, was ? "Couldn't reopen the event." : "Couldn't end the event.", ME_ERRORS) });
    } finally {
      setEndBusy(null);
    }
  };

  const removeEvent = async (name: string) => {
    if (eventBusy) return;
    setEventBusy(name);
    try {
      await saveProfile(eventRules.remove(profile, name));
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't remove the event.", ME_ERRORS) });
    } finally {
      setEventBusy(null);
    }
  };

  const addEvent = async (picked?: string, makeCurrent = false) => {
    const name = (picked ?? newEvent).replace(/\s+/g, ' ').trim();
    if (!name || eventBusy) return;
    const list: string[] = profile?.events || [];
    const existing = list.find((e) => e.toLowerCase() === name.toLowerCase());
    if (existing) {
      if (makeCurrent) { setNewEvent(''); await setActiveEvent(existing); return; } // just switch to it
      setToast({ kind: 'error', text: 'That event is already in your list.' });
      return;
    }
    setEventBusy('__add');
    try {
      // The first event becomes the active one, so scans start getting tagged right away.
      const current = makeCurrent || !profile?.current_event_name;
      await saveProfile(eventRules.add(profile, name, { makeCurrent }));
      setNewEvent('');
      setAddingEvent(false);
      setToast({ kind: 'ok', text: current ? `“${name}” is now your current event` : `Added “${name}”` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't add the event.", ME_ERRORS) });
    } finally {
      setEventBusy(null);
    }
  };

  const syncNow = async () => {
    if (syncing) return;
    setSyncing(true);
    try {
      const res = await runSync();
      setSyncState(res);
      setToast(res.error
        ? { kind: 'error', text: `Sync incomplete: ${res.error}` }
        : { kind: 'ok', text: `Synced · uploaded ${res.pushed}, downloaded ${res.pulled}` });
      load();
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't sync right now.", ME_ERRORS) });
    } finally {
      setSyncing(false);
    }
  };

  // Sign-out sheet: the real number of changes still waiting to upload (outbox + queued scans).
  const [pendingOnSignOut, setPendingOnSignOut] = useState(0);
  const askSignOut = async () => {
    let n = 0;
    try { n = await countPendingChanges(); } catch { /* unknown — the sheet says everything is synced only if 0 */ }
    setPendingOnSignOut(n);
    setConfirmSignOut(true);
  };

  const doExport = async (format: ExportFormat, mode: 'save' | 'share') => {
    if (!userId || exporting) return;
    setExporting(`${format}:${mode}`);
    const F = format.toUpperCase();
    try {
      if (mode === 'save') {
        const r = await saveContactsToDownloads(userId, format);
        const n = `${r.count} contact${r.count === 1 ? '' : 's'}`;
        setToast({ kind: 'ok', text: Platform.OS === 'ios' ? `${F} ready — choose “Save to Files”` : `Saved ${n} to ${r.where} as ${F}${r.askedForFolder ? ' · folder remembered' : ''}` });
        if (Platform.OS === 'android') setSaveFolder(r.where);
      } else {
        const rows = await exportContacts(userId, format);
        setToast({ kind: 'ok', text: `Exported ${rows} contact${rows === 1 ? '' : 's'} as ${F}` });
      }
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, `Couldn't export your contacts as ${F}.`, ME_ERRORS) });
    } finally {
      setExporting(null);
    }
  };

  const toQrData = (res: any): QrDialogData => ({
    title: res.card?.name || displayName,
    subtitle: [res.card?.title, res.card?.company].filter(Boolean).join(' · ') || res.card?.email || null,
    firstName: (res.card?.name || displayName).split(' ')[0] || null,
    lastName: (res.card?.name || displayName).split(' ').slice(1).join(' ') || null,
    photoUrl: profile?.avatar || null,
    vcard: res.vcard, qrDataUrl: res.qrDataUrl, link: res.url, linkQrDataUrl: res.linkQrDataUrl,
  });
  const openMyQr = async () => {
    setQrOpen(true);
    setQrLoading(true);
    setQrError(null);
    try {
      setQrData(toQrData(await api.getMyCard()));
    } catch (e: any) {
      setQrError(errorMessage(e, "Couldn't make your QR code.", ME_ERRORS));
    } finally {
      setQrLoading(false);
    }
  };
  const resetMyLink = async () => { setQrData(toQrData(await api.resetMyCardLink())); };

  // Profile photo: picked/cropped on the phone, saved with PUT /user/profile/ { avatar }.
  const choosePhoto = async (source: PhotoSource) => {
    setPhotoBusy(true);
    try {
      const picked = await pickContactPhoto(source);
      if (!picked) return;
      await saveProfile({ avatar: picked });
      setToast({ kind: 'ok', text: 'Photo updated' });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't save your photo.", ME_ERRORS) });
    } finally {
      setPhotoBusy(false);
    }
  };
  const removePhoto = async () => {
    setPhotoBusy(true);
    try {
      await saveProfile({ avatar: '' });
      setProfile((prev: any) => { const next = { ...(prev || {}) }; delete next.avatar; return next; });
      setToast({ kind: 'ok', text: 'Photo removed' });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't remove your photo.", ME_ERRORS) });
    } finally {
      setPhotoBusy(false);
    }
  };

  const openProfileSheet = () => {
    setForm({
      full_name: profile?.full_name || userName || '',
      job_title: profile?.job_title || '',
      company_name: profile?.company_name || '',
      company_website: profile?.company_website || '',
      linkedin_url: profile?.linkedin_url || '',
      phone: profile?.phone || '',
    });
    setSheetError('');
    setProfileSheet(true);
  };

  const saveProfileSheet = async () => {
    if (!form.full_name.trim()) { setSheetError('Add your name.'); return; }
    setSheetSaving(true);
    setSheetError('');
    try {
      await saveProfile(Object.fromEntries(Object.entries(form).map(([k, v]) => [k, v.trim()])));
      setProfileSheet(false);
      setToast({ kind: 'ok', text: 'Profile saved' });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save your profile.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  const openTargetSheet = () => {
    setIndustries([...(profile?.target_industries || [])]);
    setRoles([...(profile?.target_roles || [])]);
    setSizes([...(profile?.target_company_sizes || [])]);
    setGeos([...(profile?.target_geographies || [])]);
    setSheetError('');
    setTargetSheet(true);
  };

  const saveTargetSheet = async () => {
    setSheetSaving(true);
    setSheetError('');
    try {
      await saveProfile({ target_industries: industries, target_roles: roles, target_company_sizes: sizes, target_geographies: geos });
      setTargetSheet(false);
      setToast({ kind: 'ok', text: 'Targeting saved — new scores use it' });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save your targeting.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  /* ---- products & services ---- */
  const products: Product[] = Array.isArray(profile?.products) ? profile.products.filter((p: any) => p?.name) : [];

  const openProductSheet = (index: number | null) => {
    const p = index != null ? products[index] : null;
    setProductForm({ name: p?.name || '', description: p?.description || '', ideal_customer: p?.ideal_customer || '' });
    setSheetError('');
    setProductSheet({ index });
  };

  const saveProduct = async () => {
    const item = { name: productForm.name.trim(), description: productForm.description.trim(), ideal_customer: productForm.ideal_customer.trim() };
    if (!item.name) { setSheetError('Give it a name.'); return; }
    const idx = productSheet?.index ?? null;
    if (products.some((p, i) => i !== idx && p.name.toLowerCase() === item.name.toLowerCase())) { setSheetError('You already have one with that name.'); return; }
    const next = idx == null ? [...products, item] : products.map((p, i) => (i === idx ? item : p));
    setSheetSaving(true);
    setSheetError('');
    try {
      await saveProfile({ products: next });
      setProductSheet(null);
      setToast({ kind: 'ok', text: idx == null ? `Added “${item.name}”` : `Saved “${item.name}”` });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save it.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  const deleteProduct = async () => {
    const idx = productSheet?.index;
    if (idx == null) return;
    const name = products[idx]?.name;
    setSheetSaving(true);
    try {
      await saveProfile({ products: products.filter((_, i) => i !== idx) });
      setProductSheet(null);
      setToast({ kind: 'ok', text: `Removed “${name}”` });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't remove it.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  /* ---- sales approach ---- */
  const openSalesSheet = () => {
    setPitchStyle(profile?.pitch_style || '');
    setValueProps([...(profile?.value_propositions || [])]);
    setObjections((profile?.common_objections || []).map((o: any) => ({ objection: o?.objection || '', response: o?.response || '' })));
    setCaseStudies((profile?.case_studies || []).map((c: any) => ({ title: c?.title || '', result: c?.result || '' })));
    setSheetError('');
    setSalesSheet(true);
  };

  const saveSalesSheet = async () => {
    setSheetSaving(true);
    setSheetError('');
    try {
      await saveProfile({
        pitch_style: pitchStyle,
        value_propositions: valueProps,
        common_objections: objections.filter((o) => o.objection?.trim()).map((o) => ({ objection: o.objection!.trim(), response: (o.response || '').trim() })),
        case_studies: caseStudies.filter((c) => c.title?.trim()).map((c) => ({ title: c.title!.trim(), result: (c.result || '').trim() })),
      });
      setSalesSheet(false);
      setToast({ kind: 'ok', text: 'Sales approach saved — the AI will pitch your way' });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save your sales approach.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  /* ---- event goals ---- */
  const openGoalsSheet = (eventName?: string) => {
    const target = eventName || profile?.current_event_name;
    if (!target) { setEventPicker(true); return; } // goals belong to an event: pick one first
    const d = eventDetailFor(profile, target);
    setGoalsEvent(target);
    setEventDesc(d.description);
    setGoals([...d.goals]);
    setSheetError('');
    setGoalsSheet(true);
  };

  // Deep link from the Scout shortcuts: open a sheet once the profile has loaded (once per tap).
  const opened = useRef<string | null>(null);
  useEffect(() => {
    if (loading || !open) return;
    const key = `${open}|${openT ?? ''}`;
    if (opened.current === key) return;
    opened.current = key;
    if (open === 'event') setEventPicker(true);
    else if (open === 'targeting') openTargetSheet();
    else if (open === 'qr') openMyQr();
    else if (open === 'calendar') {
      if (openDay && /^\d{4}-\d{2}-\d{2}$/.test(openDay)) cal.select(openDay);
      // Wait a beat for the cards to lay out, then bring the calendar into view.
      setTimeout(() => scrollRef.current?.scrollTo({ y: Math.max(0, calendarY.current - 8), animated: true }), 350);
    }
  }, [loading, open, openT]);

  const saveGoalsSheet = async () => {
    setSheetSaving(true);
    setSheetError('');
    try {
      await saveProfile({ event_details: withEventDetail(profile, goalsEvent, { goals, description: eventDesc }) });
      setGoalsSheet(false);
      setToast({ kind: 'ok', text: `Goals saved for “${goalsEvent}”` });
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't save your event goals.", ME_ERRORS));
    } finally {
      setSheetSaving(false);
    }
  };

  /* ---- derived (all from the API / device, nothing made up) ---- */
  const displayName = profile?.full_name || userName || userEmail || 'You';
  const initials = displayName.split(/[\s@.]+/).filter(Boolean).slice(0, 2).map((p: string) => p[0]).join('').toUpperCase() || '?';
  const role = [profile?.job_title, profile?.company_name].filter(Boolean).join(' · ');
  const events: string[] = profile?.events || [];
  const active: string | null = profile?.current_event_name || null;
  // Your events card shows only what's live now: your active event, and events whose admin-set dates
  // include today — unless you ended them. Every other event is under All.
  const endedKeys = new Set(((profile?.ended_events || []) as string[]).map((e) => e.toLowerCase()));
  const datesToday = (name: string) => {
    const p = cal.planned.find((e) => e.name.toLowerCase() === name.toLowerCase());
    return !!p && coversDay(p.startDate, p.endDate, todayYmd());
  };
  const liveEvents = events.filter((n) => !endedKeys.has(n.toLowerCase()) && (n === active || datesToday(n)));
  const otherEventCount = events.length - liveEvents.length;
  const byEvent: Record<string, number> = dashboard?.by_event || {};
  // Real suggestions: events in the exhibitor directory + event names already on your contacts.
  const suggestions = Array.from(new Set([...directoryEvents, ...Object.keys(byEvent)]))
    .filter((n) => n && n.trim() && !events.some((e) => e.toLowerCase() === n.toLowerCase()))
    .slice(0, 8);
  const tags = [
    ...((profile?.target_industries || []) as string[]).map((t) => ({ t, kind: 'industry' as const })),
    ...((profile?.target_roles || []) as string[]).map((t) => ({ t, kind: 'role' as const })),
    ...((profile?.target_company_sizes || []) as string[]).map((t) => ({ t, kind: 'size' as const })),
    ...((profile?.target_geographies || []) as string[]).map((t) => ({ t, kind: 'geo' as const })),
  ];
  const style = PITCH_STYLES.find((s) => s.value === profile?.pitch_style) || null;
  const vps: string[] = profile?.value_propositions || [];
  const objectionCount = (profile?.common_objections || []).length;
  const caseCount = (profile?.case_studies || []).length;
  const activeDetail = eventDetailFor(profile, active);
  const eventGoals: string[] = activeDetail.goals;
  const eventDescription: string = activeDetail.description;
  const pending = syncState?.pendingChanges ?? 0;
  const failed = syncState?.failedChanges ?? 0;
  const lastSync = syncState?.lastSyncAt ? new Date(syncState.lastSyncAt).toLocaleString([], { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' }) : null;

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(260)} style={styles.header}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Me</Text>
          <Text style={styles.headerSub} numberOfLines={1}>{userEmail || 'Signed in'}</Text>
        </View>
        {/* Light ↔ dark in one tap (System / Light / Dark: Appearance card below) */}
        <Animated.View entering={ZoomIn.delay(80).duration(260)}>
          <ThemeToggleButton onResult={(r, target) => setAppearanceNote(noteFor(r, target))} />
        </Animated.View>
        <Animated.View key={failed ? 'f' : pending ? 'p' : 's'} entering={ZoomIn.duration(260)}>
          <PressableScale onPress={() => router.push('/me/sync' as any)} scaleTo={0.92} accessibilityLabel="Open sync details"
            style={[styles.syncPill, failed ? styles.syncPillBad : pending ? styles.syncPillWarn : null]}>
          {failed ? <CloudOff size={13} color={theme.colors.semantic.danger} />
            : pending ? <RefreshCw size={13} color={theme.colors.semantic.warning} />
            : <CheckCircle2 size={13} color={theme.colors.semantic.successText} />}
          <Text style={[styles.syncPillText, failed ? { color: theme.colors.semantic.danger } : pending ? { color: theme.colors.semantic.warning } : null]}>
            {failed ? `${failed} failed` : pending ? `${pending} to sync` : 'Synced'}
          </Text>
          </PressableScale>
        </Animated.View>
      </Animated.View>

      <ScrollView
        ref={scrollRef}
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl progressBackgroundColor={theme.colors.bg.surface} refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.brand[600]} colors={[theme.colors.brand[600]]} />}
      >
        {loading ? (
          <View style={styles.loading}><ActivityIndicator color={theme.colors.brand[600]} /><Text style={styles.muted}>Loading your profile…</Text></View>
        ) : (
          <>
            {loadError ? (
              <Animated.View entering={FadeInDown} style={styles.errorBanner}>
                <AlertCircle size={16} color={theme.colors.semantic.danger} />
                <Text style={styles.errorText}>{loadError} Pull down to retry.</Text>
              </Animated.View>
            ) : null}

            {/* Profile */}
            <Animated.View entering={enter(0)} style={styles.profileCard}>
              <View style={styles.profileGlow} />
              <View style={styles.profileRow}>
                <AvatarRing initials={initials} photo={profile?.avatar || null} busy={photoBusy} onPress={() => setPhotoSheet(true)} />
                <View style={{ flex: 1 }}>
                  <Text style={styles.name} numberOfLines={1}>{displayName}</Text>
                  <Text style={[styles.role, !role && { color: theme.colors.text.muted, fontStyle: 'italic' }]} numberOfLines={2}>
                    {role || 'Add your title and company'}
                  </Text>
                  {active ? (
                    <Animated.View entering={ZoomIn.delay(200)}>
                      <Pressable onPress={() => router.push(`/events/${encodeURIComponent(active)}` as any)} style={styles.activeChip} hitSlop={6}>
                        <CalendarDays size={11} color={theme.colors.brand[700]} />
                        <Text style={styles.activeChipText} numberOfLines={1}>At {active}</Text>
                      </Pressable>
                    </Animated.View>
                  ) : null}
                </View>
                <PressableScale onPress={openProfileSheet} scaleTo={0.9} style={styles.iconBtn} accessibilityLabel="Edit profile">
                  <Pencil size={16} color={theme.colors.brand[700]} />
                </PressableScale>
              </View>
            </Animated.View>

            {/* Current event — scans are tagged with it; the AI plans around it */}
            <Animated.View entering={enter(1)} layout={LinearTransition.duration(260)} style={[styles.currentCard, !active && styles.currentCardEmpty]}>
              {active ? <View style={styles.currentGlow} /> : null}
              <View style={styles.currentTop}>
                <View style={[styles.currentIcon, !active && { backgroundColor: theme.colors.brand[100] }]}>
                  <CalendarDays size={18} color={active ? 'white' : theme.colors.brand[700]} />
                </View>
                <View style={{ flex: 1 }}>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                    <Text style={[styles.currentLabel, !active && { color: theme.colors.text.muted }]}>CURRENT EVENT</Text>
                    {active ? <LiveDot /> : null}
                  </View>
                  {active ? (
                    <Animated.Text key={active} entering={FadeInDown.duration(260)} style={styles.currentName} numberOfLines={2}>{active}</Animated.Text>
                  ) : (
                    <Text style={styles.currentEmptyText}>Where are you scouting? New scans get tagged with it.</Text>
                  )}
                </View>
              </View>
              {active ? (
                <>
                  <View style={styles.currentStats}>
                    <View style={styles.currentStat}>
                      <AnimatedCount value={byEvent[active] ?? 0} style={styles.currentStatNum} />
                      <Text style={styles.currentStatLabel}>contacts here</Text>
                    </View>
                    <View style={styles.currentStatDivider} />
                    <View style={styles.currentStat}>
                      <AnimatedCount value={eventGoals.length} style={styles.currentStatNum} />
                      <Text style={styles.currentStatLabel}>goal{eventGoals.length === 1 ? '' : 's'}</Text>
                    </View>
                    <View style={styles.currentStatDivider} />
                    <View style={styles.currentStat}>
                      <AnimatedCount value={Math.max(0, events.length - 1)} style={styles.currentStatNum} />
                      <Text style={styles.currentStatLabel}>other event{events.length - 1 === 1 ? '' : 's'}</Text>
                    </View>
                  </View>
                  <View style={styles.currentActions}>
                    <PressableScale containerStyle={{ flex: 1 }} onPress={() => router.push(`/events/${encodeURIComponent(active)}` as any)} scaleTo={0.95}
                      style={styles.currentBtnGhost} accessibilityLabel={`Open ${active}`}>
                      <Text style={styles.currentBtnGhostText}>Open</Text>
                    </PressableScale>
                    <PressableScale containerStyle={{ flex: 1 }} onPress={() => setEventPicker(true)} scaleTo={0.95} style={styles.currentBtnGhost}
                      leadingIcon={<ArrowLeftRight size={14} color="white" />} accessibilityLabel="Switch the current event">
                      <Text style={styles.currentBtnGhostText}>Switch</Text>
                    </PressableScale>
                    <PressableScale containerStyle={{ flex: 1.2 }} onPress={() => router.push('/(app)/scan' as any)} scaleTo={0.95} style={styles.currentBtn}
                      leadingIcon={<ScanLine size={14} color={theme.colors.brand[700]} />} accessibilityLabel={`Scan at ${active}`}>
                      <Text style={styles.currentBtnText}>Scan here</Text>
                    </PressableScale>
                  </View>
                </>
              ) : (
                <PressableScale onPress={() => setEventPicker(true)} scaleTo={0.97} style={styles.currentSetBtn}
                  leadingIcon={<Plus size={15} color="white" />} accessibilityLabel="Set your current event">
                  <Text style={styles.currentSetBtnText}>Set your current event</Text>
                </PressableScale>
              )}
            </Animated.View>

            {/* My QR code — let people scan to add you */}
            <Animated.View entering={enter(1)}>
              <PressableScale onPress={openMyQr} scaleTo={0.98} style={styles.qrRow} accessibilityLabel="Show my QR code">
                <View style={styles.qrIcon}><QrCode size={20} color="white" /></View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.listTitle}>My QR code</Text>
                  <Text style={styles.muted}>Let people scan to save your contact — share it as an image or link</Text>
                </View>
                <ChevronRight size={17} color={theme.colors.brand[600]} />
              </PressableScale>
            </Animated.View>

            {/* Stats (GET /dashboard) */}
            <View style={styles.statsRow}>
              <StatTile index={0} icon={<Users size={16} color={theme.colors.brand[700]} />} tint={theme.colors.brand[50]}
                value={dashboard?.total_contacts ?? null} label="Contacts" onPress={() => openList('all')} />
              <StatTile index={1} icon={<Flame size={16} color={toneFg('#dc2626')} />} tint={toneBg('#fef2f2')}
                value={hotCount ?? dashboard?.by_temperature?.hot ?? null} label="Hot leads" onPress={() => openList('hot')} />
              <StatTile index={2} icon={<Sparkles size={16} color={theme.colors.semantic.successText} />} tint={theme.colors.semantic.successBg}
                value={dashboard?.scored_contacts ?? null} label="Scored" onPress={() => openList('scored')} />
              <StatTile index={3} icon={<CalendarPlus size={16} color={toneFg('#0369a1')} />} tint={toneBg('#f0f9ff')}
                value={dashboard?.added_this_week ?? null} label="This week" onPress={() => openList('week')} />
            </View>

            {/* Events */}
            <Animated.View entering={enter(2)} layout={LinearTransition.duration(260)} style={styles.card}>
              <SectionHeader
                icon={<CalendarDays size={15} color={theme.colors.brand[600]} />}
                label="YOUR EVENTS"
                hint={liveEvents.length ? 'Live now · tap a row to open it · All shows every event' : 'Live events show here · All shows every event'}
                right={
                  <View style={{ flexDirection: 'row', gap: 6 }}>
                    <PressableScale onPress={() => setAllEventsOpen(true)} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="See all events"
                      leadingIcon={<CalendarDays size={13} color={theme.colors.brand[700]} />}>
                      <Text style={styles.smallBtnText}>All</Text>
                    </PressableScale>
                    <PressableScale onPress={() => setAddingEvent((x) => !x)} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Add an event"
                      leadingIcon={addingEvent ? <X size={13} color={theme.colors.brand[700]} /> : <Plus size={13} color={theme.colors.brand[700]} />}>
                      <Text style={styles.smallBtnText}>{addingEvent ? 'Close' : 'Add'}</Text>
                    </PressableScale>
                  </View>
                }
              />
              {addingEvent ? (
                <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(120)} style={styles.addEventRow}>
                  <TextInput
                    value={newEvent}
                    onChangeText={setNewEvent}
                    placeholder="Event name, e.g. Web Summit 2026"
                    placeholderTextColor={theme.colors.text.muted}
                    style={styles.addEventInput}
                    autoFocus
                    maxLength={120}
                    returnKeyType="done"
                    onSubmitEditing={() => addEvent()}
                  />
                  <PressableScale onPress={() => addEvent()} disabled={!newEvent.trim() || !!eventBusy} scaleTo={0.9}
                    style={[styles.addEventBtn, (!newEvent.trim() || !!eventBusy) && { opacity: 0.5 }]} accessibilityLabel="Save event">
                    {eventBusy === '__add' ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" />}
                  </PressableScale>
                </Animated.View>
              ) : null}
              {addingEvent && suggestions.length ? (
                <Animated.View entering={FadeInDown.delay(60).duration(260)} style={{ marginBottom: 10 }}>
                  <Text style={[styles.subLabel, { marginTop: 0 }]}>SUGGESTED</Text>
                  <View style={styles.tags}>
                    {suggestions.map((s, i) => (
                      <Animated.View key={s} entering={ZoomIn.delay(i * 35).duration(220)}>
                        <PressableScale onPress={() => addEvent(s)} disabled={!!eventBusy} scaleTo={0.93} style={styles.suggestChip}
                          leadingIcon={<Plus size={12} color={theme.colors.brand[700]} />} accessibilityLabel={`Add ${s}`}>
                          <Text style={styles.suggestText} numberOfLines={1}>{s}</Text>
                        </PressableScale>
                      </Animated.View>
                    ))}
                  </View>
                </Animated.View>
              ) : null}
              {events.length === 0 && !addingEvent ? (
                <Text style={styles.muted}>No events yet. Add the show you're at so scans are grouped by event.</Text>
              ) : null}
              {events.length && !liveEvents.length && !addingEvent ? (
                <Animated.View entering={FadeIn.duration(220)} style={{ marginBottom: 8 }}>
                  <Text style={styles.muted}>No live events right now.</Text>
                </Animated.View>
              ) : null}
              {liveEvents.map((name, i) => {
                const on = name === active;
                return (
                  <Animated.View key={name} entering={FadeInDown.delay(i * 40).duration(260)} exiting={FadeOut.duration(150)} layout={LinearTransition.duration(260)}>
                    <Pressable onPress={() => router.push(`/events/${encodeURIComponent(name)}` as any)}
                      style={({ pressed }) => [styles.eventRow, on && styles.eventRowOn, pressed && { opacity: 0.8 }]}
                      accessibilityLabel={`Open ${name}`}>
                      {/* The circle switches the active event; the rest of the row opens the event page. */}
                      <Pressable onPress={() => setActiveEvent(name)} disabled={!!eventBusy} hitSlop={10}
                        accessibilityRole="radio" accessibilityState={{ selected: on }} accessibilityLabel={`Make ${name} the active event`}
                        style={[styles.radio, on && styles.radioOn]}>
                        {eventBusy === name ? <ActivityIndicator size="small" color={theme.colors.brand[600]} />
                          : on ? <Animated.View entering={ZoomIn.duration(260)} style={styles.radioDot} /> : null}
                      </Pressable>
                      <View style={{ flex: 1 }}>
                        <Text style={[styles.eventName, on && { color: theme.colors.brand[800] }]} numberOfLines={1}>{name}</Text>
                        <Text style={styles.eventMeta}>
                          {byEvent[name] ?? 0} contact{(byEvent[name] ?? 0) === 1 ? '' : 's'}
                          {(() => { const n = eventDetailFor(profile, name).goals.length; return n ? ` · ${n} goal${n === 1 ? '' : 's'}` : ''; })()}
                          {on ? ' · active' : ''}
                          {datesToday(name) ? ' · live now' : ''}
                        </Text>
                      </View>
                      {!on ? (
                        <Pressable onPress={() => removeEvent(name)} hitSlop={8} disabled={!!eventBusy} accessibilityLabel={`Remove ${name}`}>
                          <X size={15} color={theme.colors.text.muted} />
                        </Pressable>
                      ) : null}
                      <ChevronRight size={16} color={theme.colors.text.muted} />
                    </Pressable>
                  </Animated.View>
                );
              })}
              {otherEventCount > 0 ? (
                <Animated.View entering={FadeInDown.delay(liveEvents.length * 40).duration(260)} layout={LinearTransition.duration(260)}>
                  <PressableScale onPress={() => setAllEventsOpen(true)} scaleTo={0.97} style={styles.moreEventsRow}
                    accessibilityLabel={`See all ${events.length} of your events`} trailingIcon={<ChevronRight size={15} color={theme.colors.brand[600]} />}>
                    <CalendarDays size={14} color={theme.colors.brand[600]} />
                    <Text style={styles.moreEventsText}>
                      {liveEvents.length ? `+${otherEventCount} more event${otherEventCount === 1 ? '' : 's'}` : `See your ${otherEventCount} event${otherEventCount === 1 ? '' : 's'}`} in All
                    </Text>
                  </PressableScale>
                </Animated.View>
              ) : null}
              {events.length ? (
                <Animated.View entering={FadeInDown.delay(events.length * 40).duration(260)} layout={LinearTransition.duration(260)}>
                  <Pressable onPress={clearActiveEvent} disabled={!!eventBusy || !active}
                    style={({ pressed }) => [styles.eventRow, !active && styles.eventRowOn, pressed && { opacity: 0.8 }]}
                    accessibilityRole="radio" accessibilityState={{ selected: !active }} accessibilityLabel="No active event">
                    <View style={[styles.radio, !active && styles.radioOn]}>
                      {eventBusy === '__none' ? <ActivityIndicator size="small" color={theme.colors.brand[600]} />
                        : !active ? <Animated.View entering={ZoomIn.duration(260)} style={styles.radioDot} /> : null}
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={[styles.eventName, !active && { color: theme.colors.brand[800] }]} numberOfLines={1}>None</Text>
                      <Text style={styles.eventMeta}>No active event · new scans aren't tagged</Text>
                    </View>
                  </Pressable>
                </Animated.View>
              ) : null}
              {/* What you want from the event (event_goals + description → AI chat, scoring, research) */}
              <View style={styles.goalsBlock}>
                <View style={styles.goalsHead}>
                  <Text style={[styles.subLabel, { marginTop: 0, marginBottom: 0, flex: 1 }]}>GOALS{active ? ` · ${active.toUpperCase()}` : ''}</Text>
                  <PressableScale onPress={() => openGoalsSheet()} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Edit event goals"
                    leadingIcon={eventGoals.length || eventDescription ? <Pencil size={12} color={theme.colors.brand[700]} /> : <Plus size={13} color={theme.colors.brand[700]} />}>
                    <Text style={styles.smallBtnText}>{eventGoals.length || eventDescription ? 'Edit' : 'Add'}</Text>
                  </PressableScale>
                </View>
                {eventDescription ? <Text style={[styles.muted, { marginTop: 6 }]} numberOfLines={2}>{eventDescription}</Text> : null}
                {eventGoals.length ? eventGoals.slice(0, 4).map((g, i) => (
                  <Animated.View key={`${active}-${g}`} entering={FadeInDown.delay(i * 40).duration(260)} layout={LinearTransition.duration(260)} style={styles.goalRow}>
                    <Flag size={12} color={theme.colors.brand[600]} />
                    <Text style={styles.goalText} numberOfLines={2}>{g}</Text>
                  </Animated.View>
                )) : !eventDescription ? (
                  <Pressable onPress={() => openGoalsSheet()}><Text style={[styles.muted, { marginTop: 6 }]}>{active ? `What do you want from ${active}? The AI prioritises leads around it.` : 'Set a current event, then add what you want from it.'}</Text></Pressable>
                ) : null}
                {eventGoals.length > 4 ? <Text style={styles.muted}>+{eventGoals.length - 4} more</Text> : null}
              </View>
            </Animated.View>

            {/* Calendar: upcoming events by date, plus notes & reminders per day */}
            <View onLayout={(e) => { calendarY.current = e.nativeEvent.layout.y; }}>
              <EventCalendarCard cal={cal} entering={enter(2)} onOpenEvent={(name) => router.push(`/events/${encodeURIComponent(name)}` as any)} />
            </View>

            {/* Products & services (profile.products) */}
            <Animated.View entering={enter(3)} layout={LinearTransition.duration(260)} style={styles.card}>
              <SectionHeader
                icon={<Package size={15} color={theme.colors.brand[600]} />}
                label="PRODUCTS & SERVICES"
                hint="What you sell — the AI pitches these and scores leads against them"
                right={products.length < 12 ? (
                  <PressableScale onPress={() => openProductSheet(null)} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Add a product or service"
                    leadingIcon={<Plus size={13} color={theme.colors.brand[700]} />}>
                    <Text style={styles.smallBtnText}>Add</Text>
                  </PressableScale>
                ) : undefined}
              />
              {products.length === 0 ? (
                <Animated.View entering={FadeIn.duration(260)}>
                  <PressableScale onPress={() => openProductSheet(null)} scaleTo={0.98} style={styles.emptyCta}>
                    <View style={styles.emptyIcon}><Package size={18} color={theme.colors.brand[600]} /></View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.listTitle}>Add what you sell</Text>
                      <Text style={styles.muted}>Name, what it does and who it's for — pitches and scores get specific.</Text>
                    </View>
                    <ChevronRight size={16} color={theme.colors.brand[600]} />
                  </PressableScale>
                </Animated.View>
              ) : products.map((p, i) => (
                <Animated.View key={`${p.name}-${i}`} entering={FadeInDown.delay(i * 45).duration(260)} exiting={FadeOut.duration(150)} layout={LinearTransition.duration(260)}>
                  <PressableScale onPress={() => openProductSheet(i)} scaleTo={0.98} style={styles.productRow} accessibilityLabel={`Edit ${p.name}`}>
                    <View style={styles.productIcon}><Text style={styles.productIndex}>{i + 1}</Text></View>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.eventName} numberOfLines={1}>{p.name}</Text>
                      {p.description ? <Text style={styles.eventMeta} numberOfLines={2}>{p.description}</Text> : null}
                      {p.ideal_customer ? (
                        <View style={styles.idealChip}>
                          <Target size={10} color={toneFg('#0f766e')} />
                          <Text style={styles.idealText} numberOfLines={1}>For {p.ideal_customer}</Text>
                        </View>
                      ) : null}
                    </View>
                    <ChevronRight size={16} color={theme.colors.text.muted} />
                  </PressableScale>
                </Animated.View>
              ))}
            </Animated.View>

            {/* Lead scoring */}
            <Animated.View entering={enter(4)} layout={LinearTransition.duration(260)} style={styles.card}>
              <SectionHeader icon={<Target size={15} color={theme.colors.brand[600]} />} label="LEAD SCORING"
                right={
                  <PressableScale onPress={openTargetSheet} scaleTo={0.9} style={styles.smallBtn} leadingIcon={<Pencil size={12} color={theme.colors.brand[700]} />}>
                    <Text style={styles.smallBtnText}>Edit</Text>
                  </PressableScale>
                } />
              <View style={styles.toggleRow}>
                <View style={styles.toggleIcon}><Zap size={16} color={theme.colors.brand[700]} /></View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.toggleTitle}>Auto-score new scans</Text>
                  <Text style={styles.muted}>Score each scanned card with AI against your targeting.</Text>
                </View>
                <AnimatedToggle value={!!profile?.auto_score_on_add} onValueChange={toggleAutoScore} accessibilityLabel="Auto-score new scans" />
              </View>
              <Text style={styles.subLabel}>WHO YOU'RE TARGETING</Text>
              {tags.length ? (
                <View style={styles.tags}>
                  {tags.map(({ t, kind }, i) => (
                    <Animated.View key={`${kind}-${t}`} entering={ZoomIn.delay(i * 30).duration(260)}
                      style={[styles.tag, kind === 'role' && styles.tagRole, kind === 'size' && styles.tagSize, kind === 'geo' && styles.tagGeo]}>
                      {kind === 'role' ? <Briefcase size={11} color={toneFg('#0f766e')} />
                        : kind === 'size' ? <Users size={11} color={toneFg('#b45309')} />
                        : kind === 'geo' ? <MapPin size={11} color={toneFg('#0369a1')} />
                        : <Building2 size={11} color={theme.colors.brand[700]} />}
                      <Text style={[styles.tagText, kind === 'role' && { color: toneFg('#0f766e') }, kind === 'size' && { color: toneFg('#b45309') }, kind === 'geo' && { color: toneFg('#0369a1') }]}>{t}</Text>
                    </Animated.View>
                  ))}
                </View>
              ) : (
                <Pressable onPress={openTargetSheet}><Text style={styles.link}>Add industries, roles, company sizes and regions →</Text></Pressable>
              )}
            </Animated.View>

            {/* Sales approach (pitch_style, value_propositions, common_objections, case_studies) */}
            <Animated.View entering={enter(5)} layout={LinearTransition.duration(260)} style={styles.card}>
              <SectionHeader icon={<MessageSquareQuote size={15} color={theme.colors.brand[600]} />} label="SALES APPROACH"
                hint="How you sell — used for pitch angles and follow-ups"
                right={
                  <PressableScale onPress={openSalesSheet} scaleTo={0.9} style={styles.smallBtn} leadingIcon={<Pencil size={12} color={theme.colors.brand[700]} />}>
                    <Text style={styles.smallBtnText}>Edit</Text>
                  </PressableScale>
                } />
              {!style && !vps.length && !objectionCount && !caseCount ? (
                <Pressable onPress={openSalesSheet}><Text style={styles.link}>Tell the AI how you like to pitch →</Text></Pressable>
              ) : (
                <>
                  {style ? (
                    <Animated.View key={style.value} entering={ZoomIn.duration(260)} style={styles.styleRow}>
                      <View style={styles.stylePill}><Text style={styles.stylePillText}>{style.label}</Text></View>
                      <Text style={[styles.muted, { flex: 1 }]} numberOfLines={1}>{style.hint}</Text>
                    </Animated.View>
                  ) : null}
                  {vps.slice(0, 3).map((v, i) => (
                    <Animated.View key={v} entering={FadeInDown.delay(i * 40).duration(260)} style={styles.goalRow}>
                      <Lightbulb size={12} color={toneFg('#b45309')} />
                      <Text style={styles.goalText} numberOfLines={2}>{v}</Text>
                    </Animated.View>
                  ))}
                  {vps.length > 3 ? <Text style={styles.muted}>+{vps.length - 3} more value propositions</Text> : null}
                  <View style={styles.countRow}>
                    <View style={styles.countPill}>
                      <ShieldQuestion size={12} color={theme.colors.brand[700]} />
                      <AnimatedCount value={objectionCount} style={styles.countPillNum} />
                      <Text style={styles.countPillText}>objection{objectionCount === 1 ? '' : 's'} handled</Text>
                    </View>
                    <View style={styles.countPill}>
                      <Trophy size={12} color={theme.colors.brand[700]} />
                      <AnimatedCount value={caseCount} style={styles.countPillNum} />
                      <Text style={styles.countPillText}>case stud{caseCount === 1 ? 'y' : 'ies'}</Text>
                    </View>
                  </View>
                </>
              )}
            </Animated.View>

            {/* Team files (GET /api/files/list → open / save via GET /api/files/{id}/download) */}
            <Animated.View entering={enter(6)} style={styles.card}>
              <SectionHeader icon={<FolderOpen size={15} color={theme.colors.brand[600]} />} label="TEAM FILES"
                hint={teamFiles == null ? 'Loading…' : teamFiles.length ? `${teamFiles.length} from your admins · tap to open` : 'Sales material from your admins'} />
              {teamFiles == null ? null : teamFiles.length ? (
                <FileList
                  download={(id) => api.downloadEventFile(id)}
                  files={teamFiles.map((f) => ({
                    id: f.id, filename: f.filename, fileType: f.fileType, fileSize: f.fileSize, description: f.description,
                    meta: [f.category && f.category !== 'general' ? f.category[0]!.toUpperCase() + f.category.slice(1) : null, f.eventName].filter(Boolean).join(' · ') || null,
                  }))}
                />
              ) : (
                <Text style={styles.muted}>No files yet. Admins add brochures, price lists, floor plans and decks in the admin console → Files — they show up here to open or save.</Text>
              )}
            </Animated.View>

            {/* LinkedIn (real link / unlink / set password) */}
            <Animated.View entering={enter(6)}>
              <LinkedInConnectionCard accountEmail={userEmail} initialError={linkError} />
            </Animated.View>

            {/* Appearance */}
            <Animated.View entering={enter(7)} style={styles.card}>
              <SectionHeader icon={<Palette size={15} color={theme.colors.brand[600]} />} label="APPEARANCE"
                hint={appearance === 'system' ? `Following your phone · now ${currentScheme}` : `Always ${appearance}`} />
              <SegmentedSelect
                value={appearance === 'system' ? 'System' : appearance === 'light' ? 'Light' : 'Dark'}
                onChange={chooseAppearance}
                options={[
                  { value: 'System', icon: (on) => <Smartphone size={14} color={on ? 'white' : theme.colors.text.body} /> },
                  { value: 'Light', icon: (on) => <Sun size={14} color={on ? 'white' : theme.colors.text.body} /> },
                  { value: 'Dark', icon: (on) => <Moon size={14} color={on ? 'white' : theme.colors.text.body} /> },
                ]}
              />
              {appearanceNote ? <Animated.Text entering={FadeIn.duration(220)} style={[styles.muted, { marginTop: 8 }]}>{appearanceNote}</Animated.Text> : null}
            </Animated.View>

            {/* Sync (device outbox) */}
            <Animated.View entering={enter(7)} style={styles.card}>
              <SectionHeader icon={<RefreshCw size={15} color={theme.colors.brand[600]} />} label="SYNC"
                hint={lastSync ? `Last synced ${lastSync}` : 'Not synced from this device yet'} />
              <View style={styles.syncRow}>
                <View style={styles.syncStat}>
                  <Text style={[styles.syncNum, pending ? { color: theme.colors.semantic.warning } : null]}>{pending}</Text>
                  <Text style={styles.syncLabel}>waiting</Text>
                </View>
                <View style={styles.syncStat}>
                  <Text style={[styles.syncNum, failed ? { color: theme.colors.semantic.danger } : null]}>{failed}</Text>
                  <Text style={styles.syncLabel}>failed</Text>
                </View>
                <View style={{ flex: 1 }} />
                <PressableScale onPress={syncNow} disabled={syncing} scaleTo={0.94} style={styles.primarySmall}
                  leadingIcon={<Spin on={syncing}><RefreshCw size={14} color="white" /></Spin>}>
                  <Text style={styles.primarySmallText}>{syncing ? 'Syncing…' : 'Sync now'}</Text>
                </PressableScale>
              </View>
              <Pressable onPress={() => router.push('/me/sync' as any)} style={styles.rowLink}>
                <Text style={styles.link}>View sync details</Text>
                <ChevronRight size={15} color={theme.colors.brand[600]} />
              </Pressable>
            </Animated.View>

            {/* Data */}
            <Animated.View entering={enter(8)} style={[styles.card, { paddingVertical: 4 }]}>
              {/* Title row (same layout as the other rows) */}
              <View style={styles.listRow}>
                <View style={styles.listIcon}>{exporting ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Download size={17} color={theme.colors.brand[700]} />}</View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.listTitle}>{exporting ? `Preparing ${(exporting.split(':')[0] || '').toUpperCase()}…` : 'Export contacts'}</Text>
                  <Text style={styles.muted} numberOfLines={2}>All {dashboard?.total_contacts ?? ''} contacts — save them to Downloads or share them</Text>
                </View>
              </View>

              {/* 1. Pick a format */}
              <View style={styles.formatRow} accessibilityRole="radiogroup">
                {([
                  { f: 'csv' as const, label: 'CSV', sub: 'Excel · Sheets', Icon: FileSpreadsheet },
                  { f: 'json' as const, label: 'JSON', sub: 'Full data', Icon: FileJson },
                ]).map(({ f, label, sub, Icon }, i) => {
                  const on = exportFormat === f;
                  return (
                    <Animated.View key={f} entering={FadeInDown.delay(80 + i * 60).duration(260)} style={{ flex: 1 }}>
                      <PressableScale onPress={() => setExportFormat(f)} disabled={!!exporting} scaleTo={0.96}
                        style={[styles.formatCard, on && styles.formatCardOn]}
                        accessibilityRole="radio" accessibilityState={{ selected: on }} accessibilityLabel={`${label}, ${sub}`}>
                        <View style={[styles.formatIcon, on && { backgroundColor: theme.colors.brand[600] }]}>
                          <Icon size={16} color={on ? 'white' : theme.colors.brand[700]} />
                        </View>
                        <View style={{ flex: 1 }}>
                          <Text style={[styles.formatLabel, on && { color: theme.colors.brand[800] }]}>{label}</Text>
                          <Text style={styles.formatSub} numberOfLines={1}>{sub}</Text>
                        </View>
                        {on ? <Animated.View entering={ZoomIn.duration(200)}><CheckCircle2 size={16} color={theme.colors.brand[600]} /></Animated.View> : null}
                      </PressableScale>
                    </Animated.View>
                  );
                })}
              </View>

              {/* 2. Save it or share it */}
              <View style={styles.exportActions}>
                <PressableScale containerStyle={{ flex: 1 }} onPress={() => doExport(exportFormat, 'save')} disabled={!!exporting} scaleTo={0.97}
                  style={styles.exportSave} accessibilityLabel={`Save ${exportFormat.toUpperCase()} to Downloads`}
                  leadingIcon={exporting === `${exportFormat}:save` ? <ActivityIndicator size="small" color="white" /> : <FolderDown size={16} color="white" />}>
                  <Text style={styles.exportSaveText} numberOfLines={1}>
                    {Platform.OS === 'ios' ? 'Save to Files' : `Save ${exportFormat.toUpperCase()} to Downloads`}
                  </Text>
                </PressableScale>
                <PressableScale onPress={() => doExport(exportFormat, 'share')} disabled={!!exporting} scaleTo={0.95}
                  style={styles.exportShare} accessibilityLabel={`Share ${exportFormat.toUpperCase()}`}
                  leadingIcon={exporting === `${exportFormat}:share` ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <Share2 size={16} color={theme.colors.brand[700]} />}>
                  <Text style={styles.exportShareText}>Share</Text>
                </PressableScale>
              </View>

              {/* Where saves go */}
              {Platform.OS === 'android' ? (
                <View style={styles.exportFoot}>
                  <FolderDown size={12} color={theme.colors.text.muted} />
                  <Text style={styles.exportFootText} numberOfLines={2}>
                    {saveFolder ? `Saves to ${saveFolder}` : 'First save: create a folder in Downloads (e.g. “EventScout”) and tap “Use this folder”.'}
                  </Text>
                  {saveFolder ? (
                    <Pressable onPress={async () => { await resetDownloadsFolder(); setSaveFolder(null); setToast({ kind: 'ok', text: 'The next save will ask where to put the file' }); }}
                      hitSlop={8} accessibilityLabel="Change download folder">
                      <Text style={[styles.link, { fontSize: 12 }]}>Change</Text>
                    </Pressable>
                  ) : null}
                </View>
              ) : <View style={{ height: 10 }} />}
              <View style={styles.divider} />
              <PressableScale onPress={() => openContactsPage(router, 'deleted', '/me')} scaleTo={0.98} style={styles.listRow}>
                <View style={styles.listIcon}><Trash2 size={17} color={theme.colors.text.body} /></View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.listTitle}>Recently deleted</Text>
                  <Text style={styles.muted}>Restore within 30 days</Text>
                </View>
                {deletedCount ? (
                  <Animated.View entering={ZoomIn} style={styles.countBadge}><Text style={styles.countBadgeText}>{deletedCount}</Text></Animated.View>
                ) : null}
                <ChevronRight size={17} color={theme.colors.text.muted} />
              </PressableScale>
            </Animated.View>

            <Animated.View entering={enter(9)}>
              <PressableScale onPress={askSignOut} scaleTo={0.97} style={styles.signOut}
                leadingIcon={<LogOut size={16} color={theme.colors.semantic.danger} />}>
                <Text style={styles.signOutText}>Sign out</Text>
              </PressableScale>
            </Animated.View>
          </>
        )}
        <View style={{ height: 40 }} />
      </ScrollView>

      {toast ? (
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(200)}
          style={[styles.toast, toast.kind === 'error' && styles.toastError]} pointerEvents="none">
          {toast.kind === 'ok' ? <CheckCircle2 size={16} color="white" /> : <AlertCircle size={16} color="white" />}
          <Text style={styles.toastText}>{toast.text}</Text>
        </Animated.View>
      ) : null}

      {/* Edit profile */}
      <Sheet visible={profileSheet} title="Edit profile" icon={<Pencil size={15} color="white" />} onClose={() => setProfileSheet(false)}>
        <View style={styles.photoRow}>
          <AvatarRing initials={initials} photo={profile?.avatar || null} busy={photoBusy} onPress={() => setPhotoSheet(true)} />
          <View style={{ flex: 1, gap: 6 }}>
            <Text style={styles.toggleTitle}>Profile photo</Text>
            <View style={{ flexDirection: 'row', gap: 8 }}>
              <PressableScale onPress={() => setPhotoSheet(true)} disabled={photoBusy} scaleTo={0.94} style={styles.smallBtn}
                leadingIcon={<Camera size={13} color={theme.colors.brand[700]} />}>
                <Text style={styles.smallBtnText}>{profile?.avatar ? 'Change photo' : 'Add photo'}</Text>
              </PressableScale>
              {profile?.avatar ? (
                <PressableScale onPress={removePhoto} disabled={photoBusy} scaleTo={0.94} style={[styles.smallBtn, styles.smallBtnDanger]}
                  leadingIcon={<Trash2 size={13} color={theme.colors.semantic.danger} />}>
                  <Text style={[styles.smallBtnText, { color: theme.colors.semantic.danger }]}>Remove</Text>
                </PressableScale>
              ) : null}
            </View>
          </View>
        </View>
        <EditableField icon={<UserRound size={16} color={theme.colors.text.muted} />} label="Your name" value={form.full_name}
          onChangeText={(v) => setForm((f) => ({ ...f, full_name: v }))} placeholder="e.g. Priya Raman" autoCapitalize="words" maxLength={120} />
        <EditableField icon={<Briefcase size={16} color={theme.colors.text.muted} />} label="Your title" value={form.job_title}
          onChangeText={(v) => setForm((f) => ({ ...f, job_title: v }))} placeholder="e.g. Head of Sales" autoCapitalize="words" />
        <EditableField icon={<Building2 size={16} color={theme.colors.text.muted} />} label="Company" value={form.company_name}
          onChangeText={(v) => setForm((f) => ({ ...f, company_name: v }))} placeholder="e.g. Northwind" autoCapitalize="words" />
        <EditableField icon={<Globe size={16} color={theme.colors.text.muted} />} label="Company website" value={form.company_website}
          onChangeText={(v) => setForm((f) => ({ ...f, company_website: v }))} placeholder="northwind.com" autoCapitalize="none" keyboardType="url" autoCorrect={false} />
        <EditableField icon={<Phone size={16} color={theme.colors.text.muted} />} label="Phone (on your QR card)" value={form.phone}
          onChangeText={(v) => setForm((f) => ({ ...f, phone: v }))} placeholder="+1 415 555 0142" keyboardType="phone-pad" />
        <EditableField icon={<Linkedin size={16} color={theme.colors.text.muted} />} label="Your LinkedIn" value={form.linkedin_url}
          onChangeText={(v) => setForm((f) => ({ ...f, linkedin_url: v }))} placeholder="linkedin.com/in/you" autoCapitalize="none" keyboardType="url" autoCorrect={false} />
        <Text style={styles.muted}>The AI uses these to tailor pitches and lead scores.</Text>
        {sheetError ? <Animated.Text entering={FadeIn} style={styles.sheetError}>{sheetError}</Animated.Text> : null}
        <PressableScale onPress={saveProfileSheet} disabled={sheetSaving} style={styles.sheetSave}
          leadingIcon={sheetSaving ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" />}>
          <Text style={styles.sheetSaveText}>{sheetSaving ? 'Saving…' : 'Save profile'}</Text>
        </PressableScale>
      </Sheet>

      {/* Targeting */}
      <Sheet visible={targetSheet} title="Who you're targeting" icon={<Target size={15} color="white" />} onClose={() => setTargetSheet(false)}>
        <Text style={styles.subLabel}>INDUSTRIES</Text>
        <TagEditor tags={industries} setTags={setIndustries} tone="industry" addLabel="Add industry" placeholder="e.g. Healthcare" />
        <Text style={[styles.subLabel, { marginTop: 14 }]}>ROLES</Text>
        <TagEditor tags={roles} setTags={setRoles} tone="role" addLabel="Add role" placeholder="e.g. CTO" />
        <Text style={[styles.subLabel, { marginTop: 14 }]}>COMPANY SIZES</Text>
        <ChoiceChips multi
          options={[...COMPANY_SIZES, ...sizes.filter((x) => !COMPANY_SIZES.includes(x))].map((v) => ({ value: v, label: v }))}
          selected={sizes}
          onToggle={(v) => setSizes((prev) => (prev.includes(v) ? prev.filter((x) => x !== v) : [...prev, v]))} />
        <Text style={[styles.subLabel, { marginTop: 14 }]}>REGIONS</Text>
        <TagEditor tags={geos} setTags={setGeos} tone="industry" addLabel="Add region" placeholder="e.g. Middle East" />
        <Text style={[styles.muted, { marginTop: 10 }]}>Lead scores compare each contact against these.</Text>
        {sheetError ? <Animated.Text entering={FadeIn} style={styles.sheetError}>{sheetError}</Animated.Text> : null}
        <PressableScale onPress={saveTargetSheet} disabled={sheetSaving} style={styles.sheetSave}
          leadingIcon={sheetSaving ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" />}>
          <Text style={styles.sheetSaveText}>{sheetSaving ? 'Saving…' : 'Save targeting'}</Text>
        </PressableScale>
      </Sheet>

      {/* Current event picker */}
      <Sheet visible={eventPicker} title="Current event" icon={<CalendarDays size={15} color="white" />} onClose={() => setEventPicker(false)}>
        <Text style={[styles.muted, { marginBottom: 10 }]}>New scans are tagged with your current event, and the AI plans around it.</Text>
        {events.map((name, i) => {
          const on = name === active;
          return (
            <Animated.View key={name} entering={FadeInDown.delay(i * 40).duration(240)} layout={LinearTransition.duration(240)}>
              <PressableScale onPress={async () => { await setActiveEvent(name); setEventPicker(false); }} disabled={!!eventBusy || on} scaleTo={0.98}
                style={[styles.eventRow, on && styles.eventRowOn]} accessibilityRole="radio" accessibilityState={{ selected: on }}>
                <View style={[styles.radio, on && styles.radioOn]}>
                  {eventBusy === name ? <ActivityIndicator size="small" color={theme.colors.brand[600]} />
                    : on ? <Animated.View entering={ZoomIn.duration(240)} style={styles.radioDot} /> : null}
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.eventName, on && { color: theme.colors.brand[800] }]} numberOfLines={1}>{name}</Text>
                  <Text style={styles.eventMeta}>{byEvent[name] ?? 0} contact{(byEvent[name] ?? 0) === 1 ? '' : 's'}{on ? ' · current' : ''}</Text>
                </View>
              </PressableScale>
            </Animated.View>
          );
        })}
        {events.length ? (
          <Animated.View entering={FadeInDown.delay(events.length * 40).duration(240)} layout={LinearTransition.duration(240)}>
            <PressableScale onPress={async () => { await clearActiveEvent(); setEventPicker(false); }} disabled={!!eventBusy || !active} scaleTo={0.98}
              style={[styles.eventRow, !active && styles.eventRowOn]} accessibilityRole="radio" accessibilityState={{ selected: !active }} accessibilityLabel="No active event">
              <View style={[styles.radio, !active && styles.radioOn]}>
                {eventBusy === '__none' ? <ActivityIndicator size="small" color={theme.colors.brand[600]} />
                  : !active ? <Animated.View entering={ZoomIn.duration(240)} style={styles.radioDot} /> : null}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.eventName, !active && { color: theme.colors.brand[800] }]} numberOfLines={1}>None</Text>
                <Text style={styles.eventMeta}>New scans aren't tagged{!active ? ' · current' : ''}</Text>
              </View>
            </PressableScale>
          </Animated.View>
        ) : null}
        <Text style={[styles.subLabel, { marginTop: events.length ? 12 : 0 }]}>ADD AN EVENT</Text>
        <View style={styles.addEventRow}>
          <TextInput
            value={newEvent} onChangeText={setNewEvent} placeholder="Event name, e.g. GITEX 2026"
            placeholderTextColor={theme.colors.text.muted} style={styles.addEventInput} maxLength={120} returnKeyType="done"
            onSubmitEditing={async () => { if (newEvent.trim()) { await addEvent(undefined, true); setEventPicker(false); } }}
          />
          <PressableScale onPress={async () => { await addEvent(undefined, true); setEventPicker(false); }} disabled={!newEvent.trim() || !!eventBusy} scaleTo={0.9}
            style={[styles.addEventBtn, (!newEvent.trim() || !!eventBusy) && { opacity: 0.5 }]} accessibilityLabel="Add and make current">
            {eventBusy === '__add' ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" />}
          </PressableScale>
        </View>
        {suggestions.length ? (
          <>
            <Text style={styles.subLabel}>SUGGESTED</Text>
            <View style={styles.tags}>
              {suggestions.map((s, i) => (
                <Animated.View key={s} entering={ZoomIn.delay(i * 35).duration(220)}>
                  <PressableScale onPress={async () => { await addEvent(s, true); setEventPicker(false); }} disabled={!!eventBusy} scaleTo={0.93}
                    style={styles.suggestChip} leadingIcon={<Plus size={12} color={theme.colors.brand[700]} />} accessibilityLabel={`Make ${s} current`}>
                    <Text style={styles.suggestText} numberOfLines={1}>{s}</Text>
                  </PressableScale>
                </Animated.View>
              ))}
            </View>
          </>
        ) : null}
        <View style={{ height: 10 }} />
      </Sheet>

      {/* Product / service */}
      <Sheet visible={!!productSheet} title={productSheet?.index != null ? 'Edit product or service' : 'Add a product or service'}
        icon={<Package size={15} color="white" />} onClose={() => setProductSheet(null)}>
        <EditableField icon={<Package size={16} color={theme.colors.text.muted} />} label="Name" value={productForm.name}
          onChangeText={(v) => setProductForm((f) => ({ ...f, name: v }))} placeholder="e.g. Remote patient monitoring" autoCapitalize="sentences" maxLength={120} />
        <EditableField icon={<Lightbulb size={16} color={theme.colors.text.muted} />} label="What it does" value={productForm.description}
          onChangeText={(v) => setProductForm((f) => ({ ...f, description: v }))} placeholder="One or two lines on the problem it solves" multiline maxLength={600} />
        <EditableField icon={<Target size={16} color={theme.colors.text.muted} />} label="Ideal customer" value={productForm.ideal_customer}
          onChangeText={(v) => setProductForm((f) => ({ ...f, ideal_customer: v }))} placeholder="e.g. Hospital groups with 5+ sites" autoCapitalize="sentences" maxLength={300} />
        <Text style={styles.muted}>Used in AI chat answers, lead scores, research and pitch suggestions.</Text>
        {sheetError ? <Animated.Text entering={FadeIn} style={styles.sheetError}>{sheetError}</Animated.Text> : null}
        <PressableScale onPress={saveProduct} disabled={sheetSaving} style={styles.sheetSave}
          leadingIcon={sheetSaving ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" />}>
          <Text style={styles.sheetSaveText}>{sheetSaving ? 'Saving…' : productSheet?.index != null ? 'Save changes' : 'Add'}</Text>
        </PressableScale>
        {productSheet?.index != null ? (
          <PressableScale onPress={() => setConfirmDeleteProduct(true)} disabled={sheetSaving} scaleTo={0.97} style={styles.sheetDelete}
            leadingIcon={<Trash2 size={15} color={theme.colors.semantic.danger} />}>
            <Text style={styles.sheetDeleteText}>Remove</Text>
          </PressableScale>
        ) : null}
      </Sheet>

      {/* Sales approach */}
      <Sheet visible={salesSheet} title="How you sell" icon={<MessageSquareQuote size={15} color="white" />} onClose={() => setSalesSheet(false)}>
        <Text style={[styles.subLabel, { marginTop: 0 }]}>PITCH STYLE</Text>
        <ChoiceChips options={PITCH_STYLES} selected={pitchStyle ? [pitchStyle] : []}
          onToggle={(v) => setPitchStyle((cur) => (cur === v ? '' : v))} />
        {pitchStyle ? (
          <Animated.Text key={pitchStyle} entering={FadeIn.duration(220)} style={[styles.muted, { marginTop: 8 }]}>
            {PITCH_STYLES.find((s) => s.value === pitchStyle)?.hint}
          </Animated.Text>
        ) : null}
        <Text style={[styles.subLabel, { marginTop: 16 }]}>VALUE PROPOSITIONS</Text>
        <ListEditor items={valueProps} setItems={setValueProps} addLabel="Add a value proposition" placeholder="e.g. Cuts readmissions by 20%" />
        <Text style={[styles.subLabel, { marginTop: 16 }]}>COMMON OBJECTIONS</Text>
        <PairListEditor items={objections} setItems={setObjections} keys={['objection', 'response']} labels={['OBJECTION', 'YOUR ANSWER']}
          placeholders={['e.g. It’s too expensive', 'e.g. Pays for itself in 6 months — here’s how']} addLabel="Add an objection" />
        <Text style={[styles.subLabel, { marginTop: 16 }]}>CASE STUDIES</Text>
        <PairListEditor items={caseStudies} setItems={setCaseStudies} keys={['title', 'result']} labels={['CUSTOMER / PROJECT', 'RESULT']}
          placeholders={['e.g. Mercy Health Group', 'e.g. 30% fewer no-shows in 3 months']} addLabel="Add a case study" />
        {sheetError ? <Animated.Text entering={FadeIn} style={styles.sheetError}>{sheetError}</Animated.Text> : null}
        <PressableScale onPress={saveSalesSheet} disabled={sheetSaving} style={styles.sheetSave}
          leadingIcon={sheetSaving ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" />}>
          <Text style={styles.sheetSaveText}>{sheetSaving ? 'Saving…' : 'Save sales approach'}</Text>
        </PressableScale>
      </Sheet>

      {/* Event goals */}
      <Sheet visible={goalsSheet} title={goalsEvent ? `Goals for ${goalsEvent}` : 'Event goals'} icon={<Flag size={15} color="white" />} onClose={() => setGoalsSheet(false)}>
        <EditableField icon={<CalendarDays size={16} color={theme.colors.text.muted} />} label="About the event (optional)" value={eventDesc}
          onChangeText={setEventDesc} placeholder="e.g. Largest healthcare show in the region, 3 days, hall 5" multiline maxLength={1000} />
        <Text style={[styles.subLabel, { marginTop: 8 }]}>WHAT YOU WANT FROM IT</Text>
        <ListEditor items={goals} setItems={setGoals} addLabel="Add a goal" placeholder="e.g. Book 10 demos with hospital CIOs" />
        <Text style={[styles.muted, { marginTop: 10 }]}>The AI uses these to rank leads and suggest who to talk to next.</Text>
        {sheetError ? <Animated.Text entering={FadeIn} style={styles.sheetError}>{sheetError}</Animated.Text> : null}
        <PressableScale onPress={saveGoalsSheet} disabled={sheetSaving} style={styles.sheetSave}
          leadingIcon={sheetSaving ? <ActivityIndicator size="small" color="white" /> : <Check size={17} color="white" />}>
          <Text style={styles.sheetSaveText}>{sheetSaving ? 'Saving…' : 'Save goals'}</Text>
        </PressableScale>
      </Sheet>

      {/* Calendar: all notes & reminders (from the count tiles), then the note editor above it */}
      <EventNotesListSheet cal={cal} />
      <EventNoteSheet cal={cal} />

      {/* Your events → All: every event, yours and the admins' scheduled ones */}
      <AllEventsSheet
        visible={allEventsOpen}
        onClose={() => setAllEventsOpen(false)}
        myEvents={events}
        active={active}
        ended={profile?.ended_events || []}
        byEvent={byEvent}
        busy={endBusy}
        onToggleEnded={toggleEnded}
        onOpen={(name) => { setAllEventsOpen(false); router.push(`/events/${encodeURIComponent(name)}` as any); }}
      />

      <OptionSheet
        visible={confirmDeleteProduct}
        title={`Remove “${productSheet?.index != null ? products[productSheet.index]?.name ?? '' : ''}”?`}
        message="The AI will stop using it in pitches and scores."
        options={[{ label: 'Remove', destructive: true, icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, onPress: deleteProduct }]}
        onClose={() => setConfirmDeleteProduct(false)}
      />

      <QrDialog visible={qrOpen} loading={qrLoading} error={qrError} data={qrData} onClose={() => setQrOpen(false)} onRetry={openMyQr} onResetLink={resetMyLink}
        onLoadBusinessCard={async () => { const r = await api.getMyBusinessCard(); return { pngDataUrl: r.pngDataUrl, filename: r.filename }; }} />

      <OptionSheet
        visible={photoSheet}
        title={profile?.avatar ? 'Your photo' : 'Add your photo'}
        message="Shown on your profile."
        options={[
          { label: 'Choose from library', icon: <ImageIcon size={20} color={theme.colors.brand[600]} />, onPress: () => choosePhoto('library') },
          ...(Platform.OS !== 'web' ? [{ label: 'Take a photo', icon: <Camera size={20} color={theme.colors.brand[600]} />, onPress: () => choosePhoto('camera') }] : []),
          ...(profile?.avatar ? [{ label: 'Remove photo', destructive: true, icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, onPress: removePhoto }] : []),
        ]}
        onClose={() => setPhotoSheet(false)}
      />
      <OptionSheet
        visible={confirmSignOut}
        title="Sign out?"
        message={pendingOnSignOut > 0
          ? `${pendingOnSignOut} change${pendingOnSignOut === 1 ? '' : 's'} on this phone ${pendingOnSignOut === 1 ? "hasn't" : "haven't"} synced yet. ${pendingOnSignOut === 1 ? 'It stays' : 'They stay'} here and upload${pendingOnSignOut === 1 ? 's' : ''} when you sign back in — but if a different account signs in on this phone, ${pendingOnSignOut === 1 ? 'it is' : 'they are'} removed. Sync now to keep ${pendingOnSignOut === 1 ? 'it' : 'them'} safe.`
          : 'Everything on this phone is synced.'}
        options={[
          ...(pendingOnSignOut > 0 ? [{ label: 'Sync now', icon: <RefreshCw size={20} color={theme.colors.brand[600]} />, onPress: syncNow }] : []),
          { label: 'Sign out', destructive: true, icon: <LogOut size={20} color={theme.colors.semantic.danger} />, onPress: async () => { await logout(); router.replace('/welcome'); } },
        ]}
        onClose={() => setConfirmSignOut(false)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingTop: 8, paddingBottom: 8 },
  title: { fontSize: 28, lineHeight: 34, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.5 },
  headerSub: { fontSize: 13, color: theme.colors.text.muted },
  syncPill: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999, backgroundColor: theme.colors.semantic.successBg },
  syncPillWarn: { backgroundColor: theme.colors.semantic.warningBg },
  syncPillBad: { backgroundColor: theme.colors.semantic.dangerBg },
  syncPillText: { fontSize: 12, fontWeight: '800', color: theme.colors.semantic.successText },
  container: { paddingHorizontal: 16, paddingTop: 4 },
  loading: { alignItems: 'center', gap: 10, paddingVertical: 60 },
  muted: { fontSize: 13, lineHeight: 18, color: theme.colors.text.muted },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  errorBanner: { flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 12 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  profileCard: {
    backgroundColor: toneBg('#ffffff'), borderRadius: 24, padding: 16, marginBottom: 12, overflow: 'hidden',
    borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.active,
  },
  profileGlow: { position: 'absolute', top: -80, right: -60, width: 200, height: 200, borderRadius: 100, backgroundColor: theme.colors.brand[50] },
  profileRow: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  avatarWrap: { width: 64, height: 64, alignItems: 'center', justifyContent: 'center' },
  avatarRing: { position: 'absolute', width: 64, height: 64, borderRadius: 32, backgroundColor: theme.colors.brand[300] },
  avatar: { width: 56, height: 56, borderRadius: 28, backgroundColor: theme.colors.brand[600], alignItems: 'center', justifyContent: 'center', borderWidth: 3, borderColor: toneBg('#ffffff'), overflow: 'hidden' },
  avatarText: { fontSize: 20, fontWeight: '900', color: 'white' },
  avatarImg: { width: '100%', height: '100%', borderRadius: 28 },
  avatarBusy: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: 28, backgroundColor: 'rgba(15,23,42,0.45)', alignItems: 'center', justifyContent: 'center' },
  avatarBadge: { position: 'absolute', right: 0, bottom: 2, width: 22, height: 22, borderRadius: 11, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600], borderWidth: 2, borderColor: toneBg('#ffffff') },
  photoRow: { flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 14 },
  currentCard: { backgroundColor: theme.colors.fill.brand700, borderRadius: 22, padding: 16, marginBottom: 12, overflow: 'hidden', ...theme.shadows.active },
  currentCardEmpty: { backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[200], shadowOpacity: 0, elevation: 0 },
  currentGlow: { position: 'absolute', top: -70, right: -50, width: 180, height: 180, borderRadius: 90, backgroundColor: theme.colors.brand[500], opacity: 0.45 },
  currentTop: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  currentIcon: { width: 40, height: 40, borderRadius: 13, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.18)' },
  currentLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1.2, color: theme.colors.fg.brand100 },
  currentName: { fontSize: 19, lineHeight: 24, fontWeight: '900', color: 'white', marginTop: 2 },
  currentEmptyText: { fontSize: 13, lineHeight: 18, color: theme.colors.text.body, marginTop: 2 },
  currentStats: { flexDirection: 'row', alignItems: 'center', marginTop: 14, backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 14, paddingVertical: 10 },
  currentStat: { flex: 1, alignItems: 'center' },
  currentStatNum: { fontSize: 18, lineHeight: 23, fontWeight: '900', color: 'white' },
  currentStatLabel: { fontSize: 11, fontWeight: '600', color: theme.colors.fg.brand100 },
  currentStatDivider: { width: 1, height: 26, backgroundColor: 'rgba(255,255,255,0.2)' },
  currentActions: { flexDirection: 'row', gap: 8, marginTop: 12 },
  currentBtnGhost: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, paddingVertical: 10, borderRadius: 999, borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.45)' },
  currentBtnGhostText: { fontSize: 13, fontWeight: '800', color: 'white' },
  currentBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 5, paddingVertical: 10, borderRadius: 999, backgroundColor: toneBg('#ffffff') },
  currentBtnText: { fontSize: 13, fontWeight: '800', color: theme.colors.brand[700] },
  currentSetBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 14, paddingVertical: 12, borderRadius: 999, backgroundColor: theme.colors.brand[600] },
  currentSetBtnText: { fontSize: 14, fontWeight: '800', color: 'white' },
  liveWrap: { width: 12, height: 12, alignItems: 'center', justifyContent: 'center' },
  liveRing: { position: 'absolute', width: 8, height: 8, borderRadius: 4, backgroundColor: toneBg('#4ade80') },
  liveDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: toneBg('#4ade80') },
  suggestChip: { flexDirection: 'row', alignItems: 'center', gap: 4, maxWidth: 220, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 999, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50] },
  suggestText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700], flexShrink: 1 },
  qrRow: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: toneBg('#ffffff'), borderRadius: 20, padding: 14, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.card },
  qrIcon: { width: 42, height: 42, borderRadius: 14, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  smallBtnDanger: { backgroundColor: toneBg('#ffffff'), borderColor: theme.colors.semantic.dangerBorder },
  name: { fontSize: 19, lineHeight: 24, fontWeight: '900', color: theme.colors.text.display },
  role: { fontSize: 13, color: theme.colors.text.body, marginTop: 1 },
  activeChip: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start', marginTop: 6, backgroundColor: theme.colors.brand[50], paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, maxWidth: '100%' },
  activeChipText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700], flexShrink: 1 },
  iconBtn: { width: 38, height: 38, borderRadius: 19, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },

  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  statTile: { alignItems: 'center', gap: 4, backgroundColor: toneBg('#ffffff'), borderRadius: 18, paddingVertical: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  statIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  statValue: { fontSize: 20, lineHeight: 25, fontWeight: '900', color: theme.colors.text.display },
  statLabel: { fontSize: 11, fontWeight: '700', color: theme.colors.text.muted },

  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 20, paddingHorizontal: 14, paddingTop: 10, paddingBottom: 12, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  smallBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  smallBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  addEventRow: { flexDirection: 'row', gap: 8, marginBottom: 10 },
  addEventInput: { flex: 1, height: 44, borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[300], paddingHorizontal: 12, fontSize: 15, color: theme.colors.text.display, backgroundColor: toneBg('#fff') },
  addEventBtn: { width: 44, height: 44, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  eventRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 14, borderWidth: 1.5, borderColor: theme.colors.bg.divider, marginBottom: 8 },
  eventRowOn: { borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50] },
  radio: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center' },
  radioOn: { borderColor: theme.colors.brand[600] },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: theme.colors.brand[600] },
  eventName: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  eventMeta: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },
  moreEventsRow: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 14, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[200], backgroundColor: theme.colors.brand[50], marginBottom: 8 },
  moreEventsText: { flex: 1, fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },

  toggleRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 6 },
  toggleIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  toggleTitle: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  subLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted, marginTop: 12, marginBottom: 8 },
  tags: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  tag: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 9, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  tagRole: { backgroundColor: toneBg('#ccfbf1') },
  tagSize: { backgroundColor: toneBg('#fef3c7') },
  tagGeo: { backgroundColor: toneBg('#e0f2fe') },

  goalsBlock: { marginTop: 6, paddingTop: 10, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  goalsHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  goalRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginTop: 8 },
  goalText: { flex: 1, fontSize: 13, lineHeight: 18, color: theme.colors.text.body },

  emptyCta: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 14, borderWidth: 1.5, borderStyle: 'dashed', borderColor: theme.colors.brand[200], backgroundColor: theme.colors.brand[50] },
  emptyIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff') },
  productRow: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 12, borderRadius: 14, borderWidth: 1.5, borderColor: theme.colors.bg.divider, marginBottom: 8 },
  productIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600] },
  productIndex: { fontSize: 13, fontWeight: '900', color: 'white' },
  idealChip: { flexDirection: 'row', alignItems: 'center', gap: 4, alignSelf: 'flex-start', marginTop: 6, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: toneBg('#ccfbf1'), maxWidth: '100%' },
  idealText: { fontSize: 11, fontWeight: '700', color: toneFg('#0f766e'), flexShrink: 1 },

  styleRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 2 },
  stylePill: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999, backgroundColor: theme.colors.brand[600] },
  stylePillText: { fontSize: 12, fontWeight: '800', color: 'white' },
  countRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 12 },
  countPill: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  countPillNum: { fontSize: 12, fontWeight: '900', color: theme.colors.brand[800] },
  countPillText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  tagText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },

  syncRow: { flexDirection: 'row', alignItems: 'center', gap: 16, marginTop: 4 },
  syncStat: { alignItems: 'center' },
  syncNum: { fontSize: 20, lineHeight: 25, fontWeight: '900', color: theme.colors.text.display },
  syncLabel: { fontSize: 11, color: theme.colors.text.muted },
  primarySmall: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999, backgroundColor: theme.colors.brand[600] },
  primarySmallText: { fontSize: 13, fontWeight: '800', color: 'white' },
  rowLink: { flexDirection: 'row', alignItems: 'center', gap: 2, marginTop: 10, alignSelf: 'flex-start' },

  exportRow: { flexDirection: 'row', gap: 8, marginTop: 10 },
  exportBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  exportBtnText: { fontSize: 13, fontWeight: '800', color: theme.colors.brand[800] },
  exportBtnSub: { flex: 1, fontSize: 12, fontWeight: '600', color: theme.colors.text.muted },
  formatRow: { flexDirection: 'row', gap: 8, marginTop: 2 },
  formatCard: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 14, borderWidth: 1.5, borderColor: theme.colors.bg.divider, backgroundColor: toneBg('#ffffff') },
  formatCardOn: { borderColor: theme.colors.brand[400], backgroundColor: theme.colors.brand[50] },
  formatIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  formatLabel: { fontSize: 14, lineHeight: 18, fontWeight: '800', color: theme.colors.text.display },
  formatSub: { fontSize: 11, lineHeight: 14, fontWeight: '600', color: theme.colors.text.muted },
  exportActions: { flexDirection: 'row', gap: 8, marginTop: 10 },
  exportSave: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 46, paddingHorizontal: 14, borderRadius: 14, backgroundColor: theme.colors.brand[600], ...theme.shadows.card },
  exportSaveText: { fontSize: 14, fontWeight: '800', color: 'white', flexShrink: 1 },
  exportShare: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, height: 46, paddingHorizontal: 16, borderRadius: 14, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.brand[200] },
  exportShareText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  exportFoot: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingTop: 10, paddingBottom: 12 },
  exportFootText: { flex: 1, fontSize: 11, lineHeight: 15, color: theme.colors.text.muted },
  listRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 12 },
  listIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  listTitle: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  divider: { height: 1, backgroundColor: theme.colors.bg.divider, marginLeft: 48 },
  countBadge: { minWidth: 22, height: 22, borderRadius: 11, paddingHorizontal: 6, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.semantic.danger },
  countBadgeText: { fontSize: 11, fontWeight: '800', color: 'white' },

  signOut: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 14, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, backgroundColor: toneBg('#ffffff'), marginTop: 4 },
  signOutText: { fontSize: 15, fontWeight: '800', color: theme.colors.semantic.danger },

  toast: {
    position: 'absolute', left: 16, right: 16, bottom: 16, flexDirection: 'row', alignItems: 'center', gap: 8,
    backgroundColor: theme.colors.semantic.successText, paddingHorizontal: 14, paddingVertical: 12, borderRadius: 14, ...theme.shadows.active,
  },
  toastError: { backgroundColor: theme.colors.semantic.danger },
  toastText: { flex: 1, fontSize: 13, fontWeight: '700', color: 'white' },

  sheetError: { fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', marginTop: 10 },
  sheetSave: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 15, borderRadius: 999, backgroundColor: theme.colors.brand[600], marginTop: 16, marginBottom: 6 },
  sheetSaveText: { fontSize: 15, fontWeight: '800', color: 'white' },
  sheetDelete: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 12, borderRadius: 999, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, backgroundColor: toneBg('#ffffff'), marginBottom: 6 },
  sheetDeleteText: { fontSize: 14, fontWeight: '800', color: theme.colors.semantic.danger },
}));
