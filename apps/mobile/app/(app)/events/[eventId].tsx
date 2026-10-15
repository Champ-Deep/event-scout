import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { View, ScrollView, Pressable, RefreshControl, TextInput, Linking, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  FadeIn, FadeInDown, FadeOut, ZoomIn, LinearTransition,
  useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming, withDelay, Easing,
} from 'react-native-reanimated';
import {
  ArrowLeft, CalendarDays, MapPin, Users, Flame, Sparkles, TrendingUp, ScanLine, Store, Search, X, Globe,
  ChevronRight, ChevronDown, CheckCircle2, AlertCircle, Handshake, WifiOff, Flag, Pencil, Plus, Check, CalendarPlus, Trash2, UsersRound, FolderOpen,
} from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { api, errorMessage, type Exhibitor, type BoothInput, type Seat, type EventFile, type PlannedEvent } from '../../../src/lib/api';
import { contactsFeature } from '../../../src/features/contacts';
import { eventRules, profileStore, type Profile } from '../../../src/features/profile';
import { EventInfoCard } from '../../../src/features/events/EventInfoCard';
import { EndEventButton } from '../../../src/features/events/EndEventButton';
import { FileList } from '../../../src/features/files/FileList';
import { useSession } from '../../../src/providers/SessionProvider';
import { rankContacts } from '../../../src/features/feed/rank';
import { ContactAvatar } from '../../../src/features/contacts/components/ContactAvatar';
import { AnimatedCount, SkeletonRows } from '../../../src/features/contacts/components/ListAnimations';
import { PressableScale } from '../../../src/features/onboarding/components';
import { SCORE_LEVELS, scoreLevel, isHot } from '../../../src/features/contacts/scoreLevel';
import { ListEditor } from '../../../src/features/me/components/ProfileEditors';
import { eventDetailFor, withEventDetail, type EventDetail } from '../../../src/features/me/eventGoals';
import { KeyboardAwareView } from '../../../src/ui/keyboard';
import { FloorMap, BoothRoute, GuideComingSoon, buildGuide, located, locationLabel } from '../../../src/features/events/BoothMap';
import { Sheet } from '../../../src/ui/Sheet';
import { openContact } from '../../../src/features/contacts/open';

// Booths previewed on the page; the rest open in the all-booths sheet (rows capped there, search narrows).
const BOOTH_PREVIEW = 3;
const BOOTH_SHEET_MAX = 100;
// Upper bound on contacts loaded for one event (pages of 500 until the cursor runs out).
const MAX_EVENT_CONTACTS = 5000;

const norm = (s?: string | null) => (s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

/** Animated bar that fills to `value` (0–1). */
function Bar({ value, color, delay }: { value: number; color: string; delay: number }) {
  const w = useSharedValue(0);
  useEffect(() => { w.value = withDelay(delay, withTiming(value, { duration: 800, easing: Easing.out(Easing.cubic) })); }, [value]);
  const style = useAnimatedStyle(() => ({ width: `${w.value * 100}%`, backgroundColor: color }));
  return <View style={styles.barTrack}><Animated.View style={[styles.barFill, style]} /></View>;
}

function LiveDot() {
  const t = useSharedValue(0);
  useEffect(() => { t.value = withRepeat(withTiming(1, { duration: 1400 }), -1); }, []);
  const ring = useAnimatedStyle(() => ({ opacity: 0.6 * (1 - t.value), transform: [{ scale: 1 + t.value * 1.4 }] }));
  return (
    <View style={styles.liveWrap}>
      <Animated.View style={[styles.liveRing, ring]} />
      <View style={styles.liveDot} />
    </View>
  );
}

/** Chevron that turns smoothly instead of snapping. */
function Rotate({ open, children }: { open: boolean; children: React.ReactNode }) {
  const r = useSharedValue(open ? 180 : 0);
  useEffect(() => { r.value = withTiming(open ? 180 : 0, { duration: 220, easing: Easing.out(Easing.quad) }); }, [open]);
  const style = useAnimatedStyle(() => ({ transform: [{ rotate: `${r.value}deg` }] }));
  return <Animated.View style={style}>{children}</Animated.View>;
}

/** Hero icon: a slow, gentle float. */
function Float({ children }: { children: React.ReactNode }) {
  const y = useSharedValue(0);
  useEffect(() => {
    y.value = withRepeat(withSequence(
      withTiming(-3, { duration: 1600, easing: Easing.inOut(Easing.quad) }),
      withTiming(0, { duration: 1600, easing: Easing.inOut(Easing.quad) }),
    ), -1);
  }, []);
  const style = useAnimatedStyle(() => ({ transform: [{ translateY: y.value }] }));
  return <Animated.View style={style}>{children}</Animated.View>;
}

/** This event's own goals & notes (profile.event_details[event]) — view + inline edit. */
function GoalsCard({ eventName, detail, onSave }: { eventName: string; detail: EventDetail; onSave: (d: EventDetail) => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [desc, setDesc] = useState('');
  const [goals, setGoals] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');
  const has = detail.goals.length > 0 || !!detail.description;

  const start = () => { setDesc(detail.description); setGoals([...detail.goals]); setErr(''); setEditing(true); };
  const save = async () => {
    setSaving(true);
    setErr('');
    try {
      await onSave({ goals, description: desc });
      setEditing(false);
    } catch (e: any) {
      setErr(errorMessage(e, "Couldn't save the goals."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Animated.View entering={FadeInDown.delay(170).duration(260)} layout={LinearTransition.duration(260)} style={styles.card}>
      <View style={styles.goalsHead}>
        <View style={styles.goalsIcon}><Flag size={14} color={theme.colors.brand[700]} /></View>
        <Text style={[styles.sectionLabel, { flex: 1 }]}>YOUR GOALS HERE</Text>
        {editing ? (
          <Pressable onPress={() => setEditing(false)} hitSlop={8} disabled={saving}><Text style={styles.cancelText}>Cancel</Text></Pressable>
        ) : (
          <PressableScale onPress={start} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel={`Edit goals for ${eventName}`}
            leadingIcon={has ? <Pencil size={12} color={theme.colors.brand[700]} /> : <Plus size={13} color={theme.colors.brand[700]} />}>
            <Text style={styles.smallBtnText}>{has ? 'Edit' : 'Add'}</Text>
          </PressableScale>
        )}
      </View>

      {editing ? (
        <Animated.View entering={FadeIn.duration(220)} style={{ gap: 10 }}>
          <TextInput value={desc} onChangeText={setDesc} multiline maxLength={1000}
            placeholder={`About ${eventName} (optional) — e.g. 3 days, hall 5, our booth B12`}
            placeholderTextColor={theme.colors.text.muted} style={styles.goalInput} />
          <ListEditor items={goals} setItems={setGoals} addLabel="Add a goal" placeholder="e.g. Book 10 demos with hospital CIOs" />
          {err ? <Animated.Text entering={FadeIn} style={styles.goalErr}>{err}</Animated.Text> : null}
          <PressableScale onPress={save} disabled={saving} scaleTo={0.97} style={styles.primaryBtn}
            leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : <Check size={16} color="white" />}>
            <Text style={styles.primaryText}>{saving ? 'Saving…' : 'Save goals'}</Text>
          </PressableScale>
        </Animated.View>
      ) : has ? (
        <>
          {detail.description ? <Animated.Text entering={FadeIn.duration(260)} style={styles.goalDesc}>{detail.description}</Animated.Text> : null}
          {detail.goals.map((g, i) => (
            <Animated.View key={`${eventName}-${g}`} entering={FadeInDown.delay(i * 50).duration(260)} layout={LinearTransition.duration(260)} style={styles.goalRow}>
              <View style={styles.goalNum}><Text style={styles.goalNumText}>{i + 1}</Text></View>
              <Text style={styles.goalText}>{g}</Text>
            </Animated.View>
          ))}
        </>
      ) : (
        <Pressable onPress={start}>
          <Text style={styles.emptyText}>What do you want from {eventName}? Goals here are only for this event — the AI ranks leads and suggests who to see around them.</Text>
        </Pressable>
      )}
    </Animated.View>
  );
}

// Booth-form suggestions (GET /api/exhibitors/categories): this event's categories first, then the
// rest of the directory. Cached per event for this session.
const categoryCache = new Map<string, string[]>();

/** Add a booth to this event (one the directory doesn't list, or when there's no directory yet),
 *  or edit one you added (`editing`). */
function AddBoothCard({ eventName, editing, onAdded, onSaved, onClose }: {
  eventName: string;
  editing?: Exhibitor | null;
  onAdded: (x: Exhibitor & { existed: boolean }) => void;
  onSaved?: (x: Exhibitor) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState(editing?.name ?? '');
  const [booth, setBooth] = useState(editing?.booth ?? '');
  const [hall, setHall] = useState(editing?.hall ?? '');
  const [category, setCategory] = useState(editing?.category ?? '');
  const [visited, setVisited] = useState(true);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>(categoryCache.get(eventName) ?? []);

  useEffect(() => {
    if (categoryCache.has(eventName)) return;
    Promise.allSettled([api.getExhibitorCategories(eventName), api.getExhibitorCategories()]).then(([here, all]) => {
      const names = (r: PromiseSettledResult<{ category: string; count: number }[]>) =>
        r.status === 'fulfilled' && Array.isArray(r.value) ? r.value.map((c) => c.category).filter(Boolean) : [];
      const merged = [...new Set([...names(here), ...names(all)])].slice(0, 12);
      categoryCache.set(eventName, merged);
      setSuggestions(merged);
    });
  }, [eventName]);
  const shownSuggestions = suggestions.filter((c) => c.toLowerCase() !== category.trim().toLowerCase()
    && (!category.trim() || c.toLowerCase().includes(category.trim().toLowerCase()))).slice(0, 6);

  const save = async () => {
    if (!name.trim()) { setErr('Enter the company / booth name.'); return; }
    setSaving(true);
    setErr('');
    try {
      if (editing) {
        onSaved?.(await api.updateBooth(editing.id, { name: name.trim(), booth: booth.trim(), hall: hall.trim(), category: category.trim() }));
      } else {
        const input: BoothInput = { event: eventName, name: name.trim(), booth: booth.trim(), hall: hall.trim(), category: category.trim(), visited };
        onAdded(await api.addBooth(input));
      }
    } catch (e: any) {
      setErr(errorMessage(e, editing ? "Couldn't save the booth." : "Couldn't add the booth."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(160)} layout={LinearTransition.duration(260)} style={[styles.card, styles.addCard]}>
      <View style={styles.goalsHead}>
        <View style={styles.goalsIcon}><Store size={14} color={theme.colors.brand[700]} /></View>
        <Text style={[styles.sectionLabel, { flex: 1 }]}>{editing ? 'EDIT BOOTH' : 'ADD A BOOTH'}</Text>
        <Pressable onPress={onClose} hitSlop={8} disabled={saving}><Text style={styles.cancelText}>Cancel</Text></Pressable>
      </View>
      <TextInput value={name} onChangeText={setName} placeholder="Company / booth name *" maxLength={500} autoFocus={!editing}
        placeholderTextColor={theme.colors.text.muted} style={styles.fieldInput} returnKeyType="next" />
      <View style={{ flexDirection: 'row', gap: 8 }}>
        <TextInput value={booth} onChangeText={setBooth} placeholder="Booth (e.g. B-12)" maxLength={100} autoCapitalize="characters"
          placeholderTextColor={theme.colors.text.muted} style={[styles.fieldInput, { flex: 1 }]} />
        <TextInput value={hall} onChangeText={setHall} placeholder="Hall" maxLength={100}
          placeholderTextColor={theme.colors.text.muted} style={[styles.fieldInput, { flex: 1 }]} />
      </View>
      <TextInput value={category} onChangeText={setCategory} placeholder="Category (optional)" maxLength={500}
        placeholderTextColor={theme.colors.text.muted} style={styles.fieldInput} />
      {shownSuggestions.length ? (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} keyboardShouldPersistTaps="handled" contentContainerStyle={{ gap: 6 }}>
          {shownSuggestions.map((c, i) => (
            <Animated.View key={c} entering={FadeInDown.delay(i * 30).duration(220)}>
              <PressableScale onPress={() => setCategory(c)} scaleTo={0.92} style={styles.catChip}>
                <Text style={styles.catText} numberOfLines={1}>{c}</Text>
              </PressableScale>
            </Animated.View>
          ))}
        </ScrollView>
      ) : null}
      {!editing ? (
        <Pressable onPress={() => setVisited((v) => !v)} style={styles.checkRow} accessibilityRole="checkbox" accessibilityState={{ checked: visited }}>
          <View style={[styles.checkBox, visited && styles.checkBoxOn]}>
            {visited ? <Animated.View entering={ZoomIn.duration(180)}><Check size={13} color="white" /></Animated.View> : null}
          </View>
          <Text style={styles.checkText}>I visited this booth</Text>
        </Pressable>
      ) : null}
      {err ? <Animated.Text entering={FadeIn} style={styles.goalErr}>{err}</Animated.Text> : null}
      <PressableScale onPress={save} disabled={saving} scaleTo={0.97} style={styles.primaryBtn}
        leadingIcon={saving ? <ActivityIndicator size="small" color="white" /> : editing ? <Check size={16} color="white" /> : <Plus size={16} color="white" />}>
        <Text style={styles.primaryText}>{saving ? (editing ? 'Saving…' : 'Adding…') : editing ? 'Save booth' : 'Add booth'}</Text>
      </PressableScale>
    </Animated.View>
  );
}

export default function EventScreen() {
  const router = useRouter();
  // `section` deep-links from the Scout shortcuts (booths | map | met); `t` makes a repeat tap re-scroll.
  const { eventId, section, t } = useLocalSearchParams<{ eventId: string; section?: string; t?: string }>();
  // The route carries the event's name (events are named, like scans are tagged).
  const name = decodeURIComponent(String(eventId || '')).trim();
  const { userId } = useSession();

  const [contacts, setContacts] = useState<Contact[]>([]);
  const [exhibitors, setExhibitors] = useState<Exhibitor[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [activating, setActivating] = useState(false);
  const [toast, setToast] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<string | null>(null);
  const [openEx, setOpenEx] = useState<string | null>(null);
  const [showAllContacts, setShowAllContacts] = useState(false);
  // Exhibitors: "Met" filter (also opened from the Booths met tile), add-booth form, busy row
  const [metOnly, setMetOnly] = useState(false);
  const [addingBooth, setAddingBooth] = useState(false);
  const [busyEx, setBusyEx] = useState<string | null>(null);
  const [editingBooth, setEditingBooth] = useState<Exhibitor | null>(null);
  // Teammates who scanned here (GET /api/events/{name}/seats) and this event's files (GET /api/files/list?event=)
  const [seats, setSeats] = useState<Seat[]>([]);
  const [eventFiles, setEventFiles] = useState<EventFile[]>([]);
  // The admins' plan for this event (GET /api/me/events): dates, venue, address, booth, hours…
  const [planned, setPlanned] = useState<PlannedEvent | null>(null);
  // Floor map ↔ road map ↔ list: the booth you last tapped, and whether the all-booths sheet is open
  const [focusId, setFocusId] = useState<string | null>(null);
  const [boothsOpen, setBoothsOpen] = useState(false);
  const scrollRef = useRef<ScrollView>(null);
  const exhibitorsY = useRef(0);
  const mapY = useRef(0);
  const goalsY = useRef(0);
  const seatsY = useRef(0);
  const filesY = useRef(0);

  const load = useCallback(async () => {
    if (!name) return;
    const [c, x, p, st, fl, pe] = await Promise.allSettled([
      // Every contact from this event: every page (capped so a huge event can't stall the page).
      contactsFeature.listContacts({ eventId: name, limit: 500, allPages: true, cap: MAX_EVENT_CONTACTS }),
      api.listExhibitors({ event: name, limit: 1000 }),
      userId ? profileStore.load(userId) : Promise.resolve(null),
      api.getEventSeats(name),
      api.listEventFiles(name),
      api.listMyPlannedEvents('all'),
    ]);
    if (pe.status === 'fulfilled') {
      const k = (s: string) => s.split(/\s+/).join(' ').toLowerCase();
      setPlanned((Array.isArray(pe.value) ? pe.value : []).find((e) => k(e.name) === k(name)) ?? null);
    }
    if (st.status === 'fulfilled') setSeats(Array.isArray(st.value) ? st.value : []);
    if (fl.status === 'fulfilled') setEventFiles((Array.isArray(fl.value) ? fl.value : []).filter((f) => !f.placeholder));
    if (c.status === 'fulfilled') setContacts(rankContacts(c.value.contacts));
    if (x.status === 'fulfilled') setExhibitors(Array.isArray(x.value) ? (x.value as any) : []);
    if (p.status === 'fulfilled') setProfile(p.value);
    setError(c.status === 'rejected' ? errorMessage(c.reason, "Couldn't load this event.") : '');
    setLoading(false);
  }, [name, userId]);

  useFocusEffect(useCallback(() => { load(); }, [load]));

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2800);
    return () => clearTimeout(t);
  }, [toast]);

  const isActive = eventRules.isActive(profile, name);
  const isMine = (profile?.events || []).includes(name);
  // Ended by you (End event) — it stays in your list, marked ended; Reopen undoes it.
  const isEnded = eventRules.isEnded(profile, name);
  const [ending, setEnding] = useState(false);
  const toggleEnded = async () => {
    if (!userId || ending) return;
    setEnding(true);
    try {
      const saved = await profileStore.toggleEnded(userId, profile, name);
      setProfile((prev) => ({ ...(prev || {}), ...saved }));
      setToast({ kind: 'ok', text: isEnded ? `Reopened “${name}”` : isActive ? `Ended “${name}” · new scans aren't tagged now` : `Ended “${name}”` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, isEnded ? "Couldn't reopen this event." : "Couldn't end this event.") });
    } finally {
      setEnding(false);
    }
  };

  const makeActive = async (thenScan = false) => {
    if (!userId || activating) return;
    if (isActive) { if (thenScan) router.push('/(app)/scan' as any); return; }
    setActivating(true);
    try {
      // joins your events, and an ended event is reopened (eventRules.activate)
      const saved = await profileStore.activateEvent(userId, profile, name);
      setProfile((prev) => ({ ...(prev || {}), ...saved, current_event_name: name }));
      setToast({ kind: 'ok', text: `Scans are now tagged “${name}”` });
      if (thenScan) router.push('/(app)/scan' as any);
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't switch to this event.") });
    } finally {
      setActivating(false);
    }
  };

  const detail = eventDetailFor(profile, name);
  const [adding, setAdding] = useState(false);

  const addToMine = async () => {
    if (!userId || adding || isMine) return;
    setAdding(true);
    try {
      const saved = await profileStore.save(userId, { events: [...(profile?.events || []), name] });
      setProfile((prev) => ({ ...(prev || {}), ...saved }));
      setToast({ kind: 'ok', text: `Added “${name}” to your events` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't add this event.") });
    } finally {
      setAdding(false);
    }
  };

  // Goals are saved per event; an event needs to be in your list to keep its goals.
  const saveGoals = async (d: EventDetail) => {
    if (!userId) throw new Error('Sign in again to save.');
    const saved = await profileStore.save(userId, {
      event_details: withEventDetail(profile, name, d),
      ...eventRules.ensureListed(profile, name),
    });
    setProfile((prev) => ({ ...(prev || {}), ...saved }));
    setToast({ kind: 'ok', text: `Goals saved for “${name}”` });
  };

  /* ---- stats (from this event's real contacts) ---- */
  const stats = useMemo(() => {
    const scored = contacts.filter((c) => c.score != null);
    const by = { hot: 0, warm: 0, cool: 0, cold: 0 };
    scored.forEach((c) => { by[scoreLevel(c.score)!.key] += 1; });
    const avg = scored.length ? Math.round(scored.reduce((s, c) => s + (c.score as number), 0) / scored.length) : null;
    return { total: contacts.length, scored: scored.length, ...by, avg };
  }, [contacts]);

  /* ---- exhibitors: search, categories, "you met someone from here" ---- */
  const metByCompany = useMemo(() => {
    const m = new Map<string, number>();
    contacts.forEach((c) => { const k = norm(c.company); if (k) m.set(k, (m.get(k) || 0) + 1); });
    return m;
  }, [contacts]);
  // "Met": contacts whose company is this exhibitor. Same name, or one name containing the other
  // as whole words ("Mercy Health" ↔ "Mercy Health Group") — short names like "AI" or "ABC" must
  // match exactly, so they don't count every company that happens to contain those letters.
  const metAt = (x: Exhibitor) => {
    const k = norm(x.name);
    if (!k) return 0;
    let n = 0;
    const words = (s: string) => ` ${s} `;
    metByCompany.forEach((count, company) => {
      const same = company === k;
      const contains = k.length >= 4 && company.length >= 4 && (words(company).includes(words(k)) || words(k).includes(words(company)));
      if (same || contains) n += count;
    });
    return n;
  };
  const isMet = (x: Exhibitor) => !!x.visited || metAt(x) > 0;
  const categories = useMemo(() => {
    const m = new Map<string, number>();
    exhibitors.forEach((x) => x.category && m.set(x.category, (m.get(x.category) || 0) + 1));
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
  }, [exhibitors]);
  // Booths you've met float to the top, then the ones you marked visited.
  const boothRank = (x: Exhibitor) => metAt(x) * 2 + (x.visited ? 1 : 0);
  // The page previews the top few; the full, searchable list lives in the all-booths sheet.
  const topBooths = useMemo(
    () => [...exhibitors].sort((a, b) => boothRank(b) - boothRank(a)).slice(0, BOOTH_PREVIEW),
    [exhibitors, metByCompany],
  );
  // Sheet list (search / Met / category). Matches are counted in full; rows are capped so a
  // thousand-stall directory doesn't render at once — searching narrows it.
  const matchedExhibitors = useMemo(() => {
    const q = norm(query);
    return exhibitors
      .filter((x) => !metOnly || isMet(x))
      .filter((x) => !category || x.category === category)
      .filter((x) => !q || norm([x.name, x.category, x.description, x.booth, x.hall, x.country].join(' ')).includes(q))
      .sort((a, b) => boothRank(b) - boothRank(a));
  }, [exhibitors, query, category, metByCompany, metOnly]);
  const shownExhibitors = matchedExhibitors.slice(0, BOOTH_SHEET_MAX);
  // Booths met = booths you marked visited + booths whose company you have a contact from here
  const visitedCount = useMemo(() => exhibitors.filter(isMet).length, [exhibitors, metByCompany]);

  /* ---- floor map + road map ---- */
  // The booth guide reads the stalls against your preferences: this event's goals, then your
  // targeting from Me. It needs the organiser's floor plan; until then it shows "coming soon".
  const prefs = {
    goals: detail.goals,
    industries: (profile?.target_industries || []) as string[],
    geographies: (profile?.target_geographies || []) as string[],
  };
  const prefsKey = [...prefs.goals, '§', ...prefs.industries, '§', ...prefs.geographies].join('|');
  const guide = useMemo(() => buildGuide(exhibitors, isMet, prefs), [exhibitors, metByCompany, prefsKey]);
  const picks = useMemo(() => new Set(guide.stops.map((x) => x.id)), [guide]);
  const nextId = guide.stops.find((x) => !isMet(x))?.id ?? null;
  const floorPlans = useMemo(() => eventFiles
    // Admin console → Files category "floorplan", or (older uploads) a name / description that says so
    .filter((f) => (f.category || '').trim().toLowerCase() === 'floorplan' || /floor|plan|map|layout/i.test(`${f.filename} ${f.description || ''}`))
    .map((f) => ({ id: f.id, filename: f.filename, fileType: f.fileType, fileSize: f.fileSize, description: f.description })), [eventFiles]);

  // Map tile → open that booth in the all-booths sheet
  const pickBooth = (x: Exhibitor) => {
    setFocusId(x.id);
    setOpenEx(x.id);
    setMetOnly(false);
    setCategory(null);
    setQuery(x.name);
    setBoothsOpen(true);
  };
  // Road-map stop → show it on the floor map
  const focusOnMap = (x: Exhibitor) => {
    setFocusId(x.id);
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ y: Math.max(0, mapY.current - 8), animated: true }));
  };

  const showMetBooths = () => {
    setMetOnly(true);
    setCategory(null);
    setQuery('');
    setBoothsOpen(true);
  };
  // "Show all N booths" → the sheet, unfiltered
  const showAllBooths = () => {
    setMetOnly(false);
    setCategory(null);
    setQuery('');
    setBoothsOpen(true);
  };
  // Add / edit forms live on the page: close the sheet and bring the form into view.
  const formOnPage = (open: () => void) => {
    setBoothsOpen(false);
    open();
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ y: Math.max(0, exhibitorsY.current - 8), animated: true }));
  };

  // Deep link: once the event has loaded and laid out, jump to the requested section (once per tap).
  const jumped = useRef<string | null>(null);
  useEffect(() => {
    if (loading || !section) return;
    const key = `${section}|${t ?? ''}`;
    if (jumped.current === key) return;
    const timer = setTimeout(() => {
      jumped.current = key;
      if (section === 'met') { showMetBooths(); return; }
      // Team and files only render when there's something to show.
      if (section === 'seats' && !seats.length) { setToast({ kind: 'ok', text: 'No teammates have scanned at this event yet' }); return; }
      if (section === 'files' && !eventFiles.length) { setToast({ kind: 'ok', text: 'No files shared for this event yet' }); return; }
      const y = section === 'map' && (floorPlans.length || exhibitors.some(located)) ? mapY.current
        : section === 'seats' ? seatsY.current
        : section === 'files' ? filesY.current
        : section === 'goals' ? goalsY.current
        : exhibitorsY.current;
      scrollRef.current?.scrollTo({ y: Math.max(0, y - 8), animated: true });
    }, 350);
    return () => clearTimeout(timer);
  }, [loading, section, t, floorPlans.length]);

  const replaceEx = (x: Exhibitor) => setExhibitors((list) => list.map((e) => (e.id === x.id ? { ...e, ...x } : e)));

  const toggleVisited = async (x: Exhibitor) => {
    if (busyEx) return;
    const next = !x.visited;
    setBusyEx(x.id);
    replaceEx({ ...x, visited: next }); // optimistic
    try {
      replaceEx(await api.setBoothVisited(x.id, next));
      setToast({ kind: 'ok', text: next ? `Marked “${x.name}” as visited` : `Un-marked “${x.name}”` });
    } catch (e: any) {
      replaceEx(x);
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't update the booth.") });
    } finally {
      setBusyEx(null);
    }
  };

  const removeBooth = async (x: Exhibitor) => {
    if (busyEx) return;
    setBusyEx(x.id);
    try {
      await api.deleteBooth(x.id);
      setExhibitors((list) => list.filter((e) => e.id !== x.id));
      setOpenEx(null);
      setToast({ kind: 'ok', text: `Removed “${x.name}”` });
    } catch (e: any) {
      setToast({ kind: 'error', text: errorMessage(e, "Couldn't remove the booth.") });
    } finally {
      setBusyEx(null);
    }
  };

  const onBoothSaved = (x: Exhibitor) => {
    replaceEx(x);
    setEditingBooth(null);
    setToast({ kind: 'ok', text: `Saved “${x.name}”` });
  };

  const onBoothAdded = (x: Exhibitor & { existed: boolean }) => {
    setExhibitors((list) => (list.some((e) => e.id === x.id)
      ? list.map((e) => (e.id === x.id ? { ...e, ...x } : e))
      : [...list, x].sort((a, b) => a.name.localeCompare(b.name))));
    setAddingBooth(false);
    setOpenEx(x.id);
    setToast({ kind: 'ok', text: x.existed ? `“${x.name}” is already listed${x.visited ? ' — marked visited' : ''}` : `Added “${x.name}”` });
  };

  const onRefresh = async () => { setRefreshing(true); await load(); setRefreshing(false); };
  const visibleContacts = showAllContacts ? contacts : contacts.slice(0, 5);
  const bars = [
    ...SCORE_LEVELS.map((l) => ({ label: l.name, n: stats[l.key], color: l.ring })),
    { label: 'Not scored', n: stats.total - stats.scored, color: toneFg('#cbd5e1') },
  ];

  // One booth row (tap to expand: details, Mark visited, Website, Ask AI, Edit/Remove) — used by
  // the page preview and the all-booths sheet. In the sheet rows don't stagger in or animate their
  // layout: every keystroke in the search re-filters the list, and that made it jump around.
  const renderBooth = (x: Exhibitor, i: number, inSheet = false) => {
    const met = metAt(x);
    const done = met > 0 || !!x.visited;
    const open = openEx === x.id;
    const busy = busyEx === x.id;
    return (
      <Animated.View key={x.id}
        {...(inSheet ? {} : { entering: FadeInDown.delay(Math.min(i, 8) * 40).duration(260), layout: LinearTransition.duration(260) })}>
        <Pressable onPress={() => setOpenEx(open ? null : x.id)} style={({ pressed }) => [styles.exRow, open && styles.exRowOpen, pressed && { opacity: 0.9 }]}>
          <View style={styles.exTop}>
            <View style={[styles.exIcon, done && { backgroundColor: theme.colors.semantic.successBg }]}>
              {done ? <Handshake size={15} color={theme.colors.semantic.successText} /> : <Store size={15} color={theme.colors.brand[700]} />}
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.exName} numberOfLines={open ? undefined : 1}>{x.name}</Text>
              <Text style={styles.exSub} numberOfLines={1}>
                {[locationLabel(x), x.category].filter(Boolean).join(' · ') || x.country || 'Exhibitor'}
              </Text>
            </View>
            {met > 0 ? (
              <Animated.View entering={ZoomIn.duration(260)} style={styles.metBadge}><Text style={styles.metText}>Met {met}</Text></Animated.View>
            ) : x.visited ? (
              <Animated.View entering={ZoomIn.duration(260)} style={styles.metBadge}><Text style={styles.metText}>Visited</Text></Animated.View>
            ) : null}
            <Rotate open={open}><ChevronDown size={16} color={theme.colors.text.muted} /></Rotate>
          </View>
          {open ? (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.exBody}>
              {x.description ? <Text style={styles.exDesc}>{x.description}</Text> : <Text style={styles.emptyText}>No description in the directory.</Text>}
              <View style={styles.exMeta}>
                {x.hall || x.booth ? <View style={styles.exMetaChip}><MapPin size={11} color={theme.colors.brand[700]} /><Text style={styles.exMetaText}>{locationLabel(x)}</Text></View> : null}
                {x.country ? <View style={styles.exMetaChip}><Globe size={11} color={theme.colors.brand[700]} /><Text style={styles.exMetaText}>{x.country}</Text></View> : null}
                {(x.tags || []).slice(0, 4).map((t) => <View key={t} style={styles.exMetaChip}><Text style={styles.exMetaText}>{t}</Text></View>)}
              </View>
              {met > 0 ? (
                <Text style={styles.exHint}>You have {met} contact{met === 1 ? '' : 's'} from {x.name} at this event.</Text>
              ) : null}
              <View style={styles.exActions}>
                <PressableScale onPress={() => toggleVisited(x)} disabled={busy} scaleTo={0.94}
                  style={[styles.exBtn, x.visited && styles.exBtnOn]}
                  accessibilityLabel={x.visited ? `Un-mark ${x.name} as visited` : `Mark ${x.name} as visited`}
                  leadingIcon={busy ? <ActivityIndicator size="small" color={x.visited ? 'white' : theme.colors.brand[700]} />
                    : x.visited ? <Check size={13} color="white" /> : <Handshake size={13} color={theme.colors.brand[700]} />}>
                  <Text style={[styles.exBtnText, x.visited && { color: 'white' }]}>{x.visited ? 'Visited' : 'Mark visited'}</Text>
                </PressableScale>
                {x.website ? (
                  <PressableScale onPress={() => Linking.openURL(/^https?:/i.test(x.website) ? x.website : `https://${x.website}`)} scaleTo={0.94} style={styles.exBtn}
                    leadingIcon={<Globe size={13} color={theme.colors.brand[700]} />}>
                    <Text style={styles.exBtnText}>Website</Text>
                  </PressableScale>
                ) : null}
                <PressableScale onPress={() => router.push({ pathname: '/ask', params: { q: `Should I visit ${x.name}${x.booth ? ` (booth ${x.booth})` : ''} at ${name}? What should I pitch them?`, n: String(Date.now()) } } as any)}
                  scaleTo={0.94} style={styles.exBtn} leadingIcon={<Sparkles size={13} color={theme.colors.brand[700]} />}>
                  <Text style={styles.exBtnText}>Ask AI</Text>
                </PressableScale>
                {x.addedByMe ? (
                  <PressableScale onPress={() => formOnPage(() => { setAddingBooth(false); setEditingBooth(x); })}
                    disabled={busy} scaleTo={0.94} style={styles.exBtn} accessibilityLabel={`Edit ${x.name}`}
                    leadingIcon={<Pencil size={13} color={theme.colors.brand[700]} />}>
                    <Text style={styles.exBtnText}>Edit</Text>
                  </PressableScale>
                ) : null}
                {x.addedByMe ? (
                  <PressableScale onPress={() => removeBooth(x)} disabled={busy} scaleTo={0.94} style={styles.exBtnDanger}
                    accessibilityLabel={`Remove ${x.name}`} leadingIcon={<Trash2 size={13} color={theme.colors.semantic.danger} />}>
                    <Text style={[styles.exBtnText, { color: theme.colors.semantic.danger }]}>Remove</Text>
                  </PressableScale>
                ) : null}
              </View>
            </Animated.View>
          ) : null}
        </Pressable>
      </Animated.View>
    );
  };

  if (!name) {
    return (
      <SafeAreaView style={[styles.safeArea, styles.center]}>
        <Text style={styles.emptyTitle}>Event not found</Text>
        <Pressable onPress={() => router.back()}><Text style={styles.link}>Go back</Text></Pressable>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      <KeyboardAwareView style={{ flex: 1 }}>
      <Animated.View entering={FadeInDown.duration(260)} style={styles.header}>
        <PressableScale onPress={() => router.back()} scaleTo={0.9} style={styles.backBtn} accessibilityLabel="Back">
          <ArrowLeft size={20} color={theme.colors.text.display} />
        </PressableScale>
        <Text style={styles.headerTitle} numberOfLines={1}>Event</Text>
      </Animated.View>

      <ScrollView
        ref={scrollRef}
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        refreshControl={<RefreshControl progressBackgroundColor={theme.colors.bg.surface} refreshing={refreshing} onRefresh={onRefresh} tintColor={theme.colors.brand[600]} colors={[theme.colors.brand[600]]} />}
      >
        {/* Hero */}
        <Animated.View entering={FadeInDown.delay(60).duration(260)} style={styles.hero}>
          <View style={styles.heroGlow} />
          <Float><View style={styles.heroIcon}><CalendarDays size={22} color="white" /></View></Float>
          <Text style={styles.eventName}>{name}</Text>
          <View style={styles.heroChips}>
            {isActive ? (
              <Animated.View entering={ZoomIn.duration(260)} style={[styles.chip, styles.chipLive]}>
                <LiveDot />
                <Text style={[styles.chipText, { color: theme.colors.semantic.successText }]}>Active — scans are tagged here</Text>
              </Animated.View>
            ) : (
              <View style={styles.chip}><Text style={styles.chipText}>{isMine ? 'One of your events' : 'Not in your events'}</Text></View>
            )}
            {exhibitors.length ? (
              <View style={styles.chip}><Store size={11} color={theme.colors.brand[700]} /><Text style={styles.chipText}>{exhibitors.length} exhibitors</Text></View>
            ) : null}
            {isEnded ? (
              <Animated.View entering={ZoomIn.duration(240)} style={styles.chip}><Flag size={11} color={theme.colors.text.muted} /><Text style={styles.chipText}>Ended</Text></Animated.View>
            ) : null}
          </View>
          <View style={styles.heroActions}>
            {!isMine && !isActive ? (
              <PressableScale onPress={addToMine} disabled={adding} scaleTo={0.92} style={styles.askBtn} accessibilityLabel="Add to my events">
                {adding ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <CalendarPlus size={17} color={theme.colors.brand[700]} />}
              </PressableScale>
            ) : null}
            {!isActive ? (
              <PressableScale onPress={() => makeActive(false)} disabled={activating} containerStyle={{ flex: 1 }} style={styles.ghostBtn}
                leadingIcon={activating ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <CheckCircle2 size={16} color={theme.colors.brand[700]} />}>
                <Text style={styles.ghostText}>Make active</Text>
              </PressableScale>
            ) : null}
            <PressableScale onPress={() => makeActive(true)} disabled={activating} containerStyle={{ flex: 1 }} style={styles.primaryBtn}
              leadingIcon={<ScanLine size={16} color="white" />}>
              <Text style={styles.primaryText}>Scan here</Text>
            </PressableScale>
            <PressableScale onPress={() => router.push({ pathname: '/ask', params: { q: `Summarize the people I met at ${name} and who I should follow up with first${detail.goals.length ? `, given my goals there: ${detail.goals.join('; ')}` : ''}.`, n: String(Date.now()) } } as any)}
              scaleTo={0.92} style={styles.askBtn} accessibilityLabel="Ask AI about this event">
              <Sparkles size={17} color={theme.colors.brand[700]} />
            </PressableScale>
          </View>
          {userId ? (
            <View style={{ alignItems: 'center', marginTop: 10 }}>
              <EndEventButton ended={isEnded} busy={ending} onPress={toggleEnded} />
            </View>
          ) : null}
        </Animated.View>

        {error ? (
          <Animated.View entering={FadeInDown} style={styles.errorBanner}>
            <WifiOff size={16} color={theme.colors.semantic.danger} />
            <Text style={styles.errorText}>{error} Pull down to retry.</Text>
          </Animated.View>
        ) : null}

        {/* What the admins planned: dates, location, booth, hours, contact */}
        {planned ? (
          <EventInfoCard event={planned} onOpenFloorMap={() => {
            const y = floorPlans.length ? mapY.current : filesY.current;
            scrollRef.current?.scrollTo({ y: Math.max(0, y - 8), animated: true });
          }} />
        ) : null}

        {/* Stats */}
        <View style={styles.statsRow}>
          {[
            { icon: <Users size={15} color={theme.colors.brand[700]} />, tint: theme.colors.brand[50], v: stats.total, l: 'Met here' },
            { icon: <Flame size={15} color={toneFg('#dc2626')} />, tint: toneBg('#fef2f2'), v: stats.hot, l: 'Hot' },
            { icon: <TrendingUp size={15} color={theme.colors.semantic.successText} />, tint: theme.colors.semantic.successBg, v: stats.avg, l: 'Avg score' },
            { icon: <Handshake size={15} color={toneFg('#0369a1')} />, tint: toneBg('#f0f9ff'), v: visitedCount, l: 'Booths met', onPress: showMetBooths },
          ].map((s: { icon: React.ReactNode; tint: string; v: number | null; l: string; onPress?: () => void }, i) => {
            const body = (
              <>
                <View style={[styles.statIcon, { backgroundColor: s.tint }]}>{s.icon}</View>
                {loading || s.v == null ? <Text style={styles.statValue}>—</Text> : <AnimatedCount value={s.v} style={styles.statValue} />}
                <Text style={styles.statLabel}>{s.l}</Text>
              </>
            );
            return (
              <Animated.View key={s.l} entering={FadeInDown.delay(120 + i * 60).duration(260)} style={{ flex: 1 }}>
                {s.onPress ? (
                  <PressableScale onPress={s.onPress} scaleTo={0.95} containerStyle={{ flex: 1 }} style={[styles.statTile, { flex: 1 }]}
                    accessibilityLabel={`${s.l}: ${s.v ?? 0}. Show them`}>
                    {body}
                  </PressableScale>
                ) : <View style={[styles.statTile, { flex: 1 }]}>{body}</View>}
              </Animated.View>
            );
          })}
        </View>

        {/* Exhibitor directory for this event */}
        <View style={[styles.sectionHead, { marginTop: 4 }]} onLayout={(e) => { exhibitorsY.current = e.nativeEvent.layout.y; }}>
          <Store size={13} color={theme.colors.brand[700]} />
          <Text style={styles.sectionLabel}>BOOTHS</Text>
          <View style={{ flex: 1 }} />
          {!loading && !addingBooth && !editingBooth ? (
            <PressableScale onPress={() => setAddingBooth(true)} scaleTo={0.9} style={styles.smallBtn} accessibilityLabel="Add a booth"
              leadingIcon={<Plus size={13} color={theme.colors.brand[700]} />}>
              <Text style={styles.smallBtnText}>Add booth</Text>
            </PressableScale>
          ) : null}
        </View>
        {addingBooth ? <AddBoothCard eventName={name} onAdded={onBoothAdded} onClose={() => setAddingBooth(false)} /> : null}
        {editingBooth ? (
          <AddBoothCard key={editingBooth.id} eventName={name} editing={editingBooth} onAdded={onBoothAdded} onSaved={onBoothSaved} onClose={() => setEditingBooth(null)} />
        ) : null}
        {loading ? null : exhibitors.length === 0 ? (
          addingBooth ? null : (
            <Animated.View entering={FadeIn} onLayout={(e) => { if (floorPlans.length) mapY.current = e.nativeEvent.layout.y; }}>
              {floorPlans.length ? (
                // No booths yet, but the admins uploaded the venue's floor plan: show it to open / save.
                <View style={[styles.card, { gap: 8 }]}>
                  <View style={styles.sectionHead}>
                    <MapPin size={13} color={theme.colors.brand[700]} />
                    <Text style={styles.sectionLabel}>FLOOR MAP</Text>
                    <View style={{ flex: 1 }} />
                    <Text style={styles.sectionCount}>{floorPlans.length}</Text>
                  </View>
                  <FileList files={floorPlans} download={(id) => api.downloadEventFile(id)} />
                </View>
              ) : <GuideComingSoon stalls={0} />}
              <Pressable onPress={() => setAddingBooth(true)} style={styles.emptyCard}>
                <Store size={20} color={theme.colors.text.muted} />
                <Text style={styles.emptyText}>No booths for this event yet. Tap to add the booths you visit — or an admin can import the full exhibitor list from the admin console.</Text>
              </Pressable>
            </Animated.View>
          )
        ) : (
          <>
            {/* The map is drawn from the booths' own hall / booth numbers; the official plan file is optional. */}
            {floorPlans.length || exhibitors.some(located) ? (
              <>
                <View onLayout={(e) => { mapY.current = e.nativeEvent.layout.y; }}>
                  <FloorMap exhibitors={exhibitors} isMet={isMet} picks={picks} nextId={nextId} focusId={focusId} onPick={pickBooth}
                    planFiles={floorPlans} download={(id) => api.downloadEventFile(id)} />
                </View>
                <BoothRoute guide={guide} isMet={isMet} nextId={nextId} focusId={focusId} busyId={busyEx}
                  onFocus={focusOnMap} onToggle={toggleVisited}
                  onSetGoals={() => scrollRef.current?.scrollTo({ y: Math.max(0, goalsY.current - 8), animated: true })} />
              </>
            ) : <GuideComingSoon stalls={exhibitors.length} />}

            <View style={[styles.sectionHead, { marginTop: 6 }]}>
              <Store size={13} color={theme.colors.brand[700]} />
              <Text style={styles.sectionLabel}>ALL BOOTHS</Text>
              <View style={{ flex: 1 }} />
              <Text style={styles.sectionCount}>{exhibitors.length}</Text>
            </View>
            {topBooths.map((x, i) => renderBooth(x, i))}
            {exhibitors.length > BOOTH_PREVIEW ? (
              <Pressable onPress={showAllBooths} style={styles.moreRow} accessibilityRole="button" accessibilityLabel={`Show all ${exhibitors.length} booths`}>
                <Text style={styles.link}>Show all {exhibitors.length} booths</Text>
                <ChevronRight size={15} color={theme.colors.brand[600]} />
              </Pressable>
            ) : null}
          </>
        )}
        <View style={{ height: 14 }} />

        {/* This event's own goals */}
        <View onLayout={(e) => { goalsY.current = e.nativeEvent.layout.y; }}>
          {!loading ? <GoalsCard eventName={name} detail={detail} onSave={saveGoals} /> : null}
        </View>

        {/* Lead quality breakdown */}
        {stats.total > 0 ? (
          <Animated.View entering={FadeInDown.delay(200).duration(260)} style={styles.card}>
            <Text style={styles.sectionLabel}>LEAD QUALITY</Text>
            {bars.map((b, i) => (
              <View key={b.label} style={styles.barRow}>
                <Text style={styles.barLabel}>{b.label}</Text>
                <Bar value={stats.total ? b.n / stats.total : 0} color={b.color} delay={250 + i * 90} />
                <Text style={styles.barCount}>{b.n}</Text>
              </View>
            ))}
          </Animated.View>
        ) : null}

        {/* People met here */}
        <View style={styles.sectionHead}>
          <Users size={13} color={theme.colors.brand[700]} />
          <Text style={styles.sectionLabel}>PEOPLE YOU MET HERE</Text>
          <View style={{ flex: 1 }} />
          {contacts.length ? <Text style={styles.sectionCount}>{contacts.length}</Text> : null}
        </View>
        {loading ? (
          <SkeletonRows count={3} />
        ) : contacts.length === 0 ? (
          <Animated.View entering={FadeIn} style={styles.emptyCard}>
            <ScanLine size={20} color={theme.colors.brand[600]} />
            <Text style={styles.emptyText}>No contacts tagged with this event yet. {isActive ? 'Scan a card to add one.' : 'Make it active, then scan — new contacts get tagged automatically.'}</Text>
          </Animated.View>
        ) : (
          <>
            {visibleContacts.map((c, i) => {
              const nm = [c.firstName, c.lastName].filter(Boolean).join(' ') || 'Unnamed contact';
              const hot = isHot(c);
              return (
                <Animated.View key={c.id} entering={FadeInDown.delay(Math.min(i, 8) * 50).duration(260)} layout={LinearTransition.duration(260)}>
                  <PressableScale onPress={() => openContact(router, c.id, `/events/${encodeURIComponent(name)}`)} scaleTo={0.98} style={styles.personRow}>
                    <ContactAvatar firstName={c.firstName} lastName={c.lastName} photoUrl={c.photoUrl} size={40} />
                    <View style={{ flex: 1 }}>
                      <Text style={styles.personName} numberOfLines={1}>{nm}</Text>
                      <Text style={styles.personSub} numberOfLines={1}>{[c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No title or company yet'}</Text>
                    </View>
                    {c.score != null ? (
                      <View style={[styles.scorePill, hot && { backgroundColor: toneBg('#fee2e2') }]}>
                        <Text style={[styles.scoreText, hot && { color: toneFg('#b91c1c') }]}>{c.score}</Text>
                      </View>
                    ) : <Text style={styles.unscored}>—</Text>}
                    <ChevronRight size={15} color={theme.colors.text.muted} />
                  </PressableScale>
                </Animated.View>
              );
            })}
            {contacts.length > 5 ? (
              <Pressable onPress={() => setShowAllContacts((x) => !x)} style={styles.moreRow}>
                <Text style={styles.link}>{showAllContacts ? 'Show less' : `Show all ${contacts.length}`}</Text>
                <Rotate open={showAllContacts}><ChevronDown size={15} color={theme.colors.brand[600]} /></Rotate>
              </Pressable>
            ) : null}
          </>
        )}

        {/* Your team at this event: who scanned here, and how many */}
        {seats.length ? (
          <>
            <View style={[styles.sectionHead, { marginTop: 18 }]} onLayout={(e) => { seatsY.current = e.nativeEvent.layout.y; }}>
              <UsersRound size={13} color={theme.colors.brand[700]} />
              <Text style={styles.sectionLabel}>YOUR TEAM HERE</Text>
              <View style={{ flex: 1 }} />
              <Text style={styles.sectionCount}>{seats.length}</Text>
            </View>
            <Animated.View entering={FadeInDown.duration(260)} style={[styles.card, { gap: 4 }]}>
              {seats.map((s, i) => {
                const [first, ...rest] = (s.name || s.email || '').split(' ');
                const me = s.userId === userId;
                const top = Math.max(...seats.map((x) => x.scansCount), 1);
                return (
                  <Animated.View key={s.id} entering={FadeInDown.delay(Math.min(i, 8) * 40).duration(240)} style={styles.seatRow}>
                    <ContactAvatar firstName={first} lastName={rest.join(' ')} size={32} />
                    <View style={{ flex: 1, gap: 4 }}>
                      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                        <Text style={styles.personName} numberOfLines={1}>{s.name || s.email}</Text>
                        {me ? <View style={styles.youTag}><Text style={styles.youText}>You</Text></View> : null}
                      </View>
                      <Bar value={s.scansCount / top} color={me ? theme.colors.brand[600] : theme.colors.brand[300]} delay={120 + i * 60} />
                    </View>
                    <Text style={styles.seatCount}>{s.scansCount}</Text>
                  </Animated.View>
                );
              })}
            </Animated.View>
          </>
        ) : null}

        {/* This event's files from your admins (open / save) */}
        {eventFiles.length ? (
          <>
            <View style={[styles.sectionHead, { marginTop: 18 }]} onLayout={(e) => { filesY.current = e.nativeEvent.layout.y; }}>
              <FolderOpen size={13} color={theme.colors.brand[700]} />
              <Text style={styles.sectionLabel}>EVENT FILES</Text>
              <View style={{ flex: 1 }} />
              <Text style={styles.sectionCount}>{eventFiles.length}</Text>
            </View>
            <FileList
              download={(id) => api.downloadEventFile(id)}
              files={eventFiles.map((f) => ({ id: f.id, filename: f.filename, fileType: f.fileType, fileSize: f.fileSize, description: f.description,
                meta: f.eventName ? null : 'All events' }))}
            />
          </>
        ) : null}

        <View style={{ height: 40 }} />
      </ScrollView>
      </KeyboardAwareView>

      {/* All booths: search, Met / category filters and every booth's actions */}
      <Sheet
        visible={boothsOpen}
        tall
        deferBody
        title={`Booths · ${exhibitors.length}`}
        icon={<Store size={15} color="white" />}
        onClose={() => setBoothsOpen(false)}
        header={(
          <>
            <View style={styles.searchBox}>
              <Search size={17} color={theme.colors.text.muted} />
              <TextInput value={query} onChangeText={setQuery} placeholder="Search exhibitors, booths, categories…"
                placeholderTextColor={theme.colors.text.muted} style={styles.searchInput} autoCorrect={false} />
              {query ? (
                <Animated.View entering={ZoomIn} exiting={FadeOut}>
                  <Pressable onPress={() => setQuery('')} hitSlop={8} accessibilityLabel="Clear search"><X size={16} color={theme.colors.text.muted} /></Pressable>
                </Animated.View>
              ) : null}
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} keyboardShouldPersistTaps="handled" style={{ flexGrow: 0 }} contentContainerStyle={styles.catRow}>
              <PressableScale onPress={() => setMetOnly((m) => !m)} scaleTo={0.92} style={[styles.catChip, styles.metChip, metOnly && styles.metChipOn]}
                leadingIcon={<Handshake size={12} color={metOnly ? 'white' : theme.colors.semantic.successText} />}>
                <Text style={[styles.catText, { color: metOnly ? 'white' : theme.colors.semantic.successText }]} numberOfLines={1}>Met · {visitedCount}</Text>
              </PressableScale>
              {categories.length > 1 ? [['All', exhibitors.length] as [string, number], ...categories].map(([cat, n]) => {
                const on = (cat === 'All' && !category) || cat === category;
                return (
                  <PressableScale key={cat} onPress={() => setCategory(cat === 'All' ? null : cat)} scaleTo={0.92} style={[styles.catChip, on && styles.catChipOn]}>
                    <Text style={[styles.catText, on && { color: 'white' }]} numberOfLines={1}>{cat} · {n}</Text>
                  </PressableScale>
                );
              }) : null}
            </ScrollView>
            <View style={styles.sheetMeta}>
              <Text style={styles.sectionCount}>
                {matchedExhibitors.length === exhibitors.length ? `${exhibitors.length} booths` : `${matchedExhibitors.length} of ${exhibitors.length} booths`}
              </Text>
              <View style={{ flex: 1 }} />
              <PressableScale onPress={() => formOnPage(() => { setEditingBooth(null); setAddingBooth(true); })} scaleTo={0.9} style={styles.smallBtn}
                accessibilityLabel="Add a booth" leadingIcon={<Plus size={13} color={theme.colors.brand[700]} />}>
                <Text style={styles.smallBtnText}>Add booth</Text>
              </PressableScale>
            </View>
          </>
        )}
      >
        {shownExhibitors.length === 0 ? (
          <Text style={[styles.emptyText, { textAlign: 'center', paddingVertical: 24 }]}>
            {metOnly ? 'No booths met yet — open a booth and tap “Mark visited”, or scan someone from it.' : 'No exhibitors match.'}
          </Text>
        ) : shownExhibitors.map((x, i) => renderBooth(x, i, true))}
        {matchedExhibitors.length > shownExhibitors.length ? (
          <Text style={[styles.emptyText, { textAlign: 'center', paddingVertical: 12 }]}>
            Showing the first {shownExhibitors.length} — search to find the rest.
          </Text>
        ) : null}
        <View style={{ height: 8 }} />
      </Sheet>

      {toast ? (
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut} pointerEvents="none" style={[styles.toast, toast.kind === 'error' && styles.toastError]}>
          {toast.kind === 'ok' ? <CheckCircle2 size={16} color="white" /> : <AlertCircle size={16} color="white" />}
          <Text style={styles.toastText}>{toast.text}</Text>
        </Animated.View>
      ) : null}
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  center: { alignItems: 'center', justifyContent: 'center', gap: 10 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingHorizontal: 16, paddingVertical: 10 },
  backBtn: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider },
  headerTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  container: { paddingHorizontal: 16, paddingTop: 2 },
  link: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },

  hero: { alignItems: 'center', backgroundColor: toneBg('#ffffff'), borderRadius: 24, padding: 20, marginBottom: 12, overflow: 'hidden', borderWidth: 1, borderColor: theme.colors.brand[100], ...theme.shadows.active },
  heroGlow: { position: 'absolute', top: -90, width: 280, height: 180, borderRadius: 140, backgroundColor: theme.colors.brand[50] },
  heroIcon: { width: 52, height: 52, borderRadius: 18, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600], ...theme.shadows.active },
  eventName: { fontSize: 22, lineHeight: 28, fontWeight: '900', color: theme.colors.text.display, textAlign: 'center', marginTop: 12 },
  heroChips: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 6, marginTop: 10 },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  chipLive: { backgroundColor: theme.colors.semantic.successBg },
  chipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  liveWrap: { width: 10, height: 10, alignItems: 'center', justifyContent: 'center' },
  liveRing: { position: 'absolute', width: 10, height: 10, borderRadius: 5, backgroundColor: theme.colors.semantic.success },
  liveDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: theme.colors.semantic.success },
  heroActions: { flexDirection: 'row', alignSelf: 'stretch', gap: 8, marginTop: 16 },
  primaryBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 12, borderRadius: 999, backgroundColor: theme.colors.brand[600] },
  primaryText: { fontSize: 14, fontWeight: '800', color: 'white' },
  ghostBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, paddingVertical: 12, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.brand[200] },
  ghostText: { fontSize: 14, fontWeight: '800', color: theme.colors.brand[700] },
  askBtn: { width: 46, height: 46, borderRadius: 23, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },

  errorBanner: { flexDirection: 'row', gap: 8, alignItems: 'center', backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 12 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  statsRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  statTile: { flex: 1, alignItems: 'center', gap: 4, backgroundColor: toneBg('#ffffff'), borderRadius: 16, paddingVertical: 11, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  statIcon: { width: 28, height: 28, borderRadius: 9, alignItems: 'center', justifyContent: 'center' },
  statValue: { fontSize: 18, lineHeight: 23, fontWeight: '900', color: theme.colors.text.display },
  statLabel: { fontSize: 10, fontWeight: '700', color: theme.colors.text.muted },

  card: { backgroundColor: toneBg('#ffffff'), borderRadius: 18, padding: 14, gap: 9, marginBottom: 14, borderWidth: 1, borderColor: theme.colors.bg.divider },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  barLabel: { width: 74, fontSize: 12, fontWeight: '700', color: theme.colors.text.body },
  barTrack: { flex: 1, height: 8, borderRadius: 4, backgroundColor: theme.colors.bg.surfaceHover, overflow: 'hidden' },
  barFill: { height: 8, borderRadius: 4 },
  barCount: { width: 26, textAlign: 'right', fontSize: 12, fontWeight: '800', color: theme.colors.text.display },

  sectionHead: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8, paddingHorizontal: 2 },
  sectionLabel: { fontSize: 11, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted },
  sectionCount: { fontSize: 12, fontWeight: '800', color: theme.colors.text.muted },

  personRow: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 11, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  personName: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display },
  personSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1 },
  scorePill: { minWidth: 34, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999, alignItems: 'center', backgroundColor: theme.colors.brand[50] },
  scoreText: { fontSize: 13, fontWeight: '800', color: theme.colors.brand[700] },
  unscored: { width: 34, textAlign: 'center', color: theme.colors.text.muted },
  moreRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 4, paddingVertical: 8 },

  emptyCard: { flexDirection: 'row', alignItems: 'center', gap: 10, backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 14, borderWidth: 1, borderStyle: 'dashed', borderColor: theme.colors.bg.dashed },
  emptyTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display },
  emptyText: { flex: 1, fontSize: 13, lineHeight: 19, color: theme.colors.text.muted },

  searchBox: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: toneBg('#ffffff'), borderWidth: 1, borderColor: theme.colors.bg.divider, borderRadius: 14, paddingHorizontal: 12, height: 44, marginBottom: 10 },
  searchInput: { flex: 1, fontSize: 15, color: theme.colors.text.display },
  catRow: { gap: 6, paddingBottom: 10, paddingRight: 8 },
  catChip: { maxWidth: 200, paddingHorizontal: 11, paddingVertical: 6, borderRadius: 999, backgroundColor: toneBg('#ffffff'), borderWidth: 1.5, borderColor: theme.colors.bg.divider },
  catChipOn: { backgroundColor: theme.colors.brand[600], borderColor: theme.colors.fill.brand700 },
  catText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.display },
  exRow: { backgroundColor: toneBg('#ffffff'), borderRadius: 16, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: theme.colors.bg.divider },
  exRowOpen: { borderColor: theme.colors.brand[200] },
  exTop: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  exIcon: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  exName: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  exSub: { fontSize: 12, color: theme.colors.text.muted, marginTop: 1 },
  metBadge: { paddingHorizontal: 7, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.semantic.successBg },
  metText: { fontSize: 10, fontWeight: '800', color: theme.colors.semantic.successText },
  exBody: { marginTop: 10, gap: 10 },
  exDesc: { fontSize: 13, lineHeight: 19, color: theme.colors.text.body },
  exMeta: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  exMetaChip: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  exMetaText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700] },
  exActions: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  exBtnOn: { backgroundColor: theme.colors.semantic.success, borderColor: theme.colors.semantic.success },
  exBtnDanger: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, backgroundColor: theme.colors.semantic.dangerBg, borderWidth: 1, borderColor: theme.colors.semantic.dangerBorder },
  exHint: { fontSize: 12, fontWeight: '600', color: theme.colors.semantic.successText },
  metChip: { flexDirection: 'row', alignItems: 'center', gap: 5, borderColor: theme.colors.semantic.successBg },
  metChipOn: { backgroundColor: theme.colors.semantic.success, borderColor: theme.colors.semantic.success },
  addCard: { borderColor: theme.colors.brand[200] },
  seatRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 5 },
  seatCount: { minWidth: 30, textAlign: 'right', fontSize: 15, fontWeight: '900', color: theme.colors.text.display },
  youTag: { paddingHorizontal: 6, paddingVertical: 1, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  youText: { fontSize: 10, fontWeight: '800', color: theme.colors.brand[700] },
  fieldInput: { fontSize: 14, color: theme.colors.text.display, backgroundColor: theme.colors.bg.surfaceHover, borderRadius: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, paddingHorizontal: 12, paddingVertical: 10 },
  checkRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 2 },
  checkBox: { width: 22, height: 22, borderRadius: 7, borderWidth: 1.5, borderColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center' },
  checkBoxOn: { backgroundColor: theme.colors.semantic.success, borderColor: theme.colors.semantic.success },
  checkText: { fontSize: 14, fontWeight: '600', color: theme.colors.text.display },
  exBtn: { flexDirection: 'row', alignItems: 'center', gap: 5, paddingHorizontal: 12, paddingVertical: 7, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  exBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },

  goalsHead: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  goalsIcon: { width: 26, height: 26, borderRadius: 8, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50] },
  smallBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  smallBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  cancelText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.muted },
  goalDesc: { fontSize: 13, lineHeight: 19, color: theme.colors.text.body },
  goalRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  goalNum: { width: 22, height: 22, borderRadius: 11, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[600], marginTop: 1 },
  goalNumText: { fontSize: 11, fontWeight: '900', color: 'white' },
  goalText: { flex: 1, fontSize: 14, lineHeight: 20, color: theme.colors.text.display },
  goalInput: { minHeight: 64, borderRadius: 12, borderWidth: 1.5, borderColor: theme.colors.brand[200], paddingHorizontal: 12, paddingVertical: 10, fontSize: 14, color: theme.colors.text.display, backgroundColor: toneBg('#ffffff'), textAlignVertical: 'top' },
  goalErr: { fontSize: 13, fontWeight: '600', color: theme.colors.semantic.danger },

  sheetMeta: { flexDirection: 'row', alignItems: 'center', paddingBottom: 8 },
  // Above the booths sheet (zIndex 50) so Mark visited / Remove feedback shows there too.
  toast: { position: 'absolute', left: 16, right: 16, bottom: 16, flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.successText, paddingHorizontal: 14, paddingVertical: 12, borderRadius: 14, ...theme.shadows.active, zIndex: 60, elevation: 60 },
  toastError: { backgroundColor: theme.colors.semantic.danger },
  toastText: { flex: 1, fontSize: 13, fontWeight: '700', color: 'white' },
}));
