import React, { useEffect, useState } from 'react';
import { View, Image, Pressable, Linking, ActivityIndicator } from 'react-native';
import Animated, {
  FadeIn, FadeInDown, FadeOut, LinearTransition, ZoomIn,
  useSharedValue, useAnimatedStyle, withRepeat, withTiming, withSequence, Easing,
} from 'react-native-reanimated';
import { Linkedin, Search, ExternalLink, Check, RotateCw, AlertCircle, CheckCircle2, UserSearch, Image as ImageIcon } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { Text } from '../../../ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../ui/theme';
import { api, errorMessage, type ErrorMessageOptions, type LinkedInCandidate, type LinkedInSearchResult } from '../../../lib/api';
import { contactsFeature } from '..';
import { PressableScale } from '../../onboarding/components';
import { initialsFor } from './ContactAvatar';

const LI_BLUE = '#0a66c2';

type Phase =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'results'; result: LinkedInSearchResult }
  | { kind: 'error'; message: string };

// How a failed LinkedIn search is worded (see errorMessage).
const LINKEDIN_ERRORS: ErrorMessageOptions = {
  offline: "You're offline. Connect to the internet to search LinkedIn.",
  timeout: 'The search took too long. Please try again.',
};
const SEARCH_FAILED = "Couldn't search LinkedIn right now.";

const STEPS = ['Searching the web…', 'Looking for LinkedIn profiles…', 'Matching name, company & job title…'];

/** Pulsing LinkedIn mark + step text that changes while the search runs. */
function Searching() {
  const pulse = useSharedValue(0);
  const [step, setStep] = useState(0);
  useEffect(() => {
    pulse.value = withRepeat(withTiming(1, { duration: 1100, easing: Easing.out(Easing.quad) }), -1);
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 2600);
    return () => clearInterval(t);
  }, []);
  const ring = useAnimatedStyle(() => ({ opacity: 0.45 * (1 - pulse.value), transform: [{ scale: 1 + pulse.value * 0.8 }] }));
  return (
    <Animated.View entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} style={styles.searching}>
      <View style={styles.pulseWrap}>
        <Animated.View style={[styles.pulseRing, ring]} />
        <View style={styles.pulseCore}><Linkedin size={18} color="white" /></View>
      </View>
      <Animated.Text key={step} entering={FadeInDown.duration(220)} style={styles.searchingText}>{STEPS[step]}</Animated.Text>
      <View style={{ gap: 8, alignSelf: 'stretch', marginTop: 4 }}>
        {[0, 1].map((i) => <SkeletonRow key={i} delay={i * 120} />)}
      </View>
    </Animated.View>
  );
}

function SkeletonRow({ delay }: { delay: number }) {
  const o = useSharedValue(0.5);
  useEffect(() => {
    o.value = withRepeat(withSequence(withTiming(1, { duration: 650 + delay }), withTiming(0.5, { duration: 650 })), -1);
  }, []);
  const s = useAnimatedStyle(() => ({ opacity: o.value }));
  return (
    <Animated.View style={[styles.skRow, s]}>
      <View style={styles.skAvatar} />
      <View style={{ flex: 1, gap: 6 }}>
        <View style={[styles.skLine, { width: '55%' }]} />
        <View style={[styles.skLine, { width: '80%' }]} />
      </View>
    </Animated.View>
  );
}

function strength(c: number) {
  if (c >= 80) return { label: 'Strong match', fg: theme.colors.semantic.successText, bg: theme.colors.semantic.successBg };
  if (c >= 50) return { label: 'Likely match', fg: toneFg('#b45309'), bg: toneBg('#fffbeb') };
  return { label: 'Possible match', fg: theme.colors.text.muted, bg: theme.colors.bg.surfaceHover };
}

const MATCH_LABEL: Record<string, string> = {
  name: 'Name', 'partial name': 'Part of name', company: 'Company', 'email domain': 'Email domain', 'job title': 'Job title',
};

function CandidateRow({ c, index, busy, disabled, onPick }: {
  c: LinkedInCandidate; index: number; busy: boolean; disabled: boolean; onPick: () => void;
}) {
  const s = strength(c.confidence);
  const parts = c.name.split(/\s+/);
  const [photoFailed, setPhotoFailed] = useState(false);
  return (
    <Animated.View entering={FadeInDown.delay(index * 80).duration(280)} layout={LinearTransition.duration(220)} style={[styles.cand, index === 0 && c.confidence >= 80 && styles.candTop]}>
      <View style={styles.candHead}>
        {c.photoUrl && !photoFailed ? (
          <Animated.View entering={ZoomIn.delay(120 + index * 80).duration(260)}>
            <Image source={{ uri: c.photoUrl }} style={styles.candPhoto} accessibilityIgnoresInvertColors onError={() => setPhotoFailed(true)} />
          </Animated.View>
        ) : (
          <View style={[styles.candPhoto, styles.candInitials]}><Text style={styles.candInitialsText}>{initialsFor(parts[0], parts.slice(1).join(' '))}</Text></View>
        )}
        <View style={{ flex: 1, minWidth: 0 }}>
          <Text style={styles.candName} numberOfLines={1}>{c.name}</Text>
          {c.headline ? <Text style={styles.candHeadline} numberOfLines={2}>{c.headline}</Text> : null}
        </View>
        <Pressable onPress={() => Linking.openURL(c.url)} hitSlop={8} style={styles.openBtn} accessibilityRole="link" accessibilityLabel={`Open ${c.name}'s LinkedIn profile`}>
          <ExternalLink size={15} color={LI_BLUE} />
        </Pressable>
      </View>

      <View style={styles.chips}>
        <View style={[styles.strength, { backgroundColor: s.bg }]}><Text style={[styles.strengthText, { color: s.fg }]}>{s.label}</Text></View>
        {c.matched.map((m) => (
          <View key={m} style={styles.chip}>
            <Check size={11} color={theme.colors.semantic.successText} strokeWidth={3} />
            <Text style={styles.chipText}>{MATCH_LABEL[m] || m}</Text>
          </View>
        ))}
      </View>

      <PressableScale onPress={onPick} disabled={disabled} scaleTo={0.96} style={[styles.pickBtn, disabled && !busy && { opacity: 0.5 }]}
        leadingIcon={busy ? <ActivityIndicator size="small" color="white" /> : <Check size={15} color="white" />} accessibilityRole="button"
        accessibilityLabel={`${c.name} is the right person. Save this LinkedIn profile`}>
        <Text style={styles.pickText}>{busy ? 'Saving…' : 'This is them'}</Text>
      </PressableScale>
    </Animated.View>
  );
}

/**
 * Quick-action tile (next to Call / Email / Ask AI / Edit) for LinkedIn.
 *  - Not linked: "Find" — LinkedIn mark with a magnifier badge that bobs, and a sheen that sweeps
 *    across the icon every few seconds to invite a tap. Starts the search in the LinkedIn card.
 *  - Searching: the badge turns into a spinner.
 *  - Linked: "LinkedIn" with a green tick badge; opens the profile.
 */
export function LinkedInFindTile({ linked, searching, onPress }: { linked: boolean; searching: boolean; onPress: () => void }) {
  const sheen = useSharedValue(0);
  const bob = useSharedValue(1);
  useEffect(() => {
    if (linked) { sheen.value = 0; bob.value = withTiming(1); return; }
    sheen.value = withRepeat(withSequence(withTiming(0, { duration: 1800 }), withTiming(1, { duration: 700, easing: Easing.inOut(Easing.quad) })), -1);
    bob.value = withRepeat(withSequence(withTiming(1, { duration: 1700 }), withTiming(1.22, { duration: 160 }), withTiming(0.94, { duration: 140 }), withTiming(1, { duration: 160 })), -1);
  }, [linked]);
  const sheenStyle = useAnimatedStyle(() => ({ opacity: linked ? 0 : 0.55, transform: [{ translateX: -40 + sheen.value * 80 }, { rotate: '20deg' }] }));
  const badgeStyle = useAnimatedStyle(() => ({ transform: [{ scale: bob.value }] }));
  const label = linked ? 'LinkedIn' : 'Find';
  return (
    <PressableScale containerStyle={{ flex: 1 }} style={[styles.tile, linked && styles.tileLinked]} onPress={onPress} scaleTo={0.92}
      accessibilityRole="button" accessibilityLabel={linked ? 'Open LinkedIn profile' : 'Find on LinkedIn'}>
      <View>
        <View style={styles.tileIcon}>
          <Linkedin size={17} color="white" />
          <Animated.View pointerEvents="none" style={[styles.tileSheen, sheenStyle]} />
        </View>
        <Animated.View style={[styles.tileBadge, linked && styles.tileBadgeLinked, badgeStyle]}>
          {searching ? <View style={{ transform: [{ scale: 0.55 }] }}><ActivityIndicator size="small" color={LI_BLUE} /></View>
            : linked ? <Check size={10} color="white" strokeWidth={3.5} />
            : <Search size={10} color={LI_BLUE} strokeWidth={3} />}
        </Animated.View>
      </View>
      <Text style={styles.tileLabel} numberOfLines={1}>{searching ? 'Finding…' : label}</Text>
    </PressableScale>
  );
}

/**
 * "Find on LinkedIn" for one contact: searches the web with their name, company, job title and
 * email domain, lists the real LinkedIn profiles it found (best match first, with what matched),
 * and saves the one you pick — the link, plus the profile photo when LinkedIn shares it.
 */
export function LinkedInFinderCard({ contact, entering, onUpdated, searchSignal = 0, onSearchingChange }: {
  contact: Contact;
  entering?: any;
  onUpdated: (c: Contact) => void;
  /** Bump to start a search from outside (the "Find" quick-action tile). */
  searchSignal?: number;
  onSearchingChange?: (searching: boolean) => void;
}) {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [picking, setPicking] = useState<string | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [usePhoto, setUsePhoto] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const linked = contact.linkedin && contact.linkedin !== 'N/A' ? contact.linkedin : null;
  const first = contact.firstName || 'this person';

  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => setDone(null), 4000);
    return () => clearTimeout(t);
  }, [done]);

  useEffect(() => { onSearchingChange?.(phase.kind === 'searching'); }, [phase.kind]);
  useEffect(() => { if (searchSignal > 0 && phase.kind !== 'searching') search(); }, [searchSignal]);

  // Linked already: fetch (or refresh) the profile photo without searching again.
  const refreshPhoto = async () => {
    if (!linked || refreshing) return;
    setRefreshing(true);
    setPickError(null);
    try {
      const res = await contactsFeature.attachLinkedIn(contact.id, { url: linked, usePhoto: true });
      onUpdated(res.contact);
      if (res.photoAdded) setDone(res.photoSource === 'gravatar' ? 'Photo added from Gravatar' : 'LinkedIn photo added');
      else if (res.photoStatus === 'limited') setPickError('LinkedIn is limiting photo lookups right now. Please try again in about 15 minutes.');
      else setPickError("This profile doesn't share a public photo.");
    } catch (e: any) {
      setPickError(errorMessage(e, SEARCH_FAILED, LINKEDIN_ERRORS));
    } finally {
      setRefreshing(false);
    }
  };

  const search = async () => {
    setPickError(null);
    setDone(null);
    setPhase({ kind: 'searching' });
    try {
      const result = await api.findLinkedIn(contact.id);
      setPhase({ kind: 'results', result });
    } catch (e: any) {
      setPhase({ kind: 'error', message: errorMessage(e, SEARCH_FAILED, LINKEDIN_ERRORS) });
    }
  };

  const pick = async (c: LinkedInCandidate) => {
    setPicking(c.url);
    setPickError(null);
    try {
      // usePhoto even without a photo from the search: the server looks it up again when saving.
      const res = await contactsFeature.attachLinkedIn(contact.id, { url: c.url, photoUrl: c.photoUrl, usePhoto: contact.photoUrl ? usePhoto : true });
      onUpdated(res.contact);
      setPhase({ kind: 'idle' });
      setDone(
        res.photoAdded && res.photoSource === 'gravatar' ? 'LinkedIn profile saved, with their photo from Gravatar'
          : res.photoAdded ? 'LinkedIn profile and photo saved'
          : res.photoStatus === 'limited' ? 'LinkedIn profile saved. LinkedIn is limiting photos right now: tap "Use LinkedIn photo" later.'
          : 'LinkedIn profile saved (no public photo).',
      );
    } catch (e: any) {
      setPickError(errorMessage(e, SEARCH_FAILED, LINKEDIN_ERRORS));
    } finally {
      setPicking(null);
    }
  };

  const results = phase.kind === 'results' ? phase.result : null;

  return (
    <Animated.View entering={entering} layout={LinearTransition.duration(240)} style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={[styles.cardIcon, { backgroundColor: toneBg('#e8f1fb') }]}><Linkedin size={16} color={LI_BLUE} /></View>
        <Text style={styles.cardTitle}>LINKEDIN</Text>
        {linked && phase.kind === 'idle' ? (
          <Animated.View entering={ZoomIn.duration(240)} style={styles.linkedChip}>
            <CheckCircle2 size={12} color={theme.colors.semantic.successText} />
            <Text style={styles.linkedChipText}>Linked</Text>
          </Animated.View>
        ) : null}
      </View>

      {done ? (
        <Animated.View entering={FadeInDown.duration(260)} exiting={FadeOut.duration(200)} style={styles.doneRow}>
          <Animated.View entering={ZoomIn.delay(80).duration(300)}><CheckCircle2 size={16} color={theme.colors.semantic.successText} /></Animated.View>
          <Text style={styles.doneText}>{done}</Text>
        </Animated.View>
      ) : null}

      {phase.kind === 'idle' ? (
        linked ? (
          <Animated.View entering={FadeIn.duration(220)} style={{ gap: 10 }}>
            <Pressable onPress={() => Linking.openURL(linked)} style={styles.linkedRow} accessibilityRole="link" accessibilityLabel="Open LinkedIn profile">
              <Linkedin size={16} color={LI_BLUE} />
              <Text style={styles.linkedUrl} numberOfLines={1}>{linked.replace(/^https?:\/\/(www\.)?/, '')}</Text>
              <ExternalLink size={15} color={LI_BLUE} />
            </Pressable>
            <View style={styles.rowBtns}>
              <PressableScale onPress={refreshPhoto} disabled={refreshing} scaleTo={0.96} style={styles.secondaryBtn}
                leadingIcon={refreshing ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <ImageIcon size={14} color={theme.colors.brand[700]} />}
                accessibilityRole="button">
                <Text style={styles.secondaryText}>{refreshing ? 'Getting photo…' : 'Use LinkedIn photo'}</Text>
              </PressableScale>
              <Pressable onPress={search} hitSlop={8} style={styles.againBtn} accessibilityRole="button">
                <Search size={14} color={theme.colors.brand[600]} />
                <Text style={styles.linkText}>Search again</Text>
              </Pressable>
            </View>
            {pickError ? <Text style={styles.errText}>{pickError}</Text> : null}
          </Animated.View>
        ) : (
          <Animated.View entering={FadeIn.duration(220)}>
            <Text style={styles.muted}>Find {first} on LinkedIn using their name, company and job title. You choose the right profile before anything is saved.</Text>
            <PressableScale onPress={search} scaleTo={0.96} style={styles.findBtn} leadingIcon={<UserSearch size={16} color="white" />} accessibilityRole="button">
              <Text style={styles.findText}>Find on LinkedIn</Text>
            </PressableScale>
          </Animated.View>
        )
      ) : null}

      {phase.kind === 'searching' ? <Searching /> : null}

      {phase.kind === 'error' ? (
        <Animated.View entering={FadeIn.duration(200)} style={{ gap: 10 }}>
          <View style={styles.errRow}>
            <AlertCircle size={16} color={theme.colors.semantic.danger} />
            <Text style={styles.errText}>{phase.message}</Text>
          </View>
          <View style={styles.rowBtns}>
            <PressableScale onPress={search} scaleTo={0.96} style={styles.secondaryBtn} leadingIcon={<RotateCw size={14} color={theme.colors.brand[700]} />} accessibilityRole="button">
              <Text style={styles.secondaryText}>Try again</Text>
            </PressableScale>
            <Pressable onPress={() => setPhase({ kind: 'idle' })} hitSlop={8}><Text style={styles.cancelText}>Cancel</Text></Pressable>
          </View>
        </Animated.View>
      ) : null}

      {results ? (
        <Animated.View entering={FadeIn.duration(220)} style={{ gap: 10 }}>
          <Text style={styles.searchedFor} numberOfLines={2}>
            Searched for {[results.searchedFor.name, results.searchedFor.title, results.searchedFor.company].filter(Boolean).join(' · ')}
          </Text>

          {/* Their own photo (Gravatar, from their email): compare it with the profiles below */}
          {results.contactPhoto && results.candidates.length ? (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.refRow}>
              <Animated.View entering={ZoomIn.delay(120).duration(280)}>
                <Image source={{ uri: results.contactPhoto.url }} style={styles.refPhoto} accessibilityIgnoresInvertColors />
              </Animated.View>
              <View style={{ flex: 1, minWidth: 0 }}>
                <Text style={styles.refTitle}>{first}'s photo</Text>
                <Text style={styles.refSub}>From their email (Gravatar). Compare it with the profiles below.</Text>
              </View>
            </Animated.View>
          ) : null}

          {results.photosLimited && results.candidates.some((c) => !c.photoUrl) ? (
            <Animated.View entering={FadeInDown.duration(240)} style={styles.limitNote}>
              <AlertCircle size={14} color={toneFg('#b45309')} />
              <Text style={styles.limitText}>
                LinkedIn isn't sharing photos right now, so some show initials. Tap the link icon to check a profile before saving.
              </Text>
            </Animated.View>
          ) : null}

          {results.candidates.length && contact.photoUrl ? (
            <Pressable onPress={() => setUsePhoto((v) => !v)} style={styles.photoToggle} accessibilityRole="checkbox" accessibilityState={{ checked: usePhoto }}>
              <View style={[styles.checkbox, usePhoto && styles.checkboxOn]}>{usePhoto ? <Check size={12} color="white" strokeWidth={3} /> : null}</View>
              <Text style={styles.photoToggleText}>Replace the current photo with their LinkedIn photo</Text>
            </Pressable>
          ) : null}

          {results.candidates.length ? (
            results.candidates.map((c, i) => (
              <CandidateRow key={c.url} c={c} index={i} busy={picking === c.url} disabled={!!picking} onPick={() => pick(c)} />
            ))
          ) : (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.empty}>
              <UserSearch size={22} color={theme.colors.text.muted} />
              <Text style={styles.emptyTitle}>No LinkedIn profile found</Text>
              <Text style={styles.emptyText}>Try adding their company or job title, or search LinkedIn yourself.</Text>
            </Animated.View>
          )}

          {pickError ? <Text style={styles.errText}>{pickError}</Text> : null}

          <View style={styles.rowBtns}>
            <PressableScale onPress={() => Linking.openURL(results.linkedinSearchUrl)} scaleTo={0.96} style={styles.secondaryBtn}
              leadingIcon={<Search size={14} color={theme.colors.brand[700]} />} accessibilityRole="link">
              <Text style={styles.secondaryText}>{results.candidates.length ? 'Not here? Search LinkedIn' : 'Search on LinkedIn'}</Text>
            </PressableScale>
            <Pressable onPress={() => setPhase({ kind: 'idle' })} hitSlop={8} disabled={!!picking}><Text style={styles.cancelText}>Close</Text></Pressable>
          </View>
        </Animated.View>
      ) : null}
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  card: { backgroundColor: theme.colors.bg.surface, borderRadius: 20, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  cardIcon: { width: 30, height: 30, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  cardTitle: { flex: 1, fontSize: 11, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  linkText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  muted: { fontSize: 14, color: theme.colors.text.muted, lineHeight: 20 },

  linkedChip: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.semantic.successBg },
  linkedChipText: { fontSize: 11, fontWeight: '800', color: theme.colors.semantic.successText },
  linkedRow: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 12, borderRadius: 14, backgroundColor: toneBg('#f3f8fd'), borderWidth: 1, borderColor: toneBg('#d6e6f7') },
  linkedUrl: { flex: 1, fontSize: 14, fontWeight: '700', color: toneFg(LI_BLUE) },
  againBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start' },

  findBtn: { flexDirection: 'row', alignItems: 'center', gap: 8, alignSelf: 'flex-start', backgroundColor: LI_BLUE, paddingHorizontal: 16, paddingVertical: 10, borderRadius: 999, marginTop: 12 },
  findText: { color: 'white', fontSize: 14, fontWeight: '800' },

  searching: { alignItems: 'center', gap: 10, paddingVertical: 6 },
  pulseWrap: { width: 46, height: 46, alignItems: 'center', justifyContent: 'center' },
  pulseRing: { position: 'absolute', width: 46, height: 46, borderRadius: 23, backgroundColor: LI_BLUE },
  pulseCore: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: LI_BLUE },
  searchingText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.body },
  skRow: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10, borderRadius: 14, backgroundColor: theme.colors.bg.surfaceHover },
  skAvatar: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.bg.divider },
  skLine: { height: 10, borderRadius: 5, backgroundColor: theme.colors.bg.divider },

  searchedFor: { fontSize: 12, color: theme.colors.text.muted },
  refRow: { flexDirection: 'row', alignItems: 'center', gap: 10, padding: 10, borderRadius: 14, backgroundColor: theme.colors.semantic.successBg },
  refPhoto: { width: 44, height: 44, borderRadius: 22, borderWidth: 2, borderColor: 'white' },
  refTitle: { fontSize: 13, fontWeight: '800', color: theme.colors.semantic.successText },
  refSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 1, lineHeight: 16 },
  limitNote: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, padding: 10, borderRadius: 12, backgroundColor: toneBg('#fffbeb') },
  limitText: { flex: 1, fontSize: 12, lineHeight: 17, color: toneFg('#92400e') },
  photoToggle: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  checkbox: { width: 20, height: 20, borderRadius: 6, borderWidth: 1.5, borderColor: theme.colors.bg.divider, alignItems: 'center', justifyContent: 'center' },
  checkboxOn: { backgroundColor: LI_BLUE, borderColor: LI_BLUE },
  photoToggleText: { flex: 1, fontSize: 13, color: theme.colors.text.body },

  cand: { gap: 10, padding: 12, borderRadius: 16, borderWidth: 1, borderColor: theme.colors.bg.divider, backgroundColor: theme.colors.bg.surface },
  candTop: { borderColor: toneBg('#b7d3f0'), backgroundColor: toneBg('#f7fbff') },
  candHead: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  candPhoto: { width: 44, height: 44, borderRadius: 22, backgroundColor: theme.colors.brand[50] },
  candInitials: { alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#e8f1fb') },
  candInitialsText: { fontSize: 15, fontWeight: '800', color: toneFg(LI_BLUE) },
  candName: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  candHeadline: { fontSize: 12, color: theme.colors.text.body, marginTop: 2, lineHeight: 16 },
  openBtn: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center', backgroundColor: toneBg('#e8f1fb') },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  strength: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999 },
  strengthText: { fontSize: 11, fontWeight: '800' },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 7, paddingVertical: 3, borderRadius: 999, backgroundColor: theme.colors.bg.surfaceHover },
  chipText: { fontSize: 11, fontWeight: '700', color: theme.colors.text.body },
  pickBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, backgroundColor: LI_BLUE, paddingVertical: 9, borderRadius: 12 },
  pickText: { color: 'white', fontSize: 13, fontWeight: '800' },

  empty: { alignItems: 'center', gap: 4, paddingVertical: 12 },
  emptyTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  emptyText: { fontSize: 13, color: theme.colors.text.muted, textAlign: 'center', lineHeight: 18 },

  rowBtns: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  secondaryBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 999, backgroundColor: theme.colors.brand[50] },
  secondaryText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },
  cancelText: { fontSize: 13, fontWeight: '700', color: theme.colors.text.muted },

  errRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  errText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', lineHeight: 18 },
  // Quick-action tile (same footprint as Call / Email / Ask AI / Edit, LinkedIn-tinted)
  tile: { alignItems: 'center', gap: 6, borderRadius: 16, paddingVertical: 12, borderWidth: 1, backgroundColor: toneBg('#f3f8fd'), borderColor: toneBg('#cfe2f6') },
  tileLinked: { backgroundColor: theme.colors.bg.surface, borderColor: theme.colors.bg.divider },
  tileIcon: { width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: LI_BLUE, overflow: 'hidden' },
  tileSheen: { position: 'absolute', top: -10, bottom: -10, width: 12, backgroundColor: 'rgba(255,255,255,0.9)' },
  tileBadge: { position: 'absolute', right: -5, bottom: -5, width: 18, height: 18, borderRadius: 9, alignItems: 'center', justifyContent: 'center', backgroundColor: 'white', borderWidth: 1.5, borderColor: LI_BLUE },
  tileBadgeLinked: { backgroundColor: theme.colors.semantic.successText, borderColor: 'white' },
  tileLabel: { fontSize: 12, fontWeight: '700', color: toneFg(LI_BLUE) },

  doneRow: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 10, borderRadius: 12, backgroundColor: theme.colors.semantic.successBg, marginBottom: 10 },
  doneText: { flex: 1, fontSize: 13, fontWeight: '700', color: theme.colors.semantic.successText },
}));
