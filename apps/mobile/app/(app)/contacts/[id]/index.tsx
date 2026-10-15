import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, Pressable, ActivityIndicator, Linking, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useReturnTo } from '../../../../src/features/contacts/open';
import Animated, {
  FadeIn, FadeInDown, FadeOut, LinearTransition, ZoomIn,
  useSharedValue, useAnimatedStyle, withDelay, withTiming, Easing,
  useAnimatedScrollHandler, interpolate, Extrapolation,
} from 'react-native-reanimated';
import { ArrowLeft, MoreVertical, Pencil, Trash2, Phone, Mail, Sparkles, Briefcase, Building2, AtSign, Smartphone, CalendarDays, TrendingUp, Copy, Check, ChevronRight, Camera, UserX, ScanLine, WifiOff, RotateCw, Linkedin, Share2, QrCode } from 'lucide-react-native';
import type { Contact } from '@es/types';
import { contactsFeature, describeReasons } from '../../../../src/features/contacts';
import { Text } from '../../../../src/ui/Text';
import { theme, toneBg, makeStyles } from '../../../../src/ui/theme';
import { AskHalfSheet } from '../../../../src/features/contacts/components/AskHalfSheet';
import { EmailVerifyRow } from '../../../../src/features/contacts/components/EmailVerify';
import { ContactAvatar } from '../../../../src/features/contacts/components/ContactAvatar';
import { LinkedInFinderCard, LinkedInFindTile } from '../../../../src/features/contacts/components/LinkedInFinderCard';
import { OptionSheet } from '../../../../src/features/contacts/components/OptionSheet';
import { usePhotoEditor } from '../../../../src/features/contacts/components/usePhotoEditor';
import { NotesCard } from '../../../../src/features/contacts/components/NotesCard';
import { useEditConflict } from '../../../../src/features/contacts/components/ConflictSheet';
import { QrDialog, type QrDialogData } from '../../../../src/features/contacts/components/QrDialog';
import { PressableScale } from '../../../../src/features/onboarding/components';
import { scoreLevel } from '../../../../src/features/contacts/scoreLevel';
import { KeyboardAwareView, useRevealAboveKeyboard } from '../../../../src/ui/keyboard';
import { api, errorMessage, type DuplicateMatch } from '../../../../src/lib/api';
import { ContactFilesCard } from '../../../../src/features/files/ContactFilesCard';

const enter = (i: number) => FadeInDown.delay(80 + i * 70).duration(260);

// Same levels and colours as the Scout feed and event page (src/features/contacts/scoreLevel.ts).
function temperatureOf(score: number) {
  const l = scoreLevel(score)!;
  return { label: l.label, bg: l.bg, fg: l.fg, bar: l.ring };
}

// Fills to the real score when it appears or changes.
function ScoreBar({ score, color }: { score: number; color: string }) {
  const width = useSharedValue(0);
  useEffect(() => {
    width.value = 0;
    width.value = withDelay(250, withTiming(score, { duration: 900, easing: Easing.out(Easing.cubic) }));
  }, [score]);
  const style = useAnimatedStyle(() => ({ width: `${width.value}%`, backgroundColor: color }));
  return (
    <View style={styles.scoreTrack}>
      <Animated.View style={[styles.scoreFill, style]} />
    </View>
  );
}

const DIMENSION_LABELS: Record<string, string> = {
  role_fit: 'Role fit', company_fit: 'Company fit', need_fit: 'Needs what you sell', goal_fit: 'Event goals', intent: 'Interest shown',
  // scores made before Sept 30
  profile_fit: 'Profile fit', role_relevance: 'Role', engagement: 'Engagement', timing: 'Timing',
};

/** One 0-20 part of the score, filling in after the total. */
function DimensionBar({ label, value, index }: { label: string; value: number; index: number }) {
  const w = useSharedValue(0);
  const v = Math.max(0, Math.min(20, value));
  useEffect(() => {
    w.value = 0;
    w.value = withDelay(350 + index * 90, withTiming(v / 20, { duration: 700, easing: Easing.out(Easing.cubic) }));
  }, [v]);
  const style = useAnimatedStyle(() => ({ width: `${w.value * 100}%` }));
  const color = v >= 15 ? theme.colors.semantic.success : v >= 10 ? theme.colors.brand[500] : v >= 6 ? theme.colors.semantic.warning : theme.colors.semantic.danger;
  return (
    <Animated.View entering={FadeInDown.delay(300 + index * 70).duration(260)} style={styles.dimRow}>
      <Text style={styles.dimLabel} numberOfLines={1}>{label}</Text>
      <View style={styles.dimTrack}><Animated.View style={[styles.dimFill, { backgroundColor: color }, style]} /></View>
      <Text style={styles.dimValue}>{v}<Text style={styles.dimOf}>/20</Text></Text>
    </Animated.View>
  );
}

function QuickAction({ icon, label, onPress, disabled }: { icon: React.ReactNode; label: string; onPress: () => void; disabled?: boolean }) {
  return (
    <PressableScale
      containerStyle={{ flex: 1 }}
      style={[styles.quickAction, disabled && styles.quickActionDisabled]}
      onPress={onPress}
      disabled={disabled}
      scaleTo={0.92}
      accessibilityRole="button"
      accessibilityLabel={label}
    >
      <View style={[styles.quickIcon, disabled && { backgroundColor: theme.colors.bg.surfaceHover }]}>{icon}</View>
      <Text style={[styles.quickLabel, disabled && { color: theme.colors.text.muted }]}>{label}</Text>
    </PressableScale>
  );
}

function DetailRow({ icon, label, value, onPress, trailing }: {
  icon: React.ReactNode; label: string; value?: string | null; onPress?: () => void; trailing?: React.ReactNode;
}) {
  const content = (
    <View style={styles.detailRow}>
      <View style={styles.detailIcon}>{icon}</View>
      <View style={{ flex: 1 }}>
        <Text style={styles.detailLabel}>{label}</Text>
        <Text style={value ? (onPress ? styles.detailLink : styles.detailValue) : styles.detailEmpty} selectable={!!value}>
          {value || 'Not added'}
        </Text>
      </View>
      {value && trailing ? trailing : null}
    </View>
  );
  if (!value || !onPress) return content;
  return (
    <Pressable onPress={onPress} style={({ pressed }) => pressed && { backgroundColor: theme.colors.bg.surfaceHover }} accessibilityRole="link">
      {content}
    </Pressable>
  );
}

export default function ContactDetailScreen() {
  // `from`: opened from another tab (Scout, an event, Scan, Ask AI) — see src/features/contacts/open.ts
  const { id, from } = useLocalSearchParams<{ id: string; from?: string }>();
  const router = useRouter();
  const [showAskSheet, setShowAskSheet] = useState(false);
  // QR code (GET /api/contact/:id/qr — a real vCard any camera can scan)
  const [qrOpen, setQrOpen] = useState(false);
  const [qrData, setQrData] = useState<QrDialogData | null>(null);
  const [qrLoading, setQrLoading] = useState(false);
  const [qrError, setQrError] = useState<string | null>(null);
  const [contact, setContact] = useState<Contact | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'missing' | 'error'>('loading');
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState('');
  const [scoreReason, setScoreReason] = useState('');
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  const [duplicates, setDuplicates] = useState<DuplicateMatch[]>([]);
  const [dismissing, setDismissing] = useState<string | null>(null);
  const photo = usePhotoEditor(contact, setContact);
  // Notes saved here but changed on another device meanwhile: keep mine / use theirs / keep both.
  const notesConflict = useEditConflict({
    onSaved: (updated) => setContact((prev) => (prev ? { ...prev, ...updated } : updated)),
    onUseTheirs: (server) => { if (server) setContact((prev) => (prev ? { ...prev, ...server } : server)); },
  });
  const scrollRef = useRef<any>(null);
  const notesDirty = useRef(false);
  const [confirmLeave, setConfirmLeave] = useState(false);
  // Back returns to where you opened the contact from (src/features/contacts/open.ts); the
  // Android back button goes through goBack too, so unsaved notes still ask first.
  const leave = useReturnTo(from, () => goBack());
  const goBack = () => (notesDirty.current ? setConfirmLeave(true) : leave());

  // Header reacts to scroll: "Contact" label hands over to the avatar + name past the hero.
  const scrollY = useSharedValue(0);
  const onScroll = useAnimatedScrollHandler((e) => { scrollY.value = e.contentOffset.y; });
  // Keeps the notes editor + its Save row above the keyboard (Android doesn't resize the window).
  const keyboard = useRevealAboveKeyboard(scrollRef, () => scrollY.value);
  const headerBgStyle = useAnimatedStyle(() => ({
    opacity: interpolate(scrollY.value, [60, 140], [0, 1], Extrapolation.CLAMP),
  }));
  const headerLabelStyle = useAnimatedStyle(() => ({
    opacity: interpolate(scrollY.value, [120, 170], [1, 0], Extrapolation.CLAMP),
  }));
  const headerIdentityStyle = useAnimatedStyle(() => ({
    opacity: interpolate(scrollY.value, [150, 200], [0, 1], Extrapolation.CLAMP),
    transform: [{ translateY: interpolate(scrollY.value, [150, 200], [8, 0], Extrapolation.CLAMP) }],
  }));

  const [reloadKey, setReloadKey] = useState(0);

  // LinkedIn: the quick-action "Find" tile starts the search in the LinkedIn card below.
  const [liSignal, setLiSignal] = useState(0);
  const [liSearching, setLiSearching] = useState(false);
  const liCardY = useRef(0);
  const linkedinUrl = contact?.linkedin && contact.linkedin !== 'N/A' ? contact.linkedin : null;

  // Reload whenever the screen gains focus, so edits made on the edit screen show up.
  useFocusEffect(
    useCallback(() => {
      if (!id) return;
      let active = true;
      contactsFeature.getContact(id)
        .then((c) => {
          if (!active) return;
          setContact(c);
          setLoadState(c ? 'ready' : 'missing');
        })
        .catch(() => {
          // Server unreachable and nothing cached — say so instead of "not found".
          // (If it was already loaded, keep showing it.)
          if (active) setLoadState((prev) => (prev === 'ready' ? 'ready' : 'error'));
        });
      contactsFeature.getDuplicates(id).then((d) => { if (active) setDuplicates(d); });
      return () => { active = false; };
    }, [id, reloadKey]),
  );

  // Inline QR card: fetched with the contact, refreshed when its details change.
  const qrKey = contact ? [contact.id, contact.firstName, contact.lastName, contact.company, contact.title, contact.email, contact.phone, contact.linkedin].join('|') : '';
  useEffect(() => {
    if (!contact) return;
    let alive = true;
    api.getContactQR(contact.id)
      .then((res) => {
        if (!alive) return;
        setQrData({
          title: [contact.firstName, contact.lastName].filter(Boolean).join(' ') || res.name || 'Contact',
          subtitle: [contact.title, contact.company].filter(Boolean).join(' · ') || null,
          firstName: contact.firstName, lastName: contact.lastName, photoUrl: contact.photoUrl ?? null,
          vcard: res.vcard, qrDataUrl: res.qrDataUrl,
        });
        setQrError(null);
      })
      .catch((e: any) => alive && setQrError(errorMessage(e, "Couldn't make the QR code.")));
    return () => { alive = false; };
  }, [qrKey]);

  const openQr = async () => {
    if (!contact) return;
    setQrOpen(true);
    if (qrData) return; // already loaded for the inline card
    setQrLoading(true);
    setQrError(null);
    try {
      const res = await api.getContactQR(contact.id);
      setQrData({
        title: [contact.firstName, contact.lastName].filter(Boolean).join(' ') || res.name || 'Contact',
        subtitle: [contact.title, contact.company].filter(Boolean).join(' · ') || null,
        firstName: contact.firstName, lastName: contact.lastName, photoUrl: contact.photoUrl ?? null,
        vcard: res.vcard, qrDataUrl: res.qrDataUrl,
      });
    } catch (e: any) {
      setQrError(errorMessage(e, "Couldn't make the QR code."));
    } finally {
      setQrLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!contact || deleting) return;
    setDeleting(true);
    setDeleteError('');
    try {
      await contactsFeature.deleteContact(contact.id, fullName);
      // To the Contacts list even when opened from elsewhere: it shows "moved to Recently deleted · Undo".
      router.back();
    } catch (e: any) {
      setDeleteError(errorMessage(e, "Couldn't delete this contact. Please try again.", {
        offline: "You're offline — the contact wasn't deleted.",
      }));
    } finally {
      setDeleting(false);
    }
  };

  // Re-scoring: unchanged details keep the saved score (the AI isn't asked again, so the number
  // can't drift); "Score again anyway" forces a fresh 3-run score.
  const [scoreNote, setScoreNote] = useState<{ kind: 'kept' | 'changed'; text: string } | null>(null);
  const handleScore = async (force = false) => {
    if (!contact || scoring) return;
    setScoring(true);
    setScoreError('');
    setScoreNote(null);
    try {
      const res: any = await api.scoreContact(contact.id, force);
      setScoreReason(res?.reasoning || '');
      if (res?.unchanged) {
        setScoreNote({ kind: 'kept', text: `Nothing changed since the last score, so it stays at ${res.score}. Add notes, a title or your goals to change it.` });
      } else if (typeof res?.previousScore === 'number' && res.previousScore !== res.score) {
        const d = res.score - res.previousScore;
        setScoreNote({ kind: 'changed', text: `${d > 0 ? '+' : ''}${d} from ${res.previousScore} — the details it's based on changed.` });
      }
      const fresh = await contactsFeature.getContact(contact.id).catch(() => null);
      if (fresh) setContact(fresh);
    } catch (e: any) {
      const status = e?.response?.status;
      setScoreError(
        status === 503 ? (e?.response?.data?.detail || 'Lead scoring is not available on the server.')
          : status === 504 ? 'Scoring timed out. Please try again.'
          : errorMessage(e, 'Could not score this contact. Please try again.', { offline: "You're offline. Try again when connected." }),
      );
    } finally {
      setScoring(false);
    }
  };

  const notDuplicate = async (other: Contact) => {
    if (!contact) return;
    setDismissing(other.id);
    try {
      await api.dismissDuplicate(contact.id, other.id);
      setDuplicates((prev) => prev.filter((d) => d.contact.id !== other.id));
    } catch {
      /* leave the banner; the user can retry */
    } finally {
      setDismissing(null);
    }
  };

  if (!contact) {
    return (
      <SafeAreaView style={[styles.safeArea, styles.centered]}>
        {loadState === 'missing' ? (
          <Animated.View entering={FadeIn} style={styles.centered}>
            <View style={styles.missingIcon}><UserX size={28} color={theme.colors.text.muted} /></View>
            <Text style={styles.missingTitle}>Contact not found</Text>
            <Text style={styles.missingText}>It may have been deleted. Check Recently deleted to restore it.</Text>
            <PressableScale style={styles.missingBtn} onPress={() => router.replace({ pathname: '/contacts/deleted', params: from ? { from } : {} } as any)} accessibilityRole="button">
              <Text style={styles.missingBtnText}>Open Recently deleted</Text>
            </PressableScale>
            <Pressable onPress={leave} hitSlop={8}><Text style={styles.linkText}>{from ? 'Go back' : 'Back to contacts'}</Text></Pressable>
          </Animated.View>
        ) : loadState === 'error' ? (
          <Animated.View entering={FadeIn} style={styles.centered}>
            <View style={styles.missingIcon}><WifiOff size={28} color={theme.colors.text.muted} /></View>
            <Text style={styles.missingTitle}>Couldn't load this contact</Text>
            <Text style={styles.missingText}>The server can't be reached. Check your connection and try again.</Text>
            <PressableScale
              style={styles.missingBtn}
              onPress={() => { setLoadState('loading'); setReloadKey((k) => k + 1); }}
              leadingIcon={<RotateCw size={15} color="white" />}
              accessibilityRole="button"
            >
              <Text style={styles.missingBtnText}>Try again</Text>
            </PressableScale>
            <Pressable onPress={leave} hitSlop={8}><Text style={styles.linkText}>{from ? 'Go back' : 'Back to contacts'}</Text></Pressable>
          </Animated.View>
        ) : (
          <>
            <ActivityIndicator color={theme.colors.brand[600]} />
            <Text style={styles.missingText}>Loading contact…</Text>
          </>
        )}
      </SafeAreaView>
    );
  }

  const fullName = [contact.firstName, contact.lastName].filter(Boolean).join(' ') || 'Unnamed contact';
  const score = contact.score; // null = never scored; never show a made-up number
  const temp = score == null ? null : temperatureOf(score);
  const phoneDigits = (contact.phone || '').replace(/[^\d+]/g, '');

  return (
    <SafeAreaView style={styles.safeArea} edges={['top']}>
      {/* Floating header: fills in and shows the contact once the hero scrolls away. */}
      <View style={styles.header}>
        <Animated.View style={[styles.headerBg, headerBgStyle]} pointerEvents="none" />
        <PressableScale style={styles.iconBtn} onPress={goBack} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Back to contacts">
          <ArrowLeft size={20} color={theme.colors.text.display} />
        </PressableScale>
        <View style={styles.headerCenter}>
          <Animated.Text style={[styles.headerTitle, headerLabelStyle]}>Contact</Animated.Text>
          <Animated.View style={[styles.headerIdentity, headerIdentityStyle]} pointerEvents="none">
            <ContactAvatar firstName={contact.firstName} lastName={contact.lastName} photoUrl={contact.photoUrl} size={28} />
            <Text style={styles.headerName} numberOfLines={1}>{fullName}</Text>
          </Animated.View>
        </View>
        <PressableScale style={styles.iconBtn} onPress={openQr} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Show QR code">
          <QrCode size={19} color={theme.colors.brand[700]} />
        </PressableScale>
        <PressableScale style={styles.iconBtn} onPress={() => router.push(`/contacts/${contact.id}/share` as any)} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="Share contact">
          <Share2 size={19} color={theme.colors.brand[700]} />
        </PressableScale>
        <PressableScale style={styles.iconBtn} onPress={() => setMenuOpen(true)} scaleTo={0.9} accessibilityRole="button" accessibilityLabel="More actions">
          <MoreVertical size={20} color={theme.colors.text.display} />
        </PressableScale>
      </View>

      <KeyboardAwareView>
      <Animated.ScrollView
        ref={scrollRef}
        keyboardShouldPersistTaps="handled"
        onScroll={onScroll}
        scrollEventThrottle={16}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero */}
        <Animated.View entering={FadeInDown.duration(260)} style={styles.hero}>
          <View style={styles.heroGlow} />
          <Animated.View entering={ZoomIn.delay(60).duration(260)}>
            <ContactAvatar
              firstName={contact.firstName}
              lastName={contact.lastName}
              photoUrl={contact.photoUrl}
              size={92}
              editable
              busy={photo.busy}
              onPress={photo.open}
            />
          </Animated.View>
          <Text style={styles.heroName}>{fullName}</Text>
          {[contact.title, contact.company].some(Boolean) ? (
            <Text style={styles.heroSub}>{[contact.title, contact.company].filter(Boolean).join(' · ')}</Text>
          ) : (
            <Text style={styles.heroMuted}>No title or company yet</Text>
          )}
          <View style={styles.heroChips}>
            {contact.eventId ? (
              <Pressable onPress={() => router.push(`/events/${encodeURIComponent(contact.eventId)}` as any)} style={styles.chip} hitSlop={6} accessibilityLabel={`Open event ${contact.eventId}`}>
                <CalendarDays size={12} color={theme.colors.brand[700]} />
                <Text style={styles.chipText}>{contact.eventId}</Text>
                <ChevronRight size={12} color={theme.colors.brand[700]} />
              </Pressable>
            ) : null}
            <View style={styles.chip}>
              {(() => {
                // Backend source values: scan (card) · qr (QR / badge) · manual · import; anything else came from the admin side.
                const src = String(contact.source || 'manual').toLowerCase();
                const Icon = src === 'qr' ? QrCode : src === 'scan' ? ScanLine : Pencil;
                const label = src === 'scan' ? 'Card scan' : src === 'qr' ? 'QR scan' : src === 'manual' ? 'Added manually'
                  : src.startsWith('admin') ? 'Added by admin' : 'Imported';
                return (
                  <>
                    <Icon size={12} color={theme.colors.brand[700]} />
                    <Text style={styles.chipText}>{label}</Text>
                  </>
                );
              })()}
            </View>
            {temp ? (
              <Animated.View entering={ZoomIn.delay(200)} style={[styles.chip, { backgroundColor: temp.bg }]}>
                <TrendingUp size={12} color={temp.fg} />
                <Text style={[styles.chipText, { color: temp.fg }]}>{temp.label}</Text>
              </Animated.View>
            ) : null}
          </View>
        </Animated.View>
        {photo.error ? <Text style={styles.inlineError}>{photo.error}</Text> : null}
        {deleteError ? <Text style={styles.inlineError}>{deleteError}</Text> : null}

        {/* Quick actions */}
        <Animated.View entering={enter(1)} style={styles.quickRow}>
          <QuickAction icon={<Phone size={18} color={theme.colors.brand[700]} />} label="Call" disabled={!phoneDigits}
            onPress={() => Linking.openURL(`tel:${phoneDigits}`)} />
          <QuickAction icon={<Mail size={18} color={theme.colors.brand[700]} />} label="Email" disabled={!contact.email}
            onPress={() => Linking.openURL(`mailto:${contact.email}`)} />
          <QuickAction icon={<Sparkles size={18} color={theme.colors.brand[700]} />} label="Ask AI" onPress={() => setShowAskSheet(true)} />
          <QuickAction icon={<Pencil size={18} color={theme.colors.brand[700]} />} label="Edit" onPress={() => router.push(`/contacts/${contact.id}/edit`)} />
          <LinkedInFindTile linked={!!linkedinUrl} searching={liSearching} onPress={() => {
            if (linkedinUrl) { Linking.openURL(linkedinUrl); return; }
            // Bring the LinkedIn card into view, then start its search.
            scrollRef.current?.scrollTo?.({ y: Math.max(liCardY.current - 12, 0), animated: true });
            setLiSignal((n) => n + 1);
          }} />
        </Animated.View>

        {/* Possible duplicates — ask the user to check */}
        {duplicates.map((d) => {
          const otherName = [d.contact.firstName, d.contact.lastName].filter(Boolean).join(' ') || 'Unnamed contact';
          return (
            <Animated.View
              key={d.contact.id}
              entering={FadeInDown.duration(260)}
              exiting={FadeOut.duration(180)}
              layout={LinearTransition.duration(260)}
              style={styles.dupCard}
            >
              <View style={styles.dupHeader}>
                <View style={styles.dupIcon}><Copy size={15} color={theme.colors.semantic.warning} /></View>
                <View style={{ flex: 1 }}>
                  <Text style={styles.dupTitle}>Possible duplicate</Text>
                  <Text style={styles.dupSub}>
                    Looks like <Text style={styles.bold}>{otherName}</Text> ({describeReasons(d.reasons)}). Please check.
                  </Text>
                </View>
              </View>
              <View style={styles.dupActions}>
                <PressableScale style={styles.dupBtn} onPress={() => router.push(`/contacts/${d.contact.id}`)} scaleTo={0.94}
                  trailingIcon={<ChevronRight size={14} color={theme.colors.text.display} />} accessibilityRole="button">
                  <Text style={styles.dupBtnText}>View {d.contact.firstName || 'it'}</Text>
                </PressableScale>
                <PressableScale style={styles.dupBtn} onPress={() => notDuplicate(d.contact)} disabled={dismissing === d.contact.id} scaleTo={0.94}
                  leadingIcon={dismissing === d.contact.id ? <ActivityIndicator size="small" /> : <Check size={14} color={theme.colors.text.display} />} accessibilityRole="button">
                  <Text style={styles.dupBtnText}>Different people</Text>
                </PressableScale>
                <PressableScale style={[styles.dupBtn, styles.dupBtnDanger]} onPress={() => setConfirmDelete(true)} scaleTo={0.94}
                  leadingIcon={<Trash2 size={14} color={theme.colors.semantic.danger} />} accessibilityRole="button">
                  <Text style={[styles.dupBtnText, { color: theme.colors.semantic.danger }]}>Delete this one</Text>
                </PressableScale>
              </View>
            </Animated.View>
          );
        })}

        {/* Lead score */}
        <Animated.View entering={enter(2)} style={styles.card}>
          <View style={styles.cardHeader}>
            <View style={styles.cardIcon}><TrendingUp size={16} color={theme.colors.brand[700]} /></View>
            <Text style={styles.cardTitle}>LEAD SCORE</Text>
            <Text style={styles.cardMeta}>{contact.scoreBand ? `Band ${contact.scoreBand}` : 'Not scored'}</Text>
          </View>
          {score == null ? (
            <>
              <Text style={styles.notScored}>Not scored yet</Text>
              <Text style={styles.muted}>Score this lead against your targets, products and event goals.</Text>
              <PressableScale
                style={[styles.scoreButton, scoring && { opacity: 0.7 }]}
                onPress={() => handleScore()}
                disabled={scoring}
                leadingIcon={scoring ? <ActivityIndicator size="small" color="white" /> : <Sparkles size={15} color="white" />}
                accessibilityRole="button"
              >
                <Text style={styles.scoreButtonText}>{scoring ? 'Scoring…' : 'Score with AI'}</Text>
              </PressableScale>
            </>
          ) : (
            <>
              <View style={styles.scoreRow}>
                <Animated.Text key={score} entering={ZoomIn.duration(260)} style={styles.scoreValue}>{score}</Animated.Text>
                <Text style={styles.scoreOutOf}>/100</Text>
                <Animated.View key={`t-${temp!.label}`} entering={ZoomIn.delay(150).duration(240)} style={[styles.tempChip, { backgroundColor: temp!.bg }]}>
                  <Text style={[styles.tempChipText, { color: temp!.fg }]}>{temp!.label}</Text>
                </Animated.View>
                <View style={{ flex: 1 }} />
                <Pressable onPress={() => handleScore()} disabled={scoring} hitSlop={8} accessibilityRole="button">
                  {scoring ? <ActivityIndicator size="small" color={theme.colors.brand[600]} /> : <Text style={styles.linkText}>Re-score</Text>}
                </Pressable>
              </View>
              <ScoreBar score={score} color={temp!.bar} />
              {scoreNote ? (
                <Animated.View entering={FadeInDown.duration(260)} style={[styles.scoreNote, scoreNote.kind === 'changed' && { backgroundColor: theme.colors.brand[50] }]}>
                  <Text style={styles.scoreNoteText}>{scoreNote.text}</Text>
                  {scoreNote.kind === 'kept' ? (
                    <Pressable onPress={() => handleScore(true)} disabled={scoring} hitSlop={8} accessibilityRole="button">
                      <Text style={[styles.linkText, { fontSize: 12, marginTop: 4 }]}>Score again anyway</Text>
                    </Pressable>
                  ) : null}
                </Animated.View>
              ) : null}
              {/* How much to trust it: answers combined + how far apart they were */}
              {(() => {
                const m = (contact as any).scoreMeta as { runs?: number; spread?: number; model?: string; scoredAt?: string } | null;
                if (!m || !m.runs) return null;
                const spread = m.spread ?? 0;
                const level = m.runs < 2 ? null : spread <= 6 ? { t: 'High agreement', c: theme.colors.semantic.successText } : spread <= 14 ? { t: 'Some variation', c: theme.colors.semantic.warning } : { t: 'Low agreement — add more details', c: theme.colors.semantic.danger };
                const when = m.scoredAt ? new Date(m.scoredAt).toLocaleString([], { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' }) : null;
                return (
                  <Animated.View entering={FadeIn.delay(200).duration(260)} style={styles.trustRow}>
                    {level ? <View style={[styles.trustDot, { backgroundColor: level.c }]} /> : null}
                    <Text style={styles.trustText} numberOfLines={2}>
                      {level ? `${level.t} · ` : ''}{m.runs > 1 ? `middle of ${m.runs} AI answers (±${Math.ceil(spread / 2)})` : '1 AI answer'}
                      {when ? ` · ${when}` : ''}
                    </Text>
                  </Animated.View>
                );
              })()}
              {/* Why this score: the five parts it's made of, the AI's reasons, suggested next steps */}
              {(() => {
                const breakdown = (contact as any).scoreBreakdown as Record<string, number> | null | undefined;
                const reasoning = scoreReason || ((contact as any).scoreReasoning as string | null) || '';
                const actions = ((contact as any).scoreActions as string[] | null | undefined) || [];
                const parts = breakdown ? Object.entries(breakdown).filter(([, n]) => typeof n === 'number') : [];
                if (!parts.length && !reasoning && !actions.length) return null;
                return (
                  <View key={`why-${score}`} style={styles.whyBox}>
                    {parts.length ? <Text style={styles.whyLabel}>WHY THIS SCORE</Text> : null}
                    {parts.map(([k, n], i) => (
                      <DimensionBar key={k} index={i} value={n} label={DIMENSION_LABELS[k] || k.replace(/_/g, ' ')} />
                    ))}
                    {reasoning ? <Animated.Text entering={FadeIn.delay(500).duration(300)} style={styles.scoreReason}>{reasoning}</Animated.Text> : null}
                    {actions.length ? (
                      <>
                        <Text style={[styles.whyLabel, { marginTop: 10 }]}>NEXT STEPS</Text>
                        {actions.map((a, i) => (
                          <Animated.View key={a} entering={FadeInDown.delay(650 + i * 60).duration(260)} style={styles.actionRow}>
                            <View style={styles.actionNum}><Text style={styles.actionNumText}>{i + 1}</Text></View>
                            <Text style={styles.actionText}>{a}</Text>
                          </Animated.View>
                        ))}
                      </>
                    ) : null}
                  </View>
                );
              })()}
            </>
          )}
          {scoreError ? <Text style={styles.inlineError}>{scoreError}</Text> : null}
        </Animated.View>

        {/* Details */}
        <Animated.View entering={enter(3)} style={styles.card}>
          <View style={styles.cardHeader}>
            <View style={styles.cardIcon}><Briefcase size={16} color={theme.colors.brand[700]} /></View>
            <Text style={styles.cardTitle}>DETAILS</Text>
            <Pressable onPress={() => router.push(`/contacts/${contact.id}/edit`)} hitSlop={8}><Text style={styles.linkText}>Edit</Text></Pressable>
          </View>
          <DetailRow icon={<Briefcase size={16} color={theme.colors.text.body} />} label="TITLE" value={contact.title} />
          <View style={styles.divider} />
          <DetailRow icon={<Building2 size={16} color={theme.colors.text.body} />} label="COMPANY" value={contact.company} />
          <View style={styles.divider} />
          <DetailRow icon={<AtSign size={16} color={theme.colors.text.body} />} label="EMAIL" value={contact.email}
            onPress={() => Linking.openURL(`mailto:${contact.email}`)} trailing={<Mail size={16} color={theme.colors.brand[600]} />} />
          {/* Is the address real? (format · domain · mailbox · catch-all · disposable / role) */}
          <EmailVerifyRow email={contact.email} />
          <View style={styles.divider} />
          <DetailRow icon={<Smartphone size={16} color={theme.colors.text.body} />} label="PHONE" value={contact.phone}
            onPress={() => Linking.openURL(`tel:${phoneDigits}`)} trailing={<Phone size={16} color={theme.colors.brand[600]} />} />
          {contact.linkedin ? (
            <>
              <View style={styles.divider} />
              <DetailRow icon={<Linkedin size={16} color={theme.colors.text.body} />} label="LINKEDIN"
                value={contact.linkedin.replace(/^https?:\/\/(www\.)?/, '')}
                onPress={() => Linking.openURL(contact.linkedin!)} trailing={<ChevronRight size={16} color={theme.colors.brand[600]} />} />
            </>
          ) : null}
        </Animated.View>

        {/* Find on LinkedIn: web search by name / company / job title, pick the right person, save
            the profile link + photo */}
        <View onLayout={(e) => { liCardY.current = e.nativeEvent.layout.y; }}>
          <LinkedInFinderCard contact={contact} entering={enter(4)} searchSignal={liSignal} onSearchingChange={setLiSearching}
            onUpdated={(updated) => setContact((prev) => (prev ? { ...prev, ...updated } : updated))} />
        </View>

        {/* Notes — edited right here (used to jump to the edit screen) */}
        <NotesCard
          contactId={contact.id}
          notes={contact.notes}
          eventName={contact.eventId || null}
          entering={enter(4)}
          onSaved={(updated) => setContact((prev) => (prev ? { ...prev, ...updated } : updated))}
          onEditingChange={(_editing, dirty) => { notesDirty.current = dirty; }}
          onFocusEditor={(saveRow) => keyboard.reveal(saveRow as any)}
          conflict={notesConflict}
        />

        {/* Files attached to this contact: attach, open, save, remove */}
        <ContactFilesCard contactId={contact.id} reloadKey={reloadKey} />

        {/* QR code — scan with any phone camera to add this contact */}
        <Animated.View entering={enter(5)} style={styles.card}>
          <View style={styles.cardHeader}>
            <View style={styles.cardIcon}><QrCode size={16} color={theme.colors.brand[700]} /></View>
            <Text style={styles.cardTitle}>QR CODE</Text>
            <Pressable onPress={openQr} hitSlop={8}><Text style={styles.linkText}>Share</Text></Pressable>
          </View>
          <Pressable onPress={openQr} style={styles.qrInline} accessibilityLabel="Open QR code">
            <View style={styles.qrInlineBox}>
              {qrData ? (
                <Animated.View entering={ZoomIn.duration(260)}>
                  <Image source={{ uri: qrData.qrDataUrl }} style={styles.qrInlineImg} resizeMode="contain" />
                </Animated.View>
              ) : qrError ? (
                <QrCode size={28} color={theme.colors.text.muted} />
              ) : (
                <ActivityIndicator color={theme.colors.brand[600]} />
              )}
            </View>
            <View style={{ flex: 1, gap: 6 }}>
              <Text style={styles.qrInlineTitle}>Scan to add {contact.firstName || 'this contact'}</Text>
              <Text style={styles.muted}>
                {qrError ? qrError : 'Any phone camera can read it — name, company, phone and email. Notes stay private.'}
              </Text>
              <Text style={styles.linkText}>Tap to enlarge or share ›</Text>
            </View>
          </Pressable>
        </Animated.View>

        <Text style={styles.footerMeta}>
          Added {contact.createdAt ? new Date(contact.createdAt).toLocaleDateString() : '—'}
          {contact.updatedAt && contact.updatedAt !== contact.createdAt ? ` · Updated ${new Date(contact.updatedAt).toLocaleDateString()}` : ''}
        </Text>
      </Animated.ScrollView>
      </KeyboardAwareView>

      {showAskSheet && (
        <AskHalfSheet
          contactId={contact.id}
          contactName={fullName !== 'Unnamed contact' ? fullName : (contact.company || 'this contact')}
          contactCompany={contact.company}
          contactEmail={contact.email}
          onClose={() => setShowAskSheet(false)}
        />
      )}

      {photo.sheet}

      {notesConflict.sheet}

      <QrDialog visible={qrOpen} loading={qrLoading} error={qrError} data={qrData} onClose={() => setQrOpen(false)} onRetry={openQr} />


      <OptionSheet
        visible={menuOpen}
        title={fullName}
        options={[
          { label: 'Edit contact', icon: <Pencil size={20} color={theme.colors.brand[600]} />, onPress: () => router.push(`/contacts/${contact.id}/edit`) },
          { label: contact.photoUrl ? 'Change photo' : 'Add photo', icon: <Camera size={20} color={theme.colors.brand[600]} />, onPress: photo.open },
          { label: 'Show QR code', icon: <QrCode size={20} color={theme.colors.brand[600]} />, onPress: openQr },
          { label: 'Share contact', icon: <Share2 size={20} color={theme.colors.brand[600]} />, onPress: () => router.push(`/contacts/${contact.id}/share` as any) },
          { label: 'Ask AI about this contact', icon: <Sparkles size={20} color={theme.colors.brand[600]} />, onPress: () => setShowAskSheet(true) },
          { label: 'Delete contact', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: () => setConfirmDelete(true) },
        ]}
        onClose={() => setMenuOpen(false)}
      />

      <OptionSheet
        visible={confirmDelete}
        title={`Delete ${fullName}?`}
        message="It moves to Recently deleted, where you can restore it for 30 days."
        options={[
          { label: deleting ? 'Deleting…' : 'Delete contact', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: handleDelete },
        ]}
        onClose={() => setConfirmDelete(false)}
      />

      <OptionSheet
        visible={confirmLeave}
        title="Discard note changes?"
        message="Your unsaved changes to the notes will be lost."
        options={[
          { label: 'Discard and go back', icon: <Trash2 size={20} color={theme.colors.semantic.danger} />, destructive: true, onPress: () => { notesDirty.current = false; leave(); } },
        ]}
        onClose={() => setConfirmLeave(false)}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  scoreNote: { marginTop: 10, padding: 10, borderRadius: 12, backgroundColor: theme.colors.bg.surfaceHover },
  scoreNoteText: { fontSize: 12, lineHeight: 17, color: theme.colors.text.body },
  trustRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 8 },
  trustDot: { width: 7, height: 7, borderRadius: 4 },
  trustText: { flex: 1, fontSize: 11, lineHeight: 15, color: theme.colors.text.muted },
  tempChip: { marginLeft: 10, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, alignSelf: 'center' },
  tempChipText: { fontSize: 11, fontWeight: '900', letterSpacing: 0.6 },
  whyBox: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider },
  whyLabel: { fontSize: 10, fontWeight: '800', letterSpacing: 1, color: theme.colors.text.muted, marginBottom: 8 },
  dimRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 7 },
  dimLabel: { width: 128, fontSize: 12, fontWeight: '700', color: theme.colors.text.body },
  dimTrack: { flex: 1, height: 7, borderRadius: 4, backgroundColor: theme.colors.bg.surfaceHover, overflow: 'hidden' },
  dimFill: { height: 7, borderRadius: 4 },
  dimValue: { width: 40, textAlign: 'right', fontSize: 12, fontWeight: '800', color: theme.colors.text.display },
  dimOf: { fontSize: 10, fontWeight: '600', color: theme.colors.text.muted },
  actionRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, marginBottom: 6 },
  actionNum: { width: 20, height: 20, borderRadius: 10, alignItems: 'center', justifyContent: 'center', backgroundColor: theme.colors.brand[50], marginTop: 1 },
  actionNumText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[700] },
  actionText: { flex: 1, fontSize: 13, lineHeight: 19, color: theme.colors.text.body },
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 10, padding: 24 },
  missingIcon: { width: 64, height: 64, borderRadius: 22, backgroundColor: theme.colors.bg.surfaceHover, alignItems: 'center', justifyContent: 'center' },
  missingTitle: { fontSize: 18, fontWeight: '800', color: theme.colors.text.display },
  missingText: { fontSize: 14, color: theme.colors.text.muted, textAlign: 'center' },
  missingBtn: { marginTop: 8, backgroundColor: theme.colors.brand[600], paddingHorizontal: 18, paddingVertical: 11, borderRadius: 999 },
  missingBtnText: { color: 'white', fontWeight: '700' },

  header: { height: 60, flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, zIndex: 2 },
  headerBg: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: theme.colors.bg.surface,
    borderBottomWidth: 1, borderBottomColor: theme.colors.bg.divider, ...theme.shadows.card,
  },
  headerCenter: { flex: 1, height: 40, justifyContent: 'center' },
  headerIdentity: { position: 'absolute', left: 0, right: 0, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  headerName: { fontSize: 16, fontWeight: '800', color: theme.colors.text.display, flexShrink: 1 },
  iconBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.bg.surface, borderWidth: 1, borderColor: theme.colors.bg.divider, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontSize: 17, fontWeight: '800', color: theme.colors.text.display, textAlign: 'center' },
  scrollContent: { paddingHorizontal: 16, paddingTop: 4, paddingBottom: 110 },

  hero: {
    alignItems: 'center', backgroundColor: theme.colors.bg.surface, borderRadius: 24, paddingTop: 24, paddingBottom: 18, paddingHorizontal: 16,
    borderWidth: 1, borderColor: theme.colors.bg.divider, marginBottom: 12, overflow: 'hidden', ...theme.shadows.card,
  },
  heroGlow: { position: 'absolute', top: -80, width: 260, height: 160, borderRadius: 130, backgroundColor: theme.colors.brand[50] },
  heroName: { fontSize: 22, lineHeight: 28, fontWeight: '900', color: theme.colors.text.display, marginTop: 12, textAlign: 'center', letterSpacing: -0.3 },
  heroSub: { fontSize: 14, color: theme.colors.text.body, marginTop: 3, textAlign: 'center' },
  heroMuted: { fontSize: 14, color: theme.colors.text.muted, marginTop: 3, fontStyle: 'italic' },
  heroChips: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 6, marginTop: 12 },
  chip: { flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: theme.colors.brand[50], paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999 },
  chipText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },

  quickRow: { flexDirection: 'row', gap: 8, marginBottom: 12 },
  quickAction: { alignItems: 'center', gap: 6, backgroundColor: theme.colors.bg.surface, borderRadius: 16, paddingVertical: 12, borderWidth: 1, borderColor: theme.colors.bg.divider },
  quickActionDisabled: { opacity: 0.55 },
  quickIcon: { width: 36, height: 36, borderRadius: 12, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  quickLabel: { fontSize: 12, fontWeight: '700', color: theme.colors.text.display },

  dupCard: { backgroundColor: theme.colors.semantic.warningBg, borderWidth: 1, borderColor: theme.colors.semantic.warningBorder, borderRadius: 16, padding: 12, marginBottom: 12 },
  dupHeader: { flexDirection: 'row', gap: 10, alignItems: 'flex-start' },
  dupIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: toneBg('#ffffff'), alignItems: 'center', justifyContent: 'center' },
  dupTitle: { fontSize: 14, fontWeight: '800', color: theme.colors.text.display },
  dupSub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2, lineHeight: 18 },
  bold: { fontWeight: '800', color: theme.colors.text.display },
  dupActions: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 10 },
  dupBtn: { flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: toneBg('#ffffff'), paddingHorizontal: 10, paddingVertical: 7, borderRadius: 999, borderWidth: 1, borderColor: theme.colors.semantic.warningBorder },
  dupBtnDanger: { borderColor: theme.colors.semantic.dangerBorder },
  dupBtnText: { fontSize: 12, fontWeight: '700', color: theme.colors.text.display },

  card: { backgroundColor: theme.colors.bg.surface, borderRadius: 20, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: theme.colors.bg.divider, ...theme.shadows.card },
  cardHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 10 },
  cardIcon: { width: 30, height: 30, borderRadius: 10, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center' },
  cardTitle: { flex: 1, fontSize: 11, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  cardMeta: { fontSize: 12, fontWeight: '700', color: theme.colors.text.muted },
  linkText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[600] },
  muted: { fontSize: 14, color: theme.colors.text.muted, lineHeight: 20 },

  notScored: { fontSize: 17, fontWeight: '800', color: theme.colors.text.muted, marginBottom: 4 },
  scoreButton: { flexDirection: 'row', alignItems: 'center', gap: 6, alignSelf: 'flex-start', backgroundColor: theme.colors.brand[600], paddingHorizontal: 14, paddingVertical: 9, borderRadius: 999, marginTop: 12 },
  scoreButtonText: { color: 'white', fontSize: 13, fontWeight: '700' },
  scoreRow: { flexDirection: 'row', alignItems: 'baseline', marginBottom: 10 },
  scoreValue: { fontSize: 38, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -1 },
  scoreOutOf: { fontSize: 15, fontWeight: '700', color: theme.colors.text.muted, marginLeft: 2 },
  scoreTrack: { height: 8, borderRadius: 4, backgroundColor: theme.colors.bg.surfaceHover, overflow: 'hidden' },
  scoreFill: { height: 8, borderRadius: 4 },
  scoreReason: { fontSize: 13, color: theme.colors.text.body, lineHeight: 19, marginTop: 10 },

  detailRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingVertical: 10 },
  detailIcon: { width: 32, height: 32, borderRadius: 10, backgroundColor: theme.colors.bg.surfaceHover, alignItems: 'center', justifyContent: 'center' },
  detailLabel: { fontSize: 10, fontWeight: '800', color: theme.colors.text.muted, letterSpacing: 1, marginBottom: 2 },
  detailValue: { fontSize: 15, color: theme.colors.text.display },
  detailLink: { fontSize: 15, color: theme.colors.brand[700], fontWeight: '600' },
  detailEmpty: { fontSize: 14, color: theme.colors.text.muted, fontStyle: 'italic' },
  divider: { height: 1, backgroundColor: theme.colors.bg.divider, marginLeft: 44 },
  notesText: { fontSize: 15, color: theme.colors.text.display, lineHeight: 22 },
  inlineError: { fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600', marginTop: 8, marginBottom: 8 },
  qrInline: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  qrInlineBox: { width: 124, height: 124, borderRadius: 16, padding: 8, backgroundColor: '#ffffff' /* white in both modes: QR codes must scan */, borderWidth: 1, borderColor: theme.colors.brand[100], alignItems: 'center', justifyContent: 'center' },
  qrInlineImg: { width: 108, height: 108 },
  qrInlineTitle: { fontSize: 15, fontWeight: '800', color: theme.colors.text.display },
  footerMeta: { fontSize: 12, color: theme.colors.text.muted, textAlign: 'center', marginTop: 4 },
}));
