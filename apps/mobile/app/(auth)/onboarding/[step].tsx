import React, { useState, useRef, useEffect } from 'react';
import { View, TextInput, ScrollView, ActivityIndicator, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withDelay,
  withRepeat,
  withSequence,
  interpolateColor,
  Easing,
  FadeIn,
  FadeInDown,
  FadeInUp,
  FadeOut,
  ZoomIn,
  LinearTransition,
} from 'react-native-reanimated';
import {
  Link as LinkIcon,
  Check,
  ArrowRight,
  ArrowLeft,
  UserRound,
  Briefcase,
  Building2,
  Package,
  Factory,
  Users,
  Rocket,
  Building,
  Landmark,
  CalendarDays,
  Target,
  Zap,
  PenLine,
  AlertCircle,
  Sparkles,
  Plus,
  X,
} from 'lucide-react-native';
import { Text } from '../../../src/ui/Text';
import { theme, toneBg, toneFg, makeStyles } from '../../../src/ui/theme';
import { Card } from '../../../src/ui/Card';
import { Badge } from '../../../src/ui/Badge';
import { useSession } from '../../../src/providers/SessionProvider';
import { api, errorMessage, type Event, type PlannedEvent, type ErrorMessageOptions } from '../../../src/lib/api';
import { profileStore } from '../../../src/features/profile';
import { beginLinkedInLink } from '../../../src/features/auth';
import {
  PressableScale,
  SectionHeader,
  EditableField,
  TagEditor,
  SegmentedSelect,
  AnimatedToggle,
  useKeyboardAwareScroll,
} from '../../../src/features/onboarding/components';
import Svg, { Path, Rect } from 'react-native-svg';

// LinkedIn Icon (official design)
const LinkedInIcon = ({ size = 20, color = "#FFFFFF" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Rect width="24" height="24" rx="4" fill={color} />
    <Path d="M7.5 9.5H5V18.5H7.5V9.5Z" fill={toneFg('#0A66C2')} />
    <Path d="M6.25 8C7.07843 8 7.75 7.32843 7.75 6.5C7.75 5.67157 7.07843 5 6.25 5C5.42157 5 4.75 5.67157 4.75 6.5C4.75 7.32843 5.42157 8 6.25 8Z" fill={toneFg('#0A66C2')} />
    <Path d="M12.5 9.5H10V18.5H12.5V13.75C12.5 12.2312 14 12 14.5 12C15 12 16 12.25 16 13.75V18.5H18.5V13C18.5 10 16.5 9.5 15.5 9.5C14.5 9.5 13 10 12.5 11V9.5Z" fill={toneFg('#0A66C2')} />
  </Svg>
);

const TOTAL_STEPS = 3;
// Mirror MAX_PROFILE_EVENTS / MAX_EVENT_NAME_LENGTH in apps/backend/app.py.
const MAX_EVENTS = 20;
const MAX_EVENT_NAME_LENGTH = 120;
const STEP_TITLES = ['Account setup', 'Your profile', 'Event & goals'];

const COMPANY_SIZES = [
  { value: 'Startup', icon: Rocket },
  { value: 'Mid-market', icon: Building },
  { value: 'Enterprise', icon: Landmark },
];

// Accepts linkedin.com/in/<handle> with or without scheme / www / trailing slash / query.
const LINKEDIN_PROFILE_RE = /^(?:https?:\/\/)?(?:[a-z]{2,3}\.)?linkedin\.com\/in\/([A-Za-z0-9\-_%]+)\/?(?:\?.*)?$/i;

const enter = (delay: number) => FadeInDown.delay(delay).duration(260);

function initialsOf(fullName: string) {
  const parts = fullName.trim().split(/\s+/).filter(Boolean);
  const first = parts[0]?.[0] ?? '';
  const last = parts.length > 1 ? parts[parts.length - 1]?.[0] ?? '' : '';
  return (first + last).toUpperCase();
}

// How this screen words a failed request (see errorMessage).
const ONBOARDING_ERRORS: ErrorMessageOptions = { offline: "You're offline. Check your connection and try again." };

// Saved as profile.auto_score_on_add; the backend scores each business-card / QR scan in the
// background when it's on (see is_auto_score_enabled in apps/backend/app.py).
function AutoScoreCard({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  const progress = useSharedValue(value ? 1 : 0);
  const bolt = useSharedValue(1);

  useEffect(() => {
    progress.value = withTiming(value ? 1 : 0, { duration: 260 });
    if (value) {
      bolt.value = withSequence(withSpring(1.04, { damping: 22, stiffness: 320, overshootClamping: true }), withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true }));
    } else {
      bolt.value = withSequence(withTiming(0.85, { duration: 110 }), withSpring(1, { damping: 22, stiffness: 200, overshootClamping: true }));
    }
  }, [value]);

  const cDivider = theme.colors.bg.divider; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const cB300 = theme.colors.brand[300];
  const cSurface = theme.colors.bg.surface;
  const cB50 = theme.colors.brand[50];
  const cHover = theme.colors.bg.surfaceHover;
  const cB600 = theme.colors.brand[600];
  const cardStyle = useAnimatedStyle(() => ({
    borderColor: interpolateColor(progress.value, [0, 1], [cDivider, cB300]),
    backgroundColor: interpolateColor(progress.value, [0, 1], [cSurface, cB50]),
  }));
  const iconBoxStyle = useAnimatedStyle(() => ({
    backgroundColor: interpolateColor(progress.value, [0, 1], [cHover, cB600]),
    transform: [{ scale: bolt.value }, { rotate: `${(bolt.value - 1) * 40}deg` }],
  }));

  return (
    <Pressable
      onPress={() => onChange(!value)}
      accessibilityRole="switch"
      accessibilityState={{ checked: value }}
      accessibilityLabel="Auto-score on scan"
    >
      <Animated.View style={[styles.autoCard, cardStyle]}>
        <Animated.View style={[styles.toggleIcon, iconBoxStyle]}>
          <Zap
            size={18}
            color={value ? theme.colors.text.inverse : theme.colors.text.muted}
            fill={value ? theme.colors.text.inverse : 'transparent'}
          />
        </Animated.View>
        <View style={styles.toggleInfo}>
          <Text style={styles.toggleTitle}>Auto-score on scan</Text>
          <Animated.Text key={value ? 'on' : 'off'} entering={FadeInDown.duration(220)} style={[styles.toggleSub, value && styles.toggleSubOn]}>
            {value
              ? 'On — every scanned card or badge is scored against your targets'
              : 'Off — score leads yourself from the contact screen'}
          </Animated.Text>
        </View>
        <AnimatedToggle value={value} onValueChange={onChange} accessibilityLabel="Auto-score on scan" />
      </Animated.View>
    </Pressable>
  );
}

export default function OnboardingScreen() {
  const { step, name, linked, linkError } = useLocalSearchParams<{ step: string; name?: string; linked?: string; linkError?: string }>();
  const router = useRouter();
  const { userId, userName, authMethod } = useSession();
  const { scrollProps, keyboardHeight, onInputFocus } = useKeyboardAwareScroll();

  const currentStep = Math.min(Math.max(parseInt(step || '1', 10) || 1, 1), TOTAL_STEPS);

  const goToStep = (target: number, params: Record<string, string> = {}) =>
    router.push({ pathname: '/onboarding/[step]' as any, params: { step: String(target), ...params } });

  const goBack = () => {
    if (router.canGoBack()) router.back();
    else router.replace({ pathname: '/onboarding/[step]' as any, params: { step: String(currentStep - 1) } });
  };

  // Users who signed in with LinkedIn are already connected — skip the connect step.
  useEffect(() => {
    if (currentStep === 1 && authMethod === 'linkedin') {
      router.replace({ pathname: '/onboarding/[step]' as any, params: { step: '2', name: name || '' } });
    }
  }, []);

  // ─── Profile load (steps 2 & 3 prefill from what's already saved) ─────────
  const [profileLoading, setProfileLoading] = useState(currentStep > 1);
  const [savedProfile, setSavedProfile] = useState<Record<string, any>>({});

  // Step 1
  const [isLoadingLinkedIn, setIsLoadingLinkedIn] = useState(false);
  const [profileUrl, setProfileUrl] = useState('');
  const [urlError, setUrlError] = useState('');
  const [isSavingUrl, setIsSavingUrl] = useState(false);
  // linkError arrives from /callback when a web-redirect link attempt failed.
  const [linkedInError, setLinkedInError] = useState(linkError || '');
  const [linkedInConnected, setLinkedInConnected] = useState(false);

  // Step 2
  const [fullName, setFullName] = useState(name || userName || '');
  const [jobTitle, setJobTitle] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [youSellText, setYouSellText] = useState('');
  const [industries, setIndustries] = useState<string[]>([]);
  const [roles, setRoles] = useState<string[]>([]);
  const [selectedSize, setSelectedSize] = useState('');
  const [nameError, setNameError] = useState('');

  // Step 3
  const [events, setEvents] = useState<Event[]>([]);
  const [plannedEvents, setPlannedEvents] = useState<PlannedEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(currentStep === 3);
  // Every event the user is scouting (saved as profile.events) and the active one
  // (profile.current_event_name) that new scans get tagged with.
  const [myEvents, setMyEvents] = useState<string[]>([]);
  const [activeEvent, setActiveEvent] = useState('');
  const [newEventText, setNewEventText] = useState('');
  const [eventNotice, setEventNotice] = useState('');
  const [goalText, setGoalText] = useState('');
  const [autoScore, setAutoScore] = useState(true);

  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState('');

  const youSellRef = useRef<TextInput>(null);

  useEffect(() => {
    if (currentStep === 1 || !userId) {
      setProfileLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const p = await profileStore.load(userId);
        if (cancelled) return;
        setSavedProfile(p);
        if (currentStep === 2) {
          if (p.full_name) setFullName(p.full_name);
          if (p.job_title) setJobTitle(p.job_title);
          if (p.company_name) setCompanyName(p.company_name);
          const firstProduct = Array.isArray(p.products) ? p.products[0] : undefined;
          if (firstProduct?.description) setYouSellText(firstProduct.description);
          if (Array.isArray(p.target_industries)) setIndustries(p.target_industries);
          if (Array.isArray(p.target_roles)) setRoles(p.target_roles);
          if (Array.isArray(p.target_company_sizes) && p.target_company_sizes[0]) setSelectedSize(p.target_company_sizes[0]);
        } else {
          if (Array.isArray(p.event_goals) && p.event_goals[0]) setGoalText(p.event_goals[0]);
          if (typeof p.auto_score_on_add === 'boolean') setAutoScore(p.auto_score_on_add);
          const savedEvents: string[] = Array.isArray(p.events) ? p.events.filter((e: any) => typeof e === 'string' && e.trim()) : [];
          if (p.current_event_name && !savedEvents.some((e) => e.toLowerCase() === String(p.current_event_name).toLowerCase())) {
            savedEvents.push(p.current_event_name);
          }
          setMyEvents(savedEvents);
          setActiveEvent(p.current_event_name || savedEvents[0] || '');
        }
      } catch {
        // No saved profile yet (or offline) — start from blank fields.
      } finally {
        if (!cancelled) setProfileLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [currentStep, userId]);

  useEffect(() => {
    if (currentStep !== 1 || !userId) return;
    let cancelled = false;
    api.getLinkedInLinkStatus()
      .then((res) => { if (!cancelled) setLinkedInConnected(!!res?.connected); })
      .catch(() => { /* unknown status — show the connect button */ });
    return () => { cancelled = true; };
  }, [currentStep, userId]);

  useEffect(() => {
    if (currentStep !== 3) return;
    let cancelled = false;
    // Suggestions = events with an imported exhibitor list + events planned for you in the
    // admin console (deduped by name; the exhibitor-list entry wins).
    Promise.all([
      api.listEvents().catch(() => [] as Event[]),
      api.listMyPlannedEvents().catch(() => [] as PlannedEvent[]),
    ])
      .then(([list, planned]) => {
        if (cancelled) return;
        const withLists = Array.isArray(list) ? list : [];
        const seen = new Set(withLists.map((e) => e.name.trim().toLowerCase()));
        const plannedOnly: PlannedEvent[] = [];
        for (const p of Array.isArray(planned) ? planned : []) {
          const key = (p?.name || '').trim().toLowerCase();
          if (!key || seen.has(key)) continue;
          seen.add(key);
          plannedOnly.push(p);
        }
        setEvents(withLists);
        setPlannedEvents(plannedOnly);
      })
      .finally(() => { if (!cancelled) setEventsLoading(false); });
    return () => { cancelled = true; };
  }, [currentStep]);

  // ─── Step 3 event list ────────────────────────────────────────────────────
  const findEvent = (name: string) => myEvents.find((e) => e.toLowerCase() === name.toLowerCase());

  const addEvent = (raw: string): boolean => {
    const name = raw.trim().replace(/\s+/g, ' ');
    if (!name) {
      setEventNotice('Type an event name first');
      return false;
    }
    if (name.length > MAX_EVENT_NAME_LENGTH) {
      setEventNotice(`Event names can be at most ${MAX_EVENT_NAME_LENGTH} characters`);
      return false;
    }
    const existing = findEvent(name);
    if (existing) {
      setActiveEvent(existing);
      setEventNotice(`"${existing}" is already in your events — made it active`);
      return false;
    }
    if (myEvents.length >= MAX_EVENTS) {
      setEventNotice(`You can add up to ${MAX_EVENTS} events`);
      return false;
    }
    setEventNotice('');
    setMyEvents((prev) => [...prev, name]);
    if (!activeEvent) setActiveEvent(name);
    return true;
  };

  const removeEvent = (name: string) => {
    const rest = myEvents.filter((e) => e !== name);
    setMyEvents(rest);
    setEventNotice('');
    if (activeEvent === name) setActiveEvent(rest[0] || '');
  };

  const submitNewEvent = () => {
    if (addEvent(newEventText)) setNewEventText('');
  };

  // ─── Lively bits ──────────────────────────────────────────────────────────
  const float = useSharedValue(0);
  useEffect(() => {
    float.value = withRepeat(
      withSequence(
        withTiming(1, { duration: 1600, easing: Easing.inOut(Easing.quad) }),
        withTiming(0, { duration: 1600, easing: Easing.inOut(Easing.quad) }),
      ),
      -1,
    );
  }, []);
  const floatStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: float.value * -5 }, { rotate: `${(float.value - 0.5) * 6}deg` }],
  }));

  // Current progress segment fills in on arrival.
  const progressFill = useSharedValue(0);
  useEffect(() => {
    progressFill.value = withDelay(150, withSpring(1, { damping: 22, stiffness: 90, overshootClamping: true }));
  }, [currentStep]);
  const progressFillStyle = useAnimatedStyle(() => ({ width: `${progressFill.value * 100}%` }));

  // ─── Step 1 handlers ──────────────────────────────────────────────────────
  const handleLinkedInConnect = async () => {
    if (linkedInConnected) {
      goToStep(2, { name: name || '', linked: '1' });
      return;
    }
    setLinkedInError('');
    setIsLoadingLinkedIn(true);
    try {
      // Connects LinkedIn to the account that is signed in now — the session is unchanged.
      // Web: redirects to LinkedIn and /callback completes. Native: resolves with the result.
      const result = await beginLinkedInLink();
      if (!result) return;

      setLinkedInConnected(true);
      goToStep(2, { name: result.linkedin?.name || result.user?.name || name || '', linked: '1' });
    } catch (error: any) {
      setLinkedInError(
        error?.response?.status === 501
          ? "LinkedIn sign-in isn't set up on this server yet."
          : errorMessage(error, 'LinkedIn connection failed. Please try again.', ONBOARDING_ERRORS),
      );
    } finally {
      setIsLoadingLinkedIn(false);
    }
  };

  const handleProfileUrlSave = async () => {
    const trimmed = profileUrl.trim();
    const match = trimmed.match(LINKEDIN_PROFILE_RE);
    if (!match) {
      setUrlError('Enter a profile URL like linkedin.com/in/your-handle');
      return;
    }
    if (!userId) {
      setUrlError('Your session has expired. Please log in again.');
      return;
    }
    setUrlError('');
    setIsSavingUrl(true);
    try {
      await profileStore.save(userId, { linkedin_url: `https://www.linkedin.com/in/${match[1]}` });
      goToStep(2, { name: name || '' });
    } catch (error: any) {
      setUrlError(errorMessage(error, "Couldn't save your profile URL. Please try again.", ONBOARDING_ERRORS));
    } finally {
      setIsSavingUrl(false);
    }
  };

  // ─── Step 2 / 3 save ──────────────────────────────────────────────────────
  const handleStep2Continue = async () => {
    const trimmedName = fullName.trim();
    if (!trimmedName) {
      setNameError('Please enter your name');
      return;
    }
    if (!userId) {
      setSaveError('Your session has expired. Please log in again.');
      return;
    }
    setNameError('');
    setSaveError('');
    setIsSaving(true);
    try {
      const existingProducts: any[] = Array.isArray(savedProfile.products) ? savedProfile.products : [];
      const sell = youSellText.trim();
      const products = sell
        ? [{ ...(existingProducts[0] || {}), name: existingProducts[0]?.name || companyName.trim() || 'My offering', description: sell }, ...existingProducts.slice(1)]
        : existingProducts.slice(1);

      await profileStore.save(userId, {
        full_name: trimmedName,
        job_title: jobTitle.trim(),
        company_name: companyName.trim(),
        products,
        target_industries: industries,
        target_roles: roles,
        target_company_sizes: selectedSize ? [selectedSize] : [],
      });
      goToStep(3);
    } catch (error: any) {
      setSaveError(errorMessage(error, "Couldn't save your profile. Please try again.", ONBOARDING_ERRORS));
    } finally {
      setIsSaving(false);
    }
  };

  const handleFinish = async () => {
    if (!userId) {
      setSaveError('Your session has expired. Please log in again.');
      return;
    }
    setSaveError('');
    setIsSaving(true);
    try {
      const goal = goalText.trim();
      // A name typed but not yet added with "Add" still counts.
      const pending = newEventText.trim().replace(/\s+/g, ' ');
      const allEvents = pending && !findEvent(pending) ? [...myEvents, pending] : myEvents;
      const active = activeEvent || allEvents[0] || '';
      await profileStore.save(userId, {
        events: allEvents,
        current_event_name: active,
        event_goals: goal ? [goal] : [],
        auto_score_on_add: autoScore,
      });
      router.replace('/(app)/feed' as any);
    } catch (error: any) {
      setSaveError(errorMessage(error, "Couldn't save your event settings. Please try again.", ONBOARDING_ERRORS));
    } finally {
      setIsSaving(false);
    }
  };

  // ─── Shared UI ────────────────────────────────────────────────────────────
  const renderProgressBar = () => (
    <Animated.View entering={FadeIn.duration(300)} style={styles.progressContainer}>
      <View style={styles.progressHeader}>
        <Text style={styles.stepText}>STEP {currentStep} OF {TOTAL_STEPS}</Text>
        <Text style={styles.stepTitle}>{STEP_TITLES[currentStep - 1]}</Text>
      </View>
      <View style={styles.progressBarWrapper}>
        {Array.from({ length: TOTAL_STEPS }, (_, i) => {
          const segment = i + 1;
          return (
            <View key={segment} style={styles.progressSegment}>
              {segment < currentStep && <View style={[styles.progressFill, { width: '100%' }]} />}
              {segment === currentStep && <Animated.View style={[styles.progressFill, progressFillStyle]} />}
            </View>
          );
        })}
      </View>
    </Animated.View>
  );

  const renderHeading = (icon: React.ReactNode, title: string, description: string) => (
    <>
      <Animated.View entering={FadeInUp.duration(260)} style={styles.headingIconWrap}>
        <Animated.View style={[styles.headingIcon, floatStyle]}>{icon}</Animated.View>
      </Animated.View>
      <Animated.Text entering={enter(60)} style={styles.title}>{title}</Animated.Text>
      <Animated.Text entering={enter(120)} style={styles.description}>{description}</Animated.Text>
    </>
  );

  // LinkedIn users skip step 1, so there is nothing to go back to from step 2.
  const renderBackButton = () => currentStep === 2 && authMethod === 'linkedin' ? null : (
    <PressableScale
      containerStyle={{ flex: 1 }}
      style={styles.backButton}
      onPress={goBack}
      disabled={isSaving}
      leadingIcon={<ArrowLeft size={16} color={theme.colors.text.display} strokeWidth={2.5} />}
      accessibilityRole="button"
    >
      <Text style={styles.backButtonText}>Back</Text>
    </PressableScale>
  );

  const renderSaveError = () =>
    saveError ? (
      <Animated.View entering={FadeInDown.duration(260)} style={styles.errorBanner}>
        <AlertCircle size={16} color={theme.colors.semantic.danger} />
        <Text style={styles.errorBannerText}>{saveError}</Text>
      </Animated.View>
    ) : null;

  // ─── Step 1 ───────────────────────────────────────────────────────────────
  const renderStep1 = () => (
    <>
      {renderHeading(
        <LinkedInIcon size={26} color={theme.colors.social.linkedin} />,
        'Connect LinkedIn',
        "Connect LinkedIn to this account to prefill your name and sign in with one tap next time. You'll review and edit everything on the next screen.",
      )}

      {/* Main LinkedIn Connect Card */}
      <Animated.View entering={enter(180)}>
        <Card>
          <PressableScale
            style={styles.linkedinButton}
            onPress={handleLinkedInConnect}
            disabled={isLoadingLinkedIn}
            leadingIcon={
              isLoadingLinkedIn ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} />
                : linkedInConnected ? <Check size={20} color={toneFg('#FFFFFF')} strokeWidth={3} />
                : <LinkedInIcon size={20} color={toneFg('#FFFFFF')} />
            }
            trailingIcon={!isLoadingLinkedIn ? <ArrowRight size={20} color={toneFg('#FFFFFF')} strokeWidth={2.5} /> : undefined}
            accessibilityRole="button"
          >
            <Text style={styles.linkedinButtonText}>
              {isLoadingLinkedIn ? 'Connecting…' : linkedInConnected ? 'LinkedIn connected · Continue' : 'Connect LinkedIn'}
            </Text>
          </PressableScale>
          {linkedInError ? (
            <Animated.View entering={FadeInDown.duration(260)} style={styles.inlineError}>
              <AlertCircle size={14} color={theme.colors.semantic.danger} />
              <Text style={styles.inlineErrorText}>{linkedInError}</Text>
            </Animated.View>
          ) : null}
          <View style={styles.trustBadges}>
            <Check size={14} color={theme.colors.brand[600]} />
            <Text style={styles.trustTextHighlight}>read-only</Text>
            <Text style={styles.trustDot}>•</Text>
            <Text style={styles.trustText}>we never post</Text>
            <Text style={styles.trustDot}>•</Text>
            <Text style={styles.trustText}>revoke any time</Text>
          </View>
        </Card>
      </Animated.View>

      {/* What it unlocks Card */}
      <Animated.View entering={enter(260)}>
        <Card variant="dashed" style={{ padding: theme.spacing.lg }}>
          <View style={styles.unlocksLabelWrapper}>
            <Sparkles size={12} color={theme.colors.brand[800]} />
            <Text style={styles.unlocksLabel}>WHAT YOU GET</Text>
          </View>
          {[
            'Your name filled in for you on the next step',
            'One-tap sign-in with LinkedIn from now on',
            'Pitch suggestions written around what you sell',
            'Scanned leads scored against your target market',
          ].map((feature, i) => (
            <Animated.View key={feature} entering={enter(320 + i * 70)} style={styles.featureRow}>
              <View style={styles.checkCircle}><Check size={12} color={theme.colors.brand[600]} strokeWidth={3} /></View>
              <Text style={styles.featureText}>{feature}</Text>
            </Animated.View>
          ))}
        </Card>
      </Animated.View>

      {/* Profile URL Card */}
      <Animated.View entering={enter(420)}>
        <Card>
          <View style={styles.linkHeader}>
            <View style={styles.linkIconWrapper}>
              <LinkIcon size={16} color={theme.colors.brand[600]} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.linkTitle}>Prefer not to sign in? Add your profile URL</Text>
              <Text style={styles.linkSub}>Saved to your profile so the AI assistant can reference it.</Text>
            </View>
          </View>
          <View style={[styles.inputWithButton, !!urlError && styles.inputError]}>
            <TextInput
              style={styles.urlInput}
              placeholder="linkedin.com/in/your-handle"
              placeholderTextColor={theme.colors.text.muted}
              value={profileUrl}
              onChangeText={(t) => { setProfileUrl(t); if (urlError) setUrlError(''); }}
              onFocus={onInputFocus}
              onSubmitEditing={handleProfileUrlSave}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              returnKeyType="done"
            />
            <PressableScale
              style={styles.addBtn}
              onPress={handleProfileUrlSave}
              disabled={isSavingUrl || !profileUrl.trim()}
              scaleTo={0.92}
              accessibilityRole="button"
            >
              {isSavingUrl ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <Text style={styles.addBtnText}>Save</Text>}
            </PressableScale>
          </View>
          {urlError ? <Text style={styles.fieldErrorText}>{urlError}</Text> : null}
        </Card>
      </Animated.View>

      {/* Footer */}
      <Animated.View entering={enter(500)} style={styles.footerSingle}>
        <PressableScale
          style={styles.fillButton}
          onPress={() => goToStep(2, { name: name || '' })}
          leadingIcon={<PenLine size={17} color={theme.colors.brand[700]} strokeWidth={2.25} />}
          trailingIcon={<ArrowRight size={18} color={theme.colors.brand[700]} strokeWidth={2.5} />}
          accessibilityRole="button"
        >
          <Text style={styles.fillButtonText}>Skip and fill it in myself</Text>
        </PressableScale>
      </Animated.View>
    </>
  );

  // ─── Step 2 ───────────────────────────────────────────────────────────────
  const renderStep2 = () => {
    const fromLinkedIn = authMethod === 'linkedin' || linked === '1';
    const initials = initialsOf(fullName);

    return (
      <>
        {renderHeading(
          <UserRound size={26} color={theme.colors.brand[600]} />,
          fromLinkedIn ? 'Is this you?' : 'Tell us about yourself',
          fromLinkedIn
            ? 'We pulled your name from LinkedIn. Add your role and what you sell — every field below is editable.'
            : 'This shapes your lead scores and pitch suggestions. Every field is editable, now or later from your profile.',
        )}

        {profileLoading ? (
          <Animated.View entering={FadeIn} style={styles.loadingRow}>
            <ActivityIndicator size="small" color={theme.colors.brand[600]} />
            <Text style={styles.loadingText}>Loading your saved details…</Text>
          </Animated.View>
        ) : (
          <>
            {/* Profile Card */}
            <Animated.View entering={enter(160)}>
              <Card>
                <View style={styles.profileHeader}>
                  <View style={styles.avatar}>
                    {initials
                      ? <Text style={styles.avatarText}>{initials}</Text>
                      : <UserRound size={22} color={theme.colors.brand[700]} />}
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.profileName} numberOfLines={1}>{fullName.trim() || 'Your name'}</Text>
                    <Text style={styles.profileSub} numberOfLines={1}>
                      {[jobTitle.trim(), companyName.trim()].filter(Boolean).join(' · ') || 'Add your title and company'}
                    </Text>
                  </View>
                  {fromLinkedIn
                    ? <Badge label="LinkedIn" variant="success" icon="Check" />
                    : <View style={styles.editablePill}><PenLine size={11} color={theme.colors.brand[700]} /><Text style={styles.editablePillText}>Editable</Text></View>}
                </View>

                <View style={styles.profileFields}>
                  <EditableField
                    icon={<UserRound size={16} color={theme.colors.brand[600]} />}
                    label="Full name"
                    value={fullName}
                    onChangeText={(t) => { setFullName(t); if (nameError) setNameError(''); }}
                    placeholder="e.g. Priya Sharma"
                    autoCapitalize="words"
                    returnKeyType="next"
                    onFocus={onInputFocus}
                    error={nameError}
                  />
                  <EditableField
                    icon={<Briefcase size={16} color={theme.colors.brand[600]} />}
                    label="Job title"
                    value={jobTitle}
                    onChangeText={setJobTitle}
                    placeholder="e.g. Head of Sales"
                    autoCapitalize="words"
                    returnKeyType="next"
                    onFocus={onInputFocus}
                  />
                  <EditableField
                    icon={<Building2 size={16} color={theme.colors.brand[600]} />}
                    label="Company"
                    value={companyName}
                    onChangeText={setCompanyName}
                    placeholder="e.g. Acme Health"
                    autoCapitalize="words"
                    returnKeyType="next"
                    onSubmitEditing={() => youSellRef.current?.focus()}
                    onFocus={onInputFocus}
                  />
                </View>
              </Card>
            </Animated.View>

            {/* What You Sell */}
            <Animated.View entering={enter(240)} style={styles.section}>
              <SectionHeader
                icon={<Package size={15} color={theme.colors.brand[600]} />}
                label="WHAT YOU SELL"
                hint="Used to write pitch angles for each lead"
              />
              <EditableField
                inputRef={youSellRef}
                icon={<Sparkles size={16} color={theme.colors.brand[600]} />}
                label="Describe your product or service"
                value={youSellText}
                onChangeText={setYouSellText}
                placeholder="e.g. B2B contact data for healthcare marketing teams"
                multiline
                maxLength={500}
                onFocus={onInputFocus}
              />
            </Animated.View>

            {/* Target Industries */}
            <Animated.View entering={enter(320)} style={styles.section}>
              <SectionHeader
                icon={<Factory size={15} color={theme.colors.brand[600]} />}
                label="TARGET INDUSTRIES"
                hint="Tap a tag to remove it"
                right={industries.length > 0 ? <Text style={styles.countPill}>{industries.length}</Text> : undefined}
              />
              <TagEditor
                tags={industries}
                setTags={setIndustries}
                tone="industry"
                addLabel="Add industry"
                placeholder="e.g. Healthcare"
                onInputFocus={onInputFocus}
              />
            </Animated.View>

            {/* Target Roles */}
            <Animated.View entering={enter(400)} style={styles.section}>
              <SectionHeader
                icon={<Users size={15} color={theme.colors.brand[600]} />}
                label="TARGET ROLES"
                hint="Tap a tag to remove it"
                right={roles.length > 0 ? <Text style={[styles.countPill, styles.countPillRole]}>{roles.length}</Text> : undefined}
              />
              <TagEditor
                tags={roles}
                setTags={setRoles}
                tone="role"
                addLabel="Add role"
                placeholder="e.g. Head of Data"
                onInputFocus={onInputFocus}
              />
            </Animated.View>

            {/* Company Size */}
            <Animated.View entering={enter(480)} style={styles.section}>
              <SectionHeader
                icon={<Building2 size={15} color={theme.colors.brand[600]} />}
                label="TARGET COMPANY SIZE"
                hint="Pick the size you sell to most"
              />
              <SegmentedSelect
                value={selectedSize}
                onChange={setSelectedSize}
                options={COMPANY_SIZES.map(({ value, icon: Icon }) => ({
                  value,
                  icon: (active: boolean) => (
                    <Icon size={16} color={active ? theme.colors.text.inverse : theme.colors.text.muted} strokeWidth={2.25} />
                  ),
                }))}
              />
            </Animated.View>

            <Animated.View entering={enter(540)} style={styles.hintRow}>
              <PenLine size={12} color={theme.colors.text.muted} />
              <Text style={styles.hintText}>Saved when you tap Continue — you can change it later in your profile.</Text>
            </Animated.View>
          </>
        )}

        {renderSaveError()}

        {/* Footer */}
        <Animated.View entering={enter(600)} style={[styles.footerDual, styles.footerPush]}>
          {renderBackButton()}
          <PressableScale
            containerStyle={{ flex: 2 }}
            style={styles.continueButton}
            onPress={handleStep2Continue}
            disabled={isSaving || profileLoading}
            trailingIcon={isSaving ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <ArrowRight size={18} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
            accessibilityRole="button"
          >
            <Text style={styles.continueButtonText}>{isSaving ? 'Saving' : 'Continue'}</Text>
          </PressableScale>
        </Animated.View>
      </>
    );
  };

  // ─── Step 3 ───────────────────────────────────────────────────────────────
  const renderStep3 = () => (
    <>
      {renderHeading(
        <CalendarDays size={26} color={theme.colors.brand[600]} />,
        'Which events?',
        'Add every event you are scouting, pick the one you are at now, and tell us what a good outcome looks like.',
      )}

      <Animated.View entering={enter(160)} style={styles.section}>
        <SectionHeader
          icon={<CalendarDays size={15} color={theme.colors.brand[600]} />}
          label="YOUR EVENTS"
          hint="The active event is added to every contact you scan"
          right={myEvents.length > 0 ? <Text style={styles.countPill}>{myEvents.length}</Text> : undefined}
        />
        {eventsLoading || profileLoading ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator size="small" color={theme.colors.brand[600]} />
            <Text style={styles.loadingText}>Loading events…</Text>
          </View>
        ) : (
          <>
            {/* Events that have an exhibitor list on the server — tap to add/remove */}
            {events.length > 0 && (
              <>
                <Text style={styles.subLabel}>WITH AN IMPORTED EXHIBITOR LIST</Text>
                {events.map((ev, i) => {
                  const added = !!findEvent(ev.name);
                  return (
                    <Animated.View key={ev.id} entering={enter(200 + i * 60)}>
                      <PressableScale
                        style={[styles.eventCard, added && styles.eventCardSelected]}
                        onPress={() => (added ? removeEvent(findEvent(ev.name)!) : addEvent(ev.name))}
                        scaleTo={0.98}
                        accessibilityRole="checkbox"
                        accessibilityState={{ checked: added }}
                      >
                        <View style={{ flex: 1 }}>
                          <Text style={styles.eventName}>{ev.name}</Text>
                          <Text style={styles.eventMeta}>
                            {ev.exhibitorCount} exhibitor{ev.exhibitorCount === 1 ? '' : 's'} imported
                          </Text>
                        </View>
                        <View style={[styles.checkCircleLg, !added && styles.checkCircleLgEmpty]}>
                          {added ? (
                            <Animated.View entering={ZoomIn.duration(260)}>
                              <Check size={16} color={theme.colors.text.inverse} strokeWidth={3} />
                            </Animated.View>
                          ) : (
                            <Plus size={14} color={theme.colors.text.muted} strokeWidth={3} />
                          )}
                        </View>
                      </PressableScale>
                    </Animated.View>
                  );
                })}
              </>
            )}

            {/* Events planned for you in the admin console (no exhibitor list yet) — tap to add/remove */}
            {plannedEvents.length > 0 && (
              <>
                <Text style={styles.subLabel}>PLANNED FOR YOU</Text>
                {plannedEvents.map((ev, i) => {
                  const added = !!findEvent(ev.name);
                  const where = [ev.city, ev.country].filter(Boolean).join(', ');
                  const meta = [where, ev.startDate].filter(Boolean).join(' · ') || 'Planned event';
                  return (
                    <Animated.View key={ev.id} entering={enter(200 + (events.length + i) * 60)}>
                      <PressableScale
                        style={[styles.eventCard, added && styles.eventCardSelected]}
                        onPress={() => (added ? removeEvent(findEvent(ev.name)!) : addEvent(ev.name))}
                        scaleTo={0.98}
                        accessibilityRole="checkbox"
                        accessibilityState={{ checked: added }}
                      >
                        <View style={{ flex: 1 }}>
                          <Text style={styles.eventName}>{ev.name}</Text>
                          <Text style={styles.eventMeta}>{meta}</Text>
                        </View>
                        <View style={[styles.checkCircleLg, !added && styles.checkCircleLgEmpty]}>
                          {added ? (
                            <Animated.View entering={ZoomIn.duration(260)}>
                              <Check size={16} color={theme.colors.text.inverse} strokeWidth={3} />
                            </Animated.View>
                          ) : (
                            <Plus size={14} color={theme.colors.text.muted} strokeWidth={3} />
                          )}
                        </View>
                      </PressableScale>
                    </Animated.View>
                  );
                })}
              </>
            )}

            {/* Any other event, typed by hand — add as many as needed */}
            <Text style={styles.subLabel}>{events.length + plannedEvents.length > 0 ? 'NOT LISTED? ADD IT BY NAME' : 'ADD AN EVENT BY NAME'}</Text>
            <View style={[styles.inputWithButton, !!eventNotice && styles.inputNotice]}>
              <TextInput
                style={styles.urlInput}
                placeholder="e.g. Arab Health 2026"
                placeholderTextColor={theme.colors.text.muted}
                value={newEventText}
                onChangeText={(t) => { setNewEventText(t); if (eventNotice) setEventNotice(''); }}
                onSubmitEditing={submitNewEvent}
                submitBehavior="submit"
                onFocus={onInputFocus}
                autoCapitalize="words"
                returnKeyType="done"
                maxLength={MAX_EVENT_NAME_LENGTH}
              />
              <PressableScale
                style={[styles.addBtn, styles.addBtnRow]}
                onPress={submitNewEvent}
                disabled={!newEventText.trim()}
                scaleTo={0.92}
                leadingIcon={<Plus size={14} color={theme.colors.text.inverse} strokeWidth={3} />}
                accessibilityRole="button"
                accessibilityLabel="Add event"
              >
                <Text style={styles.addBtnText}>Add</Text>
              </PressableScale>
            </View>
            {eventNotice ? (
              <Animated.Text entering={FadeInDown.duration(200)} style={styles.eventNoticeText}>{eventNotice}</Animated.Text>
            ) : null}

            {/* The user's events: tap to make active, × to remove */}
            <View style={styles.myEvents}>
              {myEvents.length === 0 ? (
                <Animated.Text entering={FadeIn} style={styles.emptyEvents}>
                  No events yet. Pick one above or add it by name — or skip for now and add it later.
                </Animated.Text>
              ) : (
                myEvents.map((name) => {
                  const active = name === activeEvent;
                  const imported = events.find((e) => e.name.toLowerCase() === name.toLowerCase());
                  return (
                    <Animated.View
                      key={name}
                      entering={FadeInDown.duration(260)}
                      exiting={FadeOut.duration(160)}
                      layout={LinearTransition.duration(260)}
                    >
                      <PressableScale
                        style={[styles.myEventRow, active && styles.myEventRowActive]}
                        onPress={() => setActiveEvent(name)}
                        scaleTo={0.98}
                        accessibilityRole="radio"
                        accessibilityState={{ selected: active }}
                        accessibilityLabel={`${name}${active ? ', active event' : ''}`}
                      >
                        <View style={[styles.radio, active && styles.radioActive]}>
                          {active ? <Animated.View entering={ZoomIn.duration(260)} style={styles.radioDot} /> : null}
                        </View>
                        <View style={{ flex: 1 }}>
                          <Text style={styles.myEventName} numberOfLines={1}>{name}</Text>
                          <Text style={[styles.myEventMeta, active && styles.myEventMetaActive]} numberOfLines={1}>
                            {active ? 'Active · new scans are tagged with this event' : 'Tap to make active'}
                            {' · '}
                            {imported ? `${imported.exhibitorCount} exhibitors` : 'added by you'}
                          </Text>
                        </View>
                        <Pressable
                          onPress={() => removeEvent(name)}
                          hitSlop={10}
                          style={styles.removeEvent}
                          accessibilityRole="button"
                          accessibilityLabel={`Remove ${name}`}
                        >
                          <X size={14} color={theme.colors.text.muted} strokeWidth={2.75} />
                        </Pressable>
                      </PressableScale>
                    </Animated.View>
                  );
                })
              )}
            </View>
          </>
        )}
      </Animated.View>

      <Animated.View entering={enter(260)} style={styles.section}>
        <SectionHeader
          icon={<Target size={15} color={theme.colors.brand[600]} />}
          label="YOUR GOAL HERE"
          hint="The assistant uses this to prioritise leads"
        />
        <EditableField
          icon={<Target size={16} color={theme.colors.brand[600]} />}
          label="What would make this event a success?"
          value={goalText}
          onChangeText={setGoalText}
          placeholder="e.g. Meet 10 hospital groups planning multi-site data consolidation"
          multiline
          maxLength={300}
          onFocus={onInputFocus}
        />
      </Animated.View>

      {/* Auto Score Toggle */}
      <Animated.View entering={enter(320)}>
        <AutoScoreCard value={autoScore} onChange={setAutoScore} />
      </Animated.View>

      {renderSaveError()}

      {/* Footer */}
      <Animated.View entering={enter(400)} style={[styles.footerDual, styles.footerPush]}>
        {renderBackButton()}
        <PressableScale
          containerStyle={{ flex: 2 }}
          style={styles.continueButton}
          onPress={handleFinish}
          disabled={isSaving || profileLoading}
          trailingIcon={isSaving ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <ArrowRight size={18} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
          accessibilityRole="button"
        >
          <Text style={styles.continueButtonText}>{isSaving ? 'Saving' : 'Start scouting'}</Text>
        </PressableScale>
      </Animated.View>
    </>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        {...scrollProps}
        style={styles.flex1}
        contentContainerStyle={[styles.container, { paddingBottom: 40 + keyboardHeight }]}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="interactive"
        showsVerticalScrollIndicator={false}
      >
        {renderProgressBar()}

        {currentStep === 1 && renderStep1()}
        {currentStep === 2 && renderStep2()}
        {currentStep === 3 && renderStep3()}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  flex1: { flex: 1 },
  container: { paddingHorizontal: theme.spacing.lg, paddingTop: 20 },

  progressContainer: { marginBottom: 28 },
  progressHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 12 },
  stepText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[600], letterSpacing: 1 },
  stepTitle: { fontSize: 13, color: theme.colors.text.muted, fontWeight: '500' },
  progressBarWrapper: { flexDirection: 'row', gap: 8 },
  progressSegment: { flex: 1, height: 4, borderRadius: 2, backgroundColor: theme.colors.bg.divider, overflow: 'hidden' },
  progressFill: { height: 4, borderRadius: 2, backgroundColor: theme.colors.brand[600] },

  headingIconWrap: { alignSelf: 'flex-start', marginBottom: 16 },
  headingIcon: {
    width: 52,
    height: 52,
    borderRadius: 16,
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1,
    borderColor: theme.colors.brand[100],
    alignItems: 'center',
    justifyContent: 'center',
    ...theme.shadows.active,
  },
  title: { fontSize: 28, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.5, marginBottom: 8 },
  description: { fontSize: 15, color: theme.colors.text.body, lineHeight: 22, marginBottom: 24 },

  section: { marginBottom: 20 },
  loadingRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 24, justifyContent: 'center' },
  loadingText: { fontSize: 13, color: theme.colors.text.muted },

  // Step 1
  trustBadges: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', marginTop: 16, gap: 6, flexWrap: 'wrap' },
  trustTextHighlight: { fontSize: 12, color: theme.colors.brand[600], fontWeight: '600' },
  trustDot: { fontSize: 12, color: theme.colors.text.muted },
  trustText: { fontSize: 12, color: theme.colors.text.body },

  unlocksLabelWrapper: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.brand[50], alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6, marginBottom: 16 },
  unlocksLabel: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[800], letterSpacing: 0.5 },
  featureRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 12 },
  checkCircle: { width: 20, height: 20, borderRadius: 10, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center', marginRight: 12, marginTop: 2 },
  featureText: { flex: 1, fontSize: 14, color: theme.colors.text.body, lineHeight: 22 },

  linkHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  linkIconWrapper: { width: 32, height: 32, borderRadius: 8, backgroundColor: theme.colors.brand[50], alignItems: 'center', justifyContent: 'center', marginRight: 12 },
  linkTitle: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },
  linkSub: { fontSize: 12, color: theme.colors.text.body, marginTop: 2 },
  inputWithButton: { flexDirection: 'row', backgroundColor: theme.colors.bg.surfaceHover, borderWidth: 1, borderColor: theme.colors.bg.divider, borderRadius: 12, padding: 4 },
  inputError: { borderColor: theme.colors.semantic.danger },
  urlInput: { flex: 1, paddingHorizontal: 12, paddingVertical: 8, fontSize: 14, color: theme.colors.text.display },
  addBtn: { backgroundColor: theme.colors.brand[600], paddingHorizontal: 18, paddingVertical: 10, borderRadius: 8, justifyContent: 'center', alignItems: 'center', minWidth: 64 },
  addBtnText: { color: theme.colors.text.inverse, fontSize: 13, fontWeight: '700' },
  addBtnRow: { flexDirection: 'row', gap: 4 },
  fieldErrorText: { fontSize: 12, color: theme.colors.semantic.danger, marginTop: 8, fontWeight: '600' },
  inlineError: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: -4, marginBottom: 4, justifyContent: 'center' },
  inlineErrorText: { fontSize: 12, color: theme.colors.semantic.danger, fontWeight: '600', flexShrink: 1 },

  footerDual: { flexDirection: 'row', gap: 12 },
  footerSingle: { marginTop: 12 },
  footerPush: { marginTop: 24 },

  linkedinButton: {
    backgroundColor: toneBg('#0A66C2'),
    borderRadius: theme.radii.full,
    paddingVertical: 16,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 12,
    ...theme.shadows.active,
  },
  linkedinButtonText: { color: toneFg('#FFFFFF'), fontSize: 16, fontWeight: '700' },
  fillButton: {
    backgroundColor: theme.colors.brand[50],
    borderWidth: 1.5,
    borderColor: theme.colors.brand[200],
    borderRadius: theme.radii.full,
    paddingVertical: 15,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  fillButtonText: { color: theme.colors.brand[800], fontSize: 15, fontWeight: '700' },

  // Step 2
  profileHeader: { flexDirection: 'row', alignItems: 'center', gap: 14 },
  avatar: { width: 50, height: 50, borderRadius: 25, backgroundColor: theme.colors.brand[100], alignItems: 'center', justifyContent: 'center' },
  avatarText: { fontSize: 18, fontWeight: '700', color: theme.colors.brand[900] },
  profileName: { fontSize: 16, fontWeight: '700', color: theme.colors.text.display },
  profileSub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2 },
  editablePill: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: theme.colors.brand[50], paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999 },
  editablePillText: { fontSize: 11, fontWeight: '700', color: theme.colors.brand[700] },
  profileFields: { marginTop: 18, borderTopWidth: 1, borderTopColor: theme.colors.bg.divider, paddingTop: 16 },

  countPill: { minWidth: 22, textAlign: 'center', fontSize: 11, fontWeight: '800', color: theme.colors.text.inverse, backgroundColor: theme.colors.brand[600], borderRadius: 999, paddingHorizontal: 7, paddingVertical: 2, overflow: 'hidden' },
  countPillRole: { backgroundColor: toneBg('#0d9488') },

  hintRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: 4 },
  hintText: { fontSize: 12, color: theme.colors.text.muted, flexShrink: 1, textAlign: 'center' },

  errorBanner: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginTop: 16 },
  errorBannerText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  backButton: {
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5,
    borderColor: theme.colors.bg.divider,
    borderRadius: theme.radii.full,
    paddingVertical: 14,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
  },
  backButtonText: { color: theme.colors.text.display, fontSize: 15, fontWeight: '700' },
  continueButton: {
    backgroundColor: theme.colors.fill.brand900,
    borderRadius: theme.radii.full,
    paddingVertical: 15,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    ...theme.shadows.active,
  },
  continueButtonText: { color: toneFg('#FFFFFF'), fontSize: 15, fontWeight: '700' },

  // Step 3
  eventCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5,
    borderColor: theme.colors.bg.divider,
    borderRadius: theme.radii.lg,
    padding: 16,
    marginBottom: 10,
  },
  eventCardSelected: { borderColor: theme.colors.brand[500], backgroundColor: theme.colors.brand[50] },
  subLabel: { fontSize: 10, fontWeight: '800', color: theme.colors.text.muted, letterSpacing: 1, marginBottom: 8, marginTop: 4 },
  inputNotice: { borderColor: theme.colors.semantic.warningBorder },
  eventNoticeText: { fontSize: 12, color: theme.colors.semantic.warning, fontWeight: '600', marginTop: 8 },
  myEvents: { marginTop: 14 },
  emptyEvents: { fontSize: 13, color: theme.colors.text.muted, lineHeight: 19 },
  myEventRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5,
    borderColor: theme.colors.bg.divider,
    borderRadius: theme.radii.md,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 8,
  },
  myEventRowActive: { borderColor: theme.colors.brand[500], backgroundColor: theme.colors.brand[50] },
  radio: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: theme.colors.text.muted, alignItems: 'center', justifyContent: 'center' },
  radioActive: { borderColor: theme.colors.brand[600] },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: theme.colors.brand[600] },
  myEventName: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display },
  myEventMeta: { fontSize: 12, color: theme.colors.text.muted, marginTop: 2 },
  myEventMetaActive: { color: theme.colors.brand[700], fontWeight: '600' },
  removeEvent: { width: 28, height: 28, borderRadius: 14, backgroundColor: theme.colors.bg.surfaceHover, alignItems: 'center', justifyContent: 'center' },
  eventName: { fontSize: 16, fontWeight: '800', color: theme.colors.text.display, marginBottom: 2 },
  eventMeta: { fontSize: 13, color: theme.colors.text.body },
  checkCircleLg: { width: 28, height: 28, borderRadius: 14, backgroundColor: theme.colors.brand[600], alignItems: 'center', justifyContent: 'center' },
  checkCircleLgEmpty: { backgroundColor: 'transparent', borderWidth: 2, borderColor: theme.colors.text.muted },

  autoCard: { flexDirection: 'row', alignItems: 'center', padding: 18, borderRadius: theme.radii.lg, borderWidth: 1.5, marginBottom: theme.spacing.md },
  toggleIcon: { width: 36, height: 36, borderRadius: 10, alignItems: 'center', justifyContent: 'center', marginRight: 12 },
  toggleInfo: { flex: 1, paddingRight: 12 },
  toggleTitle: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display, marginBottom: 2 },
  toggleSub: { fontSize: 13, color: theme.colors.text.body, lineHeight: 18 },
  toggleSubOn: { color: theme.colors.brand[700] },
}));
