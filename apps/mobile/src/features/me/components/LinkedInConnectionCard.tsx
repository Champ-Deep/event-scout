import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, Pressable, ActivityIndicator } from 'react-native';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';
import { useRouter } from 'expo-router';
import { Linkedin, Lock, KeyRound, Unlink, AlertCircle, Check, ArrowRight } from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { Card } from '../../../ui/Card';
import { theme, toneFg, makeStyles } from '../../../ui/theme';
import { useSheetMotion, SheetOverlay } from '../../../ui/SheetOverlay';
import { api, errorMessage, type ErrorMessageOptions, type LinkedInLinkStatus } from '../../../lib/api';
import { VALIDATION } from '../../../config/constants';
import { beginLinkedInLink } from '../../auth/linkedin';
import { PressableScale, EditableField } from '../../onboarding/components';

type SheetStep = 'setPassword' | 'confirm';

interface LinkedInConnectionCardProps {
  accountEmail: string | null;
  // Error handed back by /callback after a failed web-redirect connect.
  initialError?: string | undefined;
}

// How a failed request is worded here (see errorMessage).
const LINKEDIN_ERRORS: ErrorMessageOptions = { offline: "You're offline. Check your connection and try again." };

/** LinkedIn link / unlink state shared by the card and its disconnect sheet. */
function useLinkedInConnection({ accountEmail, initialError }: LinkedInConnectionCardProps) {
  const [status, setStatus] = useState<LinkedInLinkStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState(false);
  const [cardError, setCardError] = useState(initialError || '');
  const [notice, setNotice] = useState('');

  const [sheetStep, setSheetStep] = useState<SheetStep | null>(null);
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  // Set when the user just created their password in this sheet, so we don't ask again.
  const [justSetPassword, setJustSetPassword] = useState('');
  const [sheetError, setSheetError] = useState('');
  const [working, setWorking] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await api.getLinkedInLinkStatus());
    } catch (e: any) {
      setCardError(errorMessage(e, "Couldn't load your LinkedIn connection.", LINKEDIN_ERRORS));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const handleConnect = async () => {
    setCardError('');
    setNotice('');
    setConnecting(true);
    try {
      // Web: redirects to LinkedIn and /callback returns here. Native: resolves with the result.
      const result = await beginLinkedInLink({ returnTo: '/me' });
      if (!result) return;
      setNotice('LinkedIn connected.');
      await loadStatus();
    } catch (e: any) {
      setCardError(
        e?.response?.status === 501
          ? "LinkedIn sign-in isn't set up on this server yet."
          : errorMessage(e, 'LinkedIn connection failed. Please try again.', LINKEDIN_ERRORS),
      );
    } finally {
      setConnecting(false);
    }
  };

  const openDisconnect = () => {
    setCardError('');
    setNotice('');
    setSheetError('');
    setNewPassword('');
    setConfirmPassword('');
    setCurrentPassword('');
    setJustSetPassword('');
    setSheetStep(status?.has_password ? 'confirm' : 'setPassword');
  };

  const closeSheet = () => {
    if (working) return;
    setSheetStep(null);
  };

  const handleSetPassword = async () => {
    if (newPassword.length < VALIDATION.PASSWORD_MIN_LENGTH) {
      setSheetError(`Password must be at least ${VALIDATION.PASSWORD_MIN_LENGTH} characters`);
      return;
    }
    if (newPassword !== confirmPassword) {
      setSheetError("Passwords don't match");
      return;
    }
    setSheetError('');
    setWorking(true);
    try {
      await api.setPassword(newPassword);
      setStatus((prev) => (prev ? { ...prev, has_password: true } : prev));
      setJustSetPassword(newPassword);
      setSheetStep('confirm');
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't set your password. Please try again.", LINKEDIN_ERRORS));
    } finally {
      setWorking(false);
    }
  };

  const handleDisconnect = async () => {
    const password = justSetPassword || currentPassword;
    if (!password) {
      setSheetError('Enter your password to confirm');
      return;
    }
    setSheetError('');
    setWorking(true);
    try {
      await api.unlinkLinkedIn(password);
      setStatus((prev) => (prev ? { ...prev, connected: false, linkedin_email: null, linked_at: null } : prev));
      setSheetStep(null);
      setNotice(`LinkedIn disconnected. Sign in with ${accountEmail || 'your email'} and your password from now on.`);
    } catch (e: any) {
      setSheetError(errorMessage(e, "Couldn't disconnect LinkedIn. Please try again.", LINKEDIN_ERRORS));
    } finally {
      setWorking(false);
    }
  };

  const connected = !!status?.connected;

  return {
    accountEmail, status, loading, connecting, cardError, notice, connected, handleConnect, openDisconnect,
    sheetStep, setSheetStep, closeSheet, working, sheetError, setSheetError, handleSetPassword, handleDisconnect,
    newPassword, setNewPassword, confirmPassword, setConfirmPassword, currentPassword, setCurrentPassword, justSetPassword,
  };
}

type LinkedInConnection = ReturnType<typeof useLinkedInConnection>;

export function LinkedInConnectionCard(props: LinkedInConnectionCardProps) {
  const link = useLinkedInConnection(props);
  const { status, loading, connecting, cardError, notice, connected, handleConnect, openDisconnect } = link;
  return (
    <>
    <Card>
      <View style={styles.header}>
        <View style={styles.iconBox}>
          <Linkedin size={16} color={theme.colors.social.linkedin} strokeWidth={2.25} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>LINKEDIN</Text>
          {loading ? (
            <Text style={styles.sub}>Checking connection…</Text>
          ) : connected ? (
            <Text style={styles.sub} numberOfLines={1}>
              Connected{status?.linkedin_email ? ` as ${status.linkedin_email}` : ''}
            </Text>
          ) : (
            <Text style={styles.sub}>Not connected</Text>
          )}
        </View>
        {loading ? (
          <ActivityIndicator size="small" color={theme.colors.brand[600]} />
        ) : connected ? (
          <PressableScale
            style={styles.disconnectBtn}
            onPress={openDisconnect}
            leadingIcon={<Unlink size={14} color={theme.colors.semantic.danger} strokeWidth={2.5} />}
            accessibilityRole="button"
          >
            <Text style={styles.disconnectText}>Disconnect</Text>
          </PressableScale>
        ) : (
          <PressableScale
            style={styles.connectBtn}
            onPress={handleConnect}
            disabled={connecting}
            leadingIcon={connecting ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : undefined}
            accessibilityRole="button"
          >
            <Text style={styles.connectText}>{connecting ? 'Connecting' : 'Connect'}</Text>
          </PressableScale>
        )}
      </View>

      {connected && !status?.has_password ? (
        <View style={styles.infoRow}>
          <Lock size={12} color={theme.colors.text.muted} />
          <Text style={styles.infoText}>You sign in with LinkedIn only. You'll set a password before disconnecting.</Text>
        </View>
      ) : null}
      {notice ? (
        <Animated.View entering={FadeInDown.duration(260)} style={styles.noticeRow}>
          <Check size={14} color={theme.colors.semantic.successText} strokeWidth={3} />
          <Text style={styles.noticeText}>{notice}</Text>
        </Animated.View>
      ) : null}
      {cardError ? (
        <Animated.View entering={FadeInDown.duration(260)} style={styles.errorRow}>
          <AlertCircle size={14} color={theme.colors.semantic.danger} />
          <Text style={styles.errorText}>{cardError}</Text>
        </Animated.View>
      ) : null}
    </Card>
    {/* Drawn in the app's window layer (over the whole screen and tab bar), so it can live here in the card. */}
    <LinkedInDisconnectSheet link={link} />
    </>
  );
}

/** The disconnect / set-password sheet (window layer, like the RN <Modal> it replaces). */
function LinkedInDisconnectSheet({ link }: { link: LinkedInConnection }) {
  const router = useRouter();
  const {
    accountEmail, sheetStep: step, setSheetStep, closeSheet, working, sheetError, setSheetError, handleSetPassword, handleDisconnect,
    newPassword, setNewPassword, confirmPassword, setConfirmPassword, currentPassword, setCurrentPassword, justSetPassword,
  } = link;
  // Slides up like the other sheets; can't be closed while a request is running.
  const motion = useSheetMotion({ open: step !== null, onClose: closeSheet, dismissible: !working });
  // Keep the last step on screen while the sheet slides out.
  const lastStep = useRef<SheetStep>('confirm');
  if (step) lastStep.current = step;
  const sheetStep = step ?? lastStep.current;

  return (
    <SheetOverlay motion={motion} backdropColor="rgba(0,0,0,0.4)" zIndex={60} keyboardAware layer="window">
      <Animated.View style={[styles.sheet, motion.sheetStyle]}>
        {sheetStep === 'setPassword' ? (
          <Animated.View key="set" entering={FadeIn.duration(200)}>
            <View style={styles.stepPill}>
              <Text style={styles.stepPillText}>STEP 1 OF 2</Text>
            </View>
            <Text style={styles.sheetTitle}>Set a password first</Text>
            <Text style={styles.sheetBody}>
              You signed up with LinkedIn, so your account doesn't have a password yet. Set one so you can still sign in
              with {accountEmail || 'your email'} after LinkedIn is disconnected.
            </Text>
            <EditableField
              icon={<KeyRound size={16} color={theme.colors.brand[600]} />}
              label="New password"
              value={newPassword}
              onChangeText={(t) => { setNewPassword(t); if (sheetError) setSheetError(''); }}
              placeholder={`At least ${VALIDATION.PASSWORD_MIN_LENGTH} characters`}
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
              textContentType="newPassword"
              autoComplete="new-password"
            />
            <EditableField
              icon={<KeyRound size={16} color={theme.colors.brand[600]} />}
              label="Confirm password"
              value={confirmPassword}
              onChangeText={(t) => { setConfirmPassword(t); if (sheetError) setSheetError(''); }}
              placeholder="Type it again"
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
              textContentType="newPassword"
              autoComplete="new-password"
              onSubmitEditing={handleSetPassword}
              returnKeyType="done"
            />
          </Animated.View>
        ) : (
          <Animated.View key="confirm" entering={FadeIn.duration(200)}>
            {justSetPassword ? (
              <View style={styles.stepPill}>
                <Text style={styles.stepPillText}>STEP 2 OF 2</Text>
              </View>
            ) : null}
            <Text style={styles.sheetTitle}>Disconnect LinkedIn?</Text>
            {justSetPassword ? (
              <View style={styles.noticeRow}>
                <Check size={14} color={theme.colors.semantic.successText} strokeWidth={3} />
                <Text style={styles.noticeText}>Password set.</Text>
              </View>
            ) : null}
            <Text style={styles.sheetBody}>
              You'll sign in with {accountEmail || 'your email'} and your password. Signing in with LinkedIn won't open
              this account until you connect it again. Your contacts and profile stay as they are.
            </Text>
            {!justSetPassword ? (
              <EditableField
                icon={<Lock size={16} color={theme.colors.brand[600]} />}
                label="Your password"
                value={currentPassword}
                onChangeText={(t) => { setCurrentPassword(t); if (sheetError) setSheetError(''); }}
                placeholder="Enter your password to confirm"
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                textContentType="password"
                autoComplete="current-password"
                onSubmitEditing={handleDisconnect}
                returnKeyType="done"
              />
            ) : null}
            {!justSetPassword ? (
              <Pressable
                onPress={() => {
                  setSheetStep(null);
                  router.push({ pathname: '/forgot-password' as any, params: { email: accountEmail || '' } });
                }}
                disabled={working}
                hitSlop={8}
              >
                <Text style={styles.forgotLink}>Forgot your password?</Text>
              </Pressable>
            ) : null}
          </Animated.View>
        )}

        {sheetError ? (
          <Animated.View entering={FadeInDown.duration(260)} style={styles.errorRow}>
            <AlertCircle size={14} color={theme.colors.semantic.danger} />
            <Text style={styles.errorText}>{sheetError}</Text>
          </Animated.View>
        ) : null}

        <View style={styles.sheetActions}>
          <PressableScale
            containerStyle={{ flex: 1 }}
            style={styles.cancelBtn}
            onPress={closeSheet}
            disabled={working}
            accessibilityRole="button"
          >
            <Text style={styles.cancelText}>Cancel</Text>
          </PressableScale>
          {sheetStep === 'setPassword' ? (
            <PressableScale
              containerStyle={{ flex: 2 }}
              style={styles.primaryBtn}
              onPress={handleSetPassword}
              disabled={working}
              trailingIcon={working ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <ArrowRight size={16} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
              accessibilityRole="button"
            >
              <Text style={styles.primaryText}>Set password</Text>
            </PressableScale>
          ) : (
            <PressableScale
              containerStyle={{ flex: 2 }}
              style={styles.dangerBtn}
              onPress={handleDisconnect}
              disabled={working}
              leadingIcon={working ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <Unlink size={16} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
              accessibilityRole="button"
            >
              <Text style={styles.primaryText}>Disconnect</Text>
            </PressableScale>
          )}
        </View>
      </Animated.View>
    </SheetOverlay>
  );
}

const styles = makeStyles(() => ({
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  iconBox: { width: 32, height: 32, borderRadius: 10, backgroundColor: theme.colors.semantic.infoBg, alignItems: 'center', justifyContent: 'center' },
  title: { fontSize: 12, fontWeight: '800', color: theme.colors.text.display, letterSpacing: 1 },
  sub: { fontSize: 13, color: theme.colors.text.body, marginTop: 2 },

  connectBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.social.linkedin, paddingHorizontal: 16, paddingVertical: 9, borderRadius: 999 },
  connectText: { color: toneFg('#FFFFFF'), fontSize: 13, fontWeight: '700' },
  disconnectBtn: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.bg.surface, borderWidth: 1.5, borderColor: theme.colors.semantic.dangerBorder, paddingHorizontal: 12, paddingVertical: 8, borderRadius: 999 },
  disconnectText: { color: theme.colors.semantic.danger, fontSize: 13, fontWeight: '700' },

  infoRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 12 },
  infoText: { flex: 1, fontSize: 12, color: theme.colors.text.muted },
  noticeRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 12 },
  noticeText: { flex: 1, fontSize: 13, color: theme.colors.semantic.successText, fontWeight: '600' },
  errorRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 12 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  sheet: { backgroundColor: theme.colors.bg.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: theme.spacing.lg, paddingBottom: 36 },
  stepPill: { alignSelf: 'flex-start', backgroundColor: theme.colors.brand[50], paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6, marginBottom: 10 },
  stepPillText: { fontSize: 11, fontWeight: '800', color: theme.colors.brand[800], letterSpacing: 0.5 },
  sheetTitle: { fontSize: 20, fontWeight: '900', color: theme.colors.text.display, marginBottom: 8 },
  sheetBody: { fontSize: 14, color: theme.colors.text.body, lineHeight: 21, marginTop: 4, marginBottom: 16 },
  sheetActions: { flexDirection: 'row', gap: 12, marginTop: 20 },
  cancelBtn: { borderWidth: 1.5, borderColor: theme.colors.bg.divider, borderRadius: 999, paddingVertical: 14, alignItems: 'center', justifyContent: 'center' },
  cancelText: { fontSize: 15, fontWeight: '700', color: theme.colors.text.display },
  primaryBtn: { flexDirection: 'row', gap: 8, backgroundColor: theme.colors.fill.brand900, borderRadius: 999, paddingVertical: 14, alignItems: 'center', justifyContent: 'center' },
  dangerBtn: { flexDirection: 'row', gap: 8, backgroundColor: theme.colors.semantic.danger, borderRadius: 999, paddingVertical: 14, alignItems: 'center', justifyContent: 'center' },
  primaryText: { fontSize: 15, fontWeight: '700', color: toneFg('#FFFFFF') },
  forgotLink: { fontSize: 13, fontWeight: '600', color: theme.colors.brand[600], marginTop: 2 },
}));
