import React, { useEffect, useRef, useState } from 'react';
import { View, ScrollView, TextInput, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  FadeIn,
  FadeInDown,
  ZoomIn,
  useSharedValue,
  useAnimatedStyle,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import { KeyRound, Mail, ArrowLeft, ArrowRight, AlertCircle, Check, RotateCw } from 'lucide-react-native';
import { Text } from '../../src/ui/Text';
import { theme, toneFg, makeStyles } from '../../src/ui/theme';
import { api, errorMessage, type ErrorMessageOptions } from '../../src/lib/api';
import { VALIDATION } from '../../src/config/constants';
import { useSession } from '../../src/providers/SessionProvider';
import { HeaderPill, PasswordInput } from '../../src/components/auth';
import { PressableScale, EditableField, useKeyboardAwareScroll } from '../../src/features/onboarding/components';

const CODE_LENGTH = 6;
type Stage = 'email' | 'code';

// How this screen words a failed request (see errorMessage).
const RESET_ERRORS: ErrorMessageOptions = {
  offline: "You're offline. Check your connection and try again.",
  notAvailable: 'Password reset by email is not set up on the server yet.',
  // FastAPI validation errors come back as a list.
  invalid: 'Please enter a valid email address.',
};

// Six boxes over one hidden input, so paste and SMS/email autofill still work.
function CodeInput({ value, onChange, onFocus, hasError }: { value: string; onChange: (v: string) => void; onFocus?: () => void; hasError: boolean }) {
  const inputRef = useRef<TextInput>(null);
  const [focused, setFocused] = useState(false);
  const shake = useSharedValue(0);

  useEffect(() => {
    if (hasError) {
      shake.value = withSequence(
        withTiming(-8, { duration: 50 }),
        withTiming(8, { duration: 50 }),
        withTiming(-5, { duration: 50 }),
        withTiming(0, { duration: 50 }),
      );
    }
  }, [hasError]);
  const shakeStyle = useAnimatedStyle(() => ({ transform: [{ translateX: shake.value }] }));

  return (
    <Pressable onPress={() => inputRef.current?.focus()} accessibilityLabel="Reset code">
      <Animated.View style={[styles.codeRow, shakeStyle]}>
        {Array.from({ length: CODE_LENGTH }, (_, i) => {
          const digit = value[i];
          const active = focused && i === Math.min(value.length, CODE_LENGTH - 1);
          return (
            <View
              key={i}
              style={[
                styles.codeBox,
                digit ? styles.codeBoxFilled : null,
                active ? styles.codeBoxActive : null,
                hasError ? styles.codeBoxError : null,
              ]}
            >
              {digit ? (
                <Animated.Text key={`${i}-${digit}`} entering={ZoomIn.duration(260)} style={styles.codeDigit}>
                  {digit}
                </Animated.Text>
              ) : null}
            </View>
          );
        })}
      </Animated.View>
      <TextInput
        ref={inputRef}
        value={value}
        onChangeText={(t) => onChange(t.replace(/\D/g, '').slice(0, CODE_LENGTH))}
        onFocus={() => { setFocused(true); onFocus?.(); }}
        onBlur={() => setFocused(false)}
        keyboardType="number-pad"
        textContentType="oneTimeCode"
        autoComplete="one-time-code"
        maxLength={CODE_LENGTH}
        style={styles.hiddenInput}
        caretHidden
      />
    </Pressable>
  );
}

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ email?: string }>();
  const { login } = useSession();
  const { scrollProps, keyboardHeight, onInputFocus } = useKeyboardAwareScroll();

  const [stage, setStage] = useState<Stage>('email');
  const [email, setEmail] = useState(params.email || '');
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [codeError, setCodeError] = useState(false);
  const [notice, setNotice] = useState('');
  const [resendIn, setResendIn] = useState(0);

  useEffect(() => {
    if (resendIn <= 0) return;
    const t = setTimeout(() => setResendIn((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [resendIn]);

  const normalizedEmail = email.trim().toLowerCase();

  const sendCode = async (isResend = false) => {
    if (!VALIDATION.EMAIL_REGEX.test(normalizedEmail)) {
      setError('Please enter a valid email address');
      return;
    }
    setError('');
    setWorking(true);
    try {
      const res = await api.requestPasswordReset(normalizedEmail);
      setResendIn(res?.resend_after ?? 60);
      setNotice(isResend ? 'A new code is on its way. Codes from earlier emails no longer work.' : '');
      setCode('');
      setCodeError(false);
      setStage('code');
    } catch (e: any) {
      setError(errorMessage(e, "Couldn't send the code. Please try again.", RESET_ERRORS));
    } finally {
      setWorking(false);
    }
  };

  const resetPassword = async () => {
    if (code.length !== CODE_LENGTH) {
      setCodeError(true);
      setError(`Enter the ${CODE_LENGTH}-digit code from the email`);
      return;
    }
    if (newPassword.length < VALIDATION.PASSWORD_MIN_LENGTH) {
      setError(`Password must be at least ${VALIDATION.PASSWORD_MIN_LENGTH} characters`);
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("Passwords don't match");
      return;
    }
    setError('');
    setCodeError(false);
    setWorking(true);
    try {
      const session = await api.resetPassword(normalizedEmail, code, newPassword);
      await login(session.token, session.refresh_token, {
        name: session.user.name || '',
        email: session.user.email || '',
      }, 'email');
      router.replace('/(app)/feed' as any);
    } catch (e: any) {
      const status = e?.response?.status;
      if (status === 400 || status === 429) setCodeError(true);
      if (status === 429) setCode('');
      setError(errorMessage(e, "Couldn't reset your password. Please try again.", RESET_ERRORS));
    } finally {
      setWorking(false);
    }
  };

  const goBack = () => {
    if (stage === 'code') {
      setStage('email');
      setError('');
      setNotice('');
      return;
    }
    if (router.canGoBack()) router.back();
    else router.replace('/login' as any);
  };

  const errorBanner = error ? (
    <Animated.View entering={FadeInDown.duration(260)} style={styles.errorBanner}>
      <AlertCircle size={16} color={theme.colors.semantic.danger} />
      <Text style={styles.errorText}>{error}</Text>
    </Animated.View>
  ) : null;

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView
        {...scrollProps}
        contentContainerStyle={[styles.container, { paddingBottom: 40 + keyboardHeight }]}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="interactive"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topRow}>
          <PressableScale
            style={styles.backButton}
            onPress={goBack}
            disabled={working}
            leadingIcon={<ArrowLeft size={16} color={theme.colors.text.display} strokeWidth={2.5} />}
            accessibilityRole="button"
          >
            <Text style={styles.backText}>Back</Text>
          </PressableScale>
          <HeaderPill icon={KeyRound} text="/reset-password" iconColor={theme.colors.text.inverse} />
        </View>

        {stage === 'email' ? (
          <Animated.View key="email" entering={FadeIn.duration(250)}>
            <Animated.Text entering={FadeInDown.delay(40).duration(260)} style={styles.title}>
              Forgot your password<Text style={styles.titleDot}>?</Text>
            </Animated.Text>
            <Animated.Text entering={FadeInDown.delay(100).duration(260)} style={styles.description}>
              Enter your account email and we'll send you a 6-digit code to set a new password.
            </Animated.Text>

            <Animated.View entering={FadeInDown.delay(160).duration(260)}>
              <EditableField
                icon={<Mail size={16} color={theme.colors.brand[600]} />}
                label="Work email"
                value={email}
                onChangeText={(t) => { setEmail(t); if (error) setError(''); }}
                placeholder="you@company.com"
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
                autoComplete="email"
                textContentType="emailAddress"
                returnKeyType="send"
                onSubmitEditing={() => sendCode()}
                onFocus={onInputFocus}
              />
            </Animated.View>

            {errorBanner}

            <Animated.View entering={FadeInDown.delay(220).duration(260)}>
              <PressableScale
                style={styles.primaryButton}
                onPress={() => sendCode()}
                disabled={working}
                trailingIcon={working ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <ArrowRight size={18} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
                accessibilityRole="button"
              >
                <Text style={styles.primaryText}>{working ? 'Sending' : 'Send code'}</Text>
              </PressableScale>
            </Animated.View>
          </Animated.View>
        ) : (
          <Animated.View key="code" entering={FadeIn.duration(250)}>
            <Animated.Text entering={FadeInDown.delay(40).duration(260)} style={styles.title}>
              Check your email<Text style={styles.titleDot}>.</Text>
            </Animated.Text>
            <Animated.Text entering={FadeInDown.delay(100).duration(260)} style={styles.description}>
              If an account exists for <Text style={styles.bold}>{normalizedEmail}</Text>, we've sent it a 6-digit code.
              It expires in 15 minutes.
            </Animated.Text>

            {notice ? (
              <Animated.View entering={FadeInDown.duration(260)} style={styles.noticeRow}>
                <Check size={14} color={theme.colors.semantic.successText} strokeWidth={3} />
                <Text style={styles.noticeText}>{notice}</Text>
              </Animated.View>
            ) : null}

            <Animated.View entering={FadeInDown.delay(160).duration(260)} style={styles.section}>
              <Text style={styles.label}>RESET CODE</Text>
              <CodeInput
                value={code}
                onChange={(v) => { setCode(v); if (codeError) setCodeError(false); if (error) setError(''); }}
                onFocus={onInputFocus}
                hasError={codeError}
              />
              <View style={styles.resendRow}>
                {resendIn > 0 ? (
                  <Text style={styles.resendMuted}>Resend code in {resendIn}s</Text>
                ) : (
                  <PressableScale
                    style={styles.resendButton}
                    onPress={() => sendCode(true)}
                    disabled={working}
                    leadingIcon={<RotateCw size={13} color={theme.colors.brand[700]} strokeWidth={2.5} />}
                    accessibilityRole="button"
                  >
                    <Text style={styles.resendText}>Resend code</Text>
                  </PressableScale>
                )}
              </View>
            </Animated.View>

            <Animated.View entering={FadeInDown.delay(220).duration(260)}>
              <PasswordInput
                label="NEW PASSWORD"
                placeholder={`At least ${VALIDATION.PASSWORD_MIN_LENGTH} characters`}
                value={newPassword}
                onChangeText={(t) => { setNewPassword(t); if (error) setError(''); }}
                onFocus={onInputFocus}
                showStrengthIndicator
                passwordLength={newPassword.length}
                minLength={VALIDATION.PASSWORD_MIN_LENGTH}
                autoCapitalize="none"
                autoCorrect={false}
                textContentType="newPassword"
                autoComplete="new-password"
                borderAnimStyle={styles.passwordBorder}
              />
              <PasswordInput
                label="CONFIRM NEW PASSWORD"
                placeholder="Type it again"
                value={confirmPassword}
                onChangeText={(t) => { setConfirmPassword(t); if (error) setError(''); }}
                onFocus={onInputFocus}
                autoCapitalize="none"
                autoCorrect={false}
                textContentType="newPassword"
                autoComplete="new-password"
                returnKeyType="done"
                onSubmitEditing={resetPassword}
                borderAnimStyle={styles.passwordBorder}
              />
            </Animated.View>

            {errorBanner}

            <Animated.View entering={FadeInDown.delay(280).duration(260)}>
              <PressableScale
                style={styles.primaryButton}
                onPress={resetPassword}
                disabled={working}
                trailingIcon={working ? <ActivityIndicator size="small" color={toneFg('#FFFFFF')} /> : <ArrowRight size={18} color={toneFg('#FFFFFF')} strokeWidth={2.5} />}
                accessibilityRole="button"
              >
                <Text style={styles.primaryText}>{working ? 'Resetting' : 'Reset password & sign in'}</Text>
              </PressableScale>
              <Pressable onPress={goBack} style={styles.linkRow} disabled={working}>
                <Text style={styles.linkText}>Use a different email</Text>
              </Pressable>
            </Animated.View>
          </Animated.View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: { flex: 1, backgroundColor: theme.colors.bg.app },
  container: { paddingHorizontal: theme.spacing.lg, paddingTop: 24 },
  topRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: theme.colors.bg.surface,
    borderWidth: 1.5,
    borderColor: theme.colors.bg.divider,
    borderRadius: theme.radii.full,
    paddingVertical: 8,
    paddingHorizontal: 14,
    marginBottom: theme.spacing.xl,
  },
  backText: { fontSize: 14, fontWeight: '700', color: theme.colors.text.display },

  title: { fontSize: 30, fontWeight: '900', color: theme.colors.text.display, letterSpacing: -0.5, marginBottom: 10 },
  titleDot: { color: theme.colors.brand[600] },
  description: { fontSize: 15, color: theme.colors.text.body, lineHeight: 22, marginBottom: 24 },
  bold: { fontWeight: '700', color: theme.colors.text.display },

  section: { marginBottom: 20 },
  label: { fontSize: 11, fontWeight: '700', color: theme.colors.text.display, letterSpacing: 1, marginBottom: 8 },

  codeRow: { flexDirection: 'row', gap: 8 },
  codeBox: {
    flex: 1,
    height: 56,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: theme.colors.bg.divider,
    backgroundColor: theme.colors.bg.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  codeBoxFilled: { borderColor: theme.colors.brand[300], backgroundColor: theme.colors.brand[50] },
  codeBoxActive: { borderColor: theme.colors.brand[600], borderWidth: 2 },
  codeBoxError: { borderColor: theme.colors.semantic.danger },
  codeDigit: { fontSize: 24, fontWeight: '800', color: theme.colors.text.display, fontFamily: theme.typography.fontFamily.mono },
  hiddenInput: { position: 'absolute', width: 1, height: 1, opacity: 0 },

  resendRow: { flexDirection: 'row', justifyContent: 'flex-end', marginTop: 10, minHeight: 32, alignItems: 'center' },
  resendMuted: { fontSize: 13, color: theme.colors.text.muted },
  resendButton: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: theme.colors.brand[50], paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999 },
  resendText: { fontSize: 13, fontWeight: '700', color: theme.colors.brand[700] },

  passwordBorder: { borderWidth: 1, borderColor: theme.colors.bg.divider },

  noticeRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 16 },
  noticeText: { flex: 1, fontSize: 13, color: theme.colors.semantic.successText, fontWeight: '600' },
  errorBanner: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: theme.colors.semantic.dangerBg, borderRadius: 12, padding: 12, marginBottom: 16 },
  errorText: { flex: 1, fontSize: 13, color: theme.colors.semantic.danger, fontWeight: '600' },

  primaryButton: {
    backgroundColor: theme.colors.fill.brand900,
    borderRadius: theme.radii.full,
    paddingVertical: 16,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: 8,
    ...theme.shadows.active,
  },
  primaryText: { color: toneFg('#FFFFFF'), fontSize: 16, fontWeight: '700' },
  linkRow: { alignItems: 'center', paddingVertical: 16 },
  linkText: { color: theme.colors.brand[600], fontSize: 14, fontWeight: '600' },
}));
