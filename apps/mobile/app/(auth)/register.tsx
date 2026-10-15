import React, { useState, useEffect, useRef } from 'react';
import { View, ScrollView, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withDelay,
} from 'react-native-reanimated';
import { Text } from '../../src/ui/Text';
import { KeyboardAwareView, useRevealAboveKeyboard } from '../../src/ui/keyboard';
import { Toast } from '../../src/ui/Toast';
import { Scan, User, Mail } from 'lucide-react-native';
import { VALIDATION } from '../../src/config/constants';
import { useSession } from '../../src/providers/SessionProvider';
import { beginLinkedInLogin } from '../../src/features/auth';
import { theme, makeStyles } from '../../src/ui/theme';
import { api, errorMessage } from '../../src/lib/api';
import {
  FormInput,
  PasswordInput,
  SocialButton,
  AuthButton,
  HeaderPill,
} from '../../src/components/auth';

// Input sanitization helpers
const sanitizeInput = (input: string): string => {
  return input
    .trim()
    .replace(/[<>]/g, '')
    .slice(0, 255);
};

const sanitizeEmail = (email: string): string => {
  return email
    .trim()
    .toLowerCase()
    .replace(/[<>]/g, '')
    .slice(0, 255);
};

export default function RegisterScreen() {
  // Android draws edge-to-edge (the window doesn't shrink for the keyboard): pad by the overlap
  // and scroll the focused field into view.
  const scrollRef = useRef<ScrollView>(null);
  const scrollY = useRef(0);
  useRevealAboveKeyboard(scrollRef, () => scrollY.current);
  const router = useRouter();
  const { login } = useSession();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  // Which button started the current request, and what it's doing right now (shown on the button).
  const [via, setVia] = useState<'email' | 'linkedin' | null>(null);
  const [phase, setPhase] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  // Animations
  const headerTranslateY = useSharedValue(-50);
  const headerOpacity = useSharedValue(0);
  const titleScale = useSharedValue(0.5);
  const titleOpacity = useSharedValue(0);
  const nameTranslateY = useSharedValue(30);
  const nameOpacity = useSharedValue(0);
  const emailTranslateY = useSharedValue(30);
  const emailOpacity = useSharedValue(0);
  const passwordTranslateY = useSharedValue(30);
  const passwordOpacity = useSharedValue(0);
  const buttonScale = useSharedValue(0.8);
  const buttonOpacity = useSharedValue(0);
  const nameBorderColor = useSharedValue(0);
  const emailBorderColor = useSharedValue(0);
  const passwordBorderColor = useSharedValue(0);

  useEffect(() => {
    headerTranslateY.value = withDelay(100, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    headerOpacity.value = withDelay(100, withTiming(1, { duration: 600 }));
    titleScale.value = withDelay(300, withSpring(1, { damping: 22, stiffness: 100, overshootClamping: true }));
    titleOpacity.value = withDelay(300, withTiming(1, { duration: 600 }));
    nameTranslateY.value = withDelay(500, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    nameOpacity.value = withDelay(500, withTiming(1, { duration: 600 }));
    emailTranslateY.value = withDelay(700, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    emailOpacity.value = withDelay(700, withTiming(1, { duration: 600 }));
    passwordTranslateY.value = withDelay(900, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    passwordOpacity.value = withDelay(900, withTiming(1, { duration: 600 }));
    buttonScale.value = withDelay(1100, withSpring(1, { damping: 22, stiffness: 100, overshootClamping: true }));
    buttonOpacity.value = withDelay(1100, withTiming(1, { duration: 600 }));
  }, []);

  const headerAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: headerTranslateY.value }],
    opacity: headerOpacity.value,
  }));

  const titleAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: titleScale.value }],
    opacity: titleOpacity.value,
  }));

  const nameAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: nameTranslateY.value }],
    opacity: nameOpacity.value,
  }));

  const emailAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: emailTranslateY.value }],
    opacity: emailOpacity.value,
  }));

  const passwordAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: passwordTranslateY.value }],
    opacity: passwordOpacity.value,
  }));

  const buttonAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: buttonScale.value }],
    opacity: buttonOpacity.value,
  }));

  const FOCUS_BORDER = theme.colors.brand[600]; // plain strings for the animation (a worklet keeps its first copy of the theme)
  const IDLE_BORDER = theme.colors.bg.divider;
  const nameBorderAnimStyle = useAnimatedStyle(() => {
    const color = nameBorderColor.value === 1 ? FOCUS_BORDER : IDLE_BORDER;
    return {
      borderColor: color,
      borderWidth: nameBorderColor.value === 1 ? 2 : 1,
    };
  });

  const emailBorderAnimStyle = useAnimatedStyle(() => {
    const color = emailBorderColor.value === 1 ? FOCUS_BORDER : IDLE_BORDER;
    return {
      borderColor: color,
      borderWidth: emailBorderColor.value === 1 ? 2 : 1,
    };
  });

  const passwordBorderAnimStyle = useAnimatedStyle(() => {
    const color = passwordBorderColor.value === 1 ? FOCUS_BORDER : IDLE_BORDER;
    return {
      borderColor: color,
      borderWidth: passwordBorderColor.value === 1 ? 2 : 1,
    };
  });

  const handleLinkedInRegister = async () => {
    setLoading(true);
    setVia('linkedin');
    setPhase('Connecting to LinkedIn…');
    setErrorMsg('');
    try {
      const session = await beginLinkedInLogin();
      if (!session) {
        setLoading(false);
        return;
      }
      setPhase('Setting up your account…');
      await login(session.token, session.refresh_token, {
        name: session.user?.name || '',
        email: session.user?.email || '',
      }, 'linkedin');

      const isNewUser = /created/i.test(session.message || '');
      if (isNewUser) {
        router.replace({
          pathname: '/onboarding/[step]' as any,
          params: { step: '2', name: session.user?.name || '' },
        });
      } else {
        router.replace('/(app)/feed' as any);
      }
    } catch (error: any) {
      setErrorMsg(errorMessage(error, 'LinkedIn sign-in failed. Please try again.', {
        offline: "You're offline. Check your connection and try again.",
        notAvailable: "LinkedIn sign-in isn't set up on this server yet.",
      }));
    } finally {
      setLoading(false);
      setVia(null);
    }
  };

  const handleRegister = async () => {
    const sanitizedName = sanitizeInput(name);
    const sanitizedEmail = sanitizeEmail(email);
    // The password is sent exactly as typed — never trimmed or stripped.
    if (!sanitizedName || !sanitizedEmail || !password) {
      setErrorMsg('Please fill in all fields');
      return;
    }

    if (password.length < VALIDATION.PASSWORD_MIN_LENGTH) {
      setErrorMsg(`Password must be at least ${VALIDATION.PASSWORD_MIN_LENGTH} characters`);
      return;
    }

    if (!VALIDATION.EMAIL_REGEX.test(sanitizedEmail)) {
      setErrorMsg('Please enter a valid email address');
      return;
    }

    setLoading(true);
    setVia('email');
    setPhase('Creating your account…');
    setErrorMsg('');
    try {
      const response = await api.register(sanitizedName, sanitizedEmail, password);
      if (response && response.user_id && response.token) {
        setPhase('Setting things up…');
        const refreshToken = response.refresh_token || '';
        await login(response.token, refreshToken, {
          name: response.user?.name || '',
          email: response.user?.email || ''
        }, 'email');
        router.replace({ pathname: '/onboarding/[step]' as any, params: { step: '1', name: sanitizedName } });
      } else {
        setErrorMsg('Registration failed');
      }
    } catch (error: any) {
      setErrorMsg(errorMessage(error, 'Network error. Please check your connection.', {
        offline: "You're offline. Check your connection and try again.",
        unreachable: 'Network error. Please check your connection.',
      }));
      setTimeout(() => setErrorMsg(''), 5000);
    } finally {
      setLoading(false);
      setVia(null);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAwareView style={styles.flex1}>
        <ScrollView
          ref={scrollRef}
          onScroll={(e) => { scrollY.current = e.nativeEvent.contentOffset.y; }}
          scrollEventThrottle={16}
          contentContainerStyle={styles.container}
          bounces={false}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          automaticallyAdjustKeyboardInsets={Platform.OS === 'ios'}
        >
          <HeaderPill
            icon={Scan}
            text="/register"
            animStyle={headerAnimStyle}
            iconColor={theme.colors.brand[600]}
            iconBackground="transparent"
          />

          <Animated.View style={titleAnimStyle}>
            <Text style={styles.title}>Create Your Account</Text>
            <Text style={styles.description}>
              Start scanning badges in under a minute. Import exhibitor lists and identify high-value prospects instantly.
            </Text>
          </Animated.View>

          <SocialButton
            onPress={handleLinkedInRegister}
            disabled={loading}
            loading={via === 'linkedin'}
            loadingText={phase}
            animStyle={buttonAnimStyle}
          />

          <View style={styles.dividerContainer}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>OR</Text>
            <View style={styles.dividerLine} />
          </View>

          <FormInput
            label="NAME"
            icon={User}
            placeholder="Your Name"
            value={name}
            onChangeText={setName}
            onFocus={() => nameBorderColor.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true })}
            onBlur={() => nameBorderColor.value = withSpring(0, { damping: 22, stiffness: 300, overshootClamping: true })}
            editable={!loading}
            animStyle={nameAnimStyle}
            borderAnimStyle={nameBorderAnimStyle}
          />

          <FormInput
            label="WORK EMAIL"
            icon={Mail}
            placeholder="you@company.com"
            keyboardType="email-address"
            autoCapitalize="none"
            value={email}
            onChangeText={setEmail}
            onFocus={() => emailBorderColor.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true })}
            onBlur={() => emailBorderColor.value = withSpring(0, { damping: 22, stiffness: 300, overshootClamping: true })}
            editable={!loading}
            animStyle={emailAnimStyle}
            borderAnimStyle={emailBorderAnimStyle}
          />

          <PasswordInput
            label="PASSWORD"
            placeholder="••••••••"
            value={password}
            onChangeText={setPassword}
            onFocus={() => passwordBorderColor.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true })}
            onBlur={() => passwordBorderColor.value = withSpring(0, { damping: 22, stiffness: 300, overshootClamping: true })}
            editable={!loading}
            animStyle={passwordAnimStyle}
            borderAnimStyle={passwordBorderAnimStyle}
            showStrengthIndicator
            passwordLength={password.length}
            minLength={VALIDATION.PASSWORD_MIN_LENGTH}
          />

          <AuthButton
            text="Create account"
            onPress={handleRegister}
            disabled={loading}
            loading={via === 'email'}
            loadingText={phase}
            animStyle={buttonAnimStyle}
          />

          <Text style={styles.termsText}>
            by continuing you accept the terms • we never post to LinkedIn
          </Text>
        </ScrollView>

        <View style={styles.bottomArea}>
          <Text style={styles.bottomText}>
            Already have an account?{' '}
            <Text style={styles.bottomLink} onPress={() => router.push('/login' as any)}>Log in</Text>
          </Text>
        </View>
      </KeyboardAwareView>

      <Toast
        message={errorMsg}
        type="error"
        visible={!!errorMsg}
        onDismiss={() => setErrorMsg('')}
      />
    </SafeAreaView>
  );
}

const styles = makeStyles(() => ({
  safeArea: {
    flex: 1,
    backgroundColor: theme.colors.bg.app,
  },
  flex1: {
    flex: 1,
  },
  container: {
    paddingHorizontal: theme.spacing.lg,
    paddingTop: 40,
    paddingBottom: theme.spacing.lg,
  },
  title: {
    fontSize: theme.typography.sizes.display.fontSize,
    fontWeight: '900',
    color: theme.colors.text.display,
    lineHeight: 38,
    marginBottom: 12,
    letterSpacing: -0.5,
  },
  description: {
    fontSize: theme.typography.sizes.body.fontSize,
    color: theme.colors.text.body,
    lineHeight: 22,
    marginBottom: theme.spacing.xl,
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: theme.colors.bg.divider,
  },
  dividerText: {
    marginHorizontal: theme.spacing.md,
    color: theme.colors.text.muted,
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 1,
  },
  termsText: {
    textAlign: 'center',
    color: theme.colors.text.muted,
    fontSize: 12,
    lineHeight: 18,
  },
  bottomArea: {
    borderTopWidth: 1,
    borderTopColor: theme.colors.bg.divider,
    paddingVertical: 24,
    alignItems: 'center',
    backgroundColor: theme.colors.bg.surface,
  },
  bottomText: {
    color: theme.colors.text.body,
    fontSize: 14,
  },
  bottomLink: {
    color: theme.colors.brand[900],
    fontWeight: '700',
    textDecorationLine: 'underline',
  },
}));
