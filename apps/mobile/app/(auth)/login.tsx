import React, { useState, useEffect, useRef } from 'react';
import { View, ScrollView, Platform, Pressable } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  withDelay,
} from 'react-native-reanimated';
import { Text } from '../../src/ui/Text';
import { Toast } from '../../src/ui/Toast';
import { KeyboardAwareView, useRevealAboveKeyboard } from '../../src/ui/keyboard';
import { Key, Mail } from 'lucide-react-native';
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

// Email clean-up. Passwords are never altered — they're sent exactly as typed.
const sanitizeEmail = (email: string): string => {
  return email
    .trim()
    .toLowerCase()
    .replace(/[<>]/g, '')
    .slice(0, 255);
};

export default function LoginScreen() {
  // Android draws edge-to-edge (the window doesn't shrink for the keyboard): pad by the overlap
  // and scroll the focused field into view.
  const scrollRef = useRef<ScrollView>(null);
  const scrollY = useRef(0);
  useRevealAboveKeyboard(scrollRef, () => scrollY.current);
  const router = useRouter();
  // Set when this device was signed out because the password was reset elsewhere.
  const { signedOut } = useLocalSearchParams<{ signedOut?: string }>();
  const { login } = useSession();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  // Which button started the current request, and what it's doing right now (shown on the button).
  const [via, setVia] = useState<'email' | 'linkedin' | null>(null);
  const [phase, setPhase] = useState('');
  const [errorMsg, setErrorMsg] = useState(
    signedOut === '1' ? 'You were signed out because your password was changed. Please sign in again.' : '',
  );

  // Animations
  const headerTranslateY = useSharedValue(-50);
  const headerOpacity = useSharedValue(0);
  const titleScale = useSharedValue(0.5);
  const titleOpacity = useSharedValue(0);
  const emailTranslateY = useSharedValue(30);
  const emailOpacity = useSharedValue(0);
  const passwordTranslateY = useSharedValue(30);
  const passwordOpacity = useSharedValue(0);
  const buttonScale = useSharedValue(0.8);
  const buttonOpacity = useSharedValue(0);
  const emailBorderColor = useSharedValue(0);
  const passwordBorderColor = useSharedValue(0);

  useEffect(() => {
    headerTranslateY.value = withDelay(100, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    headerOpacity.value = withDelay(100, withTiming(1, { duration: 600 }));
    titleScale.value = withDelay(300, withSpring(1, { damping: 22, stiffness: 100, overshootClamping: true }));
    titleOpacity.value = withDelay(300, withTiming(1, { duration: 600 }));
    emailTranslateY.value = withDelay(500, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    emailOpacity.value = withDelay(500, withTiming(1, { duration: 600 }));
    passwordTranslateY.value = withDelay(700, withSpring(0, { damping: 22, stiffness: 100, overshootClamping: true }));
    passwordOpacity.value = withDelay(700, withTiming(1, { duration: 600 }));
    buttonScale.value = withDelay(900, withSpring(1, { damping: 22, stiffness: 100, overshootClamping: true }));
    buttonOpacity.value = withDelay(900, withTiming(1, { duration: 600 }));
  }, []);

  const headerAnimStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: headerTranslateY.value }],
    opacity: headerOpacity.value,
  }));

  const titleAnimStyle = useAnimatedStyle(() => ({
    transform: [{ scale: titleScale.value }],
    opacity: titleOpacity.value,
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

  const handleEmailFocus = () => {
    emailBorderColor.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true });
  };

  const handleEmailBlur = () => {
    emailBorderColor.value = withSpring(0, { damping: 22, stiffness: 300, overshootClamping: true });
  };

  const handlePasswordFocus = () => {
    passwordBorderColor.value = withSpring(1, { damping: 22, stiffness: 300, overshootClamping: true });
  };

  const handlePasswordBlur = () => {
    passwordBorderColor.value = withSpring(0, { damping: 22, stiffness: 300, overshootClamping: true });
  };

  const handleLinkedInLogin = async () => {
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
      setPhase('Opening your account…');
      await login(session.token, session.refresh_token, {
        name: session.user?.name || '',
        email: session.user?.email || '',
      }, 'linkedin');
      router.replace('/(app)/feed' as any);
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

  const handleLogin = async () => {
    const sanitizedEmail = sanitizeEmail(email);

    if (!sanitizedEmail || !password) {
      setErrorMsg('Please fill in all fields');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(sanitizedEmail)) {
      setErrorMsg('Please enter a valid email address');
      return;
    }

    setLoading(true);
    setVia('email');
    setPhase('Signing in…');
    setErrorMsg('');
    try {
      const response = await api.login(sanitizedEmail, password);
      if (response && response.user_id && response.token) {
        setPhase('Opening your account…');
        const refreshToken = response.refresh_token || '';
        await login(response.token, refreshToken, {
          name: response.user?.name || '',
          email: response.user?.email || ''
        }, 'email');
        router.replace('/(app)/feed' as any);
      } else {
        setErrorMsg('Login failed. Please check your credentials.');
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
          <HeaderPill icon={Key} text="/login" animStyle={headerAnimStyle} />

          <Animated.View style={titleAnimStyle}>
            <Text style={styles.title}>
              Welcome back<Text style={styles.titleDot}>.</Text>
            </Text>
            <Text style={styles.description}>
              Log in to access your badge scanner, exhibitor lists, and event insights.
            </Text>
          </Animated.View>

          <SocialButton
            onPress={handleLinkedInLogin}
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
            label="WORK EMAIL"
            icon={Mail}
            placeholder="you@company.com"
            keyboardType="email-address"
            autoCapitalize="none"
            value={email}
            onChangeText={setEmail}
            onFocus={handleEmailFocus}
            onBlur={handleEmailBlur}
            editable={!loading}
            animStyle={emailAnimStyle}
            borderAnimStyle={emailBorderAnimStyle}
          />

          <PasswordInput
            label="PASSWORD"
            placeholder="••••••••"
            value={password}
            onChangeText={setPassword}
            onFocus={handlePasswordFocus}
            onBlur={handlePasswordBlur}
            editable={!loading}
            animStyle={passwordAnimStyle}
            borderAnimStyle={passwordBorderAnimStyle}
          />

          <AuthButton
            text="Log in"
            onPress={handleLogin}
            disabled={loading}
            loading={via === 'email'}
            loadingText={phase}
            animStyle={buttonAnimStyle}
          />

          <Pressable
            style={styles.forgotPasswordContainer}
            onPress={() => router.push({ pathname: '/forgot-password' as any, params: { email: sanitizeEmail(email) } })}
            disabled={loading}
          >
            <Text style={styles.forgotPasswordText}>Forgot password?</Text>
          </Pressable>
        </ScrollView>

        <View style={styles.bottomArea}>
          <Text style={styles.bottomText}>
            New here?{' '}
            <Text style={styles.bottomLink} onPress={() => router.push('/register' as any)}>Create an account</Text>
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
  titleDot: {
    color: theme.colors.brand[600],
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
  forgotPasswordContainer: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  forgotPasswordText: {
    color: theme.colors.brand[600],
    fontSize: 14,
    fontWeight: '500',
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
