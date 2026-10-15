import React, { useEffect, useRef, useState } from 'react';
import { View, ActivityIndicator, Platform } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSession } from '../../src/providers/SessionProvider';
import { consumeWebOAuthRequest } from '../../src/features/auth';
import { api, errorMessage } from '../../src/lib/api';
import { Text } from '../../src/ui/Text';
import { tokens } from '../../src/ui/tokens';

import { makeStyles } from '../../src/ui/theme';
/**
 * LinkedIn OAuth redirect target (web full-page flow).
 *
 * LinkedIn redirects the browser here with `?code=...&state=...`. We verify the
 * state against the value stored before the redirect, exchange the code for a real
 * JWT session, persist it via SessionProvider, then route into the app.
 */
export default function CallbackScreen() {
  const { code, state, error, error_description } = useLocalSearchParams<{
    code?: string;
    state?: string;
    error?: string;
    error_description?: string;
  }>();
  const { login } = useSession();
  const router = useRouter();
  const [message, setMessage] = useState('Finalizing authentication…');
  // Guard against the effect running twice (React 18 strict mode / re-renders).
  const handled = useRef(false);

  useEffect(() => {
    if (handled.current) return;
    handled.current = true;

    async function completeAuth() {
      // iOS/Android: the in-app auth session (src/features/auth/linkedin.ts) already received
      // eventscout://callback and exchanges the code itself. If the OS also opened this route,
      // don't use the one-time code a second time — just step back to where the sign-in started.
      if (Platform.OS !== 'web') {
        if (router.canGoBack()) router.back();
        else router.replace('/' as any);
        return;
      }

      if (error) {
        setMessage('Sign-in was cancelled. Redirecting…');
        router.replace('/login' as any);
        return;
      }

      if (!code) {
        setMessage('Missing authorization code. Redirecting…');
        router.replace('/login' as any);
        return;
      }

      // CSRF protection: the state LinkedIn echoes back must match what we stored.
      const { state: expectedState, intent, returnTo } = consumeWebOAuthRequest();
      // Where a link attempt goes back to: the caller's screen, or onboarding by default.
      const linkFailureRoute = (linkError: string): any =>
        returnTo
          ? { pathname: returnTo, params: { linkError } }
          : { pathname: '/onboarding/[step]', params: { step: '1', linkError } };

      if (expectedState && state && expectedState !== state) {
        setMessage('Security check failed. Redirecting…');
        router.replace(intent === 'link' ? linkFailureRoute('LinkedIn failed a security check. Please try again.') : ('/login' as any));
        return;
      }

      if (intent === 'link') {
        // Connecting LinkedIn to the account that is already signed in — no new session.
        setMessage('Connecting LinkedIn…');
        try {
          const result = await api.linkLinkedIn(code);
          router.replace(
            returnTo
              ? ({ pathname: returnTo, params: { linked: '1' } } as any)
              : {
                  pathname: '/onboarding/[step]' as any,
                  params: { step: '2', name: result.linkedin?.name || result.user?.name || '', linked: '1' },
                },
          );
        } catch (err: any) {
          router.replace(linkFailureRoute(errorMessage(err, 'LinkedIn connection failed. Please try again.')));
        }
        return;
      }

      try {
        const session = await api.exchangeLinkedInCode(code);
        await login(session.token, session.refresh_token, {
          name: session.user?.name || '',
          email: session.user?.email || '',
        }, 'linkedin');

        // LinkedIn is already connected, so new users start at the profile step.
        const isNewUser = /created/i.test(session.message || '');
        if (isNewUser) {
          router.replace({
            pathname: '/onboarding/[step]' as any,
            params: { step: '2', name: session.user?.name || '' },
          });
        } else {
          router.replace('/(app)/feed' as any);
        }
      } catch {
        setMessage('LinkedIn sign-in failed. Redirecting…');
        router.replace('/login' as any);
      }
    }

    completeAuth();
  }, [code, state, error, error_description, login, router]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={tokens.colors.accent.emerald} />
      <Text variant="body" color="secondary" style={styles.text}>
        {message}
      </Text>
    </View>
  );
}

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: tokens.colors.bg.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    marginTop: tokens.spacing.md,
  },
}));
