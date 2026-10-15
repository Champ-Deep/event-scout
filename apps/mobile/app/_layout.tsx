import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Appearance, StyleSheet, View } from 'react-native';
import { Stack, ThemeProvider, DarkTheme, DefaultTheme, useGlobalSearchParams, usePathname, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SessionProvider } from '../src/providers/SessionProvider';
import { QueryProvider } from '../src/providers/QueryProvider';
import { theme, getColorScheme, getAppearanceChoice, applyColorScheme, onColorSchemeChange } from '../src/ui/theme';
import { ThemeTransitionHost } from '../src/ui/ThemeSwitch';

const StackLayout = Stack as unknown as React.ComponentType<{ screenOptions?: any; children?: React.ReactNode }>;

// Navigation colours (screen background behind transitions, etc.). Light keeps the default theme
// it always had; dark uses the app's dark palette so nothing flashes white between screens.
function navTheme(scheme: 'light' | 'dark') {
  if (scheme === 'light') return DefaultTheme;
  return {
    ...DarkTheme,
    colors: {
      ...DarkTheme.colors,
      primary: theme.colors.brand[600],
      background: theme.colors.bg.app,
      card: theme.colors.bg.surface,
      text: theme.colors.text.display,
      border: theme.colors.bg.divider,
    },
  };
}

export default function RootLayout() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useGlobalSearchParams();
  const [scheme, setScheme] = useState<'light' | 'dark'>(getColorScheme());
  // Where the user is, so the redraw after a light / dark switch stays on the same page.
  const where = useRef({ pathname, params });
  where.current = { pathname, params };
  const restore = useRef<{ pathname: string; params: Record<string, any> } | null>(null);

  // Light / dark switch in place — no app reload: the palette and every style sheet update, then
  // the screens redraw (the ThemeSwitch fade covers it). Sign-in, loaded data and the app stay.
  useEffect(() => onColorSchemeChange((next) => {
    restore.current = { ...where.current };
    setScheme(next);
  }), []);

  // If the redraw landed on another page, go back to the one that was open.
  useEffect(() => {
    const r = restore.current;
    if (!r) return;
    restore.current = null;
    const id = setTimeout(() => {
      if (where.current.pathname === r.pathname) return;
      try { router.replace({ pathname: r.pathname as any, params: r.params }); } catch { /* stays on the start page */ }
    }, 30);
    return () => clearTimeout(id);
  }, [scheme]);

  // Following the phone (Me → Appearance → System): its Light / Dark change applies live too.
  useEffect(() => {
    const sub = Appearance.addChangeListener(({ colorScheme: next }) => {
      if (getAppearanceChoice() === 'system') applyColorScheme(next === 'dark' ? 'dark' : 'light');
    });
    return () => sub.remove();
  }, []);

  const nav = useMemo(() => navTheme(scheme), [scheme]);
  const dark = scheme === 'dark';

  return (
    // Needed for gestures such as swipe-down-to-close on the share dialog.
    <GestureHandlerRootView style={[styles.root, dark && { backgroundColor: theme.colors.bg.app }]}>
      <ThemeProvider value={nav}>
        <StatusBar style={dark ? 'light' : 'dark'} />
        <QueryProvider>
          <SessionProvider>
            {/* key = mode: screens remount with the new palette */}
            <View key={scheme} style={styles.root}>
              <StackLayout screenOptions={{ headerShown: false }} />
            </View>
          </SessionProvider>
        </QueryProvider>
        {/* Fade between light and dark when the mode is switched */}
        <ThemeTransitionHost />
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({ root: { flex: 1 } });
