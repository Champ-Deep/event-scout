import { Appearance, Platform } from 'react-native';

/**
 * Light / Dark / System preference (Me page → header toggle + Appearance card). Read synchronously
 * at start-up by theme.ts — styles are built once from it — so it's kept in SecureStore's sync API
 * (MMKV is in-memory in Expo Go and would forget it). Switching reloads the app (behind a fade,
 * see ThemeSwitch.tsx) so every screen redraws in the new mode.
 */
export type AppearancePref = 'system' | 'light' | 'dark';

const KEY = 'eventscout.appearance';
const FX_KEY = 'eventscout.appearance.fx'; // "a switch just happened" → fade in after the reload

/** App background per mode (same values as theme.ts) — used by the switch fade. */
export const APP_BG = { light: '#fcfaff', dark: '#0d0c14' } as const;

function store(key: string): { get: () => string | null; set: (v: string) => void; del: () => void } {
  if (Platform.OS === 'web') {
    return {
      get: () => { try { return window.localStorage.getItem(key); } catch { return null; } },
      set: (v) => { try { window.localStorage.setItem(key, v); } catch { /* private mode */ } },
      del: () => { try { window.localStorage.removeItem(key); } catch { /* */ } },
    };
  }
  const SecureStore = require('expo-secure-store');
  return {
    get: () => { try { return SecureStore.getItem(key); } catch { return null; } },
    set: (v) => { try { SecureStore.setItem(key, v); } catch { /* keychain unavailable */ } },
    del: () => { try { SecureStore.deleteItemAsync(key); } catch { /* */ } },
  };
}

export function getAppearancePref(): AppearancePref {
  const v = store(KEY).get();
  // Light is the default until the user picks Dark or System on the Me page.
  return v === 'dark' || v === 'system' ? v : 'light';
}

export function saveAppearancePref(pref: AppearancePref) {
  store(KEY).set(pref);
}

/** The scheme the app should use now: the saved choice, or the phone's setting. */
export function resolveColorScheme(pref: AppearancePref = getAppearancePref()): 'light' | 'dark' {
  if (pref !== 'system') return pref;
  return Appearance.getColorScheme() === 'dark' ? 'dark' : 'light';
}

// No Appearance.setColorScheme() here: on Android it recreates the activity, which crashed Expo Go
// while the app was loading ("Expo Go keeps stopping"). The app's colours come from its own theme.

/** Remember that a switch is in progress, so the next start fades in from the new background. */
export function markSwitchFx(on: boolean) {
  if (on) store(FX_KEY).set('1'); else store(FX_KEY).del();
}
/** True once, right after a theme switch reload. */
export function consumeSwitchFx(): boolean {
  const s = store(FX_KEY);
  const on = s.get() === '1';
  if (on) s.del();
  return on;
}
