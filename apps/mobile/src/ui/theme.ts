import { useEffect, useState } from 'react';
import { Platform, StyleSheet } from 'react-native';
import { getAppearancePref, resolveColorScheme, type AppearancePref } from './appearance';

/**
 * UI Architecture System
 * 4-Layer Color System, 4px/8px Grid, 6-Step Typography, Nested Radii
 *
 * Light & dark: the app follows the phone's Light / Dark setting. Styles are created once when
 * screens load, so the palette is picked here at start-up (switching the phone's theme while the
 * app is open reloads it in development; in a store build it applies on the next launch).
 * LIGHT is exactly the original palette. Names mean the same thing in both modes:
 *   bg.*          backgrounds           text.*        text on backgrounds
 *   brand[50–300] tints (chips, borders) brand[700–900] text on tints
 *   fill.*        solid button / toast fills that stay deep in both modes
 *   fg.*          light brand text used ON those fills (and brand-tinted icons)
 * Hard-coded colours go through toneBg() / toneFg(): unchanged in light, adapted in dark.
 */
const LIGHT = {
  // Layer 1: Neutral Foundation & Surface Elevation
  bg: {
    app: '#fcfaff', // Base app background (slightly tinted)
    surface: '#ffffff', // Elevated card surface
    surfaceHover: '#f8fafc',
    divider: '#e2e8f0', // ~85% white border
    dashed: '#c7d2fe',
  },
  text: {
    display: '#0f172a', // ~11% white (dark gray)
    body: '#475569', // 15-20% white
    muted: '#94a3b8', // 30-40% white
    inverse: '#ffffff',
  },
  // Layer 2: Functional Accent Scale (Brand Purple/Indigo)
  brand: {
    50: '#f5f3ff',
    100: '#ede9fe',
    200: '#ddd6fe',
    300: '#c4b5fd',
    400: '#a78bfa',
    500: '#8b5cf6',
    600: '#7c3aed', // Primary links/buttons
    700: '#6d28d9',
    800: '#5b21b6', // Primary active/heavy
    900: '#4c1d95',
    950: '#2e1065',
  },
  // Solid fills (buttons, toasts, the current-event card) — deep in both modes
  fill: {
    brand700: '#6d28d9',
    brand800: '#5b21b6',
    brand900: '#4c1d95',
    brand950: '#2e1065',
    ink: '#0f172a',
  },
  // Light brand text / icons (on fills and on dark surfaces)
  fg: {
    brand50: '#f5f3ff',
    brand100: '#ede9fe',
    brand200: '#ddd6fe',
    brand300: '#c4b5fd',
  },
  // Layer 3: Semantic Communication
  semantic: {
    danger: '#ef4444',
    dangerBg: '#fee2e2',
    dangerBorder: '#f87171',

    success: '#10b981',
    successText: '#059669',
    successBg: '#ecfdf5',

    warning: '#d97706',
    warningBg: '#fef3c7',
    warningBorder: '#fde68a',

    info: '#3b82f6',
    infoBg: '#eff6ff',
    infoBorder: '#bfdbfe',
  },
  social: {
    linkedin: '#0a66c2',
  },
};

type Palette = { [K in keyof typeof LIGHT]: { [P in keyof (typeof LIGHT)[K]]: string } };

const DARK: Palette = {
  bg: {
    app: '#0d0c14',
    surface: '#17161f',
    surfaceHover: '#201f2b',
    divider: '#2c2a3a',
    dashed: '#3d3a5c',
  },
  text: {
    display: '#f4f4f8',
    body: '#c3c6d4',
    muted: '#8a8ea3',
    inverse: '#ffffff',
  },
  brand: {
    50: '#1a1430',
    100: '#231a42',
    200: '#31255e',
    300: '#4a3a8a',
    400: '#8b6cf0',
    500: '#8b5cf6',
    600: '#8b5cf6', // a little lighter: readable as link text on dark
    700: '#c4b5fd',
    800: '#ddd6fe',
    900: '#ede9fe',
    950: '#f5f3ff',
  },
  fill: {
    brand700: '#6d28d9',
    brand800: '#5b21b6',
    brand900: '#4c1d95',
    brand950: '#2e1065',
    ink: '#2d3348',
  },
  fg: {
    brand50: '#f5f3ff',
    brand100: '#ede9fe',
    brand200: '#ddd6fe',
    brand300: '#c4b5fd',
  },
  semantic: {
    danger: '#f87171',
    dangerBg: '#3a1519',
    dangerBorder: '#7f2a2a',

    success: '#34d399',
    successText: '#34d399',
    successBg: '#0f2a21',

    warning: '#fbbf24',
    warningBg: '#352708',
    warningBorder: '#6b4d12',

    info: '#60a5fa',
    infoBg: '#0f1e36',
    infoBorder: '#1e3a6e',
  },
  social: {
    linkedin: '#3b8fe0',
  },
};

// Me page → Appearance: System (the phone's setting), Light or Dark.
// ONE state object, read through getters. (Reassigned `export let` values were NOT seen by other
// files under Metro, so after a switch some parts still used the old mode — the sun/moon button
// and the Appearance card disagreed, and toneBg/toneFg colours stayed dark in light mode.)
type Scheme = 'light' | 'dark';
const themeState: { pref: AppearancePref; scheme: Scheme } = (() => {
  const pref = getAppearancePref();
  return { pref, scheme: resolveColorScheme(pref) };
})();
/** The mode the app is showing now. */
export function getColorScheme(): Scheme { return themeState.scheme; }
export function getIsDark(): boolean { return themeState.scheme === 'dark'; }
/** The user's choice: 'system' follows the phone. */
export function getAppearanceChoice(): AppearancePref { return themeState.pref; }

// theme.colors.* always reads the current mode. Every key is a fixed (non-configurable) getter:
// once a colour is used inside an animation, Reanimated locks the object's configurable keys and
// silently ignores writes ("Tried to modify key `surface`…"), which left the app stuck in light.
// It skips non-configurable keys, so switching just points `activePalette` at the other palette.
let activePalette: Palette = themeState.scheme === 'dark' ? DARK : LIGHT;
const COLORS = {} as Palette;
for (const group of Object.keys(LIGHT) as Array<keyof Palette>) {
  const view = {} as Record<string, string>;
  for (const key of Object.keys(LIGHT[group])) {
    Object.defineProperty(view, key, { get: () => (activePalette[group] as Record<string, string>)[key], enumerable: true, configurable: false });
  }
  Object.defineProperty(COLORS, group, { value: view, enumerable: true, writable: false, configurable: false });
}

// ---- live styles: switch light / dark without reloading the app ----
const liveObjects: Array<() => void> = [];
type AppearanceListener = (scheme: Scheme, pref: AppearancePref) => void;
const schemeListeners = new Set<(scheme: Scheme) => void>();
const appearanceListeners = new Set<AppearanceListener>();

/**
 * Use instead of StyleSheet.create: returns a style object that rebuilds itself from the current
 * palette whenever the mode changes (screens then redraw with it). Also used for colour tables.
 */
export function makeStyles<T extends StyleSheet.NamedStyles<T> | StyleSheet.NamedStyles<any>>(
  factory: () => T & StyleSheet.NamedStyles<any>,
): T {
  return liveObject(factory) as T;
}

function liveObject<T extends object>(factory: () => T): T {
  let current = factory();
  liveObjects.push(() => { current = factory(); });
  const target = (Array.isArray(current) ? [] : {}) as T;
  return new Proxy(target, {
    get: (_t, k) => (current as any)[k],
    has: (_t, k) => k in (current as any),
    ownKeys: () => Reflect.ownKeys(current as any),
    getOwnPropertyDescriptor: (_t, k) => {
      const d = Object.getOwnPropertyDescriptor(current as any, k);
      if (d) d.configurable = true;
      return d;
    },
  });
}
/** Same as makeStyles, for non-style values that depend on colours (tables of chip colours…). */
export function live<T extends object>(factory: () => T): T {
  return liveObject(factory);
}

/**
 * Switch the whole app to light or dark in place (ThemeSwitch fades over the redraw), and/or
 * record the user's choice. Every listener hears about both, so all controls stay in sync.
 */
export function applyColorScheme(next: Scheme, pref?: AppearancePref) {
  const prefChanged = !!pref && pref !== themeState.pref;
  if (pref) themeState.pref = pref;
  const schemeChanged = next !== themeState.scheme;
  if (schemeChanged) {
    themeState.scheme = next;
    activePalette = next === 'dark' ? DARK : LIGHT;
    liveObjects.forEach((rebuild) => rebuild());
    schemeListeners.forEach((l) => l(next));
  }
  if (schemeChanged || prefChanged) appearanceListeners.forEach((l) => l(themeState.scheme, themeState.pref));
}

/** Fires when the mode actually changes (the root redraws the screens). */
export function onColorSchemeChange(listener: (scheme: Scheme) => void): () => void {
  schemeListeners.add(listener);
  return () => { schemeListeners.delete(listener); };
}

/** Fires when the mode OR the choice changes (sun/moon button, Appearance card…). */
export function onAppearanceChange(listener: AppearanceListener): () => void {
  appearanceListeners.add(listener);
  return () => { appearanceListeners.delete(listener); };
}

/** Current mode + choice, re-rendering the component whenever either changes. */
export function useAppearance(): { scheme: Scheme; pref: AppearancePref; isDark: boolean } {
  const [, force] = useState(0);
  useEffect(() => onAppearanceChange(() => force((n) => n + 1)), []);
  return { scheme: themeState.scheme, pref: themeState.pref, isDark: themeState.scheme === 'dark' };
}

// ---- hard-coded colours → dark equivalents (identity in light mode) ----
function hexToHsl(hex: string): [number, number, number] | null {
  let h = hex.replace('#', '');
  if (h.length === 3 || h.length === 4) h = h.split('').map((c) => c + c).join('');
  if (h.length !== 6 && h.length !== 8) return null;
  const r = parseInt(h.slice(0, 2), 16) / 255, g = parseInt(h.slice(2, 4), 16) / 255, b = parseInt(h.slice(4, 6), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const hue = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [hue * 60, s, l];
}
function hsl(h: number, s: number, l: number): string {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const x = (v: number) => Math.round(v * 255).toString(16).padStart(2, '0');
  return `#${x(f(0))}${x(f(8))}${x(f(4))}`;
}
/** A hard-coded BACKGROUND / border colour: white → surface, pale tints → deep tints, the rest as is. */
export function toneBg(hex: string): string {
  if (themeState.scheme !== 'dark') return hex;
  const c = hexToHsl(hex);
  if (!c) return hex;
  const [h, s, l] = c;
  if (l >= 0.97) return COLORS.bg.surface; // white / near-white (even if tinted)
  // Neutral light greys (borders, sheet handles, skeletons): keep them visible on dark surfaces.
  if (l >= 0.9 && s < 0.4) return COLORS.bg.surfaceHover;
  if (l >= 0.8 && s < 0.4) return '#3a3a4c';
  if (l >= 0.8) return hsl(h, Math.min(0.6, s * 0.55), 0.13 + (l - 0.8) * 0.3); // pastel tints → deep tints
  return hex;
}
/** A hard-coded TEXT / icon colour: dark → light enough to read on dark, the rest as is. */
export function toneFg(hex: string): string {
  if (themeState.scheme !== 'dark') return hex;
  const c = hexToHsl(hex);
  if (!c) return hex;
  const [h, s, l] = c;
  if (l <= 0.45) return hsl(h, Math.min(0.85, s), 0.8 - l * 0.35);
  return hex;
}

export const theme = {
  // --- 1 & 2: Color Architecture ---
  colors: COLORS,

  // --- 3: Spatial Architecture (4px / 8px base) ---
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },

  radii: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16, // Outer cards
    xl: 24,
    full: 9999, // Pills
  },

  // --- 4: Typography Architecture (6 sizes) ---
  typography: {
    fontFamily: {
      sans: Platform.OS === 'ios' ? 'System' : 'Roboto',
      mono: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    },
    sizes: {
      // Large-Text Kerning (-2% to -4% letter spacing)
      display: {
        fontSize: 32,
        fontWeight: '900' as const,
        lineHeight: 36,
        letterSpacing: -1,
      },
      title: {
        fontSize: 24,
        fontWeight: '800' as const,
        lineHeight: 30,
        letterSpacing: -0.5,
      },
      headline: {
        fontSize: 18,
        fontWeight: '700' as const,
        lineHeight: 24,
        letterSpacing: 0,
      },
      body: {
        fontSize: 15,
        fontWeight: '500' as const,
        lineHeight: 22,
      },
      subtext: {
        fontSize: 13,
        fontWeight: '500' as const,
        lineHeight: 18,
      },
      caption: {
        fontSize: 11,
        fontWeight: '700' as const,
        lineHeight: 14,
        letterSpacing: 0.5,
      },
    },
  },
  
  // --- 5: Icon System (microkit.io specification) ---
  icon: {
    // Icon sizing following microkit.io spec
    sizes: {
      xs: 12,    // Small inline icons
      sm: 16,    // Tab bar icons
      md: 20,    // Default icon size
      lg: 24,    // Section headers
      xl: 32,    // Feature icons
      xxl: 44,   // Tab targets (44px minimum touch target)
    },
    // Stroke widths for Lucide icons
    strokeWidths: {
      thin: 1.5,     // Small icons
      normal: 2,     // Default
      medium: 2.5,   // Medium emphasis
      bold: 3,       // High emphasis
    },
  },

  shadows: {
    card: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.03,
      shadowRadius: 8,
      elevation: 1,
    },
    active: {
      shadowColor: '#4f46e5',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.1,
      shadowRadius: 12,
      elevation: 3,
    }
  }
} as const;
