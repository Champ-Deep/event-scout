export const tokens = {
  colors: {
    bg: {
      primary: '#09090b',
      secondary: '#18181b',
      card: '#27272a',
      input: '#27272a',
    },
    text: {
      primary: '#fafafa',
      secondary: '#a1a1aa',
      muted: '#71717a',
    },
    accent: {
      emerald: '#10b981',
      blue: '#3b82f6',
      amber: '#f59e0b',
      rose: '#f43f5e',
    },
    bands: {
      A: '#10b981', // 85-100 Tier 1
      B: '#3b82f6', // 70-84 Tier 2
      C: '#f59e0b', // 50-69 Tier 3
      D: '#f97316', // 30-49 Tier 4
      F: '#71717a', // 0-29 Unqualified
    },
    border: '#3f3f46',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  radii: {
    sm: 6,
    md: 10,
    lg: 16,
    full: 9999,
  },
  typography: {
    title: {
      fontSize: 24,
      fontWeight: '700' as const,
      lineHeight: 32,
    },
    subtitle: {
      fontSize: 18,
      fontWeight: '600' as const,
      lineHeight: 24,
    },
    body: {
      fontSize: 15,
      fontWeight: '400' as const,
      lineHeight: 20,
    },
    caption: {
      fontSize: 12,
      fontWeight: '500' as const,
      lineHeight: 16,
    },
  },
} as const;
