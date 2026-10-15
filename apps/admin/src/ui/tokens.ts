// Design tokens shared by the admin UI. Icons take their size + stroke from here (never per icon).
export const ICON = { size: 16, stroke: 1.75 } as const;

// Motion: short, calm, no bounce (matches the mobile app). `motion` honours prefers-reduced-motion
// through <MotionConfig reducedMotion="user"> in main.tsx.
export const MOTION = {
  ease: [0.2, 0.8, 0.2, 1] as const,
  fast: 0.18,
  base: 0.26,
  slow: 0.9,
  press: 0.97,
  stagger: 0.06,
} as const;

// Contacts grid: virtualise past this many rows.
export const VIRTUALIZE_AFTER = 200;
