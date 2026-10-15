import { toneBg, toneFg, live } from '../../ui/theme';
/**
 * One set of score levels for the whole app (feed, contact page, event page), matching the
 * backend: hot ≥ 85 · warm 70–84 · cool 50–69 · cold < 50 — the same four levels as the
 * backend's temperature and the `level` it sends on each contact (contactLevel prefers that).
 * Before, the feed called everything under 70 "COOL" and the contact page showed HOT in green
 * while the feed showed it in red.
 */
export type ScoreLevelKey = 'hot' | 'warm' | 'cool' | 'cold';

export interface ScoreLevel {
  key: ScoreLevelKey;
  label: 'HOT' | 'WARM' | 'COOL' | 'COLD';
  name: 'Hot' | 'Warm' | 'Cool' | 'Cold';
  range: string;
  fg: string;   // text
  bg: string;   // chip background
  ring: string; // bars / rings
}

const LEVELS: Record<ScoreLevelKey, ScoreLevel> = live(() => ({
  hot: { key: 'hot', label: 'HOT', name: 'Hot', range: '85+', fg: toneFg('#b91c1c'), bg: toneBg('#fee2e2'), ring: '#ef4444' },
  warm: { key: 'warm', label: 'WARM', name: 'Warm', range: '70–84', fg: toneFg('#b45309'), bg: toneBg('#fef3c7'), ring: '#f59e0b' },
  cool: { key: 'cool', label: 'COOL', name: 'Cool', range: '50–69', fg: toneFg('#0369a1'), bg: toneBg('#e0f2fe'), ring: '#38bdf8' },
  cold: { key: 'cold', label: 'COLD', name: 'Cold', range: 'under 50', fg: toneFg('#475569'), bg: toneBg('#f1f5f9'), ring: '#94a3b8' },
}));

export const SCORE_LEVELS: ScoreLevel[] = live(() => ([LEVELS.hot, LEVELS.warm, LEVELS.cool, LEVELS.cold]));

export function scoreLevel(score: number | null | undefined): ScoreLevel | null {
  if (score == null || Number.isNaN(score)) return null;
  if (score >= 85) return LEVELS.hot;
  if (score >= 70) return LEVELS.warm;
  if (score >= 50) return LEVELS.cool;
  return LEVELS.cold;
}

/** Anything with a score — and, from newer servers, its `level` already worked out. */
export interface Scored {
  score?: number | null | undefined;
  level?: ScoreLevelKey | string | null | undefined;
}

const LEVEL_KEYS: ScoreLevelKey[] = ['hot', 'warm', 'cool', 'cold'];

/** A contact's level: the server's `level` when it sent one, else worked out from the score. */
export function contactLevel(c: Scored | null | undefined): ScoreLevel | null {
  const key = c?.level;
  if (typeof key === 'string' && (LEVEL_KEYS as string[]).includes(key)) return LEVELS[key as ScoreLevelKey];
  return scoreLevel(c?.score);
}

/** Hot lead (85+). */
export function isHot(c: Scored | null | undefined): boolean {
  return contactLevel(c)?.key === 'hot';
}

/** Warm or hot (70+): worth following up first. */
export function isWarmOrHotter(c: Scored | null | undefined): boolean {
  const key = contactLevel(c)?.key;
  return key === 'hot' || key === 'warm';
}
