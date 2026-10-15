// Lead levels — one module for the rule (hot ≥ 85, warm 70–84, cool 50–69, cold < 50, the app's
// features/contacts/scoreLevel.ts), the band letter the URL / API filter by (A–D, U = not scored),
// the label, the colours and the counts.
//
// The server sends each contact's level: in the console `level` is the ADMIN level (from the admin's
// own score) and `repLevel` the rep's. When present those win; the local rule is the fallback (older
// payloads, and bare scores such as chart buckets, breakdown bars and toasts).

export type Level = 'hot' | 'warm' | 'cool' | 'cold';
export type Band = 'A' | 'B' | 'C' | 'D';
/** Icon-tile tint for a level (ui/drawerKit TINT): hot rose, warm amber, cool sky, cold slate. */
export type LevelTint = 'rose' | 'amber' | 'sky' | 'slate';

export interface LevelInfo {
  key: Level;
  label: string;
  band: Band;
  /** lowest score in the level */
  min: number;
  /** CSS colour of the level (tokens.css --heat-*) */
  color: string;
  /** Tailwind classes: a dot, a filled bar, a pale chip with readable text */
  dot: string;
  bar: string;
  badge: string;
  tint: LevelTint;
}

export const LEVELS: readonly LevelInfo[] = [
  { key: 'hot', label: 'Hot', band: 'A', min: 85, color: 'var(--heat-hot)', tint: 'rose', dot: 'bg-heat-hot', bar: 'bg-heat-hot', badge: 'bg-red-100 text-red-700 dark:bg-heat-hot/15 dark:text-heat-hot' },
  { key: 'warm', label: 'Warm', band: 'B', min: 70, color: 'var(--heat-warm)', tint: 'amber', dot: 'bg-heat-warm', bar: 'bg-heat-warm', badge: 'bg-amber-100 text-amber-700 dark:bg-heat-warm/15 dark:text-heat-warm' },
  { key: 'cool', label: 'Cool', band: 'C', min: 50, color: 'var(--heat-cool)', tint: 'sky', dot: 'bg-heat-cool', bar: 'bg-heat-cool', badge: 'bg-sky-100 text-sky-700 dark:bg-heat-cool/15 dark:text-heat-cool' },
  { key: 'cold', label: 'Cold', band: 'D', min: -Infinity, color: 'var(--heat-cold)', tint: 'slate', dot: 'bg-heat-cold', bar: 'bg-heat-cold', badge: 'bg-secondary text-body dark:bg-white/5' },
];

export const LEVEL_BY_KEY: Readonly<Record<Level, LevelInfo>> = Object.fromEntries(LEVELS.map((l) => [l.key, l])) as Record<Level, LevelInfo>;

export const isLevel = (v: unknown): v is Level => typeof v === 'string' && v in LEVEL_BY_KEY;

/** The local rule: a score's level (null = not scored). */
export function levelOf(score: number | null | undefined): LevelInfo | null {
  if (score == null) return null;
  return LEVELS.find((l) => score >= l.min)!;
}

export const levelKeyOf = (score: number | null | undefined): Level | null => levelOf(score)?.key ?? null;

/** A server level when it sent one, else the local rule on the score. */
export function levelFrom(score: number | null | undefined, serverLevel?: string | null): LevelInfo | null {
  return isLevel(serverLevel) ? LEVEL_BY_KEY[serverLevel] : levelOf(score);
}

/** Anything that carries the admin score (a contact, a list member, a duplicate match…). */
export interface AdminScored { score?: number | null | undefined; level?: string | null | undefined }
/** Anything that carries the rep's score. */
export interface RepScored { repScore?: number | null | undefined; repLevel?: string | null | undefined }

/** A contact's ADMIN level (what the console ranks, filters and counts by). */
export const contactLevel = (c: AdminScored): LevelInfo | null => levelFrom(c.score, c.level);
/** A contact's REP level (the rep's own score from the app). */
export const repLevelOf = (c: RepScored): LevelInfo | null => levelFrom(c.repScore, c.repLevel);

/** Band letter for the URL / API filter (U = not scored). */
export const bandOf = (score: number | null | undefined, serverLevel?: string | null): Band | 'U' => levelFrom(score, serverLevel)?.band ?? 'U';

export type LevelCounts = { all: number; A: number; B: number; C: number; D: number; U: number };

/** Counts per band for the level pills: hot (A), warm (B), cool (C), cold (D), U = not scored. Takes scores or scored rows. */
export function levelCounts(items: readonly (number | null | undefined | AdminScored)[]): LevelCounts {
  const k: LevelCounts = { all: 0, A: 0, B: 0, C: 0, D: 0, U: 0 };
  items.forEach((it) => {
    k.all += 1;
    const lvl = it != null && typeof it === 'object' ? contactLevel(it) : levelOf(it);
    k[lvl?.band ?? 'U'] += 1;
  });
  return k;
}

/** How many rows are at one of these levels (e.g. hot + warm). */
export function countAtLevels(items: readonly AdminScored[], levels: readonly Level[]): number {
  return items.filter((c) => { const l = contactLevel(c); return !!l && levels.includes(l.key); }).length;
}
