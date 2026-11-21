import type { Temperature } from "@/api/types";

/** Spec 1.5. One function for ScoreBadge, ScoreCard and the feed. Bands: hot >= 70, warm 40 to 69, cold < 40. */
export const HOT = 70;
export const WARM = 40;

export interface Band {
  key: Temperature;
  word: "Hot" | "Warm" | "Cold";
  /** Tailwind classes using the @theme colors in index.css (text passes 4.5:1 on its fill). */
  className: string;
}

export function scoreBand(score: unknown): Band | null {
  if (
    typeof score !== "number" ||
    !Number.isFinite(score) ||
    score < 0 ||
    score > 100
  )
    return null;
  if (score >= HOT)
    return { key: "hot", word: "Hot", className: "bg-hot text-white" };
  if (score >= WARM)
    return { key: "warm", word: "Warm", className: "bg-warm text-foreground" };
  return { key: "cold", word: "Cold", className: "bg-cold text-foreground" };
}

export const BREAKDOWN_LABELS = {
  profile_fit: "Fits what you sell",
  role_relevance: "Role match",
  company_fit: "Industry match",
  engagement: "Reachability",
  timing: "Event timing",
} as const;
