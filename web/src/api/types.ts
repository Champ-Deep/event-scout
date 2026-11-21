// Wire types for the Hi-Fi surfaces. These mirror the backend exactly; docs/hifi-integration-brief.md section 5 is the prose copy.

export type Temperature = "hot" | "warm" | "cold";

export interface Breakdown {
  profile_fit: number;
  role_relevance: number;
  company_fit: number;
  engagement: number;
  timing: number;
}

/** The one score shape every surface displays (POST /scan/analyze, add_contact `score`, 409 `expected`). */
export interface ScoreObject {
  score: number;
  temperature: Temperature;
  reasoning: string;
  breakdown: Breakdown;
  engine?: string;
  signals?: Record<string, unknown>;
}

/** What the scan flow sends. Empty values are the string "N/A", except notes which is "". */
export interface ContactInput {
  name: string;
  email: string;
  phone: string;
  linkedin: string;
  company_name: string;
  notes: string;
  title?: string;
}

export interface Exhibitor {
  id: string;
  name: string;
  booth: string;
  hall: string;
  category: string;
  subcategory: string;
  country: string;
  website: string;
  description: string;
  products: string[];
  tags: string[];
  score: number;
  reason: string;
}

export interface RankedPage {
  status: "success";
  event: string;
  items: Exhibitor[];
  total: number;
  has_more: boolean;
}

export interface ExtractedFields {
  name: string;
  email: string;
  phone: string;
  linkedin: string;
  company_name: string;
}

export interface SavedContact {
  status: "success";
  contact_id: string;
  qr_base64: string;
  score: ScoreObject | null;
}

export interface RetrievedContact {
  id: string;
  name: string;
  title: string;
  email: string;
  phone: string;
  linkedin: string;
  company_name: string;
  lead_score: number | null;
  lead_temperature: Temperature | null;
  lead_score_reasoning: string;
  lead_score_breakdown: Partial<Breakdown>;
  notes: string;
}

export interface ConverseReply {
  status: "success";
  response: string;
  retrieved_contacts: RetrievedContact[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** Every failure the UI must render maps to exactly one kind. */
export type ApiErrorKind =
  | "offline" // fetch threw (no network). Queue-and-retry applies to scanning only.
  | "auth" // 401 or 403
  | "score_mismatch" // 409 on add_contact; `expected` is the fresh server score
  | "unreadable" // 422 on /scan/extract: nothing readable on the card
  | "invalid" // other 4xx
  | "timeout"
  | "server"; // 5xx

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    public status: number,
    message: string,
    public expected?: ScoreObject,
  ) {
    super(message);
  }
}
