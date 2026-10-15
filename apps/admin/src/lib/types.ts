// Shapes returned by the backend admin APIs (apps/backend/api_routes.py).
// The shapes the app uses too come from the wire contract (@es/types); this file keeps the console-only ones.
import type { Contact, WorkspaceDeletedContact, Exhibitor } from '@es/types';

/**
 * Contact: in the admin console, score* / level is the ADMIN's own score (made against the admin sales
 * profile) — what the console ranks, sorts and filters by. The rep's own score is repScore* / repLevel (read only).
 */
export type {
  Contact, ContactFile, WorkspaceDeletedContact, DuplicateMatch, WorkspaceDuplicatePair, Broadcast,
  EventSummary, Exhibitor, EventFile,
} from '@es/types';

/** Fields PATCH /api/contacts/{id} accepts (sent as {baseVersion, patch}). photo: data URI to set, null to remove. */
export interface ContactPatch {
  firstName?: string;
  lastName?: string | null;
  company?: string | null;
  title?: string | null;
  email?: string | null;
  phone?: string | null;
  linkedin?: string | null;
  notes?: string | null;
  eventId?: string | null;
  photo?: string | null;
}

/** POST /api/contact/{id}/score */
export interface ScoreResult {
  contactId: string;
  score: number;
  scoreBand: string;
  temperature: string | null;
  reasoning: string;
  recommendedActions: string[];
  breakdown: Record<string, number>;
  /** nothing the score depends on changed, so the saved score was kept (no AI call) */
  unchanged: boolean;
  previousScore: number | null;
  runs?: number | null;
  spread?: number | null;
  model?: string | null;
  scoredAt?: string | null;
  /** true: saved as the admin console's own score — the rep's score was not changed */
  admin?: boolean | undefined;
}

export interface Dashboard {
  totalContacts: number;
  scansToday: number;
  scansThisWeek: number;
  levels: { hot: number; warm: number; cool: number; cold: number; unscored: number };
  averageScore: number | null;
  reps: number;
  admins: number;
  activeRepsToday: number;
  byEvent: { event: string; count: number }[];
  bySource: { source: string; count: number }[];
  last7Days: { date: string; count: number }[];
  pipelines: { running: number; complete: number; failed: number };
  activeBroadcasts: number;
  files: number;
  lists: number;
  recentlyDeleted: number;
  range: DashboardRange;
  timestamp: string;
  /** contacts created today (the admin's local day, from tzOffset), by local hour 0–23 */
  todayByHour?: number[];
  /** reps who captured contacts today, most first */
  repsToday?: { userId: string; name: string; photoUrl: string | null; count: number }[];
  /** the admin's local midnight today, as an ISO time (UTC) */
  todayStart?: string;
  /** non-admin accounts that captured today / all non-admin accounts */
  activeReps?: number;
  totalReps?: number;
}

/** GET /api/contacts/stats — level counts by the console's (admin) score. */
export interface ContactStats {
  total: number;
  levels: { hot: number; warm: number; cool: number; cold: number; unscored: number };
}

/** The selected period (admin's local days) vs the same length before it. */
export interface DashboardRange {
  days: number;
  from: string;
  to: string;
  series: { date: string; added: number; hot: number; scored: number; avgScore: number | null }[];
  totals: { added: number; hot: number; scored: number; avgScore: number | null };
  previous: { added: number; hot: number; scored: number; avgScore: number | null };
  /** % change for counts; points for avgScore; null = nothing to compare with */
  change: { added: number | null; hot: number | null; scored: number | null; avgScore: number | null };
  activeRepsToday: number;
  scoreHistogram: { from: number; to: number; count: number }[];
  topReps: { userId: string; name: string; added: number; hot: number; avgScore: number | null }[];
  sources: { source: string; count: number }[];
  events: { event: string; count: number }[];
  /** contacts added in the period, split four ways (absent on an older backend) */
  audience?: Audience;
}

/** One slice of the audience mix; `other` = the long tail folded into one row. */
export interface AudienceRow { key: string; label: string; count: number; hot: number; /** score ≥ 70 (hot + warm) */ hotWarm?: number | undefined; avgScore: number | null; other?: boolean }
/** `known` = how many of the period's contacts that split could be worked out for. */
export interface AudienceSplit { known: number; rows: AudienceRow[] }
export interface Audience {
  total: number;
  /** worked out automatically by AI per company (a research run, when there is one, wins).
   *  `pending` = companies still being classified; `ai` = AI key configured; `error` = last run failed */
  industry: AudienceSplit & { pending?: number; ai?: boolean; error?: string | null };
  /** from the job title */
  function: AudienceSplit;
  seniority: AudienceSplit;
  /** top companies; `distinct` = how many companies, `repeat` = companies with 2+ contacts */
  company: AudienceSplit & { distinct: number; repeat: number };
}

export type ActivityType =
  | 'contact_added' | 'contact_scored' | 'pipeline' | 'broadcast' | 'file'
  | 'contact_deleted' | 'contact_merged' | 'user_joined';

/** GET /api/admin/activity/calendar: per local day (YYYY-MM-DD), the total and a count per type. */
export interface ActivityCalendar {
  days: Record<string, { total: number; types: Partial<Record<ActivityType, number>> }>;
}

export interface ActivityItem {
  id: string;
  type: ActivityType;
  title: string;
  subtitle: string;
  actor: string;
  timestamp: string;
  meta: Record<string, unknown>;
}

export interface AdminUser {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'rep';
  isAdmin: boolean;
  contactsCount: number;
  scansToday: number;
  lastActive: string | null;
  linkedin: boolean;
  createdAt: string | null;
  /** the picture they set on their profile in the app (null = none, show initials) */
  photoUrl?: string | null;
  /** their job title from their app profile */
  title?: string | null;
  /** their company from their app profile */
  company?: string | null;
  /** contacts scored 85+ */
  hotCount?: number;
  /** the event they're working now (app profile), and every event they listed */
  currentEvent?: string | null;
  events?: string[];
}

export interface ContactList {
  id: string;
  name: string;
  memberCount: number;
  /** null when it wasn't recorded */
  createdAt?: string | null;
  createdBy?: string | null;
}

/** Smart lists (GET /api/admin/segments): contacts grouped by industry, seniority, role (function) and company. */
export type SegmentDim = 'industry' | 'seniority' | 'function' | 'company';
export interface SegmentRow { key: string; label: string; count: number; /** score ≥ 85 */ hot: number; /** score ≥ 70 (hot + warm) */ hotWarm?: number | undefined; avgScore: number | null; other?: boolean }
export interface Segments {
  total: number;
  industry: { rows: SegmentRow[]; known: number; pending: number; ai: boolean; error: string | null };
  seniority: { rows: SegmentRow[]; known: number };
  function: { rows: SegmentRow[]; known: number };
  company: { rows: SegmentRow[]; known: number; distinct: number };
}

/** GET /api/admin/contact_lists/{id}/contacts: a contact plus when it was put in the list. */
export type ListMember = Contact & { addedToListAt?: string | null };

export interface PipelineRun {
  id?: string;
  contactId: string;
  /** false once the contact was deleted (nothing to open or re-run) */
  contactExists?: boolean;
  contactName?: string | null;
  company?: string | null;
  score?: number | null;
  /** the rep's own score (the app's) */
  repScore?: number | null;
  /** the admin console's score — what an admin-run pipeline's scoring step produces */
  adminScore?: { score: number; temperature?: string | null; reasoning?: string; scoredAt?: string | null } | null;
  /** did this run's scoring step produce a score */
  scoreCompleted?: boolean;
  /** the structured research (null when the AI's answer couldn't be read; the summary still has it) */
  research?: PipelineResearch | null;
  owner?: string | null;
  status: string;
  running: boolean;
  currentStep?: string;
  error?: string | null;
  researchSummary?: string;
  pitchAngle?: string;
  pitchSubject?: string;
  pitchEmail?: string;
  slides?: { title?: string; content?: string; speaker_notes?: string; bullets?: string[] }[];
  deckFileId?: string | null;
  gammaDeckUrl?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  settings?: PipelineSettings;
}

export interface PipelineResearch {
  person: { name: string; likely_title: string; seniority: string; authority: string };
  company: { name: string; industry: string; size_estimate: string; products: string; growth_signals: string };
  industry: string;
  painPoints: string[];
  opportunities: string[];
  talkingPoints: string[];
  competitiveNotes: string;
  confidence: string;
}

export interface PipelineSettings {
  aiConfigured: boolean;
  gammaConfigured: boolean;
  autoPipelineEnabled: boolean;
  /** what the switch goes back to after a server restart (AUTO_PIPELINE_ENABLED in .env) */
  autoPipelineFromEnv?: boolean;
}

export interface PipelineList {
  pipelines: PipelineRun[];
  statusCounts: Record<string, number>;
  /** runs matching the status filter (the list holds one page of them: offset / limit) */
  total?: number;
  settings: PipelineSettings;
}

/** GET /api/admin/broadcasts/history: one announcement, live or ended. */
export interface BroadcastLog {
  id: string;
  message: string;
  priority: 'normal' | 'urgent' | string;
  isActive: boolean;
  createdAt: string | null;
  endedAt: string | null;
  sentBy: string | null;
  recipients: { id: string; name: string }[];
  eventId: string | null;
  eventName: string | null;
}

/** Every event the workspace knows (GET /api/admin/events): booth directory ∪ contacts ∪ reps' lists ∪ files. */
export interface AdminEvent {
  name: string;
  key: string;
  booths: number;
  contacts: number;
  hot: number;
  /** reps who captured contacts there */
  reps: number;
  /** reps who list it in the app (Me → events) */
  following: number;
  /** reps for whom it's the current event */
  activeFor: number;
  files: number;
  floorPlans: number;
  hasDirectory: boolean;
  lastActivity: string | null;
}

/** A booth in GET /api/admin/events/detail: the exhibitor (contract) plus who visited and who met there. */
export interface EventBooth extends Exhibitor {
  /** reps who marked it visited */
  visitCount: number;
  visitors: string[];
  /** contacts met at this event from this company */
  metContacts: number;
  /** the rep who added it from the app */
  addedBy?: string | null;
}

export interface EventRep {
  userId: string;
  name: string;
  contacts: number;
  hot: number;
  boothsVisited: number;
  /** has it in their event list */
  following: boolean;
  /** it's their current event */
  active: boolean;
  lastCapture: string | null;
}

export interface EventFileItem {
  id: string;
  filename: string;
  fileType: string;
  mimeType?: string | null;
  category: string;
  fileSize: number;
  description: string;
  forAllEvents: boolean;
  floorPlan: boolean;
  createdAt: string | null;
}

/** One event in full (GET /api/admin/events/detail?event=). */
export interface EventDetail {
  name: string;
  key: string;
  stats: { booths: number; halls: number; categories: number; contacts: number; hot: number; reps: number; boothsVisited: number; boothsMet: number };
  booths: EventBooth[];
  reps: EventRep[];
  perDay: { date: string; count: number }[];
  files: EventFileItem[];
}

/** Event planner (GET /api/admin/planner/events): an event the admins plan and staff. */
export type PlannerStatus = 'suggested' | 'interested' | 'planned' | 'attending' | 'done' | 'dismissed';
export interface PlannerEvent {
  id: string;
  name: string;
  status: PlannerStatus;
  startDate: string | null;
  endDate: string | null;
  venue: string;
  address: string;
  city: string;
  country: string;
  website: string;
  organizer: string;
  industry: string;
  description: string;
  details: Partial<Record<'hours' | 'registration' | 'tickets' | 'booth' | 'dressCode' | 'contactName' | 'contactEmail' | 'contactPhone' | 'notes', string>>;
  source: 'manual' | 'discovery' | string;
  sourceUrl: string;
  createdBy: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  team: { userId: string; name: string; email: string; role: 'lead' | 'member'; isAdmin: boolean; photoUrl: string | null; assignedAt: string | null }[];
  files: { id: string; filename: string; fileType: string; fileSize: number; category: string; description: string; createdAt: string | null; floorPlan: boolean }[];
  contactsMet: number;
}
export type PlannerEventInput = Partial<Omit<PlannerEvent, 'id' | 'team' | 'files' | 'contactsMet' | 'source' | 'sourceUrl' | 'createdBy' | 'createdAt' | 'updatedAt'>>;

/** GET /api/admin/contacts/deleted/{id}: the full contact rebuilt from its snapshot (read-only) + why / when it was deleted. */
export interface DeletedContactDetail extends WorkspaceDeletedContact {
  contact: Contact;
  files: { id: string; filename: string; fileType: string; category: string; fileSize: number; description: string; createdAt: string | null }[];
}

/** A contact the copilot's answer drew on (its contact cards open these): POST /api/converse's `contacts`, and
 *  what api.converseStream makes of the stream's retrieved_contacts (@es/types RetrievedContact). Console only. */
export interface ConverseContact {
  id: string;
  name: string;
  company?: string | null;
  score?: number | null;
  temperature?: string | null;
}

/** POST /api/converse (and the final event of /api/converse/stream). */
export interface ConverseReply {
  reply: string;
  /** "openui": the reply holds ```openui-lang blocks between plain text */
  format: string;
  modelUsed?: string;
  contacts: ConverseContact[];
  /** follow-up questions the server suggests (its quick_replies) */
  quickReplies?: string[] | undefined;
  timestamp?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  role: string;
  uses: string[];
  default: boolean;
  configured: boolean;
}

/** GET /api/health — the database is really queried (SELECT 1). */
export interface Health {
  status: 'healthy' | 'degraded';
  api: string;
  databaseReady: boolean;
  databaseLatencyMs: number | null;
  databaseError: string | null;
  databaseKind: 'SQLite' | 'PostgreSQL' | null;
  version: string | null;
  aiConfigured: boolean | null;
  webhookConfigured: boolean | null;
  /** how often the automatic backup runs (null: no backup database) */
  backupIntervalHours: number | null;
  startedAt: string;
  uptimeSeconds: number;
  timestamp: string;
}

export interface BackupResult {
  /** partial: some tables failed (named in failedTables / message) */
  status: 'success' | 'partial' | 'error';
  kind: 'backup-database' | 'sqlite-file';
  /** who ran it: from this console, or the startup / scheduled backup */
  trigger?: 'console' | 'automatic';
  at: string;
  file?: string;
  sizeBytes?: number;
  message: string;
  /** rows copied per table (null: that table failed) */
  tables?: Record<string, number | null>;
  failedTables?: string[];
  totalRows?: number | null;
  durationSeconds?: number | null;
}

export interface BackupStatus {
  target: 'backup-database' | 'sqlite-file' | null;
  configured: boolean;
  lastBackup: BackupResult | null;
  snapshots: { file: string; sizeBytes: number; createdAt: string }[];
  lastSnapshotAt: string | null;
  keep: number;
  intervalHours: number | null;
}

export interface WebhookResult {
  status: 'delivered' | 'rejected';
  statusCode: number;
  latencyMs: number;
  target: string;
  response: string;
}

/** Exports → Send to CRM (GET /api/admin/crm). Credentials live in server env; `missing` names the unset vars. */
export type CrmKey = 'hubspot' | 'salesforce' | 'zoho' | 'pipedrive';
export interface CrmJob {
  id: string;
  crm: CrmKey;
  label: string;
  record: string;
  scope: string;
  status: 'running' | 'done' | 'failed' | 'cancelled';
  total: number;
  processed: number;
  created: number;
  updated: number;
  skipped: number;
  failed: number;
  failures: { contactId: string; name: string; reason: string }[];
  error: string | null;
  startedAt: string;
  finishedAt: string | null;
}
export interface CrmInfo {
  key: CrmKey;
  label: string;
  record: string;
  configured: boolean;
  missing: string[];
  lastPush: CrmJob | null;
}
export interface CrmTestResult { ok: boolean; label: string; account: string; latencyMs: number }

/** GET /api/admin/me — the signed-in admin's account and the profile the app's Me screen edits. */
export interface MyAccount {
  id: string;
  name: string;
  email: string;
  isAdmin: boolean;
  createdAt: string | null;
  /** false for a LinkedIn-created account that hasn't set a password yet */
  passwordSet: boolean;
  linkedin: { connected: boolean; email: string | null };
  profile: {
    full_name: string;
    job_title: string;
    company_name: string;
    company_website: string;
    linkedin_url: string;
    phone: string;
    email_signature: string;
  };
  photoUrl: string | null;
}

export type PitchStyle = '' | 'consultative' | 'direct' | 'challenger' | 'relationship';

/**
 * GET /api/admin/sales-profile — the admin console's own AI profile (shared by all admins): who's pitching,
 * what we sell, how we sell, who we target. Admin scoring/ranking, admin pipeline runs and the AI assistant
 * use ONLY this; reps' profiles are never read or changed for it.
 */
export interface AdminSalesProfile {
  company_name: string;
  job_title: string;
  products: { name: string; description: string; ideal_customer: string }[];
  pitch_style: PitchStyle;
  value_propositions: string[];
  common_objections: { objection: string; response: string }[];
  case_studies: { title: string; result: string }[];
  target_industries: string[];
  target_roles: string[];
  target_company_sizes: string[];
  target_geographies: string[];
  updatedAt: string | null;
  /** name of the admin who last saved it */
  updatedBy: string | null;
}

export type AdminSalesPatch = Partial<Omit<AdminSalesProfile, 'updatedAt' | 'updatedBy'>>;

export type MyProfilePatch = Partial<MyAccount['profile']> & {
  /** a data URI to set the photo, '' to remove it */
  avatar?: string | undefined;
};
