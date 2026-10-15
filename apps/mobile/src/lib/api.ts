import axios, { AxiosInstance } from 'axios';
import { Platform } from 'react-native';
import { isOnline } from '../sync/network';
import { secureStorage } from './secure';
import { isTokenExpired } from './jwt';
import { API_CONFIG } from '../config/constants';
import { classifyError, type ApiError } from './apiError';
import { createRequestPipeline, type RequestPipeline } from './requestPipeline';

// The shapes the admin console shares come from the wire contract; responses of those shapes are
// checked against it in development (checkResponse — a no-op in release builds).
import { z } from 'zod';
import {
  addedExhibitorSchema, broadcastSchema, contactFileSchema, contactPageSchema, contactSchema, deletedContactsSchema,
  duplicateMatchesSchema, duplicatePairsSchema, eventFileSchema, eventSummarySchema, exhibitorSchema, fileContentSchema,
  retrievedContactsSchema,
  type Broadcast, type Contact, type ContactFile, type DeletedContact, type DuplicateMatch, type DuplicatePair,
  type EventFile, type EventSummary, type Exhibitor, type FileContent, type RetrievedContact,
} from '@es/types';
import { checked, checkResponse } from './checkResponse';
export type {
  Broadcast, Contact, ContactFile, DeletedContact, DuplicateMatch, DuplicatePair, DuplicateReason, EventFile, Exhibitor,
  FileContent, RetrievedContact,
} from '@es/types';
/** GET /api/events (an event with an imported exhibitor directory). */
export type Event = EventSummary;

// Errors: every failure this adapter returns or throws is an ApiError (see ./apiError).
export {
  ApiError, SESSION_REVOKED_DETAIL, isApiError, asApiError, classifyError, errorMessage, errorKind, isOffline, isNetworkError,
  type ApiErrorKind, type ErrorMessageOptions,
} from './apiError';

export interface ListContactsParams {
  search?: string | undefined;
  stage?: string | undefined;
  scoreBand?: string | undefined;
  eventId?: string | undefined;
  userId?: string | undefined;
  limit?: number | undefined;
  cursor?: string | undefined;
  /** 'score' = best lead score first (unscored last); default = newest first */
  sort?: 'recent' | 'score' | undefined;
}

export interface Seat {
  id: string;
  eventId: string;
  userId: string;
  name: string;
  email: string;
  role: 'admin' | 'rep';
  status: 'active' | 'inactive';
  scansCount: number;
}

/** An event the admins planned (Admin console → Events), as a rep on it sees it (GET /api/me/events). */
export interface PlannedEvent {
  id: string;
  name: string;
  status: string;
  /** YYYY-MM-DD, the event's own local dates */
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
  /** hours, registration, tickets, booth, dressCode, contactName, contactEmail, contactPhone, notes */
  details: Record<string, string>;
  updatedAt: string | null;
  /** your role on the event's team; null = it's in your list but you aren't on the team */
  myRole: 'lead' | 'member' | null;
  lead: string | null;
  team: { name: string; role: string }[];
  floorPlans: number;
}

/** Your own note / reminder for an event (Me → Calendar). Private to you. */
export interface EventNote {
  id: string;
  event: string;
  /** YYYY-MM-DD the note is pinned to on the calendar */
  date: string | null;
  text: string;
  /** ISO date-time; set = a reminder */
  remindAt: string | null;
  done: boolean;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface EventNoteInput {
  event?: string;
  date?: string;
  text?: string;
  remindAt?: string;
  done?: boolean;
}

/** safe = the mailbox exists · risky = exists but catch-all / disposable / role / full ·
 *  invalid = bad address, domain takes no mail, or mailbox doesn't exist · unknown = couldn't ask. */
export type EmailReachable = 'safe' | 'risky' | 'invalid' | 'unknown';

/** Result of POST /api/email/verify — the check-if-email-exists checks (backend email_verify.py). */
export interface EmailCheck {
  input: string;
  is_reachable: EmailReachable;
  /** plain-English findings, in order */
  reasons: string[];
  syntax: { address: string; is_valid_syntax: boolean; username: string; domain: string; normalized_email: string | null; error?: string };
  mx: { accepts_mail: boolean; records: string[]; error?: string };
  misc: { is_disposable: boolean; is_role_account: boolean; is_free_provider: boolean };
  smtp: { can_connect_smtp: boolean; has_full_inbox: boolean; is_catch_all: boolean; is_deliverable: boolean; is_disabled: boolean; error?: string | null } | null;
  smtp_checked: boolean;
  source: 'builtin' | 'reacher';
  checked_at: string | null;
  cached: boolean;
}

/** A possible LinkedIn profile for a contact (from POST /api/contacts/{id}/linkedin/search). */
export interface LinkedInCandidate {
  url: string;
  name: string;
  headline: string;
  confidence: number;        // 0-100, computed from what matched
  matched: string[];         // 'name' | 'partial name' | 'company' | 'email domain' | 'job title'
  photoUrl: string | null;   // profile picture, when one could be found
  photoSource?: 'linkedin' | 'brave' | null;
}

export interface LinkedInSearchResult {
  candidates: LinkedInCandidate[];
  searchedFor: { name?: string; company?: string; title?: string };
  linkedinSearchUrl: string; // LinkedIn's own people search, for "not here?"
  photosLimited?: boolean;   // LinkedIn is refusing automated page loads for now → photos may be missing
  contactPhoto?: { url: string; source: 'gravatar' } | null; // the contact's own photo (from their email)
  imageSearchConfigured?: boolean;
}

export interface BoothInput {
  event: string;
  name: string;
  booth?: string;
  hall?: string;
  category?: string;
  website?: string;
  description?: string;
  /** also mark it visited for me */
  visited?: boolean;
  note?: string;
}

export interface AIResponse {
  reply: string;
  modelUsed: string;
  timestamp: string;
}

// Shape returned by the scan endpoints (/add_contact_from_image/, /scan_qr/).
// The backend runs OCR/QR decode, persists a ContactDB row, and returns its id.
export interface ScanResult {
  status: string; // 'success' | 'duplicate' | ...
  message?: string;
  contact_id?: string;
  existing_contact_id?: string;
  // status === 'duplicate': who it matched and on what
  existing_name?: string;
  matched_on?: 'email' | 'phone' | string;
  // true when "Auto-score on scan" is on and scoring started in the background
  auto_score_started?: boolean;
  // true when this client_id was already saved (a retried upload): contact_id is that contact
  already_saved?: boolean;
  extracted_fields?: Record<string, any>;
  qr_base64?: string;
  photo_base64?: string;
}

// Extra query params for the scan endpoints. `clientId` (a uuid) makes the upload safe to repeat:
// the server saves the contact with that id, and a second upload with it returns `already_saved`.
export interface ScanOptions {
  clientId?: string;
  /** event active when the photo was taken (offline scans upload later) */
  eventName?: string | null;
  /** ISO 8601 time the photo was taken */
  capturedAt?: string | null;
}

function scanQuery(userId: string, opts?: ScanOptions): string {
  const q = [`user_id=${encodeURIComponent(userId)}`];
  if (opts?.clientId) q.push(`client_id=${encodeURIComponent(opts.clientId)}`);
  if (opts?.eventName) q.push(`event_name=${encodeURIComponent(opts.eventName)}`);
  if (opts?.capturedAt) q.push(`captured_at=${encodeURIComponent(opts.capturedAt)}`);
  return q.join('&');
}

// Shape returned by POST /converse/ (answered by an OpenRouter model). retrieved_contacts: the contract's RetrievedContact.
export interface ConverseResponse {
  status: string;
  response: string;
  format?: 'openui' | 'text'; // 'openui': response may contain ```openui-lang blocks between plain text
  model: string; // the model that actually answered, e.g. "google/gemini-2.5-flash"
  retrieved_contacts: RetrievedContact[];
  ui_components?: Array<{ type: string; data: any }>;
  intent?: string;
  query?: string;
  attachments?: Array<{ name: string; mime_type: string }>;
}

// Optional parts of a contact that can go into a share link (the name is always included).
export type ShareField = 'title' | 'company' | 'email' | 'phone' | 'linkedin' | 'event' | 'score' | 'notes' | 'photo';

export interface ContactShareLink {
  token: string;
  url: string;       // page anyone with the link can open
  vcardUrl: string;  // "Save contact" download
  fields: ShareField[];
  createdAt: string | null;
  expiresAt: string | null; // null = never
  viewCount: number;
  lastViewedAt: string | null;
  state: 'active' | 'revoked' | 'expired';
}

export interface MyCard {
  card: { name: string; title: string | null; company: string | null; email: string; phone: string | null; linkedin: string | null; website: string | null };
  hasPhoto: boolean;
  vcard: string;
  qrDataUrl: string;     // vCard QR: scan with any camera to add you
  url: string;           // public card page with Save contact
  linkQrDataUrl: string; // QR of that page link
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

// A file picked in the Ask composer (photo library, camera or document picker).
export interface ChatAttachment {
  uri: string;
  name: string;
  mimeType?: string | undefined;
  size?: number | undefined;
}

export interface AuthSession {
  token: string;
  refreshToken: string;
  user: {
    id: string;
    email: string;
    firstName: string | null;
    lastName: string | null;
  };
}

// Shape returned by POST /auth/linkedin/callback (snake_case, matches backend)
export interface LinkedInSession {
  status: string;
  message: string;
  token: string;
  refresh_token: string;
  expires_in: number;
  user: {
    id: string;
    name: string;
    email: string;
    is_admin?: boolean;
  };
}

// Shape returned by POST /auth/linkedin/link
export interface LinkedInLinkResult {
  status: string;
  message: string;
  linkedin: { name: string; email: string; picture?: string | null };
  user: { id: string; name: string; email: string };
}

// Body for POST /api/contacts (manual add). `photo` is a base64 / data-URI image.
export interface CreateContactInput {
  /** client-made uuid: sending the same one again returns the contact already created (no duplicate) */
  id?: string;
  firstName: string;
  lastName?: string | null;
  company?: string | null;
  title?: string | null;
  email?: string | null;
  phone?: string | null;
  notes?: string | null;
  eventId?: string | null;
  source?: 'manual' | 'scan' | 'import';
  photo?: string | null;
}

// Shape returned by GET /auth/linkedin/link
export interface LinkedInLinkStatus {
  connected: boolean;
  linkedin_email: string | null;
  linked_at: string | null;
  has_password: boolean;
}

class EventScoutAPI {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || API_CONFIG.BASE_URL;

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    // Headers, token renewal and revoked-session detection: one pipeline, shared with the streams.
    this.pipeline = createRequestPipeline({
      getToken: () => this.getAuthToken(),
      getRefreshToken: () => secureStorage.getRefreshToken(),
      saveTokens: async (access, refresh) => {
        await this.setAuthToken(access);
        await this.setRefreshToken(refresh);
      },
      postRefresh: (rt) => this.client.post<any, { token: string; refresh_token: string; expires_in?: number }>('/auth/refresh', { refreshToken: rt }),
      isTokenExpired,
      apiKey: () => process.env.EXPO_PUBLIC_API_KEY || process.env.API_KEY,
      warn: (m) => console.warn(m),
    });

    this.client.interceptors.request.use(
      async (config) => {
        if (config.url) {
          const headers = await this.pipeline.headers(config.url);
          Object.entries(headers).forEach(([k, v]) => { config.headers[k] = v; });
        } else {
          const token = await this.getAuthToken();
          if (token) config.headers.Authorization = `Bearer ${token}`;
        }

        // If sending FormData, delete the global Content-Type so Axios computes the multipart boundary
        if (config.data instanceof FormData) {
          delete config.headers['Content-Type'];
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Every failure leaves as one classified ApiError (after renew-and-replay of an expired token).
    this.client.interceptors.response.use(
      (response) => response.data,
      async (error) => {
        if (API_CONFIG.DEBUG) {
          console.log('API Error:', {
            url: error.config?.url,
            method: error.config?.method,
            status: error.response?.status,
            data: error.response?.data,
            message: error.message,
          });
        }

        const url = String(error.config?.url || '');
        const status = error.response?.status;
        const detail = error.response?.data?.detail;

        // Access token expired (the server enforces it): renew once — shared by every request that
        // failed at the same time — and replay this request with the new token. A refresh the
        // server refuses (401) signs out through checkRevoked below.
        if (status === 401 && error.config && !error.config._authRetried) {
          error.config._authRetried = true;
          const sent = String(error.config.headers?.Authorization || '').replace(/^Bearer /, '');
          const renewed = await this.pipeline.recoverFrom401({ url, detail, sentToken: sent });
          if (renewed) {
            error.config.headers = error.config.headers || {};
            error.config.headers.Authorization = `Bearer ${renewed}`;
            return this.client.request(error.config);
          }
        }

        // Signed out elsewhere, or the session check / renewal refused: end the local session once.
        this.pipeline.checkRevoked({ url, status, detail });

        throw await this.classify(error, url);
      }
    );
  }

  private pipeline: RequestPipeline;

  /** The one error type: no answer + no connection = offline (when offline mode is on). */
  private async classify(error: any, url: string): Promise<ApiError> {
    const noAnswer = !error?.response;
    const deviceOffline = noAnswer && API_CONFIG.OFFLINE_MODE_ENABLED && !(await this.checkNetworkConnection());
    return classifyError(error, { deviceOffline, sessionRoute: this.pipeline.isSessionRoute(url) });
  }

  // ==================== SESSION REVOCATION ====================
  /** The session module (src/features/session) registers this to sign out when the server reports a revoked session. */
  setSessionRevokedHandler(handler: (() => void) | null) {
    this.pipeline.setSessionRevokedHandler(handler);
  }

  /** Call after a successful sign-in so a later revocation is reported again. */
  resetSessionRevoked() {
    this.pipeline.resetSessionRevoked();
  }

  /** Resolves if this device's session is still valid; a revoked one triggers the handler. */
  async checkSession() {
    return this.client.get<any, { status: string; valid: boolean; user_id: string; name?: string; email?: string; is_admin?: boolean }>('/auth/session');
  }

  // Check network connection for React Native
  private async checkNetworkConnection(): Promise<boolean> {
    if (Platform.OS === 'web') {
      return navigator.onLine;
    }
    
    try {
      // Real connectivity (expo-network) — it used to assume native apps were always online.
      return await isOnline();
    } catch {
      return true;
    }
  }

  // Auth token management
  private async getAuthToken(): Promise<string | null> {
    try {
      return await secureStorage.getAuthToken();
    } catch (error) {
      console.warn('Failed to get auth token:', error);
      return null;
    }
  }

  private async setAuthToken(token: string): Promise<void> {
    try {
      await secureStorage.setAuthToken(token);
    } catch (error) {
      console.warn('Failed to set auth token:', error);
    }
  }

  private async setRefreshToken(token: string): Promise<void> {
    try {
      await secureStorage.setRefreshToken(token);
    } catch (error) {
      console.warn('Failed to set refresh token:', error);
    }
  }

  // ==================== HEALTH & STATUS ====================
  async healthCheck() {
    try {
      return await this.client.get('/api/health');
    } catch (error) {
      console.warn('Health check failed, but continuing in offline mode');
      return {
        status: 'offline',
        api: 'EventScout Backend API',
        databaseReady: false,
        timestamp: new Date().toISOString(),
      };
    }
  }

  // ==================== PROFILE ====================
  async getProfile(userId: string) {
    return this.client.get<any, any>(`/user/profile/`, { params: { user_id: userId } });
  }

  async updateProfile(userId: string, profile: any) {
    return this.client.put<any, any>(`/user/profile/`, profile, { params: { user_id: userId } });
  }

  // ==================== DASHBOARD ====================
  async getDashboard(userId: string) {
    // tz_offset: so "Met today" is the phone's local day, not the UTC day.
    return this.client.get<any, any>(`/dashboard`, { params: { user_id: userId, tz_offset: new Date().getTimezoneOffset() } });
  }

  // ==================== CONTACTS MANAGEMENT ====================
  async listContacts(params?: ListContactsParams) {
    return this.client.get<any, { contacts: Contact[]; cursor: string | null }>('/api/contacts', { params })
      .then(checked(contactPageSchema, 'GET /api/contacts'));
  }

  async getContact(contactId: string) {
    return this.client.get<Contact>(`/api/contacts/${contactId}`).then(checked(contactSchema, 'GET /api/contacts/{id}'));
  }

  // POST /api/contacts — owner is the signed-in user; starts unscored.
  async createContact(contact: CreateContactInput) {
    return this.client.post<any, Contact>('/api/contacts', contact).then(checked(contactSchema, 'POST /api/contacts'));
  }

  // `patch` may include `photo`: a base64 / data-URI image to set, or null to remove it.
  // `base`: each edited field's value when the edit started. If one of them was changed elsewhere
  // since, the server saves nothing and answers 409 (see ContactEditConflict); `force` saves anyway.
  async updateContact(
    contactId: string,
    patch: Partial<Contact> & { photo?: string | null },
    opts: { base?: Record<string, unknown>; force?: boolean } = {},
  ) {
    return this.client.patch<any, Contact>(`/api/contacts/${contactId}`, {
      baseVersion: 1,
      patch,
      ...(opts.base && Object.keys(opts.base).length ? { base: opts.base } : {}),
      ...(opts.force ? { force: true } : {}),
    }).then(checked(contactSchema, 'PATCH /api/contacts/{id}'));
  }

  // ---- Find on LinkedIn ----
  // Live web search for this contact's LinkedIn profile. Only profiles that really appear in the
  // search results come back (nothing is saved); the user picks one, then attachLinkedIn saves it.
  async findLinkedIn(contactId: string) {
    return this.client.post<any, LinkedInSearchResult>(`/api/contacts/${contactId}/linkedin/search`, null, { timeout: 120000 });
  }

  // Save the picked profile link, plus its photo when found (`usePhoto: false` keeps the current one).
  async attachLinkedIn(contactId: string, pick: { url: string; photoUrl?: string | null; usePhoto?: boolean }) {
    return this.client.post<any, { contact: Contact; photoAdded: boolean; photoStatus?: 'added' | 'limited' | 'none'; photoSource?: 'search' | 'linkedin' | 'brave' | 'gravatar' | null }>(
      `/api/contacts/${contactId}/linkedin/attach`, pick, { timeout: 45000 });
  }

  // Moves the contact to Recently deleted (restorable for `daysLeft` days).
  async deleteContact(contactId: string) {
    return this.client.delete<any, { status: string; id: string; daysLeft: number; purgeAfter: string }>(`/api/contacts/${contactId}`);
  }

  // ---- Recently deleted (30-day archive) ----
  async listDeletedContacts() {
    return this.client.get<any, { contacts: DeletedContact[]; retentionDays: number }>('/api/contacts/deleted')
      .then(checked(deletedContactsSchema, 'GET /api/contacts/deleted'));
  }

  async restoreContact(contactId: string) {
    return this.client.post<any, Contact>(`/api/contacts/deleted/${contactId}/restore`)
      .then(checked(contactSchema, 'POST /api/contacts/deleted/{id}/restore'));
  }

  async purgeDeletedContact(contactId: string) {
    return this.client.delete(`/api/contacts/deleted/${contactId}`);
  }

  async emptyDeletedContacts() {
    return this.client.delete<any, { status: string; deleted: number }>('/api/contacts/deleted');
  }

  // ---- Duplicates ----
  async getContactDuplicates(contactId: string) {
    return this.client.get<any, { matches: DuplicateMatch[] }>(`/api/contacts/${contactId}/duplicates`)
      .then(checked(duplicateMatchesSchema, 'GET /api/contacts/{id}/duplicates'));
  }

  async listDuplicatePairs() {
    return this.client.get<any, { pairs: DuplicatePair[]; count: number }>('/api/contacts/duplicates')
      .then(checked(duplicatePairsSchema, 'GET /api/contacts/duplicates'));
  }

  async checkDuplicates(details: {
    firstName?: string; lastName?: string | null; company?: string | null;
    email?: string | null; phone?: string | null; excludeId?: string;
  }) {
    return this.client.post<any, { matches: DuplicateMatch[] }>('/api/contacts/check-duplicates', details)
      .then(checked(duplicateMatchesSchema, 'POST /api/contacts/check-duplicates'));
  }

  async dismissDuplicate(contactId: string, otherId: string) {
    return this.client.post(`/api/contacts/${contactId}/duplicates/${otherId}/dismiss`);
  }

  // ==================== SHARE LINKS ====================
  // View-only link showing the name + the ticked fields (live; revocable). expiresInDays null = never.
  async createShareLink(contactId: string, fields: ShareField[], expiresInDays: number | null) {
    return this.client.post<any, ContactShareLink>(`/api/contacts/${contactId}/share-links`, { fields, expiresInDays });
  }

  async listShareLinks(contactId: string) {
    return this.client.get<any, { links: ContactShareLink[] }>(`/api/contacts/${contactId}/share-links`);
  }

  async revokeShareLink(contactId: string, token: string) {
    return this.client.delete<any, ContactShareLink>(`/api/contacts/${contactId}/share-links/${encodeURIComponent(token)}`);
  }

  // ==================== EVENTS & SEATS ====================
  async listEvents() {
    try {
      return await this.client.get<any, Event[]>('/api/events').then(checked(z.array(eventSummarySchema), 'GET /api/events'));
    } catch (error) {
      console.warn('Failed to load events', error);
      // Return empty array when offline/error
      return [];
    }
  }

  /** Teammates who scanned contacts at this event, with their real scan counts. */
  async getEventSeats(eventId: string) {
    return this.client.get<any, Seat[]>(`/api/events/${encodeURIComponent(eventId)}/seats`);
  }

  /** Planned events you're on the team for (or have in your list): dates, venue, address, booth, hours…
   *  With scope 'all': every event the admins scheduled. */
  async listMyPlannedEvents(scope?: 'all') {
    return this.client.get<any, PlannedEvent[]>('/api/me/events', { params: scope ? { scope } : undefined });
  }

  /** Your notes & reminders — all, or one event's. */
  async listEventNotes(event?: string) {
    return this.client.get<any, EventNote[]>('/api/me/event-notes', { params: event ? { event } : undefined });
  }

  async addEventNote(input: EventNoteInput & { event: string; text: string }) {
    return this.client.post<any, EventNote>('/api/me/event-notes', input);
  }

  async updateEventNote(id: string, patch: EventNoteInput) {
    return this.client.patch<any, EventNote>(`/api/me/event-notes/${id}`, patch);
  }

  async deleteEventNote(id: string) {
    return this.client.delete<any, { status: string; id: string }>(`/api/me/event-notes/${id}`);
  }

  // ==================== EMAIL VERIFICATION ====================
  /** Is this address real? `smtp: false` = quick (format + domain only). Cached a week unless `force`. */
  async verifyEmail(email: string, opts?: { force?: boolean; smtp?: boolean }) {
    return this.client.post<any, EmailCheck>('/api/email/verify', { email, ...opts }, { timeout: 60000 });
  }

  /** Cached results for many addresses (no new checks): {email(lower-case): {is_reachable, checked_at}}. */
  async emailStatus(emails: string[]) {
    return this.client.post<any, Record<string, { is_reachable: EmailReachable; checked_at: string | null }>>('/api/email/status', { emails });
  }

  // ==================== SYNC (OFFLINE-FIRST) ====================
  // Batches of up to 25 changes (photos included): give it longer than the default timeout.
  async syncPush(batch: any) {
    return this.client.post('/api/sync/push', batch, { timeout: 90000 });
  }

  async syncPull(params?: { cursor?: string; eventId?: string; limit?: number }) {
    return this.client.get<{ contacts: Contact[]; cursor: string | null; serverTimestamp: string }>('/api/sync/pull', { params });
  }

  // ==================== AI COPILOT & ASSISTANT ====================
  // The response interceptor unwraps `.data`, so this resolves to AIResponse itself.
  async converseAI(query: string, model?: string, conversationHistory?: any[]) {
    return this.client.post<any, AIResponse>('/api/converse', {
      query,
      model: model || 'claude-opus',
      conversationHistory,
    });
  }

  // ==================== EXHIBITORS DIRECTORY ====================
  async listExhibitors(params?: { search?: string; category?: string; hall?: string; event?: string; limit?: number }) {
    return this.client.get<Exhibitor[]>('/api/exhibitors', { params }).then(checked(z.array(exhibitorSchema), 'GET /api/exhibitors'));
  }

  /** Add a booth to an event (returns the existing one if that event already lists the name). */
  async addBooth(input: BoothInput) {
    return this.client.post<any, Exhibitor & { existed: boolean }>('/api/exhibitors', input)
      .then(checked(addedExhibitorSchema, 'POST /api/exhibitors'));
  }

  /** Edit a booth you added. */
  async updateBooth(id: string, patch: Partial<Omit<BoothInput, 'event' | 'visited' | 'note'>>) {
    return this.client.patch<any, Exhibitor>(`/api/exhibitors/${id}`, patch).then(checked(exhibitorSchema, 'PATCH /api/exhibitors/{id}'));
  }

  /** Remove a booth you added. */
  async deleteBooth(id: string) {
    return this.client.delete<any, { status: string; id: string }>(`/api/exhibitors/${id}`);
  }

  /** Mark / un-mark a booth as visited by you. */
  async setBoothVisited(id: string, visited: boolean, note?: string) {
    return visited
      ? this.client.put<any, Exhibitor>(`/api/exhibitors/${id}/visit`, note != null ? { note } : {}).then(checked(exhibitorSchema, 'PUT /api/exhibitors/{id}/visit'))
      : this.client.delete<any, Exhibitor>(`/api/exhibitors/${id}/visit`).then(checked(exhibitorSchema, 'DELETE /api/exhibitors/{id}/visit'));
  }

  /** Exhibitor categories with counts, most common first — for one event with `event`, else all. */
  async getExhibitorCategories(event?: string) {
    return this.client.get<any, { category: string; count: number }[]>('/api/exhibitors/categories', { params: event ? { event } : undefined });
  }

  // ==================== FILES & COLLATERAL ====================
  /** Team files; with `event`, that event's files plus the ones for every event. */
  async listEventFiles(event?: string) {
    return this.client.get<any, EventFile[]>('/api/files/list', { params: event ? { event } : undefined })
      .then(checked(z.array(eventFileSchema), 'GET /api/files/list'));
  }

  async downloadEventFile(fileId: string) {
    return this.client.get<any, FileContent>(`/api/files/${fileId}/download`, { timeout: 120000 })
      .then(checked(fileContentSchema, 'GET /api/files/{id}/download'));
  }

  async getContactFiles(contactId: string) {
    return this.client.get<any, ContactFile[]>(`/api/contact/${contactId}/files`)
      .then(checked(z.array(contactFileSchema), 'GET /api/contact/{id}/files'));
  }

  async downloadContactFile(fileId: string) {
    return this.client.get<any, FileContent>(`/api/contact/files/${fileId}/download`, { timeout: 120000 })
      .then(checked(fileContentSchema, 'GET /api/contact/files/{id}/download'));
  }

  /** Attach a document to one of your contacts (base64 content, max 20 MB). */
  async attachContactFile(contactId: string, file: { filename: string; contentBase64: string; description?: string }) {
    return this.client.post<any, ContactFile>(`/api/contact/${contactId}/files`, file, { timeout: 120000 })
      .then(checked(contactFileSchema, 'POST /api/contact/{id}/files'));
  }

  async deleteContactFile(fileId: string) {
    return this.client.delete<any, { status: string; id: string }>(`/api/contact/files/${fileId}`);
  }

  // ==================== TEAM ANNOUNCEMENTS ====================
  /** Every announcement sent to you, live and ended, newest first (the announcements dropdown). */
  async getMyBroadcasts(limit = 50) {
    return this.client.get<any, Broadcast[]>('/api/broadcasts/mine', { params: { limit } })
      .then(checked(z.array(broadcastSchema), 'GET /api/broadcasts/mine'));
  }

  /**
   * Live announcements feed (Server-Sent Events): onChange fires whenever an admin sends or ends
   * an announcement. onClose fires once when the stream ends for any reason (network, sign-out,
   * server restart) — reconnecting is the caller's job. Returns a function that closes it.
   * XMLHttpRequest, not fetch/EventSource: React Native has neither for streams.
   */
  async openBroadcastStream(handlers: { onOpen?: () => void; onChange: () => void; onClose: (status: number) => void }) {
    const path = '/api/broadcasts/stream';
    let closed = false;
    let current: XMLHttpRequest | null = null;
    const close = (status: number) => {
      if (closed) return;
      closed = true;
      handlers.onClose(status);
    };

    // One connection; an expired token (401) renews once and connects again.
    const connect = async (canRenew: boolean) => {
      const headers = await this.streamHeaders(path, { Accept: 'text/event-stream' });
      if (closed) return;
      const xhr = new XMLHttpRequest();
      current = xhr;
      let seen = 0;
      let pending = '';
      const consume = () => {
        if (xhr.status >= 400) return;
        const text = xhr.responseText || '';
        if (text.length <= seen) return;
        pending += text.slice(seen).replace(/\r\n/g, '\n');
        seen = text.length;
        let cut: number;
        while ((cut = pending.indexOf('\n\n')) !== -1) {
          const block = pending.slice(0, cut);
          pending = pending.slice(cut + 2);
          const event = block.split('\n').find((l) => l.startsWith('event:'))?.slice(6).trim();
          if (event === 'ready') handlers.onOpen?.();
          else if (event === 'announcements') handlers.onChange();
        }
        // a phone left open for hours would keep the whole stream in memory: start a fresh one
        if (seen > 512 * 1024) { xhr.abort(); close(0); }
      };
      xhr.open('GET', `${this.baseURL}${path}`);
      Object.entries(headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));
      xhr.onreadystatechange = () => { if (xhr.readyState >= 3) consume(); };
      xhr.onload = async () => {
        if (xhr.status === 401 && canRenew && !closed) {
          const renewed = await this.renewAfterStream401(path, xhr.responseText || '', headers).catch(() => false);
          if (renewed && !closed) { connect(false).catch(() => close(0)); return; }
        } else if (xhr.status === 401) {
          this.pipeline.checkRevoked({ url: path, status: 401, detail: parseDetail(xhr.responseText || '') });
        }
        close(xhr.status);
      };
      xhr.onerror = () => close(0);
      xhr.onabort = () => close(0);
      xhr.send();
    };

    await connect(true).catch(() => close(0));
    return () => { closed = true; current?.abort(); };
  }

  // ==================== AI SCORING & ENRICHMENT ====================
  // force: ask the AI again even when nothing changed (normally the saved score is kept then).
  async scoreContact(contactId: string, force = false) {
    return this.client.post(`/api/contact/${contactId}/score`, null, { timeout: 60000, params: force ? { force: true } : undefined });
  }

  // Scores up to 25 unscored contacts per call (each is an AI request). 503 = AI unavailable.
  async scoreAllContacts() {
    return this.client.post<any, { scoredCount: number; errors: number; pending: number }>('/api/contacts/score_all', null, { timeout: 180000 });
  }

  // Real, scannable vCard QR for a contact (any phone camera can add it as a contact).
  async getContactQR(contactId: string) {
    return this.client.get<any, { contactId: string; name: string; vcard: string; qrDataUrl: string }>(`/api/contact/${contactId}/qr`);
  }

  // Your own card: vCard QR + public page link (/c/{token}) and that link's QR.
  async getMyCard() {
    return this.client.get<any, MyCard>('/api/me/card');
  }

  // Premium business card PNG (QR left, details right) — 2100×1200 @ 600 dpi.
  async getMyBusinessCard() {
    return this.client.get<any, { pngDataUrl: string; filename: string; width: number; height: number; printed: Record<string, boolean> }>('/api/me/business-card', { timeout: 60000 });
  }

  // New card link; the old /c/… link stops working.
  async resetMyCardLink() {
    return this.client.post<any, MyCard>('/api/me/card/reset-link');
  }

  // ==================== AUTHENTICATION ====================
  // platform 'native' (iOS/Android): LinkedIn returns to the server's https page, which bounces to
  // eventscout://callback; the response's redirectUri is then that app link. Web sends no platform.
  async getLinkedInAuthUrl(state?: string, platform?: 'native'): Promise<{ url: string; redirectUri: string; state: string }> {
    const params: Record<string, string> = {};
    if (state) params.state = state;
    if (platform) params.platform = platform;
    return this.client.get<any, { url: string; redirectUri: string; state: string }>('/auth/linkedin/url', {
      params: Object.keys(params).length ? params : undefined,
    });
  }

  async exchangeLinkedInCode(code: string, platform?: 'native'): Promise<LinkedInSession> {
    // Returns the JWT session issued by the backend. Token persistence is handled by
    // the session module (useSession().login → session.signIn) so that session state stays in sync.
    return this.client.post<any, LinkedInSession>('/auth/linkedin/callback', platform ? { code, platform } : { code });
  }

  // Connects LinkedIn to the signed-in account (Bearer JWT). Does not issue a new session.
  async linkLinkedIn(code: string, platform?: 'native'): Promise<LinkedInLinkResult> {
    return this.client.post<any, LinkedInLinkResult>('/auth/linkedin/link', platform ? { code, platform } : { code });
  }

  async getLinkedInLinkStatus(): Promise<LinkedInLinkStatus> {
    return this.client.get<any, LinkedInLinkStatus>('/auth/linkedin/link');
  }

  // Requires the account password; LinkedIn-created accounts must call setPassword first.
  async unlinkLinkedIn(password: string): Promise<{ status: string; message: string; email: string }> {
    return this.client.post<any, { status: string; message: string; email: string }>('/auth/linkedin/unlink', { password });
  }

  // Emails a 6-digit reset code. Same response whether or not the email has an account.
  async requestPasswordReset(email: string): Promise<{ status: string; message: string; expires_in: number; resend_after: number }> {
    return this.client.post<any, { status: string; message: string; expires_in: number; resend_after: number }>('/auth/password/forgot', { email });
  }

  // Sets the new password with the emailed code and returns a login-shaped session.
  async resetPassword(email: string, code: string, newPassword: string) {
    const response = await this.client.post<any, {
      status: string;
      message: string;
      user_id: string;
      name: string;
      email: string;
      is_admin: boolean;
      token: string;
      refresh_token: string;
    }>('/auth/password/reset', { email, code, new_password: newPassword });
    return {
      token: response.token,
      refresh_token: response.refresh_token,
      user: { id: response.user_id, email: response.email, name: response.name, isAdmin: response.is_admin },
    };
  }

  // First password for a LinkedIn-created account (rejected if one already exists).
  async setPassword(newPassword: string): Promise<{ status: string; message: string; email: string }> {
    return this.client.post<any, { status: string; message: string; email: string }>('/auth/password/set', { new_password: newPassword });
  }

  /** New access + refresh tokens from the saved refresh token (the backend answers `refresh_token`). */
  async refreshToken(refreshToken: string) {
    // One renewal at a time (shared with the 401 renew-and-replay): see requestPipeline.
    return this.pipeline.refreshToken(refreshToken);
  }

  // The streams (XMLHttpRequest — React Native's fetch can't read a response while it arrives) go
  // through the same pipeline as axios: the same headers (the token renewed first if it already
  // expired), and a 401 renews the token once and opens the stream again, like an axios replay.

  private async streamHeaders(path: string, extra: Record<string, string>): Promise<Record<string, string>> {
    return { ...extra, ...(await this.pipeline.headers(path, { fresh: true })) };
  }

  /** After a stream's 401: true when the token was renewed (open it again); a revoked session is reported. */
  private async renewAfterStream401(path: string, responseText: string, sentHeaders: Record<string, string>): Promise<boolean> {
    const detail = parseDetail(responseText);
    const sentToken = (sentHeaders.Authorization || '').replace(/^Bearer /, '') || null;
    const renewed = await this.pipeline.recoverFrom401({ url: path, detail, sentToken });
    if (!renewed) this.pipeline.checkRevoked({ url: path, status: 401, detail });
    return !!renewed;
  }

  // ==================== SCANNING APIs (legacy - app.py) ====================
  async addContactFromImage(imageUri: string, userId: string, opts?: ScanOptions): Promise<ScanResult> {
    // For React Native, we need to handle image upload using FormData
    const formData = new FormData();
    
    // Extract filename from URI
    const filename = imageUri.split('/').pop() || 'scan.jpg';
    const fileType = filename.endsWith('.png') ? 'image/png' : 'image/jpeg';
    
    if (Platform.OS === 'web') {
      const response = await fetch(imageUri);
      const blob = await response.blob();
      formData.append('file', blob, filename);
    } else {
      // Append the file (React Native format)
      formData.append('file', {
        uri: imageUri,
        type: fileType,
        name: filename,
      } as any);
    }
    
    // Append user_id as query parameter since FastAPI expects it in the URL.
    // OCR (OpenRouter vision model, then a fallback model — up to ~35 s each) can take a while;
    // wait longer than the server's worst case so a slow read isn't mistaken for "offline".
    return this.client.post<any, ScanResult>(`/add_contact_from_image/?${scanQuery(userId, opts)}`, formData, { timeout: 90000 });
  }

  async scanQRCode(imageUri: string, userId: string, opts?: ScanOptions): Promise<ScanResult> {
    const formData = new FormData();
    
    const filename = imageUri.split('/').pop() || 'qr.jpg';
    const fileType = filename.endsWith('.png') ? 'image/png' : 'image/jpeg';
    
    if (Platform.OS === 'web') {
      const response = await fetch(imageUri);
      const blob = await response.blob();
      formData.append('file', blob, filename);
    } else {
      formData.append('file', {
        uri: imageUri,
        type: fileType,
        name: filename,
      } as any);
    }
    
    // The backend reads user_id from the query string (like /add_contact_from_image/);
    // sending it as a form field made every QR scan fail with 422.
    return this.client.post<any, ScanResult>(`/scan_qr/?${scanQuery(userId, opts)}`, formData);
  }

  // The AI answer can take a while (retrieval + LLM), so these get longer timeouts than the default.
  // ui: 'openui' lets the answer include OpenUI Lang blocks (rendered by src/features/ask/openui).
  async converse(query: string, userId: string, conversationHistory?: ChatTurn[], ui?: 'openui') {
    return this.client.post<any, ConverseResponse>('/converse/', {
      query,
      user_id: userId,
      conversation_history: conversationHistory,
      ...(ui ? { ui_format: ui } : {}),
    }, { timeout: 90000 }).then((r) => checkRetrieved(r, 'POST /converse/'));
  }

  // Ask about attached files (images, PDFs, text/CSV — max 3, 10 MB each) plus your contacts.
  async converseWithFiles(query: string, userId: string, files: ChatAttachment[], conversationHistory?: ChatTurn[], ui?: 'openui') {
    const formData = await this.converseForm(query, userId, files, conversationHistory, ui);
    return this.client.post<any, ConverseResponse>('/converse/with_files/', formData, { timeout: 120000 })
      .then((r) => checkRetrieved(r, 'POST /converse/with_files/'));
  }

  private async converseForm(query: string, userId: string, files: ChatAttachment[], conversationHistory?: ChatTurn[], ui?: 'openui') {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('user_id', userId);
    if (ui) formData.append('ui_format', ui);
    if (conversationHistory?.length) formData.append('conversation_history', JSON.stringify(conversationHistory));
    for (const f of files) {
      if (Platform.OS === 'web') {
        const blob = await (await fetch(f.uri)).blob();
        formData.append('files', blob, f.name);
      } else {
        formData.append('files', { uri: f.uri, name: f.name, type: f.mimeType || 'application/octet-stream' } as any);
      }
    }
    return formData;
  }

  /**
   * converse / converseWithFiles, streamed (Server-Sent Events): onDelta gets the whole answer so far
   * each time more arrives, and the promise resolves with the full ConverseResponse. Errors are the
   * same ApiError as every other call (kind 'timeout' / 'canceled' / 'offline'…). A server
   * without the streaming routes (404/405) gets the normal request instead.
   * XMLHttpRequest, not fetch: React Native's fetch can't read a response while it arrives.
   */
  async converseStream(query: string, userId: string, opts: {
    history?: ChatTurn[];
    files?: ChatAttachment[];
    ui?: 'openui';
    signal?: AbortSignal;
    onMeta?: (meta: { format?: 'openui' | 'text'; retrieved_contacts: RetrievedContact[] }) => void;
    onModel?: (model: string) => void;
    onDelta?: (answerSoFar: string) => void;
  } = {}): Promise<ConverseResponse> {
    const files = opts.files || [];
    const path = files.length ? '/converse/with_files/stream/' : '/converse/stream/';
    const extra: Record<string, string> = { Accept: 'text/event-stream' };
    if (!files.length) extra['Content-Type'] = 'application/json';

    const IDLE_MS = 90000; // no bytes for this long → give up (the AI can think a while before it writes)

    // One attempt; 'renew' = the token had expired (401): renew it once and ask again.
    const attempt = async (canRenew: boolean): Promise<ConverseResponse | 'fallback' | 'renew'> => {
      // FormData / JSON body made per attempt so a retry sends it again from scratch.
      const body = files.length
        ? await this.converseForm(query, userId, files, opts.history, opts.ui)
        : JSON.stringify({ query, user_id: userId, conversation_history: opts.history, ...(opts.ui ? { ui_format: opts.ui } : {}) });
      const headers = await this.streamHeaders(path, extra);

      return new Promise<ConverseResponse | 'fallback' | 'renew'>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        let seen = 0;
        let pending = '';
        let answer = '';
        let settled = false;
        let idle: ReturnType<typeof setTimeout> | null = null;

        const fail = (err: any) => {
          if (settled) return;
          settled = true;
          if (idle) clearTimeout(idle);
          opts.signal?.removeEventListener('abort', onAbort);
          xhr.abort();
          reject(classifyError(err, { sessionRoute: false }));
        };
        const succeed = (res: ConverseResponse | 'fallback' | 'renew') => {
          if (settled) return;
          settled = true;
          if (idle) clearTimeout(idle);
          opts.signal?.removeEventListener('abort', onAbort);
          resolve(res);
        };
        const httpError = (status: number, detail?: string) =>
          Object.assign(new Error(detail || `HTTP ${status}`), { response: { status, data: detail ? { detail } : {} } });
        const resetIdle = () => {
          if (idle) clearTimeout(idle);
          idle = setTimeout(() => fail(Object.assign(new Error('timeout'), { code: 'ECONNABORTED' })), IDLE_MS);
        };
        const onAbort = () => fail(Object.assign(new Error('aborted'), { code: 'ERR_CANCELED' }));

        const handleEvent = (block: string) => {
          let event = 'message';
          const data: string[] = [];
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) event = line.slice(6).trim();
            else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
          }
          if (!data.length) return; // comment / keep-alive
          let payload: any;
          try { payload = JSON.parse(data.join('\n')); } catch { return; }
          if (event === 'meta') {
            opts.onMeta?.({
              format: payload.format,
              retrieved_contacts: checkResponse(retrievedContactsSchema, payload.retrieved_contacts, `SSE meta ${path} (retrieved_contacts)`) || [],
            });
          } else if (event === 'model') opts.onModel?.(payload.model);
          else if (event === 'delta' && typeof payload.text === 'string') {
            answer += payload.text;
            opts.onDelta?.(answer);
          } else if (event === 'done') succeed(checkRetrieved(payload as ConverseResponse, `SSE done ${path}`));
          else if (event === 'error') fail(httpError(503, payload.detail));
        };

        // Parses whatever arrived since last time; complete events end with a blank line.
        const consume = () => {
          if (xhr.status && xhr.status >= 400) return;
          const text = xhr.responseText || '';
          if (text.length <= seen) return;
          pending += text.slice(seen).replace(/\r\n/g, '\n');
          seen = text.length;
          resetIdle();
          let cut: number;
          while (!settled && (cut = pending.indexOf('\n\n')) !== -1) {
            const block = pending.slice(0, cut);
            pending = pending.slice(cut + 2);
            handleEvent(block);
          }
        };

        xhr.open('POST', `${this.baseURL}${path}`);
        Object.entries(headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));
        xhr.onreadystatechange = () => {
          if (xhr.readyState >= 3) consume();
        };
        xhr.onload = async () => {
          if (xhr.status === 404 || xhr.status === 405) return succeed('fallback');
          if (xhr.status >= 400) {
            const detail = parseDetail(xhr.responseText || '');
            if (xhr.status === 401) {
              if (canRenew && !settled) {
                const renewed = await this.renewAfterStream401(path, xhr.responseText || '', headers).catch(() => false);
                if (renewed) return succeed('renew');
              } else {
                this.pipeline.checkRevoked({ url: path, status: 401, detail });
              }
            }
            return fail(httpError(xhr.status, typeof detail === 'string' ? detail : undefined));
          }
          consume();
          if (pending.trim()) handleEvent(pending.trim());
          fail(httpError(502, 'The answer was cut off. Please try again.'));
        };
        xhr.onerror = async () => {
          const offline = API_CONFIG.OFFLINE_MODE_ENABLED && !(await this.checkNetworkConnection());
          fail(offline ? new Error('OFFLINE') : new Error('Network Error'));
        };

        if (opts.signal?.aborted) return onAbort();
        opts.signal?.addEventListener('abort', onAbort);
        resetIdle();
        xhr.send(body as any);
      });
    };

    let streamed = await attempt(true);
    if (streamed === 'renew') streamed = await attempt(false);
    if (streamed === 'renew') throw classifyError({ message: 'Unauthorized', response: { status: 401, data: {} } });

    if (streamed !== 'fallback') return streamed;
    return files.length
      ? this.converseWithFiles(query, userId, files, opts.history, opts.ui)
      : this.converse(query, userId, opts.history, opts.ui);
  }

  // Voice question → text (OpenRouter audio input). The recording is an .m4a on iOS/Android.
  async transcribeAudio(audioUri: string, userId: string): Promise<{ status: string; transcript: string }> {
    const formData = new FormData();
    const name = audioUri.split('/').pop() || 'question.m4a';
    if (Platform.OS === 'web') {
      const blob = await (await fetch(audioUri)).blob();
      formData.append('file', blob, name.includes('.') ? name : 'question.webm');
    } else {
      formData.append('file', { uri: audioUri, name, type: name.endsWith('.3gp') ? 'audio/3gpp' : 'audio/mp4' } as any);
    }
    return this.client.post<any, { status: string; transcript: string }>(
      `/transcribe_audio/?user_id=${encodeURIComponent(userId)}`, formData, { timeout: 45000 },
    );
  }

  // ==================== AUTHENTICATION (legacy endpoints) ====================
  async register(name: string, email: string, password: string) {
    const response = await this.client.post<any, {
      status: string;
      message: string;
      user_id: string;
      name: string;
      email: string;
      is_admin: boolean;
      token?: string;
      refresh_token?: string;
    }>('/register/', { name, email, password });
    
    return {
      status: response.status,
      message: response.message,
      user_id: response.user_id,
      token: response.token,
      refresh_token: response.refresh_token,
      user: {
        id: response.user_id,
        email: response.email,
        name: response.name,
        isAdmin: response.is_admin
      }
    };
  }

  async login(email: string, password: string) {
    const response = await this.client.post<any, {
      status: string;
      message: string;
      user_id: string;
      name: string;
      email: string;
      is_admin: boolean;
      token?: string;
      refresh_token?: string;
    }>('/login/', { email, password });
    
    return {
      status: response.status,
      message: response.message,
      user_id: response.user_id,
      token: response.token,
      refresh_token: response.refresh_token,
      user: {
        id: response.user_id,
        email: response.email,
        name: response.name,
        isAdmin: response.is_admin
      }
    };
  }

  async exportContacts(userId: string, format: 'csv' | 'json') {
    return this.client.get(`/export_contacts/`, {
      params: { user_id: userId, format },
    });
  }
}

/** An assistant answer, with its retrieved_contacts checked against the contract (development only). */
function checkRetrieved<T extends { retrieved_contacts?: unknown } | null | undefined>(response: T, endpoint: string): T {
  checkResponse(retrievedContactsSchema, response?.retrieved_contacts, `${endpoint} (retrieved_contacts)`);
  return response;
}

/** `detail` from an error body, if it is JSON. */
function parseDetail(text: string): unknown {
  try { return JSON.parse(text)?.detail; } catch { return undefined; }
}

// Singleton instance
export const api = new EventScoutAPI();