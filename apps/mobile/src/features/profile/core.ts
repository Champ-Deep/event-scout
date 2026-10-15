/**
 * The rep's profile (GET / PUT /user/profile/) and the active event.
 *
 * One module owns loading and saving the profile, the phone's offline copy of the active event's
 * name (offline scans are tagged with it), and the rules for the rep's event list: add, remove,
 * end, reopen, activate. Pure module: the backend and the offline copy are adapters (./index.ts
 * wires the real ones).
 */

export interface Product {
  name: string;
  description?: string;
  ideal_customer?: string;
}

export interface EventDetailEntry {
  goals: string[];
  description: string;
}

/** The profile fields the app reads (the server may send more). */
export interface Profile {
  full_name?: string;
  job_title?: string;
  company_name?: string;
  company_website?: string;
  linkedin_url?: string;
  phone?: string;
  /** data-URI photo ('' / missing = none) */
  avatar?: string;
  products?: Product[];
  target_industries?: string[];
  target_roles?: string[];
  target_company_sizes?: string[];
  target_geographies?: string[];
  pitch_style?: string;
  value_propositions?: string[];
  common_objections?: { objection: string; response: string }[];
  case_studies?: { title: string; result: string }[];
  /** the rep's events */
  events?: string[];
  /** the active event: new scans are tagged with it ('' = none) */
  current_event_name?: string;
  current_event_description?: string;
  /** events the rep ended (still in the list, marked ended) */
  ended_events?: string[];
  /** goals per event */
  event_details?: Record<string, EventDetailEntry>;
  event_goals?: string[];
  auto_score_on_add?: boolean;
  [key: string]: unknown;
}

export type ProfilePatch = Partial<Profile>;

export interface ProfileRemote {
  get(userId: string): Promise<any>;
  update(userId: string, patch: ProfilePatch): Promise<any>;
}

/** Where the active event's name is kept for offline scans. */
export interface ActiveEventCopy {
  get(): string | undefined;
  set(name: string): void;
}

// ─── event-list rules ───────────────────────────────────────────────────────

const same = (a: string, b: string) => a.toLowerCase() === b.toLowerCase();

export const eventRules = {
  inList(profile: Profile | null | undefined, name: string): boolean {
    return (profile?.events || []).some((e) => same(e, name));
  },

  isEnded(profile: Profile | null | undefined, name: string): boolean {
    return (profile?.ended_events || []).some((e) => same(e, name));
  },

  isActive(profile: Profile | null | undefined, name: string): boolean {
    return profile?.current_event_name === name;
  },

  /** Make `name` the active event: it joins the rep's events, and an ended event is reopened. */
  activate(profile: Profile | null | undefined, name: string): ProfilePatch {
    const events = profile?.events || [];
    return {
      current_event_name: name,
      events: events.includes(name) || eventRules.inList(profile, name) ? events : [...events, name],
      // making an ended event active reopens it
      ...(eventRules.isEnded(profile, name)
        ? { ended_events: (profile?.ended_events || []).filter((e) => !same(e, name)) }
        : {}),
    };
  },

  /** Add `name` to the rep's events. `makeCurrent` (or no active event yet) also activates it. */
  add(profile: Profile | null | undefined, name: string, opts: { makeCurrent?: boolean } = {}): ProfilePatch {
    const current = !!opts.makeCurrent || !profile?.current_event_name;
    if (current) return eventRules.activate(profile, name);
    return { events: [...(profile?.events || []), name] };
  },

  remove(profile: Profile | null | undefined, name: string): ProfilePatch {
    return { events: (profile?.events || []).filter((e) => e !== name) };
  },

  /** End the event (it stays listed, marked ended); ending the active one clears it. */
  end(profile: Profile | null | undefined, name: string): ProfilePatch {
    return {
      ended_events: [...(profile?.ended_events || []), name],
      ...(eventRules.isActive(profile, name) ? { current_event_name: '' } : {}),
    };
  },

  reopen(profile: Profile | null | undefined, name: string): ProfilePatch {
    return { ended_events: (profile?.ended_events || []).filter((e) => !same(e, name)) };
  },

  toggleEnded(profile: Profile | null | undefined, name: string): ProfilePatch {
    return eventRules.isEnded(profile, name) ? eventRules.reopen(profile, name) : eventRules.end(profile, name);
  },

  /** No active event: new scans aren't tagged. */
  clearActive(): ProfilePatch {
    return { current_event_name: '' };
  },

  /** `events` patch that lists `name` if it isn't yet (e.g. to keep its goals). */
  ensureListed(profile: Profile | null | undefined, name: string): ProfilePatch {
    return eventRules.inList(profile, name) ? {} : { events: [...(profile?.events || []), name] };
  },
};

// ─── the store ──────────────────────────────────────────────────────────────

/** The profile inside a GET / PUT answer ({ profile } or the profile itself). */
function profileOf(res: any): Profile | null {
  const p = res && typeof res === 'object' ? (res.profile ?? res) : null;
  return p && typeof p === 'object' ? (p as Profile) : null;
}

export function createProfileStore(deps: { remote: ProfileRemote; activeEvent: ActiveEventCopy }) {
  const { remote, activeEvent } = deps;
  let last: { userId: string; profile: Profile } | null = null;

  /** Keep the offline copy of the active event in step with what the server holds. */
  const remember = (profile: Profile | null, patch?: ProfilePatch) => {
    const name = profile && 'current_event_name' in profile ? profile.current_event_name
      : patch && 'current_event_name' in patch ? patch.current_event_name
        : undefined;
    if (name === undefined) return;
    activeEvent.set(typeof name === 'string' ? name : '');
  };

  const store = {
    /** GET the profile ({} when the server sent none). Also refreshes the active event's offline copy. */
    async load(userId: string): Promise<Profile> {
      const profile = profileOf(await remote.get(userId)) ?? {};
      remember(profile);
      last = { userId, profile };
      return profile;
    },

    /** PUT a change; resolves with the saved profile (or the patch, if the server sent none back). */
    async save(userId: string, patch: ProfilePatch): Promise<Profile> {
      if (!userId) throw new Error('Sign in again to save.');
      const saved = profileOf(await remote.update(userId, patch));
      remember(saved, patch);
      const result = saved ?? patch;
      if (last?.userId === userId) last = { userId, profile: { ...last.profile, ...result } };
      return result;
    },

    /** The last profile loaded / saved for this account (no request). */
    peek(userId: string | null | undefined): Profile | null {
      return userId && last?.userId === userId ? last.profile : null;
    },

    /** The active event's name as kept on the phone (for offline scans); null = none. */
    activeEventName(): string | null {
      return activeEvent.get() || null;
    },

    // Event-list actions: each saves the rule's patch and resolves with what to merge into the profile.
    activateEvent: (userId: string, profile: Profile | null | undefined, name: string) =>
      store.save(userId, eventRules.activate(profile, name)),
    addEvent: (userId: string, profile: Profile | null | undefined, name: string, opts: { makeCurrent?: boolean } = {}) =>
      store.save(userId, eventRules.add(profile, name, opts)),
    removeEvent: (userId: string, profile: Profile | null | undefined, name: string) =>
      store.save(userId, eventRules.remove(profile, name)),
    toggleEnded: (userId: string, profile: Profile | null | undefined, name: string) =>
      store.save(userId, eventRules.toggleEnded(profile, name)),
    clearActiveEvent: (userId: string) => store.save(userId, eventRules.clearActive()),
  };
  return store;
}

export type ProfileStore = ReturnType<typeof createProfileStore>;
