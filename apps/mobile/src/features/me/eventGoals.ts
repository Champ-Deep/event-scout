/**
 * Goals / notes are kept PER EVENT in the profile (profile.event_details[eventName]). They used to
 * be one global list (event_goals), so every event showed the same goals. The backend mirrors the
 * current event's entry into event_goals / current_event_description for the AI prompts.
 */
export interface EventDetail { goals: string[]; description: string }

const EMPTY: EventDetail = { goals: [], description: '' };

function findKey(details: Record<string, any>, name: string): string | undefined {
  const n = name.trim().toLowerCase();
  return Object.keys(details).find((k) => k.trim().toLowerCase() === n);
}

/** This event's goals. Profiles saved before per-event goals: the old list belongs to the current event. */
export function eventDetailFor(profile: any, name: string | null | undefined): EventDetail {
  if (!name) return EMPTY;
  const details = profile?.event_details;
  if (details && typeof details === 'object') {
    const k = findKey(details, name);
    const d = k ? details[k] : null;
    return d ? { goals: Array.isArray(d.goals) ? d.goals : [], description: d.description || '' } : EMPTY;
  }
  if (name === profile?.current_event_name) {
    return { goals: profile?.event_goals || [], description: profile?.current_event_description || '' };
  }
  return EMPTY;
}

/** The full event_details map to save, with this event's entry replaced. */
export function withEventDetail(profile: any, name: string, detail: EventDetail): Record<string, EventDetail> {
  const out: Record<string, EventDetail> = {};
  const details = profile?.event_details;
  if (details && typeof details === 'object') {
    for (const [k, v] of Object.entries(details)) out[k] = v as EventDetail;
  } else if (profile?.current_event_name && (profile?.event_goals?.length || profile?.current_event_description)) {
    // keep the old global goals on the event they were written for
    out[profile.current_event_name] = { goals: profile.event_goals || [], description: profile.current_event_description || '' };
  }
  const k = findKey(out, name);
  if (k && k !== name) delete out[k];
  out[name] = { goals: detail.goals, description: detail.description.trim() };
  return out;
}
