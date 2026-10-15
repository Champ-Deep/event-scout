// Query options for every piece of server data. Routes `ensureQueryData(...)` in their loader,
// components `useSuspenseQuery(...)` the same options (they read the cache, never fetch on their
// own), mutations and the SSE stream write to the same keys. All keys start with 'admin'.
import { queryOptions } from '@tanstack/react-query';
import { api, type ContactQuery, type ContactStatsQuery } from './api';
import type { SegmentDim } from './types';
import { dayKey, localDayRange } from './dates';
import type { ActivityCalendar } from './types';

export const keys = {
  all: ['admin'] as const,
  dashboard: (days: number) => ['admin', 'dashboard', days] as const,
  activity: (limit: number) => ['admin', 'activity', limit] as const,
  activityDay: (date: string) => ['admin', 'activity', 'day', date] as const,
  activityCalendar: (start: string) => ['admin', 'activity', 'calendar', start] as const,
  contacts: (q: ContactQuery) => ['admin', 'contacts', 'list', q] as const,
  contactsAll: ['admin', 'contacts'] as const,
  contactStats: (f: ContactStatsQuery) => ['admin', 'contacts', 'stats', f] as const,
  contact: (id: string) => ['admin', 'contacts', 'one', id] as const,
  contactFiles: (id: string) => ['admin', 'contacts', 'files', id] as const,
  contactQr: (id: string, v?: string) => ['admin', 'contacts', 'qr', id, v] as const,
  contactDuplicates: (id: string) => ['admin', 'contacts', 'duplicates', id] as const,
  users: ['admin', 'users'] as const,
  events: ['admin', 'events'] as const,
  lists: ['admin', 'lists'] as const,
  listMembers: (id: string) => ['admin', 'lists', id, 'members'] as const,
  duplicates: ['admin', 'duplicates'] as const,
  deleted: ['admin', 'deleted'] as const,
  deletedOne: (id: string) => ['admin', 'deleted', id] as const,
  pipelines: (status: string | undefined, offset: number) => ['admin', 'pipelines', 'all', status ?? 'all', offset] as const,
  pipelineFor: (id: string) => ['admin', 'pipelines', 'contact', id] as const,
  broadcasts: ['admin', 'broadcasts'] as const,
  exhibitors: (event: string) => ['admin', 'exhibitors', event] as const,
  adminEvents: ['admin', 'events', 'all'] as const,
  eventDetail: (event: string) => ['admin', 'events', 'detail', event] as const,
  files: ['admin', 'files'] as const,
  planner: ['admin', 'planner'] as const,
  segments: ['admin', 'segments'] as const,
  segmentContacts: (dim: string, key: string) => ['admin', 'segments', dim, key] as const,
  broadcastHistory: ['admin', 'broadcasts', 'history'] as const,
  broadcastHistoryDay: (date: string) => ['admin', 'broadcasts', 'history', 'day', date] as const,
  broadcastCalendar: (start: string) => ['admin', 'broadcasts', 'history', 'calendar', start] as const,
  plannerDiscovery: ['admin', 'planner', 'discovery'] as const,
  models: ['admin', 'models'] as const,
  me: ['admin', 'me'] as const,
  salesProfile: ['admin', 'sales-profile'] as const,
  health: ['admin', 'health'] as const,
  backup: ['admin', 'backup'] as const,
  crms: ['admin', 'crm'] as const,
};

export const q = {
  dashboard: (days: number) => queryOptions({ queryKey: keys.dashboard(days), queryFn: () => api.dashboard(days), refetchInterval: 60_000 }),
  activity: (limit: number) => queryOptions({ queryKey: keys.activity(limit), queryFn: () => api.activity(limit), refetchInterval: 60_000 }),
  /** everything on one local day (YYYY-MM-DD), from local midnight to the next */
  activityDay: (date: string) => queryOptions({ queryKey: keys.activityDay(date), queryFn: () => api.activity(500, localDayRange(date)), refetchInterval: 60_000 }),
  /** per-day totals for the calendar grid that starts on start (YYYY-MM-DD, 42 days) */
  activityCalendar: (start: string) => queryOptions({
    queryKey: keys.activityCalendar(start),
    queryFn: () => { const r = localDayRange(start, 42); return api.activityCalendar(r.start, r.end, new Date(r.start).getTimezoneOffset()); },
    staleTime: 60_000,
  }),
  contacts: (query: ContactQuery) => queryOptions({ queryKey: keys.contacts(query), queryFn: () => api.contacts(query) }),
  contactStats: (f: ContactStatsQuery) => queryOptions({ queryKey: keys.contactStats(f), queryFn: () => api.contactStats(f) }),
  contact: (id: string) => queryOptions({ queryKey: keys.contact(id), queryFn: () => api.contact(id) }),
  contactFiles: (id: string) => queryOptions({ queryKey: keys.contactFiles(id), queryFn: () => api.contactFiles(id) }),
  // the QR encodes the contact's current details: keyed by updatedAt so an edit refreshes it
  contactQr: (id: string, v?: string) => queryOptions({ queryKey: keys.contactQr(id, v), queryFn: () => api.contactQr(id) }),
  contactDuplicates: (id: string) => queryOptions({ queryKey: keys.contactDuplicates(id), queryFn: () => api.contactDuplicates(id) }),
  users: () => queryOptions({ queryKey: keys.users, queryFn: () => api.users() }),
  events: () => queryOptions({ queryKey: keys.events, queryFn: () => api.events() }),
  lists: () => queryOptions({ queryKey: keys.lists, queryFn: () => api.lists() }),
  listMembers: (id: string) => queryOptions({ queryKey: keys.listMembers(id), queryFn: () => api.listMembers(id) }),
  // always re-checked when the page opens (the cached list shows meanwhile): pairs change with every edit
  duplicates: () => queryOptions({ queryKey: keys.duplicates, queryFn: () => api.duplicates(), staleTime: 0 }),
  deleted: () => queryOptions({ queryKey: keys.deleted, queryFn: () => api.deletedContacts() }),
  deletedContact: (id: string) => queryOptions({ queryKey: keys.deletedOne(id), queryFn: () => api.deletedContact(id) }),
  /** one page of runs (newest first) for a status filter (running / complete / failed, or every run) */
  pipelines: (status?: string, offset = 0) => queryOptions({ queryKey: keys.pipelines(status, offset), queryFn: () => api.pipelines(status, offset) }),
  pipelineFor: (id: string) => queryOptions({ queryKey: keys.pipelineFor(id), queryFn: () => api.pipelineFor(id) }),
  broadcasts: () => queryOptions({ queryKey: keys.broadcasts, queryFn: () => api.broadcasts(), refetchInterval: 30_000 }),
  exhibitors: (event: string) => queryOptions({ queryKey: keys.exhibitors(event), queryFn: () => api.exhibitors(event || undefined) }),
  adminEvents: () => queryOptions({ queryKey: keys.adminEvents, queryFn: () => api.adminEvents() }),
  eventDetail: (event: string) => queryOptions({ queryKey: keys.eventDetail(event), queryFn: () => api.eventDetail(event), refetchInterval: 60_000 }),
  files: () => queryOptions({ queryKey: keys.files, queryFn: () => api.files() }),
  broadcastHistory: () => queryOptions({ queryKey: keys.broadcastHistory, queryFn: () => api.broadcastHistory() }),
  /** announcements sent on one local day (YYYY-MM-DD) */
  broadcastHistoryDay: (date: string) => queryOptions({ queryKey: keys.broadcastHistoryDay(date), queryFn: () => api.broadcastHistory(localDayRange(date), 2000) }),
  /** per local day: how many announcements, split normal / urgent — for the 42-day calendar grid starting on start */
  broadcastCalendar: (start: string) => queryOptions({
    queryKey: keys.broadcastCalendar(start),
    queryFn: async (): Promise<ActivityCalendar> => {
      const list = await api.broadcastHistory(localDayRange(start, 42), 2000);
      const days: ActivityCalendar['days'] = {};
      for (const b of list) {
        if (!b.createdAt) continue;
        const k = dayKey(new Date(b.createdAt));
        const d = (days[k] ??= { total: 0, types: {} }) as { total: number; types: Record<string, number> };
        d.total += 1;
        const kind = b.priority === 'urgent' ? 'urgent' : 'normal';
        d.types[kind] = (d.types[kind] ?? 0) + 1;
      }
      return { days };
    },
    staleTime: 60_000,
  }),
  // re-sorted on the server on every request; poll fast while the AI is still sorting companies into industries
  segments: () => queryOptions({ queryKey: keys.segments, queryFn: () => api.segments(), refetchInterval: (query) => (query.state.data?.industry.pending ? 8_000 : 60_000) }),
  segmentContacts: (dim: SegmentDim, key: string) => queryOptions({ queryKey: keys.segmentContacts(dim, key), queryFn: () => api.segmentContacts(dim, key) }),
  plannerEvents: () => queryOptions({ queryKey: keys.planner, queryFn: () => api.plannerEvents() }),
  plannerDiscovery: () => queryOptions({ queryKey: keys.plannerDiscovery, queryFn: () => api.plannerDiscovery(), staleTime: 5 * 60_000 }),
  models: () => queryOptions({ queryKey: keys.models, queryFn: () => api.models() }),
  me: () => queryOptions({ queryKey: keys.me, queryFn: () => api.me(), staleTime: 60_000 }),
  salesProfile: () => queryOptions({ queryKey: keys.salesProfile, queryFn: () => api.salesProfile(), staleTime: 60_000 }),
  health: () => queryOptions({ queryKey: keys.health, queryFn: () => api.health(), refetchInterval: 30_000 }),
  backup: () => queryOptions({ queryKey: keys.backup, queryFn: () => api.backupStatus() }),
  crms: () => queryOptions({ queryKey: keys.crms, queryFn: () => api.crms() }),
};
