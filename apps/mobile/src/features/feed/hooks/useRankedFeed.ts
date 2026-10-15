import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type { Contact } from '@es/types';
import type { FeedFilters } from '../../../db/queries/feed';
import { rankContacts } from '../rank';
import { errorMessage, type ErrorMessageOptions } from '../../../lib/api';
import { syncEvents } from '../../../sync/offline';
import { contactsFeature, contactEvents } from '../../contacts';

// Safety cap for `allPages` (a rep's own contacts; the list is rendered virtualised).
const MAX_ALL_PAGES_CONTACTS = 5000;

// How a list that couldn't load is worded (see errorMessage).
const LOAD_ERRORS: ErrorMessageOptions = {
  unreachable: "Couldn't reach the server.",
  unauthorized: 'Your session has ended. Please sign in again.',
};

export function useRankedFeed(filters: FeedFilters = {}) {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loading, setLoading] = useState(true);
  // Set when the server couldn't be reached; `contacts` then holds whatever the device has
  // cached locally (possibly nothing), so screens must not present it as the full list.
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  // Memoize filters to prevent unnecessary reloads
  const memoizedFilters = useMemo(() => filters, [
    filters.eventId,
    filters.userId,
    filters.stage,
    filters.scoreBand,
    filters.search,
    filters.limit,
    filters.offset,
    filters.sort,
    filters.allPages,
  ]);

  const loadData = useCallback(async (filters: FeedFilters) => {
    try {
      setLoading(true);
      // Server first (every page for the whole list, so search / counts cover every contact); the
      // repository falls back to the device cache when the server can't be reached.
      const res = await contactsFeature.listContacts({
        search: filters.search,
        stage: filters.stage,
        scoreBand: filters.scoreBand,
        eventId: filters.eventId,
        userId: filters.userId,
        limit: filters.limit,
        cursor: filters.cursor,
        sort: filters.sort,
        allPages: filters.allPages,
        cap: MAX_ALL_PAGES_CONTACTS,
      }, { fallbackToCache: true });
      if (mountedRef.current) {
        setContacts(rankContacts(res.contacts));
        setError(res.error ? errorMessage(res.error, "Couldn't load contacts.", LOAD_ERRORS) : null);
      }
    } catch (err) {
      console.error('Failed to load ranked feed', err);
      if (mountedRef.current) setContacts([]);
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    loadData(memoizedFilters);

    return () => {
      mountedRef.current = false;
    };
  }, [loadData, memoizedFilters]);

  const reload = useCallback(() => {
    loadData(memoizedFilters);
  }, [loadData, memoizedFilters]);

  // Offline saves and background uploads change the list: refresh it.
  useEffect(() => syncEvents.subscribe((e) => {
    if (e.type === 'queued' || e.pushed || e.scans) loadData(memoizedFilters);
  }), [loadData, memoizedFilters]);

  // A scan discarded on the review screen leaves the list right away.
  useEffect(() => contactEvents.subscribe((e) => {
    if (e.type === 'discarded' && mountedRef.current) setContacts((prev) => prev.filter((c) => c.id !== e.id));
  }), []);

  return { contacts, loading, error, reload };
}
