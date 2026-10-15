// Every event name the console knows: the admin events list (booth directory ∪ contacts' "met at" ∪
// reps' event lists ∪ event files) merged with the planner's events. /api/events alone only lists
// events that have a booth directory, so pickers built on it missed most events.
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { q } from './queries';

export function useEventNames(enabled = true): string[] {
  const all = useQuery({ ...q.adminEvents(), enabled });
  const planned = useQuery({ ...q.plannerEvents(), enabled });
  return useMemo(() => {
    const seen = new Map<string, string>();
    for (const n of [...(all.data ?? []).map((e) => e.name), ...(planned.data ?? []).map((e) => e.name)]) {
      const name = (n || '').trim();
      const k = name.toLowerCase();
      if (name && !seen.has(k)) seen.set(k, name);
    }
    return [...seen.values()].sort((a, b) => a.localeCompare(b));
  }, [all.data, planned.data]);
}
