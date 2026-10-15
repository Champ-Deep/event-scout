import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import * as Popover from '@radix-ui/react-popover';
import { AnimatePresence, animate, motion, useMotionValue, useTransform } from 'motion/react';
import { useReduceMotion } from '../../lib/motion';
import { q } from '../../lib/queries';
import type { StreamStatus } from '../../lib/sse';
import type { ActivityItem, ActivityType } from '../../lib/types';
import { Icon, type IconName } from '../../ui/icons';
import { cx, fmtDateTime, ink, relTime } from '../../ui/kit';
import { ACTIVITY_STYLE, activityLink, dayLabel } from './ActivityFeed';
import { dayKey } from '../../lib/dates';
import { levelOf } from '../../lib/levels';

const SEEN_KEY = 'es.activity.seenAt';
const readSeen = () => { try { return Number(localStorage.getItem(SEEN_KEY)) || null; } catch { return null; } };
const writeSeen = (t: number) => { try { localStorage.setItem(SEEN_KEY, String(t)); } catch { /* private window */ } };
const ts = (a: ActivityItem) => Date.parse(a.timestamp) || 0;

// the lead levels' rule and ramp (lib/levels), like the dashboard and the app's band chips
const band = (s: number) => { const l = levelOf(s)!; return { label: l.label, color: l.color }; };
const tint = (c: string, pct: number) => `color-mix(in oklab, ${c} ${pct}%, transparent)`;

/** Newest items shown per tab; "View all" opens the full timeline. */
const PER_TAB = 40;

type Filter = 'all' | 'contacts' | 'scores' | 'pipelines' | 'team';
const FILTERS: { value: Filter; label: string; icon: IconName; types: ActivityType[]; empty: [string, string] }[] = [
  { value: 'all', label: 'All', icon: 'activity', types: [],
    empty: ['Nothing yet', 'Scans, scores, merges and announcements show up here as they happen.'] },
  { value: 'contacts', label: 'Contacts', icon: 'contacts', types: ['contact_added', 'contact_deleted', 'contact_merged'],
    empty: ['No contact activity yet', 'New, deleted and merged contacts show up here.'] },
  { value: 'scores', label: 'Scores', icon: 'flame', types: ['contact_scored'],
    empty: ['No scores yet', 'AI lead scores show up here as contacts are scored.'] },
  { value: 'pipelines', label: 'Pipelines', icon: 'pipeline', types: ['pipeline'],
    empty: ['No pipeline runs yet', 'Research and pitch pipelines show up here when they start and finish.'] },
  { value: 'team', label: 'Team', icon: 'users', types: ['broadcast', 'file', 'user_joined'],
    empty: ['No team updates yet', 'Announcements, shared files and new teammates show up here.'] },
];

// Thumb edge springs (same as the mobile feed's toggle): the edge moving toward the new tab leads
// (snappy), the other trails (soft), so the thumb stretches toward where it's going, then settles.
const EDGE_LEAD = { type: 'spring', damping: 20, stiffness: 340, mass: 0.6 } as const;
const EDGE_TRAIL = { type: 'spring', damping: 17, stiffness: 130, mass: 0.9 } as const;

const SLOTS = 5; // one per filter; the toggle is five equal slots
// one slot's width inside the toggle's padding box (3px padding each side)
const SLOT = `((100% - 6px) / ${SLOTS})`;

/**
 * The filter toggle, ported from the mobile feed's rail toggle:
 *  - "worm" thumb: the leading edge shoots to the new tab, the trailing edge catches up, and the
 *    thumb squashes a little while stretched;
 *  - a ring pulses out from where it lands;
 *  - the new tab's icon pops (overshoot) with a quick wiggle; tabs squish when pressed.
 * Positions are slot numbers turned into CSS calc(), so nothing is measured and the thumb is
 * always exactly on its tab. Every tab is clickable; an empty one shows "No data found".
 */
function FilterToggle({ value, onChange, newIn, counts }: {
  value: Filter; onChange: (f: Filter) => void; newIn: (f: typeof FILTERS[number]) => boolean; counts: Record<Filter, number>;
}) {
  const reduce = useReduceMotion();
  const idx = Math.max(0, FILTERS.findIndex((f) => f.value === value));
  const l = useMotionValue(idx);     // thumb's left edge, in slots
  const r = useMotionValue(idx + 1); // thumb's right edge, in slots
  const prev = useRef(idx);
  const [pulse, setPulse] = useState<{ key: number; idx: number } | null>(null);

  useEffect(() => {
    const from = prev.current;
    prev.current = idx;
    if (from === idx) return;
    if (reduce) { l.set(idx); r.set(idx + 1); return; }
    const forward = idx > from;
    animate(r, idx + 1, forward ? EDGE_LEAD : EDGE_TRAIL);
    animate(l, idx, forward ? EDGE_TRAIL : EDGE_LEAD);
    setPulse({ key: Date.now(), idx });
  }, [idx, reduce, l, r]);

  const left = useTransform(l, (v) => `calc(3px + ${v} * ${SLOT})`);
  const width = useTransform([l, r], ([a, b]: number[]) => `calc(${Math.max(b! - a!, 0)} * ${SLOT})`);
  const scaleY = useTransform([l, r], ([a, b]: number[]) => 1 - 0.12 * Math.min(Math.max(b! - a! - 1, 0), 1));

  return (
    // the one-row grid is set inline too, so the five tabs can never fall back to a stacked list
    <div role="tablist" aria-label="Filter activity" className="relative isolate grid grid-cols-5 rounded-full border bg-secondary p-[3px]"
      style={{ display: 'grid', gridTemplateColumns: `repeat(${SLOTS}, minmax(0, 1fr))`, padding: 3, position: 'relative' }}>
      {pulse ? (
        <motion.span key={pulse.key} aria-hidden className="pointer-events-none absolute bottom-[3px] top-[3px] rounded-full border-2 border-primary"
          style={{ left: `calc(3px + ${pulse.idx} * ${SLOT})`, width: `calc(${SLOT})` }}
          initial={{ opacity: 0.45, scale: 1 }} animate={{ opacity: 0, scale: 1.12 }}
          transition={{ delay: 0.09, duration: 0.5, ease: [0.33, 1, 0.68, 1] }} />
      ) : null}
      <motion.span aria-hidden className="pointer-events-none absolute bottom-[3px] top-[3px] rounded-full bg-primary shadow-card"
        style={{ left, width, scaleY }} />
      {FILTERS.map((f) => {
        const on = value === f.value;
        return (
          <motion.button key={f.value} role="tab" aria-selected={on} onClick={() => onChange(f.value)} whileTap={{ scale: 0.92 }}
            title={`${f.label} · ${counts[f.value] || 'no'} item${counts[f.value] === 1 ? '' : 's'}`}
            className={cx('relative z-[1] flex h-8 min-w-0 items-center justify-center gap-1.5 rounded-full px-1.5 text-xs font-semibold transition-colors duration-200 cursor-pointer',
              on ? 'text-primary-foreground' : 'text-muted-foreground hover:text-foreground')}>
            <motion.span className="flex shrink-0" initial={false}
              animate={on && !reduce ? { scale: [1, 1.3, 1], rotate: [0, -16, 12, 0] } : { scale: 1, rotate: 0 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}>
              <Icon name={f.icon} size={14} />
            </motion.span>
            <span className="whitespace-nowrap max-[499px]:hidden">{f.label}</span>
            {newIn(f) ? <span className={cx('absolute right-2 top-1.5 h-1.5 w-1.5 rounded-full', on ? 'bg-primary-foreground' : 'bg-primary')} aria-label="has new" /> : null}
          </motion.button>
        );
      })}
    </div>
  );
}

/** "No data found" for an empty tab: the tab's icon springs in and floats, with a dashed frame. */
function NoData({ f, onShowAll }: { f: typeof FILTERS[number]; onShowAll?: (() => void) | undefined }) {
  const reduce = useReduceMotion();
  return (
    <div className="flex flex-col items-center px-6 py-10 text-center">
      <div className="relative mb-3 flex h-16 w-16 items-center justify-center">
        <motion.span aria-hidden className="absolute inset-0 rounded-[20px] border-2 border-dashed border-primary/30"
          initial={{ scale: 0.6, opacity: 0, rotate: -20 }} animate={{ scale: 1, opacity: 1, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 260, damping: 18 }} />
        <motion.span className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-accent text-primary"
          initial={{ scale: 0.5, opacity: 0 }} animate={reduce ? { scale: 1, opacity: 1 } : { scale: 1, opacity: 1, y: [0, -3, 0] }}
          transition={{ scale: { type: 'spring', stiffness: 420, damping: 16, delay: 0.06 }, opacity: { duration: 0.2, delay: 0.06 },
            y: { duration: 2.4, repeat: Infinity, ease: 'easeInOut', delay: 0.6 } }}>
          <Icon name={f.icon} size={22} />
        </motion.span>
        <motion.span aria-hidden className="absolute -bottom-1 -right-1 flex h-6 w-6 items-center justify-center rounded-full bg-card text-muted-foreground shadow-card ring-1 ring-border"
          initial={{ scale: 0, rotate: -45 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 500, damping: 18, delay: 0.22 }}>
          <Icon name="search" size={12} />
        </motion.span>
      </div>
      <motion.p className="text-sm font-semibold text-foreground" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12, duration: 0.2 }}>
        No data found
      </motion.p>
      <motion.p className="mt-1 max-w-64 text-xs leading-[18px] text-muted-foreground" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18, duration: 0.2 }}>
        {f.empty[0]}. {f.empty[1]}
      </motion.p>
      {onShowAll ? (
        <motion.button onClick={onShowAll} whileTap={{ scale: 0.94 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.24 }}
          className="mt-3 inline-flex items-center gap-1 rounded-full border bg-card px-3 py-1.5 text-xs font-semibold text-primary transition hover:bg-accent cursor-pointer">
          <Icon name="activity" size={13} />Show all activity
        </motion.button>
      ) : null}
    </div>
  );
}

/**
 * Header "Activity ▾" menu: a compact panel with one row of sliding filter tabs and the latest
 * workspace activity grouped by day. The badge counts items newer than the last time the admin
 * opened it (kept per browser); the live stream refreshes the activity query, so new scans bump
 * the count right away.
 */
export function ActivityMenu({ status }: { status: StreamStatus }) {
  const navigate = useNavigate();
  // the same 200-item window as the Activity page (shared cache): with only the newest few, frequent
  // types (contacts added) pushed every score / pipeline / team item out and those tabs came up empty
  const feed = useQuery(q.activity(200));
  const items = feed.data || [];
  const newest = items[0] ? ts(items[0]) : 0;
  const [open, setOpen] = useState(false);
  const [seenAt, setSeenAt] = useState<number | null>(readSeen);
  // what was "seen" when the panel opened, so this visit's new items stay marked while it is open
  const [openedSeenAt, setOpenedSeenAt] = useState<number | null>(null);
  const [filter, setFilter] = useState<Filter>('all');

  // first visit in this browser: start from now instead of flagging the whole history as new
  useEffect(() => {
    if (seenAt == null && newest) { setSeenAt(newest); writeSeen(newest); }
  }, [seenAt, newest]);

  const markSeen = () => { if (newest) { setSeenAt(newest); writeSeen(newest); } };
  const unread = open || seenAt == null ? 0 : items.filter((a) => ts(a) > seenAt).length;
  const isNew = (a: ActivityItem) => openedSeenAt != null && ts(a) > openedSeenAt;

  const onOpenChange = (o: boolean) => {
    setOpen(o);
    if (o) { setOpenedSeenAt(seenAt); markSeen(); } else { markSeen(); setFilter('all'); }
  };

  const counts = useMemo(() => {
    const m = {} as Record<Filter, number>;
    FILTERS.forEach((f) => { m[f.value] = f.types.length ? items.filter((a) => f.types.includes(a.type)).length : items.length; });
    return m;
  }, [items]);
  const activeFilter = FILTERS.find((f) => f.value === filter)!;
  const groups = useMemo(() => {
    const types = FILTERS.find((f) => f.value === filter)!.types;
    const out: { day: string; items: ActivityItem[] }[] = [];
    items.filter((a) => !types.length || types.includes(a.type)).slice(0, PER_TAB).forEach((a) => {
      const day = dayLabel(a.timestamp);
      const last = out[out.length - 1];
      if (last && last.day === day) last.items.push(a);
      else out.push({ day, items: [a] });
    });
    return out;
  }, [items, filter]);
  const freshCount = open ? items.filter(isNew).length : 0;

  const close = () => { setOpen(false); markSeen(); };
  // the same targets as the Activity page (only real ones); nothing to open → that day on /activity
  const go = (a: ActivityItem) => {
    const link = activityLink(a);
    close();
    if (link) navigate({ to: link.to as never, params: link.params as never, search: (link.search ?? {}) as never });
    else navigate({ to: '/activity', search: { date: dayKey(new Date(a.timestamp)) } as never });
  };

  const live = status === 'open';

  return (
    <Popover.Root open={open} onOpenChange={onOpenChange}>
      <Popover.Trigger asChild>
        <button aria-label={unread ? `Activity, ${unread} new` : 'Activity'}
          className="relative inline-flex h-8 items-center gap-1.5 rounded-lg border bg-card px-2.5 text-xs font-medium text-body transition hover:bg-secondary data-[state=open]:bg-secondary cursor-pointer">
          <Icon name="activity" size={14} className="text-muted-foreground" />
          <span className="hidden sm:inline">Activity</span>
          <Icon name="chevronDown" size={14} className={cx('text-muted-foreground transition-transform duration-200', open && 'rotate-180')} />
          {unread ? (
            <span key={unread} className="es-pop absolute -right-2 -top-2 flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-primary px-1 text-[10px] font-bold leading-none tabular-nums text-primary-foreground ring-2 ring-background">
              {unread > 99 ? '99+' : unread}
            </span>
          ) : null}
        </button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content align="end" sideOffset={6} collisionPadding={12}
          className="es-fade z-50 flex max-h-[min(540px,calc(100vh-64px))] w-[480px] max-w-[calc(100vw-24px)] flex-col overflow-hidden rounded-xl border bg-popover text-popover-foreground shadow-xl">

          {/* Header: one line */}
          <div className="flex shrink-0 items-center gap-2 px-3 pt-2.5">
            <h2 className="text-sm font-semibold text-foreground">Activity</h2>
            <span title={live ? 'Connected to the live feed' : status === 'offline' ? 'Live feed offline, reconnecting' : 'Connecting to the live feed'}
              className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <span className="relative flex h-1.5 w-1.5">
                {live ? <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-live opacity-60" /> : null}
                <span className={cx('relative inline-flex h-1.5 w-1.5 rounded-full', live ? 'bg-live' : status === 'offline' ? 'bg-heat-hot' : 'bg-heat-warm')} />
              </span>
              {live ? 'Live' : status === 'offline' ? 'Offline' : 'Connecting'}
            </span>
            {freshCount ? <span className="rounded-full bg-accent px-1.5 py-px text-[10px] font-semibold text-accent-foreground">{freshCount} new</span> : null}
            <div className="ml-auto flex items-center">
              <button onClick={() => feed.refetch()} aria-label="Refresh activity" title="Refresh"
                className={cx('flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition hover:bg-secondary hover:text-foreground cursor-pointer', feed.isFetching && '[&_svg]:animate-spin')}>
                <Icon name="refresh" size={13} />
              </button>
              <button onClick={() => { close(); navigate({ to: '/activity' }); }}
                className="inline-flex h-7 items-center gap-0.5 rounded-md pl-2 pr-1 text-xs font-medium text-primary transition hover:bg-accent cursor-pointer">
                View all<Icon name="chevronRight" size={13} />
              </button>
            </div>
          </div>

          {/* Filter toggle (mobile feed style): every tab active; an empty one shows "No data found". */}
          <div className="mx-3 mb-2.5 mt-2.5 shrink-0">
            <FilterToggle value={filter} onChange={setFilter} counts={counts}
              newIn={(f) => open && items.some((a) => isNew(a) && (!f.types.length || f.types.includes(a.type)))} />
          </div>

          {/* List */}
          <div className="es-scroll min-h-0 flex-1 overflow-y-auto border-t">
            {feed.isLoading ? (
              <div className="space-y-1 p-1.5">{[0, 1, 2, 3, 4].map((i) => <div key={i} className="h-12 animate-pulse rounded-lg bg-secondary" />)}</div>
            ) : feed.error ? (
              <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                Couldn't load activity. <button onClick={() => feed.refetch()} className="font-medium text-primary cursor-pointer">Try again</button>
              </div>
            ) : (
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={filter} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.16, ease: 'easeOut' }}>
                  {!groups.length ? (
                    <NoData f={activeFilter} onShowAll={filter !== 'all' ? () => setFilter('all') : undefined} />
                  ) : groups.map((g) => (
                    <section key={g.day}>
                      <h3 className="sticky top-0 z-[1] bg-popover/95 px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground backdrop-blur-sm">{g.day}</h3>
                      <ul className="px-1.5 pb-1">
                        {g.items.map((a) => {
                          const s = ACTIVITY_STYLE[a.type] || ACTIVITY_STYLE.contact_added;
                          const fresh = isNew(a);
                          const score = a.type === 'contact_scored' && typeof a.meta?.score === 'number' ? a.meta.score : null;
                          const b = score != null ? band(score) : null;
                          return (
                            <li key={a.id}>
                              <button onClick={() => go(a)}
                                className={cx('group flex w-full items-start gap-2.5 rounded-lg px-1.5 py-2 text-left transition-colors cursor-pointer hover:bg-secondary',
                                  fresh && 'bg-accent/60 hover:bg-accent')}>
                                <span className={cx('flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', s.tone)}><Icon name={s.icon} size={15} /></span>
                                {/* full text, wrapped: nothing is cut off */}
                                <div className="min-w-0 flex-1">
                                  <p className="break-words text-[13px] font-medium leading-5 text-foreground">{a.title}</p>
                                  {a.subtitle ? <p className="mt-0.5 whitespace-pre-line break-words text-xs leading-[18px] text-muted-foreground">{a.subtitle}</p> : null}
                                </div>
                                <div className="flex shrink-0 flex-col items-end gap-1 pt-0.5">
                                  {b ? (
                                    <span className="inline-flex items-center gap-1 rounded-md px-1.5 py-px text-[11px] font-semibold tabular-nums"
                                      style={{ color: ink(b.color), background: tint(b.color, 14) }}>{score}<span className="text-[9.5px] font-medium">{b.label}</span></span>
                                  ) : null}
                                  <span className="flex items-center gap-1 text-[10.5px] tabular-nums text-muted-foreground" title={fmtDateTime(a.timestamp)}>
                                    {fresh ? <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-label="New" /> : null}
                                    {relTime(a.timestamp)}
                                  </span>
                                </div>
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    </section>
                  ))}
                </motion.div>
              </AnimatePresence>
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
