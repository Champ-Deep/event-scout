// The Activity page's date picker: a pill at the right end of the filter row that opens a month grid.
// Every day is shaded by how much happened on it (GET /api/admin/activity/calendar, bucketed by the
// admin's own local day) with dots for the kinds of activity; picking a day shows just that day.
import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, type UseQueryOptions } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { q } from '../../lib/queries';
import { addDays, dayKey, fromDayKey, gridStart } from '../../lib/dates';
import type { ActivityCalendar as DayCounts } from '../../lib/types';
import { Icon } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { cx } from '../../ui/kit';
import { ACTIVITY_STYLE } from './ActivityFeed';

const WEEKDAYS = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'];

/** What the dots stand for: key → colour and label (Activity: the activity types). */
export type CalendarKinds = Record<string, { color: string; label: string }>;
const ACTIVITY_KINDS: CalendarKinds = Object.fromEntries(Object.entries(ACTIVITY_STYLE).map(([k, s]) => [k, { color: s.color, label: s.label }]));

/**
 * A day picker whose days are shaded by how much happened. Defaults to the Activity page's counts;
 * other pages pass `counts` (per-day totals for the 42-day grid starting at a YYYY-MM-DD key),
 * `kinds` (what the dots mean) and `noun` (what's being counted).
 */
export function ActivityCalendar({ value, onChange, counts, kinds = ACTIVITY_KINDS, noun = 'event', align = 'end' }: {
  value: string | undefined; onChange: (day: string | undefined) => void;
  // (any query key: each page's queryOptions carries its own literal key tuple)
  counts?: (gridStartKey: string) => UseQueryOptions<DayCounts, Error, DayCounts, any>;
  kinds?: CalendarKinds; noun?: string; align?: 'start' | 'end';
}) {
  const [open, setOpen] = useState(false);
  const today = dayKey(new Date());
  // the month on screen: the picked day's month, else this month
  const [month, setMonth] = useState(() => { const d = value ? fromDayKey(value) : new Date(); return new Date(d.getFullYear(), d.getMonth(), 1); });
  useEffect(() => { if (value) { const d = fromDayKey(value); setMonth(new Date(d.getFullYear(), d.getMonth(), 1)); } }, [value]);

  const first = gridStart(month);
  const firstKey = dayKey(first);
  const cal = useQuery({ ...(counts ? counts(firstKey) : (q.activityCalendar(firstKey) as unknown as UseQueryOptions<DayCounts, Error, DayCounts, readonly unknown[]>)), enabled: open });
  const days = cal.data?.days ?? {};
  const cells = useMemo(() => Array.from({ length: 42 }, (_, i) => addDays(first, i)), [firstKey]);
  const inMonth = (d: Date) => d.getMonth() === month.getMonth();
  const max = Math.max(1, ...cells.filter(inMonth).map((d) => days[dayKey(d)]?.total ?? 0));
  const monthTotal = cells.filter(inMonth).reduce((n, d) => n + (days[dayKey(d)]?.total ?? 0), 0);
  const isThisMonth = month.getFullYear() === new Date().getFullYear() && month.getMonth() === new Date().getMonth();
  const step = (n: number) => setMonth((m) => new Date(m.getFullYear(), m.getMonth() + n, 1));

  const picked = value ? fromDayKey(value) : null;
  const label = picked
    ? (value === today ? 'Today' : value === dayKey(addDays(new Date(), -1)) ? 'Yesterday' : picked.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short', ...(picked.getFullYear() !== new Date().getFullYear() ? { year: 'numeric' } : {}) }))
    : 'All dates';

  return (
    <span className={cx('inline-flex shrink-0 items-stretch rounded-full', value && 'shadow-active')}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button aria-label={value ? `Showing ${label}. Pick another day` : 'Pick a day to see its activity'}
            className={cx('inline-flex h-9 cursor-pointer items-center gap-2 rounded-full border-[1.5px] pl-2 pr-3 text-[13px] font-bold transition-[background-color,border-color,transform] duration-200 active:scale-[0.97]',
              value ? 'rounded-r-none border-r-0 border-brand-700 bg-brand-600 text-white hover:bg-brand-700'
                : 'border-border bg-card text-foreground hover:border-brand-200 hover:bg-brand-50/60 data-[state=open]:border-brand-300 data-[state=open]:bg-brand-50')}>
            <span className={cx('flex size-6 items-center justify-center rounded-full', value ? 'bg-white/20' : 'bg-brand-50 text-brand-700')}>
              <Icon name="calendar" size={13} />
            </span>
            {label}
            {!value ? <Icon name="chevronDown" size={13} className="-mr-0.5 text-muted-foreground transition-transform duration-200 [[data-state=open]_&]:rotate-180" /> : null}
          </button>
        </PopoverTrigger>
        <PopoverContent align={align} sideOffset={8} className="w-[320px] overflow-hidden rounded-2xl p-0 shadow-xl">
          {/* month · arrows */}
          <div className="flex items-center justify-between border-b px-3 py-2.5">
            <button onClick={() => step(-1)} aria-label="Previous month" className="flex size-8 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground">
              <Icon name="chevronRight" size={15} className="rotate-180" />
            </button>
            <div className="text-center">
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={firstKey} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.fast }}
                  className="text-sm font-extrabold text-foreground">{month.toLocaleDateString([], { month: 'long', year: 'numeric' })}</motion.div>
              </AnimatePresence>
              <div className="text-[11px] font-semibold text-muted-foreground" aria-live="polite">
                {cal.isFetching && !cal.data ? 'Loading…' : cal.isError ? "Couldn't load" : `${monthTotal.toLocaleString()} ${noun}${monthTotal === 1 ? '' : 's'} this month`}
              </div>
            </div>
            <button onClick={() => step(1)} disabled={isThisMonth} aria-label="Next month" className="flex size-8 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:cursor-default disabled:opacity-30 disabled:hover:bg-transparent">
              <Icon name="chevronRight" size={15} />
            </button>
          </div>

          <div className="p-3">
            <div className="mb-1 grid grid-cols-7 text-center text-[10px] font-bold uppercase tracking-[0.06em] text-muted-foreground">
              {WEEKDAYS.map((w) => <span key={w} className="py-1">{w}</span>)}
            </div>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={firstKey} className={cx('grid grid-cols-7 gap-1 transition-opacity', cal.isFetching && !cal.data && 'opacity-50')}
                initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: MOTION.fast, ease: MOTION.ease }}>
                {cells.map((d, i) => {
                  const key = dayKey(d);
                  const info = days[key];
                  const n = info?.total ?? 0;
                  const future = key > today;
                  const sel = key === value;
                  const types = info ? (Object.entries(info.types) as [string, number][]).sort((a, b) => b[1] - a[1]).slice(0, 3) : [];
                  // heat: a brand tint that deepens with how busy the day was
                  const heat = n ? 10 + Math.round((n / max) * 40) : 0;
                  return (
                    <motion.button key={key} disabled={future} onClick={() => { onChange(sel ? undefined : key); setOpen(false); }}
                      initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: MOTION.fast, delay: i * 0.006 }}
                      whileHover={future ? {} : { scale: 1.08 }} whileTap={future ? {} : { scale: 0.92 }}
                      title={future ? undefined : n ? `${d.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' })}: ${n} ${noun}${n === 1 ? '' : 's'}\n${types.map(([t, c]) => `${kinds[t]?.label ?? t}: ${c}`).join('\n')}` : 'Nothing happened'}
                      aria-label={`${d.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' })}, ${n} events`} aria-pressed={sel}
                      className={cx('relative flex h-10 cursor-pointer flex-col items-center justify-center rounded-xl text-[12.5px] font-bold tabular-nums transition-colors disabled:cursor-default',
                        sel ? 'bg-brand-600 text-white shadow-active' : inMonth(d) ? (n ? 'text-foreground' : 'text-body') : 'text-muted-foreground/45',
                        future && 'opacity-35', !sel && key === today && 'ring-[1.5px] ring-brand-400')}
                      style={!sel && n && inMonth(d) ? { background: `color-mix(in oklab, var(--brand-500) ${heat}%, transparent)` } : {}}>
                      {d.getDate()}
                      {/* up to three dots: the day's most common kinds of activity */}
                      <span className="mt-0.5 flex h-1 gap-[2px]">
                        {types.map(([t]) => <span key={t} className="size-1 rounded-full" style={{ background: sel ? 'white' : kinds[t]?.color ?? 'var(--muted-foreground)' }} />)}
                      </span>
                    </motion.button>
                  );
                })}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* jump back · legend */}
          <div className="flex items-center justify-between border-t bg-secondary/40 px-3 py-2">
            <div className="flex gap-1.5">
              <button onClick={() => { onChange(today); setOpen(false); }} className="cursor-pointer rounded-full bg-card px-2.5 py-1 text-xs font-bold text-foreground ring-1 ring-border transition-colors hover:ring-brand-300">Today</button>
              {value ? <button onClick={() => { onChange(undefined); setOpen(false); }} className="cursor-pointer rounded-full px-2.5 py-1 text-xs font-bold text-brand-700 transition-colors hover:bg-brand-50">All dates</button> : null}
            </div>
            <span className="flex items-center gap-1 text-[10px] font-semibold text-muted-foreground">
              Less
              {[0, 15, 30, 50].map((h) => <span key={h} className="size-2.5 rounded-[3px] ring-1 ring-border" style={{ background: h ? `color-mix(in oklab, var(--brand-500) ${h}%, transparent)` : 'transparent' }} />)}
              More
            </span>
          </div>
        </PopoverContent>
      </Popover>
      {value ? (
        <button onClick={() => onChange(undefined)} aria-label="Show every date"
          className="flex cursor-pointer items-center rounded-r-full border-[1.5px] border-l-0 border-brand-700 bg-brand-600 pl-0.5 pr-1.5 text-white transition-colors hover:bg-brand-700">
          <span className="flex size-5 items-center justify-center rounded-full bg-white/20 transition-colors hover:bg-white/35"><Icon name="x" size={11} strokeWidth={3} /></span>
        </button>
      ) : null}
    </span>
  );
}
