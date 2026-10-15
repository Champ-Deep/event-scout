// /activity — the workspace timeline, all from the API:
//   latest     GET /api/admin/activity?limit=200                          (q.activity(200), shared with the header menu)
//   one day    GET /api/admin/activity?limit=500&start=…&end=…            (q.activityDay(date), local midnight → midnight)
//   calendar   GET /api/admin/activity/calendar?start=…&end=…&tzOffset=…  (q.activityCalendar, per-day totals)
// The type filter (?type=) and the picked day (?date=YYYY-MM-DD) live in the URL; the live stream
// refreshes every ['admin','activity',…] query when something happens.
import React, { useMemo } from 'react';
import { useSuspenseQuery } from '@tanstack/react-query';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { q as queries } from '../lib/queries';
import { clean } from '../lib/search';
import { addDays, dayKey, fromDayKey } from '../lib/dates';
import type { ActivityItem, ActivityType } from '../lib/types';
import { ActivityFeed, ACTIVITY_STYLE, dayLabel } from '../features/activity/ActivityFeed';
import { ActivityCalendar } from '../features/activity/ActivityCalendar';
import { Icon } from '../ui/icons';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import { Button, Card, EmptyState, PageHeader, cx, deep, ink } from '../ui/kit';

const TYPES = Object.keys(ACTIVITY_STYLE) as ActivityType[];

export function ActivityComponent() {
  const search = useSearch({ from: '/_auth/activity' });
  const type: ActivityType | 'all' = search.type ?? 'all';
  const date = search.date;
  const navigate = useNavigate();
  const setSearch = (patch: { type?: ActivityType | undefined; date?: string | undefined }) =>
    navigate({ to: '/activity', search: (s) => clean({ ...s, ...patch }) as never, replace: true });
  const setType = (t: ActivityType | 'all') => setSearch({ type: t === 'all' ? undefined : t });
  const setDate = (d: string | undefined) => setSearch({ date: d });

  // the loader filled whichever one the URL asks for
  // (both return ActivityItem[]; the cast only unifies the two query-key tuple types)
  const feed = useSuspenseQuery((date ? queries.activityDay(date) : queries.activity(200)) as ReturnType<typeof queries.activity>);
  const all: ActivityItem[] = feed.data ?? [];
  const items = useMemo(() => all.filter((a) => type === 'all' || a.type === type), [all, type]);
  const counts = useMemo(() => {
    const m: Partial<Record<ActivityType, number>> = {};
    all.forEach((a) => { m[a.type] = (m[a.type] ?? 0) + 1; });
    return m;
  }, [all]);
  const groups = useMemo(() => {
    const out: { key: string; day: string; items: ActivityItem[] }[] = [];
    items.forEach((a) => {
      const key = dayKey(new Date(a.timestamp));
      const last = out[out.length - 1];
      if (last && last.key === key) last.items.push(a);
      else out.push({ key, day: dayLabel(a.timestamp), items: [a] });
    });
    return out;
  }, [items]);

  const today = dayKey(new Date());
  const picked = date ? fromDayKey(date) : null;
  const pickedLabel = picked ? picked.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }) : '';

  return (
    <>
      <PageHeader title="Activity"
        description="Everything that happened in the workspace — contacts added, AI scores, pipelines, merges, announcements, files and new accounts. Pick a day on the calendar to see what happened then."
        actions={<Button icon="refresh" onClick={() => feed.refetch()} loading={feed.isFetching && !feed.isLoading}>Refresh</Button>} />

      {/* filters ······ calendar (right end) */}
      <div className="mb-4 flex items-center gap-2">
        <div role="tablist" aria-label="Kind of activity" className="no-scrollbar -mx-1 flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto px-1 py-1">
          {(['all', ...TYPES] as (ActivityType | 'all')[]).map((t, i) => {
            const on = type === t;
            const n = t === 'all' ? all.length : counts[t] ?? 0;
            if (t !== 'all' && !n && !on) return null;
            const s = t === 'all' ? null : ACTIVITY_STYLE[t];
            const color = s?.color ?? 'var(--brand-600)';
            return (
              <motion.button key={t} role="tab" aria-selected={on} onClick={() => setType(t)}
                initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: i * 0.03 }}
                whileHover={{ y: -1 }} whileTap={{ scale: 0.96 }}
                className={cx('relative inline-flex h-9 shrink-0 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] pl-1 pr-1.5 text-[13px] font-bold transition-colors',
                  on ? 'text-foreground shadow-card' : 'border-border bg-card text-foreground hover:border-brand-200')}
                style={on ? { borderColor: color, background: `color-mix(in oklab, ${color} 10%, var(--card))` } : {}}>
                <span className="flex size-6 items-center justify-center rounded-full" style={{ background: on ? deep(color) : `color-mix(in oklab, ${color} 14%, transparent)`, color: on ? 'white' : ink(color) }}>
                  <Icon name={s?.icon ?? 'activity'} size={13} />
                </span>
                {t === 'all' ? 'Everything' : s!.label}
                <span className="es-num min-w-6 rounded-full px-1.5 text-center text-[12px] font-black leading-[22px]"
                  style={on ? { background: deep(color), color: 'white' } : { background: 'var(--secondary)', color: 'var(--body)' }}>
                  <AnimatedNumber value={n} duration={500} />
                </span>
              </motion.button>
            );
          })}
        </div>
        <ActivityCalendar value={date} onChange={setDate} />
      </div>

      {/* the picked day: what happened, by kind (tap one to filter) */}
      <AnimatePresence initial={false}>
        {picked ? (
          <motion.div key={date} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}
            className="mb-4 overflow-hidden rounded-2xl border bg-card shadow-card">
            <div className="flex flex-wrap items-center gap-3 border-b px-4 py-3" style={{ background: 'color-mix(in oklab, var(--brand-500) 6%, var(--card))' }}>
              <span className="flex size-11 flex-col items-center justify-center rounded-xl bg-brand-600 text-white shadow-active">
                <span className="text-[9px] font-bold uppercase leading-none tracking-[0.08em] opacity-80">{picked.toLocaleDateString([], { month: 'short' })}</span>
                <span className="text-[17px] font-black leading-5">{picked.getDate()}</span>
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-[15px] font-extrabold text-foreground">{date === today ? 'Today' : pickedLabel}</div>
                <div className="text-xs font-semibold text-muted-foreground">
                  {all.length ? `${all.length} event${all.length === 1 ? '' : 's'}${all.length >= 500 ? ' (first 500)' : ''}` : 'Nothing happened'}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <button onClick={() => setDate(dayKey(addDays(picked, -1)))} aria-label="Previous day" title="Previous day"
                  className="flex size-8 cursor-pointer items-center justify-center rounded-full border bg-card text-muted-foreground transition-colors hover:border-brand-200 hover:text-foreground">
                  <Icon name="chevronRight" size={15} className="rotate-180" />
                </button>
                <button onClick={() => setDate(dayKey(addDays(picked, 1)))} disabled={date! >= today} aria-label="Next day" title="Next day"
                  className="flex size-8 cursor-pointer items-center justify-center rounded-full border bg-card text-muted-foreground transition-colors hover:border-brand-200 hover:text-foreground disabled:cursor-default disabled:opacity-35">
                  <Icon name="chevronRight" size={15} />
                </button>
                <Button size="sm" variant="ghost" icon="x" onClick={() => setDate(undefined)} className="ml-1 rounded-full">All dates</Button>
              </div>
            </div>
            {all.length ? (
              <div className="grid grid-cols-2 gap-2 p-3 sm:grid-cols-4 xl:grid-cols-8">
                {TYPES.map((t, i) => {
                  const s = ACTIVITY_STYLE[t];
                  const n = counts[t] ?? 0;
                  return (
                    <motion.button key={t} disabled={!n} onClick={() => setType(type === t ? 'all' : t)}
                      initial={{ opacity: 0, y: 8, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.05 + i * 0.03 }}
                      whileHover={n ? { y: -2 } : {}} whileTap={n ? { scale: 0.96 } : {}}
                      className={cx('flex items-center gap-2.5 rounded-xl border p-2.5 text-left transition-colors disabled:cursor-default disabled:opacity-45',
                        n && 'cursor-pointer hover:border-brand-200', type === t && 'border-brand-300 bg-brand-50')}>
                      <span className={cx('flex size-8 shrink-0 items-center justify-center rounded-lg', s.tone)}><Icon name={s.icon} size={15} /></span>
                      <span className="min-w-0">
                        <span className="es-num block text-[17px] font-black leading-5 text-foreground"><AnimatedNumber value={n} duration={600} /></span>
                        <span className="block truncate text-[11px] font-semibold text-muted-foreground">{s.label}</span>
                      </span>
                    </motion.button>
                  );
                })}
              </div>
            ) : null}
          </motion.div>
        ) : null}
      </AnimatePresence>

      <Card className="rounded-2xl p-4 sm:p-5">
        {groups.length ? (
          <div className="space-y-6">
            {!date ? (
              <p className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
                <Icon name="info" size={13} />Showing the latest {all.length} events. Pick a day on the calendar to see older activity.
              </p>
            ) : null}
            {groups.map((g, gi) => {
              const before = groups.slice(0, gi).reduce((n, x) => n + x.items.length, 0);
              return (
                <section key={g.key}>
                  {/* day header: stays on screen while its items scroll */}
                  <div className="sticky top-12 z-10 -mx-1 mb-3 flex items-center gap-2 bg-card/95 px-1 py-1.5 backdrop-blur supports-[not(backdrop-filter:blur(1px))]:bg-card">
                    <h2 className="text-[13px] font-extrabold text-foreground">{g.day}</h2>
                    {g.day === 'Today' || g.day === 'Yesterday' ? <span className="text-xs font-medium text-muted-foreground">{fromDayKey(g.key).toLocaleDateString([], { day: 'numeric', month: 'long' })}</span> : null}
                    <span className="h-px flex-1 bg-border" aria-hidden />
                    <span className="es-num rounded-full bg-secondary px-2 text-[11px] font-black leading-5 text-body">{g.items.length}</span>
                    {!date ? (
                      <button onClick={() => setDate(g.key)} className="cursor-pointer rounded-full px-2 text-[11px] font-bold leading-5 text-brand-700 transition-colors hover:bg-brand-50" title="See only this day">
                        Open day
                      </button>
                    ) : null}
                  </div>
                  <ActivityFeed items={g.items} startIndex={before} />
                </section>
              );
            })}
          </div>
        ) : (
          <EmptyState icon={date ? 'calendar' : 'activity'}
            title={date ? (type === 'all' ? `Nothing happened on ${date === today ? 'today' : pickedLabel}` : `No ${ACTIVITY_STYLE[type as ActivityType].label.toLowerCase()} on this day`) : type === 'all' ? 'Nothing here yet' : `No ${ACTIVITY_STYLE[type as ActivityType].label.toLowerCase()} yet`}
            description={date ? 'Pick another day on the calendar, or go back to the latest activity.' : 'Activity appears as your team scans and works their leads.'}
            action={date ? <Button icon="refresh" onClick={() => setDate(undefined)}>Back to latest</Button> : type !== 'all' ? <Button onClick={() => setType('all')}>Show everything</Button> : undefined} />
        )}
      </Card>
    </>
  );
}
