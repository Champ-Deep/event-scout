import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { q } from '../../lib/queries';
import type { LiveEventMessage } from '../../lib/sse';
import { Icon, type IconName } from '../../ui/icons';
import { AnimatedNumber } from '../../ui/micro';
import { cx, relTime } from '../../ui/kit';

type Tile = {
  key: string; icon: IconName; label: string; value: number | null | undefined; of?: number | undefined;
  /** theme-aware tint for the icon chip (light in light mode, dark in dark mode) */
  tint: string; to: string; search?: Record<string, string>; hint: string; show: string;
};

/**
 * "Today at a glance" in the top bar, the admin version of the mobile feed's stat tiles: a tinted
 * icon, the number and a short label, each opening the matching list. Numbers are today's (the
 * admin's local day, from /api/admin/dashboard?days=1, refreshed by the live stream). Zero reads as a
 * dimmed 0, never as a sentence, so a quiet morning is calm instead of repetitive.
 */
export function TodayGlance({ events, describe }: { events: LiveEventMessage[]; describe: (e: LiveEventMessage) => string }) {
  const today = useQuery(q.dashboard(1));
  const d = today.data;
  const r = d?.range;
  const met = r ? (r.series.at(-1)?.added ?? r.totals.added) : undefined;

  // the group says "Today", so each segment only needs a short word after its number
  const tiles: Tile[] = [
    { key: 'met', icon: 'userPlus', label: 'met', value: met, tint: 'text-primary',
      to: '/contacts', search: { since: 'today' }, hint: 'Contacts added today. Opens them.', show: 'flex' },
    { key: 'hot', icon: 'flame', label: 'hot', value: r?.totals.hot, tint: 'text-heat-hot',
      to: '/contacts', search: { since: 'today', band: 'A' }, hint: 'Contacts added today that scored 85 or more. Opens them.', show: 'flex' },
    { key: 'score', icon: 'sparkles', label: 'to score', value: d?.levels.unscored, tint: 'text-warning',
      to: '/contacts', search: { band: 'U' }, hint: 'Contacts still waiting for an AI score (all time). Opens them.', show: 'hidden lg:flex' },
    { key: 'reps', icon: 'users', label: 'reps', value: d?.activeReps ?? r?.activeRepsToday, of: d?.totalReps ?? d?.reps, tint: 'text-info',
      to: '/seats', hint: 'Reps who added at least one contact today, out of all reps. Opens the team.', show: 'hidden xl:flex' },
  ];
  const latest = events[0];
  const quiet = !!r && !met && !r.activeRepsToday;

  return (
    <div className="hidden min-w-0 flex-1 items-center gap-3 md:flex">
      {/* one segmented pill: TODAY · 0 met · 0 hot · 30 to score · 0 of 79 reps */}
      <div role="group" aria-label="Today at a glance"
        className="flex h-8 shrink-0 items-stretch divide-x overflow-hidden rounded-lg border bg-card shadow-card">
        <span className="es-eyebrow flex items-center gap-1.5 bg-secondary/60 px-2.5 text-[10px] text-body">
          <span className="relative flex h-1.5 w-1.5">
            {!quiet && r ? <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-live opacity-60" /> : null}
            <span className={cx('relative inline-flex h-1.5 w-1.5 rounded-full', quiet || !r ? 'bg-muted-foreground/50' : 'bg-live')} />
          </span>
          Today
        </span>
        {tiles.map((t) => {
          const zero = t.value === 0;
          return (
            <Link key={t.key} to={t.to} search={t.search as never} title={t.hint}
              className={cx(t.show, 'group items-center gap-1.5 whitespace-nowrap px-2.5 text-xs transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring')}>
              <Icon name={t.icon} size={13} className={cx('shrink-0 transition-opacity', t.tint, zero && 'opacity-50')} />
              <span className={cx('es-num text-[13px] font-semibold', zero ? 'text-muted-foreground' : 'text-foreground')}>
                {t.value == null ? '—' : <AnimatedNumber value={t.value} duration={500} />}
              </span>
              <span className="text-muted-foreground group-hover:text-body">
                {t.of != null && t.value != null ? <>of <span className="tabular-nums">{t.of}</span> </> : null}{t.label}
              </span>
            </Link>
          );
        })}
      </div>

      {/* The newest live update (scan, score, merge…), or a calm note before the first scan. */}
      <span className="hidden min-w-0 items-center gap-1.5 text-xs text-muted-foreground 2xl:flex">
        {latest ? (
          <>
            <Icon name="bolt" size={12} className="shrink-0 text-primary" />
            <span className="truncate"><span className="text-body">{describe(latest)}</span> · {relTime(latest.timestamp)}</span>
          </>
        ) : quiet ? <span className="truncate">Waiting for the first scan of the day</span> : null}
      </span>
    </div>
  );
}
