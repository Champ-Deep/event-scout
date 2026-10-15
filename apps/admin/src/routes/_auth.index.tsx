import React, { useMemo, useState } from 'react';
import { useQuery, useSuspenseQuery } from '@tanstack/react-query';
import { Link, useNavigate, useSearch } from '@tanstack/react-router';
import { q } from '../lib/queries';
import { getSession } from '../lib/auth';
import { Icon, type IconName } from '../ui/icons';
import { LiveIcon } from '../ui/live-icon';
import { AnimatePresence, motion } from 'motion/react';
import { BarList, Columns, Donut, Ring, Sparkline } from '../ui/charts';
import type { ActivityItem, Audience, AudienceRow, Dashboard } from '../lib/types';
import { AnimatedNumber, Delta, Reveal, SlidingTabs, SwapLink } from '../ui/micro';
import { Card, EmptyState, ErrorState, IconButton, cx, ink, relTime } from '../ui/kit';
import { levelOf as levelInfo } from '../lib/levels';

type Range = '7' | '14' | '30' | '90';
const RANGES: { value: Range; label: string }[] = [{ value: '7', label: '7d' }, { value: '14', label: '14d' }, { value: '30', label: '30d' }, { value: '90', label: '90d' }];

// Lead heat has its own ramp (tokens.css --heat-*), separate from the violet brand accent.
const LEVEL_COLORS = { hot: 'var(--heat-hot)', warm: 'var(--heat-warm)', cool: 'var(--heat-cool)', cold: 'var(--heat-cold)', unscored: 'var(--heat-none)' } as const;
const tint = (c: string, pct: number) => `color-mix(in oklab, ${c} ${pct}%, transparent)`;
const LEVEL_ROWS: { key: keyof typeof LEVEL_COLORS; label: string; band: string; range: string }[] = [
  { key: 'hot', label: 'Hot', band: 'A', range: '85–100' }, { key: 'warm', label: 'Warm', band: 'B', range: '70–84' },
  { key: 'cool', label: 'Cool', band: 'C', range: '50–69' }, { key: 'cold', label: 'Cold', band: 'D', range: '0–49' },
  { key: 'unscored', label: 'Not scored', band: 'U', range: '—' },
];
const SOURCE_LABEL: Record<string, string> = { scan: 'Card scan', qr: 'QR code', manual: 'Added by hand', import: 'Import', linkedin: 'LinkedIn', sync: 'Phone (offline)', unknown: 'Other' };
// the lead levels' rule and colours (lib/levels): hot ≥ 85, warm 70–84, cool 50–69, cold < 50
const scoreColor = (s: number) => levelInfo(s)!.color;
/** A histogram bucket's fill; the server buckets by 10, so 80–89 straddles warm / hot and is split
 *  left → right in proportion (80–84 warm, 85–89 hot) instead of reading as all one level. */
const bucketColor = (from: number, to: number) => {
  const hi = Math.max(from, to);
  const span = hi - from + 1;
  const parts: { color: string; n: number }[] = [];
  for (let s = from; s <= hi; s++) {
    const c = scoreColor(s);
    const last = parts[parts.length - 1];
    if (last && last.color === c) last.n += 1; else parts.push({ color: c, n: 1 });
  }
  if (parts.length === 1) return parts[0]!.color;
  let at = 0;
  const stops = parts.map((p) => { const a = (at / span) * 100; at += p.n; return `${p.color} ${a}% ${(at / span) * 100}%`; });
  return `linear-gradient(to right, ${stops.join(', ')})`;
};

type AudienceDim = 'industry' | 'function' | 'seniority' | 'company';
const AUDIENCE_TABS: { value: AudienceDim; label: string }[] = [
  { value: 'industry', label: 'Industry' }, { value: 'function', label: 'Role' }, { value: 'seniority', label: 'Seniority' }, { value: 'company', label: 'Companies' },
];

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? 'Working late' : h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
}
const longDay = (iso: string) => new Date(`${iso}T00:00:00`).toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'short' });

/** Loading placeholder with a light sweep (tokens.css .es-shimmer) instead of a flat pulse. */
function Shimmer({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return <span className={cx('es-shimmer block rounded-md', className)} style={style} aria-hidden />;
}

/** Text that arrives word by word out of a soft blur (MicroKit "blur glide"). */
function GlideWords({ text, delay = 0 }: { text: string; delay?: number }) {
  return (
    <span aria-label={text}>
      {text.split(' ').map((w, i) => (
        <motion.span key={`${w}-${i}`} aria-hidden className="inline-block whitespace-pre"
          initial={{ opacity: 0, y: 10, filter: 'blur(6px)' }} animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.45, ease: [0.2, 0.8, 0.2, 1], delay: delay + i * 0.07 }}>
          {w}{i < text.split(' ').length - 1 ? ' ' : ''}
        </motion.span>
      ))}
    </span>
  );
}

/** Feeds the cursor position to an .es-spot element so its highlight follows the pointer. */
const spotMove = (e: React.MouseEvent<HTMLElement>) => {
  const r = e.currentTarget.getBoundingClientRect();
  e.currentTarget.style.setProperty('--es-x', `${e.clientX - r.left}px`);
  e.currentTarget.style.setProperty('--es-y', `${e.clientY - r.top}px`);
};

const isToday = (iso: string) => new Date(iso).toDateString() === new Date().toDateString();
const initials = (name: string) => name.split(/\s+/).filter(Boolean).map((p) => p[0]).join('').slice(0, 2).toUpperCase() || '?';
const hourLabel = (h: number) => `${h % 12 || 12}${h < 12 ? 'a' : 'p'}`;

/**
 * "On the floor now": who is capturing leads today and when. Left, a live headline with the faces of
 * today's active reps and the latest update; middle, today's captures hour by hour (now marked);
 * right, four compact stats that open the matching list. The hours and reps come from the dashboard
 * endpoint (todayByHour / repsToday: contacts created today, the admin's local day, counted on the
 * server); an older server without them falls back to the activity feed.
 */
function FloorNow({ loading, activity, todayByHour, todayReps, repsToday, repsTotal, addedToday, hotToday, totalContacts, onOpen }: {
  loading: boolean; activity: ActivityItem[]; todayByHour: number[] | undefined; todayReps: Dashboard['repsToday'];
  repsToday: number; repsTotal: number | undefined; addedToday: number | undefined;
  hotToday: number | undefined; totalContacts: number | undefined; onOpen: (to: 'added' | 'hot' | 'reps' | 'all') => void;
}) {
  const nowHour = new Date().getHours();
  const { hours, reps } = useMemo(() => {
    if (todayByHour?.length === 24 && todayReps) {
      return { hours: todayByHour, reps: todayReps.map((r) => ({ name: r.name, n: r.count, photo: r.photoUrl })) };
    }
    const h = Array.from({ length: 24 }, () => 0);
    const who = new Map<string, number>();
    activity.forEach((a) => {
      if (a.type !== 'contact_added' || !isToday(a.timestamp)) return;
      h[new Date(a.timestamp).getHours()]! += 1;
      who.set(a.actor, (who.get(a.actor) ?? 0) + 1); // newest first, so the order is "most recently active"
    });
    return { hours: h, reps: [...who.entries()].map(([name, n]) => ({ name, n, photo: null as string | null })) };
  }, [activity, todayByHour, todayReps]);
  const peak = Math.max(1, ...hours);
  const any = hours.some(Boolean);
  const latest = activity[0];
  const live = repsToday > 0;
  const accent = live ? 'var(--live)' : 'var(--heat-warm)';
  const stats: { key: 'added' | 'hot' | 'reps' | 'all'; label: string; icon: IconName; color: string; value: number | undefined; of?: number | undefined }[] = [
    { key: 'added', label: 'Added today', icon: 'userPlus', color: 'var(--primary)', value: addedToday },
    { key: 'hot', label: 'Hot today', icon: 'flame', color: 'var(--heat-hot)', value: hotToday },
    { key: 'reps', label: 'Active reps', icon: 'users', color: 'var(--live)', value: repsToday, of: repsTotal },
    { key: 'all', label: 'All contacts', icon: 'contacts', color: 'var(--heat-cool)', value: totalContacts },
  ];

  return (
    // laid out by the space the band has (container queries), not the window: with the sidebar open it stacks cleanly
    <div className="@container relative isolate overflow-hidden rounded-2xl border border-panel-border bg-panel text-panel-foreground shadow-card">
      {/* soft glow in the live colour, top-left */}
      <span aria-hidden className="pointer-events-none absolute -left-20 -top-24 -z-10 h-56 w-72 rounded-full opacity-60 blur-3xl"
        style={{ background: `radial-gradient(closest-side, ${tint(accent, 28)}, transparent)` }} />

      <div className="grid items-stretch gap-3 p-3 @2xl:grid-cols-2 @5xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)_minmax(0,1.15fr)]">
        {/* who */}
        <div className="flex min-w-0 flex-col justify-center gap-2 px-1.5">
          <div className="flex items-center gap-2">
            <span className="es-eyebrow text-[10px] text-panel-muted">On the floor now</span>
            <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-bold tracking-[0.06em]"
              style={{ background: tint(accent, 16), color: ink(accent) }}>
              <span className="relative flex h-1.5 w-1.5">
                {live ? <span className="absolute inset-0 animate-ping rounded-full opacity-70" style={{ background: accent }} /> : null}
                <span className="relative h-1.5 w-1.5 rounded-full" style={{ background: accent }} />
              </span>
              {live ? 'LIVE' : 'QUIET'}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <div className="flex items-baseline gap-2">
              <span className="es-num text-[34px] font-bold leading-none tracking-[-0.03em]">{loading ? '—' : <AnimatedNumber value={repsToday} />}</span>
              <span className="text-sm font-medium leading-tight text-panel-muted">
                {loading ? 'loading…' : live ? <>rep{repsToday === 1 ? '' : 's'} capturing<br /> leads today</> : <>no captures<br /> yet today</>}
              </span>
            </div>
            {reps.length ? (
              <span className="flex items-center -space-x-1.5" aria-label={`Active today: ${reps.map((r) => r.name).join(', ')}`}>
                {reps.slice(0, 5).map((r, i) => (
                  <motion.span key={r.name} title={`${r.name} · ${r.n} today`}
                    initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 420, damping: 20, delay: 0.1 + i * 0.05 }}
                    className="flex h-7 w-7 items-center justify-center rounded-full text-[10px] font-bold tracking-tight text-white ring-2 ring-panel"
                    style={{ background: SEG_COLORS[i % SEG_COLORS.length], zIndex: 10 - i }}>
                    {r.photo ? <img src={r.photo} alt="" className="size-full rounded-full object-cover" /> : initials(r.name)}
                  </motion.span>
                ))}
                {reps.length > 5 ? <span className="flex h-7 min-w-7 items-center justify-center rounded-full bg-panel px-1 text-[10px] font-bold text-panel-muted ring-2 ring-panel-border">+{reps.length - 5}</span> : null}
              </span>
            ) : null}
          </div>

          {latest ? (
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={latest.id} className="flex min-w-0 max-w-full items-center gap-1.5 self-start rounded-full border border-panel-border bg-panel-foreground/[0.03] py-1 pl-1.5 pr-2.5 text-xs text-panel-muted"
                initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.2 }}>
                <span className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full" style={{ background: tint(accent, 20), color: accent }}>
                  <Icon name="bolt" size={10} strokeWidth={2.5} />
                </span>
                <span className="truncate"><span className="font-medium text-panel-foreground">{latest.title}</span> · {relTime(latest.timestamp)}</span>
              </motion.div>
            </AnimatePresence>
          ) : null}
        </div>

        {/* when: today's captures by hour */}
        <div className="flex min-w-0 flex-col rounded-xl border border-panel-border bg-panel-foreground/[0.025] px-3 pb-2 pt-2.5">
          <div className="flex items-baseline justify-between gap-2 text-[10px] text-panel-muted">
            <span className="es-eyebrow">Today by hour</span>
            <span className="font-mono">{any ? `peak ${peak}/h` : 'no captures yet'}</span>
          </div>
          <div className="relative mt-auto flex h-12 items-end gap-[3px] pt-3" role="img"
            aria-label={`Contacts added today by hour: ${hours.map((n, h) => (n ? `${hourLabel(h)} ${n}` : null)).filter(Boolean).join(', ') || 'none yet'}`}>
            {hours.map((n, h) => {
              const future = h > nowHour;
              const now = h === nowHour;
              return (
                <motion.span key={h} title={`${hourLabel(h)}–${hourLabel((h + 1) % 24)} · ${n} contact${n === 1 ? '' : 's'}`}
                  className={cx('relative flex-1 origin-bottom rounded-t-[3px] rounded-b-[1px]',
                    future ? 'border border-dashed border-panel-border' : !n ? 'bg-panel-foreground/[0.12]' : '',
                    now && 'ring-2 ring-offset-1 ring-offset-panel')}
                  style={{
                    height: n ? `${Math.max(16, (n / peak) * 100)}%` : future ? '6px' : '3px',
                    ...(n ? { background: `linear-gradient(to top, ${tint('var(--live)', 55)}, var(--live))` } : {}),
                    ...(now ? { ['--tw-ring-color' as string]: tint('var(--live)', 60) } : {}),
                  }}
                  initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ duration: 0.45, ease: [0.2, 0.8, 0.2, 1], delay: 0.15 + h * 0.015 }}>
                  {now ? (
                    <span className="absolute -top-4 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-[4px] bg-panel-foreground px-1 font-mono text-[8.5px] font-bold leading-3 text-panel">NOW</span>
                  ) : null}
                </motion.span>
              );
            })}
          </div>
          <div className="mt-1 flex justify-between font-mono text-[9.5px] text-panel-muted" aria-hidden>
            <span>12a</span><span>6a</span><span>12p</span><span>6p</span><span>12a</span>
          </div>
        </div>

        {/* how many */}
        <div className="grid grid-cols-2 gap-2 @lg:grid-cols-4 @2xl:col-span-2 @5xl:col-span-1 @5xl:grid-cols-2">
          {stats.map((s, i) => (
            <motion.button key={s.key} onClick={() => onOpen(s.key)} whileTap={{ scale: 0.97 }}
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.1 + i * 0.05 }}
              className="group flex min-w-0 items-center gap-2.5 rounded-xl border border-panel-border bg-panel-foreground/[0.025] px-2.5 py-2 text-left transition-colors hover:bg-panel-foreground/[0.06] cursor-pointer">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-transform duration-300 group-hover:-rotate-6 group-hover:scale-110"
                style={{ background: tint(s.color, 16), color: s.color }}>
                <Icon name={s.icon} size={15} />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-[11px] font-medium text-panel-muted">{s.label}</span>
                <span className={cx('es-num block text-lg font-bold leading-5', s.key === 'hot' && s.value ? 'text-heat-hot' : '')}>
                  {s.value == null || loading ? '—' : <AnimatedNumber value={s.value} />}
                  {s.of != null && !loading ? <span className="text-xs font-medium text-panel-muted">/{s.of}</span> : null}
                </span>
              </span>
              <Icon name="chevronRight" size={13} className="ml-auto shrink-0 -translate-x-1 text-panel-muted opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" />
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Card heading: a quiet title and caption, no icon tile. */
function CardTitle({ title, subtitle, right }: { title: string; subtitle?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2 px-4 pt-3">
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold text-foreground">{title}</h3>
        {subtitle ? <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p> : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
  );
}

/** One cell of the ledger strip: label, big figure, change and a trend line. */
function LedgerCell({ color, label, value, decimals, change, unit, series, foot, onClick, loading, index }: {
  color: string; label: string; value: number | null | undefined; decimals?: number;
  change: number | null | undefined; unit?: string; series: number[]; foot?: React.ReactNode; onClick?: () => void; loading: boolean; index: number;
}) {
  const Comp: any = onClick ? 'button' : 'div';
  return (
    <Reveal index={index} className="flex">
      <Comp onClick={onClick} onMouseMove={spotMove}
        className={cx('es-spot group flex w-full flex-col bg-card text-left transition-colors', onClick && 'cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring')}>
        <div className="px-4 pt-2.5">
          <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <span className="h-2 w-2 rounded-[2px] transition-transform duration-300 group-hover:scale-125" style={{ background: color }} aria-hidden />
            {label}
            {onClick ? <Icon name="chevronRight" size={14} className="ml-auto text-muted-foreground/0 transition group-hover:translate-x-0.5 group-hover:text-muted-foreground" /> : null}
          </div>
          {loading ? <Shimmer className="mt-1 h-7.5 w-24" /> : (
            <div className="es-num mt-1 text-[30px] font-semibold leading-none text-foreground"><AnimatedNumber value={value} decimals={decimals ?? 0} /></div>
          )}
          <div className="mt-1 min-h-5">{loading ? <Shimmer className="h-4 w-28" /> : <Delta value={change} unit={unit ?? '%'} />}</div>
        </div>
        <div className="mt-auto px-1">{loading ? <div className="h-6" /> : <Sparkline values={series} color={color} height={24} />}</div>
        {foot ? <div className="border-t px-4 py-1.5 text-xs text-muted-foreground">{foot}</div> : null}
      </Comp>
    </Reveal>
  );
}

type LevelKey = keyof typeof LEVEL_COLORS;

/**
 * Lead quality: a headline share of Hot + Warm leads, one stacked bar of every level, a tile per
 * scored level and a call-out for contacts still waiting for a score. Hovering a tile or a bar
 * segment highlights the other; clicking either opens Contacts filtered to that band.
 */
function LeadQuality({ levels, average, loading, active, setActive, onOpen }: {
  levels: Record<LevelKey, number> | undefined; average: number | null | undefined; loading: boolean;
  active: string | null; setActive: (k: string | null) => void; onOpen: (band?: string) => void;
}) {
  const count = (k: LevelKey) => levels?.[k] ?? 0;
  const scoredRows = LEVEL_ROWS.filter((l) => l.key !== 'unscored');
  const scored = scoredRows.reduce((s, l) => s + count(l.key), 0);
  const unscored = count('unscored');
  const total = scored + unscored;
  const quality = scored ? Math.round(((count('hot') + count('warm')) / scored) * 100) : null;
  const pct = (v: number, of: number) => (of ? Math.round((v / of) * 100) : 0);
  const hover = (k: string | null) => () => setActive(k);

  return (
    <Card className="es-lift flex h-full flex-col">
      <CardTitle title="Lead quality" subtitle="Every contact, by AI score level" right={<SwapLink onClick={() => onOpen()}>All contacts</SwapLink>} />

      {loading ? (
        <div className="space-y-3 p-4"><Shimmer className="h-10 w-36" /><Shimmer className="h-2.5" />
          <div className="grid grid-cols-2 gap-2">{[0, 1, 2, 3].map((i) => <Shimmer key={i} className="h-21" />)}</div></div>
      ) : !total ? (
        <EmptyState icon="flame" title="No contacts yet" description="Lead levels appear once your reps add and score contacts." className="flex-1 py-8" />
      ) : (
        <div className="flex flex-1 flex-col gap-2.5 px-4 pb-4 pt-3">
          <div className="flex items-end justify-between gap-3">
            <div>
              <div className="es-num flex items-baseline gap-1 text-[32px] font-semibold leading-none text-foreground">
                {quality == null ? '—' : <><AnimatedNumber value={quality} /><span className="text-xl text-muted-foreground">%</span></>}
              </div>
              <p className="mt-1.5 text-xs text-muted-foreground">of scored leads are <b className="font-semibold text-foreground">Hot</b> or <b className="font-semibold text-foreground">Warm</b></p>
            </div>
            {average != null ? (
              <div className="rounded-lg border px-2.5 py-1.5 text-right">
                <div className="es-num text-lg font-semibold text-foreground">{average}</div>
                <div className="es-eyebrow text-[9.5px] text-muted-foreground">avg score</div>
              </div>
            ) : null}
          </div>

          <div className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-[3px]" role="list" aria-label="Contacts by level">
            {LEVEL_ROWS.map((l) => {
              const v = count(l.key);
              if (!v) return null;
              return (
                <button key={l.key} role="listitem" title={`${l.label}: ${v.toLocaleString()} (${pct(v, total)}%)`} aria-label={`${l.label}: ${v} contacts`}
                  onMouseEnter={hover(l.key)} onMouseLeave={hover(null)} onFocus={hover(l.key)} onBlur={hover(null)} onClick={() => onOpen(l.band)}
                  className="es-grow-x h-full min-w-1.5 cursor-pointer transition-opacity duration-200"
                  style={{ flexGrow: v, background: LEVEL_COLORS[l.key], opacity: active && active !== l.key ? 0.3 : 1 }} />
              );
            })}
          </div>

          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border">
            {scoredRows.map((l, i) => {
              const v = count(l.key);
              const c = LEVEL_COLORS[l.key];
              const on = active === l.key;
              const share = pct(v, scored);
              return (
                <motion.button key={l.key} onMouseEnter={hover(l.key)} onMouseLeave={hover(null)} onFocus={hover(l.key)} onBlur={hover(null)} onClick={() => onOpen(l.band)}
                  initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} whileTap={{ scale: 0.98 }}
                  transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1], delay: 0.15 + i * 0.05 }}
                  className="group relative cursor-pointer bg-card p-2.5 text-left transition-colors duration-200"
                  style={on ? { background: tint(c, 9) } : {}}>
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded-lg font-mono text-[10px] font-semibold transition-transform duration-300 group-hover:scale-110" style={{ background: tint(c, 18), color: ink(c) }}>{l.band}</span>
                    <span className="text-sm font-medium text-body">{l.label}</span>
                    <Icon name="chevronRight" size={14} className={cx('ml-auto text-muted-foreground transition', on ? 'translate-x-0 opacity-100' : '-translate-x-1 opacity-0')} />
                  </div>
                  <div className="mt-1.5 flex items-baseline justify-between gap-2">
                    <span className="es-num text-2xl font-semibold text-foreground"><AnimatedNumber value={v} /></span>
                    <span className="font-mono text-xs tabular-nums text-muted-foreground">{share}%</span>
                  </div>
                  <div className="mt-0.5 font-mono text-[10.5px] text-muted-foreground">score {l.range}</div>
                  <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-secondary">
                    <div className="es-grow-x h-full rounded-full" style={{ width: `${share}%`, background: c, animationDelay: `${200 + i * 60}ms` }} />
                  </div>
                </motion.button>
              );
            })}
          </div>

          {unscored ? (
            <button onMouseEnter={hover('unscored')} onMouseLeave={hover(null)} onClick={() => onOpen('U')}
              className={cx('group mt-auto flex cursor-pointer items-center gap-2.5 rounded-lg border border-dashed px-3 py-2 text-left text-sm transition',
                active === 'unscored' ? 'border-foreground/40 bg-secondary' : 'hover:border-foreground/30')}
              style={{ borderColor: active === 'unscored' ? undefined : 'var(--dashed)' }}>
              <Icon name="sparkles" size={15} className="shrink-0 text-warning transition-transform duration-300 group-hover:rotate-12 group-hover:scale-110" />
              <span className="flex-1 text-body"><b className="font-semibold tabular-nums text-foreground">{unscored.toLocaleString()}</b> {unscored === 1 ? 'contact is' : 'contacts are'} waiting for a score</span>
              <span className="flex items-center gap-0.5 text-xs font-semibold text-foreground">Review<Icon name="chevronRight" size={13} className="transition-transform group-hover:translate-x-0.5" /></span>
            </button>
          ) : (
            <div className="mt-auto flex items-center gap-2 rounded-lg bg-success-bg px-3 py-2 text-sm text-success">
              <Icon name="check" size={15} />Every contact has a score
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

/** Average-score zones behind the dot plot: the lead levels, as faint columns. */
const SCORE_ZONES: { key: LevelKey; label: string; from: number; to: number }[] = [
  { key: 'cold', label: 'Cold', from: 0, to: 50 }, { key: 'cool', label: 'Cool', from: 50, to: 70 },
  { key: 'warm', label: 'Warm', from: 70, to: 85 }, { key: 'hot', label: 'Hot', from: 85, to: 100 },
];
const levelKeyOf = (score: number): LevelKey => levelInfo(score)!.key;
const levelLabel = (score: number) => levelInfo(score)!.label;
// categorical slots in their fixed (validated) order; seniority is ordered, so it wears one ramp
const SEG_COLORS = ['var(--seg-1)', 'var(--seg-2)', 'var(--seg-3)', 'var(--seg-4)', 'var(--seg-5)', 'var(--seg-6)'];
const SENIORITY_KEYS = ['owner', 'clevel', 'vp', 'director', 'manager', 'ic', 'student'];
const PLOT_COLS = 'grid-cols-[minmax(0,6.5rem)_minmax(0,1fr)_2.25rem_1.75rem] sm:grid-cols-[minmax(0,9.5rem)_minmax(0,1fr)_2.75rem_2rem]';
// mobile caption: 10–11px, heavy, tracked capitals (the app's "LEAD QUALITY" section labels)
const CAPTION = 'text-[10px] font-extrabold uppercase tracking-[0.08em] text-muted-foreground';

/** The app's level badge: a small rounded tag in the level's tint (HOT · WARM · COOL · COLD). */
function LevelBadge({ score, withScore }: { score: number; withScore?: boolean }) {
  const c = LEVEL_COLORS[levelKeyOf(score)];
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-[5px] px-1.5 py-px text-[9.5px] font-black tracking-[0.06em]"
      style={{ background: tint(c, 18), color: `color-mix(in oklab, ${c} 62%, var(--foreground))` }}>
      {withScore ? <span className="tabular-nums">{score}</span> : null}{levelLabel(score).toUpperCase()}
    </span>
  );
}

/**
 * Loading state shaped like the chart it stands in for: shimmering tiles, a donut with a spinning
 * arc, and rows whose dots glide along the score track (staggered), so the wait reads as "working".
 */
function AudienceLoading() {
  return (
    <div className="flex flex-1 flex-col gap-3 px-3.5 pb-3.5 pt-3" aria-busy="true" aria-label="Loading who your team is meeting">
      <div className="grid gap-2 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="flex items-center gap-2.5 rounded-2xl border bg-card py-2.5 pl-3.5 pr-3">
            <span className="es-shimmer h-7.5 w-7.5 shrink-0 rounded-[10px]" style={{ animationDelay: `${i * 120}ms` }} />
            <div className="flex-1 space-y-1.5">
              <span className="es-shimmer block h-2 w-14 rounded-full" style={{ animationDelay: `${i * 120 + 60}ms` }} />
              <span className="es-shimmer block h-3 w-28 rounded-full" style={{ animationDelay: `${i * 120 + 120}ms` }} />
            </div>
            <span className="es-shimmer h-6 w-10 rounded-md" style={{ animationDelay: `${i * 120 + 180}ms` }} />
          </div>
        ))}
      </div>
      <div className="grid flex-1 items-center gap-4 sm:grid-cols-[152px_minmax(0,1fr)]">
        <div className="relative mx-auto h-38 w-38">
          <svg viewBox="0 0 152 152" className="h-full w-full">
            <circle cx="76" cy="76" r="63" fill="none" strokeWidth="20" className="text-secondary" stroke="currentColor" />
            <g className="es-spin-slow">
              <circle cx="76" cy="76" r="63" fill="none" strokeWidth="20" strokeLinecap="round" strokeDasharray="70 326" style={{ stroke: 'var(--seg-1)', opacity: 0.85 }} />
              <circle cx="76" cy="76" r="63" fill="none" strokeWidth="20" strokeLinecap="round" strokeDasharray="34 362" strokeDashoffset="-96" style={{ stroke: 'var(--seg-1)', opacity: 0.35 }} />
            </g>
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5">
            <span className="es-shimmer h-5 w-12 rounded-md" /><span className="es-shimmer h-2 w-14 rounded-full" />
          </div>
        </div>
        <ul className="space-y-1.5">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <li key={i} className={cx('grid h-7.5 items-center gap-x-2.5', PLOT_COLS)}>
              <span className="es-shimmer h-2.5 rounded-full" style={{ width: `${82 - (i % 3) * 16}%`, animationDelay: `${i * 90}ms` }} />
              <span className="relative h-full overflow-hidden rounded-lg bg-secondary/60">
                <span className="es-glide absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card"
                  style={{ background: SEG_COLORS[i % SEG_COLORS.length], animationDelay: `${i * 160}ms` }} />
              </span>
              <span className="es-shimmer ml-auto h-2.5 w-6 rounded-full" style={{ animationDelay: `${i * 90 + 40}ms` }} />
              <span className="es-shimmer ml-auto h-2.5 w-4 rounded-full" style={{ animationDelay: `${i * 90 + 80}ms` }} />
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

const glide = { initial: { opacity: 0, y: 6, filter: 'blur(4px)' }, animate: { opacity: 1, y: 0, filter: 'blur(0px)' }, exit: { opacity: 0, y: -4, filter: 'blur(4px)' } };

/**
 * Audience mix, in the mobile app's card language: the period's contacts split by industry (from
 * AI research), role and seniority (from the job title) or company. Two insight tiles lead (most
 * met, best leads); the donut shows each group's share; the dot plot puts every group on the 0–100
 * score axis (dot size = contacts). Hovering a tile, a slice or a row highlights that group
 * everywhere; companies open in Contacts.
 */
function AudienceMix({ audience, subtitle, loading, onCompany }: {
  audience: Audience | undefined; subtitle: string; loading: boolean; onCompany: (name: string) => void;
}) {
  const [picked, setPicked] = useState<AudienceDim | null>(null);
  const [active, setActive] = useState<string | null>(null);
  // until the admin picks a tab: industry when research has found any, else role
  const dim: AudienceDim = picked ?? (audience?.industry.known ? 'industry' : 'function');
  const split = audience?.[dim];
  const total = audience?.total ?? 0;
  // the "no job title" row always goes last, after the real groups
  const rows = split ? [...split.rows].sort((a, b) => Number(a.key === 'unknown') - Number(b.key === 'unknown')) : [];
  const isReal = (x: AudienceRow) => !x.other && x.key !== 'unknown';
  const real = rows.filter(isReal);
  const colorOf = (x: AudienceRow) => {
    if (!isReal(x)) return 'var(--seg-other)';
    if (dim === 'seniority') return `var(--rank-${SENIORITY_KEYS.indexOf(x.key) + 1})`;
    return SEG_COLORS[real.indexOf(x)] ?? 'var(--seg-other)';
  };
  const maxCount = Math.max(1, ...rows.map((x) => x.count));
  const known = split?.known ?? 0;
  // shares are of everything the donut shows, so the tiles, the donut and the rows agree
  const base = rows.reduce((s, x) => s + x.count, 0);
  const pct = (v: number) => (base ? Math.round((v / base) * 100) : 0);
  const biggest = real.length ? real.reduce((a, b) => (b.count > a.count ? b : a)) : undefined;
  // best leads = best average among groups of two or more contacts (one contact is noise)
  const strongest = real.filter((x) => x.count >= 2 && x.avgScore != null).sort((a, b) => b.avgScore! - a.avgScore! || b.hot - a.hot)[0];
  const clickable = (x: AudienceRow) => dim === 'company' && !x.other;
  const pick = (k: AudienceDim) => { setPicked(k); setActive(null); };
  const hoverOn = (k: string) => () => setActive(k);
  const hoverOff = () => setActive(null);
  const centerText = dim === 'industry' ? 'researched' : dim === 'company' ? 'with a company' : 'contacts';
  const chipCount = (k: AudienceDim) => (!audience ? undefined : k === 'company' ? audience.company.distinct : audience[k].known);

  const insights: { label: string; icon: IconName; row: AudienceRow | undefined; big: number; suffix: string; small: React.ReactNode; empty: string }[] = [
    { label: 'Most met', icon: 'users', row: biggest, big: biggest ? pct(biggest.count) : 0, suffix: '%',
      small: biggest ? `${biggest.count.toLocaleString()} contact${biggest.count === 1 ? '' : 's'}` : null, empty: 'No groups yet' },
    { label: 'Best leads', icon: 'flame', row: strongest, big: strongest?.avgScore ?? 0, suffix: '',
      small: strongest ? <span className="inline-flex items-center gap-1.5"><LevelBadge score={strongest.avgScore!} />{strongest.hot ? `${strongest.hot} hot` : 'avg'}</span> : null,
      empty: 'Not enough scored contacts' },
  ];

  // footer: how much of the period this split covers, as words + a small meter
  const coverageText = !audience ? null : dim === 'company'
    ? <><b className="font-bold text-foreground">{audience.company.distinct.toLocaleString()}</b> companies{audience.company.repeat ? <> · {audience.company.repeat} with 2+ contacts</> : null}</>
    : <><b className="font-bold text-foreground">{known.toLocaleString()}</b> of {total.toLocaleString()} {dim === 'industry' ? 'have an industry · detected by AI' : 'have a job title'}</>;
  const coverageShare = dim === 'company' ? null : total ? known / total : 0;
  // industry is worked out automatically in the background; say what the AI is doing
  const ind = audience?.industry;
  const pending = ind && !ind.error ? ind.pending ?? 0 : 0;
  const aiStatus = dim !== 'industry' || !ind ? null : pending ? (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full bg-brand-50 px-2 py-0.5 font-bold text-brand-700">
      <span className="relative flex h-2 w-2" aria-hidden>
        <span className="absolute inset-0 animate-ping rounded-full bg-brand-500/60" /><span className="relative h-2 w-2 rounded-full bg-brand-600" />
      </span>
      AI is classifying {pending.toLocaleString()} {pending === 1 ? 'company' : 'companies'}
    </span>
  ) : ind.error ? <span className="font-semibold text-warning">AI is unavailable right now — it will retry by itself</span>
    : ind.ai === false ? <span className="font-semibold text-warning">Set the AI key to detect industries</span> : null;

  return (
    <Card className="flex h-full flex-col rounded-[18px]">
      {/* header: the app's icon-square title, then its filter chips with counts */}
      <div className="px-3.5 pt-3.5">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[9px] bg-brand-600 text-white" aria-hidden><Icon name="users" size={15} /></span>
          <div className="min-w-0">
            <h3 className="font-sans text-[15px] font-extrabold leading-5 tracking-[-0.01em] text-foreground">Who your team is meeting</h3>
            <p className="truncate text-xs font-medium text-muted-foreground">{subtitle}</p>
          </div>
        </div>
        <div role="tablist" className="mt-2.5 flex flex-wrap gap-1.5">
          {AUDIENCE_TABS.map((t) => {
            const on = dim === t.value;
            const n = chipCount(t.value);
            return (
              <motion.button key={t.value} role="tab" aria-selected={on} onClick={() => pick(t.value)} whileTap={{ scale: 0.96 }}
                className={cx('relative inline-flex h-7.5 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] pl-3 text-[13px] font-bold transition-colors duration-200',
                  n != null ? 'pr-1.5' : 'pr-3',
                  on ? 'border-transparent text-white' : 'border-border bg-card text-foreground hover:border-brand-300')}>
                {/* the selected fill slides from chip to chip (MicroKit "spotlight indicator") */}
                {on ? <motion.span layoutId="audience-chip" className="absolute -inset-[1.5px] rounded-full bg-brand-600 shadow-[0_4px_12px_-4px_var(--brand-600)]"
                  transition={{ type: 'spring', stiffness: 480, damping: 38 }} aria-hidden /> : null}
                <span className="relative">{t.label}</span>
                {n != null ? (
                  <span className={cx('relative min-w-5.5 rounded-full px-1.5 text-center text-[11px] font-extrabold leading-5 tabular-nums transition-colors duration-200', on ? 'bg-black/25 text-white' : 'bg-secondary text-body')}>
                    <AnimatedNumber value={n} duration={600} />
                  </span>
                ) : null}
              </motion.button>
            );
          })}
        </div>
      </div>

      {loading ? <AudienceLoading /> : !audience ? (
        <EmptyState icon="info" title="Not available yet" description="Update the backend to see who your team is meeting." className="flex-1 py-8" />
      ) : !total ? (
        <EmptyState icon="users" title="Nothing in this period" description="Contacts your reps add show up here by industry, role and company." className="flex-1 py-8" />
      ) : !real.length && dim === 'industry' ? (
        pending ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-8 text-center">
            <span className="relative flex h-11 w-11 items-center justify-center rounded-full bg-brand-50 text-brand-700" aria-hidden>
              <span className="es-spin-slow absolute inset-0 rounded-full border-2 border-transparent border-t-brand-600" />
              <LiveIcon name="sparkles" size={17} />
            </span>
            <div>
              <p className="text-sm font-extrabold text-foreground">Working out industries…</p>
              <p className="mt-1 max-w-sm text-xs text-muted-foreground">The AI is classifying {pending.toLocaleString()} {pending === 1 ? 'company' : 'companies'} from their names, email domains and the people met. This fills in by itself.</p>
            </div>
          </div>
        ) : (
          <EmptyState icon="sparkles" title="No industries yet" className="flex-1 py-8"
            description={ind?.ai === false ? 'Industries are detected by AI. Set the AI key (OPENROUTER_API_KEY) on the backend and they fill in by themselves.'
              : 'None of these contacts has a company name or work email to classify.'} />
        )
      ) : (
        <div className="flex flex-1 flex-col gap-3 px-3.5 pb-3.5 pt-3">
          <AnimatePresence mode="wait" initial={false}>
          {/* switching tabs: the old split blurs out, the new one glides in (MicroKit "blur glide") */}
          <motion.div key={dim} {...glide} transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }} className="flex flex-1 flex-col gap-3">
          {/* the two answers worth having at a glance — the app's lead-card row */}
          <div className="grid gap-2 sm:grid-cols-2">
            {insights.map((s, si) => {
              const color = s.row ? colorOf(s.row) : 'var(--border)';
              const on = !!s.row && active === s.row.key;
              return (
                <motion.div key={s.label} onMouseEnter={s.row ? hoverOn(s.row.key) : undefined} onMouseLeave={hoverOff}
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} whileHover={{ y: -2 }}
                  transition={{ type: 'spring', stiffness: 380, damping: 30, delay: si * 0.06 }}
                  className={cx('relative flex items-center gap-2.5 overflow-hidden rounded-2xl border py-2.5 pl-3.5 pr-3 shadow-card transition-[background-color,border-color,box-shadow] duration-200 hover:shadow-md',
                    on ? 'border-brand-300 bg-brand-50' : 'bg-card')}>
                  <span className="absolute bottom-3 left-0 top-3 w-[3px] rounded-r-[3px]" style={{ background: color }} aria-hidden />
                  <span className="flex h-7.5 w-7.5 shrink-0 items-center justify-center rounded-[10px] text-foreground" style={{ background: tint(color, 18) }} aria-hidden>
                    <Icon name={s.icon} size={15} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className={CAPTION}>{s.label}</div>
                    <div className={cx('break-words leading-tight [overflow-wrap:anywhere]', s.row ? 'text-sm font-extrabold text-foreground' : 'text-[13px] font-medium text-muted-foreground')}>{s.row?.label ?? s.empty}</div>
                  </div>
                  {s.row ? (
                    <div className="flex shrink-0 flex-col items-end gap-0.5">
                      <div className="text-xl font-black leading-6 tabular-nums tracking-[-0.02em] text-foreground">
                        <AnimatedNumber value={s.big} decimals={Number.isInteger(s.big) ? 0 : 1} suffix={s.suffix} duration={700} />
                      </div>
                      <div className="whitespace-nowrap text-[11px] font-bold text-muted-foreground">{s.small}</div>
                    </div>
                  ) : null}
                </motion.div>
              );
            })}
          </div>

          <div className="grid flex-1 items-center gap-4 sm:grid-cols-[152px_minmax(0,1fr)]">
            {/* share of contacts */}
            <div className="flex justify-center">
              <Donut key={dim} size={152} thickness={20} active={active} setActive={setActive} centerLabel={centerText}
                segments={rows.map((x) => ({ key: x.key, label: x.label, value: x.count, color: colorOf(x) }))}
                center={(h, t) => (
                  <>
                    <div className="text-2xl font-black leading-none tabular-nums tracking-[-0.02em] text-foreground">{h ? `${pct(h.value)}%` : <AnimatedNumber value={t} duration={800} />}</div>
                    <div className={cx('mt-1 line-clamp-2 text-[11px] font-bold leading-tight', h ? 'text-body' : 'text-muted-foreground')}>{h ? h.label : centerText}</div>
                  </>
                )}
                {...(dim === 'company' ? { onSelect: (k: string) => { const x = rows.find((r) => r.key === k); if (x && clickable(x)) onCompany(x.label); } } : {})} />
            </div>

            {/* where each group scores */}
            <div className="min-w-0">
              <div className={cx('grid items-end gap-x-2.5 pb-1', PLOT_COLS, CAPTION)}>
                <span className="truncate pl-1">{AUDIENCE_TABS.find((t) => t.value === dim)!.label}</span>
                <span className="relative h-3.5 max-sm:invisible">
                  {SCORE_ZONES.map((z) => (
                    <span key={z.key} className="absolute bottom-0 text-center" style={{ left: `${z.from}%`, width: `${z.to - z.from}%` }}>{z.label}</span>
                  ))}
                </span>
                <span className="text-right">Count</span><span className="pr-1 text-right">Hot</span>
              </div>

              <div className="relative">
                {/* the level zones, as one continuous band behind every row */}
                <div className={cx('pointer-events-none absolute inset-0 grid gap-x-2.5', PLOT_COLS)} aria-hidden>
                  <span />
                  <span className="relative overflow-hidden rounded-lg">
                    {SCORE_ZONES.map((z) => (
                      <span key={z.key} className="absolute inset-y-0 border-l-2 border-card first:border-l-0"
                        style={{ left: `${z.from}%`, width: `${z.to - z.from}%`, background: tint(LEVEL_COLORS[z.key], z.key === 'cold' ? 10 : 13) }} />
                    ))}
                  </span>
                </div>

                <ul className="relative">
                  {rows.map((x, i) => {
                    const c = colorOf(x);
                    const on = active === x.key;
                    const faded = active != null && !on;
                    const size = 8 + Math.round(12 * Math.sqrt(x.count / maxCount));
                    const Comp: any = clickable(x) ? 'button' : 'div';
                    const avg = x.avgScore;
                    // the value badge sits beside the dot, on whichever side has room — never over another row
                    const badgeLeft = avg != null && avg > 66;
                    const tip = `${x.label}: ${x.count.toLocaleString()} contact${x.count === 1 ? '' : 's'} (${pct(x.count)}%)`
                      + (avg != null ? `, average score ${avg} (${levelLabel(avg)})` : ', not scored yet') + (x.hot ? `, ${x.hot} hot` : '');
                    return (
                      <motion.li key={`${dim}:${x.key}`} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1], delay: 0.08 + i * 0.045 }}>
                        <Comp aria-label={tip} title={x.label} onClick={clickable(x) ? () => onCompany(x.label) : undefined}
                          onMouseEnter={hoverOn(x.key)} onMouseLeave={hoverOff} onFocus={hoverOn(x.key)} onBlur={hoverOff}
                          className={cx('group grid h-9 w-full items-center gap-x-2.5 rounded-lg text-left transition-colors duration-150', PLOT_COLS,
                            on && 'bg-foreground/[0.045]', clickable(x) && 'cursor-pointer')}>
                          <span className="flex min-w-0 items-center gap-2 pl-1">
                            <span className="h-2.5 w-2.5 shrink-0 rounded-[3px] transition-opacity duration-200" style={{ background: c, opacity: faded ? 0.35 : 1 }} aria-hidden />
                            <span className={cx('line-clamp-2 text-xs font-bold leading-[1.15] transition-colors duration-200 [overflow-wrap:anywhere]',
                              !isReal(x) || faded ? 'text-muted-foreground' : 'text-body', on && 'text-foreground', clickable(x) && 'underline-offset-2 group-hover:underline')}>{x.label}</span>
                          </span>

                          <span className="relative h-9">
                            {avg != null ? (
                              <>
                                {/* stem from 0 to the average, then the dot */}
                                <span className="es-grow-x absolute left-0 top-1/2 h-0.5 -translate-y-1/2 rounded-full transition-opacity duration-200"
                                  style={{ width: `${avg}%`, background: tint(c, 50), opacity: faded ? 0.3 : 1, animationDelay: `${i * 60}ms` }} />
                                <motion.span className="absolute top-1/2 rounded-full ring-2 ring-card"
                                  style={{ width: size, height: size, marginLeft: -size / 2, marginTop: -size / 2, background: c }}
                                  initial={{ left: '0%', scale: 0.4, opacity: 0 }} animate={{ left: `${avg}%`, scale: on ? 1.2 : 1, opacity: faded ? 0.3 : 1 }}
                                  transition={{ type: 'spring', stiffness: 170, damping: 22, delay: active == null ? i * 0.06 : 0 }} />
                                {on ? (
                                  <span className="es-fade pointer-events-none absolute top-1/2 z-10 -translate-y-1/2"
                                    style={badgeLeft ? { right: `calc(${100 - avg}% + ${size / 2 + 6}px)` } : { left: `calc(${avg}% + ${size / 2 + 6}px)` }}>
                                    <LevelBadge score={avg} withScore />
                                  </span>
                                ) : null}
                              </>
                            ) : (
                              <span className="absolute left-1.5 top-1/2 -translate-y-1/2 text-[11px] font-medium italic text-muted-foreground">not scored yet</span>
                            )}
                          </span>

                          <span className={cx('text-right text-xs font-extrabold tabular-nums transition-colors', faded ? 'text-muted-foreground' : 'text-foreground')}>{x.count.toLocaleString()}</span>
                          <span className={cx('pr-1 text-right text-xs font-extrabold tabular-nums', x.hot && !faded ? 'text-heat-hot' : 'text-muted-foreground')}>{x.hot || '—'}</span>
                        </Comp>
                      </motion.li>
                    );
                  })}
                </ul>
              </div>

              <div className={cx('grid gap-x-2.5 pt-1 text-[10px] font-bold tabular-nums text-muted-foreground', PLOT_COLS)} aria-hidden>
                <span className="truncate pl-1">Avg score</span>
                <span className="relative h-3">
                  {[0, 50, 70, 85, 100].map((t) => (
                    <span key={t} className={cx('absolute', t === 0 ? '' : t === 100 ? '-translate-x-full' : '-translate-x-1/2', (t === 70 || t === 85) && 'max-sm:hidden')} style={{ left: `${t}%` }}>{t}</span>
                  ))}
                </span>
              </div>
            </div>
          </div>

          </motion.div>
          </AnimatePresence>
        </div>
      )}

      {/* footer: its own strip, the card's last child — it can only ever sit at the bottom */}
      {!loading && audience && total ? (
        <footer className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t px-3.5 py-2.5 text-[11.5px] font-medium text-muted-foreground">
          {coverageShare != null ? (
            <span className="h-1.5 w-14 shrink-0 overflow-hidden rounded-full bg-secondary" role="meter" aria-valuemin={0} aria-valuemax={100}
              aria-valuenow={Math.round(coverageShare * 100)} aria-label="Coverage">
              <motion.span key={dim} className="block h-full rounded-full bg-brand-600" initial={{ width: 0 }} animate={{ width: `${Math.max(coverageShare * 100, coverageShare ? 4 : 0)}%` }}
                transition={{ duration: 0.8, ease: [0.2, 0.8, 0.2, 1], delay: 0.15 }} />
            </span>
          ) : null}
          <span className="min-w-0">{coverageText}</span>
          {aiStatus}
          <span className="ml-auto inline-flex items-center gap-1.5 whitespace-nowrap">
            <span className="inline-flex items-end gap-0.5" aria-hidden>
              <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60" /><span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/60" />
            </span>
            Dot size = contacts
          </span>
        </footer>
      ) : null}
    </Card>
  );
}

export function EventBoardComponent() {
  const navigate = useNavigate();
  // the range lives in the URL (?days=30); the loader warmed the cache for it. Switching tabs keeps
  // the previous range on screen (placeholderData) instead of flashing the skeleton.
  const { days } = useSearch({ from: '/_auth/' });
  const range = String(days) as Range;
  const setRange = (v: Range) => navigate({ to: '/', search: { days: Number(v) }, replace: true });
  const dash = useQuery({
    ...q.dashboard(days), placeholderData: (prev) => prev,
    // while the AI is still working out industries, refresh so they fill in by themselves
    refetchInterval: (query) => {
      const ind = query.state.data?.range.audience?.industry;
      return ind?.pending && !ind.error ? 6000 : false;
    },
  });
  // shared with the header's Activity menu (same query), used here for "Last activity"
  const activity = useSuspenseQuery(q.activity(200));
  const [activeLevel, setActiveLevel] = useState<string | null>(null);
  // the top progress sweep shows for a refresh the admin asked for or a range switch — not for
  // the quiet background polls while industries fill in
  const [refreshing, setRefreshing] = useState(false);
  const refresh = () => { setRefreshing(true); Promise.allSettled([dash.refetch(), activity.refetch()]).finally(() => setRefreshing(false)); };
  const busy = refreshing || dash.isPlaceholderData;

  const d = dash.data;
  const r = d?.range;
  const loading = dash.isLoading;
  const goContacts = (search: Record<string, string>) => navigate({ to: '/contacts', search } as any);
  const name = (getSession()?.name || '').split(' ')[0];

  // average-score sparkline: carry the last known value over days with no scored contacts
  const avgSeries = useMemo(() => {
    let last = 0;
    return (r?.series || []).map((p) => (p.avgScore != null ? (last = p.avgScore) : last));
  }, [r]);
  const pipeTotal = (d?.pipelines.complete ?? 0) + (d?.pipelines.failed ?? 0);
  // non-admin accounts only (activeReps / totalReps); older servers send activeRepsToday / reps
  const repsToday = d?.activeReps ?? r?.activeRepsToday ?? 0;

  return (
    <div className="space-y-3">
      {/* refresh / range switch: a thin sweep along the top of the window */}
      <AnimatePresence>
        {busy ? (
          <motion.div key="progress" className="pointer-events-none fixed inset-x-0 top-0 z-50 h-0.5 overflow-hidden"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} aria-hidden>
            <div className="es-progress h-full w-1/3 bg-[linear-gradient(90deg,transparent,var(--primary),transparent)]" />
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* Header: a tinted hero with two bubbles drifting behind it (same as every page) */}
      <div className="es-in relative isolate flex flex-wrap items-end justify-between gap-3 overflow-hidden rounded-3xl border p-5 shadow-card sm:p-6"
        style={{ background: 'linear-gradient(135deg, color-mix(in oklab, var(--brand-500) 12%, var(--card)), var(--card) 62%)' }}>
        <span aria-hidden className="es-bubble pointer-events-none absolute -right-16 -top-24 -z-10 size-64 rounded-full" style={{ background: 'color-mix(in oklab, var(--brand-500) 13%, transparent)' }} />
        <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-28 right-52 -z-10 size-52 rounded-full" style={{ background: 'color-mix(in oklab, #0ea5e9 8%, transparent)' }} />
        <span aria-hidden className="es-bubble-2 pointer-events-none absolute -left-10 -top-16 -z-10 size-32 rounded-full" style={{ background: 'color-mix(in oklab, #10b981 6%, transparent)' }} />
        <div className="flex min-w-0 items-center gap-3.5">
          {/* the page's live icon, like every other page header */}
          <span className="es-pop-in hidden size-12 shrink-0 items-center justify-center rounded-2xl text-brand-700 shadow-card sm:flex"
            style={{ background: 'color-mix(in oklab, var(--brand-600) 14%, var(--card))' }}>
            <LiveIcon name="dashboard" size={22} />
          </span>
          <div className="min-w-0">
            <motion.p className="es-eyebrow es-eyebrow-mark text-muted-foreground" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.35 }}>
              {new Date().toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' })}
            </motion.p>
            <h1 className="mt-0.5 font-display text-[24px] font-bold leading-8 tracking-[-0.03em] text-foreground">
              <GlideWords text={`${greeting()}${name ? `, ${name}` : ''}`} delay={0.05} />
            </h1>
            <motion.p className="text-sm text-body" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, delay: 0.3 }}>
              How your team's leads are coming in. Every number is live from the database.
            </motion.p>
          </div>
        </div>
        <motion.div className="flex flex-wrap items-center gap-2" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.15 }}>
          <SlidingTabs value={range} onChange={setRange} options={RANGES} size="sm" />
          <IconButton icon="refresh" label="Refresh" onClick={refresh} className={cx('border bg-card transition-transform active:scale-90', busy ? '[&_svg]:animate-spin' : '')} />
        </motion.div>
      </div>

      {dash.error ? <ErrorState error={dash.error} onRetry={() => dash.refetch()} /> : null}

      {/* On the floor now: the one highlighted band on the page (light in light mode, dark in dark mode) */}
      <Reveal index={1}>
        <FloorNow loading={loading} activity={activity.data ?? []} todayByHour={d?.todayByHour} todayReps={d?.repsToday}
          repsToday={repsToday} repsTotal={d?.totalReps ?? d?.reps}
          addedToday={r?.series.at(-1)?.added} hotToday={r?.series.at(-1)?.hot} totalContacts={d?.totalContacts}
          // "today" tiles open the grid filtered to contacts created since local midnight (?since=today)
          onOpen={(k) => (k === 'reps' ? navigate({ to: '/seats' })
            : goContacts(k === 'added' ? { since: 'today' } : k === 'hot' ? { since: 'today', band: 'A' } : {}))} />
      </Reveal>

      {/* Ledger strip: the four numbers that matter, one surface split by hairlines */}
      <div className="grid gap-px overflow-hidden rounded-xl border bg-border shadow-card sm:grid-cols-2 xl:grid-cols-4">
        <LedgerCell index={2} color="var(--chart-1)" label={`Contacts added · ${days}d`} value={r?.totals.added} change={r?.change.added} loading={loading}
          series={(r?.series || []).map((p) => p.added)} onClick={() => goContacts({})} foot={r ? `${r.previous.added.toLocaleString()} in the ${days} days before` : undefined} />
        <LedgerCell index={3} color="var(--heat-hot)" label={`Hot leads · ${days}d`} value={r?.totals.hot} change={r?.change.hot} loading={loading}
          series={(r?.series || []).map((p) => p.hot)} onClick={() => goContacts({ band: 'A' })} foot={d ? `${d.levels.hot.toLocaleString()} hot leads overall` : undefined} />
        <LedgerCell index={4} color="var(--heat-warm)" label={`AI-scored · ${days}d`} value={r?.totals.scored} change={r?.change.scored} loading={loading}
          series={(r?.series || []).map((p) => p.scored)} onClick={() => goContacts({ sort: 'score' })} foot={d ? <span className={d.levels.unscored ? 'text-warning' : ''}>{d.levels.unscored.toLocaleString()} still waiting for a score</span> : undefined} />
        <LedgerCell index={5} color="var(--heat-cool)" label={`Average score · ${days}d`} value={r?.totals.avgScore} decimals={1} change={r?.change.avgScore} unit=" pts" loading={loading}
          series={avgSeries} foot={d?.averageScore != null ? `${d.averageScore} across every scored contact` : 'No scored contacts yet'} />
      </div>

      {/* Audience + quality */}
      <div className="grid gap-3 xl:grid-cols-3">
        <Reveal index={6} className="xl:col-span-2">
          <AudienceMix audience={r?.audience} loading={loading} subtitle={r ? `Contacts added ${longDay(r.from)} – ${longDay(r.to)}` : ' '}
            onCompany={(name) => goContacts({ q: name })} />
        </Reveal>

        <Reveal index={7}>
          <LeadQuality levels={d?.levels} average={d?.averageScore} loading={loading} active={activeLevel} setActive={setActiveLevel}
            onOpen={(band) => goContacts(band ? { band } : {})} />
        </Reveal>
      </div>

      {/* Distribution · reps · events */}
      <div className="grid gap-3 lg:grid-cols-3">
        <Reveal index={8}>
          <Card className="es-lift h-full">
            <CardTitle title="Score distribution" subtitle="Contacts in each score band" />
            <div className="px-4 pb-4 pt-3">
              {loading ? <div className="flex h-42 items-end gap-1.5">{[38, 52, 70, 88, 64, 96, 72, 50, 34, 22].map((h, i) => <Shimmer key={i} className="flex-1 rounded-b-none" style={{ height: `${h}%`, animationDelay: `${i * 70}ms` }} />)}</div>
                : r && r.scoreHistogram.some((b) => b.count) ? (
                <Columns height={168} data={r.scoreHistogram.map((b) => ({ label: `${b.from}`, value: b.count, tip: `${b.from}–${b.to}: ${b.count}` }))}
                  color={(i) => bucketColor(r.scoreHistogram[i]!.from, r.scoreHistogram[i]!.to)} />
              ) : <EmptyState icon="flame" title="No scores yet" description="Score contacts to see the spread." />}
            </div>
          </Card>
        </Reveal>

        <Reveal index={9}>
          <Card className="es-lift h-full">
            <CardTitle title="Top reps" subtitle={`Most contacts added · ${days} days`} right={<SwapLink onClick={() => navigate({ to: '/seats' })}>Team</SwapLink>} />
            <div className="px-4 pb-4 pt-3">
              {loading ? <div className="space-y-3">{[0, 1, 2, 3, 4].map((i) => <div key={i} className="flex items-center gap-3"><Shimmer className="h-6 w-6" /><div className="flex-1 space-y-1.5"><Shimmer className="h-2.5 w-2/3" /><Shimmer className="h-1.5" /></div></div>)}</div>
                : r?.topReps.length ? (
                <ol className="space-y-2">
                  {r.topReps.map((p, i) => {
                    const max = r.topReps[0]!.added || 1;
                    return (
                      <motion.li key={p.userId} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1], delay: 0.1 + i * 0.05 }}>
                        <button onClick={() => goContacts({ owner: p.userId })} className="group flex w-full cursor-pointer items-center gap-3 text-left">
                          <motion.span initial={{ scale: 0.5, rotate: -12 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.15 + i * 0.05 }}
                            className={cx('flex h-6 w-6 shrink-0 items-center justify-center rounded-[5px] font-mono text-[11px] font-semibold transition-transform group-hover:scale-110',
                            i === 0 ? 'bg-signal text-signal-foreground ring-1 ring-black/15' : 'bg-secondary text-muted-foreground')}>{i + 1}</motion.span>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="truncate text-sm font-medium text-foreground underline-offset-2 group-hover:underline">{p.name}</span>
                              <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{p.hot ? <span className="text-heat-hot">{p.hot} hot · </span> : null}{p.avgScore != null ? `avg ${p.avgScore}` : 'unscored'}</span>
                            </div>
                            <div className="mt-1.5 flex items-center gap-2">
                              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
                                <div className="es-grow-x h-full rounded-full bg-foreground/80" style={{ width: `${(p.added / max) * 100}%`, animationDelay: `${i * 80}ms` }} />
                              </div>
                              <span className="es-num w-8 text-right text-sm font-semibold text-foreground"><AnimatedNumber value={p.added} duration={700} /></span>
                            </div>
                          </div>
                        </button>
                      </motion.li>
                    );
                  })}
                </ol>
              ) : <EmptyState icon="users" title="No contacts in this period" />}
            </div>
          </Card>
        </Reveal>

        <Reveal index={10}>
          <Card className="es-lift h-full">
            <CardTitle title="Top events" subtitle={`Where contacts were met · ${days} days`} />
            <div className="px-4 pb-4 pt-3">
              {loading ? <div className="space-y-3">{[0, 1, 2, 3].map((i) => <div key={i} className="space-y-1.5"><Shimmer className="h-2.5" style={{ width: `${70 - i * 12}%` }} /><Shimmer className="h-1.5" /></div>)}</div> : r?.events.length ? (
                <BarList items={r.events.map((e) => ({ key: e.event, label: e.event, value: e.count }))} onSelect={(k) => goContacts({ event: k })} />
              ) : <EmptyState icon="calendar" title="No event-tagged contacts" description="Contacts are tagged with the rep's active event." />}
            </div>
          </Card>
        </Reveal>
      </div>

      {/* Pipelines · sources · workspace (activity lives in the header's Activity menu) */}
      <div className="grid gap-3 lg:grid-cols-3">
        <Reveal index={11}>
          <Card className="es-lift h-full">
            <CardTitle title="AI pipelines" subtitle="Research → score → pitch → deck" right={<SwapLink onClick={() => navigate({ to: '/pipelines' })}>Open</SwapLink>} />
            <div className="flex items-center gap-4 px-4 pb-4 pt-3">
              {loading ? <Shimmer className="h-23 w-23 rounded-full" /> : (
                <Ring value={pipeTotal ? (d!.pipelines.complete / pipeTotal) : 0} color={d?.pipelines.failed ? 'var(--heat-warm)' : 'var(--success)'}>
                  <span className="es-num text-xl font-semibold text-foreground">{pipeTotal ? `${Math.round((d!.pipelines.complete / pipeTotal) * 100)}%` : '—'}</span>
                  <span className="es-eyebrow text-[9px] text-muted-foreground">success</span>
                </Ring>
              )}
              <dl className="flex-1 space-y-2 text-sm">
                {[['Running', d?.pipelines.running, 'var(--heat-cool)'], ['Finished', d?.pipelines.complete, 'var(--success)'], ['Failed', d?.pipelines.failed, 'var(--heat-hot)']].map(([l, v, c]) => (
                  <div key={l as string} className="flex items-center gap-2">
                    <span className={cx('h-2 w-2 rounded-[2px]', l === 'Running' && v ? 'animate-pulse' : '')} style={{ background: c as string }} />
                    <dt className="flex-1 text-muted-foreground">{l}</dt>
                    <dd className="es-num font-semibold text-foreground">{loading ? '—' : <AnimatedNumber value={v as number} duration={700} />}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </Card>
        </Reveal>

        <Reveal index={12}>
          <Card className="es-lift h-full">
            <CardTitle title="How leads came in" subtitle={`${days} days`} />
            <div className="px-4 pb-4 pt-3">
              {loading ? <div className="space-y-3">{[0, 1, 2].map((i) => <div key={i} className="space-y-1.5"><Shimmer className="h-2.5" style={{ width: `${60 - i * 14}%` }} /><Shimmer className="h-1.5" /></div>)}</div> : r?.sources.length ? (
                <BarList color="bg-heat-cool" items={r.sources.map((s) => ({ key: s.source, label: SOURCE_LABEL[s.source] || s.source, value: s.count }))} />
              ) : <p className="text-sm text-muted-foreground">No contacts in this period.</p>}
            </div>
          </Card>
        </Reveal>

        <Reveal index={13}>
          <div className="grid h-full grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border shadow-card">
            {[
              { label: 'Live announcements', value: d?.activeBroadcasts, to: '/broadcasts', icon: 'megaphone' as IconName },
              { label: 'Team files', value: d?.files, to: '/files', icon: 'file' as IconName },
              { label: 'Lists', value: d?.lists, to: '/lists', icon: 'list' as IconName },
              { label: 'Recently deleted', value: d?.recentlyDeleted, to: '/deleted', icon: 'trash' as IconName },
            ].map((x) => (
              <Link key={x.label} to={x.to} onMouseMove={spotMove} className="es-spot group bg-card px-3 py-2.5 transition-colors">
                <div className="flex items-center justify-between">
                  <Icon name={x.icon} size={15} className="text-muted-foreground transition-[color,transform] duration-300 group-hover:-rotate-6 group-hover:scale-110 group-hover:text-foreground" />
                  <Icon name="chevronRight" size={13} className="-translate-x-1 text-muted-foreground/0 transition group-hover:translate-x-0 group-hover:text-muted-foreground" />
                </div>
                <div className="es-num mt-1.5 text-xl font-semibold text-foreground">{loading ? '—' : <AnimatedNumber value={x.value ?? 0} />}</div>
                <div className="text-xs text-muted-foreground">{x.label}</div>
              </Link>
            ))}
          </div>
        </Reveal>
      </div>
    </div>
  );
}
