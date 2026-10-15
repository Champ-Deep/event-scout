import React, { useMemo, useState } from 'react';
import { useSuspenseQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, LayoutGroup, motion } from 'motion/react';
import { api } from '../lib/api';
import { q as queries } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { patchAll, removeContacts, removeFromArrays } from '../lib/optimistic';
import type { Contact, WorkspaceDuplicatePair } from '../lib/types';
import { Avatar } from '../ui/Avatar';
import { Icon, type IconName } from '../ui/icons';
import { MOTION } from '../ui/tokens';
import { AnimatedNumber } from '../ui/micro';
import { Button, Card, ErrorState, PageHeader, Skeleton, contactName, cx, errorMessage, relTime, useUi } from '../ui/kit';
import { contactLevel } from '../lib/levels';
import { DrawerRing, IconTile, Rise, TINT, type Tint } from '../ui/drawerKit';

type Reason = WorkspaceDuplicatePair['reasons'][number];
type Side = 'primary' | 'duplicate';
type Filter = 'all' | 'cross' | Reason;

const REASON: Record<Reason, { label: string; icon: IconName; tint: Tint }> = {
  email: { label: 'Same email', icon: 'mail', tint: 'violet' },
  phone: { label: 'Same phone', icon: 'phone', tint: 'emerald' },
  name: { label: 'Same name + company', icon: 'idCard', tint: 'sky' },
};

/** The details compared side by side, each with its own icon tile (as in the contact overview). */
const FIELDS: { key: keyof Contact; label: string; icon: IconName; tint: Tint; mono?: boolean }[] = [
  { key: 'email', label: 'Email', icon: 'mail', tint: 'violet' },
  { key: 'phone', label: 'Phone', icon: 'phone', tint: 'emerald', mono: true },
  { key: 'company', label: 'Company', icon: 'building', tint: 'sky' },
  { key: 'title', label: 'Designation', icon: 'briefcase', tint: 'violet' },
  { key: 'eventId', label: 'Event', icon: 'calendar', tint: 'amber' },
  { key: 'linkedin', label: 'LinkedIn', icon: 'link', tint: 'linkedin' },
];

const val = (c: Contact, k: keyof Contact) => String((c[k] as string | null | undefined) ?? '').trim();
const same = (a: string, b: string) => a.replace(/\s+/g, ' ').toLowerCase() === b.replace(/\s+/g, ' ').toLowerCase();

// Compare the way the server matches, so "+1 (555) 234-5678" = "+1 555 234 5678" and "Stripe Inc." = "Stripe"
const COMPANY_SUFFIXES = new Set(['inc', 'incorporated', 'llc', 'llp', 'lp', 'ltd', 'limited', 'pvt', 'private', 'plc', 'corp', 'corporation',
  'co', 'company', 'gmbh', 'ag', 'sa', 'bv', 'nv', 'pte', 'pty', 'srl', 'oy', 'ab', 'the']);
const words = (v: string) => v.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(Boolean);
const phones = (v: string) => new Set(v.split(/\s*[,;|\n]\s*|\s+\/\s+/)
  .map((p) => p.replace(/(?:\s*(?:ext|extn|extension|x)\.?\s*|\s*#\s*)\d{1,6}\s*$/i, '').replace(/\D/g, ''))
  .filter((d) => d.length >= 7).map((d) => d.slice(-10)));
const emails = (v: string) => new Set(v.toLowerCase().match(/[a-z0-9._%+'-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+/g) ?? []);
const overlap = (a: Set<string>, b: Set<string>) => [...a].some((x) => b.has(x));
function sameValue(key: keyof Contact, a: string, b: string): boolean {
  if (key === 'phone') return overlap(phones(a), phones(b));
  if (key === 'email') return overlap(emails(a), emails(b));
  if (key === 'company') return words(a).filter((w) => !COMPANY_SUFFIXES.has(w)).join(' ') === words(b).filter((w) => !COMPANY_SUFFIXES.has(w)).join(' ');
  if (key === 'linkedin') return same(a.replace(/^https?:\/\/(www\.)?/i, '').replace(/\/$/, ''), b.replace(/^https?:\/\/(www\.)?/i, '').replace(/\/$/, ''));
  return same(a, b);
}
const confColor = (n: number) => (n >= 95 ? 'var(--heat-hot)' : n >= 90 ? 'var(--heat-warm)' : 'var(--primary)');

/** Confidence as a ring that draws to the percentage. */
function MatchRing({ value, size = 52 }: { value: number; size?: number }) {
  const stroke = 5;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const color = confColor(value);
  return (
    <span className="relative inline-flex shrink-0 items-center justify-center" style={{ width: size, height: size }} title={`${value}% match`}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} style={{ stroke: 'var(--secondary)' }} />
        <motion.circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} strokeLinecap="round" style={{ stroke: color }}
          strokeDasharray={`${circ} ${circ}`} initial={{ strokeDashoffset: circ }} animate={{ strokeDashoffset: circ * (1 - value / 100) }}
          transition={{ duration: 1, ease: MOTION.ease, delay: 0.15 }} />
      </svg>
      <span className="absolute flex flex-col items-center leading-none">
        <span className={cx('es-num font-bold text-foreground', size < 48 ? 'text-[13px]' : 'text-[15px]')}><AnimatedNumber value={value} duration={900} /></span>
        <span className={cx('font-bold uppercase tracking-[0.06em] text-muted-foreground', size < 48 ? 'text-[7px]' : 'text-[8.5px]')}>match</span>
      </span>
    </span>
  );
}

/** One compared detail on a single line: icon · label · value · how it compares with the other contact.
 *  Both contacts list the same details in the same order, so each line reads straight across. */
function FieldRow({ f, mine, theirs, keep, i }: { f: (typeof FIELDS)[number]; mine: string; theirs: string; keep: boolean; i: number }) {
  const fill = keep && !mine && !!theirs; // the kept contact gets this from the other one
  const status = mine && theirs ? (sameValue(f.key, mine, theirs) ? 'same' : 'differs') : fill ? 'fill' : null;
  return (
    <motion.div initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.08 + i * 0.025 }}
      className={cx('flex min-w-0 items-center gap-2 rounded-lg px-1.5 py-[3px] transition-colors hover:bg-secondary/60',
        status === 'same' && 'bg-emerald-50/50 dark:bg-emerald-500/[0.06]', status === 'differs' && 'bg-amber-50/40 dark:bg-amber-500/[0.05]')}>
      <span className={cx('flex h-5 w-5 shrink-0 items-center justify-center rounded-md', TINT[f.tint], !mine && !fill && 'opacity-45 grayscale')}>
        <Icon name={f.icon} size={11} />
      </span>
      <span className="w-[74px] shrink-0 text-[11px] font-medium text-muted-foreground">{f.label}</span>
      <span className={cx('min-w-0 flex-1 break-words text-[12.5px] font-medium leading-5 [overflow-wrap:anywhere]', f.mono && 'font-mono tabular-nums',
        mine ? 'text-foreground' : fill ? 'italic text-brand-700 dark:text-brand-800' : 'font-normal italic text-muted-foreground')}>
        {mine || (fill ? theirs : 'Not added')}
      </span>
      {status ? (
        <motion.span initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.2 + i * 0.025 }}
          title={status === 'same' ? 'Same on both' : status === 'differs' ? 'Different on the other one' : 'Filled in from the other one when merged'}
          className={cx('inline-flex shrink-0 items-center gap-0.5 rounded-full px-1.5 text-[9.5px] font-bold leading-4',
            status === 'same' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300'
              : status === 'differs' ? 'bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-200'
                : 'bg-brand-50 text-brand-700 dark:bg-brand-100 dark:text-brand-800')}>
          <Icon name={status === 'same' ? 'check' : status === 'differs' ? 'alert' : 'plus'} size={9} strokeWidth={3} />
          {status === 'same' ? 'Same' : status === 'differs' ? 'Differs' : 'Filled in'}
        </motion.span>
      ) : null}
    </motion.div>
  );
}
/** One contact of the pair; click it to keep it. The kept one wears a green ring, the other is merged in. */
function ContactSide({ c, other, rep, keep, onPick, index }: { c: Contact; other: Contact; rep: string; keep: boolean; onPick: () => void; index: number }) {
  const lvl = contactLevel(c);
  return (
    <motion.button type="button" layout onClick={onPick} aria-pressed={keep} aria-label={`Keep ${contactName(c)}`}
      title={keep ? 'Kept: gets the other one’s notes, files and any missing details' : 'Click to keep this one instead (the other is merged into it)'}
      whileHover={{ y: keep ? 0 : -2 }} whileTap={{ scale: 0.99 }}
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.05 + index * 0.05 }}
      className={cx('relative flex min-w-0 flex-1 cursor-pointer flex-col overflow-hidden rounded-xl border bg-card text-left shadow-card transition-[border-color,box-shadow,opacity] duration-200',
        keep ? 'border-emerald-400 ring-2 ring-emerald-400/30 dark:border-emerald-500/60' : 'border-dashed opacity-90 hover:border-brand-300 hover:opacity-100')}>
      {/* heat strip, like the contact cards */}
      <span className="absolute inset-x-0 top-0 h-[3px]" style={{ background: lvl ? `var(--heat-${lvl.key})` : 'var(--secondary)' }} />

      {/* who · keep / merge-in tag · score */}
      <div className="flex items-center gap-2.5 px-2.5 pb-1 pt-2.5">
        <span className="shrink-0 rounded-full p-[2px]" style={{ background: lvl ? `var(--heat-${lvl.key})` : 'transparent', boxShadow: lvl ? undefined : 'inset 0 0 0 1.5px var(--dashed)' }}>
          <span className="block rounded-full bg-card p-px"><Avatar src={c.photoUrl || null} name={contactName(c)} size="sm" /></span>
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className="break-words text-sm font-semibold leading-5 text-foreground [overflow-wrap:anywhere]">{contactName(c)}</span>
            <AnimatePresence mode="wait" initial={false}>
              <motion.span key={keep ? 'keep' : 'merge'} initial={{ scale: 0.6, opacity: 0, y: -3 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.6, opacity: 0, y: 3 }}
                transition={{ type: 'spring', stiffness: 500, damping: 24 }}
                className={cx('inline-flex shrink-0 items-center gap-0.5 rounded-full px-1.5 py-px text-[9.5px] font-black uppercase tracking-[0.06em]',
                  keep ? 'bg-emerald-700 text-white' : 'bg-secondary text-muted-foreground')}>
                <Icon name={keep ? 'check' : 'merge'} size={10} strokeWidth={3} />{keep ? 'Keep' : 'Merge in'}
              </motion.span>
            </AnimatePresence>
          </div>
          <div className="flex min-w-0 items-center gap-1 truncate text-[11px] text-muted-foreground">
            <Icon name="user" size={10} className="shrink-0" /><span className="truncate">{rep || 'Unknown rep'}</span><span className="shrink-0">· {relTime(c.createdAt)}</span>
          </div>
        </div>
        <DrawerRing score={c.score} level={c.level} size={36} />
      </div>

      {/* details: one column, same order on both sides */}
      <div className="flex flex-1 flex-col gap-px px-1 pb-1.5">
        {FIELDS.map((f, i) => <FieldRow key={f.key} f={f} mine={val(c, f.key)} theirs={val(other, f.key)} keep={keep} i={i} />)}
      </div>
    </motion.button>
  );
}

function PairCard({ pair, index, keepSide, onKeep, onMerge, onDismiss }: {
  pair: WorkspaceDuplicatePair; index: number; keepSide: Side; onKeep: (s: Side) => void; onMerge: () => void; onDismiss: () => void;
}) {
  const keepLeft = keepSide === 'primary';
  const keepC = keepLeft ? pair.primary : pair.duplicate;
  const color = confColor(pair.confidence);
  return (
    <motion.div layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: 60, scale: 0.97, transition: { duration: MOTION.base } }}
      transition={{ duration: MOTION.base * 1.4, ease: MOTION.ease, delay: Math.min(index, 8) * 0.05 }}>
      <Card className="relative overflow-hidden rounded-2xl p-0">
        <span className="absolute inset-y-0 left-0 w-1" style={{ background: color }} aria-hidden />

        {/* why they look alike · actions */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b px-3 py-2 pl-4">
          <MatchRing value={pair.confidence} size={40} />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold leading-5 text-foreground">
              {contactName(pair.primary)}{!same(contactName(pair.primary), contactName(pair.duplicate)) ? <> &amp; {contactName(pair.duplicate)}</> : null}
            </div>
            <div className="flex flex-wrap items-center gap-1">
              {pair.reasons.map((r, ri) => (
                <motion.span key={r} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.12 + ri * 0.05 }}
                  className={cx('inline-flex items-center gap-1 rounded-full px-1.5 py-px text-[10.5px] font-semibold', TINT[REASON[r].tint])}>
                  <Icon name={REASON[r].icon} size={10} />{REASON[r].label}
                </motion.span>
              ))}
              {pair.crossRep ? (
                <motion.span initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.27 }}
                  className={cx('inline-flex items-center gap-1 rounded-full px-1.5 py-px text-[10.5px] font-semibold', TINT.amber)}>
                  <Icon name="users" size={10} />Two reps · {pair.owner || 'Unknown'} &amp; {pair.duplicateOwner || 'Unknown'}
                </motion.span>
              ) : (
                <span className="text-[10.5px] text-muted-foreground">· {pair.owner || 'Unknown rep'}'s contacts</span>
              )}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <Button size="sm" variant="ghost" icon="x" onClick={onDismiss}>Not a duplicate</Button>
            <motion.span whileHover={{ y: -1 }} whileTap={{ scale: 0.96 }}>
              <Button size="sm" variant="primary" icon="merge" onClick={onMerge}>Merge · keep {keepC.firstName || contactName(keepC)}</Button>
            </motion.span>
          </div>
        </div>

        {/* the two contacts: click one to keep it, or swap */}
        <LayoutGroup id={pair.id}>
          <div className="flex flex-col items-stretch gap-2 p-2 md:flex-row">
            <ContactSide c={pair.primary} other={pair.duplicate} rep={pair.owner} keep={keepLeft} onPick={() => onKeep('primary')} index={0} />

            {/* the flow: the merged-in one → the kept one; click to swap */}
            <motion.button type="button" onClick={() => onKeep(keepLeft ? 'duplicate' : 'primary')} aria-label="Swap which contact is kept"
              title="Swap which one is kept" whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
              className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center self-center rounded-full border bg-card text-muted-foreground shadow-card transition-colors hover:border-brand-300 hover:text-brand-700">
              <motion.span animate={{ rotate: keepLeft ? 180 : 0 }} transition={{ type: 'spring', stiffness: 300, damping: 18 }} className="flex max-md:rotate-90">
                <Icon name="chevronRight" size={16} strokeWidth={2.5} />
              </motion.span>
            </motion.button>

            <ContactSide c={pair.duplicate} other={pair.primary} rep={pair.crossRep ? pair.duplicateOwner || '' : pair.owner} keep={!keepLeft} onPick={() => onKeep('duplicate')} index={1} />
          </div>
        </LayoutGroup>
      </Card>
    </motion.div>
  );
}

export function DuplicatesComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const q = useSuspenseQuery(queries.duplicates());
  const pairs = q.data ?? [];
  // which contact of each pair to keep (default: the older one, the primary)
  const [keepOf, setKeepOf] = useState<Record<string, Side>>({});
  const [filter, setFilter] = useState<Filter>('all');

  const counts = useMemo(() => ({
    all: pairs.length,
    cross: pairs.filter((p) => p.crossRep).length,
    email: pairs.filter((p) => p.reasons.includes('email')).length,
    phone: pairs.filter((p) => p.reasons.includes('phone')).length,
    name: pairs.filter((p) => p.reasons.includes('name')).length,
  }), [pairs]);
  const shown = pairs.filter((p) => filter === 'all' || (filter === 'cross' ? p.crossRep : p.reasons.includes(filter)));

  // optimistic: the pair leaves the list at once (and a merged duplicate leaves the contact grid);
  // both come back if the server rejects
  const act = async (pair: WorkspaceDuplicatePair, kind: 'merge' | 'dismiss') => {
    const keepSide = keepOf[pair.id] ?? 'primary';
    const keep = keepSide === 'primary' ? pair.primary : pair.duplicate;
    const drop = keepSide === 'primary' ? pair.duplicate : pair.primary;
    const keepRep = (keepSide === 'primary' ? pair.owner : pair.duplicateOwner) || 'the rep';
    const dropRep = (keepSide === 'primary' ? (pair.crossRep ? pair.duplicateOwner : pair.owner) : pair.owner) || 'the rep';
    if (kind === 'merge') {
      const ok = await confirm({
        title: `Keep ${contactName(keep)} and merge the other into it?`,
        message: pair.crossRep ? (
          <>These were captured by <b>two different reps</b>. {keepRep}'s contact is kept (empty fields filled, notes and files combined, the higher score kept), and
            the copy is removed from <b>{dropRep}'s</b> contacts and app. It goes to their Recently deleted for 30 days, so this can be undone.</>
        ) : (
          <>Empty fields on the kept contact are filled from the other one, notes and files are combined, and the higher score is kept. The other one goes to {dropRep}'s Recently deleted for 30 days, so this can be undone.</>
        ),
        confirmLabel: 'Merge',
      });
      if (!ok) return;
    }
    const rollback = await patchAll(
      removeFromArrays<WorkspaceDuplicatePair>(qc, ['admin', 'duplicates'], [pair.id]),
      ...(kind === 'merge' ? [removeContacts(qc, [drop.id])] : []),
    );
    try {
      if (kind === 'merge') {
        const r = await api.merge(keep.id, drop.id);
        toast('success', 'Merged', r.message);
      } else {
        await api.notDuplicate(pair.primary.id, pair.duplicate.id);
        toast('success', 'Marked as different people', "They won't be suggested again.");
      }
    } catch (e) {
      rollback();
      toast('error', kind === 'merge' ? "Couldn't merge — restored" : "Couldn't save — restored", errorMessage(e));
    } finally {
      // either way the pair leaves this page and both contacts' Duplicates tab
      void applyChange(qc, kind === 'merge' ? ['contact_merged', 'duplicate_dismissed'] : 'duplicate_dismissed');
    }
  };

  const tiles: { key: Filter; label: string; icon: IconName; tint: Tint }[] = [
    { key: 'all', label: 'Possible duplicates', icon: 'duplicate', tint: 'amber' },
    { key: 'cross', label: 'Across two reps', icon: 'users', tint: 'rose' },
    { key: 'email', label: 'Same email', icon: 'mail', tint: 'violet' },
    { key: 'phone', label: 'Same phone', icon: 'phone', tint: 'emerald' },
    { key: 'name', label: 'Same name + company', icon: 'idCard', tint: 'sky' },
  ];

  return (
    <>
      <PageHeader title="Duplicates" description="Contacts that look like the same person — within one rep's contacts or captured by two different reps. Pick the one to keep, then merge."
        actions={<Button icon="refresh" onClick={() => q.refetch()} loading={q.isFetching && !q.isLoading}>Re-check</Button>} />

      {q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : q.isLoading ? (
        <div className="space-y-3">{[0, 1].map((i) => <Skeleton key={i} className="h-72 w-full rounded-2xl" />)}</div>
      ) : !pairs.length ? (
        <Rise>
          <Card className="flex flex-col items-center gap-2 rounded-2xl px-6 py-12 text-center">
            <motion.span initial={{ scale: 0.4, rotate: -20, opacity: 0 }} animate={{ scale: 1, rotate: 0, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 16 }}
              className={cx('mb-1 flex h-14 w-14 items-center justify-center rounded-2xl', TINT.emerald)}>
              <Icon name="check" size={26} strokeWidth={2.5} />
            </motion.span>
            <p className="text-base font-semibold text-foreground">No duplicates found</p>
            <p className="max-w-sm text-sm text-muted-foreground">Every contact looks unique across all reps. New captures and edits are checked automatically.</p>
          </Card>
        </Rise>
      ) : (
        <div className="space-y-3">
          {/* at a glance — tap a tile to show only those pairs */}
          <Rise className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
            {tiles.map((t, k) => {
              const on = filter === t.key;
              const n = counts[t.key];
              return (
                <motion.button key={t.key} type="button" onClick={() => setFilter(on && t.key !== 'all' ? 'all' : t.key)} aria-pressed={on}
                  whileHover={{ y: -3 }} whileTap={{ scale: 0.96 }}
                  initial={{ opacity: 0, y: 10, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.05 + k * 0.05 }}
                  className={cx('group relative flex cursor-pointer items-center gap-2.5 overflow-hidden rounded-xl border bg-card px-2.5 py-2 text-left shadow-card transition-colors',
                    on ? 'border-brand-400 ring-2 ring-brand-400/25' : 'hover:border-brand-200', !n && t.key !== 'all' && 'opacity-60')}>
                  <motion.span initial={{ scale: 0, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 15, delay: 0.15 + k * 0.05 }}>
                    <IconTile icon={t.icon} tint={t.tint} size="sm" />
                  </motion.span>
                  <span className="min-w-0">
                    <span className="es-num block text-base font-bold leading-5 text-foreground"><AnimatedNumber value={n} duration={700} /></span>
                    <span className="block truncate text-[11px] font-medium text-muted-foreground">{t.label}</span>
                  </span>
                </motion.button>
              );
            })}
          </Rise>

          <AnimatePresence mode="popLayout" initial={true}>
            {shown.map((pair, i) => (
              <PairCard key={pair.id} pair={pair} index={i} keepSide={keepOf[pair.id] ?? 'primary'}
                onKeep={(s) => setKeepOf((m) => ({ ...m, [pair.id]: s }))}
                onMerge={() => act(pair, 'merge')} onDismiss={() => act(pair, 'dismiss')} />
            ))}
          </AnimatePresence>
          {!shown.length ? (
            <Card className="rounded-2xl px-6 py-8 text-center text-sm text-muted-foreground">
              No pairs of this kind. <button type="button" onClick={() => setFilter('all')} className="font-semibold text-primary hover:underline cursor-pointer">Show all</button>
            </Card>
          ) : null}
        </div>
      )}
    </>
  );
}
