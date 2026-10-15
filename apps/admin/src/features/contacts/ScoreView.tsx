// A contact's AI score: the big ring + level, how it adds up (five dimensions out of 20), why, and
// the next steps. ScoreView draws one score (the contact drawer's Score tab and the Recently deleted
// drawer); ScoreTab is the drawer's tab (admin score / rep's score switch, re-score with AI).
//
// In the console the contact's own score* fields ARE the admin score (the server's console view), so
// the drawer header and this tab both read it from the one contact query — adminScoreOf(c).
import React from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { motion } from 'motion/react';
import { api } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { levelFrom, levelOf } from '../../lib/levels';
import type { Contact } from '../../lib/types';
import type { ScoreMeta } from '@es/types';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { AnimatedNumber } from '../../ui/micro';
import { Button, ErrorState, Skeleton, cx, errorMessage, relTime, useUi } from '../../ui/kit';
import { CopyAction, DrawerRing, EmptyPanel, Fade, IconTile, RadarPulse, Rise, SectionTitle, TINT, useCopy } from '../../ui/drawerKit';

export const DIMENSIONS: [string, string, IconName, string][] = [
  ['role_fit', 'Role fit', 'briefcase', 'Does their job match who you sell to?'],
  ['company_fit', 'Company fit', 'building', 'Is the company your kind of customer?'],
  ['need_fit', 'Need fit', 'bolt', 'Do they have a problem your products solve?'],
  ['goal_fit', 'Goal fit', 'flame', 'Does it serve this event’s goals?'],
  ['intent', 'Intent', 'send', 'How ready they seemed to talk business'],
];

/** One score as the views draw it. `level`: the server's level for it, when sent. */
export interface ScoreData {
  score: number;
  level?: string | null | undefined;
  reasoning: string | null | undefined;
  breakdown: Record<string, number> | null | undefined;
  actions: string[] | null | undefined;
  meta: ScoreMeta | null | undefined;
}

/** The admin console's own score of a contact (its score* fields in the console), or null if not scored here. */
export const adminScoreOf = (c: Contact): ScoreData | null => (c.score == null ? null
  : { score: c.score, level: c.level, reasoning: c.scoreReasoning, breakdown: c.scoreBreakdown, actions: c.scoreActions, meta: c.scoreMeta });

/** The rep's own score from the app (read only here), or null if they haven't scored it. */
export const repScoreOf = (c: Contact): ScoreData | null => (c.repScore == null ? null
  : { score: c.repScore, level: c.repLevel, reasoning: c.repScoreReasoning, breakdown: c.repScoreBreakdown, actions: c.repScoreActions, meta: c.repScoreMeta });

/** Light list rows (?lite=1) leave the score write-up out: the full contact is still loading. */
export const hasScoreDetail = (c: Contact) => c.scoreMeta !== undefined;

/**
 * One score. variant 'live' (the contact drawer) animates every part and explains each dimension;
 * 'snapshot' (a deleted contact) is the same panel, stiller and narrower.
 */
export function ScoreView({ sv, title, icon, action, variant = 'live', onCopy }: {
  sv: ScoreData; title: string; icon: IconName; action?: React.ReactNode; variant?: 'live' | 'snapshot'; onCopy: (what: string, value: string) => void;
}) {
  const live = variant === 'live';
  const lvl = levelFrom(sv.score, sv.level);
  const heat = lvl ? lvl.color : 'var(--primary)';
  const meta = (
    <>
      {sv.meta?.runs ? (
        <span className="inline-flex items-center gap-1"><Icon name="sparkles" size={12} className="text-brand-600" />Middle of {sv.meta.runs} AI answers{sv.meta.spread != null ? ` (±${sv.meta.spread})` : ''}</span>
      ) : <span className="inline-flex items-center gap-1"><Icon name="sparkles" size={12} className="text-brand-600" />AI score</span>}
      {sv.meta?.scoredAt ? <span className="inline-flex items-center gap-1"><Icon name="clock" size={12} />{relTime(sv.meta.scoredAt)}</span> : null}
      {sv.meta?.model ? <span className="inline-flex items-center gap-1"><Icon name="server" size={12} />{sv.meta.model}</span> : null}
    </>
  );
  return (
    <div className="space-y-5">
      {/* one row: the score (≈1/3) beside how it adds up (≈2/3); stacks on narrow screens */}
      <div className={cx('grid items-stretch gap-4', sv.breakdown && (live ? 'md:grid-cols-[minmax(240px,1fr)_2fr]' : 'md:grid-cols-[minmax(220px,1fr)_2fr]'))}>
      {/* the score: big ring that draws in, level chip, how it was worked out */}
      <Rise i={0} className="flex flex-col">
        <SectionTitle icon={icon} right={<span className="text-xs font-medium text-muted-foreground">out of 100</span>}>{title}</SectionTitle>
        <div className="relative flex flex-1 flex-col items-center justify-center gap-3 overflow-hidden rounded-2xl border p-5 text-center shadow-card"
          style={{ background: `color-mix(in oklab, ${heat} 6%, var(--card))` }}>
          <span className="relative">
            {lvl?.key === 'hot' ? <RadarPulse color="var(--heat-hot)" /> : null}
            <DrawerRing key={`${title}:${sv.score}`} score={sv.score} level={sv.level} size={112} />
          </span>
          {live ? (
            <div className="min-w-0">
              <div className="flex flex-wrap items-center justify-center gap-2">
                {lvl ? (
                  <motion.span initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.35 }}
                    className={cx('inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.05em]', lvl.badge)}>
                    <motion.span animate={lvl.key === 'hot' ? { rotate: [0, -12, 12, 0] } : {}} transition={{ duration: 1.2, repeat: Infinity, repeatDelay: 1.5 }} className="flex">
                      <Icon name="flame" size={12} />
                    </motion.span>{lvl.label} lead
                  </motion.span>
                ) : null}
              </div>
              <div className="mt-2.5 flex flex-col items-center gap-1 text-xs text-muted-foreground">{meta}</div>
            </div>
          ) : (
            <>
              {lvl ? (
                <span className={cx('inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.05em]', lvl.badge)}>
                  <Icon name="flame" size={12} />{lvl.label} lead
                </span>
              ) : null}
              <div className="flex flex-col items-center gap-1 text-xs text-muted-foreground">{meta}</div>
            </>
          )}
          {action}
        </div>
      </Rise>

      {/* breakdown: one row per dimension, icon + bar that fills + number that counts up */}
      {sv.breakdown ? (
        <Rise i={1} className="flex min-w-0 flex-col">
          <SectionTitle icon="sliders" right={<span className="text-xs font-medium text-muted-foreground">each out of 20</span>}>How the score adds up</SectionTitle>
          <div className="flex flex-1 flex-col justify-around gap-1 rounded-2xl border bg-card p-2 shadow-card">
            {DIMENSIONS.map(([k, label, dimIcon, help], i) => {
              const v = Number(sv.breakdown?.[k] ?? 0);
              const dl = levelOf(v * 5);
              const tile = <IconTile icon={dimIcon} tint={dl?.tint ?? 'slate'} />;
              return (
                <motion.div key={k} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.15 + i * 0.05 }}
                  className="group flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-secondary/60">
                  {live ? (
                    <motion.span initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.2 + i * 0.05 }}>{tile}</motion.span>
                  ) : tile}
                  <div className="min-w-0 flex-1">
                    <div className="mb-1.5 flex items-baseline justify-between gap-2">
                      <span className="text-sm font-semibold text-foreground">{label}{live ? <span className="ml-2 hidden text-xs font-normal text-muted-foreground lg:inline">{help}</span> : null}</span>
                      <span className="es-num shrink-0 text-sm font-bold text-foreground"><AnimatedNumber value={v} duration={800} /><span className="text-xs font-medium text-muted-foreground">/20</span></span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-secondary">
                      <motion.div className={cx('h-full origin-left rounded-full', dl?.bar || 'bg-border')} initial={{ scaleX: 0 }} animate={{ scaleX: v / 20 }}
                        transition={{ duration: MOTION.slow * 0.8, ease: MOTION.ease, delay: 0.25 + i * 0.07 }} />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </Rise>
      ) : null}
      </div>

      {sv.reasoning ? (
        <Rise i={2}>
          <SectionTitle icon="info" right={<CopyAction label="Reasoning" value={sv.reasoning} onCopy={onCopy} />}>Why this score</SectionTitle>
          <div className="relative rounded-2xl border bg-card p-4 pl-11 shadow-card">
            <span className="absolute left-3.5 top-3.5 font-display text-3xl leading-none text-brand-300" aria-hidden>“</span>
            <p className="text-sm leading-relaxed text-body">{sv.reasoning}</p>
          </div>
        </Rise>
      ) : null}

      {sv.actions?.length ? (
        <Rise i={3}>
          <SectionTitle icon="check" right={<span className="text-xs font-medium text-muted-foreground">{sv.actions.length} suggested</span>}>Next steps</SectionTitle>
          <ol className="space-y-2">{sv.actions.map((a, i) => (live ? (
            <motion.li key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.3 + i * 0.06 }}
              whileHover={{ x: 3 }}
              className="group flex items-start gap-3 rounded-xl border bg-card px-3.5 py-3 text-sm text-body shadow-card transition-colors hover:border-emerald-200 dark:hover:border-emerald-500/30">
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 18, delay: 0.4 + i * 0.06 }}
                className={cx('flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold tabular-nums transition-transform group-hover:scale-110', TINT.emerald)}>{i + 1}</motion.span>
              <span className="pt-0.5">{a}</span>
            </motion.li>
          ) : (
            <li key={i} className="flex items-start gap-3 rounded-xl border bg-card px-3.5 py-3 text-sm text-body shadow-card">
              <span className={cx('flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-bold tabular-nums', TINT.emerald)}>{i + 1}</span>
              <span className="pt-0.5">{a}</span>
            </li>
          )))}</ol>
        </Rise>
      ) : null}
    </div>
  );
}

/**
 * Score the contact with AI as the admin (POST /api/contact/{id}/score[?force]). Saved as the admin
 * console's own score; the rep's score in the app is never changed from here.
 */
export function useRescore(id: string | undefined, onScored: () => void) {
  const qc = useQueryClient();
  const { toast } = useUi();
  return useMutation({
    mutationFn: (force: boolean) => api.scoreContact(id!, force),
    onSuccess: (r) => {
      const delta = r.previousScore != null && !r.unchanged ? r.score - r.previousScore : 0;
      toast('success', r.unchanged ? `Nothing changed — admin score stays at ${r.score}` : `Admin score ${r.score} · ${levelOf(r.score)?.label}`,
        r.unchanged ? 'Use "Score again" to ask the AI anyway.'
          : `${delta ? `${delta > 0 ? '+' : ''}${delta} from ${r.previousScore}. ` : ''}The rep's score is unchanged.`);
      // the console ranks by the admin score: this contact (header and Score tab), the grid, lists, counts
      void applyChange(qc, 'admin_score_updated');
      onScored();
    },
    onError: (e) => toast('error', "Couldn't score this contact", errorMessage(e)),
  });
}

export type ScoreSide = 'admin' | 'rep';

/** The drawer's Score tab: the admin score (default once there is one) or the rep's, side by side in a switch. */
export function ScoreTab({ c, ownerName, view: picked, onView, detail, rescore }: {
  c: Contact;
  ownerName?: string | undefined;
  /** the side picked in the switch (null = the default) */
  view: ScoreSide | null;
  onView: (v: ScoreSide) => void;
  /** the full contact's load (a light list row has the score but not its write-up yet) */
  detail: { fetching: boolean; error: unknown; retry: () => void };
  rescore: { run: (force: boolean) => void; pending: boolean };
}) {
  const copy = useCopy();
  // two separate scores: the admin console's own, and the rep's from the app (read-only here)
  const mine = adminScoreOf(c);
  const rep = repScoreOf(c);
  const view: ScoreSide = picked ?? (mine || !rep ? 'admin' : 'rep');
  const sv = view === 'admin' ? mine : rep;
  const repLabel = `${ownerName ? ownerName.split(' ')[0] : 'Rep'}'s score`;
  const waiting = !hasScoreDetail(c);
  return (
    <div className="space-y-4">
      <Rise i={0} className="flex flex-wrap items-center justify-between gap-3">
        <div role="tablist" aria-label="Which score" className="inline-flex rounded-2xl bg-secondary p-1">
          {([['admin', 'Admin score', 'shield', mine?.score], ['rep', repLabel, 'smartphone', rep?.score]] as const).map(([v, label, icon, value]) => (
            <button key={v} type="button" role="tab" aria-selected={view === v} onClick={() => onView(v)}
              className={cx('relative cursor-pointer rounded-xl px-3 py-1.5 text-xs font-semibold transition-colors', view === v ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
              {view === v ? <motion.span layoutId="es-score-view" className="absolute inset-0 rounded-xl bg-card shadow-card" transition={{ type: 'spring', stiffness: 500, damping: 34 }} /> : null}
              <span className="relative flex items-center gap-1.5">
                <Icon name={icon} size={13} className={view === v ? 'text-primary' : ''} />{label}
                {value != null ? <span className="es-num rounded-full bg-brand-50 px-1.5 text-[10px] font-bold text-brand-700 dark:bg-brand-100 dark:text-brand-800">{value}</span> : null}
              </span>
            </button>
          ))}
        </div>
        <Fade k={view}>
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <Icon name={view === 'admin' ? 'shield' : 'eye'} size={12} className={view === 'admin' ? 'text-emerald-600' : 'text-sky-600'} />
            {view === 'admin' ? "Admin only · the rep's score in the app never changes" : 'Set in the app · read-only here'}
          </span>
        </Fade>
      </Rise>
      {sv && waiting && detail.fetching ? <Skeleton className="h-56 w-full rounded-2xl" />
        // a failed load must not read as "No admin score yet"
        : sv && waiting && detail.error ? <ErrorState error={detail.error} onRetry={detail.retry} />
        : !sv ? (
          view === 'admin' ? (
            <EmptyPanel icon="shield" tint="violet" title="No admin score yet"
              action={<Button variant="primary" icon="sparkles" loading={rescore.pending} onClick={() => rescore.run(false)}>Score with AI</Button>}>
              Scores the lead against your admin sales profile — what you sell, how you sell and who you target. It's saved for the admin console only, so {ownerName || 'the rep'}'s score in the app stays exactly as it is.
            </EmptyPanel>
          ) : (
            <EmptyPanel icon="smartphone" tint="sky" title="Not scored in the app">
              {ownerName || 'The rep'} hasn't scored this contact yet. Their score shows here once they do.
            </EmptyPanel>
          )
        ) : (
          <ScoreView sv={sv} title={view === 'admin' ? 'Admin AI score' : repLabel} icon={view === 'admin' ? 'shield' : 'smartphone'} onCopy={copy}
            action={view === 'admin' ? (
              <motion.span whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.94 }} className="mt-1 inline-flex">
                <Button size="sm" icon="refresh" loading={rescore.pending} onClick={() => rescore.run(true)}>Score again</Button>
              </motion.span>
            ) : undefined} />
        )}
    </div>
  );
}
