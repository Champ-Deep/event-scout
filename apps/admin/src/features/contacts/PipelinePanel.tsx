// The contact drawer's Pipeline tab: GET/POST /api/contact/{id}/pipeline[/run|/retry]. Owns the run
// (polled every 2 s while it runs), starting / retrying it, and the refresh when it finishes.
import React, { useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, LayoutGroup, motion } from 'motion/react';
import { api } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { q } from '../../lib/queries';
import type { Contact } from '../../lib/types';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { AnimatedNumber } from '../../ui/micro';
import { Button, ErrorState, Notice, Skeleton, Spinner, contactName, cx, errorMessage, relTime, useUi } from '../../ui/kit';
import { CopyAction, EmptyPanel, Fade, IconTile, RadarPulse, Rise, SectionTitle, TINT, useCopy } from '../../ui/drawerKit';
import { PIPELINE_STEPS, stepStates, type StepState } from './pipelineSteps';

/** Each step's own "working" motion while it runs (magnifier searches, flame flickers, mail flies…). */
export const STEP_MOTION: Record<string, { animate: Record<string, number[]>; transition: Record<string, unknown> }> = {
  researching: { animate: { x: [0, 3, 0, -3, 0], y: [0, -2, 0, -2, 0] }, transition: { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } },
  scoring: { animate: { scale: [1, 1.2, 0.94, 1.12, 1], rotate: [0, -7, 5, -3, 0] }, transition: { duration: 1.1, repeat: Infinity, ease: 'easeInOut' } },
  pitching: { animate: { x: [0, 5, -2, 0], y: [0, -3, 0, 0], rotate: [0, -10, 0, 0] }, transition: { duration: 1.3, repeat: Infinity, ease: 'easeInOut' } },
  generating_deck: { animate: { y: [0, -4, 0], scale: [1, 1.1, 1] }, transition: { duration: 1.2, repeat: Infinity, ease: 'easeInOut' } },
  attaching: { animate: { rotate: [0, 20, -16, 0] }, transition: { duration: 1.4, repeat: Infinity, ease: 'easeInOut' } },
};

/**
 * One step's node. It keeps the same layoutId in every layout, so when the pipeline moves from
 * "running" (big, centred) to "finished" (compact strip) each node glides to its new place.
 */
function StepNode({ step, state, size, delay = 0 }: { step: (typeof PIPELINE_STEPS)[number]; state: StepState; size: 'lg' | 'sm'; delay?: number }) {
  const lg = size === 'lg';
  const m = STEP_MOTION[step.key];
  return (
    <motion.span layoutId={`es-pl-node-${step.key}`} transition={{ type: 'spring', stiffness: 260, damping: 28 }}
      className={cx('relative flex shrink-0 items-center justify-center rounded-full ring-4 ring-card', lg ? 'h-14 w-14' : 'h-8 w-8',
        state === 'done' ? 'bg-emerald-500 text-white' : state === 'now' ? 'bg-sky-500 text-white' : state === 'failed' ? 'bg-rose-500 text-white' : 'bg-secondary text-muted-foreground')}>
      {state === 'now' ? (
        <>
          <RadarPulse color="#0ea5e9" />
          <motion.span aria-hidden className="pointer-events-none absolute -inset-1.5 rounded-full border-2 border-dashed border-sky-400/80"
            animate={{ rotate: 360 }} transition={{ duration: 4, repeat: Infinity, ease: 'linear' }} />
        </>
      ) : null}
      {state === 'done' ? (
        <motion.span key="burst" aria-hidden className="pointer-events-none absolute inset-0 rounded-full ring-2 ring-emerald-400"
          initial={{ scale: 1, opacity: 0.9 }} animate={{ scale: 1.9, opacity: 0 }} transition={{ duration: 0.7, ease: 'easeOut', delay }} />
      ) : null}
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={state} className="flex"
          initial={{ scale: 0, rotate: -90 }} animate={state === 'failed' ? { scale: 1, rotate: 0, x: [0, -3, 3, -2, 2, 0] } : { scale: 1, rotate: 0 }}
          exit={{ scale: 0, rotate: 90 }} transition={{ type: 'spring', stiffness: 480, damping: 20, delay }}>
          {state === 'now' && m ? (
            <motion.span className="flex" animate={m.animate} transition={m.transition}><Icon name={step.icon} size={lg ? 22 : 14} /></motion.span>
          ) : (
            <Icon name={state === 'done' ? 'check' : state === 'failed' ? 'x' : step.icon} size={lg ? 22 : 14} {...(state === 'done' || state === 'failed' ? { strokeWidth: 3 } : {})} />
          )}
        </motion.span>
      </AnimatePresence>
    </motion.span>
  );
}

/** The line between two steps: fills green when the left step is done, shimmers while it runs. */
function StepLink({ from, size }: { from: StepState; size: 'lg' | 'sm' }) {
  return (
    <span className={cx('relative overflow-hidden rounded-full bg-secondary', size === 'lg' ? 'mx-1 mt-[26px] h-1 flex-1' : 'h-0.5 w-3 shrink-0 sm:w-5')}>
      <motion.span className="absolute inset-0 origin-left rounded-full bg-emerald-500" initial={false} animate={{ scaleX: from === 'done' ? 1 : 0 }} transition={{ duration: 0.5, ease: MOTION.ease }} />
      {from === 'now' ? (
        <motion.span className="absolute inset-y-0 w-1/2 rounded-full bg-gradient-to-r from-transparent via-sky-400 to-transparent"
          animate={{ x: ['-100%', '220%'] }} transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }} />
      ) : null}
    </span>
  );
}

const STEP_NOTE: Record<StepState, string> = { done: 'Done', now: 'Working on it…', failed: 'Failed', wait: 'Waiting' };

/** Vertical step timeline (finished / failed): small nodes stacked, a line between them that fills green. */
export function StepTrackVertical({ states, finished = true }: { states: StepState[]; finished?: boolean }) {
  return (
    <ol>
      {PIPELINE_STEPS.map((s, i) => (
        <li key={s.key} className={cx('relative flex items-start gap-3', i < PIPELINE_STEPS.length - 1 && 'pb-3.5')}>
          {i < PIPELINE_STEPS.length - 1 ? (
            <span className="absolute bottom-0 left-[15px] top-9 w-0.5 overflow-hidden rounded-full bg-secondary">
              <motion.span className="block h-full w-full origin-top bg-emerald-500" initial={{ scaleY: 0 }} animate={{ scaleY: states[i] === 'done' ? 1 : 0 }}
                transition={{ duration: 0.45, ease: MOTION.ease, delay: 0.3 + i * 0.1 }} />
            </span>
          ) : null}
          <StepNode step={s} state={states[i]!} size="sm" delay={0.15 + i * 0.08} />
          <span className="min-w-0 pt-0.5">
            <motion.span layoutId={`es-pl-label-${s.key}`} className={cx('block text-sm font-semibold', states[i] === 'wait' ? 'text-muted-foreground' : states[i] === 'failed' ? 'text-rose-600' : 'text-foreground')}>{s.label}</motion.span>
            <Fade k={states[i]!}>
              <span className={cx('block text-[11px]', states[i] === 'done' ? 'text-emerald-700 dark:text-emerald-400' : states[i] === 'failed' ? 'text-rose-600'
                : states[i] === 'now' ? 'font-medium text-sky-600 dark:text-sky-400' : 'text-muted-foreground')}>
                {finished && states[i] === 'wait' && (s.key === 'generating_deck' || s.key === 'attaching') ? 'Skipped' : STEP_NOTE[states[i]!]}
              </span>
            </Fade>
          </span>
        </li>
      ))}
    </ol>
  );
}

/** Compact step strip (finished / failed / not run): small nodes in a row with their labels. */
function StepTrackCompact({ states }: { states: StepState[] }) {
  return (
    <ol className="flex flex-wrap items-center gap-y-2">
      {PIPELINE_STEPS.map((s, i) => (
        <React.Fragment key={s.key}>
          <li className="flex items-center gap-1.5" title={`${s.label}: ${STEP_NOTE[states[i]!]}`}>
            <StepNode step={s} state={states[i]!} size="sm" delay={i * 0.06} />
            <motion.span layoutId={`es-pl-label-${s.key}`} className={cx('text-xs font-semibold', states[i] === 'wait' ? 'text-muted-foreground' : states[i] === 'failed' ? 'text-rose-600' : 'text-foreground')}>{s.label}</motion.span>
          </li>
          {i < PIPELINE_STEPS.length - 1 ? <span className="mx-1.5"><StepLink from={states[i]!} size="sm" /></span> : null}
        </React.Fragment>
      ))}
    </ol>
  );
}

export function PipelinePanel({ c, ownerName }: { c: Contact; ownerName?: string | undefined }) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const copy = useCopy();
  const pipeline = useQuery({ ...q.pipelineFor(c.id), refetchInterval: (s) => (s.state.data?.running ? 2000 : false) });
  const runPipeline = useMutation({
    mutationFn: (retry: boolean) => (retry ? api.retryPipeline(c.id) : api.runPipeline(c.id)),
    // the run, this contact's tab and the AI pipelines page pick it up
    onSuccess: (r) => { toast(r.status === 'already_running' ? 'info' : 'success', r.message); void applyChange(qc, 'pipeline_started'); },
    onError: (e) => toast('error', "Couldn't start the pipeline", errorMessage(e)),
  });
  // a run finished while this tab watched it → the contact (score, notes, files) changed. Only on the
  // running → done transition: opening the tab on a finished run must not refresh everything.
  const wasRunning = useRef<boolean | undefined>(undefined);
  useEffect(() => {
    const running = pipeline.data?.running;
    if (wasRunning.current === true && running === false) void applyChange(qc, 'pipeline_finished');
    if (running !== undefined) wasRunning.current = running;
  }, [pipeline.data?.running]);

  if (pipeline.isLoading) return <Skeleton className="h-40 w-full rounded-2xl" />;
  if (pipeline.error) return <ErrorState error={pipeline.error} onRetry={() => pipeline.refetch()} />;
  const p = pipeline.data;
  // the layout follows the run: not run → a hero; running → a big live step track;
  // finished / failed → the steps shrink into a strip at the top and the results take the page
  const phase: 'idle' | 'running' | 'done' = !p || p.status === 'none' ? 'idle' : p.running ? 'running' : 'done';
  const states = p && phase !== 'idle' ? stepStates(p) : PIPELINE_STEPS.map((): StepState => 'wait');
  const doneCount = states.filter((s) => s === 'done').length;
  const failed = p?.status === 'failed';
  const aiOff = p?.settings ? !p.settings.aiConfigured : false;
  const runButton = (
    <motion.span whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.95 }} className="inline-flex">
      <Button variant="primary" icon={failed ? 'refresh' : 'play'} loading={runPipeline.isPending} disabled={aiOff}
        onClick={() => runPipeline.mutate(failed)}>{failed ? 'Retry' : phase === 'idle' ? 'Run pipeline' : 'Run again'}</Button>
    </motion.span>
  );
  const adminNote = (
    <p className="flex items-start gap-1.5 text-[11px] leading-4 text-muted-foreground">
      <Icon name="shield" size={12} className="mt-0.5 shrink-0 text-emerald-600" />
      Admin run: results stay here. {ownerName || 'The rep'}'s score, notes and links in the app are not changed.
    </p>
  );
  const hasResults = !!(p && (p.researchSummary || p.pitchSubject || p.pitchEmail || p.slides?.length || p.gammaDeckUrl));
  // a shimmering placeholder for the output a running step is about to produce
  const pending = (icon: IconName, label: string) => (
    <div className="relative overflow-hidden rounded-2xl border border-dashed bg-card/60 p-4">
      <motion.span className="pointer-events-none absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-sky-100/70 to-transparent dark:via-sky-400/10"
        animate={{ x: ['-100%', '400%'] }} transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }} />
      <div className="relative flex items-center gap-2 text-sm font-medium text-muted-foreground"><Spinner size={14} className="text-sky-500" /><Icon name={icon} size={14} />{label}</div>
      <div className="relative mt-3 space-y-2"><Skeleton className="h-3 w-11/12" /><Skeleton className="h-3 w-4/5" /><Skeleton className="h-3 w-2/3" /></div>
    </div>
  );
  return (
  <LayoutGroup id={`es-pl-${c.id}`}>
  <div className="space-y-5">
    {aiOff ? (
      <Rise i={0}><Notice tone="amber" icon="alert">The AI isn't set up on the server (<code className="font-mono text-xs">OPENROUTER_API_KEY</code>), so the pipeline can't run.</Notice></Rise>
    ) : null}

    <AnimatePresence initial={false}>
      {phase === 'idle' ? (
        /* not run yet: a hero with the five steps waiting in a row */
        <motion.div key="idle" layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease }}
          className="flex flex-col items-center gap-4 rounded-3xl border bg-card px-6 py-8 text-center shadow-card"
          style={{ background: 'color-mix(in oklab, var(--primary) 5%, var(--card))' }}>
          <motion.span animate={{ y: [0, -5, 0], rotate: [0, -4, 4, 0] }} transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
            className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-card"><Icon name="sparkles" size={24} /></motion.span>
          <div>
            <h3 className="font-display text-lg font-bold text-foreground">Let the AI do the homework</h3>
            <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">One run researches {contactName(c)}, scores them, writes a pitch email and drafts a deck.</p>
          </div>
          <StepTrackCompact states={states} />
          {runButton}
          {adminNote}
        </motion.div>
      ) : null}
    </AnimatePresence>

    {/* running / finished / failed: one slim status + step timeline column on the left (pinned
        inside its own column only, so it never covers anything); results get the wider column */}
    <div className={cx(phase !== 'idle' && 'grid items-start gap-5 md:grid-cols-[minmax(240px,1fr)_2.2fr]')}>
    {phase !== 'idle' ? (
      // layout="size": it's sticky, so its position changes with every scroll; animating
      // position too made it slide on each 2 s refetch. layoutScroll: it scrolls its own step track
      <motion.aside layout="size" layoutScroll initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease }}
        className="es-scroll relative space-y-4 overflow-hidden rounded-2xl border bg-card p-4 shadow-card md:sticky md:top-[76px] md:max-h-[calc(100cqh-96px)] md:overflow-y-auto">
        {/* header: changes from live (spinning, step x of 5, progress) to the result */}
        <AnimatePresence mode="wait" initial={false}>
          {phase === 'running' ? (
            <motion.div key="live" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: MOTION.base }} className="space-y-3">
              <div className="flex items-center gap-2.5">
                <span className="relative flex h-10 w-10 shrink-0 items-center justify-center">
                  <motion.span aria-hidden className="absolute inset-0 rounded-xl border-2 border-dashed border-sky-400/70" animate={{ rotate: 360 }} transition={{ duration: 5, repeat: Infinity, ease: 'linear' }} />
                  <motion.span animate={{ rotate: 360 }} transition={{ duration: 2.2, repeat: Infinity, ease: 'linear' }}
                    className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-500 text-white"><Icon name="refresh" size={15} /></motion.span>
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground">
                    Running
                    <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-70" /><span className="relative inline-flex h-2 w-2 rounded-full bg-sky-500" /></span>
                  </div>
                  <div className="text-xs text-muted-foreground">Step <AnimatedNumber value={Math.min(doneCount + 1, PIPELINE_STEPS.length)} duration={300} /> of {PIPELINE_STEPS.length}</div>
                </div>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
                <motion.div className="h-full origin-left rounded-full bg-gradient-to-r from-sky-500 to-emerald-500" initial={false}
                  animate={{ scaleX: (doneCount + 0.5) / PIPELINE_STEPS.length }} transition={{ duration: 0.6, ease: MOTION.ease }} />
              </div>
              <Fade k={p?.currentStep || 'run'}><p className="break-words text-xs text-muted-foreground">{p?.currentStep || 'Working…'}</p></Fade>
            </motion.div>
          ) : (
            <motion.div key="result" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: MOTION.base }} className="flex items-center gap-2.5">
              <motion.span initial={{ scale: 0, rotate: -45 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 16, delay: 0.2 }}
                className={cx('flex h-10 w-10 shrink-0 items-center justify-center rounded-xl', failed ? TINT.rose : TINT.emerald)}>
                <Icon name={failed ? 'alert' : 'check'} size={18} strokeWidth={2.5} />
              </motion.span>
              <div className="min-w-0">
                <div className="text-sm font-semibold text-foreground">{failed ? 'Pipeline failed' : 'Pipeline complete'}</div>
                <p className="break-words text-xs text-muted-foreground">
                  {failed ? (p?.error || p?.currentStep || 'Something went wrong') : `Finished ${relTime(p?.completedAt)}${p?.status === 'complete_no_deck' ? ' · no deck (Gamma not set up)' : ''}`}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div className="border-t pt-4"><StepTrackVertical states={states} finished={phase === 'done'} /></div>
        {phase === 'done' ? <div className="flex border-t pt-4 [&>span]:w-full [&_button]:w-full">{runButton}</div> : null}
        {adminNote}
      </motion.aside>
    ) : null}

    {/* results: grow in as steps finish */}
    {p && phase !== 'idle' ? (
      <motion.div layout className="min-w-0 space-y-5">
        {phase === 'done' && !hasResults ? (
          <EmptyPanel icon="info" tint="slate" title="No results from this run">Run it again to research, pitch and draft a deck.</EmptyPanel>
        ) : null}
        <div className="grid items-start gap-5">
          {p.researchSummary ? (
            <Rise i={1}>
              <SectionTitle icon="search" right={<CopyAction label="Research" value={p.researchSummary} onCopy={copy} />}>Research</SectionTitle>
              <p className="whitespace-pre-wrap rounded-2xl border bg-card p-4 text-sm leading-relaxed text-body shadow-card">{p.researchSummary}</p>
            </Rise>
          ) : phase === 'running' ? <Rise i={1}>{pending('search', 'Researching the company and role…')}</Rise> : null}
          {p.pitchSubject || p.pitchEmail ? (
            <Rise i={2}>
              <SectionTitle icon="mail" right={<CopyAction label="Email" value={`${p.pitchSubject}\n\n${p.pitchEmail}`} onCopy={copy} />}>Pitch email</SectionTitle>
              <div className="overflow-hidden rounded-2xl border bg-card shadow-card">
                <div className="flex items-center gap-2.5 border-b bg-secondary/50 px-4 py-2.5">
                  <IconTile icon="send" tint="violet" size="sm" />
                  <div className="min-w-0">
                    <div className="text-[11px] font-medium text-muted-foreground">Subject</div>
                    <div className="break-words text-sm font-semibold text-foreground">{p.pitchSubject || 'No subject'}</div>
                  </div>
                </div>
                <p className="whitespace-pre-wrap px-4 py-3.5 text-sm leading-relaxed text-body">{p.pitchEmail}</p>
              </div>
            </Rise>
          ) : phase === 'running' ? <Rise i={2}>{pending('mail', 'Writing the pitch email…')}</Rise> : null}
        </div>
        {p.slides?.length ? (
          <Rise i={3}>
            <SectionTitle icon="file" right={<span className="text-xs font-medium text-muted-foreground">{p.slides.length} slides</span>}>Deck outline</SectionTitle>
            <ol className="grid gap-2 sm:grid-cols-2">{p.slides.map((s, i) => (
              <motion.li key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.3 + Math.min(i, 12) * 0.035 }}
                whileHover={{ y: -2 }}
                className="group flex items-center gap-3 rounded-xl border bg-card px-3 py-2.5 text-sm shadow-card transition-colors hover:border-amber-200">
                <span className={cx('flex h-7 w-9 shrink-0 items-center justify-center rounded-md text-[11px] font-bold tabular-nums transition-transform group-hover:scale-110', TINT.amber)}>{i + 1}</span>
                <span className="min-w-0 break-words text-body">{s.title || 'Untitled'}</span>
              </motion.li>
            ))}</ol>
          </Rise>
        ) : null}
        {p.gammaDeckUrl ? (
          <Rise i={4}>
            <motion.a whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} href={p.gammaDeckUrl} target="_blank" rel="noreferrer"
              className="group flex items-center gap-3 rounded-2xl border bg-card p-3.5 shadow-card transition-colors hover:border-brand-300">
              <IconTile icon="external" tint="violet" />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-foreground">Open deck in Gamma</div>
                <div className="truncate text-xs text-muted-foreground">{p.gammaDeckUrl}</div>
              </div>
              <Icon name="chevronRight" size={16} className="text-muted-foreground transition-transform group-hover:translate-x-1" />
            </motion.a>
          </Rise>
        ) : null}
      </motion.div>
    ) : null}
    </div>
  </div>
  </LayoutGroup>
  );
}
