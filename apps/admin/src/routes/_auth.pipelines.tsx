import React, { useEffect, useState } from 'react';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { PIPELINE_PAGE, api, base64ToBlob, saveBlob } from '../lib/api';
import { q as queries } from '../lib/queries';
import { applyChange } from '../lib/changes';
import type { PipelineList, PipelineResearch, PipelineRun, PipelineSettings } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { Avatar } from '../ui/Avatar';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import {
  Button, Card, Drawer, ErrorState, Modal, Notice, PageHeader, ScoreBadge, SearchInput, Skeleton, Spinner, Table, Td, Th,
  contactName, cx, errorMessage, fmtDateTime, relTime, useUi,
} from '../ui/kit';
import { CopyAction, EmptyPanel, IconTile, Rise, SectionTitle, TINT, type Tint } from '../ui/drawerKit';
import { PIPELINE_STEPS, stepStates, type StepState } from '../features/contacts/pipelineSteps';
import { StepTrackVertical } from '../features/contacts/PipelinePanel';

const STATUS: Record<string, { label: string; tint: Tint }> = {
  pending: { label: 'Starting', tint: 'sky' }, researching: { label: 'Researching', tint: 'sky' }, scoring: { label: 'Scoring', tint: 'sky' },
  pitching: { label: 'Writing pitch', tint: 'sky' }, generating_deck: { label: 'Making deck', tint: 'sky' }, attaching: { label: 'Attaching', tint: 'sky' },
  complete: { label: 'Complete', tint: 'emerald' }, complete_no_deck: { label: 'Complete · no deck', tint: 'emerald' }, failed: { label: 'Failed', tint: 'rose' },
};
const statusOf = (s: string) => STATUS[s] || { label: s.replace(/_/g, ' '), tint: 'slate' as Tint };

/** Status pill: spins + pings while running, the tick pops in when complete, the alert shakes once on failure. */
function StatusPill({ run }: { run: PipelineRun }) {
  const st = statusOf(run.status);
  const failed = run.status === 'failed';
  return (
    <span className={cx('inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold', TINT[st.tint])}>
      {run.running ? (
        <>
          <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-70" /><span className="relative inline-flex h-2 w-2 rounded-full bg-sky-500" /></span>
          <motion.span className="flex" animate={{ rotate: 360 }} transition={{ duration: 1.6, repeat: Infinity, ease: 'linear' }}><Icon name="refresh" size={12} /></motion.span>
        </>
      ) : (
        <motion.span className="flex" initial={{ scale: 0, rotate: -60 }} animate={failed ? { scale: 1, rotate: 0, x: [0, -2, 2, -1, 1, 0] } : { scale: 1, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 500, damping: 18, delay: 0.15 }}>
          <Icon name={failed ? 'alert' : 'check'} size={12} strokeWidth={2.6} />
        </motion.span>
      )}
      {st.label}
    </span>
  );
}

/** Five tiny step dots for a table row: green done, blue (pulsing) running, red failed, grey waiting. */
function MiniSteps({ run }: { run: PipelineRun }) {
  const states = stepStates(run);
  const tone = (s: StepState) => (s === 'done' ? 'bg-emerald-500 text-white' : s === 'now' ? 'bg-sky-500 text-white' : s === 'failed' ? 'bg-rose-500 text-white' : 'bg-secondary text-muted-foreground');
  return (
    <span className="flex items-center" aria-label={PIPELINE_STEPS.map((s, i) => `${s.label}: ${states[i]}`).join(', ')}>
      {PIPELINE_STEPS.map((s, i) => (
        <React.Fragment key={s.key}>
          <motion.span title={s.label} initial={{ scale: 0 }} animate={states[i] === 'now' ? { scale: [1, 1.15, 1] } : { scale: 1 }}
            transition={states[i] === 'now' ? { duration: 1.2, repeat: Infinity } : { type: 'spring', stiffness: 500, damping: 20, delay: 0.1 + i * 0.05 }}
            className={cx('relative flex h-6 w-6 items-center justify-center rounded-full ring-2 ring-card', tone(states[i]!))}>
            {states[i] === 'now' ? <span className="absolute inset-0 animate-ping rounded-full bg-sky-400/40" /> : null}
            <Icon name={states[i] === 'done' ? 'check' : states[i] === 'failed' ? 'x' : s.icon} size={11} {...(states[i] === 'done' || states[i] === 'failed' ? { strokeWidth: 3 } : {})} />
          </motion.span>
          {i < PIPELINE_STEPS.length - 1 ? (
            <span className="relative h-0.5 w-2.5 overflow-hidden bg-secondary">
              <motion.span className="absolute inset-0 origin-left bg-emerald-500" initial={{ scaleX: 0 }} animate={{ scaleX: states[i] === 'done' ? 1 : 0 }}
                transition={{ duration: 0.4, delay: 0.2 + i * 0.08 }} />
            </span>
          ) : null}
        </React.Fragment>
      ))}
    </span>
  );
}

/** Filter tile (like the contacts level tiles): icon, count that counts up, press feedback. */
function StatTile({ active, label, icon, tint, count, spin, onClick }: { active: boolean; label: string; icon: IconName; tint: Tint; count: number; spin?: boolean; onClick: () => void }) {
  return (
    <motion.button type="button" onClick={onClick} aria-pressed={active} whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
      className={cx('group flex min-w-[150px] flex-1 cursor-pointer items-center gap-3 rounded-2xl border bg-card px-3.5 py-3 text-left shadow-card transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active ? 'border-primary/60 ring-2 ring-primary/20' : 'hover:border-brand-200')}>
      <motion.span animate={spin ? { rotate: 360 } : { rotate: 0 }} transition={spin ? { duration: 2.4, repeat: Infinity, ease: 'linear' } : { type: 'spring' }}
        className={cx('flex h-10 w-10 shrink-0 items-center justify-center rounded-xl transition-transform group-hover:scale-110', TINT[tint])}>
        <Icon name={icon} size={18} />
      </motion.span>
      <span className="min-w-0">
        <span className="es-num block text-xl font-bold leading-6 text-foreground"><AnimatedNumber value={count} duration={600} /></span>
        <span className={cx('block text-xs font-medium', active ? 'text-primary' : 'text-muted-foreground')}>{label}</span>
      </span>
      {active ? <motion.span layoutId="es-pl-tile-dot" className="ml-auto h-2 w-2 shrink-0 rounded-full bg-primary" /> : null}
    </motion.button>
  );
}

/** The research as plain text (for Copy). */
function researchText(run: PipelineRun): string {
  const r = run.research;
  if (!r) return run.researchSummary || '';
  const sec = (title: string, items: string[]) => (items.length ? `${title}:\n${items.map((x) => `- ${x}`).join('\n')}` : '');
  return [
    [r.person.likely_title && `Role: ${r.person.likely_title}${r.person.seniority ? ` (${r.person.seniority})` : ''}`, r.person.authority && `Decision power: ${r.person.authority}`,
      (r.industry || r.company.industry) && `Industry: ${r.industry || r.company.industry}`, r.company.size_estimate && `Company size: ${r.company.size_estimate}`,
      r.company.products && `What they sell: ${r.company.products}`, r.company.growth_signals && `Growth signals: ${r.company.growth_signals}`].filter(Boolean).join('\n'),
    sec('Pain points', r.painPoints), sec('Opportunities', r.opportunities), sec('Conversation starters', r.talkingPoints),
    r.competitiveNotes && `Competition:\n${r.competitiveNotes}`,
  ].filter(Boolean).join('\n\n');
}

/** A titled list of points (pain points, opportunities, talking points) — each point slides in. */
function PointList({ icon, tint, title, items, start = 0 }: { icon: IconName; tint: Tint; title: string; items: string[]; start?: number }) {
  if (!items.length) return null;
  return (
    <div className="rounded-xl border bg-card p-3 shadow-card">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-foreground"><IconTile icon={icon} tint={tint} size="sm" />{title}</div>
      <ul className="space-y-1.5">
        {items.map((t, i) => (
          <motion.li key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.1 + (start + i) * 0.03 }}
            className="flex gap-2 text-[13px] leading-snug text-body">
            <span className={cx('mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full', tint === 'rose' ? 'bg-rose-500' : tint === 'emerald' ? 'bg-emerald-500' : 'bg-brand-500')} />
            <span className="min-w-0 break-words">{t}</span>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}

/** Everything the research step found: who they are, the company, pains, openings, what to say. */
function ResearchDetails({ r }: { r: PipelineResearch }) {
  const facts: [IconName, Tint, string, string][] = ([
    ['user', 'violet', 'Likely role', [r.person.likely_title, r.person.seniority && `${r.person.seniority} level`].filter(Boolean).join(' · ')],
    ['key', 'amber', 'Decision power', r.person.authority],
    ['briefcase', 'sky', 'Industry', r.industry || r.company.industry],
    ['users', 'emerald', 'Company size', r.company.size_estimate],
    ['store', 'violet', 'What they sell', r.company.products],
    ['bolt', 'amber', 'Growth signals', r.company.growth_signals],
  ] as [IconName, Tint, string, string][]).filter(([, , , v]) => !!v);
  return (
    <div className="space-y-2.5">
      {facts.length ? (
        <div className="grid gap-1.5 sm:grid-cols-2">
          {facts.map(([icon, tint, label, value], i) => (
            <motion.div key={label} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.08 + i * 0.03 }}
              className="group flex items-start gap-2.5 rounded-xl border bg-card px-2.5 py-2 shadow-card">
              <IconTile icon={icon} tint={tint} size="sm" />
              <div className="min-w-0">
                <div className="text-[10.5px] font-medium text-muted-foreground">{label}</div>
                <div className="break-words text-[13px] font-medium text-foreground">{value}</div>
              </div>
            </motion.div>
          ))}
        </div>
      ) : null}
      <div className="grid gap-2 sm:grid-cols-2">
        <PointList icon="alert" tint="rose" title="Pain points" items={r.painPoints} />
        <PointList icon="flame" tint="emerald" title="Opportunities" items={r.opportunities} start={r.painPoints.length} />
      </div>
      <PointList icon="send" tint="violet" title="Conversation starters" items={r.talkingPoints} start={r.painPoints.length + r.opportunities.length} />
      {r.competitiveNotes ? (
        <div className="rounded-xl border bg-card p-3 shadow-card">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-foreground"><IconTile icon="shield" tint="slate" size="sm" />Competition</div>
          <p className="whitespace-pre-wrap text-[13px] leading-snug text-body">{r.competitiveNotes}</p>
        </div>
      ) : null}
      {r.confidence ? (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground"><Icon name="info" size={12} />
          The AI's confidence in this research: <b className="font-semibold capitalize text-foreground">{r.confidence}</b>
        </p>
      ) : null}
    </div>
  );
}

/** One deck slide: number + title; click to see its content and the speaker notes. */
function SlideItem({ s, i }: { s: NonNullable<PipelineRun['slides']>[number]; i: number }) {
  const [open, setOpen] = useState(false);
  const body = [s.content, ...(s.bullets ?? []).map((b) => `• ${b}`)].filter(Boolean).join('\n');
  const more = !!body || !!s.speaker_notes;
  return (
    <motion.li initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.25 + Math.min(i, 12) * 0.035 }}
      className="overflow-hidden rounded-xl border bg-card shadow-card">
      <button type="button" disabled={!more} onClick={() => setOpen((o) => !o)} aria-expanded={open}
        className={cx('group flex w-full items-center gap-3 px-3 py-2.5 text-left text-sm transition-colors', more && 'cursor-pointer hover:bg-secondary/60')}>
        <span className={cx('flex h-7 w-9 shrink-0 items-center justify-center rounded-md text-[11px] font-bold tabular-nums transition-transform group-hover:scale-110', TINT.amber)}>{i + 1}</span>
        <span className="min-w-0 flex-1 break-words font-medium text-body">{s.title || 'Untitled'}</span>
        {more ? (
          <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: MOTION.base }} className="flex text-muted-foreground">
            <Icon name="chevronDown" size={15} />
          </motion.span>
        ) : null}
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}>
            <div className="space-y-2 border-t px-3 py-2.5">
              {body ? <p className="whitespace-pre-wrap text-[13px] leading-snug text-body">{body}</p> : null}
              {s.speaker_notes ? (
                <p className="rounded-lg bg-secondary/60 px-2.5 py-2 text-xs leading-snug text-muted-foreground">
                  <span className="mr-1 font-semibold text-foreground">Say:</span>{s.speaker_notes}
                </p>
              ) : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.li>
  );
}

/** Both scores, labelled: an admin run scores into the admin console's own score; the rep's stays theirs. */
function ScoresCard({ run }: { run: PipelineRun }) {
  const admin = run.adminScore;
  return (
    <div className="space-y-2 rounded-2xl border bg-card p-3 shadow-card">
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-2 text-sm font-medium text-body"><IconTile icon="shield" tint="violet" size="sm" />Admin score</span>
        {admin ? <ScoreBadge score={admin.score} /> : <span className="text-xs text-muted-foreground">{run.running ? 'Not yet' : 'None'}</span>}
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-2 text-sm font-medium text-body"><IconTile icon="smartphone" tint="slate" size="sm" />Rep's score</span>
        {run.repScore != null ? <ScoreBadge score={run.repScore} /> : <span className="text-xs text-muted-foreground">Not scored</span>}
      </div>
      <p className="text-[11px] leading-snug text-muted-foreground">
        {run.scoreCompleted === false && !run.running ? 'This run couldn’t score (the AI was unavailable). '
          : null}Runs started here save the admin score only. The rep's score is never changed.
      </p>
    </div>
  );
}

/** On/off switch for the auto-pipeline (runs after every card / QR scan, for the rep). */
function AutoPipelineSwitch({ settings }: { settings: PipelineSettings | undefined }) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const [busy, setBusy] = useState(false);
  if (!settings) return null;
  const on = settings.autoPipelineEnabled;
  const flip = async () => {
    const next = !on;
    if (next) {
      const ok = await confirm({
        title: 'Turn on the auto-pipeline?',
        message: <>After every card or QR scan the pipeline runs for that rep, as their own run: the research is added to the contact's notes, the rep's score is set, and the deck link is added to the contact. Each run makes several AI calls{settings.gammaConfigured ? ' and uses Gamma credits' : ''}.</>,
        confirmLabel: 'Turn on',
      });
      if (!ok) return;
    }
    setBusy(true);
    // optimistic: the switch moves at once; back if the server says no
    qc.setQueriesData<PipelineList>({ queryKey: ['admin', 'pipelines'] }, (d) => (d ? { ...d, settings: { ...d.settings, autoPipelineEnabled: next } } : d));
    try {
      await api.setPipelineSettings({ autoPipelineEnabled: next });
      toast('success', next ? 'Auto-pipeline on' : 'Auto-pipeline off',
        next === !!settings.autoPipelineFromEnv ? undefined : 'Until the server restarts. Set AUTO_PIPELINE_ENABLED in the server .env to keep it.');
    } catch (e) {
      toast('error', "Couldn't change the auto-pipeline", errorMessage(e));
    } finally {
      setBusy(false);
      void applyChange(qc, 'pipeline_settings_changed');
    }
  };
  return (
    <button type="button" role="switch" aria-checked={on} onClick={flip} disabled={busy || !settings.aiConfigured}
      title={settings.aiConfigured ? 'Run the pipeline automatically after every card / QR scan' : 'Needs the AI set up on the server'}
      className="inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg border bg-card px-2.5 text-sm font-medium text-body shadow-card transition-colors hover:border-brand-300 disabled:cursor-not-allowed disabled:opacity-60">
      <span className={cx('relative inline-flex h-5 w-9 shrink-0 items-center rounded-full p-0.5 transition-colors duration-200', on ? 'bg-emerald-500' : 'bg-secondary ring-1 ring-inset ring-border')}>
        <motion.span className="h-4 w-4 rounded-full bg-white shadow" animate={{ x: on ? 16 : 0 }} transition={{ type: 'spring', stiffness: 500, damping: 32 }} />
      </span>
      Auto-pipeline {busy ? <Spinner size={13} /> : <span className={cx('text-xs font-semibold', on ? 'text-emerald-700 dark:text-emerald-400' : 'text-muted-foreground')}>{on ? 'On' : 'Off'}</span>}
    </button>
  );
}

/** "Run pipeline": find a contact and start an admin run for them (writes only admin-side results). */
function NewRunModal({ open, onClose, onStarted }: { open: boolean; onClose: () => void; onStarted: (contactId: string) => void }) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const [text, setText] = useState('');
  const [term, setTerm] = useState('');
  const [startingId, setStartingId] = useState<string | null>(null);
  useEffect(() => { const t = setTimeout(() => setTerm(text.trim()), 250); return () => clearTimeout(t); }, [text]);
  useEffect(() => { if (!open) { setText(''); setTerm(''); } }, [open]);
  const found = useQuery({ ...queries.contacts({ search: term || undefined, limit: 30, lite: 1 }), enabled: open });
  const list = found.data?.contacts ?? [];
  const start = async (id: string) => {
    setStartingId(id);
    try {
      const r = await api.runPipeline(id);
      toast(r.status === 'already_running' ? 'info' : 'success', r.message);
      void applyChange(qc, 'pipeline_started');
      onStarted(id);
    } catch (e) {
      toast('error', "Couldn't start the pipeline", errorMessage(e));
    } finally {
      setStartingId(null);
    }
  };
  return (
    <Modal open={open} onOpenChange={(o) => { if (!o) onClose(); }} title="Run the pipeline"
      description="Pick a contact. The pipeline researches them, scores them (admin score), writes a pitch email and builds a deck. The rep's own contact isn't changed.">
      <SearchInput value={text} onChange={setText} placeholder="Search name, company or email…" />
      <div className="mt-3 max-h-[46vh] space-y-1 overflow-y-auto pr-1">
        {found.isLoading ? <div className="space-y-1.5">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full rounded-xl" />)}</div>
          : found.error ? <ErrorState error={found.error} onRetry={() => found.refetch()} />
            : !list.length ? <p className="py-6 text-center text-sm text-muted-foreground">{term ? 'No contact matches.' : 'No contacts yet.'}</p>
              : list.map((c, i) => (
                <motion.div key={c.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 10) * 0.025 }}
                  className="group flex items-center gap-3 rounded-xl border bg-card px-3 py-2 transition-colors hover:border-brand-200">
                  <Avatar src={c.photoUrl || null} name={contactName(c)} size="sm" />
                  <div className="min-w-0 flex-1">
                    <div className="break-words text-sm font-semibold text-foreground">{contactName(c)}</div>
                    <div className="break-words text-xs text-muted-foreground">{[c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No details'}</div>
                  </div>
                  <Button size="sm" variant="primary" icon="play" loading={startingId === c.id} disabled={!!startingId} onClick={() => start(c.id)}>Run</Button>
                </motion.div>
              ))}
      </div>
    </Modal>
  );
}

function RunModal({ run, onClose }: { run: PipelineRun | null; onClose: () => void }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { toast } = useUi();
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  if (!run) return null;
  const downloadDeck = async () => {
    if (!run.deckFileId) return;
    setDownloading(true);
    try {
      const f = await api.downloadContactFile(run.deckFileId);
      saveBlob(base64ToBlob(f.contentBase64, f.mimeType), f.filename);
    } catch (e) {
      toast('error', "Couldn't download the deck", errorMessage(e));
    } finally {
      setDownloading(false);
    }
  };
  const copy = async (what: string, value: string) => {
    try { await navigator.clipboard.writeText(value); toast('success', `${what} copied`); } catch { toast('error', `Couldn't copy the ${what.toLowerCase()}`); }
  };
  const retry = async () => {
    setBusy(true);
    try {
      const r = await api.retryPipeline(run.contactId);
      toast('success', r.message);
      void applyChange(qc, 'pipeline_started');
      onClose();
    } catch (e) {
      toast('error', "Couldn't retry", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const failed = run.status === 'failed';
  const states = stepStates(run);
  const empty = !run.researchSummary && !run.pitchEmail && !run.pitchSubject && !run.slides?.length && !run.error;
  return (
    <Drawer mode="center" open={!!run} onOpenChange={(o) => { if (!o) onClose(); }} title={run.contactName || 'Pipeline run'}
      header={
        <div className="flex items-center gap-3 pr-8">
          <motion.span initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 22 }}>
            <Avatar name={run.contactName || 'Deleted contact'} size="md" />
          </motion.span>
          <div className="min-w-0 flex-1">
            <h2 className="break-words font-display text-lg font-bold text-foreground">{run.contactName || 'Deleted contact'}</h2>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
              {run.company ? <span className="inline-flex items-center gap-1"><Icon name="building" size={12} className="text-sky-600" />{run.company}</span> : null}
              {run.owner ? <span className="inline-flex items-center gap-1"><Icon name="user" size={12} className="text-brand-600" />{run.owner}'s contact</span> : null}
            </div>
          </div>
          <StatusPill run={run} />
        </div>
      }
      footer={<>
        {run.contactExists === false ? (
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground"><Icon name="trash" size={13} />The contact was deleted — see Recently deleted to restore it.</span>
        ) : (
          <Button variant="ghost" icon="contacts" onClick={() => { onClose(); navigate({ to: '/contacts/$id', params: { id: run.contactId }, search: { tab: 'pipeline' } }); }}>Open contact</Button>
        )}
        {!run.running && run.contactExists !== false ? (
          <motion.span whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.95 }} className="inline-flex">
            <Button variant="primary" icon={failed ? 'refresh' : 'play'} loading={busy} onClick={retry}>{failed ? 'Retry' : 'Run again'}</Button>
          </motion.span>
        ) : null}
      </>}>
      <div className="grid items-start gap-5 px-6 py-5 md:grid-cols-[minmax(240px,1fr)_2.2fr]">
        {/* left: when, the steps, the score — pinned in its own column */}
        <aside className="space-y-4 md:sticky md:top-5">
          <Rise i={0} className="grid grid-cols-2 gap-2">
            {([['play', 'violet', 'Started', fmtDateTime(run.startedAt)], [run.running ? 'refresh' : failed ? 'alert' : 'check', run.running ? 'sky' : failed ? 'rose' : 'emerald', run.running ? 'Now' : 'Finished',
              run.completedAt ? fmtDateTime(run.completedAt) : run.running ? run.currentStep || 'Running…' : '—']] as [IconName, Tint, string, string][]).map(([icon, tint, label, value]) => (
              <div key={label} className="group rounded-2xl border bg-card p-3 shadow-card">
                <IconTile icon={icon} tint={tint} size="sm" />
                <div className="mt-2 text-[11px] font-medium text-muted-foreground">{label}</div>
                <div className="break-words text-xs font-semibold text-foreground">{value}</div>
              </div>
            ))}
          </Rise>
          <Rise i={1} className="rounded-2xl border bg-card p-4 shadow-card">
            <SectionTitle icon="pipeline">Steps</SectionTitle>
            <StepTrackVertical states={states} finished={!run.running} />
          </Rise>
          <Rise i={2}><ScoresCard run={run} /></Rise>
        </aside>

        {/* right: what the run produced */}
        <div className="min-w-0 space-y-5">
          {run.error ? <Rise i={1}><Notice tone="red" icon="alert">{run.error}</Notice></Rise> : null}
          {run.research || run.researchSummary ? (
            <Rise i={2}>
              <SectionTitle icon="search" right={<CopyAction label="Research" value={researchText(run)} onCopy={copy} />}>Research</SectionTitle>
              {run.research ? <ResearchDetails r={run.research} />
                : <p className="whitespace-pre-wrap rounded-2xl border bg-card p-4 text-sm leading-relaxed text-body shadow-card">{run.researchSummary}</p>}
            </Rise>
          ) : null}
          {run.pitchSubject || run.pitchEmail ? (
            <Rise i={3}>
              <SectionTitle icon="mail" right={<CopyAction label="Email" value={`${run.pitchSubject}\n\n${run.pitchEmail}`} onCopy={copy} />}>Pitch email</SectionTitle>
              <div className="overflow-hidden rounded-2xl border bg-card shadow-card">
                <div className="flex items-center gap-2.5 border-b bg-secondary/50 px-4 py-2.5">
                  <IconTile icon="send" tint="violet" size="sm" />
                  <div className="min-w-0">
                    <div className="text-[11px] font-medium text-muted-foreground">Subject</div>
                    <div className="break-words text-sm font-semibold text-foreground">{run.pitchSubject || 'No subject'}</div>
                  </div>
                </div>
                <p className="whitespace-pre-wrap px-4 py-3.5 text-sm leading-relaxed text-body">{run.pitchEmail}</p>
              </div>
            </Rise>
          ) : null}
          {run.slides?.length ? (
            <Rise i={4}>
              <SectionTitle icon="file" right={<span className="text-xs font-medium text-muted-foreground">{run.slides.length} slides · click one for its content</span>}>Deck outline</SectionTitle>
              <ol className="grid items-start gap-2 sm:grid-cols-2">{run.slides.map((s, i) => <SlideItem key={i} s={s} i={i} />)}</ol>
            </Rise>
          ) : null}
          {run.deckFileId ? (
            <Rise i={5}>
              <motion.button type="button" whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} disabled={downloading} onClick={downloadDeck}
                className="group flex w-full cursor-pointer items-center gap-3 rounded-2xl border bg-card p-3.5 text-left shadow-card transition-colors hover:border-brand-300 disabled:opacity-60">
                <IconTile icon="download" tint="emerald" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold text-foreground">Download the deck (PowerPoint)</div>
                  <div className="text-xs text-muted-foreground">Kept in the admin console only — the rep doesn't see it</div>
                </div>
                {downloading ? <Spinner size={16} /> : <Icon name="chevronRight" size={16} className="text-muted-foreground transition-transform group-hover:translate-x-1" />}
              </motion.button>
            </Rise>
          ) : null}
          {run.gammaDeckUrl ? (
            <Rise i={5}>
              <motion.a whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} href={run.gammaDeckUrl} target="_blank" rel="noreferrer"
                className="group flex items-center gap-3 rounded-2xl border bg-card p-3.5 shadow-card transition-colors hover:border-brand-300">
                <IconTile icon="external" tint="violet" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold text-foreground">Open deck in Gamma</div>
                  <div className="truncate text-xs text-muted-foreground">{run.gammaDeckUrl}</div>
                </div>
                <Icon name="chevronRight" size={16} className="text-muted-foreground transition-transform group-hover:translate-x-1" />
              </motion.a>
            </Rise>
          ) : null}
          {empty ? (
            run.running
              ? <EmptyPanel icon="refresh" tint="sky" title="Working on it">Results appear here as each step finishes.</EmptyPanel>
              : <EmptyPanel icon="info" tint="slate" title="This run produced no output" />
          ) : null}
        </div>
      </div>
    </Drawer>
  );
}

export function PipelinesComponent() {
  // the status filter lives in the URL (?status=failed)
  const search = useSearch({ from: '/_auth/pipelines' });
  const navigate = useNavigate();
  type Filter = 'all' | 'running' | 'complete' | 'failed';
  const filter: Filter = search.status ?? 'all';
  const setFilter = (f: Filter) => navigate({ to: '/pipelines', search: f === 'all' ? {} : { status: f }, replace: true });
  const qc = useQueryClient();
  const [openId, setOpenId] = useState<string | null>(null);
  const [newRun, setNewRun] = useState(false);
  // real paging: the server returns one page of the filter's runs (offset / limit) and how many there are
  const status = search.status;
  // the page belongs to its filter: switching the filter starts at its first page
  const [paging, setPaging] = useState<{ status: string | undefined; offset: number }>({ status, offset: 0 });
  const offset = paging.status === status ? paging.offset : 0;
  const setOffset = (n: number) => setPaging({ status, offset: Math.max(0, n) });
  const q = useQuery({
    ...queries.pipelines(status, offset),
    // a page / filter switch keeps the current rows up until the new ones arrive
    placeholderData: keepPreviousData,
    refetchInterval: (query) => (query.state.data?.pipelines.some((p) => p.running) ? 3000 : false),
  });
  const runs = q.data?.pipelines || [];
  const c = q.data?.statusCounts || {};
  const PIPELINE_RUNNING = ['pending', 'researching', 'scoring', 'pitching', 'generating_deck', 'attaching'];
  // counts come from statusCounts (every run), not from the page on screen
  const running = PIPELINE_RUNNING.reduce((n, st) => n + (c[st] || 0), 0);
  const complete = (c.complete || 0) + (c.complete_no_deck || 0);
  const allRuns = Object.values(c).reduce((n, v) => n + (v || 0), 0);
  const shown = runs.filter((r) => filter === 'all' || (filter === 'running' ? r.running : filter === 'complete' ? r.status.startsWith('complete') : r.status === 'failed'));
  const total = q.data?.total ?? shown.length;
  const from = shown.length ? offset + 1 : 0;
  const to = offset + runs.length;
  const s = q.data?.settings;
  // the modal follows the live list, so an open running run updates as it goes
  const open = openId ? runs.find((r) => (r.id || r.contactId) === openId) ?? null : null;

  const tiles: { value: Filter; label: string; icon: IconName; tint: Tint; count: number }[] = [
    { value: 'all', label: 'All runs', icon: 'pipeline', tint: 'violet', count: allRuns },
    { value: 'running', label: 'Running', icon: 'refresh', tint: 'sky', count: running },
    { value: 'complete', label: 'Complete', icon: 'check', tint: 'emerald', count: complete },
    { value: 'failed', label: 'Failed', icon: 'alert', tint: 'rose', count: c.failed || 0 },
  ];

  return (
    <>
      <PageHeader title="AI pipelines" description="For one contact the pipeline researches the company, scores the lead, writes a personal pitch email and builds a deck. Runs started here never change the rep's own contact."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <AutoPipelineSwitch settings={s} />
            <motion.span whileTap={{ scale: 0.95 }} className="inline-flex">
              <Button icon="refresh" onClick={() => q.refetch()} loading={q.isFetching && !q.isLoading}>Refresh</Button>
            </motion.span>
            <motion.span whileHover={{ y: -1 }} whileTap={{ scale: 0.95 }} className="inline-flex">
              <Button variant="primary" icon="play" disabled={!!s && !s.aiConfigured} onClick={() => setNewRun(true)}>Run pipeline</Button>
            </motion.span>
          </div>
        } />

      {s && !s.aiConfigured ? <Rise className="mb-4"><Notice tone="amber" icon="alert">The AI isn't set up on the server (<code className="font-mono text-xs">OPENROUTER_API_KEY</code>) — pipelines can't run until it is.</Notice></Rise> : null}
      {s && s.aiConfigured && !s.gammaConfigured ? <Rise className="mb-4"><Notice icon="info">Decks need <code className="font-mono text-xs">GAMMA_API_KEY</code> on the server. Without it pipelines still research, score and write the pitch ("Complete · no deck").</Notice></Rise> : null}

      {/* filters as tiles */}
      <Rise i={1} className="mb-4 flex flex-wrap gap-2.5">
        {tiles.map((t) => (
          <StatTile key={t.value} active={filter === t.value} label={t.label} icon={t.icon} tint={t.tint} count={t.count}
            spin={t.value === 'running' && running > 0} onClick={() => setFilter(t.value)} />
        ))}
      </Rise>

      {q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : (
        <Rise i={2}>
          <Card className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <IconTile icon="pipeline" tint="violet" size="sm" />
                {tiles.find((t) => t.value === filter)?.label}
                <span className="es-num rounded-full bg-secondary px-2 text-xs font-semibold text-muted-foreground"><AnimatedNumber value={total} duration={400} /></span>
              </div>
              <AnimatePresence>
                {running ? (
                  <motion.span key="live" initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                    className="inline-flex items-center gap-1.5 rounded-full bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-700 dark:bg-sky-500/10 dark:text-sky-300">
                    <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-70" /><span className="relative inline-flex h-2 w-2 rounded-full bg-sky-500" /></span>
                    Live · {running} run{running === 1 ? '' : 's'} in progress
                  </motion.span>
                ) : null}
              </AnimatePresence>
            </div>
            {q.isLoading ? <div className="space-y-2 p-4">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>
              : !shown.length ? (
                <div className="p-6">
                  <EmptyPanel icon="pipeline" tint="violet" title={allRuns ? 'Nothing in this view' : 'No pipeline runs yet'}>
                    {allRuns ? 'Try another filter above.' : 'Open a contact and press "Run pipeline" to research them and draft a pitch.'}
                  </EmptyPanel>
                </div>
              ) : (
                <Table>
                  <thead><tr><Th>Contact</Th><Th>Representative</Th><Th>Progress</Th><Th>Status</Th><Th>Step</Th><Th>Started</Th><Th className="w-8" /></tr></thead>
                  <tbody>
                    {shown.map((r) => {
                      const cur = PIPELINE_STEPS.find((x) => x.key === r.status);
                      return (
                        // rows cascade in through the shared table animation (ui/kit Table)
                        <tr key={r.id || r.contactId} onClick={() => setOpenId(r.id || r.contactId)} className="group cursor-pointer transition-colors hover:bg-secondary/60">
                          <Td>
                            <div className="flex items-center gap-2.5">
                              <span className="transition-transform duration-300 group-hover:scale-110"><Avatar name={r.contactName || 'Deleted contact'} size="sm" /></span>
                              <div className="min-w-0">
                                <div className="font-semibold text-foreground">{r.contactName || 'Deleted contact'}</div>
                                <div className="flex items-center gap-1 text-xs text-muted-foreground"><Icon name="building" size={11} className="text-sky-600" />{r.company || '—'}</div>
                              </div>
                            </div>
                          </Td>
                          <Td>
                            {r.owner ? (
                              <span className="inline-flex items-center gap-1.5 text-body"><Avatar name={r.owner} size="xs" className="size-6 border-0 text-[9px] shadow-none" />{r.owner}</span>
                            ) : <span className="text-muted-foreground">—</span>}
                          </Td>
                          <Td><MiniSteps run={r} /></Td>
                          <Td><StatusPill run={r} /></Td>
                          <Td className="max-w-[260px] text-body" title={r.error || r.currentStep}>
                            <span className="flex items-center gap-1.5">
                              <Icon name={r.error ? 'alert' : cur?.icon || (r.status.startsWith('complete') ? 'check' : 'info')} size={13}
                                className={cx('shrink-0', r.error ? 'text-rose-600' : r.running ? 'animate-pulse text-sky-600' : 'text-muted-foreground')} />
                              <span className="truncate">{r.error || r.currentStep || '—'}</span>
                            </span>
                          </Td>
                          <Td className="whitespace-nowrap text-muted-foreground"><span className="inline-flex items-center gap-1"><Icon name="clock" size={12} />{relTime(r.startedAt)}</span></Td>
                          <Td><Icon name="chevronRight" size={16} className="text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" /></Td>
                        </tr>
                      );
                    })}
                  </tbody>
                </Table>
              )}
            {/* paging: the server sends one page of this filter's runs (newest first) and how many there are */}
            {total > PIPELINE_PAGE || offset > 0 ? (
              <div className="flex flex-wrap items-center justify-between gap-2 border-t px-4 py-2 text-xs text-muted-foreground">
                <span role="status" className="inline-flex items-center gap-1.5">
                  {q.isPlaceholderData ? <Spinner size={12} /> : null}
                  Runs <span className="tabular-nums">{from.toLocaleString()}–{to.toLocaleString()}</span> of <span className="tabular-nums">{total.toLocaleString()}</span>, newest first
                </span>
                <span className="flex items-center gap-1.5">
                  <Button size="sm" disabled={offset === 0 || q.isPlaceholderData} onClick={() => setOffset(offset - PIPELINE_PAGE)}>
                    <Icon name="chevronRight" size={14} className="rotate-180" />Newer
                  </Button>
                  <Button size="sm" disabled={to >= total || q.isPlaceholderData} onClick={() => setOffset(offset + PIPELINE_PAGE)}>
                    Older<Icon name="chevronRight" size={14} />
                  </Button>
                </span>
              </div>
            ) : null}
          </Card>
        </Rise>
      )}
      <RunModal run={open} onClose={() => setOpenId(null)} />
      <NewRunModal open={newRun} onClose={() => setNewRun(false)}
        onStarted={async (contactId) => {
          setNewRun(false);
          // back to the newest page; open the new run as soon as the list has it (the server waits for the run record)
          setOffset(0);
          const fresh = await qc.fetchQuery({ ...queries.pipelines(status, 0), staleTime: 0 });
          const r = fresh.pipelines.find((p) => p.contactId === contactId);
          if (r) setOpenId(r.id || r.contactId);
        }} />
    </>
  );
}
