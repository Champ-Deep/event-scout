// /broadcasts — Announcements (and the Events tab), all on the real API:
//   live        GET  /api/broadcasts/active            send   POST /api/admin/broadcast {message, priority, recipientIds, eventId}
//   end         POST /api/broadcasts/{id}/dismiss       log    GET  /api/admin/broadcasts/history[?start&end]
//   events      features/events/EventPlanner (GET /api/admin/planner/events …), AI fill POST /api/admin/planner/extract
import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { EventFormModal, EventPlanner, eventMessage, fmtRange } from '../features/events/EventPlanner';
import { SearchField } from '../features/contacts/Toolbar';
import { ActivityCalendar, type CalendarKinds } from '../features/activity/ActivityCalendar';
import { addDays, dayKey, fromDayKey } from '../lib/dates';
import type { BroadcastLog, PlannerEventInput } from '../lib/types';
import { clean } from '../lib/search';
import { MOTION } from '../ui/tokens';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { removeFromArrays } from '../lib/optimistic';
import { api } from '../lib/api';
import { Avatar } from '../ui/Avatar';
import { Icon, type IconName } from '../ui/icons';
import { LiveIcon } from '../ui/live-icon';
import { AnimatedNumber } from '../ui/micro';
import { Button, Card, EmptyState, ErrorState, Skeleton, cx, deep, errorMessage, fmtDateTime, relTime, useUi } from '../ui/kit';

const MAX = 2000;
/** What the calendar's dots mean on this page. */
const ANNOUNCE_KINDS: CalendarKinds = { normal: { color: 'var(--brand-600)', label: 'Normal' }, urgent: { color: '#f43f5e', label: 'Urgent' } };
const URGENT = '#f43f5e';
const spring = { type: 'spring', stiffness: 420, damping: 26 } as const;

/** Who an announcement went to, in a few words. */
const toWhom = (r: { name: string }[] | undefined) => (r?.length ? `to ${r.length <= 2 ? r.map((x) => x.name).join(' & ') : `${r.length} people`}` : 'to everyone');

/** A ring that fills as the message grows; amber near the limit, red at it. */
function CharRing({ n }: { n: number }) {
  const size = 30, stroke = 3, r = (size - stroke) / 2, c = 2 * Math.PI * r;
  const pct = Math.min(1, n / MAX);
  const color = pct >= 1 ? URGENT : pct > 0.85 ? '#f59e0b' : 'var(--brand-600)';
  return (
    <span className="flex items-center gap-2" title={`${n.toLocaleString()} of ${MAX.toLocaleString()} characters`}>
      <AnimatePresence>{pct > 0.85 ? <motion.span initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="text-xs font-bold tabular-nums" style={{ color }}>{(MAX - n).toLocaleString()} left</motion.span> : null}</AnimatePresence>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} style={{ stroke: 'var(--border)' }} />
        <motion.circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} strokeLinecap="round" strokeDasharray={c}
          animate={{ strokeDashoffset: c * (1 - pct), stroke: color }} transition={{ duration: MOTION.base, ease: MOTION.ease }} />
      </svg>
    </span>
  );
}

/** One of the two choice cards (priority / audience): icon, title, line; the picked one glows. */
function ChoiceCard({ on, onClick, icon, title, sub, color, layoutId }: { on: boolean; onClick: () => void; icon: IconName; title: string; sub: string; color: string; layoutId: string }) {
  return (
    <motion.button type="button" onClick={onClick} aria-pressed={on} whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
      className={cx('relative flex flex-1 cursor-pointer items-center gap-3 rounded-2xl border-[1.5px] p-3 text-left transition-colors', on ? 'border-transparent' : 'border-border bg-card hover:border-brand-200')}>
      {on ? <motion.span layoutId={layoutId} className="absolute inset-0 rounded-2xl border-[1.5px]" style={{ borderColor: color, background: `color-mix(in oklab, ${color} 9%, var(--card))` }} transition={spring} /> : null}
      <motion.span animate={on ? { rotate: [0, -12, 10, -6, 0], scale: [1, 1.12, 1] } : { rotate: 0, scale: 1 }} transition={{ duration: 0.5 }}
        className="relative flex size-10 shrink-0 items-center justify-center rounded-xl" style={{ background: on ? color : `color-mix(in oklab, ${color} 13%, transparent)`, color: on ? 'white' : color }}>
        <Icon name={icon} size={18} />
      </motion.span>
      <span className="relative min-w-0">
        <span className="block text-sm font-extrabold text-foreground">{title}</span>
        <span className="block text-xs text-muted-foreground">{sub}</span>
      </span>
      <AnimatePresence>{on ? <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={spring} className="relative ml-auto flex size-5 items-center justify-center rounded-full text-white" style={{ background: color }}><Icon name="check" size={11} strokeWidth={3} /></motion.span> : null}</AnimatePresence>
    </motion.button>
  );
}

/** What reps see: the feed banner inside a phone, updating as you type. */
function PhonePreview({ message, urgent, eventName }: { message: string; urgent: boolean; eventName?: string | undefined }) {
  const color = urgent ? URGENT : 'var(--brand-600)';
  return (
    <div className="mx-auto w-[260px] shrink-0">
      <div className="relative rounded-[2.2rem] border-[6px] border-foreground/85 bg-background p-3 pt-7 shadow-xl">
        <span className="absolute left-1/2 top-2 h-4 w-20 -translate-x-1/2 rounded-full bg-foreground/85" aria-hidden />
        <div className="mb-2 flex items-center justify-between px-1 text-[10px] font-bold text-muted-foreground"><span>Scout</span><span>{new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span></div>
        <AnimatePresence mode="popLayout" initial={false}>
          {message.trim() ? (
            <motion.div key="card" layout initial={{ opacity: 0, y: -14, scale: 0.94 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.96 }} transition={spring}
              className="flex items-start gap-2 rounded-2xl border p-2.5" style={{ borderColor: `color-mix(in oklab, ${color} 30%, transparent)`, background: `color-mix(in oklab, ${color} 8%, var(--card))` }}>
              <motion.span key={String(urgent)} initial={{ rotate: -20, scale: 0.6 }} animate={{ rotate: 0, scale: 1 }} transition={spring}
                className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-card" style={{ color }}>
                <Icon name={urgent ? 'alert' : 'megaphone'} size={13} />
              </motion.span>
              <div className="min-w-0 flex-1">
                <div className="text-[8.5px] font-black tracking-[0.08em]" style={{ color }}>{urgent ? 'URGENT · TEAM' : 'TEAM ANNOUNCEMENT'} · JUST NOW</div>
                <p className="line-clamp-4 whitespace-pre-line break-words text-[11.5px] font-semibold leading-[15px] text-foreground">{message}</p>
                {message.length > 220 || message.split('\n').length > 4 ? <div className="mt-0.5 text-[10px] font-black" style={{ color }}>Show more</div> : null}
                {eventName ? <div className="mt-1 inline-flex items-center gap-1 rounded-full bg-card px-1.5 text-[9px] font-bold leading-4 text-brand-700"><Icon name="calendar" size={8} />{eventName}</div> : null}
              </div>
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="flex h-20 items-center justify-center rounded-2xl border border-dashed text-center text-[11px] text-muted-foreground">
              Start typing — your announcement<br />appears here
            </motion.div>
          )}
        </AnimatePresence>
        {/* the rest of the feed, faded */}
        {[0, 1, 2].map((i) => (
          <div key={i} className="mt-2 flex items-center gap-2 rounded-xl bg-secondary/60 p-2 opacity-60" aria-hidden>
            <span className="size-6 rounded-full bg-border" />
            <span className="flex-1 space-y-1"><span className="block h-1.5 w-2/3 rounded-full bg-border" /><span className="block h-1.5 w-1/3 rounded-full bg-border" /></span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-center text-[11px] font-semibold text-muted-foreground">Top of every rep's Scout feed</p>
    </div>
  );
}

export function BroadcastsCenterComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const live = useSuspenseQuery(q.broadcasts());
  const users = useSuspenseQuery(q.users());
  const planner = useSuspenseQuery(q.plannerEvents());
  const historyAll = useQuery(q.broadcastHistory());
  // ?tab=events shows the event planner; ?event=<id> opens one event's panel
  const search = useSearch({ from: '/_auth/broadcasts' });
  const navigate = useNavigate();
  const tab = search.tab ?? (search.event ? 'events' : 'announce');
  const setTab = (t: 'announce' | 'events') => navigate({ to: '/broadcasts', search: (s) => clean({ ...s, tab: t === 'announce' ? undefined : t, event: t === 'announce' ? undefined : s.event }) as never, replace: true });
  const openEvent = (id: string | undefined) => navigate({ to: '/broadcasts', search: (s) => clean({ ...s, tab: 'events', event: id }) as never });
  const upcoming = planner.data.filter((e) => !['done', 'dismissed'].includes(e.status));
  const [message, setMessage] = useState('');
  const [priority, setPriority] = useState<'normal' | 'urgent'>('normal');
  const [audience, setAudience] = useState<'everyone' | 'some'>('everyone');
  const [picked, setPicked] = useState<Record<string, boolean>>({});
  const [repQuery, setRepQuery] = useState('');
  const [sending, setSending] = useState(false);
  const [justSent, setJustSent] = useState(false);
  const [liveOpen, setLiveOpen] = useState<Record<string, boolean>>({});
  // the event this announcement is about (linked: it shows on the event and in the log)
  const [eventId, setEventId] = useState<string | undefined>(undefined);
  const linked = eventId ? planner.data.find((e) => e.id === eventId) : undefined;
  const [eventForm, setEventForm] = useState<{ value: PlannerEventInput } | null>(null);
  const [extracting, setExtracting] = useState(false);
  const recipients = Object.keys(picked).filter((k) => picked[k]);
  const reps = (users.data || []).filter((u) => !u.isAdmin);
  const shownReps = reps.filter((u) => !repQuery || `${u.name} ${u.email}`.toLowerCase().includes(repQuery.toLowerCase()));
  const weekAgo = Date.now() - 7 * 86_400_000;
  const sentThisWeek = (historyAll.data ?? []).filter((b) => b.createdAt && new Date(b.createdAt).getTime() >= weekAgo).length;
  useEffect(() => { if (!justSent) return; const t = setTimeout(() => setJustSent(false), 1800); return () => clearTimeout(t); }, [justSent]);

  // "Save as a new event": the AI reads the details out of the message, the admin checks them in the form
  const saveAsEvent = async () => {
    setExtracting(true);
    try {
      const value = await api.extractPlannerEvent(message.trim());
      setEventForm({ value: { status: 'planned', ...value } });
    } catch (e) {
      toast('info', "Couldn't read the details automatically", `${errorMessage(e)} The form is empty — fill it in.`);
      setEventForm({ value: { status: 'planned', name: '', details: {} } });
    } finally { setExtracting(false); }
  };

  const send = async () => {
    setSending(true);
    try {
      await api.sendBroadcast(message.trim(), priority, audience === 'some' ? recipients : undefined, eventId);
      toast('success', 'Announcement sent', audience === 'some' ? `To ${recipients.length} rep${recipients.length === 1 ? '' : 's'}` : 'Reps see it at the top of their Scout feed.');
      setJustSent(true);
      setMessage('');
      setPicked({});
      setEventId(undefined);
      void applyChange(qc, 'broadcast_alert');
    } catch (e) {
      toast('error', "Couldn't send", errorMessage(e));
    } finally {
      setSending(false);
    }
  };
  const end = async (id: string) => {
    if (!(await confirm({ title: 'End this announcement?', message: 'It disappears from every phone.', confirmLabel: 'End it' }))) return;
    // optimistic: it leaves the live list at once and comes back if the server rejects
    const rollback = await removeFromArrays<{ id: string }>(qc, ['admin', 'broadcasts'], [id]);
    try {
      await api.endBroadcast(id);
      toast('success', 'Announcement ended');
    } catch (e) {
      rollback();
      toast('error', "Couldn't end it — restored", errorMessage(e));
    } finally {
      void applyChange(qc, 'broadcast_alert');
    }
  };

  const canSend = message.trim().length > 0 && (audience === 'everyone' || recipients.length > 0);
  const liveN = live.data?.length ?? 0;
  return (
    <>
      {/* hero: what this page is, three live numbers, and the tabs */}
      <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.4, ease: MOTION.ease }}
        className="relative mb-5 overflow-hidden rounded-3xl border p-5 shadow-card sm:p-6"
        style={{ background: 'linear-gradient(135deg, color-mix(in oklab, var(--brand-500) 13%, var(--card)), var(--card) 60%)' }}>
        <motion.span aria-hidden className="pointer-events-none absolute -right-16 -top-20 size-64 rounded-full" style={{ background: 'color-mix(in oklab, var(--brand-500) 12%, transparent)' }}
          animate={{ scale: [1, 1.1, 1], rotate: [0, 12, 0] }} transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.span aria-hidden className="pointer-events-none absolute -bottom-24 right-40 size-48 rounded-full" style={{ background: 'color-mix(in oklab, #0ea5e9 8%, transparent)' }}
          animate={{ scale: [1, 1.15, 1] }} transition={{ duration: 7, repeat: Infinity, ease: 'easeInOut', delay: 1 }} />
        <div className="relative flex flex-wrap items-center gap-5">
          <motion.span initial={{ scale: 0.5, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={spring}
            className="flex size-14 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-active">
            <LiveIcon name="megaphone" size={26} />
          </motion.span>
          <div className="min-w-0 flex-1">
            <h1 className="font-display text-[26px] font-black leading-8 tracking-[-0.03em] text-foreground">Announcements</h1>
            <p className="text-sm text-body">Messages on your reps' phones, and the events your team is going to.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {([
              { icon: 'bolt' as IconName, n: liveN, label: 'live now', color: '#10b981' },
              { icon: 'send' as IconName, n: sentThisWeek, label: 'sent this week', color: 'var(--brand-600)' },
              { icon: 'calendar' as IconName, n: upcoming.length, label: 'upcoming events', color: '#0ea5e9' },
            ]).map((s, i) => (
              <motion.div key={s.label} initial={{ opacity: 0, y: 8, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ ...spring, delay: 0.15 + i * 0.07 }} whileHover={{ y: -2 }}
                className="flex items-center gap-2.5 rounded-2xl border bg-card/80 px-3 py-2 shadow-card backdrop-blur">
                <span className="flex size-8 items-center justify-center rounded-xl" style={{ background: `color-mix(in oklab, ${s.color} 14%, transparent)`, color: s.color }}>
                  <Icon name={s.icon} size={15} />
                </span>
                <span>
                  <span className="es-num block text-lg font-black leading-5 text-foreground"><AnimatedNumber value={s.n} duration={700} /></span>
                  <span className="block text-[11px] font-semibold text-muted-foreground">{s.label}</span>
                </span>
              </motion.div>
            ))}
          </div>
        </div>
        <div role="tablist" aria-label="Announcements or events" className="relative mt-5 inline-flex rounded-2xl bg-card/80 p-1 shadow-card backdrop-blur">
          {([['announce', 'Announcements', 'megaphone', liveN], ['events', 'Events', 'calendar', upcoming.length]] as const).map(([v, label, icon, n]) => (
            <motion.button key={v} role="tab" aria-selected={tab === v} onClick={() => setTab(v)} whileTap={{ scale: 0.96 }}
              className={cx('relative inline-flex cursor-pointer items-center gap-2 rounded-xl px-4 py-2 text-sm font-bold transition-colors', tab === v ? 'text-white' : 'text-muted-foreground hover:text-foreground')}>
              {tab === v ? <motion.span layoutId="es-bc-tab" className="absolute inset-0 rounded-xl bg-brand-600 shadow-active" transition={spring} /> : null}
              <Icon name={icon} size={15} className="relative" />
              <span className="relative">{label}</span>
              <span className={cx('relative min-w-6 rounded-full px-1.5 text-center text-[11px] font-black leading-5', tab === v ? 'bg-black/25 text-white' : 'bg-brand-50 text-brand-700')}>{n}</span>
            </motion.button>
          ))}
        </div>
      </motion.section>

      <AnimatePresence mode="wait" initial={false}>
        {tab === 'events' ? (
          <motion.div key="events" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -24 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}>
            <EventPlanner openId={search.event} onOpen={openEvent} />
          </motion.div>
        ) : (
          <motion.div key="announce" initial={{ opacity: 0, x: -24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 24 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_400px]">
              {/* composer */}
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.05 }}>
                <Card className="overflow-hidden rounded-3xl">
                  <div className="flex items-center gap-3 border-b px-5 py-4">
                    <span className="flex size-9 items-center justify-center rounded-xl bg-brand-50 text-brand-700"><Icon name="pencil" size={16} /></span>
                    <div className="min-w-0 flex-1">
                      <div className="text-[15px] font-extrabold text-foreground">New announcement</div>
                      <div className="text-xs text-muted-foreground">Write it, pick who gets it, check the preview, send</div>
                    </div>
                  </div>
                  <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_260px]">
                    <div className="min-w-0 space-y-5">
                      {/* the message */}
                      <div className="group/msg rounded-2xl border bg-card shadow-xs transition-[border-color,box-shadow] focus-within:border-brand-400 focus-within:shadow-[0_0_0_4px_color-mix(in_oklab,var(--brand-400)_16%,transparent)]">
                        <textarea rows={7} maxLength={MAX} value={message} onChange={(e) => setMessage(e.target.value)} aria-label="Message"
                          placeholder="What should the team know? e.g. Stand-up at booth B12 in 15 minutes."
                          className="block w-full resize-y rounded-t-2xl bg-transparent px-4 pt-3.5 text-[14.5px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground" />
                        <div className="flex items-center gap-2 px-3 pb-2.5 pt-1">
                          <Icon name="megaphone" size={14} className="text-muted-foreground transition-colors group-focus-within/msg:text-brand-600" />
                          <span className="text-[11.5px] text-muted-foreground">Line breaks are kept on the phone</span>
                          <span className="flex-1" />
                          <AnimatePresence>{message ? (
                            <motion.button initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }} onClick={() => setMessage('')}
                              className="cursor-pointer rounded-full px-2 py-0.5 text-[11px] font-bold text-muted-foreground hover:bg-secondary hover:text-foreground">Clear</motion.button>
                          ) : null}</AnimatePresence>
                          <CharRing n={message.length} />
                        </div>
                      </div>

                      {/* about an event */}
                      <div className="rounded-2xl border border-dashed p-3.5">
                        <div className="mb-2.5 flex items-center gap-2 text-[13px] font-bold text-foreground">
                          <span className="flex size-6 items-center justify-center rounded-lg bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="calendar" size={13} /></span>
                          About an event?<span className="text-xs font-normal text-muted-foreground">optional — shows on the event and in the log</span>
                        </div>
                        <AnimatePresence mode="wait" initial={false}>
                          {linked ? (
                            <motion.div key="linked" initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }} transition={spring}
                              className="flex flex-wrap items-center gap-2 rounded-xl bg-brand-50 p-2.5">
                              <motion.span initial={{ rotate: -20 }} animate={{ rotate: 0 }} transition={spring} className="flex size-10 flex-col items-center justify-center rounded-lg bg-brand-600 text-white">
                                {linked.startDate ? <>
                                  <span className="text-[8px] font-bold uppercase leading-none opacity-80">{fromDayKey(linked.startDate).toLocaleDateString([], { month: 'short' })}</span>
                                  <span className="text-sm font-black leading-4">{fromDayKey(linked.startDate).getDate()}</span>
                                </> : <Icon name="calendar" size={15} />}
                              </motion.span>
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-bold text-foreground">{linked.name}</div>
                                <div className="truncate text-xs text-body">{fmtRange(linked.startDate, linked.endDate)}{linked.city ? ` · ${linked.city}` : ''}</div>
                              </div>
                              <Button size="sm" variant="ghost" icon="refresh" onClick={() => setMessage(eventMessage(linked))} title="Replace the message with the event's details">Fill from event</Button>
                              <Button size="sm" variant="ghost" icon="external" onClick={() => openEvent(linked.id)}>Open</Button>
                              <motion.button whileHover={{ rotate: 90 }} whileTap={{ scale: 0.8 }} onClick={() => setEventId(undefined)} aria-label="Unlink the event" className="flex size-7 cursor-pointer items-center justify-center rounded-full text-muted-foreground hover:bg-card hover:text-foreground"><Icon name="x" size={13} /></motion.button>
                            </motion.div>
                          ) : (
                            <motion.div key="pick" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: MOTION.fast }} className="flex flex-wrap items-center gap-2">
                              {upcoming.length ? (
                                <select value="" aria-label="Link an event" onChange={(e) => { const ev = upcoming.find((x) => x.id === e.target.value); if (!ev) return; setEventId(ev.id); if (!message.trim()) setMessage(eventMessage(ev)); }}
                                  className="h-9 min-w-[220px] flex-1 cursor-pointer rounded-xl border border-input bg-card px-3 text-sm">
                                  <option value="">Link one of your events…</option>
                                  {upcoming.map((ev) => <option key={ev.id} value={ev.id}>{ev.name} · {fmtRange(ev.startDate, ev.endDate)}</option>)}
                                </select>
                              ) : null}
                              <motion.span whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.96 }} className="inline-flex">
                                <Button size="sm" icon="sparkles" loading={extracting} disabled={!message.trim()} onClick={saveAsEvent}
                                  title={message.trim() ? 'The AI reads the event details out of your message; you check them before saving' : 'Write about the event first'}>
                                  Save as a new event
                                </Button>
                              </motion.span>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      {/* priority */}
                      <div>
                        <div className="mb-2 text-[12px] font-bold uppercase tracking-[0.06em] text-muted-foreground">Priority</div>
                        <div className="flex flex-col gap-2 sm:flex-row">
                          <ChoiceCard layoutId="es-prio" on={priority === 'normal'} onClick={() => setPriority('normal')} icon="megaphone" title="Normal" sub="A heads-up at the top of the feed" color="var(--brand-600)" />
                          <ChoiceCard layoutId="es-prio" on={priority === 'urgent'} onClick={() => setPriority('urgent')} icon="alert" title="Urgent" sub="Red, for things that can't wait" color={URGENT} />
                        </div>
                      </div>

                      {/* audience */}
                      <div>
                        <div className="mb-2 text-[12px] font-bold uppercase tracking-[0.06em] text-muted-foreground">Send to</div>
                        <div className="flex flex-col gap-2 sm:flex-row">
                          <ChoiceCard layoutId="es-aud" on={audience === 'everyone'} onClick={() => setAudience('everyone')} icon="users" title="Everyone" sub={`All ${reps.length} rep${reps.length === 1 ? '' : 's'}`} color="#10b981" />
                          <ChoiceCard layoutId="es-aud" on={audience === 'some'} onClick={() => setAudience('some')} icon="userPlus" title="Chosen reps" sub={recipients.length ? `${recipients.length} picked` : 'Pick who gets it'} color="#0ea5e9" />
                        </div>
                        <AnimatePresence initial={false}>
                          {audience === 'some' ? (
                            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="overflow-hidden">
                              <div className="mt-3 rounded-2xl border bg-card p-2 shadow-card">
                                <div className="flex items-center gap-2 p-1">
                                  <SearchField value={repQuery} onChange={setRepQuery} placeholder="Find a rep…" className="h-9 flex-1" />
                                  <Button size="sm" variant="ghost" onClick={() => setPicked(Object.fromEntries(shownReps.map((u) => [u.id, true])))}>All</Button>
                                  <Button size="sm" variant="ghost" disabled={!recipients.length} onClick={() => setPicked({})}>None</Button>
                                </div>
                                <ul className="mt-1 max-h-60 space-y-0.5 overflow-y-auto">
                                  {shownReps.length ? shownReps.map((u, i) => {
                                    const on = !!picked[u.id];
                                    return (
                                      <motion.li key={u.id} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.fast, delay: Math.min(i, 10) * 0.02 }}>
                                        <button type="button" onClick={() => setPicked((p) => ({ ...p, [u.id]: !p[u.id] }))} aria-pressed={on}
                                          className={cx('flex w-full cursor-pointer items-center gap-3 rounded-xl px-2 py-1.5 text-left transition-colors', on ? 'bg-sky-50 dark:bg-sky-500/10' : 'hover:bg-secondary/60')}>
                                          <span className="relative">
                                            <Avatar src={u.photoUrl || null} name={u.name} size="xs" className="size-8 border-0 text-[10px] shadow-none" />
                                            <AnimatePresence>{on ? <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={spring} className="absolute -bottom-0.5 -right-0.5 flex size-4 items-center justify-center rounded-full bg-sky-500 text-white ring-2 ring-card"><Icon name="check" size={9} strokeWidth={3} /></motion.span> : null}</AnimatePresence>
                                          </span>
                                          <span className="min-w-0 flex-1">
                                            <span className="block truncate text-sm font-semibold text-foreground">{u.name}</span>
                                            <span className="block truncate text-xs text-muted-foreground">{u.email}</span>
                                          </span>
                                        </button>
                                      </motion.li>
                                    );
                                  }) : <li className="p-3 text-sm text-muted-foreground">{reps.length ? 'No one matches.' : 'No reps yet.'}</li>}
                                </ul>
                              </div>
                            </motion.div>
                          ) : null}
                        </AnimatePresence>
                      </div>
                    </div>

                    {/* preview + send */}
                    <div className="flex flex-col items-center gap-4">
                      <PhonePreview message={message} urgent={priority === 'urgent'} eventName={linked?.name} />
                      <motion.div className="w-full" whileHover={canSend ? { scale: 1.02 } : {}} whileTap={canSend ? { scale: 0.97 } : {}}>
                        <button type="button" onClick={send} disabled={!canSend || sending}
                          className={cx('relative flex h-12 w-full cursor-pointer items-center justify-center gap-2 overflow-hidden rounded-2xl text-[15px] font-extrabold text-white shadow-active transition-[background-color,opacity] disabled:cursor-not-allowed disabled:opacity-50',
                            justSent ? 'bg-emerald-600' : priority === 'urgent' ? 'bg-rose-600 hover:bg-rose-700' : 'bg-brand-600 hover:bg-brand-700')}>
                          <AnimatePresence mode="wait" initial={false}>
                            {justSent ? (
                              <motion.span key="sent" initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ opacity: 0 }} transition={spring} className="flex items-center gap-2">
                                <Icon name="check" size={18} strokeWidth={3} />Sent!
                              </motion.span>
                            ) : sending ? (
                              <motion.span key="sending" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-2">
                                <motion.span animate={{ x: [0, 40], y: [0, -24], opacity: [1, 0] }} transition={{ duration: 0.7, repeat: Infinity }} className="flex"><Icon name="send" size={17} /></motion.span>Sending…
                              </motion.span>
                            ) : (
                              <motion.span key="idle" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="group/send flex items-center gap-2">
                                <Icon name="send" size={17} className="transition-transform duration-300 group-hover/send:-translate-y-0.5 group-hover/send:translate-x-1 group-hover/send:-rotate-12" />
                                Send {audience === 'some' ? (recipients.length ? `to ${recipients.length}` : '') : 'to everyone'}
                              </motion.span>
                            )}
                          </AnimatePresence>
                        </button>
                      </motion.div>
                      <p className="text-center text-[11px] text-muted-foreground">{!message.trim() ? 'Write a message to send' : audience === 'some' && !recipients.length ? 'Pick at least one rep' : 'It stays on phones until you end it'}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>

              {/* live now */}
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.12 }}>
                <Card className="h-fit overflow-hidden rounded-3xl">
                  <div className="flex items-center gap-3 border-b px-5 py-4">
                    <span className="flex size-9 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15">
                      <Icon name="bolt" size={16} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="text-[15px] font-extrabold text-foreground">Live now</div>
                      <div className="text-xs text-muted-foreground">Showing on phones until you end them</div>
                    </div>
                    <span className="es-num rounded-full bg-secondary px-2.5 text-sm font-black leading-7 text-body"><AnimatedNumber value={liveN} duration={500} /></span>
                  </div>
                  {live.error ? <div className="p-4"><ErrorState error={live.error} onRetry={() => live.refetch()} /></div>
                    : !liveN ? (
                      <div className="flex flex-col items-center px-6 py-10 text-center">
                        <motion.span animate={{ y: [0, -6, 0] }} transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
                          className="mb-3 flex size-12 items-center justify-center rounded-2xl bg-secondary text-muted-foreground"><LiveIcon name="megaphone" size={20} /></motion.span>
                        <div className="text-sm font-bold text-foreground">Nothing live</div>
                        <p className="mt-1 max-w-[240px] text-xs text-muted-foreground">Announcements you send show here until you end them.</p>
                      </div>
                    ) : (
                      // a calm list: one steady dot per item, two lines of message, who / when, End
                      <motion.ul layoutScroll className="max-h-[520px] divide-y overflow-y-auto">
                        <AnimatePresence initial={false}>
                          {live.data.map((b) => {
                            const urgent = b.priority === 'urgent';
                            const color = urgent ? URGENT : '#10b981';
                            const isOpen = !!liveOpen[b.id];
                            const long = b.message.length > 140 || b.message.split('\n').length > 2;
                            return (
                              <motion.li key={b.id} layout="position" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: MOTION.base, ease: MOTION.ease }} className="px-5 py-3.5 transition-colors hover:bg-secondary/40">
                                <div className="flex items-start gap-3">
                                  <span className="mt-[7px] size-2 shrink-0 rounded-full" style={{ background: color }} title={urgent ? 'Urgent' : 'Live'} aria-hidden />
                                  <div className="min-w-0 flex-1">
                                    <p className={cx('whitespace-pre-line break-words text-[13.5px] leading-5 text-foreground', !isOpen && 'line-clamp-2')}>{b.message}</p>
                                    <div className="mt-1 flex flex-wrap items-center gap-x-1.5 text-[11.5px] text-muted-foreground">
                                      {urgent ? <span className="font-bold text-rose-700 dark:text-rose-400">Urgent ·</span> : null}
                                      <span title={fmtDateTime(b.createdAt)}>{relTime(b.createdAt)}</span>
                                      <span>·</span>
                                      <span className="truncate" title={b.recipients?.map((r) => r.name).join(', ')}>{toWhom(b.recipients)}</span>
                                      {long ? <button onClick={() => setLiveOpen((m) => ({ ...m, [b.id]: !m[b.id] }))} className="cursor-pointer font-bold text-brand-700 hover:underline">{isOpen ? 'Less' : 'More'}</button> : null}
                                    </div>
                                  </div>
                                  <button onClick={() => end(b.id)} aria-label="End this announcement" title="End it — it disappears from every phone"
                                    className="shrink-0 cursor-pointer rounded-lg border px-2.5 py-1 text-xs font-bold text-body transition-colors hover:border-rose-200 hover:bg-rose-50 hover:text-rose-700 dark:hover:bg-rose-500/10">
                                    End
                                  </button>
                                </div>
                              </motion.li>
                            );
                          })}
                        </AnimatePresence>
                      </motion.ul>
                    )}
                </Card>
              </motion.div>
            </div>
            <PastAnnouncements onOpenEvent={openEvent}
              onReuse={(b) => { setMessage(b.message); setPriority(b.priority === 'urgent' ? 'urgent' : 'normal'); setEventId(b.eventId ?? undefined); window.scrollTo({ top: 0, behavior: 'smooth' }); }} />
          </motion.div>
        )}
      </AnimatePresence>
      <EventFormModal form={eventForm} onClose={() => setEventForm(null)}
        note="Filled in from your announcement by the AI — check every field before saving. Anything it couldn't find is left empty."
        onSaved={(saved) => setEventId(saved.id)} />
    </>
  );
}

/** Every announcement ever sent — live and ended — as a timeline grouped by day. */
function PastAnnouncements({ onOpenEvent, onReuse }: { onOpenEvent: (id: string) => void; onReuse: (b: BroadcastLog) => void }) {
  // a day picked on the calendar shows just that day (every announcement sent then, from the server)
  const [day, setDay] = useState<string | undefined>(undefined);
  const latest = useQuery({ ...q.broadcastHistory(), enabled: !day });
  const onDay = useQuery({ ...q.broadcastHistoryDay(day ?? ''), enabled: !!day });
  const history = day ? onDay : latest;
  const [show, setShow] = useState<'all' | 'live' | 'ended'>('all');
  const [text, setText] = useState('');
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const all = history.data ?? [];
  const shown = useMemo(() => all.filter((b) => (show === 'all' || (show === 'live' ? b.isActive : !b.isActive))
    && (!text || [b.message, b.sentBy, b.eventName, ...b.recipients.map((r) => r.name)].join(' ').toLowerCase().includes(text.toLowerCase()))), [all, show, text]);
  const n = { all: all.length, live: all.filter((b) => b.isActive).length, ended: all.filter((b) => !b.isActive).length };
  const groups = useMemo(() => {
    const out: { key: string; items: BroadcastLog[] }[] = [];
    shown.forEach((b) => {
      const key = b.createdAt ? dayKey(new Date(b.createdAt)) : 'unknown';
      const last = out[out.length - 1];
      if (last && last.key === key) last.items.push(b); else out.push({ key, items: [b] });
    });
    return out;
  }, [shown]);
  const today = dayKey(new Date());
  const dayTitle = (k: string) => (k === today ? 'Today' : k === dayKey(addDays(new Date(), -1)) ? 'Yesterday' : k === 'unknown' ? 'Unknown date' : fromDayKey(k).toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' }));

  return (
    <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.2 }}>
      <Card className="mt-5 overflow-hidden rounded-3xl">
        <div className="flex flex-wrap items-center gap-3 border-b px-5 py-4">
          <span className="flex size-9 items-center justify-center rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-500/15"><Icon name="clock" size={16} /></span>
          <div className="min-w-0 flex-1">
            <div className="text-[15px] font-extrabold text-foreground">Past announcements</div>
            <div className="text-xs text-muted-foreground">Everything sent from this console, newest first</div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-b px-5 py-3">
          <SearchField value={text} onChange={setText} placeholder="Search messages, people, events…" className="w-full sm:w-[300px]" />
          <div role="tablist" className="relative inline-flex rounded-full bg-secondary p-1">
            {(['all', 'live', 'ended'] as const).map((s) => (
              <button key={s} role="tab" aria-selected={show === s} onClick={() => setShow(s)}
                className={cx('relative inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-full pl-3 pr-1.5 text-[12.5px] font-bold transition-colors', show === s ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
                {show === s ? <motion.span layoutId="es-log-filter" className="absolute inset-0 rounded-full bg-card shadow-card" transition={spring} /> : null}
                <span className="relative">{s === 'all' ? 'All' : s === 'live' ? 'Live' : 'Ended'}</span>
                <span className={cx('relative min-w-5 rounded-full px-1.5 text-center text-[11px] font-black leading-[18px]', show === s ? 'bg-brand-600 text-white' : 'bg-card text-body')}>
                  <AnimatedNumber value={n[s]} duration={400} />
                </span>
              </button>
            ))}
          </div>
          <ActivityCalendar value={day} onChange={setDay} align="start" noun="announcement" counts={q.broadcastCalendar} kinds={ANNOUNCE_KINDS} />
        </div>
        <AnimatePresence initial={false}>
          {day ? (
            <motion.div key={day} initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}
              className="overflow-hidden border-b bg-brand-50/50">
              <div className="flex flex-wrap items-center gap-2 px-5 py-2.5 text-sm">
                <Icon name="calendar" size={14} className="text-brand-600" />
                <span className="font-bold text-foreground">{fromDayKey(day).toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })}</span>
                <span className="text-muted-foreground">· {history.isLoading ? 'loading…' : `${all.length} announcement${all.length === 1 ? '' : 's'} sent`}</span>
                <span className="flex-1" />
                <Button size="sm" variant="ghost" onClick={() => setDay(dayKey(addDays(fromDayKey(day), -1)))}><Icon name="chevronRight" size={14} className="rotate-180" />Previous day</Button>
                <Button size="sm" variant="ghost" disabled={day >= today} onClick={() => setDay(dayKey(addDays(fromDayKey(day), 1)))}>Next day<Icon name="chevronRight" size={14} /></Button>
                <Button size="sm" variant="ghost" icon="x" onClick={() => setDay(undefined)}>All dates</Button>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        {history.error ? <div className="p-4"><ErrorState error={history.error} onRetry={() => history.refetch()} /></div>
          : history.isLoading ? <div className="space-y-3 p-5">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-20 w-full rounded-2xl" />)}</div>
            : !shown.length ? <EmptyState icon={day ? 'calendar' : 'megaphone'}
                title={all.length ? 'Nothing matches' : day ? 'Nothing sent that day' : 'No announcements yet'}
                description={all.length ? 'Try another filter or search.' : day ? 'Pick another day on the calendar, or show all dates.' : 'Announcements you send are kept here.'}
                action={day && !all.length ? <Button icon="x" onClick={() => setDay(undefined)}>All dates</Button> : undefined} />
              : (
                <div className="space-y-5 p-5">
                  {groups.map((g, gi) => {
                    const before = groups.slice(0, gi).reduce((m, x) => m + x.items.length, 0);
                    return (
                      <section key={g.key}>
                        <div className="mb-2.5 flex items-center gap-2">
                          <span className="text-[13px] font-extrabold text-foreground">{dayTitle(g.key)}</span>
                          <span className="h-px flex-1 bg-border" />
                          <span className="es-num rounded-full bg-secondary px-2 text-[11px] font-black leading-5 text-body">{g.items.length}</span>
                        </div>
                        <ol className="relative">
                          {g.items.map((b, i) => {
                            const urgent = b.priority === 'urgent';
                            const color = urgent ? URGENT : 'var(--brand-600)';
                            const long = b.message.length > 220 || b.message.split('\n').length > 3;
                            const k = before + i;
                            return (
                              <motion.li key={b.id} className="relative flex gap-3 pb-2.5 last:pb-0"
                                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(k, 14) * 0.03 }}>
                                {i < g.items.length - 1 ? <span className="absolute bottom-0 left-[17px] top-10 w-[2px] rounded-full bg-border/70" aria-hidden /> : null}
                                <motion.span initial={{ scale: 0.4, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={{ ...spring, delay: 0.05 + Math.min(k, 14) * 0.03 }}
                                  className="relative z-[1] mt-1.5 flex size-9 shrink-0 items-center justify-center rounded-xl ring-4 ring-card"
                                  style={{ background: `color-mix(in oklab, ${color} 13%, var(--card))`, color }}>
                                  <Icon name={urgent ? 'alert' : 'megaphone'} size={16} />
                                </motion.span>
                                <motion.div whileHover={{ y: -2 }} className="group min-w-0 flex-1 rounded-2xl border bg-card px-3.5 py-3 shadow-card transition-[border-color,box-shadow] hover:border-brand-200 hover:shadow-active">
                                  <div className="mb-1 flex flex-wrap items-center gap-1.5">
                                    {b.isActive
                                      ? <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 text-[10.5px] font-bold leading-5 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300"><span className="size-1.5 animate-pulse rounded-full bg-emerald-500" />Live</span>
                                      : <span className="rounded-full bg-secondary px-2 text-[10.5px] font-bold leading-5 text-muted-foreground">Ended{b.endedAt ? ` ${relTime(b.endedAt)}` : ''}</span>}
                                    {urgent ? <span className="rounded-full px-2 text-[10.5px] font-black uppercase leading-5 tracking-[0.05em] text-white" style={{ background: deep(URGENT) }}>Urgent</span> : null}
                                    {b.eventId && b.eventName ? (
                                      <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={() => onOpenEvent(b.eventId!)}
                                        className="inline-flex cursor-pointer items-center gap-1 rounded-full border border-brand-100 bg-brand-50 px-2 text-[10.5px] font-bold leading-5 text-brand-700 hover:border-brand-300">
                                        <Icon name="calendar" size={10} />{b.eventName}
                                      </motion.button>
                                    ) : null}
                                    <span className="ml-auto text-[11px] font-semibold text-muted-foreground" title={b.createdAt ? fmtDateTime(b.createdAt) : undefined}>
                                      {b.createdAt ? new Date(b.createdAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : ''}
                                    </span>
                                  </div>
                                  <motion.p layout className={cx('whitespace-pre-line break-words text-sm leading-relaxed text-foreground', long && !open[b.id] && 'line-clamp-3')}>{b.message}</motion.p>
                                  {long ? (
                                    <button onClick={() => setOpen((m) => ({ ...m, [b.id]: !m[b.id] }))} className="mt-0.5 inline-flex cursor-pointer items-center gap-0.5 text-xs font-bold text-brand-700 hover:underline">
                                      {open[b.id] ? 'Show less' : 'Show more'}<Icon name="chevronDown" size={12} className={cx('transition-transform duration-200', open[b.id] && 'rotate-180')} />
                                    </button>
                                  ) : null}
                                  <div className="mt-2 flex items-center gap-2 border-t pt-2">
                                    <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground" title={b.recipients.map((r) => r.name).join(', ') || undefined}>
                                      {b.sentBy ? `By ${b.sentBy} · ` : ''}{toWhom(b.recipients)}
                                    </span>
                                    <motion.button whileTap={{ scale: 0.92 }} onClick={() => onReuse(b)} title="Put this message back in the composer"
                                      className="group/re inline-flex cursor-pointer items-center gap-1 rounded-full px-2.5 py-1 text-xs font-bold text-brand-700 transition-colors hover:bg-brand-50">
                                      <Icon name="refresh" size={12} className="transition-transform duration-500 group-hover/re:-rotate-180" />Use again
                                    </motion.button>
                                  </div>
                                </motion.div>
                              </motion.li>
                            );
                          })}
                        </ol>
                      </section>
                    );
                  })}
                </div>
              )}
      </Card>
    </motion.div>
  );
}
