// /exports — every file is built from the live database when you click:
//   contacts (all · hot & warm · one event · one rep · a list)  GET /api/admin/export[?event|userId|minScore|listId]
//                                                               (server-side CSV, no row cap, same columns every time)
//   team        GET /api/admin/users            activity  GET /api/admin/activity?limit=500[&start&end]
// Pickers come from the API too: every event (GET /api/admin/events), people (users), lists.
import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useQuery, useSuspenseQuery } from '@tanstack/react-query';
import { q } from '../lib/queries';
import { MAX_ACTIVITY_ITEMS, api, saveBlob } from '../lib/api';
import { addDays, dayKey, localDayRange } from '../lib/dates';
import { Icon, type IconName } from '../ui/icons';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import { PageHeader, Select, deep, errorMessage, fmtNum, ink, useUi } from '../ui/kit';
import { CrmSection } from '../features/exports/CrmSection';

const csvCell = (v: unknown) => { const s = v == null ? '' : String(v); return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
const stamp = () => dayKey(new Date());

/**
 * One export: a tinted header with its icon floating over soft bubbles, what's in the file, how many
 * rows (counting up), an optional picker, and a Download button that goes downloading → done.
 */
function ExportCard({ icon, tint, title, description, format, count, countLabel = 'row', children, onExport, disabled, index = 0 }: {
  icon: IconName; tint: string; title: string; description: string; format: 'CSV' | 'JSON'; count?: number | undefined; countLabel?: string;
  children?: React.ReactNode; onExport: () => Promise<void>; disabled?: boolean; index?: number;
}) {
  const [state, setState] = useState<'idle' | 'busy' | 'done'>('idle');
  const { toast } = useUi();
  useEffect(() => { if (state !== 'done') return; const t = setTimeout(() => setState('idle'), 1800); return () => clearTimeout(t); }, [state]);
  const run = async () => {
    setState('busy');
    try { await onExport(); setState('done'); } catch (e) { toast('error', "Couldn't export", errorMessage(e)); setState('idle'); }
  };
  return (
    <motion.div initial={{ opacity: 0, y: 18, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: MOTION.base * 1.4, ease: MOTION.ease, delay: 0.06 + index * 0.06 }}
      whileHover={{ y: -4 }} className="group h-full">
      <div className="relative flex h-full flex-col overflow-hidden rounded-3xl border bg-card shadow-card transition-[border-color,box-shadow] duration-300 group-hover:shadow-active"
        style={{ ['--tint' as string]: tint }}>
        {/* tinted header with drifting bubbles and the floating icon */}
        <div className="relative isolate overflow-hidden px-5 pb-4 pt-5" style={{ background: `linear-gradient(135deg, color-mix(in oklab, ${tint} 16%, var(--card)), color-mix(in oklab, ${tint} 4%, var(--card)))` }}>
          <span aria-hidden className="es-bubble pointer-events-none absolute -right-8 -top-10 -z-10 size-28 rounded-full" style={{ background: `color-mix(in oklab, ${tint} 16%, transparent)` }} />
          <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-12 left-16 -z-10 size-20 rounded-full" style={{ background: `color-mix(in oklab, ${tint} 10%, transparent)` }} />
          <div className="flex items-start justify-between gap-3">
            <motion.span initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 16, delay: 0.15 + index * 0.06 }}
              className="es-wiggle-hover flex size-12 items-center justify-center rounded-2xl bg-card shadow-card" style={{ color: ink(tint) }}>
              <motion.span animate={{ y: [0, -3, 0] }} transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: index * 0.3 }} className="flex"><Icon name={icon} size={21} /></motion.span>
            </motion.span>
            <span className="inline-flex items-center gap-1 rounded-full bg-card/80 px-2 py-0.5 text-[10.5px] font-black tracking-[0.06em] shadow-card" style={{ color: ink(tint) }}>
              <Icon name={format === 'CSV' ? 'columns' : 'file'} size={11} />{format}
            </span>
          </div>
          <h3 className="mt-3 text-[15px] font-extrabold text-foreground">{title}</h3>
          <div className="mt-0.5 flex items-baseline gap-1.5">
            {count != null ? (
              <>
                <span className="es-num text-[22px] font-black leading-7" style={{ color: ink(tint) }}><AnimatedNumber value={count} duration={900} /></span>
                <span className="text-xs font-semibold text-muted-foreground">{countLabel}{count === 1 ? '' : 's'}</span>
              </>
            ) : <span className="text-xs font-semibold text-muted-foreground">{children ? 'Pick one below' : 'Built when you click'}</span>}
          </div>
        </div>

        <div className="flex flex-1 flex-col p-5 pt-4">
          <p className="flex-1 text-sm leading-relaxed text-body">{description}</p>
          {children ? <div className="mt-3">{children}</div> : null}
          <motion.button type="button" onClick={run} disabled={disabled || state === 'busy'} whileTap={disabled ? {} : { scale: 0.97 }}
            className="relative mt-4 flex h-11 w-full cursor-pointer items-center justify-center gap-2 overflow-hidden rounded-2xl text-sm font-extrabold text-white shadow-card transition-[filter,opacity] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
            style={{ background: deep(state === 'done' ? '#10b981' : tint) }}>
            {state === 'busy' ? <motion.span className="absolute inset-y-0 left-0 w-1/3 bg-white/25" animate={{ x: ['-100%', '300%'] }} transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }} /> : null}
            <AnimatePresence mode="wait" initial={false}>
              {state === 'done' ? (
                <motion.span key="done" initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ opacity: 0 }} transition={{ type: 'spring', stiffness: 460, damping: 20 }} className="relative flex items-center gap-2">
                  <Icon name="check" size={17} strokeWidth={3} />Downloaded
                </motion.span>
              ) : state === 'busy' ? (
                <motion.span key="busy" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative flex items-center gap-2">
                  <motion.span animate={{ y: [0, 4, 0] }} transition={{ duration: 0.6, repeat: Infinity }} className="flex"><Icon name="download" size={17} /></motion.span>Building file…
                </motion.span>
              ) : (
                <motion.span key="idle" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="group/dl relative flex items-center gap-2">
                  <Icon name="download" size={17} className="transition-transform duration-300 group-hover/dl:translate-y-0.5" />
                  {disabled ? 'Pick one first' : `Download ${format}`}
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}

export function ExportsComponent() {
  const { toast } = useUi();
  const dash = useSuspenseQuery(q.dashboard(7));
  const users = useSuspenseQuery(q.users());
  const lists = useSuspenseQuery(q.lists());
  // every event with contacts (not just the dashboard's top 6)
  const allEvents = useQuery(q.adminEvents());
  const events = (allEvents.data ?? []).filter((e) => e.contacts > 0).sort((a, b) => b.contacts - a.contacts);
  const [event, setEvent] = useState('');
  const [rep, setRep] = useState('');
  const [list, setList] = useState('');
  const [activityDays, setActivityDays] = useState<'latest' | '7' | '30' | '90'>('latest');
  // set when the last period export hit the server's cap (it holds only the newest events)
  const [activityCapped, setActivityCapped] = useState<number | null>(null);
  const d = dash.data;

  /** a contacts CSV from the server; tells the admin how many rows it really had */
  const serverCsv = async (f: Parameters<typeof api.exportAll>[0]) => {
    const { blob, filename, rows } = await api.exportAll(f);
    if (rows === 0) { toast('info', 'Nothing to export', 'No contacts match.'); return; }
    saveBlob(blob, filename);
    toast('success', rows != null ? `Exported ${fmtNum(rows)} contact${rows === 1 ? '' : 's'}` : 'Export downloaded', filename);
  };

  return (
    <>
      <PageHeader title="Exports" description="Download your data as spreadsheets (CSV opens in Excel and Google Sheets) or JSON. Every file is built from the live database when you click — never cut off." />
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <ExportCard index={0} icon="contacts" tint="var(--brand-600)" title="All contacts" format="CSV" count={d?.totalContacts}
          description="Every contact in the workspace with title, company, email, phone, LinkedIn, event, owner, score, level, source and notes."
          onExport={() => serverCsv({})} />

        <ExportCard index={1} icon="flame" tint="#ef4444" title="Hot & warm leads" format="CSV" count={d ? d.levels.hot + d.levels.warm : undefined}
          description="Every lead scoring 70 or more, best first — ready to import into your CRM."
          onExport={() => serverCsv({ minScore: 70 })} />

        <ExportCard index={2} icon="calendar" tint="#0ea5e9" title="One event" format="CSV" count={events.find((e) => e.name === event)?.contacts}
          description="Everyone met at an event." disabled={!event}
          onExport={() => serverCsv({ event })}>
          <Select value={event} onChange={(e) => setEvent(e.target.value)} aria-label="Event" disabled={allEvents.isLoading}>
            <option value="">{allEvents.isLoading ? 'Loading events…' : allEvents.isError ? "Couldn't load events" : events.length ? 'Choose an event…' : 'No events with contacts yet'}</option>
            {events.map((e) => <option key={e.key} value={e.name}>{e.name} ({fmtNum(e.contacts)})</option>)}
          </Select>
        </ExportCard>

        <ExportCard index={3} icon="user" tint="#10b981" title="One rep's contacts" format="CSV" count={users.data?.find((u) => u.id === rep)?.contactsCount}
          description="Everything a single team member captured." disabled={!rep}
          onExport={() => serverCsv({ userId: rep })}>
          <Select value={rep} onChange={(e) => setRep(e.target.value)} aria-label="Rep">
            <option value="">{users.data?.length ? 'Choose a person…' : 'No one yet'}</option>
            {(users.data || []).map((u) => <option key={u.id} value={u.id}>{u.name} ({fmtNum(u.contactsCount)})</option>)}
          </Select>
        </ExportCard>

        <ExportCard index={4} icon="list" tint="#d946ef" title="A list" format="CSV" count={lists.data?.find((l) => l.id === list)?.memberCount}
          description="The members of one of your lists." disabled={!list}
          onExport={() => serverCsv({ listId: list })}>
          <Select value={list} onChange={(e) => setList(e.target.value)} aria-label="List">
            <option value="">{lists.data?.length ? 'Choose a list…' : 'No lists yet'}</option>
            {(lists.data || []).map((l) => <option key={l.id} value={l.id}>{l.name} ({fmtNum(l.memberCount)})</option>)}
          </Select>
        </ExportCard>

        <ExportCard index={5} icon="users" tint="#f59e0b" title="Team" format="CSV" count={users.data?.length} countLabel="person"
          description="Everyone who can sign in, with role, contacts, added today, last activity and when they joined."
          onExport={async () => {
            const u = await api.users();
            const lines = [['Name', 'Email', 'Role', 'Contacts', 'Added today', 'Last activity', 'Joined'].join(','),
              ...u.map((x) => [x.name, x.email, x.role, x.contactsCount, x.scansToday, x.lastActive || '', x.createdAt || ''].map(csvCell).join(','))];
            saveBlob(new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' }), `eventscout-team-${stamp()}.csv`);
            toast('success', `Exported ${u.length} ${u.length === 1 ? 'person' : 'people'}`);
          }} />

        <ExportCard index={6} icon="activity" tint="#6366f1" title="Activity log" format="JSON"
          description="Workspace events — contacts added, scores, pipelines, merges, announcements, files, new accounts."
          onExport={async () => {
            // "latest" = the newest 500; a period = everything in it (up to the server's 5,000), from local midnight that many days ago until now
            const range = activityDays === 'latest' ? undefined
              : { start: localDayRange(dayKey(addDays(new Date(), -(Number(activityDays) - 1)))).start, end: new Date().toISOString() };
            const cap = range ? MAX_ACTIVITY_ITEMS : 500;
            const a = await api.activity(cap, range);
            setActivityCapped(range && a.length >= cap ? a.length : null);
            if (!a.length) { toast('info', 'Nothing to export', 'No activity in that period.'); return; }
            saveBlob(new Blob([JSON.stringify({ format: 'eventscout.activity.v1', exportedAt: new Date().toISOString(), period: activityDays === 'latest' ? 'latest' : `last ${activityDays} days`, items: a }, null, 2)],
              { type: 'application/json' }), `eventscout-activity-${activityDays === 'latest' ? 'latest' : `${activityDays}d`}-${stamp()}.json`);
            toast(a.length >= cap ? 'info' : 'success', `Exported ${a.length.toLocaleString()} events`,
              a.length >= cap ? `That's the most one export holds (the newest ${cap.toLocaleString()}) — pick a shorter period for everything in it.` : undefined);
          }}>
          <Select value={activityDays} onChange={(e) => { setActivityDays(e.target.value as typeof activityDays); setActivityCapped(null); }} aria-label="Period">
            <option value="latest">The latest 500 events</option>
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </Select>
          {activityCapped != null ? (
            <p className="mt-1.5 flex items-center gap-1.5 text-xs text-warning"><Icon name="alert" size={12} />
              Only the newest {activityCapped.toLocaleString()} events fit in one export — pick a shorter period for the rest.</p>
          ) : null}
        </ExportCard>
      </div>
      <p className="mt-4 flex items-center gap-1.5 text-xs text-muted-foreground"><Icon name="info" size={13} />Contact files are built on the server with every matching row, and use the same columns each time.</p>
      <CrmSection totalContacts={d?.totalContacts} hotWarm={d ? d.levels.hot + d.levels.warm : undefined}
        events={events} users={users.data || []} lists={lists.data || []} />
    </>
  );
}
