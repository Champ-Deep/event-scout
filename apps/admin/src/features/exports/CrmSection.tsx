// Exports → Send to CRM: one card per CRM (HubSpot · Salesforce · Zoho CRM · Pipedrive), all sharing one
// "which contacts" picker (same filters as the contact exports above).
//   Send     POST /api/admin/crm/{crm}/push → job, polled at GET /api/admin/crm/jobs/{id} until it ends
//   CSV      GET  /api/admin/crm/{crm}/csv  (that CRM's import column names; works without credentials)
//   Test     POST /api/admin/crm/{crm}/test
// Credentials are server env vars only; a CRM that isn't set up says which ones are missing.
import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { q } from '../../lib/queries';
import { applyChange } from '../../lib/changes';
import { api, saveBlob, type ExportFilters } from '../../lib/api';
import type { AdminEvent, AdminUser, ContactList, CrmInfo, CrmJob, CrmKey } from '../../lib/types';
import { Icon } from '../../ui/icons';
import { AnimatedNumber } from '../../ui/micro';
import { MOTION } from '../../ui/tokens';
import { Select, Skeleton, deep, errorMessage, fmtNum, ink, relTime, useUi } from '../../ui/kit';

const TINTS: Record<CrmKey, string> = { hubspot: '#ff7a59', salesforce: '#0ea5e9', zoho: '#e42527', pipedrive: '#10b981' };

type Scope = 'all' | 'hot' | 'event' | 'rep' | 'list';

export function CrmSection({ totalContacts, hotWarm, events, users, lists }: {
  totalContacts: number | undefined; hotWarm: number | undefined;
  events: AdminEvent[]; users: AdminUser[]; lists: ContactList[];
}) {
  const crms = useQuery(q.crms());
  const [scope, setScope] = useState<Scope>('hot');
  const [pick, setPick] = useState('');
  useEffect(() => setPick(''), [scope]);

  const filters: ExportFilters | null =
    scope === 'all' ? {} : scope === 'hot' ? { minScore: 70 }
      : !pick ? null : scope === 'event' ? { event: pick } : scope === 'rep' ? { userId: pick } : { listId: pick };
  const count = scope === 'all' ? totalContacts : scope === 'hot' ? hotWarm
    : scope === 'event' ? events.find((e) => e.name === pick)?.contacts
      : scope === 'rep' ? users.find((u) => u.id === pick)?.contactsCount
        : lists.find((l) => l.id === pick)?.memberCount;

  return (
    <section className="mt-8">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-[15px] font-extrabold text-foreground">Send to your CRM</h2>
          <p className="mt-0.5 text-sm text-body">Push contacts straight into HubSpot, Salesforce, Zoho CRM or Pipedrive — or download a file in that CRM's import format.</p>
        </div>
        <div className="flex w-full flex-wrap gap-2 sm:w-auto">
          <Select value={scope} onChange={(e) => setScope(e.target.value as Scope)} aria-label="Which contacts" className="w-full sm:w-48">
            <option value="hot">Hot & warm leads (70+)</option>
            <option value="all">All contacts</option>
            <option value="event">One event</option>
            <option value="rep">One rep's contacts</option>
            <option value="list">A list</option>
          </Select>
          {scope === 'event' || scope === 'rep' || scope === 'list' ? (
            <Select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Pick one" className="w-full sm:w-56">
              <option value="">{scope === 'event' ? 'Choose an event…' : scope === 'rep' ? 'Choose a person…' : 'Choose a list…'}</option>
              {scope === 'event' ? events.map((e) => <option key={e.key} value={e.name}>{e.name} ({fmtNum(e.contacts)})</option>)
                : scope === 'rep' ? users.map((u) => <option key={u.id} value={u.id}>{u.name} ({fmtNum(u.contactsCount)})</option>)
                  : lists.map((l) => <option key={l.id} value={l.id}>{l.name} ({fmtNum(l.memberCount)})</option>)}
            </Select>
          ) : null}
        </div>
      </div>
      {crms.isError ? (
        <p className="flex items-center gap-1.5 text-sm text-destructive"><Icon name="alert" size={14} />Couldn't load CRM settings: {errorMessage(crms.error)}</p>
      ) : crms.isPending ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4" aria-busy>
          {['HubSpot', 'Salesforce', 'Zoho CRM', 'Pipedrive'].map((label) => (
            <div key={label} className="flex h-64 flex-col rounded-3xl border bg-card p-5 shadow-card">
              <Skeleton className="size-12 rounded-2xl" />
              <span className="mt-3 text-[15px] font-extrabold text-foreground">{label}</span>
              <span className="mt-1 text-xs font-semibold text-muted-foreground">Checking the connection…</span>
              <Skeleton className="mt-auto h-11 w-full rounded-2xl" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {(crms.data?.crms ?? []).map((c, i) => <CrmCard key={c.key} info={c} index={i} filters={filters} count={count} />)}
        </div>
      )}
    </section>
  );
}

function CrmCard({ info, index, filters, count }: { info: CrmInfo; index: number; filters: ExportFilters | null; count: number | undefined }) {
  const tint = TINTS[info.key];
  const { toast, confirm } = useUi();
  const qc = useQueryClient();
  const [job, setJob] = useState<CrmJob | null>(info.lastPush);
  const [csvState, setCsvState] = useState<'idle' | 'busy'>('idle');
  const [testing, setTesting] = useState(false);
  const [showFailures, setShowFailures] = useState(false);
  const running = job?.status === 'running';

  useEffect(() => { if (!job && info.lastPush) setJob(info.lastPush); }, [info.lastPush, job]);

  // follow a running push until it ends (also one that was already running when the page opened)
  useEffect(() => {
    if (!running || !job) return;
    let stop = false;
    const tick = async () => {
      while (!stop) {
        await new Promise((r) => setTimeout(r, 900));
        try {
          const next = await api.crmJob(job.id);
          if (stop) return;
          setJob(next);
          if (next.status !== 'running') {
            void applyChange(qc, 'crm_pushed');
            if (next.status === 'failed') toast('error', `${info.label} push stopped`, next.error || undefined);
            else toast(next.failed ? 'info' : 'success', `Sent to ${info.label}`,
              `${next.created} created · ${next.updated} updated${next.skipped ? ` · ${next.skipped} skipped` : ''}${next.failed ? ` · ${next.failed} failed` : ''}`);
            return;
          }
        } catch (e) {
          if (!stop) { toast('error', `Lost track of the ${info.label} push`, errorMessage(e)); return; }
        }
      }
    };
    void tick();
    return () => { stop = true; };
  }, [running, job?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // between confirming and the job starting, a second Send would ask again and start a second push
  const starting = useRef(false);
  const send = async () => {
    if (!filters || starting.current) return;
    starting.current = true;
    try { await startPush(); } finally { starting.current = false; }
  };
  const startPush = async () => {
    if (!filters) return;
    const n = count != null ? `${fmtNum(count)} contact${count === 1 ? '' : 's'}` : 'these contacts';
    const okd = await confirm({
      title: `Send ${n} to ${info.label}?`,
      message: `Each becomes a ${info.label} ${info.record}. Contacts sent before, or with an email ${info.label} already has, update that ${info.record} instead of creating a new one.`,
      confirmLabel: `Send to ${info.label}`,
    });
    if (!okd) return;
    try {
      setShowFailures(false);
      setJob(await api.pushToCrm(info.key, filters));
    } catch (e) { toast('error', `Couldn't send to ${info.label}`, errorMessage(e)); }
  };

  const csv = async () => {
    if (!filters) return;
    setCsvState('busy');
    try {
      const { blob, filename, rows } = await api.crmCsv(info.key, filters);
      if (rows === 0) toast('info', 'Nothing to export', 'No contacts match.');
      else { saveBlob(blob, filename); toast('success', `${info.label} file ready`, rows != null ? `${fmtNum(rows)} contacts · ${filename}` : filename); }
    } catch (e) { toast('error', "Couldn't export", errorMessage(e)); }
    setCsvState('idle');
  };

  const test = async () => {
    setTesting(true);
    try { const r = await api.testCrm(info.key); toast('success', `${info.label} is connected`, `${r.account} · ${r.latencyMs} ms`); }
    catch (e) { toast('error', `${info.label} test failed`, errorMessage(e)); }
    setTesting(false);
  };

  const pct = job && job.total ? Math.round((job.processed / job.total) * 100) : 0;

  return (
    <motion.div initial={{ opacity: 0, y: 18, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: MOTION.base * 1.4, ease: MOTION.ease, delay: 0.06 + index * 0.06 }}
      whileHover={{ y: -4 }} className="group h-full">
      <div className="relative flex h-full flex-col overflow-hidden rounded-3xl border bg-card shadow-card transition-[border-color,box-shadow] duration-300 group-hover:shadow-active">
        <div className="relative isolate overflow-hidden px-5 pb-4 pt-5" style={{ background: `linear-gradient(135deg, color-mix(in oklab, ${tint} 16%, var(--card)), color-mix(in oklab, ${tint} 4%, var(--card)))` }}>
          <span aria-hidden className="es-bubble pointer-events-none absolute -right-8 -top-10 -z-10 size-28 rounded-full" style={{ background: `color-mix(in oklab, ${tint} 16%, transparent)` }} />
          <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-12 left-16 -z-10 size-20 rounded-full" style={{ background: `color-mix(in oklab, ${tint} 10%, transparent)` }} />
          <div className="flex items-start justify-between gap-3">
            <motion.span initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 16, delay: 0.15 + index * 0.06 }}
              className="es-wiggle-hover flex size-12 items-center justify-center rounded-2xl bg-card shadow-card" style={{ color: ink(tint) }}>
              <motion.span animate={{ y: [0, -3, 0] }} transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: index * 0.3 }} className="flex"><Icon name="send" size={21} /></motion.span>
            </motion.span>
            <span className="inline-flex items-center gap-1 rounded-full bg-card/80 px-2 py-0.5 text-[10.5px] font-black tracking-[0.06em] shadow-card backdrop-blur"
              style={{ color: info.configured ? '#10b981' : 'var(--muted-foreground)' }}>
              <span className="size-1.5 rounded-full" style={{ background: info.configured ? '#10b981' : 'currentColor' }} />
              {info.configured ? 'CONNECTED' : 'NOT CONNECTED'}
            </span>
          </div>
          <h3 className="mt-3 text-[15px] font-extrabold text-foreground">{info.label}</h3>
          <div className="mt-0.5 flex items-baseline gap-1.5">
            {count != null && filters ? (
              <>
                <span className="es-num text-[22px] font-black leading-7" style={{ color: ink(tint) }}><AnimatedNumber value={count} duration={900} /></span>
                <span className="text-xs font-semibold text-muted-foreground">contact{count === 1 ? '' : 's'} → {info.record}s</span>
              </>
            ) : <span className="text-xs font-semibold text-muted-foreground">{filters ? `Contacts become ${info.record}s` : 'Pick one above'}</span>}
          </div>
        </div>

        <div className="flex flex-1 flex-col p-5 pt-4">
          <div className="flex-1 text-sm leading-relaxed text-body">
            {info.configured ? (
              <p>Creates or updates {info.record}s, matched by email. Event, score, LinkedIn and notes travel with them.</p>
            ) : (
              <p>To send directly, set <span className="font-mono text-[12px] text-foreground">{info.missing.join(', ')}</span> on the server. The import file works now.</p>
            )}
            <AnimatePresence initial={false}>
              {job ? (
                <motion.div key="job" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                  <div className="mt-3 rounded-2xl border bg-secondary/40 p-3 text-xs">
                    <div className="flex items-center justify-between gap-2 font-semibold text-foreground">
                      <span className="truncate">{running ? `Sending · ${job.scope}` : job.status === 'failed' ? 'Last push stopped' : job.status === 'cancelled' ? 'Last push cancelled' : 'Last push'}</span>
                      {running ? (
                        <button type="button" onClick={() => void api.cancelCrmJob(job.id).then(setJob).catch((e) => toast('error', "Couldn't cancel the push", errorMessage(e)))} className="cursor-pointer text-muted-foreground hover:text-foreground">Cancel</button>
                      ) : <span className="shrink-0 font-normal text-muted-foreground">{relTime(job.finishedAt || job.startedAt)}</span>}
                    </div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
                      <motion.div className="h-full rounded-full" style={{ background: job.status === 'failed' ? 'var(--destructive)' : tint }}
                        initial={false} animate={{ width: `${running ? pct : 100}%` }} transition={{ duration: 0.4, ease: MOTION.ease }} />
                    </div>
                    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                      <span className="es-num">{fmtNum(job.processed)}/{fmtNum(job.total)}</span>
                      <span><b className="text-foreground">{fmtNum(job.created)}</b> created</span>
                      <span><b className="text-foreground">{fmtNum(job.updated)}</b> updated</span>
                      {job.skipped ? <span><b className="text-foreground">{fmtNum(job.skipped)}</b> skipped</span> : null}
                      {job.failed ? (
                        <button type="button" onClick={() => setShowFailures((v) => !v)} className="cursor-pointer text-destructive hover:underline">
                          {fmtNum(job.failed)} failed
                        </button>
                      ) : null}
                    </div>
                    {job.error ? <p className="mt-2 text-destructive">{job.error}</p> : null}
                    {showFailures && job.failures.length ? (
                      <ul className="es-scroll mt-2 max-h-32 space-y-1 overflow-y-auto">
                        {job.failures.map((f) => <li key={f.contactId}><b className="text-foreground">{f.name}</b> — {f.reason}</li>)}
                      </ul>
                    ) : null}
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>

          <motion.button type="button" onClick={send} disabled={!info.configured || !filters || running} whileTap={!info.configured || !filters ? {} : { scale: 0.97 }}
            className="relative mt-4 flex h-11 w-full cursor-pointer items-center justify-center gap-2 overflow-hidden rounded-2xl text-sm font-extrabold text-white shadow-card transition-[filter,opacity] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
            style={{ background: deep(tint) }}>
            {running ? <motion.span className="absolute inset-y-0 left-0 w-1/3 bg-white/25" animate={{ x: ['-100%', '300%'] }} transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }} /> : null}
            <span className="relative flex items-center gap-2">
              <Icon name="send" size={16} />
              {running ? `Sending… ${pct}%` : !filters ? 'Pick one first' : !info.configured ? 'Not connected' : `Send to ${info.label}`}
            </span>
          </motion.button>
          <div className="mt-2 flex items-center gap-2">
            <button type="button" onClick={csv} disabled={!filters || csvState === 'busy'}
              className="flex h-9 flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-xl border text-xs font-bold text-foreground transition hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-45">
              <Icon name="download" size={14} />{csvState === 'busy' ? 'Building file…' : `${info.label} CSV`}
            </button>
            {info.configured ? (
              <button type="button" onClick={test} disabled={testing}
                className="flex h-9 cursor-pointer items-center justify-center gap-1.5 rounded-xl border px-3 text-xs font-bold text-foreground transition hover:bg-secondary disabled:opacity-45">
                <Icon name={testing ? 'loader' : 'bolt'} size={14} className={testing ? 'animate-spin' : ''} />Test
              </button>
            ) : null}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
