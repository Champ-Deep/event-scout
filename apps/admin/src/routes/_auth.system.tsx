// /system — server health, AI setup, backups and integrations. Everything is the real API:
//   health   GET  /api/health                      (the database is really queried; version, uptime, flags)
//   models   GET  /api/models                      (the OpenRouter models this server calls, and for what)
//   backups  GET  /api/admin/backup/status · POST /api/admin/backup · GET /api/admin/backup/snapshots/{file}
//   webhook  POST /api/admin/webhook/test          (sends a real eventscout.ping to WEBHOOK_URL)
//   export   GET  /api/admin/export                (every contact as CSV)
// Each check loads on its own: one that fails shows on its own tile instead of blanking the page.
import React, { useState } from 'react';
import { useIsFetching, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { api, saveBlob } from '../lib/api';
import type { BackupResult, WebhookResult } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { MOTION } from '../ui/tokens';
import { AnimatedNumber } from '../ui/micro';
import {
  Badge, Button, Card, EmptyState, ErrorState, Notice, PageHeader, Skeleton, cx, errorMessage, fmtBytes, fmtDateTime, relTime, useUi,
} from '../ui/kit';

const spring = { type: 'spring', stiffness: 420, damping: 28 } as const;
type Tone = 'ok' | 'warn' | 'bad' | 'off';
const TONE: Record<Tone, { dot: string; tile: string }> = {
  ok: { dot: 'bg-emerald-500', tile: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300' },
  warn: { dot: 'bg-amber-500', tile: 'bg-amber-50 text-amber-600 dark:bg-amber-500/15 dark:text-amber-300' },
  bad: { dot: 'bg-rose-500', tile: 'bg-rose-50 text-rose-600 dark:bg-rose-500/15 dark:text-rose-300' },
  off: { dot: 'bg-slate-300 dark:bg-slate-600', tile: 'bg-secondary text-muted-foreground' },
};

const uptime = (s: number) => {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
};

/** One status tile: icon, label, value, a line under it; the dot pulses while healthy. */
function StatusTile({ icon, label, tone, value, hint, loading, index }: {
  icon: IconName; label: string; tone: Tone; value: string; hint?: string | undefined; loading: boolean; index: number;
}) {
  const t = TONE[tone];
  return (
    <motion.div initial={{ opacity: 0, y: 10, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ ...spring, delay: 0.05 + index * 0.06 }} whileHover={{ y: -3 }}>
      <Card className="group h-full rounded-2xl p-3.5 transition-colors hover:border-brand-200">
        <div className="flex items-center gap-2.5">
          <span className={cx('flex size-9 shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110', loading ? TONE.off.tile : t.tile)}>
            <Icon name={icon} size={16} />
          </span>
          <span className="min-w-0 flex-1 text-xs font-bold uppercase tracking-[0.06em] text-muted-foreground">{label}</span>
          {loading ? null : (
            <span className="relative flex size-2.5">
              {tone === 'ok' ? <span className={cx('absolute inline-flex size-full animate-ping rounded-full opacity-50', t.dot)} /> : null}
              <span className={cx('relative inline-flex size-2.5 rounded-full', t.dot)} />
            </span>
          )}
        </div>
        {loading ? <Skeleton className="mt-2.5 h-6 w-24" /> : (
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={value} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: MOTION.fast }}
              className="mt-2 truncate text-lg font-extrabold text-foreground">{value}</motion.div>
          </AnimatePresence>
        )}
        {hint && !loading ? <div className="mt-0.5 truncate text-xs text-muted-foreground" title={hint}>{hint}</div> : null}
      </Card>
    </motion.div>
  );
}

/** A section card: tinted icon, title and line, actions on the right. */
function Section({ icon, tint, title, subtitle, actions, children, index }: {
  icon: IconName; tint: string; title: string; subtitle: React.ReactNode; actions?: React.ReactNode; children: React.ReactNode; index: number;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.2, ease: MOTION.ease, delay: 0.15 + index * 0.06 }}>
      <Card className="overflow-hidden rounded-2xl">
        <div className="group flex flex-wrap items-center gap-3 border-b px-4 py-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-12" style={{ background: `color-mix(in oklab, ${tint} 14%, transparent)`, color: tint }}>
            <Icon name={icon} size={16} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[15px] font-extrabold text-foreground">{title}</div>
            <div className="text-xs text-muted-foreground">{subtitle}</div>
          </div>
          {actions}
        </div>
        {children}
      </Card>
    </motion.div>
  );
}

const backupTone = (b: BackupResult) => (b.status === 'success' ? 'green' : b.status === 'partial' ? 'amber' : 'red');

export function SystemComponent() {
  const qc = useQueryClient();
  const { toast } = useUi();
  const health = useQuery(q.health());
  const backup = useQuery(q.backup());
  const models = useQuery(q.models());
  const fetching = useIsFetching({ queryKey: ['admin'], predicate: (x) => ['health', 'backup', 'models'].includes(String(x.queryKey[1])) }) > 0;
  const [backingUp, setBackingUp] = useState(false);
  const [hook, setHook] = useState<{ result?: WebhookResult; error?: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [showTables, setShowTables] = useState(false);

  const h = health.data;
  const b = backup.data;
  const last = b?.lastBackup ?? null;
  const aiOn = !!models.data?.some((m) => m.configured);
  const chat = models.data?.find((m) => m.default);

  // backups: the last run decides; with no run yet, a snapshot on disk (SQLite target only) counts
  const backupState: Tone = !b ? 'off' : !b.configured ? 'off' : last ? (last.status === 'success' ? 'ok' : last.status === 'partial' ? 'warn' : 'bad')
    : b.target === 'sqlite-file' && b.lastSnapshotAt ? 'ok' : 'warn';
  const backupValue = !b ? '—' : !b.configured ? 'Off' : last ? relTime(last.at) : b.target === 'sqlite-file' && b.lastSnapshotAt ? relTime(b.lastSnapshotAt) : 'None yet';

  const refresh = () => qc.invalidateQueries({ predicate: (x) => x.queryKey[0] === 'admin' && ['health', 'backup', 'models'].includes(String(x.queryKey[1])) });

  const backupNow = async () => {
    setBackingUp(true);
    try {
      const r = await api.backupNow();
      if (r.status === 'partial') toast('info', 'Backup finished with problems', r.message);
      else if (r.status === 'error') toast('error', 'Backup failed', r.message);
      else toast('success', 'Backup complete', r.message);
    } catch (e) {
      toast('error', 'Backup failed', errorMessage(e));
    } finally {
      setBackingUp(false);
      void applyChange(qc, 'backup_done');
    }
  };
  const testHook = async () => {
    setTesting(true);
    try {
      const result = await api.testWebhook();
      setHook({ result });
      toast(result.status === 'delivered' ? 'success' : 'error', result.status === 'delivered' ? 'Webhook answered' : 'Webhook rejected the test', `HTTP ${result.statusCode} in ${result.latencyMs} ms`);
    } catch (e) {
      setHook({ error: errorMessage(e) });
      toast('error', "Webhook test didn't go through", errorMessage(e));
    } finally {
      setTesting(false);
    }
  };
  const exportCsv = async () => {
    setExporting(true);
    try {
      const { blob, filename, rows } = await api.exportAll();
      saveBlob(blob, filename);
      toast('success', 'Export downloaded', rows == null ? filename : `${rows.toLocaleString()} contact${rows === 1 ? '' : 's'} · ${filename}`);
    } catch (e) {
      toast('error', "Couldn't export", errorMessage(e));
    } finally {
      setExporting(false);
    }
  };
  const download = async (file: string) => {
    setDownloading(file);
    try { saveBlob(await api.downloadSnapshot(file), file); } catch (e) { toast('error', "Couldn't download the snapshot", errorMessage(e)); } finally { setDownloading(null); }
  };

  const tables = Object.entries(last?.tables ?? {});

  return (
    <>
      <PageHeader title="System" description="Server health, AI setup, backups and integrations — all checked live." compact
        actions={(
          <motion.span whileTap={{ scale: 0.95 }} className="inline-flex">
            <Button icon="refresh" onClick={refresh} disabled={fetching} className="[&_svg]:transition-transform [&_svg]:duration-700 hover:[&_svg]:rotate-180">
              {fetching ? 'Checking…' : 'Check again'}
            </Button>
          </motion.span>
        )} />

      {/* ---------- status tiles ---------- */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatusTile index={0} icon="server" label="API" loading={health.isLoading}
          tone={health.error ? 'bad' : h?.status === 'healthy' ? 'ok' : 'warn'}
          value={health.error ? 'Unreachable' : h?.status === 'healthy' ? 'Healthy' : 'Degraded'}
          hint={health.error ? errorMessage(health.error) : h ? `${h.version ? `v${h.version} · ` : ''}up ${uptime(h.uptimeSeconds)} · checked every 30 s` : undefined} />
        <StatusTile index={1} icon="database" label="Database" loading={health.isLoading}
          tone={health.error ? 'off' : h?.databaseReady ? 'ok' : 'bad'}
          value={health.error ? 'Unknown' : h?.databaseReady ? 'Connected' : 'Down'}
          hint={health.error ? "Can't tell while the API is unreachable" : h?.databaseReady ? `${h.databaseKind ?? 'Database'} · answered in ${h.databaseLatencyMs} ms` : h?.databaseError ?? undefined} />
        <StatusTile index={2} icon="sparkles" label="AI (OpenRouter)" loading={models.isLoading}
          tone={models.error ? 'bad' : aiOn ? 'ok' : 'warn'}
          value={models.error ? 'Unknown' : aiOn ? 'Configured' : 'Not configured'}
          hint={models.error ? errorMessage(models.error) : aiOn ? (chat ? `Chat: ${chat.name}` : undefined) : 'Set OPENROUTER_API_KEY on the server'} />
        <StatusTile index={3} icon="save" label="Backups" loading={backup.isLoading}
          tone={backup.error ? 'bad' : backupState} value={backup.error ? 'Unknown' : backupValue}
          hint={backup.error ? errorMessage(backup.error)
            : b?.target === 'backup-database' ? `To the backup database${b.intervalHours ? ` · every ${b.intervalHours} h` : ''}`
              : b?.target === 'sqlite-file' ? 'SQLite snapshots on the server' : 'Not configured'} />
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        {/* ---------- backups ---------- */}
        <Section index={0} icon="database" tint="var(--brand-600)" title="Backups"
          subtitle={b?.target === 'backup-database' ? `Copies every table to the backup database (BACKUP_DATABASE_URL)${b.intervalHours ? `, also automatically every ${b.intervalHours} hours and at startup` : ''}.`
            : b?.target === 'sqlite-file' ? `Saves a consistent copy of the database file to backend/backups/ (newest ${b.keep} kept).`
              : backup.isLoading ? 'Checking how backups are set up…' : 'Set BACKUP_DATABASE_URL to enable backups.'}
          actions={(
            <motion.span whileHover={b?.configured ? { scale: 1.03 } : {}} whileTap={b?.configured ? { scale: 0.96 } : {}} className="inline-flex">
              <Button variant="primary" icon="database" loading={backingUp} disabled={!b?.configured} onClick={backupNow}
                title={b?.configured ? undefined : 'No backup target is set up on the server'}>Back up now</Button>
            </motion.span>
          )}>
          <div className="space-y-3 p-4">
            {backup.error ? <ErrorState error={backup.error} onRetry={() => backup.refetch()} /> : backup.isLoading ? <Skeleton className="h-24 w-full" /> : (
              <>
                {last ? (
                  <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={spring}>
                    <Notice tone={backupTone(last)} icon={last.status === 'success' ? 'check' : 'alert'}>
                      <span className="font-semibold">{last.status === 'success' ? 'Last backup' : last.status === 'partial' ? 'Last backup was incomplete' : 'Last backup failed'}</span>
                      {' '}{fmtDateTime(last.at)}{last.trigger ? ` (${last.trigger === 'automatic' ? 'automatic' : 'from the console'})` : ''} — {last.message}
                    </Notice>
                  </motion.div>
                ) : b?.configured ? (
                  <p className="text-sm text-muted-foreground">No backup has run since the server started.</p>
                ) : null}

                {/* rows copied per table, from the last backup to the backup database */}
                {tables.length ? (
                  <div>
                    <button type="button" onClick={() => setShowTables((v) => !v)} aria-expanded={showTables}
                      className="group flex w-full cursor-pointer items-center gap-2 rounded-xl px-1 py-1 text-left text-xs font-bold uppercase tracking-[0.06em] text-muted-foreground hover:text-foreground">
                      <Icon name="chevronDown" size={14} className={cx('transition-transform duration-200', showTables ? '' : '-rotate-90')} />
                      {tables.length} tables{last?.totalRows != null ? ` · ${last.totalRows.toLocaleString()} rows` : ''}{last?.durationSeconds != null ? ` · ${last.durationSeconds}s` : ''}
                      {last?.failedTables?.length ? <Badge tone="red" className="ml-1">{last.failedTables.length} failed</Badge> : null}
                    </button>
                    <AnimatePresence initial={false}>
                      {showTables ? (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="overflow-hidden">
                          <div className="mt-1.5 grid gap-1.5 sm:grid-cols-2">
                            {tables.map(([name, n], i) => (
                              <motion.div key={name} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: Math.min(i, 12) * 0.02 }}
                                className={cx('flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-xs', n == null && 'border-rose-200 bg-rose-50 dark:border-rose-500/30 dark:bg-rose-500/10')}>
                                <span className="truncate font-mono text-body">{name}</span>
                                {n == null ? <span className="font-bold text-rose-600">failed</span> : <span className="font-bold tabular-nums text-foreground">{n.toLocaleString()}</span>}
                              </motion.div>
                            ))}
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>
                  </div>
                ) : null}

                {b?.snapshots.length ? (
                  <ul className="divide-y rounded-xl border">
                    {b.snapshots.map((s, i) => (
                      <motion.li key={s.file} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: Math.min(i, 8) * 0.04 }}
                        className="group flex items-center gap-3 px-3 py-2 transition-colors hover:bg-secondary/50">
                        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-700 transition-transform duration-300 group-hover:-rotate-6 dark:text-brand-800"><Icon name="database" size={14} /></span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-mono text-xs text-foreground">{s.file}</span>
                          <span className="block text-[11px] text-muted-foreground">{fmtDateTime(s.createdAt)} · {relTime(s.createdAt)}</span>
                        </span>
                        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{fmtBytes(s.sizeBytes)}</span>
                        <Button size="sm" variant="ghost" icon="download" loading={downloading === s.file} onClick={() => download(s.file)} aria-label={`Download ${s.file}`}>Download</Button>
                      </motion.li>
                    ))}
                  </ul>
                ) : b?.target === 'sqlite-file' ? <EmptyState icon="database" title="No snapshots yet" description="Press “Back up now” to save the first one." /> : null}
              </>
            )}
          </div>
        </Section>

        <div className="space-y-3">
          {/* ---------- webhook ---------- */}
          <Section index={1} icon="webhook" tint="#0ea5e9" title="Webhook"
            subtitle={h?.webhookConfigured === false ? 'No WEBHOOK_URL is set on the server.' : 'Sends a real test event (eventscout.ping) to WEBHOOK_URL, e.g. your n8n workflow.'}
            actions={(
              <span className="flex items-center gap-2">
                {h ? <Badge tone={h.webhookConfigured ? 'green' : 'amber'}>{h.webhookConfigured ? 'Set up' : 'Not set'}</Badge> : null}
                <motion.span whileTap={{ scale: 0.95 }} className="group/s inline-flex">
                  <Button icon="send" loading={testing} disabled={h?.webhookConfigured === false} onClick={testHook}
                    className="[&_svg]:transition-transform [&_svg]:duration-300 group-hover/s:[&_svg]:-translate-y-0.5 group-hover/s:[&_svg]:translate-x-0.5">Send test</Button>
                </motion.span>
              </span>
            )}>
            <div className="p-4">
              <AnimatePresence mode="wait" initial={false}>
                {!hook ? (
                  <motion.p key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-sm text-muted-foreground">
                    {h?.webhookConfigured === false ? 'Set WEBHOOK_URL on the server to connect your automations.' : 'Send a test to check the integration answers.'}
                  </motion.p>
                ) : hook.error ? (
                  <motion.div key="err" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: [0, -4, 4, 0] }} exit={{ opacity: 0 }}><Notice tone="red" icon="alert">{hook.error}</Notice></motion.div>
                ) : hook.result ? (
                  <motion.div key="ok" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={spring} className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={hook.result.status === 'delivered' ? 'green' : 'red'} icon={hook.result.status === 'delivered' ? 'check' : 'alert'}>{hook.result.status === 'delivered' ? 'Delivered' : 'Rejected'}</Badge>
                      <Badge>HTTP {hook.result.statusCode}</Badge>
                      <Badge icon="clock">{hook.result.latencyMs} ms</Badge>
                      <span className="truncate text-xs text-muted-foreground">{hook.result.target}</span>
                    </div>
                    {hook.result.response ? <pre className="max-h-40 overflow-auto rounded-lg bg-secondary p-3 font-mono text-xs text-body">{hook.result.response}</pre> : null}
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>
          </Section>

          {/* ---------- AI models ---------- */}
          <Section index={2} icon="sparkles" tint="var(--brand-600)" title="AI models" subtitle="What this server calls on OpenRouter, and for what">
            <div className="p-3">
              {models.isLoading ? <Skeleton className="h-20 w-full" /> : models.error ? <ErrorState error={models.error} onRetry={() => models.refetch()} /> : !models.data?.length ? (
                <EmptyState icon="sparkles" title="No models" description="The server didn't report any AI models." />
              ) : (
                <ul className="space-y-1.5">
                  {models.data.map((m, i) => (
                    <motion.li key={`${m.role}-${m.id}`} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ ...spring, delay: 0.2 + i * 0.04 }} whileHover={{ x: 2 }}
                      className="group flex items-center gap-2.5 rounded-xl border px-2.5 py-2 transition-colors hover:border-brand-200">
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-secondary text-muted-foreground transition-colors group-hover:bg-brand-50 group-hover:text-brand-700">
                        <Icon name={m.role === 'audio' ? 'mic' : m.role === 'vision' ? 'camera' : m.role === 'fallback' ? 'refresh' : m.default ? 'sparkles' : 'bolt'} size={14} className="transition-transform duration-300 group-hover:-rotate-12" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-bold text-foreground">{m.name}</span>
                        <span className="block truncate text-[11px] text-muted-foreground"><span className="font-mono">{m.id}</span>{m.uses.length ? ` · ${m.uses.join(' · ')}` : ''}</span>
                      </span>
                      {m.default ? <Badge tone="violet">Chat</Badge> : null}
                      <Badge tone={m.configured ? 'green' : 'amber'}>{m.configured ? 'Ready' : 'No key'}</Badge>
                    </motion.li>
                  ))}
                </ul>
              )}
            </div>
          </Section>

          {/* ---------- export ---------- */}
          <Section index={3} icon="download" tint="#10b981" title="Data export" subtitle="Every contact, from every rep, as one CSV file"
            actions={(
              <motion.span whileTap={{ scale: 0.95 }} className="group/e inline-flex">
                <Button icon="download" loading={exporting} onClick={exportCsv} className="[&_svg]:transition-transform [&_svg]:duration-300 group-hover/e:[&_svg]:translate-y-0.5">Export CSV</Button>
              </motion.span>
            )}>
            <Link to="/exports" className="group flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground transition-colors hover:bg-secondary/50 hover:text-foreground">
              <Icon name="info" size={15} />
              <span className="flex-1">More exports — by event, rep, list, team or activity — are on the Exports page.</span>
              <Icon name="chevronRight" size={15} className="transition-transform duration-200 group-hover:translate-x-1 group-hover:text-brand-600" />
            </Link>
          </Section>
        </div>
      </div>

      {h ? (
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-3 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1"><Icon name="clock" size={11} />Server started {fmtDateTime(h.startedAt)}</span>
          <span>·</span>
          <span>Last checked {relTime(h.timestamp)}</span>
          {models.data ? <><span>·</span><span><AnimatedNumber value={models.data.length} duration={500} /> AI model{models.data.length === 1 ? '' : 's'}</span></> : null}
        </motion.p>
      ) : null}
    </>
  );
}
