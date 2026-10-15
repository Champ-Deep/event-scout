// Recently deleted → click a name: the contact's overview in the SAME design as the Contacts page's
// contact view (ui/drawerKit's hero pieces, stat tiles and detail rows, and ScoreView's score panel
// are reused, not copied), opened as a panel from the right. It's read-only: the contact
// is rebuilt from the snapshot the server keeps for 30 days (GET /api/admin/contacts/deleted/{id});
// the one action is Restore (back to its owner, same ID, fields, photo and files).
import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'motion/react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { q } from '../../lib/queries';
import type { WorkspaceDeletedContact } from '../../lib/types';
import { Avatar } from '../../ui/Avatar';
import { Icon, type IconName } from '../../ui/icons';
import { AnimatedNumber } from '../../ui/micro';
import { MOTION } from '../../ui/tokens';
import { Button, Drawer, ErrorState, Notice, Skeleton, cx, fmtBytes, fmtDateTime, relTime, wasEdited } from '../../ui/kit';
import { contactLevel } from '../../lib/levels';
import {
  CopyAction, DrawerRing, EmptyPanel, Fade, IconTile, PILL, QuickAction, RadarPulse, RepAvatar, Rise, Row, SectionTitle, useCopy, type Tint,
} from '../../ui/drawerKit';
import { ScoreView } from './ScoreView';

type Tab = 'overview' | 'score' | 'files';

/** Card scans store "N/A" (or "-", "none") for a missing field; treat those as empty. */
const PLACEHOLDER = /^\s*(n\/?a|none|null|nil|-+|–|—)?\s*$/i;
const blank = (v?: string | null) => (v == null || PLACEHOLDER.test(v) ? null : v);
/** One field can hold several values ("+91 124…, +91 993…"). */
const split = (v?: string | null) => (blank(v) ?? '').split(/\s*[,;|\n]\s*|\s+\/\s+/).map((s) => s.trim()).filter((s) => s && !PLACEHOLDER.test(s));
const linkedinHref = (v: string) => (/^https?:/i.test(v) ? v : `https://${v}`);

export function DeletedContactDrawer({ summary, onClose, onRestore }: {
  /** the row that was clicked (its header shows at once while the full record loads); null = closed */
  summary: WorkspaceDeletedContact | null;
  onClose: () => void;
  onRestore: (c: WorkspaceDeletedContact) => void;
}) {
  const copy = useCopy();
  const id = summary?.id ?? '';
  const [tab, setTab] = useState<Tab>('overview');
  useEffect(() => { setTab('overview'); }, [id]);
  const detail = useQuery({ ...q.deletedContact(id), enabled: !!summary });
  const { data: users } = useQuery(q.users());
  const d = detail.data;
  const s = d ?? summary;
  const raw = d?.contact;
  // the snapshot, with "N/A" placeholders read as empty (like the contacts grid)
  const c = raw ? { ...raw, company: blank(raw.company), title: blank(raw.title), email: blank(raw.email), phone: blank(raw.phone), linkedin: blank(raw.linkedin) } : null;
  const ownerUser = users?.find((u) => u.id === s?.ownerId);
  const ownerName = s?.owner;
  const files = d?.files ?? [];

  const name = c ? [c.firstName, c.lastName].filter(Boolean).join(' ') || s?.name || 'Contact' : s?.name ?? 'Contact';
  const title = c ? c.title : blank(s?.title);
  const company = c ? c.company : blank(s?.company);
  const lvl = c ? contactLevel(c) : null;
  const li = c?.linkedin ? linkedinHref(c.linkedin) : undefined;
  const emails = split(c?.email);
  const phones = split(c?.phone);

  return (
    <Drawer open={!!summary} onOpenChange={(o) => { if (!o) onClose(); }} title={name}
      // a panel from the right (the Contacts page opens a centred dialog; same content and look)
      mode="side" width="sm:max-w-2xl" scrollKey={`${id}:${tab}`}
      header={s ? (
        // the contact view's hero: tinted by the lead's heat, photo in its heat ring, ring + level on the right
        <div className="-mx-6 -mb-4 -mt-4 px-6 pb-4 pt-5" style={{ background: `color-mix(in oklab, ${lvl ? `var(--heat-${lvl.key})` : 'var(--primary)'} 7%, var(--panel))` }}>
          <div className="flex items-start gap-4 pr-8">
            <motion.span initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 24 }} className="relative shrink-0 rounded-full">
              {lvl?.key === 'hot' ? <RadarPulse color="var(--heat-hot)" /> : null}
              <span className="block rounded-full p-[3px]" style={{ background: lvl ? `var(--heat-${lvl.key})` : 'var(--dashed)' }}>
                <span className="block rounded-full bg-card p-[2px]"><Avatar src={c?.photoUrl ?? s.photoUrl} name={name} size="xl" className="size-16 border-0 text-xl shadow-none" /></span>
              </span>
              {/* it's in the bin */}
              <span className="absolute -bottom-0.5 -right-0.5 flex size-6 items-center justify-center rounded-full bg-heat-hot text-white ring-[3px] ring-card" title="Deleted"><Icon name="trash" size={11} /></span>
            </motion.span>

            <div className="min-w-0 flex-1 pt-0.5">
              <h2 className="break-words font-display text-[22px] font-bold leading-7 tracking-[-0.02em] text-foreground [overflow-wrap:anywhere]">{name}</h2>
              <div className="mt-1 space-y-0.5 text-[13.5px]">
                <div className="flex items-start gap-1.5 text-body"><Icon name="briefcase" size={14} className="mt-[3px] shrink-0 text-brand-600" /><span className={cx('break-words', !title && 'italic text-muted-foreground')}>{title || 'No designation yet'}</span></div>
                <div className="flex items-start gap-1.5 text-muted-foreground"><Icon name="building" size={14} className="mt-[3px] shrink-0 text-sky-600 dark:text-sky-400" /><span className={cx('break-words', !company && 'italic')}>{company || 'No company yet'}</span></div>
              </div>
            </div>

            <div className="flex shrink-0 flex-col items-center gap-1">
              <DrawerRing score={c?.score} level={c?.level} />
              {lvl ? (
                <span className={cx('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.05em]', lvl.badge)}><Icon name="flame" size={10} />{lvl.label}</span>
              ) : <span className="text-[10px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">{c ? 'Not scored' : ' '}</span>}
            </div>
          </div>

          {/* the same matching pills: deleted · event · representative · source · added ······ quick actions */}
          <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
            <span className={cx(PILL, 'font-semibold text-destructive ring-heat-hot/40')} title={s.deletedAt ? fmtDateTime(s.deletedAt) : undefined}>
              <Icon name={s.reason ? 'merge' : 'trash'} size={13} />{s.reason ? 'Merged' : 'Deleted'}{s.deletedAt ? ` ${relTime(s.deletedAt)}` : ''}
            </span>
            {c?.eventId ? <span className={PILL}><Icon name="calendar" size={13} className="text-amber-600 dark:text-amber-400" />{c.eventId}</span> : null}
            {ownerName ? (
              <span className={cx(PILL, 'pl-0.5')} title={`Representative: ${ownerName}`}>
                <Avatar src={ownerUser?.photoUrl || null} name={ownerName} size="xs" className="size-6 border-0 text-[9px] shadow-none ring-1 ring-brand-200 dark:ring-brand-300/40" />
                {ownerName}
              </span>
            ) : null}
            {c?.source ? <span className={cx(PILL, 'capitalize')}><Icon name="bolt" size={13} className="text-brand-600 dark:text-brand-700" />{c.source}</span> : null}
            {c?.createdAt ? <span className={PILL} title={fmtDateTime(c.createdAt)}><Icon name="clock" size={13} className="text-sky-600 dark:text-sky-400" />Added {relTime(c.createdAt)}</span> : null}
            <span className="ml-auto flex items-center gap-1.5">
              {emails[0] ? <QuickAction icon="mail" label={`Email ${emails[0]}`} href={`mailto:${emails[0]}`} tint="violet" /> : null}
              {phones[0] ? <QuickAction icon="phone" label={`Call ${phones[0]}`} href={`tel:${phones[0].replace(/[^\d+]/g, '')}`} tint="emerald" /> : null}
              {li ? <QuickAction icon="link" label="Open LinkedIn" href={li} tint="linkedin" /> : null}
            </span>
          </div>
        </div>
      ) : undefined}
      footer={s ? (
        <>
          <Button variant="ghost" onClick={onClose}>Close</Button>
          <Button variant="primary" icon="refresh" onClick={() => onRestore(s)}>Restore to {s.owner.split(/\s+/)[0]}</Button>
        </>
      ) : undefined}>
      {s ? (
        <div className="px-6 pb-5">
          {/* the contact view's tabs, pinned at the top of the scroll area (read-only ones only) */}
          <div className="sticky top-0 z-20 -mx-6 border-b border-transparent bg-card/95 px-6 pb-2.5 pt-4 backdrop-blur supports-[not(backdrop-filter:blur(1px))]:bg-card">
            <Tabs value={tab} onValueChange={(t) => setTab(t as Tab)} className="max-w-full overflow-x-auto no-scrollbar">
              <TabsList className="h-10 rounded-2xl bg-secondary p-1">
                {([
                  { value: 'overview', label: 'Overview', icon: 'user' },
                  { value: 'score', label: 'AI score', icon: 'flame' },
                  { value: 'files', label: 'Files', icon: 'paperclip', count: files.length },
                ] as { value: Tab; label: string; icon: IconName; count?: number }[]).map((t) => (
                  <TabsTrigger key={t.value} value={t.value}
                    className="group cursor-pointer gap-1.5 rounded-xl px-3 text-xs font-semibold text-muted-foreground transition-all data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-card">
                    <Icon name={t.icon} size={13} className="transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110 group-data-[state=active]:text-primary" />
                    {t.label}
                    {t.count ? <span className="rounded-full bg-brand-50 px-1.5 text-[10px] font-semibold tabular-nums text-brand-700">{t.count}</span> : null}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>

          <div className="mt-3">
            {detail.isPending ? (
              <div className="space-y-3" aria-busy>
                <Skeleton className="h-20 w-full rounded-2xl" />
                <Skeleton className="h-72 w-full rounded-2xl" />
                <Skeleton className="h-24 w-full rounded-2xl" />
              </div>
            ) : detail.isError ? (
              <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
            ) : c ? (
              <Fade k={`${id}:${tab}`}>
                {tab === 'overview' ? (
                  <div className="space-y-5">
                    {/* why it's here and how long it stays */}
                    <Rise>
                      <Notice tone={s.reason ? 'violet' : 'red'} icon={s.reason ? 'merge' : 'trash'}>
                        <b className="font-semibold">{s.reason || 'Deleted'}</b>{s.deletedAt ? ` on ${fmtDateTime(s.deletedAt)}` : ''}.{' '}
                        Restorable for <b className="font-semibold">{s.daysLeft} more day{s.daysLeft === 1 ? '' : 's'}</b>, then it's removed for good.
                      </Notice>
                    </Rise>

                    {/* at a glance — the contact view's stat tiles */}
                    <Rise i={1} className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {([
                        { icon: 'flame', tint: lvl?.tint ?? 'slate', value: c.score ?? '—', label: c.score == null ? 'Not scored' : `${lvl?.label} lead`, onClick: () => setTab('score') },
                        { icon: 'trash', tint: 'rose', value: s.deletedAt ? relTime(s.deletedAt) : '—', label: s.reason ? 'Merged' : 'Deleted' },
                        { icon: 'clock', tint: s.daysLeft <= 3 ? 'rose' : 'amber', value: s.daysLeft, label: `Day${s.daysLeft === 1 ? '' : 's'} to restore` },
                        { icon: 'paperclip', tint: 'sky', value: files.length, label: files.length === 1 ? 'File' : 'Files', onClick: () => setTab('files') },
                      ] as { icon: IconName; tint: Tint; value: React.ReactNode; label: string; onClick?: () => void }[]).map((t, k) => (
                        <motion.button key={t.label} type="button" onClick={t.onClick} disabled={!t.onClick} whileHover={t.onClick ? { y: -3 } : {}} whileTap={t.onClick ? { scale: 0.95 } : {}}
                          initial={{ opacity: 0, y: 10, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.08 + k * 0.05 }}
                          className={cx('group relative flex flex-col items-start gap-2 overflow-hidden rounded-2xl border bg-card p-3 text-left shadow-card transition-colors disabled:cursor-default', t.onClick && 'cursor-pointer hover:border-brand-200')}>
                          <motion.span initial={{ scale: 0, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 15, delay: 0.18 + k * 0.05 }}>
                            <IconTile icon={t.icon} tint={t.tint} size="sm" />
                          </motion.span>
                          <span>
                            <span className="es-num block text-lg font-bold leading-6 text-foreground">{typeof t.value === 'number' ? <AnimatedNumber value={t.value} duration={800} /> : t.value}</span>
                            <span className="block text-[11px] font-medium text-muted-foreground">{t.label}</span>
                          </span>
                          {t.onClick ? <Icon name="chevronRight" size={14} className="absolute right-2.5 top-3 text-muted-foreground/0 transition-all duration-200 group-hover:translate-x-0.5 group-hover:text-muted-foreground" /> : null}
                        </motion.button>
                      ))}
                    </Rise>

                    {/* contact details — the same rows, colours and actions */}
                    <Rise i={2}>
                      <SectionTitle icon="idCard" right={
                        <CopyAction label="All details" onCopy={copy}
                          value={[name, c.title, c.company, emails.length && `Email: ${emails.join(', ')}`, phones.length && `Phone: ${phones.join(', ')}`,
                            li && `LinkedIn: ${li}`, c.eventId && `Met at: ${c.eventId}`, ownerName && `Representative: ${ownerName}`].filter(Boolean).join('\n')} />
                      }>Contact details</SectionTitle>
                      <div className="rounded-2xl border bg-card p-1.5 shadow-card">
                        <Row i={0} icon="mail" tint="violet" label={emails.length > 1 ? `Email · ${emails.length}` : 'Email'}
                          action={emails[0] ? <>
                            <CopyAction label="Email" value={emails.join(', ')} onCopy={copy} />
                            <motion.a whileHover={{ x: 2, y: -2 }} whileTap={{ scale: 0.85 }} href={`mailto:${emails[0]}`} aria-label="Send an email" className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-card hover:text-primary"><Icon name="send" size={14} /></motion.a>
                          </> : null}>
                          {emails.length ? <span className="flex flex-col">{emails.map((e) => <a key={e} href={`mailto:${e}`} className="text-primary underline-offset-2 hover:underline">{e}</a>)}</span> : null}
                        </Row>
                        <Row i={1} icon="phone" tint="emerald" label={phones.length > 1 ? `Phone · ${phones.length}` : 'Phone'}
                          action={phones[0] ? <>
                            <CopyAction label="Phone" value={phones.join(', ')} onCopy={copy} />
                            <motion.a whileHover={{ rotate: [0, -14, 12, -8, 0] }} transition={{ duration: 0.5 }} whileTap={{ scale: 0.85 }} href={`tel:${phones[0].replace(/[^\d+]/g, '')}`} aria-label="Call" className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-card hover:text-emerald-600"><Icon name="phone" size={14} /></motion.a>
                          </> : null}>
                          {phones.length ? <span className="flex flex-col">{phones.map((p) => <a key={p} href={`tel:${p.replace(/[^\d+]/g, '')}`} className="font-mono tabular-nums underline-offset-2 hover:underline">{p}</a>)}</span> : null}
                        </Row>
                        <Row i={2} icon="link" tint="linkedin" label="LinkedIn" action={li ? <CopyAction label="LinkedIn" value={li} onCopy={copy} /> : null}>
                          {li ? <a href={li} target="_blank" rel="noreferrer" className="group/li inline-flex items-center gap-1 text-[#0a66c2] underline-offset-2 hover:underline dark:text-[#7fb2ff]">{c.linkedin!.replace(/^https?:\/\/(www\.)?/, '')} <Icon name="external" size={12} className="transition-transform group-hover/li:-translate-y-0.5 group-hover/li:translate-x-0.5" /></a> : null}
                        </Row>
                        <Row i={3} icon="briefcase" tint="violet" label="Designation" action={c.title ? <CopyAction label="Designation" value={c.title} onCopy={copy} /> : null}>{c.title}</Row>
                        <Row i={4} icon="building" tint="sky" label="Company" action={c.company ? <CopyAction label="Company" value={c.company} onCopy={copy} /> : null}>{c.company}</Row>
                        <Row i={5} icon="calendar" tint="amber" label="Event">{c.eventId || null}</Row>
                        <Row i={6} icon="user" tint="violet" label="Representative">{ownerName ? (
                          <span className="inline-flex items-center gap-2">
                            <RepAvatar name={ownerName} photo={ownerUser?.photoUrl} />
                            <span className="leading-tight">
                              <span className="block">{ownerName}</span>
                              {ownerUser?.title ? <span className="block text-[11px] font-normal text-muted-foreground">{ownerUser.title}</span> : null}
                            </span>
                          </span>
                        ) : null}</Row>
                        <Row i={7} icon="bolt" tint="slate" label="Source">{c.source ? <span className="capitalize">{c.source}</span> : null}</Row>
                        <Row i={8} icon="clock" tint="slate" label="Added">{c.createdAt ? `${fmtDateTime(c.createdAt)} · ${relTime(c.createdAt)}` : null}</Row>
                        {wasEdited(c.createdAt, c.updatedAt) ? <Row i={9} icon="refresh" tint="slate" label="Last updated">{`${fmtDateTime(c.updatedAt)} · ${relTime(c.updatedAt)}`}</Row> : null}
                        {s.deletedAt ? <Row i={10} icon="trash" tint="rose" label={s.reason ? 'Merged' : 'Deleted'}>{`${fmtDateTime(s.deletedAt)} · ${relTime(s.deletedAt)}`}</Row> : null}
                      </div>
                    </Rise>

                    {/* the rep's notes, as a quote */}
                    <Rise i={3}>
                      <SectionTitle icon="note">Rep's notes</SectionTitle>
                      {c.notes ? (
                        <blockquote className="relative whitespace-pre-wrap rounded-2xl bg-brand-50/70 p-4 pl-10 text-sm leading-relaxed text-body dark:bg-brand-50">
                          <span className="absolute left-3.5 top-2 font-display text-3xl leading-none text-brand-300" aria-hidden>“</span>{c.notes}
                        </blockquote>
                      ) : <p className="rounded-2xl border border-dashed p-4 text-sm text-muted-foreground">No notes from the rep.</p>}
                    </Rise>

                    {/* admin-only note (read-only while deleted) */}
                    <Rise i={4}>
                      <SectionTitle icon="shield">Admin note <span className="ml-1 text-xs font-normal text-muted-foreground">only visible here</span></SectionTitle>
                      {c.adminNotes ? (
                        <p className="whitespace-pre-wrap rounded-2xl border bg-card p-4 text-sm leading-relaxed text-body shadow-card">{c.adminNotes}</p>
                      ) : <p className="rounded-2xl border border-dashed p-4 text-sm text-muted-foreground">No admin note. Restore the contact to add one.</p>}
                    </Rise>
                  </div>
                ) : null}

                {tab === 'score' ? (
                  c.score == null ? (
                    <EmptyPanel icon="smartphone" tint="sky" title="Not scored">
                      {ownerName || 'The rep'} hadn't scored this contact before it was deleted. Restore it to score it with AI.
                    </EmptyPanel>
                  ) : (
                    <ScoreView variant="snapshot" title={`${ownerName ? ownerName.split(' ')[0] : 'Rep'}'s score`} icon="smartphone" onCopy={copy}
                      sv={{ score: c.score, level: c.level, reasoning: c.scoreReasoning, breakdown: c.scoreBreakdown, actions: c.scoreActions, meta: c.scoreMeta }} />
                  )
                ) : null}

                {tab === 'files' ? (
                  files.length ? (
                    <Rise>
                      <SectionTitle icon="paperclip" right={<span className="text-xs font-medium text-muted-foreground">come back on restore</span>}>Attached files</SectionTitle>
                      <div className="rounded-2xl border bg-card p-1.5 shadow-card">
                        {files.map((f, i) => (
                          <Row key={f.id || f.filename} i={i} icon="file" tint={f.category === 'pitch' ? 'violet' : f.category === 'brief' ? 'amber' : 'sky'} label={`${f.category} · ${f.fileType.toUpperCase()} · ${fmtBytes(f.fileSize)}`}>
                            <span title={f.description || undefined}>{f.filename}</span>
                          </Row>
                        ))}
                      </div>
                    </Rise>
                  ) : (
                    <EmptyPanel icon="paperclip" tint="sky" title="No files">Nothing was attached to this contact.</EmptyPanel>
                  )
                ) : null}
              </Fade>
            ) : null}
          </div>
        </div>
      ) : null}
    </Drawer>
  );
}
