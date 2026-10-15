// The contact drawer (/contacts/$id): the contact's header, the Overview tab and the tab bar. Each
// other tab is its own module that owns its data and writes:
//   overview   GET /api/contacts/{id} · PUT /api/admin/contact/{id}/note        (here)
//   edit       EditTab          PATCH /api/contacts/{id}
//   score      ScoreView        POST /api/contact/{id}/score[?force]
//   duplicates DuplicatesPanel  GET /api/contacts/{id}/duplicates · merge · dismiss
//   pipeline   PipelinePanel    GET/POST /api/contact/{id}/pipeline[/run|/retry]
//   files      FilesPanel       GET/POST /api/contact/{id}/files · DELETE · download
//   qr         QrPanel          GET /api/contact/{id}/qr
import React, { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { levelFrom } from '../../lib/levels';
import { q } from '../../lib/queries';
import { patchContact, removeContacts } from '../../lib/optimistic';
import type { ContactTab } from '../../lib/search';
import type { Contact } from '../../lib/types';
import { Avatar } from '../../ui/Avatar';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { AnimatedNumber } from '../../ui/micro';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Button, Drawer, Notice, Textarea, contactName, cx, errorMessage, fmtDateTime, relTime, useUi, wasEdited,
} from '../../ui/kit';
import {
  CopyAction, DrawerRing, Fade, IconTile, PILL, QuickAction, RadarPulse, RepAvatar, Rise, Row, SectionTitle, useCopy, type Tint,
} from '../../ui/drawerKit';
import { AddToListModal } from './AddToListModal';
import { DuplicatesPanel, useDuplicateMatches } from './DuplicatesPanel';
import { EditTab } from './EditTab';
import { FilesPanel } from './FilesPanel';
import { PipelinePanel } from './PipelinePanel';
import { QrPanel } from './QrPanel';
import { ScoreTab, adminScoreOf, useRescore, type ScoreSide } from './ScoreView';

export interface ContactDrawerProps {
  contact: Contact | null;
  ownerName?: string | undefined;
  /** the representative's own profile picture (null/undefined → their initials) */
  ownerPhoto?: string | null | undefined;
  /** the representative's job title from their app profile */
  ownerTitle?: string | null | undefined;
  tab: ContactTab;
  onTab: (tab: ContactTab) => void;
  onClose: () => void;
  /** open another contact in this drawer (duplicates) */
  onOpenContact?: ((id: string) => void) | undefined;
  /** narrow the grid behind the drawer */
  onFilter?: ((patch: { event?: string; owner?: string }) => void) | undefined;
  eventNames?: string[] | undefined;
}

export function ContactDrawer({ contact, ownerName, ownerPhoto, ownerTitle, tab, onTab, onClose, onOpenContact, onFilter, eventNames = [] }: ContactDrawerProps) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const copy = useCopy();
  const [note, setNote] = useState('');
  const [listFor, setListFor] = useState<string[] | null>(null);
  const dirty = useRef(false);
  const id = contact?.id;

  // fresh copy of the contact (score / notes / edits change while the drawer is open). It is the ONE
  // source of the admin score: the header ring, the stat tile, the footer and the Score tab all read it.
  const live = useQuery({ ...q.contact(id || ''), enabled: !!id, ...(contact ? { initialData: contact } : {}) });
  const c = live.data ?? contact;
  useEffect(() => { setNote(c?.adminNotes || ''); }, [c?.id, c?.adminNotes]);

  const requestClose = async () => {
    if (dirty.current) {
      const ok = await confirm({ title: 'Discard unsaved changes?', message: 'Your edits to this contact have not been saved.', confirmLabel: 'Discard', danger: true });
      if (!ok) return;
      dirty.current = false;
    }
    onClose();
  };

  // which of the two scores the Score tab shows (null = its default); re-scoring shows the admin one
  const [scoreView, setScoreView] = useState<ScoreSide | null>(null);
  useEffect(() => { setScoreView(null); }, [id]);
  const score = useRescore(id, () => {
    setScoreView('admin');
    if (tab !== 'score') onTab('score');
  });
  // optimistic: the note shows as saved at once; restored if the server rejects it
  const saveNote = useMutation({
    mutationFn: (notes: string) => api.saveAdminNote(id!, notes),
    onMutate: async (notes) => ({ rollback: await patchContact(qc, id!, { adminNotes: notes || null }) }),
    onError: (e, _n, ctx) => { ctx?.rollback(); toast('error', "Couldn't save the note — restored", errorMessage(e)); },
    onSuccess: () => toast('success', 'Admin note saved'),
    onSettled: () => { void applyChange(qc, 'contact_updated'); },
  });
  const del = useMutation({
    mutationFn: (cid: string) => api.deleteContact(cid),
    onMutate: async (cid) => { const rollback = await removeContacts(qc, [cid]); dirty.current = false; onClose(); return { rollback }; },
    onError: (e, _id, ctx) => { ctx?.rollback(); toast('error', "Couldn't delete — restored", errorMessage(e)); },
    onSuccess: () => toast('success', `${c ? contactName(c) : 'Contact'} deleted`, 'Restorable for 30 days from Recently deleted.'),
    onSettled: () => { void applyChange(qc, 'contact_deleted'); },
  });

  const dups = useDuplicateMatches(id);

  const remove = async () => {
    if (!c) return;
    const ok = await confirm({ title: `Delete ${contactName(c)}?`, message: "It goes to the owner's Recently deleted for 30 days, so it can be restored.", confirmLabel: 'Delete', danger: true });
    if (ok) del.mutate(c.id);
  };

  const dupCount = dups.data?.matches.length;
  const admin = c ? adminScoreOf(c) : null;
  const adminLevel = levelFrom(admin?.score, admin?.level);

  return (
    <Drawer open={!!contact} onOpenChange={(o) => { if (!o) void requestClose(); }} title={c ? contactName(c) : 'Contact'}
      // every contact opens as a large centred dialog over a frosted-glass backdrop (no side panel)
      mode="center"
      scrollKey={c ? `${c.id}:${tab}` : undefined}
      header={c ? (() => {
        const lvl = adminLevel;
        const li = c.linkedin ? (/^https?:/.test(c.linkedin) ? c.linkedin : `https://${c.linkedin}`) : undefined;
        return (
          // lavender hero tinted by the lead's heat (light in light mode, dark in dark mode)
          <div className="-mx-6 -mb-4 -mt-4 px-6 pb-4 pt-5" style={{ background: `color-mix(in oklab, ${lvl ? `var(--heat-${lvl.key})` : 'var(--primary)'} 7%, var(--panel))` }}>
            <div>
            <div className="flex items-start gap-4 pr-8">
              {/* photo or initials in its heat ring; click to change */}
              <motion.button initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 24 }}
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.96 }}
                className="group relative shrink-0 cursor-pointer rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60" onClick={() => onTab('edit')} aria-label="Change photo" title="Change photo">
                {lvl?.key === 'hot' ? <RadarPulse color="var(--heat-hot)" /> : null}
                <span className="block rounded-full p-[3px]" style={{ background: lvl ? `var(--heat-${lvl.key})` : 'var(--dashed)' }}>
                  <span className="block rounded-full bg-card p-[2px]"><Avatar src={c.photoUrl || null} name={contactName(c)} size="xl" className="size-16 border-0 text-xl shadow-none" /></span>
                </span>
                <span className="absolute inset-[5px] flex items-center justify-center rounded-full bg-slate-900/55 text-white opacity-0 transition-opacity group-hover:opacity-100"><Icon name="camera" size={18} /></span>
              </motion.button>

              <div className="min-w-0 flex-1 pt-0.5">
                <h2 className="break-words font-display text-[22px] font-bold leading-7 tracking-[-0.02em] text-foreground [overflow-wrap:anywhere]">{contactName(c)}</h2>
                <div className="mt-1 space-y-0.5 text-[13.5px]">
                  <div className="flex items-start gap-1.5 text-body"><Icon name="briefcase" size={14} className="mt-[3px] shrink-0 text-brand-600" /><span className={cx('break-words', !c.title && 'italic text-muted-foreground')}>{c.title || 'No designation yet'}</span></div>
                  <div className="flex items-start gap-1.5 text-muted-foreground"><Icon name="building" size={14} className="mt-[3px] shrink-0 text-sky-600 dark:text-sky-400" /><span className={cx('break-words', !c.company && 'italic')}>{c.company || 'No company yet'}</span></div>
                </div>
              </div>

              {/* score ring + level */}
              <div className="flex shrink-0 flex-col items-center gap-1">
                <DrawerRing score={admin?.score} level={admin?.level} />
                {lvl ? (
                  <motion.span key={lvl.key} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 24, delay: 0.3 }}
                    className={cx('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.05em]', lvl.badge)}>
                    <Icon name="flame" size={10} />{lvl.label}
                  </motion.span>
                ) : <span className="text-[10px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">Not scored</span>}
              </div>
            </div>

            {/* chips: where · who · how · when ······ quick actions */}
            {/* four matching pills: same height, shape, text size and border; each with its own icon colour */}
            <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
              {c.eventId ? (
                <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.96 }} onClick={() => onFilter?.({ event: c.eventId })} title="Show only contacts met at this event"
                  className={cx(PILL, 'cursor-pointer hover:ring-amber-300')}>
                  <Icon name="calendar" size={13} className="text-amber-600 dark:text-amber-400" />{c.eventId}
                </motion.button>
              ) : null}
              {ownerName ? (
                <motion.button whileHover={{ y: -1 }} whileTap={{ scale: 0.96 }} onClick={() => c.userId && onFilter?.({ owner: c.userId })} title={`Representative: ${ownerName} · show only their contacts`}
                  className={cx(PILL, 'group cursor-pointer pl-0.5 hover:ring-brand-300')}>
                  {/* the rep's picture in a circle (initials when they have none), filling the pill's height */}
                  <Avatar src={ownerPhoto || null} name={ownerName} size="xs"
                    className="size-6 border-0 text-[9px] shadow-none ring-1 ring-brand-200 transition-transform duration-300 group-hover:scale-110 dark:ring-brand-300/40" />
                  {ownerName}
                </motion.button>
              ) : null}
              {c.source ? (
                <span className={cx(PILL, 'capitalize')}><Icon name="bolt" size={13} className="text-brand-600 dark:text-brand-700" />{c.source}</span>
              ) : null}
              {c.createdAt ? (
                <span className={PILL} title={fmtDateTime(c.createdAt)}><Icon name="clock" size={13} className="text-sky-600 dark:text-sky-400" />Added {relTime(c.createdAt)}</span>
              ) : null}
              <span className="ml-auto flex items-center gap-1.5">
                {c.email ? <QuickAction icon="mail" label={`Email ${c.email}`} href={`mailto:${c.email}`} tint="violet" /> : null}
                {c.phone ? <QuickAction icon="phone" label={`Call ${c.phone}`} href={`tel:${c.phone}`} tint="emerald" /> : null}
                {li ? <QuickAction icon="link" label="Open LinkedIn" href={li} tint="linkedin" /> : null}
              </span>
            </div>
            </div>
          </div>
        );
      })() : undefined}
      // the Edit tab has its own save bar, so the dialog footer steps aside there (no two button rows)
      footer={c && tab !== 'edit' ? <>
        <Button variant="ghost" icon="trash" className="text-rose-600 hover:bg-rose-50 hover:text-rose-700 dark:text-rose-400 dark:hover:bg-rose-500/10" onClick={remove}>Delete</Button>
        <div className="flex gap-2">
          <Button icon="listPlus" onClick={() => setListFor([c.id])}>Add to list</Button>
          <Button icon="pencil" onClick={() => onTab('edit')}>Edit</Button>
          <Button icon="flame" loading={score.isPending} onClick={() => score.mutate(false)}>{admin ? 'Re-score (admin)' : 'Score with AI'}</Button>
          <Button variant="primary" icon="pipeline" onClick={() => onTab('pipeline')}>Pipeline</Button>
        </div>
      </> : undefined}>
      {c ? (
        <div className="px-6 pb-5">
          {/* tabs stay pinned at the top of the scroll area, so switching never needs a scroll back up */}
          <div className="sticky top-0 z-20 -mx-6 border-b border-transparent bg-card/95 px-6 pb-2.5 pt-4 backdrop-blur supports-[not(backdrop-filter:blur(1px))]:bg-card">
          <Tabs value={tab} onValueChange={(t) => onTab(t as ContactTab)} className="max-w-full overflow-x-auto no-scrollbar">
            <TabsList className="h-10 rounded-2xl bg-secondary p-1">
              {([
                { value: 'overview', label: 'Overview', icon: 'user' }, { value: 'edit', label: 'Edit', icon: 'pencil' }, { value: 'score', label: 'AI score', icon: 'flame' },
                { value: 'duplicates', label: 'Duplicates', icon: 'duplicate', count: dupCount }, { value: 'pipeline', label: 'Pipeline', icon: 'pipeline' },
                { value: 'files', label: 'Files', icon: 'paperclip' }, { value: 'qr', label: 'QR', icon: 'qr' },
              ] as { value: ContactTab; label: string; icon: IconName; count?: number | undefined }[]).map((t) => (
                <TabsTrigger key={t.value} value={t.value}
                  className="group cursor-pointer gap-1.5 rounded-xl px-3 text-xs font-semibold text-muted-foreground transition-all data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-card">
                  <Icon name={t.icon} size={13} className="transition-transform duration-300 group-hover:-rotate-12 group-hover:scale-110 group-data-[state=active]:text-primary" />
                  {t.label}
                  {t.count ? <span className="rounded-full bg-amber-100 px-1.5 text-[10px] font-semibold tabular-nums text-amber-800 dark:bg-amber-500/20 dark:text-amber-200">{t.count}</span> : null}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          </div>

          <div className="mt-3">
            <Fade k={`${c.id}:${tab}`}>
              {tab === 'overview' ? (
                <div className="space-y-5">
                  {dupCount ? (
                    <Rise>
                      <Notice tone="amber" icon="duplicate">
                        {dupCount} other contact{dupCount === 1 ? '' : 's'} look{dupCount === 1 ? 's' : ''} like this one.{' '}
                        <button className="font-medium underline cursor-pointer" onClick={() => onTab('duplicates')}>Review</button>
                      </Notice>
                    </Rise>
                  ) : null}

                  {/* at a glance — the app's stat tiles: tinted icon, value, label; tap to act */}
                  <Rise i={1} className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {([
                      { icon: 'flame', tint: adminLevel?.tint ?? 'slate', value: admin?.score ?? '—', label: admin ? `${adminLevel?.label} lead` : 'Not scored', onClick: () => onTab('score') },
                      { icon: 'duplicate', tint: dupCount ? 'amber' : 'emerald', value: dupCount ?? 0, label: dupCount ? 'Look-alikes' : 'No duplicates', onClick: () => onTab('duplicates') },
                      { icon: 'pipeline', tint: 'violet', value: 'Run', label: 'AI pipeline', onClick: () => onTab('pipeline') },
                      { icon: 'paperclip', tint: 'sky', value: 'Files', label: 'Decks & briefs', onClick: () => onTab('files') },
                    ] as { icon: IconName; tint: Tint; value: React.ReactNode; label: string; onClick: () => void }[]).map((s, k) => (
                      <motion.button key={s.label} type="button" onClick={s.onClick} whileHover={{ y: -3 }} whileTap={{ scale: 0.95 }}
                        initial={{ opacity: 0, y: 10, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.08 + k * 0.05 }}
                        className="group relative flex cursor-pointer flex-col items-start gap-2 overflow-hidden rounded-2xl border bg-card p-3 text-left shadow-card transition-colors hover:border-brand-200">
                        <motion.span initial={{ scale: 0, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 15, delay: 0.18 + k * 0.05 }}
                          className="transition-transform duration-300 group-hover:-translate-y-0.5">
                          <IconTile icon={s.icon} tint={s.tint} size="sm" />
                        </motion.span>
                        <span>
                          <span className="es-num block text-lg font-bold leading-6 text-foreground">{typeof s.value === 'number' ? <AnimatedNumber value={s.value} duration={800} /> : s.value}</span>
                          <span className="block text-[11px] font-medium text-muted-foreground">{s.label}</span>
                        </span>
                        <Icon name="chevronRight" size={14} className="absolute right-2.5 top-3 text-muted-foreground/0 transition-all duration-200 group-hover:translate-x-0.5 group-hover:text-muted-foreground" />
                      </motion.button>
                    ))}
                  </Rise>

                  {/* contact details — every field with its own colour and its actions */}
                  <Rise i={2}>
                    <SectionTitle icon="idCard" right={
                      <CopyAction label="All details" onCopy={copy}
                        value={[contactName(c), c.title, c.company, c.email && `Email: ${c.email}`, c.phone && `Phone: ${c.phone}`,
                          c.linkedin && `LinkedIn: ${/^https?:/.test(c.linkedin) ? c.linkedin : `https://${c.linkedin}`}`, c.eventId && `Met at: ${c.eventId}`,
                          ownerName && `Representative: ${ownerName}`].filter(Boolean).join('\n')} />
                    }>Contact details</SectionTitle>
                    <div className="rounded-2xl border bg-card p-1.5 shadow-card">
                      <Row i={0} icon="mail" tint="violet" label="Email"
                        action={c.email ? <>
                          <CopyAction label="Email" value={c.email} onCopy={copy} />
                          <motion.a whileHover={{ x: 2, y: -2 }} whileTap={{ scale: 0.85 }} href={`mailto:${c.email}`} aria-label="Send an email" className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-card hover:text-primary"><Icon name="send" size={14} /></motion.a>
                        </> : null}>
                        {c.email ? <a href={`mailto:${c.email}`} className="text-primary underline-offset-2 hover:underline">{c.email}</a> : null}
                      </Row>
                      <Row i={1} icon="phone" tint="emerald" label="Phone"
                        action={c.phone ? <>
                          <CopyAction label="Phone" value={c.phone} onCopy={copy} />
                          <motion.a whileHover={{ rotate: [0, -14, 12, -8, 0] }} transition={{ duration: 0.5 }} whileTap={{ scale: 0.85 }} href={`tel:${c.phone}`} aria-label="Call" className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-card hover:text-emerald-600"><Icon name="phone" size={14} /></motion.a>
                        </> : null}>
                        {c.phone ? <a href={`tel:${c.phone}`} className="font-mono tabular-nums underline-offset-2 hover:underline">{c.phone}</a> : null}
                      </Row>
                      <Row i={2} icon="link" tint="linkedin" label="LinkedIn"
                        action={c.linkedin ? <CopyAction label="LinkedIn" value={/^https?:/.test(c.linkedin) ? c.linkedin : `https://${c.linkedin}`} onCopy={copy} /> : null}>
                        {c.linkedin ? <a href={/^https?:/.test(c.linkedin) ? c.linkedin : `https://${c.linkedin}`} target="_blank" rel="noreferrer" className="group/li inline-flex items-center gap-1 text-[#0a66c2] underline-offset-2 hover:underline dark:text-[#7fb2ff]">{c.linkedin.replace(/^https?:\/\/(www\.)?/, '')} <Icon name="external" size={12} className="transition-transform group-hover/li:-translate-y-0.5 group-hover/li:translate-x-0.5" /></a> : null}
                      </Row>
                      <Row i={3} icon="briefcase" tint="violet" label="Designation"
                        action={c.title ? <CopyAction label="Designation" value={c.title} onCopy={copy} /> : null}>{c.title}</Row>
                      <Row i={4} icon="building" tint="sky" label="Company"
                        action={c.company ? <CopyAction label="Company" value={c.company} onCopy={copy} /> : null}>{c.company}</Row>
                      <Row i={5} icon="calendar" tint="amber" label="Event">{c.eventId ? <button className="underline-offset-2 hover:underline cursor-pointer" onClick={() => onFilter?.({ event: c.eventId })}>{c.eventId}</button> : null}</Row>
                      <Row i={6} icon="user" tint="violet" label="Representative">{ownerName ? (
                        <button className="group inline-flex cursor-pointer items-center gap-2 text-left" onClick={() => c.userId && onFilter?.({ owner: c.userId })} title="Show only this representative's contacts">
                          <RepAvatar name={ownerName} photo={ownerPhoto} />
                          <span className="leading-tight">
                            <span className="block underline-offset-2 group-hover:underline">{ownerName}</span>
                            {ownerTitle ? <span className="block text-[11px] font-normal text-muted-foreground">{ownerTitle}</span> : null}
                          </span>
                        </button>
                      ) : null}</Row>
                      <Row i={7} icon="bolt" tint="slate" label="Source">{c.source ? <span className="capitalize">{c.source}</span> : null}</Row>
                      <Row i={8} icon="clock" tint="slate" label="Added">{c.createdAt ? `${fmtDateTime(c.createdAt)} · ${relTime(c.createdAt)}` : null}</Row>
                      {wasEdited(c.createdAt, c.updatedAt) ? <Row i={9} icon="refresh" tint="slate" label="Last updated">{`${fmtDateTime(c.updatedAt)} · ${relTime(c.updatedAt)}`}</Row> : null}
                    </div>
                  </Rise>

                  {/* the rep's notes, as a quote */}
                  <Rise i={3}>
                    <SectionTitle icon="note" right={<button className="text-xs font-medium text-primary hover:underline cursor-pointer" onClick={() => onTab('edit')}>Edit</button>}>Rep's notes</SectionTitle>
                    {c.notes ? (
                      <blockquote className="relative whitespace-pre-wrap rounded-2xl bg-brand-50/70 p-4 pl-10 text-sm leading-relaxed text-body dark:bg-brand-50">
                        <motion.span initial={{ opacity: 0, y: 6, scale: 0.6 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ type: 'spring', stiffness: 300, damping: 18, delay: 0.3 }}
                          className="absolute left-3.5 top-2 font-display text-3xl leading-none text-brand-300" aria-hidden>“</motion.span>{c.notes}
                      </blockquote>
                    ) : <p className="rounded-2xl border border-dashed p-4 text-sm text-muted-foreground">No notes from the rep yet.</p>}
                  </Rise>

                  {/* admin-only note */}
                  <Rise i={4}>
                    <SectionTitle icon="shield">Admin note <span className="ml-1 text-xs font-normal text-muted-foreground">only visible here</span></SectionTitle>
                    <Textarea rows={3} value={note} onChange={(e) => setNote(e.target.value)} placeholder="Internal context, next steps, who should follow up…" className="rounded-2xl" />
                    <div className="mt-2 flex items-center justify-end gap-2">
                      <AnimatePresence mode="wait">{note !== (c.adminNotes || '') ? (
                        <motion.span key="unsaved" initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />Unsaved
                        </motion.span>
                      ) : saveNote.isSuccess ? (
                        <motion.span key="saved" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ type: 'spring', stiffness: 500, damping: 20 }}
                          className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 dark:text-emerald-400">
                          <Icon name="check" size={13} />Saved
                        </motion.span>
                      ) : null}</AnimatePresence>
                      <Button size="sm" variant="primary" icon="save" loading={saveNote.isPending} disabled={note === (c.adminNotes || '')} onClick={() => saveNote.mutate(note)}>Save note</Button>
                    </div>
                  </Rise>
                </div>
              ) : null}

              {tab === 'edit' ? <EditTab key={c.id} c={c} eventNames={eventNames} onDirty={(d) => { dirty.current = d; }} onSaved={() => onTab('overview')} /> : null}

              {tab === 'score' ? (
                <ScoreTab c={c} ownerName={ownerName} view={scoreView} onView={setScoreView}
                  detail={{ fetching: live.isFetching, error: live.error, retry: () => { void live.refetch(); } }}
                  rescore={{ run: (force) => score.mutate(force), pending: score.isPending }} />
              ) : null}

              {tab === 'duplicates' ? <DuplicatesPanel c={c} onOpen={onOpenContact} /> : null}

              {tab === 'pipeline' ? <PipelinePanel c={c} ownerName={ownerName} /> : null}

              {tab === 'files' ? <FilesPanel c={c} /> : null}

              {tab === 'qr' ? <QrPanel c={c} /> : null}
            </Fade>
          </div>
        </div>
      ) : null}
      <AddToListModal contactIds={listFor} onClose={() => setListFor(null)} />
    </Drawer>
  );
}
