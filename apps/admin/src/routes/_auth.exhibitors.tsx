// /exhibitors — the Events page. Everything comes from the API:
//   events     GET /api/admin/events            every event: booth directory ∪ contacts' "met at" ∪ reps' event lists ∪ event files
//   one event  GET /api/admin/events/detail     booths (rep visits, contacts met per company), reps, captures per day, files
//   booths     POST /api/exhibitors · PATCH /api/exhibitors/{id} · DELETE /api/exhibitors/{id} · POST /api/admin/import_exhibitors
//   files      GET /api/files/{id}/download     (floor plans are shown inline: images and PDFs)
// The event, tab, hall, search and category live in the URL (?event=…&tab=map&hall=…&q=…&category=…).
import React, { useEffect, useMemo, useState } from 'react';
import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { Link, useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { api, base64ToBlob, saveBlob } from '../lib/api';
import { q as queries } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { clean } from '../lib/search';
import { useEventNames } from '../lib/eventNames';
import type { AdminEvent, EventBooth, EventDetail, EventFileItem, EventRep } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { Avatar } from '../ui/Avatar';
import { AnimatedNumber, SlidingTabs } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import {
  Badge, Button, Card, EmptyState, ErrorState, Field, IconButton, Input, Modal, Notice, PageHeader, SearchInput, Spinner,
  Textarea, cx, errorMessage, fmtBytes, fmtDateTime, relTime, useUi, safeHref } from '../ui/kit';
import { IconTile, TINT, type Tint } from '../ui/drawerKit';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';

const SAMPLE = `[
  { "name": "Mercy Health", "booth": "B-12", "hall": "5", "category": "Hospitals", "country": "USA", "website": "mercy.net" }
]`;
type Tab = 'booths' | 'map' | 'team' | 'files';
const NO_HALL = '—';
const byCode = (a?: string | null, b?: string | null) => (a || '').localeCompare(b || '', undefined, { numeric: true, sensitivity: 'base' });
const hallOf = (b: EventBooth) => (b.hall || '').trim() || NO_HALL;
const hallLabel = (h: string) => (h === NO_HALL ? 'No hall' : /^hall\b/i.test(h) ? h : `Hall ${h}`);
const site = (w: string) => safeHref(w); // web addresses only (javascript:/data: links render no link)
const initials = (n: string) => n.split(/\s+/).filter(Boolean).map((p) => p[0]).join('').slice(0, 2).toUpperCase() || '?';

/* ------------------------------------------------------------------ event picker */

/** The event dropdown: every event the workspace knows, searchable, with what's in each. */
function EventPicker({ events, value, onChange }: { events: AdminEvent[]; value: string; onChange: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const cur = events.find((e) => e.name === value);
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button aria-label={`Event: ${value || 'none'}. Change`}
          className="group inline-flex h-11 max-w-full cursor-pointer items-center gap-2.5 rounded-xl border bg-card pl-2 pr-3 text-left shadow-card transition-colors hover:border-brand-300 data-[state=open]:border-brand-400">
          <span className={cx('flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', TINT.violet)}><Icon name="calendar" size={16} /></span>
          <span className="min-w-0">
            <span className="block text-[10px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">Event · {events.length} in all</span>
            <span className="block truncate text-[15px] font-bold leading-5 text-foreground">{value || 'Pick an event'}</span>
          </span>
          <Icon name="chevronDown" size={15} className="ml-1 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={8} className="w-[400px] max-w-[calc(100vw-24px)] overflow-hidden rounded-2xl p-0 shadow-xl">
        <Command>
          <CommandInput placeholder="Find an event…" />
          <CommandList className="max-h-[360px]">
            <CommandEmpty>No event matches.</CommandEmpty>
            <CommandGroup>
              {events.map((e) => {
                const sel = e.name === value;
                return (
                  <CommandItem key={e.key} value={`${e.name} ${e.key}`} onSelect={() => { onChange(e.name); setOpen(false); }}
                    className={cx('cursor-pointer items-start gap-2.5 rounded-xl px-2 py-2', sel && 'bg-accent text-accent-foreground')}>
                    <span className={cx('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg', e.hasDirectory ? TINT.violet : TINT.slate)}>
                      <Icon name={e.hasDirectory ? 'store' : 'calendar'} size={13} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className={cx('block break-words text-[13px]', sel ? 'font-bold' : 'font-semibold')}>{e.name}</span>
                      <span className="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
                        <span>{e.booths} booth{e.booths === 1 ? '' : 's'}</span>
                        <span>· {e.contacts} contact{e.contacts === 1 ? '' : 's'}</span>
                        {e.reps ? <span>· {e.reps} rep{e.reps === 1 ? '' : 's'}</span> : null}
                        {e.floorPlans ? <span className="font-semibold text-emerald-700 dark:text-emerald-400">· floor plan</span> : null}
                        {e.activeFor ? <span className="font-semibold text-brand-700">· current for {e.activeFor}</span> : null}
                      </span>
                    </span>
                    {sel ? <Icon name="check" size={14} className="mt-1 text-brand-600" /> : null}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
        {cur ? <div className="border-t px-3 py-2 text-[11px] text-muted-foreground">Events come from the booth directory, contacts' “met at”, reps' event lists and event files.</div> : null}
      </PopoverContent>
    </Popover>
  );
}

/* ------------------------------------------------------------------ summary */

/** A summary tile: every one does something (opens a tab, filters the booths or opens Contacts),
 *  the one you're looking at is highlighted, and hovering says what it opens. */
function StatTile({ icon, tint, label, value, i, onClick, active, hint, external }: {
  icon: IconName; tint: Tint; label: string; value: number; i: number; onClick: () => void; active?: boolean; hint: string; external?: boolean;
}) {
  return (
    <motion.button type="button" onClick={onClick} aria-pressed={!!active} title={hint}
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.04 + i * 0.04 }}
      whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
      className={cx('group relative flex min-w-[124px] flex-1 cursor-pointer items-center gap-2 rounded-xl border bg-card px-2.5 py-2 text-left shadow-card transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active ? 'border-brand-400 ring-2 ring-brand-400/25' : 'hover:border-brand-200')}>
      <IconTile icon={icon} tint={tint} size="sm" />
      <span className="min-w-0 flex-1">
        <span className="es-num block text-lg font-bold leading-5 text-foreground"><AnimatedNumber value={value} duration={600} /></span>
        <span className={cx('block whitespace-nowrap text-[11px] font-medium leading-4', active ? 'text-brand-700' : 'text-muted-foreground')}>{label}</span>
      </span>
      <Icon name={external ? 'external' : 'chevronRight'} size={13}
        className={cx('shrink-0 transition-all', active ? 'text-brand-600' : 'text-muted-foreground/0 group-hover:translate-x-0.5 group-hover:text-muted-foreground')} />
    </motion.button>
  );
}

/** Contacts captured per day at this event, as small bars (hover for the date). */
function CaptureBars({ perDay }: { perDay: EventDetail['perDay'] }) {
  if (!perDay.length) return null;
  const days = perDay.slice(-21);
  const max = Math.max(1, ...days.map((d) => d.count));
  return (
    <div className="min-w-[180px] flex-1 rounded-xl border bg-card px-3 py-2 shadow-card">
      <div className="flex items-baseline justify-between text-[10.5px] font-medium text-muted-foreground">
        <span>Contacts per day</span><span className="tabular-nums">{days.length} day{days.length === 1 ? '' : 's'}</span>
      </div>
      <div className="mt-1.5 flex h-9 items-end gap-[3px]">
        {days.map((d, i) => (
          <motion.span key={d.date} title={`${new Date(`${d.date}T00:00:00`).toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' })}: ${d.count}`}
            className="flex-1 origin-bottom rounded-t-[3px] bg-brand-500/80" style={{ height: `${Math.max(12, (d.count / max) * 100)}%` }}
            initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ duration: 0.4, ease: MOTION.ease, delay: 0.1 + i * 0.02 }} />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ booths */

/** A booth as a card: the booth code tile, name, category, where, and what the team did there. */
function BoothCard({ b, i, onEdit, onRemove, onOpen }: { b: EventBooth; i: number; onEdit: () => void; onRemove: () => void; onOpen: () => void }) {
  const done = b.visitCount > 0 || b.metContacts > 0;
  return (
    <motion.div layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 16) * 0.025 }}
      className={cx('group relative flex flex-col gap-2 overflow-hidden rounded-xl border bg-card p-3 shadow-card transition-[border-color,box-shadow] duration-200 hover:border-brand-200 hover:shadow-active',
        done && 'border-emerald-200 dark:border-emerald-500/30')}>
      {done ? <span className="absolute inset-x-0 top-0 h-[3px] bg-emerald-500" aria-hidden /> : null}
      <div className="flex items-start gap-2.5">
        <button type="button" onClick={onOpen} title="Show on the floor map"
          className={cx('flex h-11 min-w-[52px] shrink-0 cursor-pointer flex-col items-center justify-center rounded-lg px-1.5 transition-transform group-hover:scale-105',
            done ? TINT.emerald : TINT.violet)}>
          <span className="text-[8.5px] font-bold uppercase leading-none tracking-[0.06em] opacity-70">{b.hall ? hallLabel(b.hall).replace(/^Hall /, 'H') : 'Booth'}</span>
          <span className="max-w-[64px] truncate font-mono text-[13px] font-bold leading-5">{b.booth || '—'}</span>
        </button>
        <div className="min-w-0 flex-1">
          <div className="break-words text-sm font-semibold leading-5 text-foreground">{b.name}</div>
          <div className="mt-0.5 flex flex-wrap items-center gap-1">
            {b.category ? <span className={cx('rounded-full px-1.5 text-[10.5px] font-semibold leading-[18px]', TINT.sky)}>{b.category}</span> : null}
            {b.country ? <span className="inline-flex items-center gap-0.5 text-[11px] text-muted-foreground"><Icon name="mapPin" size={10} />{b.country}</span> : null}
            {b.userAdded ? <span className={cx('rounded-full px-1.5 text-[10px] font-semibold leading-[18px]', TINT.amber)} title={b.addedBy ? `Added by ${b.addedBy}` : 'Added by a rep'}>Added by {b.addedBy || 'a rep'}</span> : null}
          </div>
        </div>
        <div className="flex shrink-0 gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <IconButton icon="pencil" label={`Edit ${b.name}`} onClick={onEdit} />
          <IconButton icon="trash" label={`Remove ${b.name}`} onClick={onRemove} className="hover:text-rose-600" />
        </div>
      </div>
      {b.description ? <p className="line-clamp-2 text-xs leading-snug text-muted-foreground" title={b.description}>{b.description}</p> : null}
      <div className="mt-auto flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className={cx('inline-flex items-center gap-1 rounded-full px-1.5 py-px font-semibold', b.visitCount ? TINT.emerald : 'bg-secondary text-muted-foreground')}
          title={b.visitors.length ? `Visited by ${b.visitors.join(', ')}` : 'No rep has marked it visited'}>
          <Icon name="check" size={10} strokeWidth={3} />{b.visitCount ? `${b.visitCount} visit${b.visitCount === 1 ? '' : 's'}` : 'Not visited'}
        </span>
        {b.metContacts ? (
          <span className={cx('inline-flex items-center gap-1 rounded-full px-1.5 py-px font-semibold', TINT.violet)} title="Contacts met at this event from this company">
            <Icon name="contacts" size={10} />{b.metContacts} contact{b.metContacts === 1 ? '' : 's'} met
          </span>
        ) : null}
        {b.visitors.length ? (
          <span className="ml-auto flex -space-x-1" title={b.visitors.join(', ')}>
            {b.visitors.slice(0, 4).map((v, k) => (
              <span key={`${v}-${k}`} className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-600 text-[8.5px] font-bold text-white ring-2 ring-card" style={{ zIndex: 5 - k }}>{initials(v)}</span>
            ))}
          </span>
        ) : null}
        {b.website ? (
          <a href={site(b.website)} target="_blank" rel="noreferrer" className={cx('inline-flex items-center gap-0.5 font-medium text-brand-700 hover:underline', !b.visitors.length && 'ml-auto')}>
            {b.website.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '')}<Icon name="external" size={10} />
          </a>
        ) : null}
      </div>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ floor map */

/** Shows an uploaded floor plan inline: images as a picture, PDFs in the browser's viewer, others as a download. */
function PlanViewer({ file }: { file: EventFileItem }) {
  const { toast } = useUi();
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const viewable = /^(png|jpe?g|pdf)$/i.test(file.fileType);
  useEffect(() => {
    if (!viewable) return;
    let alive = true;
    let made: string | null = null;
    setUrl(null); setErr(null);
    api.downloadFile(file.id)
      .then((f) => { if (!alive) return; made = URL.createObjectURL(base64ToBlob(f.contentBase64, f.mimeType)); setUrl(made); })
      .catch((e) => { if (alive) setErr(e); });
    return () => { alive = false; if (made) URL.revokeObjectURL(made); };
  }, [file.id, viewable]);
  const save = async () => {
    try { const f = await api.downloadFile(file.id); saveBlob(base64ToBlob(f.contentBase64, f.mimeType), f.filename); } catch (e) { toast('error', "Couldn't download", errorMessage(e)); }
  };
  return (
    <div className="overflow-hidden rounded-2xl border bg-card shadow-card">
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <IconTile icon="map" tint="emerald" size="sm" />
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-semibold text-foreground">{file.filename}</div>
          <div className="text-[11px] text-muted-foreground">Official floor plan · {fmtBytes(file.fileSize)}{file.forAllEvents ? ' · for every event' : ''}</div>
        </div>
        {url ? <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-semibold text-brand-700 hover:bg-brand-50"><Icon name="maximize" size={12} />Open full size</a> : null}
        <Button size="sm" icon="download" onClick={save}>Download</Button>
      </div>
      <div className="bg-secondary/40">
        {!viewable ? <p className="px-4 py-8 text-center text-sm text-muted-foreground">This file type can't be shown here — download it to view.</p>
          : err ? <div className="p-4"><ErrorState error={err} /></div>
            : !url ? <div className="flex h-72 items-center justify-center"><Spinner size={20} /></div>
              : file.fileType.toLowerCase() === 'pdf'
                ? <iframe title={file.filename} src={url} className="h-[560px] w-full border-0 bg-white" />
                : <a href={url} target="_blank" rel="noreferrer" className="block cursor-zoom-in"><img src={url} alt={`Floor plan: ${file.filename}`} className="mx-auto max-h-[560px] w-auto object-contain" /></a>}
      </div>
    </div>
  );
}

/**
 * The booth layout, hall by hall, from each booth's hall and number (sorted the way booths are numbered).
 * Green = a rep visited or contacts were met there; violet ring = the booth you picked.
 */
function BoothLayout({ booths, hall, onHall, focusId, onPick }: {
  booths: EventBooth[]; hall: string | undefined; onHall: (h: string) => void; focusId: string | null; onPick: (b: EventBooth) => void;
}) {
  const halls = useMemo(() => {
    const m = new Map<string, EventBooth[]>();
    booths.forEach((b) => { const h = hallOf(b); m.set(h, [...(m.get(h) || []), b]); });
    m.forEach((list) => list.sort((a, b) => byCode(a.booth, b.booth) || a.name.localeCompare(b.name)));
    return [...m.keys()].sort((a, b) => (a === NO_HALL ? 1 : b === NO_HALL ? -1 : byCode(a, b))).map((h) => ({ hall: h, booths: m.get(h)! }));
  }, [booths]);
  const current = halls.find((h) => h.hall === hall) ?? halls[0];
  if (!current) return <EmptyState icon="map" title="No booths to lay out" description="Add booths with a hall and booth number and they're laid out here, hall by hall." />;
  const done = current.booths.filter((b) => b.visitCount || b.metContacts).length;
  return (
    <div className="space-y-3">
      {halls.length > 1 ? (
        <div className="flex flex-wrap gap-1.5">
          {halls.map((h) => {
            const on = h.hall === current.hall;
            return (
              <motion.button key={h.hall} type="button" onClick={() => onHall(h.hall)} whileTap={{ scale: 0.95 }}
                className={cx('relative inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] pl-3 pr-1.5 text-[12.5px] font-bold transition-colors',
                  on ? 'border-transparent text-white' : 'border-border bg-card text-foreground hover:border-brand-200')}>
                {on ? <motion.span layoutId="es-hall-chip" className="absolute -inset-[1.5px] rounded-full bg-brand-600" transition={{ type: 'spring', stiffness: 480, damping: 38 }} /> : null}
                <span className="relative">{hallLabel(h.hall)}</span>
                <span className={cx('relative rounded-full px-1.5 text-[11px] leading-5', on ? 'bg-white/25' : 'bg-secondary text-body')}>{h.booths.length}</span>
              </motion.button>
            );
          })}
        </div>
      ) : null}
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={current.hall} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}
          className="rounded-2xl border-[1.5px] border-dashed bg-secondary/30 p-3" style={{ borderColor: 'var(--dashed)' }}>
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-sm font-extrabold text-foreground">{hallLabel(current.hall)}</span>
            <span className="text-xs text-muted-foreground">{done}/{current.booths.length} with team activity</span>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(96px,1fr))] gap-1.5">
            {current.booths.map((b, i) => {
              const active = !!(b.visitCount || b.metContacts);
              const focus = b.id === focusId;
              return (
                <motion.button key={b.id} type="button" onClick={() => onPick(b)} title={`${b.name}${b.booth ? ` · booth ${b.booth}` : ''}`}
                  initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.2, delay: Math.min(i, 30) * 0.012 }}
                  whileHover={{ y: -2 }} whileTap={{ scale: 0.95 }}
                  className={cx('relative flex h-14 cursor-pointer flex-col items-center justify-center rounded-lg border px-1 text-center transition-colors',
                    active ? 'border-emerald-300 bg-emerald-50 dark:border-emerald-500/40 dark:bg-emerald-500/10' : 'bg-card hover:border-brand-200',
                    focus && 'ring-2 ring-brand-500 ring-offset-1 ring-offset-background')}>
                  {active ? <Icon name="check" size={9} strokeWidth={3} className="absolute right-1 top-1 text-emerald-600" /> : null}
                  <span className={cx('max-w-full truncate font-mono text-[12px] font-bold', active ? 'text-emerald-700 dark:text-emerald-300' : 'text-foreground')}>{b.booth || '—'}</span>
                  <span className="max-w-full truncate text-[10px] text-muted-foreground">{b.name}</span>
                </motion.button>
              );
            })}
          </div>
          <div className="mt-2 flex items-center justify-center gap-1 text-[10px] font-bold uppercase tracking-[0.08em] text-muted-foreground"><Icon name="door" size={11} />Entrance</div>
        </motion.div>
      </AnimatePresence>
      <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-4 rounded border border-emerald-300 bg-emerald-50" />Visited or contacts met</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-4 rounded border bg-card" />Other booths</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-3 w-4 rounded border bg-card ring-2 ring-brand-500" />Picked</span>
        <span className="ml-auto">Laid out from each booth's hall and number — not to scale.</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ page */

const EMPTY_BOOTH = { name: '', booth: '', hall: '', category: '', website: '', description: '' };
type ImportState = { json: string; event: string; replace: boolean };

/** The exhibitor-list import (shared by the empty page and an event's page). */
function useImport(onImported: (firstEvent: string | undefined) => void) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const [importing, setImporting] = useState(false);
  const [imp, setImp] = useState<ImportState>({ json: '', event: '', replace: true });
  const [busy, setBusy] = useState(false);
  const open = (event: string) => { setImp({ json: '', event, replace: true }); setImporting(true); };
  const run = async () => {
    let list: unknown[];
    try {
      const parsed = JSON.parse(imp.json);
      list = Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      toast('error', "That isn't valid JSON", 'Paste an array like the example.');
      return;
    }
    setBusy(true);
    try {
      const r = await api.importExhibitors(list, imp.event.trim() || undefined, imp.replace);
      const where = Object.entries(r.events).map(([k, v]) => `${v} for ${k}`).join(', ');
      toast('success', `Imported ${r.count} exhibitor${r.count === 1 ? '' : 's'}`, [where, r.replaced ? `replaced ${r.replaced}` : '', r.skipped ? `${r.skipped} skipped (no name)` : ''].filter(Boolean).join(' · '));
      setImporting(false);
      void applyChange(qc, 'exhibitors_changed');
      onImported(Object.keys(r.events)[0]);
    } catch (e) {
      toast('error', 'Import failed', errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  return { importing, setImporting, imp, setImp, busy, open, run };
}

export function ExhibitorsComponent() {
  const navigate = useNavigate();
  const events = useSuspenseQuery(queries.adminEvents());
  const search = useSearch({ from: '/_auth/exhibitors' });
  const allEvents = events.data ?? [];
  const event = search.event || allEvents[0]?.name || '';
  const imports = useImport((first) => { if (first) navigate({ to: '/exhibitors', search: { event: first } as never, replace: true }); });
  // import suggestions: every known event, planned ones included
  const eventNames = useEventNames();

  if (!event) {
    return (
      <>
        <PageHeader title="Events" description="Every event your team works: the booth directory, the floor map, the reps there and the event's files." />
        <Card><EmptyState icon="calendar" title="No events yet" description="An event appears when reps tag contacts with it, add it in the app, or you import its exhibitor list."
          action={<Button variant="primary" icon="upload" onClick={() => imports.open('')}>Import an exhibitor list</Button>} /></Card>
        <ImportModal open={imports.importing} onOpenChange={imports.setImporting} imp={imports.imp} setImp={imports.setImp} busy={imports.busy} eventNames={eventNames} onRun={imports.run} />
      </>
    );
  }
  // keyed by event: switching events starts that event's view fresh (search box, picked booth…)
  return <EventView key={event} event={event} allEvents={allEvents} eventNames={eventNames} />;
}

/** The selected event: summary, then Booths · Floor map · Team · Files. */
function EventView({ event, allEvents, eventNames }: { event: string; allEvents: AdminEvent[]; eventNames: string[] }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { toast, confirm } = useUi();
  const search = useSearch({ from: '/_auth/exhibitors' });
  const tab: Tab = search.tab ?? 'booths';
  const category = search.category ?? '';
  const go = (patch: Record<string, string | undefined>) => navigate({ to: '/exhibitors', search: (s) => clean({ ...s, ...patch }) as never, replace: true });
  const imports = useImport((first) => { if (first) go({ event: first, tab: undefined }); });
  const d = useSuspenseQuery(queries.eventDetail(event));
  const detail = d.data;
  const booths = detail.booths;

  const [text, setText] = useState(search.q ?? '');
  useEffect(() => {
    const id = setTimeout(() => { if ((search.q ?? '') !== text.trim()) go({ q: text.trim() || undefined }); }, 300);
    return () => clearTimeout(id);
  }, [text]);
  const [editing, setEditing] = useState<EventBooth | 'new' | null>(null);
  const [form, setForm] = useState(EMPTY_BOOTH);
  const [busy, setBusy] = useState(false);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [planId, setPlanId] = useState<string | null>(null);
  const refresh = () => applyChange(qc, 'exhibitors_changed');

  const categories = useMemo(() => [...new Set(booths.map((b) => b.category).filter(Boolean))].sort(), [booths]);
  const needle = (search.q ?? '').toLowerCase();
  const show = search.show;
  const shown = useMemo(() => booths.filter((b) => (!category || b.category === category)
    && (show !== 'visited' || b.visitCount > 0) && (show !== 'met' || b.metContacts > 0)
    && (!needle || [b.name, b.booth, b.hall, b.category, b.country, b.description, b.website].join(' ').toLowerCase().includes(needle))), [booths, category, needle, show]);
  // the tiles: open a tab, or the booths list with one filter on (clearing the others)
  const boothsWith = (patch: { show?: 'visited' | 'met' | undefined; category?: string | undefined }) => {
    setText('');
    go({ tab: undefined, q: undefined, show: patch.show, category: patch.category });
  };
  const byHall = useMemo(() => {
    const m = new Map<string, EventBooth[]>();
    shown.forEach((b) => { const h = hallOf(b); m.set(h, [...(m.get(h) || []), b]); });
    m.forEach((l) => l.sort((a, b) => byCode(a.booth, b.booth) || a.name.localeCompare(b.name)));
    return [...m.entries()].sort(([a], [b]) => (a === NO_HALL ? 1 : b === NO_HALL ? -1 : byCode(a, b)));
  }, [shown]);
  const plans = detail.files.filter((f) => f.floorPlan);
  const plan = plans.find((f) => f.id === planId) ?? plans[0];
  const s = detail.stats;
  const showOnMap = (b: EventBooth) => { setFocusId(b.id); go({ tab: 'map', hall: hallOf(b) }); };
  const edit = (b: EventBooth) => { setForm({ name: b.name, booth: b.booth || '', hall: b.hall || '', category: b.category || '', website: b.website || '', description: b.description || '' }); setEditing(b); };

  const saveBooth = async () => {
    setBusy(true);
    try {
      if (editing === 'new') {
        const r = await api.addExhibitor({ event, name: form.name, booth: form.booth, hall: form.hall, category: form.category, website: form.website, description: form.description });
        toast('success', r.existed ? `${r.name} was already listed` : `Added ${r.name}`, r.existed ? 'Empty booth / hall / category were filled in.' : undefined);
      } else if (editing) {
        await api.updateExhibitor(editing.id, form);
        toast('success', `Saved ${form.name}`);
      }
      setEditing(null);
      refresh();
    } catch (e) {
      toast('error', "Couldn't save the booth", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (b: EventBooth) => {
    if (!(await confirm({ title: `Remove ${b.name}?`, message: "It's removed from this event's booth list, and reps' visits to it are cleared.", confirmLabel: 'Remove', danger: true }))) return;
    // optimistic: the booth disappears at once and comes back if the server says no
    const key = queries.eventDetail(event).queryKey;
    const before = qc.getQueryData<EventDetail>(key);
    if (before) qc.setQueryData<EventDetail>(key, { ...before, booths: before.booths.filter((x) => x.id !== b.id), stats: { ...before.stats, booths: before.stats.booths - 1 } });
    try {
      await api.deleteExhibitor(b.id);
      toast('success', `Removed ${b.name}`);
      if (focusId === b.id) setFocusId(null);
    } catch (e) {
      if (before) qc.setQueryData(key, before);
      toast('error', "Couldn't remove — restored", errorMessage(e));
    } finally {
      refresh();
    }
  };

  const focused = focusId ? booths.find((x) => x.id === focusId) : undefined;

  return (
    <>
      <PageHeader title="Events"
        description="Each event your team works: its booth directory (reps see it in the app, mark booths visited and get a route), the floor map, the reps there and the event's files."
        actions={<>
          <Button icon="upload" onClick={() => imports.open(event)}>Import list</Button>
          <Button variant="primary" icon="plus" onClick={() => { setForm(EMPTY_BOOTH); setEditing('new'); }}>Add booth</Button>
        </>} />

      {/* the event, and what happened there */}
      <div className="mb-3 flex flex-wrap items-stretch gap-2">
        <EventPicker events={allEvents} value={event} onChange={(name) => go({ event: name, category: undefined, q: undefined, hall: undefined, show: undefined })} />
        <CaptureBars perDay={detail.perDay} />
        <Link to="/contacts" search={{ event } as never}
          className="inline-flex items-center gap-2 rounded-xl border bg-card px-3 text-sm font-semibold text-brand-700 shadow-card transition-colors hover:border-brand-300">
          <Icon name="contacts" size={15} />Contacts met here<Icon name="chevronRight" size={14} />
        </Link>
      </div>
      {/* one row of tiles; on a narrow window the row scrolls sideways rather than cutting labels off */}
      <div className="no-scrollbar -mx-1 mb-3 flex gap-2 overflow-x-auto px-1 py-1">
        <StatTile i={0} icon="store" tint="violet" label="Booths" value={s.booths} hint="Show every booth"
          active={tab === 'booths' && !show && !category && !search.q} onClick={() => boothsWith({})} />
        <StatTile i={1} icon="map" tint="sky" label="Halls" value={s.halls} hint="Open the floor map, hall by hall"
          active={tab === 'map'} onClick={() => go({ tab: 'map' })} />
        <StatTile i={2} icon="list" tint="slate" label="Categories" value={s.categories} hint="Show the booths with the category filters"
          active={tab === 'booths' && !!category} onClick={() => boothsWith({ category: categories[0] })} />
        <StatTile i={3} icon="check" tint="emerald" label="Booths visited" value={s.boothsVisited} hint="Show only booths a rep marked visited"
          active={tab === 'booths' && show === 'visited'} onClick={() => boothsWith({ show: show === 'visited' ? undefined : 'visited' })} />
        <StatTile i={4} icon="building" tint="emerald" label="Booths met" value={s.boothsMet} hint="Show only booths whose company the team met (contacts from this event)"
          active={tab === 'booths' && show === 'met'} onClick={() => boothsWith({ show: show === 'met' ? undefined : 'met' })} />
        <StatTile i={5} icon="users" tint="amber" label="Reps" value={s.reps} hint="Open the team at this event"
          active={tab === 'team'} onClick={() => go({ tab: 'team' })} />
        <StatTile i={6} icon="contacts" tint="violet" label="Contacts" value={s.contacts} hint="Open Contacts filtered to this event" external
          onClick={() => navigate({ to: '/contacts', search: { event } as never })} />
        <StatTile i={7} icon="flame" tint="rose" label="Hot leads" value={s.hot} hint="Open this event's hot leads (score 85+) in Contacts" external
          onClick={() => navigate({ to: '/contacts', search: { event, band: 'A' } as never })} />
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <SlidingTabs value={tab} onChange={(t) => go({ tab: t === 'booths' ? undefined : t })} size="sm"
          options={[
            { value: 'booths', label: 'Booths', count: s.booths },
            { value: 'map', label: 'Floor map', count: plans.length || undefined },
            // the same number as the Reps tile (stats.reps = the reps listed on the Team tab)
            { value: 'team', label: 'Team', count: s.reps },
            { value: 'files', label: 'Files', count: detail.files.length },
          ]} />
        {tab === 'booths' ? (
          <>
            <SearchInput value={text} onChange={setText} placeholder="Search booths…" className="w-full sm:w-64" />
            <AnimatePresence>
              {show ? (
                <motion.button key={show} type="button" onClick={() => go({ show: undefined })} title="Show every booth again"
                  initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} transition={{ duration: MOTION.fast }}
                  className="inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] border-emerald-600 bg-emerald-600 pl-2.5 pr-1.5 text-xs font-bold text-white">
                  <Icon name={show === 'visited' ? 'check' : 'building'} size={12} />
                  {show === 'visited' ? 'Visited by a rep' : 'Contacts met'}
                  <span className="flex size-5 items-center justify-center rounded-full bg-white/20"><Icon name="x" size={11} strokeWidth={3} /></span>
                </motion.button>
              ) : null}
            </AnimatePresence>
            {categories.length ? (
              <div className="no-scrollbar flex min-w-0 flex-1 gap-1 overflow-x-auto">
                {['', ...categories].map((c) => (
                  <button key={c || 'all'} type="button" onClick={() => go({ category: c || undefined })}
                    className={cx('h-8 shrink-0 cursor-pointer rounded-full border-[1.5px] px-2.5 text-xs font-bold transition-colors',
                      category === c ? 'border-brand-600 bg-brand-600 text-white' : 'border-border bg-card text-foreground hover:border-brand-200')}>{c || 'All categories'}</button>
                ))}
              </div>
            ) : null}
            <span className="ml-auto text-xs text-muted-foreground">{shown.length} of {booths.length}</span>
          </>
        ) : null}
      </div>

      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}>
          {tab === 'booths' ? (
            !booths.length ? (
              <Card><EmptyState icon="store" title="No booth directory for this event yet"
                description="Import the organiser's exhibitor list or add booths one by one. Reps then see them on the event page in the app."
                action={<Button variant="primary" icon="upload" onClick={() => imports.open(event)}>Import list</Button>} /></Card>
            ) : !shown.length ? (
              <Card><EmptyState icon={show === 'visited' ? 'check' : show === 'met' ? 'building' : 'search'}
                title={show === 'visited' ? 'No booth has been marked visited yet' : show === 'met' ? 'No contacts met from these companies yet' : 'No booths match'}
                description={show === 'visited' ? 'Reps mark booths visited on the event page in the app.'
                  : show === 'met' ? "A booth counts once a contact from this event works at that company."
                    : 'Try another search or category.'}
                action={<Button icon="x" onClick={() => boothsWith({})}>Show every booth</Button>} /></Card>
            ) : (
              <div className="space-y-4">
                {byHall.map(([h, list]) => (
                  <section key={h}>
                    <div className="mb-2 flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.08em] text-muted-foreground">
                      <Icon name="map" size={12} />{hallLabel(h)}<span className="font-semibold normal-case tracking-normal">· {list.length} booth{list.length === 1 ? '' : 's'}</span>
                    </div>
                    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
                      <AnimatePresence initial={true}>
                        {list.map((b, i) => <BoothCard key={b.id} b={b} i={i} onOpen={() => showOnMap(b)} onEdit={() => edit(b)} onRemove={() => remove(b)} />)}
                      </AnimatePresence>
                    </div>
                  </section>
                ))}
              </div>
            )
          ) : tab === 'map' ? (
            <div className="space-y-3">
              {plan ? (
                <>
                  {plans.length > 1 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {plans.map((p) => (
                        <button key={p.id} type="button" onClick={() => setPlanId(p.id)}
                          className={cx('h-8 cursor-pointer rounded-full border-[1.5px] px-3 text-xs font-bold', p.id === plan.id ? 'border-emerald-600 bg-emerald-600 text-white' : 'border-border bg-card hover:border-emerald-300')}>{p.filename}</button>
                      ))}
                    </div>
                  ) : null}
                  <PlanViewer file={plan} />
                </>
              ) : (
                <Notice tone="amber" icon="map">
                  No official floor plan for {event} yet. Upload it (PNG, JPG or PDF) in <Link to="/files" className="font-semibold underline">Team files</Link> with
                  the category <b>Floor plan</b> and this event, and it shows here and in the reps' app.
                </Notice>
              )}
              <Card className="p-4">
                <div className="mb-3 flex items-center gap-2">
                  <IconTile icon="store" tint="violet" size="sm" />
                  <div>
                    <div className="text-sm font-semibold text-foreground">Booth layout</div>
                    <div className="text-xs text-muted-foreground">Every booth by hall and number. Click one to see it.</div>
                  </div>
                </div>
                <BoothLayout booths={booths} hall={search.hall} onHall={(h) => go({ hall: h })} focusId={focusId}
                  onPick={(b) => setFocusId(b.id === focusId ? null : b.id)} />
                <AnimatePresence>
                  {focused ? (
                    <motion.div key={focused.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }} transition={{ duration: MOTION.base }} className="mt-3 max-w-md">
                      <BoothCard b={focused} i={0} onOpen={() => {}} onEdit={() => edit(focused)} onRemove={() => remove(focused)} />
                    </motion.div>
                  ) : null}
                </AnimatePresence>
              </Card>
            </div>
          ) : tab === 'team' ? (
            !detail.reps.length ? <Card><EmptyState icon="users" title="No reps at this event yet" description="Reps show up once they capture a contact here, mark a booth visited or add the event in the app." /></Card> : (
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {detail.reps.map((r, i) => <RepCard key={r.userId} r={r} i={i} event={event} />)}
              </div>
            )
          ) : (
            !detail.files.length ? (
              <Card><EmptyState icon="file" title="No files for this event" description="Upload brochures, price lists or the floor plan in Team files and pick this event."
                action={<Link to="/files"><Button icon="upload">Open Team files</Button></Link>} /></Card>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {detail.files.map((f, i) => <FileCard key={f.id} f={f} i={i} onView={f.floorPlan ? () => { setPlanId(f.id); go({ tab: 'map' }); } : undefined} />)}
              </div>
            )
          )}
        </motion.div>
      </AnimatePresence>

      <Modal open={editing !== null} onOpenChange={(o) => { if (!o) setEditing(null); }}
        title={editing === 'new' ? `Add a booth to ${event}` : `Edit ${editing ? editing.name : 'booth'}`}
        footer={<>
          <Button variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
          <Button variant="primary" loading={busy} disabled={!form.name.trim()} onClick={saveBooth}>{editing === 'new' ? 'Add booth' : 'Save'}</Button>
        </>}>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Company / booth name *" className="sm:col-span-2"><Input autoFocus value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="Booth number"><Input value={form.booth} onChange={(e) => setForm({ ...form, booth: e.target.value })} placeholder="B-12" /></Field>
          <Field label="Hall"><Input value={form.hall} onChange={(e) => setForm({ ...form, hall: e.target.value })} placeholder="5" /></Field>
          <Field label="Category"><Input list="es-cats" value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })} /><datalist id="es-cats">{categories.map((c) => <option key={c} value={c} />)}</datalist></Field>
          <Field label="Website"><Input value={form.website} onChange={(e) => setForm({ ...form, website: e.target.value })} placeholder="example.com" /></Field>
          <Field label="Description" className="sm:col-span-2"><Textarea rows={3} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></Field>
        </div>
      </Modal>
      <ImportModal open={imports.importing} onOpenChange={imports.setImporting} imp={imports.imp} setImp={imports.setImp} busy={imports.busy}
        eventNames={eventNames.length ? eventNames : allEvents.map((e) => e.name)} onRun={imports.run} />
    </>
  );
}

/** A rep at this event: what they captured, booths they visited, and whether it's their current event. */
function RepCard({ r, i, event }: { r: EventRep; i: number; event: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 12) * 0.03 }}
      className="flex items-center gap-3 rounded-xl border bg-card p-3 shadow-card">
      <Avatar name={r.name} size="md" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="break-words text-sm font-semibold text-foreground">{r.name}</span>
          {r.active ? <Badge tone="green">Current event</Badge> : r.following ? <Badge tone="sky">In their events</Badge> : null}
        </div>
        <div className="mt-0.5 flex flex-wrap gap-x-2.5 text-xs text-muted-foreground">
          <span><b className="font-semibold text-foreground">{r.contacts}</b> contact{r.contacts === 1 ? '' : 's'}</span>
          {r.hot ? <span className="text-heat-hot"><b className="font-semibold">{r.hot}</b> hot</span> : null}
          <span><b className="font-semibold text-foreground">{r.boothsVisited}</b> booth{r.boothsVisited === 1 ? '' : 's'} visited</span>
          {r.lastCapture ? <span title={fmtDateTime(r.lastCapture)}>last {relTime(r.lastCapture)}</span> : null}
        </div>
      </div>
      {r.contacts ? (
        <Link to="/contacts" search={{ event, owner: r.userId } as never} title="Their contacts from this event"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-brand-700"><Icon name="chevronRight" size={16} /></Link>
      ) : null}
    </motion.div>
  );
}

function FileCard({ f, i, onView }: { f: EventFileItem; i: number; onView?: (() => void) | undefined }) {
  const { toast } = useUi();
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    try { const r = await api.downloadFile(f.id); saveBlob(base64ToBlob(r.contentBase64, r.mimeType), r.filename); } catch (e) { toast('error', "Couldn't download", errorMessage(e)); } finally { setBusy(false); }
  };
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 12) * 0.03 }}
      className="group flex items-center gap-3 rounded-xl border bg-card p-3 shadow-card">
      <IconTile icon={f.floorPlan ? 'map' : 'file'} tint={f.floorPlan ? 'emerald' : 'sky'} />
      <div className="min-w-0 flex-1">
        <div className="break-words text-sm font-semibold text-foreground">{f.filename}</div>
        <div className="flex flex-wrap gap-x-2 text-xs text-muted-foreground">
          <span>{f.fileType.toUpperCase()} · {fmtBytes(f.fileSize)}</span>
          {f.floorPlan ? <span className="font-semibold text-emerald-700 dark:text-emerald-400">Floor plan</span> : null}
          {f.forAllEvents ? <span>For every event</span> : null}
        </div>
      </div>
      {onView ? <Button size="sm" variant="ghost" icon="eye" onClick={onView}>View</Button> : null}
      {/* disabled while busy (it only dimmed before), so a double-click downloads once */}
      <IconButton icon="download" label={`Download ${f.filename}`} onClick={save} disabled={busy} className={busy ? 'opacity-50' : ''} />
    </motion.div>
  );
}

function ImportModal({ open, onOpenChange, imp, setImp, busy, eventNames, onRun }: {
  open: boolean; onOpenChange: (o: boolean) => void; imp: { json: string; event: string; replace: boolean };
  setImp: React.Dispatch<React.SetStateAction<ImportState>>; busy: boolean; eventNames: string[]; onRun: () => void;
}) {
  const { toast } = useUi();
  return (
    <Modal open={open} onOpenChange={onOpenChange} size="lg" title="Import an exhibitor list" description="Paste a JSON array, or load a .json file."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" icon="upload" loading={busy} disabled={!imp.json.trim()} onClick={onRun}>Import</Button>
      </>}>
      <div className="space-y-4">
        <Field label="Event name" hint="Exactly as reps pick it in the app. An item's own event_name overrides it.">
          <Input list="es-import-events" value={imp.event} onChange={(e) => setImp({ ...imp, event: e.target.value })} placeholder="e.g. GITEX 2026" />
          <datalist id="es-import-events">{eventNames.map((n) => <option key={n} value={n} />)}</datalist>
        </Field>
        <Field label="Exhibitors (JSON)" hint="Each needs a name; booth, hall, category, subcategory, country, website, description, products, tags are optional.">
          <Textarea rows={10} className="font-mono text-xs" value={imp.json} onChange={(e) => setImp({ ...imp, json: e.target.value })} placeholder={SAMPLE} />
        </Field>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-sm text-body">
            <input type="checkbox" className="h-4 w-4 accent-violet-600" checked={imp.replace} onChange={(e) => setImp({ ...imp, replace: e.target.checked })} />
            Replace this event's imported list (booths reps added are kept)
          </label>
          <label className="cursor-pointer text-sm font-medium text-accent-foreground hover:underline">
            Load .json file
            <input type="file" accept=".json,application/json" className="hidden" onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) f.text().then((t) => setImp((s) => ({ ...s, json: t }))).catch(() => toast('error', "Couldn't read that file"));
              e.target.value = '';
            }} />
          </label>
        </div>
        {!imp.event.trim() ? <Notice tone="amber" icon="alert">Without an event name, every item must have its own <code className="font-mono text-xs">event_name</code>.</Notice> : null}
      </div>
    </Modal>
  );
}
