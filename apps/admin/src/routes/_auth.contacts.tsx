// /contacts — every rep's contacts in one TanStack Table over the real API:
//   rows      GET /api/contacts?lite=1 (search · level · event · rep · added today · sort; pages followed up to lib/api MAX_CONTACT_ROWS)
//   counts    GET /api/contacts/stats (same filters without the level)
//   add       POST /api/contacts (+ PATCH for LinkedIn)      score   POST /api/contacts/score_all · /api/contact/{id}/score
//   delete    POST /api/admin/contacts/batch_delete · DELETE /api/contacts/{id}       export  GET /api/admin/export
//   lists     POST /api/admin/contact_lists/{id}/contacts    drawer  /contacts/$id (features/contacts/ContactDrawer)
// The grid is the shadcn/ui Data Table pattern (components/data-table): column-header menus
// (sort / hide), faceted filters, a View menu and pagination, all driven by TanStack Table.
// Filters, sort column + direction and the drawer tab live in the URL; column visibility, widths,
// density and rows per page are remembered per browser (features/contacts/table.ts).
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { Link, Outlet, useMatch, useNavigate, useSearch } from '@tanstack/react-router';
import {
  useReactTable, getCoreRowModel, getSortedRowModel, getPaginationRowModel, flexRender,
  type Column, type ColumnDef, type PaginationState, type Row as TRow, type SortingState,
} from '@tanstack/react-table';
import { AnimatePresence, motion } from 'motion/react';
import { Checkbox as ShadCheckbox } from '@/components/ui/checkbox';
import { TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { DataTablePagination } from '@/components/data-table/data-table-pagination';
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useIsMobile } from '@/hooks/use-mobile';
import { ApiError, MAX_CONTACT_ROWS, api, fileToBase64, saveBlob } from '../lib/api';
import type { Contact } from '../lib/types';
import { getSession } from '../lib/auth';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { clean, contactQuery, contactCountsQuery, sortDir, type ContactSort, type ContactsSearch } from '../lib/search';
import { removeContacts } from '../lib/optimistic';
import { AddToListModal } from '../features/contacts/AddToListModal';
import { ColumnsMenu, FilterChip, FilterPill, LayoutToggle, SearchField, SortMenu, type PillOption } from '../features/contacts/Toolbar';
import { contactsToCsv } from '../features/contacts/csv';
import { moveFocus, selectRange, selectedIn, shouldHandleKey, toggle, type Selection } from '../features/contacts/selection';
import {
  COLUMN_IDS, COLUMN_LABELS, DEFAULT_PREFS, PAGE_SIZES, loadPrefs, savePrefs, searchForSorting, sortingFor, type ColumnId, type TablePrefs,
} from '../features/contacts/table';
import { contactLevel, levelCounts, levelFrom } from '../lib/levels';
import { Avatar } from '../ui/Avatar';
import { Icon, type IconName } from '../ui/icons';
import { LiveIcon } from '../ui/live-icon';
import { Menu, type MenuItem } from '../ui/menu';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import {
  Button, Card, EmptyState, ErrorState, Field, IconButton, Input, LEVELS, Modal, Select, Skeleton, Spinner,
  Textarea, contactName, cx, deep, errorMessage, fmtDateTime, ink, levelOf, parseTime, relTime, useUi, wasEdited,
} from '../ui/kit';

type Row = Contact & { ownerName: string; ownerPhoto: string | null };
type Band = 'A' | 'B' | 'C' | 'D' | 'U';
const EMPTY_LEAD = { firstName: '', lastName: '', company: '', title: '', email: '', phone: '', linkedin: '', eventId: '', notes: '', userId: '', photo: '' };
const SOURCE_LABEL: Record<string, string> = { scan: 'Card scan', qr: 'QR code', manual: 'Added by hand', import: 'Import', linkedin: 'LinkedIn', sync: 'Phone (offline)' };
const SOURCE_ICON: Record<string, IconName> = { scan: 'camera', qr: 'qr', manual: 'pencil', import: 'upload', linkedin: 'link', sync: 'smartphone' };
const time = (iso?: string | null) => parseTime(iso)?.getTime() ?? 0;
const NO_CONTACTS: { contacts: Contact[] } = { contacts: [] };
const EMPTY = <span className="text-muted-foreground">—</span>;

/** Sticky left columns (checkbox + name) stay put while the rest scrolls sideways. */
function pinStyle(col: Column<Row>): React.CSSProperties {
  return col.getIsPinned() ? { position: 'sticky', left: col.getStart('left'), zIndex: 2 } : {};
}

/** The contact's own photo, else their initials, like the representative (never the generic stock avatar). */
function ContactAvatar({ c, size = 'sm', className }: { c: Contact; size?: 'xs' | 'sm' | 'md' | 'lg'; className?: string }) {
  return <Avatar src={c.photoUrl || null} name={contactName(c)} size={size} className={cx('border-0 shadow-none', className)} />;
}

const linkedinHref = (v?: string | null) => (v ? (/^https?:/i.test(v) ? v : `https://${v}`) : '');
/** "linkedin.com/in/jane-doe" → "in/jane-doe" for display. */
const linkedinLabel = (v?: string | null) => (v || '').replace(/^https?:\/\//i, '').replace(/^(www\.)?linkedin\.com\//i, '').replace(/\/$/, '');

/** Minimum column widths (px) in the table; a column grows past its minimum to fit its longest value. */
const COLUMN_MIN: Record<ColumnId, number> = {
  name: 240, company: 170, title: 170, phone: 150, email: 210, linkedin: 84,
  owner: 150, event: 150, score: 110, source: 120, added: 100, updated: 110,
};

/** Card scans often store "N/A" (or "-", "none") for a missing field; the console treats those as empty. */
const PLACEHOLDER = /^\s*(n\/?a|none|null|nil|-+|–|—)?\s*$/i;
const blank = (v?: string | null) => (v == null || PLACEHOLDER.test(v) ? null : v);
/** One field can hold several values ("+91 124…, +91 993…"): split on , ; | newlines and " / ". */
const splitValues = (v?: string | null) => (blank(v) ?? '').split(/\s*[,;|\n]\s*|\s+\/\s+/).map((s) => s.trim()).filter((s) => s && !PLACEHOLDER.test(s));

/** LinkedIn's "in" mark (lucide has no brand icons). `muted` = a grey outline version for "no profile". */
function LinkedInLogo({ size = 16, muted }: { size?: number; muted?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden className="shrink-0">
      <rect width="24" height="24" rx="5" fill={muted ? 'currentColor' : '#0a66c2'} opacity={muted ? 0.35 : 1} />
      <path fill="#fff" d="M7.2 9.6h2.3V17H7.2zM8.35 6a1.33 1.33 0 1 1 0 2.66 1.33 1.33 0 0 1 0-2.66zM11 9.6h2.2v1h.03c.31-.58 1.06-1.2 2.18-1.2 2.33 0 2.76 1.53 2.76 3.53V17h-2.3v-3.6c0-.86-.02-1.97-1.2-1.97-1.2 0-1.38.94-1.38 1.9V17H11z" />
    </svg>
  );
}

/**
 * Phone / email cell: the first value on one line (never wrapped or cut), and "+N" when the field
 * holds more. The chip opens a dropdown with every value, each with copy and call / email.
 */
function MultiValue({ kind, raw, onCopy }: { kind: 'phone' | 'email'; raw: string | null; onCopy: (what: string, value: string) => void }) {
  const values = splitValues(raw);
  const [open, setOpen] = useState(false);
  if (!values.length) return EMPTY;
  const icon: IconName = kind === 'phone' ? 'phone' : 'mail';
  const what = kind === 'phone' ? 'Phone' : 'Email';
  const href = (v: string) => (kind === 'phone' ? `tel:${v.replace(/[^\d+]/g, '')}` : `mailto:${v}`);
  const first = values[0]!;
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
      <a href={href(first)} title={`${kind === 'phone' ? 'Call' : 'Email'} ${first}`}
        className={cx('text-body underline-offset-2 hover:text-foreground hover:underline', kind === 'phone' && 'tabular-nums')}>{first}</a>
      {values.length > 1 ? (
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <button aria-label={`Show all ${values.length} ${kind === 'phone' ? 'phone numbers' : 'emails'}`}
              className="inline-flex h-[18px] cursor-pointer items-center rounded-full bg-brand-50 px-1.5 text-[10.5px] font-black text-brand-700 transition-colors hover:bg-brand-100 data-[state=open]:bg-brand-600 data-[state=open]:text-white">
              +{values.length - 1}
            </button>
          </PopoverTrigger>
          <PopoverContent align="start" sideOffset={6} className="w-auto min-w-[240px] rounded-xl p-1" onClick={(e) => e.stopPropagation()}>
            <div className="px-2 pb-1 pt-1.5 text-[10px] font-bold uppercase tracking-[0.07em] text-muted-foreground">
              {values.length} {kind === 'phone' ? 'phone numbers' : 'emails'}
            </div>
            {values.map((v, i) => (
              <div key={`${v}-${i}`} className="group/val flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-secondary">
                <span className={cx('flex size-6 shrink-0 items-center justify-center rounded-md', kind === 'phone' ? 'bg-success-bg text-success' : 'bg-info-bg text-info')}>
                  <Icon name={icon} size={12} />
                </span>
                <a href={href(v)} className={cx('flex-1 whitespace-nowrap text-[12.5px] font-medium text-foreground hover:underline', kind === 'phone' && 'tabular-nums')}>{v}</a>
                {i === 0 ? <span className="rounded-full bg-secondary px-1.5 text-[9.5px] font-bold uppercase text-muted-foreground">Main</span> : null}
                <button onClick={() => onCopy(what, v)} aria-label={`Copy ${v}`}
                  className="flex size-6 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-card hover:text-foreground"><Icon name="copy" size={12} /></button>
              </div>
            ))}
          </PopoverContent>
        </Popover>
      ) : null}
    </span>
  );
}

type ReachField = 'email' | 'phone' | 'linkedin';

/**
 * Email · phone · LinkedIn icons. Clicking any opens a panel with every detail of the contact
 * (the clicked one highlighted), each with copy / open actions and a link to the full profile.
 */
function ContactReach({ c, ownerName, onCopy, onOpenProfile, size = 'md', hideEmptyLinkedIn }: {
  c: Contact; ownerName?: string; onCopy: (what: string, value: string) => void; onOpenProfile: () => void; size?: 'sm' | 'md';
  /** cards: leave the LinkedIn button (and its row) out entirely when there's no profile */
  hideEmptyLinkedIn?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [focusOn, setFocusOn] = useState<ReachField>('email');
  const li = linkedinHref(c.linkedin);
  const lvl = contactLevel(c);
  const allIcons: { field: ReachField; icon: IconName; label: string; has: boolean; tint: string }[] = [
    { field: 'email', icon: 'mail', label: c.email ? `Email · ${c.email}` : 'No email', has: !!c.email, tint: 'hover:bg-info-bg hover:text-info' },
    { field: 'phone', icon: 'phone', label: c.phone ? `Phone · ${c.phone}` : 'No phone number', has: !!c.phone, tint: 'hover:bg-success-bg hover:text-success' },
    { field: 'linkedin', icon: 'link', label: c.linkedin ? 'LinkedIn profile' : 'No LinkedIn', has: !!c.linkedin, tint: 'hover:bg-[#e8f0fb] hover:border-[#0a66c2]/40 dark:hover:bg-[#0a66c2]/20' },
  ];
  const icons = allIcons.filter((it) => !(hideEmptyLinkedIn && it.field === 'linkedin' && !it.has));
  const rows: { field?: ReachField; icon: IconName; label: string; value?: string | null | undefined; href?: string | undefined; copy?: boolean }[] = [
    { field: 'email', icon: 'mail', label: 'Email', value: c.email, href: c.email ? `mailto:${c.email}` : undefined, copy: true },
    { field: 'phone', icon: 'phone', label: 'Phone', value: c.phone, href: c.phone ? `tel:${c.phone}` : undefined, copy: true },
    ...(hideEmptyLinkedIn && !c.linkedin ? [] : [{ field: 'linkedin' as const, icon: 'link' as IconName, label: 'LinkedIn', value: c.linkedin ? linkedinLabel(c.linkedin) : null, href: li || undefined, copy: true }]),
    { icon: 'building', label: 'Company', value: c.company },
    { icon: 'idCard', label: 'Designation', value: c.title },
    { icon: 'calendar', label: 'Event', value: c.eventId },
    { icon: 'user', label: 'Representative', value: ownerName },
    { icon: SOURCE_ICON[c.source || ''] || 'bolt', label: 'Source', value: SOURCE_LABEL[c.source || ''] || c.source },
    { icon: 'clock', label: 'Added', value: c.createdAt ? `${fmtDateTime(c.createdAt)} · ${relTime(c.createdAt)}` : null },
  ];
  const btn = size === 'sm' ? 'h-7 w-7' : 'h-8 w-8';
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverAnchor asChild>
        <span className="inline-flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          {icons.map((it) => (
            <Tooltip key={it.field}>
              <TooltipTrigger asChild>
                <button aria-label={it.label} onClick={() => { setFocusOn(it.field); setOpen(true); }}
                  className={cx('flex cursor-pointer items-center justify-center rounded-lg border transition-[color,background-color,border-color,transform] duration-200 hover:-translate-y-px hover:scale-110 active:scale-90', btn,
                    it.has ? cx('border-border bg-card text-body', it.tint) : 'border-dashed text-muted-foreground hover:text-muted-foreground',
                    open && focusOn === it.field && 'border-primary bg-accent text-accent-foreground')}>
                  {it.field === 'linkedin'
                    ? <LinkedInLogo size={size === 'sm' ? 14 : 15} muted={!it.has} />
                    : <Icon name={it.icon} size={size === 'sm' ? 13 : 14} />}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">{it.label}</TooltipContent>
            </Tooltip>
          ))}
        </span>
      </PopoverAnchor>
      <PopoverContent align="start" sideOffset={6} className="w-[340px] rounded-xl p-0" onClick={(e) => e.stopPropagation()}>
        {/* who */}
        <div className="flex items-start gap-3 border-b p-3.5">
          <span className="rounded-full p-[2px]" style={{ background: lvl ? heatVar(lvl.key) : 'transparent', boxShadow: lvl ? undefined : 'inset 0 0 0 1.5px var(--dashed)' }}>
            <span className="block rounded-full bg-card p-px"><ContactAvatar c={c} size="md" /></span>
          </span>
          <div className="min-w-0 flex-1">
            <div className="break-words text-[15px] font-semibold leading-5 text-foreground">{contactName(c)}</div>
            {c.title ? <div className="break-words text-[13px] text-body">{c.title}</div> : null}
            {c.company ? <div className="break-words text-[13px] text-muted-foreground">{c.company}</div> : null}
          </div>
          <span className="flex shrink-0 flex-col items-center gap-0.5">
            <ScoreRing score={c.score} level={c.level} size={36} />
            <span className="font-mono text-[9px] font-semibold uppercase tracking-[0.06em]" style={{ color: lvl ? heatVar(lvl.key) : 'var(--muted-foreground)' }}>{lvl?.label ?? 'Unscored'}</span>
          </span>
        </div>
        {/* every detail */}
        <dl className="max-h-[320px] overflow-y-auto p-1.5">
          {rows.map((r) => (
            <div key={r.label} className={cx('group/detail flex items-start gap-2.5 rounded-lg px-2 py-1.5', r.field && r.field === focusOn && 'bg-accent/60')}>
              {r.field === 'linkedin'
                ? <span className="mt-[3px] flex shrink-0 text-muted-foreground"><LinkedInLogo size={14} muted={!r.value} /></span>
                : <Icon name={r.icon} size={14} className="mt-[3px] shrink-0 text-muted-foreground" />}
              <div className="min-w-0 flex-1">
                <dt className="font-mono text-[9.5px] uppercase tracking-[0.07em] text-muted-foreground">{r.label}</dt>
                <dd className={cx('break-words text-[13px] [overflow-wrap:anywhere]', r.value ? 'text-foreground' : 'italic text-muted-foreground')}>
                  {r.value ? (r.href ? <a href={r.href} target={r.href.startsWith('http') ? '_blank' : undefined} rel="noopener" className="underline-offset-2 hover:underline">{r.value}</a> : r.value) : 'Not added'}
                </dd>
              </div>
              {r.copy && r.value ? (
                <button onClick={() => onCopy(r.label, r.field === 'linkedin' ? li : r.value!)} aria-label={`Copy ${r.label.toLowerCase()}`}
                  className="mt-0.5 flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground opacity-60 transition hover:bg-card hover:text-foreground group-hover/detail:opacity-100">
                  <Icon name="copy" size={12} />
                </button>
              ) : null}
            </div>
          ))}
        </dl>
        <div className="flex items-center gap-1.5 border-t p-2">
          {c.email ? <a href={`mailto:${c.email}`} className="inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-lg border bg-card text-xs font-medium text-body hover:border-foreground/25 hover:text-foreground"><Icon name="mail" size={13} />Email</a> : null}
          {c.phone ? <a href={`tel:${c.phone}`} className="inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-lg border bg-card text-xs font-medium text-body hover:border-foreground/25 hover:text-foreground"><Icon name="phone" size={13} />Call</a> : null}
          <button onClick={() => { setOpen(false); onOpenProfile(); }} className="inline-flex h-8 flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-lg bg-fill text-xs font-semibold text-fill-foreground hover:bg-fill-hover">
            Full profile<Icon name="chevronRight" size={13} />
          </button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

/** Heat colour of a lead level (tokens.css --heat-*), the same ramp as the dashboard. */
const heatVar = (key?: string | null) => (key ? `var(--heat-${key})` : 'var(--heat-none)');

/** The mobile feed's score ring: a heat-coloured arc that draws to the score, the number inside. */
function ScoreRing({ score, level, size = 32 }: { score: number | null | undefined; level?: string | null | undefined; size?: number }) {
  const lvl = levelFrom(score, level);
  const heat = heatVar(lvl?.key);
  // the app's ring: a disc tinted with the level, a faint track and an arc that draws to the score
  const stroke = Math.max(3, Math.round(size / 10));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <span className="relative inline-flex shrink-0 items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} strokeWidth={stroke}
          style={score == null ? { stroke: 'var(--border)', fill: 'none' } : { stroke: `color-mix(in oklab, ${heat} 22%, transparent)`, fill: `color-mix(in oklab, ${heat} 9%, var(--card))` }}
          strokeDasharray={score == null ? '2 3' : undefined} />
        {score != null ? (
          <motion.circle key={score} cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} strokeLinecap="round"
            style={{ stroke: heat }} strokeDasharray={`${c} ${c}`}
            initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - Math.max(0, Math.min(100, score)) / 100) }}
            transition={{ duration: MOTION.slow, ease: MOTION.ease }} />
        ) : null}
      </svg>
      <span className={cx('es-num absolute font-black', size >= 40 ? 'text-[13px]' : size >= 28 ? 'text-[11px]' : 'text-[9.5px]', score == null ? 'text-muted-foreground' : 'text-foreground')}>{score ?? '–'}</span>
    </span>
  );
}

/**
 * Lead-level switch beside the page title: All · Hot · Warm · Cool · Cold · Not scored, as one compact
 * segmented bar (a thumb slides to the picked level), then Deleted. It sizes to the room it gets
 * (container query): wide, every segment shows its label; narrow, only dots + counts (the picked one
 * keeps its label) and the bar scrolls sideways rather than wrapping.
 */
function LevelTiles({ counts, band, onBand, deleted }: { counts: ReturnType<typeof levelCounts>; band: Band | undefined; onBand: (b: Band | undefined) => void; deleted: number }) {
  const tabs: { band: Band | undefined; label: string; count: number; color: string; hint: string }[] = [
    { band: undefined, label: 'All', count: counts.all, color: 'var(--brand-600)', hint: 'Every contact' },
    ...LEVELS.map((l) => ({ band: l.band as Band, label: l.label, count: counts[l.band], color: heatVar(l.key),
      hint: { hot: 'Score 85–100', warm: 'Score 70–84', cool: 'Score 50–69', cold: 'Score below 50' }[l.key] })),
    { band: 'U', label: 'Not scored', count: counts.U, color: 'var(--muted-foreground)', hint: 'Waiting for an AI score' },
  ];
  const share = (n: number) => (counts.all ? Math.round((n / counts.all) * 100) : 0);
  return (
    <div className="@container min-w-0 flex-1">
      <div className="no-scrollbar flex items-center gap-1.5 overflow-x-auto py-0.5">
        <div role="tablist" aria-label="Lead level" className="inline-flex h-8 shrink-0 items-center gap-px rounded-full border bg-secondary/70 p-[3px]">
          {tabs.map((t) => {
            const on = band === t.band;
            return (
              <button key={t.label} role="tab" aria-selected={on} aria-label={`${t.label}, ${t.count} contacts`} onClick={() => onBand(t.band)}
                title={`${t.label} · ${t.hint} · ${t.count.toLocaleString()} contacts${t.band ? ` · ${share(t.count)}%` : ''}`}
                className={cx('relative inline-flex h-full shrink-0 cursor-pointer items-center gap-1.5 rounded-full px-2 text-[12px] font-bold transition-colors duration-200 active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  on ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
                {on ? <motion.span layoutId="es-level-thumb" className="absolute inset-0 rounded-full bg-card shadow-card" transition={{ duration: MOTION.base, ease: MOTION.ease }} aria-hidden /> : null}
                {t.band === 'U' ? (
                  <span className="relative size-2 shrink-0 rounded-full border-[1.5px] border-dashed" style={{ borderColor: 'var(--muted-foreground)' }} aria-hidden />
                ) : t.band ? (
                  <span className="relative size-2 shrink-0 rounded-full" style={{ background: t.color }} aria-hidden />
                ) : null}
                {/* "All" always reads as a word; level labels show when there's room, or when picked */}
                <span className={cx('relative whitespace-nowrap', !t.band || on ? 'inline' : 'hidden @[560px]:inline')}>{t.label}</span>
                <span className="es-num relative tabular-nums" style={{ color: on ? ink(t.color) : undefined }}><AnimatedNumber value={t.count} duration={500} /></span>
              </button>
            );
          })}
        </div>
        <Link to="/deleted" title={`Recently deleted · ${deleted} contacts (restorable for 30 days)`} aria-label={`Recently deleted, ${deleted}`}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border border-dashed px-2.5 text-[12px] font-bold text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground">
          <Icon name="trash" size={12} />
          <span className="hidden @[680px]:inline">Deleted</span>
          <span className="es-num tabular-nums">{deleted}</span>
        </Link>
      </div>
    </div>
  );
}

const isNew = (iso?: string | null) => Date.now() - time(iso) < 24 * 3600 * 1000;


/** What "asc" / "desc" mean for each kind of column, in the header menu. */
const SORT_WORDS: Partial<Record<ColumnId, [asc: string, desc: string]>> = {
  added: ['Oldest first', 'Newest first'], updated: ['Oldest first', 'Newest first'], score: ['Lowest first', 'Highest first'],
};

/**
 * Column header: clicks cycle ascending → descending → default (Added, newest first). The default
 * order is Added ↓, so on that column the cycle is ascending → default. The arrow shows the
 * direction; hiding columns lives in View options.
 */
const header = (id: ColumnId) => ({ column }: { column: Column<Row, unknown> }) => {
  const label = COLUMN_LABELS[id];
  const cls = 'whitespace-nowrap text-[10.5px] font-bold uppercase tracking-[0.06em]';
  if (!column.getCanSort()) return <span className={cx(cls, 'text-muted-foreground')}>{label}</span>;
  const sorted = column.getIsSorted();
  const [asc, desc] = SORT_WORDS[id] ?? ['A → Z', 'Z → A'];
  const isDefault = id === 'added' && sorted === 'desc';
  const next = !sorted || isDefault ? 'asc' : sorted === 'asc' && id !== 'added' ? 'desc' : 'default';
  return (
    <button onClick={() => (next === 'default' ? column.clearSorting() : column.toggleSorting(next === 'desc'))}
      title={`Sort: ${next === 'default' ? 'default (newest added first)' : next === 'asc' ? asc : desc}`}
      className={cx('group/sort -ml-1.5 inline-flex h-6 cursor-pointer items-center gap-1 rounded-full px-1.5 transition-colors', cls,
        sorted ? 'bg-brand-50 text-brand-700' : 'text-muted-foreground hover:bg-card hover:text-foreground')}>
      {label}
      {/* always shown: ↕ when the column isn't the sort, ↑ / ↓ when it is */}
      <Icon name={sorted ? (sorted === 'asc' ? 'arrowUp' : 'arrowDown') : 'sort'} size={11} strokeWidth={2.5}
        className={cx('shrink-0 transition-opacity', sorted ? 'opacity-100' : 'opacity-45 group-hover/sort:opacity-90')} />
    </button>
  );
};

export function ContactsGridComponent() {
  const qc = useQueryClient();
  const navigate = useNavigate({ from: '/contacts' });
  const { toast, confirm } = useUi();
  const search = useSearch({ from: '/_auth/contacts' });
  const drawer = useMatch({ from: '/_auth/contacts/$id', shouldThrow: false });
  const openId = drawer?.params.id ?? null;
  const isMobile = useIsMobile();

  // every filter / sort lives in the URL
  const setSearch = useCallback((patch: Partial<ContactsSearch>, replace = false) =>
    navigate({ search: (prev) => clean({ ...prev, ...patch }) as ContactsSearch, replace }), [navigate]);

  // search box: local while typing, written to the URL after a pause
  const [text, setText] = useState(search.q ?? '');
  useEffect(() => { setText(search.q ?? ''); }, [search.q]);
  useEffect(() => {
    const t = setTimeout(() => { if ((text.trim() || undefined) !== search.q) setSearch({ q: text.trim() || undefined }, true); }, 300);
    return () => clearTimeout(t);
  }, [text]);

  // keepPreviousData: a level / filter / sort change keeps the current rows on screen (dimmed) until the
  // new ones arrive, instead of suspending the page (which looked like a reload)
  const pageQ = useQuery({ ...q.contacts(contactQuery(search)), placeholderData: keepPreviousData });
  // level counts are counted on the server (GET /api/contacts/stats), not from a second full load
  const statsQ = useQuery({ ...q.contactStats(contactCountsQuery(search)), placeholderData: keepPreviousData });
  const page = pageQ.data ?? NO_CONTACTS;
  const updating = pageQ.isPlaceholderData || (pageQ.isFetching && !pageQ.data);
  const { data: users } = useSuspenseQuery(q.users());
  const { data: events } = useSuspenseQuery(q.events());
  const { data: dash } = useSuspenseQuery(q.dashboard(7));

  const owners = useMemo(() => Object.fromEntries(users.map((u) => [u.id, u.name])), [users]);
  // the rep's own profile picture (from the app), when they've set one
  const ownerPhotos = useMemo(() => Object.fromEntries(users.map((u) => [u.id, u.photoUrl || null])), [users]);
  // owner name lives in the row data: TanStack caches accessor values per row
  // "N/A" placeholders from card scans read as empty everywhere in the grid
  const rows: Row[] = useMemo(() => page.contacts.map((c) => ({
    ...c, company: blank(c.company), title: blank(c.title), email: blank(c.email), phone: blank(c.phone), linkedin: blank(c.linkedin),
    ownerName: owners[c.userId || ''] || '', ownerPhoto: ownerPhotos[c.userId || ''] || null,
  })), [page, owners, ownerPhotos]);
  const eventNames = useMemo(() => {
    const s = new Set<string>([...events.map((e) => e.name), ...dash.byEvent.map((e) => e.event), ...rows.map((r) => r.eventId)]);
    if (search.event) s.add(search.event);
    return [...s].filter(Boolean).sort();
  }, [events, dash, rows, search.event]);
  const counts = useMemo(() => {
    const s = statsQ.data;
    if (s) return { all: s.total, A: s.levels.hot, B: s.levels.warm, C: s.levels.cool, D: s.levels.cold, U: s.levels.unscored };
    // no counts from the server (yet): the loaded rows, when they're every level
    return statsQ.error && !search.band ? levelCounts(page.contacts) : levelCounts([]);
  }, [statsQ.data, statsQ.error, search.band, page]);

  // table preferences (columns, widths, density) — remembered per browser
  const [prefs, setPrefs] = useState<TablePrefs>(() => loadPrefs());
  useEffect(() => { savePrefs(prefs); }, [prefs]);
  const density = prefs.density;

  // selection (TanStack rowSelection) + keyboard focus + page
  const [selected, setSelected] = useState<Selection>({});
  const [anchor, setAnchor] = useState<string | null>(null);
  const [focus, setFocus] = useState(-1);
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: prefs.pageSize });
  useEffect(() => { setSelected({}); setFocus(-1); setAnchor(null); }, [search.band, search.event, search.owner, search.q, search.since]);
  useEffect(() => { setPagination((p) => ({ ...p, pageIndex: 0 })); setFocus(-1); }, [search.band, search.event, search.owner, search.q, search.since, search.sort, search.dir]);
  const open = (id: string, tab?: ContactsSearch['tab']) => navigate({ to: '/contacts/$id', params: { id }, search: (p) => clean({ ...p, tab }) as ContactsSearch });
  const close = () => navigate({ to: '/contacts', search: (p) => clean({ ...p, tab: undefined }) as ContactsSearch });


  /* ---------------------------------------------------------------- mutations */
  const scoreAll = useMutation({
    mutationFn: api.scoreAll,
    onSuccess: (r) => { toast(r.errors ? 'info' : 'success', r.scoredCount ? `Admin-scored ${r.scoredCount} contact${r.scoredCount === 1 ? '' : 's'}` : 'Admin scores are up to date', [r.errors ? `${r.errors} failed` : '', r.pending ? `${r.pending} still to check — run again` : '', "Reps' scores are unchanged"].filter(Boolean).join(' · ')); },
    onError: (e) => toast('error', "Couldn't score", errorMessage(e)),
    onSettled: () => { void applyChange(qc, 'batch_score_completed'); },
  });
  const scoreOne = useMutation({
    mutationFn: (id: string) => api.scoreContact(id),
    onSuccess: (r, id) => {
      const c = rows.find((x) => x.id === id);
      toast('success', `${c ? contactName(c) : 'Contact'} · admin score ${r.unchanged ? `stays at ${r.score}` : `${r.score} · ${levelOf(r.score)?.label}`}`,
        "Saved for the admin console only. The rep's score is unchanged.");
    },
    onError: (e) => toast('error', "Couldn't score", errorMessage(e)),
    onSettled: () => { void applyChange(qc, 'admin_score_updated'); },
  });
  // optimistic: rows disappear at once, come back if the server says no
  const batchDelete = useMutation({
    mutationFn: (ids: string[]) => api.batchDelete(ids),
    onMutate: async (ids) => { const rollback = await removeContacts(qc, ids); setSelected({}); if (openId && ids.includes(openId)) close(); return { rollback }; },
    onError: (e, _ids, ctx) => { ctx?.rollback(); toast('error', "Couldn't delete — restored", errorMessage(e)); },
    onSuccess: (r) => toast('success', r.message),
    onSettled: () => { void applyChange(qc, 'contact_deleted'); },
  });

  // score the selected rows one by one (each is an AI call); stops early if the AI is down
  const [scoring, setScoring] = useState<{ done: number; total: number; failed: number } | null>(null);
  const stopScoring = useRef(false);
  const scoreSelected = async (ids: string[]) => {
    stopScoring.current = false;
    setScoring({ done: 0, total: ids.length, failed: 0 });
    let failed = 0; let firstError = '';
    for (const id of ids) {
      if (stopScoring.current) break;
      try { await api.scoreContact(id); }
      catch (e) { failed += 1; firstError ||= errorMessage(e); if (e instanceof ApiError && e.status === 503) stopScoring.current = true; }
      setScoring((s) => (s ? { ...s, done: s.done + 1, failed } : s));
    }
    setScoring(null);
    const done = stopScoring.current ? 'Stopped' : 'Done';
    toast(failed ? 'info' : 'success', `${done}: ${ids.length - failed} scored${failed ? `, ${failed} failed` : ''}`, firstError || undefined);
    void applyChange(qc, 'admin_score_updated');
  };

  const [lead, setLead] = useState(EMPTY_LEAD);
  const [leadPhotoBusy, setLeadPhotoBusy] = useState(false);
  const create = useMutation({
    mutationFn: async () => {
      const { linkedin, photo, ...rest } = lead;
      const c = await api.createContact({ ...rest, photo: photo || undefined, source: 'manual', userId: lead.userId || getSession()?.userId });
      if (linkedin.trim()) {
        try { return await api.updateContact(c.id, { linkedin: linkedin.trim() }); }
        catch (e) { toast('info', `${contactName(c)} added without the LinkedIn link`, errorMessage(e)); }
      }
      return c;
    },
    onSuccess: (c) => { toast('success', `${contactName(c)} added`); setSearch({ add: undefined }); setLead(EMPTY_LEAD); open(c.id); },
    onError: (e) => toast('error', "Couldn't add the contact", errorMessage(e)),
    onSettled: () => { void applyChange(qc, 'contact_scanned'); },
  });
  const pickLeadPhoto = async (f: File | undefined) => {
    if (!f) return;
    setLeadPhotoBusy(true);
    try {
      if (!f.type.startsWith('image/')) throw new Error('Pick an image file.');
      if (f.size > 8 * 1024 * 1024) throw new Error('That photo is over 8 MB.');
      setLead((l) => ({ ...l, photo: `data:${f.type};base64,` }));
      const b64 = await fileToBase64(f);
      setLead((l) => ({ ...l, photo: `data:${f.type};base64,${b64}` }));
    } catch (e) { setLead((l) => ({ ...l, photo: '' })); toast('error', "Couldn't use that photo", errorMessage(e)); }
    finally { setLeadPhotoBusy(false); }
  };
  const [listFor, setListFor] = useState<string[] | null>(null);

  const exportAll = async () => {
    try { const { blob, filename } = await api.exportAll(); saveBlob(blob, filename); } catch (e) { toast('error', "Couldn't export", errorMessage(e)); }
  };
  const exportRows = (which: Row[]) => {
    saveBlob(new Blob([contactsToCsv(which, owners)], { type: 'text/csv;charset=utf-8' }), `eventscout-contacts-${which.length}.csv`);
    toast('success', `Exported ${which.length} contact${which.length === 1 ? '' : 's'}`);
  };
  const deleteIds = async (ids: string[]) => {
    const one = ids.length === 1 ? rows.find((r) => r.id === ids[0]) : null;
    const ok = await confirm({ title: one ? `Delete ${contactName(one)}?` : `Delete ${ids.length} contacts?`, message: "Each goes to its owner's Recently deleted for 30 days, so it can be restored.", confirmLabel: 'Delete', danger: true });
    if (ok) batchDelete.mutate(ids);
  };
  const copy = async (what: string, value: string) => {
    try { await navigator.clipboard.writeText(value); toast('success', `${what} copied`); } catch { toast('error', `Couldn't copy the ${what.toLowerCase()}`); }
  };

  /* ---------------------------------------------------------------- table */
  const order = useMemo(() => rows.map((r) => r.id), [rows]);
  const selectedIds = selectedIn(selected, order);
  // shift-click ranges follow what's on screen (sorted order), so keep it in a ref the cells can read
  const sortedOrder = useRef<string[]>(order);
  const anchorRef = useRef(anchor);
  anchorRef.current = anchor;
  const pick = useCallback((id: string, shift: boolean) => {
    setSelected((s) => (shift ? selectRange(s, sortedOrder.current, anchorRef.current, id) : toggle(s, id)));
    setAnchor(id);
  }, []);
  const rowMenu = (c: Row): MenuItem[] => [
    { label: 'Open', icon: 'eye', onSelect: () => open(c.id), hint: '↵' },
    { label: 'Edit details', icon: 'pencil', onSelect: () => open(c.id, 'edit'), hint: 'e' },
    { label: c.score == null ? 'Score with AI' : 'Re-score with AI', icon: 'flame', onSelect: () => scoreOne.mutate(c.id), hint: 's' },
    { label: 'Add to list', icon: 'listPlus', onSelect: () => setListFor([c.id]) },
    { label: 'Copy email', icon: 'mail', onSelect: () => copy('Email', c.email || ''), disabled: !c.email, separator: true },
    { label: 'Copy phone', icon: 'phone', onSelect: () => copy('Phone', c.phone || ''), disabled: !c.phone },
    { label: 'Open LinkedIn', icon: 'external', onSelect: () => window.open(/^https?:/.test(c.linkedin || '') ? c.linkedin! : `https://${c.linkedin}`, '_blank', 'noopener'), disabled: !c.linkedin },
    { label: 'Export this row (CSV)', icon: 'download', onSelect: () => exportRows([c]) },
    { label: 'Delete', icon: 'trash', onSelect: () => deleteIds([c.id]), danger: true, separator: true },
  ];

  const vis = prefs.visibility;
  const columns = useMemo<ColumnDef<Row>[]>(() => [
    {
      id: 'select', size: 40, minSize: 40, maxSize: 40, enableSorting: false, enableResizing: false, enableHiding: false,
      header: ({ table: t }) => (
        <ShadCheckbox aria-label="Select all on this page" className="cursor-pointer translate-y-[2px]"
          checked={t.getIsAllPageRowsSelected() || (t.getIsSomePageRowsSelected() && 'indeterminate')}
          onCheckedChange={(v) => t.toggleAllPageRowsSelected(!!v)} />
      ),
      cell: ({ row }) => (
        <ShadCheckbox aria-label={`Select ${contactName(row.original)}`} className="cursor-pointer translate-y-[2px]"
          checked={row.getIsSelected()} onCheckedChange={(v) => { row.toggleSelected(!!v); setAnchor(row.id); }}
          // shift-click selects the range from the last row clicked
          onClick={(e) => { e.stopPropagation(); if (e.shiftKey) { e.preventDefault(); pick(row.id, true); } }} />
      ),
    },
    {
      id: 'name', header: header('name'), size: 250, minSize: 190, enableHiding: false, accessorFn: (c) => contactName(c).toLowerCase(),
      cell: ({ row }) => {
        const c = row.original;
        const lvl = contactLevel(c);
        const ring = density === 'compact' ? 22 : 26;
        // one line, full length: photo · name · level tag · New ······ score
        return (
          <div className="flex items-center gap-2 whitespace-nowrap">
            <ContactAvatar c={c} size="xs" />
            <span className="text-[12.5px] font-bold text-foreground">{contactName(c)}</span>
            {/* level tag like the app's card (white on the heat colour); cold leads go untagged */}
            {lvl && lvl.key !== 'cold' ? (
              <span className="rounded px-1 text-[9px] font-black uppercase leading-[15px] tracking-[0.07em] text-white" style={{ background: deep(heatVar(lvl.key)) }}>{lvl.label}</span>
            ) : null}
            {isNew(c.createdAt) ? (
              <span className="rounded bg-signal px-1 text-[9px] font-black uppercase leading-[15px] tracking-[0.07em] text-white" title="Added in the last 24 hours">New</span>
            ) : null}
            <span className="ml-auto pl-2">
              {c.score != null ? (
                <span className="flex" title={`Admin score ${c.score} · ${lvl?.label}`}><ScoreRing score={c.score} level={c.level} size={ring} /></span>
              ) : (
                <button onClick={(e) => { e.stopPropagation(); scoreOne.mutate(c.id); }} title="Score with AI" aria-label={`Score ${contactName(c)} with AI`}
                  disabled={scoreOne.isPending && scoreOne.variables === c.id} style={{ borderColor: 'var(--dashed)', width: ring, height: ring }}
                  className="flex cursor-pointer items-center justify-center rounded-full border-[1.5px] border-dashed text-brand-600 transition-colors hover:bg-brand-50">
                  <Icon name="sparkles" size={11} className={scoreOne.isPending && scoreOne.variables === c.id ? 'animate-spin' : ''} />
                </button>
              )}
            </span>
          </div>
        );
      },
    },
    // every cell below is ONE line at its full length (the table scrolls sideways instead of wrapping or cutting)
    { id: 'company', header: header('company'), size: 180, accessorFn: (c) => (c.company || '').toLowerCase(),
      cell: ({ row }) => row.original.company ? <span className="whitespace-nowrap font-semibold text-foreground">{row.original.company}</span> : EMPTY },
    { id: 'title', header: header('title'), size: 180, accessorFn: (c) => (c.title || '').toLowerCase(),
      cell: ({ row }) => row.original.title ? <span className="whitespace-nowrap text-body">{row.original.title}</span> : EMPTY },
    { id: 'phone', header: header('phone'), size: 160, enableSorting: false,
      cell: ({ row }) => <MultiValue kind="phone" raw={row.original.phone} onCopy={copy} /> },
    { id: 'email', header: header('email'), size: 220, accessorFn: (c) => (c.email || '').toLowerCase(),
      cell: ({ row }) => <MultiValue kind="email" raw={row.original.email} onCopy={copy} /> },
    { id: 'linkedin', header: header('linkedin'), size: 80, enableSorting: false,
      cell: ({ row }) => row.original.linkedin ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <a href={linkedinHref(row.original.linkedin)} target="_blank" rel="noopener" onClick={(e) => e.stopPropagation()} aria-label={`LinkedIn · ${linkedinLabel(row.original.linkedin)}`}
              className="inline-flex rounded-[5px] transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <LinkedInLogo size={18} />
            </a>
          </TooltipTrigger>
          <TooltipContent side="top">{linkedinLabel(row.original.linkedin)}</TooltipContent>
        </Tooltip>
      ) : EMPTY },
    { id: 'owner', header: header('owner'), size: 170, accessorFn: (c) => c.ownerName.toLowerCase(),
      cell: ({ row }) => row.original.ownerName ? (
        <button className="-ml-1 inline-flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full py-0.5 pl-0.5 pr-2 transition-colors hover:bg-secondary" title={`Show only contacts added by ${row.original.ownerName}`}
          onClick={(e) => { e.stopPropagation(); setSearch({ owner: row.original.userId || undefined }); }}>
          <Avatar src={row.original.ownerPhoto} name={row.original.ownerName} size="xs" className="size-5 shrink-0 border-0 text-[8px] shadow-none" />
          <span className="text-body">{row.original.ownerName}</span>
        </button>
      ) : EMPTY },
    { id: 'event', header: header('event'), size: 160, accessorFn: (c) => (c.eventId || '').toLowerCase(),
      cell: ({ row }) => row.original.eventId ? (
        <button title={`Show only contacts met at ${row.original.eventId}`} onClick={(e) => { e.stopPropagation(); setSearch({ event: row.original.eventId }); }}
          className="inline-flex cursor-pointer items-center gap-1 whitespace-nowrap rounded-full border border-brand-100 bg-brand-50 px-2 text-[11px] font-bold leading-[18px] text-brand-700 transition-colors hover:border-brand-300 hover:bg-brand-100">
          <Icon name="calendar" size={11} className="text-brand-600" />{row.original.eventId}
        </button>
      ) : EMPTY },
    { id: 'score', header: header('score'), size: 150, accessorFn: (c) => c.score ?? undefined, sortUndefined: 'last',
      cell: ({ row }) => {
        const s = row.original.score;
        const lvl = contactLevel(row.original);
        return (
          <div className="flex items-center gap-2 whitespace-nowrap">
            <ScoreRing score={s} level={row.original.level} size={density === 'compact' ? 22 : 26} />
            {lvl ? (
              <span className="text-[10px] font-bold uppercase tracking-[0.06em]" style={{ color: heatVar(lvl.key) }}>{lvl.label}</span>
            ) : (
              <button onClick={(e) => { e.stopPropagation(); scoreOne.mutate(row.original.id); }}
                // like the Score buttons elsewhere: a double-click no longer scores (and toasts) twice
                disabled={scoreOne.isPending && scoreOne.variables === row.original.id}
                className="inline-flex h-5 cursor-pointer items-center gap-1 rounded-full px-1.5 text-[11px] font-semibold text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground">
                <Icon name="sparkles" size={11} />Score
              </button>
            )}
          </div>
        );
      } },
    { id: 'source', header: header('source'), size: 140, enableSorting: false,
      cell: ({ row }) => (
        <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-body">
          <Icon name={SOURCE_ICON[row.original.source || ''] || 'bolt'} size={12} className="text-muted-foreground" />
          {SOURCE_LABEL[row.original.source || ''] || row.original.source || EMPTY}
        </span>
      ) },
    { id: 'added', header: header('added'), size: 120, accessorFn: (c) => time(c.createdAt),
      cell: ({ row }) => (
        <span className={cx('inline-flex items-center gap-1.5 whitespace-nowrap', isNew(row.original.createdAt) ? 'font-bold text-foreground' : 'text-muted-foreground')} title={fmtDateTime(row.original.createdAt)}>
          {isNew(row.original.createdAt) ? <span className="size-1.5 rounded-full bg-brand-600" aria-hidden /> : null}{relTime(row.original.createdAt)}
        </span>
      ) },
    { id: 'updated', header: header('updated'), size: 120, accessorFn: (c) => time(c.updatedAt || c.createdAt),
      // never edited (updated within a minute of being added): a dash, as the contact drawer leaves the row out
      cell: ({ row }) => !wasEdited(row.original.createdAt, row.original.updatedAt)
        ? <span className="whitespace-nowrap text-muted-foreground" title="Not edited since it was added">—</span>
        : <span className="whitespace-nowrap text-muted-foreground" title={fmtDateTime(row.original.updatedAt)}>{relTime(row.original.updatedAt)}</span> },
    {
      id: 'actions', size: 52, minSize: 52, maxSize: 52, enableSorting: false, enableResizing: false, enableHiding: false,
      header: () => <span className="sr-only">Actions</span>,
      cell: ({ row }) => (
        <span onClick={(e) => e.stopPropagation()} className="flex justify-end">
          <Menu label={`Actions for ${contactName(row.original)}`} items={rowMenu(row.original)}
            trigger={<IconButton icon="more" label="More actions" className="size-6 text-muted-foreground group-hover/row:text-foreground data-[state=open]:text-foreground" />} />
        </span>
      ),
    },
  ], [density, rows, pick, vis]);

  const sorting = useMemo<SortingState>(() => sortingFor(search), [search.sort, search.dir]);
  const table = useReactTable({
    data: rows, columns, getRowId: (r) => r.id,
    // column order: Name · Company · Email · Event · Representative · Status, then the optional ones, Actions last
    state: { sorting, rowSelection: selected, pagination, columnVisibility: prefs.visibility, columnSizing: prefs.sizing, columnPinning: { left: ['select', 'name'] },
      columnOrder: ['select', 'name', ...COLUMN_IDS.filter((id) => id !== 'name'), 'actions'] },
    onSortingChange: (u) => setSearch(searchForSorting(typeof u === 'function' ? u(sorting) : u)),
    onRowSelectionChange: setSelected,
    onPaginationChange: (u) => {
      const next = typeof u === 'function' ? u(pagination) : u;
      setPagination(next);
      if (next.pageSize !== pagination.pageSize) setPrefs((p) => ({ ...p, pageSize: next.pageSize }));
    },
    onColumnVisibilityChange: (u) => setPrefs((p) => ({ ...p, visibility: typeof u === 'function' ? u(p.visibility) : u })),
    onColumnSizingChange: (u) => setPrefs((p) => ({ ...p, sizing: typeof u === 'function' ? u(p.sizing) : u })),
    // columns size to their content (one line, full length), so there's nothing to drag-resize
    enableRowSelection: true, enableMultiSort: false, enableSortingRemoval: false, enableColumnResizing: false,
    // the page only resets on a filter / sort change (above), not when a row is deleted or re-scored
    autoResetPageIndex: false,
    defaultColumn: { minSize: 70, size: 160, maxSize: 640 },
    getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel(), getPaginationRowModel: getPaginationRowModel(),
  });
  const tableRows = table.getRowModel().rows;
  const visibleCols = table.getVisibleLeafColumns();
  const sortedRows = table.getSortedRowModel().rows;
  sortedOrder.current = sortedRows.map((r) => r.id);
  const pageCount = table.getPageCount();
  // deleting the last rows of the last page: step back to a page that exists
  useEffect(() => { if (pagination.pageIndex > 0 && pagination.pageIndex >= pageCount) setPagination((p) => ({ ...p, pageIndex: Math.max(0, pageCount - 1) })); }, [pageCount]);

  const scroller = useRef<HTMLDivElement>(null);
  const focusRow = (i: number) => {
    setFocus(i);
    scroller.current?.querySelector<HTMLElement>(`[data-index="${i}"]`)?.scrollIntoView({ block: 'nearest' });
  };

  // keyboard: j/k move · x select · enter open · e edit · s score · [ ] page · / search · esc clear
  const searchBox = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!shouldHandleKey(e)) return;
      const ids = tableRows.map((r) => r.id);
      const cur = focus >= 0 ? ids[focus] : undefined;
      if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); focusRow(moveFocus(focus, 1, ids.length)); }
      else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); focusRow(moveFocus(focus, -1, ids.length)); }
      else if (e.key === 'x' && cur) { e.preventDefault(); pick(cur, e.shiftKey); }
      else if (e.key === 'Enter' && cur) { e.preventDefault(); open(cur); }
      else if (e.key === 'e' && cur) { e.preventDefault(); open(cur, 'edit'); }
      else if (e.key === 's' && cur) { e.preventDefault(); scoreOne.mutate(cur); }
      else if (e.key === '[' && table.getCanPreviousPage()) { e.preventDefault(); table.previousPage(); setFocus(-1); }
      else if (e.key === ']' && table.getCanNextPage()) { e.preventDefault(); table.nextPage(); setFocus(-1); }
      else if (e.key === '/') { e.preventDefault(); searchBox.current?.querySelector('input')?.focus(); }
      else if (e.key === 'Escape' && selectedIds.length) { setSelected({}); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tableRows, focus, selectedIds.length, anchor, pagination]);
  // keep the focus on the open contact — turning to its page if it's on another one (⌘K, deep link)
  useEffect(() => {
    if (!openId) return;
    const at = sortedRows.findIndex((r) => r.id === openId);
    if (at < 0) return;
    const pageIndex = Math.floor(at / pagination.pageSize);
    if (pageIndex !== pagination.pageIndex) setPagination((p) => ({ ...p, pageIndex }));
    setFocus(at % pagination.pageSize);
  }, [openId]);

  /* ---------------------------------------------------------------- render */
  const filtered = !!(search.q || search.band || search.event || search.owner || search.since);
  const clearFilters = () => { setText(''); navigate({ search: (p) => clean({ sort: p.sort, dir: p.dir }) as ContactsSearch }); };

  // filter pills: events with their all-time count (top events from the dashboard), reps with theirs
  const eventCounts = useMemo(() => Object.fromEntries(dash.byEvent.map((e) => [e.event, e.count])), [dash]);
  const eventOptions: PillOption[] = eventNames.map((n) => ({
    value: n, label: n, ...(eventCounts[n] != null ? { count: eventCounts[n] } : {}),
    icon: <span className="flex size-6 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600"><Icon name="calendar" size={12} /></span>,
  }));
  const repOptions: PillOption[] = users.map((u) => ({
    value: u.id, label: u.name, ...(u.contactsCount != null ? { count: u.contactsCount } : {}),
    icon: <Avatar src={u.photoUrl || null} name={u.name} size="xs" className="size-6 shrink-0 border-0 text-[9px] shadow-none" />,
  }));
  // tight rows (~40px comfortable, ~32px compact) so a 15-row page fits without scrolling
  const cellPad = density === 'compact' ? 'px-2.5 py-1' : 'px-2.5 py-1.5';
  /**
   * Checkbox and row menu are fixed; every other column has a minimum width so short or empty
   * values still line up in tidy columns, and grows past it for longer values (one line, never cut).
   */
  const fixedWidth = (id: string): React.CSSProperties =>
    id === 'select' ? { width: 40, minWidth: 40, maxWidth: 40 } : id === 'actions' ? { width: 44, minWidth: 44 } : { minWidth: COLUMN_MIN[id as ColumnId] ?? 110 };

  const renderRow = (row: TRow<Row>, i: number) => {
    const c = row.original;
    const lvl = contactLevel(c);
    // hot and warm leads carry the app card's accent bar down their left edge
    const accent = lvl && (lvl.key === 'hot' || lvl.key === 'warm') ? `inset 3px 0 0 ${heatVar(lvl.key)}` : undefined;
    const cls = cx('group/row cursor-pointer bg-card transition-colors hover:bg-brand-50/45',
      selected[row.id] && 'bg-brand-50/80 hover:bg-brand-50',
      openId === row.id && 'bg-accent hover:bg-accent',
      // keyboard focus: a full inset outline (not a side stripe)
      focus === i && 'shadow-[inset_0_0_0_1.5px_var(--ring)]');
    const cells = row.getVisibleCells().map((cell) => (
      <TableCell key={cell.id} style={{ ...pinStyle(cell.column), ...fixedWidth(cell.column.id), ...(cell.column.id === 'select' && accent ? { boxShadow: accent } : {}) }}
        // row lines live on the cells (the table is border-separate), so they move with the sticky
        // checkbox + Name cells instead of breaking up under them while the table scrolls sideways
        className={cx('whitespace-nowrap border-b border-border/60 align-middle text-[12.5px] leading-[18px] first:pl-3.5 last:pr-2 [tr:last-child>&]:border-b-0', cellPad, cell.column.getIsPinned() && 'bg-inherit',
          // the pinned Name column keeps a hairline on its right edge while the rest scrolls under it
          cell.column.id === 'name' && 'es-pin-edge shadow-[inset_-1px_0_0_var(--border)]')}>
        {flexRender(cell.column.columnDef.cell, cell.getContext())}
      </TableCell>
    ));
    const common = {
      'data-index': i, 'data-state': selected[row.id] ? 'selected' : undefined, 'data-focused': focus === i ? 'true' : undefined, className: cls, onMouseEnter: () => setFocus(i),
      onClick: (e: React.MouseEvent) => (e.shiftKey ? pick(row.id, true) : open(row.id)),
      title: c.title || c.company ? `${contactName(c)} · ${[c.title, c.company].filter(Boolean).join(' at ')}` : contactName(c),
    };
    return (
      <motion.tr key={row.id} {...common} layout="position" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
        transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 14) * 0.02 }}>
        {cells}
      </motion.tr>
    );
  };

  // top-3 rank badges, as in the mobile feed, when the list is ranked by score (best first)
  const rankedByScore = search.sort === 'score' && sortDir(search) === 'desc';
  /** The mobile feed's lead card, for the card view. */
  const renderCard = (row: TRow<Row>, i: number) => {
    const c = row.original;
    const lvl = contactLevel(c);
    const rank = rankedByScore && c.score != null ? pagination.pageIndex * pagination.pageSize + i + 1 : null;
    const sub = [c.title, c.company].filter(Boolean).join(' · ');
    const isSel = !!selected[row.id];
    const selecting = isSel || selectedIds.length > 0; // once anything is selected, every card shows its checkbox
    return (
      <motion.div key={row.id} layout="position" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.98 }}
        transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 12) * 0.025 }}
        data-index={i} data-state={isSel ? 'selected' : undefined} data-focused={focus === i ? 'true' : undefined}
        onMouseEnter={() => setFocus(i)} onClick={(e) => (e.shiftKey ? pick(row.id, true) : open(row.id))}
        title={sub ? `${contactName(c)} · ${sub}` : contactName(c)}
        whileHover={{ y: -2 }} whileTap={{ scale: 0.985 }}
        className={cx('group/row relative flex h-full cursor-pointer flex-col gap-2 overflow-hidden rounded-2xl border bg-card p-2.5 pt-3 shadow-card transition-[border-color,box-shadow] duration-200 hover:border-brand-200 hover:shadow-active dark:hover:border-brand-300',
          rank === 1 && 'border-brand-300',
          isSel && 'border-primary bg-accent/40',
          openId === row.id && 'border-primary ring-2 ring-primary/40',
          focus === i && openId !== row.id && !isSel && 'ring-2 ring-ring/40')}
        style={{ ['--heat' as string]: lvl ? heatVar(lvl.key) : 'var(--heat-none)' }}>
        {/* a thin strip of the lead's heat colour along the top */}
        <span className="absolute inset-x-0 top-0 h-[3px]" style={{ background: 'var(--heat)' }} aria-hidden />

        {/* who · how hot */}
        <div className="flex items-center gap-2.5">
          {/* photo or initials in its heat ring; turns into a checkbox on hover or while selecting */}
          <span className="relative shrink-0" onClick={(e) => e.stopPropagation()}>
            <span className={cx('block rounded-full p-[2px] transition-opacity', selecting ? 'opacity-0' : 'group-hover/row:opacity-0')}
              style={{ background: lvl ? 'var(--heat)' : 'transparent', boxShadow: lvl ? undefined : 'inset 0 0 0 1.5px var(--dashed)' }}>
              <span className="block rounded-full bg-card p-[1.5px]"><ContactAvatar c={c} size="md" /></span>
            </span>
            <span className={cx('absolute inset-0 flex items-center justify-center rounded-full bg-accent transition-opacity', selecting ? 'opacity-100' : 'opacity-0 group-hover/row:opacity-100')}>
              <ShadCheckbox aria-label={`Select ${contactName(c)}`} className="size-5 cursor-pointer bg-card"
                checked={isSel} onCheckedChange={(v) => { row.toggleSelected(!!v); setAnchor(row.id); }}
                onClick={(e) => { if (e.shiftKey) { e.preventDefault(); pick(row.id, true); } }} />
            </span>
            {rank && rank <= 3 ? (
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 22, delay: 0.15 + i * 0.03 }}
                className={cx('absolute -left-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full font-mono text-[10px] font-bold text-white ring-2 ring-card',
                  rank === 1 ? 'bg-amber-500' : rank === 2 ? 'bg-slate-400' : 'bg-orange-400')} title={`#${rank} by score`}>{rank}</motion.span>
            ) : null}
          </span>

          <div className="min-w-0 flex-1">
            <div className="break-words text-[15px] font-semibold leading-5 text-foreground [overflow-wrap:anywhere]">{contactName(c)}</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-1">
              {lvl ? (
                <motion.span initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 520, damping: 26, delay: 0.1 + Math.min(i, 12) * 0.03 }}
                  className={cx('inline-flex items-center gap-1 rounded-full px-1.5 py-px text-[10px] font-bold uppercase tracking-[0.05em]', lvl.badge)}>
                  <Icon name="flame" size={10} />{lvl.label}
                </motion.span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-secondary px-1.5 py-px text-[10px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">Not scored</span>
              )}
              {isNew(c.createdAt) ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-brand-600 px-1.5 py-px text-[10px] font-bold uppercase tracking-[0.05em] text-white" title="Added in the last 24 hours">
                  <span className="relative flex h-1.5 w-1.5"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-70" /><span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-white" /></span>New
                </span>
              ) : null}
            </div>
          </div>

          {/* score ring (draws itself), or the app's "Score" tile */}
          {c.score != null ? (
            <span className="shrink-0" title={`Admin score ${c.score} · ${lvl?.label}`}><ScoreRing score={c.score} level={c.level} size={42} /></span>
          ) : (
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.9 }}
              onClick={(e) => { e.stopPropagation(); scoreOne.mutate(c.id); }} disabled={scoreOne.isPending && scoreOne.variables === c.id}
              className="flex h-[42px] w-[46px] shrink-0 cursor-pointer flex-col items-center justify-center gap-0.5 rounded-xl border border-brand-100 bg-brand-50 text-brand-700 transition-colors hover:bg-brand-100 disabled:opacity-60"
              aria-label={`Score ${contactName(c)} with AI`} title="Score with AI">
              <Icon name="sparkles" size={14} className={scoreOne.isPending && scoreOne.variables === c.id ? 'animate-spin' : ''} />
              <span className="text-[9.5px] font-bold">Score</span>
            </motion.button>
          )}
        </div>

        {/* what they do — coloured icon tiles, like the app's stat tiles */}
        <div className="flex-1 space-y-1">
          <div className="flex items-start gap-2">
            <span className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-brand-50 text-brand-700 transition-transform duration-300 group-hover/row:-rotate-6 group-hover/row:scale-110"><Icon name="briefcase" size={11} /></span>
            <span className={cx('break-words text-[13px] leading-5 [overflow-wrap:anywhere]', c.title ? 'font-medium text-body' : 'italic text-muted-foreground')}>{c.title || 'No designation yet'}</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-sky-50 text-sky-700 transition-transform delay-75 duration-300 group-hover/row:rotate-6 group-hover/row:scale-110 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="building" size={11} /></span>
            <span className={cx('break-words text-[13px] leading-5 [overflow-wrap:anywhere]', c.company ? 'text-muted-foreground' : 'italic text-muted-foreground')}>{c.company || 'No company yet'}</span>
          </div>
          {c.eventId ? (
            <div className="flex items-start gap-2" onClick={(e) => e.stopPropagation()}>
              <span className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-amber-50 text-amber-700 transition-transform delay-100 duration-300 group-hover/row:-rotate-6 group-hover/row:scale-110 dark:bg-amber-500/15 dark:text-amber-300"><Icon name="calendar" size={11} /></span>
              <button title={`Show only contacts met at ${c.eventId}`} onClick={() => setSearch({ event: c.eventId })}
                className="cursor-pointer break-words text-left text-[13px] leading-5 text-muted-foreground underline-offset-2 hover:text-foreground hover:underline">{c.eventId}</button>
            </div>
          ) : null}
        </div>

        {/* who added them · when ······ reach them */}
        <div className="mt-auto flex items-center gap-2 border-t pt-2.5" onClick={(e) => e.stopPropagation()}>
          {c.ownerName ? (
            <button title={`Show only contacts added by ${c.ownerName}`} onClick={() => setSearch({ owner: c.userId || undefined })}
              className="group/rep flex min-w-0 cursor-pointer items-center gap-2 rounded-lg py-0.5 pr-1.5 text-left transition-colors hover:bg-secondary">
              {/* the representative's initials (they have no photo here), in their own colour */}
              <Avatar src={c.ownerPhoto} name={c.ownerName} size="xs" className="size-6 shrink-0 border-0 text-[10px] font-bold shadow-none ring-2 ring-card transition-transform group-hover/rep:scale-110" />
              <span className="min-w-0 leading-tight">
                <span className="block truncate text-[12px] font-semibold text-foreground">{c.ownerName}</span>
                <span className="block text-[10.5px] text-muted-foreground" title={`Added ${fmtDateTime(c.createdAt)}`}>Added {relTime(c.createdAt)}</span>
              </span>
            </button>
          ) : (
            <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground"><Icon name="clock" size={12} />Added {relTime(c.createdAt)}</span>
          )}
          <span className="ml-auto flex shrink-0 items-center gap-0.5">
            <ContactReach c={c} ownerName={c.ownerName} onCopy={copy} onOpenProfile={() => open(c.id)} size="sm" hideEmptyLinkedIn />
            <Menu label={`Actions for ${contactName(c)}`} items={rowMenu(c)}
              trigger={<IconButton icon="more" label="More actions" className="h-7 w-7 text-muted-foreground hover:text-foreground" />} />
          </span>
        </div>
      </motion.div>
    );
  };
  /** Phones: the app's own ContactCard — accent bar, name + level tag, title · company, event · when, score ring. */
  const renderListCard = (row: TRow<Row>, i: number) => {
    const c = row.original;
    const lvl = contactLevel(c);
    const loud = !!lvl && (lvl.key === 'hot' || lvl.key === 'warm');
    const sub = [c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No company yet';
    const isSel = !!selected[row.id];
    const selecting = selectedIds.length > 0;
    return (
      <motion.div key={row.id} layout="position" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.98 }}
        transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 12) * 0.03 }} whileTap={{ scale: 0.98 }}
        data-index={i} data-state={isSel ? 'selected' : undefined}
        onClick={() => (selecting ? row.toggleSelected(!isSel) : open(row.id))}
        className={cx('relative flex cursor-pointer items-center gap-3 overflow-hidden rounded-[18px] border bg-card py-3 pl-3.5 pr-3 shadow-card',
          loud && 'border-[1.5px]', isSel && 'border-primary bg-accent/50', openId === row.id && 'ring-2 ring-primary/40')}
        style={loud && !isSel ? { borderColor: `color-mix(in oklab, ${heatVar(lvl!.key)} 50%, transparent)` } : {}}>
        {lvl ? <span className="absolute inset-y-0 left-0 w-1" style={{ background: heatVar(lvl.key) }} aria-hidden /> : null}
        {/* tap the photo to select (then every tap selects) */}
        <button className="relative shrink-0 cursor-pointer rounded-full" aria-label={isSel ? `Unselect ${contactName(c)}` : `Select ${contactName(c)}`} aria-pressed={isSel}
          onClick={(e) => { e.stopPropagation(); row.toggleSelected(!isSel); setAnchor(row.id); }}>
          <ContactAvatar c={c} size="md" />
          <AnimatePresence>
            {isSel ? (
              <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: 'spring', stiffness: 520, damping: 26 }}
                className="absolute inset-0 flex items-center justify-center rounded-full bg-brand-600 text-white"><Icon name="check" size={18} strokeWidth={3} /></motion.span>
            ) : null}
          </AnimatePresence>
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className="truncate text-[15.5px] font-extrabold leading-5 text-foreground">{contactName(c)}</span>
            {lvl && lvl.key !== 'cold' ? <span className="shrink-0 rounded-md px-1.5 text-[9.5px] font-black uppercase leading-4 tracking-[0.07em] text-white" style={{ background: heatVar(lvl.key) }}>{lvl.label}</span> : null}
            {isNew(c.createdAt) ? <span className="shrink-0 rounded-md bg-brand-600 px-1.5 text-[9.5px] font-black uppercase leading-4 tracking-[0.07em] text-white">New</span> : null}
          </div>
          <div className="truncate text-[13px] text-body">{sub}</div>
          <div className="mt-0.5 flex min-w-0 items-center gap-1 text-[11.5px] text-muted-foreground">
            {c.eventId ? <><Icon name="calendar" size={11} className="text-brand-600" /><span className="truncate font-bold text-brand-700">{c.eventId}</span><span>·</span></> : null}
            <span className="shrink-0">{relTime(c.createdAt)}</span>
            {c.ownerName ? <><span>·</span><span className="truncate">{c.ownerName.split(/\s+/)[0]}</span></> : null}
          </div>
        </div>
        {c.score != null ? (
          <ScoreRing score={c.score} level={c.level} size={46} />
        ) : (
          <button onClick={(e) => { e.stopPropagation(); scoreOne.mutate(c.id); }} disabled={scoreOne.isPending && scoreOne.variables === c.id} aria-label={`Score ${contactName(c)} with AI`}
            className="flex h-[46px] w-[46px] shrink-0 cursor-pointer flex-col items-center justify-center gap-0.5 rounded-xl border border-brand-100 bg-brand-50 text-brand-700 disabled:opacity-60">
            <Icon name="sparkles" size={14} className={scoreOne.isPending && scoreOne.variables === c.id ? 'animate-spin' : ''} />
            <span className="text-[9.5px] font-bold">Score</span>
          </button>
        )}
      </motion.div>
    );
  };

  // phones always get the app's card list; wider screens use the saved Table / Cards choice
  const view: TablePrefs['view'] | 'list' = isMobile ? 'list' : prefs.view;
  const sortValue = `${search.sort}:${sortDir(search)}`;
  const setSort = (value: string) => {
    const [sort, dir] = value.split(':') as [ContactSort, 'asc' | 'desc'];
    setSearch({ sort, dir: dir === (sort === 'added' || sort === 'updated' || sort === 'score' ? 'desc' : 'asc') ? undefined : dir });
  };
  // a column-header sort that isn't one of the presets, e.g. "Email ↑"
  const customSortLabel = `${COLUMN_LABELS[search.sort as ColumnId] ?? search.sort} ${sortDir(search) === 'asc' ? '↑' : '↓'}`;
  // every column a user can show / hide, in display order, for the Columns menu
  const columnToggles = table.getAllLeafColumns()
    .filter((col) => (COLUMN_IDS as readonly string[]).includes(col.id))
    .sort((a, b) => COLUMN_IDS.indexOf(a.id as ColumnId) - COLUMN_IDS.indexOf(b.id as ColumnId))
    .map((col) => ({ id: col.id, label: COLUMN_LABELS[col.id as ColumnId], visible: col.getIsVisible(), locked: !col.getCanHide(), toggle: (v: boolean) => col.toggleVisibility(v) }));
  const resetView = () => { setPrefs({ ...DEFAULT_PREFS, visibility: { ...DEFAULT_PREFS.visibility } }); setPagination({ pageIndex: 0, pageSize: DEFAULT_PREFS.pageSize }); setSearch({ sort: 'added', dir: undefined }); };
  // a failed load must not read as "No contacts yet"
  const empty = pageQ.error && !pageQ.data ? (
    <div className="p-4"><ErrorState error={pageQ.error} onRetry={() => pageQ.refetch()} /></div>
  ) : pageQ.isPending ? (
    // still loading (nothing cached for these filters yet): not "no contacts"
    <div className="space-y-2 p-4" aria-busy="true">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-11 w-full rounded-xl" />)}</div>
  ) : (
    <EmptyState icon="contacts" title={filtered ? 'No contacts match these filters' : 'No contacts yet'}
      description={filtered ? 'Try widening the search, another event or rep, or every level.' : 'Contacts your reps scan in the app show up here the moment they sync.'}
      action={filtered ? <Button icon="eraser" onClick={clearFilters}>Clear filters</Button> : <Button variant="primary" icon="userPlus" onClick={() => setSearch({ add: '1' })}>Add a contact</Button>} />
  );

  return (
    <>
      {/* the page fills the screen: header, toolbar, then the list, which scrolls inside while pagination stays put */}
      <div className="flex h-[calc(100dvh-72px)] min-h-[520px] flex-col gap-2.5">
      {/* header, always ONE row: title · level bar (takes the room that's left, scrolls if it runs out) ······
          Export · Score with AI · Add contact (labels drop to icons as the screen narrows) */}
      <div className="relative isolate flex shrink-0 flex-nowrap items-center gap-2 overflow-hidden rounded-2xl border px-2.5 py-1.5 shadow-card sm:gap-3"
        style={{ background: 'linear-gradient(135deg, color-mix(in oklab, var(--brand-500) 11%, var(--card)), var(--card) 55%)' }}>
        {/* the header's bubbles, like every page hero (kept thin so the table keeps its rows) */}
        <span aria-hidden className="es-bubble pointer-events-none absolute -right-10 -top-16 -z-10 size-40 rounded-full" style={{ background: 'color-mix(in oklab, var(--brand-500) 12%, transparent)' }} />
        <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-20 left-1/3 -z-10 size-32 rounded-full" style={{ background: 'color-mix(in oklab, #0ea5e9 8%, transparent)' }} />
        <h1 className="group flex shrink-0 items-center gap-2.5 font-display text-[20px] font-black leading-8 tracking-[-0.03em] text-foreground sm:text-[24px]">
          <span className="es-pop-in hidden size-9 items-center justify-center rounded-xl bg-brand-50 text-brand-700 shadow-card sm:flex"><LiveIcon name="contacts" size={18} /></span>
          <span className="es-in" style={{ ['--es-d' as string]: '60ms' }}>Contacts</span>
        </h1>
        <LevelTiles counts={counts} band={search.band as Band | undefined} onBand={(b) => setSearch({ band: b }, true)} deleted={dash.recentlyDeleted ?? 0} />
        <div className="flex shrink-0 items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <span><Button size="sm" icon="download" onClick={exportAll} aria-label="Export every contact as CSV" className="h-8 rounded-full px-2.5"><span className="hidden 2xl:inline">Export</span></Button></span>
            </TooltipTrigger>
            <TooltipContent side="bottom">Export every contact (CSV)</TooltipContent>
          </Tooltip>
          <Button size="sm" icon="sparkles" loading={scoreAll.isPending} onClick={() => scoreAll.mutate()} aria-label="Score with AI"
            title={`Ranks contacts against your Sales profile: those without an admin score first, then ones whose admin score is out of date (up to 25 AI calls per click). Saved for the admin console only — reps’ scores never change`} className="h-8 rounded-full px-2.5">
            <span className="hidden xl:inline">Score with AI</span>{counts.U ? <span className="es-num rounded-full bg-brand-50 px-1.5 text-[11px] font-black text-brand-700">{Math.min(counts.U, 25)}</span> : null}
          </Button>
          <Button size="sm" variant="primary" icon="userPlus" onClick={() => setSearch({ add: '1' })} aria-label="Add contact" className="h-8 rounded-full px-2.5 shadow-active lg:px-3.5">
            <span className="hidden lg:inline">Add contact</span>
          </Button>
        </div>
      </div>

      {/* toolbar, one compact row: search · Event · Representative · Sort · (clear) ······ status · Table/Cards · Columns.
          It wraps instead of overflowing, so with the sidebar (or the contact panel) open the view controls
          drop to the next line rather than being pushed off the edge. Phones: the filter pills swipe. */}
      <div className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1.5">
        <SearchField containerRef={searchBox} value={text} onChange={setText} placeholder="Search name, company, email…" className="w-full sm:w-auto sm:min-w-[220px] sm:max-w-[360px] sm:flex-1" />
        <div className="no-scrollbar flex min-w-0 items-center gap-1.5 max-sm:-mx-3 max-sm:w-[calc(100%+1.5rem)] max-sm:overflow-x-auto max-sm:px-3 sm:flex-wrap">
          {view !== 'table' ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <label className="inline-flex h-8 shrink-0 cursor-pointer items-center gap-2 rounded-full border-[1.5px] bg-card px-2.5 text-[12.5px] font-bold text-foreground">
                  <ShadCheckbox aria-label="Select every contact on this page" className="cursor-pointer"
                    checked={table.getIsAllPageRowsSelected() || (table.getIsSomePageRowsSelected() && 'indeterminate')}
                    onCheckedChange={(v) => table.toggleAllPageRowsSelected(!!v)} />
                  <span className="hidden sm:inline">Page</span>
                </label>
              </TooltipTrigger>
              <TooltipContent side="bottom">Select every contact on this page</TooltipContent>
            </Tooltip>
          ) : null}
          <FilterPill icon="calendar" tint="var(--heat-cool)" title="Event" noun="event" options={eventOptions}
            value={search.event} onChange={(v) => setSearch({ event: v })} />
          <FilterPill icon="users" tint="var(--brand-600)" title="Representative" noun="rep" options={repOptions}
            value={search.owner} onChange={(v) => setSearch({ owner: v })} />
          {search.since === 'today' ? (
            <FilterChip icon="clock" title="Added" label="Today" hint="Contacts added since midnight (your time)" onClear={() => setSearch({ since: undefined })} />
          ) : null}
          <SortMenu value={sortValue} onChange={setSort} customLabel={customSortLabel} />
          <AnimatePresence>
            {filtered ? (
              <motion.button key="clear" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ duration: MOTION.fast }}
                onClick={clearFilters} className="inline-flex h-8 shrink-0 cursor-pointer items-center gap-1 rounded-full px-2.5 text-xs font-bold text-brand-700 transition-colors hover:bg-brand-50">
                <Icon name="eraser" size={13} />Clear all
              </motion.button>
            ) : null}
          </AnimatePresence>
        </div>
        {/* view controls: one group pinned right; it wraps as a unit, never off-screen */}
        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          {updating ? (
            <span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-semibold text-muted-foreground" role="status"><Spinner size={13} />Updating…</span>
          ) : search.q ? (
            <span className="shrink-0 text-xs font-bold text-muted-foreground" role="status">{rows.length.toLocaleString()} match{rows.length === 1 ? '' : 'es'}</span>
          ) : null}
          {view !== 'list' ? (
            <>
              <LayoutToggle view={prefs.view} onView={(v) => setPrefs((p) => ({ ...p, view: v }))} />
              <ColumnsMenu columns={columnToggles} density={density} onDensity={(d) => setPrefs((p) => ({ ...p, density: d }))} onReset={resetView} disabled={view === 'cards'} />
            </>
          ) : null}
        </div>
      </div>

      <Card className={cx('flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl', view === 'list' && 'border-0 bg-transparent shadow-none')}>
        {view === 'list' ? (
          <motion.div ref={scroller} layoutScroll aria-busy={updating} className={cx('min-h-0 flex-1 overflow-auto transition-opacity duration-200', updating && 'opacity-60')}>
            {tableRows.length === 0 ? empty : (
              <div className="flex flex-col gap-2.5 pb-2">
                <AnimatePresence initial={true}>{tableRows.map(renderListCard)}</AnimatePresence>
              </div>
            )}
          </motion.div>
        ) : view === 'cards' ? (
          // columns follow the space the list has (container queries), not the window, so the sidebar
          // or the contact panel opening never squeezes two cards into too little room
          <motion.div ref={scroller} layoutScroll aria-busy={updating} className={cx('@container min-h-0 flex-1 overflow-auto bg-secondary/40 transition-opacity duration-200', updating && 'opacity-60')}>
            {tableRows.length === 0 ? empty : (
              <div className="grid auto-rows-fr gap-2 p-2 @xl:grid-cols-2 @4xl:grid-cols-3">
                <AnimatePresence initial={true}>{tableRows.map(renderCard)}</AnimatePresence>
              </div>
            )}
          </motion.div>
        ) : (
        // layoutScroll: the rows' layout animation measures them relative to this scroller, so a hover
        // after scrolling sideways (or down) doesn't read the scroll offset as a move and jolt the rows
        // data-scrolled-x: once the columns slide under Name, its edge casts a soft shadow (styles/tokens.css)
        <motion.div ref={scroller} layoutScroll aria-busy={updating} className={cx('es-grid-scroll min-h-0 flex-1 overflow-auto transition-opacity duration-200', updating && 'opacity-60')}
          onScroll={(e) => { const el = e.currentTarget; const x = el.scrollLeft > 0 ? '1' : ''; if (el.dataset.scrolledX !== x) el.dataset.scrolledX = x; }}>
          {/* auto layout: every column is as wide as its longest value, so nothing wraps or gets cut.
              border-separate: a sticky header / column keeps its own lines (collapsed borders stay behind on scroll) */}
          <table data-slot="table" className="es-table es-grid caption-bottom border-separate border-spacing-0 text-left text-[12.5px]" style={{ minWidth: '100%', tableLayout: 'auto' }}>
            <TableHeader>
              {table.getHeaderGroups().map((hg) => (
                <TableRow key={hg.id} className="border-0 hover:bg-transparent">
                  {hg.headers.map((h) => (
                    <TableHead key={h.id} style={{ ...pinStyle(h.column), ...fixedWidth(h.column.id), zIndex: h.column.getIsPinned() ? 4 : 3 }}
                      aria-sort={h.column.getIsSorted() ? (h.column.getIsSorted() === 'asc' ? 'ascending' : 'descending') : undefined}
                      className={cx('sticky top-0 h-8 whitespace-nowrap border-b bg-secondary px-2.5 first:pl-3.5 last:pr-2', h.column.id === 'name' && 'es-pin-edge shadow-[inset_-1px_0_0_var(--border)]')}>
                      {h.isPlaceholder ? null : flexRender(h.column.columnDef.header, h.getContext())}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {tableRows.length === 0 ? (
                <TableRow className="hover:bg-transparent"><TableCell colSpan={visibleCols.length} className="whitespace-normal">{empty}</TableCell></TableRow>
              ) : (
                <AnimatePresence initial={true}>{tableRows.map(renderRow)}</AnimatePresence>
              )}
            </TableBody>
          </table>
        </motion.div>
        )}
        {/* pagination: always on screen, under the scrolling list */}
        <div className={cx('shrink-0 py-1 pl-1 pr-2', view === 'list' ? 'rounded-2xl border bg-card shadow-card' : 'border-t bg-card')}>
          <DataTablePagination table={table} pageSizes={PAGE_SIZES}
            note={rows.length >= MAX_CONTACT_ROWS ? `Showing the first ${MAX_CONTACT_ROWS.toLocaleString()} — narrow with filters.` : undefined} />
        </div>
      </Card>
      </div>

      {/* bulk actions */}
      <AnimatePresence>
        {selectedIds.length ? (
          <motion.div key="bulk" initial={{ opacity: 0, y: 24, x: '-50%' }} animate={{ opacity: 1, y: 0, x: '-50%' }} exit={{ opacity: 0, y: 16, x: '-50%' }} transition={{ duration: MOTION.base, ease: MOTION.ease }}
            className="fixed bottom-6 left-1/2 z-30 flex max-w-[calc(100vw-2rem)] flex-wrap items-center gap-2 rounded-xl border bg-popover p-2 pl-2.5 text-popover-foreground shadow-2xl lg:ml-30">
            <span className="flex items-center gap-2 pr-1 text-sm font-medium">
              <span className="es-num rounded-md bg-signal px-2 py-0.5 text-[13px] font-semibold text-signal-foreground">{selectedIds.length}</span>
              selected <span className="hidden text-xs text-muted-foreground sm:inline">· shift-click for ranges</span>
            </span>
            <Separator orientation="vertical" className="h-5" />
            {scoring ? (
              <span className="flex items-center gap-2 px-2 text-xs">
                <span className="h-1.5 w-28 overflow-hidden rounded-full bg-primary/20"><motion.span className="block h-full bg-primary" animate={{ width: `${(scoring.done / scoring.total) * 100}%` }} transition={{ duration: MOTION.fast }} /></span>
                scoring {scoring.done}/{scoring.total}{scoring.failed ? ` · ${scoring.failed} failed` : ''}
                <button className="underline cursor-pointer" onClick={() => { stopScoring.current = true; }}>Stop</button>
              </span>
            ) : <>
              <Button size="sm" variant="subtle" icon="flame" onClick={() => scoreSelected(selectedIds)}>Score with AI</Button>
              <Button size="sm" variant="subtle" icon="listPlus" onClick={() => setListFor(selectedIds)}>Add to list</Button>
              <Button size="sm" variant="subtle" icon="download" onClick={() => exportRows(rows.filter((r) => selected[r.id]))}>Export CSV</Button>
              <Button size="sm" variant="danger" icon="trash" onClick={() => deleteIds(selectedIds)}>Delete</Button>
            </>}
            <IconButton icon="x" label="Clear selection" onClick={() => setSelected({})} />
          </motion.div>
        ) : null}
      </AnimatePresence>

      {/* /contacts/$id renders the drawer here, so the grid never remounts */}
      <Outlet />
      <AddToListModal contactIds={listFor} onClose={() => setListFor(null)} onDone={() => setSelected({})} />

      <Modal open={search.add === '1'} onOpenChange={(o) => setSearch({ add: o ? '1' : undefined })} title="Add a contact" description="It starts unscored — score it from the contact's drawer." size="lg"
        footer={<>
          <Button variant="ghost" onClick={() => setSearch({ add: undefined })}>Cancel</Button>
          <Button variant="primary" icon="userPlus" loading={create.isPending} disabled={!lead.firstName.trim() || leadPhotoBusy} onClick={() => create.mutate()}>Add contact</Button>
        </>}>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex items-center gap-4 sm:col-span-2">
            <Avatar src={lead.photo && !lead.photo.endsWith('base64,') ? lead.photo : null} name={[lead.firstName, lead.lastName].filter(Boolean).join(' ') || 'New contact'} size="xl" rounded="xl" />
            <div>
              <div className="text-sm font-medium text-foreground">Photo <span className="font-normal text-muted-foreground">optional</span></div>
              <div className="mt-1.5 flex gap-2">
                <label className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg bg-card px-2.5 text-xs font-medium text-body ring-1 ring-inset ring-border hover:bg-secondary">
                  <Icon name="imagePlus" size={14} />{lead.photo ? 'Change' : 'Add photo'}
                  <input type="file" accept="image/*" className="hidden" onChange={(e) => pickLeadPhoto(e.target.files?.[0])} />
                </label>
                {lead.photo ? <Button size="sm" variant="ghost" icon="x" onClick={() => setLead({ ...lead, photo: '' })}>Remove</Button> : null}
              </div>
            </div>
          </div>
          <Field label="First name *"><Input autoFocus value={lead.firstName} onChange={(e) => setLead({ ...lead, firstName: e.target.value })} /></Field>
          <Field label="Last name"><Input value={lead.lastName} onChange={(e) => setLead({ ...lead, lastName: e.target.value })} /></Field>
          <Field label="Company"><Input value={lead.company} onChange={(e) => setLead({ ...lead, company: e.target.value })} /></Field>
          <Field label="Job title"><Input value={lead.title} onChange={(e) => setLead({ ...lead, title: e.target.value })} /></Field>
          <Field label="Email"><Input type="email" value={lead.email} onChange={(e) => setLead({ ...lead, email: e.target.value })} /></Field>
          <Field label="Phone"><Input value={lead.phone} onChange={(e) => setLead({ ...lead, phone: e.target.value })} /></Field>
          <Field label="LinkedIn" hint="linkedin.com/in/…" className="sm:col-span-2"><Input value={lead.linkedin} onChange={(e) => setLead({ ...lead, linkedin: e.target.value })} placeholder="https://www.linkedin.com/in/…" /></Field>
          <Field label="Event" hint="Exactly as reps pick it in the app">
            <Input list="es-events" value={lead.eventId} onChange={(e) => setLead({ ...lead, eventId: e.target.value })} />
            <datalist id="es-events">{eventNames.map((n) => <option key={n} value={n} />)}</datalist>
          </Field>
          <Field label="Rep">
            <Select value={lead.userId} onChange={(e) => setLead({ ...lead, userId: e.target.value })}>
              <option value="">Me ({getSession()?.name || 'admin'})</option>
              {users.filter((u) => u.id !== getSession()?.userId).map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
            </Select>
          </Field>
          <Field label="Notes" className="sm:col-span-2"><Textarea rows={3} value={lead.notes} onChange={(e) => setLead({ ...lead, notes: e.target.value })} /></Field>
        </div>
      </Modal>
    </>
  );
}
