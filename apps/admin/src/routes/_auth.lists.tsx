// /lists — named groups of contacts for follow-up, all on the real API:
//   lists     GET/POST /api/admin/contact_lists · PATCH/DELETE /api/admin/contact_lists/{id}
//   members   GET  /api/admin/contact_lists/{id}/contacts (with addedToListAt)
//             POST /api/admin/contact_lists/{id}/contacts {contactIds} · DELETE …/contacts/{contactId}
//   export    GET  /api/admin/export?listId= (server CSV, same columns as Exports)
//   picker    GET  /api/contacts?search=&limit=200
//   smart     GET  /api/admin/segments · /api/admin/segments/{dim}/contacts?key=  (industry · seniority · role · company)
// What's open is in the URL (?tab= ?seg= ?list=); the live stream ("list_changed") refreshes lists for every admin.
import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { patchAll, patchInArrays, removeFromArrays } from '../lib/optimistic';
import type { ContactList, ListMember, SegmentDim } from '../lib/types';
import { clean } from '../lib/search';
import { dayKey } from '../lib/dates';
import { contactsToCsv } from '../features/contacts/csv';
import { api, saveBlob } from '../lib/api';
import { SearchField } from '../features/contacts/Toolbar';
import { Avatar } from '../ui/Avatar';
import { Icon, type IconName } from '../ui/icons';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import { contactLevel, countAtLevels } from '../lib/levels';
import {
  Button, Card, EmptyState, ErrorState, IconButton, Modal, PageHeader, ScoreBadge, Skeleton, Table, Td, Th, contactName, cx, deep, errorMessage, fmtDateTime, ink, relTime, useUi,
} from '../ui/kit';

const spring = { type: 'spring', stiffness: 420, damping: 28 } as const;
const PALETTE = ['#8b5cf6', '#0ea5e9', '#10b981', '#f59e0b', '#f43f5e', '#d946ef', '#14b8a6', '#6366f1'];
/** A stable colour per list (from its name), for its tile. */
const listColor = (name: string) => PALETTE[[...name].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7) % PALETTE.length]!;
const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join('').toUpperCase() || '#';

/* ------------------------------------------------------------------ add contacts */

function AddContactsModal({ listId, listName, existing, open, onClose }: { listId: string; listName: string; existing: Set<string>; open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const [text, setText] = useState('');
  const [query, setQuery] = useState('');
  const [picked, setPicked] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (open) { setText(''); setQuery(''); setPicked({}); } }, [open]);
  useEffect(() => { const t = setTimeout(() => setQuery(text.trim()), 250); return () => clearTimeout(t); }, [text]);
  const found = useQuery({ queryKey: ['admin', 'contacts', 'list', { pick: query }], queryFn: () => api.contacts({ search: query || undefined, limit: 200, lite: 1 }), enabled: open });
  const all = found.data?.contacts || [];
  const rows = all.filter((c) => !existing.has(c.id));
  const ids = Object.keys(picked).filter((k) => picked[k]);
  const allPicked = rows.length > 0 && rows.every((c) => picked[c.id]);

  const save = async () => {
    setBusy(true);
    try {
      const r = await api.addToList(listId, ids);
      toast('success', `Added ${r.count} to “${listName}”`, [r.alreadyInList ? `${r.alreadyInList} already in it` : '', r.notFound ? `${r.notFound} no longer exist` : ''].filter(Boolean).join(' · ') || undefined);
      void applyChange(qc, 'list_changed');
      onClose();
    } catch (e) {
      toast('error', "Couldn't add", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onOpenChange={(o) => { if (!o) onClose(); }} size="lg" title={`Add contacts to “${listName}”`} description="Search by name, company, email or notes, tick the people, add them."
      footer={<>
        <AnimatePresence mode="popLayout">
          <motion.span key={ids.length} initial={{ y: -8, opacity: 0 }} animate={{ y: 0, opacity: 1 }} className="mr-auto text-sm font-semibold text-muted-foreground">{ids.length} selected</motion.span>
        </AnimatePresence>
        <Button variant="ghost" onClick={onClose}>Cancel</Button>
        <Button variant="primary" icon="plus" disabled={!ids.length} loading={busy} onClick={save}>Add {ids.length || ''}</Button>
      </>}>
      <div className="mb-3 flex items-center gap-2">
        <SearchField value={text} onChange={setText} placeholder="Search contacts…" className="flex-1" />
        <Button size="sm" variant="ghost" disabled={!rows.length} onClick={() => setPicked(allPicked ? {} : Object.fromEntries(rows.map((c) => [c.id, true])))}>{allPicked ? 'None' : 'All shown'}</Button>
      </div>
      <div className="max-h-[50vh] overflow-y-auto rounded-2xl border bg-card p-1.5">
        {found.isLoading ? <div className="space-y-2 p-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full rounded-xl" />)}</div>
          : found.error ? <div className="p-3"><ErrorState error={found.error} onRetry={() => found.refetch()} /></div>
            : rows.length === 0 ? <EmptyState icon="contacts" title={query ? 'No matches' : all.length ? 'Everyone shown is already in this list' : 'No contacts yet'} description={query ? 'Try another name, company or email.' : undefined} />
              : rows.map((c, i) => {
                const on = !!picked[c.id];
                return (
                  <motion.button key={c.id} type="button" onClick={() => setPicked((p) => ({ ...p, [c.id]: !p[c.id] }))} aria-pressed={on}
                    initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.fast, delay: Math.min(i, 14) * 0.02 }} whileTap={{ scale: 0.99 }}
                    className={cx('flex w-full cursor-pointer items-center gap-3 rounded-xl px-2.5 py-2 text-left transition-colors', on ? 'bg-brand-50' : 'hover:bg-secondary/60')}>
                    <span className={cx('flex size-5 shrink-0 items-center justify-center rounded-md border-[1.5px] transition-colors', on ? 'border-brand-600 bg-brand-600 text-white' : 'border-border')}>
                      <AnimatePresence>{on ? <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={spring} className="flex"><Icon name="check" size={12} strokeWidth={3} /></motion.span> : null}</AnimatePresence>
                    </span>
                    <Avatar src={c.photoUrl || null} name={contactName(c)} size="sm" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-semibold text-foreground">{contactName(c)}</div>
                      <div className="truncate text-xs text-muted-foreground">{[c.title, c.company].filter(Boolean).join(' · ') || c.email || 'No details yet'}</div>
                    </div>
                    <ScoreBadge score={c.score} level={c.level} />
                  </motion.button>
                );
              })}
      </div>
      {all.length >= 200 ? <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground"><Icon name="info" size={12} />Showing the first 200 matches — search to narrow it down.</p> : null}
    </Modal>
  );
}


/* ------------------------------------------------------------------ tiles */

type Tab = 'mine' | SegmentDim;
type Sort = 'name' | 'score' | 'added';

const TABS: { key: Tab; label: string; icon: IconName; tint: string }[] = [
  { key: 'mine', label: 'My lists', icon: 'list', tint: '#d946ef' },
  { key: 'industry', label: 'Industry', icon: 'briefcase', tint: '#0ea5e9' },
  { key: 'seniority', label: 'Seniority', icon: 'flag', tint: '#f59e0b' },
  { key: 'function', label: 'Role', icon: 'idCard', tint: '#10b981' },
  { key: 'company', label: 'Company', icon: 'building', tint: '#6366f1' },
];

/** An icon for each group, from its key / label (industry names, seniority and role keys are fixed). */
function tileIcon(dim: SegmentDim, key: string, label: string): IconName {
  const l = label.toLowerCase();
  if (dim === 'company') return 'building';
  if (dim === 'seniority') return ({ clevel: 'key', owner: 'flag', vp: 'flame', director: 'briefcase', manager: 'users', ic: 'user', student: 'idCard' } as Record<string, IconName>)[key] ?? 'info';
  if (dim === 'function') return ({ sales: 'send', marketing: 'megaphone', data: 'sparkles', product: 'columns', engineering: 'server', operations: 'pipeline', finance: 'database', people: 'users', success: 'check', leadership: 'flag' } as Record<string, IconName>)[key] ?? (key === 'unknown' ? 'info' : 'more');
  if (/health|medic|pharma/.test(l)) return 'activity';
  if (/financ|bank/.test(l)) return 'database';
  if (/software|saas/.test(l)) return 'server';
  if (/data|ai\b/.test(l)) return 'sparkles';
  if (/it services|tele/.test(l)) return 'smartphone';
  if (/market|media/.test(l)) return 'megaphone';
  if (/retail|commerce/.test(l)) return 'store';
  if (/educat/.test(l)) return 'idCard';
  if (/energy/.test(l)) return 'bolt';
  if (/logist|transport|travel/.test(l)) return 'send';
  if (/real estate|construct/.test(l)) return 'building';
  if (/government|non-profit/.test(l)) return 'shield';
  if (/manufactur|automotive/.test(l)) return 'sliders';
  if (/professional|consult/.test(l)) return 'briefcase';
  return 'tag';
}

/** One tile: icon, name, count (counting up), hot & warm, average score bar, share of every contact. */
function Tile({ i, on, onClick, icon, color, label, count, hot, avgScore, share, sub }: {
  i: number; on: boolean; onClick: () => void; icon: IconName; color: string; label: string; count: number; hot?: number; avgScore?: number | null; share?: number; sub?: string | undefined;
}) {
  return (
    <motion.button type="button" onClick={onClick} aria-pressed={on} layout
      initial={{ opacity: 0, y: 14, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: Math.min(i, 16) * 0.035 }}
      whileHover={{ y: -3 }} whileTap={{ scale: 0.97 }}
      className={cx('group relative flex min-h-[96px] cursor-pointer flex-col overflow-hidden rounded-2xl border bg-card p-2.5 text-left shadow-card transition-[border-color,box-shadow] duration-200 hover:shadow-active',
        on ? 'border-transparent' : 'hover:border-brand-200')}>
      {on ? <motion.span layoutId="es-tile-on" className="absolute inset-0 rounded-2xl border-2" style={{ borderColor: color, background: `color-mix(in oklab, ${color} 7%, transparent)` }} transition={spring} /> : null}
      <span aria-hidden className="es-bubble pointer-events-none absolute -right-8 -top-8 size-24 rounded-full opacity-70" style={{ background: `color-mix(in oklab, ${color} 12%, transparent)` }} />
      <div className="relative flex items-start justify-between gap-2">
        <span className="es-wiggle-hover flex size-8 items-center justify-center rounded-lg shadow-card" style={{ background: on ? color : `color-mix(in oklab, ${color} 15%, var(--card))`, color: on ? 'white' : color }}>
          <Icon name={icon} size={15} />
        </span>
        {hot ? <span className="inline-flex items-center gap-0.5 rounded-full bg-rose-50 px-1.5 text-[10.5px] font-bold leading-5 text-rose-600 dark:bg-rose-500/15" title="Scoring 70 or more"><Icon name="flame" size={10} />{hot}</span> : null}
      </div>
      <div className="relative mt-1.5 line-clamp-2 text-[13px] font-bold leading-[17px] text-foreground">{label}</div>
      <div className="relative mt-auto flex items-baseline gap-1.5 pt-1">
        <span className="es-num text-[18px] font-black leading-6" style={{ color: ink(color) }}><AnimatedNumber value={count} duration={800} /></span>
        <span className="text-[11px] font-semibold text-muted-foreground">{sub ?? `contact${count === 1 ? '' : 's'}`}</span>
        {avgScore != null ? <span className="ml-auto text-[11px] font-bold text-muted-foreground" title="Average score">avg {Math.round(avgScore)}</span> : null}
      </div>
      {share != null ? (
        <div className="relative mt-1.5 h-1 overflow-hidden rounded-full bg-secondary" title={`${share}% of every contact`}>
          <motion.div className="h-full rounded-full" style={{ background: color }} initial={{ width: 0 }} animate={{ width: `${Math.max(share, 2)}%` }} transition={{ duration: 0.9, ease: MOTION.ease, delay: 0.15 + Math.min(i, 16) * 0.035 }} />
        </div>
      ) : null}
    </motion.button>
  );
}

/* ------------------------------------------------------------------ the people in a list / smart list */

function MembersPanel({ members, loading, error, onRetry, owners, emptyText, onRemove, emptyAction }: {
  members: ListMember[] | undefined; loading: boolean; error: unknown; onRetry: () => void; owners: Record<string, string>;
  emptyText: string; onRemove?: (id: string, name: string) => void; emptyAction?: React.ReactNode;
}) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<Sort>('name');
  const shown = useMemo(() => {
    const t = query.toLowerCase();
    const list = (members || []).filter((c) => !t || [contactName(c), c.company, c.title, c.email, c.eventId, owners[c.userId || '']].join(' ').toLowerCase().includes(t));
    return [...list].sort((a, b) => sort === 'score' ? (b.score ?? -1) - (a.score ?? -1)
      : sort === 'added' ? ((b.addedToListAt || b.createdAt || '').localeCompare(a.addedToListAt || a.createdAt || ''))
        : contactName(a).localeCompare(contactName(b)));
  }, [members, query, sort, owners]);
  return (
    <Card className="overflow-hidden rounded-3xl">
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <SearchField value={query} onChange={setQuery} placeholder="Search these people…" className="w-full sm:w-[280px]" />
        <div role="tablist" aria-label="Sort" className="inline-flex rounded-full bg-secondary p-1">
          {([['name', 'Name', 'user'], ['score', 'Score', 'flame'], ['added', 'Newest', 'clock']] as const).map(([v, label, icon]) => (
            <button key={v} role="tab" aria-selected={sort === v} onClick={() => setSort(v)}
              className={cx('relative inline-flex h-7 cursor-pointer items-center gap-1.5 rounded-full px-3 text-[12.5px] font-bold transition-colors', sort === v ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
              {sort === v ? <motion.span layoutId="es-list-sort" className="absolute inset-0 rounded-full bg-card shadow-card" transition={spring} /> : null}
              <Icon name={icon} size={12} className="relative" /><span className="relative">{label}</span>
            </button>
          ))}
        </div>
        <span className="ml-auto text-xs font-semibold text-muted-foreground">{query ? `${shown.length} of ${members?.length ?? 0}` : ''}</span>
      </div>
      {error ? <div className="p-4"><ErrorState error={error} onRetry={onRetry} /></div>
        : loading ? <div className="space-y-2 p-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full rounded-xl" />)}</div>
          : !members?.length ? <EmptyState icon="contacts" title={emptyText} action={emptyAction} />
            : !shown.length ? <EmptyState icon="search" title="No one matches" description={`Nothing matches “${query}”.`} />
              : (
                <Table>
                  <thead><tr><Th>Contact</Th><Th>Company</Th><Th>Event</Th><Th>Owner</Th><Th>Score</Th><Th>{onRemove ? 'Added' : 'Met'}</Th>{onRemove ? <Th className="w-12" /> : null}</tr></thead>
                  <tbody>
                    {shown.map((c) => {
                      const lvl = contactLevel(c);
                      const when = onRemove ? c.addedToListAt : c.createdAt;
                      return (
                        <tr key={c.id} className="group cursor-pointer" onClick={() => navigate({ to: '/contacts/$id', params: { id: c.id } })} title={`Open ${contactName(c)}`}>
                          <Td>
                            <div className="flex items-center gap-3">
                              <span className="relative transition-transform duration-200 group-hover:scale-110">
                                <Avatar src={c.photoUrl || null} name={contactName(c)} size="sm" />
                                {lvl ? <span className="absolute -bottom-0.5 -right-0.5 size-2.5 rounded-full ring-2 ring-card" style={{ background: `var(--heat-${lvl.key})` }} /> : null}
                              </span>
                              <div className="min-w-0">
                                <div className="truncate font-semibold text-foreground group-hover:text-brand-700">{contactName(c)}</div>
                                <div className="truncate text-xs text-muted-foreground">{c.title || c.email || '—'}</div>
                              </div>
                            </div>
                          </Td>
                          <Td className="text-body">{c.company || '—'}</Td>
                          <Td>{c.eventId ? <span className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 text-[11px] font-bold leading-5 text-brand-700"><Icon name="calendar" size={10} />{c.eventId}</span> : <span className="text-muted-foreground">—</span>}</Td>
                          <Td className="text-body">{owners[c.userId || ''] || '—'}</Td>
                          <Td><ScoreBadge score={c.score} level={c.level} /></Td>
                          <Td className="whitespace-nowrap text-xs text-muted-foreground" title={when ? fmtDateTime(when) : undefined}>{when ? relTime(when) : '—'}</Td>
                          {onRemove ? (
                            <Td onClick={(e) => e.stopPropagation()}>
                              <IconButton icon="x" label={`Remove ${contactName(c)} from the list`} onClick={() => onRemove(c.id, contactName(c))}
                                className="opacity-50 transition-opacity hover:text-rose-600 group-hover:opacity-100" />
                            </Td>
                          ) : null}
                        </tr>
                      );
                    })}
                  </tbody>
                </Table>
              )}
    </Card>
  );
}

/** The header of whatever is open: big tile, name, stats, actions. */
function OpenHeader({ color, icon, initialsText, title, sub, stats, actions }: {
  color: string; icon?: IconName; initialsText?: string; title: string; sub: React.ReactNode; stats: { icon: IconName; label: string; n: number | null; tint: string }[]; actions: React.ReactNode;
}) {
  return (
    <div className="relative isolate overflow-hidden rounded-3xl border p-4 shadow-card" style={{ background: `linear-gradient(135deg, color-mix(in oklab, ${color} 14%, var(--card)), var(--card) 65%)` }}>
      <span aria-hidden className="es-bubble pointer-events-none absolute -right-12 -top-16 -z-10 size-48 rounded-full" style={{ background: `color-mix(in oklab, ${color} 14%, transparent)` }} />
      <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-20 right-48 -z-10 size-36 rounded-full" style={{ background: `color-mix(in oklab, ${color} 8%, transparent)` }} />
      <div className="flex flex-wrap items-center gap-3">
        <motion.span initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 380, damping: 16 }}
          className="flex size-11 shrink-0 items-center justify-center rounded-xl text-base font-black text-white shadow-active" style={{ background: deep(color) }}>
          {icon ? <Icon name={icon} size={20} /> : initialsText}
        </motion.span>
        <div className="min-w-0 flex-1">
          <h2 className="break-words font-display text-[19px] font-bold leading-6 tracking-[-0.02em] text-foreground">{title}</h2>
          <p className="text-[13px] text-body">{sub}</p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">{actions}</div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {stats.map((s, i) => (
          <motion.div key={s.label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring, delay: 0.08 + i * 0.05 }} whileHover={{ y: -2 }}
            className="group flex items-center gap-2 rounded-xl border bg-card/85 px-2.5 py-1.5 shadow-card backdrop-blur">
            <span className="es-wiggle-hover flex size-7 items-center justify-center rounded-lg" style={{ background: `color-mix(in oklab, ${s.tint} 14%, transparent)`, color: s.tint }}><Icon name={s.icon} size={15} /></span>
            <span>
              <span className="es-num block text-lg font-black leading-5 text-foreground">{s.n == null ? '–' : <AnimatedNumber value={s.n} duration={700} />}</span>
              <span className="block text-[11px] font-semibold text-muted-foreground">{s.label}</span>
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

const memberStats = (m: { score: number | null; level?: string | null | undefined; eventId: string; userId: string | null }[] | undefined) => m ? {
  // the tile reads "Hot & warm": hot + warm leads by the admin level
  hot: countAtLevels(m, ['hot', 'warm']),
  events: new Set(m.map((c) => c.eventId).filter(Boolean)).size,
  reps: new Set(m.map((c) => c.userId).filter(Boolean)).size,
} : null;

/* ------------------------------------------------------------------ page */

export function ContactListsComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const lists = useSuspenseQuery(q.lists());
  const users = useSuspenseQuery(q.users());
  const segments = useQuery(q.segments());
  const search = useSearch({ from: '/_auth/lists' });
  const navigate = useNavigate();
  // which tiles (?tab=), and which one is open: ?list=<id> for your lists, ?seg=<key> for a smart list
  const tab: Tab = search.tab ?? (search.seg ? 'industry' : 'mine');
  const go = (patch: { tab?: Tab; seg?: string | undefined; list?: string | undefined }) =>
    navigate({ to: '/lists', search: (s) => clean({ ...s, ...patch }) as never, replace: true });
  // one search over everything: list names, industries, seniority levels, roles and companies
  const [query, setQuery] = useState('');
  const [naming, setNaming] = useState<{ id?: string; name: string; fromSegment?: { dim: SegmentDim; key: string } } | null>(null);
  const [adding, setAdding] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const owners = useMemo(() => Object.fromEntries((users.data || []).map((u) => [u.id, u.name])), [users.data]);

  // your own lists
  const activeId = tab === 'mine' ? ((search.list && lists.data.some((l) => l.id === search.list) ? search.list : lists.data[0]?.id) ?? null) : null;
  const active = lists.data.find((l) => l.id === activeId) || null;
  const members = useQuery({ ...q.listMembers(activeId || ''), enabled: !!activeId });
  const memberIds = useMemo(() => new Set((members.data || []).map((m) => m.id)), [members.data]);

  // smart lists
  const dim = tab === 'mine' ? null : tab;
  const rows = dim && segments.data ? segments.data[dim].rows.filter((r) => !r.other) : [];
  const segKey = dim ? (search.seg && rows.some((r) => r.key === search.seg) ? search.seg : rows[0]?.key) : undefined;
  const segRow = rows.find((r) => r.key === segKey);
  const segMembers = useQuery({ ...q.segmentContacts(dim ?? 'industry', segKey ?? ''), enabled: !!dim && !!segKey });

  const total = segments.data?.total ?? 0;
  const tabCount = (t: Tab) => t === 'mine' ? lists.data.length : segments.data ? segments.data[t].rows.filter((r) => !r.other).length : null;
  const tint = TABS.find((t) => t.key === tab)!.tint;

  const saveName = async () => {
    if (!naming) return;
    setSaving(true);
    try {
      if (naming.id) {
        // optimistic rename: the new name shows at once, the old one returns on failure
        const rollback = await patchInArrays<ContactList>(qc, ['admin', 'lists'], naming.id, { name: naming.name.trim() });
        const id = naming.id;
        setNaming(null);
        try { await api.renameList(id, naming.name); toast('success', 'List renamed'); }
        catch (e) { rollback(); throw e; }
      } else {
        const l = await api.createList(naming.name);
        // "Save as list" from a smart list: put everyone in it straight away
        if (naming.fromSegment) {
          const people = await api.segmentContacts(naming.fromSegment.dim, naming.fromSegment.key);
          if (people.length) await api.addToList(l.id, people.map((c) => c.id));
          toast('success', `Saved “${l.name}” with ${people.length} contact${people.length === 1 ? '' : 's'}`, "It's a normal list now — it won't change when new contacts arrive.");
        } else {
          toast('success', `Created “${l.name}”`, 'Add contacts to it, here or from the Contacts table.');
        }
        go({ tab: 'mine', list: l.id, seg: undefined });
        setNaming(null);
      }
      void applyChange(qc, 'list_changed');
    } catch (e) {
      toast('error', "Couldn't save the list", errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const removeList = async () => {
    if (!active) return;
    if (!(await confirm({ title: `Delete “${active.name}”?`, message: 'The list is removed; the contacts in it are kept.', confirmLabel: 'Delete list', danger: true }))) return;
    const rollback = await removeFromArrays<ContactList>(qc, ['admin', 'lists'], [active.id]);
    go({ list: undefined });
    try {
      await api.deleteList(active.id);
      toast('success', 'List deleted');
    } catch (e) {
      rollback();
      go({ list: active.id });
      toast('error', "Couldn't delete — restored", errorMessage(e));
    } finally {
      void applyChange(qc, 'list_changed');
    }
  };

  const removeMember = async (contactId: string, name: string) => {
    if (!active) return;
    // optimistic: the row and the count update at once; both come back on failure
    const rollback = await patchAll(
      removeFromArrays<ListMember>(qc, q.listMembers(active.id).queryKey, [contactId]),
      patchInArrays<ContactList>(qc, ['admin', 'lists'], active.id, { memberCount: Math.max(0, active.memberCount - 1) }),
    );
    try {
      await api.removeFromList(active.id, contactId);
      toast('success', `Removed ${name}`);
    } catch (e) {
      rollback();
      toast('error', "Couldn't remove — restored", errorMessage(e));
    } finally {
      void applyChange(qc, 'list_changed');
    }
  };

  const exportList = async () => {
    if (!active) return;
    setExporting(true);
    try {
      const { blob, filename, rows: n } = await api.exportAll({ listId: active.id });
      saveBlob(blob, filename);
      toast('success', `Exported ${n ?? active.memberCount} contact${(n ?? active.memberCount) === 1 ? '' : 's'}`, filename);
    } catch (e) { toast('error', "Couldn't export", errorMessage(e)); }
    finally { setExporting(false); }
  };
  const exportSegment = () => {
    if (!segRow || !segMembers.data?.length) return;
    saveBlob(new Blob([contactsToCsv(segMembers.data, owners)], { type: 'text/csv;charset=utf-8' }), `eventscout-${segRow.label.replace(/[^\w-]+/g, '-').toLowerCase()}-${dayKey(new Date())}.csv`);
    toast('success', `Exported ${segMembers.data.length} contact${segMembers.data.length === 1 ? '' : 's'}`);
  };

  // the tiles for this tab
  const listTiles = lists.data;
  const segTiles = rows;
  const segColor = (i: number) => PALETTE[i % PALETTE.length]!;
  const needle = query.trim().toLowerCase();
  // search results across every tab (matches by name / label)
  const results = needle ? [
    { tab: 'mine' as Tab, items: lists.data.filter((l) => l.name.toLowerCase().includes(needle)).map((l) => ({ id: l.id, label: l.name, count: l.memberCount, color: listColor(l.name), icon: 'list' as IconName, hot: undefined as number | undefined, avg: null as number | null })) },
    ...(['industry', 'seniority', 'function', 'company'] as SegmentDim[]).map((d) => ({
      tab: d as Tab,
      items: (segments.data?.[d].rows ?? []).filter((r) => !r.other && r.label.toLowerCase().includes(needle))
        .map((r, i) => ({ id: r.key, label: r.label, count: r.count, color: segColor(i), icon: tileIcon(d, r.key, r.label), hot: (r.hotWarm ?? r.hot) as number | undefined, avg: r.avgScore })),
    })),
  ].filter((s) => s.items.length) : [];
  const pick = (t: Tab, id: string) => { setQuery(''); go(t === 'mine' ? { tab: t, list: id, seg: undefined } : { tab: t, seg: id, list: undefined }); };

  return (
    <>
      <PageHeader compact title="Lists" description="Every contact sorted automatically by industry, seniority, role and company — re-sorted the moment new contacts arrive — plus your own lists."
        actions={<Button variant="primary" icon="plus" onClick={() => setNaming({ name: '' })}>New list</Button>} />

      {/* which tiles */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div role="tablist" aria-label="Group contacts by" className="no-scrollbar inline-flex max-w-full overflow-x-auto rounded-2xl bg-secondary p-1">
          {TABS.map((t) => {
            const on = tab === t.key;
            const n = tabCount(t.key);
            return (
              <motion.button key={t.key} role="tab" aria-selected={on} whileTap={{ scale: 0.96 }}
                onClick={() => go({ tab: t.key, seg: undefined, list: undefined })}
                className={cx('relative inline-flex shrink-0 cursor-pointer items-center gap-2 rounded-xl px-3.5 py-2 text-sm font-bold transition-colors', on ? 'text-white' : 'text-muted-foreground hover:text-foreground')}>
                {on ? <motion.span layoutId="es-lists-tab" className="absolute inset-0 rounded-xl shadow-active" style={{ background: deep(t.tint) }} transition={spring} /> : null}
                <Icon name={t.icon} size={15} className="relative" />
                <span className="relative">{t.label}</span>
                {n != null ? <span className={cx('relative min-w-6 rounded-full px-1.5 text-center text-[11px] font-black leading-5', on ? 'bg-black/25 text-white' : 'bg-card text-body')}>{n}</span> : null}
              </motion.button>
            );
          })}
        </div>
        <SearchField value={query} onChange={setQuery} placeholder="Search lists, industries, roles, seniority, companies…" className="w-full sm:w-[340px]" />
        {dim && segments.data ? (
          <span className="ml-auto text-xs font-semibold text-muted-foreground">
            {segments.data[dim].known.toLocaleString()} of {total.toLocaleString()} contacts {dim === 'industry' ? 'have an industry' : dim === 'company' ? 'have a company' : 'have a job title'}
          </span>
        ) : null}
      </div>

      {/* industry still being worked out */}
      {dim === 'industry' && segments.data?.industry.pending ? (
        <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="mb-3 flex items-center gap-2 rounded-2xl border border-sky-200 bg-sky-50 px-3.5 py-2.5 text-sm text-sky-800 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200">
          <Icon name="sparkles" size={15} className="es-spin-slow" />
          The AI is still sorting {segments.data.industry.pending} compan{segments.data.industry.pending === 1 ? 'y' : 'ies'} into industries — tiles update by themselves.
        </motion.div>
      ) : dim === 'industry' && segments.data && !segments.data.industry.ai ? (
        <div className="mb-3 rounded-2xl border border-dashed p-3 text-sm text-muted-foreground">Industries come from the AI, which isn't set up on the server (OPENROUTER_API_KEY) — only researched contacts have one.</div>
      ) : null}

      {/* the tiles — or, while searching, the matches from every tab */}
      {needle ? (
        <motion.div key="results" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="space-y-3">
          {results.length ? results.map((sec) => {
            const meta = TABS.find((x) => x.key === sec.tab)!;
            return (
              <section key={sec.tab}>
                <div className="mb-1.5 flex items-center gap-2 text-[12px] font-bold uppercase tracking-[0.06em] text-muted-foreground">
                  <span className="flex size-5 items-center justify-center rounded-md" style={{ background: `color-mix(in oklab, ${meta.tint} 15%, transparent)`, color: meta.tint }}><Icon name={meta.icon} size={11} /></span>
                  {meta.label}<span className="rounded-full bg-secondary px-1.5 text-[10.5px] leading-4 text-body">{sec.items.length}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-7">
                  <AnimatePresence initial>
                    {sec.items.slice(0, 14).map((it, i) => (
                      <Tile key={`${sec.tab}:${it.id}`} i={i} on={false} onClick={() => pick(sec.tab, it.id)} icon={it.icon} color={it.color} label={it.label} count={it.count}
                        {...(it.hot != null ? { hot: it.hot } : {})} avgScore={it.avg} />
                    ))}
                  </AnimatePresence>
                </div>
                {sec.items.length > 14 ? <p className="mt-1 text-xs text-muted-foreground">+{sec.items.length - 14} more — keep typing to narrow it down.</p> : null}
              </section>
            );
          }) : (
            <Card className="rounded-3xl"><EmptyState icon="search" title={`No list, industry, role, seniority or company matches “${query.trim()}”`}
              description={segments.isLoading ? 'Still loading the smart lists…' : 'Try part of a company or industry name.'} /></Card>
          )}
          {/* a person's name? search the contacts */}
          <button type="button" onClick={() => navigate({ to: '/contacts', search: { q: query.trim() } as never })}
            className="group inline-flex cursor-pointer items-center gap-1.5 rounded-full border bg-card px-3 py-1.5 text-xs font-bold text-body shadow-card transition-colors hover:border-brand-200 hover:text-brand-700">
            <Icon name="contacts" size={13} />Search contacts for “{query.trim()}”<Icon name="chevronRight" size={13} className="transition-transform group-hover:translate-x-0.5" />
          </button>
        </motion.div>
      ) : tab === 'mine' ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-7">
          <AnimatePresence initial>
            {listTiles.map((l, i) => (
              <Tile key={l.id} i={i} on={l.id === activeId} onClick={() => go({ list: l.id })} icon="list" color={listColor(l.name)} label={l.name}
                count={l.memberCount} sub={l.createdBy ? `· ${l.createdBy}` : undefined} />
            ))}
          </AnimatePresence>
          <motion.button type="button" onClick={() => setNaming({ name: '' })} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(listTiles.length, 16) * 0.035 }}
            whileHover={{ y: -4 }} whileTap={{ scale: 0.97 }}
            className="group flex min-h-[96px] cursor-pointer flex-col items-center justify-center gap-1.5 rounded-2xl border-2 border-dashed text-muted-foreground transition-colors hover:border-brand-300 hover:bg-brand-50/50 hover:text-brand-700">
            <span className="es-wiggle-hover flex size-8 items-center justify-center rounded-lg bg-secondary group-hover:bg-brand-100"><Icon name="plus" size={16} /></span>
            <span className="text-sm font-bold">New list</span>
          </motion.button>
        </div>
      ) : segments.isLoading ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-7">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-[96px] rounded-2xl" />)}</div>
      ) : segments.error ? <ErrorState error={segments.error} onRetry={() => segments.refetch()} />
        : !rows.length ? (
          <Card className="rounded-3xl"><EmptyState icon={TABS.find((t) => t.key === tab)!.icon} title="Nothing to group yet"
            description={dim === 'industry' ? 'Industries appear as companies get classified.' : 'Contacts need a job title or company for this.'} /></Card>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5 2xl:grid-cols-7">
            <AnimatePresence initial>
              {segTiles.slice(0, tab === 'company' ? 24 : undefined).map((r, i) => (
                <Tile key={`${dim}:${r.key}`} i={i} on={r.key === segKey} onClick={() => go({ seg: r.key })} icon={tileIcon(dim!, r.key, r.label)} color={segColor(i)}
                  label={r.label} count={r.count} hot={r.hotWarm ?? r.hot} avgScore={r.avgScore} share={total ? Math.round((r.count / total) * 100) : 0} />
              ))}
            </AnimatePresence>
          </div>
        )}
      {!needle && tab === 'company' && segTiles.length > 24 ? <p className="mt-2 text-xs text-muted-foreground">Showing the 24 biggest of {segTiles.length} companies — search to find the rest.</p> : null}

      {/* what's open */}
      <div className="mt-3">
        <AnimatePresence mode="wait" initial={false}>
          {tab === 'mine' ? (
            active ? (
              <motion.div key={`list:${active.id}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="space-y-4">
                <OpenHeader color={listColor(active.name)} initialsText={initials(active.name)} title={active.name}
                  sub={<>{active.createdAt ? <span title={fmtDateTime(active.createdAt)}>Created {relTime(active.createdAt)}</span> : 'Your list'}{active.createdBy ? ` by ${active.createdBy}` : ''}</>}
                  stats={[
                    { icon: 'contacts', label: 'Contacts', n: active.memberCount, tint: listColor(active.name) },
                    { icon: 'flame', label: 'Hot & warm', n: memberStats(members.data)?.hot ?? null, tint: '#ef4444' },
                    { icon: 'calendar', label: 'Events', n: memberStats(members.data)?.events ?? null, tint: '#0ea5e9' },
                    { icon: 'users', label: 'Reps', n: memberStats(members.data)?.reps ?? null, tint: '#10b981' },
                  ]}
                  actions={<>
                    <IconButton icon="pencil" label="Rename" onClick={() => setNaming({ id: active.id, name: active.name })} className="border bg-card" />
                    <IconButton icon="trash" label="Delete list" onClick={removeList} className="border bg-card hover:text-rose-600" />
                    <Button size="sm" icon="download" loading={exporting} disabled={!active.memberCount} onClick={exportList}>Export CSV</Button>
                    <Button size="sm" variant="primary" icon="plus" onClick={() => setAdding(true)}>Add contacts</Button>
                  </>} />
                <MembersPanel members={members.data} loading={members.isLoading} error={members.error} onRetry={() => members.refetch()} owners={owners}
                  emptyText="This list is empty" onRemove={removeMember}
                  emptyAction={<Button variant="primary" icon="plus" onClick={() => setAdding(true)}>Add contacts</Button>} />
              </motion.div>
            ) : (
              <motion.div key="no-list" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <Card className="rounded-3xl"><EmptyState icon="list" title="No lists yet" description="Make one for a campaign, an account or a follow-up day — or save any smart list from the other tabs."
                  action={<Button variant="primary" icon="plus" onClick={() => setNaming({ name: '' })}>Create a list</Button>} /></Card>
              </motion.div>
            )
          ) : segRow && dim ? (
            <motion.div key={`seg:${dim}:${segRow.key}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: MOTION.base, ease: MOTION.ease }} className="space-y-4">
              <OpenHeader color={segColor(rows.findIndex((r) => r.key === segRow.key))} icon={tileIcon(dim, segRow.key, segRow.label)} title={segRow.label}
                sub={<>Smart list · {TABS.find((t) => t.key === dim)!.label.toLowerCase()} · updates by itself as contacts come in</>}
                stats={[
                  { icon: 'contacts', label: 'Contacts', n: segRow.count, tint: tint },
                  { icon: 'flame', label: 'Hot & warm', n: segRow.hotWarm ?? segRow.hot, tint: '#ef4444' },
                  { icon: 'calendar', label: 'Events', n: memberStats(segMembers.data)?.events ?? null, tint: '#0ea5e9' },
                  { icon: 'users', label: 'Reps', n: memberStats(segMembers.data)?.reps ?? null, tint: '#10b981' },
                ]}
                actions={<>
                  <Button size="sm" icon="download" disabled={!segMembers.data?.length} onClick={exportSegment}>Export CSV</Button>
                  <Button size="sm" variant="primary" icon="save" onClick={() => setNaming({ name: segRow.label, fromSegment: { dim, key: segRow.key } })}>Save as list</Button>
                </>} />
              <MembersPanel members={segMembers.data} loading={segMembers.isLoading} error={segMembers.error} onRetry={() => segMembers.refetch()} owners={owners}
                emptyText="No one here any more" />
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      {active ? <AddContactsModal open={adding} onClose={() => setAdding(false)} listId={active.id} listName={active.name} existing={memberIds} /> : null}

      <Modal open={!!naming} onOpenChange={(o) => { if (!o) setNaming(null); }} size="sm"
        title={naming?.id ? 'Rename list' : naming?.fromSegment ? 'Save as a list' : 'New list'}
        description={naming?.fromSegment ? 'A copy of who is in it right now, as one of your lists.' : naming?.id ? undefined : 'A campaign, an account, a follow-up day…'}
        footer={<>
          <Button variant="ghost" onClick={() => setNaming(null)}>Cancel</Button>
          <Button variant="primary" icon={naming?.id ? 'save' : 'plus'} loading={saving} disabled={!naming?.name.trim()} onClick={saveName}>{naming?.id ? 'Save' : naming?.fromSegment ? 'Save list' : 'Create list'}</Button>
        </>}>
        <div className="flex items-center gap-3">
          <motion.span key={naming?.name ? listColor(naming.name) : 'x'} initial={{ scale: 0.6, rotate: -10 }} animate={{ scale: 1, rotate: 0 }} transition={spring}
            className="flex size-11 shrink-0 items-center justify-center rounded-xl text-sm font-black text-white shadow-card" style={{ background: naming?.name.trim() ? listColor(naming.name) : 'var(--muted-foreground)' }}>
            {naming?.name.trim() ? initials(naming.name) : <Icon name="list" size={18} />}
          </motion.span>
          <input autoFocus maxLength={255} value={naming?.name || ''} onChange={(e) => setNaming((n) => (n ? { ...n, name: e.target.value } : n))}
            // !saving: Enter pressed again (or held) mid-request created duplicate lists
            onKeyDown={(e) => { if (e.key === 'Enter' && !saving && naming?.name.trim()) void saveName(); }} placeholder="e.g. Hospital CIOs — follow up Monday" aria-label="List name"
            className="h-11 flex-1 rounded-xl border border-input bg-card px-3 text-sm outline-none transition-[border-color,box-shadow] focus:border-brand-400 focus:shadow-[0_0_0_4px_color-mix(in_oklab,var(--brand-400)_16%,transparent)]" />
        </div>
      </Modal>
    </>
  );
}
