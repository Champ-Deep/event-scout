// Announcements → Events: plan the events the team goes to, all on the real API.
//   list / add / edit / delete   GET · POST · PATCH · DELETE /api/admin/planner/events[/id]
//   team (who goes, the lead)    PUT  /api/admin/planner/events/{id}/team  (adds the event to their app's event list)
//   floor plans & files          POST /api/admin/files/upload {event, category: "floorplan"} · GET /api/files/{id}/download
//   announce                     POST /api/admin/planner/events/{id}/announce  (to the team, or everyone)
//   discovery                    GET  /api/admin/planner/discovery  (no scraper connected yet → says so; nothing is invented)
import React, { useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api, base64ToBlob, fileToBase64, saveBlob } from '../../lib/api';
import { q } from '../../lib/queries';
import { applyChange, type Change } from '../../lib/changes';
import type { AdminUser, PlannerEvent, PlannerEventInput, PlannerStatus } from '../../lib/types';
import { Avatar } from '../../ui/Avatar';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import {
  Button, Card, Drawer, EmptyState, Field, Select, Textarea, cx, errorMessage, fmtBytes, relTime, useUi, useWhileOpen, safeHref } from '../../ui/kit';
import { SearchField } from '../contacts/Toolbar';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '@/components/ui/dialog';

export const STATUS: Record<PlannerStatus, { label: string; color: string; icon: IconName }> = {
  suggested: { label: 'Suggested', color: '#64748b', icon: 'sparkles' },
  interested: { label: 'Interested', color: '#0ea5e9', icon: 'eye' },
  planned: { label: 'Planned', color: 'var(--brand-600)', icon: 'calendar' },
  attending: { label: 'Attending', color: '#10b981', icon: 'check' },
  done: { label: 'Done', color: '#94a3b8', icon: 'clock' },
  dismissed: { label: 'Dismissed', color: '#f43f5e', icon: 'x' },
};
const STATUS_ORDER: PlannerStatus[] = ['planned', 'attending', 'interested', 'suggested', 'done', 'dismissed'];
const DETAIL_FIELDS: { key: keyof PlannerEvent['details']; label: string; icon: IconName; placeholder: string }[] = [
  { key: 'booth', label: 'Our booth / stand', icon: 'store', placeholder: 'e.g. Hall 3, B12' },
  { key: 'hours', label: 'Opening hours', icon: 'clock', placeholder: 'e.g. 10:00–18:00, last day till 16:00' },
  { key: 'registration', label: 'Registration / badges', icon: 'idCard', placeholder: 'How and where to pick up badges' },
  { key: 'tickets', label: 'Tickets / passes', icon: 'key', placeholder: 'Exhibitor passes, codes, links' },
  { key: 'dressCode', label: 'Dress code', icon: 'briefcase', placeholder: 'e.g. Business, company T-shirt day 2' },
  { key: 'contactName', label: 'Organiser contact', icon: 'user', placeholder: 'Name' },
  { key: 'contactEmail', label: 'Contact email', icon: 'mail', placeholder: 'name@organiser.com' },
  { key: 'contactPhone', label: 'Contact phone', icon: 'phone', placeholder: '+971…' },
  { key: 'notes', label: 'Notes for the team', icon: 'note', placeholder: 'Parking, hotel, meeting point…' },
];

const fromKey = (d: string) => { const [y, m, day] = d.split('-').map(Number); return new Date(y!, (m ?? 1) - 1, day ?? 1); };
const todayKey = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; };

/** "12–14 Feb 2027", "28 Feb – 2 Mar 2027", "12 Feb 2027" or "Dates not set". */
export function fmtRange(start: string | null, end: string | null) {
  if (!start) return 'Dates not set';
  const s = fromKey(start);
  const e = end ? fromKey(end) : null;
  const full = (d: Date) => d.toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' });
  if (!e || end === start) return full(s);
  if (s.getFullYear() === e.getFullYear() && s.getMonth() === e.getMonth()) return `${s.getDate()}–${full(e)}`;
  return `${s.toLocaleDateString([], { day: 'numeric', month: 'short' })} – ${full(e)}`;
}
const place = (e: Pick<PlannerEvent, 'venue' | 'city' | 'country'>) => [e.venue, e.city, e.country].filter(Boolean).join(', ');
const mapsLink = (e: PlannerEvent) => {
  const where = [e.venue, e.address, e.city, e.country].filter(Boolean).join(', ');
  return where ? `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(where)}` : null;
};
/** When it is relative to today: "In 12 days", "On now", "Ended 3 weeks ago". */
function when(e: PlannerEvent) {
  if (!e.startDate) return null;
  const t = todayKey();
  const end = e.endDate || e.startDate;
  if (t >= e.startDate && t <= end) return { text: 'On now', tone: '#10b981' };
  const days = Math.round((fromKey(e.startDate).getTime() - fromKey(t).getTime()) / 86_400_000);
  if (days > 0) return { text: days === 1 ? 'Tomorrow' : days < 45 ? `In ${days} days` : `In ${Math.round(days / 30)} months`, tone: days <= 14 ? '#f59e0b' : 'var(--brand-600)' };
  return { text: `Ended ${relTime(end + 'T12:00:00')}`, tone: '#94a3b8' };
}

/** The announcement text for an event: what, when, where, booth, link, who leads (≤ 2,000 chars). */
export function eventMessage(e: PlannerEvent) {
  const lead = e.team.find((m) => m.role === 'lead');
  const lines = [
    `Event: ${e.name}`,
    `When: ${fmtRange(e.startDate, e.endDate)}${e.details.hours ? ` · ${e.details.hours}` : ''}`,
    place(e) ? `Where: ${place(e)}${e.address ? ` (${e.address})` : ''}` : '',
    e.details.booth ? `Our booth: ${e.details.booth}` : '',
    e.details.registration ? `Badges: ${e.details.registration}` : '',
    lead ? `Team lead: ${lead.name}` : '',
    e.files.some((f) => f.floorPlan) ? 'Floor plan: in the app under the event → Files' : '',
    e.website ? e.website : '',
  ].filter(Boolean);
  return lines.join('\n').slice(0, 2000);
}

const EMPTY_FORM: PlannerEventInput = { name: '', status: 'planned', startDate: '', endDate: '', venue: '', address: '', city: '', country: '', website: '', organizer: '', industry: '', description: '', details: {} };

function StatusPill({ status, small }: { status: PlannerStatus; small?: boolean }) {
  const s = STATUS[status] ?? STATUS.planned;
  return (
    <span className={cx('inline-flex items-center gap-1 rounded-full font-bold', small ? 'px-1.5 text-[10px] leading-[18px]' : 'px-2 py-0.5 text-[11px]')}
      style={{ color: s.color, background: `color-mix(in oklab, ${s.color} 13%, transparent)` }}>
      <Icon name={s.icon} size={small ? 10 : 11} />{s.label}
    </span>
  );
}

function TeamStack({ team, max = 4 }: { team: PlannerEvent['team']; max?: number }) {
  if (!team.length) return <span className="text-xs font-semibold text-muted-foreground">No team yet</span>;
  return (
    <span className="flex items-center">
      {team.slice(0, max).map((m, i) => (
        <span key={m.userId} title={`${m.name}${m.role === 'lead' ? ' · lead' : ''}`} className={cx('relative rounded-full ring-2 ring-card', i && '-ml-2')}>
          <Avatar src={m.photoUrl} name={m.name} size="xs" className="size-7 border-0 text-[10px] shadow-none" />
          {m.role === 'lead' ? <span className="absolute -right-0.5 -top-0.5 size-2.5 rounded-full bg-amber-400 ring-2 ring-card" /> : null}
        </span>
      ))}
      {team.length > max ? <span className="-ml-2 flex size-7 items-center justify-center rounded-full bg-secondary text-[10px] font-black text-body ring-2 ring-card">+{team.length - max}</span> : null}
    </span>
  );
}

/* ------------------------------------------------------------------ form */

/** A labelled input with its own tinted icon that lights up while you type in it. */
function IconField({ label, icon, tint = 'var(--brand-600)', hint, required, className, children }: {
  label: string; icon: IconName; tint?: string; hint?: React.ReactNode; required?: boolean; className?: string; children: React.ReactElement;
}) {
  return (
    <label className={cx('group/f block', className)}>
      <span className="mb-1.5 flex items-center justify-between gap-2 text-[12px] font-semibold text-body">
        <span>{label}{required ? <span className="ml-0.5 text-rose-500">*</span> : null}</span>
        {hint ? <span className="text-[11px] font-medium text-muted-foreground">{hint}</span> : null}
      </span>
      <span className="relative block">
        <span className="pointer-events-none absolute left-2 top-2 flex size-6 items-center justify-center rounded-lg transition-[transform,background-color] duration-200 group-focus-within/f:scale-110"
          style={{ color: tint, background: `color-mix(in oklab, ${tint} 12%, transparent)` }}>
          <Icon name={icon} size={13} />
        </span>
        {children}
      </span>
    </label>
  );
}
const FIELD = 'w-full rounded-xl border border-input bg-card pl-10 pr-3 text-sm text-foreground shadow-xs outline-none transition-[border-color,box-shadow] placeholder:text-muted-foreground focus:border-brand-400 focus:shadow-[0_0_0_4px_color-mix(in_oklab,var(--brand-400)_16%,transparent)]';

type Step = 'basics' | 'where' | 'team' | 'review';
const STEPS: { key: Step; label: string; sub: string; icon: IconName; tint: string }[] = [
  { key: 'basics', label: 'The event', sub: 'Name, status, what it is', icon: 'flag', tint: 'var(--brand-600)' },
  { key: 'where', label: 'When & where', sub: 'Dates, venue, website', icon: 'mapPin', tint: '#0ea5e9' },
  { key: 'team', label: 'For the team', sub: 'Booth, badges, contacts', icon: 'users', tint: '#f59e0b' },
  { key: 'review', label: 'Review', sub: 'Check and save', icon: 'check', tint: '#10b981' },
];
const DETAIL_TINT: Partial<Record<keyof PlannerEvent['details'], string>> = {
  booth: 'var(--brand-600)', hours: '#0ea5e9', registration: '#f59e0b', tickets: '#d946ef', dressCode: '#64748b',
  contactName: '#10b981', contactEmail: '#6366f1', contactPhone: '#10b981', notes: '#f43f5e',
};
const DETAIL_ICON: Partial<Record<keyof PlannerEvent['details'], IconName>> = { tickets: 'ticket', dressCode: 'shirt' };

/** How many days an event runs, for the little badge next to the dates. */
const spanDays = (s?: string | null, e?: string | null) => (s && e && e >= s ? Math.round((fromKey(e).getTime() - fromKey(s).getTime()) / 86_400_000) + 1 : s ? 1 : 0);

function EventForm({ value, onChange, step }: { value: PlannerEventInput; onChange: (v: PlannerEventInput) => void; step: Step }) {
  const set = (patch: PlannerEventInput) => onChange({ ...value, ...patch });
  const setDetail = (k: string, v: string) => onChange({ ...value, details: { ...(value.details ?? {}), [k]: v } });
  const days = spanDays(value.startDate, value.endDate);
  const badDates = !!(value.startDate && value.endDate && value.endDate < value.startDate);
  const site = (value.website ?? '').trim();
  const badSite = !!site && (!safeHref(site) || !/^(https?:\/\/)?[^\s/$.?#:]+\.[^\s]+$/i.test(site)); // a web address only (no javascript:/data: links)
  const stagger = (i: number) => ({ initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: MOTION.base, ease: MOTION.ease, delay: 0.04 + i * 0.035 } });

  if (step === 'basics') return (
    <div className="space-y-5">
      <motion.div {...stagger(0)}>
        <IconField label="Event name" icon="flag" required hint={`${(value.name ?? '').length}/255`}>
          <input autoFocus maxLength={255} value={value.name ?? ''} onChange={(e) => set({ name: e.target.value })} placeholder="e.g. WHX Dubai 2027" className={cx(FIELD, 'h-11 text-[15px] font-semibold')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(1)}>
        <div className="mb-1.5 text-[12px] font-semibold text-body">Status</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {STATUS_ORDER.map((s) => {
            const on = (value.status ?? 'planned') === s;
            const c = STATUS[s].color;
            return (
              <motion.button key={s} type="button" onClick={() => set({ status: s })} aria-pressed={on} whileHover={{ y: -2 }} whileTap={{ scale: 0.96 }}
                className={cx('relative flex cursor-pointer items-center gap-2 rounded-xl border-[1.5px] px-3 py-2.5 text-left text-[13px] font-bold transition-colors', on ? 'shadow-card' : 'border-border bg-card hover:border-brand-200')}
                style={on ? { borderColor: c, background: `color-mix(in oklab, ${c} 10%, var(--card))` } : {}}>
                <span className="flex size-7 items-center justify-center rounded-lg transition-transform" style={{ background: on ? c : `color-mix(in oklab, ${c} 14%, transparent)`, color: on ? 'white' : c }}>
                  <Icon name={STATUS[s].icon} size={14} />
                </span>
                {STATUS[s].label}
                {on ? <motion.span layoutId="es-status-check" className="ml-auto" style={{ color: c }}><Icon name="check" size={14} strokeWidth={3} /></motion.span> : null}
              </motion.button>
            );
          })}
        </div>
      </motion.div>
      <motion.div {...stagger(2)} className="grid gap-4 sm:grid-cols-2">
        <IconField label="Industry / theme" icon="tag" tint="#f59e0b">
          <input value={value.industry ?? ''} onChange={(e) => set({ industry: e.target.value })} placeholder="e.g. Healthcare, medical devices" className={cx(FIELD, 'h-10')} />
        </IconField>
        <IconField label="Organiser" icon="users" tint="#10b981">
          <input value={value.organizer ?? ''} onChange={(e) => set({ organizer: e.target.value })} placeholder="Who runs it" className={cx(FIELD, 'h-10')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(3)}>
        <IconField label="About the event" icon="note" tint="#6366f1" hint="Why we're going, who exhibits">
          <textarea rows={4} value={value.description ?? ''} onChange={(e) => set({ description: e.target.value })} placeholder="What it is, who exhibits, why we're going…" className={cx(FIELD, 'resize-y py-2.5 leading-relaxed')} />
        </IconField>
      </motion.div>
    </div>
  );

  if (step === 'where') return (
    <div className="space-y-5">
      <motion.div {...stagger(0)} className="grid items-end gap-3 sm:grid-cols-[1fr_auto_1fr]">
        <IconField label="Starts" icon="calendar">
          <input type="date" value={value.startDate ?? ''} onChange={(e) => set({ startDate: e.target.value, ...(value.endDate && e.target.value > value.endDate ? { endDate: e.target.value } : {}) })} className={cx(FIELD, 'h-10')} />
        </IconField>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span key={badDates ? 'bad' : days} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.6, opacity: 0 }}
            className={cx('mb-1.5 hidden h-7 items-center justify-center whitespace-nowrap rounded-full px-2.5 text-[11px] font-black sm:flex',
              badDates ? 'bg-rose-50 text-rose-600' : days ? 'bg-brand-50 text-brand-700' : 'bg-secondary text-muted-foreground')}>
            {badDates ? 'Ends before it starts' : days ? `${days} day${days === 1 ? '' : 's'}` : '→'}
          </motion.span>
        </AnimatePresence>
        <IconField label="Ends" icon="calendar" tint="#0ea5e9">
          <input type="date" min={value.startDate || undefined} value={value.endDate ?? ''} onChange={(e) => set({ endDate: e.target.value })} className={cx(FIELD, 'h-10', badDates && 'border-rose-400')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(1)}>
        <IconField label="Venue" icon="building" tint="#0ea5e9">
          <input value={value.venue ?? ''} onChange={(e) => set({ venue: e.target.value })} placeholder="e.g. Dubai World Trade Centre" className={cx(FIELD, 'h-10')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(2)}>
        <IconField label="Address" icon="mapPin" tint="#f43f5e">
          <input value={value.address ?? ''} onChange={(e) => set({ address: e.target.value })} placeholder="Street, area" className={cx(FIELD, 'h-10')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(3)} className="grid gap-4 sm:grid-cols-2">
        <IconField label="City" icon="building" tint="#10b981">
          <input value={value.city ?? ''} onChange={(e) => set({ city: e.target.value })} placeholder="e.g. Dubai" className={cx(FIELD, 'h-10')} />
        </IconField>
        <IconField label="Country" icon="flag" tint="#f59e0b">
          <input value={value.country ?? ''} onChange={(e) => set({ country: e.target.value })} placeholder="e.g. United Arab Emirates" className={cx(FIELD, 'h-10')} />
        </IconField>
      </motion.div>
      <motion.div {...stagger(4)}>
        <IconField label="Website" icon="globe" tint="#6366f1" hint={badSite ? <span className="text-rose-600">That doesn't look like a web address</span> : 'https:// is added for you'}>
          <input inputMode="url" value={value.website ?? ''} onChange={(e) => set({ website: e.target.value })} placeholder="e.g. worldhealthexpo.com" className={cx(FIELD, 'h-10', badSite && 'border-rose-400')} />
        </IconField>
      </motion.div>
    </div>
  );

  if (step === 'team') return (
    <div className="grid gap-4 sm:grid-cols-2">
      {DETAIL_FIELDS.map((f, i) => (
        <motion.div key={f.key} {...stagger(i)} className={f.key === 'notes' ? 'sm:col-span-2' : ''}>
          <IconField label={f.label} icon={DETAIL_ICON[f.key] ?? f.icon} tint={DETAIL_TINT[f.key] ?? 'var(--brand-600)'}>
            {f.key === 'notes'
              ? <textarea rows={3} value={value.details?.[f.key] ?? ''} onChange={(e) => setDetail(f.key, e.target.value)} placeholder={f.placeholder} className={cx(FIELD, 'resize-y py-2.5 leading-relaxed')} />
              : <input type={f.key === 'contactEmail' ? 'email' : f.key === 'contactPhone' ? 'tel' : 'text'} value={value.details?.[f.key] ?? ''} onChange={(e) => setDetail(f.key, e.target.value)} placeholder={f.placeholder} className={cx(FIELD, 'h-10')} />}
          </IconField>
        </motion.div>
      ))}
    </div>
  );

  // review: the card as it will look in Events, and every filled-in detail
  const status = STATUS[(value.status ?? 'planned') as PlannerStatus] ?? STATUS.planned;
  const where = [value.venue, value.city, value.country].filter(Boolean).join(', ');
  const filled = DETAIL_FIELDS.filter((f) => (value.details?.[f.key] ?? '').trim());
  return (
    <div className="space-y-4">
      <motion.div {...stagger(0)} className="relative overflow-hidden rounded-2xl border bg-card p-4 shadow-card">
        <span className="absolute inset-x-0 top-0 h-[3px]" style={{ background: status.color }} aria-hidden />
        <div className="flex items-start gap-3">
          <motion.span initial={{ rotate: -8, scale: 0.8 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: 'spring', stiffness: 360, damping: 18 }}
            className="flex size-14 shrink-0 flex-col items-center justify-center rounded-xl bg-brand-600 text-white shadow-active">
            {value.startDate ? <>
              <span className="text-[9px] font-bold uppercase leading-none tracking-[0.08em] opacity-80">{fromKey(value.startDate).toLocaleDateString([], { month: 'short' })}</span>
              <span className="text-xl font-black leading-6">{fromKey(value.startDate).getDate()}</span>
            </> : <Icon name="calendar" size={20} />}
          </motion.span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="break-words text-[16px] font-extrabold text-foreground">{value.name?.trim() || <span className="italic text-muted-foreground">No name yet</span>}</span>
              <StatusPill status={(value.status ?? 'planned') as PlannerStatus} small />
            </div>
            <div className="mt-0.5 text-[13px] text-body">{fmtRange(value.startDate || null, value.endDate || null)}</div>
            <div className="text-[13px] text-muted-foreground">{where || 'Venue not set'}</div>
            {value.industry ? <div className="mt-1.5 inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 text-[11px] font-bold leading-5 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300"><Icon name="tag" size={10} />{value.industry}</div> : null}
          </div>
        </div>
        {value.description ? <p className="mt-3 line-clamp-3 whitespace-pre-line border-t pt-3 text-[13px] leading-relaxed text-body">{value.description}</p> : null}
      </motion.div>
      <motion.div {...stagger(1)} className="rounded-2xl border bg-card p-1.5 shadow-card">
        {[
          { icon: 'globe' as IconName, label: 'Website', v: site },
          { icon: 'mapPin' as IconName, label: 'Address', v: value.address ?? '' },
          { icon: 'users' as IconName, label: 'Organiser', v: value.organizer ?? '' },
          ...filled.map((f) => ({ icon: DETAIL_ICON[f.key] ?? f.icon, label: f.label, v: value.details?.[f.key] ?? '' })),
        ].filter((r) => r.v.trim()).map((r, i) => (
          <motion.div key={r.label} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, delay: 0.1 + i * 0.03 }}
            className="flex items-start gap-3 rounded-xl px-2 py-1.5">
            <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-secondary text-muted-foreground"><Icon name={r.icon} size={13} /></span>
            <div className="min-w-0">
              <div className="text-[11px] font-medium text-muted-foreground">{r.label}</div>
              <div className="whitespace-pre-line break-words text-sm font-medium text-foreground">{r.v}</div>
            </div>
          </motion.div>
        ))}
        {!site && !value.address && !value.organizer && !filled.length ? <p className="p-3 text-sm text-muted-foreground">No extra details yet — you can add them any time.</p> : null}
      </motion.div>
      <motion.p {...stagger(2)} className="flex items-start gap-2 text-xs text-muted-foreground">
        <Icon name="info" size={13} className="mt-0.5 shrink-0" />After saving, open the event to upload the floor plan, pick the team and announce it.
      </motion.p>
    </div>
  );
}

const toInput = (e: PlannerEvent): PlannerEventInput => ({
  name: e.name, status: e.status, startDate: e.startDate ?? '', endDate: e.endDate ?? '', venue: e.venue, address: e.address, city: e.city,
  country: e.country, website: e.website, organizer: e.organizer, industry: e.industry, description: e.description, details: { ...e.details },
});

/* ------------------------------------------------------------------ panel */

type PanelTab = 'overview' | 'files' | 'team' | 'announce';

function EventPanel({ ev, users, onClose, onEdit }: { ev: PlannerEvent | null; users: AdminUser[]; onClose: () => void; onEdit: (e: PlannerEvent) => void }) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const [tab, setTab] = useState<PanelTab>('overview');
  const [busy, setBusy] = useState<string | null>(null);
  // team being edited (null = not editing)
  const [draft, setDraft] = useState<Record<string, 'lead' | 'member'> | null>(null);
  const [teamQuery, setTeamQuery] = useState('');
  const [msg, setMsg] = useState('');
  const [prio, setPrio] = useState<'normal' | 'urgent'>('normal');
  const [audience, setAudience] = useState<'team' | 'everyone'>('team');
  const [upCat, setUpCat] = useState<'floorplan' | 'general'>('floorplan');
  const fileInput = useRef<HTMLInputElement>(null);
  const id = ev?.id;
  // announcements already sent about this event
  const history = useQuery({ ...q.broadcastHistory(), enabled: !!ev });
  const sentAbout = (history.data ?? []).filter((b) => b.eventId === id);
  React.useEffect(() => { setTab('overview'); setDraft(null); setMsg(''); setAudience('team'); setPrio('normal'); }, [id]);
  React.useEffect(() => { if (tab === 'announce' && ev && !msg) setMsg(eventMessage(ev)); if (tab === 'team' && ev && !draft) setDraft(Object.fromEntries(ev.team.map((m) => [m.userId, m.role]))); }, [tab, id]);

  const refresh = (change: Change | Change[] = 'event_planned') => { void applyChange(qc, change); };

  if (!ev) return <Drawer open={false} onOpenChange={() => {}} title="Event" mode="side">{null}</Drawer>;
  const w = when(ev);
  const map = mapsLink(ev);
  const floor = ev.files.filter((f) => f.floorPlan);
  const others = ev.files.filter((f) => !f.floorPlan);

  const setStatus = async (status: PlannerStatus) => {
    setBusy('status');
    try { await api.updatePlannerEvent(ev.id, { status }); toast('success', `${ev.name} · ${STATUS[status].label}`); refresh(); }
    catch (e) { toast('error', "Couldn't change the status", errorMessage(e)); }
    finally { setBusy(null); }
  };
  const saveTeam = async () => {
    if (!draft) return;
    setBusy('team');
    try {
      const members = Object.entries(draft).map(([userId, role]) => ({ userId, role }));
      await api.setPlannerTeam(ev.id, members);
      toast('success', `Team saved · ${members.length} ${members.length === 1 ? 'person' : 'people'}`, 'New members see the event in their app under Events.');
      refresh();
    } catch (e) { toast('error', "Couldn't save the team", errorMessage(e)); }
    finally { setBusy(null); }
  };
  const upload = async (f: File | undefined) => {
    if (!f) return;
    if (f.size > 20 * 1024 * 1024) { toast('error', 'That file is over 20 MB'); return; }
    setBusy('upload');
    try {
      await api.uploadFile({ filename: f.name, contentBase64: await fileToBase64(f), mimeType: f.type || undefined, category: upCat, description: upCat === 'floorplan' ? `Floor plan · ${ev.name}` : '', event: ev.name });
      toast('success', `${upCat === 'floorplan' ? 'Floor plan' : 'File'} added`, `Reps see it on ${ev.name} in the app.`);
      refresh('files_changed');
    } catch (e) { toast('error', "Couldn't upload", errorMessage(e)); }
    finally { setBusy(null); if (fileInput.current) fileInput.current.value = ''; }
  };
  const download = async (fileId: string) => {
    setBusy(fileId);
    try { const d = await api.downloadFile(fileId); saveBlob(base64ToBlob(d.contentBase64, d.mimeType), d.filename); }
    catch (e) { toast('error', "Couldn't download", errorMessage(e)); }
    finally { setBusy(null); }
  };
  const announce = async () => {
    setBusy('announce');
    try {
      const r = await api.announcePlannerEvent(ev.id, { message: msg.trim(), priority: prio, audience });
      toast('success', 'Announcement sent', r.recipients ? `To the ${r.recipients}-person team` : 'To everyone');
      setMsg(eventMessage(ev));
      refresh(['broadcast_alert', 'event_planned']);
    } catch (e) { toast('error', "Couldn't send", errorMessage(e)); }
    finally { setBusy(null); }
  };
  const remove = async () => {
    if (!(await confirm({ title: `Remove ${ev.name}?`, message: 'The plan and its team are removed (it leaves the team’s app event list). Contacts met there and the event’s files stay.', confirmLabel: 'Remove', danger: true }))) return;
    try { await api.deletePlannerEvent(ev.id); toast('success', `${ev.name} removed`); onClose(); refresh(); }
    catch (e) { toast('error', "Couldn't remove it", errorMessage(e)); }
  };

  const people = users.filter((u) => !teamQuery || `${u.name} ${u.email}`.toLowerCase().includes(teamQuery.toLowerCase()));
  const teamChanged = !!draft && JSON.stringify(Object.entries(draft).sort()) !== JSON.stringify(ev.team.map((m) => [m.userId, m.role]).sort());

  const tabs: { v: PanelTab; label: string; icon: IconName; n?: number }[] = [
    { v: 'overview', label: 'Overview', icon: 'info' },
    { v: 'files', label: 'Floor plan & files', icon: 'file', n: ev.files.length },
    { v: 'team', label: 'Team', icon: 'users', n: ev.team.length },
    { v: 'announce', label: 'Announce', icon: 'megaphone' },
  ];

  return (
    <Drawer open={!!ev} onOpenChange={(o) => { if (!o) onClose(); }} title={ev.name} mode="side" width="sm:max-w-2xl" scrollKey={`${ev.id}:${tab}`}
      header={(
        <div className="-mx-6 -mb-4 -mt-4 px-6 pb-4 pt-5" style={{ background: `color-mix(in oklab, ${STATUS[ev.status]?.color ?? 'var(--primary)'} 8%, var(--panel))` }}>
          <div className="flex items-start gap-4 pr-8">
            <motion.span initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 22 }}
              className="flex size-16 shrink-0 flex-col items-center justify-center rounded-2xl bg-brand-600 text-white shadow-active">
              {ev.startDate ? <>
                <span className="text-[10px] font-bold uppercase tracking-[0.08em] opacity-80">{fromKey(ev.startDate).toLocaleDateString([], { month: 'short' })}</span>
                <span className="text-2xl font-black leading-6">{fromKey(ev.startDate).getDate()}</span>
              </> : <Icon name="calendar" size={24} />}
            </motion.span>
            <div className="min-w-0 flex-1">
              <h2 className="break-words font-display text-[22px] font-bold leading-7 tracking-[-0.02em] text-foreground">{ev.name}</h2>
              <div className="mt-1 space-y-0.5 text-[13.5px]">
                <div className="flex items-center gap-1.5 text-body"><Icon name="calendar" size={14} className="text-brand-600" />{fmtRange(ev.startDate, ev.endDate)}{w ? <span className="font-bold" style={{ color: w.tone }}>· {w.text}</span> : null}</div>
                <div className="flex items-start gap-1.5 text-muted-foreground"><Icon name="building" size={14} className="mt-[3px] text-sky-600" /><span className={cx(!place(ev) && 'italic')}>{place(ev) || 'Venue not set'}</span></div>
              </div>
            </div>
          </div>
          <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
            <Select aria-label="Status" value={ev.status} disabled={busy === 'status'} onChange={(e) => setStatus(e.target.value as PlannerStatus)} className="h-8 rounded-full text-xs font-bold">
              {STATUS_ORDER.map((s) => <option key={s} value={s}>{STATUS[s].label}</option>)}
            </Select>
            {ev.industry ? <span className="inline-flex h-7 items-center gap-1.5 rounded-full bg-card px-2.5 text-xs font-medium text-body ring-1 ring-inset ring-border"><Icon name="briefcase" size={12} className="text-amber-600" />{ev.industry}</span> : null}
            <span className="inline-flex h-7 items-center gap-1.5 rounded-full bg-card px-2.5 text-xs font-medium text-body ring-1 ring-inset ring-border"><Icon name="contacts" size={12} className="text-brand-600" />{ev.contactsMet} contact{ev.contactsMet === 1 ? '' : 's'} met</span>
            <span className="ml-auto flex items-center gap-1.5">
              {safeHref(ev.website) ? <a href={safeHref(ev.website)} target="_blank" rel="noreferrer" title="Event website" className="flex size-9 items-center justify-center rounded-full bg-sky-50 text-sky-700 ring-1 ring-inset ring-black/5 transition-transform hover:scale-110 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="external" size={15} /></a> : null}
              {map ? <a href={map} target="_blank" rel="noreferrer" title="Open the venue in Google Maps" className="flex size-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-700 ring-1 ring-inset ring-black/5 transition-transform hover:scale-110 dark:bg-emerald-500/15 dark:text-emerald-300"><Icon name="search" size={15} /></a> : null}
            </span>
          </div>
        </div>
      )}
      footer={<>
        <Button variant="ghost" icon="trash" className="text-rose-600 hover:bg-rose-50 hover:text-rose-700" onClick={remove}>Remove</Button>
        <div className="flex gap-2">
          <Button icon="pencil" onClick={() => onEdit(ev)}>Edit details</Button>
          <Button variant="primary" icon="megaphone" onClick={() => setTab('announce')}>Announce</Button>
        </div>
      </>}>
      <div className="px-6 pb-5">
        <div className="sticky top-0 z-20 -mx-6 bg-card/95 px-6 pb-2.5 pt-4 backdrop-blur">
          <div role="tablist" className="no-scrollbar inline-flex max-w-full overflow-x-auto rounded-2xl bg-secondary p-1">
            {tabs.map((t) => (
              <button key={t.v} role="tab" aria-selected={tab === t.v} onClick={() => setTab(t.v)}
                className={cx('relative inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition-colors', tab === t.v ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
                {tab === t.v ? <motion.span layoutId="es-event-tab" className="absolute inset-0 rounded-xl bg-card shadow-card" transition={{ duration: MOTION.base, ease: MOTION.ease }} /> : null}
                <Icon name={t.icon} size={13} className={cx('relative', tab === t.v && 'text-primary')} />
                <span className="relative">{t.label}</span>
                {t.n ? <span className="relative rounded-full bg-brand-50 px-1.5 text-[10px] font-bold text-brand-700">{t.n}</span> : null}
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={tab} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.fast }} className="mt-3 space-y-4">
            {tab === 'overview' ? <>
              {ev.description ? <p className="whitespace-pre-line rounded-2xl border bg-card p-4 text-sm leading-relaxed text-body shadow-card">{ev.description}</p> : null}
              <div className="rounded-2xl border bg-card p-1.5 shadow-card">
                {[
                  { icon: 'calendar' as IconName, label: 'Dates', value: fmtRange(ev.startDate, ev.endDate) },
                  { icon: 'building' as IconName, label: 'Venue', value: ev.venue },
                  { icon: 'search' as IconName, label: 'Address', value: [ev.address, ev.city, ev.country].filter(Boolean).join(', '), href: map ?? undefined },
                  { icon: 'external' as IconName, label: 'Website', value: ev.website, href: safeHref(ev.website) },
                  { icon: 'users' as IconName, label: 'Organiser', value: ev.organizer },
                  ...DETAIL_FIELDS.map((f) => ({ icon: f.icon, label: f.label, value: ev.details[f.key] ?? '', href: f.key === 'contactEmail' && ev.details.contactEmail ? `mailto:${ev.details.contactEmail}` : f.key === 'contactPhone' && ev.details.contactPhone ? `tel:${ev.details.contactPhone}` : undefined })),
                ].map((r, i) => (
                  <motion.div key={r.label} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, delay: i * 0.02 }}
                    className="flex items-start gap-3 rounded-xl px-2 py-2 hover:bg-secondary/60">
                    <span className={cx('mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-secondary text-muted-foreground', !r.value && 'opacity-45')}><Icon name={r.icon} size={13} /></span>
                    <div className="min-w-0 flex-1">
                      <div className="text-[11px] font-medium text-muted-foreground">{r.label}</div>
                      <div className="whitespace-pre-line break-words text-sm font-medium text-foreground">
                        {r.value ? (r.href ? <a href={r.href} target={r.href.startsWith('http') ? '_blank' : undefined} rel="noreferrer" className="text-primary underline-offset-2 hover:underline">{r.value}</a> : r.value)
                          : <span className="font-normal italic text-muted-foreground">Not added</span>}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">{ev.source === 'manual' ? 'Added by hand' : 'Found by event discovery'}{ev.createdBy ? ` by ${ev.createdBy}` : ''}{ev.createdAt ? ` · ${relTime(ev.createdAt)}` : ''}{ev.updatedAt && ev.updatedAt !== ev.createdAt ? ` · updated ${relTime(ev.updatedAt)}` : ''}</p>
            </> : null}

            {tab === 'files' ? <>
              <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-dashed p-3">
                <Icon name="upload" size={16} className="text-brand-600" />
                <span className="flex-1 text-sm text-body">Add a floor plan or other file — reps assigned to the event see it in the app.</span>
                <Select aria-label="Kind of file" value={upCat} onChange={(e) => setUpCat(e.target.value as 'floorplan' | 'general')} className="h-8 text-xs">
                  <option value="floorplan">Floor plan</option>
                  <option value="general">Other file</option>
                </Select>
                <Button size="sm" variant="primary" icon="upload" loading={busy === 'upload'} onClick={() => fileInput.current?.click()}>Upload</Button>
                <input ref={fileInput} type="file" accept=".pdf,.png,.jpg,.jpeg,.ppt,.pptx,.doc,.docx,.xls,.xlsx,.csv,.txt" className="hidden" onChange={(e) => upload(e.target.files?.[0])} />
              </div>
              {[{ title: 'Floor plans', list: floor, icon: 'columns' as IconName, empty: 'No floor plan yet — upload the organiser’s hall map so the team can find booths.' },
                { title: 'Other files', list: others, icon: 'file' as IconName, empty: 'Nothing else attached to this event.' }].map((g) => (
                <div key={g.title}>
                  <div className="mb-2 flex items-center gap-2 text-[13px] font-semibold text-foreground"><Icon name={g.icon} size={14} className="text-brand-600" />{g.title}<span className="text-xs font-normal text-muted-foreground">{g.list.length}</span></div>
                  {g.list.length ? (
                    <ul className="space-y-1.5">
                      {g.list.map((f) => (
                        <li key={f.id} className="flex items-center gap-3 rounded-xl border bg-card px-3 py-2 shadow-card">
                          <span className="flex size-9 items-center justify-center rounded-lg bg-secondary text-[10px] font-black uppercase text-muted-foreground">{f.fileType}</span>
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-semibold text-foreground">{f.filename}</div>
                            <div className="text-xs text-muted-foreground">{fmtBytes(f.fileSize)}{f.createdAt ? ` · ${relTime(f.createdAt)}` : ''}</div>
                          </div>
                          <Button size="sm" icon="download" loading={busy === f.id} onClick={() => download(f.id)}>Get</Button>
                        </li>
                      ))}
                    </ul>
                  ) : <p className="rounded-xl border border-dashed p-3 text-sm text-muted-foreground">{g.empty}</p>}
                </div>
              ))}
              <p className="text-xs text-muted-foreground">Files live in Team files too — delete or re-upload them there.</p>
            </> : null}

            {tab === 'team' && draft ? <>
              <p className="text-sm text-body">Pick who goes. Each person gets <b className="font-semibold">{ev.name}</b> in their app's event list; star one as the team lead.</p>
              <SearchField value={teamQuery} onChange={setTeamQuery} placeholder="Find a teammate…" />
              <ul className="max-h-[46vh] space-y-1 overflow-y-auto rounded-2xl border bg-card p-1.5 shadow-card">
                {people.map((u) => {
                  const role = draft[u.id];
                  return (
                    <li key={u.id} className={cx('flex items-center gap-3 rounded-xl px-2 py-1.5 transition-colors', role ? 'bg-brand-50/70' : 'hover:bg-secondary/60')}>
                      <button onClick={() => setDraft((d) => { const n = { ...(d ?? {}) }; if (n[u.id]) delete n[u.id]; else n[u.id] = 'member'; return n; })}
                        aria-pressed={!!role} className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-left">
                        <span className={cx('flex size-5 shrink-0 items-center justify-center rounded-md border-[1.5px]', role ? 'border-brand-600 bg-brand-600 text-white' : 'border-border')}>{role ? <Icon name="check" size={12} strokeWidth={3} /> : null}</span>
                        <Avatar src={u.photoUrl || null} name={u.name} size="xs" className="size-7 border-0 text-[10px] shadow-none" />
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-semibold text-foreground">{u.name}{u.isAdmin ? <span className="ml-1.5 text-[10px] font-bold uppercase text-muted-foreground">admin</span> : null}</span>
                          <span className="block truncate text-xs text-muted-foreground">{u.email}</span>
                        </span>
                      </button>
                      {role ? (
                        <button onClick={() => setDraft((d) => Object.fromEntries(Object.entries(d ?? {}).map(([k, r]) => [k, k === u.id ? (r === 'lead' ? 'member' : 'lead') : (r === 'lead' ? 'member' : r)])) as Record<string, 'lead' | 'member'>)}
                          title={role === 'lead' ? 'Team lead — click to make a member' : 'Make team lead'}
                          className={cx('inline-flex h-7 cursor-pointer items-center gap-1 rounded-full px-2.5 text-xs font-bold transition-colors', role === 'lead' ? 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200' : 'text-muted-foreground hover:bg-secondary')}>
                          <Icon name="flame" size={12} />{role === 'lead' ? 'Lead' : 'Make lead'}
                        </button>
                      ) : null}
                    </li>
                  );
                })}
                {!people.length ? <li className="p-3 text-sm text-muted-foreground">No one matches.</li> : null}
              </ul>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-muted-foreground">{Object.keys(draft).length} selected</span>
                <div className="flex gap-2">
                  {teamChanged ? <Button variant="ghost" onClick={() => setDraft(Object.fromEntries(ev.team.map((m) => [m.userId, m.role])))}>Undo</Button> : null}
                  <Button variant="primary" icon="save" loading={busy === 'team'} disabled={!teamChanged} onClick={saveTeam}>Save team</Button>
                </div>
              </div>
            </> : null}

            {tab === 'announce' ? <>
              <Field label="Message" hint={`${msg.length.toLocaleString()}/2,000`}>
                <Textarea rows={10} maxLength={2000} value={msg} onChange={(e) => setMsg(e.target.value)} />
              </Field>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="ghost" icon="refresh" onClick={() => setMsg(eventMessage(ev))}>Rebuild from details</Button>
                <span className="flex-1" />
                <Select aria-label="Send to" value={audience} onChange={(e) => setAudience(e.target.value as 'team' | 'everyone')} className="h-8 text-xs">
                  <option value="team">The event team ({ev.team.length})</option>
                  <option value="everyone">Everyone</option>
                </Select>
                <Select aria-label="Priority" value={prio} onChange={(e) => setPrio(e.target.value as 'normal' | 'urgent')} className="h-8 text-xs">
                  <option value="normal">Normal</option>
                  <option value="urgent">Urgent</option>
                </Select>
              </div>
              {audience === 'team' && !ev.team.length ? <p className="rounded-xl bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-500/10 dark:text-amber-200">This event has no team yet — add people on the Team tab, or send it to everyone.</p> : null}
              <div className="flex justify-end">
                <Button variant="primary" icon="send" loading={busy === 'announce'} disabled={!msg.trim() || (audience === 'team' && !ev.team.length)} onClick={announce}>
                  Send {audience === 'team' ? `to the team (${ev.team.length})` : 'to everyone'}
                </Button>
              </div>
              <div>
                <div className="mb-2 flex items-center gap-2 text-[13px] font-semibold text-foreground"><Icon name="clock" size={14} className="text-brand-600" />Sent about this event<span className="text-xs font-normal text-muted-foreground">{sentAbout.length}</span></div>
                {history.isLoading ? <p className="text-sm text-muted-foreground">Loading…</p> : sentAbout.length ? (
                  <ul className="space-y-1.5">
                    {sentAbout.map((b) => (
                      <li key={b.id} className="rounded-xl border bg-card px-3 py-2 shadow-card">
                        <p className="line-clamp-3 whitespace-pre-line break-words text-sm text-foreground">{b.message}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {b.isActive ? <span className="font-bold text-emerald-600">Live</span> : <span className="font-semibold">Ended{b.endedAt ? ` ${relTime(b.endedAt)}` : ''}</span>}
                          {' · '}{b.createdAt ? relTime(b.createdAt) : ''}{b.sentBy ? ` · by ${b.sentBy}` : ''} · {b.recipients.length ? `to ${b.recipients.length} ${b.recipients.length === 1 ? 'person' : 'people'}` : 'to everyone'}
                        </p>
                      </li>
                    ))}
                  </ul>
                ) : <p className="rounded-xl border border-dashed p-3 text-sm text-muted-foreground">Nothing sent about it yet.</p>}
              </div>
            </> : null}
          </motion.div>
        </AnimatePresence>
      </div>
    </Drawer>
  );
}

/* ------------------------------------------------------------------ add / edit (also opened from the announcement composer) */

/** The add / edit event form in a dialog. `form` null = closed; with `id` it edits, without it creates. */
export function EventFormModal({ form, onClose, onSaved, note }: {
  form: { id?: string; value: PlannerEventInput } | null;
  onClose: () => void;
  onSaved?: (saved: PlannerEvent, created: boolean) => void;
  /** a line above the form (e.g. "Filled in from your announcement — check it") */
  note?: React.ReactNode;
}) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const [value, setValue] = useState<PlannerEventInput>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [step, setStep] = useState<Step>('basics');
  const [dir, setDir] = useState(1);
  React.useEffect(() => { if (form) { setValue({ ...EMPTY_FORM, ...form.value, details: { ...(form.value.details ?? {}) } }); setStep('basics'); } }, [form]);
  const idx = STEPS.findIndex((s) => s.key === step);
  const go = (to: Step) => { setDir(STEPS.findIndex((s) => s.key === to) >= idx ? 1 : -1); setStep(to); };
  const named = !!value.name?.trim();
  const badDates = !!(value.startDate && value.endDate && value.endDate < value.startDate);
  // how much of each step is filled in (for the rail)
  const filledIn: Record<Step, number> = {
    basics: [value.name, value.industry, value.organizer, value.description].filter((v) => (v ?? '').trim()).length,
    where: [value.startDate, value.endDate, value.venue, value.address, value.city, value.country, value.website].filter((v) => (v ?? '').trim()).length,
    team: DETAIL_FIELDS.filter((f) => (value.details?.[f.key] ?? '').trim()).length,
    review: 0,
  };
  // while it fades out after closing, keep showing what it was (Edit vs Add, the form) — not the reset state
  const shown = useWhileOpen(!!form, form);
  const save = async () => {
    // saving: Ctrl+Enter pressed again mid-request would create the event twice
    if (!form || !named || badDates || saving) return;
    setSaving(true);
    try {
      const body = { ...value, name: (value.name ?? '').trim() };
      const saved = form.id ? await api.updatePlannerEvent(form.id, body) : await api.createPlannerEvent(body);
      toast('success', form.id ? 'Event updated' : `${saved.name} added to Events`, form.id ? undefined : 'Add the floor plan and pick the team on the Events tab.');
      void applyChange(qc, 'event_planned');
      onClose();
      onSaved?.(saved, !form.id);
    } catch (e) { toast('error', "Couldn't save the event", errorMessage(e)); }
    finally { setSaving(false); }
  };
  const cur = STEPS[idx]!;
  return (
    <Dialog open={!!form} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="flex h-[min(680px,92vh)] max-w-[calc(100vw-1.5rem)] flex-col gap-0 overflow-hidden rounded-3xl p-0 sm:max-w-4xl md:flex-row"
        onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); void save(); } }}>
        {/* the rail: what this is, the four steps, how far along */}
        <aside className="relative flex shrink-0 flex-col gap-4 overflow-hidden border-b p-5 md:w-[250px] md:border-b-0 md:border-r"
          style={{ background: 'color-mix(in oklab, var(--brand-500) 7%, var(--card))' }}>
          <motion.span aria-hidden className="pointer-events-none absolute -right-10 -top-10 size-40 rounded-full" style={{ background: 'color-mix(in oklab, var(--brand-500) 14%, transparent)' }}
            animate={{ scale: [1, 1.12, 1] }} transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }} />
          <div className="relative flex items-center gap-3">
            <motion.span initial={{ scale: 0.6, rotate: -15, opacity: 0 }} animate={{ scale: 1, rotate: 0, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 16 }}
              className="flex size-11 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-active">
              <Icon name={shown?.id ? 'pencil' : 'calendar'} size={20} />
            </motion.span>
            <div className="min-w-0">
              <DialogTitle className="font-display text-[17px] font-bold leading-6 text-foreground">{shown?.id ? 'Edit event' : 'Add an event'}</DialogTitle>
              <DialogDescription className="text-[11.5px] text-muted-foreground">{shown?.id ? value.name : 'Everything the team needs'}</DialogDescription>
            </div>
          </div>
          <nav className="relative flex gap-1.5 overflow-x-auto md:flex-col md:overflow-visible" aria-label="Steps">
            {STEPS.map((s, i) => {
              const on = s.key === step;
              const done = i < idx || (s.key !== 'review' && filledIn[s.key] > 0 && !on);
              return (
                <button key={s.key} type="button" onClick={() => go(s.key)} aria-current={on ? 'step' : undefined}
                  className={cx('relative flex shrink-0 cursor-pointer items-center gap-2.5 rounded-2xl px-2.5 py-2 text-left transition-colors', on ? 'text-foreground' : 'text-muted-foreground hover:bg-card/60 hover:text-foreground')}>
                  {on ? <motion.span layoutId="es-step" className="absolute inset-0 rounded-2xl bg-card shadow-card" transition={{ type: 'spring', stiffness: 420, damping: 34 }} /> : null}
                  <span className="relative flex size-8 shrink-0 items-center justify-center rounded-xl transition-colors"
                    style={{ background: on || done ? s.tint : `color-mix(in oklab, ${s.tint} 14%, transparent)`, color: on || done ? 'white' : s.tint }}>
                    <AnimatePresence mode="wait" initial={false}>
                      <motion.span key={done && !on ? 'done' : 'icon'} initial={{ scale: 0, rotate: -90 }} animate={{ scale: 1, rotate: 0 }} exit={{ scale: 0 }} transition={{ type: 'spring', stiffness: 500, damping: 22 }} className="flex">
                        {done && !on ? <Icon name="check" size={15} strokeWidth={3} /> : <Icon name={s.icon} size={15} />}
                      </motion.span>
                    </AnimatePresence>
                  </span>
                  <span className="relative hidden min-w-0 md:block">
                    <span className="block text-[13px] font-bold">{s.label}</span>
                    <span className="block truncate text-[11px] font-medium text-muted-foreground">{s.key !== 'review' && filledIn[s.key] ? `${filledIn[s.key]} filled in` : s.sub}</span>
                  </span>
                </button>
              );
            })}
          </nav>
          <div className="relative mt-auto hidden md:block">
            <div className="mb-1.5 flex justify-between text-[11px] font-semibold text-muted-foreground"><span>Step {idx + 1} of {STEPS.length}</span><span>{Math.round(((idx + 1) / STEPS.length) * 100)}%</span></div>
            <div className="h-1.5 overflow-hidden rounded-full bg-card">
              <motion.div className="h-full rounded-full bg-brand-600" animate={{ width: `${((idx + 1) / STEPS.length) * 100}%` }} transition={{ duration: MOTION.base, ease: MOTION.ease }} />
            </div>
            <p className="mt-3 flex items-center gap-1.5 text-[10.5px] text-muted-foreground"><kbd className="rounded border bg-card px-1 font-mono text-[10px]">Ctrl</kbd>+<kbd className="rounded border bg-card px-1 font-mono text-[10px]">↵</kbd> saves</p>
          </div>
        </aside>

        {/* the step */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-3 border-b px-6 py-4 pr-12">
            <AnimatePresence mode="wait" initial={false}>
              <motion.span key={step} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 20 }}
                className="flex size-9 items-center justify-center rounded-xl" style={{ background: `color-mix(in oklab, ${cur.tint} 14%, transparent)`, color: cur.tint }}>
                <Icon name={cur.icon} size={17} />
              </motion.span>
            </AnimatePresence>
            <div className="min-w-0">
              <div className="text-[15px] font-extrabold text-foreground">{cur.label}</div>
              <div className="text-xs text-muted-foreground">{cur.sub}</div>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
            {note && step === 'basics' ? (
              <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="mb-4 flex items-start gap-2 rounded-xl bg-brand-50 p-3 text-sm text-brand-800">
                <Icon name="sparkles" size={15} className="mt-0.5 shrink-0" />{note}
              </motion.div>
            ) : null}
            <AnimatePresence mode="wait" initial={false} custom={dir}>
              <motion.div key={step} custom={dir}
                initial={{ opacity: 0, x: dir * 28 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: dir * -28 }} transition={{ duration: MOTION.base, ease: MOTION.ease }}>
                {shown ? <EventForm value={value} onChange={setValue} step={step} /> : null}
              </motion.div>
            </AnimatePresence>
          </div>
          <div className="flex items-center gap-2 border-t px-6 py-3.5">
            {idx ? (
              <Button variant="ghost" onClick={() => go(STEPS[idx - 1]!.key)}><Icon name="chevronRight" size={15} className="rotate-180" />Back</Button>
            ) : (
              <Button variant="ghost" onClick={onClose}>Cancel</Button>
            )}
            <span className="flex-1 truncate text-xs text-muted-foreground">
              {!named ? 'Give the event a name to save it' : badDates ? <span className="text-rose-600">The end date is before the start date</span> : 'You can save from any step'}
            </span>
            {idx < STEPS.length - 1 ? (
              <>
                <Button onClick={save} loading={saving} disabled={!named || badDates} icon="save" className="hidden sm:inline-flex">{shown?.id ? 'Save' : 'Save now'}</Button>
                <Button variant="primary" onClick={() => go(STEPS[idx + 1]!.key)} disabled={step === 'basics' && !named}>
                  Next<Icon name="chevronRight" size={15} />
                </Button>
              </>
            ) : (
              <motion.span initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 420, damping: 18 }}>
                <Button variant="primary" icon={shown?.id ? 'save' : 'plus'} loading={saving} disabled={!named || badDates} onClick={save}>{shown?.id ? 'Save changes' : 'Add event'}</Button>
              </motion.span>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* ------------------------------------------------------------------ the tab */

export function EventPlanner({ openId, onOpen }: { openId: string | undefined; onOpen: (id: string | undefined) => void }) {
  const { data: events } = useSuspenseQuery(q.plannerEvents());
  const { data: users } = useSuspenseQuery(q.users());
  const discovery = useQuery(q.plannerDiscovery());
  const [text, setText] = useState('');
  const [filter, setFilter] = useState<PlannerStatus | 'active' | 'all'>('active');
  const [form, setForm] = useState<{ id?: string; value: PlannerEventInput } | null>(null);

  const shown = useMemo(() => events.filter((e) => (filter === 'all' || (filter === 'active' ? !['done', 'dismissed'].includes(e.status) : e.status === filter))
    && (!text || [e.name, e.city, e.country, e.venue, e.industry, e.organizer].join(' ').toLowerCase().includes(text.toLowerCase()))), [events, filter, text]);
  const counts = useMemo(() => {
    const m: Record<string, number> = { all: events.length, active: events.filter((e) => !['done', 'dismissed'].includes(e.status)).length };
    events.forEach((e) => { m[e.status] = (m[e.status] ?? 0) + 1; });
    return m;
  }, [events]);
  const open = openId ? events.find((e) => e.id === openId) ?? null : null;

  return (
    <>
      {/* discovery: honest about whether a scraper is connected (nothing is suggested until one is) */}
      <Card className="mb-4 flex flex-wrap items-center gap-3 rounded-2xl border-dashed p-4">
        <span className="flex size-10 items-center justify-center rounded-xl bg-brand-50 text-brand-700"><Icon name="sparkles" size={18} /></span>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-bold text-foreground">Event discovery {discovery.data?.connected ? <span className="text-emerald-600">· connected</span> : <span className="font-semibold text-muted-foreground">· not connected</span>}</div>
          <p className="text-xs text-muted-foreground">{discovery.isLoading ? 'Checking…' : discovery.data?.message ?? (discovery.error ? errorMessage(discovery.error) : '')}</p>
        </div>
        <Button variant="primary" icon="plus" onClick={() => setForm({ value: { ...EMPTY_FORM, details: {} } })}>Add event</Button>
      </Card>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <SearchField value={text} onChange={setText} placeholder="Search events, cities, venues…" className="w-full sm:w-[320px]" />
        <div className="no-scrollbar flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto py-1">
          {(['active', ...STATUS_ORDER, 'all'] as const).map((s) => {
            const n = counts[s] ?? 0;
            if (s !== 'active' && s !== 'all' && !n) return null;
            const on = filter === s;
            const color = s === 'active' || s === 'all' ? 'var(--brand-600)' : STATUS[s].color;
            return (
              <button key={s} onClick={() => setFilter(s)} aria-pressed={on}
                className={cx('inline-flex h-9 shrink-0 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] pl-3 pr-1.5 text-[13px] font-bold transition-colors', on ? 'shadow-card' : 'border-border bg-card hover:border-brand-200')}
                style={on ? { borderColor: color, background: `color-mix(in oklab, ${color} 10%, var(--card))` } : undefined}>
                {s === 'active' ? 'Upcoming & planned' : s === 'all' ? 'All' : STATUS[s].label}
                <span className="min-w-6 rounded-full px-1.5 text-center text-[12px] font-black leading-[22px]" style={on ? { background: color, color: 'white' } : { background: 'var(--secondary)', color: 'var(--body)' }}>{n}</span>
              </button>
            );
          })}
        </div>
      </div>

      {shown.length ? (
        <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
          <AnimatePresence initial>
            {shown.map((e, i) => {
              const w = when(e);
              const floor = e.files.filter((f) => f.floorPlan).length;
              return (
                <motion.button key={e.id} layout="position" onClick={() => onOpen(e.id)}
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 12) * 0.04 }}
                  whileHover={{ y: -3 }} whileTap={{ scale: 0.985 }}
                  className={cx('group relative flex cursor-pointer flex-col gap-3 overflow-hidden rounded-2xl border bg-card p-4 text-left shadow-card transition-[border-color,box-shadow] hover:border-brand-200 hover:shadow-active',
                    openId === e.id && 'border-primary ring-2 ring-primary/30')}>
                  <span className="absolute inset-x-0 top-0 h-[3px]" style={{ background: STATUS[e.status]?.color }} aria-hidden />
                  <div className="flex items-start gap-3">
                    <span className="flex size-12 shrink-0 flex-col items-center justify-center rounded-xl bg-brand-50 text-brand-700 transition-transform duration-300 group-hover:-rotate-3 group-hover:scale-105">
                      {e.startDate ? <>
                        <span className="text-[9px] font-bold uppercase leading-none tracking-[0.08em]">{fromKey(e.startDate).toLocaleDateString([], { month: 'short' })}</span>
                        <span className="text-lg font-black leading-5">{fromKey(e.startDate).getDate()}</span>
                      </> : <Icon name="calendar" size={18} />}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="break-words text-[15px] font-extrabold leading-5 text-foreground">{e.name}</span>
                        <StatusPill status={e.status} small />
                      </div>
                      <div className="mt-0.5 text-[12.5px] text-body">{fmtRange(e.startDate, e.endDate)}{w ? <span className="font-bold" style={{ color: w.tone }}> · {w.text}</span> : null}</div>
                      <div className="truncate text-[12.5px] text-muted-foreground">{place(e) || 'Venue not set'}</div>
                    </div>
                  </div>
                  <div className="mt-auto flex items-center gap-2 border-t pt-3">
                    <TeamStack team={e.team} />
                    <span className="ml-auto flex items-center gap-2 text-[11px] font-semibold text-muted-foreground">
                      <span className={cx('inline-flex items-center gap-1', floor ? 'text-emerald-600' : '')} title={floor ? 'Floor plan attached' : 'No floor plan yet'}><Icon name="columns" size={12} />{floor ? 'Floor plan' : 'No plan'}</span>
                      {e.contactsMet ? <span className="inline-flex items-center gap-1" title="Contacts met there"><Icon name="contacts" size={12} />{e.contactsMet}</span> : null}
                    </span>
                  </div>
                </motion.button>
              );
            })}
          </AnimatePresence>
        </div>
      ) : (
        <Card className="rounded-2xl">
          <EmptyState icon="calendar" title={events.length ? 'No events match' : 'No events yet'}
            description={events.length ? 'Try another status or search.' : 'Add the events your team is going to — dates, venue, floor plan, the team — and announce them to the reps.'}
            action={<Button variant="primary" icon="plus" onClick={() => setForm({ value: { ...EMPTY_FORM, details: {} } })}>Add event</Button>} />
        </Card>
      )}

      <EventPanel ev={open} users={users} onClose={() => onOpen(undefined)} onEdit={(e) => setForm({ id: e.id, value: toInput(e) })} />

      <EventFormModal form={form} onClose={() => setForm(null)} onSaved={(saved, created) => { if (created) onOpen(saved.id); }} />
    </>
  );
}
