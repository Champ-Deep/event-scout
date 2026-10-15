import React from 'react';
import { Link } from '@tanstack/react-router';
import { motion } from 'motion/react';
import type { ActivityItem, ActivityType } from '../../lib/types';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { cx, ink, relTime, fmtDateTime } from '../../ui/kit';

/** Every activity type: its icon, tinted tile, solid colour (chips, calendar) and plural label. */
export const ACTIVITY_STYLE: Record<ActivityType, { icon: IconName; tone: string; color: string; label: string; one: string }> = {
  contact_added: { icon: 'userPlus', tone: 'bg-brand-50 text-brand-700', color: 'var(--brand-600)', label: 'Contacts added', one: 'Contact added' },
  contact_scored: { icon: 'flame', tone: 'bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-300', color: '#f59e0b', label: 'AI scores', one: 'AI score' },
  pipeline: { icon: 'pipeline', tone: 'bg-sky-50 text-sky-600 dark:bg-sky-500/10 dark:text-sky-300', color: '#0ea5e9', label: 'Pipelines', one: 'Pipeline' },
  broadcast: { icon: 'megaphone', tone: 'bg-fuchsia-50 text-fuchsia-600 dark:bg-fuchsia-500/10 dark:text-fuchsia-300', color: '#d946ef', label: 'Announcements', one: 'Announcement' },
  file: { icon: 'file', tone: 'bg-teal-50 text-teal-600 dark:bg-teal-500/10 dark:text-teal-300', color: '#14b8a6', label: 'Files', one: 'File' },
  contact_deleted: { icon: 'trash', tone: 'bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-300', color: '#f43f5e', label: 'Deleted', one: 'Deleted' },
  contact_merged: { icon: 'merge', tone: 'bg-brand-50 text-brand-700', color: '#8b5cf6', label: 'Merged', one: 'Merged' },
  user_joined: { icon: 'user', tone: 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-300', color: '#10b981', label: 'New accounts', one: 'New account' },
};

/** "Today", "Yesterday" or a long date, for grouping activity by day. */
export function dayLabel(iso: string) {
  const d = new Date(iso);
  const today = new Date();
  const y = new Date(); y.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === y.toDateString()) return 'Yesterday';
  return d.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' });
}

const str = (v: unknown) => (typeof v === 'string' && v ? v : null);

/**
 * Where an activity item leads, from the ids the server sends — or null when there's nothing real to
 * open (e.g. a pipeline whose contact was deleted, or a contact no longer in Recently deleted).
 */
export function activityLink(a: ActivityItem): { to: string; params?: Record<string, string>; search?: Record<string, string>; label: string } | null {
  const contactId = str(a.meta?.contactId);
  switch (a.type) {
    case 'contact_added': case 'contact_scored':
      return contactId ? { to: '/contacts/$id', params: { id: contactId }, label: 'Open contact', ...(a.type === 'contact_scored' ? { search: { tab: 'score' } } : {}) } : null;
    case 'pipeline':
      return contactId ? { to: '/contacts/$id', params: { id: contactId }, search: { tab: 'pipeline' }, label: 'Open pipeline' } : null;
    case 'contact_deleted': case 'contact_merged': {
      const id = str(a.meta?.deletedId);
      return id ? { to: '/deleted', search: { view: id }, label: 'View in Recently deleted' } : null;
    }
    case 'broadcast': return { to: '/broadcasts', label: 'Open announcements' };
    case 'file': return { to: '/files', label: 'Open files' };
    case 'user_joined': {
      const q = str(a.meta?.email) ?? a.actor;
      return { to: '/seats', ...(q ? { search: { q } } : {}), label: 'Open in Team & seats' };
    }
    default: return null;
  }
}

/** Initials avatar for the person behind an item (the server sends their name, not a photo). */
function ActorChip({ name }: { name: string }) {
  const initials = name.split(/\s+/).filter(Boolean).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary py-0.5 pl-0.5 pr-2 text-[11px] font-semibold text-body">
      <span className="flex size-[18px] items-center justify-center rounded-full bg-card text-[8.5px] font-black text-brand-700 ring-1 ring-border">{initials || '•'}</span>
      {name}
    </span>
  );
}

/**
 * The activity timeline: a tinted icon tile per item on a vertical rail, title, details, who and when.
 * Items that lead somewhere real are links (hover lifts them and shows where they go).
 */
export function ActivityFeed({ items, compact, startIndex = 0 }: { items: ActivityItem[]; compact?: boolean; startIndex?: number }) {
  return (
    <ol className="relative">
      {items.map((a, i) => {
        const s = ACTIVITY_STYLE[a.type] || ACTIVITY_STYLE.contact_added;
        const link = activityLink(a);
        const k = startIndex + i;
        const body = (
          <>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                <p className="min-w-0 break-words text-[13.5px] font-semibold leading-5 text-foreground">{a.title}</p>
                <span className="rounded-full px-1.5 text-[9.5px] font-black uppercase leading-4 tracking-[0.06em]" style={{ color: ink(s.color), background: `color-mix(in oklab, ${s.color} 12%, transparent)` }}>{s.one}</span>
              </div>
              {a.subtitle && !compact ? <p className="mt-0.5 line-clamp-3 whitespace-pre-line break-words text-[12.5px] leading-[18px] text-body">{a.subtitle}</p> : null}
              {!compact && a.actor && a.actor !== 'Someone' ? <div className="mt-1.5"><ActorChip name={a.actor} /></div> : null}
            </div>
            <div className="flex shrink-0 flex-col items-end gap-1">
              <time className="whitespace-nowrap text-[11.5px] font-semibold text-muted-foreground" dateTime={a.timestamp} title={fmtDateTime(a.timestamp)}>{relTime(a.timestamp)}</time>
              <span className="whitespace-nowrap text-[10.5px] tabular-nums text-muted-foreground">{new Date(a.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</span>
              {link ? <Icon name="chevronRight" size={14} className="mt-0.5 text-muted-foreground/0 transition-all duration-200 group-hover:translate-x-0.5 group-hover:text-brand-600" /> : null}
            </div>
          </>
        );
        const card = cx('group flex min-w-0 flex-1 items-start gap-3 rounded-2xl border bg-card px-3.5 py-3 shadow-card transition-[border-color,box-shadow,transform] duration-200',
          link && 'cursor-pointer hover:-translate-y-0.5 hover:border-brand-200 hover:shadow-active');
        return (
          <motion.li key={a.id} className="relative flex gap-3 pb-2.5 last:pb-0"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(k, 16) * 0.03 }}>
            {/* the rail between icons */}
            {i < items.length - 1 ? <span className="absolute bottom-0 left-[17px] top-10 w-[2px] rounded-full bg-border/70" aria-hidden /> : null}
            <motion.span initial={{ scale: 0.4, rotate: -20 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.05 + Math.min(k, 16) * 0.03 }}
              className={cx('relative z-[1] mt-1.5 flex size-9 shrink-0 items-center justify-center rounded-xl ring-4 ring-background', s.tone)}>
              <Icon name={s.icon} size={16} />
            </motion.span>
            {link ? (
              <Link to={link.to as never} params={link.params as never} search={(link.search ?? {}) as never} className={card} title={link.label} aria-label={`${a.title} — ${link.label}`}>{body}</Link>
            ) : (
              <div className={card}>{body}</div>
            )}
          </motion.li>
        );
      })}
    </ol>
  );
}
