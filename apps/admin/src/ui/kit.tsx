// The console's shared primitives, built on shadcn/ui (src/components/ui). Pages import from here so
// they keep one small API (Button variant/icon/loading, Modal, Drawer, toast/confirm…) while the
// look, focus rings, motion and dark mode come from shadcn. Formatting helpers live here too.
import React, { createContext, useCallback, useContext, useState } from 'react';
import { toast as sonner } from 'sonner';
import { motion } from 'motion/react';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button as ShadButton, type buttonVariants } from '@/components/ui/button';
import { Badge as ShadBadge } from '@/components/ui/badge';
import { Card as ShadCard } from '@/components/ui/card';
import { Input as ShadInput } from '@/components/ui/input';
import { Textarea as ShadTextarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Checkbox as ShadCheckbox } from '@/components/ui/checkbox';
import { Skeleton as ShadSkeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table as ShadTable, TableCell, TableHead } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Dialog as DialogPrimitive } from 'radix-ui';
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import type { VariantProps } from 'class-variance-authority';
import { Icon, type IconName } from './icons';
import { LiveIcon } from './live-icon';
import { levelFrom } from '../lib/levels';

/** Class helper (shadcn's `cn`: conditional + Tailwind-conflict aware). */
export const cx = cn;

/* Readable colour helpers (any CSS colour or var). Bright tints (amber, green, sky, pink…) are too light to be
   text on a pale tint, and too light under white text; these keep WCAG AA in light AND dark mode:
     ink(c)  — c pulled toward the theme's text colour: darker in light mode, lighter in dark mode (text / icons on tints)
     deep(c) — a darker shade of c for a solid fill under white text */
/** A link that can only open a web page: http(s) kept, "northwind.com" → https://northwind.com, anything else
 *  (javascript:, data:, vbscript:… — even with hidden whitespace) → undefined, so no link is rendered. */
export function safeHref(raw?: string | null): string | undefined {
  const v = (raw || '').replace(/[\u0000-\u001f\u007f\s]+/g, '');
  if (!v) return undefined;
  if (/^https?:\/\//i.test(v)) return v;
  if (/^[a-z][a-z0-9+.-]*:/i.test(v) && !/^[^:/]+\.[^:/]+:\d/.test(v)) return undefined; // another scheme (host:port is fine)
  return `https://${v}`;
}

export const ink = (c: string) => `color-mix(in oklab, ${c} 50%, var(--foreground))`;
export const deep = (c: string) => `color-mix(in oklab, ${c} 64%, #000)`;

/* ------------------------------------------------------------------ formatting */

/** Backend times without a zone (SQLite) are UTC — read them as UTC, not local time. */
export function parseTime(iso?: string | null): Date | null {
  if (!iso) return null;
  const s = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) || !/\d{2}:\d{2}/.test(iso) ? iso : `${iso.replace(' ', 'T')}Z`;
  const d = new Date(s);
  return isFinite(d.getTime()) ? d : null;
}

/**
 * Was the record edited after it was added? created_at and updated_at are stamped separately on
 * insert (microseconds to seconds apart), so "updated" within a minute of "added" counts as not edited.
 */
export function wasEdited(createdAt?: string | null, updatedAt?: string | null): boolean {
  const u = parseTime(updatedAt)?.getTime();
  if (u == null) return false;
  const c = parseTime(createdAt)?.getTime();
  return c == null || Math.abs(u - c) >= 60_000;
}

export function relTime(iso?: string | null): string {
  if (!iso) return '—';
  const t = parseTime(iso)?.getTime() ?? NaN;
  if (!isFinite(t)) return '—';
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 45) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d} d ago`;
  return new Date(t).toLocaleDateString([], { day: 'numeric', month: 'short', year: d > 300 ? 'numeric' : undefined });
}

export function fmtDateTime(iso?: string | null) {
  const d = parseTime(iso);
  return d ? d.toLocaleString([], { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
}

export function fmtBytes(n?: number | null) {
  if (!n) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export const fmtNum = (n?: number | null) => (n == null ? '—' : n.toLocaleString());

export function contactName(c: { firstName?: string | null; lastName?: string | null }) {
  return [c.firstName, c.lastName].filter(Boolean).join(' ').trim() || 'Unnamed contact';
}

export function errorMessage(e: unknown, fallback = 'Something went wrong.') {
  return e instanceof Error && e.message ? e.message : fallback;
}

/* ------------------------------------------------------------------ levels (same as the app) */

// The rule, labels and colours live in lib/levels (one module); re-exported here for the pages.
export { LEVELS, levelOf, type Level } from '../lib/levels';

/** `level`: the server's level for this score (contact.level), when it sent one. */
export function ScoreBadge({ score, level, className }: { score: number | null | undefined; level?: string | null | undefined; className?: string }) {
  const l = levelFrom(score, level);
  // the app's score chip: a pill with the number in bold, tinted by level
  if (!l) return <span className={cn('inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground', className)}>Not scored</span>;
  return (
    <span className={cn('inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-bold tabular-nums', l.badge, className)}>
      {score}
      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.06em]">{l.label}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ primitives */

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'subtle';
type ShadVariant = VariantProps<typeof buttonVariants>['variant'];
// Buttons: squared 8px corners, semibold labels. Primary is the solid --fill (brand violet with white text, like the app).
const VARIANT: Record<Variant, { variant: ShadVariant; className?: string }> = {
  primary: { variant: 'default', className: 'bg-fill text-fill-foreground shadow-none hover:bg-fill-hover' },
  secondary: { variant: 'outline', className: 'border-input bg-card text-foreground shadow-none hover:bg-secondary dark:bg-card' },
  ghost: { variant: 'ghost', className: 'text-body' },
  danger: { variant: 'destructive' },
  subtle: { variant: 'secondary', className: 'bg-accent text-accent-foreground hover:bg-brand-200 dark:hover:bg-brand-200' },
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: 'sm' | 'md';
  icon?: IconName;
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon, loading, className, children, disabled, ...rest }, ref,
) {
  const v = VARIANT[variant];
  return (
    <ShadButton ref={ref} variant={v.variant} size={size === 'sm' ? 'sm' : 'default'} disabled={disabled || loading}
      className={cn('es-btn cursor-pointer rounded-lg font-semibold transition-[transform,background-color,color,box-shadow] hover:-translate-y-px active:translate-y-0 active:scale-[0.97]', size === 'sm' ? 'px-3 text-xs' : 'px-4', v.className, className)} {...rest}>
      {loading ? <Loader2 className="animate-spin" aria-label="Loading" /> : icon ? <Icon name={icon} size={size === 'sm' ? 14 : 16} /> : null}
      {children}
    </ShadButton>
  );
});

export function IconButton({ icon, label, className, size = 'sm', ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { icon: IconName; label: string; size?: 'xs' | 'sm' | 'md' }) {
  return (
    <ShadButton variant="ghost" size={size === 'xs' ? 'icon-xs' : size === 'md' ? 'icon' : 'icon-sm'} aria-label={label} title={label}
      className={cn('es-btn cursor-pointer rounded-lg text-muted-foreground transition-[transform,color,background-color] hover:text-foreground active:scale-90', className)} {...rest}>
      <Icon name={icon} size={size === 'xs' ? 13 : 16} />
    </ShadButton>
  );
}

export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={cn('animate-spin', className)} aria-label="Loading" />;
}

export function Skeleton({ className }: { className?: string }) {
  // a light sweep over the placeholder instead of a plain blink
  return <ShadSkeleton className={cn('es-shimmer', className)} />;
}

/** A plain surface (shadcn Card without its default padding/gap, so pages lay out their own sections). Rises in. */
export function Card({ children, className, ...rest }: React.HTMLAttributes<HTMLDivElement>) {
  return <ShadCard className={cn('es-card-in gap-0 rounded-xl py-0 shadow-card', className)} {...rest}>{children}</ShadCard>;
}

export function CardHeader({ title, subtitle, icon, actions, className }: { title: React.ReactNode; subtitle?: React.ReactNode; icon?: IconName; actions?: React.ReactNode; className?: string }) {
  const [hover, setHover] = React.useState(false); // the live icon plays while the header is hovered
  return (
    <div className={cn('group flex items-start justify-between gap-3 border-b px-5 py-3.5', className)} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <div className="flex min-w-0 items-start gap-2.5">
        {icon ? (
          <span className="es-pop-in mt-px flex size-7 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-700">
            <LiveIcon name={icon} size={14} play={hover} />
          </span>
        ) : null}
        <div className="min-w-0">
          <h3 className="text-[15px] font-semibold leading-6 text-foreground">{title}</h3>
          {subtitle ? <p className="text-xs text-muted-foreground">{subtitle}</p> : null}
        </div>
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}

/** Each page's own icon and colour in its header (pages can pass `icon` / `tint` to override). */
const PAGE_ICONS: Record<string, { icon: IconName; tint: string }> = {
  'Activity': { icon: 'activity', tint: '#0ea5e9' },
  'Recently deleted': { icon: 'trash', tint: '#f43f5e' },
  'Duplicates': { icon: 'duplicate', tint: '#f59e0b' },
  'Team & seats': { icon: 'users', tint: '#10b981' },
  'AI pipelines': { icon: 'pipeline', tint: 'var(--brand-600)' },
  'Team files': { icon: 'file', tint: '#14b8a6' },
  'Exports': { icon: 'download', tint: '#6366f1' },
  'Lists': { icon: 'list', tint: '#d946ef' },
  'Events': { icon: 'store', tint: '#0ea5e9' },
  'System': { icon: 'server', tint: '#64748b' },
  'Sales profile': { icon: 'target', tint: 'var(--brand-600)' },
};

export function PageHeader({ title, description, actions, eyebrow, compact, icon, tint, live = true }: {
  title: string; description?: React.ReactNode; actions?: React.ReactNode; eyebrow?: React.ReactNode; compact?: boolean; icon?: IconName; tint?: string;
  /** the header icon plays its looping live animation (ui/live-icon); on for every page unless false */
  live?: boolean;
}) {
  const look = PAGE_ICONS[title];
  const ic = icon ?? look?.icon;
  const color = tint ?? look?.tint ?? 'var(--brand-600)';
  return (
    // a hero tinted with the page's own colour, two bubbles drifting behind it (styles/tokens.css .es-bubble)
    <div className={cn('es-in group relative isolate flex flex-wrap items-end justify-between overflow-hidden rounded-3xl border shadow-card',
      compact ? 'mb-3 gap-3 p-4' : 'mb-5 gap-3 p-5 sm:p-6')}
      style={{ background: `linear-gradient(135deg, color-mix(in oklab, ${color} 12%, var(--card)), var(--card) 62%)` }}>
      <span aria-hidden className="es-bubble pointer-events-none absolute -right-16 -top-24 -z-10 size-64 rounded-full" style={{ background: `color-mix(in oklab, ${color} 13%, transparent)` }} />
      <span aria-hidden className="es-bubble-2 pointer-events-none absolute -bottom-28 right-44 -z-10 size-52 rounded-full" style={{ background: `color-mix(in oklab, ${color} 8%, transparent)` }} />
      <span aria-hidden className="es-bubble-2 pointer-events-none absolute -left-10 -top-16 -z-10 size-32 rounded-full" style={{ background: `color-mix(in oklab, ${color} 6%, transparent)` }} />
      <div className="flex min-w-0 items-center gap-3.5">
        {ic ? (
          <span className={cn('es-pop-in es-wiggle-hover flex shrink-0 items-center justify-center rounded-2xl shadow-card', compact ? 'size-10' : 'size-12')}
            style={{ background: `color-mix(in oklab, ${color} 14%, var(--card))`, color }}>
            {live ? <LiveIcon name={ic} size={compact ? 18 : 22} /> : <Icon name={ic} size={compact ? 18 : 22} />}
          </span>
        ) : null}
        <div className="min-w-0">
          {eyebrow ? <p className={cn('es-in es-eyebrow es-eyebrow-mark text-muted-foreground', compact ? 'mb-1' : 'mb-2')}>{eyebrow}</p> : null}
          <h1 className={cn('es-in font-display font-bold tracking-[-0.025em] text-foreground', compact ? 'text-[24px] leading-7' : 'text-[28px] leading-8')} style={{ ['--es-d' as string]: '60ms' }}>{title}</h1>
          {description ? <p className={cn('es-in max-w-2xl text-sm text-body', compact ? 'mt-0.5' : 'mt-1')} style={{ ['--es-d' as string]: '120ms' }}>{description}</p> : null}
        </div>
      </div>
      {actions ? <div className="es-in flex flex-wrap items-center gap-2" style={{ ['--es-d' as string]: '180ms' }}>{actions}</div> : null}
    </div>
  );
}

type Tone = 'gray' | 'violet' | 'green' | 'amber' | 'red' | 'sky';
// The app's semantic tints (theme.ts semantic.* / brand-50), each with a soft border for banners.
const TONES: Record<Tone, string> = {
  gray: 'bg-secondary text-body border-border',
  violet: 'bg-brand-50 text-brand-700 border-brand-100',
  green: 'bg-success-bg text-success border-success/20',
  amber: 'bg-warning-bg text-warning border-warning-border',
  red: 'bg-danger-bg text-destructive border-danger-border/40',
  sky: 'bg-info-bg text-info border-info-border',
};
/** The app's Badge: small uppercase label on a tint, no outline. */
export function Badge({ tone = 'gray', children, className, icon }: { tone?: Tone; children: React.ReactNode; className?: string; icon?: IconName }) {
  return (
    <ShadBadge variant="outline" className={cn('es-badge-in rounded-md border-transparent px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-[0.06em]', TONES[tone], className)}>
      {icon ? <Icon name={icon} size={12} /> : null}
      {children}
    </ShadBadge>
  );
}

export function Stat({ label, value, hint, icon, tone = 'violet', onClick, loading }: {
  label: string; value: React.ReactNode; hint?: React.ReactNode; icon: IconName; tone?: Tone; onClick?: () => void; loading?: boolean;
}) {
  const Comp: any = onClick ? 'button' : 'div';
  const [hover, setHover] = React.useState(false); // the live icon plays while the tile is hovered
  return (
    <Comp onClick={onClick} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      className={cn('es-card-in group flex w-full items-start gap-3 rounded-xl border bg-card p-4 text-left shadow-card transition-[transform,border-color,box-shadow] duration-200 hover:-translate-y-0.5',
        onClick && 'cursor-pointer hover:border-foreground/25 hover:shadow-active active:scale-[0.99]')}>
      <span className={cn('es-pop-in flex size-9 shrink-0 items-center justify-center rounded-xl', TONES[tone])}>
        <LiveIcon name={icon} size={16} play={hover} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">{label}</div>
        {loading ? <Skeleton className="mt-2 h-7 w-16" /> : <div className="es-num mt-1 text-[28px] font-semibold leading-8 text-foreground">{value}</div>}
        {hint ? <div className="mt-1 truncate text-xs text-muted-foreground">{hint}</div> : null}
      </div>
      {onClick ? <Icon name="chevronRight" size={16} className="mt-0.5 text-muted-foreground/50 transition group-hover:translate-x-0.5 group-hover:text-foreground" /> : null}
    </Comp>
  );
}

export function EmptyState({ icon = 'info', title, description, action, className }: { icon?: IconName; title: string; description?: React.ReactNode; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn('es-in flex flex-col items-center justify-center px-6 py-12 text-center', className)}>
      <span className="es-pop-in mb-3">
        <span className="es-float flex size-14 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 shadow-card">
          <LiveIcon name={icon} size={22} />
        </span>
      </span>
      <h3 className="es-in text-base font-semibold text-foreground" style={{ ['--es-d' as string]: '80ms' }}>{title}</h3>
      {description ? <p className="es-in mt-1 max-w-sm text-sm text-body" style={{ ['--es-d' as string]: '140ms' }}>{description}</p> : null}
      {action ? <div className="es-in mt-4" style={{ ['--es-d' as string]: '200ms' }}>{action}</div> : null}
    </div>
  );
}

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  return (
    <Alert variant="destructive" className={cn('es-shake border-destructive/30 bg-destructive/5', className)}>
      <Icon name="alert" size={16} />
      <AlertTitle>Couldn't load this</AlertTitle>
      <AlertDescription className="flex w-full flex-wrap items-center justify-between gap-2">
        <span>{errorMessage(error)}</span>
        {onRetry ? <Button size="sm" variant="secondary" icon="refresh" onClick={onRetry}>Retry</Button> : null}
      </AlertDescription>
    </Alert>
  );
}

export function Notice({ tone = 'gray', icon = 'info', children }: { tone?: Tone; icon?: IconName; children: React.ReactNode }) {
  return (
    <Alert className={cn('es-in', TONES[tone], 'rounded-xl has-[>svg]:gap-x-2.5')}>
      <Icon name={icon} size={16} />
      <AlertDescription className="text-current [&_p]:text-current">{children}</AlertDescription>
    </Alert>
  );
}

/* ------------------------------------------------------------------ forms */

export function Field({ label, hint, error, children, className }: { label: string; hint?: React.ReactNode; error?: string | null; children: React.ReactNode; className?: string }) {
  return (
    <label className={cn('block', className)}>
      <Label asChild><span className="mb-1.5 block text-xs font-semibold text-foreground">{label}</span></Label>
      {children}
      {error ? <span className="mt-1 block text-xs text-destructive">{error}</span>
        : hint ? <span className="mt-1 block text-xs text-muted-foreground">{hint}</span> : null}
    </label>
  );
}

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...rest }, ref) {
  return <ShadInput ref={ref} className={cn('rounded-lg bg-card', className)} {...rest} />;
});

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  return <ShadTextarea ref={ref} className={cn('rounded-lg bg-card leading-relaxed', className)} {...rest} />;
});

/** Native <select> (pages pass <option> children) styled like shadcn's SelectTrigger. */
export function Select({ className, children, ...rest }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn('h-9 cursor-pointer rounded-lg border border-input bg-card px-3 pr-8 text-sm shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-input/30',
      !/(^|\s)w-/.test(className || '') && 'w-full', className)} {...rest}>{children}</select>
  );
}

export function SearchInput({ value, onChange, placeholder, className, inputRef }: { value: string; onChange: (v: string) => void; placeholder?: string; className?: string; inputRef?: React.Ref<HTMLInputElement> }) {
  return (
    <div className={cn('relative', className)}>
      <Icon name="search" size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
      <Input ref={inputRef} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className="pl-9 pr-8" />
      {value ? (
        <button onClick={() => onChange('')} aria-label="Clear search" className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-muted-foreground hover:text-foreground cursor-pointer">
          <Icon name="x" size={14} />
        </button>
      ) : null}
    </div>
  );
}

export function Checkbox({ checked, onChange, label, indeterminate, className }: { checked: boolean; onChange: (v: boolean) => void; label?: string; indeterminate?: boolean; className?: string }) {
  return (
    <ShadCheckbox checked={indeterminate ? 'indeterminate' : checked} aria-label={label} onCheckedChange={(v) => onChange(v === true)}
      onClick={(e) => e.stopPropagation()} className={cn('cursor-pointer', className)} />
  );
}

/** Segmented control (shadcn Tabs used as a value picker; optional counts). */
export function Segmented<T extends string>({ value, onChange, options, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode; count?: number | undefined }[]; className?: string }) {
  return (
    <Tabs value={value} onValueChange={(v) => onChange(v as T)} className={className}>
      <TabsList className="h-8 rounded-lg">
        {options.map((o) => (
          <TabsTrigger key={o.value} value={o.value} className="cursor-pointer rounded-md px-3 text-xs font-semibold">
            {o.label}
            {o.count != null ? <span className="rounded bg-muted px-1.5 font-mono text-[10px] tabular-nums text-muted-foreground group-data-[state=active]:bg-signal group-data-[state=active]:text-signal-foreground">{o.count}</span> : null}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}

/* ------------------------------------------------------------------ table */

export function Table({ children, className }: { children: React.ReactNode; className?: string }) {
  // es-table: rows cascade in and warm up on hover (styles/tokens.css)
  return <ShadTable className={cn('es-table text-left', className)}>{children}</ShadTable>;
}
export function Th({ children, className, ...rest }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return <TableHead className={cn('sticky top-0 z-[1] h-auto border-b bg-secondary px-3 py-2.5 font-mono text-[10.5px] font-medium uppercase tracking-[0.07em] text-muted-foreground first:pl-4 last:pr-4', className)} {...rest}>{children}</TableHead>;
}
export function Td({ children, className, ...rest }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <TableCell className={cn('whitespace-normal border-b px-3 py-3 align-middle first:pl-4 last:pr-4', className)} {...rest}>{children}</TableCell>;
}

/* ------------------------------------------------------------------ dialogs */

/**
 * What a dialog showed the last time it was open. Callers usually clear the state that drives the
 * dialog (the row being edited, the form) as they close it, but the dialog stays on screen while it
 * animates out — rendering the cleared state made the title flip ("Edit…" → "Add…"), the body go
 * blank or the buttons change mid-fade. Closed, Modal and Drawer keep showing this snapshot.
 */
export function useWhileOpen<T>(open: boolean, value: T): T {
  const last = React.useRef(value);
  if (open) last.current = value;
  return open ? value : last.current;
}

type ModalProps = {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: React.ReactNode; children: React.ReactNode; footer?: React.ReactNode; size?: 'sm' | 'md' | 'lg' | 'xl';
};

export function Modal(props: ModalProps) {
  const { open, onOpenChange } = props;
  const { title, description, children, footer, size = 'md' } = useWhileOpen(open, props);
  const w = { sm: 'sm:max-w-sm', md: 'sm:max-w-lg', lg: 'sm:max-w-2xl', xl: 'sm:max-w-4xl' }[size];
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className={cn('flex max-h-[88vh] flex-col gap-0 rounded-xl p-0', w)}>
        <DialogHeader className="border-b px-6 py-4 text-left">
          <DialogTitle className="font-display text-lg font-semibold">{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : <DialogDescription className="sr-only">{title}</DialogDescription>}
        </DialogHeader>
        {/* layoutScroll (here and in the Drawer bodies): animated rows inside are measured relative to
            this scroll area, so a re-render after scrolling (typing, a refetch) doesn't make them jump */}
        <motion.div layoutScroll className="flex-1 overflow-y-auto px-6 py-5">{children}</motion.div>
        {footer ? <DialogFooter className="border-t px-6 py-3.5">{footer}</DialogFooter> : null}
      </DialogContent>
    </Dialog>
  );
}

type DrawerProps = {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; subtitle?: React.ReactNode; header?: React.ReactNode; children: React.ReactNode; footer?: React.ReactNode; width?: string;
  /** when this changes (e.g. another tab or contact), the body jumps back to the top instead of keeping the old position */
  scrollKey?: string | undefined;
  /** extra buttons in the top-right corner, left of the close button (e.g. "Open full page") */
  headerActions?: React.ReactNode;
  /** 'side' = panel from the right; 'center' = a large centred dialog over a frosted-glass backdrop */
  mode?: 'side' | 'center';
};

export function Drawer(props: DrawerProps) {
  const { open, onOpenChange } = props;
  // closing keeps the last open content on screen while the panel slides / fades out (useWhileOpen)
  const { title, subtitle, header, children, footer, width = 'sm:max-w-xl', headerActions, mode = 'side', scrollKey } = useWhileOpen(open, props);
  const bodyRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => { bodyRef.current?.scrollTo({ top: 0 }); }, [scrollKey]);
  if (mode === 'center') {
    return (
      <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="es-glass fixed inset-0 z-50 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:duration-300" />
          <DialogPrimitive.Content
            className="fixed left-1/2 top-1/2 z-50 flex h-[min(90dvh,980px)] w-[min(1120px,calc(100vw-1.5rem))] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-3xl border bg-card text-card-foreground shadow-[0_32px_80px_-24px_rgb(30_16_70/0.45)] outline-none data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=open]:slide-in-from-bottom-3 data-[state=open]:duration-300">
            {headerActions ? <div className="absolute right-12 top-3 z-10 flex items-center gap-1">{headerActions}</div> : null}
            <DialogPrimitive.Close aria-label="Close"
              className="absolute right-3 top-3 z-10 flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg bg-card/80 text-muted-foreground ring-1 ring-inset ring-border transition-[color,transform] hover:text-foreground active:scale-90">
              <Icon name="x" size={16} />
            </DialogPrimitive.Close>
            <div className="border-b px-6 py-4 pr-12">
              {header ?? (
                <>
                  <DialogPrimitive.Title className="font-display text-lg font-semibold">{title}</DialogPrimitive.Title>
                  {subtitle ? <DialogPrimitive.Description className="text-sm text-muted-foreground">{subtitle}</DialogPrimitive.Description> : null}
                </>
              )}
              {header ? <DialogPrimitive.Title className="sr-only">{title}</DialogPrimitive.Title> : null}
              <DialogPrimitive.Description className="sr-only">{title}</DialogPrimitive.Description>
            </div>
            {/* one scroll area: stays inside the dialog (no page scroll behind), stable gutter, thin scrollbar */}
            {/* container-type: size lets children size themselves to the visible body height with cqh units */}
            <motion.div ref={bodyRef} layoutScroll className="es-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain [container-type:size] [scrollbar-gutter:stable]">{children}</motion.div>
            {footer ? <div className="flex items-center justify-between gap-2 border-t px-6 py-3.5">{footer}</div> : null}
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    );
  }
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className={cn('w-full gap-0 p-0 transition-[max-width] data-[state=open]:duration-300', width.replace(/^max-w-/, 'sm:max-w-'))}>
        {headerActions ? <div className="absolute right-11 top-3 z-10 flex items-center gap-1">{headerActions}</div> : null}
        <SheetHeader className="border-b px-6 py-4 pr-12">
          {header ?? (
            <>
              <SheetTitle className="font-display text-lg font-semibold">{title}</SheetTitle>
              {subtitle ? <SheetDescription>{subtitle}</SheetDescription> : null}
            </>
          )}
          {header ? <SheetTitle className="sr-only">{title}</SheetTitle> : null}
          <SheetDescription className="sr-only">{title}</SheetDescription>
        </SheetHeader>
        <motion.div layoutScroll className="flex-1 overflow-y-auto">{children}</motion.div>
        {footer ? <SheetFooter className="mt-0 flex-row items-center justify-between gap-2 border-t px-6 py-3.5">{footer}</SheetFooter> : null}
      </SheetContent>
    </Sheet>
  );
}

/* ------------------------------------------------------------------ toasts + confirm */

type ToastTone = 'success' | 'error' | 'info';
interface ConfirmOpts { title: string; message?: React.ReactNode; confirmLabel?: string; danger?: boolean }

const UiContext = createContext<{
  toast: (tone: ToastTone, title: string, description?: string) => void;
  confirm: (opts: ConfirmOpts) => Promise<boolean>;
} | null>(null);

/** Toasts go to sonner (<Toaster /> in main.tsx); confirm() renders a shadcn AlertDialog. */
export function UiProvider({ children }: { children: React.ReactNode }) {
  const [dialog, setDialog] = useState<(ConfirmOpts & { resolve: (v: boolean) => void }) | null>(null);
  // open is separate from the content: closing keeps the last title / buttons on screen while the
  // dialog animates out (clearing them showed an empty "Confirm / Cancel" dialog, like a second prompt)
  const [open, setOpen] = useState(false);

  const toast = useCallback((tone: ToastTone, title: string, description?: string) => {
    sonner[tone](title, { description, duration: tone === 'error' ? 7000 : 4000 });
  }, []);
  const confirm = useCallback((opts: ConfirmOpts) => new Promise<boolean>((resolve) => { setDialog({ ...opts, resolve }); setOpen(true); }), []);
  // the button's click and the dialog's own close both land here; a promise only takes the first answer
  const close = (v: boolean) => { dialog?.resolve(v); setOpen(false); };

  return (
    <UiContext.Provider value={{ toast, confirm }}>
      {children}
      <AlertDialog open={open} onOpenChange={(o) => { if (!o) close(false); }}>
        <AlertDialogContent className="rounded-xl">
          <AlertDialogHeader>
            <AlertDialogTitle>{dialog?.title}</AlertDialogTitle>
            <AlertDialogDescription asChild><div className="text-sm text-muted-foreground">{dialog?.message}</div></AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="cursor-pointer rounded-lg font-semibold" onClick={() => close(false)}>Cancel</AlertDialogCancel>
            <AlertDialogAction variant={dialog?.danger ? 'destructive' : 'default'} className="cursor-pointer rounded-lg font-semibold" onClick={() => close(true)} autoFocus>{dialog?.confirmLabel || 'Confirm'}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </UiContext.Provider>
  );
}

export function useUi() {
  const ctx = useContext(UiContext);
  if (!ctx) throw new Error('useUi must be used inside UiProvider');
  return ctx;
}
