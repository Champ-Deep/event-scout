// The contact view's visual primitives (tinted icon tiles, detail rows, copy buttons, section titles,
// the score ring, the staggered rise…). Shared by the contact drawer and its panels, the Recently
// deleted drawer, and the Duplicates / Events / Pipelines / Team pages.
import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { levelFrom } from '../lib/levels';
import { Avatar } from './Avatar';
import { Icon, type IconName } from './icons';
import { cx, useUi } from './kit';
import { AnimatedNumber } from './micro';
import { MOTION } from './tokens';

/** Tinted icon tiles, like the mobile app's stat / shortcut tiles (each detail gets its own colour). */
export const TINT = {
  violet: 'bg-brand-50 text-brand-700 dark:bg-brand-100 dark:text-brand-800',
  emerald: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300',
  sky: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300',
  amber: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300',
  rose: 'bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
  slate: 'bg-slate-100 text-slate-600 dark:bg-white/5 dark:text-slate-300',
  linkedin: 'bg-[#e8f0fb] text-[#0a66c2] dark:bg-[#0a66c2]/20 dark:text-[#7fb2ff]',
} as const;
export type Tint = keyof typeof TINT;

/** Tint → just the icon colour (for icons sitting inside inputs). */
export const TINT_TEXT: Record<Tint, string> = {
  violet: 'text-brand-600 dark:text-brand-700', emerald: 'text-emerald-600 dark:text-emerald-300', sky: 'text-sky-600 dark:text-sky-300',
  amber: 'text-amber-600 dark:text-amber-300', rose: 'text-rose-600 dark:text-rose-300', slate: 'text-slate-500 dark:text-slate-300',
  linkedin: 'text-[#0a66c2] dark:text-[#7fb2ff]',
};

/** One look for every chip in the contact header (event · representative · source · added): equal height. */
export const PILL = 'inline-flex h-7 items-center gap-1.5 whitespace-nowrap rounded-full bg-card px-2.5 text-xs font-medium text-body ring-1 ring-inset ring-border transition-colors';

export function IconTile({ icon, tint, size = 'md' }: { icon: IconName; tint: Tint; size?: 'sm' | 'md' }) {
  return (
    <span className={cx('flex shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-6 group-hover:scale-110',
      size === 'sm' ? 'h-7 w-7' : 'h-9 w-9', TINT[tint])}>
      <Icon name={icon} size={size === 'sm' ? 13 : 16} />
    </span>
  );
}

/**
 * One contact detail: a tinted icon tile, the label above the full value, and actions that are always
 * there. Rows slide in one after another (i = position) and the tile pops in with a spring.
 */
export function Row({ icon, tint = 'slate', label, children, action, i = 0 }: { icon: IconName; tint?: Tint; label: string; children: React.ReactNode; action?: React.ReactNode; i?: number }) {
  const empty = !children;
  return (
    <motion.div initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.12 + i * 0.035 }}
      className="group flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-secondary/60">
      <motion.span initial={{ scale: 0.5, rotate: -12 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.16 + i * 0.035 }}
        className={cx(empty && 'opacity-45 grayscale')}>
        <IconTile icon={icon} tint={tint} size="sm" />
      </motion.span>
      <div className="min-w-0 flex-1">
        <div className="text-[11px] font-medium text-muted-foreground">{label}</div>
        <div className="break-words text-sm font-medium text-foreground [overflow-wrap:anywhere]">{children || <span className="font-normal italic text-muted-foreground">Not added</span>}</div>
      </div>
      {action ? <div className="flex shrink-0 items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100 focus-within:opacity-100">{action}</div> : null}
    </motion.div>
  );
}

/** Copy to the clipboard with a toast ("Email copied"); the CopyAction buttons call it. */
export function useCopy() {
  const { toast } = useUi();
  return async (what: string, value: string) => {
    try { await navigator.clipboard.writeText(value); toast('success', `${what} copied`); } catch { toast('error', `Couldn't copy the ${what.toLowerCase()}`); }
  };
}

/** Copy button whose icon flips to a green tick for a moment after copying. */
export function CopyAction({ label, value, onCopy }: { label: string; value: string; onCopy: (what: string, value: string) => void }) {
  const [done, setDone] = useState(false);
  useEffect(() => { if (!done) return; const t = setTimeout(() => setDone(false), 1500); return () => clearTimeout(t); }, [done]);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.button type="button" whileTap={{ scale: 0.85 }} aria-label={`Copy ${label.toLowerCase()}`}
          onClick={() => { onCopy(label, value); setDone(true); }}
          className={cx('relative flex h-8 w-8 cursor-pointer items-center justify-center overflow-hidden rounded-lg transition-colors',
            done ? 'bg-emerald-50 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-300' : 'text-muted-foreground hover:bg-card hover:text-foreground')}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={done ? 'done' : 'copy'} initial={{ scale: 0.4, rotate: -90, opacity: 0 }} animate={{ scale: 1, rotate: 0, opacity: 1 }}
              exit={{ scale: 0.4, rotate: 90, opacity: 0 }} transition={{ type: 'spring', stiffness: 500, damping: 24 }}>
              <Icon name={done ? 'check' : 'copy'} size={14} />
            </motion.span>
          </AnimatePresence>
        </motion.button>
      </TooltipTrigger>
      <TooltipContent side="top">{done ? 'Copied' : `Copy ${label.toLowerCase()}`}</TooltipContent>
    </Tooltip>
  );
}

export function SectionTitle({ children, right, icon }: { children: React.ReactNode; right?: React.ReactNode; icon?: IconName }) {
  return (
    <h4 className="mb-2 flex items-center justify-between gap-2 text-[13px] font-semibold text-foreground">
      <span className="flex items-center gap-2">
        {icon ? (
          <motion.span initial={{ scale: 0, rotate: -30 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 400, damping: 16, delay: 0.1 }}
            className="flex h-6 w-6 items-center justify-center rounded-lg bg-brand-50 text-brand-700 dark:bg-brand-100 dark:text-brand-800">
            <Icon name={icon} size={13} />
          </motion.span>
        ) : null}
        {children}
      </span>{right}
    </h4>
  );
}

/** The mobile feed's radar pulse: rings ripple out from the photo (used for hot leads). */
export function RadarPulse({ color }: { color: string }) {
  return (
    <span className="pointer-events-none absolute inset-0" aria-hidden>
      {[0, 1].map((k) => (
        <motion.span key={k} className="absolute inset-0 rounded-full" style={{ boxShadow: `0 0 0 2px ${color}` }}
          initial={{ scale: 1, opacity: 0.55 }} animate={{ scale: 1.45, opacity: 0 }}
          transition={{ duration: 2.2, ease: 'easeOut', repeat: Infinity, delay: k * 1.1 }} />
      ))}
    </span>
  );
}

/** Sections slide up one after another (the app's FadeInDown stagger). */
export function Rise({ i = 0, children, className }: { i?: number; children: React.ReactNode; className?: string }) {
  return (
    <motion.div className={className} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.04 + i * 0.05 }}>
      {children}
    </motion.div>
  );
}

/** Round quick-action button with a springy press (the app's PressableScale). */
export function QuickAction({ icon, label, href, onClick, tint }: { icon: IconName; label: string; href?: string | undefined; onClick?: () => void; tint: Tint }) {
  const cls = cx('flex h-9 w-9 items-center justify-center rounded-full ring-1 ring-inset ring-black/5 dark:ring-white/10', TINT[tint]);
  const inner = <Icon name={icon} size={15} />;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {href ? (
          <motion.a whileHover={{ scale: 1.1, y: -1 }} whileTap={{ scale: 0.9 }} href={href} target={href.startsWith('http') ? '_blank' : undefined} rel="noreferrer" aria-label={label} className={cls}>{inner}</motion.a>
        ) : (
          <motion.button whileHover={{ scale: 1.1, y: -1 }} whileTap={{ scale: 0.9 }} type="button" onClick={onClick} aria-label={label} className={cx(cls, 'cursor-pointer')}>{inner}</motion.button>
        )}
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

/**
 * Score ring that draws to the score, the number inside (same as the contacts list and the app).
 * `level`: the server's level for this score (contact.level), when it sent one.
 */
export function DrawerRing({ score, level, size = 60 }: { score: number | null | undefined; level?: string | null | undefined; size?: number }) {
  const lvl = levelFrom(score, level);
  const big = size > 80;
  const stroke = big ? 7 : 5;
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  return (
    <span className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} style={{ stroke: 'var(--secondary)' }} strokeDasharray={score == null ? '3 4' : undefined} />
        {score != null ? (
          <motion.circle key={score} cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke} strokeLinecap="round"
            style={{ stroke: lvl ? lvl.color : 'var(--heat-none)' }} strokeDasharray={`${circ} ${circ}`}
            initial={{ strokeDashoffset: circ }} animate={{ strokeDashoffset: circ * (1 - Math.max(0, Math.min(100, score)) / 100) }}
            transition={{ duration: 1, ease: MOTION.ease, delay: 0.15 }} />
        ) : null}
      </svg>
      <span className={cx('es-num absolute font-bold text-foreground', big ? 'text-3xl' : 'text-lg')}>{score == null ? '–' : <AnimatedNumber value={score} duration={1000} />}</span>
    </span>
  );
}

export function Fade({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div key={k} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.fast, ease: MOTION.ease }}>
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

/** Empty state with a floating icon in a soft ring (Duplicates, Files). */
export function EmptyPanel({ icon, tint, title, children, action }: { icon: IconName; tint: Tint; title: string; children?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <Rise className="flex flex-col items-center justify-center rounded-2xl border border-dashed bg-card/60 px-6 py-10 text-center">
      <span className="relative mb-3 flex h-14 w-14 items-center justify-center">
        <motion.span className={cx('absolute inset-0 rounded-full opacity-40', TINT[tint])} animate={{ scale: [1, 1.18, 1] }} transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.span initial={{ scale: 0, rotate: -20 }} animate={{ scale: 1, rotate: 0, y: [0, -3, 0] }}
          transition={{ scale: { type: 'spring', stiffness: 360, damping: 16 }, rotate: { type: 'spring', stiffness: 360, damping: 16 }, y: { duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 0.6 } }}
          className={cx('relative flex h-11 w-11 items-center justify-center rounded-2xl', TINT[tint])}>
          <Icon name={icon} size={20} />
        </motion.span>
      </span>
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      {children ? <p className="mt-1 max-w-sm text-xs text-muted-foreground">{children}</p> : null}
      {action ? <motion.div className="mt-4" whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.95 }}>{action}</motion.div> : null}
    </Rise>
  );
}

/**
 * The representative's picture, or their initials when they haven't set one — with a small
 * "rep" badge so it never reads as the contact's own photo. Pops in, lifts on hover.
 */
export function RepAvatar({ name, photo, size = 'sm' }: { name: string; photo?: string | null | undefined; size?: 'xs' | 'sm' }) {
  return (
    <motion.span initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 460, damping: 20, delay: 0.15 }}
      className="relative inline-flex shrink-0 transition-transform duration-300 group-hover:scale-110">
      <span className="rounded-full bg-gradient-to-br from-brand-400 to-brand-600 p-[1.5px]">
        <Avatar src={photo || null} name={name} size={size}
          className={cx('border-2 border-card shadow-none', size === 'xs' ? 'size-6 text-[9px]' : 'size-7 text-[10px]')} />
      </span>
      <span className="absolute -bottom-0.5 -right-0.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-brand-600 text-white ring-2 ring-card">
        <Icon name="user" size={8} strokeWidth={3} />
      </span>
    </motion.span>
  );
}
