// Micro-interactions in the spirit of MicroKit (microkit.co): shine CTA, text-swap links, cursor
// spotlight cards, sliding tabs, count-up numbers. Motion (`motion/react`) drives the count-up,
// entrances, press-scale and the tab pill; hover effects are CSS (styles/tokens.css). Timings come
// from ui/tokens. Everything calms down when the OS asks for reduced motion (<MotionConfig> in main).
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { animate, motion, type HTMLMotionProps } from 'motion/react';
import { useReduceMotion } from '../lib/motion';
import { Icon, type IconName } from './icons';
import { cx } from './kit';
import { MOTION } from './tokens';

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Counts from the last shown value to `value` (ease-out, ~0.9 s). */
export function AnimatedNumber({ value, decimals = 0, className, duration = 900, prefix = '', suffix = '' }: {
  value: number | null | undefined; decimals?: number; className?: string; duration?: number; prefix?: string; suffix?: string;
}) {
  const [shown, setShown] = useState(0);
  const from = useRef(0);
  const reduce = useReduceMotion();
  useEffect(() => {
    if (value == null) return;
    if (reduce) { setShown(value); from.current = value; return; }
    const ctl = animate(from.current, value, {
      duration: duration / 1000, ease: MOTION.ease,
      onUpdate: (v) => { from.current = v; setShown(v); },
    });
    return () => ctl.stop();
  }, [value, duration, reduce]);
  if (value == null) return <span className={className}>—</span>;
  return <span className={cx('tabular-nums', className)}>{prefix}{shown.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}

/** A card surface. Hover firms up the hairline (no glow); clickable ones show a pointer. `tone` is kept for callers. */
export function SpotlightCard({ children, className, onClick, style }: {
  children: React.ReactNode; className?: string | undefined; onClick?: (() => void) | undefined; tone?: 'violet' | 'sky' | 'amber' | 'emerald' | 'rose' | undefined; style?: React.CSSProperties | undefined;
}) {
  const Comp: any = onClick ? 'button' : 'div';
  return (
    <Comp onClick={onClick} style={style}
      className={cx('group relative w-full overflow-hidden rounded-xl border bg-card text-left shadow-card transition duration-200',
        onClick && 'cursor-pointer hover:border-foreground/25 hover:shadow-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring', className)}>
      {children}
    </Comp>
  );
}

/** The page's one call to action: brand violet with white text. The icon swaps to a send arrow on hover. */
export function ShineButton({ children, icon, className, ...rest }: Omit<HTMLMotionProps<'button'>, 'children'> & { icon?: IconName; children?: React.ReactNode }) {
  return (
    <motion.button whileTap={{ scale: MOTION.press }} transition={{ duration: MOTION.fast }}
      className={cx('group relative inline-flex h-9 cursor-pointer items-center gap-2 rounded-lg bg-signal px-4 text-sm font-semibold text-signal-foreground shadow-[inset_0_-1px_0_rgb(23_24_20/0.18)] transition-[filter] hover:brightness-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground focus-visible:ring-offset-2 focus-visible:ring-offset-background', className)} {...rest}>
      {icon ? <span className="relative flex h-4 w-4 overflow-hidden">
        <span className="absolute inset-0 transition-transform duration-300 group-hover:-translate-y-full"><Icon name={icon} size={16} /></span>
        <span className="absolute inset-0 translate-y-full transition-transform duration-300 group-hover:translate-y-0"><Icon name="send" size={16} /></span>
      </span> : null}
      <span className="relative">{children}</span>
    </motion.button>
  );
}

/** Label swaps vertically on hover; an arrow slides through (MicroKit "text swap" + "arrow slide"). */
export function SwapLink({ children, onClick, className }: { children: string; onClick?: () => void; className?: string }) {
  return (
    <button onClick={onClick} className={cx('group inline-flex cursor-pointer items-center gap-1 text-xs font-medium text-primary', className)}>
      <span className="relative inline-flex h-4 overflow-hidden">
        <span className="transition-transform duration-300 ease-out group-hover:-translate-y-full">{children}</span>
        <span className="absolute inset-x-0 top-full select-none transition-transform duration-300 ease-out group-hover:-translate-y-full" aria-hidden>{children}</span>
      </span>
      <span className="relative flex h-3.5 w-3.5 overflow-hidden">
        <span className="absolute inset-0 transition-transform duration-300 group-hover:translate-x-full"><Icon name="chevronRight" size={14} /></span>
        <span className="absolute inset-0 -translate-x-full transition-transform duration-300 group-hover:translate-x-0"><Icon name="chevronRight" size={14} /></span>
      </span>
    </button>
  );
}

/** Segmented control with a sliding active surface (MicroKit "sliding content tabs"). Optional per-tab counts. */
export function SlidingTabs<T extends string>({ value, onChange, options, size = 'md', className }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode; count?: number | undefined; dot?: string | undefined }[]; size?: 'sm' | 'md'; className?: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);
  const counts = options.map((o) => o.count).join('|');
  useLayoutEffect(() => {
    const el = wrap.current?.querySelector<HTMLButtonElement>(`[data-value="${CSS.escape(value)}"]`);
    if (el) setPill({ left: el.offsetLeft, width: el.offsetWidth });
  }, [value, options.length, counts]);
  return (
    <div ref={wrap} role="tablist" className={cx('relative inline-flex max-w-full rounded-lg border bg-secondary p-0.5', className)}>
      {pill ? <motion.span className="absolute bottom-0.5 top-0.5 left-0 rounded-md bg-card shadow-card ring-1 ring-border" initial={false}
        animate={{ x: pill.left, width: pill.width }} transition={{ type: 'spring', stiffness: 520, damping: 40 }} aria-hidden /> : null}
      {options.map((o) => (
        <button key={o.value} data-value={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
          className={cx('relative z-[1] inline-flex cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors', size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
            value === o.value ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
          {o.dot ? <span className={cx('h-1.5 w-1.5 rounded-full', o.dot)} aria-hidden /> : null}
          {o.label}
          {o.count != null ? (
            <span className={cx('rounded-full px-1.5 text-[10px] tabular-nums transition-colors', value === o.value ? 'bg-signal text-signal-foreground' : 'bg-border/80 text-body')}>
              <AnimatedNumber value={o.count} duration={500} />
            </span>
          ) : null}
        </button>
      ))}
    </div>
  );
}

/** ▲ 12% / ▼ 3% vs the previous period (points for scores). */
export function Delta({ value, unit = '%', invert, label = 'vs previous' }: { value: number | null | undefined; unit?: string; invert?: boolean; label?: string }) {
  if (value == null) return <span className="text-xs text-muted-foreground">No earlier data</span>;
  const up = value > 0;
  const flat = value === 0;
  const good = invert ? !up : up;
  return (
    <span className="inline-flex items-center gap-1 text-xs">
      <span className={cx('inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-semibold tabular-nums',
        flat ? 'bg-secondary text-muted-foreground' : good ? 'bg-success-bg text-success' : 'bg-danger-bg text-destructive')}>
        {flat ? '' : <svg width="8" height="8" viewBox="0 0 8 8" className={up ? '' : 'rotate-180'} aria-hidden><path d="M4 1 7 6H1z" fill="currentColor" /></svg>}
        {flat ? '0' : Math.abs(value)}{unit}
      </span>
      <span className="text-muted-foreground">{label}</span>
    </span>
  );
}

/** A live dot that breathes. */
export function PulseDot({ tone = 'emerald' }: { tone?: 'emerald' | 'amber' | 'rose' | 'violet' }) {
  const c = { emerald: 'bg-live', amber: 'bg-heat-warm', rose: 'bg-heat-hot', violet: 'bg-primary' }[tone];
  return (
    <span className="relative flex h-2 w-2">
      <span className={cx('absolute inline-flex h-full w-full animate-ping rounded-full opacity-60', c)} />
      <span className={cx('relative inline-flex h-2 w-2 rounded-full', c)} />
    </span>
  );
}

/** Staggered entrance for a block (index → delay): rises out of a soft blur (MicroKit "blur glide").
 *  The blur is dropped once done, so nothing inside keeps a filter (it would trap fixed popovers). */
export function Reveal({ index = 0, children, className }: { index?: number; children: React.ReactNode; className?: string }) {
  return (
    <motion.div className={className} initial={{ opacity: 0, y: 10, filter: 'blur(6px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)', transitionEnd: { filter: 'none' } }}
      transition={{ duration: MOTION.base * 1.8, ease: MOTION.ease, delay: index * MOTION.stagger }}>
      {children}
    </motion.div>
  );
}
