// Small SVG charts with hover tooltips; lines draw on with `motion` (pathLength), bars grow in (CSS).
import React, { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { cx } from './kit';
import { MOTION } from './tokens';

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return;
    setW(ref.current.clientWidth);
    const ro = new ResizeObserver(([e]) => setW(e!.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** A round tick step so the axis reads 0 · 10 · 20 · 30 · 40 (4 steps) — never 13 / 38. */
export function niceStep(max: number, steps = 4) {
  const raw = Math.max(1, max) / steps;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / p;
  return Math.max(1, (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p);
}
export const niceMax = (v: number, steps = 4) => niceStep(v, steps) * steps;

/** Monotone cubic path (no overshoot below zero) through the points. */
function monotonePath(pts: [number, number][]) {
  const n = pts.length;
  if (n === 0) return '';
  if (n === 1) return `M${pts[0]![0]},${pts[0]![1]}`;
  const dx = (i: number) => pts[i + 1]![0] - pts[i]![0];
  const slope = (i: number) => (pts[i + 1]![1] - pts[i]![1]) / (dx(i) || 1);
  const m: number[] = [];
  for (let i = 0; i < n; i++) {
    if (i === 0) m.push(slope(0));
    else if (i === n - 1) m.push(slope(n - 2));
    else {
      const a = slope(i - 1), b = slope(i);
      m.push(a * b <= 0 ? 0 : (3 * (dx(i - 1) + dx(i))) / ((2 * dx(i) + dx(i - 1)) / a + (dx(i) + 2 * dx(i - 1)) / b));
    }
  }
  let d = `M${pts[0]![0]},${pts[0]![1]}`;
  for (let i = 0; i < n - 1; i++) {
    const [x0, y0] = pts[i]!, [x1, y1] = pts[i + 1]!;
    const h = (x1 - x0) / 3;
    d += ` C${x0 + h},${y0 + m[i]! * h} ${x1 - h},${y1 - m[i + 1]! * h} ${x1},${y1}`;
  }
  return d;
}

export interface Series { key: string; label: string; color: string }

/** Area / line chart over days, with a crosshair tooltip. */
export function AreaChart({ data, series, height = 260, formatX, formatTip }: {
  data: Record<string, any>[]; series: Series[]; height?: number;
  formatX: (row: Record<string, any>) => string; formatTip?: (row: Record<string, any>) => string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const uid = useId().replace(/:/g, '');
  const pad = { l: 34, r: 12, t: 14, b: 28 };
  const iw = Math.max(1, width - pad.l - pad.r);
  const ih = height - pad.t - pad.b;
  const max = niceMax(Math.max(1, ...data.flatMap((r) => series.map((s) => Number(r[s.key]) || 0))));
  const x = (i: number) => pad.l + (data.length <= 1 ? iw / 2 : (i / (data.length - 1)) * iw);
  const y = (v: number) => pad.t + ih - (v / max) * ih;
  const ticks = [0, 1, 2, 3, 4].map((i) => (max / 4) * i);
  const labelEvery = Math.max(1, Math.ceil(data.length / Math.max(2, Math.floor(iw / 72))));
  // label every n-th day, plus the last one when it isn't too close to the previous label
  const showLabel = (i: number) => i % labelEvery === 0 || (i === data.length - 1 && (data.length - 1) % labelEvery >= labelEvery / 2);
  const key = `${data.length}-${data.map((r) => series.map((s) => r[s.key]).join('.')).join('|')}`;

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    const px = e.clientX - r.left;
    const i = data.length <= 1 ? 0 : Math.round((px / r.width) * (data.length - 1));
    setHover(Math.max(0, Math.min(data.length - 1, i)));
  };

  return (
    <div ref={ref} className="relative w-full" style={{ height }}>
      {width > 0 ? (
        <svg width={width} height={height} className="overflow-visible" key={key}>
          <defs>
            {series.map((s) => (
              <linearGradient key={s.key} id={`${uid}-${s.key}`} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" style={{ stopColor: s.color }} stopOpacity={0.16} />
                <stop offset="100%" style={{ stopColor: s.color }} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} className={t === 0 ? 'text-foreground/35' : 'text-border'} stroke="currentColor" strokeDasharray={t === 0 ? undefined : '2 4'} />
              <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" className="fill-muted-foreground font-mono text-[10px] tabular-nums">{t}</text>
            </g>
          ))}
          {data.map((r, i) => showLabel(i) ? (
            <text key={i} x={x(i)} y={height - 8} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px]">{formatX(r)}</text>
          ) : null)}
          {series.map((s, si) => {
            const pts = data.map((r, i) => [x(i), y(Number(r[s.key]) || 0)] as [number, number]);
            const line = monotonePath(pts);
            const area = pts.length ? `${line} L${pts[pts.length - 1]![0]},${y(0)} L${pts[0]![0]},${y(0)} Z` : '';
            return (
              <g key={s.key}>
                <path d={area} fill={`url(#${uid}-${s.key})`} className="es-fade-late" />
                <motion.path d={line} fill="none" style={{ stroke: s.color }} strokeWidth={2.25} strokeLinecap="round" strokeLinejoin="round"
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.1, ease: MOTION.ease, delay: si * 0.15 }} />
              </g>
            );
          })}
          {hover != null ? (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={pad.t + ih} className="text-foreground/40" stroke="currentColor" />
              {series.map((s) => (
                <circle key={s.key} cx={x(hover)} cy={y(Number(data[hover]![s.key]) || 0)} r={4.5} style={{ stroke: s.color }} strokeWidth={2.5} className="fill-card" />
              ))}
            </g>
          ) : null}
          <rect x={pad.l} y={pad.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
      ) : null}
      {hover != null && width > 0 ? (
        <div className="es-fade pointer-events-none absolute z-10 min-w-[150px] -translate-x-1/2 rounded-lg border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-lg"
          style={{ left: Math.min(Math.max(x(hover), 80), width - 80), top: 4 }}>
          <div className="mb-1 font-semibold text-foreground">{formatTip ? formatTip(data[hover]!) : formatX(data[hover]!)}</div>
          {series.map((s) => (
            <div key={s.key} className="flex items-center justify-between gap-4 py-0.5">
              <span className="flex items-center gap-1.5 text-muted-foreground"><span className="h-2 w-2 rounded-[2px]" style={{ background: s.color }} />{s.label}</span>
              <span className="font-mono font-medium tabular-nums text-foreground">{Number(data[hover]![s.key]) || 0}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export interface Segment { key: string; label: string; value: number; color: string }

/**
 * Donut with animated sweep; hover a slice (or a linked row) to see it in the centre. The hovered
 * slice grows into a margin kept free for it, so it never clips; `center` replaces the default
 * centre read-out (it gets the hovered slice, if any, and the total).
 */
export function Donut({ segments, size = 188, thickness = 22, centerLabel, onSelect, active, setActive, center }: {
  segments: Segment[]; size?: number; thickness?: number; centerLabel: string;
  onSelect?: (key: string) => void; active: string | null; setActive: (k: string | null) => void;
  center?: (hovered: Segment | undefined, total: number) => React.ReactNode;
}) {
  const [on, setOn] = useState(false);
  useEffect(() => { const t = requestAnimationFrame(() => setOn(true)); return () => cancelAnimationFrame(t); }, []);
  const grow = 6;
  const total = segments.reduce((s, x) => s + x.value, 0);
  const r = (size - thickness - grow) / 2;
  const c = 2 * Math.PI * r;
  let acc = 0;
  const gap = total && segments.filter((s) => s.value).length > 1 ? 3 : 0;
  const hovered = segments.find((s) => s.key === active);
  const inner = size - grow - thickness * 2; // the hole's diameter
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={thickness} className="text-secondary" stroke="currentColor" />
        {total ? segments.map((s) => {
          const len = (s.value / total) * c;
          const dash = Math.max(0, len - gap);
          const el = (
            <circle key={s.key} cx={size / 2} cy={size / 2} r={r} fill="none" strokeLinecap="butt"
              strokeWidth={active === s.key ? thickness + grow : thickness}
              strokeDasharray={`${on ? dash : 0} ${c}`} strokeDashoffset={-acc}
              style={{ stroke: s.color, transition: 'stroke-dasharray .9s cubic-bezier(.2,.8,.2,1), stroke-width .2s, opacity .2s', opacity: active && active !== s.key ? 0.3 : 1, cursor: onSelect ? 'pointer' : undefined }}
              onMouseEnter={() => setActive(s.key)} onMouseLeave={() => setActive(null)} onClick={() => onSelect?.(s.key)} />
          );
          acc += len;
          return s.value ? el : null;
        }) : null}
      </svg>
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="flex flex-col items-center justify-center text-center" style={{ width: inner * 0.82 }}>
          {center ? center(hovered, total) : (
            <>
              <div className="text-2xl font-semibold tabular-nums text-foreground">{hovered ? hovered.value.toLocaleString() : total.toLocaleString()}</div>
              <div className="line-clamp-2 text-xs leading-tight text-muted-foreground">{hovered ? `${hovered.label} · ${total ? Math.round((hovered.value / total) * 100) : 0}%` : centerLabel}</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/** Vertical columns (histograms) that grow in, staggered, with hover values. */
export function Columns({ data, height = 168, color = 'var(--chart-1)', highlight }: {
  data: { label: string; value: number; tip?: string }[]; height?: number; color?: string | ((i: number) => string); highlight?: (i: number) => boolean;
}) {
  const max = Math.max(1, ...data.map((d) => d.value));
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div className="flex items-end gap-1.5" style={{ height }}>
      {data.map((d, i) => {
        const h = d.value ? Math.max(4, (d.value / max) * (height - 34)) : 2;
        const fill = typeof color === 'function' ? color(i) : color;
        return (
          <div key={i} className="group relative flex flex-1 flex-col items-center justify-end gap-1.5" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <span className={cx('text-[10px] font-semibold tabular-nums transition-opacity', hover === i || highlight?.(i) ? 'opacity-100 text-body' : 'opacity-0')}>{d.value}</span>
            <div className="es-grow-y w-full rounded-t-[3px] transition-[filter,opacity] group-hover:brightness-110"
              style={{ height: h, background: fill, opacity: d.value ? (hover == null || hover === i ? 1 : 0.55) : 0.2, animationDelay: `${i * 45}ms` }} title={d.tip} />
            <span className="w-full truncate text-center font-mono text-[10px] text-muted-foreground">{d.label}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Ranked horizontal bars (top events / reps / sources). */
export function BarList({ items, color = 'bg-foreground/80', onSelect, format = (n: number) => n.toLocaleString() }: {
  items: { key: string; label: React.ReactNode; value: number; sub?: React.ReactNode }[]; color?: string; onSelect?: (key: string) => void; format?: (n: number) => string;
}) {
  const max = Math.max(1, ...items.map((i) => i.value));
  return (
    <ul className="space-y-3">
      {items.map((it, i) => {
        const Comp: any = onSelect ? 'button' : 'div';
        return (
          <li key={it.key}>
            <Comp onClick={onSelect ? () => onSelect(it.key) : undefined} className={cx('group block w-full text-left', onSelect && 'cursor-pointer')}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="min-w-0 flex-1 truncate font-medium text-body transition-colors group-hover:text-accent-foreground">{it.label}</span>
                {it.sub ? <span className="shrink-0 text-xs text-muted-foreground">{it.sub}</span> : null}
                <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">{format(it.value)}</span>
              </div>
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-secondary">
                <div className={cx('es-grow-x h-full rounded-full', color)} style={{ width: `${(it.value / max) * 100}%`, animationDelay: `${i * 70}ms` }} />
              </div>
            </Comp>
          </li>
        );
      })}
    </ul>
  );
}

/** Tiny trend line for a KPI card. */
export function Sparkline({ values, color = 'var(--chart-1)', height = 36 }: { values: number[]; color?: string; height?: number }) {
  const uid = useId().replace(/:/g, '');
  const pts = useMemo(() => {
    const max = Math.max(1, ...values);
    return values.map((v, i) => [values.length <= 1 ? 50 : (i / (values.length - 1)) * 100, 30 - (v / max) * 26 - 2] as [number, number]);
  }, [values]);
  if (!values.length) return null;
  const line = monotonePath(pts);
  return (
    <svg viewBox="0 0 100 32" preserveAspectRatio="none" className="w-full" style={{ height }} key={values.join(',')}>
      <defs>
        <linearGradient id={`sp-${uid}`} x1="0" x2="0" y1="0" y2="1"><stop offset="0%" style={{ stopColor: color }} stopOpacity={0.2} /><stop offset="100%" style={{ stopColor: color }} stopOpacity={0} /></linearGradient>
      </defs>
      <path d={`${line} L100,32 L0,32 Z`} fill={`url(#sp-${uid})`} className="es-fade-late" />
      <motion.path d={line} fill="none" style={{ stroke: color }} strokeWidth={1.75} vectorEffect="non-scaling-stroke"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: MOTION.slow, ease: MOTION.ease }} />
    </svg>
  );
}

/** Ring showing a 0–1 share, sweeping in. */
export function Ring({ value, size = 92, thickness = 9, color = 'var(--success)', children }: { value: number; size?: number; thickness?: number; color?: string; children?: React.ReactNode }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const t = requestAnimationFrame(() => setOn(true)); return () => cancelAnimationFrame(t); }, []);
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={thickness} className="text-secondary" stroke="currentColor" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={thickness} strokeLinecap="round"
          strokeDasharray={`${on ? Math.max(0, Math.min(1, value)) * c : 0} ${c}`} style={{ stroke: color, transition: 'stroke-dasharray 1s cubic-bezier(.2,.8,.2,1)' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">{children}</div>
    </div>
  );
}
