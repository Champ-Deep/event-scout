// The contacts toolbar, in the mobile app's language (apps/mobile app/(app)/contacts):
//   SearchField  — the app's AnimatedSearch: a rounded field whose border and icon turn brand on focus
//   FilterPill   — the app's FilterPill: icon + label pill, filled brand when a value is picked, × to clear
//   SortMenu     — "Newest ▾" pill with every sort as one tap
//   LayoutToggle — Table / Cards segmented control (a sliding thumb)
//   ColumnsMenu  — table columns, row height, reset and the keyboard shortcuts
import React, { useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { cx } from '../../ui/kit';
import type { TablePrefs } from './table';

/** Every toolbar control is a compact 32px pill, like the app's chips. */
const PILL = 'inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 rounded-full border-[1.5px] text-[12.5px] font-bold transition-[background-color,border-color,color,box-shadow,transform] duration-200 active:scale-[0.97] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';
const PILL_IDLE = 'border-border bg-card text-foreground hover:border-brand-200 hover:bg-brand-50/60 data-[state=open]:border-brand-300 data-[state=open]:bg-brand-50';

/* ------------------------------------------------------------------ search */

export function SearchField({ value, onChange, placeholder, containerRef, className }: {
  value: string; onChange: (v: string) => void; placeholder: string; containerRef?: React.Ref<HTMLDivElement>; className?: string;
}) {
  return (
    <div ref={containerRef}
      className={cx('group/search flex h-9 min-w-0 items-center gap-2 rounded-xl border bg-card pl-2.5 pr-1.5 shadow-card transition-[border-color,box-shadow] duration-200',
        'focus-within:border-brand-400 focus-within:shadow-[0_0_0_4px_color-mix(in_oklab,var(--brand-400)_18%,transparent)]', className)}>
      <Icon name="search" size={15}
        className={cx('transition-[transform,color] duration-300 group-focus-within/search:-rotate-12 group-focus-within/search:scale-110 group-focus-within/search:text-brand-600',
          value ? 'text-brand-600' : 'text-muted-foreground')} />
      <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} aria-label={placeholder.replace(/…$/, '')}
        type="search" autoComplete="off" spellCheck={false}
        className="h-full min-w-0 flex-1 bg-transparent text-[13px] font-medium text-foreground outline-none placeholder:font-normal placeholder:text-muted-foreground [&::-webkit-search-cancel-button]:hidden" />
      <AnimatePresence mode="wait" initial={false}>
        {value ? (
          <motion.button key="clear" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: 'spring', stiffness: 520, damping: 26 }}
            onClick={() => onChange('')} aria-label="Clear search"
            className="flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-full bg-muted-foreground/70 text-white transition-colors hover:bg-foreground">
            <Icon name="x" size={11} strokeWidth={3} />
          </motion.button>
        ) : (
          <motion.kbd key="kbd" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="pointer-events-none hidden rounded-md border bg-secondary px-1.5 font-mono text-[10px] leading-5 text-muted-foreground sm:block">/</motion.kbd>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ filter pill */

export interface PillOption { value: string; label: string; count?: number; icon?: React.ReactNode }

/**
 * One single-choice filter. Idle: an outlined pill with a tinted icon. Picked: a filled brand pill
 * showing the choice, with its own × to clear. Opens a searchable list with counts.
 */
export function FilterPill({ icon, tint, title, options, value, onChange, noun }: {
  icon: IconName; tint: string; title: string; options: PillOption[]; value: string | undefined;
  onChange: (v: string | undefined) => void; noun: string;
}) {
  const [open, setOpen] = useState(false);
  const chosen = value ? options.find((o) => o.value === value) : undefined;
  const on = !!value;
  return (
    <span className={cx('inline-flex shrink-0 items-stretch rounded-full transition-shadow duration-200', on && 'shadow-active')}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button aria-label={on ? `${title}: ${chosen?.label ?? value}. Change` : `Filter by ${title.toLowerCase()}`}
            className={cx(PILL, on ? 'rounded-r-none border-r-0 border-brand-700 bg-brand-600 pl-3 pr-2 text-white hover:bg-brand-700 dark:text-white' : cx(PILL_IDLE, 'pl-3 pr-2.5'))}>
            <span className="flex" style={on ? undefined : { color: tint }}><Icon name={icon} size={14} /></span>
            {on ? (
              <span className="flex min-w-0 items-baseline gap-1">
                <span className="hidden font-semibold text-white/75 sm:inline">{title}</span>
                <span className="max-w-[150px] truncate">{chosen?.label ?? value}</span>
              </span>
            ) : (
              <>
                <span>{title}</span>
                <Icon name="chevronDown" size={13} className="-mr-0.5 text-muted-foreground transition-transform duration-200 [[data-state=open]_&]:rotate-180" />
              </>
            )}
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" sideOffset={8} className="w-72 overflow-hidden rounded-2xl p-0 shadow-xl">
          <div className="flex items-center justify-between border-b px-3.5 py-2.5">
            <span className="flex items-center gap-2 text-sm font-extrabold text-foreground">
              <span className="flex size-6 items-center justify-center rounded-lg" style={{ background: `color-mix(in oklab, ${tint} 14%, var(--card))`, color: tint }}><Icon name={icon} size={13} /></span>
              {title}
            </span>
            <span className="text-[11px] font-semibold text-muted-foreground">{options.length} {noun}{options.length === 1 ? '' : 's'}</span>
          </div>
          <Command>
            <CommandInput placeholder={`Find ${noun}…`} />
            <CommandList className="max-h-72">
              <CommandEmpty>No {noun} matches.</CommandEmpty>
              <CommandGroup>
                {options.map((o) => {
                  const sel = o.value === value;
                  return (
                    <CommandItem key={o.value} value={`${o.label} ${o.value}`} onSelect={() => { onChange(sel ? undefined : o.value); setOpen(false); }}
                      className={cx('cursor-pointer gap-2.5 rounded-xl px-2 py-1.5', sel && 'bg-accent text-accent-foreground')}>
                      {o.icon}
                      <span className={cx('min-w-0 flex-1 truncate text-[13px]', sel ? 'font-bold' : 'font-medium')}>{o.label}</span>
                      {o.count != null ? (
                        <span className={cx('es-num min-w-6 rounded-full px-1.5 text-center text-[11px] font-black leading-5', sel ? 'bg-brand-600 text-white' : 'bg-secondary text-body')}>{o.count}</span>
                      ) : null}
                      {sel ? <Icon name="check" size={14} className="text-brand-600" /> : null}
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
          </Command>
          {on ? (
            <button onClick={() => { onChange(undefined); setOpen(false); }}
              className="flex w-full cursor-pointer items-center justify-center gap-1.5 border-t py-2.5 text-xs font-bold text-brand-700 transition-colors hover:bg-brand-50">
              <Icon name="x" size={12} />Show every {noun}
            </button>
          ) : null}
        </PopoverContent>
      </Popover>
      {on ? (
        <button onClick={() => onChange(undefined)} aria-label={`Clear the ${title.toLowerCase()} filter`}
          className="flex cursor-pointer items-center rounded-r-full border-[1.5px] border-l-0 border-brand-700 bg-brand-600 pl-0.5 pr-1.5 text-white transition-colors hover:bg-brand-700">
          <span className="flex size-5 items-center justify-center rounded-full bg-white/20 transition-colors hover:bg-white/35"><Icon name="x" size={11} strokeWidth={3} /></span>
        </button>
      ) : null}
    </span>
  );
}

/**
 * A filter that came in from a link (e.g. "Added · Today" from the dashboard): shown like a picked
 * FilterPill — filled brand, its value, and × to remove it.
 */
export function FilterChip({ icon, title, label, hint, onClear }: { icon: IconName; title: string; label: string; hint?: string; onClear: () => void }) {
  return (
    <span className="inline-flex shrink-0 items-stretch rounded-full shadow-active" title={hint}>
      <span className={cx(PILL, 'cursor-default rounded-r-none border-r-0 border-brand-700 bg-brand-600 pl-3 pr-2 text-white active:scale-100 dark:text-white')}>
        <span className="flex"><Icon name={icon} size={14} /></span>
        <span className="flex min-w-0 items-baseline gap-1">
          <span className="hidden font-semibold text-white/75 sm:inline">{title}</span>
          <span className="max-w-[150px] truncate">{label}</span>
        </span>
      </span>
      <button onClick={onClear} aria-label={`Clear the ${title.toLowerCase()} ${label.toLowerCase()} filter`}
        className="flex cursor-pointer items-center rounded-r-full border-[1.5px] border-l-0 border-brand-700 bg-brand-600 pl-0.5 pr-1.5 text-white transition-colors hover:bg-brand-700">
        <span className="flex size-5 items-center justify-center rounded-full bg-white/20 transition-colors hover:bg-white/35"><Icon name="x" size={11} strokeWidth={3} /></span>
      </button>
    </span>
  );
}

/* ------------------------------------------------------------------ sort */

export const SORT_OPTIONS: { value: string; label: string; short: string; icon: IconName }[] = [
  { value: 'added:desc', label: 'Newest first', short: 'Newest', icon: 'clock' },
  { value: 'added:asc', label: 'Oldest first', short: 'Oldest', icon: 'clock' },
  { value: 'score:desc', label: 'Highest score', short: 'Top score', icon: 'flame' },
  { value: 'score:asc', label: 'Lowest score', short: 'Low score', icon: 'flame' },
  { value: 'name:asc', label: 'Name A → Z', short: 'Name', icon: 'user' },
  { value: 'company:asc', label: 'Company A → Z', short: 'Company', icon: 'building' },
  { value: 'updated:desc', label: 'Recently updated', short: 'Updated', icon: 'refresh' },
];

export function SortMenu({ value, onChange, customLabel }: { value: string; onChange: (v: string) => void; customLabel: string }) {
  const [open, setOpen] = useState(false);
  const cur = SORT_OPTIONS.find((o) => o.value === value);
  const desc = value.endsWith(':desc');
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button aria-label={`Sort: ${cur?.label ?? customLabel}. Change`} className={cx(PILL, PILL_IDLE, 'pl-2.5 pr-2.5')}>
          <span className="flex size-5 items-center justify-center rounded-full bg-brand-50 text-brand-700">
            <Icon name={desc ? 'arrowDown' : 'arrowUp'} size={12} strokeWidth={2.5} />
          </span>
          <span className="hidden font-semibold text-muted-foreground sm:inline">Sort</span>
          <span>{cur?.short ?? customLabel}</span>
          <Icon name="chevronDown" size={13} className="-mr-0.5 text-muted-foreground transition-transform duration-200 [[data-state=open]_&]:rotate-180" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={8} className="w-60 rounded-2xl p-1.5 shadow-xl">
        <div className="px-2 pb-1 pt-1.5 text-[10.5px] font-bold uppercase tracking-[0.07em] text-muted-foreground">Sort contacts by</div>
        {SORT_OPTIONS.map((o) => {
          const sel = o.value === value;
          return (
            <button key={o.value} onClick={() => { onChange(o.value); setOpen(false); }} aria-pressed={sel}
              className={cx('flex w-full cursor-pointer items-center gap-2.5 rounded-xl px-2 py-1.5 text-left text-[13px] transition-colors',
                sel ? 'bg-accent font-bold text-accent-foreground' : 'font-medium text-body hover:bg-secondary hover:text-foreground')}>
              <span className={cx('flex size-6 items-center justify-center rounded-lg', sel ? 'bg-brand-600 text-white' : 'bg-secondary text-muted-foreground')}>
                <Icon name={o.icon} size={13} />
              </span>
              <span className="flex-1">{o.label}</span>
              <Icon name={o.value.endsWith(':desc') ? 'arrowDown' : 'arrowUp'} size={12} className={sel ? 'text-brand-600' : 'text-muted-foreground/50'} />
            </button>
          );
        })}
      </PopoverContent>
    </Popover>
  );
}

/* ------------------------------------------------------------------ layout */

export function LayoutToggle({ view, onView }: { view: TablePrefs['view']; onView: (v: TablePrefs['view']) => void }) {
  const items = [['table', 'rows', 'Table'], ['cards', 'columns', 'Cards']] as const;
  return (
    <div role="radiogroup" aria-label="Layout" className="inline-flex h-8 shrink-0 items-center rounded-full border-[1.5px] bg-secondary p-[2px]">
      {items.map(([v, icon, label]) => {
        const on = view === v;
        return (
          <Tooltip key={v}>
            <TooltipTrigger asChild>
              <button role="radio" aria-checked={on} aria-label={`${label} view`} onClick={() => onView(v)}
                className={cx('relative inline-flex h-full cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-xs font-bold transition-colors duration-200',
                  on ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
                {on ? <motion.span layoutId="es-layout-thumb" className="absolute inset-0 rounded-full bg-card shadow-card" transition={{ duration: MOTION.base, ease: MOTION.ease }} /> : null}
                <Icon name={icon} size={14} className={cx('relative', on && 'text-brand-600')} />
                <span className="relative hidden xl:inline">{label}</span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">{label} view</TooltipContent>
          </Tooltip>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ columns */

const SHORTCUTS: [string, string][] = [['j / k', 'move'], ['x', 'select'], ['↵', 'open'], ['e', 'edit'], ['s', 'score'], ['[ ]', 'page'], ['/', 'search']];

export function ColumnsMenu({ columns, density, onDensity, onReset, disabled }: {
  columns: { id: string; label: string; visible: boolean; locked: boolean; toggle: (v: boolean) => void }[];
  density: TablePrefs['density']; onDensity: (d: TablePrefs['density']) => void; onReset: () => void; disabled?: boolean;
}) {
  const shown = columns.filter((c) => c.visible).length;
  const chip = (on: boolean) => cx('inline-flex h-7 cursor-pointer items-center gap-1 rounded-full border-[1.5px] px-2.5 text-xs font-bold transition-colors',
    on ? 'border-brand-300 bg-brand-50 text-brand-700' : 'border-border bg-card text-muted-foreground hover:border-brand-200 hover:text-foreground');
  return (
    <Popover>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <button aria-label="Columns and row height" className={cx(PILL, PILL_IDLE, 'gap-1.5 px-2.5', disabled && 'opacity-60')}>
              <Icon name="sliders" size={14} />
              <span className="hidden 2xl:inline">Columns</span>
              <span className="es-num rounded-full bg-secondary px-1.5 text-[11px] font-black leading-5 text-body">{shown}</span>
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom">Columns &amp; row height</TooltipContent>
      </Tooltip>
      <PopoverContent align="end" sideOffset={8} className="w-[320px] overflow-hidden rounded-2xl p-0 shadow-xl">
        <div className="flex items-center justify-between border-b px-3.5 py-2.5">
          <span className="text-sm font-extrabold text-foreground">Table columns</span>
          <button onClick={onReset} className="inline-flex cursor-pointer items-center gap-1 rounded-full px-2 py-0.5 text-xs font-bold text-muted-foreground hover:bg-secondary hover:text-foreground">
            <Icon name="undo" size={12} />Reset view
          </button>
        </div>
        <div className="space-y-4 p-3.5">
          {disabled ? <p className="rounded-xl bg-secondary px-3 py-2 text-xs text-muted-foreground">Columns apply to the Table view.</p> : null}
          <div className="flex flex-wrap gap-1.5">
            {columns.map((c) => (
              <button key={c.id} disabled={c.locked} onClick={() => c.toggle(!c.visible)} aria-pressed={c.visible}
                title={c.locked ? 'Always shown' : c.visible ? `Hide ${c.label}` : `Show ${c.label}`}
                className={cx(chip(c.visible), c.locked && 'cursor-default opacity-70')}>
                <Icon name={c.locked ? 'key' : c.visible ? 'check' : 'plus'} size={11} strokeWidth={2.5} />{c.label}
              </button>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10.5px] font-bold uppercase tracking-[0.07em] text-muted-foreground">Row height</span>
            <div className="inline-flex rounded-full border bg-secondary p-0.5">
              {(['comfortable', 'compact'] as const).map((d) => (
                <button key={d} onClick={() => onDensity(d)} aria-pressed={density === d}
                  className={cx('cursor-pointer rounded-full px-2.5 py-0.5 text-xs font-bold capitalize transition-colors',
                    density === d ? 'bg-card text-foreground shadow-card' : 'text-muted-foreground hover:text-foreground')}>{d}</button>
              ))}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-t bg-secondary/40 px-3.5 py-2 text-[10.5px] text-muted-foreground">
          <Icon name="keyboard" size={12} />
          {SHORTCUTS.map(([k, what]) => (
            <span key={k} className="inline-flex items-center gap-1"><kbd className="rounded border bg-card px-1 font-mono text-[10px] text-body">{k}</kbd>{what}</span>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
