# EventScout Admin — DESIGN.md

Tokens live in `src/styles/tokens.css`. This file is the contract; the CSS is the source of truth for values.

## 1. Visual theme & atmosphere

**The EventScout app, on a desk.** The console uses the reps' mobile app theme (`apps/mobile/src/ui/theme.ts`)
so both products read as one: violet brand, a lavender-tinted page, white cards with soft 18 px corners, slate
text, tinted icon tiles, and the same lead-level colours, score rings and motion. Admins switch between the two
all day; a contact must look the same in both.

The memorable things are the app's: score rings that draw themselves, heat-coloured avatar rings, tinted icon
tiles, and springy presses.

## 2. Color palette & roles

**Theme rule (from the product owner): in light mode every surface is light, in dark mode every surface is
dark.** No always-dark sidebar, band, tooltip, chip or bubble in light mode, and no light surface in dark mode.

| Role | Light | Dark | Use |
|---|---|---|---|
| background (page) | `#f8f6fe` lavender | `#0d0c14` | the app's `bg.app` |
| card | `#ffffff` | `#17161f` | the app's `bg.surface` |
| secondary | `#f5f3fb` | `#201f2b` | pressed / table header / tracks |
| border | `#e7e3f3` | `#2c2a3a` | hairlines (`bg.divider`, lavender-shifted) |
| foreground | `#0f172a` | `#f4f4f8` | headings, numbers (`text.display`) |
| body | `#475569` | `#c3c6d4` | paragraphs (`text.body`) |
| muted-foreground | `#64748b` | `#9da1b6` | captions (darker than the app's `#94a3b8` so it passes AA) |
| primary | `#7c3aed` | `#a78bfa` | links, rings, active text (`brand.600`) |
| fill | `#6d28d9` | `#6d28d9` | solid buttons, white text (`fill.brand700`) |
| signal | `#7c3aed` | `#8b5cf6` | the one CTA, "New" tags — white text |
| accent | `#f5f3ff` / `#6d28d9` | `#1a1430` / `#c4b5fd` | chips, active nav (`brand.50` + `brand.700`) |
| panel | `#f3effe` | `#1a1430` | highlighted bands (floor status, login hero, drawer hero) |
| live | `#10b981` | `#34d399` | live dots (the app's emerald) |

**Lead levels** (the app's `scoreLevel.ts`): hot `#ef4444` · warm `#f59e0b` · cool `#0ea5e9` · cold `#94a3b8`
(`--heat-*`, rings / bars / dots); chips use the app's pale fills with dark text (`bg-red-100 text-red-700` …).
Icon tiles use one hue per kind of data: designation violet, company sky, event amber, phone emerald,
LinkedIn `#0a66c2`.

60-30-10: lavender and white neutrals, slate text for structure, violet for actions and focus.

**Motion** (the app's Reanimated patterns, done with `motion`): lists fade up with a 30–50 ms stagger; cards and
buttons lift on hover and squeeze on press (`whileTap` 0.96–0.985); badges zoom in with a spring; score rings
draw to their value; icon tiles tilt on hover; live dots ping. All of it is off under `prefers-reduced-motion`
(`<MotionConfig reducedMotion="user">`).

## 3. Typography

| Role | Family | Size / weight |
|---|---|---|
| Display, page titles, big numbers | Bricolage Grotesque | 30–44px, 600–700, tracking −0.025em |
| UI and body | Geist | 13–15px, 450 base (crisp on Windows), 500–600 for labels |
| Eyebrows, table heads, badges, figures in lists | Geist Mono | 10–11px, 500, caps, +0.06–0.08em |

Helpers: `.es-num` (display face, tabular figures), `.es-eyebrow` (mono caps), `.es-eyebrow-mark` (adds the
signal square). No Inter, no font-black.

## 4. Component stylings

- **Buttons:** 8px radius, semibold. Primary = `--fill`. Secondary = card + input border. `ShineButton` is the
  single signal-lime CTA on a page. No pills for actions.
- **Cards:** 12px radius, 1px border, `shadow-card` hairline lift. Hover on clickable cards firms the border
  (`border-foreground/25`); no glow, no lift-and-bloom.
- **Card headers:** title + caption. No rounded icon tile above or beside headings.
- **Grouped figures:** one surface split by hairlines (`gap-px bg-border` grid), not a row of floating cards.
- **Badges / score chips:** 6px radius, mono caps label, tinted by tone or heat.
- **Tables:** header row on `secondary`, mono caps 10.5px.
- **Sidebar:** ink, mono section labels, active item = accent fill + signal icon. Collapses to icons.
- **Charts:** hairline grid, mono axis labels, ink tooltip, colours from `--chart-*` / `--heat-*` via style.

## 5. Layout principles

4px grid. Page column max 1400px, 16–32px gutters. Sections 20px apart; cards pad 20px. Headers are
left-aligned with actions on the right. Tight inside groups, generous between them.

## 6. Depth & elevation

Flat paper. `shadow-card` (1px hairline + 3px soft) for surfaces, `shadow-active` for hover on clickable
surfaces, `shadow-lg/xl` only for floating menus, dialogs and tooltips. The sticky top bar is the only
translucent surface (it floats over scrolling content) with a solid fallback.

## 7. Do's and don'ts

- Do keep signal lime rare and always pair it with ink text.
- Do use the heat ramp only for lead levels.
- Don't use gradients, gradient text, coloured glows, glass cards, or side-stripe borders.
- Don't put icon tiles on every heading or stat.
- Don't use pure white or pure black surfaces.

## 8. Responsive behaviour

Device class: pointer surface (desktop-first web), usable down to 360px. Sidebar becomes a sheet below
768px; ledger strip goes 4 → 2 → 1 columns. Touch targets ≥ 36px visual / 44px hit on mobile.
Motion budget: transform/opacity only, 150–300ms, `cubic-bezier(.16,1,.3,1)`; everything off under
`prefers-reduced-motion`. No expressive-tier effects.

## 9. Agent prompt guide

"Warm paper (`bg-background`, `bg-card`), ink text, hairline borders, 12px cards, 8px controls. Titles in
`font-display`, numbers in `.es-num`, labels in `.es-eyebrow`. One `ShineButton` (signal lime) per page.
Lead levels use `--heat-*`. No gradients, glows or icon tiles."
