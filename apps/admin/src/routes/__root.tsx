import React, { useEffect, useState, useSyncExternalStore } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Outlet, Link, useLocation } from '@tanstack/react-router';
import { q } from '../lib/queries';
import { useEventStream } from '../features/live/useEventStream';
import { ActivityMenu } from '../features/activity/ActivityMenu';
import { useTheme } from '../lib/theme';
import { useMotionPref, type MotionPref } from '../lib/motion';
import { getSession, signOut, subscribeSession } from '../lib/auth';
import type { LiveEventMessage } from '../lib/sse';
import { motion } from 'motion/react';
import { Icon, type IconName } from '../ui/icons';
import { LiveIcon } from '../ui/live-icon';
import { CommandPalette, usePaletteHotkey } from '../features/palette/CommandPalette';
import { TodayGlance } from '../features/live/TodayGlance';
import { cx } from '../ui/kit';
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupLabel, SidebarHeader, SidebarInset,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarProvider, SidebarRail, SidebarTrigger, useSidebar,
} from '@/components/ui/sidebar';
import { Separator } from '@/components/ui/separator';
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

// tint: each page's own colour (the same one its page header uses, ui/kit PAGE_ICONS)
type NavLink = { to: string; label: string; icon: IconName; tint: string };
const NAV: { section: string; items: NavLink[] }[] = [
  { section: 'Overview', items: [
    { to: '/', label: 'Live board', icon: 'dashboard', tint: 'var(--brand-600)' },
    { to: '/activity', label: 'Activity', icon: 'activity', tint: '#0ea5e9' },
  ] },
  { section: 'Leads', items: [
    { to: '/contacts', label: 'Contacts', icon: 'contacts', tint: '#8b5cf6' },
    { to: '/duplicates', label: 'Duplicates', icon: 'duplicate', tint: '#f59e0b' },
    { to: '/deleted', label: 'Recently deleted', icon: 'trash', tint: '#f43f5e' },
    { to: '/lists', label: 'Lists', icon: 'list', tint: '#d946ef' },
    { to: '/pipelines', label: 'AI pipelines', icon: 'pipeline', tint: 'var(--brand-600)' },
    { to: '/sales-profile', label: 'Sales profile', icon: 'target', tint: '#10b981' },
  ] },
  { section: 'Event', items: [
    { to: '/exhibitors', label: 'Events', icon: 'store', tint: '#0ea5e9' },
    { to: '/files', label: 'Team files', icon: 'file', tint: '#14b8a6' },
    { to: '/broadcasts', label: 'Announcements', icon: 'megaphone', tint: '#f97316' },
  ] },
  { section: 'Team', items: [
    { to: '/seats', label: 'Team & seats', icon: 'users', tint: '#10b981' },
  ] },
  { section: 'Tools', items: [
    { to: '/assistant', label: 'AI assistant', icon: 'sparkles', tint: '#ec4899' },
    { to: '/exports', label: 'Exports', icon: 'download', tint: '#6366f1' },
    { to: '/system', label: 'System', icon: 'server', tint: '#64748b' },
  ] },
];

/** A live event in plain words. */
function describe(evt: LiveEventMessage): string {
  const d = (evt.data || {}) as Record<string, any>;
  switch (evt.type) {
    case 'contact_scanned': return `New contact: ${d.name || 'someone'}`;
    case 'score_updated': return d.score == null ? `${d.name || 'A contact'}'s score was cleared` : `${d.name || 'A contact'} scored ${d.score}`;
    case 'contact_updated': return `${d.name || 'A contact'} was updated`;
    case 'admin_score_updated': return `Admin score for ${d.name || 'a contact'}: ${d.score ?? ''}`;
    case 'contact_deleted': return `${d.name || 'A contact'} was deleted`;
    case 'contact_restored': return `${d.name || 'A contact'} was restored`;
    case 'contact_merged': return `Duplicate merged into ${d.name || 'a contact'}`;
    case 'batch_score_completed': return `Batch scoring finished (${d.scoredCount ?? 0} scored)`;
    case 'pipeline_started': return `Pipeline started for ${d.name || 'a contact'}`;
    case 'pipeline_finished': {
      const who = d.name || 'a contact';
      if (d.status === 'failed' || d.status === 'error') return `Pipeline failed for ${who}`;
      return `Pipeline finished for ${who}`;
    }
    case 'broadcast_alert':
      if (d.ended) return 'Announcement ended';
      return d.message ? `Announcement sent: ${String(d.message).slice(0, 60)}` : 'Announcement sent';
    case 'contact_purged': {
      const n = Array.isArray(d.ids) ? d.ids.length : 0;
      return n === 1 ? 'A deleted contact was permanently removed' : n > 1 ? `${n} deleted contacts were permanently removed` : 'Deleted contacts were permanently removed';
    }
    case 'files_changed': {
      const file = d.filename ?? d.name; // the server sends `filename`
      return d.deleted ? 'A team file was removed' : file ? `Team files updated: ${String(file).slice(0, 60)}` : 'Team files were updated';
    }
    case 'list_changed': return d.deleted ? 'A contact list was deleted' : `List ${d.name ? `"${d.name}" ` : ''}was updated`;
    case 'event_planned': return d.deleted ? 'A planned event was removed' : `Event plan updated: ${d.name || 'an event'}`;
    case 'industries_updated': return 'Industries re-sorted by AI';
    case 'user_created': return `${d.name || 'Someone'} was added to the team`;
    case 'user_updated': return `${d.name || 'A team member'}'s profile was updated`;
    case 'user_deleted': return `${d.name || 'An account'} was removed`;
    default: return evt.type.replace(/_/g, ' ');
  }
}

const isActivePath = (pathname: string, to: string) =>
  to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(`${to}/`);

/** Account menu → Animations: On · Off · Follow system (lib/motion.tsx). */
function AnimationsSetting() {
  const { pref, setPref, systemReduce } = useMotionPref();
  const opts: { value: MotionPref; label: string }[] = [{ value: 'on', label: 'On' }, { value: 'off', label: 'Off' }, { value: 'system', label: 'System' }];
  return (
    <div className="px-2 py-1.5" onKeyDown={(e) => e.stopPropagation()}>
      <div className="mb-1.5 flex items-center gap-2 text-xs font-medium text-muted-foreground"><Icon name="sparkles" size={13} />Animations</div>
      <div role="radiogroup" aria-label="Animations" className="grid grid-cols-3 gap-0.5 rounded-lg bg-secondary p-0.5">
        {opts.map((o) => (
          <button key={o.value} type="button" role="radio" aria-checked={pref === o.value} onClick={() => setPref(o.value)}
            className={cx('cursor-pointer rounded-md px-2 py-1 text-xs font-semibold transition-colors',
              pref === o.value ? 'bg-card text-foreground shadow-card' : 'text-muted-foreground hover:text-foreground')}>{o.label}</button>
        ))}
      </div>
      {pref === 'system' ? <p className="mt-1 text-[10.5px] text-muted-foreground">Your system asks for {systemReduce ? 'less motion, so animations are off' : 'full motion'}.</p> : null}
    </div>
  );
}

/**
 * When animations are switched off (the setting is Off, or "System" on a PC whose Windows "Show
 * animations" is off — Chrome then asks every site for less motion), say so once, with a one-click
 * "Turn on", instead of the console silently looking static. Hidden for the session when dismissed.
 */
function MotionOffNotice() {
  const { pref, setPref, reduce } = useMotionPref();
  const [hidden, setHidden] = useState(() => { try { return sessionStorage.getItem('es_motion_notice') === 'hidden'; } catch { return false; } });
  if (!reduce || hidden) return null;
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2 rounded-2xl border border-brand-200 bg-brand-50 px-3.5 py-2.5 text-sm text-brand-800" role="status">
      <Icon name="sparkles" size={15} className="shrink-0" />
      <span className="min-w-0 flex-1">
        Animations are off {pref === 'system' ? 'because Windows asks apps for less motion ("Show animations" is off)' : 'in your settings'}.
      </span>
      <button type="button" onClick={() => setPref('on')} className="cursor-pointer rounded-full bg-brand-600 px-3 py-1 text-xs font-bold text-white hover:bg-brand-700">Turn animations on</button>
      <button type="button" aria-label="Hide this notice" onClick={() => { setHidden(true); try { sessionStorage.setItem('es_motion_notice', 'hidden'); } catch { /* not remembered */ } }}
        className="flex size-7 cursor-pointer items-center justify-center rounded-full text-brand-700 hover:bg-brand-100"><Icon name="x" size={13} /></button>
    </div>
  );
}

/**
 * One page in the sidebar: its icon in a tile of the page's colour. The active tab gets a filled tile whose
 * live icon loops (ui/live-icon) and a highlight that slides between tabs; other icons play only while hovered.
 */
function NavItem({ it, active }: { it: NavLink; active: boolean }) {
  const [hover, setHover] = useState(false);
  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={active} tooltip={it.label}
        className="relative h-10 gap-3 rounded-xl px-1.5 hover:bg-transparent active:bg-transparent data-[active=true]:bg-transparent group-data-[collapsible=icon]:p-0.5!">
        <Link to={it.to} aria-current={active ? 'page' : undefined}
          onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)} onFocus={() => setHover(true)} onBlur={() => setHover(false)}>
          {active ? (
            <motion.span layoutId="es-nav-active" aria-hidden className="absolute inset-0 rounded-xl"
              style={{ background: `color-mix(in oklab, ${it.tint} 12%, var(--sidebar))`, boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${it.tint} 26%, transparent)` }}
              transition={{ type: 'spring', stiffness: 520, damping: 40 }} />
          ) : <span aria-hidden className="absolute inset-0 rounded-xl bg-sidebar-accent opacity-0 transition-opacity duration-200 group-hover/menu-item:opacity-100" />}
          <span className="relative flex size-7 shrink-0 items-center justify-center rounded-lg transition-[background-color,color,box-shadow,transform] duration-300 group-hover/menu-item:scale-105"
            // icon colour is the tint pulled toward the theme's text colour (darker in light, lighter in dark) so
            // bright tints (amber, green, sky) stay clearly visible on their pale tile; the active fill is a
            // deeper shade so its white icon keeps contrast
            style={active
              ? { background: `color-mix(in oklab, ${it.tint} 80%, #000)`, color: '#fff', boxShadow: `0 5px 14px -5px ${it.tint}` }
              : { background: `color-mix(in oklab, ${it.tint} 13%, transparent)`, color: `color-mix(in oklab, ${it.tint} 62%, var(--sidebar-foreground))` }}>
            <LiveIcon name={it.icon} size={16} play={active || hover} />
          </span>
          <span className={cx('relative min-w-0 flex-1 truncate transition-colors', active ? 'font-semibold text-sidebar-foreground' : 'text-sidebar-foreground/80 group-hover/menu-item:text-sidebar-foreground')}>{it.label}</span>
          {active ? (
            <span aria-hidden className="relative mr-1.5 flex size-1.5 group-data-[collapsible=icon]:hidden">
              <span className="absolute inline-flex size-full animate-ping rounded-full opacity-60" style={{ background: it.tint }} />
              <span className="relative inline-flex size-1.5 rounded-full" style={{ background: it.tint }} />
            </span>
          ) : null}
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

/** shadcn-style app sidebar: collapses to an icon rail (labels become tooltips), a sheet on mobile. */
function AppSidebar() {
  const { pathname } = useLocation();
  const { isMobile, setOpenMobile } = useSidebar();
  // re-renders when the name changes on My profile
  const session = useSyncExternalStore(subscribeSession, getSession);
  const me = useQuery(q.me()); // your photo, from your profile
  const photo = me.data?.photoUrl ?? null;
  const name = session?.name || 'Admin';
  const initials = (session?.name || session?.email || '?').split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  const avatar = (cls: string) => (photo
    ? <img src={photo} alt="" className={cx('size-8 shrink-0 rounded-md object-cover', cls)} />
    : <span className={cx('flex size-8 shrink-0 items-center justify-center rounded-md font-mono text-xs font-semibold', cls)}>{initials}</span>);
  useEffect(() => { setOpenMobile(false); }, [pathname, setOpenMobile]);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border/70">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild tooltip="EventScout admin console" className="hover:bg-transparent">
              <Link to="/">
                <div className="relative flex aspect-square size-8 items-center justify-center rounded-xl text-white shadow-[0_6px_16px_-6px_#7c3aed]"
                  style={{ background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)' }}>
                  <Icon name="bolt" size={16} strokeWidth={2.25} />
                </div>
                <div className="relative grid flex-1 text-left leading-tight">
                  <span className="truncate font-display text-[16px] font-bold tracking-[-0.02em] text-sidebar-foreground">EventScout</span>
                  <span className="es-eyebrow truncate text-[10px] text-sidebar-muted">Floor control</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent className="gap-0 pt-1">
        {NAV.map((g) => (
          <SidebarGroup key={g.section} className="py-1">
            {/* collapsed: the label only fades out and slides up over the previous section's last icon, so it must not catch clicks */}
            <SidebarGroupLabel className="es-eyebrow gap-2 text-[10px] text-sidebar-muted after:h-px after:flex-1 after:bg-sidebar-border after:content-[''] group-data-[collapsible=icon]:pointer-events-none">{g.section}</SidebarGroupLabel>
            <SidebarMenu className="gap-0.5">
              {g.items.map((it) => <NavItem key={it.to} it={it} active={isActivePath(pathname, it.to)} />)}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border/70">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton size="lg" tooltip={name} className="rounded-xl border border-sidebar-border bg-sidebar-accent/40 hover:bg-sidebar-accent data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground group-data-[collapsible=icon]:border-0 group-data-[collapsible=icon]:bg-transparent">
                  {avatar('bg-sidebar-accent text-sidebar-mark ring-1 ring-sidebar-border')}
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold text-sidebar-foreground">{name}</span>
                    <span className="truncate text-xs text-sidebar-muted">{session?.email}</span>
                  </div>
                  <Icon name="chevronsUpDown" size={16} className="ml-auto text-sidebar-muted" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="min-w-56 rounded-lg" side={isMobile ? 'bottom' : 'right'} align="end" sideOffset={4}>
                <DropdownMenuLabel className="p-0 font-normal">
                  <div className="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                    {avatar('bg-accent text-accent-foreground')}
                    <div className="grid flex-1 leading-tight">
                      <span className="truncate font-semibold">{name}</span>
                      <span className="truncate text-xs text-muted-foreground">{session?.email}</span>
                    </div>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild className="cursor-pointer">
                  <Link to="/account"><Icon name="user" size={16} /> My profile &amp; password</Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <AnimationsSetting />
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={signOut} className="cursor-pointer">
                  <Icon name="logout" size={16} /> Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}

export function RootComponent() {
  const location = useLocation();
  const { recentEvents, status } = useEventStream('/api/live');
  const { theme, toggleTheme } = useTheme();
  const [paletteOpen, setPaletteOpen] = usePaletteHotkey();

  return (
    <SidebarProvider className="bg-background text-foreground">
      <AppSidebar />

      <SidebarInset>
        {/* Top bar */}
        {/* solid (no see-through blur): what scrolls under the bar never shows through it */}
        <header className="sticky top-0 z-20 flex h-12 items-center gap-2.5 border-b bg-background px-3 sm:px-4 lg:px-5">
          <SidebarTrigger className="-ml-1 text-muted-foreground hover:text-foreground cursor-pointer" />
          <Separator orientation="vertical" className="mr-1 data-[orientation=vertical]:h-4" />
          <button onClick={() => setPaletteOpen(true)} aria-label="Open command palette (Ctrl+K)"
            className="inline-flex h-8 shrink-0 items-center gap-2 rounded-lg border bg-card px-2.5 text-xs font-medium text-muted-foreground transition hover:border-foreground/25 hover:text-foreground cursor-pointer sm:w-64">
            <Icon name="search" size={14} />
            <span className="hidden flex-1 text-left sm:block">Search or jump to…</span>
            <kbd className="hidden items-center gap-0.5 rounded border bg-secondary px-1 font-mono text-[10px] font-medium text-body sm:inline-flex"><Icon name="command" size={10} />K</kbd>
          </button>

          {/* Today, live */}
          <TodayGlance events={recentEvents} describe={describe} />
          <div className="flex-1 md:hidden" />

          {/* the activity menu with its unread count (the live feed keeps running; no status badge) */}
          <ActivityMenu status={status} />

          <button onClick={toggleTheme} className="flex h-8 w-8 items-center justify-center rounded-lg border bg-card text-muted-foreground hover:text-foreground cursor-pointer"
            aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'} title={theme === 'light' ? 'Dark mode' : 'Light mode'}>
            <Icon name={theme === 'light' ? 'moon' : 'sun'} size={18} />
          </button>
        </header>

        {/* SidebarInset is the page's <main>; this is just the content column. */}
        <div key={location.pathname.split('/')[1] || 'home'} className="es-rise mx-auto w-full max-w-[1680px] px-3 py-3 sm:px-4 lg:px-5">
          <MotionOffNotice />
          <Outlet />
        </div>
      </SidebarInset>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} pages={NAV.flatMap((g) => g.items)} />
    </SidebarProvider>
  );
}
