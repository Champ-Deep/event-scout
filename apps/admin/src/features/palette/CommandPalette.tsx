import React, { useEffect, useState } from 'react';
import { Command } from 'cmdk';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import { api } from '../../lib/api';
import { q } from '../../lib/queries';
import { applyChange } from '../../lib/changes';
import { useEventNames } from '../../lib/eventNames';
import { signOut } from '../../lib/auth';
import { useTheme } from '../../lib/theme';
import { Icon, type IconName } from '../../ui/icons';
import { ScoreBadge, contactName, errorMessage, useUi } from '../../ui/kit';

export type PaletteLink = { to: string; label: string; icon: IconName };

const item = 'flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-body data-[selected=true]:bg-accent data-[selected=true]:text-brand-800';
const group = '[&_[cmdk-group-heading]]:px-2.5 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:pt-2.5 [&_[cmdk-group-heading]]:text-[11px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted-foreground';

/** ⌘K / Ctrl+K: jump to a page, run an action, open a contact, or filter contacts by rep or event. */
export function CommandPalette({ open, onOpenChange, pages }: { open: boolean; onOpenChange: (o: boolean) => void; pages: PaletteLink[] }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { toast } = useUi();
  const { theme, toggleTheme } = useTheme();
  const [text, setText] = useState('');
  const [term, setTerm] = useState('');
  useEffect(() => { if (!open) { setText(''); setTerm(''); } }, [open]);
  useEffect(() => { const id = setTimeout(() => setTerm(text.trim()), 200); return () => clearTimeout(id); }, [text]);

  // reps and events usually sit in the cache already; contacts are searched on the server as you type
  const users = useQuery({ ...q.users(), enabled: open });
  const eventNames = useEventNames(open);
  const found = useQuery({ ...q.contacts({ search: term, limit: 8, lite: 1 }), enabled: open && term.length >= 2 });

  // contact hits arrive after cmdk has already highlighted a page; move the highlight to the first hit
  const contactValue = (c: { id: string; firstName?: string | null; lastName?: string | null }) => `${term} ${contactName(c)} ${c.id}`;
  const [selected, setSelected] = useState('');
  const first = found.data?.contacts[0];
  useEffect(() => { if (first && term.length >= 2) setSelected(contactValue(first)); }, [first?.id, term]);

  const run = (fn: () => unknown) => { onOpenChange(false); fn(); };
  const go = (to: string, search?: Record<string, string>) => run(() => navigate({ to, ...(search ? { search } : {}) } as never));

  const actions: { label: string; icon: IconName; keywords: string[]; fn: () => unknown }[] = [
    { label: 'Add a contact', icon: 'plus', keywords: ['new', 'create', 'lead'], fn: () => navigate({ to: '/contacts', search: { add: '1' } } as never) },
    { label: 'Send an announcement', icon: 'megaphone', keywords: ['broadcast', 'message', 'alert'], fn: () => navigate({ to: '/broadcasts' }) },
    { label: 'Export contacts', icon: 'download', keywords: ['csv', 'download'], fn: () => navigate({ to: '/exports' }) },
    { label: 'My profile & password', icon: 'user', keywords: ['account', 'me', 'settings', 'password', 'photo', 'name'], fn: () => navigate({ to: '/account' }) },
    { label: 'Sales profile (admin AI)', icon: 'target', keywords: ['products', 'services', 'sell', 'pitch', 'sales approach', 'targeting', 'icp', 'score'], fn: () => navigate({ to: '/sales-profile' }) },
    {
      label: 'Back up the database now', icon: 'database', keywords: ['backup', 'snapshot'],
      // same handling as System → Back up now: a partial / failed run is not a success
      fn: async () => {
        try {
          const r = await api.backupNow();
          if (r.status === 'partial') toast('info', 'Backup finished with problems', r.message);
          else if (r.status === 'error') toast('error', 'Backup failed', r.message);
          else toast('success', 'Backup complete', r.message);
        } catch (e) {
          toast('error', 'Backup failed', errorMessage(e));
        } finally {
          void applyChange(qc, 'backup_done');
        }
      },
    },
    { label: theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode', icon: theme === 'light' ? 'moon' : 'sun', keywords: ['theme', 'appearance'], fn: toggleTheme },
    { label: 'Sign out', icon: 'logout', keywords: ['logout', 'exit'], fn: signOut },
  ];

  return (
    <Command.Dialog open={open} onOpenChange={onOpenChange} label="Command palette" loop value={selected} onValueChange={setSelected}
      overlayClassName="es-fade fixed inset-0 z-50 bg-slate-950/40 backdrop-blur-[2px]"
      contentClassName="es-fade fixed left-1/2 top-[18%] z-50 w-[calc(100%-32px)] max-w-xl -translate-x-1/2 overflow-hidden rounded-2xl border border-border bg-card shadow-2xl">
      <div className="flex items-center gap-2.5 border-b border-border px-4">
        <Icon name="search" className="text-muted-foreground" />
        <Command.Input value={text} onValueChange={setText} placeholder="Search pages, actions, contacts, reps, events…"
          className="h-12 flex-1 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground" />
        <kbd className="rounded-md border border-border px-1.5 text-[10px] font-medium text-muted-foreground">Esc</kbd>
      </div>
      <Command.List className="max-h-[min(60vh,420px)] overflow-y-auto p-1.5">
        <Command.Empty className="px-3 py-8 text-center text-sm text-muted-foreground">{found.isFetching ? 'Searching…' : 'Nothing matches.'}</Command.Empty>

        {term.length >= 2 && found.data?.contacts.length ? (
          <Command.Group heading="Contacts" className={group} forceMount>
            {found.data.contacts.map((c) => (
              // the server already matched these: the value starts with the typed term so they rank first
              <Command.Item key={c.id} value={contactValue(c)} forceMount onSelect={() => go(`/contacts/${c.id}`)} className={item}>
                <Icon name="user" className="text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate">{contactName(c)}<span className="text-muted-foreground">{c.company ? ` · ${c.company}` : ''}</span></span>
                <ScoreBadge score={c.score} level={c.level} />
              </Command.Item>
            ))}
          </Command.Group>
        ) : null}

        <Command.Group heading="Go to" className={group}>
          {pages.map((p) => (
            <Command.Item key={p.to} value={`page ${p.label}`} onSelect={() => go(p.to)} className={item}>
              <Icon name={p.icon} className="text-muted-foreground" />{p.label}
            </Command.Item>
          ))}
        </Command.Group>

        <Command.Group heading="Actions" className={group}>
          {actions.map((a) => (
            <Command.Item key={a.label} value={a.label} keywords={a.keywords} onSelect={() => run(a.fn)} className={item}>
              <Icon name={a.icon} className="text-muted-foreground" />{a.label}
            </Command.Item>
          ))}
        </Command.Group>

        {text && users.data?.length ? (
          <Command.Group heading="Contacts by rep" className={group}>
            {users.data.map((u) => (
              <Command.Item key={u.id} value={`rep ${u.name} ${u.email}`} onSelect={() => go('/contacts', { owner: u.id })} className={item}>
                <Icon name="users" className="text-muted-foreground" />
                <span className="flex-1 truncate">{u.name}</span>
                <span className="text-xs tabular-nums text-muted-foreground">{u.contactsCount}</span>
              </Command.Item>
            ))}
          </Command.Group>
        ) : null}

        {text && eventNames.length ? (
          <Command.Group heading="Contacts at event" className={group}>
            {eventNames.map((name) => (
              <Command.Item key={name} value={`event ${name}`} onSelect={() => go('/contacts', { event: name })} className={item}>
                <Icon name="calendar" className="text-muted-foreground" />{name}
              </Command.Item>
            ))}
          </Command.Group>
        ) : null}
      </Command.List>
      <div className="flex items-center gap-3 border-t border-border px-4 py-2 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1"><Icon name="arrowUp" size={12} /><Icon name="arrowDown" size={12} />move</span>
        <span className="inline-flex items-center gap-1"><Icon name="enter" size={12} />open</span>
        <span className="ml-auto inline-flex items-center gap-1"><Icon name="command" size={12} />K to toggle</span>
      </div>
    </Command.Dialog>
  );
}

/** ⌘K / Ctrl+K toggles; returns [open, setOpen]. */
export function usePaletteHotkey(): [boolean, (o: boolean) => void] {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setOpen((o) => !o); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return [open, setOpen];
}
