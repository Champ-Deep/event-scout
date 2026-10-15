// /seats — Team & seats. Everything comes from the API:
//   list        GET    /api/admin/users              (contacts, hot, added today, last activity, profile photo / title / company / events)
//   add         POST   /api/admin/users
//   rename · role · password   PATCH /api/admin/users/{id}   (a role or password change signs them out everywhere)
//   delete      DELETE /api/admin/users/{id}         (reps only; their contacts go with them)
//   merge       POST   /api/admin/users/merge        (everything moves to the kept account, LinkedIn sign-in too)
// Search, role, tile filter and sort live in the URL (?q=…&role=admin&show=active&sort=active).
import React, { useEffect, useMemo, useState } from 'react';
import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { Link, useNavigate, useSearch } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import * as Popover from '@radix-ui/react-popover';
import { q as queries } from '../lib/queries';
import { patchInArrays, removeFromArrays, type Rollback } from '../lib/optimistic';
import { clean } from '../lib/search';
import { api } from '../lib/api';
import { applyChange, type Change } from '../lib/changes';
import type { AdminUser } from '../lib/types';
import { getSession } from '../lib/auth';
import { Icon, type IconName } from '../ui/icons';
import { Avatar } from '../ui/Avatar';
import { AnimatedNumber } from '../ui/micro';
import { MOTION } from '../ui/tokens';
import {
  Badge, Button, Card, EmptyState, ErrorState, Field, IconButton, Input, Modal, Notice, PageHeader, SearchInput, Segmented, Select,
  cx, errorMessage, fmtDateTime, relTime, useUi,
} from '../ui/kit';
import { IconTile, type Tint } from '../ui/drawerKit';

type Sort = 'contacts' | 'active' | 'newest' | 'name';
const SORTS: { value: Sort; label: string }[] = [
  { value: 'contacts', label: 'Most contacts' }, { value: 'active', label: 'Recently active' }, { value: 'newest', label: 'Newest' }, { value: 'name', label: 'Name A → Z' },
];
const DAY = 24 * 3600 * 1000;
const activeToday = (u: AdminUser) => !!u.lastActive && Date.now() - new Date(u.lastActive).getTime() < DAY;

function strongPassword() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789';
  const a = new Uint32Array(14);
  crypto.getRandomValues(a);
  return Array.from(a, (n) => chars[n % chars.length]).join('');
}

function MenuItem({ icon, label, onClick, danger, disabled, hint }: { icon: IconName; label: string; onClick: () => void; danger?: boolean; disabled?: boolean; hint?: string | undefined }) {
  return (
    <Popover.Close asChild>
      <button type="button" disabled={disabled} onClick={onClick} title={hint}
        className={cx('flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm cursor-pointer disabled:cursor-not-allowed disabled:opacity-40',
          danger ? 'text-rose-600 hover:bg-rose-50 dark:text-rose-400 dark:hover:bg-rose-500/10' : 'text-body hover:bg-secondary')}>
        <Icon name={icon} size={15} /> {label}
      </button>
    </Popover.Close>
  );
}

/** A summary tile: each one filters the list or opens the matching page; the active one is highlighted. */
function Tile({ icon, tint, label, value, i, onClick, active, hint, external }: {
  icon: IconName; tint: Tint; label: string; value: number; i: number; onClick: () => void; active?: boolean; hint: string; external?: boolean;
}) {
  return (
    <motion.button type="button" onClick={onClick} aria-pressed={!!active} title={hint}
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.04 + i * 0.04 }}
      whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
      className={cx('group flex min-w-[132px] flex-1 cursor-pointer items-center gap-2 rounded-xl border bg-card px-2.5 py-2 text-left shadow-card transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active ? 'border-brand-400 ring-2 ring-brand-400/25' : 'hover:border-brand-200')}>
      <IconTile icon={icon} tint={tint} size="sm" />
      <span className="min-w-0 flex-1">
        <span className="es-num block text-lg font-bold leading-5 text-foreground"><AnimatedNumber value={value} duration={600} /></span>
        <span className={cx('block whitespace-nowrap text-[11px] font-medium leading-4', active ? 'text-brand-700' : 'text-muted-foreground')}>{label}</span>
      </span>
      <Icon name={external ? 'external' : 'chevronRight'} size={13}
        className={cx('shrink-0 transition-all', active ? 'text-brand-600' : 'text-muted-foreground/0 group-hover:translate-x-0.5 group-hover:text-muted-foreground')} />
    </motion.button>
  );
}

/** One person: photo, name, title, what they captured, where they are, when they were last active. */
function PersonRow({ u, i, me, maxContacts, onAction }: {
  u: AdminUser; i: number; me: string | undefined; maxContacts: number;
  onAction: (a: 'rename' | 'password' | 'role' | 'delete') => void;
}) {
  const navigate = useNavigate();
  const today = activeToday(u);
  const sub = [u.title, u.company].filter(Boolean).join(' · ');
  return (
    <motion.li layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: 30 }}
      transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(i, 14) * 0.02 }}
      className="group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2 border-b px-4 py-3 last:border-0 hover:bg-secondary/40 lg:grid-cols-[minmax(0,2.2fr)_minmax(0,1.3fr)_minmax(0,1.2fr)_minmax(0,1fr)_auto]">
      {/* who */}
      <div className="flex min-w-0 items-center gap-3">
        <span className="relative shrink-0">
          <Avatar src={u.photoUrl || null} name={u.name} size="md" />
          <span className={cx('absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full ring-2 ring-card', today ? 'bg-emerald-500' : u.lastActive ? 'bg-slate-300 dark:bg-slate-600' : 'bg-transparent ring-0')}
            title={today ? 'Added a contact in the last 24 hours' : undefined} />
        </span>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="break-words text-sm font-semibold text-foreground">{u.name}</span>
            {u.id === me ? <Badge tone="violet">You</Badge> : null}
            {u.isAdmin ? <Badge tone="violet" icon="shield">Admin</Badge> : <Badge>Rep</Badge>}
            {u.linkedin ? <Badge tone="sky" icon="link">LinkedIn</Badge> : null}
          </div>
          {sub ? <div className="break-words text-xs text-body">{sub}</div> : null}
          <a href={`mailto:${u.email}`} className="break-all text-xs text-muted-foreground hover:text-brand-700 hover:underline">{u.email}</a>
        </div>
      </div>

      {/* what they captured */}
      <div className="col-span-2 min-w-0 lg:col-span-1">
        <div className="flex items-baseline gap-2 text-xs text-muted-foreground">
          <button type="button" onClick={() => navigate({ to: '/contacts', search: { owner: u.id } as never })} disabled={!u.contactsCount}
            className="es-num text-base font-bold text-foreground enabled:cursor-pointer enabled:hover:text-brand-700 enabled:hover:underline disabled:cursor-default"
            title={u.contactsCount ? `Open ${u.name}'s contacts` : undefined}>{u.contactsCount.toLocaleString()}</button>
          contact{u.contactsCount === 1 ? '' : 's'}
          {u.hotCount ? <button type="button" onClick={() => navigate({ to: '/contacts', search: { owner: u.id, band: 'A' } as never })}
            className="cursor-pointer font-semibold text-heat-hot hover:underline" title={`Open ${u.name}'s hot leads`}>· {u.hotCount} hot</button> : null}
          {u.scansToday ? <span className="font-semibold text-emerald-700 dark:text-emerald-400">· +{u.scansToday} today</span> : null}
        </div>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-secondary">
          <motion.div className="h-full rounded-full bg-brand-500" initial={{ width: 0 }} animate={{ width: `${maxContacts ? (u.contactsCount / maxContacts) * 100 : 0}%` }}
            transition={{ duration: 0.6, ease: MOTION.ease, delay: 0.1 + Math.min(i, 14) * 0.02 }} />
        </div>
      </div>

      {/* where */}
      <div className="min-w-0 text-xs">
        {u.currentEvent ? (
          <Link to="/exhibitors" search={{ event: u.currentEvent } as never} title={`Open ${u.currentEvent}`}
            className="inline-flex max-w-full items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 font-semibold text-brand-700 hover:bg-brand-100 dark:bg-brand-100 dark:text-brand-800">
            <Icon name="calendar" size={11} className="shrink-0" /><span className="truncate">{u.currentEvent}</span>
          </Link>
        ) : <span className="text-muted-foreground">No current event</span>}
        {u.events && u.events.length > 1 ? <div className="mt-0.5 text-[11px] text-muted-foreground" title={u.events.join(', ')}>{u.events.length} events listed</div> : null}
      </div>

      {/* when */}
      <div className="min-w-0 text-xs text-muted-foreground">
        <div className={today ? 'font-semibold text-emerald-700 dark:text-emerald-400' : ''} title={u.lastActive ? fmtDateTime(u.lastActive) : undefined}>
          {u.lastActive ? `Active ${relTime(u.lastActive)}` : 'No contacts yet'}
        </div>
        <div title={u.createdAt ? fmtDateTime(u.createdAt) : undefined}>Joined {u.createdAt ? relTime(u.createdAt) : '—'}</div>
      </div>

      {/* actions */}
      <div className="row-start-1 flex justify-end lg:row-start-auto">
        <Popover.Root>
          <Popover.Trigger asChild><IconButton icon="more" label={`Actions for ${u.name}`} /></Popover.Trigger>
          <Popover.Portal>
            <Popover.Content align="end" sideOffset={4} className="es-fade z-50 w-60 rounded-xl border border-border bg-card p-1.5 shadow-xl">
              <MenuItem icon="contacts" label="View their contacts" disabled={!u.contactsCount} onClick={() => navigate({ to: '/contacts', search: { owner: u.id } as never })} />
              {u.currentEvent ? <MenuItem icon="calendar" label={`Open ${u.currentEvent}`} onClick={() => navigate({ to: '/exhibitors', search: { event: u.currentEvent } as never })} /> : null}
              <div className="my-1 h-px bg-border" />
              <MenuItem icon="pencil" label="Rename" onClick={() => onAction('rename')} />
              <MenuItem icon="key" label="Set a new password" onClick={() => onAction('password')} hint="They're signed out everywhere and sign in with the new one" />
              <MenuItem icon="shield" label={u.isAdmin ? 'Make rep' : 'Make admin'} disabled={u.id === me} onClick={() => onAction('role')}
                hint={u.id === me ? "You can't change your own role" : undefined} />
              <MenuItem icon="trash" label="Delete account" danger disabled={u.id === me || u.isAdmin} onClick={() => onAction('delete')}
                hint={u.id === me ? "You can't delete yourself" : u.isAdmin ? 'Make them a rep first' : undefined} />
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      </div>
    </motion.li>
  );
}

export function SeatsComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const me = getSession()?.userId;
  const q = useSuspenseQuery(queries.users());
  const search = useSearch({ from: '/_auth/seats' });
  const navigate = useNavigate();
  const role: 'all' | 'rep' | 'admin' = search.role ?? 'all';
  const show = search.show;
  const sort: Sort = search.sort ?? 'contacts';
  const go = (patch: Record<string, string | undefined>) => navigate({ to: '/seats', search: (s) => clean({ ...s, ...patch }) as never, replace: true });
  const [text, setText] = useState(search.q ?? '');
  useEffect(() => {
    const id = setTimeout(() => { if ((search.q ?? '') !== text.trim()) go({ q: text.trim() || undefined }); }, 300);
    return () => clearTimeout(id);
  }, [text]);

  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ name: '', email: '', password: '', isAdmin: false });
  const [renaming, setRenaming] = useState<AdminUser | null>(null);
  const [newName, setNewName] = useState('');
  const [pwFor, setPwFor] = useState<AdminUser | null>(null);
  const [newPw, setNewPw] = useState('');
  const [merging, setMerging] = useState(false);
  const [merge, setMerge] = useState({ source: '', target: '' });
  const [busy, setBusy] = useState(false);

  const users = q.data || [];
  const admins = users.filter((u) => u.isAdmin).length;
  const activeNow = users.filter(activeToday).length;
  const idle = users.filter((u) => !u.contactsCount).length;
  const totalContacts = users.reduce((n, u) => n + u.contactsCount, 0);
  const totalHot = users.reduce((n, u) => n + (u.hotCount ?? 0), 0);
  const maxContacts = Math.max(0, ...users.map((u) => u.contactsCount));
  const shown = useMemo(() => {
    const needle = text.trim().toLowerCase();
    const list = users.filter((u) => (role === 'all' || u.role === role)
      && (show !== 'active' || activeToday(u)) && (show !== 'idle' || !u.contactsCount)
      && (!needle || [u.name, u.email, u.title, u.company, u.currentEvent].join(' ').toLowerCase().includes(needle)));
    const t = (s?: string | null) => (s ? new Date(s).getTime() : 0);
    return [...list].sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name)
      : sort === 'newest' ? t(b.createdAt) - t(a.createdAt)
        : sort === 'active' ? t(b.lastActive) - t(a.lastActive) || b.contactsCount - a.contactsCount
          : b.contactsCount - a.contactsCount || t(b.lastActive) - t(a.lastActive));
  }, [users, role, show, text, sort]);
  // deleting / merging / renaming a rep moves or relabels their contacts: lib/changes refreshes every contact-derived view too
  const run = async (change: Change, fn: () => Promise<unknown>, ok: string, fail: string, detail?: (r: any) => string | undefined) => {
    setBusy(true);
    try {
      const r = await fn();
      toast('success', ok, detail?.(r));
      void applyChange(qc, change);
      return true;
    } catch (e) {
      toast('error', fail, errorMessage(e));
      return false;
    } finally {
      setBusy(false);
    }
  };
  // optimistic edits: the list changes at once and is restored exactly if the server rejects
  const runOptimistic = async (change: Change, patch: () => Promise<Rollback>, fn: () => Promise<unknown>, ok: string, fail: string) => {
    const rollback = await patch();
    try {
      await fn();
      toast('success', ok);
      return true;
    } catch (e) {
      rollback();
      toast('error', `${fail} — restored`, errorMessage(e));
      return false;
    } finally {
      void applyChange(qc, change);
    }
  };

  const create = async () => {
    if (await run('user_created', () => api.createUser(form), `${form.name} can now sign in`, "Couldn't create the account")) {
      setCreating(false);
      setForm({ name: '', email: '', password: '', isAdmin: false });
    }
  };
  const action = async (u: AdminUser, a: 'rename' | 'password' | 'role' | 'delete') => {
    if (a === 'rename') { setRenaming(u); setNewName(u.name); return; }
    if (a === 'password') { setPwFor(u); setNewPw(strongPassword()); return; }
    if (a === 'role') {
      const ok = await confirm({
        title: u.isAdmin ? `Make ${u.name} a rep?` : `Make ${u.name} an admin?`,
        message: u.isAdmin ? 'They lose access to this console and are signed out.' : 'They get full access to this console, including every contact. They need to sign in again.',
        confirmLabel: u.isAdmin ? 'Make rep' : 'Make admin',
      });
      if (ok) await runOptimistic('user_updated', () => patchInArrays<AdminUser>(qc, ['admin', 'users'], u.id, { isAdmin: !u.isAdmin, role: u.isAdmin ? 'rep' : 'admin' }),
        () => api.updateUser(u.id, { isAdmin: !u.isAdmin }), u.isAdmin ? `${u.name} is now a rep` : `${u.name} is now an admin`, "Couldn't change the role");
      return;
    }
    const ok = await confirm({
      title: `Delete ${u.name}'s account?`,
      message: <>This permanently deletes the account and <b>all {u.contactsCount} of their contacts</b>, files and settings. It can't be undone. To keep their contacts, merge the account into someone else's instead.</>,
      confirmLabel: 'Delete account', danger: true,
    });
    if (ok) await runOptimistic('user_deleted', () => removeFromArrays<AdminUser>(qc, ['admin', 'users'], [u.id]), () => api.deleteUser(u.id), `${u.name}'s account was deleted`, "Couldn't delete the account");
  };
  const savePassword = async () => {
    if (!pwFor) return;
    const u = pwFor;
    if (await run('user_updated', () => api.updateUser(u.id, { password: newPw }), `New password set for ${u.name}`, "Couldn't set the password",
      () => 'They were signed out everywhere. Share the new password with them securely.')) setPwFor(null);
  };
  const src = users.find((u) => u.id === merge.source);
  const tgt = users.find((u) => u.id === merge.target);
  const doMerge = async () => {
    if (!src || !tgt) return;
    // no second "Merge X into Y?" prompt: this dialog already is the confirmation (what happens,
    // "This can't be undone", a red Merge button) — asking again was a double prompt
    if (await run('user_deleted', () => api.mergeUsers(src.id, tgt.id), `Merged into ${tgt.name}`, "Couldn't merge", (r) => r?.message)) {
      setMerging(false);
      setMerge({ source: '', target: '' });
    }
  };

  return (
    <>
      <PageHeader title="Team & seats" description="Everyone who can sign in: reps use the mobile app, admins use this console. See what each person captured, where they are, and manage their account."
        actions={<>
          <Button icon="merge" onClick={() => setMerging(true)} disabled={users.length < 2}>Merge accounts</Button>
          <Button variant="primary" icon="userPlus" onClick={() => { setForm({ name: '', email: '', password: strongPassword(), isAdmin: false }); setCreating(true); }}>Add user</Button>
        </>} />

      {q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : (
        <>
          {/* one row of tiles: each filters the list or opens the matching page */}
          <div className="no-scrollbar -mx-1 mb-3 flex gap-2 overflow-x-auto px-1 py-1">
            <Tile i={0} icon="users" tint="violet" label="Everyone" value={users.length} hint="Show everyone" active={role === 'all' && !show} onClick={() => go({ role: undefined, show: undefined })} />
            <Tile i={1} icon="smartphone" tint="sky" label="Reps" value={users.length - admins} hint="Show the reps (mobile app)" active={role === 'rep' && !show} onClick={() => go({ role: 'rep', show: undefined })} />
            <Tile i={2} icon="shield" tint="violet" label="Admins" value={admins} hint="Show the admins (this console)" active={role === 'admin' && !show} onClick={() => go({ role: 'admin', show: undefined })} />
            <Tile i={3} icon="bolt" tint="emerald" label="Active today" value={activeNow} hint="People who added a contact in the last 24 hours" active={show === 'active'}
              onClick={() => go({ show: show === 'active' ? undefined : 'active', role: undefined })} />
            <Tile i={4} icon="clock" tint="slate" label="No contacts yet" value={idle} hint="People who haven't added a contact yet" active={show === 'idle'}
              onClick={() => go({ show: show === 'idle' ? undefined : 'idle', role: undefined })} />
            <Tile i={5} icon="contacts" tint="violet" label="Contacts" value={totalContacts} hint="Open every contact" external onClick={() => navigate({ to: '/contacts' })} />
            <Tile i={6} icon="flame" tint="rose" label="Hot leads" value={totalHot} hint="Open the hot leads (score 85+)" external onClick={() => navigate({ to: '/contacts', search: { band: 'A' } as never })} />
          </div>

          <Card className="overflow-hidden">
            <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-3">
              <SearchInput value={text} onChange={setText} placeholder="Search name, email, title, company, event…" className="w-full sm:w-80" />
              <Segmented value={role} onChange={(r) => go({ role: r === 'all' ? undefined : r })} options={[
                { value: 'all', label: 'Everyone', count: users.length }, { value: 'rep', label: 'Reps', count: users.length - admins }, { value: 'admin', label: 'Admins', count: admins },
              ]} />
              <AnimatePresence>
                {show ? (
                  <motion.button key={show} type="button" onClick={() => go({ show: undefined })} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
                    className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-full bg-emerald-600 pl-2.5 pr-1.5 text-xs font-bold text-white">
                    {show === 'active' ? 'Active today' : 'No contacts yet'}
                    <span className="flex size-5 items-center justify-center rounded-full bg-white/20"><Icon name="x" size={11} strokeWidth={3} /></span>
                  </motion.button>
                ) : null}
              </AnimatePresence>
              <div className="ml-auto flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{shown.length} of {users.length}</span>
                <Select value={sort} onChange={(e) => go({ sort: e.target.value === 'contacts' ? undefined : e.target.value })} className="h-8 w-44 text-xs" aria-label="Sort">
                  {SORTS.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
                </Select>
              </div>
            </div>
            <div className="hidden grid-cols-[minmax(0,2.2fr)_minmax(0,1.3fr)_minmax(0,1.2fr)_minmax(0,1fr)_auto] gap-x-4 border-b bg-secondary/50 px-4 py-2 text-[10.5px] font-semibold uppercase tracking-[0.06em] text-muted-foreground lg:grid">
              <span>Person</span><span>Contacts captured</span><span>Current event</span><span>Activity</span><span className="w-8" />
            </div>
            {!shown.length ? (
              <EmptyState icon="users" title="No one matches" description="Try another search or filter."
                action={<Button icon="x" onClick={() => { setText(''); go({ q: undefined, role: undefined, show: undefined }); }}>Clear filters</Button>} />
            ) : (
              <ul>
                <AnimatePresence initial={true}>
                  {shown.map((u, i) => <PersonRow key={u.id} u={u} i={i} me={me} maxContacts={maxContacts} onAction={(a) => action(u, a)} />)}
                </AnimatePresence>
              </ul>
            )}
          </Card>
        </>
      )}

      <Modal open={creating} onOpenChange={setCreating} title="Add a user" description="They sign in to the app (or this console, for admins) with this email and password."
        footer={<>
          <Button variant="ghost" onClick={() => setCreating(false)}>Cancel</Button>
          <Button variant="primary" loading={busy} disabled={!form.name.trim() || !form.email.trim() || form.password.length < 8} onClick={create}>Create account</Button>
        </>}>
        <div className="space-y-4">
          <Field label="Full name"><Input autoFocus value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="Email"><Input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></Field>
          <PasswordField value={form.password} onChange={(v) => setForm({ ...form, password: v })} />
          <Field label="Role">
            <Select value={form.isAdmin ? 'admin' : 'rep'} onChange={(e) => setForm({ ...form, isAdmin: e.target.value === 'admin' })}>
              <option value="rep">Rep — uses the mobile app</option>
              <option value="admin">Admin — full access to this console</option>
            </Select>
          </Field>
        </div>
      </Modal>

      <Modal open={!!renaming} onOpenChange={(o) => { if (!o) setRenaming(null); }} size="sm" title="Rename"
        footer={<>
          <Button variant="ghost" onClick={() => setRenaming(null)}>Cancel</Button>
          <Button variant="primary" disabled={!newName.trim()} onClick={() => {
            if (!renaming) return;
            const u = renaming;
            setRenaming(null);
            void runOptimistic('user_updated', () => patchInArrays<AdminUser>(qc, ['admin', 'users'], u.id, { name: newName.trim() }), () => api.updateUser(u.id, { name: newName }), 'Name updated', "Couldn't rename");
          }}>Save</Button>
        </>}>
        <Field label="Full name"><Input autoFocus value={newName} onChange={(e) => setNewName(e.target.value)} /></Field>
      </Modal>

      <Modal open={!!pwFor} onOpenChange={(o) => { if (!o) setPwFor(null); }} title={`Set a new password for ${pwFor?.name ?? ''}`}
        description="They're signed out on every device and sign in with this password. They can change it later with “Forgot password”."
        footer={<>
          <Button variant="ghost" onClick={() => setPwFor(null)}>Cancel</Button>
          <Button variant="primary" icon="key" loading={busy} disabled={newPw.length < 8} onClick={savePassword}>Set password</Button>
        </>}>
        <PasswordField value={newPw} onChange={setNewPw} />
      </Modal>

      <Modal open={merging} onOpenChange={setMerging} size="lg" title="Merge two accounts" description="For when one person ended up with two accounts (e.g. LinkedIn and email)."
        footer={<>
          <Button variant="ghost" onClick={() => setMerging(false)}>Cancel</Button>
          <Button variant="danger" icon="merge" loading={busy} disabled={!src || !tgt || src.id === tgt.id} onClick={doMerge}>Merge</Button>
        </>}>
        <div className="space-y-4">
          <div className="grid items-end gap-3 sm:grid-cols-[1fr_auto_1fr]">
            <Field label="Move everything from (removed)">
              <Select value={merge.source} onChange={(e) => setMerge({ ...merge, source: e.target.value })}>
                <option value="">Choose the account to remove…</option>
                {users.filter((u) => u.id !== me && u.id !== merge.target).map((u) => <option key={u.id} value={u.id}>{u.name} — {u.email} ({u.contactsCount})</option>)}
              </Select>
            </Field>
            <span className="mb-2 hidden h-8 w-8 items-center justify-center self-end rounded-full bg-secondary text-muted-foreground sm:flex"><Icon name="chevronRight" size={16} /></span>
            <Field label="Into (kept)">
              <Select value={merge.target} onChange={(e) => setMerge({ ...merge, target: e.target.value })}>
                <option value="">Choose the account to keep…</option>
                {users.filter((u) => u.id !== merge.source).map((u) => <option key={u.id} value={u.id}>{u.name} — {u.email} ({u.contactsCount})</option>)}
              </Select>
            </Field>
          </div>
          {src && tgt ? (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="rounded-xl border bg-secondary/40 p-3 text-sm text-body">
              <div className="mb-1.5 font-semibold text-foreground">What happens</div>
              <ul className="space-y-1 text-[13px]">
                <li className="flex gap-2"><Icon name="check" size={14} className="mt-0.5 shrink-0 text-emerald-600" />{src.name}'s <b>{src.contactsCount}</b> contact{src.contactsCount === 1 ? '' : 's'}, chats, Recently deleted, booth visits, share links and event team spots move to {tgt.name}.</li>
                {src.linkedin && !tgt.linkedin ? <li className="flex gap-2"><Icon name="check" size={14} className="mt-0.5 shrink-0 text-emerald-600" />Their LinkedIn sign-in moves too, so it opens {tgt.name}'s account.</li> : null}
                <li className="flex gap-2"><Icon name="check" size={14} className="mt-0.5 shrink-0 text-emerald-600" />{tgt.name} keeps their own profile, role and password{src.isAdmin && !tgt.isAdmin ? ` (${src.name} was an admin; ${tgt.name} stays a rep)` : ''}.</li>
                <li className="flex gap-2"><Icon name="trash" size={14} className="mt-0.5 shrink-0 text-rose-600" />{src.name}'s account ({src.email}) is removed.</li>
              </ul>
            </motion.div>
          ) : null}
          <Notice tone="amber" icon="alert">This can't be undone.</Notice>
        </div>
      </Modal>
    </>
  );
}

function PasswordField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { toast } = useUi();
  return (
    <Field label="Password" hint="At least 8 characters. Share it with them securely; they can change it with “Forgot password”."
      error={value && value.length < 8 ? 'At least 8 characters' : null}>
      <div className="flex gap-2">
        <Input value={value} onChange={(e) => onChange(e.target.value)} className="font-mono" />
        <Button type="button" icon="refresh" onClick={() => onChange(strongPassword())}>Generate</Button>
        <Button type="button" icon="copy" onClick={() => { navigator.clipboard.writeText(value).then(() => toast('success', 'Password copied'), () => toast('error', "Couldn't copy")); }}>Copy</Button>
      </div>
    </Field>
  );
}
