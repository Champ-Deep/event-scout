// /account — My profile: the signed-in admin's own account, like the app's Me screen. All real API:
//   load     GET   /api/admin/me                (account + the profile the app's Me screen edits)
//   save     PATCH /api/admin/me                (name, title, company, website, LinkedIn, phone, signature, photo)
//   password POST  /api/admin/me/password       (current + new; other devices are signed out, this one gets new tokens)
//   forgot   POST  /auth/password/forgot → /auth/password/reset   (emailed 6-digit code, the app's flow)
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api } from '../lib/api';
import { q } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { requestPasswordReset, resetPasswordWithCode, updateSession } from '../lib/auth';
import type { MyAccount } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { MOTION } from '../ui/tokens';
import { Badge, Button, Card, ErrorState, Field, Input, Notice, PageHeader, Skeleton, Textarea, cx, errorMessage, fmtDateTime, useUi } from '../ui/kit';

type Profile = MyAccount['profile'];
const spring = { type: 'spring', stiffness: 420, damping: 28 } as const;
const FIELDS: { key: keyof Profile; label: string; icon: IconName; placeholder: string; type?: string; hint?: string }[] = [
  { key: 'full_name', label: 'Your name', icon: 'user', placeholder: 'e.g. Priya Raman', hint: 'Also the name your team sees on announcements and in People.' },
  { key: 'job_title', label: 'Title', icon: 'briefcase', placeholder: 'e.g. Head of Events' },
  { key: 'company_name', label: 'Company', icon: 'building', placeholder: 'e.g. Northwind' },
  { key: 'company_website', label: 'Company website', icon: 'globe', placeholder: 'northwind.com', type: 'url' },
  { key: 'linkedin_url', label: 'Your LinkedIn', icon: 'link', placeholder: 'linkedin.com/in/you', type: 'url' },
  { key: 'phone', label: 'Phone', icon: 'phone', placeholder: '+1 415 555 0142', type: 'tel' },
];
const MAX_PHOTO = 8 * 1024 * 1024;

/** 0–4 from length and variety, with a word for it. */
function strength(pw: string): { score: number; label: string; color: string } {
  let s = 0;
  if (pw.length >= 8) s++;
  if (pw.length >= 12) s++;
  if (/[a-z]/.test(pw) && /[A-Z]/.test(pw)) s++;
  if (/\d/.test(pw) && /[^A-Za-z0-9]/.test(pw)) s++;
  return [
    { score: 0, label: 'Too short', color: '#f43f5e' }, { score: 1, label: 'Weak', color: '#f97316' }, { score: 2, label: 'Fair', color: '#f59e0b' },
    { score: 3, label: 'Good', color: '#10b981' }, { score: 4, label: 'Strong', color: '#059669' },
  ][pw.length < 8 ? 0 : s]!;
}

function PasswordInput({ value, onChange, placeholder, autoComplete, autoFocus }: { value: string; onChange: (v: string) => void; placeholder: string; autoComplete: string; autoFocus?: boolean }) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative">
      <Input type={show ? 'text' : 'password'} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} autoComplete={autoComplete} autoFocus={autoFocus} maxLength={128} className="pr-10" />
      <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? 'Hide password' : 'Show password'} title={show ? 'Hide' : 'Show'}
        className="absolute right-1.5 top-1/2 flex size-7 -translate-y-1/2 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-secondary hover:text-foreground">
        <Icon name={show ? 'eye' : 'key'} size={14} />
      </button>
    </div>
  );
}

function StrengthBar({ pw }: { pw: string }) {
  const st = strength(pw);
  if (!pw) return null;
  return (
    <div className="mt-1.5 flex items-center gap-2">
      <div className="flex flex-1 gap-1">
        {[0, 1, 2, 3].map((i) => (
          <motion.span key={i} className="h-1.5 flex-1 rounded-full bg-border" animate={{ backgroundColor: i < Math.max(1, st.score) ? st.color : 'var(--border)' }} transition={{ duration: 0.25 }} />
        ))}
      </div>
      <motion.span key={st.label} initial={{ opacity: 0, x: 4 }} animate={{ opacity: 1, x: 0 }} className="w-16 text-right text-[11px] font-bold" style={{ color: st.color }}>{st.label}</motion.span>
    </div>
  );
}

function SectionCard({ icon, tint, title, subtitle, children, index, actions }: { icon: IconName; tint: string; title: string; subtitle: string; children: React.ReactNode; index: number; actions?: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.2, ease: MOTION.ease, delay: 0.08 + index * 0.06 }}>
      <Card className="overflow-hidden rounded-2xl">
        <div className="group flex items-center gap-3 border-b px-4 py-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-12" style={{ background: `color-mix(in oklab, ${tint} 14%, transparent)`, color: tint }}>
            <Icon name={icon} size={16} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[15px] font-extrabold text-foreground">{title}</div>
            <div className="text-xs text-muted-foreground">{subtitle}</div>
          </div>
          {actions}
        </div>
        {children}
      </Card>
    </motion.div>
  );
}

export function AccountComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const me = useQuery(q.me());
  const data = me.data;

  const setMe = (next: MyAccount) => {
    qc.setQueryData(q.me().queryKey, next);
    updateSession({ name: next.name });
    void applyChange(qc, 'user_updated'); // People shows the new name / photo
  };

  if (me.error) return (<><PageHeader title="My profile" icon="user" compact /><ErrorState error={me.error} onRetry={() => me.refetch()} /></>);
  return (
    <>
      <PageHeader title="My profile" icon="user" compact description="Your details and password — the same profile you edit on the Me screen in the app." />
      {!data ? (
        <div className="grid gap-3 lg:grid-cols-[300px_minmax(0,1fr)]"><Skeleton className="h-72 w-full rounded-2xl" /><Skeleton className="h-96 w-full rounded-2xl" /></div>
      ) : (
        <div className="grid gap-3 lg:grid-cols-[300px_minmax(0,1fr)]">
          <IdentityCard me={data} onSaved={setMe} toast={toast} confirm={confirm} />
          <div className="space-y-3">
            <ProfileForm me={data} onSaved={setMe} toast={toast} />
            <PasswordCard me={data} toast={toast} onChanged={() => { void applyChange(qc, 'user_updated'); }} />
          </div>
        </div>
      )}
    </>
  );
}

type Toast = ReturnType<typeof useUi>['toast'];

/** Photo, name, role and account facts. Click the photo to change it (shrunk on the server, like the app). */
function IdentityCard({ me, onSaved, toast, confirm }: { me: MyAccount; onSaved: (m: MyAccount) => void; toast: Toast; confirm: ReturnType<typeof useUi>['confirm'] }) {
  const picker = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const name = me.profile.full_name || me.name || me.email;
  const initials = name.split(/\s+/).map((p) => p[0]).join('').slice(0, 2).toUpperCase();
  const role = [me.profile.job_title, me.profile.company_name].filter(Boolean).join(' · ');

  const pick = async (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) { toast('error', 'Choose an image', 'JPG, PNG or WebP'); return; }
    if (file.size > MAX_PHOTO) { toast('error', 'That photo is too large', 'Pick one under 8 MB.'); return; }
    setBusy(true);
    try {
      const dataUrl = await new Promise<string>((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = () => rej(new Error('Could not read the file.')); r.readAsDataURL(file); });
      onSaved(await api.updateMe({ avatar: dataUrl }));
      toast('success', 'Photo updated');
    } catch (e) {
      toast('error', "Couldn't change your photo", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    if (!(await confirm({ title: 'Remove your photo?', message: 'Your initials show instead.', confirmLabel: 'Remove', danger: true }))) return;
    setBusy(true);
    try { onSaved(await api.updateMe({ avatar: '' })); toast('success', 'Photo removed'); } catch (e) { toast('error', "Couldn't remove your photo", errorMessage(e)); } finally { setBusy(false); }
  };

  const facts: { icon: IconName; label: string; value: React.ReactNode }[] = [
    { icon: 'mail', label: 'Email', value: <span className="break-all">{me.email}</span> },
    { icon: 'shield', label: 'Role', value: me.isAdmin ? 'Admin' : 'Rep' },
    { icon: 'calendar', label: 'Member since', value: me.createdAt ? fmtDateTime(me.createdAt) : '—' },
    { icon: 'link', label: 'LinkedIn sign-in', value: me.linkedin.connected ? `Connected${me.linkedin.email ? ` · ${me.linkedin.email}` : ''}` : 'Not connected' },
    { icon: 'key', label: 'Password', value: me.passwordSet ? 'Set' : 'Not set yet' },
  ];

  return (
    <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base * 1.2, ease: MOTION.ease }}>
      <Card className="overflow-hidden rounded-2xl">
        <div className="relative px-5 pb-5 pt-6 text-center" style={{ background: 'linear-gradient(160deg, color-mix(in oklab, var(--brand-500) 14%, var(--card)), var(--card) 70%)' }}>
          <div className="relative mx-auto size-24">
            <motion.span aria-hidden className="absolute -inset-1.5 rounded-full bg-brand-400/40" animate={{ scale: [1, 1.08, 1], opacity: [0.5, 0.15, 0.5] }} transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }} />
            <motion.button type="button" onClick={() => picker.current?.click()} disabled={busy} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
              aria-label={me.photoUrl ? 'Change your photo' : 'Add your photo'} title={me.photoUrl ? 'Change your photo' : 'Add your photo'}
              className="group relative flex size-24 cursor-pointer items-center justify-center overflow-hidden rounded-full border-4 border-card bg-brand-600 text-2xl font-black text-white shadow-active">
              <AnimatePresence mode="wait" initial={false}>
                {me.photoUrl ? <motion.img key={me.photoUrl.slice(-24)} src={me.photoUrl} alt="" initial={{ opacity: 0, scale: 1.1 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="size-full object-cover" />
                  : <motion.span key="initials" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>{initials}</motion.span>}
              </AnimatePresence>
              <span className="absolute inset-0 flex items-center justify-center bg-black/45 opacity-0 transition-opacity group-hover:opacity-100"><Icon name="camera" size={22} /></span>
              {busy ? <span className="absolute inset-0 flex items-center justify-center bg-black/45"><motion.span animate={{ rotate: 360 }} transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }} className="flex"><Icon name="loader" size={22} /></motion.span></span> : null}
            </motion.button>
            <span className="pointer-events-none absolute bottom-0.5 right-0.5 flex size-7 items-center justify-center rounded-full border-2 border-card bg-brand-600 text-white"><Icon name="camera" size={12} /></span>
          </div>
          <input ref={picker} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => { void pick(e.target.files?.[0]); e.target.value = ''; }} />
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={name} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="mt-3 font-display text-xl font-black tracking-[-0.02em] text-foreground">{name}</motion.div>
          </AnimatePresence>
          <div className="mt-0.5 text-sm text-body">{role || <span className="text-muted-foreground">Add your title and company</span>}</div>
          <div className="mt-2 flex justify-center gap-1.5">
            <Badge tone="violet" icon="shield">{me.isAdmin ? 'Admin' : 'Rep'}</Badge>
            {me.linkedin.connected ? <Badge tone="sky" icon="link">LinkedIn</Badge> : null}
          </div>
          <div className="mt-3 flex justify-center gap-2">
            <Button size="sm" icon="camera" onClick={() => picker.current?.click()} disabled={busy}>{me.photoUrl ? 'Change photo' : 'Add photo'}</Button>
            {me.photoUrl ? <Button size="sm" variant="ghost" icon="trash" onClick={remove} disabled={busy}>Remove</Button> : null}
          </div>
        </div>
        <ul className="divide-y border-t">
          {facts.map((f, i) => (
            <motion.li key={f.label} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 + i * 0.04 }} className="group flex items-start gap-3 px-4 py-2.5">
              <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-secondary text-muted-foreground transition-colors group-hover:bg-brand-50 group-hover:text-brand-700"><Icon name={f.icon} size={13} /></span>
              <span className="min-w-0 flex-1">
                <span className="block text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">{f.label}</span>
                <span className="block text-sm text-foreground">{f.value}</span>
              </span>
            </motion.li>
          ))}
        </ul>
      </Card>
    </motion.div>
  );
}

/** Name, title, company, website, LinkedIn, phone, email signature — saved together, with unsaved-changes tracking. */
function ProfileForm({ me, onSaved, toast }: { me: MyAccount; onSaved: (m: MyAccount) => void; toast: Toast }) {
  const [form, setForm] = useState<Profile>(me.profile);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { setForm(me.profile); }, [me.profile]);
  const changed = useMemo(() => (Object.keys(form) as (keyof Profile)[]).filter((k) => form[k].trim() !== (me.profile[k] || '').trim()), [form, me.profile]);
  const dirty = changed.length > 0;

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.full_name.trim()) { setError('Add your name.'); return; }
    setSaving(true);
    setError('');
    try {
      const patch = Object.fromEntries(changed.map((k) => [k, form[k].trim()]));
      onSaved(await api.updateMe(patch));
      toast('success', 'Profile saved');
    } catch (err) {
      setError(errorMessage(err, "Couldn't save your profile."));
    } finally {
      setSaving(false);
    }
  };

  return (
    <SectionCard index={0} icon="user" tint="var(--brand-600)" title="Profile" subtitle="Shown to your team, and used by the AI when it writes for you"
      actions={<AnimatePresence>{dirty ? <motion.span initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }} transition={spring}><Badge tone="amber">{changed.length} unsaved</Badge></motion.span> : null}</AnimatePresence>}>
      <form onSubmit={save} className="space-y-4 p-4">
        <div className="grid gap-3 sm:grid-cols-2">
          {FIELDS.map((f, i) => (
            <motion.div key={f.key} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 + i * 0.03 }} className={f.key === 'full_name' ? 'sm:col-span-2' : ''}>
              <Field label={f.label} hint={f.hint}>
                <div className="group relative">
                  <Icon name={f.icon} size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-brand-600" />
                  <Input value={form[f.key]} onChange={(e) => setForm((p) => ({ ...p, [f.key]: e.target.value }))} placeholder={f.placeholder}
                    type={f.type === 'tel' ? 'tel' : 'text'} inputMode={f.type === 'url' ? 'url' : undefined} autoComplete={f.key === 'full_name' ? 'name' : f.key === 'phone' ? 'tel' : 'off'}
                    maxLength={f.key === 'phone' ? 40 : f.key === 'full_name' || f.key === 'job_title' ? 120 : 300}
                    className={cx('pl-9 transition-[border-color]', changed.includes(f.key) && 'border-amber-300')} />
                </div>
              </Field>
            </motion.div>
          ))}
        </div>
        <Field label="Email signature" hint="Added to follow-up emails the AI drafts for you.">
          <Textarea rows={4} value={form.email_signature} onChange={(e) => setForm((p) => ({ ...p, email_signature: e.target.value }))} maxLength={2000}
            placeholder={'Priya Raman\nHead of Events · Northwind\n+1 415 555 0142'} className={cx(changed.includes('email_signature') && 'border-amber-300')} />
        </Field>
        <AnimatePresence>{error ? <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0, x: [0, -4, 4, 0] }} exit={{ opacity: 0 }}><Notice tone="red" icon="alert">{error}</Notice></motion.div> : null}</AnimatePresence>
        <div className="flex items-center justify-end gap-2">
          <Button type="button" variant="ghost" icon="undo" disabled={!dirty || saving} onClick={() => { setForm(me.profile); setError(''); }}>Discard</Button>
          <motion.span whileTap={dirty ? { scale: 0.96 } : {}} className="inline-flex">
            <Button type="submit" variant="primary" icon="save" loading={saving} disabled={!dirty}>Save changes</Button>
          </motion.span>
        </div>
      </form>
    </SectionCard>
  );
}

/** Change your password (needs the current one), or reset it with a code emailed to you — the app's flow. */
function PasswordCard({ me, toast, onChanged }: { me: MyAccount; toast: Toast; onChanged: () => void }) {
  const [mode, setMode] = useState<'change' | 'code'>('change');
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [again, setAgain] = useState('');
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [sending, setSending] = useState(false);
  const [sentAt, setSentAt] = useState<number | null>(null);
  const [cooldown, setCooldown] = useState(0);
  const [error, setError] = useState('');
  const [done, setDone] = useState('');
  useEffect(() => {
    if (!cooldown) return;
    const t = setTimeout(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const reset = () => { setCurrent(''); setNext(''); setAgain(''); setCode(''); setError(''); };
  const check = (): string => {
    if (next.length < 8) return 'Your new password needs at least 8 characters.';
    if (next !== again) return "The two new passwords don't match.";
    return '';
  };

  const change = async (e: React.FormEvent) => {
    e.preventDefault();
    const problem = (me.passwordSet && !current ? 'Enter your current password.' : '') || check();
    if (problem) { setError(problem); return; }
    setBusy(true);
    setError('');
    try {
      const r = await api.changeMyPassword(next, me.passwordSet ? current : undefined);
      updateSession({ token: r.token, refreshToken: r.refreshToken }); // this console stays signed in
      reset();
      setDone(r.message);
      toast('success', 'Password changed', 'Other devices were signed out.');
      onChanged();
    } catch (err) {
      setError(errorMessage(err, "Couldn't change your password."));
    } finally {
      setBusy(false);
    }
  };

  const sendCode = async () => {
    setSending(true);
    setError('');
    try {
      const r = await requestPasswordReset(me.email);
      setSentAt(Date.now());
      setCooldown(r.resendAfter);
      toast('success', 'Code sent', `Check ${me.email} — it expires in ${Math.round(r.expiresIn / 60)} minutes.`);
    } catch (err) {
      setError(errorMessage(err, "Couldn't send a reset code."));
    } finally {
      setSending(false);
    }
  };
  const resetWithCode = async (e: React.FormEvent) => {
    e.preventDefault();
    const problem = (!/^\d{6}$/.test(code.trim()) ? 'Enter the 6-digit code from the email.' : '') || check();
    if (problem) { setError(problem); return; }
    setBusy(true);
    setError('');
    try {
      await resetPasswordWithCode(me.email, code, next); // saves the fresh session
      reset();
      setSentAt(null);
      setMode('change');
      setDone('Password reset. Other devices were signed out.');
      toast('success', 'Password reset', 'Other devices were signed out.');
      onChanged();
    } catch (err) {
      setError(errorMessage(err, "Couldn't reset your password."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <SectionCard index={1} icon="key" tint="#f59e0b" title="Password" subtitle="Changing it signs you out on every other device"
      actions={(
        <div role="tablist" className="relative flex rounded-xl bg-secondary p-0.5 text-xs font-bold">
          {([['change', me.passwordSet ? 'Change' : 'Set'], ['code', 'Email me a code']] as const).map(([v, label]) => (
            <button key={v} type="button" role="tab" aria-selected={mode === v} onClick={() => { setMode(v); setError(''); setDone(''); }}
              className={cx('relative cursor-pointer rounded-lg px-2.5 py-1 transition-colors', mode === v ? 'text-foreground' : 'text-muted-foreground hover:text-foreground')}>
              {mode === v ? <motion.span layoutId="es-pw-tab" className="absolute inset-0 rounded-lg bg-card shadow-card" transition={spring} /> : null}
              <span className="relative">{label}</span>
            </button>
          ))}
        </div>
      )}>
      <div className="p-4">
        <AnimatePresence>{done ? (
          <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={spring} className="mb-3"><Notice tone="green" icon="check">{done}</Notice></motion.div>
        ) : null}</AnimatePresence>
        <AnimatePresence mode="wait" initial={false}>
          {mode === 'change' ? (
            <motion.form key="change" onSubmit={change} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }} transition={{ duration: MOTION.fast }} className="space-y-3">
              {me.passwordSet ? (
                <Field label="Current password"><PasswordInput value={current} onChange={setCurrent} placeholder="Your current password" autoComplete="current-password" /></Field>
              ) : <Notice tone="sky" icon="info">You signed up with LinkedIn, so there's no password yet. Set one to sign in with your email too.</Notice>}
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="New password"><PasswordInput value={next} onChange={setNext} placeholder="At least 8 characters" autoComplete="new-password" /><StrengthBar pw={next} /></Field>
                <Field label="New password again" error={again && next !== again ? "Doesn't match" : null}><PasswordInput value={again} onChange={setAgain} placeholder="Type it again" autoComplete="new-password" /></Field>
              </div>
              <AnimatePresence>{error ? <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1, x: [0, -4, 4, 0] }} exit={{ opacity: 0 }}><Notice tone="red" icon="alert">{error}</Notice></motion.div> : null}</AnimatePresence>
              <div className="flex flex-wrap items-center justify-between gap-2">
                {me.passwordSet ? <button type="button" onClick={() => { setMode('code'); setError(''); }} className="cursor-pointer text-xs font-semibold text-brand-700 hover:underline">Forgot your current password?</button> : <span />}
                <Button type="submit" variant="primary" icon="key" loading={busy}>{me.passwordSet ? 'Change password' : 'Set password'}</Button>
              </div>
            </motion.form>
          ) : (
            <motion.form key="code" onSubmit={resetWithCode} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: MOTION.fast }} className="space-y-3">
              <div className="flex flex-wrap items-center gap-3 rounded-xl border border-dashed p-3">
                <motion.span animate={sentAt ? { rotate: [0, -12, 10, 0] } : {}} transition={{ duration: 0.5 }} className="flex size-9 items-center justify-center rounded-xl bg-sky-50 text-sky-600 dark:bg-sky-500/15 dark:text-sky-300"><Icon name="mail" size={16} /></motion.span>
                <div className="min-w-0 flex-1 text-sm">
                  <div className="font-semibold text-foreground">{sentAt ? 'Code sent' : 'Get a 6-digit code by email'}</div>
                  <div className="truncate text-xs text-muted-foreground">{sentAt ? `Sent to ${me.email} — it expires in 15 minutes.` : `To ${me.email}`}</div>
                </div>
                {/* type="button": inside the reset form, a plain button also submitted it (an instant "Enter the
                    6-digit code" error) and was the form's default, so Enter in the code field re-sent the email */}
                <Button type="button" size="sm" icon="send" loading={sending} disabled={cooldown > 0} onClick={sendCode}>{cooldown > 0 ? `Resend in ${cooldown}s` : sentAt ? 'Send again' : 'Send code'}</Button>
              </div>
              <AnimatePresence initial={false}>
                {sentAt ? (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: MOTION.base }} className="space-y-3 overflow-hidden">
                    <Field label="Code from the email">
                      <Input value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))} inputMode="numeric" autoComplete="one-time-code" placeholder="123456"
                        className="font-mono text-lg tracking-[0.4em]" autoFocus />
                    </Field>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field label="New password"><PasswordInput value={next} onChange={setNext} placeholder="At least 8 characters" autoComplete="new-password" /><StrengthBar pw={next} /></Field>
                      <Field label="New password again" error={again && next !== again ? "Doesn't match" : null}><PasswordInput value={again} onChange={setAgain} placeholder="Type it again" autoComplete="new-password" /></Field>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
              <AnimatePresence>{error ? <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1, x: [0, -4, 4, 0] }} exit={{ opacity: 0 }}><Notice tone="red" icon="alert">{error}</Notice></motion.div> : null}</AnimatePresence>
              {sentAt ? <div className="flex justify-end"><Button type="submit" variant="primary" icon="check" loading={busy}>Reset password</Button></div> : null}
            </motion.form>
          )}
        </AnimatePresence>
      </div>
    </SectionCard>
  );
}
