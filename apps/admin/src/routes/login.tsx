import React, { useEffect, useState } from 'react';
import { useRouter, useSearch } from '@tanstack/react-router';
import { requestPasswordReset, resetPasswordWithCode, signIn } from '../lib/auth';
import { Icon } from '../ui/icons';
import { Button, Field, Input } from '../ui/kit';

const POINTS: [Parameters<typeof Icon>[0]['name'], string][] = [
  ['contacts', 'Every lead your team captures, with AI scores'],
  ['pipeline', 'Research, pitch emails and decks in one click'],
  ['megaphone', "Announcements straight to reps' phones"],
];

export function LoginComponent() {
  const router = useRouter();
  const search = useSearch({ from: '/login' });
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // forgot password: email → 6-digit code + new password (the app's flow), then you're signed in
  const [mode, setMode] = useState<'signin' | 'forgot'>('signin');
  const [codeSent, setCodeSent] = useState(false);
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [cooldown, setCooldown] = useState(0);
  useEffect(() => {
    if (!cooldown) return;
    const t = setTimeout(() => setCooldown((c) => Math.max(0, c - 1)), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);
  const goNext = () => {
    // Only follow same-site paths from ?next=.
    const next = search.next && search.next.startsWith('/') && !search.next.startsWith('//') ? search.next : '/';
    router.history.push(next.startsWith('/login') ? '/' : next);
  };
  const failText = (err: unknown) => (err instanceof TypeError ? "Can't reach the server. Is the backend running?" : (err as Error).message);

  const sendCode = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim())) { setError('Enter the email you sign in with.'); return; }
    setLoading(true);
    setError('');
    try {
      const r = await requestPasswordReset(email);
      setCodeSent(true);
      setCooldown(r.resendAfter);
    } catch (err) {
      setError(failText(err));
    } finally {
      setLoading(false);
    }
  };
  const reset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!/^\d{6}$/.test(code.trim())) { setError('Enter the 6-digit code from the email.'); return; }
    if (newPassword.length < 8) { setError('Your new password needs at least 8 characters.'); return; }
    setLoading(true);
    setError('');
    try {
      await resetPasswordWithCode(email, code, newPassword);
      goNext();
    } catch (err) {
      setError(failText(err));
    } finally {
      setLoading(false);
    }
  };
  const backToSignIn = () => { setMode('signin'); setCodeSent(false); setCode(''); setNewPassword(''); setError(''); };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Enter your email and password.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await signIn(email, password);
      // history.push takes a full path (with ?query), unlike navigate's typed `to`.
      goNext();
    } catch (err) {
      setError(failText(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid min-h-screen bg-background lg:grid-cols-[1.05fr_1fr]">
      <div className="relative hidden overflow-hidden border-r border-panel-border bg-panel p-12 text-panel-foreground lg:flex lg:flex-col lg:justify-between">
        <div className="relative flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-signal text-signal-foreground"><Icon name="bolt" size={18} strokeWidth={2.25} /></span>
          <span className="font-display text-lg font-bold tracking-[-0.02em]">EventScout</span>
        </div>
        <div className="relative max-w-md">
          <p className="es-eyebrow es-eyebrow-mark text-panel-muted">Floor control</p>
          <h2 className="mt-4 font-display text-[44px] font-bold leading-[1.05] tracking-[-0.035em]">Run your event team from one desk.</h2>
          <ol className="mt-10 divide-y divide-panel-border border-y border-panel-border">
            {POINTS.map(([icon, text], i) => (
              <li key={text} className="flex items-center gap-4 py-3.5 text-[15px]">
                <span className="font-mono text-xs text-primary">0{i + 1}</span>
                <span className="flex-1">{text}</span>
                <Icon name={icon} size={16} className="text-panel-muted" />
              </li>
            ))}
          </ol>
        </div>
        <p className="es-eyebrow relative text-[10px] text-panel-muted">Admin console</p>
      </div>

      <div className="flex items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2.5 lg:hidden">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-signal text-signal-foreground"><Icon name="bolt" size={18} /></span>
            <span className="font-display text-lg font-bold text-foreground">EventScout</span>
          </div>
          <span key={mode} className="es-pop-in mb-4 flex size-12 items-center justify-center rounded-2xl bg-brand-50 text-brand-700 shadow-card"><Icon name={mode === 'signin' ? 'shield' : 'key'} size={22} /></span>
          <h1 key={`t-${mode}`} className="es-in font-display text-[30px] font-bold tracking-[-0.025em] text-foreground" style={{ ['--es-d' as string]: '60ms' }}>{mode === 'signin' ? 'Sign in' : 'Reset your password'}</h1>
          <p key={`s-${mode}-${codeSent}`} className="es-in mt-1 text-sm text-body" style={{ ['--es-d' as string]: '120ms' }}>
            {mode === 'signin' ? 'Use an admin account.' : codeSent ? `If ${email.trim()} has an account, a 6-digit code is on its way. It expires in 15 minutes.` : "We'll email you a 6-digit code."}
          </p>

          {error ? (
            <div key={error} className="es-shake mt-6 flex items-start gap-2 rounded-lg border border-danger-border bg-danger-bg p-3 text-sm text-destructive" role="alert">
              <Icon name="alert" size={16} className="mt-0.5" />{error}
            </div>
          ) : null}

          {mode === 'signin' ? (
            <>
              <form onSubmit={handleSubmit} className="es-in mt-6 space-y-4" style={{ ['--es-d' as string]: '180ms' }}>
                <Field label="Email"><Input type="email" autoComplete="email" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} disabled={loading} placeholder="you@company.com" /></Field>
                <Field label="Password"><Input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} disabled={loading} placeholder="••••••••" /></Field>
                <Button type="submit" variant="primary" loading={loading} className="w-full">{loading ? 'Signing in…' : 'Sign in'}</Button>
              </form>
              <p className="mt-6 text-center text-sm">
                <button type="button" onClick={() => { setMode('forgot'); setError(''); }} className="cursor-pointer font-semibold text-brand-700 hover:underline">Forgot your password?</button>
              </p>
            </>
          ) : !codeSent ? (
            <form key="email" onSubmit={sendCode} className="es-in mt-6 space-y-4">
              <Field label="Email"><Input type="email" autoComplete="email" autoFocus value={email} onChange={(e) => setEmail(e.target.value)} disabled={loading} placeholder="you@company.com" /></Field>
              <Button type="submit" variant="primary" icon="send" loading={loading} className="w-full">{loading ? 'Sending…' : 'Send code'}</Button>
              <button type="button" onClick={backToSignIn} className="block w-full cursor-pointer text-center text-sm font-semibold text-muted-foreground hover:text-foreground">← Back to sign in</button>
            </form>
          ) : (
            <form key="code" onSubmit={reset} className="es-in mt-6 space-y-4">
              <Field label="Code from the email">
                <Input value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))} inputMode="numeric" autoComplete="one-time-code" autoFocus disabled={loading} placeholder="123456" className="font-mono text-lg tracking-[0.4em]" />
              </Field>
              <Field label="New password" hint="At least 8 characters. You'll be signed out on every other device.">
                <Input type="password" autoComplete="new-password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} disabled={loading} maxLength={128} placeholder="••••••••" />
              </Field>
              <Button type="submit" variant="primary" icon="check" loading={loading} className="w-full">{loading ? 'Resetting…' : 'Reset password & sign in'}</Button>
              <div className="flex items-center justify-between text-sm">
                <button type="button" onClick={backToSignIn} className="cursor-pointer font-semibold text-muted-foreground hover:text-foreground">← Back to sign in</button>
                <button type="button" onClick={() => sendCode()} disabled={cooldown > 0 || loading} className="cursor-pointer font-semibold text-brand-700 hover:underline disabled:cursor-default disabled:text-muted-foreground disabled:no-underline">
                  {cooldown > 0 ? `Resend in ${cooldown}s` : 'Send a new code'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
