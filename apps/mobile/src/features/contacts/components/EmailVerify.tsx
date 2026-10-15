import React, { useEffect, useState } from 'react';
import { View, Pressable, ActivityIndicator } from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeOut, LinearTransition, ZoomIn } from 'react-native-reanimated';
import {
  ShieldCheck, ShieldAlert, ShieldX, ShieldQuestion, Check, X, AlertTriangle, Minus, RefreshCw, ChevronDown, ChevronUp,
} from 'lucide-react-native';
import { Text } from '../../../ui/Text';
import { theme, toneBg, makeStyles } from '../../../ui/theme';
import { api, errorMessage, type EmailCheck, type EmailReachable } from '../../../lib/api';
import { PressableScale } from '../../onboarding/components';

/* Under a contact's email: is the address real? (backend email_verify.py — the
   check-if-email-exists checks). Shows the last result straight away (a quick format + domain check
   when there's none yet); "Check mailbox" asks the mail server; tap the line for every check. */

const LOOK: Record<EmailReachable, { label: string; Icon: typeof ShieldCheck; tone: 'ok' | 'warn' | 'bad' | 'muted' }> = {
  safe: { label: 'Valid email', Icon: ShieldCheck, tone: 'ok' },
  risky: { label: 'Risky email', Icon: ShieldAlert, tone: 'warn' },
  invalid: { label: 'Invalid email', Icon: ShieldX, tone: 'bad' },
  unknown: { label: 'Not confirmed', Icon: ShieldQuestion, tone: 'muted' },
};

function ago(iso?: string | null) {
  if (!iso) return '';
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

const CHECK_FAILED = "Couldn't check the email right now.";

export function EmailVerifyRow({ email }: { email?: string | null }) {
  const [res, setRes] = useState<EmailCheck | null>(null);
  const [busy, setBusy] = useState<'quick' | 'full' | null>(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState(false);
  const addr = (email || '').trim();

  useEffect(() => {
    setRes(null); setError(''); setOpen(false);
    if (!addr) return;
    let cancelled = false;
    setBusy('quick');
    api.verifyEmail(addr, { smtp: false })   // the cached full result if there is one, else format + domain
      .then((r) => { if (!cancelled) setRes(r); })
      .catch((e) => { if (!cancelled) setError(errorMessage(e, CHECK_FAILED)); })
      .finally(() => { if (!cancelled) setBusy(null); });
    return () => { cancelled = true; };
  }, [addr]);

  const checkMailbox = async () => {
    if (busy) return;
    setBusy('full'); setError('');
    try {
      setRes(await api.verifyEmail(addr, { force: !!res?.smtp_checked }));
      setOpen(true);
    } catch (e) {
      setError(errorMessage(e, CHECK_FAILED));
    } finally {
      setBusy(null);
    }
  };

  if (!addr) return null;
  const look = res ? LOOK[res.is_reachable] : null;
  const quickOnly = !!res && !res.smtp_checked;
  const color = !look ? theme.colors.text.muted : look.tone === 'ok' ? theme.colors.semantic.successText
    : look.tone === 'warn' ? theme.colors.semantic.warning : look.tone === 'bad' ? theme.colors.semantic.danger : theme.colors.text.muted;
  const bg = !look ? toneBg('#f1f5f9') : look.tone === 'ok' ? theme.colors.semantic.successBg
    : look.tone === 'warn' ? theme.colors.semantic.warningBg : look.tone === 'bad' ? theme.colors.semantic.dangerBg : toneBg('#f1f5f9');
  const label = !look ? '' : quickOnly && res!.is_reachable === 'unknown' ? 'Domain OK · mailbox not checked' : look.label;

  return (
    <Animated.View layout={LinearTransition.springify().damping(20)} style={styles.wrap}>
      <View style={styles.row}>
        <Pressable onPress={() => res && setOpen((o) => !o)} disabled={!res} style={styles.status} hitSlop={6}
          accessibilityRole="button" accessibilityLabel={res ? `${label}. ${open ? 'Hide' : 'Show'} the checks` : 'Checking email'}>
          {busy === 'quick' && !res ? (
            <><ActivityIndicator size="small" color={theme.colors.text.muted} /><Text style={styles.muted}>Checking…</Text></>
          ) : look ? (
            <>
              <Animated.View key={`${res!.is_reachable}-${res!.smtp_checked}`} entering={ZoomIn.springify().damping(14)} style={[styles.badge, { backgroundColor: bg }]}>
                <look.Icon size={13} color={color} />
                <Text style={[styles.badgeText, { color }]} numberOfLines={1}>{label}</Text>
              </Animated.View>
              {res!.checked_at ? <Text style={styles.muted} numberOfLines={1}>{ago(res!.checked_at)}</Text> : null}
              {open ? <ChevronUp size={14} color={theme.colors.text.muted} /> : <ChevronDown size={14} color={theme.colors.text.muted} />}
            </>
          ) : error ? <Text style={styles.error} numberOfLines={2}>{error}</Text> : null}
        </Pressable>
        <PressableScale onPress={checkMailbox} disabled={!!busy} scaleTo={0.92} style={styles.btn}
          accessibilityLabel={res?.smtp_checked ? 'Check this email again' : 'Check the mailbox exists'}
          leadingIcon={busy === 'full' ? <ActivityIndicator size="small" color={theme.colors.brand[700]} /> : <RefreshCw size={12} color={theme.colors.brand[700]} />}>
          <Text style={styles.btnText}>{busy === 'full' ? 'Checking…' : res?.smtp_checked ? 'Recheck' : 'Check mailbox'}</Text>
        </PressableScale>
      </View>

      {open && res ? (
        <Animated.View entering={FadeInDown.duration(220)} exiting={FadeOut.duration(140)} style={styles.details}>
          <Item i={0} state={res.syntax.is_valid_syntax ? 'yes' : 'no'} text={res.syntax.is_valid_syntax ? 'Well-formed address' : (res.syntax.error || 'Not a valid address')} />
          {res.syntax.is_valid_syntax ? (
            <Item i={1} state={res.mx.accepts_mail ? 'yes' : 'no'}
              text={res.mx.accepts_mail ? `Domain accepts email${res.mx.records[0] ? ` (${res.mx.records[0]})` : ''}` : (res.mx.error || "Domain doesn't accept email")} />
          ) : null}
          {res.mx.accepts_mail ? (
            <Item i={2} state={!res.smtp_checked || !res.smtp?.can_connect_smtp ? 'skip' : res.smtp.is_deliverable ? 'yes' : 'no'}
              text={!res.smtp_checked ? 'Mailbox not checked yet — tap Check mailbox'
                : !res.smtp?.can_connect_smtp ? (res.smtp?.error || "Couldn't ask the mail server")
                : res.smtp.is_deliverable ? 'The mailbox exists' : res.smtp.is_disabled ? 'The mailbox is disabled'
                : res.smtp.has_full_inbox ? 'The mailbox is full' : (res.smtp.error || "The mail server says this mailbox doesn't exist")} />
          ) : null}
          {res.smtp?.is_catch_all ? <Item i={3} state="warn" text="Catch-all domain — it accepts any address, so the mailbox can't be confirmed" /> : null}
          {res.misc.is_disposable ? <Item i={4} state="warn" text="Disposable / throwaway address" /> : null}
          {res.misc.is_role_account ? <Item i={5} state="warn" text={`Role address (${res.syntax.username}@) — reaches a team, not a person`} /> : null}
          {res.misc.is_free_provider ? <Item i={6} state="info" text="Personal mail provider, not a company domain" /> : null}
        </Animated.View>
      ) : null}
    </Animated.View>
  );
}

function Item({ state, text, i }: { state: 'yes' | 'no' | 'warn' | 'skip' | 'info'; text: string; i: number }) {
  const c = state === 'yes' ? theme.colors.semantic.successText : state === 'no' ? theme.colors.semantic.danger
    : state === 'warn' ? theme.colors.semantic.warning : theme.colors.text.muted;
  const Icon = state === 'yes' ? Check : state === 'no' ? X : state === 'warn' ? AlertTriangle : Minus;
  return (
    <Animated.View entering={FadeIn.delay(i * 40).duration(200)} style={styles.item}>
      <Icon size={13} color={c} strokeWidth={2.6} />
      <Text style={styles.itemText}>{text}</Text>
    </Animated.View>
  );
}

const styles = makeStyles(() => ({
  wrap: { marginLeft: 44, marginTop: -4, paddingBottom: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  status: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 6, minHeight: 28 },
  badge: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999, flexShrink: 1 },
  badgeText: { fontSize: 12, fontWeight: '800', flexShrink: 1 },
  muted: { fontSize: 11, color: theme.colors.text.muted },
  error: { flex: 1, fontSize: 12, color: theme.colors.semantic.danger },
  btn: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, backgroundColor: theme.colors.brand[50], borderWidth: 1, borderColor: theme.colors.brand[100] },
  btnText: { fontSize: 12, fontWeight: '700', color: theme.colors.brand[700] },
  details: { marginTop: 8, padding: 10, gap: 6, borderRadius: 12, backgroundColor: theme.colors.bg.surfaceHover },
  item: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  itemText: { flex: 1, fontSize: 12, lineHeight: 17, color: theme.colors.text.body },
}));
