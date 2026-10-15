// The contact drawer's Duplicates tab: GET /api/contacts/{id}/duplicates · POST /api/admin/contacts/merge
// · …/duplicates/dismiss. Owns the look-alikes, merge and "not a duplicate" (optimistic, with rollback).
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { contactLevel } from '../../lib/levels';
import { keys, q } from '../../lib/queries';
import { patchQueries, removeContacts } from '../../lib/optimistic';
import type { Contact, DuplicateMatch } from '../../lib/types';
import { Avatar } from '../../ui/Avatar';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { AnimatedNumber } from '../../ui/micro';
import { Badge, Button, ErrorState, ScoreBadge, Skeleton, contactName, cx, errorMessage, relTime, useUi } from '../../ui/kit';
import { EmptyPanel, IconTile, Rise, SectionTitle, TINT_TEXT, type Tint } from '../../ui/drawerKit';

/** The look-alikes of one contact (the drawer's tab badge and overview notice read the same query). */
export const useDuplicateMatches = (id: string | undefined) => useQuery({ ...q.contactDuplicates(id || ''), enabled: !!id });

export function DuplicatesPanel({ c, onOpen }: { c: Contact; onOpen?: ((id: string) => void) | undefined }) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const dups = useDuplicateMatches(c.id);
  const matches = dups.data?.matches;
  const dropMatch = (otherId: string) => patchQueries<{ matches: DuplicateMatch[] } | undefined>(qc, keys.contactDuplicates(c.id), (d) => (d ? { ...d, matches: d.matches.filter((m) => m.contact.id !== otherId) } : d));
  // optimistic: the match leaves at once (and a merged duplicate leaves the grid); back if the server says no
  const merge = async (other: Contact, otherRep?: string, sameRep = true) => {
    const ok = await confirm({
      title: `Merge ${contactName(other)} into ${contactName(c)}?`,
      message: sameRep ? (
        <>Empty fields on {contactName(c)} are filled from the duplicate, notes and files are combined, and the higher score is kept. {contactName(other)} goes to Recently deleted for 30 days, so this can be undone.</>
      ) : (
        <>{contactName(other)} was captured by <b>another rep{otherRep ? ` (${otherRep})` : ''}</b>. Merging fills {contactName(c)}'s empty fields from it, combines notes and files, keeps the higher score — and
          removes it from <b>{otherRep || 'that rep'}'s</b> contacts and app (it goes to their Recently deleted for 30 days, so this can be undone).</>
      ),
      confirmLabel: 'Merge',
    });
    if (!ok) return;
    const r1 = await dropMatch(other.id);
    const r2 = await removeContacts(qc, [other.id]);
    try {
      const r = await api.merge(c.id, other.id);
      toast('success', 'Merged', r.message);
    } catch (e) {
      r1(); r2();
      toast('error', "Couldn't merge — restored", errorMessage(e));
    } finally { void applyChange(qc, 'contact_merged'); }
  };
  const dismiss = async (other: Contact) => {
    const rollback = await dropMatch(other.id);
    try {
      await api.notDuplicate(c.id, other.id);
      toast('success', 'Marked as different people', "They won't be suggested again.");
    } catch (e) {
      rollback();
      toast('error', "Couldn't save — restored", errorMessage(e));
    } finally { void applyChange(qc, 'duplicate_dismissed'); }
  };

  if (dups.isLoading) return <div className="space-y-3">{[0, 1].map((i) => <Skeleton key={i} className="h-28 w-full rounded-2xl" />)}</div>;
  if (dups.error) return <ErrorState error={dups.error} onRetry={() => dups.refetch()} />;
  if (!matches?.length) {
    // what was compared, each with its own icon and a tick that pops in
    const checks: { icon: IconName; tint: Tint; label: string; value: string | null | undefined }[] = [
      { icon: 'mail', tint: 'violet', label: 'Same email', value: c.email },
      { icon: 'phone', tint: 'emerald', label: 'Same phone', value: c.phone },
      { icon: 'idCard', tint: 'sky', label: 'Same name + company', value: [contactName(c), c.company].filter(Boolean).join(' · ') },
    ];
    return (
      <div className="grid items-start gap-4 md:grid-cols-[minmax(240px,1fr)_2fr]">
        <EmptyPanel icon="duplicate" tint="emerald" title="No look-alikes">
          Nothing else in any rep's contacts matches {contactName(c)}.
        </EmptyPanel>
        <Rise i={1}>
          <SectionTitle icon="search" right={<span className="text-xs font-medium text-muted-foreground">across every rep's contacts</span>}>What we checked</SectionTitle>
          <div className="space-y-1 rounded-2xl border bg-card p-2 shadow-card">
            {checks.map((k, i) => (
              <motion.div key={k.label} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.12 + i * 0.06 }}
                className="group flex items-center gap-3 rounded-xl px-2 py-2.5 transition-colors hover:bg-secondary/60">
                <motion.span initial={{ scale: 0.5, rotate: -15 }} animate={{ scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.16 + i * 0.06 }}
                  className={cx(!k.value && 'opacity-45 grayscale')}>
                  <IconTile icon={k.icon} tint={k.tint} />
                </motion.span>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold text-foreground">{k.label}</div>
                  <div className="break-words text-xs text-muted-foreground [overflow-wrap:anywhere]">{k.value || 'Not on this contact, so nothing to compare'}</div>
                </div>
                <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 520, damping: 16, delay: 0.35 + i * 0.1 }}
                  className={cx('flex h-6 w-6 shrink-0 items-center justify-center rounded-full', k.value ? 'bg-emerald-500 text-white' : 'bg-secondary text-muted-foreground')}
                  title={k.value ? 'No match found' : 'Skipped'}>
                  <Icon name={k.value ? 'check' : 'x'} size={12} strokeWidth={3} />
                </motion.span>
              </motion.div>
            ))}
          </div>
          <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground"><Icon name="info" size={12} />New captures are checked again automatically.</p>
        </Rise>
      </div>
    );
  }
  const reasonLabel = (r: DuplicateMatch['reasons'][number]) => (r === 'email' ? 'Same email' : r === 'phone' ? 'Same phone' : 'Same name + company');
  const reasonIcon = (r: DuplicateMatch['reasons'][number]): IconName => (r === 'email' ? 'mail' : r === 'phone' ? 'phone' : 'idCard');
  return (
    <div className="space-y-3">
      <Rise i={0}>
        <div className="flex items-center gap-3 rounded-2xl bg-amber-50 px-4 py-3 ring-1 ring-inset ring-amber-200 dark:bg-amber-500/10 dark:ring-amber-500/25">
          <motion.span initial={{ rotate: -25, scale: 0.6 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 14 }}>
            <IconTile icon="duplicate" tint="amber" />
          </motion.span>
          <div className="min-w-0 flex-1 text-sm">
            <div className="font-semibold text-amber-900 dark:text-amber-100">
              <AnimatedNumber value={matches.length} duration={500} /> possible duplicate{matches.length === 1 ? '' : 's'}
            </div>
            <div className="text-xs text-amber-800 dark:text-amber-200">Merging keeps {contactName(c)} and fills its empty fields from the other one.</div>
          </div>
        </div>
      </Rise>
      <AnimatePresence initial={false}>
        {matches.map((m, idx) => {
          const o = m.contact;
          const lvl = contactLevel(o);
          return (
            <motion.div key={o.id} layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: 40, scale: 0.96, transition: { duration: MOTION.base } }}
              transition={{ duration: MOTION.base * 1.3, ease: MOTION.ease, delay: 0.08 + idx * 0.06 }}
              whileHover={{ y: -2 }}
              className="group relative overflow-hidden rounded-2xl border bg-card p-4 shadow-card transition-shadow hover:shadow-md">
              {/* heat strip */}
              <span className="absolute inset-x-0 top-0 h-1" style={{ background: lvl ? lvl.color : 'var(--secondary)' }} />
              <div className="flex items-start gap-3">
                {/* this contact ← match */}
                <div className="flex shrink-0 items-center">
                  <Avatar src={c.photoUrl || null} name={contactName(c)} size="md" />
                  <motion.span className="mx-1 text-muted-foreground" animate={{ x: [0, -3, 0] }} transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}>
                    <Icon name="merge" size={14} className="rotate-90" />
                  </motion.span>
                  <Avatar src={o.photoUrl || null} name={contactName(o)} size="md" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <button type="button" className="cursor-pointer text-sm font-semibold text-foreground hover:text-primary hover:underline" onClick={() => onOpen?.(o.id)}>{contactName(o)}</button>
                    <ScoreBadge score={o.score} level={o.level} />
                  </div>
                  <div className="mt-1 grid gap-1 text-xs text-muted-foreground sm:grid-cols-2">
                    <span className="flex min-w-0 items-center gap-1.5"><Icon name="briefcase" size={12} className={TINT_TEXT.violet} /><span className="break-words">{o.title || 'No designation'}</span></span>
                    <span className="flex min-w-0 items-center gap-1.5"><Icon name="building" size={12} className={TINT_TEXT.sky} /><span className="break-words">{o.company || 'No company'}</span></span>
                    {o.email ? <span className="flex min-w-0 items-center gap-1.5"><Icon name="mail" size={12} className={TINT_TEXT.violet} /><span className="break-all">{o.email}</span></span> : null}
                    {o.phone ? <span className="flex min-w-0 items-center gap-1.5"><Icon name="phone" size={12} className={TINT_TEXT.emerald} /><span className="font-mono">{o.phone}</span></span> : null}
                  </div>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {m.reasons.map((r, ri) => (
                      <motion.span key={r} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 20, delay: 0.2 + idx * 0.06 + ri * 0.05 }}>
                        <Badge tone="amber" icon={reasonIcon(r)}>{reasonLabel(r)}</Badge>
                      </motion.span>
                    ))}
                    {m.sameRep === false ? <Badge tone="sky" icon="users">Another rep{m.ownerName ? ` · ${m.ownerName}` : ''}</Badge> : null}
                    <Badge icon="clock">Added {relTime(o.createdAt)}</Badge>
                  </div>
                </div>
              </div>
              <div className="mt-3 flex flex-wrap justify-end gap-2 border-t pt-3">
                <Button size="sm" variant="ghost" icon="eye" onClick={() => onOpen?.(o.id)}>Open</Button>
                <Button size="sm" variant="ghost" icon="x" onClick={() => dismiss(o)}>Not a duplicate</Button>
                <motion.span whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.95 }} className="inline-flex">
                  <Button size="sm" variant="primary" icon="merge" onClick={() => merge(o, m.ownerName, m.sameRep !== false)}>Merge into {c.firstName}</Button>
                </motion.span>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
