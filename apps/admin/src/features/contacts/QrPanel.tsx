// The contact drawer's QR tab: GET /api/contact/{id}/qr (a vCard QR of the contact's current details,
// keyed by updatedAt so an edit refreshes it). Owns the code, its downloads and the vCard copy.
import { useQuery } from '@tanstack/react-query';
import { motion } from 'motion/react';
import { base64ToBlob, saveBlob } from '../../lib/api';
import { q } from '../../lib/queries';
import type { Contact } from '../../lib/types';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { ErrorState, Skeleton, contactName, cx } from '../../ui/kit';
import { IconTile, Rise, Row, SectionTitle, useCopy, type Tint } from '../../ui/drawerKit';

export function QrPanel({ c }: { c: Contact }) {
  const copy = useCopy();
  const qr = useQuery(q.contactQr(c.id, c.updatedAt));
  if (qr.isLoading) {
    return <div className="grid gap-5 md:grid-cols-[auto_1fr]"><Skeleton className="mx-auto h-72 w-72 rounded-3xl" /><div className="space-y-2">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12 w-full rounded-xl" />)}</div></div>;
  }
  if (qr.error) return <ErrorState error={qr.error} onRetry={() => qr.refetch()} />;
  const data = qr.data;
  if (!data) return null;
  const fileBase = contactName(c).replace(/\s+/g, '_');
  const inCard: { icon: IconName; tint: Tint; label: string; value: string | null | undefined }[] = [
    { icon: 'user', tint: 'violet', label: 'Name', value: contactName(c) },
    { icon: 'briefcase', tint: 'violet', label: 'Designation', value: c.title },
    { icon: 'building', tint: 'sky', label: 'Company', value: c.company },
    { icon: 'mail', tint: 'violet', label: 'Email', value: c.email },
    { icon: 'phone', tint: 'emerald', label: 'Phone', value: c.phone },
    { icon: 'link', tint: 'linkedin', label: 'LinkedIn', value: c.linkedin },
  ];
  const actions: { icon: IconName; tint: Tint; label: string; sub: string; run: () => void }[] = [
    { icon: 'download', tint: 'violet', label: 'Download QR', sub: 'PNG image', run: () => saveBlob(base64ToBlob(data.qrDataUrl.split(',', 2)[1] || '', 'image/png'), `${fileBase}-qr.png`) },
    { icon: 'idCard', tint: 'emerald', label: 'Download vCard', sub: '.vcf for any address book', run: () => saveBlob(new Blob([data.vcard], { type: 'text/vcard' }), `${fileBase}.vcf`) },
    { icon: 'copy', tint: 'sky', label: 'Copy vCard', sub: 'Paste it anywhere', run: () => copy('vCard', data.vcard) },
  ];
  return (
    <div className="grid items-start gap-5 md:grid-cols-[auto_1fr]">
      {/* the code: framed with corner brackets, a scan line sweeps over it */}
      {/* pinned on wider screens (just under the sticky tab bar) while the details beside it scroll */}
      <Rise i={0} className="flex flex-col items-center gap-3 md:sticky md:top-[76px] md:z-10 md:self-start">
        <motion.div initial={{ scale: 0.9, opacity: 0, rotate: -2 }} animate={{ scale: 1, opacity: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 260, damping: 22 }}
          whileHover={{ scale: 1.02 }}
          className="relative rounded-3xl bg-gradient-to-br from-brand-100 to-brand-50 p-4 shadow-card dark:from-brand-200/40 dark:to-brand-100/30">
          <div className="relative overflow-hidden rounded-2xl bg-white p-3">
            <img src={data.qrDataUrl} alt={`QR code for ${contactName(c)}`} className="block h-60 w-60" />
            <motion.span aria-hidden className="pointer-events-none absolute inset-x-2 h-10 bg-gradient-to-b from-transparent via-brand-500/25 to-transparent"
              initial={{ top: '-15%' }} animate={{ top: ['-15%', '100%'] }} transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut', repeatDelay: 0.6 }} />
          </div>
          {(['left-2 top-2 border-l-[3px] border-t-[3px] rounded-tl-xl', 'right-2 top-2 border-r-[3px] border-t-[3px] rounded-tr-xl',
            'bottom-2 left-2 border-b-[3px] border-l-[3px] rounded-bl-xl', 'bottom-2 right-2 border-b-[3px] border-r-[3px] rounded-br-xl'] as const).map((pos, i) => (
            <motion.span key={pos} aria-hidden className={cx('pointer-events-none absolute h-7 w-7 border-brand-600', pos)}
              initial={{ opacity: 0, scale: 1.4 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 400, damping: 20, delay: 0.2 + i * 0.06 }} />
          ))}
        </motion.div>
        <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <motion.span animate={{ y: [0, -2, 0] }} transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }} className="flex text-brand-600"><Icon name="smartphone" size={14} /></motion.span>
          Point a phone camera at it
        </div>
      </Rise>

      <div className="min-w-0 space-y-5">
        <Rise i={1}>
          <SectionTitle icon="idCard" right={<span className="text-xs font-medium text-muted-foreground">updates when you edit</span>}>What the code saves</SectionTitle>
          <div className="rounded-2xl border bg-card p-2 shadow-card">
            {inCard.map((r, i) => (
              <Row key={r.label} i={i} icon={r.icon} tint={r.tint} label={r.label}>{r.value || ''}</Row>
            ))}
          </div>
        </Rise>
        <Rise i={2}>
          <SectionTitle icon="send">Share it</SectionTitle>
          <div className="grid gap-2 sm:grid-cols-3">
            {actions.map((a, i) => (
              <motion.button key={a.label} type="button" onClick={a.run}
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base, ease: MOTION.ease, delay: 0.3 + i * 0.06 }}
                whileHover={{ y: -3 }} whileTap={{ scale: 0.96 }}
                className="group flex cursor-pointer items-center gap-3 rounded-2xl border bg-card p-3 text-left shadow-card transition-colors hover:border-brand-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                <IconTile icon={a.icon} tint={a.tint} />
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-foreground">{a.label}</span>
                  <span className="block text-xs text-muted-foreground">{a.sub}</span>
                </span>
              </motion.button>
            ))}
          </div>
        </Rise>
      </div>
    </div>
  );
}
