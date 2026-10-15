// The contact drawer's Files tab: GET/POST /api/contact/{id}/files · DELETE /api/contact/files/{id}
// · GET …/download. Owns the file list, uploads (with progress rows) and removal (optimistic).
import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api, base64ToBlob, fileToBase64, saveBlob } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { keys, q } from '../../lib/queries';
import type { Contact, ContactFile } from '../../lib/types';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { AnimatedNumber } from '../../ui/micro';
import { Button, ErrorState, IconButton, Skeleton, Spinner, cx, errorMessage, fmtBytes, relTime, useUi } from '../../ui/kit';
import { EmptyPanel, IconTile, Rise, SectionTitle, TINT, type Tint } from '../../ui/drawerKit';

const FILE_ACCEPT = '.pdf,.ppt,.pptx,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg,.webp';
const MAX_FILE = 20 * 1024 * 1024;

/** File type → icon tile colour and label (PDF rose, slides amber, sheets emerald, images sky, docs violet). */
function fileKind(f: { fileType?: string | null; filename: string }): { tint: Tint; label: string; icon: IconName } {
  const ext = (f.fileType || f.filename.split('.').pop() || '').toLowerCase();
  if (ext.includes('pdf')) return { tint: 'rose', label: 'PDF', icon: 'file' };
  if (/ppt|key|slide|deck/.test(ext)) return { tint: 'amber', label: 'Slides', icon: 'file' };
  if (/xls|csv|sheet/.test(ext)) return { tint: 'emerald', label: 'Sheet', icon: 'file' };
  if (/png|jpe?g|webp|gif|image/.test(ext)) return { tint: 'sky', label: 'Image', icon: 'imagePlus' };
  if (/doc|txt|word/.test(ext)) return { tint: 'violet', label: 'Doc', icon: 'file' };
  return { tint: 'slate', label: ext.slice(0, 4).toUpperCase() || 'File', icon: 'paperclip' };
}

export function FilesPanel({ c }: { c: Contact }) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const files = useQuery(q.contactFiles(c.id));
  const [uploads, setUploads] = useState<{ name: string; status: 'uploading' | 'failed'; error?: string }[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = async (list: FileList | null) => {
    if (!list?.length) return;
    const picked = Array.from(list);
    if (inputRef.current) inputRef.current.value = '';
    for (const f of picked) {
      if (f.size > MAX_FILE) { setUploads((u) => [...u, { name: f.name, status: 'failed', error: `Over 20 MB (${fmtBytes(f.size)})` }]); continue; }
      setUploads((u) => [...u, { name: f.name, status: 'uploading' }]);
      try {
        const saved = await api.uploadContactFile(c.id, { filename: f.name, contentBase64: await fileToBase64(f) });
        qc.setQueryData<ContactFile[]>(keys.contactFiles(c.id), (d) => [saved, ...(d || []).filter((x) => x.id !== saved.id)]);
        setUploads((u) => u.filter((x) => !(x.name === f.name && x.status === 'uploading')));
        toast('success', `${f.name} attached`);
      } catch (e) {
        setUploads((u) => u.map((x) => (x.name === f.name && x.status === 'uploading' ? { ...x, status: 'failed', error: errorMessage(e) } : x)));
      }
    }
    // a file on the contact is a contact change (the server says contact_updated too)
    void applyChange(qc, 'contact_updated');
  };
  const download = async (f: ContactFile) => {
    setBusy(f.id);
    try { const d = await api.downloadContactFile(f.id); saveBlob(base64ToBlob(d.contentBase64, d.mimeType), d.filename); }
    catch (e) { toast('error', "Couldn't download the file", errorMessage(e)); }
    finally { setBusy(null); }
  };
  // optimistic: the row leaves at once; it comes back if the server refuses
  const remove = async (f: ContactFile) => {
    const ok = await confirm({ title: `Remove ${f.filename}?`, message: 'The rep loses it in the app too. This cannot be undone.', confirmLabel: 'Remove', danger: true });
    if (!ok) return;
    const before = qc.getQueryData<ContactFile[]>(keys.contactFiles(c.id));
    qc.setQueryData<ContactFile[]>(keys.contactFiles(c.id), (d) => (d || []).filter((x) => x.id !== f.id));
    try { await api.deleteContactFile(f.id); toast('success', `${f.filename} removed`); }
    catch (e) { qc.setQueryData(keys.contactFiles(c.id), before); toast('error', "Couldn't remove — restored", errorMessage(e)); }
    finally { void applyChange(qc, 'contact_updated'); }
  };

  const pick = () => inputRef.current?.click();
  return (
    // about 1/3 : 2/3 — the upload column stays pinned under the tab bar, the file list scrolls beside it
    <div className="grid items-start gap-5 md:grid-cols-[minmax(240px,1fr)_2fr]">
      {/* full visible height (body height − sticky tab bar − bottom padding), pinned while the list scrolls */}
      <aside className="flex flex-col gap-3 md:sticky md:top-[76px] md:z-10 md:h-[calc(100cqh-96px)] md:min-h-[280px]">
      {/* dropzone: lights up while a file is dragged over it, the icon hops */}
      <Rise i={0} className="flex min-h-0 flex-1 flex-col">
        <motion.div role="button" tabIndex={0} aria-label="Upload files" onClick={pick} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } }}
          onDragOver={(e) => { e.preventDefault(); if (!drag) setDrag(true); }} onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}
          animate={{ scale: drag ? 1.015 : 1 }} transition={{ type: 'spring', stiffness: 400, damping: 26 }}
          className={cx('group flex flex-1 cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-7 text-center outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring',
            drag ? 'border-primary bg-brand-50/70 dark:bg-brand-100/40' : 'border-border bg-card hover:border-brand-300 hover:bg-brand-50/40 dark:hover:bg-brand-100/20')}>
          <motion.span animate={drag ? { y: -6, scale: 1.12 } : { y: [0, -4, 0] }}
            transition={drag ? { type: 'spring', stiffness: 400, damping: 15 } : { duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
            className="flex h-12 w-12 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-card transition-transform group-hover:rotate-6">
            <Icon name="upload" size={20} />
          </motion.span>
          <div className="text-sm font-semibold text-foreground">{drag ? 'Drop to attach' : 'Drop files here or click to choose'}</div>
          <div className="flex flex-wrap items-center justify-center gap-1.5">
            {(['rose', 'amber', 'emerald', 'sky', 'violet'] as Tint[]).map((t, i) => (
              <span key={t} className={cx('rounded-md px-1.5 py-0.5 text-[10px] font-semibold', TINT[t])}>{['PDF', 'Slides', 'Sheets', 'Images', 'Docs'][i]}</span>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">Up to 20 MB each · the rep sees them in the app</p>
          <input ref={inputRef} type="file" multiple accept={FILE_ACCEPT} className="hidden" onClick={(e) => e.stopPropagation()} onChange={(e) => upload(e.target.files)} />
        </motion.div>
      </Rise>

      {/* uploads in progress / failed: capped so the dropzone keeps most of the column */}
      {uploads.length ? <motion.div layoutScroll className="es-scroll max-h-[40%] shrink-0 space-y-2 overflow-y-auto">
      <AnimatePresence initial={false}>
        {uploads.map((u, i) => (
          <motion.div key={`${u.name}-${i}`} layout initial={{ opacity: 0, y: 6, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, x: 30 }}
            className={cx('relative flex items-center gap-3 overflow-hidden rounded-xl px-3.5 py-2.5 text-sm ring-1 ring-inset', u.status === 'failed' ? 'bg-rose-50 text-rose-700 ring-rose-200 dark:bg-rose-500/10 dark:text-rose-200 dark:ring-rose-500/30' : 'bg-card text-body ring-border')}>
            {u.status === 'uploading' ? <Spinner size={15} className="text-primary" /> : <Icon name="alert" size={15} />}
            <span className="min-w-0 flex-1 truncate">{u.name}{u.status === 'uploading' ? ' — uploading…' : ` — ${u.error}`}</span>
            {u.status === 'failed' ? <button type="button" className="cursor-pointer text-xs underline" onClick={() => setUploads((x) => x.filter((_, j) => j !== i))}>Dismiss</button> : null}
            {u.status === 'uploading' ? (
              <motion.span className="absolute bottom-0 left-0 h-0.5 w-1/3 bg-primary" animate={{ x: ['-100%', '300%'] }} transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }} />
            ) : null}
          </motion.div>
        ))}
      </AnimatePresence>
      </motion.div> : null}
      </aside>

      <div className="min-w-0">
      {files.isLoading ? <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-16 w-full rounded-xl" />)}</div>
        : files.error ? <ErrorState error={files.error} onRetry={() => files.refetch()} />
        : files.data?.length ? (
          <div>
            <SectionTitle icon="paperclip" right={<span className="text-xs font-medium text-muted-foreground"><AnimatedNumber value={files.data.length} duration={500} /> file{files.data.length === 1 ? '' : 's'}</span>}>Attached files</SectionTitle>
            <ul className="space-y-2">
              <AnimatePresence initial={false}>
                {files.data.map((f, idx) => {
                  const k = fileKind(f);
                  return (
                    <motion.li key={f.id} layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -30, scale: 0.97, transition: { duration: MOTION.base } }}
                      transition={{ duration: MOTION.base, ease: MOTION.ease, delay: Math.min(idx, 8) * 0.04 }}
                      className="group flex items-center gap-3 rounded-xl border bg-card px-3.5 py-2.5 shadow-card transition-colors hover:border-brand-200 hover:bg-secondary/40">
                      <span className="relative">
                        <IconTile icon={k.icon} tint={k.tint} />
                        <span className={cx('absolute -bottom-1 -right-1.5 rounded px-1 text-[8px] font-bold uppercase leading-3 ring-2 ring-card', TINT[k.tint])}>{k.label}</span>
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="break-words text-sm font-medium text-foreground [overflow-wrap:anywhere]">{f.filename}</div>
                        <div className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
                          <span>{fmtBytes(f.fileSize)}</span>
                          {f.category !== 'general' ? <span className="rounded bg-brand-50 px-1.5 text-[10px] font-semibold text-brand-700 dark:bg-brand-100 dark:text-brand-800">{f.category}</span> : null}
                          <span className="inline-flex items-center gap-1"><Icon name="clock" size={11} />{relTime(f.createdAt)}</span>
                          {f.description ? <span className="break-words">{f.description}</span> : null}
                        </div>
                      </div>
                      <motion.span whileHover={{ y: -1 }} whileTap={{ scale: 0.92 }} className="inline-flex">
                        <Button size="sm" icon="download" loading={busy === f.id} onClick={() => download(f)}>Download</Button>
                      </motion.span>
                      <motion.span whileHover={{ rotate: -10 }} whileTap={{ scale: 0.85 }} className="inline-flex">
                        <IconButton icon="trash" label="Remove file" className="hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-500/10" onClick={() => remove(f)} />
                      </motion.span>
                    </motion.li>
                  );
                })}
              </AnimatePresence>
            </ul>
          </div>
        ) : (
          <EmptyPanel icon="paperclip" tint="violet" title="No files yet">
            Files the rep attaches in the app, and decks the pipeline makes, appear here too.
          </EmptyPanel>
        )}
      </div>
    </div>
  );
}
