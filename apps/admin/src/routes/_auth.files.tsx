import React, { useRef, useState } from 'react';
import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { q as queries } from '../lib/queries';
import { applyChange } from '../lib/changes';
import { removeFromArrays } from '../lib/optimistic';
import { useEventNames } from '../lib/eventNames';
import { api, base64ToBlob, fileToBase64, saveBlob } from '../lib/api';
import type { EventFile } from '../lib/types';
import { Icon } from '../ui/icons';
import {
  Badge, Button, Card, EmptyState, ErrorState, Field, IconButton, Input, Modal, PageHeader, Select, Skeleton, Table, Td, Th, Textarea,
  cx, errorMessage, fmtBytes, relTime, useUi,
} from '../ui/kit';

const ACCEPT = '.pdf,.ppt,.pptx,.doc,.docx,.xls,.xlsx,.csv,.txt,.png,.jpg,.jpeg';
const CATEGORIES = [['general', 'General'], ['brochure', 'Brochure / one-pager'], ['pricing', 'Pricing'], ['catalog', 'Deck / presentation'], ['floorplan', 'Floor plan']];
// no category on file = an ordinary file (the backend treats null like "general")
const catLabel = (c: string | null) => CATEGORIES.find(([k]) => k === (c || 'general'))?.[1] || c;

export function FilesComponent() {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const q = useSuspenseQuery(queries.files());
  // every event (admin events ∪ planner), not just the ones with a booth directory
  const eventNames = useEventNames();
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [meta, setMeta] = useState({ category: 'general', description: '', event: '' });
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const pick = (f: File | null | undefined) => {
    if (!f) return;
    if (f.size > 20 * 1024 * 1024) { toast('error', 'That file is over 20 MB'); return; }
    setFile(f);
  };
  const upload = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const r = await api.uploadFile({
        filename: file.name, contentBase64: await fileToBase64(file), mimeType: file.type || undefined,
        category: meta.category, description: meta.description, event: meta.event.trim() || undefined,
      });
      toast('success', `Uploaded ${r.filename}`, 'Reps see it in the app under Me → Team files.');
      setOpen(false);
      setFile(null);
      setMeta({ category: 'general', description: '', event: '' });
      // the Events page (files / floor plans per event) and the Activity feed show it too
      void applyChange(qc, 'files_changed');
    } catch (e) {
      toast('error', "Couldn't upload", errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const download = async (f: EventFile) => {
    setDownloading(f.id);
    try {
      const d = await api.downloadFile(f.id);
      saveBlob(base64ToBlob(d.contentBase64, d.mimeType), d.filename);
      void applyChange(qc, 'file_downloaded');
    } catch (e) {
      toast('error', "Couldn't download", errorMessage(e));
    } finally {
      setDownloading(null);
    }
  };
  const remove = async (f: EventFile) => {
    if (!(await confirm({ title: `Delete ${f.filename}?`, message: 'Reps will no longer see it in the app.', confirmLabel: 'Delete', danger: true }))) return;
    // optimistic: the file leaves the list at once and comes back if the server rejects
    const rollback = await removeFromArrays<EventFile>(qc, ['admin', 'files'], [f.id]);
    try {
      await api.deleteFile(f.id);
      toast('success', 'File deleted');
    } catch (e) {
      rollback();
      toast('error', "Couldn't delete — restored", errorMessage(e));
    } finally {
      void applyChange(qc, 'files_changed');
    }
  };

  const files = q.data || [];
  const real = files.filter((f) => !f.placeholder);
  return (
    <>
      <PageHeader title="Team files" description="Brochures, price lists, floor plans and decks your reps can open or save on their phone (Me → Team files, and on the event's page)."
        actions={<Button variant="primary" icon="upload" onClick={() => setOpen(true)}>Upload file</Button>} />

      {q.error ? <ErrorState error={q.error} onRetry={() => q.refetch()} /> : (
        <Card>
          {q.isLoading ? <div className="space-y-2 p-4">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
            : !files.length ? <EmptyState icon="file" title="No files yet" description="Upload sales material for the event — it shows up in the app right away."
              action={<Button variant="primary" icon="upload" onClick={() => setOpen(true)}>Upload file</Button>} />
              : (
                <Table>
                  <thead><tr><Th>File</Th><Th>Category</Th><Th>Event</Th><Th className="text-right">Size</Th><Th className="text-right">Downloads</Th><Th>Added</Th><Th className="w-28" /></tr></thead>
                  <tbody>
                    {files.map((f) => (
                      <tr key={f.id} className={cx('hover:bg-secondary', f.placeholder && 'opacity-70')}>
                        <Td>
                          <div className="flex items-center gap-3">
                            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-secondary text-[10px] font-bold uppercase text-muted-foreground">{f.fileType}</span>
                            <div className="min-w-0">
                              <div className="flex items-center gap-2 truncate font-medium text-foreground">{f.filename}
                                {f.placeholder ? <Badge tone="amber">Placeholder</Badge> : null}</div>
                              <div className="truncate text-xs text-muted-foreground">{f.placeholder ? 'Old demo row with no real file — hidden in the app. Delete it.' : f.description || '—'}</div>
                            </div>
                          </div>
                        </Td>
                        <Td><Badge>{catLabel(f.category)}</Badge></Td>
                        <Td className="text-body">{f.eventName || 'All events'}</Td>
                        <Td className="text-right tabular-nums text-body">{fmtBytes(f.fileSize)}</Td>
                        <Td className="text-right tabular-nums text-body">{f.downloadCount}</Td>
                        <Td className="whitespace-nowrap text-muted-foreground">{relTime(f.createdAt)}</Td>
                        <Td>
                          <div className="flex justify-end gap-1">
                            <Button size="sm" icon="download" disabled={f.placeholder} loading={downloading === f.id} onClick={() => download(f)}>Get</Button>
                            <IconButton icon="trash" label={`Delete ${f.filename}`} onClick={() => remove(f)} className="hover:text-rose-600" />
                          </div>
                        </Td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
          {files.length ? <div className="px-4 py-3 text-xs text-muted-foreground">{real.length} file{real.length === 1 ? '' : 's'} reps can open{files.length - real.length ? ` · ${files.length - real.length} placeholder${files.length - real.length === 1 ? '' : 's'}` : ''}</div> : null}
        </Card>
      )}

      <Modal open={open} onOpenChange={setOpen} title="Upload a file" description="PDF, slides, Word, Excel, CSV, text or images — up to 20 MB."
        footer={<>
          <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant="primary" icon="upload" loading={busy} disabled={!file} onClick={upload}>Upload</Button>
        </>}>
        <div className="space-y-4">
          <button type="button" onClick={() => input.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
            className={cx('flex w-full flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition cursor-pointer',
              drag ? 'border-brand-400 bg-accent' : 'border-border hover:border-brand-300 hover:bg-secondary')}>
            <Icon name={file ? 'file' : 'upload'} size={22} className="mb-2 text-primary" />
            {file ? (
              <><span className="text-sm font-medium text-foreground">{file.name}</span><span className="text-xs text-muted-foreground">{fmtBytes(file.size)} · click to change</span></>
            ) : (
              <><span className="text-sm font-medium text-body">Drop a file here, or click to browse</span><span className="text-xs text-muted-foreground">Max 20 MB</span></>
            )}
          </button>
          <input ref={input} type="file" accept={ACCEPT} className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Category">
              <Select value={meta.category} onChange={(e) => setMeta({ ...meta, category: e.target.value })}>
                {CATEGORIES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
              </Select>
            </Field>
            <Field label="Event" hint="Empty = every event">
              <Input list="es-file-events" value={meta.event} onChange={(e) => setMeta({ ...meta, event: e.target.value })} placeholder="All events" />
              <datalist id="es-file-events">{eventNames.map((n) => <option key={n} value={n} />)}</datalist>
            </Field>
          </div>
          <Field label="Description"><Textarea rows={2} value={meta.description} onChange={(e) => setMeta({ ...meta, description: e.target.value })} placeholder="What's in it, when to use it…" /></Field>
        </div>
      </Modal>
    </>
  );
}
