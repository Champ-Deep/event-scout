// The contact drawer's Edit tab: PATCH /api/contacts/{id} (name, company, title, email, phone,
// LinkedIn, event, notes, photo). Owns its form, the save and its refresh.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'motion/react';
import { api, fileToBase64 } from '../../lib/api';
import { applyChange } from '../../lib/changes';
import { keys } from '../../lib/queries';
import { patchContact } from '../../lib/optimistic';
import type { Contact, ContactPatch } from '../../lib/types';
import { Avatar } from '../../ui/Avatar';
import { Icon, type IconName } from '../../ui/icons';
import { MOTION } from '../../ui/tokens';
import { Button, Input, Spinner, Textarea, contactName, cx, errorMessage, useUi } from '../../ui/kit';
import { Rise, SectionTitle, TINT_TEXT, type Tint } from '../../ui/drawerKit';
import { diffDraft, draftOf, type Draft } from './draft';

const MAX_PHOTO = 8 * 1024 * 1024;
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const readImage = async (file: File): Promise<string> => {
  if (!file.type.startsWith('image/')) throw new Error('Pick an image file (JPG, PNG, WebP…).');
  if (file.size > MAX_PHOTO) throw new Error('That photo is over 8 MB.');
  return `data:${file.type};base64,${await fileToBase64(file)}`;
};

/**
 * A form field with a coloured icon inside the input. The icon lifts when the field is focused,
 * a dot by the label marks an unsaved change, and an error nudges the field sideways once.
 */
function IconField({ label, icon, tint, changed, hint, error, className, multiline, children }: {
  label: string; icon: IconName; tint: Tint; changed?: boolean; hint?: React.ReactNode; error?: string | null; className?: string; multiline?: boolean; children: React.ReactNode;
}) {
  return (
    <label className={cx('group/field block', className)}>
      <span className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-foreground">
        {label}
        <AnimatePresence>{changed ? (
          <motion.span key="dot" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }} transition={{ type: 'spring', stiffness: 600, damping: 20 }}
            className="h-1.5 w-1.5 rounded-full bg-primary" title="Changed" />
        ) : null}</AnimatePresence>
      </span>
      <motion.div className="relative" animate={error ? { x: [0, -5, 5, -3, 3, 0] } : { x: 0 }} transition={{ duration: 0.35 }}>
        {children}
        {/* drawn after the input, so it sits on top without a z-index that could poke through sticky bars */}
        <span className={cx('pointer-events-none absolute left-3 transition-transform duration-200 group-focus-within/field:scale-115',
          multiline ? 'top-2.5 group-focus-within/field:-translate-y-px' : 'top-1/2 -translate-y-1/2 group-focus-within/field:-translate-y-[62%]', error ? 'text-destructive' : TINT_TEXT[tint])}>
          <Icon name={icon} size={15} />
        </span>
      </motion.div>
      <AnimatePresence mode="wait" initial={false}>
        {error ? (
          <motion.span key="err" initial={{ opacity: 0, y: -3 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="mt-1 flex items-center gap-1 text-xs text-destructive">
            <Icon name="alert" size={12} />{error}
          </motion.span>
        ) : hint ? <motion.span key="hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-1 block text-xs text-muted-foreground">{hint}</motion.span> : null}
      </AnimatePresence>
    </label>
  );
}

export function EditTab({ c, eventNames, onDirty, onSaved }: { c: Contact; eventNames: string[]; onDirty: (d: boolean) => void; onSaved: () => void }) {
  const qc = useQueryClient();
  const { toast } = useUi();
  const [draft, setDraft] = useState<Draft>(() => draftOf(c));
  const [photoBusy, setPhotoBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => { setDraft(draftOf(c)); }, [c.id]);
  const patch = useMemo(() => diffDraft(c, draft), [c, draft]);
  const dirty = Object.keys(patch).length > 0;
  useEffect(() => { onDirty(dirty); return () => onDirty(false); }, [dirty]);
  const emailBad = !!draft.email.trim() && !EMAIL_RE.test(draft.email.trim());
  const nameBad = !draft.firstName.trim();
  const set = (k: keyof Draft) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setDraft((d) => ({ ...d, [k]: e.target.value }));

  // optimistic for the text fields; the photo waits for the server (it shrinks it to a thumbnail)
  const save = useMutation({
    mutationFn: (fields: ContactPatch) => api.updateContact(c.id, fields),
    onMutate: async (fields) => {
      const { photo: _photo, ...rest } = fields;
      return { rollback: await patchContact(qc, c.id, { ...(rest as Partial<Contact>), eventId: rest.eventId ?? c.eventId }) };
    },
    onError: (e, _f, ctx) => { ctx?.rollback(); toast('error', "Couldn't save — restored", errorMessage(e)); },
    onSuccess: (updated) => {
      qc.setQueryData(keys.contact(c.id), updated);
      setDraft(draftOf(updated));
      toast('success', `${contactName(updated)} saved`);
      onSaved();
    },
    onSettled: () => { void applyChange(qc, 'contact_updated'); },
  });

  const pickPhoto = async (file: File | undefined) => {
    if (!file) return;
    setPhotoBusy(true);
    try { const uri = await readImage(file); setDraft((d) => ({ ...d, photo: uri })); }
    catch (e) { toast('error', "Couldn't use that photo", errorMessage(e)); }
    finally { setPhotoBusy(false); if (fileRef.current) fileRef.current.value = ''; }
  };
  const preview = draft.photo === undefined ? c.photoUrl : draft.photo || null;

  // a field whose value differs from the saved contact gets a violet dot on its label
  const changed = (k: keyof ContactPatch) => k in patch;
  return (
    <form className="space-y-5" onSubmit={(e) => { e.preventDefault(); if (!nameBad && !emailBad && dirty) save.mutate(patch); }}>
      {/* photo */}
      <Rise i={0}>
        <div className="group flex items-center gap-4 rounded-2xl border bg-card p-4 shadow-card">
          <motion.div key={preview || 'none'} initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: 'spring', stiffness: 380, damping: 20 }} className="relative">
            <Avatar src={preview} name={contactName(c)} size="xl" rounded="xl" />
            <span className="absolute -bottom-1.5 -right-1.5 flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-white shadow-card ring-2 ring-card transition-transform duration-300 group-hover:rotate-12 group-hover:scale-110"><Icon name="camera" size={13} /></span>
            {photoBusy ? <span className="absolute inset-0 flex items-center justify-center rounded-xl bg-card/70"><Spinner size={18} className="text-primary" /></span> : null}
          </motion.div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 text-sm font-semibold text-foreground">Photo{changed('photo') ? <span className="h-1.5 w-1.5 rounded-full bg-primary" title="Changed" /> : null}</div>
            <p className="text-xs text-muted-foreground">JPG, PNG or WebP up to 8 MB — stored as a 512 px thumbnail.</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button size="sm" icon="imagePlus" type="button" onClick={() => fileRef.current?.click()}>{preview ? 'Change' : 'Add photo'}</Button>
              {preview ? <Button size="sm" variant="ghost" icon="x" type="button" onClick={() => setDraft((d) => ({ ...d, photo: null }))}>Remove</Button> : null}
              {draft.photo !== undefined ? <Button size="sm" variant="ghost" icon="undo" type="button" onClick={() => setDraft((d) => ({ ...d, photo: undefined }))}>Undo</Button> : null}
            </div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => pickPhoto(e.target.files?.[0])} />
          </div>
        </div>
      </Rise>

      {/* identity */}
      <Rise i={1}>
        <SectionTitle icon="user">Identity</SectionTitle>
        <div className="grid gap-3.5 rounded-2xl border bg-card p-4 shadow-card sm:grid-cols-2">
          <IconField label="First name *" icon="user" tint="violet" changed={changed('firstName')} error={nameBad ? 'A contact needs a name' : null}>
            <Input value={draft.firstName} onChange={set('firstName')} autoFocus className="pl-9" />
          </IconField>
          <IconField label="Last name" icon="user" tint="violet" changed={changed('lastName')}><Input value={draft.lastName} onChange={set('lastName')} className="pl-9" /></IconField>
          <IconField label="Designation" icon="briefcase" tint="violet" changed={changed('title')}><Input value={draft.title} onChange={set('title')} className="pl-9" /></IconField>
          <IconField label="Company" icon="building" tint="sky" changed={changed('company')}><Input value={draft.company} onChange={set('company')} className="pl-9" /></IconField>
        </div>
      </Rise>

      {/* how to reach them */}
      <Rise i={2}>
        <SectionTitle icon="send">How to reach them</SectionTitle>
        <div className="grid gap-3.5 rounded-2xl border bg-card p-4 shadow-card sm:grid-cols-2">
          <IconField label="Email" icon="mail" tint="violet" changed={changed('email')} error={emailBad ? "That doesn't look like an email address" : null}>
            <Input type="email" value={draft.email} onChange={set('email')} className="pl-9" />
          </IconField>
          <IconField label="Phone" icon="phone" tint="emerald" changed={changed('phone')}><Input value={draft.phone} onChange={set('phone')} className="pl-9 font-mono" /></IconField>
          <IconField label="LinkedIn" icon="link" tint="linkedin" changed={changed('linkedin')} hint="linkedin.com/in/… (checked when you save)" className="sm:col-span-2">
            <Input value={draft.linkedin} onChange={set('linkedin')} placeholder="https://www.linkedin.com/in/…" className="pl-9" />
          </IconField>
        </div>
      </Rise>

      {/* where they met · notes */}
      <Rise i={3}>
        <SectionTitle icon="calendar">Where they met</SectionTitle>
        <div className="space-y-3.5 rounded-2xl border bg-card p-4 shadow-card">
          <IconField label="Event" icon="calendar" tint="amber" changed={changed('eventId')} hint="Exactly as reps pick it in the app">
            <Input list="es-edit-events" value={draft.eventId} onChange={set('eventId')} className="pl-9" />
            <datalist id="es-edit-events">{eventNames.map((n) => <option key={n} value={n} />)}</datalist>
          </IconField>
          <IconField label="Rep's notes" icon="note" tint="slate" changed={changed('notes')} hint="Visible to the rep in the app" multiline>
            <Textarea rows={4} value={draft.notes} onChange={set('notes')} className="pl-9" />
          </IconField>
        </div>
      </Rise>

      {/* save bar — sticks to the bottom while editing */}
      {/* z-30 keeps it above the animated fields scrolling under it; -mb-5 sits it flush with the dialog's bottom edge */}
      <div className="sticky bottom-0 z-30 -mx-6 -mb-5 flex flex-wrap items-center justify-between gap-3 border-t bg-card px-6 py-3 shadow-[0_-8px_20px_-14px_rgb(30_16_70/0.25)]">
        <AnimatePresence mode="wait">{dirty ? (
          <motion.span key="dirty" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: MOTION.fast }}
            className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-800 dark:bg-amber-500/10 dark:text-amber-200">
            <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-70" /><span className="relative inline-flex h-2 w-2 rounded-full bg-amber-500" /></span>
            {Object.keys(patch).length} unsaved change{Object.keys(patch).length === 1 ? '' : 's'}
          </motion.span>
        ) : save.isSuccess ? (
          <motion.span key="saved" initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 500, damping: 22 }}
            className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-700 dark:text-emerald-400"><Icon name="check" size={14} />All changes saved</motion.span>
        ) : <span key="none" className="inline-flex items-center gap-1 text-xs text-muted-foreground"><Icon name="pencil" size={12} />Edit any field, then save</span>}</AnimatePresence>
        <div className="flex gap-2">
          <Button type="button" variant="ghost" icon="undo" disabled={!dirty || save.isPending} onClick={() => setDraft(draftOf(c))}>Discard</Button>
          <Button type="submit" variant="primary" icon="save" loading={save.isPending} disabled={!dirty || nameBad || emailBad || photoBusy}>Save changes</Button>
        </div>
      </div>
    </form>
  );
}
