// /sales-profile — the admin console's own AI profile. All real API:
//   load  GET   /api/admin/sales-profile   (who's pitching, what we sell, how we sell, who we target)
//   save  PATCH /api/admin/sales-profile   (only the fields sent change)
// Admin scores (and so the Contacts ranking), admin pipeline pitches and the AI assistant use ONLY this.
// It lives in its own table: saving it never changes a rep's profile, score or data, and the reps' own
// profiles (Me screen in the app) never feed the admin's AI.
import React, { useEffect, useMemo, useState } from 'react';
import { useMutation, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'motion/react';
import { api } from '../lib/api';
import { keys, q } from '../lib/queries';
import type { AdminSalesPatch, AdminSalesProfile, PitchStyle } from '../lib/types';
import { Icon, type IconName } from '../ui/icons';
import { LiveIcon } from '../ui/live-icon';
import { MOTION } from '../ui/tokens';
import { Badge, Button, Card, Field, Input, Modal, Notice, PageHeader, Textarea, cx, errorMessage, relTime, useUi } from '../ui/kit';

type Product = AdminSalesProfile['products'][number];
type Toast = ReturnType<typeof useUi>['toast'];

const PITCH_STYLES: { value: Exclude<PitchStyle, ''>; label: string; hint: string }[] = [
  { value: 'consultative', label: 'Consultative', hint: 'Ask questions first, act as a partner' },
  { value: 'direct', label: 'Direct', hint: 'Lead with the solution and ROI' },
  { value: 'challenger', label: 'Challenger', hint: 'Reframe their thinking with insights' },
  { value: 'relationship', label: 'Relationship', hint: 'Build trust first, long-term play' },
];
const COMPANY_SIZES = ['Startup', 'Mid-market', 'Enterprise'];
const MAX_PRODUCTS = 12;
const MAX_ITEMS = 10;
const EMPTY_PRODUCT: Product = { name: '', description: '', ideal_customer: '' };

/** Saves part of the profile and puts the answer in the cache (admin scores re-rank on the next score). */
/** `inlineErrors`: the caller shows the error in its form, so no toast on top of it. */
function useSave(toast: Toast, inlineErrors = false) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patch: AdminSalesPatch) => api.updateSalesProfile(patch),
    onSuccess: (next) => { qc.setQueryData(keys.salesProfile, next); },
    onError: (e) => { if (!inlineErrors) toast('error', "Couldn't save", errorMessage(e)); },
  });
}

export function SalesProfileComponent() {
  const { toast } = useUi();
  const { data } = useSuspenseQuery(q.salesProfile());
  const filled = [
    !!(data.company_name || data.job_title),
    data.products.length > 0,
    !!(data.target_industries.length || data.target_roles.length || data.target_company_sizes.length || data.target_geographies.length),
    !!(data.pitch_style || data.value_propositions.length || data.common_objections.length || data.case_studies.length),
  ];

  return (
    <>
      <PageHeader title="Sales profile" compact live
        description="What the admin console's AI works from when it scores, ranks and pitches every contact. Yours alone — separate from each rep's profile in the app." />
      <div className="space-y-3">
        <HowItWorks data={data} filled={filled} />
        <div className="grid gap-3 xl:grid-cols-2">
          <div className="space-y-3">
            <PitchingAsCard data={data} toast={toast} index={0} />
            <ProductsCard data={data} toast={toast} index={1} />
          </div>
          <div className="space-y-3">
            <TargetingCard data={data} toast={toast} index={2} />
            <SalesApproachCard data={data} toast={toast} index={3} />
          </div>
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ how it works */

/** Admin vs rep, side by side: which profile feeds which scores — so nobody wonders whose data changes. */
function HowItWorks({ data, filled }: { data: AdminSalesProfile; filled: boolean[] }) {
  const done = filled.filter(Boolean).length;
  const lanes: { icon: IconName; title: string; tint: string; from: string; to: string[] }[] = [
    { icon: 'shield', title: 'Admin console', tint: 'var(--brand-600)', from: 'This sales profile',
      to: ['Admin score on every contact — Contacts ranks, sorts and filters by it', 'Admin AI pipeline: research, pitch email and deck (admin only)', 'AI assistant answers and pitch angles'] },
    { icon: 'smartphone', title: 'Reps (mobile app)', tint: '#0ea5e9', from: "Each rep's own Me screen",
      to: ["The rep's own score on their contacts", "The rep's pitches and follow-ups in the app", 'Shown here read-only as “Rep score”'] },
  ];
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.2, ease: MOTION.ease }}>
      <Card className="overflow-hidden rounded-2xl">
        <div className="grid gap-px bg-border md:grid-cols-[1fr_1fr_auto]">
          {lanes.map((l) => (
            <div key={l.title} className="bg-card p-4">
              <div className="flex items-center gap-2">
                <span className="flex size-8 items-center justify-center rounded-lg" style={{ background: `color-mix(in oklab, ${l.tint} 14%, transparent)`, color: l.tint }}><LiveIcon name={l.icon} size={15} /></span>
                <div>
                  <div className="text-sm font-extrabold text-foreground">{l.title}</div>
                  <div className="text-[11px] text-muted-foreground">Uses: <span className="font-semibold text-body">{l.from}</span></div>
                </div>
              </div>
              <ul className="mt-3 space-y-1.5">
                {l.to.map((t, i) => (
                  <li key={t} className="flex items-start gap-2 text-xs text-body">
                    <span className="mt-px shrink-0" style={{ color: l.tint }}><LiveIcon name="check" size={13} delay={i * 0.45} /></span>{t}
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <div className="flex flex-col justify-center gap-2 bg-card p-4 md:w-60">
            <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">Profile filled</div>
            <div className="flex items-center gap-2">
              <div className="flex flex-1 gap-1">
                {filled.map((f, i) => (
                  <motion.span key={i} className="h-1.5 flex-1 rounded-full" animate={{ backgroundColor: f ? 'var(--brand-600)' : 'var(--border)' }} transition={{ duration: 0.3, delay: i * 0.05 }} />
                ))}
              </div>
              <span className="es-num text-sm font-black text-foreground">{done}/4</span>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Neither side reads or changes the other.{data.updatedAt ? ` Saved ${relTime(data.updatedAt)}${data.updatedBy ? ` by ${data.updatedBy}` : ''}.` : ' Not saved yet.'}
            </p>
            <Link to="/contacts" search={{ sort: 'score' }} className="inline-flex items-center gap-1 text-xs font-semibold text-brand-700 hover:underline">
              See contacts ranked by admin score <LiveIcon name="chevronRight" size={12} />
            </Link>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ shared bits */

function SectionCard({ icon, tint, title, subtitle, children, index, actions }: { icon: IconName; tint: string; title: string; subtitle: string; children: React.ReactNode; index: number; actions?: React.ReactNode }) {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: MOTION.base * 1.2, ease: MOTION.ease, delay: 0.08 + index * 0.06 }}>
      <Card className="overflow-hidden rounded-2xl">
        <div className="group flex items-center gap-3 border-b px-4 py-3">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-xl transition-transform duration-300 group-hover:-rotate-12" style={{ background: `color-mix(in oklab, ${tint} 14%, transparent)`, color: tint }}>
            <LiveIcon name={icon} size={16} delay={index * 0.3} />
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

function EmptyCta({ icon, title, text, onClick }: { icon: IconName; title: string; text: string; onClick: () => void }) {
  return (
    <button type="button" onClick={onClick} className="group flex w-full cursor-pointer items-center gap-3 rounded-xl border border-dashed p-3 text-left transition-colors hover:border-brand-300 hover:bg-brand-50/50">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-brand-50 text-brand-600"><LiveIcon name={icon} size={16} /></span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold text-foreground">{title}</span>
        <span className="block text-xs text-muted-foreground">{text}</span>
      </span>
      <span className="text-brand-600"><LiveIcon name="chevronRight" size={16} /></span>
    </button>
  );
}

function SubLabel({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="mb-1.5 flex items-center justify-between gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-[0.06em] text-muted-foreground">{children}</span>
      {right}
    </div>
  );
}

function ErrorLine({ error }: { error: string }) {
  return <AnimatePresence>{error ? <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0, x: [0, -4, 4, 0] }} exit={{ opacity: 0 }}><Notice tone="red" icon="alert">{error}</Notice></motion.div> : null}</AnimatePresence>;
}

/** Type a value and press Enter (or comma) to add it; × removes. */
function TagInput({ values, onChange, placeholder, max = 20, maxLength = 80, suggestions = [] }: {
  values: string[]; onChange: (v: string[]) => void; placeholder: string; max?: number; maxLength?: number; suggestions?: string[] | undefined;
}) {
  const [draft, setDraft] = useState('');
  const add = (raw: string) => {
    const v = raw.trim().replace(/\s+/g, ' ').slice(0, maxLength);
    if (!v || values.length >= max || values.some((x) => x.toLowerCase() === v.toLowerCase())) { setDraft(''); return; }
    onChange([...values, v]);
    setDraft('');
  };
  const free = suggestions.filter((s) => !values.some((v) => v.toLowerCase() === s.toLowerCase()));
  return (
    <div>
      <div className="flex min-h-9 flex-wrap items-center gap-1.5 rounded-lg border border-input bg-card px-2 py-1.5 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50">
        <AnimatePresence initial={false}>
          {values.map((v) => (
            <motion.span key={v} layout initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.7, opacity: 0 }}
              className="inline-flex items-center gap-1 rounded-md bg-brand-50 py-0.5 pl-2 pr-1 text-xs font-semibold text-brand-700">
              {v}
              <button type="button" onClick={() => onChange(values.filter((x) => x !== v))} aria-label={`Remove ${v}`} className="cursor-pointer rounded p-0.5 hover:bg-brand-100"><Icon name="x" size={11} /></button>
            </motion.span>
          ))}
        </AnimatePresence>
        {values.length < max ? (
          <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={values.length ? '' : placeholder} maxLength={maxLength}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); add(draft); }
              else if (e.key === 'Backspace' && !draft && values.length) onChange(values.slice(0, -1));
            }}
            onBlur={() => add(draft)}
            className="min-w-[8rem] flex-1 bg-transparent px-1 text-sm outline-none placeholder:text-muted-foreground" />
        ) : null}
      </div>
      {free.length ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {free.map((s) => (
            <button key={s} type="button" onClick={() => add(s)} className="inline-flex cursor-pointer items-center gap-1 rounded-md border border-dashed px-2 py-0.5 text-xs text-muted-foreground hover:border-brand-300 hover:text-brand-700">
              <Icon name="plus" size={11} />{s}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ChipRow({ icon, values, tone }: { icon: IconName; values: string[]; tone: string }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {values.map((v, i) => (
        <motion.span key={v} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: i * 0.025 }}
          className={cx('inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium', tone)}>
          <Icon name={icon} size={11} />{v}
        </motion.span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ who's pitching */

function PitchingAsCard({ data, toast, index }: { data: AdminSalesProfile; toast: Toast; index: number }) {
  const save = useSave(toast);
  const [form, setForm] = useState({ company_name: data.company_name, job_title: data.job_title });
  useEffect(() => { setForm({ company_name: data.company_name, job_title: data.job_title }); }, [data.company_name, data.job_title]);
  const dirty = form.company_name.trim() !== data.company_name || form.job_title.trim() !== data.job_title;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    save.mutate({ company_name: form.company_name.trim(), job_title: form.job_title.trim() }, { onSuccess: () => toast('success', 'Saved', 'Admin pitches now come from this company.') });
  };
  return (
    <SectionCard index={index} icon="building" tint="#0ea5e9" title="Who's pitching" subtitle="The company and role admin pitches and research are written for">
      <form onSubmit={submit} className="space-y-3 p-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Company"><Input value={form.company_name} onChange={(e) => setForm((f) => ({ ...f, company_name: e.target.value }))} maxLength={160} placeholder="e.g. Northwind" /></Field>
          <Field label="Your role"><Input value={form.job_title} onChange={(e) => setForm((f) => ({ ...f, job_title: e.target.value }))} maxLength={120} placeholder="e.g. Head of Sales" /></Field>
        </div>
        <div className="flex justify-end gap-2">
          <Button type="button" variant="ghost" icon="undo" disabled={!dirty || save.isPending} onClick={() => setForm({ company_name: data.company_name, job_title: data.job_title })}>Discard</Button>
          <Button type="submit" variant="primary" icon="save" loading={save.isPending} disabled={!dirty}>Save</Button>
        </div>
      </form>
    </SectionCard>
  );
}

/* ------------------------------------------------------------------ products & services */

function ProductsCard({ data, toast, index }: { data: AdminSalesProfile; toast: Toast; index: number }) {
  const save = useSave(toast, true);
  const products = data.products;
  const [editing, setEditing] = useState<{ index: number | null } | null>(null);
  const [form, setForm] = useState<Product>(EMPTY_PRODUCT);
  const [error, setError] = useState('');

  const open = (i: number | null) => { setForm(i == null ? EMPTY_PRODUCT : { ...products[i]! }); setError(''); setEditing({ index: i }); };
  const commit = (next: Product[], done: string) => {
    setError('');
    save.mutate({ products: next }, {
      onSuccess: () => { setEditing(null); toast('success', done, 'Admin scores use it from the next score.'); },
      onError: (e) => setError(errorMessage(e, "Couldn't save it.")),
    });
  };
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const item = { name: form.name.trim(), description: form.description.trim(), ideal_customer: form.ideal_customer.trim() };
    if (!item.name) { setError('Give it a name.'); return; }
    const idx = editing?.index ?? null;
    if (products.some((p, i) => i !== idx && p.name.toLowerCase() === item.name.toLowerCase())) { setError('You already have one with that name.'); return; }
    commit(idx == null ? [...products, item] : products.map((p, i) => (i === idx ? item : p)), idx == null ? `Added “${item.name}”` : `Saved “${item.name}”`);
  };
  const remove = () => {
    const idx = editing?.index;
    if (idx == null) return;
    commit(products.filter((_, i) => i !== idx), `Removed “${products[idx]!.name}”`);
  };

  return (
    <SectionCard index={index} icon="package" tint="var(--brand-600)" title="Products & services" subtitle="What you sell — the admin AI pitches these and scores leads against them"
      actions={products.length < MAX_PRODUCTS ? <Button size="sm" icon="plus" onClick={() => open(null)}>Add</Button> : <Badge tone="gray">{MAX_PRODUCTS} max</Badge>}>
      <div className="p-3">
        {products.length === 0 ? (
          <EmptyCta icon="package" title="Add what you sell" text="Name, what it does and who it's for — admin scores and pitches get specific." onClick={() => open(null)} />
        ) : (
          <ul className="space-y-0.5">
            <AnimatePresence initial={false}>
              {products.map((p, i) => (
                <motion.li key={p.name} layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, height: 0 }} transition={{ duration: MOTION.base }}>
                  <button type="button" onClick={() => open(i)} aria-label={`Edit ${p.name}`} className="group flex w-full cursor-pointer items-start gap-3 rounded-xl px-2 py-2.5 text-left transition-colors hover:bg-secondary">
                    <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-brand-50 font-mono text-xs font-bold text-brand-700">{i + 1}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-semibold text-foreground">{p.name}</span>
                      {p.description ? <span className="line-clamp-2 block text-xs text-body">{p.description}</span> : null}
                      {p.ideal_customer ? (
                        <span className="mt-1 inline-flex max-w-full items-center gap-1 rounded-md bg-teal-50 px-1.5 py-0.5 text-[11px] font-medium text-teal-700 dark:bg-teal-500/15 dark:text-teal-300">
                          <LiveIcon name="target" size={11} delay={i * 0.35} /><span className="truncate">For {p.ideal_customer}</span>
                        </span>
                      ) : null}
                    </span>
                    <Icon name="pencil" size={14} className="mt-1 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                  </button>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        )}
      </div>

      <Modal open={!!editing} onOpenChange={(o) => { if (!o && !save.isPending) setEditing(null); }}
        title={editing?.index == null ? 'Add a product or service' : 'Edit product or service'} description="Admin console only — reps' products in the app don't change."
        footer={<>
          {editing?.index != null ? <Button variant="ghost" icon="trash" className="mr-auto text-destructive" onClick={remove} disabled={save.isPending}>Remove</Button> : null}
          <Button variant="ghost" onClick={() => setEditing(null)} disabled={save.isPending}>Cancel</Button>
          <Button variant="primary" icon="save" type="submit" form="es-product-form" loading={save.isPending}>{editing?.index == null ? 'Add' : 'Save'}</Button>
        </>}>
        <form id="es-product-form" onSubmit={submit} className="space-y-3">
          <Field label="Name"><Input autoFocus value={form.name} onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} maxLength={120} placeholder="e.g. Verified B2B contact data" /></Field>
          <Field label="What it does" hint="A sentence or two — the AI uses it in pitches.">
            <Textarea rows={3} value={form.description} onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))} maxLength={600} placeholder="e.g. 70M+ verified decision-maker contacts, refreshed monthly." />
          </Field>
          <Field label="Who it's for" hint="The ideal customer — leads that match score higher.">
            <Input value={form.ideal_customer} onChange={(e) => setForm((f) => ({ ...f, ideal_customer: e.target.value }))} maxLength={300} placeholder="e.g. Sales teams at mid-market SaaS companies" />
          </Field>
          <ErrorLine error={error} />
        </form>
      </Modal>
    </SectionCard>
  );
}

/* ------------------------------------------------------------------ who you're targeting */

function TargetingCard({ data, toast, index }: { data: AdminSalesProfile; toast: Toast; index: number }) {
  const save = useSave(toast, true);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ target_industries: [] as string[], target_roles: [] as string[], target_company_sizes: [] as string[], target_geographies: [] as string[] });
  const [error, setError] = useState('');
  const start = () => {
    setForm({ target_industries: [...data.target_industries], target_roles: [...data.target_roles], target_company_sizes: [...data.target_company_sizes], target_geographies: [...data.target_geographies] });
    setError('');
    setOpen(true);
  };
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    save.mutate(form, {
      onSuccess: () => { setOpen(false); toast('success', 'Targeting saved', 'Press Score with AI on Contacts to re-rank against it.'); },
      onError: (err) => setError(errorMessage(err, "Couldn't save your targeting.")),
    });
  };
  const groups: { key: keyof typeof form; label: string; icon: IconName; tone: string; placeholder: string; suggestions?: string[]; max?: number }[] = [
    { key: 'target_industries', label: 'Industries', icon: 'building', tone: 'bg-brand-50 text-brand-700', placeholder: 'e.g. Healthcare, SaaS — press Enter' },
    { key: 'target_roles', label: 'Roles', icon: 'briefcase', tone: 'bg-teal-50 text-teal-700 dark:bg-teal-500/15 dark:text-teal-300', placeholder: 'e.g. VP Sales, CMO — press Enter' },
    { key: 'target_company_sizes', label: 'Company sizes', icon: 'users', tone: 'bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300', placeholder: 'Pick or type', suggestions: COMPANY_SIZES, max: 6 },
    { key: 'target_geographies', label: 'Regions', icon: 'mapPin', tone: 'bg-sky-50 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300', placeholder: 'e.g. North America, UAE — press Enter', max: 20 },
  ];
  const any = groups.some((g) => data[g.key].length);

  return (
    <SectionCard index={index} icon="target" tint="#10b981" title="Who you're targeting" subtitle="Your reach — leads that match rank higher in the admin score"
      actions={<Button size="sm" icon="pencil" onClick={start}>Edit</Button>}>
      <div className="space-y-3 p-4">
        {!any ? (
          <EmptyCta icon="target" title="Add who you want to reach" text="Industries, roles, company sizes and regions — the admin score ranks matching leads first." onClick={start} />
        ) : groups.filter((g) => data[g.key].length).map((g) => (
          <div key={g.key}><SubLabel>{g.label}</SubLabel><ChipRow icon={g.icon} values={data[g.key]} tone={g.tone} /></div>
        ))}
      </div>

      <Modal open={open} onOpenChange={(o) => { if (!o && !save.isPending) setOpen(false); }} size="lg" title="Who you're targeting"
        description="Admin console only — reps' targeting in the app doesn't change."
        footer={<>
          <Button variant="ghost" onClick={() => setOpen(false)} disabled={save.isPending}>Cancel</Button>
          <Button variant="primary" icon="save" type="submit" form="es-targeting-form" loading={save.isPending}>Save targeting</Button>
        </>}>
        <form id="es-targeting-form" onSubmit={submit} className="space-y-4">
          {groups.map((g) => (
            <Field key={g.key} label={g.label}>
              <TagInput values={form[g.key]} onChange={(v) => setForm((f) => ({ ...f, [g.key]: v }))} placeholder={g.placeholder} suggestions={g.suggestions} max={g.max ?? 20} maxLength={g.key === 'target_company_sizes' ? 40 : 80} />
            </Field>
          ))}
          <ErrorLine error={error} />
        </form>
      </Modal>
    </SectionCard>
  );
}

/* ------------------------------------------------------------------ sales approach */

function SalesApproachCard({ data, toast, index }: { data: AdminSalesProfile; toast: Toast; index: number }) {
  const save = useSave(toast, true);
  const [open, setOpen] = useState(false);
  const [style, setStyle] = useState<PitchStyle>('');
  const [vps, setVps] = useState<string[]>([]);
  const [objections, setObjections] = useState<AdminSalesProfile['common_objections']>([]);
  const [cases, setCases] = useState<AdminSalesProfile['case_studies']>([]);
  const [error, setError] = useState('');

  const start = () => {
    setStyle(data.pitch_style);
    setVps(data.value_propositions.length ? [...data.value_propositions] : ['']);
    setObjections(data.common_objections.map((o) => ({ ...o })));
    setCases(data.case_studies.map((c) => ({ ...c })));
    setError('');
    setOpen(true);
  };
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    save.mutate({
      pitch_style: style,
      value_propositions: vps.map((v) => v.trim()).filter(Boolean),
      common_objections: objections.filter((o) => o.objection.trim()).map((o) => ({ objection: o.objection.trim(), response: o.response.trim() })),
      case_studies: cases.filter((c) => c.title.trim()).map((c) => ({ title: c.title.trim(), result: c.result.trim() })),
    }, {
      onSuccess: () => { setOpen(false); toast('success', 'Sales approach saved', 'Admin pitches and the AI assistant use it from now on.'); },
      onError: (err) => setError(errorMessage(err, "Couldn't save the sales approach.")),
    });
  };

  const current = PITCH_STYLES.find((s) => s.value === data.pitch_style);
  const empty = !current && !data.value_propositions.length && !data.common_objections.length && !data.case_studies.length;
  const counts = useMemo(() => [
    { icon: 'objection' as IconName, n: data.common_objections.length, one: 'objection handled', many: 'objections handled' },
    { icon: 'trophy' as IconName, n: data.case_studies.length, one: 'case study', many: 'case studies' },
  ], [data.common_objections.length, data.case_studies.length]);

  return (
    <SectionCard index={index} icon="quote" tint="#f59e0b" title="Sales approach" subtitle="How you sell — used for pitch angles and follow-ups"
      actions={<Button size="sm" icon="pencil" onClick={start}>Edit</Button>}>
      <div className="space-y-3 p-4">
        {empty ? (
          <EmptyCta icon="quote" title="Tell the AI how you like to pitch" text="Pitch style, value propositions, objections you hear and proof points." onClick={start} />
        ) : (
          <>
            {current ? (
              <div className="flex items-center gap-2">
                <span className="rounded-full bg-fill px-2.5 py-0.5 text-xs font-bold text-fill-foreground">{current.label}</span>
                <span className="truncate text-xs text-muted-foreground">{current.hint}</span>
              </div>
            ) : null}
            {data.value_propositions.length ? (
              <ul className="space-y-1.5">
                {data.value_propositions.slice(0, 3).map((v, i) => (
                  <li key={v} className="flex items-start gap-2 text-sm text-body">
                    <span className="mt-0.5 shrink-0 text-amber-600"><LiveIcon name="lightbulb" size={13} delay={i * 0.6} /></span><span className="line-clamp-2">{v}</span>
                  </li>
                ))}
                {data.value_propositions.length > 3 ? <li className="pl-5 text-xs text-muted-foreground">+{data.value_propositions.length - 3} more value propositions</li> : null}
              </ul>
            ) : null}
            <div className="flex flex-wrap gap-2">
              {counts.map((c, i) => (
                <span key={c.one} className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1 text-xs text-body">
                  <span className="text-brand-700"><LiveIcon name={c.icon} size={12} delay={0.4 + i * 0.7} /></span><span className="es-num font-bold text-foreground">{c.n}</span>{c.n === 1 ? c.one : c.many}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      <Modal open={open} onOpenChange={(o) => { if (!o && !save.isPending) setOpen(false); }} size="lg" title="Sales approach"
        description="Admin console only — reps' sales approach in the app doesn't change."
        footer={<>
          <Button variant="ghost" onClick={() => setOpen(false)} disabled={save.isPending}>Cancel</Button>
          <Button variant="primary" icon="save" type="submit" form="es-sales-form" loading={save.isPending}>Save sales approach</Button>
        </>}>
        <form id="es-sales-form" onSubmit={submit} className="space-y-5">
          <div>
            <SubLabel>Pitch style</SubLabel>
            <div className="grid gap-2 sm:grid-cols-2">
              {PITCH_STYLES.map((s) => {
                const on = style === s.value;
                return (
                  <button key={s.value} type="button" aria-pressed={on} onClick={() => setStyle(on ? '' : s.value)}
                    className={cx('cursor-pointer rounded-xl border p-3 text-left transition-colors', on ? 'border-brand-400 bg-brand-50' : 'hover:border-foreground/25 hover:bg-secondary')}>
                    <span className="flex items-center justify-between gap-2">
                      <span className={cx('text-sm font-bold', on ? 'text-brand-700' : 'text-foreground')}>{s.label}</span>
                      {on ? <Icon name="check" size={14} className="text-brand-700" /> : null}
                    </span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">{s.hint}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <ListEditor title="Value propositions" addLabel="Add value proposition" rows={vps} setRows={setVps} blank=""
            render={(v, set) => <Input value={v} onChange={(e) => set(e.target.value)} maxLength={200} placeholder="e.g. Cut list-building time by 80%" />} />

          <ListEditor title="Common objections" addLabel="Add objection" rows={objections} setRows={setObjections} blank={{ objection: '', response: '' }}
            render={(o, set) => (
              <div className="grid flex-1 gap-1.5">
                <Input value={o.objection} onChange={(e) => set({ ...o, objection: e.target.value })} maxLength={400} placeholder="What they say — e.g. “We already have a data vendor”" />
                <Textarea rows={2} value={o.response} onChange={(e) => set({ ...o, response: e.target.value })} maxLength={400} placeholder="How you answer it" />
              </div>
            )} />

          <ListEditor title="Case studies & proof points" addLabel="Add case study" rows={cases} setRows={setCases} blank={{ title: '', result: '' }}
            render={(c, set) => (
              <div className="grid flex-1 gap-1.5">
                <Input value={c.title} onChange={(e) => set({ ...c, title: e.target.value })} maxLength={400} placeholder="Customer or project — e.g. Acme Health" />
                <Textarea rows={2} value={c.result} onChange={(e) => set({ ...c, result: e.target.value })} maxLength={400} placeholder="The result — e.g. 3× pipeline in one quarter" />
              </div>
            )} />
          <ErrorLine error={error} />
        </form>
      </Modal>
    </SectionCard>
  );
}

/** Up to 10 rows with add / remove; `render` draws one row's inputs. */
function ListEditor<T>({ title, addLabel, rows, setRows, blank, render }: {
  title: string; addLabel: string; rows: T[]; setRows: (r: T[]) => void; blank: T; render: (row: T, set: (v: T) => void) => React.ReactNode;
}) {
  return (
    <div>
      <SubLabel right={<span className="es-num text-[11px] text-muted-foreground">{rows.length}/{MAX_ITEMS}</span>}>{title}</SubLabel>
      <div className="space-y-2">
        <AnimatePresence initial={false}>
          {rows.map((row, i) => (
            <motion.div key={i} layout initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: MOTION.fast }} className="flex items-start gap-2">
              {render(row, (v) => setRows(rows.map((r, k) => (k === i ? v : r))))}
              <button type="button" onClick={() => setRows(rows.filter((_, k) => k !== i))} aria-label="Remove" title="Remove"
                className="mt-1.5 flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted-foreground hover:bg-secondary hover:text-destructive">
                <Icon name="x" size={14} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
        {rows.length < MAX_ITEMS ? (
          <Button type="button" size="sm" variant="ghost" icon="plus" onClick={() => setRows([...rows, blank])}>{addLabel}</Button>
        ) : null}
      </div>
    </div>
  );
}
