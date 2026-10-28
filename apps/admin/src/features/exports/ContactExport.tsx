import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, saveBlob, type ExportFilters, type ExportPreset, type ExportPresetConfig } from '../../lib/api';
import type { AdminEvent, AdminUser, ContactList } from '../../lib/types';
import { Button, Checkbox, Field, Select, errorMessage, useUi } from '../../ui/kit';

const presetKey = ['export-presets'];
type Props = { events: AdminEvent[]; users: AdminUser[]; lists: ContactList[] };

export function ContactExport(props: Props) {
  const config = useQuery({ queryKey: presetKey, queryFn: api.exportPresets.get });
  if (config.isPending) return <p role="status" className="mb-5 text-body">Loading export columns and presets…</p>;
  if (config.isError) return <div className="mb-5 space-y-2">
    <p role="alert" className="text-destructive">Could not load export settings: {errorMessage(config.error)}</p>
    <Button onClick={() => void config.refetch()}>Retry export settings</Button>
  </div>;
  return <ContactExportEditor {...props} config={config.data} />;
}

function ContactExportEditor({ config, events, users, lists }: Props & { config: ExportPresetConfig }) {
  const qc = useQueryClient();
  const { toast, confirm } = useUi();
  const [columns, setColumns] = useState(config.default);
  const [filters, setFilters] = useState<ExportFilters>({});
  const [selected, setSelected] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const preset = config.presets.find((p) => p.name === selected);
  const saved = config.presets.filter((p) => !p.builtin);

  const changeColumns = (next: string[]) => { setColumns(next); setSelected(''); };
  const changeFilter = (next: ExportFilters) => { setFilters(next); setSelected(''); };
  const load = (name: string) => {
    const p = config.presets.find((item) => item.name === name);
    setSelected(name);
    setColumns(p ? p.columns : config.default);
    setFilters(p ? p.filters : {});
    setError('');
  };
  const move = (index: number, direction: number) => {
    const next = [...columns];
    const target = index + direction;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target]!, next[index]!];
    changeColumns(next);
  };
  const persist = async (presets: ExportPreset[], name: string) => {
    setBusy(true); setError('');
    try {
      const next = await api.exportPresets.put(presets);
      qc.setQueryData(presetKey, next);
      setSelected(name);
      toast('success', name ? 'Preset saved' : 'Preset deleted');
    } catch (e) { setError(errorMessage(e)); }
    finally { setBusy(false); }
  };
  const save = () => {
    const name = window.prompt('Name this export preset')?.trim();
    if (!name) return;
    if (name.length > 60) { setError('Use a preset name of 60 characters or fewer.'); return; }
    if (config.presets.some((p) => p.name.toLowerCase() === name.toLowerCase())) {
      setError('That preset name already exists. Choose a different name.'); return;
    }
    void persist([...saved, { name, columns, filters }], name);
  };
  const remove = async () => {
    if (!preset || preset.builtin) return;
    if (await confirm({ title: `Delete ${preset.name}?`, message: 'Only this saved preset is removed. Contacts are not changed.', confirmLabel: 'Delete preset', danger: true })) {
      await persist(saved.filter((p) => p.name !== preset.name), '');
    }
  };
  const download = async () => {
    setBusy(true); setError('');
    try {
      const { blob, filename, rows } = await api.exportAll({ ...filters, columns });
      saveBlob(blob, filename);
      toast('success', 'Export downloaded', rows == null ? filename : `${rows} contacts. ${filename}`);
    } catch (e) { setError(errorMessage(e)); }
    finally { setBusy(false); }
  };

  return <section aria-labelledby="contact-export-title" className="mb-5 rounded-xl border bg-card p-5 text-foreground shadow-card">
    <h2 id="contact-export-title" className="text-lg font-semibold">Contact export</h2>
    <p className="mt-1 text-sm text-body">Choose columns in CSV order, narrow the contacts, then download. Presets are saved to your admin account.</p>
    <fieldset disabled={busy} className="mt-4 min-w-0 space-y-5 disabled:opacity-70">
      <legend className="sr-only">Export configuration</legend>
      <div className="flex flex-wrap items-end gap-2">
        <Field label="Load preset" className="min-w-0 flex-1 basis-64">
          <Select aria-label="Load preset" className="h-11 text-base text-foreground sm:text-sm" value={selected} onChange={(e) => load(e.target.value)}>
            <option value="">Custom / default columns</option>
            {config.presets.map((p) => <option key={p.name} value={p.name}>{p.name}{p.builtin ? ' (built-in)' : ''}</option>)}
          </Select>
        </Field>
        <Button className="min-h-11" variant="secondary" disabled={!columns.length} onClick={save}>Save as preset</Button>
        {preset && !preset.builtin ? <Button className="min-h-11" variant="danger" onClick={() => void remove()}>Delete preset</Button> : null}
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold">Columns ({columns.length} selected)</h3>
          <p className="mb-2 mt-1 text-sm text-body">Use Up and Down to change the CSV order.</p>
          <ul aria-label="Export columns" className="divide-y rounded-lg border">
            {[...columns, ...config.columns.filter((c) => !columns.includes(c))].map((column) => {
              const index = columns.indexOf(column);
              return <li key={column} className="flex min-h-12 items-center gap-2 px-2">
                <label className="flex min-h-11 min-w-0 flex-1 cursor-pointer items-center gap-2 text-sm">
                  <Checkbox label={`Include ${column}`} checked={index >= 0} className="size-11 shrink-0"
                    onChange={(checked) => changeColumns(checked ? [...columns, column] : columns.filter((c) => c !== column))} />
                  <span>{index >= 0 ? `${index + 1}. ` : ''}{column}</span>
                </label>
                {index >= 0 ? <div className="flex gap-1">
                  <Button className="min-h-11 min-w-11 px-2" variant="ghost" aria-label={`Move ${column} up`} disabled={index === 0} onClick={() => move(index, -1)}>Up</Button>
                  <Button className="min-h-11 min-w-11 px-2" variant="ghost" aria-label={`Move ${column} down`} disabled={index === columns.length - 1} onClick={() => move(index, 1)}>Down</Button>
                </div> : null}
              </li>;
            })}
          </ul>
        </div>
        <div className="min-w-0 space-y-4">
          <h3 className="text-sm font-semibold">Which contacts</h3>
          <p className="text-sm text-body">Leave filters on All for the whole workspace. Combined filters must all match.</p>
          <Field label="Event">
            <Select aria-label="Event" className="h-11 text-base text-foreground sm:text-sm" value={filters.event ?? ''} onChange={(e) => changeFilter({ ...filters, event: e.target.value || undefined })}>
              <option value="">All events</option>
              {filters.event && !events.some((e) => e.name === filters.event) ? <option value={filters.event}>{filters.event} (saved filter)</option> : null}
              {events.map((e) => <option key={e.key} value={e.name}>{e.name}</option>)}
            </Select>
          </Field>
          <Field label="Rep">
            <Select aria-label="Rep" className="h-11 text-base text-foreground sm:text-sm" value={filters.userId ?? ''} onChange={(e) => changeFilter({ ...filters, userId: e.target.value || undefined })}>
              <option value="">All reps</option>
              {filters.userId && !users.some((u) => u.id === filters.userId) ? <option value={filters.userId}>Unavailable rep (saved filter)</option> : null}
              {users.map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
            </Select>
          </Field>
          <Field label="List">
            <Select aria-label="List" className="h-11 text-base text-foreground sm:text-sm" value={filters.listId ?? ''} onChange={(e) => changeFilter({ ...filters, listId: e.target.value || undefined })}>
              <option value="">All lists</option>
              {filters.listId && !lists.some((l) => l.id === filters.listId) ? <option value={filters.listId}>Unavailable list (saved filter)</option> : null}
              {lists.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
            </Select>
          </Field>
          <Field label="Minimum score">
            <Select aria-label="Minimum score" className="h-11 text-base text-foreground sm:text-sm" value={filters.minScore ?? ''} onChange={(e) => changeFilter({ ...filters, minScore: e.target.value === '' ? undefined : Number(e.target.value) })}>
              <option value="">All scores</option>
              <option value="70">Hot &amp; warm leads (70+)</option>
              {filters.minScore != null && filters.minScore !== 70 ? <option value={filters.minScore}>{filters.minScore}+ (saved filter)</option> : null}
            </Select>
          </Field>
          <Button className="min-h-11 w-full" variant="primary" disabled={!columns.length} icon="download" onClick={() => void download()}>Download contacts CSV</Button>
          {!columns.length ? <p role="status" className="text-sm text-destructive">Select at least one column to save or download.</p> : null}
        </div>
      </div>
    </fieldset>
    {busy ? <p role="status" className="mt-3 text-sm text-body">Working…</p> : null}
    {error ? <p role="alert" className="mt-3 text-sm text-destructive">{error}</p> : null}
  </section>;
}
