import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, type ExportPresetConfig } from '../../lib/api';
import { apiFetch } from '../../lib/auth';
import { UiProvider } from '../../ui/kit';
import { ContactExport } from './ContactExport';

vi.mock('../../lib/auth', () => ({ apiFetch: vi.fn() }));
const config: ExportPresetConfig = {
  columns: ['Name', 'Email', 'Website'], default: ['Name', 'Email'],
  presets: [{ name: 'ChampHarbinger', columns: ['Name', 'Email', 'Website'], filters: {}, builtin: true }],
};
let host: HTMLDivElement;
let root: Root;
let qc: QueryClient;
const button = (text: string) => Array.from(host.querySelectorAll('button')).find((b) => b.textContent === text)!;
const click = async (element: HTMLElement) => { await act(async () => { element.click(); }); };
const select = async (value: string) => {
  await act(async () => {
    const el = host.querySelector('select')!;
    el.value = value;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
};

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.mocked(apiFetch).mockReset();
  vi.mocked(apiFetch).mockImplementation(async (path) => path.includes('export-presets')
    ? Response.json(config)
    : new Response('Name,Email\r\n', { headers: { 'content-type': 'text/csv', 'x-row-count': '0' } }));
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL() { return 'blob:test-export'; }
    static revokeObjectURL() {}
  });
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  host = document.createElement('div'); document.body.append(host);
  root = createRoot(host);
  qc = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  qc.setQueryData(['export-presets'], config);
  await act(async () => root.render(<QueryClientProvider client={qc}><UiProvider><ContactExport events={[]} users={[]} lists={[]} /></UiProvider></QueryClientProvider>));
});
afterEach(async () => {
  await act(async () => root.unmount()); host.remove(); qc.clear();
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

describe('contact export configuration', () => {
  it('sends the checked columns in picker order in the request URL', async () => {
    await click(host.querySelector('[aria-label="Include Website"]')!);
    await click(host.querySelector('[aria-label="Move Website up"]')!);
    await click(host.querySelector('[aria-label="Include Email"]')!);
    await click(button('Download contacts CSV'));
    const path = vi.mocked(apiFetch).mock.calls.find(([url]) => url.startsWith('/api/admin/export?'))![0];
    expect(new URL(path, 'http://localhost').searchParams.get('columns')).toBe('Name,Website');
  });

  it('loads the built-in preset and never offers deletion for it', async () => {
    await select('ChampHarbinger');
    expect(button('Delete preset')).toBeUndefined();
    await click(button('Download contacts CSV'));
    expect(vi.mocked(apiFetch).mock.calls.at(-1)![0]).toContain('columns=Name%2CEmail%2CWebsite');
  });

  it('saves the chosen columns and filters then restores them when loaded', async () => {
    vi.spyOn(window, 'prompt').mockReturnValue('Sales');
    await click(host.querySelector('[aria-label="Move Email up"]')!);
    await act(async () => {
      const score = host.querySelectorAll('select')[4]!;
      score.value = '70'; score.dispatchEvent(new Event('change', { bubbles: true }));
    });
    const saved = { name: 'Sales', columns: ['Email', 'Name'], filters: { minScore: 70 }, builtin: false };
    vi.mocked(apiFetch).mockImplementation(async (path, init) => {
      if (init?.method === 'PUT') return Response.json({ ...config, presets: [...config.presets, saved] });
      return path.includes('export-presets') ? Response.json(config) : new Response('Email,Name\r\n');
    });
    await click(button('Save as preset'));
    const call = vi.mocked(apiFetch).mock.calls.find(([, init]) => init?.method === 'PUT')!;
    expect(JSON.parse(call[1]!.body as string)).toEqual({ presets: [{ name: 'Sales', columns: ['Email', 'Name'], filters: { minScore: 70 } }] });
    await select(''); await select('Sales');
    await click(button('Download contacts CSV'));
    const params = new URL(vi.mocked(apiFetch).mock.calls.at(-1)![0], 'http://localhost').searchParams;
    expect(params.get('columns')).toBe('Email,Name'); expect(params.get('minScore')).toBe('70');
    expect(button('Delete preset')).toBeDefined();
  });

  it('disables empty exports rather than silently exporting all columns', async () => {
    await click(host.querySelector('[aria-label="Include Name"]')!);
    await click(host.querySelector('[aria-label="Include Email"]')!);
    expect(button('Download contacts CSV').disabled).toBe(true);
    expect(button('Save as preset').disabled).toBe(true);
    await expect(api.exportAll({ columns: [] })).rejects.toThrow('Select at least one column.');
  });

  it('keeps omitted columns compatible and surfaces server 400 detail', async () => {
    await api.exportAll({ listId: 'list-one' });
    expect(vi.mocked(apiFetch).mock.calls.at(-1)![0]).toBe('/api/admin/export?listId=list-one');
    vi.mocked(apiFetch).mockResolvedValue(Response.json({ detail: 'Unknown export columns: Retired' }, { status: 400 }));
    await click(button('Download contacts CSV'));
    expect(host.querySelector('[role="alert"]')?.textContent).toBe('Unknown export columns: Retired');
    expect(button('Download contacts CSV').disabled).toBe(false);
  });
});
