import { afterEach, describe, expect, it, vi } from 'vitest';
import { contactFileSchema, createResponseCheck, exhibitorSchema } from '@es/types';
import { checked, checkResponse } from './checkResponse';

const file = { id: 'f1', filename: 'Proposal.pdf', fileType: 'pdf', category: 'general', fileSize: 48213, description: '', createdAt: null };

describe('checkResponse (the app adapter seam)', () => {
  afterEach(() => vi.restoreAllMocks());

  it('outside a dev build (__DEV__ unset, as in release) it is a no-op: same data, no warning', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const bad = { nope: true };
    expect(checkResponse(contactFileSchema, bad, 'GET /api/contact/{id}/files')).toBe(bad);
    expect(checked(contactFileSchema, 'x')(bad)).toBe(bad);
    expect(warn).not.toHaveBeenCalled();
  });

  it('in development it reports drift with the endpoint and the paths, and passes the data on', () => {
    const warn = vi.fn();
    const devCheck = createResponseCheck({ enabled: true, warn });
    const rows = [file, { ...file, fileSize: '48 KB' }];
    expect(devCheck(contactFileSchema.array(), rows, 'GET /api/contact/{id}/files')).toBe(rows);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toMatch(/GET \/api\/contact\/\{id\}\/files[\s\S]*\*\.fileSize/);
  });

  it('extra fields from a newer backend are not drift', () => {
    const warn = vi.fn();
    const devCheck = createResponseCheck({ enabled: true, warn });
    devCheck(exhibitorSchema, {
      id: 'x1', event: 'HIMSS 2026', name: 'Epic', booth: '', hall: '', category: '', country: '', website: '', description: '',
      subcategory: null, tags: [], products: [], addedByMe: true, userAdded: true, visited: false, visitedAt: null, visitNote: '',
      boothSize: '3x3',
    }, 'GET /api/exhibitors');
    expect(warn).not.toHaveBeenCalled();
  });
});
