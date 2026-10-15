import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { contactName, fmtBytes, levelOf, parseTime, relTime } from './kit';
import { niceMax, niceStep } from './charts';

describe('parseTime', () => {
  it('reads zone-less backend (SQLite) times as UTC', () => {
    expect(parseTime('2026-09-30 10:00:00')?.toISOString()).toBe('2026-09-30T10:00:00.000Z');
    expect(parseTime('2026-09-30T10:00:00')?.toISOString()).toBe('2026-09-30T10:00:00.000Z');
  });
  it('keeps an explicit zone', () => {
    expect(parseTime('2026-09-30T10:00:00Z')?.toISOString()).toBe('2026-09-30T10:00:00.000Z');
    expect(parseTime('2026-09-30T10:00:00+05:30')?.toISOString()).toBe('2026-09-30T04:30:00.000Z');
  });
  it('returns null for empty or invalid input', () => {
    expect(parseTime(null)).toBeNull();
    expect(parseTime('')).toBeNull();
    expect(parseTime('not a date')).toBeNull();
  });
});

describe('relTime', () => {
  beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(new Date('2026-10-01T12:00:00Z')); });
  afterEach(() => { vi.useRealTimers(); });
  it('describes recent times in plain words', () => {
    expect(relTime('2026-10-01 11:59:50')).toBe('just now');
    expect(relTime('2026-10-01 11:55:00')).toBe('5 min ago');
    expect(relTime('2026-10-01 09:00:00')).toBe('3 h ago');
    expect(relTime('2026-09-29 12:00:00')).toBe('2 d ago');
  });
  it('shows a dash for missing times', () => {
    expect(relTime(undefined)).toBe('—');
    expect(relTime('garbage')).toBe('—');
  });
});

describe('niceStep / niceMax (chart axes)', () => {
  it('rounds steps to 1 · 2 · 5 × 10ⁿ', () => {
    expect(niceStep(38)).toBe(10);
    expect(niceStep(7)).toBe(2);
    expect(niceStep(130)).toBe(50);
    expect(niceStep(0)).toBe(1);
  });
  it('the axis top always covers the max', () => {
    for (const v of [1, 3, 9, 13, 38, 99, 101, 777, 4321]) expect(niceMax(v)).toBeGreaterThanOrEqual(v);
  });
});

describe('small formatters', () => {
  it('contactName / fmtBytes / levelOf', () => {
    expect(contactName({ firstName: 'Ana', lastName: null })).toBe('Ana');
    expect(contactName({})).toBe('Unnamed contact');
    expect(fmtBytes(512)).toBe('512 B');
    expect(fmtBytes(2048)).toBe('2 KB');
    expect(fmtBytes(3 * 1024 * 1024)).toBe('3.0 MB');
    expect(levelOf(85)?.key).toBe('hot');
    expect(levelOf(49)?.key).toBe('cold');
    expect(levelOf(null)).toBeNull();
  });
});
