import { describe, expect, it, afterEach } from 'vitest';
import { moveFocus, selectRange, selectedIn, shouldHandleKey, toggle } from './selection';

const order = ['a', 'b', 'c', 'd', 'e'];

describe('toggle', () => {
  it('adds and removes an id without mutating the input', () => {
    const sel = { a: true };
    expect(toggle(sel, 'b')).toEqual({ a: true, b: true });
    expect(toggle(sel, 'a')).toEqual({});
    expect(sel).toEqual({ a: true });
  });
});

describe('selectRange (shift-click)', () => {
  it('selects every row between anchor and target, either direction', () => {
    expect(selectRange({}, order, 'b', 'd')).toEqual({ b: true, c: true, d: true });
    expect(selectRange({}, order, 'd', 'b')).toEqual({ b: true, c: true, d: true });
  });
  it('keeps the rows that were already selected', () => {
    expect(selectRange({ e: true }, order, 'a', 'b')).toEqual({ a: true, b: true, e: true });
  });
  it('falls back to a toggle without an anchor in view', () => {
    expect(selectRange({}, order, null, 'c')).toEqual({ c: true });
    expect(selectRange({}, order, 'zz', 'c')).toEqual({ c: true });
  });
});

describe('selectedIn', () => {
  it('returns the selected ids still present, in list order', () => {
    expect(selectedIn({ d: true, a: true, gone: true }, order)).toEqual(['a', 'd']);
  });
});

describe('moveFocus (j / k)', () => {
  it('starts at the first row going down and the last going up', () => {
    expect(moveFocus(-1, 1, 5)).toBe(0);
    expect(moveFocus(-1, -1, 5)).toBe(4);
  });
  it('clamps at both ends', () => {
    expect(moveFocus(4, 1, 5)).toBe(4);
    expect(moveFocus(0, -1, 5)).toBe(0);
    expect(moveFocus(2, 1, 5)).toBe(3);
  });
  it('returns -1 for an empty list', () => {
    expect(moveFocus(0, 1, 0)).toBe(-1);
  });
});

describe('shouldHandleKey', () => {
  afterEach(() => { document.body.innerHTML = ''; });
  const key = (target: EventTarget, init: KeyboardEventInit = {}) => {
    const e = new KeyboardEvent('keydown', { key: 'j', ...init });
    Object.defineProperty(e, 'target', { value: target });
    return e;
  };
  it('handles plain keys on the page', () => {
    expect(shouldHandleKey(key(document.body))).toBe(true);
  });
  it('ignores keys while typing', () => {
    const input = document.createElement('input');
    document.body.append(input);
    expect(shouldHandleKey(key(input))).toBe(false);
  });
  it('ignores modified keys (⌘K belongs to the palette)', () => {
    expect(shouldHandleKey(key(document.body, { metaKey: true }))).toBe(false);
    expect(shouldHandleKey(key(document.body, { ctrlKey: true }))).toBe(false);
  });
  it('ignores keys while a dialog is open', () => {
    const d = document.createElement('div');
    d.setAttribute('role', 'dialog');
    document.body.append(d);
    expect(shouldHandleKey(key(document.body))).toBe(false);
  });
});
