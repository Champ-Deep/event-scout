// Row selection + keyboard focus for the contacts grid (pure, unit-tested in selection.test.ts).
export type Selection = Record<string, boolean>;

export function toggle(sel: Selection, id: string): Selection {
  const next = { ...sel };
  if (next[id]) delete next[id];
  else next[id] = true;
  return next;
}

/**
 * Shift-click: select every row between the anchor and the clicked row (inclusive), keeping
 * what was already selected. Without an anchor in view it just toggles the clicked row.
 */
export function selectRange(sel: Selection, order: string[], anchor: string | null, target: string): Selection {
  const a = anchor ? order.indexOf(anchor) : -1;
  const b = order.indexOf(target);
  if (a < 0 || b < 0) return toggle(sel, target);
  const [lo, hi] = a < b ? [a, b] : [b, a];
  const next = { ...sel };
  for (let i = lo; i <= hi; i++) next[order[i]!] = true;
  return next;
}

/** Selected ids that are still in the list, in list order. */
export function selectedIn(sel: Selection, order: string[]): string[] {
  return order.filter((id) => sel[id]);
}

/** j / k: move the focused row, clamped; -1 (nothing focused) starts at the first row. */
export function moveFocus(index: number, delta: number, length: number): number {
  if (length <= 0) return -1;
  if (index < 0) return delta > 0 ? 0 : length - 1;
  return Math.max(0, Math.min(length - 1, index + delta));
}

/** Ignore shortcuts while typing or while a dialog or menu is open. */
export function shouldHandleKey(e: KeyboardEvent): boolean {
  if (e.metaKey || e.ctrlKey || e.altKey) return false;
  const t = e.target as HTMLElement | null;
  if (t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName))) return false;
  return !document.querySelector('[role="dialog"], [role="alertdialog"], [role="menu"]');
}
