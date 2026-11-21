import { useLayoutEffect, useRef } from "react";

/** Auto-grows a textarea to its content, between `rows` and the CSS max-height. Replaces the private registry hook. */
export function useTextareaResize(value: string, rows = 1) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    const line = parseFloat(getComputedStyle(el).lineHeight) || 24;
    el.style.height = `${Math.max(el.scrollHeight, line * rows)}px`;
  }, [value, rows]);
  return ref;
}
