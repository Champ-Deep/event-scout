import { useEffect, useState } from 'react';

/** True once `busy` has stayed on for `afterMs` — used to say "still working" on slow requests. */
export function useSlowHint(busy: boolean, afterMs = 5000): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!busy) {
      setSlow(false);
      return;
    }
    const t = setTimeout(() => setSlow(true), afterMs);
    return () => clearTimeout(t);
  }, [busy, afterMs]);
  return slow;
}
