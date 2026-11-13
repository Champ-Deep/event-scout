import { useSyncExternalStore } from "react";

let reachable: boolean | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((fn) => fn());
const browserChanged = () => {
  reachable = null;
  notify();
};

// A completed read is stronger evidence than a stale navigator.onLine flag.
export function reportConnection(value: boolean | null) {
  if (reachable !== value) {
    reachable = value;
    notify();
  }
}
const subscribe = (fn: () => void) => {
  listeners.add(fn);
  if (listeners.size === 1) {
    window.addEventListener("online", browserChanged);
    window.addEventListener("offline", browserChanged);
  }
  return () => {
    listeners.delete(fn);
    if (!listeners.size) {
      window.removeEventListener("online", browserChanged);
      window.removeEventListener("offline", browserChanged);
      reachable = null;
    }
  };
};
export const connectionAvailable = () => reachable ?? navigator.onLine;

export function useConnection() {
  return useSyncExternalStore(subscribe, connectionAvailable, () => true);
}
