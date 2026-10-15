import React, { createContext, useContext, useId, useLayoutEffect, useMemo, useSyncExternalStore } from 'react';
import { View, StyleSheet } from 'react-native';

/**
 * Window-level layer for overlays that must cover the whole app, tab bar included (what an RN
 * <Modal> used to do, without a Modal's unreliable taps on Android). Mount <OverlayProvider>
 * once around the tabs; <WindowPortal> renders its children in the host drawn after them.
 * The host renders nothing while no overlay is open.
 *
 * The overlay's state, hooks and back-button handling stay with the component that renders the
 * portal; only the elements are drawn in the host.
 */

type Store = {
  set: (key: string, node: React.ReactNode) => void;
  remove: (key: string) => void;
  subscribe: (fn: () => void) => () => void;
  snapshot: () => number;
  layers: () => Array<[string, React.ReactNode]>;
};

function createStore(): Store {
  const layers = new Map<string, React.ReactNode>(); // insertion order = stacking order
  const listeners = new Set<() => void>();
  let version = 0;
  const emit = () => { version += 1; listeners.forEach((fn) => fn()); };
  return {
    set: (key, node) => { layers.set(key, node); emit(); },
    remove: (key) => { if (layers.delete(key)) emit(); },
    subscribe: (fn) => { listeners.add(fn); return () => { listeners.delete(fn); }; },
    snapshot: () => version,
    layers: () => Array.from(layers.entries()),
  };
}

// The store object never changes, so portals don't re-render when the host does.
const OverlayContext = createContext<Store | null>(null);

export function OverlayProvider({ children }: { children: React.ReactNode }) {
  const store = useMemo(createStore, []);
  return (
    <OverlayContext.Provider value={store}>
      {children}
      <OverlayHost store={store} />
    </OverlayContext.Provider>
  );
}

function OverlayHost({ store }: { store: Store }) {
  useSyncExternalStore(store.subscribe, store.snapshot, store.snapshot);
  const layers = store.layers();
  if (!layers.length) return null;
  return (
    <View pointerEvents="box-none" style={StyleSheet.absoluteFill}>
      {layers.map(([key, node]) => <React.Fragment key={key}>{node}</React.Fragment>)}
    </View>
  );
}

/** Draws `children` in the window-level host (above the tab bar). Without a provider it renders in place. */
export function WindowPortal({ children }: { children: React.ReactNode }) {
  const store = useContext(OverlayContext);
  const key = useId();
  // Every render pushes the latest elements to the host.
  useLayoutEffect(() => { store?.set(key, children); });
  useLayoutEffect(() => () => store?.remove(key), [store, key]);
  return store ? null : <>{children}</>;
}
