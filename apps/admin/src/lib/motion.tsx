// The console's animation setting: On (default) · Off · Follow system.
//
// Windows turns "Animation effects" off on many work PCs (and over remote desktop), and Chrome then
// reports prefers-reduced-motion for every site — which used to switch off nearly every animation
// here without anyone choosing that. Now the admin picks (account menu → Animations); "Follow system"
// keeps the old behaviour for anyone who needs reduced motion.
//   - <MotionConfig> gets reducedMotion "never" / "always" from this setting;
//   - <html class="es-reduce-motion"> switches off the CSS animations (styles/tokens.css);
//   - components ask useReduceMotion() instead of reading the OS setting themselves.
import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { MotionConfig } from 'motion/react';

export type MotionPref = 'on' | 'off' | 'system';
const KEY = 'es_admin_motion';
const QUERY = '(prefers-reduced-motion: reduce)';

interface MotionPrefValue { pref: MotionPref; setPref: (p: MotionPref) => void; reduce: boolean; systemReduce: boolean }
const Ctx = createContext<MotionPrefValue>({ pref: 'on', setPref: () => {}, reduce: false, systemReduce: false });

function readPref(): MotionPref {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'off' || v === 'system' ? v : 'on';
  } catch {
    return 'on';
  }
}

export function MotionPrefProvider({ children }: { children: React.ReactNode }) {
  const [pref, setPrefState] = useState<MotionPref>(readPref);
  const [systemReduce, setSystemReduce] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(QUERY);
    if (!mq) return;
    const on = () => setSystemReduce(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  const reduce = pref === 'off' || (pref === 'system' && systemReduce);
  useEffect(() => { document.documentElement.classList.toggle('es-reduce-motion', reduce); }, [reduce]);
  const value = useMemo<MotionPrefValue>(() => ({
    pref, reduce, systemReduce,
    setPref: (p) => { setPrefState(p); try { localStorage.setItem(KEY, p); } catch { /* not remembered */ } },
  }), [pref, reduce, systemReduce]);
  return (
    <Ctx.Provider value={value}>
      <MotionConfig reducedMotion={reduce ? 'always' : 'never'}>{children}</MotionConfig>
    </Ctx.Provider>
  );
}

export const useMotionPref = () => useContext(Ctx);
/** True when animations should be calm (the admin turned them off, or follows a system that asks for it). */
export const useReduceMotion = () => useContext(Ctx).reduce;
