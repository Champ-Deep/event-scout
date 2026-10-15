// Live icons: a Lucide icon with its own small motion on a loop (the target pings, the box bounces,
// the speech bubble talks, the phone buzzes…). Each loop rests between beats so a page full of them
// stays calm. Static when the console's Animations setting is off (useReduceMotion).
import React from 'react';
import { motion, type TargetAndTransition, type Transition } from 'motion/react';
import { useReduceMotion } from '../lib/motion';
import { Icon, type IconName } from './icons';

interface Loop { animate: TargetAndTransition; transition: Transition; ring?: boolean }

const ease = 'easeInOut' as const;
const forever = (duration: number, repeatDelay = 0): Transition => ({ duration, repeat: Infinity, repeatDelay, ease });

const LOOPS: Partial<Record<IconName, Loop>> = {
  target: { animate: { scale: [1, 1.14, 1] }, transition: forever(1.8), ring: true },
  shield: { animate: { scale: [1, 1.08, 1], rotate: [0, -5, 0] }, transition: forever(2.4), ring: true },
  package: { animate: { y: [0, -3.5, 0, -1, 0], rotate: [0, -7, 5, 0, 0] }, transition: forever(1.3, 1.4) },
  quote: { animate: { rotate: [0, -12, 9, -6, 0], scale: [1, 1.1, 1, 1.05, 1] }, transition: forever(1.2, 1.6) },
  building: { animate: { y: [0, -2.5, 0], scaleY: [1, 1.07, 1] }, transition: forever(2.2) },
  smartphone: { animate: { x: [0, -1.5, 1.5, -1.5, 1.5, 0], rotate: [0, -7, 7, -7, 7, 0] }, transition: { duration: 0.6, repeat: Infinity, repeatDelay: 2.4 } },
  check: { animate: { scale: [1, 1.3, 1], rotate: [0, -10, 0] }, transition: forever(0.55, 2.6) },
  lightbulb: { animate: { opacity: [1, 0.35, 1, 0.6, 1], scale: [1, 0.92, 1.12, 1, 1] }, transition: forever(1.6, 1.4) },
  trophy: { animate: { y: [0, -3, 0], rotate: [0, -10, 10, 0] }, transition: forever(1.2, 1.8) },
  objection: { animate: { rotate: [0, 14, -10, 6, 0] }, transition: forever(1.1, 2) },
  chevronRight: { animate: { x: [0, 3, 0] }, transition: forever(1.2) },
  sparkles: { animate: { rotate: [0, 15, -10, 0], scale: [1, 1.15, 1] }, transition: forever(1.6, 1) },
  mapPin: { animate: { y: [0, -3, 0] }, transition: forever(1, 1.6) },
  // sidebar pages
  dashboard: { animate: { scale: [1, 0.86, 1.1, 1], rotate: [0, 0, -6, 0] }, transition: forever(1.2, 1.2) },
  activity: { animate: { scaleY: [1, 1.35, 0.8, 1.15, 1] }, transition: forever(1, 0.8) },
  contacts: { animate: { y: [0, -2.5, 0], rotate: [0, -8, 0] }, transition: forever(1.1, 1.4) },
  duplicate: { animate: { x: [0, 2, -1, 0], y: [0, -2, 1, 0] }, transition: forever(1.1, 1.4) },
  trash: { animate: { rotate: [0, -14, 10, -6, 0], y: [0, -1.5, 0, 0, 0] }, transition: forever(1, 1.6) },
  list: { animate: { x: [0, 2.5, 0], scaleX: [1, 1.08, 1] }, transition: forever(1.1, 1.2) },
  pipeline: { animate: { rotate: [0, -16, 12, 0], scale: [1, 1.1, 1, 1] }, transition: forever(1.2, 1.2) },
  store: { animate: { y: [0, -2.5, 0], scale: [1, 1.08, 1] }, transition: forever(1.3, 1.2) },
  file: { animate: { rotate: [0, -10, 6, 0], y: [0, -2, 0, 0] }, transition: forever(1.1, 1.5) },
  megaphone: { animate: { rotate: [0, -14, 10, -8, 0], scale: [1, 1.15, 1, 1.08, 1] }, transition: forever(1, 1.4) },
  users: { animate: { scale: [1, 1.14, 1], y: [0, -1.5, 0] }, transition: forever(1.2, 1.2) },
  download: { animate: { y: [0, 3, 0] }, transition: forever(0.9, 1.1) },
  server: { animate: { opacity: [1, 0.4, 1, 0.7, 1], scale: [1, 0.94, 1, 1, 1] }, transition: forever(1.4, 1) },
  bolt: { animate: { scale: [1, 1.22, 1], rotate: [0, -10, 0], opacity: [1, 0.75, 1] }, transition: forever(1, 1.6) },
  // cards, stat tiles, empty states
  flame: { animate: { scale: [1, 1.15, 0.95, 1.1, 1], rotate: [0, -6, 4, -3, 0] }, transition: forever(1.1, 0.8) },
  mail: { animate: { y: [0, -2.5, 0], rotate: [0, -8, 0] }, transition: forever(1, 1.4) },
  send: { animate: { x: [0, 2.5, 0], y: [0, -2.5, 0] }, transition: forever(1, 1.4) },
  clock: { animate: { rotate: [0, 20, -10, 0] }, transition: forever(1.2, 1.4) },
  calendar: { animate: { scaleY: [1, 0.85, 1.08, 1], y: [0, 1, -1, 0] }, transition: forever(1.1, 1.4) },
  database: { animate: { scaleY: [1, 1.12, 0.94, 1] }, transition: forever(1.2, 1.2) },
  key: { animate: { rotate: [0, -25, 15, 0] }, transition: forever(1.2, 1.6) },
  user: { animate: { y: [0, -2, 0], scale: [1, 1.06, 1] }, transition: forever(1.2, 1.4) },
  eye: { animate: { scaleY: [1, 0.1, 1] }, transition: { duration: 0.35, repeat: Infinity, repeatDelay: 2.6 } },
  upload: { animate: { y: [0, -3, 0] }, transition: forever(0.9, 1.1) },
  refresh: { animate: { rotate: [0, 360] }, transition: { duration: 1.2, repeat: Infinity, repeatDelay: 1.4, ease } },
  alert: { animate: { rotate: [0, -10, 10, -6, 6, 0] }, transition: forever(0.7, 1.8) },
  info: { animate: { scale: [1, 1.15, 1] }, transition: forever(1.2, 1.4) },
  link: { animate: { rotate: [0, -20, 0], scale: [1, 1.08, 1] }, transition: forever(1.2, 1.4) },
  webhook: { animate: { rotate: [0, -18, 14, 0] }, transition: forever(1.2, 1.4) },
  globe: { animate: { rotate: [0, 18, -18, 0] }, transition: forever(1.8, 1) },
  phone: { animate: { rotate: [0, -14, 14, -10, 10, 0] }, transition: { duration: 0.7, repeat: Infinity, repeatDelay: 2 } },
  qr: { animate: { scale: [1, 0.9, 1.08, 1] }, transition: forever(1.1, 1.4) },
  merge: { animate: { scaleX: [1, 0.85, 1.05, 1] }, transition: forever(1.1, 1.4) },
  ticket: { animate: { rotate: [0, -10, 8, 0] }, transition: forever(1.1, 1.4) },
  flag: { animate: { skewX: [0, -10, 8, 0], rotate: [0, -4, 0, 0] }, transition: forever(1.2, 1.2) },
  chats: { animate: { y: [0, -2, 0], scale: [1, 1.08, 1] }, transition: forever(1.1, 1.4) },
};
const DEFAULT: Loop = { animate: { scale: [1, 1.1, 1] }, transition: forever(2) };
const REST: TargetAndTransition = { x: 0, y: 0, rotate: 0, scale: 1, scaleX: 1, scaleY: 1, skewX: 0, opacity: 1 };

/**
 * A looping animated icon. `delay` offsets the loop (stagger a list so the icons go off one after another).
 * `play={false}` holds it still (it eases back to rest) — e.g. only the active sidebar tab plays.
 * Same props as <Icon>; the colour comes from the surrounding text colour.
 */
export function LiveIcon({ name, size, className, delay = 0, play = true }: { name: IconName; size?: number; className?: string; delay?: number; play?: boolean }) {
  const reduce = useReduceMotion();
  const loop = LOOPS[name] ?? DEFAULT;
  const on = play && !reduce;
  // The same elements whether playing or not — only the animation changes. Swapping the <svg> while
  // someone is pressing it makes the browser drop the click (the sidebar tabs toggle play on hover).
  // pointer-events-none: clicks always land on the button / link around the icon, never the moving svg.
  return (
    <span className="pointer-events-none relative inline-flex shrink-0 items-center justify-center">
      {loop.ring ? (
        <motion.span aria-hidden className="absolute -inset-[40%] rounded-full border-2 border-current"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={on ? { opacity: [0.55, 0], scale: [0.5, 1.45] } : { opacity: 0, scale: 0.5 }}
          transition={on ? { duration: 1.8, repeat: Infinity, ease: 'easeOut', delay } : { duration: 0.2 }} />
      ) : null}
      <motion.span className="inline-flex" animate={on ? loop.animate : REST}
        transition={on ? { ...loop.transition, delay } : { duration: 0.3, ease }}>
        <Icon name={name} {...(size ? { size } : {})} {...(className ? { className } : {})} />
      </motion.span>
    </span>
  );
}
