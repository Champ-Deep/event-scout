import React from 'react';
import { Avatar as ShadAvatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';

interface AvatarProps {
  src?: string | null | undefined;
  name?: string | null | undefined;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  rounded?: 'full' | 'xl';
  className?: string;
}

const SIZES = { xs: 'size-6 text-[10px]', sm: 'size-8 text-xs', md: 'size-10 text-sm', lg: 'size-12 text-base', xl: 'size-14 text-lg' } as const;
// a stable tint per name, so the same person always gets the same colour
const TINTS = [
  'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200', 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-200',
  'bg-brand-100 text-accent-foreground', 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200',
  'bg-sky-100 text-sky-700 dark:bg-sky-500/20 dark:text-sky-200', 'bg-pink-100 text-pink-700 dark:bg-pink-500/20 dark:text-pink-200',
  'bg-teal-100 text-teal-700 dark:bg-teal-500/20 dark:text-teal-200',
];

export function initialsOf(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return parts.length > 1 ? `${parts[0]![0]}${parts[parts.length - 1]![0]}`.toUpperCase() : (parts[0] || 'LE').slice(0, 2).toUpperCase();
}

/** shadcn Avatar: the contact's photo (or the server's generated image), falling back to tinted initials. */
export function Avatar({ src, name, size = 'md', rounded = 'full', className = '' }: AvatarProps) {
  const clean = (name || 'Lead').trim();
  const tint = TINTS[Math.abs(clean.split('').reduce((a, c) => a + c.charCodeAt(0), 0)) % TINTS.length];
  const shape = rounded === 'full' ? 'rounded-full' : 'rounded-xl';
  return (
    <ShadAvatar className={cn(SIZES[size], shape, 'border border-border/70 shadow-2xs', className)}>
      {src ? <AvatarImage src={src} alt={clean} className={cn('object-cover', shape)} /> : null}
      <AvatarFallback className={cn('font-semibold', shape, tint)} delayMs={src ? 300 : 0}>{initialsOf(clean)}</AvatarFallback>
    </ShadAvatar>
  );
}
