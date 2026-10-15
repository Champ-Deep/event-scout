import React from 'react';
import { useRouter, type ErrorComponentProps } from '@tanstack/react-router';
import { useQueryErrorResetBoundary } from '@tanstack/react-query';
import { Button, Card, Skeleton } from './kit';
import { Icon } from './icons';

/** Skeleton shown while a route's loader runs (pendingComponent). */
export function PageSkeleton({ kind = 'table' }: { kind?: 'table' | 'dashboard' | 'cards' | 'split' }) {
  const header = (
    <div className="mb-6 flex items-end justify-between">
      <div className="space-y-2"><Skeleton className="h-6 w-48" /><Skeleton className="h-4 w-80" /></div>
      <Skeleton className="h-9 w-36 rounded-xl" />
    </div>
  );
  if (kind === 'dashboard') {
    return (
      <div aria-busy="true">
        {header}
        <Skeleton className="mb-5 h-12 w-full rounded-2xl" />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-40 rounded-2xl" />)}</div>
        <div className="mt-4 grid gap-4 xl:grid-cols-3"><Skeleton className="h-96 rounded-2xl xl:col-span-2" /><Skeleton className="h-96 rounded-2xl" /></div>
      </div>
    );
  }
  if (kind === 'cards') {
    return <div aria-busy="true">{header}<div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{[0, 1, 2, 3, 4, 5].map((i) => <Skeleton key={i} className="h-52 rounded-2xl" />)}</div></div>;
  }
  if (kind === 'split') {
    return <div aria-busy="true">{header}<div className="grid gap-4 lg:grid-cols-[280px_1fr]"><Skeleton className="h-72 rounded-2xl" /><Skeleton className="h-96 rounded-2xl" /></div></div>;
  }
  return (
    <div aria-busy="true">
      {header}
      <Card>
        <div className="flex gap-3 border-b border-border p-4"><Skeleton className="h-9 w-72" /><Skeleton className="h-9 w-96" /></div>
        <div className="divide-y divide-border">
          {Array.from({ length: 10 }, (_, i) => (
            <div key={i} className="flex items-center gap-4 px-4 py-3">
              <Skeleton className="h-4 w-4" /><Skeleton className="h-8 w-8 rounded-full" /><Skeleton className="h-4 w-48" /><Skeleton className="h-4 w-32" />
              <Skeleton className="ml-auto h-5 w-20 rounded-full" />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/** Error shown when a route's loader (or a suspense query) fails (errorComponent). Retry reloads it. */
export function RouteError({ error, reset }: ErrorComponentProps) {
  const router = useRouter();
  const boundary = useQueryErrorResetBoundary();
  const message = error instanceof Error ? error.message : 'Something went wrong.';
  return (
    <Card className="mx-auto mt-10 max-w-lg p-8 text-center" role="alert">
      <span className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-300">
        <Icon name="alert" size={22} />
      </span>
      <h2 className="text-base font-semibold text-foreground">This page couldn't load</h2>
      <p className="mt-1 text-sm text-muted-foreground">{message}</p>
      <div className="mt-5 flex justify-center gap-2">
        <Button icon="refresh" variant="primary" onClick={() => { boundary.reset(); reset(); router.invalidate(); }}>Try again</Button>
      </div>
    </Card>
  );
}
