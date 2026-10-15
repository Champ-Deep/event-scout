import { z } from 'zod';

/**
 * Building blocks of the wire contract.
 *
 * Every response object is `loose`: a field the backend adds later never fails validation (zod
 * `.passthrough()`). The TypeScript types are inferred from the same schemas through `Wire<>`, which drops
 * the `[key: string]: unknown` that passthrough adds to the inferred type — so `keyof`, `Omit` and typo
 * checks keep working on the types the apps use.
 */

/** An object schema that accepts (and keeps) fields it doesn't name. */
export const loose = <S extends z.ZodRawShape>(shape: S) => z.object(shape).passthrough();

type NoExtraKeys<T> = {
  [K in keyof T as string extends K ? (unknown extends T[K] ? never : K) : number extends K ? never : K]: Wire<T[K]>;
};

/** The app-facing type of a schema's output: the passthrough index signature removed, all the way down. */
export type Wire<T> = T extends readonly (infer U)[]
  ? Wire<U>[]
  : T extends object
    ? NoExtraKeys<T>
    : T;

/** The app-facing type a contract schema describes. */
export type WireOf<S extends z.ZodTypeAny> = Wire<z.output<S>>;

/** Most issue lines a drift report prints (the rest are counted). */
const MAX_REPORTED = 8;

/**
 * A short report of how `data` differs from the contract, or null when it matches.
 * Array indices are folded (`contacts.*.scoreBand`), so 500 rows with the same drift are one line.
 */
export function describeDrift(schema: z.ZodTypeAny, data: unknown, endpoint: string): string | null {
  const result = schema.safeParse(data);
  if (result.success) return null;
  const counts = new Map<string, number>();
  for (const issue of result.error.issues) {
    const path = issue.path.map((p) => (typeof p === 'number' ? '*' : p)).join('.') || '(root)';
    const line = `${path}: ${issue.message}`;
    counts.set(line, (counts.get(line) ?? 0) + 1);
  }
  const lines = [...counts.entries()];
  const shown = lines.slice(0, MAX_REPORTED).map(([line, n]) => `  - ${line}${n > 1 ? ` (x${n})` : ''}`);
  if (lines.length > MAX_REPORTED) shown.push(`  - …and ${lines.length - MAX_REPORTED} more`);
  return `[contract drift] ${endpoint} — the response doesn't match @es/types:\n${shown.join('\n')}`;
}

export type ResponseCheck = <T>(schema: z.ZodTypeAny, data: T, endpoint: string) => T;

/**
 * The response check an HTTP adapter runs at its seam. Enabled (development): validates `data` against
 * the contract and warns about drift. Disabled (production): returns at once, nothing is parsed.
 * Either way it returns `data` unchanged and never throws.
 */
export function createResponseCheck(opts: { enabled: boolean; warn?: (message: string) => void }): ResponseCheck {
  return function checkResponse<T>(schema: z.ZodTypeAny, data: T, endpoint: string): T {
    if (!opts.enabled) return data;
    try {
      const report = describeDrift(schema, data, endpoint);
      if (report) (opts.warn ?? console.warn)(report);
    } catch {
      // a check must never break a request
    }
    return data;
  };
}
