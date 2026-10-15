export const logger = {
  info: (msg: string, meta?: Record<string, unknown>) => {
    console.log(`[INFO] ${msg}`, meta ?? '');
  },
  warn: (msg: string, meta?: Record<string, unknown>) => {
    console.warn(`[WARN] ${msg}`, meta ?? '');
  },
  error: (msg: string, error?: unknown) => {
    console.error(`[ERROR] ${msg}`, error ?? '');
  },
};
