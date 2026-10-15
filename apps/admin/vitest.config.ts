import path from 'path';
import { defineConfig } from 'vitest/config';

// Unit tests for the pure admin logic (selection, URL search schemas, CSV, time/axis helpers,
// optimistic cache patches, table prefs). jsdom for the few DOM-aware helpers (keyboard guard).
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
