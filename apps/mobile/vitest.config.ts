import path from 'node:path';
import { defineConfig } from 'vitest/config';

/**
 * Unit tests for the app's pure modules (sync engine, contact repository, profile rules, backend
 * error classifier, request pipeline). They run in Node: the modules under test take their backend
 * and storage as adapters, so no React Native / Expo code is loaded. The aliases below are a guard —
 * if a tested module ever imports react-native or an expo module, the test fails loudly instead of
 * trying to load native code.
 */
const stub = path.resolve(__dirname, 'test/native-stub.ts');

export default defineConfig({
  resolve: {
    alias: [
      { find: /^react-native$/, replacement: stub },
      { find: /^react-native-mmkv$/, replacement: stub },
      { find: /^expo-[a-z-]+(\/.*)?$/, replacement: stub },
    ],
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts', 'test/**/*.test.ts'],
  },
});
