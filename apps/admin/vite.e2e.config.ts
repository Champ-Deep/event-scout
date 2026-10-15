// TEMPORARY (render check): preview the build against the test backend on :8001. Deleted afterwards.
import { defineConfig, mergeConfig } from 'vite';
import base from './vite.config';

const t = { target: 'http://127.0.0.1:8001', changeOrigin: true };
export default mergeConfig(base, defineConfig({
  preview: { proxy: { '/api': t, '^/login/$': t, '/auth/refresh': t, '^/auth/password/(forgot|reset)$': t, '/register/': t } },
}));
