import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
    tailwindcss() as any,
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@es/types': path.resolve(__dirname, '../../packages/types/src/index.ts'),
    },
  },
  server: {
    port: 3000,
    // Default "localhost" binds IPv6 ::1 only on this Windows box, so 127.0.0.1:3000 / the LAN IP were refused.
    host: true,
    // Fail loudly if 3000 is taken instead of silently moving to 3001.
    strictPort: true,
    // Every backend call goes through this proxy (no separate API URL / port to keep in sync).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Sign-in + token refresh. Regex so the /login page itself is still served by Vite.
      '^/login/$': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/auth/refresh': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Forgot password on the sign-in page (emailed 6-digit code, the app's flow).
      '^/auth/password/(forgot|reset)$': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
