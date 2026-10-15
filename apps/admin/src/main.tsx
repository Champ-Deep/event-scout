import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from '@tanstack/react-router';
import { MotionPrefProvider } from './lib/motion';
import { router } from './router';
import { queryClient } from './lib/queryClient';
import { ThemeProvider } from './lib/theme';
import { UiProvider } from './ui/kit';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/sonner';
import './styles/tokens.css';

export function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delayDuration={250}>
          <UiProvider>
            {/* animations follow the console's own setting (account menu → Animations: On · Off · Follow system) */}
            <MotionPrefProvider>
              <RouterProvider router={router} />
            </MotionPrefProvider>
            <Toaster />
          </UiProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

declare global {
  interface Window {
    __es_root?: ReactDOM.Root;
  }
}

const rootEl = document.getElementById('root');
if (rootEl) {
  if (!window.__es_root) {
    window.__es_root = ReactDOM.createRoot(rootEl);
  }
  window.__es_root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}