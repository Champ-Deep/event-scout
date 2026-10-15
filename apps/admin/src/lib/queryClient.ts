import { QueryClient } from '@tanstack/react-query';

// The cache is the store (no client-side state library). Loaders fill it; SSE and optimistic
// mutations write to it.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 30,
      refetchOnWindowFocus: false,
      retry: 1, // a failing loader shows its error screen quickly instead of retrying 3×
    },
  },
});
