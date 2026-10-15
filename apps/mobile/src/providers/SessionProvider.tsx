import React, { createContext, useContext, useEffect, useMemo, useSyncExternalStore, ReactNode } from 'react';
import { AppState } from 'react-native';
import { router } from 'expo-router';
import { session } from '../features/session';

/**
 * React binding over the session module (src/features/session): screens read the session and call
 * login / logout through useSession(); the module owns the tokens, the stored account and the
 * offline-cache hand-over.
 */
interface SessionContextValue {
  isAuthenticated: boolean;
  userId: string | null;
  userEmail: string | null;
  userName: string | null;
  authMethod: string | null;
  login: (token: string, refreshToken: string, user?: { name: string, email: string }, authMethod?: string) => Promise<void>;
  logout: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

const login: SessionContextValue['login'] = (token, refreshToken, user, authMethod) =>
  session.signIn({ token, refreshToken, user, authMethod });

// Unsynced changes stay on the phone for when this account signs back in (the Me sign-out sheet
// says so); a different account signing in wipes them first.
const logout = () => session.signOut({ keepOfflineData: true });

export const SessionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const state = useSyncExternalStore(session.subscribe, session.current);

  // Signed out on another device (e.g. password reset): the session has ended — say why.
  useEffect(() => session.onRevoked(() => {
    router.replace({ pathname: '/login' as any, params: { signedOut: '1' } });
  }), []);

  // Check on launch and whenever the app returns to the foreground, so a revoked session is
  // noticed even before the next data request, and renew the sign-in when it's due.
  useEffect(() => {
    if (!state.isAuthenticated) return;
    const check = () => {
      session.renewIfDue();
      session.check();
    };
    check();
    const sub = AppState.addEventListener('change', (s) => {
      if (s === 'active') check();
    });
    return () => sub.remove();
  }, [state.isAuthenticated]);

  const value = useMemo<SessionContextValue>(() => ({ ...state, login, logout }), [state]);

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
};

export const useSession = () => {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error('useSession must be used within SessionProvider');
  }
  return ctx;
};
