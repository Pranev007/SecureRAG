import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import { ApiRequestError, api, tokenStore } from '../services/api';
import type { User } from '../types/api';

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // On boot, a stored token is *verified against the server* rather than
  // trusted. The token may be expired, or the account deactivated; asking
  // /auth/me is the only way to know, and it keeps the client's idea of "signed
  // in" identical to the server's.
  useEffect(() => {
    if (!tokenStore.get()) {
      setLoading(false);
      return;
    }
    api
      .me()
      .then(setUser)
      .catch(() => tokenStore.clear())
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const result = await api.login(email, password);
    tokenStore.set(result.access_token);
    setUser(result.user);
  }, []);

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      const result = await api.register(email, password, fullName);
      tokenStore.set(result.access_token);
      setUser(result.user);
    },
    [],
  );

  const logout = useCallback(() => {
    tokenStore.clear();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout }),
    [user, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside an AuthProvider');
  return context;
}

/** Turn an unknown thrown value into something safe to show a user. */
export function describeError(error: unknown): string {
  if (error instanceof ApiRequestError) {
    if (error.fields?.length) {
      return error.fields.map((f) => `${f.field}: ${f.message}`).join('; ');
    }
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'Something went wrong.';
}
