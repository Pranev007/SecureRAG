import { NavLink, Navigate, Route, Routes } from 'react-router-dom';

import { Spinner } from './components/ui';
import { useAuth } from './hooks/useAuth';
import { ChatPage } from './pages/ChatPage';
import { DashboardPage } from './pages/DashboardPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { LoginPage } from './pages/LoginPage';
import { PlaygroundPage } from './pages/PlaygroundPage';

const NAV = [
  { to: '/chat', label: 'Chat' },
  { to: '/documents', label: 'Documents' },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/playground', label: 'Playground' },
];

function ShieldMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
      <path
        d="M12 2.5 4.5 5.6v5.7c0 4.6 3.2 8.9 7.5 10.2 4.3-1.3 7.5-5.6 7.5-10.2V5.6L12 2.5Z"
        className="fill-accent/15 stroke-accent"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path
        d="m8.8 12 2.2 2.2 4.2-4.4"
        className="stroke-accent-glow"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}

function AppShell({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-30 border-b border-ink-800/80 bg-ink-950/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-5">
          <div className="flex items-center gap-2.5">
            <ShieldMark />
            <span className="text-[15px] font-semibold tracking-tight">
              Secure<span className="text-accent-glow">RAG</span>
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  [
                    'rounded-lg px-3 py-1.5 text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-ink-800 text-ink-100'
                      : 'text-ink-400 hover:text-ink-200',
                  ].join(' ')
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <div className="hidden text-right sm:block">
              <p className="text-xs font-medium text-ink-200">{user?.email}</p>
              <p className="text-[11px] text-ink-500">
                {user?.role === 'admin' ? 'Administrator' : 'User'}
              </p>
            </div>
            <button onClick={logout} className="btn-ghost btn-sm">
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1400px] flex-1 px-5 py-6">{children}</main>
    </div>
  );
}

export default function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-accent" />
      </div>
    );
  }

  if (!user) return <LoginPage />;

  return (
    <AppShell>
      <Routes>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </AppShell>
  );
}
