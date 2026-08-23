import { useState, type FormEvent } from 'react';

import { ErrorBanner, Spinner } from '../components/ui';
import { describeError, useAuth } from '../hooks/useAuth';

const FEATURES = [
  ['Input guardrails', 'Layered prompt-injection detection before any model call'],
  ['Access-scoped retrieval', 'Ownership enforced in SQL, never in the prompt'],
  ['Context sanitisation', 'Instructions hidden inside documents are quarantined'],
  ['Output verification', 'Grounding, citations and PII checked before you see it'],
];

export function LoginPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === 'login') await login(email, password);
      else await register(email, password, fullName);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-full lg:grid-cols-[1.1fr_1fr]">
      {/* Narrative panel: explains what the project is before asking for a login. */}
      <aside className="hidden flex-col justify-center border-r border-ink-800/60 px-14 lg:flex">
        <div className="max-w-lg">
          <h1 className="text-4xl font-semibold tracking-tight">
            Secure<span className="text-accent-glow">RAG</span>
          </h1>
          <p className="mt-4 text-[15px] leading-relaxed text-ink-300">
            A retrieval-augmented generation system built on the assumption that{' '}
            <strong className="font-semibold text-ink-100">
              both the user and the documents are untrusted
            </strong>
            .
          </p>

          <ul className="mt-9 space-y-5">
            {FEATURES.map(([title, description]) => (
              <li key={title} className="flex gap-3.5">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                <div>
                  <p className="text-sm font-medium text-ink-100">{title}</p>
                  <p className="mt-0.5 text-[13px] leading-relaxed text-ink-400">
                    {description}
                  </p>
                </div>
              </li>
            ))}
          </ul>

          <p className="mt-10 border-t border-ink-800/60 pt-5 text-xs leading-relaxed text-ink-500">
            No security control here is claimed to be complete. Every layer is
            measured, and the measurements — including the failures — are in the
            evaluation report.
          </p>
        </div>
      </aside>

      <div className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <h2 className="text-xl font-semibold">
            {mode === 'login' ? 'Sign in' : 'Create an account'}
          </h2>
          <p className="mt-1.5 text-sm text-ink-400">
            {mode === 'login'
              ? 'Access your private document workspace.'
              : 'The first account created becomes the administrator.'}
          </p>

          <form onSubmit={handleSubmit} className="mt-7 space-y-4">
            {mode === 'register' && (
              <div>
                <label className="label" htmlFor="fullName">Name</label>
                <input
                  id="fullName"
                  className="input"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Ada Lovelace"
                  autoComplete="name"
                />
              </div>
            )}

            <div>
              <label className="label" htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                required
                className="input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
              />
            </div>

            <div>
              <label className="label" htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                required
                className="input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 10 characters"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              {mode === 'register' && (
                <p className="mt-1.5 text-[11px] text-ink-500">
                  Minimum 10 characters, with letters and digits.
                </p>
              )}
            </div>

            <ErrorBanner message={error} />

            <button type="submit" disabled={busy} className="btn-primary w-full">
              {busy && <Spinner />}
              {mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>

          <button
            onClick={() => {
              setMode(mode === 'login' ? 'register' : 'login');
              setError(null);
            }}
            className="mt-5 w-full text-center text-xs text-ink-400 transition-colors hover:text-ink-200"
          >
            {mode === 'login'
              ? "Don't have an account? Create one"
              : 'Already have an account? Sign in'}
          </button>
        </div>
      </div>
    </div>
  );
}
