import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';

import {
  DecisionBadge,
  EmptyState,
  ErrorBanner,
  Panel,
  ScoreMeter,
  Spinner,
  cx,
  formatRelative,
  humanise,
} from '../components/ui';
import { describeError } from '../hooks/useAuth';
import { api } from '../services/api';
import type {
  ChatResponse,
  ChatSession,
  Citation,
  DocumentSummary,
  SecurityStatus,
} from '../types/api';

interface Turn {
  id: string;
  question: string;
  response: ChatResponse | null;
  pending: boolean;
  error?: string;
}

const SUGGESTIONS = [
  'What is the annual leave policy?',
  'How often must passwords be rotated?',
  'Summarise the security policy.',
];

// ---------------------------------------------------------------------------

function SourceCard({ source }: { source: Citation }) {
  return (
    <li className="rounded-lg border border-ink-800 bg-ink-950/50 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-2 text-xs font-medium text-ink-100">
            <span className="badge-accent">[{source.index}]</span>
            <span className="truncate">{source.filename}</span>
          </p>
          <p className="mt-1 text-[11px] text-ink-400">
            {source.page !== null && `Page ${source.page}`}
            {source.page !== null && source.section && ' · '}
            {source.section}
          </p>
        </div>
        <span
          className={source.verified ? 'badge-safe shrink-0' : 'badge-warn shrink-0'}
          title={
            source.verified
              ? 'The quoted span was found in this chunk'
              : 'The quoted span could not be located in this chunk'
          }
        >
          {source.verified ? 'verified' : 'unverified'}
        </span>
      </div>
      {source.quote && (
        <blockquote className="mt-2 border-l-2 border-ink-700 pl-2.5 text-[11px] italic leading-relaxed text-ink-400">
          {source.quote}
        </blockquote>
      )}
    </li>
  );
}

function SecurityPanel({
  security,
  retrieved,
  latency,
  timings,
}: {
  security: SecurityStatus;
  retrieved: number;
  latency: number;
  timings: Record<string, number>;
}) {
  const decision = security.blocked
    ? 'BLOCKED'
    : security.refused
      ? 'REFUSED'
      : security.pii_detected
        ? 'REDACTED'
        : 'ALLOWED';

  return (
    <div className="mt-3 rounded-lg border border-ink-800 bg-ink-950/40 p-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <DecisionBadge decision={decision} />
        {security.reason && (
          <span className="badge-neutral">{humanise(security.reason)}</span>
        )}
        {security.pii_types.map((type) => (
          <span key={type} className="badge-warn">
            {type} redacted
          </span>
        ))}
        {security.warnings.map((warning) => (
          <span key={warning} className="badge-neutral">
            {humanise(warning)}
          </span>
        ))}
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <ScoreMeter
          label="Input risk"
          value={security.risk_score}
          hint="Combined prompt-injection risk. Higher is worse."
        />
        <ScoreMeter
          label="Grounding"
          value={security.grounding_score}
          invert
          hint="How well the answer is supported by the retrieved text. Higher is better."
        />
        <ScoreMeter
          label="Confidence"
          value={security.confidence}
          invert
          hint="The model's own reported confidence."
        />
      </div>

      <p className="mt-3 border-t border-ink-800/70 pt-2.5 font-mono text-[10px] leading-relaxed text-ink-500">
        {retrieved} chunk{retrieved === 1 ? '' : 's'} · {latency.toFixed(0)} ms total
        {Object.entries(timings).length > 0 && ' · '}
        {Object.entries(timings)
          .map(([stage, value]) => `${stage.replace(/_ms$/, '')} ${value.toFixed(0)}`)
          .join(' · ')}
      </p>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <article className="animate-slide-up space-y-3">
      <div className="flex justify-end">
        <p className="max-w-[75%] rounded-2xl rounded-br-md bg-accent/12 px-4 py-2.5 text-sm text-ink-100">
          {turn.question}
        </p>
      </div>

      {turn.pending && (
        <div className="flex items-center gap-2.5 text-sm text-ink-400">
          <Spinner className="h-3.5 w-3.5" />
          Running guardrails, retrieval and verification…
        </div>
      )}

      {turn.error && <ErrorBanner message={turn.error} />}

      {turn.response && (
        <div className="max-w-[85%]">
          <div
            className={cx(
              'rounded-2xl rounded-bl-md px-4 py-3 text-sm leading-relaxed',
              turn.response.security.blocked
                ? 'border border-danger/40 bg-danger/8 text-ink-200'
                : turn.response.security.refused
                  ? 'border border-warn/35 bg-warn/8 text-ink-200'
                  : 'border border-ink-800 bg-ink-900/80 text-ink-100',
            )}
          >
            {turn.response.answer}
          </div>

          {turn.response.sources.length > 0 && (
            <div className="mt-3">
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-400">
                Sources
              </p>
              <ul className="grid gap-2 sm:grid-cols-2">
                {turn.response.sources.map((source) => (
                  <SourceCard key={source.chunk_id} source={source} />
                ))}
              </ul>
            </div>
          )}

          <SecurityPanel
            security={turn.response.security}
            retrieved={turn.response.retrieved_chunk_count}
            latency={turn.response.latency_ms}
            timings={turn.response.timings_ms}
          />
        </div>
      )}
    </article>
  );
}

// ---------------------------------------------------------------------------

export function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshSidebar = useCallback(async () => {
    try {
      const [sessionPage, documentPage] = await Promise.all([
        api.listSessions(),
        api.listDocuments(),
      ]);
      setSessions(sessionPage.items);
      setDocuments(documentPage.items);
    } catch {
      // The sidebar is supporting information; a failure here must not take
      // down the chat surface, which is the point of the page.
    }
  }, []);

  useEffect(() => {
    void refreshSidebar();
  }, [refreshSidebar]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns]);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    const turnId = crypto.randomUUID();
    setTurns((current) => [
      ...current,
      { id: turnId, question: trimmed, response: null, pending: true },
    ]);
    setQuestion('');
    setBusy(true);
    setError(null);

    try {
      const response = await api.ask(trimmed, sessionId);
      setSessionId(response.session_id);
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId ? { ...turn, response, pending: false } : turn,
        ),
      );
      void refreshSidebar();
    } catch (caught) {
      const message = describeError(caught);
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId ? { ...turn, pending: false, error: message } : turn,
        ),
      );
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  async function openSession(id: string) {
    setBusy(true);
    try {
      const detail = await api.getSession(id);
      setSessionId(id);
      // Stored messages are replayed as turns. Assistant metadata is
      // reconstructed from what was persisted, so reopening a conversation
      // shows the same security verdicts it showed live.
      const restored: Turn[] = [];
      for (let i = 0; i < detail.messages.length; i += 2) {
        const userMessage = detail.messages[i];
        const assistant = detail.messages[i + 1];
        if (!userMessage) break;
        restored.push({
          id: userMessage.id,
          question: userMessage.content,
          pending: false,
          response: assistant
            ? {
                answer: assistant.content,
                session_id: id,
                message_id: assistant.id,
                sources: assistant.citations ?? [],
                security: {
                  blocked: assistant.was_blocked,
                  refused: Boolean(
                    (assistant.meta as { refused?: boolean })?.refused,
                  ),
                  reason: assistant.block_reason,
                  risk_score: assistant.risk_score,
                  grounding_score: assistant.grounding_score ?? 0,
                  confidence:
                    (assistant.meta as { confidence?: number })?.confidence ?? 0,
                  pii_detected: assistant.pii_detected,
                  pii_types: [],
                  warnings:
                    (assistant.meta as { warnings?: string[] })?.warnings ?? [],
                },
                retrieved_chunk_count: assistant.retrieved_chunk_count,
                latency_ms: assistant.latency_ms,
                timings_ms:
                  (assistant.meta as { timings_ms?: Record<string, number> })
                    ?.timings_ms ?? {},
                request_id: null,
              }
            : null,
        });
      }
      setTurns(restored);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setBusy(false);
    }
  }

  const readyDocuments = documents.filter((d) => d.status === 'ready');

  return (
    <div className="grid gap-5 lg:grid-cols-[260px_1fr]">
      <aside className="space-y-4">
        <Panel
          title="Conversations"
          actions={
            <button
              onClick={() => {
                setSessionId(null);
                setTurns([]);
              }}
              className="btn-ghost btn-sm"
            >
              New
            </button>
          }
        >
          <ul className="max-h-64 overflow-y-auto p-2">
            {sessions.length === 0 && (
              <li className="px-2 py-3 text-xs text-ink-500">No conversations yet.</li>
            )}
            {sessions.map((session) => (
              <li key={session.id}>
                <button
                  onClick={() => void openSession(session.id)}
                  className={cx(
                    'w-full truncate rounded-md px-2.5 py-2 text-left text-xs transition-colors',
                    session.id === sessionId
                      ? 'bg-ink-800 text-ink-100'
                      : 'text-ink-400 hover:bg-ink-800/60 hover:text-ink-200',
                  )}
                  title={session.title}
                >
                  <span className="block truncate">{session.title}</span>
                  <span className="text-[10px] text-ink-500">
                    {formatRelative(session.updated_at)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Corpus" subtitle={`${readyDocuments.length} document(s) searchable`}>
          <ul className="max-h-56 space-y-1 overflow-y-auto p-2">
            {readyDocuments.length === 0 && (
              <li className="px-2 py-3 text-xs text-ink-500">
                Upload a document to start asking questions.
              </li>
            )}
            {readyDocuments.map((document) => (
              <li
                key={document.id}
                className="flex items-center justify-between gap-2 rounded-md px-2.5 py-1.5"
              >
                <span className="truncate text-xs text-ink-300" title={document.filename}>
                  {document.filename}
                </span>
                {document.quarantined_chunk_count > 0 && (
                  <span
                    className="badge-warn shrink-0"
                    title={`${document.quarantined_chunk_count} section(s) quarantined`}
                  >
                    {document.quarantined_chunk_count}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </Panel>
      </aside>

      <Panel className="flex min-h-[calc(100vh-8rem)] flex-col">
        <div className="flex-1 space-y-6 overflow-y-auto p-5">
          {turns.length === 0 ? (
            <EmptyState
              title="Ask a question about your documents"
              description="Every request passes through input guardrails, access-scoped retrieval, context sanitisation and output verification before you see an answer."
              action={
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => void submit(suggestion)}
                      className="btn-ghost btn-sm"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              }
            />
          ) : (
            turns.map((turn) => <TurnView key={turn.id} turn={turn} />)
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={(event: FormEvent) => {
            event.preventDefault();
            void submit(question);
          }}
          className="border-t border-ink-800/80 p-4"
        >
          <ErrorBanner message={error} />
          <div className="mt-2 flex gap-2">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about your documents…"
              className="input"
              disabled={busy}
              maxLength={4000}
            />
            <button type="submit" disabled={busy || !question.trim()} className="btn-primary">
              {busy ? <Spinner /> : 'Ask'}
            </button>
          </div>
          <p className="mt-2 text-[11px] text-ink-500">
            Answers are drawn only from documents you own, and are checked for
            grounding, citation validity and personal data before display.
          </p>
        </form>
      </Panel>
    </div>
  );
}
