/**
 * API client.
 *
 * One place that knows how to talk to the backend, so error handling and token
 * attachment cannot drift between call sites.
 *
 * The token is held in `localStorage`. That is the pragmatic choice for a
 * portfolio project with a JWT backend, and it is worth being clear about the
 * trade-off rather than pretending it is free: localStorage is readable by any
 * script running on the page, so it is vulnerable to XSS in a way that an
 * httpOnly cookie is not. The production-grade answer is an httpOnly, SameSite
 * cookie plus CSRF protection; that is recorded in docs/security.md under
 * Limitations rather than glossed over here.
 */

import type {
  AttackScenario,
  ChatResponse,
  ChatSession,
  ChatSessionDetail,
  DocumentDetail,
  DocumentSummary,
  DocumentUploadResponse,
  Page,
  PlaygroundResult,
  PlaygroundSuite,
  SecurityEvent,
  SecurityStats,
  TimeseriesPoint,
  TokenResponse,
  User,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';
const TOKEN_KEY = 'securerag.token';

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly requestId?: string | null,
    readonly fields?: { field: string; message: string }[],
  ) {
    super(message);
    this.name = 'ApiRequestError';
  }
}

export const tokenStore = {
  get: (): string | null => localStorage.getItem(TOKEN_KEY),
  set: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

type RequestOptions = Omit<RequestInit, 'body'> & { body?: unknown };

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, headers, ...rest } = options;
  const token = tokenStore.get();

  const finalHeaders: Record<string, string> = {
    ...(headers as Record<string, string>),
  };
  if (token) finalHeaders.Authorization = `Bearer ${token}`;

  let payload: BodyInit | undefined;
  if (body instanceof FormData) {
    // Let the browser set the multipart boundary; setting Content-Type by hand
    // here produces a boundary-less header and a 400 from the server.
    payload = body;
  } else if (body !== undefined) {
    finalHeaders['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: finalHeaders,
    body: payload,
  });

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const data = text ? JSON.parse(text) : {};

  if (!response.ok) {
    const detail = data?.error ?? {};
    // A 401 means the token is gone or expired. Clearing it here means the
    // whole app converges on the login screen without every caller needing to
    // handle the case.
    if (response.status === 401) tokenStore.clear();
    throw new ApiRequestError(
      detail.message ?? `Request failed (${response.status})`,
      response.status,
      detail.code ?? 'unknown_error',
      detail.request_id,
      detail.fields,
    );
  }

  return data as T;
}

export const api = {
  // --- auth ---
  register: (email: string, password: string, fullName?: string) =>
    request<TokenResponse>('/auth/register', {
      method: 'POST',
      body: { email, password, full_name: fullName || null },
    }),

  login: (email: string, password: string) =>
    request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: { email, password },
    }),

  me: () => request<User>('/auth/me'),

  // --- documents ---
  listDocuments: () => request<Page<DocumentSummary>>('/documents?limit=200'),

  getDocument: (id: string, includeChunks = false) =>
    request<DocumentDetail>(
      `/documents/${id}${includeChunks ? '?include_chunks=true' : ''}`,
    ),

  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<DocumentUploadResponse>('/documents', {
      method: 'POST',
      body: form,
    });
  },

  deleteDocument: (id: string) =>
    request<{ message: string }>(`/documents/${id}`, { method: 'DELETE' }),

  // --- chat ---
  ask: (question: string, sessionId?: string | null, documentIds?: string[]) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      body: {
        question,
        session_id: sessionId ?? null,
        document_ids: documentIds?.length ? documentIds : null,
      },
    }),

  listSessions: () => request<Page<ChatSession>>('/chat/sessions?limit=100'),
  getSession: (id: string) => request<ChatSessionDetail>(`/chat/${id}`),
  deleteSession: (id: string) =>
    request<{ message: string }>(`/chat/${id}`, { method: 'DELETE' }),

  // --- security ---
  stats: (windowDays = 30) =>
    request<SecurityStats>(`/security/stats?window_days=${windowDays}`),

  timeseries: (days = 14) =>
    request<TimeseriesPoint[]>(`/security/timeseries?days=${days}`),

  events: (params: { limit?: number; severity?: string[]; eventType?: string[] } = {}) => {
    const query = new URLSearchParams();
    query.set('limit', String(params.limit ?? 100));
    params.severity?.forEach((s) => query.append('severity', s));
    params.eventType?.forEach((t) => query.append('event_type', t));
    return request<Page<SecurityEvent>>(`/security/events?${query}`);
  },

  // --- playground ---
  scenarios: () => request<AttackScenario[]>('/security/playground/scenarios'),

  runScenario: (input: { scenarioId?: string; payload?: string; surface?: string }) =>
    request<PlaygroundResult>('/security/playground/run', {
      method: 'POST',
      body: {
        scenario_id: input.scenarioId ?? null,
        payload: input.payload ?? null,
        surface: input.surface ?? null,
      },
    }),

  runAllScenarios: () =>
    request<PlaygroundSuite>('/security/playground/run-all', { method: 'POST' }),
};
