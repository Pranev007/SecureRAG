/**
 * API types.
 *
 * These mirror the Pydantic response models in `backend/app/schemas/`. They are
 * written by hand rather than generated so the file stays readable, but they are
 * the contract: if the backend changes a field, `npm run typecheck` fails here
 * rather than the UI silently rendering `undefined`.
 */

export interface ApiError {
  error: {
    code: string;
    message: string;
    request_id?: string | null;
    blocked?: boolean;
    reason?: string;
    fields?: { field: string; message: string }[];
  };
}

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: 'user' | 'admin';
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in_seconds: number;
  user: User;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface DocumentSummary {
  id: string;
  filename: string;
  extension: string;
  content_type: string;
  file_size_bytes: number;
  status: DocumentStatus;
  error_message: string | null;
  visibility: string;
  page_count: number;
  chunk_count: number;
  char_count: number;
  max_injection_risk: number;
  quarantined_chunk_count: number;
  created_at: string;
}

export interface DocumentChunk {
  id: string;
  chunk_index: number;
  content: string;
  page_number: number | null;
  section: string | null;
  token_count: number;
  char_count: number;
  injection_risk: number;
  is_quarantined: boolean;
  injection_labels: string[];
}

export interface DocumentDetail extends DocumentSummary {
  chunks: DocumentChunk[];
}

export interface DocumentUploadResponse {
  document: DocumentSummary;
  message: string;
  warnings: string[];
}

export interface Citation {
  index: number;
  document_id: string;
  chunk_id: string;
  filename: string;
  page: number | null;
  section: string | null;
  quote: string;
  verified: boolean;
  label: string;
}

export interface SecurityStatus {
  blocked: boolean;
  refused: boolean;
  reason: string | null;
  risk_score: number;
  grounding_score: number;
  confidence: number;
  pii_detected: boolean;
  pii_types: string[];
  warnings: string[];
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  message_id: string;
  sources: Citation[];
  security: SecurityStatus;
  retrieved_chunk_count: number;
  latency_ms: number;
  timings_ms: Record<string, number>;
  request_id: string | null;
}

export interface StoredMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  was_blocked: boolean;
  block_reason: string | null;
  risk_score: number;
  grounding_score: number | null;
  pii_detected: boolean;
  citations: Citation[];
  retrieved_chunk_count: number;
  latency_ms: number;
  created_at: string;
  meta: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: StoredMessage[];
}

export interface SecurityStats {
  scope: 'user' | 'system';
  window_days: number;
  queries: { total: number; blocked: number; block_rate: number };
  security: {
    prompt_injection_attempts: number;
    indirect_injection_detections: number;
    grounding_failures: number;
    pii_detections: number;
    rate_limit_violations: number;
    authorization_denials: number;
  };
  documents: { total: number; chunks: number; quarantined_chunks: number };
  performance: {
    average_latency_ms: number;
    average_grounding_score: number;
    average_retrieved_chunks: number;
  };
  events: {
    by_severity: Record<string, number>;
    by_type: Record<string, number>;
    total: number;
  };
}

export interface SecurityEvent {
  id: string;
  request_id: string | null;
  user_id: string | null;
  event_type: string;
  layer: string;
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  action: string;
  risk_score: number;
  detector: string | null;
  resource_type: string | null;
  resource_id: string | null;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface TimeseriesPoint {
  date: string;
  total: number;
  attacks: number;
}

export interface AttackScenario {
  id: string;
  category: string;
  surface: 'user_input' | 'document' | 'model_output';
  name: string;
  payload: string;
  description: string;
  expected: string;
}

export interface DetectorFinding {
  layer: string;
  detector: string;
  score: number;
  detail: string;
}

export interface PlaygroundResult {
  scenario_id: string | null;
  category: string;
  surface: string;
  name: string;
  payload_preview: string;
  decision: string;
  risk_score: number;
  classification: string;
  findings: DetectorFinding[];
  explanation: string;
  expected: string;
  matched_expectation: boolean | null;
  thresholds: Record<string, number>;
}

export interface PlaygroundSuite {
  results: PlaygroundResult[];
  total: number;
  matched_expectation: number;
  attack_scenarios: number;
  control_scenarios: number;
}
