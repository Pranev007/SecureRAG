import { useCallback, useEffect, useRef, useState } from 'react';

import {
  EmptyState,
  ErrorBanner,
  Panel,
  Spinner,
  cx,
  formatBytes,
  formatRelative,
} from '../components/ui';
import { describeError } from '../hooks/useAuth';
import { api } from '../services/api';
import type { DocumentDetail, DocumentSummary } from '../types/api';

const ACCEPTED = '.pdf,.txt,.md,.markdown,.docx';

function StatusBadge({ status }: { status: DocumentSummary['status'] }) {
  const map = {
    ready: 'badge-safe',
    processing: 'badge-warn',
    pending: 'badge-neutral',
    failed: 'badge-danger',
  } as const;
  return <span className={map[status]}>{status}</span>;
}

function ChunkInspector({ document }: { document: DocumentDetail }) {
  return (
    <div className="border-t border-ink-800/80 bg-ink-950/40">
      <div className="max-h-96 space-y-2 overflow-y-auto p-4">
        {document.chunks.map((chunk) => (
          <div
            key={chunk.id}
            className={cx(
              'rounded-lg border p-3',
              chunk.is_quarantined
                ? 'border-danger/40 bg-danger/6'
                : 'border-ink-800 bg-ink-900/50',
            )}
          >
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className="badge-neutral">#{chunk.chunk_index}</span>
              {chunk.page_number !== null && (
                <span className="badge-neutral">page {chunk.page_number}</span>
              )}
              {chunk.section && (
                <span className="text-[11px] text-ink-400">{chunk.section}</span>
              )}
              <span className="ml-auto font-mono text-[10px] text-ink-500">
                {chunk.token_count} tok
              </span>
              {chunk.is_quarantined && (
                <span className="badge-danger">
                  quarantined · risk {chunk.injection_risk.toFixed(2)}
                </span>
              )}
            </div>

            <p className="whitespace-pre-wrap text-xs leading-relaxed text-ink-300">
              {chunk.content}
            </p>

            {chunk.injection_labels.length > 0 && (
              <p className="mt-2 flex flex-wrap gap-1.5 border-t border-ink-800/60 pt-2">
                {chunk.injection_labels.map((label) => (
                  <span key={label} className="badge-warn font-mono">
                    {label}
                  </span>
                ))}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [expanded, setExpanded] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ message: string; warnings: string[] } | null>(
    null,
  );
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setDocuments((await api.listDocuments()).items);
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    setNotice(null);
    try {
      for (const file of Array.from(files)) {
        const result = await api.uploadDocument(file);
        setNotice({ message: result.message, warnings: result.warnings });
      }
      await load();
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  }

  async function toggle(id: string) {
    if (expanded?.id === id) {
      setExpanded(null);
      return;
    }
    try {
      setExpanded(await api.getDocument(id, true));
    } catch (caught) {
      setError(describeError(caught));
    }
  }

  async function remove(id: string) {
    try {
      await api.deleteDocument(id);
      if (expanded?.id === id) setExpanded(null);
      await load();
    } catch (caught) {
      setError(describeError(caught));
    }
  }

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(e.dataTransfer.files);
        }}
        className={cx(
          'panel flex flex-col items-center justify-center border-dashed px-6 py-10 transition-colors',
          dragging ? 'border-accent bg-accent/6' : 'border-ink-700',
        )}
      >
        <input
          ref={fileInput}
          type="file"
          accept={ACCEPTED}
          multiple
          className="hidden"
          onChange={(e) => void upload(e.target.files)}
        />
        <p className="text-sm font-medium text-ink-100">
          Drop documents here, or{' '}
          <button
            onClick={() => fileInput.current?.click()}
            className="text-accent-glow underline underline-offset-2"
          >
            browse
          </button>
        </p>
        <p className="mt-1.5 text-xs text-ink-400">
          PDF, DOCX, TXT or Markdown · up to 20 MB
        </p>
        <p className="mt-3 max-w-md text-center text-[11px] leading-relaxed text-ink-500">
          Uploads are parsed, cleaned, chunked and scanned for embedded
          instructions before indexing. Anything that reads as an instruction
          aimed at an AI is quarantined and excluded from retrieval.
        </p>
        {uploading && (
          <p className="mt-3 flex items-center gap-2 text-xs text-accent-glow">
            <Spinner className="h-3.5 w-3.5" /> Ingesting…
          </p>
        )}
      </div>

      <ErrorBanner message={error} />

      {notice && (
        <div className="animate-fade-in rounded-lg border border-safe/35 bg-safe/8 px-4 py-3">
          <p className="text-sm text-safe">{notice.message}</p>
          {notice.warnings.map((warning) => (
            <p key={warning} className="mt-1.5 text-xs leading-relaxed text-warn">
              ⚠ {warning}
            </p>
          ))}
        </div>
      )}

      <Panel title="Your documents" subtitle={`${documents.length} uploaded`}>
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner className="h-5 w-5 text-accent" />
          </div>
        ) : documents.length === 0 ? (
          <EmptyState
            title="No documents yet"
            description="Upload a policy, handbook or report to build your private searchable corpus."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-ink-800/80">
                <tr>
                  <th className="table-head">Document</th>
                  <th className="table-head">Status</th>
                  <th className="table-head">Chunks</th>
                  <th className="table-head">Security</th>
                  <th className="table-head">Uploaded</th>
                  <th className="table-head" />
                </tr>
              </thead>
              <tbody>
                {documents.map((document) => (
                  <>
                    <tr
                      key={document.id}
                      className="border-b border-ink-800/50 transition-colors hover:bg-ink-800/25"
                    >
                      <td className="table-cell">
                        <button
                          onClick={() => void toggle(document.id)}
                          className="text-left font-medium text-ink-100 hover:text-accent-glow"
                        >
                          {document.filename}
                        </button>
                        <p className="text-[11px] text-ink-500">
                          {formatBytes(document.file_size_bytes)}
                          {document.page_count > 1 && ` · ${document.page_count} pages`}
                        </p>
                      </td>
                      <td className="table-cell">
                        <StatusBadge status={document.status} />
                      </td>
                      <td className="table-cell font-mono text-xs tabular-nums text-ink-300">
                        {document.chunk_count}
                      </td>
                      <td className="table-cell">
                        {document.quarantined_chunk_count > 0 ? (
                          <span
                            className="badge-danger"
                            title={`Highest injection risk: ${document.max_injection_risk.toFixed(2)}`}
                          >
                            {document.quarantined_chunk_count} quarantined
                          </span>
                        ) : (
                          <span className="badge-safe">clean</span>
                        )}
                      </td>
                      <td className="table-cell text-xs text-ink-400">
                        {formatRelative(document.created_at)}
                      </td>
                      <td className="table-cell text-right">
                        <button
                          onClick={() => void remove(document.id)}
                          className="btn-danger btn-sm"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                    {expanded?.id === document.id && (
                      <tr key={`${document.id}-detail`}>
                        <td colSpan={6} className="p-0">
                          <ChunkInspector document={expanded} />
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
