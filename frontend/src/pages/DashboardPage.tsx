import { useEffect, useState } from 'react';

import {
  BarChart,
  EmptyState,
  ErrorBanner,
  Panel,
  SeverityBadge,
  Spinner,
  StatCard,
  cx,
  formatRelative,
  humanise,
} from '../components/ui';
import { describeError, useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import type { SecurityEvent, SecurityStats, TimeseriesPoint } from '../types/api';

const SEVERITIES = ['critical', 'high', 'medium', 'low', 'info'];

export function DashboardPage() {
  const { user } = useAuth();
  const [stats, setStats] = useState<SecurityStats | null>(null);
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [series, setSeries] = useState<TimeseriesPoint[]>([]);
  const [severityFilter, setSeverityFilter] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.stats(),
      api.events({ limit: 100, severity: severityFilter }),
      api.timeseries(14),
    ])
      .then(([statsResult, eventPage, timeseries]) => {
        if (cancelled) return;
        setStats(statsResult);
        setEvents(eventPage.items);
        setSeries(timeseries);
      })
      .catch((caught) => !cancelled && setError(describeError(caught)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [severityFilter]);

  if (loading && !stats) {
    return (
      <div className="flex justify-center py-20">
        <Spinner className="h-6 w-6 text-accent" />
      </div>
    );
  }

  const blockRatePercent = stats ? (stats.queries.block_rate * 100).toFixed(1) : '0.0';

  return (
    <div className="space-y-5">
      <div className="flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold">Security dashboard</h1>
          <p className="mt-0.5 text-xs text-ink-400">
            {stats?.scope === 'system'
              ? 'System-wide view (administrator)'
              : 'Your activity'}
            {' · last '}
            {stats?.window_days} days
          </p>
        </div>
        {user?.role === 'admin' && <span className="badge-accent">admin</span>}
      </div>

      <ErrorBanner message={error} />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Queries"
          value={stats?.queries.total ?? 0}
          hint="Questions asked"
        />
        <StatCard
          label="Blocked"
          value={stats?.queries.blocked ?? 0}
          tone={stats?.queries.blocked ? 'danger' : 'neutral'}
          hint={`${blockRatePercent}% of all queries`}
        />
        <StatCard
          label="Injection attempts"
          value={stats?.security.prompt_injection_attempts ?? 0}
          tone={stats?.security.prompt_injection_attempts ? 'danger' : 'neutral'}
          hint="Detected at the input layer"
        />
        <StatCard
          label="Indirect injections"
          value={stats?.security.indirect_injection_detections ?? 0}
          tone={stats?.security.indirect_injection_detections ? 'warn' : 'neutral'}
          hint="Found inside uploaded documents"
        />
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Grounding failures"
          value={stats?.security.grounding_failures ?? 0}
          tone={stats?.security.grounding_failures ? 'warn' : 'neutral'}
          hint="Answers withheld as unsupported"
        />
        <StatCard
          label="PII detections"
          value={stats?.security.pii_detections ?? 0}
          tone={stats?.security.pii_detections ? 'warn' : 'neutral'}
          hint="Redacted before display"
        />
        <StatCard
          label="Authorisation denials"
          value={stats?.security.authorization_denials ?? 0}
          tone={stats?.security.authorization_denials ? 'danger' : 'neutral'}
          hint="Cross-user access refused"
        />
        <StatCard
          label="Rate limited"
          value={stats?.security.rate_limit_violations ?? 0}
          hint="Requests refused for volume"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Panel
          title="Event volume"
          subtitle="Daily totals over 14 days"
          className="lg:col-span-2"
        >
          <BarChart points={series} />
        </Panel>

        <Panel title="Corpus & performance">
          <dl className="divide-y divide-ink-800/60">
            {[
              ['Documents', stats?.documents.total ?? 0],
              ['Indexed chunks', stats?.documents.chunks ?? 0],
              ['Quarantined chunks', stats?.documents.quarantined_chunks ?? 0],
              [
                'Avg latency',
                `${(stats?.performance.average_latency_ms ?? 0).toFixed(0)} ms`,
              ],
              [
                'Avg grounding',
                (stats?.performance.average_grounding_score ?? 0).toFixed(3),
              ],
              [
                'Avg chunks retrieved',
                (stats?.performance.average_retrieved_chunks ?? 0).toFixed(1),
              ],
            ].map(([label, value]) => (
              <div key={String(label)} className="flex items-center justify-between px-5 py-2.5">
                <dt className="text-xs text-ink-400">{label}</dt>
                <dd className="font-mono text-sm tabular-nums text-ink-100">{value}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      </div>

      <Panel
        title="Security events"
        subtitle="Decisions and metadata only — never query or document content"
        actions={
          <div className="flex gap-1.5">
            {SEVERITIES.map((severity) => {
              const active = severityFilter.includes(severity);
              return (
                <button
                  key={severity}
                  onClick={() =>
                    setSeverityFilter((current) =>
                      active
                        ? current.filter((s) => s !== severity)
                        : [...current, severity],
                    )
                  }
                  className={cx(
                    'rounded-md px-2 py-1 text-[11px] font-medium transition-colors',
                    active
                      ? 'bg-accent/20 text-accent-glow'
                      : 'text-ink-500 hover:text-ink-300',
                  )}
                >
                  {severity}
                </button>
              );
            })}
          </div>
        }
      >
        {events.length === 0 ? (
          <EmptyState
            title="No security events"
            description="Events appear here as you upload documents and ask questions. Try the Playground to generate some."
          />
        ) : (
          <div className="max-h-[28rem] overflow-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-ink-900/95 backdrop-blur">
                <tr className="border-b border-ink-800">
                  <th className="table-head">Time</th>
                  <th className="table-head">Event</th>
                  <th className="table-head">Layer</th>
                  <th className="table-head">Severity</th>
                  <th className="table-head">Action</th>
                  <th className="table-head">Risk</th>
                  <th className="table-head">Detector</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => (
                  <tr
                    key={event.id}
                    className="border-b border-ink-800/40 transition-colors hover:bg-ink-800/25"
                  >
                    <td className="table-cell whitespace-nowrap text-xs text-ink-400">
                      {formatRelative(event.created_at)}
                    </td>
                    <td className="table-cell text-xs font-medium text-ink-100">
                      {humanise(event.event_type)}
                    </td>
                    <td className="table-cell">
                      <span className="badge-neutral">{event.layer}</span>
                    </td>
                    <td className="table-cell">
                      <SeverityBadge severity={event.severity} />
                    </td>
                    <td className="table-cell text-xs text-ink-300">{event.action}</td>
                    <td className="table-cell font-mono text-xs tabular-nums text-ink-300">
                      {event.risk_score > 0 ? event.risk_score.toFixed(2) : '—'}
                    </td>
                    <td
                      className="table-cell max-w-[16rem] truncate font-mono text-[11px] text-ink-500"
                      title={event.detector ?? ''}
                    >
                      {event.detector ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
