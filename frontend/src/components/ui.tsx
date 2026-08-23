/** Shared presentational primitives. */

import type { ReactNode } from 'react';

export function cx(...values: (string | false | null | undefined)[]): string {
  return values.filter(Boolean).join(' ');
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx('panel', className)}>
      {(title || actions) && (
        <header className="panel-header">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink-100">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-400">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      {icon && <div className="mb-3 text-ink-600">{icon}</div>}
      <p className="text-sm font-medium text-ink-200">{title}</p>
      {description && (
        <p className="mt-1.5 max-w-sm text-xs leading-relaxed text-ink-400">
          {description}
        </p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cx('animate-spin', className ?? 'h-4 w-4')}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-20"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M12 2a10 10 0 0 1 10 10h-3a7 7 0 0 0-7-7V2z"
      />
    </svg>
  );
}

export function ErrorBanner({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className="animate-fade-in rounded-lg border border-danger/40 bg-danger/10 px-3.5 py-2.5 text-sm text-danger"
    >
      {message}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Security presentation
// ---------------------------------------------------------------------------

export type Severity = 'info' | 'low' | 'medium' | 'high' | 'critical';

const SEVERITY_CLASS: Record<Severity, string> = {
  info: 'badge-neutral',
  low: 'badge-neutral',
  medium: 'badge-warn',
  high: 'badge-danger',
  critical: 'badge-critical',
};

export function SeverityBadge({ severity }: { severity: string }) {
  const key = (severity as Severity) in SEVERITY_CLASS ? (severity as Severity) : 'info';
  return <span className={SEVERITY_CLASS[key]}>{severity}</span>;
}

const DECISION_CLASS: Record<string, string> = {
  BLOCKED: 'badge-danger',
  WITHHELD: 'badge-danger',
  QUARANTINED: 'badge-danger',
  FLAGGED: 'badge-warn',
  NEUTRALISED: 'badge-warn',
  REDACTED: 'badge-warn',
  ALLOWED: 'badge-safe',
};

export function DecisionBadge({ decision }: { decision: string }) {
  return <span className={DECISION_CLASS[decision] ?? 'badge-neutral'}>{decision}</span>;
}

/**
 * A labelled 0-1 meter.
 *
 * `invert` flips the colour scale for metrics where low is bad (grounding,
 * confidence) rather than where high is bad (risk). Getting this backwards
 * would show a green bar for a dangerous request, so it is an explicit prop
 * rather than something inferred from the label.
 */
export function ScoreMeter({
  label,
  value,
  invert = false,
  hint,
}: {
  label: string;
  value: number;
  invert?: boolean;
  hint?: string;
}) {
  const clamped = Math.max(0, Math.min(1, value));
  const severity = invert ? 1 - clamped : clamped;
  const colour =
    severity >= 0.75
      ? 'bg-danger'
      : severity >= 0.45
        ? 'bg-warn'
        : 'bg-safe';

  return (
    <div title={hint}>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">
          {label}
        </span>
        <span className="font-mono text-xs tabular-nums text-ink-200">
          {clamped.toFixed(2)}
        </span>
      </div>
      <div className="meter">
        <div
          className={cx('meter-fill', colour)}
          style={{ width: `${clamped * 100}%` }}
        />
      </div>
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: 'neutral' | 'safe' | 'warn' | 'danger' | 'accent';
}) {
  const toneClass = {
    neutral: 'text-ink-100',
    safe: 'text-safe',
    warn: 'text-warn',
    danger: 'text-danger',
    accent: 'text-accent-glow',
  }[tone];

  return (
    <div className="panel p-4">
      <p className="stat-label">{label}</p>
      <p className={cx('stat-value mt-1.5', toneClass)}>{value}</p>
      {hint && <p className="mt-1 text-[11px] leading-snug text-ink-500">{hint}</p>}
    </div>
  );
}

/** Compact sparkline-style bar chart for the events timeseries. */
export function BarChart({
  points,
  height = 88,
}: {
  points: { date: string; total: number; attacks: number }[];
  height?: number;
}) {
  if (!points.length) {
    return (
      <p className="px-5 py-8 text-center text-xs text-ink-500">
        No activity recorded yet.
      </p>
    );
  }
  const max = Math.max(...points.map((p) => p.total), 1);

  return (
    <div className="px-5 py-4">
      {/* `max-w` matters more than it looks: on a fresh install there is one
          day of data, and an unconstrained flex-1 bar fills the whole panel as
          a single solid block that reads as a broken chart rather than as one
          data point. Capping the width keeps a sparse series legible. */}
      <div className="flex items-end gap-1" style={{ height }}>
        {points.map((point) => {
          const totalHeight = (point.total / max) * 100;
          const attackHeight = (point.attacks / max) * 100;
          return (
            <div
              key={point.date}
              className="group relative max-w-[2.5rem] flex-1"
              style={{ height: '100%' }}
              title={`${point.date}: ${point.total} events, ${point.attacks} attacks`}
            >
              <div className="absolute bottom-0 w-full rounded-sm bg-ink-700 transition-colors group-hover:bg-ink-600"
                   style={{ height: `${totalHeight}%` }} />
              <div className="absolute bottom-0 w-full rounded-sm bg-danger/80"
                   style={{ height: `${attackHeight}%` }} />
            </div>
          );
        })}
      </div>
      <div className="mt-3 flex items-center justify-between text-[11px] text-ink-500">
        <span>{points[0]?.date}</span>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <i className="h-2 w-2 rounded-sm bg-ink-700" /> all events
          </span>
          <span className="flex items-center gap-1.5">
            <i className="h-2 w-2 rounded-sm bg-danger/80" /> attacks
          </span>
        </span>
        <span>{points[points.length - 1]?.date}</span>
      </div>
    </div>
  );
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatRelative(iso: string): string {
  const then = new Date(iso).getTime();
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(iso).toLocaleDateString();
}

export function humanise(value: string): string {
  return value.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}
