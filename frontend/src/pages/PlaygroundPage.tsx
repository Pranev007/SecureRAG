import { useEffect, useMemo, useState } from 'react';

import {
  DecisionBadge,
  ErrorBanner,
  Panel,
  ScoreMeter,
  Spinner,
  cx,
  humanise,
} from '../components/ui';
import { describeError } from '../hooks/useAuth';
import { api } from '../services/api';
import type { AttackScenario, PlaygroundResult, PlaygroundSuite } from '../types/api';

const SURFACE_LABEL: Record<string, string> = {
  user_input: 'User input',
  document: 'Document content',
  model_output: 'Model output',
};

const CATEGORY_ORDER = [
  'direct_injection',
  'instruction_override',
  'prompt_extraction',
  'jailbreak',
  'indirect_injection',
  'data_exfiltration',
  'pii_leakage',
  'unauthorized_access',
  'benign_control',
];

function ResultView({ result }: { result: PlaygroundResult }) {
  const isControl = result.category === 'benign_control';

  return (
    <div className="animate-slide-up space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <DecisionBadge decision={result.decision} />
        <span className="badge-neutral">{SURFACE_LABEL[result.surface] ?? result.surface}</span>
        <span className="badge-neutral">{humanise(result.classification)}</span>
        {result.matched_expectation !== null && (
          <span
            className={result.matched_expectation ? 'badge-safe' : 'badge-danger'}
            title={result.expected}
          >
            {result.matched_expectation ? 'matched expectation' : 'UNEXPECTED'}
          </span>
        )}
      </div>

      <div className="rounded-lg border border-ink-800 bg-ink-950/50 p-3">
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-ink-500">
          Payload
        </p>
        <p className="break-words font-mono text-xs leading-relaxed text-ink-300">
          {result.payload_preview}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <ScoreMeter
          label={isControl ? 'Risk (should be low)' : 'Risk score'}
          value={result.risk_score}
        />
        <div className="rounded-lg border border-ink-800 bg-ink-950/40 px-3 py-2">
          <p className="text-[11px] font-medium uppercase tracking-wide text-ink-400">
            Thresholds in force
          </p>
          <p className="mt-1 font-mono text-[11px] text-ink-300">
            {Object.entries(result.thresholds)
              .map(([name, value]) => `${name.replace(/_/g, ' ')} ${value}`)
              .join(' · ')}
          </p>
        </div>
      </div>

      <div>
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-ink-400">
          Detectors that fired
        </p>
        {result.findings.length === 0 ? (
          <p className="rounded-lg border border-ink-800 bg-ink-950/40 px-3 py-2.5 text-xs text-ink-400">
            None. Nothing in this payload matched a signature or a structural signal.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {result.findings.map((finding, index) => (
              <li
                key={`${finding.detector}-${index}`}
                className="flex items-start gap-3 rounded-lg border border-ink-800 bg-ink-950/40 px-3 py-2"
              >
                <span className="badge-neutral shrink-0">{finding.layer}</span>
                <div className="min-w-0 flex-1">
                  <p className="font-mono text-xs text-ink-200">{finding.detector}</p>
                  {finding.detail && (
                    <p className="mt-0.5 text-[11px] leading-relaxed text-ink-400">
                      {finding.detail}
                    </p>
                  )}
                </div>
                <span className="shrink-0 font-mono text-xs tabular-nums text-ink-300">
                  {finding.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="rounded-lg border border-accent/25 bg-accent/6 p-3.5">
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-accent-glow">
          Why this decision
        </p>
        <p className="text-xs leading-relaxed text-ink-200">{result.explanation}</p>
        {result.expected && (
          <p className="mt-2 border-t border-accent/20 pt-2 text-[11px] leading-relaxed text-ink-400">
            <span className="font-medium text-ink-300">Expected: </span>
            {result.expected}
          </p>
        )}
      </div>
    </div>
  );
}

export function PlaygroundPage() {
  const [scenarios, setScenarios] = useState<AttackScenario[]>([]);
  const [selected, setSelected] = useState<AttackScenario | null>(null);
  const [result, setResult] = useState<PlaygroundResult | null>(null);
  const [suite, setSuite] = useState<PlaygroundSuite | null>(null);
  const [custom, setCustom] = useState('');
  const [customSurface, setCustomSurface] = useState('user_input');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .scenarios()
      .then((list) => {
        setScenarios(list);
        setSelected(list[0] ?? null);
      })
      .catch((caught) => setError(describeError(caught)));
  }, []);

  const grouped = useMemo(() => {
    const byCategory = new Map<string, AttackScenario[]>();
    for (const scenario of scenarios) {
      const list = byCategory.get(scenario.category) ?? [];
      list.push(scenario);
      byCategory.set(scenario.category, list);
    }
    return CATEGORY_ORDER.filter((c) => byCategory.has(c)).map((category) => ({
      category,
      items: byCategory.get(category)!,
    }));
  }, [scenarios]);

  async function run(scenario: AttackScenario) {
    setSelected(scenario);
    setBusy(true);
    setError(null);
    setSuite(null);
    try {
      setResult(await api.runScenario({ scenarioId: scenario.id }));
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setBusy(false);
    }
  }

  async function runCustom() {
    if (!custom.trim()) return;
    setBusy(true);
    setError(null);
    setSuite(null);
    setSelected(null);
    try {
      setResult(await api.runScenario({ payload: custom, surface: customSurface }));
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setBusy(false);
    }
  }

  async function runAll() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setSuite(await api.runAllScenarios());
    } catch (caught) {
      setError(describeError(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-2xl">
          <h1 className="text-lg font-semibold">Security playground</h1>
          <p className="mt-1 text-xs leading-relaxed text-ink-400">
            Run real attacks against the live guardrails. Each payload is scored by
            the same detector objects the request path uses — nothing here is
            scripted. Attacks are <strong className="text-ink-300">analysed, not
            executed</strong>: no retrieval runs and no model is called.
          </p>
        </div>
        <button onClick={() => void runAll()} disabled={busy} className="btn-primary">
          {busy ? <Spinner /> : null} Run all scenarios
        </button>
      </div>

      <ErrorBanner message={error} />

      {suite && (
        <Panel
          title="Suite results"
          subtitle={`${suite.attack_scenarios} attack scenarios · ${suite.control_scenarios} benign controls`}
        >
          <div className="border-b border-ink-800/70 px-5 py-3.5">
            <p className="text-sm">
              <span
                className={cx(
                  'font-mono text-lg font-semibold',
                  suite.matched_expectation === suite.total ? 'text-safe' : 'text-warn',
                )}
              >
                {suite.matched_expectation}/{suite.total}
              </span>{' '}
              <span className="text-ink-400">
                scenarios behaved as documented
              </span>
            </p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-500">
              Controls must be allowed and attacks must be stopped. A mismatch here
              is a genuine detector regression, not a display quirk.
            </p>
          </div>
          <div className="max-h-[30rem] overflow-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-ink-900/95 backdrop-blur">
                <tr className="border-b border-ink-800">
                  <th className="table-head">Scenario</th>
                  <th className="table-head">Category</th>
                  <th className="table-head">Surface</th>
                  <th className="table-head">Decision</th>
                  <th className="table-head">Risk</th>
                  <th className="table-head">As expected</th>
                </tr>
              </thead>
              <tbody>
                {suite.results.map((item) => (
                  <tr
                    key={item.scenario_id ?? item.name}
                    className="cursor-pointer border-b border-ink-800/40 hover:bg-ink-800/25"
                    onClick={() => {
                      setResult(item);
                      setSuite(null);
                    }}
                  >
                    <td className="table-cell text-xs font-medium text-ink-100">
                      {item.name}
                    </td>
                    <td className="table-cell text-xs text-ink-400">
                      {humanise(item.category)}
                    </td>
                    <td className="table-cell text-xs text-ink-400">
                      {SURFACE_LABEL[item.surface] ?? item.surface}
                    </td>
                    <td className="table-cell">
                      <DecisionBadge decision={item.decision} />
                    </td>
                    <td className="table-cell font-mono text-xs tabular-nums text-ink-300">
                      {item.risk_score.toFixed(2)}
                    </td>
                    <td className="table-cell">
                      <span
                        className={item.matched_expectation ? 'badge-safe' : 'badge-danger'}
                      >
                        {item.matched_expectation ? 'yes' : 'no'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
        <div className="space-y-4">
          <Panel title="Attack catalogue">
            <div className="max-h-[26rem] space-y-3 overflow-y-auto p-3">
              {grouped.map(({ category, items }) => (
                <div key={category}>
                  <p className="mb-1.5 px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-500">
                    {humanise(category)}
                  </p>
                  <ul className="space-y-1">
                    {items.map((scenario) => (
                      <li key={scenario.id}>
                        <button
                          onClick={() => void run(scenario)}
                          className={cx(
                            'w-full rounded-md px-2.5 py-2 text-left transition-colors',
                            selected?.id === scenario.id
                              ? 'bg-ink-800 text-ink-100'
                              : 'text-ink-300 hover:bg-ink-800/60',
                          )}
                        >
                          <span className="block text-xs font-medium">
                            {scenario.name}
                          </span>
                          <span className="mt-0.5 block text-[10px] leading-snug text-ink-500">
                            {scenario.description}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Custom payload" subtitle="Try your own attack">
            <div className="space-y-2.5 p-3">
              <textarea
                value={custom}
                onChange={(e) => setCustom(e.target.value)}
                rows={4}
                maxLength={8000}
                placeholder="Type a payload to analyse…"
                className="input resize-none font-mono text-xs"
              />
              <select
                value={customSurface}
                onChange={(e) => setCustomSurface(e.target.value)}
                className="input"
              >
                <option value="user_input">Surface: user input</option>
                <option value="document">Surface: document content</option>
                <option value="model_output">Surface: model output</option>
              </select>
              <button
                onClick={() => void runCustom()}
                disabled={busy || !custom.trim()}
                className="btn-ghost w-full"
              >
                Analyse
              </button>
            </div>
          </Panel>
        </div>

        <Panel
          title={result ? result.name : 'Result'}
          subtitle={
            result ? SURFACE_LABEL[result.surface] : 'Select a scenario to run it'
          }
        >
          <div className="p-5">
            {busy && !result ? (
              <div className="flex justify-center py-12">
                <Spinner className="h-5 w-5 text-accent" />
              </div>
            ) : result ? (
              <ResultView result={result} />
            ) : (
              <p className="py-12 text-center text-sm text-ink-500">
                Pick an attack from the catalogue, or write your own.
              </p>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
