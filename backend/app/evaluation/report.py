"""Report rendering: JSON for machines, Markdown for humans."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.runner import EvaluationReport


def write_json(report: EvaluationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _setting(value: object) -> str:
    """Render one configuration value for the Markdown table.

    Nested values (the NLI status block) are flattened to `k=v` pairs rather
    than dumped as a Python repr, which renders as an unreadable wall of quotes
    and braces in a table cell.
    """
    if isinstance(value, dict):
        parts = [f"{k}={v}" for k, v in value.items() if v is not None]
        return " · ".join(f"`{p}`" for p in parts) if parts else "—"
    return f"`{value}`"


def render_markdown(report: EvaluationReport) -> str:
    data = report.as_dict()
    config = data["configuration"]
    security = data["security"]
    ingestion = data["ingestion"]
    quality = data["quality"]
    relevance = data["relevance"]
    retrieval = data["retrieval"]
    latency = data["latency"]
    totals = data["totals"]

    lines: list[str] = []
    add = lines.append

    add("# SecureRAG Evaluation Report")
    add("")
    add(f"Generated: `{data['finished_at']}`  ")
    add(f"Duration: {data['duration_seconds']}s  ")
    add(f"Cases: {totals['cases']}")
    add("")

    # The configuration banner is not decoration. A detection rate produced
    # with the offline stubs is a different claim from one produced against a
    # real model, and the report must never be quotable without that context.
    add("## Configuration under test")
    add("")
    add("| Setting | Value |")
    add("|---|---|")
    for key, value in config.items():
        add(f"| `{key}` | {_setting(value)} |")
    add("")
    if config["llm_provider"] == "echo" or config["embedding_provider"] == "hashing":
        add(
            "> **These numbers were produced with offline stand-ins.** "
            "`echo` is a deterministic extractive responder, not a language "
            "model, and `hashing` is a lexical embedder that cannot match "
            "paraphrases. Security-layer metrics (detection, false positives, "
            "quarantine) are unaffected by this, because those controls are "
            "deterministic server-side code. Retrieval and answer-quality "
            "metrics **are** affected and should be read as a floor, not as a "
            "measure of what the system does with a real model."
        )
        add("")

    nli = config.get("nli") or {}
    if nli.get("requested") in {"nli", "hybrid"} and not nli.get("active"):
        add(
            "> **NLI grounding was requested but did not run** "
            f"(`{nli.get('unavailable_reason')}`), so every grounding number "
            "below was produced by the lexical verifier. The `method` field on "
            "each case records what actually scored it."
        )
        add("")

    add("## Headline results")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(
        f"| Overall case pass rate | **{_pct(totals['pass_rate'])}** ({totals['passed']}/{totals['cases']}) |"
    )
    add(
        f"| Attack detection rate | **{_pct(security['detection_rate'])}** ({security['true_positives']}/{security['attack_cases']}) |"
    )
    add(
        f"| False positive rate | **{_pct(security['false_positive_rate'])}** ({security['false_positives']}/{security['benign_cases']}) |"
    )
    add(f"| False negative rate | {_pct(security['false_negative_rate'])} |")
    add(
        f"| Benign refusal rate | {_pct(security['benign_refusal_rate'])} "
        f"({security['benign_refusals']}/{security['benign_cases']}) |"
    )
    add(
        f"| Indirect injection detection | {_pct(ingestion['indirect_detection_rate'])} ({ingestion['poisoned_chunks_quarantined']}/{ingestion['poisoned_chunks_present']} chunks) |"
    )
    add(f"| Answer faithfulness (mean grounding) | {quality['faithfulness']:.3f} |")
    add(
        f"| Answer relevance | {relevance['answer_relevance']:.3f} "
        f"({relevance['scored_answers']} answered, "
        f"{relevance['refusals_excluded']} refusals excluded) |"
    )
    add(f"| Citation accuracy | {_pct(quality['citation_accuracy'])} |")
    add(f"| Retrieval precision@{retrieval['k']} | {retrieval['precision_at_k']:.3f} |")
    add(f"| Retrieval recall@{retrieval['k']} | {retrieval['recall_at_k']:.3f} |")
    add(f"| Mean end-to-end latency | {latency['mean_ms']:.1f} ms |")
    add("")

    add("## Answer relevance")
    add("")
    add(
        "Faithfulness asks whether an answer is *supported* by the sources; "
        "relevance asks whether it *addresses the question*. An answer can "
        "score 1.0 on the first and 0.0 on the second, so both are reported. "
        "Refusals are excluded rather than scored zero -- a refusal is often "
        "the correct behaviour, and counting it as irrelevant would make this "
        "metric move whenever the guardrails did."
    )
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Mean answer relevance | **{relevance['answer_relevance']:.3f}** |")
    add(f"| Answers scored | {relevance['scored_answers']} |")
    add(f"| Refusals excluded | {relevance['refusals_excluded']} |")
    add(f"| Answers below threshold | {relevance['below_threshold']} |")
    for component, value in relevance["components_mean"].items():
        add(f"| &nbsp;&nbsp;component: {component} | {value:.3f} |")
    add("")
    if relevance.get("caveat"):
        add(f"> {relevance['caveat']}")
        add("")

    add("## Security classification")
    add("")
    add(
        "Positive = the system took a protective action. Detection and false "
        "positives are reported together; neither means anything alone. The "
        "**benign refusal rate** sits beside them because a refusal is not a "
        "block -- the case still passes and the confusion matrix stays clean, "
        "so a guardrail that buys faithfulness by answering less would "
        "otherwise look free."
    )
    add("")
    add("| | Predicted attack | Predicted benign |")
    add("|---|---|---|")
    add(
        f"| **Actual attack** | {security['true_positives']} (TP) | {security['false_negatives']} (FN) |"
    )
    add(
        f"| **Actual benign** | {security['false_positives']} (FP) | {security['true_negatives']} (TN) |"
    )
    add("")
    add(
        f"Precision {security['precision']:.3f} · Recall {security['detection_rate']:.3f} · F1 {security['f1']:.3f}"
    )
    add("")

    add("## Ingest-time indirect injection")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Documents ingested | {ingestion['documents']} |")
    add(f"| Chunks produced | {ingestion['chunks']} |")
    add(f"| Poisoned chunks present | {ingestion['poisoned_chunks_present']} |")
    add(f"| Poisoned chunks quarantined | {ingestion['poisoned_chunks_quarantined']} |")
    add(
        f"| Clean chunks wrongly quarantined | {ingestion['clean_chunks_wrongly_quarantined']} |"
    )
    add(f"| Quarantine precision | {_pct(ingestion['quarantine_precision'])} |")
    add("")

    add("## Results by case type")
    add("")
    add("| Case type | Passed | Total | Rate |")
    add("|---|---|---|---|")
    for kind, counts in sorted(totals["by_kind"].items()):
        rate = counts["passed"] / counts["total"] if counts["total"] else 0.0
        add(f"| {kind} | {counts['passed']} | {counts['total']} | {_pct(rate)} |")
    add("")

    add("## Latency breakdown")
    add("")
    add(
        f"mean {latency['mean_ms']:.1f} ms · median {latency['median_ms']:.1f} ms · p95 {latency['p95_ms']:.1f} ms · max {latency['max_ms']:.1f} ms"
    )
    add("")
    add("| Stage | Mean (ms) |")
    add("|---|---|")
    for stage, value in latency["by_stage_mean_ms"].items():
        add(f"| `{stage}` | {value:.2f} |")
    add("")

    failures = [case for case in data["cases"] if not case["passed"]]
    add("## Failing cases")
    add("")
    if not failures:
        add("None: every case in the dataset behaved as expected.")
    else:
        add(
            f"{len(failures)} of {totals['cases']} cases did not meet their "
            "expectation. These are recorded, not hidden -- a suite that always "
            "passes is usually measuring the wrong thing."
        )
        add("")
        add("| Case | Type | Expected | Actual | Why it failed |")
        add("|---|---|---|---|---|")
        for case in failures:
            add(
                f"| `{case['case_id']}` | {case['kind']} | {case['expected_behaviour']} "
                f"| {case['actual_behaviour']} | {case['failure_detail']} |"
            )
    add("")

    add("## All cases")
    add("")
    add("| Case | Type | Expected | Actual | Pass | Risk | Grounding | Sources | ms |")
    add("|---|---|---|---|---|---|---|---|---|")
    for case in data["cases"]:
        decision = case["security_decision"]
        add(
            f"| `{case['case_id']}` | {case['kind']} | {case['expected_behaviour']} "
            f"| {case['actual_behaviour']} | {'yes' if case['passed'] else '**no**'} "
            f"| {decision['risk_score']:.2f} | {case['grounding_score']:.2f} "
            f"| {case['citations']['emitted']} | {case['latency_ms']:.0f} |"
        )
    add("")

    add("---")
    add("")
    add(
        "Regenerate with `python -m app.evaluation.run` from `backend/`. "
        "See `docs/evaluation.md` for the methodology and the limitations of "
        "each metric."
    )
    add("")

    return "\n".join(lines)


def write_markdown(report: EvaluationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def print_summary(report: EvaluationReport) -> None:
    """Terminal summary. Deliberately compact and honest."""
    data = report.as_dict()
    security = data["security"]
    totals = data["totals"]
    ingestion = data["ingestion"]
    quality = data["quality"]
    relevance = data["relevance"]
    retrieval = data["retrieval"]
    latency = data["latency"]
    config = data["configuration"]

    def line(label: str, value: str) -> None:
        print(f"  {label:<34} {value}")

    print()
    print("=" * 66)
    print("  SecureRAG evaluation")
    print("=" * 66)
    line("LLM provider", config["llm_provider"])
    line("Embedding provider", config["embedding_provider"])
    line("Retrieval mode", f"{config['retrieval_mode']} (top_k={config['top_k']})")
    print("-" * 66)
    line(
        "Cases passed",
        f"{totals['passed']}/{totals['cases']}  ({_pct(totals['pass_rate'])})",
    )
    line(
        "Attack detection rate",
        f"{_pct(security['detection_rate'])}  "
        f"({security['true_positives']}/{security['attack_cases']})",
    )
    line(
        "False positive rate",
        f"{_pct(security['false_positive_rate'])}  "
        f"({security['false_positives']}/{security['benign_cases']})",
    )
    line("False negative rate", _pct(security["false_negative_rate"]))
    line(
        "Benign refusal rate",
        f"{_pct(security['benign_refusal_rate'])}  "
        f"({security['benign_refusals']}/{security['benign_cases']})",
    )
    line(
        "Indirect injection detection",
        f"{_pct(ingestion['indirect_detection_rate'])}  "
        f"({ingestion['poisoned_chunks_quarantined']}/"
        f"{ingestion['poisoned_chunks_present']} chunks)",
    )
    line("Quarantine precision", _pct(ingestion["quarantine_precision"]))
    print("-" * 66)
    line(
        "Faithfulness (mean grounding)",
        f"{quality['faithfulness']:.3f}  [{config.get('grounding_method', 'lexical')}]",
    )
    line(
        "Answer relevance",
        f"{relevance['answer_relevance']:.3f}  "
        f"({relevance['scored_answers']} answered, "
        f"{relevance['refusals_excluded']} refused)",
    )
    line("Citation accuracy", _pct(quality["citation_accuracy"]))
    line("Answer correctness", _pct(quality["answer_correctness"]))
    line(
        f"Retrieval P@{retrieval['k']} / R@{retrieval['k']}",
        f"{retrieval['precision_at_k']:.3f} / {retrieval['recall_at_k']:.3f}",
    )
    line(
        "Latency mean / p95",
        f"{latency['mean_ms']:.0f} ms / {latency['p95_ms']:.0f} ms",
    )
    print("=" * 66)

    failures = [c for c in data["cases"] if not c["passed"]]
    if failures:
        print(f"\n  {len(failures)} failing case(s):")
        for case in failures:
            print(f"    - {case['case_id']} ({case['kind']}): {case['failure_detail']}")
    print()
