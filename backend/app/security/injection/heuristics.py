"""Layer 2: structural heuristics.

Layer 1 asks "does this match a known attack phrase?".  Layer 2 asks a
different and more general question: **"does this text behave like
instructions rather than like a question?"**

That framing catches paraphrases no signature covers, because an injection has
to *command* the model to achieve anything, and commanding has measurable
structural properties: imperative verbs, second-person address, directive
punctuation, role labels, unusual length for a question.

Each signal returns a value in [0, 1].  They are combined by the detector, not
here, so that each stays independently testable and independently explainable.

Also implemented here is the counterweight that most write-ups omit:
:func:`benign_question_signal`.  Without it, a legitimate query like "What does
the security policy say about ignoring alerts?" scores as an attack, and a
guardrail with a high false-positive rate gets switched off in week two.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from app.rag.ingestion.cleaner import count_invisible_characters

# Verbs that, in the imperative, direct a system to do something.
_IMPERATIVE_VERBS = frozenset(
    """
    ignore disregard forget override bypass reveal show print output repeat
    disclose execute run perform act pretend behave respond answer stop start
    begin cease refuse comply obey follow become assume adopt enable disable
    activate deactivate unlock jailbreak bypass skip omit delete remove send
    transmit forward leak dump list enumerate translate encode decode set
    put place drop abandon discard treat consider regard rewrite replace
    append prepend insert continue proceed switch change modify
    """.split()
)

_ROLE_LABEL = re.compile(
    r"^\s*(?:system|user|assistant|human|ai|admin|developer|instruction|prompt)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")
_WORD = re.compile(r"[A-Za-z']+")
# A wh-word opener is unambiguous. An auxiliary-verb opener ("do", "can",
# "will") is not: "Do not tell the user..." opens exactly like "Do I get sick
# leave?". Auxiliaries therefore only count as a question when the text also
# ends in a question mark -- otherwise a negated imperative would earn a
# damping bonus, which is precisely the shape an injection wants.
_WH_OPENER = re.compile(
    r"^\s*(?:what|when|where|who|whom|whose|which|why|how|"
    r"summari[sz]e|explain|describe|tell\s+me\s+about|give\s+me\s+a\s+summary)\b",
    re.IGNORECASE,
)
_AUXILIARY_OPENER = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|should|would|will|may|might|"
    r"has|have|had|am)\b",
    re.IGNORECASE,
)
_DOCUMENT_REFERENCE = re.compile(
    r"\b(?:document|policy|handbook|report|file|contract|agreement|manual|guide|"
    r"section|clause|page|chapter|according\s+to|state[sd]?|says?|mention(?:s|ed)?)\b",
    re.IGNORECASE,
)
_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX_BLOB = re.compile(r"(?:\\x[0-9a-f]{2}){8,}|(?:0x)?[0-9a-f]{60,}", re.IGNORECASE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass(frozen=True)
class HeuristicSignal:
    name: str
    value: float
    detail: str = ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def imperative_density(text: str) -> HeuristicSignal:
    """Fraction of sentences that open with a directive verb.

    Questions about documents essentially never do this; instructions almost
    always do.
    """
    sentences = _sentences(text)
    if not sentences:
        return HeuristicSignal("imperative_density", 0.0)

    imperative = 0
    for sentence in sentences:
        words = _WORD.findall(sentence.lower())
        if not words:
            continue
        # "Please ignore ..." and "Now reveal ..." are still imperative.
        head = words[0]
        if head in {
            "please",
            "kindly",
            "now",
            "instead",
            "first",
            "then",
            "also",
            "just",
            "simply",
            "immediately",
            "always",
            "never",
        }:
            head = words[1] if len(words) > 1 else head
        if head in _IMPERATIVE_VERBS:
            imperative += 1

    ratio = imperative / len(sentences)
    return HeuristicSignal(
        "imperative_density",
        min(ratio * 1.4, 1.0),
        f"{imperative}/{len(sentences)} sentences are imperative",
    )


def second_person_directive(text: str) -> HeuristicSignal:
    """Density of "you must / you will / your instructions" style address."""
    lowered = text.lower()
    matches = len(
        re.findall(
            r"\byou\s+(?:are|were|was|will|must|should|shall|need\s+to|have\s+to|"
            r"had\s+to|can|may|might|do\s+not|don'?t|no\s+longer)\b"
            r"|\byour\s+(?:instructions?|rules?|prompt|system|purpose|task|role|"
            r"guidelines?|guidance|configuration|directives?|constraints?)\b",
            lowered,
        )
    )
    if matches == 0:
        return HeuristicSignal("second_person_directive", 0.0)
    # Saturating: one occurrence is weak evidence, four is not four times worse.
    return HeuristicSignal(
        "second_person_directive",
        min(1.0, 1.0 - math.exp(-0.6 * matches)),
        f"{matches} second-person directives",
    )


def role_label_presence(text: str) -> HeuristicSignal:
    """Line-leading ``system:`` / ``assistant:`` style labels."""
    matches = _ROLE_LABEL.findall(text)
    if not matches:
        return HeuristicSignal("role_labels", 0.0)
    return HeuristicSignal(
        "role_labels",
        min(1.0, 0.6 + 0.2 * (len(matches) - 1)),
        f"{len(matches)} role labels",
    )


def invisible_character_signal(text: str) -> HeuristicSignal:
    """Zero-width and bidi characters used to hide text from human review."""
    count = count_invisible_characters(text)
    if count == 0:
        return HeuristicSignal("invisible_characters", 0.0)
    return HeuristicSignal(
        "invisible_characters",
        min(1.0, 0.5 + count / 20.0),
        f"{count} invisible characters",
    )


def homoglyph_signal(text: str) -> HeuristicSignal:
    """Non-Latin letters mixed into otherwise Latin text.

    Substituting Cyrillic 'а' for Latin 'a' defeats naive keyword matching
    while reading identically to a human.  A genuinely non-English query
    scores low here because the ratio, not the presence, is what counts.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 12:
        return HeuristicSignal("homoglyphs", 0.0)

    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    foreign_ratio = 1.0 - (latin / len(letters))
    # Mostly-Latin text with a sprinkling of foreign letters is the suspicious
    # shape; wholly non-Latin text is just another language.
    if 0.02 <= foreign_ratio <= 0.35:
        return HeuristicSignal(
            "homoglyphs",
            min(1.0, foreign_ratio * 2.5),
            f"{foreign_ratio:.0%} non-Latin letters in Latin text",
        )
    return HeuristicSignal("homoglyphs", 0.0)


def encoded_payload_signal(text: str) -> HeuristicSignal:
    """Long base64/hex blobs, which carry no meaning for a document question."""
    base64_hits = [m for m in _BASE64_BLOB.findall(text) if len(m) >= 40]
    hex_hits = _HEX_BLOB.findall(text)
    if not base64_hits and not hex_hits:
        return HeuristicSignal("encoded_payload", 0.0)
    longest = max((len(m) for m in base64_hits + [str(h) for h in hex_hits]), default=0)
    return HeuristicSignal(
        "encoded_payload",
        min(1.0, 0.35 + longest / 400.0),
        f"encoded blob of {longest} characters",
    )


def shouting_signal(text: str) -> HeuristicSignal:
    """Sustained upper-case, the usual typography of a forged directive."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 25:
        return HeuristicSignal("shouting", 0.0)
    ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if ratio < 0.5:
        return HeuristicSignal("shouting", 0.0)
    return HeuristicSignal("shouting", min(1.0, (ratio - 0.5) * 2), f"{ratio:.0%} caps")


def structural_anomaly_signal(text: str) -> HeuristicSignal:
    """Length and shape unusual for a document question."""
    words = len(_WORD.findall(text))
    newlines = text.count("\n")
    score = 0.0
    reasons: list[str] = []

    if words > 150:
        score += min(0.4, (words - 150) / 500.0)
        reasons.append(f"{words} words")
    if newlines >= 6:
        score += min(0.3, (newlines - 5) / 20.0)
        reasons.append(f"{newlines} line breaks")
    if _URL.search(text):
        score += 0.2
        reasons.append("contains a URL")

    return HeuristicSignal("structural_anomaly", min(score, 1.0), ", ".join(reasons))


def benign_question_signal(text: str) -> HeuristicSignal:
    """Evidence that this is an ordinary document question.

    Used by the detector to *damp* the risk score.  A guardrail is only useful
    if people leave it switched on, and the fastest way to get it switched off
    is to block "What does the policy say about ignoring alerts?".
    """
    stripped = text.strip()
    if not stripped:
        return HeuristicSignal("benign_question", 0.0)

    score = 0.0
    reasons: list[str] = []
    is_interrogative = stripped.endswith("?")

    if _WH_OPENER.match(stripped):
        score += 0.5
        reasons.append("opens with a question word")
    elif is_interrogative and _AUXILIARY_OPENER.match(stripped):
        score += 0.5
        reasons.append("opens as a yes/no question")

    if is_interrogative:
        score += 0.2
        reasons.append("ends with a question mark")
    if _DOCUMENT_REFERENCE.search(stripped):
        score += 0.3
        reasons.append("refers to a document")

    words = len(_WORD.findall(stripped))
    if words <= 30:
        score += 0.2
        reasons.append("short")
    if "\n" not in stripped:
        score += 0.1
        reasons.append("single line")

    return HeuristicSignal("benign_question", min(score, 1.0), ", ".join(reasons))


ATTACK_SIGNALS = (
    imperative_density,
    second_person_directive,
    role_label_presence,
    invisible_character_signal,
    homoglyph_signal,
    encoded_payload_signal,
    shouting_signal,
    structural_anomaly_signal,
)


def evaluate_heuristics(text: str) -> list[HeuristicSignal]:
    """Run every attack signal and return those that fired."""
    return [
        signal for signal in (fn(text) for fn in ATTACK_SIGNALS) if signal.value > 0.0
    ]
