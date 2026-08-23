"""Grounding verification: is the answer actually supported by the context?

WHY THIS EXISTS
---------------
RAG reduces hallucination; it does not eliminate it.  A model handed five
relevant chunks will still, routinely:

* add a plausible detail the sources never state ("...and unused leave expires
  after 18 months");
* answer confidently when the retrieved context is only tangentially related;
* cite a real chunk beside a claim that chunk does not support.

Telling the model "only use the context" is a request, not a control.  So the
answer is verified *after* generation, against the exact text that was
supplied, by code that cannot be talked out of it.

HOW IT WORKS
------------
Claim-level, not answer-level.  The answer is split into sentences, and each
factual sentence is scored for lexical support against the context that was
actually sent to the model:

* **content-word overlap** -- what fraction of a claim's meaningful words
  appear in the context;
* **number and date agreement** -- weighted heavily and checked separately,
  because "24 days" versus "28 days" is the failure mode that matters most and
  it barely moves a word-overlap score;
* **n-gram containment** -- consecutive 3-word spans shared with the context,
  which distinguishes genuine paraphrase from coincidental vocabulary reuse.

Hedging and refusal sentences are exempt: "I could not find this in your
documents" is *correct* behaviour and must not be scored as an unsupported
claim.

HONEST LIMITATIONS
------------------
This is a lexical entailment proxy, not an NLI model.  It will:

* **miss** a fluent contradiction that reuses the source's vocabulary
  ("employees may *not* carry leave forward" scores well against a chunk
  saying they may);
* **penalise** a correct answer that paraphrases heavily in different words.

The negation check below covers the most common instance of the first failure.
A cross-encoder NLI model would be materially better and is the documented
upgrade path.  The score is a filter against *obvious* fabrication -- which is
the bulk of real hallucination -- not a proof of correctness, and the README
says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.rag.ingestion.chunker import split_sentences
from app.rag.retrieval.keyword import tokenize
from app.rag.retrieval.types import ScoredChunk

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
# Inline citation markers. These must be removed before scoring: "[1]" is
# citation apparatus, not a factual claim, and leaving it in makes the numeric
# check treat the citation index as an unsupported number -- penalising every
# correctly-cited answer for doing the right thing.
_CITATION_MARKER = re.compile(r"\[\d{1,3}\]")
_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can't|won't|shall\s+not|must\s+not|may\s+not|"
    r"is\s+not|are\s+not|does\s+not|do\s+not|without|excluded|prohibited|"
    r"forbidden|denied|ineligible)\b",
    re.IGNORECASE,
)

# Sentences that make no factual claim about the documents.
_HEDGE = re.compile(
    r"\b(?:i\s+(?:could\s+not|couldn'?t|cannot|can'?t|don'?t|do\s+not)\s+"
    r"(?:find|see|locate|determine|answer)|"
    r"(?:the\s+)?(?:provided\s+|supplied\s+|retrieved\s+)?documents?\s+"
    r"(?:do\s+not|don'?t|does\s+not|doesn'?t)\s+(?:contain|cover|mention|include|say)|"
    r"no\s+(?:sufficient\s+)?(?:evidence|information)|"
    r"not\s+(?:enough|sufficient)\s+(?:evidence|information|context)|"
    r"insufficient\s+(?:evidence|information)|"
    r"unable\s+to\s+(?:find|answer|determine))\b",
    re.IGNORECASE,
)

# Weights for the three signals. Numeric agreement dominates because numeric
# fabrication is both the most common and the most damaging failure mode.
W_OVERLAP = 0.4
W_NUMERIC = 0.35
W_NGRAM = 0.25

# A claim below this is treated as unsupported when computing the answer score.
CLAIM_SUPPORT_FLOOR = 0.45


@dataclass
class ClaimScore:
    sentence: str
    score: float
    overlap: float
    numeric: float
    ngram: float
    contradicts: bool = False
    is_hedge: bool = False

    @property
    def supported(self) -> bool:
        return not self.contradicts and self.score >= CLAIM_SUPPORT_FLOOR


@dataclass
class GroundingReport:
    score: float
    claims: list[ClaimScore] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradicted_claims: list[str] = field(default_factory=list)
    context_chars: int = 0
    method: str = "lexical_v1"
    notes: list[str] = field(default_factory=list)

    @property
    def factual_claim_count(self) -> int:
        return sum(1 for c in self.claims if not c.is_hedge)

    def as_detail(self) -> dict:
        """Audit-safe summary. Counts and scores only -- never the claim text."""
        return {
            "score": round(self.score, 4),
            "method": self.method,
            "claims": self.factual_claim_count,
            "unsupported": len(self.unsupported_claims),
            "contradicted": len(self.contradicted_claims),
            "notes": self.notes[:5],
        }


def _ngrams(tokens: list[str], size: int = 3) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _numeric_agreement(claim: str, context: str) -> tuple[float, bool]:
    """Fraction of the claim's numbers that appear in the context."""
    claim_numbers = {n.replace(",", "") for n in _NUMBER.findall(claim)}
    if not claim_numbers:
        # No numbers to get wrong: neutral, and flagged as such.
        return 1.0, False
    context_numbers = {n.replace(",", "") for n in _NUMBER.findall(context)}
    matched = sum(1 for n in claim_numbers if n in context_numbers)
    ratio = matched / len(claim_numbers)
    return ratio, matched < len(claim_numbers)


def score_claim(sentence: str, context: str, context_tokens: set[str]) -> ClaimScore:
    if _HEDGE.search(sentence):
        # An explicit non-answer is fully "grounded": it asserts nothing.
        return ClaimScore(sentence, 1.0, 1.0, 1.0, 1.0, is_hedge=True)

    claim_tokens = tokenize(sentence)
    if not claim_tokens:
        return ClaimScore(sentence, 1.0, 1.0, 1.0, 1.0, is_hedge=True)

    overlap = sum(1 for t in claim_tokens if t in context_tokens) / len(claim_tokens)

    numeric, numeric_mismatch = _numeric_agreement(sentence, context)

    claim_ngrams = _ngrams(claim_tokens)
    # n-grams must be built from the context's token *sequence*: the token set
    # loses ordering, and ordering is the whole point of an n-gram.
    context_ngrams = _ngrams(tokenize(context))
    ngram = (
        len(claim_ngrams & context_ngrams) / len(claim_ngrams) if claim_ngrams else 0.0
    )

    score = W_OVERLAP * overlap + W_NUMERIC * numeric + W_NGRAM * ngram

    # Polarity check: high vocabulary overlap with the opposite polarity is the
    # signature of a fluent contradiction, which pure overlap scoring rewards.
    contradicts = False
    if overlap > 0.6:
        claim_negated = bool(_NEGATION.search(sentence))
        # Compare against the best-matching context sentence, not the whole
        # context, which almost always contains a negation somewhere.
        best_sentence = _best_matching_sentence(claim_tokens, context)
        context_negated = (
            bool(_NEGATION.search(best_sentence)) if best_sentence else False
        )
        if claim_negated != context_negated:
            contradicts = True
            score *= 0.4

    if numeric_mismatch:
        score = min(score, 0.5)

    return ClaimScore(
        sentence=sentence,
        score=round(min(max(score, 0.0), 1.0), 4),
        overlap=round(overlap, 4),
        numeric=round(numeric, 4),
        ngram=round(ngram, 4),
        contradicts=contradicts,
    )


def _best_matching_sentence(claim_tokens: list[str], context: str) -> str:
    claim_set = set(claim_tokens)
    best = ""
    best_score = 0.0
    for sentence in split_sentences(context):
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        score = len(claim_set & tokens) / len(claim_set)
        if score > best_score:
            best_score, best = score, sentence
    return best


def verify_grounding(answer: str, chunks: list[ScoredChunk]) -> GroundingReport:
    """Score how well ``answer`` is supported by the supplied ``chunks``."""
    context = "\n".join(chunk.content for chunk in chunks)

    if not context.strip():
        # No evidence was supplied, so nothing can be grounded in it. An answer
        # produced from an empty context is by definition ungrounded.
        return GroundingReport(
            score=0.0,
            context_chars=0,
            notes=["no context supplied"],
        )

    prose = _CITATION_MARKER.sub("", answer)
    sentences = split_sentences(prose)
    if not sentences:
        return GroundingReport(
            score=0.0, context_chars=len(context), notes=["empty answer"]
        )

    context_tokens = set(tokenize(context))
    claims = [score_claim(s, context, context_tokens) for s in sentences]

    factual = [c for c in claims if not c.is_hedge]
    if not factual:
        # Pure hedge / refusal: correctly grounded by construction.
        return GroundingReport(
            score=1.0,
            claims=claims,
            context_chars=len(context),
            notes=["answer asserts no factual claims"],
        )

    # Mean over factual claims, then penalised by the share that failed.
    # A single fabricated sentence in an otherwise accurate answer is exactly
    # the case that must not be averaged away.
    mean_score = sum(c.score for c in factual) / len(factual)
    unsupported = [c for c in factual if not c.supported]
    penalty = 1.0 - 0.5 * (len(unsupported) / len(factual))
    score = mean_score * penalty

    notes: list[str] = []
    if any(c.contradicts for c in factual):
        notes.append("possible contradiction of the source")
    if len(factual) == 1 and factual[0].score < CLAIM_SUPPORT_FLOOR:
        notes.append("single unsupported claim")

    return GroundingReport(
        score=round(min(max(score, 0.0), 1.0), 4),
        claims=claims,
        unsupported_claims=[c.sentence for c in unsupported],
        contradicted_claims=[c.sentence for c in factual if c.contradicts],
        context_chars=len(context),
        notes=notes,
    )
