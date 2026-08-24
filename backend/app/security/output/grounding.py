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

HONEST LIMITATIONS OF THE LEXICAL SIGNAL
----------------------------------------
Lexical scoring is an entailment *proxy*.  On its own it will:

* **miss** a fluent contradiction that reuses the source's vocabulary
  ("employees may *not* carry leave forward" scores well against a chunk
  saying they may);
* **penalise** a correct answer that paraphrases heavily in different words.

The negation check below covers the most common instance of the first failure.

NLI
---
Setting ``GROUNDING_METHOD`` to ``nli`` or ``hybrid`` adds a cross-encoder
entailment model (see ``nli.py``) that addresses both of those directly.  It is
opt-in because it pulls in torch, and it does **not** replace the lexical
signal: the cross-encoder is good on numbers but not dependable, and a
confidently-entailed wrong figure is the most damaging hallucination in a
document-QA system.  ``combine_signals`` below therefore keeps the lexical
numeric check authoritative over numbers and lets NLI decide paraphrase and
polarity.  Neither signal is trusted outside the region where it has been shown
to work.

The score is a filter against fabrication, not a proof of correctness, in every
mode.  The README says so, and ``method`` in the report always records which
combination actually ran -- including a silent degradation to lexical when the
NLI dependency is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.ingestion.chunker import split_sentences
from app.rag.retrieval.keyword import tokenize
from app.rag.retrieval.types import ScoredChunk
from app.security.output.nli import EntailmentResult, get_nli_verifier

logger = get_logger("app.security.output.grounding")

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
# Inline citation markers. These must be removed before scoring: "[1]" is
# citation apparatus, not a factual claim, and leaving it in makes the numeric
# check treat the citation index as an unsupported number -- penalising every
# correctly-cited answer for doing the right thing.
_CITATION_MARKER = re.compile(r"\[\d{1,3}\]")
# Where one clause ends and the next begins, for the polarity check. Only
# coordinating boundaries: a subordinate clause ("leave that does not carry
# forward") qualifies the same assertion and must stay attached to it.
_CLAUSE_BOUNDARY = re.compile(
    r"(?:\s*;\s*)|(?:,?\s+(?:and|but|however|whereas|although|though|while)\s+)",
    re.IGNORECASE,
)
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

# A claim citing a number its context does not contain must never count as
# supported, whatever the other signals say.
#
# Derived from the floor rather than set independently, because when the two
# *were* independent (ceiling 0.50, floor 0.45) the "gate" did not gate: a
# confident NLI entailment of a fabricated figure was capped at 0.50, which is
# still above the floor, so the claim passed. Lexical scoring alone had given
# the same claim 0.42 and blocked it -- adding the better model made that case
# strictly worse. Measured, not theorised: see
# tests/security/test_nli_numeric_behaviour.py, which fails if this inequality
# is ever inverted again.
NUMERIC_MISMATCH_CEILING = CLAIM_SUPPORT_FLOOR - 0.05
# Multiplier applied when a claim is judged to contradict its source.
CONTRADICTION_PENALTY = 0.4


@dataclass
class ClaimScore:
    sentence: str
    score: float
    overlap: float
    numeric: float
    ngram: float
    contradicts: bool = False
    is_hedge: bool = False
    # Populated only when an NLI verifier ran for this claim.
    entailment: float | None = None
    nli_contradiction: float | None = None
    lexical_score: float | None = None

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
    if overlap > 0.6 and not _polarity_is_corroborated(sentence, context):
        contradicts = True
        score *= CONTRADICTION_PENALTY

    if numeric_mismatch:
        score = min(score, NUMERIC_MISMATCH_CEILING)

    final = round(min(max(score, 0.0), 1.0), 4)
    return ClaimScore(
        sentence=sentence,
        score=final,
        overlap=round(overlap, 4),
        numeric=round(numeric, 4),
        ngram=round(ngram, 4),
        contradicts=contradicts,
        lexical_score=final,
    )


def combine_signals(claim: ClaimScore, entailment: EntailmentResult) -> ClaimScore:
    """Fold an NLI verdict into a lexically-scored claim.

    The combination is not a weighted average, because the two signals are not
    interchangeable estimates of the same quantity -- they are reliable on
    disjoint things.  Each is therefore allowed to decide only what it is good
    at:

    * **Numbers stay lexical.**  A claim asserting a figure absent from the
      context is capped below the support floor, whatever the model says.  The
      cross-encoder is mostly right about numbers -- 12 of 13 fabricated
      figures caught in ``tests/security/test_nli_numeric_behaviour.py`` -- but
      the one it missed, it entailed at p=0.94, and a confidently-served wrong
      figure is the worst hallucination class in document QA.  The check is
      free on the twelve it already catches, so it costs almost nothing to keep
      and covers the case where the model is confidently wrong.
    * **Support may come from either signal** in ``hybrid``.  Lexical overlap
      proves the answer reuses the source; entailment proves it follows from
      the source.  Either is sufficient evidence of grounding, so taking the
      maximum raises recall on heavily-paraphrased answers without weakening
      the numeric gate.
    * **Contradiction is a veto from either side.**  The lexical polarity check
      and the model's contradiction probability catch different phrasings, and
      a claim flagged by either is penalised.
    """
    if claim.is_hedge:
        return claim

    contradicted = claim.contradicts or (
        entailment.contradiction >= settings.NLI_CONTRADICTION_THRESHOLD
    )

    lexical = claim.lexical_score if claim.lexical_score is not None else claim.score
    if settings.GROUNDING_METHOD == "nli":
        score = entailment.entailment
    else:  # hybrid
        score = max(entailment.entailment, lexical)

    # The lexical numeric gate outranks the model in both modes.
    if claim.numeric < 1.0:
        score = min(score, NUMERIC_MISMATCH_CEILING)

    if contradicted:
        score *= CONTRADICTION_PENALTY

    claim.score = round(min(max(score, 0.0), 1.0), 4)
    claim.contradicts = contradicted
    claim.entailment = round(entailment.entailment, 4)
    claim.nli_contradiction = round(entailment.contradiction, 4)
    return claim


def _clauses(context: str) -> list[str]:
    """Split the context into clauses for polarity comparison.

    Polarity is a property of a clause, not of the sentence containing it: a
    sentence can assert one thing and deny another in the same breath.
    """
    parts: list[str] = []
    for sentence in split_sentences(context):
        parts.extend(p for p in _CLAUSE_BOUNDARY.split(sentence) if p and p.strip())
    return parts


# A clause counts as comparable evidence if its overlap is within this fraction
# of the best clause's. Both bounds matter: the ratio keeps a weakly-related
# clause from vetoing a strong match, and the floor keeps an incidental
# one-word overlap from counting as corroboration at all.
_COMPARABLE_CLAUSE_RATIO = 0.75
_COMPARABLE_CLAUSE_FLOOR = 0.30


def _polarity_is_corroborated(claim: str, context: str) -> bool:
    """Is every assertion in ``claim`` matched in polarity by the source?

    Comparison is **clause to clause on both sides**, and each of the three
    cheaper designs failed on a real answer:

    * *whole claim vs whole sentence* (the original): a source reading "granted
      at 12 days per calendar year **and does not carry forward**" made the
      correct concise answer "12 days per calendar year" look negation-mismatched
      and blocked it. Only a real model hits this -- the offline stub copies
      whole sentences, negated clause included, so its polarity always matched.
    * *whole claim vs best clause*: a correct **negated** answer ("sick leave
      does not carry forward") lost a near-tie to the affirmative clause beside
      it and was flagged instead.
    * *whole claim vs comparable clauses*: an answer quoting **both** clauses of
      a mixed-polarity sentence reads as negated overall, while the clause that
      dominates the token overlap is affirmative -- so nothing corroborated it.

    All three are the same error: treating a span with mixed polarity as though
    it had one. Splitting both sides removes the mismatch at its source. A claim
    contradicts only when one of its own clauses asserts something no
    comparably-supporting context clause agrees with.
    """
    context_clauses = [
        (set(tokenize(clause)), bool(_NEGATION.search(clause)))
        for clause in _clauses(context)
    ]
    context_clauses = [entry for entry in context_clauses if entry[0]]
    if not context_clauses:
        # Nothing to compare against, so there is nothing to contradict. The
        # overlap and n-gram signals already scored this claim on its merits.
        return True

    for claim_clause in _clauses(claim):
        claim_set = set(tokenize(claim_clause))
        if not claim_set:
            continue
        claim_negated = bool(_NEGATION.search(claim_clause))

        scored = [
            (len(claim_set & tokens) / len(claim_set), negated)
            for tokens, negated in context_clauses
        ]
        best = max(score for score, _ in scored)
        threshold = max(best * _COMPARABLE_CLAUSE_RATIO, _COMPARABLE_CLAUSE_FLOOR)
        if not any(
            negated == claim_negated for score, negated in scored if score >= threshold
        ):
            return False

    return True


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
        # Pure hedge / refusal: correctly grounded by construction. Returned
        # before the NLI stage so a refusal never pays for a model call.
        return GroundingReport(
            score=1.0,
            claims=claims,
            method=_method_label(nli_ran=False),
            context_chars=len(context),
            notes=["answer asserts no factual claims"],
        )

    method_notes: list[str] = []
    nli_ran = False
    if settings.GROUNDING_METHOD != "lexical":
        verifier = get_nli_verifier()
        if verifier is None:
            # Degradation is recorded, not hidden: a report claiming NLI
            # grounding while running lexical scoring would misattribute every
            # number in it.
            method_notes.append("nli requested but unavailable; used lexical")
        else:
            try:
                verdicts = verifier.verify_claims([c.sentence for c in factual], context)
                for claim in factual:
                    verdict = verdicts.get(claim.sentence)
                    if verdict is not None:
                        combine_signals(claim, verdict)
                nli_ran = True
            except Exception as exc:
                # Inference failure must not fail the request open *or* closed
                # on a technicality -- the lexical score is still valid, so it
                # stands, and the substitution is disclosed.
                logger.warning("nli_scoring_failed", extra={"error": type(exc).__name__})
                method_notes.append("nli error; fell back to lexical")

    # Mean over factual claims, then penalised by the share that failed.
    # A single fabricated sentence in an otherwise accurate answer is exactly
    # the case that must not be averaged away.
    mean_score = sum(c.score for c in factual) / len(factual)
    unsupported = [c for c in factual if not c.supported]
    penalty = 1.0 - 0.5 * (len(unsupported) / len(factual))
    score = mean_score * penalty

    notes: list[str] = list(method_notes)
    if any(c.contradicts for c in factual):
        notes.append("possible contradiction of the source")
    if len(factual) == 1 and factual[0].score < CLAIM_SUPPORT_FLOOR:
        notes.append("single unsupported claim")

    return GroundingReport(
        score=round(min(max(score, 0.0), 1.0), 4),
        claims=claims,
        unsupported_claims=[c.sentence for c in unsupported],
        contradicted_claims=[c.sentence for c in factual if c.contradicts],
        method=_method_label(nli_ran=nli_ran),
        context_chars=len(context),
        notes=notes,
    )


def _method_label(*, nli_ran: bool) -> str:
    """What actually scored this answer, not what was configured.

    The distinction matters for the evaluation report: ``nli`` in the config
    and ``lexical_v1`` in the method field is exactly how a run gets attributed
    to a model that never loaded.
    """
    if not nli_ran:
        return "lexical_v1"
    return "nli_v1" if settings.GROUNDING_METHOD == "nli" else "hybrid_v1"
