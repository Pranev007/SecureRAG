"""Answer relevance: did the system answer *the question that was asked*?

WHY THIS IS A SEPARATE METRIC
-----------------------------
The suite already measures three things that sound like this one and are not:

* **Faithfulness / grounding** -- is the answer supported by the retrieved
  context?  An answer can be perfectly grounded and completely beside the
  point: quoting the leave policy verbatim in response to a question about
  password rotation is 1.0 faithful and 0.0 useful.
* **Answer correctness** -- does the answer contain the expected substring?
  High precision, but it only exists for cases where a short canonical answer
  could be written down, and it says nothing about the surrounding text.
* **Retrieval precision@k** -- did the right *document* come back?  Retrieval
  can be perfect and generation still wander.

Relevance is the missing axis, and it is the one that degrades first when a
small local model is swapped in: weaker models drift, pad, restate the
question, or answer a nearby question instead.  Without this metric a
provider swap that made answers materially worse could leave every other
number in the report unchanged.

WHY NOT LLM-AS-JUDGE
--------------------
The usual implementation (RAGAS-style) asks an LLM to reverse-engineer
questions from the answer and compares those to the original.  Rejected here
for two reasons:

1. **Circularity.**  The obvious judge is the model under test.  A model that
   misreads a question is likely to misread it the same way twice and score
   its own off-target answer highly.
2. **Reproducibility.**  A judged number cannot be recomputed without the same
   provider, temperature and weights.  This project's evaluation is meant to
   run offline with no credentials and produce the same numbers twice.

So relevance is computed from three deterministic signals instead.  This is
weaker than a good LLM judge on nuance and stronger on reproducibility, and
that trade is the point rather than an accident.

THE SIGNALS
-----------
* **Semantic similarity** between question and answer, using the *configured
  embedding provider* -- the same one retrieval uses, so it needs no extra
  dependency and is meaningful exactly to the degree the deployment's
  embeddings are.
* **Question-term coverage**: the share of the question's content words the
  answer engages with.  A directly-responsive answer nearly always names the
  thing it was asked about.
* **Answer-type match**: "how many" wants a number, "when" wants a time,
  "who" wants an agent.  Cheap, and it catches the specific failure where a
  model produces fluent on-topic prose that never actually answers.

CAVEAT THAT MUST TRAVEL WITH THE NUMBER
---------------------------------------
Under ``EMBEDDING_PROVIDER=hashing`` the "semantic" term is lexical, because
the hashing embedder is a bag-of-words feature hash.  Relevance then measures
vocabulary agreement, not meaning, and a correct answer phrased in entirely
different words scores low.  :func:`relevance_caveat` returns the disclosure
and the report prints it, so the number is never read as more than it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
from app.rag.embeddings.base import cosine_similarity
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.retrieval.keyword import tokenize

# Weights. Semantic similarity leads, but it is deliberately not dominant:
# under the offline hashing embedder it degenerates to lexical overlap, and a
# metric that collapsed to a single degraded signal would be worse than one
# built from three partly-independent weak ones.
W_SEMANTIC = 0.45
W_COVERAGE = 0.30
W_TYPE_MATCH = 0.25

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_CITATION_MARKER = re.compile(r"\[\d{1,3}\]")

_TIME = re.compile(
    r"\b(?:\d{1,2}:\d{2}|\d{4}|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|day|days|week|weeks|month|months|"
    r"year|years|hour|hours|minute|minutes|quarterly|annually|immediately)\b",
    re.IGNORECASE,
)
_AGENT = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|team|department|manager|officer|"
    r"employee|employees|administrator|admin|staff|chief|executive|security|"
    r"anyone|everyone|they|he|she)\b"
)
_PLACE_OR_THING = re.compile(r"\b(?:in|at|on|via|through|using|to|from)\b", re.I)

# (name, question trigger, what a responsive answer must contain).
#
# The name is written out rather than derived from the pattern: it is reported
# per case in the JSON, and a label like `how\s+(?:many|much` helps nobody
# reading the report. Ordered — "how many" must be tested before a bare "how".
_ANSWER_TYPES: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    ("quantity", re.compile(r"^\s*how\s+(?:many|much|long|often)\b", re.I), _NUMBER),
    ("time", re.compile(r"^\s*when\b|\bby\s+when\b", re.I), _TIME),
    ("agent", re.compile(r"^\s*who\b|\bwho\s+(?:is|are|must|should)\b", re.I), _AGENT),
    ("place", re.compile(r"^\s*where\b", re.I), _PLACE_OR_THING),
)


@dataclass
class RelevanceScore:
    score: float
    semantic: float
    coverage: float
    type_match: float
    answer_type: str = "none"
    is_refusal: bool = False

    @property
    def relevant(self) -> bool:
        return self.score >= settings.ANSWER_RELEVANCE_MIN_SCORE

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "semantic": round(self.semantic, 4),
            "coverage": round(self.coverage, 4),
            "type_match": round(self.type_match, 4),
            "answer_type": self.answer_type,
        }


def _expected_answer_type(question: str) -> tuple[str, re.Pattern[str] | None]:
    for name, trigger, expected in _ANSWER_TYPES:
        if trigger.search(question):
            return name, expected
    return "none", None


def _semantic_similarity(question: str, answer: str) -> float:
    """Cosine between question and answer under the configured embedder.

    Clamped to [0, 1]: cosine is defined on [-1, 1] and a negative value is
    not "less relevant than nothing", it is noise at this scale.
    """
    provider = get_embedding_provider()
    vectors = provider.embed_documents([question, answer])
    if len(vectors) != 2:
        return 0.0
    return max(0.0, min(1.0, cosine_similarity(vectors[0], vectors[1])))


def score_relevance(
    question: str, answer: str, *, refused: bool = False
) -> RelevanceScore:
    """Score how directly ``answer`` responds to ``question``.

    A refusal is reported with ``is_refusal=True`` and a zero score, and the
    aggregate excludes it. Scoring "I could not find that in your documents"
    as irrelevant would penalise the system for the behaviour the rest of the
    suite demands of it, and mixing refusals into the mean would make
    relevance move whenever the *guardrails* changed.
    """
    text = _CITATION_MARKER.sub("", answer or "").strip()
    if refused or not text:
        return RelevanceScore(0.0, 0.0, 0.0, 0.0, is_refusal=True)

    semantic = _semantic_similarity(question, text)

    question_terms = set(tokenize(question))
    answer_terms = set(tokenize(text))
    coverage = (
        len(question_terms & answer_terms) / len(question_terms)
        if question_terms
        else 0.0
    )

    # No inferable type means this signal has nothing to say, so it takes the
    # neutral value rather than zero: an open question ("what is the policy on
    # X") must not be scored down for failing a test never applicable to it.
    type_name, expected = _expected_answer_type(question)
    type_match = 1.0 if expected is None or expected.search(text) else 0.0

    score = W_SEMANTIC * semantic + W_COVERAGE * coverage + W_TYPE_MATCH * type_match

    return RelevanceScore(
        score=round(min(max(score, 0.0), 1.0), 4),
        semantic=semantic,
        coverage=coverage,
        type_match=type_match,
        answer_type=type_name,
    )


def relevance_caveat() -> str | None:
    """The disclosure that must accompany a relevance number, if any."""
    if settings.EMBEDDING_PROVIDER == "hashing":
        return (
            "Answer relevance was computed with the offline hashing embedder, "
            "so its semantic term measures vocabulary agreement rather than "
            "meaning. Treat it as a lower bound: correct answers phrased in "
            "different words score low."
        )
    return None
