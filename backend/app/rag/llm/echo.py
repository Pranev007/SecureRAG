"""Deterministic offline LLM stand-in.

WHAT THIS IS
------------
An *extractive* responder.  It parses the fenced data blocks out of the prompt,
scores each sentence against the question by lexical overlap, and returns the
best-supported sentences as the answer, with citations pointing at the blocks
they came from.  It emits exactly the JSON contract the real prompt asks for.

WHY IT EXISTS
-------------
Three concrete reasons, in order of importance:

1. The test suite must be deterministic.  Assertions like "a blocked query
   never reaches generation" or "citations must reference supplied blocks" are
   statements about *the system*, and testing them against a sampling language
   model would make the suite flaky for reasons unrelated to the code.
2. The repository must be runnable with no API key, so a reviewer can clone it
   and see the pipeline work end to end.
3. The evaluation suite can then report real measurements offline.

WHAT IT IS NOT
--------------
It is not a language model and it is not a security control.  It cannot
paraphrase, reason, or synthesise across chunks, and its answers are stilted.
Crucially, the fact that an extractive responder does not follow injected
instructions proves nothing about a real model -- so the indirect-injection
tests assert on *server-side* behaviour (was the chunk quarantined? was it kept
out of the prompt? did the output guardrail reject an ungrounded answer?) and
never on "the stub behaved itself".  This distinction is spelled out in
docs/evaluation.md.

The application refuses to start in production with ``LLM_PROVIDER=echo``.
"""

from __future__ import annotations

import json
import re
import time

from app.rag.ingestion.chunker import split_sentences
from app.rag.llm.base import LLMProvider, LLMResponse
from app.rag.retrieval.keyword import tokenize

_BLOCK = re.compile(
    r"\[(\d+)\][^\n]*\n--- BEGIN DATA (\w+) ---\n(.*?)\n--- END DATA \2 ---",
    re.DOTALL,
)
_QUESTION = re.compile(
    r"--- BEGIN QUESTION (\w+) ---\n(.*?)\n--- END QUESTION \1 ---", re.DOTALL
)

# A sentence must cover a *majority* of the question's content words before this
# responder will offer it as an answer. The principle: an extractive responder
# has no judgement, so lexical coverage is the only evidence it has that a
# sentence is responsive -- and a minority overlap is not evidence, it is a
# coincidence. Below the bar it reports insufficient evidence, which is the
# honest answer for a component that cannot read.
_MIN_OVERLAP = 0.5


class EchoLLMProvider(LLMProvider):
    name = "echo"

    @property
    def model(self) -> str:
        return "echo-extractive-v1"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()

        if "security classifier" in system_prompt.lower():
            payload = self._classify(user_prompt)
        else:
            payload = self._answer(user_prompt)

        text = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # ------------------------------------------------------------------
    def _answer(self, user_prompt: str) -> dict:
        question_match = _QUESTION.search(user_prompt)
        question = question_match.group(2).strip() if question_match else ""
        question_terms = set(tokenize(question))

        blocks = [
            (int(index), body.strip())
            for index, _nonce, body in _BLOCK.findall(user_prompt)
        ]

        if not blocks or not question_terms:
            return {
                "answer": "The provided documents do not cover this question.",
                "citations": [],
                "confidence": 0.0,
                "sufficient_evidence": False,
                "observed_injection_attempt": False,
            }

        scored: list[tuple[float, int, str]] = []
        for index, body in blocks:
            for sentence in split_sentences(body):
                sentence_terms = set(tokenize(sentence))
                if not sentence_terms:
                    continue
                overlap = len(question_terms & sentence_terms) / len(question_terms)
                if overlap >= _MIN_OVERLAP:
                    scored.append((overlap, index, sentence))

        if not scored:
            return {
                "answer": "The provided documents do not cover this question.",
                "citations": [],
                "confidence": 0.0,
                "sufficient_evidence": False,
                "observed_injection_attempt": False,
            }

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:3]

        used_indices: list[int] = []
        fragments: list[str] = []
        for _score, index, sentence in selected:
            if index not in used_indices:
                used_indices.append(index)
            fragments.append(f"{sentence} [{index}]")

        citations = [
            {
                "index": index,
                "quote": next(s for _o, i, s in selected if i == index)[:200],
            }
            for index in used_indices
        ]

        return {
            "answer": " ".join(fragments),
            "citations": citations,
            "confidence": round(min(0.95, max(s for s, _i, _t in selected)), 2),
            "sufficient_evidence": True,
            "observed_injection_attempt": False,
        }

    # ------------------------------------------------------------------
    def _classify(self, user_prompt: str) -> dict:
        """Stand-in classifier: defers to the deterministic detectors.

        It deliberately returns low confidence so that, offline, layer 3
        contributes nothing and the reported detection rate reflects the
        pattern and heuristic layers only -- which is the honest measurement.
        """
        return {"is_injection": False, "confidence": 0.0, "category": "not_evaluated"}
