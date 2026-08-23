"""LLM provider interface."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def was_truncated(self) -> bool:
        """Whether the model stopped because it hit the token ceiling.

        Worth surfacing: a truncated answer is usually invalid JSON, and
        retrying blindly on a parse failure would burn a second call for a
        cause that retrying cannot fix.
        """
        return self.finish_reason in {"length", "max_tokens"}


class LLMProvider(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Produce a completion for a system/user prompt pair.

        The two prompts are kept as separate arguments all the way down to the
        provider so that the system instructions occupy the API's own system
        role.  Concatenating them into one string would erase the only
        structural boundary the model has between policy and payload.
        """


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from model output.

    Models wrap JSON in prose or code fences even when told not to.  Rather
    than failing the request over formatting, three strategies are tried in
    order of confidence.  If none succeed the caller treats it as a schema
    violation, which is a *security* outcome (fail closed), not a parse retry.
    """
    if not text:
        return None

    candidate = text.strip()

    # 1. The whole response is JSON.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. JSON inside a fenced code block.
    fence = _FENCE.search(candidate)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. The outermost balanced {...} span, tracking string state so that a
    #    brace inside a quoted value does not end the object early.
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(candidate[start : index + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
