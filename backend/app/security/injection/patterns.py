"""Layer 1: rule-based prompt-injection signatures.

WHY REGEX IS NOT ENOUGH -- AND WHY IT IS STILL HERE
---------------------------------------------------
Pattern matching cannot solve prompt injection.  The attack surface is natural
language, so for any finite rule set there are unlimited paraphrases that mean
the same thing and match nothing ("kindly set aside the guidance you were given
earlier").  Anyone claiming regex solves this is wrong.

It earns its place as the *first* layer for three reasons:

* **It is nearly free.**  Microseconds, deterministic, no network call.  The
  overwhelming majority of real attempts are copy-pasted from a handful of
  public jailbreak lists and match a known signature verbatim.
* **It is explainable.**  When a request is blocked, an operator can see which
  named rule fired -- which matters when tuning a false-positive rate.
* **It is auditable.**  The rules are data, versioned in this file, and each
  one is covered by a test.

Every rule carries a weight rather than being a hard block, because the
detector combines evidence across layers instead of trusting any single hit.
Weights are per-signal probabilities, combined with a noisy-OR in
:mod:`app.security.injection.detector`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class InjectionCategory(StrEnum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    PROMPT_EXTRACTION = "prompt_extraction"
    ROLE_HIJACK = "role_hijack"
    JAILBREAK = "jailbreak"
    AUTHORITY_SPOOF = "authority_spoof"
    DELIMITER_INJECTION = "delimiter_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    SCOPE_VIOLATION = "scope_violation"
    ENCODING_EVASION = "encoding_evasion"
    OUTPUT_MANIPULATION = "output_manipulation"


@dataclass(frozen=True)
class InjectionPattern:
    name: str
    category: InjectionCategory
    regex: re.Pattern[str]
    weight: float
    description: str


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Instruction override
# ---------------------------------------------------------------------------

_OVERRIDE_VERB = (
    r"(?:ignore|disregard|forget|discard|override|bypass|skip|omit|dismiss|"
    r"abandon|set\s+aside)"
)
_PRIOR = (
    r"(?:previous|prior|above|preceding|earlier|initial|original|foregoing|"
    r"former|old|existing|all)"
)
_DIRECTIVE = (
    r"(?:instruction|prompt|rule|directive|guideline|guidance|constraint|"
    r"restriction|limitation|command|order|policy|policies|context|"
    r"configuration|briefing|framing|advice|direction)"
)

PATTERNS: tuple[InjectionPattern, ...] = (
    InjectionPattern(
        name="ignore_previous_instructions",
        category=InjectionCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            rf"\b{_OVERRIDE_VERB}\b[^.!?\n]{{0,40}}?\b{_PRIOR}\b[^.!?\n]{{0,20}}?\b{_DIRECTIVE}s?\b"
        ),
        weight=0.85,
        description="Directs the model to abandon its prior instructions",
    ),
    InjectionPattern(
        name="ignore_everything",
        category=InjectionCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            rf"\b{_OVERRIDE_VERB}\s+(?:everything|all)\b[^.!?\n]{{0,30}}\b(?:before|above|said|told|earlier|so\s+far)\b"
        ),
        weight=0.8,
        description="Blanket instruction to discard prior context",
    ),
    InjectionPattern(
        name="new_instructions_follow",
        category=InjectionCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            r"\b(?:new|updated|revised|real|actual|true)\s+(?:instruction|directive|rule|task|prompt)s?\b\s*[:\-]|"
            r"\byour\s+(?:new|real|actual|true)\s+(?:task|instruction|purpose|goal|job)\b"
        ),
        weight=0.7,
        description="Announces a replacement instruction set",
    ),
    InjectionPattern(
        name="unrestricted_operation_request",
        category=InjectionCategory.JAILBREAK,
        regex=_compile(
            r"\b(?:answer|respond|reply|behave|operate|speak|write)\b[^.!?\n]{0,30}"
            r"\b(?:without|with\s+no|free\s+of|free\s+from)\s+"
            r"(?:any\s+)?(?:limits?|limitations?|restrictions?|filters?|constraints?|"
            r"rules?|guidelines?|censorship|boundaries)\b|"
            r"\b(?:no|zero)\s+(?:limits?|restrictions?|filters?|rules?)\s+apply\b"
        ),
        weight=0.75,
        description="Requests operation without the configured constraints",
    ),
    InjectionPattern(
        name="rules_no_longer_apply",
        category=InjectionCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            r"\b(?:no\s+longer|not)\s+(?:bound|restricted|limited|constrained)\s+by\b|"
            r"\b(?:rules?|restrictions?|guidelines?|filters?)\s+(?:do\s+not|don'?t|no\s+longer)\s+apply\b"
        ),
        weight=0.75,
        description="Asserts that constraints have been lifted",
    ),
    # ------------------------------------------------------------------
    # System prompt extraction
    # ------------------------------------------------------------------
    InjectionPattern(
        name="reveal_system_prompt",
        category=InjectionCategory.PROMPT_EXTRACTION,
        regex=_compile(
            r"\b(?:reveal|show|display|print|output|repeat|reproduce|recite|disclose|reveal|"
            r"tell\s+me|give\s+me|what\s+(?:is|are|were))\b[^.!?\n]{0,40}?"
            r"\b(?:system\s+(?:prompt|message|instruction)|initial\s+(?:prompt|instruction)|"
            r"your\s+(?:prompt|instructions|rules|directives|guidelines|configuration|"
            r"system\s+message)|the\s+prompt\s+above)"
        ),
        weight=0.85,
        description="Attempts to extract the system prompt",
    ),
    InjectionPattern(
        name="repeat_everything_above",
        category=InjectionCategory.PROMPT_EXTRACTION,
        regex=_compile(
            r"\b(?:repeat|print|output|echo|copy)\b[^.!?\n]{0,30}"
            r"\b(?:everything|all\s+(?:the\s+)?text|the\s+text)\b[^.!?\n]{0,20}"
            r"\b(?:above|before|preceding|prior)\b"
        ),
        weight=0.8,
        description="Attempts to dump preceding prompt content",
    ),
    InjectionPattern(
        name="verbatim_prompt_request",
        category=InjectionCategory.PROMPT_EXTRACTION,
        regex=_compile(
            r"\b(?:verbatim|word\s+for\s+word|exactly\s+as\s+(?:written|given))\b"
            r"[^.!?\n]{0,40}\b(?:instruction|prompt|system|rules?)\b|"
            r"\b(?:instruction|prompt|system\s+message)\b[^.!?\n]{0,30}\bverbatim\b"
        ),
        weight=0.7,
        description="Requests exact reproduction of instructions",
    ),
    # ------------------------------------------------------------------
    # Role hijacking
    # ------------------------------------------------------------------
    InjectionPattern(
        name="you_are_now",
        category=InjectionCategory.ROLE_HIJACK,
        regex=_compile(
            r"\byou\s+are\s+(?:now|no\s+longer|actually|really)\b|"
            r"\bfrom\s+now\s+on,?\s+you\b|"
            r"\byou\s+will\s+(?:now\s+)?(?:act|behave|respond|operate)\s+as\b"
        ),
        weight=0.7,
        description="Reassigns the assistant's identity",
    ),
    InjectionPattern(
        name="act_as_persona",
        category=InjectionCategory.ROLE_HIJACK,
        regex=_compile(
            r"\b(?:act|behave|respond|roleplay|role-play|pretend)\s+(?:as|like|to\s+be)\s+"
            r"(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken|evil|amoral|"
            r"different|new)\b|"
            r"\bpretend\s+(?:that\s+)?you\s+(?:are|have|can)\b"
        ),
        weight=0.65,
        description="Assigns an alternate, less-restricted persona",
    ),
    InjectionPattern(
        name="named_jailbreak_persona",
        category=InjectionCategory.JAILBREAK,
        regex=_compile(
            r"\b(?:DAN|STAN|DUDE|AIM|developer\s+mode|do\s+anything\s+now|"
            r"god\s+mode|sudo\s+mode|root\s+mode|debug\s+mode)\b"
        ),
        weight=0.75,
        description="References a known jailbreak persona or mode",
    ),
    InjectionPattern(
        name="hypothetical_bypass",
        category=InjectionCategory.JAILBREAK,
        regex=_compile(
            r"\b(?:hypothetically|in\s+a\s+fictional|imagine\s+(?:that\s+)?you|"
            r"for\s+(?:educational|research)\s+purposes\s+only|"
            r"this\s+is\s+(?:just\s+)?a\s+(?:test|simulation|game))\b"
            r"[^.!?\n]{0,60}\b(?:no\s+restrictions?|without\s+(?:any\s+)?(?:filter|restriction|limit)|"
            r"anything|ignore|bypass)\b"
        ),
        weight=0.55,
        description="Framing device used to license restricted behaviour",
    ),
    # ------------------------------------------------------------------
    # Authority spoofing and delimiter injection
    # ------------------------------------------------------------------
    InjectionPattern(
        name="fake_authority_header",
        category=InjectionCategory.AUTHORITY_SPOOF,
        regex=_compile(
            r"^\s*(?:system|admin|administrator|developer|root|openai|anthropic|"
            r"important\s+ai\s+instruction)\s*[:>\]]|"
            r"\b(?:important|urgent|critical)\s+(?:ai|system|assistant)\s+(?:instruction|message|notice|directive)\b"
        ),
        weight=0.8,
        description="Impersonates a privileged speaker",
    ),
    InjectionPattern(
        name="chat_template_token",
        category=InjectionCategory.DELIMITER_INJECTION,
        regex=_compile(
            r"<\|(?:im_start|im_end|system|user|assistant|endoftext|eot_id|start_header_id)\|>|"
            r"\[/?INST\]|<<\s*/?SYS\s*>>|<\|begin_of_text\|>|###\s*(?:system|instruction)\s*:"
        ),
        weight=0.85,
        description="Injects chat-template control tokens",
    ),
    InjectionPattern(
        name="fenced_role_block",
        category=InjectionCategory.DELIMITER_INJECTION,
        regex=_compile(r"```\s*(?:system|assistant|instructions?)\b"),
        weight=0.65,
        description="Opens a fenced block impersonating a privileged role",
    ),
    InjectionPattern(
        name="forged_data_fence",
        category=InjectionCategory.DELIMITER_INJECTION,
        regex=_compile(r"-{2,}\s*(?:BEGIN|END)\s+(?:DATA|QUESTION|CONTEXT)\b"),
        weight=0.7,
        description="Imitates SecureRAG's own context fences",
    ),
    # ------------------------------------------------------------------
    # Exfiltration and scope violation
    # ------------------------------------------------------------------
    InjectionPattern(
        name="exfiltrate_to_url",
        category=InjectionCategory.DATA_EXFILTRATION,
        regex=_compile(
            r"\b(?:send|post|transmit|upload|forward|leak|exfiltrate|report)\b"
            r"[^.!?\n]{0,60}\b(?:to|at)\b\s*(?:https?://|www\.|[\w.-]+\.[a-z]{2,}/)|"
            r"\b(?:append|add|include|encode)\b[^.!?\n]{0,40}\b(?:to\s+the\s+)?url\b|"
            r"!\[[^\]]*\]\(\s*https?://"
        ),
        weight=0.9,
        description="Attempts to move data to an external endpoint",
    ),
    InjectionPattern(
        name="access_other_users_data",
        category=InjectionCategory.SCOPE_VIOLATION,
        regex=_compile(
            r"\b(?:all|every|other|another|any)\s+(?:users?|people|accounts?|tenants?|customers?)"
            r"(?:'s|s')?\s+(?:documents?|files?|data|records?|information)\b|"
            r"\b(?:documents?|files?|data)\s+(?:of|belonging\s+to|from)\s+(?:other|all|another|every)\s+users?\b"
        ),
        weight=0.8,
        description="Requests data outside the caller's authorisation scope",
    ),
    InjectionPattern(
        name="list_entire_corpus",
        category=InjectionCategory.SCOPE_VIOLATION,
        regex=_compile(
            r"\b(?:list|show|dump|reveal|display|enumerate|output)\b[^.!?\n]{0,30}"
            r"\b(?:all|every|entire|complete|full)\b[^.!?\n]{0,20}"
            r"\b(?:documents?|files?|database|corpus|index|knowledge\s*base|records?)\b"
        ),
        weight=0.6,
        description="Attempts to enumerate the whole corpus",
    ),
    # ------------------------------------------------------------------
    # Encoding evasion and output manipulation
    # ------------------------------------------------------------------
    InjectionPattern(
        name="encoding_instruction",
        category=InjectionCategory.ENCODING_EVASION,
        regex=_compile(
            r"\b(?:decode|base64|rot13|hex\s*decode|from\s+base64|atob)\b"
            r"[^.!?\n]{0,50}\b(?:and\s+(?:then\s+)?(?:execute|run|follow|obey|do)|instruction|command)\b|"
            r"\b(?:execute|run|follow|obey)\b[^.!?\n]{0,30}\b(?:decoded|encoded)\b"
        ),
        weight=0.8,
        description="Hides an instruction behind an encoding step",
    ),
    InjectionPattern(
        name="suppress_disclosure",
        category=InjectionCategory.OUTPUT_MANIPULATION,
        regex=_compile(
            r"\b(?:do\s+not|don'?t|never)\b[^.!?\n]{0,30}"
            r"\b(?:mention|tell|reveal|disclose|inform|warn|say)\b[^.!?\n]{0,30}"
            r"\b(?:the\s+)?(?:user|anyone|them|this)\b|"
            r"\bwithout\s+(?:telling|informing|notifying|alerting)\s+(?:the\s+)?user\b"
        ),
        weight=0.85,
        description="Asks the model to conceal its behaviour from the user",
    ),
    InjectionPattern(
        name="ignore_context_answer_freely",
        category=InjectionCategory.OUTPUT_MANIPULATION,
        regex=_compile(
            r"\b(?:ignore|disregard)\b[^.!?\n]{0,30}\b(?:context|documents?|sources?|retrieved)\b|"
            r"\banswer\s+(?:from\s+)?(?:your\s+own\s+knowledge|without\s+(?:the\s+)?(?:context|documents?|sources?))\b"
        ),
        weight=0.7,
        description="Attempts to detach the answer from its evidence",
    ),
)


@dataclass(frozen=True)
class PatternHit:
    pattern: InjectionPattern
    matched_span: tuple[int, int]

    @property
    def name(self) -> str:
        return self.pattern.name

    @property
    def category(self) -> str:
        return self.pattern.category.value

    @property
    def weight(self) -> float:
        return self.pattern.weight


def scan_patterns(text: str) -> list[PatternHit]:
    """Return every pattern that matches ``text``."""
    if not text:
        return []
    hits: list[PatternHit] = []
    for pattern in PATTERNS:
        match = pattern.regex.search(text)
        if match:
            hits.append(PatternHit(pattern=pattern, matched_span=match.span()))
    return hits
