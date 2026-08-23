"""PII detection patterns with checksum validation.

WHY CHECKSUMS AND NOT JUST REGEX
--------------------------------
"Sixteen digits" matches an order number, a serial number, a timestamp, and a
credit card.  A regex-only PII detector produces so many false positives that
in ``redact`` mode it destroys the answer, and teams turn it off.  Where a real
validation algorithm exists, it is applied:

* **Credit cards** -- Luhn (ISO/IEC 7812).
* **Aadhaar** -- Verhoeff, the checksum UIDAI actually specifies.
* **PAN** -- structural validation of the 5-letter/4-digit/1-letter form.
* **IBAN** -- mod-97 (ISO 13616).
* **US SSN** -- structural rules (no 000/666/900+ area, no 00 group, no 0000
  serial).

Context words ("card", "aadhaar", "ssn", "invoice") adjust confidence further.

LIMITATIONS -- stated plainly, and repeated in docs/security.md
---------------------------------------------------------------
This is a pattern-and-checksum detector, not a named-entity model.  It does
**not** detect: person names, physical addresses, dates of birth, or any
identifier whose format it has not been taught.  It is biased toward
identifiers with machine-checkable structure, and it will miss free-text
disclosures entirely ("my account is the one under my wife's maiden name").
Set ``PII_ENGINE=presidio`` for NER-backed detection.  No PII detector,
including that one, should be treated as a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class PIIType(StrEnum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    IBAN = "IBAN"
    API_KEY = "API_KEY"
    PASSPORT = "PASSPORT"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"


@dataclass(frozen=True)
class PIIPattern:
    type: PIIType
    regex: re.Pattern[str]
    base_confidence: float
    # Words that, appearing near a match, make it much more likely to be real.
    context_words: frozenset[str] = frozenset()
    validator: str | None = None


# ---------------------------------------------------------------------------
# Checksum validators
# ---------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """Luhn (mod-10) check used by all major card networks."""
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 12:
        return False
    total = 0
    parity = len(numbers) % 2
    for index, digit in enumerate(numbers):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Verhoeff tables (dihedral group D5). Catches all single-digit errors and all
# adjacent transpositions -- which is why UIDAI chose it over Luhn for Aadhaar.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum, as specified for Aadhaar numbers."""
    cleaned = [int(d) for d in digits if d.isdigit()]
    if len(cleaned) != 12:
        return False
    if cleaned[0] in (0, 1):
        # UIDAI does not issue numbers beginning 0 or 1.
        return False
    check = 0
    for index, digit in enumerate(reversed(cleaned)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][digit]]
    return check == 0


def iban_valid(candidate: str) -> bool:
    """ISO 13616 mod-97 check."""
    cleaned = re.sub(r"[\s-]", "", candidate).upper()
    if not 15 <= len(cleaned) <= 34 or not cleaned[:2].isalpha():
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    digits = "".join(
        str(ord(char) - 55) if char.isalpha() else char for char in rearranged
    )
    if not digits.isdigit():
        return False
    return int(digits) % 97 == 1


def ssn_valid(candidate: str) -> bool:
    """Structural validity of a US Social Security Number."""
    digits = re.sub(r"[\s-]", "", candidate)
    if len(digits) != 9 or not digits.isdigit():
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area[0] == "9":
        return False
    return group != "00" and serial != "0000"


def pan_valid(candidate: str) -> bool:
    """Indian PAN: AAAAA9999A, with a valid holder-type character."""
    cleaned = candidate.strip().upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", cleaned):
        return False
    return cleaned[3] in "ABCFGHLJPTK"


VALIDATORS = {
    "luhn": luhn_valid,
    "verhoeff": verhoeff_valid,
    "iban": iban_valid,
    "ssn": ssn_valid,
    "pan": pan_valid,
}


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

PATTERNS: tuple[PIIPattern, ...] = (
    PIIPattern(
        type=PIIType.EMAIL,
        regex=re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        base_confidence=0.95,
        context_words=frozenset({"email", "e-mail", "contact", "mail", "address"}),
    ),
    PIIPattern(
        type=PIIType.CREDIT_CARD,
        # Major-network prefixes, with optional space/dash grouping.
        regex=re.compile(
            r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|6(?:011|5[0-9]{2})|3[47][0-9]{2})"
            r"[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{1,4}\b"
        ),
        base_confidence=0.5,
        context_words=frozenset(
            {"card", "credit", "debit", "visa", "mastercard", "amex", "payment", "cvv"}
        ),
        validator="luhn",
    ),
    PIIPattern(
        type=PIIType.AADHAAR,
        regex=re.compile(r"\b[2-9][0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b"),
        base_confidence=0.45,
        context_words=frozenset({"aadhaar", "aadhar", "uid", "uidai", "आधार"}),
        validator="verhoeff",
    ),
    PIIPattern(
        type=PIIType.PAN,
        regex=re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        base_confidence=0.6,
        context_words=frozenset({"pan", "permanent account number", "income tax"}),
        validator="pan",
    ),
    PIIPattern(
        type=PIIType.SSN,
        regex=re.compile(r"\b[0-9]{3}[\s\-][0-9]{2}[\s\-][0-9]{4}\b"),
        base_confidence=0.6,
        context_words=frozenset({"ssn", "social security", "social-security"}),
        validator="ssn",
    ),
    PIIPattern(
        type=PIIType.IBAN,
        regex=re.compile(r"\b[A-Z]{2}[0-9]{2}[\s\-]?(?:[A-Z0-9][\s\-]?){11,30}\b"),
        base_confidence=0.4,
        context_words=frozenset({"iban", "bank", "account", "swift", "transfer"}),
        validator="iban",
    ),
    PIIPattern(
        type=PIIType.PHONE,
        regex=re.compile(
            r"(?:\+[0-9]{1,3}[\s\-.]?)?(?:\([0-9]{2,4}\)[\s\-.]?)?"
            r"[0-9]{3,5}[\s\-.][0-9]{3,4}(?:[\s\-.][0-9]{3,4})?\b"
        ),
        base_confidence=0.35,
        context_words=frozenset(
            {"phone", "mobile", "tel", "telephone", "call", "contact", "fax", "cell"}
        ),
    ),
    PIIPattern(
        type=PIIType.IP_ADDRESS,
        regex=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\b"
        ),
        base_confidence=0.7,
        context_words=frozenset({"ip", "address", "host", "server", "client"}),
    ),
    PIIPattern(
        type=PIIType.API_KEY,
        # Well-known credential prefixes plus generic long high-entropy tokens
        # that are explicitly labelled as secrets.
        regex=re.compile(
            r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|"
            r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{30,}|xox[baprs]-[A-Za-z0-9\-]{10,})\b"
        ),
        base_confidence=0.9,
        context_words=frozenset({"key", "token", "secret", "api", "credential"}),
    ),
    PIIPattern(
        type=PIIType.PASSPORT,
        regex=re.compile(r"\b[A-PR-WY][0-9]{7}\b"),
        base_confidence=0.35,
        context_words=frozenset({"passport", "travel document"}),
    ),
    PIIPattern(
        type=PIIType.DATE_OF_BIRTH,
        regex=re.compile(
            r"\b(?:0?[1-9]|[12][0-9]|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.]"
            r"(?:19|20)[0-9]{2}\b"
        ),
        base_confidence=0.3,
        context_words=frozenset({"dob", "birth", "born", "birthday", "birthdate"}),
    ),
)
