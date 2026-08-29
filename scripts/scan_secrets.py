#!/usr/bin/env python
"""Refuse to let a credential reach a commit.

    python scripts/scan_secrets.py                 # staged changes (pre-commit)
    python scripts/scan_secrets.py --all-tracked   # every tracked file (CI)
    python scripts/scan_secrets.py FILE [FILE ...]

WHY THIS EXISTS
---------------
A live Groq API key was pasted into ``render.yaml`` in place of the
``sync: false`` directive that exists precisely so the value never lands in
git. It was caught before ``git add`` by luck, not by process.

The irony is the point: this repository already contains a detector that
recognises exactly that pattern -- ``app/security/pii/patterns.py`` classifies
``sk-...`` as an ``API_KEY`` and has a test asserting so. It was pointed at
model output and never at the codebase. This script points it at the codebase.

WHAT IT CANNOT DO
-----------------
Pattern matching finds credentials that *look* like credentials. It will miss a
password that looks like a word, a base64 blob, or a key format not listed
below. It is a seatbelt, not a guarantee, and a clean run is not evidence that
a diff is free of secrets. Treat a hit as certain and a pass as unproven.

Exit codes: 0 nothing found, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Vendor-specific prefixes are worth listing individually rather than as one
# loose "long random string" rule: the prefix is what makes a match certain
# rather than probable, which is the difference between a hook people keep and
# a hook people disable.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{40,}")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    (
        "database URL with inline password",
        re.compile(
            r"\b(?:postgres|postgresql|mysql|mongodb)(?:\+\w+)?://"
            r"(?P<user>[^\s:@/]+):(?P<password>[^\s:@/]+)@(?P<host>[^\s:/?]+)"
        ),
    ),
)


def _is_local_placeholder(match: re.Match[str]) -> bool:
    """Is this database URL an obvious development stand-in, not a secret?

    Without this the rule fires on every compose file and ``.env.example`` in
    the repository. A scanner that cries wolf on strings that are committed by
    design is one that gets bypassed with ``--no-verify``, after which it never
    catches anything again -- so a false positive here is more dangerous than
    the narrow rule it would have bought.

    Two signals, both strong:

    * the host is unreachable from anywhere else -- ``localhost``, a loopback
      address, or a bare name with no dot, which is a compose service alias;
    * the username equals the password, which no real deployment does and
      almost every throwaway fixture does.
    """
    groups = match.groupdict()
    host = (groups.get("host") or "").split(":")[0].lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or "." not in host:
        return True
    return groups.get("user") == groups.get("password")


# Findings a rule can prove benign. Keyed by label so the exemption stays
# visible beside the rule it softens rather than hiding in the matcher.
VALIDATORS: dict[str, object] = {
    "database URL with inline password": _is_local_placeholder,
}

# Files whose whole purpose is to contain credential-shaped strings.
ALLOWLISTED_PATHS = {
    "scripts/scan_secrets.py",
    "backend/tests/security/test_output_guardrails.py",
    "backend/app/security/pii/patterns.py",
    "backend/tests/security/test_secret_scanning.py",
}

# Marks a line as a deliberate example. Kept explicit and greppable so an
# allowlist entry is a visible decision rather than a silent exemption.
INLINE_ALLOW = re.compile(r"#\s*noqa:\s*secret|#\s*pragma:\s*allowlist secret")

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".ico",
    ".lock",
}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def _git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def staged_files() -> list[Path]:
    return [
        Path(p) for p in _git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    ]


def tracked_files() -> list[Path]:
    return [Path(p) for p in _git("ls-files")]


def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    return path.as_posix() not in ALLOWLISTED_PATHS


def scan(path: Path) -> list[tuple[int, str, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    findings: list[tuple[int, str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if INLINE_ALLOW.search(line):
            continue
        for label, pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                validator = VALIDATORS.get(label)
                if validator is not None and validator(match):
                    continue
                # Never echo the secret in full: this output goes to terminals,
                # CI logs and screenshots, all of which outlive the incident.
                found = match.group(0)
                masked = f"{found[:7]}...{found[-3:]}" if len(found) > 14 else "***"
                findings.append((number, label, masked))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="scan every tracked file instead of the staged set",
    )
    args = parser.parse_args(argv)

    if args.paths:
        candidates = args.paths
    elif args.all_tracked:
        candidates = tracked_files()
    else:
        candidates = staged_files()

    targets = [p for p in candidates if should_scan(p)]
    total = 0
    for path in targets:
        for number, label, masked in scan(path):
            total += 1
            print(f"{path}:{number}: {label}  ({masked})", file=sys.stderr)

    if total:
        print(
            f"\n{total} possible credential(s) found. The commit was refused.\n"
            "\n"
            "If a hit is real:\n"
            "  1. REVOKE the credential at the provider. Removing it from the\n"
            "     file is not enough -- assume anything written down has leaked.\n"
            "  2. Put the value in the deployment platform's dashboard instead.\n"
            "  3. If it was already committed, rewriting history is necessary\n"
            "     but NOT sufficient; revocation is the control that matters.\n"
            "\n"
            "If it is a deliberate example, append '# pragma: allowlist secret'\n"
            "to the line, or add the path to ALLOWLISTED_PATHS in this script.\n",
            file=sys.stderr,
        )
        return 1

    print(f"scan_secrets: {len(targets)} file(s) checked, nothing found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
