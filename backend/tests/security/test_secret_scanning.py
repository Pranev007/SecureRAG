"""The commit-time secret scanner.

This exists because a live Groq API key was pasted into ``render.yaml``, in
place of the ``sync: false`` directive that is there precisely so the value
never reaches git. It was caught before ``git add`` by luck.

The tests are shaped around the two ways a scanner fails:

* **Misses** are the obvious failure. One case per credential format.
* **False positives** are the quieter and worse one. A hook that fires on
  strings committed by design gets bypassed with ``--no-verify``, and a
  bypassed hook catches nothing ever again. The repository's own compose
  files and ``.env.example`` are the fixtures for that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCANNER = REPO_ROOT / "scripts" / "scan_secrets.py"

pytestmark = pytest.mark.security


def run_scanner(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCANNER), *(str(p) for p in paths)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )


def write(tmp_path: Path, content: str, name: str = "config.yaml") -> Path:
    target = tmp_path / name
    target.write_text(content, encoding="utf-8")
    return target


# ----------------------------------------------------------------------
# It must catch these
# ----------------------------------------------------------------------

CREDENTIALS = [
    ("groq", "LLM_API_KEY: gsk_" + "a1B2c3D4e5F6g7H8i9J0" * 2 + "kLmN"),
    ("openai", "OPENAI_API_KEY=sk-proj-" + "abcdefghij" * 3),
    ("anthropic", "key = 'sk-ant-api03-" + "xyz123ABC_" * 3 + "'"),
    ("github", "token: ghp_" + "0123456789abcdefghij" * 2),
    ("aws", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"),
    # Google keys are AIza + exactly 35 characters; a shorter fixture would
    # pass the test while proving nothing about a real key.
    ("google", "GOOGLE_API_KEY=AIza" + "SyD_a1B2c3D4e5F6g7H8i9J0kLmN-oPqRst"),
    # Split, like the others above, so the source text never contains a
    # contiguous token. GitHub push protection scans the blob, not the
    # runtime value, and rejected the push when this was one literal. The
    # test is unchanged in substance: the halves are joined before they
    # reach the scanner, so it still sees a whole token.
    ("slack", "SLACK_TOKEN=xoxb-" + "123456789012-abcdefghijklmnop"),
    ("huggingface", "HF_TOKEN=hf_" + "aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"),
    ("private key", "-----BEGIN RSA PRIVATE KEY-----"),
    (
        "remote db url",
        "DATABASE_URL=postgresql://admin:Hunter2Real@db.prod.example.com:5432/app",
    ),
]


@pytest.mark.parametrize(("label", "line"), CREDENTIALS, ids=[c[0] for c in CREDENTIALS])
def test_a_credential_is_refused(tmp_path, label, line):
    result = run_scanner(write(tmp_path, line + "\n"))

    assert result.returncode == 1, f"{label} was not caught:\n{result.stdout}"
    assert "refused" in result.stderr


def test_the_exact_key_that_caused_this(tmp_path):
    """The real shape of the incident: a value where a directive belonged."""
    # Synthetic. An earlier version of this fixture spliced in 20 real
    # characters of an actual leaked key so the test could prove the scanner
    # caught "the real shape" -- which is the exact instinct this scanner
    # exists to stop. The shape is what matters, not the provenance.
    leaked = "gsk_" + "Z9y8X7w6V5u4T3s2R1q0" * 2 + "PoNm"
    result = run_scanner(
        write(tmp_path, f"      - key: LLM_API_KEY\n        sync: {leaked}\n")
    )

    assert result.returncode == 1
    assert "Groq API key" in result.stderr


def test_the_finding_never_prints_the_whole_secret(tmp_path):
    """Scanner output reaches terminals, CI logs and screenshots.

    All three outlive the incident, so echoing the credential in full would
    turn a caught leak into a wider one.
    """
    secret = "gsk_" + "a1B2c3D4e5F6g7H8i9J0" * 2 + "kLmN"
    result = run_scanner(write(tmp_path, f"key: {secret}\n"))

    assert result.returncode == 1
    assert secret not in result.stderr
    assert secret not in result.stdout
    assert secret[:7] in result.stderr  # enough to locate it, not to use it


# ----------------------------------------------------------------------
# It must NOT catch these
# ----------------------------------------------------------------------

BENIGN = [
    (
        "compose service host",
        "postgresql+psycopg://securerag:securerag@postgres:5432/securerag",
    ),
    ("localhost", "postgresql://user:password@localhost:5432/dev"),
    ("loopback", "postgres://app:app@127.0.0.1/test"),
    ("prose", "Set LLM_API_KEY in the dashboard, never in this file."),
    ("placeholder", "LLM_API_KEY=<your key here>"),
    ("empty declaration", "      - key: LLM_API_KEY\n        sync: false"),
]


@pytest.mark.parametrize(("label", "line"), BENIGN, ids=[c[0] for c in BENIGN])
def test_a_benign_string_is_allowed(tmp_path, label, line):
    """False positives are how a hook gets disabled, so they are a defect."""
    result = run_scanner(write(tmp_path, line + "\n"))

    assert result.returncode == 0, f"{label} wrongly flagged:\n{result.stderr}"


def test_an_inline_pragma_exempts_a_deliberate_example(tmp_path):
    secret = "gsk_" + "a1B2c3D4e5F6g7H8i9J0" * 2 + "kLmN"
    result = run_scanner(
        write(tmp_path, f"example = '{secret}'  # pragma: allowlist secret\n")
    )

    assert result.returncode == 0


# ----------------------------------------------------------------------
# The repository itself
# ----------------------------------------------------------------------


def test_no_tracked_file_contains_a_credential():
    """The check CI runs. Fails the build rather than the review.

    Committed-by-design database URLs must not trip this, or the whole thing
    gets switched off -- which is why the scanner distinguishes a compose
    service alias from a reachable host.
    """
    result = subprocess.run(
        [sys.executable, str(SCANNER), "--all-tracked"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"credential found in a tracked file:\n{result.stderr}"
