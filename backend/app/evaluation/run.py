"""CLI entry point for the evaluation suite.

    python -m app.evaluation.run

Runs against a **temporary, isolated database** by default so a run never
touches development or production data, and so results are reproducible from a
clean state. Point it at a real database with ``--database-url`` if you want to
evaluate a live configuration.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.evaluation.run",
        description="Run the SecureRAG evaluation suite and write a report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write reports (default: <repo>/evaluation/reports)",
    )
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        metavar="KIND",
        help=(
            "Restrict to case kinds: answerable unanswerable ambiguous "
            "direct_injection indirect_injection pii authorization benign_control"
        ),
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Evaluate against an existing database instead of a temporary one",
    )
    parser.add_argument(
        "--json-only", action="store_true", help="Skip the Markdown report"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the terminal summary"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "Exit non-zero if the overall pass rate falls below RATE (0-1). "
            "Use this in CI to catch guardrail regressions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # The environment must be configured before app.core.config is imported,
    # because Settings is a cached singleton built at first import.
    import os

    temporary_dir: tempfile.TemporaryDirectory | None = None
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    else:
        temporary_dir = tempfile.TemporaryDirectory(prefix="securerag-eval-")
        database_path = Path(temporary_dir.name) / "evaluation.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{database_path.as_posix()}"
        os.environ.setdefault("STORAGE_DIR", str(Path(temporary_dir.name) / "storage"))

    os.environ.setdefault("LOG_LEVEL", "WARNING")
    os.environ.setdefault("LOG_FORMAT", "console")
    # The suite deliberately sends bursts of adversarial input; rate limiting
    # would truncate the run and is measured separately by the test suite.
    os.environ["RATE_LIMIT_ENABLED"] = "false"

    from alembic import command
    from alembic.config import Config as AlembicConfig
    from app.core.config import BACKEND_DIR, REPO_ROOT
    from app.db.session import SessionLocal, engine
    from app.evaluation.datasets import cases_for
    from app.evaluation.report import (
        print_summary,
        write_json,
        write_markdown,
    )
    from app.evaluation.runner import EvaluationRunner

    alembic_config = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(alembic_config, "head")

    output_dir = args.output_dir or (REPO_ROOT / "evaluation" / "reports")
    cases = cases_for(set(args.kinds)) if args.kinds else None
    if args.kinds and not cases:
        print(f"No cases match {args.kinds}", file=sys.stderr)
        return 2

    session = SessionLocal()
    try:
        report = EvaluationRunner(session).run(cases)
    finally:
        session.close()
        engine.dispose()
        if temporary_dir is not None:
            temporary_dir.cleanup()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    written = [write_json(report, output_dir / f"evaluation-{stamp}.json")]
    # "latest" is always refreshed, including under --json-only. A stale
    # latest.json is worse than none: it is the file the docs link to, and
    # reading it after a run silently reports the *previous* run's numbers.
    written.append(write_json(report, output_dir / "latest.json"))
    if not args.json_only:
        written.append(write_markdown(report, output_dir / f"evaluation-{stamp}.md"))
        written.append(write_markdown(report, output_dir / "latest.md"))

    if not args.quiet:
        print_summary(report)
        print("  Reports written:")
        for path in written:
            print(f"    {path}")
        print()

    if args.fail_under is not None:
        pass_rate = report.totals["pass_rate"]
        if pass_rate < args.fail_under:
            print(
                f"FAIL: pass rate {pass_rate:.2%} is below the required "
                f"{args.fail_under:.2%}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
