"""Command-line interface for promptdiff."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from promptdiff import __version__
from promptdiff.compare import Thresholds, compare_paths
from promptdiff.errors import PromptdiffError
from promptdiff.report import render_json, render_text

_EPILOG = """\
examples:
  promptdiff examples/base.jsonl examples/new.jsonl
  promptdiff base.jsonl new.jsonl --min-score-drop 0.05 --max-regressions 2
  promptdiff base.jsonl new.jsonl --max-latency-increase 20 --max-cost-increase 15
  promptdiff base.jsonl new.jsonl --format json

exit codes:
  0  within thresholds
  1  a regression count or an increase exceeds its threshold
  2  the files or the flags are invalid
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptdiff",
        description="Compare two LLM eval runs and report regressions.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("base", help="baseline JSONL eval run")
    parser.add_argument("new", help="candidate JSONL eval run")
    parser.add_argument(
        "--pass-score",
        type=float,
        default=0.5,
        help="score at or above this is a pass (default: 0.5)",
    )
    parser.add_argument(
        "--min-score-drop",
        type=float,
        default=0.0,
        help="score decrease must be greater than this to count as a regression (default: 0)",
    )
    parser.add_argument(
        "--max-regressions",
        type=int,
        default=0,
        help="fail when score regressions exceed this count (default: 0)",
    )
    parser.add_argument(
        "--max-new-failures",
        type=int,
        default=0,
        help="fail when newly failing cases exceed this count (default: 0)",
    )
    parser.add_argument(
        "--max-removed",
        type=int,
        default=None,
        help="fail when removed cases exceed this count (off unless set)",
    )
    parser.add_argument(
        "--max-latency-increase",
        type=float,
        default=None,
        metavar="PCT",
        help="fail when mean latency rises by more than this percent (off unless set)",
    )
    parser.add_argument(
        "--max-cost-increase",
        type=float,
        default=None,
        metavar="PCT",
        help="fail when total cost rises by more than this percent (off unless set)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="also list latency and cost decreases",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print the prompt and output for regressed and newly failing cases",
    )
    parser.add_argument("--version", action="version", version=f"promptdiff {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two eval files. Returns 0, 1, or 2."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        thresholds = Thresholds(
            pass_score=args.pass_score,
            min_score_drop=args.min_score_drop,
            max_regressions=args.max_regressions,
            max_new_failures=args.max_new_failures,
            max_removed=args.max_removed,
            max_latency_increase_pct=args.max_latency_increase,
            max_cost_increase_pct=args.max_cost_increase,
        )
        report = compare_paths(args.base, args.new, thresholds)
    except PromptdiffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(
            render_text(report, show_all=args.show_all, verbose=args.verbose)
        )
    return 1 if report.summary.failed else 0
