"""Command-line interface for VizJudge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vizjudge.core.loader import load_dataset
from vizjudge.core.profiler import profile_dataframe
from vizjudge.core.report import analyze_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vizjudge", description="Judge chart value and turn evidence into ML actions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the complete VizJudge workflow")
    analyze.add_argument("dataset", type=Path)
    analyze.add_argument("--target")
    analyze.add_argument("--output", type=Path, default=Path("outputs/vizjudge-report"))
    analyze.add_argument("--top-k", type=int, default=8)
    analyze.add_argument("--no-render", action="store_true")

    profile = subparsers.add_parser("profile", help="Print a dataset profile as JSON")
    profile.add_argument("dataset", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        result = profile_dataframe(load_dataset(args.dataset))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    result = analyze_dataset(
        args.dataset,
        target=args.target,
        output_dir=args.output,
        top_k=args.top_k,
        render=not args.no_render,
    )
    print(
        f"Judged {result['candidate_count']} candidates; "
        f"saved {len(result['ranked_charts'])} ranked charts to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
