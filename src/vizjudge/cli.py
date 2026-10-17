"""Command-line interface for VizJudge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vizjudge._version import __version__
from vizjudge.agent import ExecutionBudget, run_agent
from vizjudge.core.loader import load_dataset
from vizjudge.core.profiler import profile_dataframe
from vizjudge.core.report import analyze_dataset
from vizjudge.integrations import (
    MCP_CLIENTS,
    SKILL_TARGETS,
    doctor,
    install_skill,
    mcp_config,
    skill_root,
    skill_source,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vizjudge", description="Judge chart value and turn evidence into ML actions."
    )
    parser.add_argument("--version", action="version", version=f"VizJudge {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the complete VizJudge workflow")
    analyze.add_argument("dataset", type=Path)
    analyze.add_argument("--target")
    analyze.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    analyze.add_argument("--role", action="append", default=[], metavar="COLUMN=ROLE")
    analyze.add_argument("--output", type=Path, default=Path("outputs/vizjudge-report"))
    analyze.add_argument("--top-k", type=int, default=8)
    analyze.add_argument("--no-render", action="store_true")
    analyze.add_argument("--no-validation", action="store_true")
    analyze.add_argument("--cv-folds", type=int, default=5)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--baseline", action="append", default=[], metavar="COLUMN")
    analyze.add_argument("--no-diversity", action="store_true")

    profile = subparsers.add_parser("profile", help="Print a dataset profile as JSON")
    profile.add_argument("dataset", type=Path)
    agent = subparsers.add_parser("agent", help="Run the bounded plan/analyze/judge/refine loop")
    agent.add_argument("dataset", type=Path)
    agent.add_argument("--target")
    agent.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    agent.add_argument("--role", action="append", default=[], metavar="COLUMN=ROLE")
    agent.add_argument("--output", type=Path, default=Path("outputs/vizjudge-agent"))
    agent.add_argument("--top-k", type=int, default=6)
    agent.add_argument("--no-render", action="store_true")
    agent.add_argument("--no-validation", action="store_true")
    agent.add_argument("--cv-folds", type=int, default=5)
    agent.add_argument("--seed", type=int, default=42)
    agent.add_argument("--baseline", action="append", default=[], metavar="COLUMN")
    adapter = agent.add_mutually_exclusive_group()
    adapter.add_argument("--judge-file", type=Path, help="Replay an offline LLM response")
    adapter.add_argument("--judge-url", help="Call an explicit JSON Judge endpoint")
    agent.add_argument("--max-tool-calls", type=int, default=10)
    agent.add_argument("--max-iterations", type=int, default=2)
    agent.add_argument("--max-seconds", type=float, default=120)
    agent.add_argument("--python-timeout", type=float, default=30)
    agent.add_argument("--max-rows", type=int, default=10000)
    agent.add_argument("--max-candidates", type=int, default=200)
    agent.add_argument("--max-rendered-charts", type=int, default=8)
    agent.add_argument("--max-input-bytes", type=int, default=50000000)

    skill = subparsers.add_parser("skill", help="Install the bundled agent skill")
    skill_commands = skill.add_subparsers(dest="skill_command", required=True)
    install = skill_commands.add_parser(
        "install", help="Copy the skill into a host's skills folder"
    )
    install.add_argument("--target", choices=sorted(SKILL_TARGETS), default="agents")
    install.add_argument(
        "--project", action="store_true", help="Install under the current directory, not home"
    )
    install.add_argument("--dest", type=Path, help="Explicit skills root; overrides --target")
    install.add_argument("--force", action="store_true", help="Replace an installed copy")
    skill_commands.add_parser("path", help="Print the bundled skill folder")

    config = subparsers.add_parser("mcp-config", help="Print MCP server config for a host agent")
    config.add_argument("--client", choices=MCP_CLIENTS, default="dsh")
    subparsers.add_parser("doctor", help="Check the installation for host-agent use")
    return parser


def _integration(args: argparse.Namespace) -> int:
    if args.command == "skill":
        if args.skill_command == "path":
            print(skill_source())
            return 0
        installed = install_skill(skill_root(args.target, args.project, args.dest), args.force)
        print(f"Installed the VizJudge skill to {installed.resolve()}")
        return 0
    if args.command == "mcp-config":
        print(mcp_config(args.client), end="")
        return 0
    rows = doctor()
    for check, ok, detail in rows:
        print(f"[{'ok' if ok else 'missing'}] {check}: {detail}")
    return 0 if all(ok for _, ok, _ in rows) else 1


def _run(args: argparse.Namespace) -> int:
    if args.command in {"skill", "mcp-config", "doctor"}:
        return _integration(args)
    if args.command == "profile":
        result = profile_dataframe(load_dataset(args.dataset))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    overrides = {}
    for item in args.role:
        column, separator, role = item.rpartition("=")
        if not separator or not column or not role:
            raise ValueError("--role must use COLUMN=ROLE")
        if column in overrides:
            raise ValueError(f"Duplicate role override: {column}")
        overrides[column] = role
    common = dict(
        target=args.target,
        task=args.task,
        role_overrides=overrides,
        validation=not args.no_validation,
        cv_folds=args.cv_folds,
        seed=args.seed,
        baseline_features=args.baseline,
        output_dir=args.output,
        top_k=args.top_k,
        render=not args.no_render,
    )
    if args.command == "agent":
        budget = ExecutionBudget(
            max_tool_calls=args.max_tool_calls,
            max_iterations=args.max_iterations,
            max_seconds=args.max_seconds,
            python_timeout_seconds=args.python_timeout,
            max_rows=args.max_rows,
            max_candidates=args.max_candidates,
            max_rendered_charts=args.max_rendered_charts,
            max_input_bytes=args.max_input_bytes,
        )
        judge = None
        if args.judge_file:
            judge = {"kind": "file", "path": str(args.judge_file)}
        elif args.judge_url:
            judge = {"kind": "http", "url": args.judge_url}
        result = run_agent(args.dataset, **common, budget=budget, judge=judge)
        print(f"Agent {result['agent']['status']}; report saved to {args.output.resolve()}")
        return 0 if result["agent"]["status"] == "completed" else 3
    result = analyze_dataset(
        args.dataset,
        target=args.target,
        task=args.task,
        role_overrides=overrides,
        validation=not args.no_validation,
        cv_folds=args.cv_folds,
        seed=args.seed,
        baseline_features=args.baseline,
        diversity=not args.no_diversity,
        output_dir=args.output,
        top_k=args.top_k,
        render=not args.no_render,
    )
    print(
        f"Judged {result['candidate_count']} candidates; "
        f"saved {len(result['ranked_charts'])} ranked charts to {args.output.resolve()}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
