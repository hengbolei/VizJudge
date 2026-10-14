"""Reproducible offline v0.4 demo. The Judge fixture simulates an LLM response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vizjudge.agent import ExecutionBudget, run_agent
from vizjudge.agent.judge import build_evidence
from vizjudge.core.report import analyze_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo-v04"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    signal = rng.normal(size=300)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "signal_copy": signal * 2 + rng.normal(0, 0.01, 300),
            "segment": np.where(signal > 0, "positive", "negative"),
            "noise": rng.normal(size=300),
            "target": 3 * signal + rng.normal(0, 0.3, 300),
        }
    )
    dataset = args.output / "data.csv"
    frame.to_csv(dataset, index=False)
    # Build a deterministic, explicitly simulated response using the exact CSV
    # roundtrip used by the Agent. This exercises grounding without a model key.
    initial = analyze_frame(
        pd.read_csv(dataset), target="target", task="regression", top_k=6, render=False
    )
    context = build_evidence(initial)
    response = {
        "refinement": "reduce_charts",
        "reviews": [
            {
                "chart_id": chart["chart_id"],
                "evidence_ids": [item["id"] for item in chart["evidence"]],
                "priority": min(chart["score"], 90),
                "verdict": "keep",
                "rationale": "Simulated Judge: use cited association and held-out evidence.",
                "next_step": "Compare retained signals on a domain-appropriate validation split.",
            }
            for chart in context["charts"]
        ],
    }
    fixture = args.output / "simulated-judge.json"
    fixture.write_text(json.dumps(response, indent=2), encoding="utf-8")
    report = run_agent(
        dataset,
        target="target",
        task="regression",
        top_k=6,
        output_dir=args.output,
        judge={"kind": "file", "path": str(fixture)},
        budget=ExecutionBudget(max_iterations=2, max_rendered_charts=3),
    )
    print(
        f"VizJudge {report['vizjudge_version']}: {report['agent']['status']}; "
        f"{len(report['agent']['iterations'])} iterations; "
        f"{report['agent']['tool_calls']} tool calls; "
        f"{len(report['ranked_charts'])} final charts."
    )
    print(f"Report: {(args.output / 'summary.md').resolve()}")


if __name__ == "__main__":
    main()
