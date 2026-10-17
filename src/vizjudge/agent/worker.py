"""Subprocess worker for a fixed allowlist of analysis tools; never eval user code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vizjudge.agent.judge import file_judge, http_judge, judge_report
from vizjudge.agent.runner import _refine
from vizjudge.agent.tools import list_candidates, render_chart_ids, validate_features
from vizjudge.charts.generator import ChartCandidate
from vizjudge.core.loader import load_dataset
from vizjudge.core.profiler import profile_dataframe
from vizjudge.core.report import _render, _safe_filename, analyze_frame
from vizjudge.core.type_infer import infer_types


def execute(request: dict) -> dict:
    tool = request["tool"]
    if tool == "refine":
        return _refine(request["report"], request["judgment"], request["action"])
    if tool == "judge":
        adapter = request.get("judge")
        llm = None
        if adapter:
            if adapter["kind"] == "file":
                if Path(adapter["path"]).stat().st_size > 1_000_000:
                    raise ValueError("Judge response file exceeds 1 MB")
                llm = file_judge(adapter["path"])
            elif adapter["kind"] == "http":
                llm = http_judge(adapter["url"], timeout=request["timeout"])
            else:
                raise ValueError("Unsupported Judge adapter")
        return judge_report(request["report"], llm)
    if tool not in {"profile", "analyze", "render", "candidates", "validate", "render_ids"}:
        raise ValueError(f"Tool is not allowed: {tool}")
    source = Path(request["path"])
    if source.stat().st_size > request["max_input_bytes"]:
        raise ValueError("Dataset exceeds max_input_bytes")
    frame = load_dataset(source)
    original_rows = len(frame)
    if original_rows > request["max_rows"]:
        frame = frame.sample(request["max_rows"], random_state=request["seed"])
    sampling = {
        "original_rows": original_rows,
        "analyzed_rows": len(frame),
        "sampled": original_rows > len(frame),
        "seed": request["seed"],
    }
    if tool == "render":
        report = request["report"]
        for index, chart in enumerate(report["ranked_charts"], start=1):
            if index > request["max_charts"]:
                chart["image"] = None
                continue
            fields = {
                name: value
                for name, value in chart.items()
                if name in ChartCandidate.__dataclass_fields__
            }
            candidate = ChartCandidate(**fields)
            filename = f"{index:02d}-{_safe_filename(candidate.key)}.png"
            _render(frame, candidate, Path(request["output_dir"]) / filename)
            chart["image"] = filename
        return report
    if tool == "profile":
        overrides = request.get("role_overrides")
        roles = infer_types(frame, overrides) if overrides else None
        return {"profile": profile_dataframe(frame, roles), "sampling": sampling}
    if tool == "candidates":
        return {**list_candidates(frame, **request["options"]), "sampling": sampling}
    if tool == "validate":
        return {**validate_features(frame, **request["options"]), "sampling": sampling}
    if tool == "render_ids":
        return {
            **render_chart_ids(
                frame,
                output_dir=request["output_dir"],
                max_charts=request["max_charts"],
                **request["options"],
            ),
            "sampling": sampling,
        }
    report = analyze_frame(frame, source=str(source.resolve()), **request["options"])
    report["sampling"] = sampling
    if sampling["sampled"]:
        report["warnings"].append(
            "Analysis uses a seeded row sample imposed by the execution budget."
        )
    return report


def main() -> int:
    try:
        request = json.load(sys.stdin)
        result = {"ok": True, "result": execute(request)}
    except Exception as exc:
        result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
