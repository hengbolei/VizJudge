"""An auditable plan -> tool -> judge -> refine loop with execution limits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from vizjudge._version import SCHEMA_VERSION, __version__
from vizjudge.agent.judge import chart_key
from vizjudge.charts.generator import ChartCandidate
from vizjudge.core.report import markdown_summary
from vizjudge.core.validation import validate_top_k
from vizjudge.recommendations.feature_engineering import recommend_features
from vizjudge.recommendations.modeling import recommend_models
from vizjudge.scoring.predictive import validate_options


@dataclass(frozen=True)
class ExecutionBudget:
    max_tool_calls: int = 10
    max_iterations: int = 2
    max_seconds: float = 120.0
    python_timeout_seconds: float = 30.0
    max_rows: int = 10000
    max_candidates: int = 200
    max_rendered_charts: int = 8
    max_input_bytes: int = 50_000_000

    def validate(self) -> None:
        for name in (
            "max_tool_calls",
            "max_iterations",
            "max_rows",
            "max_candidates",
            "max_input_bytes",
        ):
            validate_top_k(getattr(self, name))
        if (
            isinstance(self.max_rendered_charts, bool)
            or not isinstance(self.max_rendered_charts, int)
            or self.max_rendered_charts < 0
        ):
            raise ValueError("max_rendered_charts must be a non-negative integer")
        for name in ("max_seconds", "python_timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value <= 3600
            ):
                raise ValueError(f"{name} must be a finite number in (0, 3600]")


class BudgetExceeded(RuntimeError):
    pass


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Executor:
    """Run allowlisted tools in bounded worker subprocesses and record an audit trace."""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self.started = time.monotonic()
        self.trace: list[dict[str, Any]] = []

    def call(self, tool: str, request: dict[str, Any], reason: str) -> dict[str, Any]:
        remaining = self.budget.max_seconds - (time.monotonic() - self.started)
        if len(self.trace) >= self.budget.max_tool_calls or remaining <= 0:
            raise BudgetExceeded("Total tool-call or wall-clock budget exhausted")
        timeout = min(remaining, self.budget.python_timeout_seconds)
        request = {**request, "tool": tool, "timeout": timeout}
        event = {
            "id": f"T{len(self.trace) + 1}",
            "tool": tool,
            "reason": reason,
            "started_at": datetime.now(UTC).isoformat(),
            "input_sha256": _digest(request),
            "status": "running",
        }
        event["parameters"] = {
            key: value for key, value in request.items() if key not in {"report", "judgment"}
        }
        self.trace.append(event)
        started = time.monotonic()
        # Preserve only runtime variables; tokens are passed solely to the Judge tool.
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "HOME",
                "USERPROFILE",
                "LOCALAPPDATA",
                "APPDATA",
                "MPLCONFIGDIR",
                "XDG_CACHE_HOME",
                "VIZJUDGE_CACHE_DIR",
                "VIZJUDGE_CACHE_MAX_BYTES",
                "VIZJUDGE_NO_CACHE",
            }
        }
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        if tool == "judge" and "VIZJUDGE_JUDGE_TOKEN" in os.environ:
            env["VIZJUDGE_JUDGE_TOKEN"] = os.environ["VIZJUDGE_JUDGE_TOKEN"]
        try:
            result = subprocess.run(
                [sys.executable, "-m", "vizjudge.agent.worker"],
                input=json.dumps(request, ensure_ascii=False, allow_nan=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if len(result.stdout.encode("utf-8")) > 10_000_000:
                raise ValueError("Tool output exceeds 10 MB")
            envelope = json.loads(result.stdout)
            if not envelope["ok"]:
                raise ValueError(envelope["error"])
            if result.returncode:
                raise ValueError("Python worker exited unsuccessfully")
            if "cache" in envelope:
                event["cache"] = envelope["cache"]
            event.update(status="completed", output_sha256=_digest(envelope["result"]))
            return envelope["result"]
        except subprocess.TimeoutExpired as exc:
            event.update(status="timeout", error="Python worker timed out and was terminated")
            raise BudgetExceeded(event["error"]) from exc
        except Exception as exc:
            event.update(status="failed", error=str(exc))
            raise
        finally:
            event["duration_seconds"] = round(time.monotonic() - started, 6)


def refine_report(report: dict[str, Any], judgment: dict[str, Any], action: str) -> dict[str, Any]:
    """Apply one allowed refinement action and refresh the dependent recommendations."""
    priorities = {item["chart_id"]: item for item in judgment["reviews"]}
    # Unreviewed leakage suspects are deferred as features but stay visible under every
    # refinement; dropping them would hide exactly what needs a human decision. Once a
    # host has reviewed one, its verdict applies like any other chart's.
    leaks = [
        chart
        for chart in report["ranked_charts"]
        if chart["metrics"].get("leakage_suspect")
        and priorities[chart_key(chart)]["llm_rationale"] is None
    ]
    charts = [chart for chart in report["ranked_charts"] if chart not in leaks]
    if action == "prefer_validated":
        charts.sort(
            key=lambda chart: (
                priorities[chart_key(chart)]["verdict"] == "defer",
                -priorities[chart_key(chart)].get("combined_priority", chart["selection_score"]),
                chart_key(chart),
            )
        )
        # Keep at least one chart, but avoid spending attention on unsupported signals.
        supported = [chart for chart in charts if priorities[chart_key(chart)]["verdict"] == "keep"]
        charts = supported or (charts[:1] if not leaks else [])
    elif action == "increase_diversity":
        seen = set()
        diverse = []
        for chart in charts:
            feature = chart["feature"] or chart["x"]
            if any("redundant with" in reason for reason in chart["selection_reasons"]):
                continue
            if feature not in seen:
                seen.add(feature)
                diverse.append(chart)
        charts = diverse
    elif action == "reduce_charts":
        charts.sort(
            key=lambda chart: (
                -priorities[chart_key(chart)].get("combined_priority", chart["selection_score"]),
                chart_key(chart),
            )
        )
        charts = charts[: max(1, len(charts) // 2)]
    charts = leaks + charts
    report["ranked_charts"] = charts
    specs = [
        ChartCandidate(
            **{
                key: value
                for key, value in chart.items()
                if key in ChartCandidate.__dataclass_fields__
            }
        )
        for chart in charts
    ]
    report["recommendations"]["feature_engineering"] = recommend_features(report["profile"], specs)
    report["recommendations"]["modeling"] = recommend_models(
        report["target"], report["roles"], specs
    )
    return report


def run_agent(
    path: str | Path,
    *,
    target: str | None = None,
    task: str = "auto",
    role_overrides: dict[str, str] | None = None,
    output_dir: str | Path = "outputs/vizjudge-agent",
    top_k: int = 6,
    render: bool = True,
    validation: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    baseline_features: list[str] | None = None,
    budget: ExecutionBudget | None = None,
    judge: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a reproducible local Agent loop. External Judge calls require explicit configuration."""
    limits = budget or ExecutionBudget()
    limits.validate()
    validate_top_k(top_k)
    validate_options(cv_folds, seed, baseline_features)
    if any(not isinstance(value, bool) for value in (render, validation)):
        raise ValueError("render and validation must be booleans")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > limits.max_input_bytes:
        raise ValueError("Dataset exceeds max_input_bytes")
    dataset_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if judge is not None and (
        not isinstance(judge, dict) or judge.get("kind") not in {"file", "http"}
    ):
        raise ValueError("judge must configure a file or HTTP adapter")
    if judge and judge["kind"] == "file":
        judge = {**judge, "path": str(Path(judge["path"]).expanduser().resolve())}
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    executor = Executor(limits)
    plan = [
        {"tool": "profile", "purpose": "Inspect data quality and establish sampling scope."},
        {"tool": "analyze", "purpose": "Validate candidate signals before rendering."},
        {
            "tool": "judge",
            "purpose": "Combine statistical confidence with grounded chart judgments.",
        },
        {"tool": "refine", "purpose": "Apply an allowed improvement and evaluate it again."},
        {"tool": "render", "purpose": "Render only the final selected evidence."},
    ]
    request = {
        "path": str(source),
        "dataset_sha256": dataset_hash,
        "max_input_bytes": limits.max_input_bytes,
        "max_rows": limits.max_rows,
        "seed": seed,
    }
    options = {
        "target": target,
        "task": task,
        "role_overrides": role_overrides,
        "validation": validation,
        "cv_folds": cv_folds,
        "seed": seed,
        "baseline_features": baseline_features,
        "top_k": min(top_k, limits.max_candidates),
        "max_candidates": limits.max_candidates,
        "render": False,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vizjudge_version": __version__,
        "source": str(source),
        "ranked_charts": [],
    }
    iterations = []
    judgment = None
    status, stop_reason = "completed", "converged"
    failure = None
    try:
        profile = executor.call("profile", request, plan[0]["purpose"])
        report = executor.call("analyze", {**request, "options": options}, plan[1]["purpose"])
        report["sampling"] = profile["sampling"]
        for iteration in range(1, limits.max_iterations + 1):
            # A file is a fixed recording of the first Judge pass. Subsequent checks
            # use statistics so IDs cannot silently point at a different chart.
            adapter = judge if iteration == 1 or (judge and judge["kind"] == "http") else None
            judgment = executor.call(
                "judge",
                {"report": report, "judge": adapter},
                f"Evaluate iteration {iteration} with cited evidence.",
            )
            action = judgment["refinement"]
            entry = {
                "iteration": iteration,
                "judgment": judgment,
                "action": action,
                "before": [chart_key(chart) for chart in report["ranked_charts"]],
            }
            iterations.append(entry)
            if action == "none":
                entry["after"] = entry["before"]
                break
            if iteration == limits.max_iterations:
                entry["after"] = entry["before"]
                stop_reason = "iteration_budget"
                break
            report = executor.call(
                "refine",
                {"report": report, "judgment": judgment, "action": action},
                f"Apply {action} using reviewed evidence.",
            )
            entry["after"] = [chart_key(chart) for chart in report["ranked_charts"]]
            if entry["before"] == entry["after"]:
                stop_reason = "no_change"
                break
        if render and report["ranked_charts"] and limits.max_rendered_charts:
            # Render the selected specs in a worker without rescoring/reordering.
            report = executor.call(
                "render",
                {
                    **request,
                    "report": report,
                    "output_dir": str(destination),
                    "max_charts": limits.max_rendered_charts,
                },
                "Render final selected charts within the image budget.",
            )
        elif render and limits.max_rendered_charts == 0:
            stop_reason = "render_budget"
    except BudgetExceeded as exc:
        status, stop_reason = "budget_exhausted", str(exc)
    except Exception as exc:
        status, stop_reason, failure = "failed", str(exc), exc
    try:
        unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == dataset_hash
    except OSError:
        unchanged = False
    if not unchanged:
        status, stop_reason = "failed", "Dataset changed during Agent execution"
        failure = ValueError(stop_reason)
    report["agent"] = {
        "status": status,
        "stop_reason": stop_reason,
        "plan": plan,
        "budget": asdict(limits),
        "iterations": iterations,
        "tool_calls": len(executor.trace),
        "trace": executor.trace,
        "duration_seconds": round(time.monotonic() - executor.started, 6),
        "dataset_sha256": dataset_hash,
        "judge_adapter": judge["kind"] if judge else "statistical_only",
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "matplotlib")},
        },
    }
    if judgment:
        reviews = {item["chart_id"]: deepcopy(item) for item in judgment["reviews"]}
        for iteration in iterations:
            if iteration["judgment"]["mode"] != "statistical_and_llm":
                continue
            for review in iteration["judgment"]["reviews"]:
                if review["chart_id"] in reviews and review["llm_priority"] is not None:
                    for key in (
                        "llm_priority",
                        "llm_rationale",
                        "llm_evidence_ids",
                        "combined_priority",
                    ):
                        if key in review:
                            reviews[review["chart_id"]][key] = review[key]
                    reviews[review["chart_id"]]["llm_source_iteration"] = iteration["iteration"]
                    reviews[review["chart_id"]]["next_step"] = review["next_step"]
        for chart in report["ranked_charts"]:
            chart["judgment"] = reviews.get(chart_key(chart))
    if "configuration" in report:
        report["configuration"]["render"] = render
        report["configuration"]["requested_top_k"] = top_k
        report["configuration"]["rendered_chart_count"] = sum(
            chart["image"] is not None for chart in report["ranked_charts"]
        )
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (destination / "execution.json").write_text(
        json.dumps(report["agent"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    summary = markdown_summary(report) if "profile" in report else "# VizJudge Agent report\n"
    summary += f"\n## Agent execution\n\n- Status: {status}\n- Stop reason: {stop_reason}\n"
    for chart in report["ranked_charts"]:
        review = chart.get("judgment")
        if review:
            summary += f"\n- {chart['title']}: {review['confidence']['level']} confidence. "
            summary += f"{review['confidence']['basis']} Next: {review['next_step']}\n"
    (destination / "summary.md").write_text(summary, encoding="utf-8")
    if failure:
        raise ValueError(
            f"Agent failed; execution record saved to {destination}: {stop_reason}"
        ) from failure
    return report
