import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from vizjudge.agent import ExecutionBudget, run_agent
from vizjudge.agent.judge import build_evidence, judge_report
from vizjudge.agent.runner import BudgetExceeded, _Executor
from vizjudge.agent.worker import execute
from vizjudge.core.report import analyze_frame


def _frame():
    rng = np.random.default_rng(4)
    x = rng.normal(size=120)
    return pd.DataFrame(
        {"signal": x, "noise": rng.normal(size=120), "target": 2 * x + rng.normal(0, 0.1, 120)}
    )


def _response(report):
    return {
        "refinement": "reduce_charts",
        "reviews": [
            {
                "chart_id": chart["chart_id"],
                "evidence_ids": [chart["evidence"][0]["id"]],
                "priority": 75,
                "verdict": "keep",
                "rationale": "Prioritize cited evidence.",
                "next_step": "Validate the retained signal with a domain split.",
            }
            for chart in build_evidence(report)["charts"]
        ],
    }


def test_agent_runs_python_judge_refinement_and_records_provenance(tmp_path):
    data = tmp_path / "data.csv"
    _frame().to_csv(data, index=False)
    initial = analyze_frame(pd.read_csv(data), target="target", top_k=4, render=False)
    fixture = tmp_path / "judge.json"
    fixture.write_text(json.dumps(_response(initial)), encoding="utf-8")
    report = run_agent(
        data,
        target="target",
        top_k=4,
        render=False,
        output_dir=tmp_path / "report",
        judge={"kind": "file", "path": str(fixture)},
    )
    assert report["agent"]["status"] == "completed"
    assert [event["tool"] for event in report["agent"]["trace"]] == [
        "profile",
        "analyze",
        "judge",
        "refine",
        "judge",
    ]
    assert len(report["ranked_charts"]) == 2
    assert report["agent"]["iterations"][0]["judgment"]["mode"] == "statistical_and_llm"
    assert all(len(event["output_sha256"]) == 64 for event in report["agent"]["trace"])
    assert all(chart["judgment"]["llm_source_iteration"] == 1 for chart in report["ranked_charts"])
    assert (tmp_path / "report" / "execution.json").is_file()


def test_budget_stops_and_saves_partial_execution(tmp_path):
    data = tmp_path / "data.csv"
    _frame().to_csv(data, index=False)
    report = run_agent(
        data,
        target="target",
        output_dir=tmp_path / "report",
        render=False,
        budget=ExecutionBudget(max_tool_calls=1),
    )
    assert report["agent"]["status"] == "budget_exhausted"
    assert report["agent"]["tool_calls"] == 1
    assert (tmp_path / "report" / "report.json").is_file()


def test_worker_timeout_is_recorded(monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 0.01)

    monkeypatch.setattr(subprocess, "run", timed_out)
    executor = _Executor(ExecutionBudget(python_timeout_seconds=0.01))
    with pytest.raises(BudgetExceeded):
        executor.call("profile", {}, "Test timeout")
    assert executor.trace[0]["status"] == "timeout"


def test_judge_cannot_invent_evidence_or_raise_confidence():
    report = analyze_frame(_frame(), target="target", render=False)
    response = _response(report)
    response["reviews"][0]["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="existing evidence"):
        judge_report(report, lambda context: response)
    valid = judge_report(report, lambda context: _response(report))
    statistical = judge_report(report)
    assert [review["confidence"] for review in valid["reviews"]] == [
        review["confidence"] for review in statistical["reviews"]
    ]
    context = valid["evidence"]
    assert "top_values" not in json.dumps(context)
    assert all(
        review["combined_priority"] <= review["statistical_priority"] for review in valid["reviews"]
    )


def test_worker_rejects_arbitrary_code():
    with pytest.raises(ValueError, match="not allowed"):
        execute({"tool": "exec", "code": "print('unsafe')"})


def test_row_candidate_and_render_budgets(tmp_path):
    data = tmp_path / "data.csv"
    _frame().to_csv(data, index=False)
    report = run_agent(
        data,
        target="target",
        output_dir=tmp_path / "report",
        top_k=4,
        budget=ExecutionBudget(max_rows=80, max_candidates=2, max_rendered_charts=1),
    )
    assert report["sampling"] == {
        "original_rows": 120,
        "analyzed_rows": 80,
        "sampled": True,
        "seed": 42,
    }
    assert report["candidate_count"] == 2
    assert report["configuration"]["render"] is True
    assert report["configuration"]["rendered_chart_count"] == 1
    assert len(list((tmp_path / "report").glob("*.png"))) == 1
    assert all(chart["target"] == "target" for chart in report["ranked_charts"])


def test_invalid_judge_saves_failure_trace(tmp_path):
    data = tmp_path / "data.csv"
    _frame().to_csv(data, index=False)
    fixture = tmp_path / "judge.json"
    fixture.write_text('{"reviews": []}')
    with pytest.raises(ValueError, match="execution record saved"):
        run_agent(
            data,
            target="target",
            output_dir=tmp_path / "report",
            render=False,
            judge={"kind": "file", "path": str(fixture)},
        )
    trace = json.loads((tmp_path / "report" / "execution.json").read_text())
    assert trace["status"] == "failed"
    assert trace["trace"][-1]["tool"] == "judge"
    assert trace["trace"][-1]["status"] == "failed"


@pytest.mark.parametrize(
    "options",
    [
        {"max_tool_calls": 0},
        {"max_iterations": True},
        {"max_seconds": float("nan")},
        {"max_rendered_charts": -1},
    ],
)
def test_budget_validation(options):
    with pytest.raises(ValueError):
        ExecutionBudget(**options).validate()
