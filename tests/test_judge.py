import io
import json

import pandas as pd
import pytest

from vizjudge.agent.judge import build_evidence, http_judge, judge_report
from vizjudge.core.report import analyze_frame


def test_http_adapter_sends_only_grounded_aggregates(monkeypatch):
    report = analyze_frame(
        pd.DataFrame({"secret_category": ["private-A", "private-B"] * 30, "target": [0, 1] * 30}),
        target="target",
        render=False,
    )
    context = build_evidence(report)
    recorded = {}
    response = {
        "reviews": [
            {
                "chart_id": chart["chart_id"],
                "evidence_ids": [chart["evidence"][0]["id"]],
                "priority": 60,
                "verdict": "keep",
                "rationale": "Use the cited coverage.",
                "next_step": "Check generalization.",
            }
            for chart in context["charts"]
        ],
        "refinement": "none",
    }

    class Opener:
        def open(self, request, timeout):
            recorded["body"] = json.loads(request.data)
            recorded["headers"] = request.headers
            return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setenv("VIZJUDGE_JUDGE_TOKEN", "test-token")
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    judgment = judge_report(report, http_judge("https://judge.example.test/evaluate"))
    assert judgment["mode"] == "statistical_and_llm"
    assert recorded["body"]["evidence"] == context
    assert "private-A" not in json.dumps(recorded["body"])
    assert "private-B" not in json.dumps(recorded["body"])
    assert recorded["headers"]["Authorization"] == "Bearer test-token"


@pytest.mark.parametrize(
    "url",
    [
        "file:///data",
        "http://remote.example/judge",
        "http://localhost:80@remote.example/judge",
        "https://user:password@remote.example/judge",
    ],
)
def test_judge_rejects_unsupported_endpoint_urls(url):
    with pytest.raises(ValueError):
        http_judge(url)
