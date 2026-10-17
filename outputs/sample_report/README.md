# Sample report: Breast Cancer Wisconsin

Real, unedited VizJudge output (except that absolute paths were rewritten as
repository-relative ones). Reproduce it from the repository root:

```bash
uv run vizjudge agent examples/breast_cancer/breast_cancer.csv --target diagnosis --task classification --seed 42 --output outputs/sample_report
```

| File | Contents |
| --- | --- |
| [`summary.md`](summary.md) | Ranked charts, evidence, confidence labels and next-step recommendations |
| [`report.json`](report.json) | Full structured report (schema 0.4): profile, roles, all 60 scored candidates, fold-level validation |
| [`execution.json`](execution.json) | Agent trace: tool calls, budgets, iterations and stop reason |
| `01-…06-*.png` | The six charts VizJudge chose to render, in rank order |

No LLM Judge was used; ranking comes from the deterministic heuristic scores plus
5-fold seeded incremental predictive validation. Timestamps and timings differ between
runs; chart choices, scores and fold gains are reproducible for the same seed and
dependency versions (see [`uv.lock`](../../uv.lock)).
