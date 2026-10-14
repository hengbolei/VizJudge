# Reproducible v0.4 demo

From the repository root, after installing the project:

```bash
python examples/agent_loop/demo.py --output outputs/demo-v04
```

This generates 300 seeded synthetic regression rows with a signal, a redundant
copy, a categorical segment and noise. It creates a **simulated offline LLM Judge
response** from stable evidence IDs, runs two review iterations, and narrows six
selected candidates to three charts. The expected trace is:

```text
profile -> analyze -> judge -> refine -> judge -> render
```

Inspect `summary.md`, `report.json`, `execution.json`, and three PNG charts. The
same seed and dependency versions reproduce chart choices, evidence, fold gains,
and simulated judgments. Timestamps, timings, absolute paths and hashes of objects
containing timestamps vary between executions. This is an integration demo, not
a downstream model benchmark or evidence of live LLM quality.

For a real model, implement the [JSON Judge provider contract](../../docs/agent.md)
and use `vizjudge agent ... --judge-url https://your-service/judge`.
