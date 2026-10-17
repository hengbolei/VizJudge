# VizJudge

**Generate candidate visualizations, judge which charts are worth attention, and turn the
evidence into machine-learning actions.**

VizJudge is not another automatic EDA report generator. Its central question is not
"how many charts can we draw?" but **"which charts are useful, and what do they imply for
data quality, feature engineering, and modeling?"**

> 面向机器学习流程的图表价值判断器：少画图、看重点、给出下一步。

## v0.4: an evidence-driven Agent loop

VizJudge now runs **planning → tool calls → controlled Python analysis → chart
judgment → iterative improvement**. Statistical validation and an optional LLM
Judge prioritize a small chart set, explain confidence, and recommend next steps.
Execution budgets and a trace make the process inspectable and reproducible.

```bash
vizjudge agent data.csv --target outcome --task classification --output outputs/agent
python examples/agent_loop/demo.py --output outputs/demo-v04
```

The demo uses seeded synthetic data and an explicitly simulated offline Judge;
it needs no model credentials. A live LLM can be connected through a configured
JSON Judge endpoint. See [Agent workflow, Judge contract, and budgets](docs/agent.md),
the [demo](examples/agent_loop/README.md), and [release notes](CHANGELOG.md).

## Analysis capabilities

1. Loads CSV, JSON/JSONL, or Parquet data.
2. Infers numeric, categorical, datetime, identifier, constant, and free-text roles.
3. Generates compact candidate chart specifications.
4. Scores candidates using signal strength, coverage, actionability and seeded
   incremental predictive validation, with diversity and redundancy penalties.
5. Renders only the highest-value charts.
6. Writes structured observations and ML recommendations to JSON and Markdown.
7. Exposes direct analysis and the Agent loop through Python, CLI, and MCP.

VizJudge's scores are decision aids, not statistical proof. They intentionally remain
transparent and deterministic; see [Scoring](docs/scoring.md). Confidence labels
describe evidence for prioritization, rather than statistical probabilities.

## Quick start

[`uv`](https://docs.astral.sh/uv/) is the recommended package manager because it provides
fast, reproducible environments and can manage the project, lockfile, tools, and publishing
workflow in one place.

```bash
uv sync --extra dev
uv run vizjudge analyze data.csv --target survived --output outputs/run
```

Traditional `pip` also works:

```bash
python -m venv .venv
python -m pip install -e .
vizjudge analyze data.csv --target survived --output outputs/run
```

Parquet input uses the optional `parquet` extra: `uv sync --extra parquet`.

The output directory contains `report.json`, `summary.md`, and PNG files for the ranked
charts. Agent runs also write `execution.json`. Use `--no-render` for structured
analysis only. `--task regression` or `--task classification` resolves ambiguous
targets; repeat `--role COLUMN=ROLE` for domain-specific role overrides.

## MCP server

The MCP integration targets the stable 2.x Python SDK:

```bash
uv sync --extra mcp
uv run vizjudge-mcp
```

Example client configuration:

```json
{
  "mcpServers": {
    "vizjudge": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/vizjudge", "run", "vizjudge-mcp"]
    }
  }
}
```

The server preserves `analyze_dataset(path, target, output_dir, top_k, render)` and
adds `run_agent`, including task/role controls, predictive validation, a budget
object and optional Judge adapter. Only analyze
datasets you are authorized to read; generated files are written to the requested output
directory.

## Agent Skill

The reusable skill lives at [`skills/vizjudge`](skills/vizjudge). Copy that folder into a
compatible agent's skill directory, or invoke it from the repository as `$vizjudge`.

## CLI reference

```text
vizjudge analyze DATASET [--target COLUMN] [--output DIR] [--top-k N] [--no-render]
vizjudge profile DATASET
vizjudge agent DATASET [--target COLUMN] [--task TASK] [--output DIR]
               [--judge-file FILE | --judge-url URL] [--max-iterations N]
               [--max-tool-calls N] [--max-seconds S] [--python-timeout S]
               [--max-rows N] [--max-candidates N] [--max-rendered-charts N]
vizjudge --version
```

## Project status

This is an alpha-quality, local-first v0.4 implementation. The Agent loop,
predictive checks, evidence-grounded Judge integration and execution budgets are
implemented. A first [downstream-model benchmark](docs/benchmark.md) shows VizJudge's
feature priorities beat random selection but not mutual information or AutoViz for
gradient boosting; Live Judge quality still needs benchmarks, and image/OCR ingestion and
OS-level sandboxing are future work. See the
[roadmap](docs/roadmap.md) and [schema migration guide](docs/report-schema.md).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## License

[MIT](LICENSE)
