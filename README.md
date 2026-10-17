# VizJudge

**Generate candidate visualizations, judge which charts are worth attention, and turn the
evidence into machine-learning actions.**

VizJudge is not another automatic EDA report generator. Its central question is not
"how many charts can we draw?" but **"which charts are useful, and what do they imply for
data quality, feature engineering, and modeling?"**

> 面向机器学习流程的图表价值判断器：少画图、看重点、给出下一步。

## A statistical toolset for host agents

VizJudge's core is deterministic statistical judgment: candidate charts, signal
metrics, held-out incremental gain, diversity selection and stable evidence IDs.
Host agents such as Claude Code already have an LLM plus the user's domain context
and code, so VizJudge is built to be **called by them**: the
[skill](skills/vizjudge/SKILL.md) makes the host model the Judge, and fine-grained
MCP tools (`profile_dataset` → `generate_candidates` → `validate_features` →
`select_charts` → `render_charts`) supply evidence it must cite. The host can only
lower statistical priority, never raise it, and turns kept evidence into concrete
changes in the user's project.

For reproducible batch runs and CI, `vizjudge agent` runs the full
**plan → profile → analyze → judge → refine → render** loop deterministically with
execution budgets and a trace:

```bash
vizjudge agent data.csv --target outcome --task classification --output outputs/agent
python examples/agent_loop/demo.py --output outputs/demo-v04
```

The batch loop is statistical-only by default. An LLM Judge is optional: replay a
recorded JSON response, or configure your own HTTP Judge endpoint. The demo uses
seeded synthetic data and a simulated offline Judge; it needs no credentials. See
[Agent workflow, host tools, Judge contract, and budgets](docs/agent.md), the
[demo](examples/agent_loop/README.md), and [release notes](CHANGELOG.md).

## Analysis capabilities

1. Loads CSV, JSON/JSONL, or Parquet data.
2. Infers numeric, categorical, datetime, identifier, constant, and free-text roles.
3. Generates compact candidate chart specifications.
4. Scores candidates using signal strength, coverage, actionability and seeded
   incremental predictive validation, with diversity and redundancy penalties.
5. Renders only the highest-value charts.
6. Writes structured observations and ML recommendations to JSON and Markdown.
7. Exposes fine-grained host-agent tools, direct analysis and the batch Agent loop
   through Python, CLI, and MCP.

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

| Tool | Purpose |
| --- | --- |
| `profile_dataset` | Roles, missingness, duplicates, data-quality actions |
| `generate_candidates` | All candidate charts with scores, warnings, chart and evidence IDs |
| `validate_features` | Cross-validated incremental gain for named features |
| `select_charts` | Diverse, validated selection; accepts host verdicts/priorities that can only lower priority |
| `render_charts` | Render chosen chart IDs within the image budget |
| `analyze_dataset` | One-shot analysis (unchanged, for compatibility) |
| `run_agent` | Full deterministic batch loop with budget and optional Judge adapter |

Every tool accepts a `budget` object (rows, candidates, timeouts, images, input
bytes) and runs in a bounded worker subprocess. Only analyze
datasets you are authorized to read; generated files are written to the requested output
directory.

## Agent Skill

The reusable skill lives at [`skills/vizjudge`](skills/vizjudge). Copy that folder into a
compatible agent's skill directory, or invoke it from the repository as `$vizjudge`.
It makes the host model the Judge: recommended tool order, evidence-citation rules,
a leakage review based on column semantics, and how to turn results into code changes.

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
implemented. Live Judge quality and downstream model outcomes still require
benchmarks; image/OCR ingestion and OS-level sandboxing are future work. See the
[roadmap](docs/roadmap.md) and [schema migration guide](docs/report-schema.md).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## License

[MIT](LICENSE)
