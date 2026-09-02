# VizJudge

**Generate candidate visualizations, judge which charts are worth attention, and turn the
evidence into machine-learning actions.**

VizJudge is not another automatic EDA report generator. Its central question is not
"how many charts can we draw?" but **"which charts are useful, and what do they imply for
data quality, feature engineering, and modeling?"**

> 面向机器学习流程的图表价值判断器：少画图、看重点、给出下一步。

## What the MVP does

1. Loads CSV, JSON/JSONL, or Parquet data.
2. Infers numeric, categorical, datetime, identifier, constant, and free-text roles.
3. Generates compact candidate chart specifications.
4. Scores candidates using signal strength, information content, sample coverage, and
   actionability heuristics.
5. Renders only the highest-value charts.
6. Writes structured observations and ML recommendations to JSON and Markdown.
7. Exposes the same workflow through a CLI and an optional MCP tool.

VizJudge's scores are decision aids, not statistical proof. They intentionally remain
transparent and deterministic in the first release; see [Scoring](docs/scoring.md).

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
charts. Use `--no-render` when only structured analysis is needed.

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

The server exposes `analyze_dataset(path, target, output_dir, top_k, render)`. Only analyze
datasets you are authorized to read; generated files are written to the requested output
directory.

## Agent Skill

The reusable skill lives at [`skills/vizjudge`](skills/vizjudge). Copy that folder into a
compatible agent's skill directory, or invoke it from the repository as `$vizjudge`.

## CLI reference

```text
vizjudge analyze DATASET [--target COLUMN] [--output DIR] [--top-k N] [--no-render]
vizjudge profile DATASET
```

## Project status

This is an alpha-quality, local-first MVP. Planned work includes predictive validation of
chart rankings, task-aware scoring, image/chart ingestion, richer statistical safeguards,
and adapters for agent ecosystems. See the [roadmap](docs/roadmap.md).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## License

[MIT](LICENSE)
