# Design

VizJudge separates four decisions that automatic EDA tools often collapse:

```text
dataset -> profile/types -> candidate chart specs -> value scoring
        -> ranked evidence -> observations -> ML recommendations
```

## Design principles

- **Rank before rendering:** candidate specifications are cheap; only top candidates become
  image files.
- **Explain every score:** each score includes human-readable reasons and observations.
- **Separate evidence from advice:** chart-level observations feed data-quality, feature, and
  modeling recommendation modules.
- **Local first:** no dataset content leaves the process.
- **Agent friendly:** the report is a JSON-serializable object exposed consistently through
  Python, CLI, and MCP.

## Public interfaces

- `vizjudge.core.report.analyze_dataset`: canonical Python API.
- `vizjudge analyze`: human and automation-friendly CLI.
- `vizjudge-mcp`: stdio MCP server with an `analyze_dataset` tool.
- `skills/vizjudge/SKILL.md`: reusable agent workflow.

The first release evaluates tabular data. Chart-image understanding is deliberately left for
a later phase because it requires a separate vision/OCR evidence model.

