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
- `vizjudge.agent.run_agent`, `vizjudge agent`, and MCP `run_agent`: the bounded
  plan/tool/judge/refine loop.
- `skills/vizjudge/SKILL.md`: reusable agent workflow.

Version 0.4 evaluates tabular data with task-aware validation and optional LLM
review of structured evidence. Chart-image understanding remains a later phase
because it requires a separate vision/OCR evidence model. See [Agent execution](agent.md)
for process boundaries, provider integration, iteration and trace semantics.

