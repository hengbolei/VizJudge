---
name: vizjudge
description: Rank candidate visualizations for tabular machine-learning work and turn the strongest chart evidence into data-quality, feature-engineering, and modeling actions. Use when an agent must decide which EDA charts are worth attention rather than generate a broad automatic EDA report. Do not use for interpreting an existing chart image without its source data.
---

# VizJudge

Use VizJudge to narrow a tabular dataset into a small, evidence-backed set of visual findings.

## Workflow

1. Confirm the local dataset path and identify the target when the user has provided one. Never
   invent a target; an untargeted run is valid.
2. Prefer MCP `run_agent` for the v0.4 plan/analyze/judge/refine workflow. Set explicit
   budgets and task/role overrides when needed. Otherwise run:
   `vizjudge agent <path> [--target <column>] --output <directory>`.
   MCP `analyze_dataset` and CLI `analyze` remain available for direct analysis.
3. Inspect `report.json` or the returned structured result. Lead with the highest-scoring charts,
   their observations, and the recommendation that each observation changes.
4. Check `agent.status` before treating a partial result as complete. Inspect `execution.json`
   for budgets, skips, iteration changes, and tool failures. Distinguish statistical-only,
   recorded/offline, and live LLM review; never claim a live model was called for a replay.
5. Distinguish statistical association from causal or out-of-sample evidence. Call out small
   samples, leakage risks, missingness, and high cardinality when the report surfaces them.
6. Report evidence IDs, confidence basis, and a concrete next validation step, such as a
   leakage-safe split, a transform inside
   cross-validation, or incremental predictive testing. Do not claim model improvement until it
   is measured.

Treat chart scores as transparent prioritization heuristics, not ground truth. Ask before reading
data outside the user's authorized scope or writing output to a location they did not select.
Only invoke an external Judge endpoint when the user has authorized that endpoint and sending
schema names and aggregate evidence to it. Raw rows are excluded from the Judge payload.
