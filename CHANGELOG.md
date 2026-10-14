# Changelog

## 0.4.0 — 2026-10-14

- Added a deterministic Agent loop: plan, profile, bounded Python analysis,
  statistical/optional LLM Judge, refinement, re-evaluation and final rendering.
- Added Python, CLI `agent`, and MCP `run_agent` interfaces, without removing the
  existing direct analysis APIs.
- Added seeded incremental predictive checks for classification and regression,
  training-fold-only preprocessing, baseline feature controls and fold variation.
- Added chart deduplication, diversity/redundancy penalties, numeric evidence,
  sparse-group/leakage warnings and explainable prioritization confidence.
- Added recorded/HTTP LLM Judge adapters with evidence reference validation,
  constrained improvements and explicit statistical-only/offline/live modes.
- Added execution budgets, subprocess timeouts, dataset/tool hashes, structured
  iteration records and partial reports on budget exhaustion.
- Hardened input/task/role validation, categorical regression rendering and
  recommendations. Preserved and completed the existing uncommitted work.
- Added an offline reproducible demo, regression tests and schema migration docs.

This release rolls validation and task-awareness work into 0.4. Image OCR/vision
ingestion from the earlier 0.3 proposal remains planned; no 0.2 or 0.3 release is
claimed. Report consumers should switch from schema `0.1` to `0.4` and tolerate
new fields and non-descending raw scores when diversity selection is enabled.

## 0.1.0

- Initial transparent tabular MVP: profiling, heuristic chart ranking, selective
  rendering, JSON/Markdown reports, CLI, MCP and an agent skill.
