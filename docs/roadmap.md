# Roadmap

## 0.1 — Transparent tabular MVP

- Deterministic type inference and profiling
- Candidate generation, scoring, selective rendering
- JSON/Markdown report, CLI, MCP tool, and Agent Skill

## Validation and task awareness — included in 0.4

- Cross-validated incremental predictive value
- Regression/classification-specific scoring
- Redundancy and chart-diversity penalties
- Explainable confidence labels, fold variability, and minimum-sample warnings

## 0.4 — Evidence-driven Agent loop (implemented)

- Deterministic plan and allowlisted Python tool execution in timed subprocesses
- Statistical and optional LLM Judge with evidence IDs and constrained refinements
- Re-evaluation, selective final rendering, confidence and concrete next actions
- CLI/MCP/Python interfaces, execution budgets and traceable reports
- Seeded offline demo, regression tests and report schema migration guidance

No separate 0.2 or 0.3 release is claimed; the implementation moves from 0.1 to 0.4.

## Chart understanding — deferred from the original 0.3 proposal

- Accept existing chart images and structured chart metadata
- OCR/vision extraction with evidence provenance
- Detect misleading scales, aggregation artifacts, and visual uncertainty

## 1.0 — Extensible agent component

- Plugin API for chart generators, scorers, and recommendation policies
- Stable report schema and compatibility tests
- Stronger dataset privacy controls and OS-level execution isolation
- Benchmark suite linking chart selection to downstream model outcomes (first version: [benchmark](benchmark.md); next: re-run after
  scoring fixes and add Judge-quality tasks)

