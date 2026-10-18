# Changelog

## Unreleased

- Workers now cache the loaded, budget-sampled dataset per user (keyed by source
  SHA-256, size, mtime, `max_rows`, `seed` and pandas version; 7-day expiry, 512 MB LRU
  cap, `VIZJUDGE_CACHE_DIR` / `VIZJUDGE_CACHE_MAX_BYTES` / `VIZJUDGE_NO_CACHE`), so a
  fine-grained MCP flow parses the file once instead of five times. Trace events record
  `cache.status`, and workers reject a source whose SHA-256 differs from the one the
  caller recorded.
- `render_charts`, and `run_agent` / `analyze_dataset` when rendering, return the PNGs
  as MCP image content (at most 8, 1 MB each, 4 MB total; `include_images: false` to
  opt out). The structured result is unchanged apart from an additive `image_content`
  summary.
- Timing of the breast_cancer MCP flow (profile → candidates → validate → select →
  render 4 charts, seed 42, cold cache per flow, 5 interleaved runs, Windows,
  Python 3.12): the 125 KB example is unchanged within noise (median 5.12 s before,
  4.77 s after) because each call is dominated by ~0.8 s of worker start-up; parsing it
  takes 10 ms. On the same data replicated to 100,144 rows (28 MB, `max_rows` 10,000)
  a cached load takes 0.03 s instead of 0.62 s per call and the flow median drops from
  9.36 s to 8.41 s (a warm-cache flow: 6.0–6.6 s).
- Renamed helpers that were imported across modules: `Executor`, `refine_report`,
  `render_chart`, `safe_filename` and `markdown_summary` (previously underscored).
- Added `.gitattributes` (`* text=auto eol=lf`, binary PNG/Parquet/pickle).

## 0.5.0 — 2026-10-17

- Made the agent integration distributable: the wheel bundles the skill, and new CLI
  commands `vizjudge skill install`, `vizjudge mcp-config --client dsh|json|claude` and
  `vizjudge doctor` install the skill and print an MCP registration for the current
  interpreter.
- Added a DeepSeek Harness guide (`docs/integrations/deepseek-harness.md`) and a skill
  setup check: find the tools, otherwise detect the CLI, ask before installing, and use
  absolute paths.
- Added a tag-triggered release workflow that attaches the sdist and wheel to a GitHub
  Release, with opt-in PyPI trusted publishing (`PUBLISH_TO_PYPI`), and project URLs.
- Added `uv.lock` and a GitHub Actions CI workflow (Linux/Windows × Python 3.11/3.12,
  plus a wheel build and installed-wheel smoke test).
- Added a runnable offline Breast Cancer Wisconsin example and a committed sample report
  in `outputs/sample_report`; removed the download-only Titanic placeholder.
- README now shows sample output and compares VizJudge with dabl, AutoViz, LIDA and
  ydata-profiling.
- Repositioned VizJudge as a statistical toolset for host agents: the host model
  is the Judge and `vizjudge agent` is the reproducible batch/CI mode.
- Added MCP tools `profile_dataset`, `generate_candidates`, `validate_features`,
  `select_charts` (partial host reviews that can only lower priority) and
  `render_charts`, with shared evidence IDs, budgets and execution traces.
  `analyze_dataset` and `run_agent` are unchanged.
- Rewrote the skill around host judging, leakage review and project code changes.

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
