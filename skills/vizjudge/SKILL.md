---
name: vizjudge
description: Decide which EDA charts in a local tabular ML dataset deserve attention, using VizJudge's deterministic statistical evidence with you (the host model) as the Judge, then turn the strongest evidence into concrete data-quality, feature-engineering, leakage and modeling changes in the user's project. Use when an agent must prioritize a few charts and next steps rather than generate a broad automatic EDA report. Do not use for interpreting an existing chart image without its source data.
---

# VizJudge

VizJudge measures; you judge. Its tools return deterministic statistics, held-out
incremental gains and stable evidence IDs. You add what it cannot know: column
semantics, the user's domain, how the data was collected, and the project's code.
Your judgment may only **lower** statistical priority or confidence, never raise it.

## Call order (MCP)

Pass the same `target`, `task`, `role_overrides`, `seed` and `budget.max_rows` to every
call; otherwise chart and evidence IDs will not line up.

1. `profile_dataset(path)` — roles, missingness, duplicates, data-quality actions.
   Choose the target only from what the user said or what the code clearly predicts;
   never invent one (an untargeted run is valid). Fix wrong roles with `role_overrides`
   (e.g. an integer ID column → `identifier`, a numeric code → `categorical`).
2. **Leakage review** (below) on the profile's column names before going further.
3. `generate_candidates(path, target, ...)` — every candidate with heuristic score,
   warnings, `chart_id` and evidence IDs. Use it to see what exists, not what matters.
4. `validate_features(path, target, features=[...])` — cross-validated incremental
   gain over a ridge baseline for up to 20 named columns. Use `baseline_features` for
   columns the user's model already has, so gain is measured *on top of* them.
5. `select_charts(path, target, top_k=...)` — diverse, validated selection with
   statistical verdicts. Then judge (below) and call `select_charts` again with your
   `reviews` and `refinement`.
6. `render_charts(path, chart_ids=[...], output_dir=...)` — render only the final
   selection, in the order you will present it.

`run_agent` (and CLI `vizjudge agent`) runs the whole deterministic loop as a single
reproducible batch; prefer it for CI or when the user wants an audit record rather
than an interactive review. `analyze_dataset` is the one-shot analysis without judging.
Without MCP, run `vizjudge agent <path> [--target <column>] --output <dir>` and read
`report.json`.

## Judging rules

- Cite **exact** evidence IDs from the tool output (e.g. `C1a2b3c4d5e6f.incremental_gain`);
  each review must cite at least one ID belonging to that chart. Never quote a statistic
  that is not in the output, and never compute your own "confidence".
- `priority` (0–100) is combined as `min(statistical_priority, your priority)`. Use it
  to demote, not to promote. Leave a chart unreviewed to keep its statistical priority.
- `verdict` is `keep` or `defer`. Defer when column semantics make the signal suspect
  (leakage, proxy for an ID, post-outcome data), when coverage or sample size is small,
  or when it duplicates a stronger chart. You cannot overturn a statistical `defer`
  (non-positive held-out gain).
- `refinement` is one of `none`, `prefer_validated` (drop deferred/unsupported charts),
  `increase_diversity` (drop repeated or redundant features), `reduce_charts` (halve the
  set). A deferred chart forces at least `prefer_validated`.
- `rationale` and `next_step` are 1–2,000 characters; `next_step` must be a concrete
  action (a split, a transform, a test), not "investigate further".
- `confidence` labels are prioritization labels, not probabilities. High requires ≥100
  complete observations, positive gain in every fold and no chart warnings.

Example second `select_charts` call:

```json
{
  "reviews": [
    {
      "chart_id": "[\"target_box\",\"days_since_last_payment\",null,\"defaulted\"]",
      "evidence_ids": ["C1a2b3c4d5e6f.incremental_gain", "C1a2b3c4d5e6f.positive_folds"],
      "priority": 20,
      "verdict": "defer",
      "rationale": "Large gain, but this column is only populated after a default event.",
      "next_step": "Rebuild the feature from data available at application time and re-validate."
    }
  ],
  "refinement": "prefer_validated"
}
```

(Copy `chart_id` strings verbatim from the output; they are JSON arrays.)

## Leakage review

VizJudge's IID folds cannot detect semantic, temporal or grouped leakage — that is
your job. For every column with a strong target relationship or high gain, ask:

- **Timing:** Is this value known at prediction time? Names with `_after`, `final_`,
  `outcome`, `status`, `closed`, `paid`, `refund`, `days_since_*` (relative to an
  outcome), or approval/decision fields are suspects.
- **Target proxy:** Is it derived from the target, a renamed copy, or computed by the
  process that sets the target (scores, labels, review flags)?
- **Identity / groups:** Does an ID, account, patient, device or session repeat across
  rows? Then random folds leak; recommend `GroupKFold` on that column.
- **Time order:** Is there a date column? Recommend a time-based split, and check the
  feature does not use future aggregates.
- **Collection artifact:** Missingness, rounding or category coding that differs by
  outcome (e.g. a field only filled for churned users).

A "too good" signal (near-perfect gain, every fold positive with tiny `gain_std`) is a
leakage flag first and a feature second. Defer it with a rationale until the user
confirms the column's semantics.

## Turning results into code changes

Read the user's project (training script, feature pipeline, config) before
recommending changes, and propose edits in their code, not generic advice:

- **Data quality:** add the specific imputation, deduplication or type fix the profile
  calls for, inside the existing preprocessing step (fit on training data only).
- **Leaky columns:** remove them from the feature list / column selector, or rebuild them
  from point-in-time data; add a comment citing the evidence ID and the reason.
- **Validated features:** add kept features with measured positive gain; apply skew or
  high-cardinality transforms *inside* the CV pipeline (e.g. a `Pipeline` /
  `ColumnTransformer`), never on the full dataset.
- **Validation scheme:** replace random splits with `GroupKFold` / `TimeSeriesSplit`
  where the leakage review found groups or time order.
- **Confirmation:** after edits, rerun the user's own evaluation (or `validate_features`
  with the new `baseline_features`). Do not claim model improvement until it is measured.

## Reporting

Lead with the few kept charts: what each shows, the evidence IDs behind it, its
confidence basis, and the code change it motivates. List deferred charts briefly with
why. Distinguish association from causal or out-of-sample evidence, and state sampling
(`sampling.sampled`) and budget limits when they apply. For `run_agent`/batch runs,
check `agent.status` before treating a result as complete, and never claim a live model
judged a file replay.

Ask before reading data outside the user's authorized scope or writing outputs to a
location they did not choose. Only use an external HTTP Judge (`run_agent` `judge`
option) when the user has authorized that endpoint to receive schema names and
aggregate evidence; raw rows are never sent.
