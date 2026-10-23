# Report schema 0.5

Reports are JSON objects. `schema_version` is `"0.5"`; `vizjudge_version` is
the package version. Schema 0.5 only adds fields and changes two enumerated values
(see [Changes in 0.5](#changes-in-05)); a 0.4 reader that tolerates additional fields
keeps working, but must not compare 0.4 and 0.5 gains directly. Existing `source`, `target`, `roles`, `profile`, `candidate_count`,
`ranked_charts`, `recommendations`, and `generated_at` fields remain in complete
reports. Profiles include frequent category values and should be handled as data.

Added complete-report fields:

- `task`: `classification`, `regression`, or null for an untargeted run.
- `role_overrides` and `configuration`: reproducible task/ranking/validation settings.
- `predictive_validation`: completed/skipped status, reason, method, split, metric,
  row/feature limits, sampling, fold settings, warnings, baseline and feature results,
  and `null_test` (permutation p-value of the best feature's gain).
- `signal_assessment`: `status` (`signal`, `no_signal`, `not_assessed`), the
  permutation `p_value` when assessed, and a `message`.
- `leakage_alerts`: every candidate flagged as a leakage suspect, selected or not:
  `chart`, `feature`, `warning`.
- `warnings`: the no-signal message when applicable, then chart-level warnings for
  the selected charts.

Each chart preserves `kind`, `title`, `x`, `y`, `target`, `score`, `reasons`,
`observations`, and `image`; added fields are:

- `feature`: the non-target feature in a targeted chart; otherwise null.
- `metrics`: numeric evidence including sample count, coverage, association, initial
  heuristic score, and predictive score adjustment when available.
- `warnings`: small samples, sparse groups, rendering truncation and leakage checks.
- `predictive`: completed feature validation or an explicit skip; null for untargeted charts.
- `selection_score`, `selection_penalty`, `selection_reasons`: chart ordering evidence.

`score` is the heuristic score adjusted by predictive evidence. `selection_score`
subtracts greedy repetition/redundancy penalties. The returned order follows
selection priority, so raw scores need not descend. `diversity=False` or
`--no-diversity` restores plain score ordering.

Agent reports add `sampling`, `agent`, and chart `judgment`. A judgment includes
cited evidence, keep/defer verdict, confidence level/basis/scope, next step,
statistical and optional LLM priorities, rationale and source iteration. The
`agent` object records execution status, stop reason, plan, budgets, runtime,
dataset hash, traces, and complete iteration judgments.

Partial Agent reports may contain only schema/version/source/ranked_charts and
the `agent` object. Consumers must check `agent.status` and the presence of
`profile` before treating them as a complete analysis. Missing images are null.
Numbers are finite; reports are serialized with `allow_nan=False`.

See [Agent execution](agent.md) for status and budget semantics. Consumers should
check the schema version, read known fields, and tolerate additional fields.

## Changes in 0.5

Scoring changed after the [downstream-model benchmark](benchmark.md); see
[Scoring](scoring.md). Report fields:

- `predictive_validation.method` is `binned_ridge_baseline` (was `ridge_baseline`):
  numeric features are winsorized and quantile-binned on the training fold.
- `predictive_validation.metric` for classification is `roc_auc` (was
  `balanced_accuracy`); macro one-vs-rest for multiclass. Regression stays `rmse`.
- New `predictive_validation.forward_selection` for complete reports:
  `status`, `max_steps`, `note`, and `steps` (each `feature`, `conditional_gain`,
  `score`). `validate_features` (MCP) does not run forward selection.
- Feature results may add `conditional_gain`, `conditional_score` and
  `conditioned_on` (the feature list the gain is conditional on).
- Chart `metrics` may add `eta` (sqrt of `eta_squared`, now the scored value for
  box charts), `information_coefficient`, `relevance`, `held_out_strength` and
  `conditional_gain`. Existing metric names, and therefore evidence IDs
  (`C<hash>.<metric>`), are unchanged.
