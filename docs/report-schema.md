# Report schema 0.4

Reports are JSON objects. `schema_version` is `"0.4"`; `vizjudge_version` is
the package version (for example `"0.5.0"`; the schema is unchanged since 0.4.0). Existing `source`, `target`, `roles`, `profile`, `candidate_count`,
`ranked_charts`, `recommendations`, and `generated_at` fields remain in complete
reports. Profiles include frequent category values and should be handled as data.

Added complete-report fields:

- `task`: `classification`, `regression`, or null for an untargeted run.
- `role_overrides` and `configuration`: reproducible task/ranking/validation settings.
- `predictive_validation`: completed/skipped status, reason, method, split, metric,
  row/feature limits, sampling, fold settings, warnings, baseline and feature results.
- `warnings`: chart-level warnings for the selected charts.

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
