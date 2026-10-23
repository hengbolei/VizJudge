# Scoring

Every candidate receives a score from 0 to 100. The score is a transparent heuristic that
combines coverage, information content, relationship strength, and actionability.

## Candidate families

| Family | Evidence measured | Typical ML consequence |
| --- | --- | --- |
| Numeric distribution | spread, skew, missingness | transforms, robust scaling, imputation |
| Category frequency | normalized entropy, rare levels | encoding and rare-level grouping |
| Numeric relationship | \|Pearson r\| + information coefficient | signal, collinearity, leakage review |
| Numeric by class | correlation ratio eta + information coefficient | class separation and nonlinear models |
| Category by class | bias-corrected Cramer's V + information coefficient | categorical signal and leakage review |
| Numeric target by category | eta, information coefficient and held-out RMSE | category effects, encoding |

Scores are clamped to `[0, 100]`. Missingness reduces coverage but can itself create an
actionable observation.

With a target, distribution charts carry no target evidence, so they score only what
makes them actionable: `20 × min(|skew| / 2, 1)` for histograms and `20 × max(rare-level
share, high-cardinality excess)` for category bars, on top of the same coverage/sample
base. A well-behaved noise column therefore scores like a targeted chart with zero
relevance (about 45) and can no longer outrank a feature with measured target signal.
Without a target, the original spread/skew and entropy points apply. Constant, identifier, and free-text columns are profiled but excluded
from ordinary chart generation.

## Target relevance (schema 0.5)

Targeted charts start from `10 + 25 × coverage + 10 × min(1, n / 100)` and add
`55 × relevance`. Relevance is an even blend of two numbers on the |Pearson r| scale:

- **Linear association** (`absolute_correlation`, `eta`, or `cramers_v`). Box charts
  use eta, the square root of the reported `eta_squared`, because eta² is a
  variance share and was systematically smaller than |r| or Cramér's V for the same
  strength of signal; the mix biased mixed-type datasets towards categoricals.
- **Information coefficient** (`information_coefficient`). Mutual information from a
  contingency table (levels for categories, up to 16 quantile bins for continuous
  values, ties kept together), minus the Miller–Madow bias `(kx−1)(ky−1)/2n`, mapped
  to Linfoot's `sqrt(1 − exp(−2·MI))`. It equals |r| for a bivariate normal pair and
  also detects non-monotone and heavy-tailed relationships (for example longitude in
  house prices). An observation is added when it exceeds the linear value by 0.15.

`relevance = 0.5 × linear + 0.5 × information_coefficient` is stored in `metrics`.
When the feature has missing values, `missingness_association` (Cramér's V, or eta for
regression, between "feature is missing" and the target) is also measured, and
relevance becomes the larger of the two. Complete-case metrics are blind to a column
that is only filled after the outcome: it is missing for one class, so eta and the
information coefficient see a single group.

**Leakage.** A targeted chart is a `leakage_suspect` (metric `1`, plus a warning) when
any of `absolute_correlation`, `eta`, `cramers_v`, `information_coefficient`,
or `missingness_association` is at least 0.95, or `held_out_strength` is at least 0.98
(ROC-AUC 0.99; 0.95 would be AUC 0.975, which clean strong features reach). Suspects are selected
ahead of the score ranking (one per leaking signal), listed in the report's
`leakage_alerts`, get the statistical verdict `defer`, and stay in the chart set under
statistical-only refinement until a host review covers them. Sparse groups
(< 5 rows) now cost `15 × share of sparse groups` instead of a flat 15, so one rare
level no longer sinks a feature.

## Limits

- Association is not causation or guaranteed out-of-sample utility.
- The information coefficient is a binned estimate; it is conservative for small
  samples and for features with many levels.
- High-cardinality categories are summarized and may need domain-aware grouping.
- Small samples can produce unstable relationship scores.
- Target leakage requires semantic and temporal review beyond statistics.

## Is there any signal? (permutation null)

Chart scores are relative, so some chart always ranks first. After validation, the
target is shuffled 49 times (seeded) on the same folds and the best feature's mean
held-out gain is recomputed. The p-value is `(1 + #{permuted ≥ observed}) / 50`,
reported in `predictive_validation.null_test`. The report's `signal_assessment` is
`signal` when p ≤ 0.05, `no_signal` otherwise, or `not_assessed` when validation did
not run. With `no_signal`, the report's first warning says the charts are descriptive
only, and every chart's confidence is `low`. On synthetic pure-noise data the false
positive rate is 3–5%. It costs about one extra second at the 5,000-row × 20-feature cap.

## Predictive adjustment and selection (schema 0.5)

Targeted charts can receive a held-out predictive adjustment. With identical seeded
folds, compare a ridge baseline against baseline plus the chart feature. Numeric
features are clipped to training-fold 1st–99th percentiles and expanded into 8
training-quantile bin indicators, so the additive model captures step-shaped,
tree-like effects and outliers cannot dominate it. Classification gain is the change
in threshold-free ROC-AUC (macro one-vs-rest for multiclass); regression gain is RMSE
reduction.

Complete reports then run a greedy forward selection (up to 10 steps, stopping when
no feature improves): each step adds the feature with the largest mean held-out gain
over the baseline plus the features already picked. Every validated feature records
its `conditional_gain` and the `conditioned_on` list. Near-duplicates (for example
`temp` after `feel_temp`) drop to about zero; complementary features keep their value.

Gains are mapped to the relevance scale as `held_out_strength`: ROC-AUC gains become
Somers' D (`2 × gain`); RMSE gains become a signed `sqrt(1 − (rmse_with/rmse_without)²)`,
a held-out partial correlation. With forward selection the marginal and conditional
strengths are averaged. The adjustment moves relevance halfway towards this held-out
value: `0.5 × 55 × (held_out_strength − relevance)`, clamped to `[−15, 15]` points,
and is skipped for features already in the baseline. Raw fold gains are reported
separately. Validation may explicitly skip under sample or feature budgets.

Greedy selection subtracts 12 points for an already represented feature, 20 for
a redundant target feature, and 5 for a repeated chart family. A feature is redundant
with a selected one when numeric correlation is at least 0.95, nonnumeric values match
in at least 95% of 30 complete rows, or the two columns are a one-to-one re-encoding
of each other (at least 99% of complete rows consistent with a level bijection, both
columns with at most 100 levels; for example `education` and `education-num`). The
first two penalties use their maximum; the family penalty is additive. Duplicate
specs are removed. Plain scores remain available alongside `selection_score`,
penalty and reasons.

Groups smaller than 5 observations cost up to 15 points (scaled by their share);
boxplots with over 15 groups lose 10. Categorical associations with over 100 levels are skipped and lose 15.
Rendering truncation and near-perfect target associations produce warnings.

An optional LLM Judge cites these existing metrics. Its combined priority cannot
exceed statistical selection priority, and it cannot raise statistical confidence.
See [validation and confidence semantics](agent.md) for sampling, leakage and
interpretation limits.

