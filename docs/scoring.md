# Scoring

Every candidate receives a score from 0 to 100. The score is a transparent heuristic that
combines coverage, information content, relationship strength, and actionability.

## Candidate families

| Family | Evidence measured | Typical ML consequence |
| --- | --- | --- |
| Numeric distribution | spread, skew, missingness | transforms, robust scaling, imputation |
| Category frequency | normalized entropy, rare levels | encoding and rare-level grouping |
| Numeric relationship | absolute Pearson correlation | signal, collinearity, leakage review |
| Numeric by class | correlation ratio (eta squared) | class separation and nonlinear models |
| Category by class | bias-corrected Cramer's V | categorical signal and leakage review |
| Numeric target by category | eta squared and held-out RMSE | category effects, encoding |

Scores are clamped to `[0, 100]`. Missingness reduces coverage but can itself create an
actionable observation. Constant, identifier, and free-text columns are profiled but excluded
from ordinary chart generation.

## Limits

- Association is not causation or guaranteed out-of-sample utility.
- Pearson correlation can miss nonlinear structure.
- High-cardinality categories are summarized and may need domain-aware grouping.
- Small samples can produce unstable relationship scores.
- Target leakage requires semantic and temporal review beyond statistics.

## v0.4 predictive adjustment and selection

Targeted charts can receive a held-out predictive adjustment. With identical
seeded folds, compare a ridge baseline against baseline plus the chart feature.
Classification gain is balanced accuracy improvement; regression gain is RMSE
reduction, normalized by baseline RMSE for the score adjustment. Multiply this
gain by 30 and clamp the adjustment to `[-10, 15]` score points. Report raw fold
gains separately. Validation may explicitly skip under sample or feature budgets.

Greedy selection subtracts 12 points for an already represented feature, 20 for
a target feature whose numeric correlation is at least 0.95 with a selected
feature (or whose nonnumeric values match in at least 95% of 30 complete rows),
and 5 for a repeated chart family. The first two penalties use their maximum;
the family penalty is additive. Duplicate specs are removed. Plain scores remain
available alongside `selection_score`, penalty and reasons.

Groups smaller than 5 observations lose 15 points; boxplots with over 15 groups
lose 10. Categorical associations with over 100 levels are skipped and lose 15.
Rendering truncation and near-perfect target associations produce warnings.

An optional LLM Judge cites these existing metrics. Its combined priority cannot
exceed statistical selection priority, and it cannot raise statistical confidence.
See [validation and confidence semantics](agent.md) for sampling, leakage and
interpretation limits.

