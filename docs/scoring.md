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

Scores are clamped to `[0, 100]`. Missingness reduces coverage but can itself create an
actionable observation. Constant, identifier, and free-text columns are profiled but excluded
from ordinary chart generation.

## Limits

- Association is not causation or guaranteed out-of-sample utility.
- Pearson correlation can miss nonlinear structure.
- High-cardinality categories are summarized and may need domain-aware grouping.
- Small samples can produce unstable relationship scores.
- Target leakage requires semantic and temporal review beyond statistics.

