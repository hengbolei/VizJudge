# Downstream-model benchmark

**Question.** VizJudge claims that the charts it ranks highest are the ones most useful for
machine learning. If so, a model trained only on the features behind its top-K charts should
beat one trained on features chosen by other EDA tools or at random.

**Short answer (after the schema 0.5 scoring changes).** VizJudge's selection is now
level with mutual information, not clearly better. With linear models it has the best
average rank of the seven selectors (2.76 vs 3.11 for MI) and beats MI in 12 of 26 cells,
loses 8, mean +0.009. With gradient boosting (HGB) it is at parity with MI: average rank
3.24 vs 3.33, mean +0.003, but it loses more cells than it wins (9 wins, 4 ties, 13 losses),
and AutoViz still has the best HGB rank (3.09). The predictive check now changes the
ranking (+0.003 HGB / +0.005 linear over heuristic-only, from about 0). Most of the gain
comes from greedy forward selection. VizJudge still does not show a reliable advantage
over MI for tree models, so its value should not rest on selection quality alone. See
[After the scoring changes](#after-the-scoring-changes).

**Original answer (v0.5.0 scoring).** Partly. VizJudge's feature order was clearly better
than random (it won 22 of 26 non-saturated cells with gradient boosting and 23 of 26 with
linear models). It was roughly on par with mutual information, ppscore and dabl for linear
models, and **behind** mutual information and AutoViz with gradient boosting. Its own
predictive-validation step had almost no effect on the ranking. The causes are listed under
[Suspected causes](#suspected-causes).

Reproduce with `pip install -e ".[benchmark]"` then `python benchmarks/run.py --quick`
(~2–4 min) or `python benchmarks/run.py --jobs 4` (full, ~4 min on 4 workers). The protocol
and selector definitions are in [`benchmarks/README.md`](../benchmarks/README.md). The full
tables are in [`benchmarks/results/summary_full.md`](../benchmarks/results/summary_full.md);
the v0.5.0 tables are kept in [`benchmarks/results/v0.5.0/`](../benchmarks/results/v0.5.0/).

## Setup

- **Datasets (10).** Classification: adult, bank-marketing (`duration` dropped as leakage),
  credit-g, titanic (`boat`, `body`, `name` dropped), breast cancer. Regression: California
  housing, diabetes, red wine quality, abalone, bike sharing. All come from OpenML or
  scikit-learn and are publicly licensed (mostly UCI, CC BY 4.0). They are cached locally and
  not committed.
- **Leakage control.** Seeded 70/30 split; selectors see only the training split; models are
  fit on training rows and scored once on the untouched test split.
- **Selectors.** `vizjudge` (default, with ridge validation), `vizjudge_heuristic`
  (`validation=False`), mutual information, ppscore, dabl's ranking criteria, AutoViz's
  XGBoost selection step, and seeded random (10 draws averaged).
- **Models.** `HistGradientBoosting` (HGB) and logistic regression / ridge.
- **Metrics.** ROC-AUC (classification) and R² (regression) as the primary metrics;
  balanced accuracy and RMSE are also in the CSVs. K = 3, 5, 10; 5 seeds; 20k-row cap.
- **Saturated cells.** These are cells where K ≥ the number of features (for example K=10
  on California housing), so every selector picks everything. They are excluded, which leaves
  26 cells per model.

## Results before the changes (v0.5.0 scoring, full run: 5 seeds)

Average rank among the 7 selectors (1 = best), over the 23 cells where every selector ran.
AutoViz crashes on red wine; see the caveats.

| model | vizjudge | vizjudge_heuristic | mutual_info | ppscore | dabl | autoviz | random |
|---|---|---|---|---|---|---|---|
| HGB | 4.17 | 4.04 | 3.13 | 3.48 | 4.09 | **2.91** | 6.17 |
| linear | 3.52 | 3.30 | 3.00 | 4.87 | 4.09 | **2.96** | 6.26 |

VizJudge head-to-head as wins/ties/losses (|Δ| ≤ 0.002 counts as a tie), plus the mean
difference in the primary metric:

| model | vs heuristic-only | vs mutual_info | vs ppscore | vs dabl | vs autoviz | vs random |
|---|---|---|---|---|---|---|
| HGB | 4/18/4 (−0.001) | 9/1/16 (−0.012) | 7/1/18 (−0.002) | 8/4/14 (+0.005) | 7/2/14 (−0.018) | 22/0/4 (+0.078) |
| linear | 2/22/2 (−0.003) | 11/4/11 (−0.001) | 17/4/5 (+0.016) | 12/9/5 (+0.011) | 8/4/11 (+0.006) | 23/1/2 (+0.070) |

### Where VizJudge did well (v0.5.0)

- **Adult, K=3.** Its top 3 (`relationship`, `marital-status`, `education-num`) give the best
  linear ROC-AUC of any selector (0.863 vs 0.826 for MI/AutoViz) and the second-best HGB
  ROC-AUC.
- **Bike sharing, K=3 (linear).** R² is 0.329 vs 0.285 for MI/ppscore/dabl.
- **Breast cancer, abalone, wine, titanic at small K.** VizJudge is level with the best
  selector, within one standard deviation.
- **Against random.** VizJudge's feature order is reliably informative: +0.07–0.08 mean
  ROC-AUC/R² over random.

### Where VizJudge lost (v0.5.0)

| case | VizJudge | best | gap |
|---|---|---|---|
| California housing, K=3, HGB (R²) | 0.627 | 0.819 (MI, ppscore) | −0.19 |
| Bike sharing, K=5, HGB (R²) | 0.687 | 0.895 (AutoViz) | −0.21 |
| Adult, K=5, HGB (ROC-AUC) | 0.867 | 0.910 (AutoViz) | −0.04 |
| Bank marketing, K=10, HGB (ROC-AUC) | 0.733 | 0.785 (AutoViz); **random 0.763** | −0.05 |
| Credit-g, K=5, HGB (ROC-AUC) | 0.729 | 0.756 (MI) | −0.03 |

## Suspected causes

Each one comes from inspecting the selected features in `raw_full.csv` (seed 0) and the
report metrics.

1. **Association measures sit on different scales.** Numeric-vs-class charts use eta², a
   squared, variance-explained quantity. Categorical-vs-class charts use Cramér's V and
   numeric-vs-numeric charts use |Pearson r|, which are both unsquared. All three are then
   multiplied by the same 55 points. On credit-g, `duration` has eta² = 0.046 (≈ |r| 0.21),
   yet it ranks below `savings_status` (V = 0.18) and `purpose` (V = 0.16). On bank
   marketing, no numeric feature except `pdays` and `campaign` makes VizJudge's top 10, while
   every other selector picks `age`, `balance` or `previous`. That is why VizJudge loses to
   random at K=10 there. Classification datasets with mixed types are biased toward
   categoricals.
2. **Only linear or monotone signal is measured.** Pearson and eta² miss non-monotone or
   heavy-tailed relationships. In California housing, `Longitude` has |r| = 0.04 and is
   ranked 6th, but together with `Latitude` it explains most of the location effect. MI and
   ppscore pick both and score R² 0.82 vs 0.63. In adult, the heavily skewed `capital-gain`
   (eta² 0.05) is picked by MI, ppscore and AutoViz but not by VizJudge at K ≤ 5. This is the
   main argument for adding a nonlinear measure (MI or a small cross-validated tree score).
3. **Scoring is univariate.** Complementary features are invisible to per-chart scores. In
   bike sharing, VizJudge prefers `temp`, `feel_temp` and `humidity`, which are near
   duplicates in signal. AutoViz picks `year`, `workingday` and `hour`, which interact
   strongly. HGB exploits that interaction (R² 0.90 vs 0.69).
4. **Redundancy detection misses re-encodings.** `education` (strings) and `education-num`
   (integers) are the same variable. VizJudge's redundancy check only compares numeric
   correlation or string equality, so both are selected (identical V = 0.364), wasting a slot
   at K=5 in adult.
5. **The predictive check rarely changes the order.** `vizjudge` and `vizjudge_heuristic`
   tie in 18/26 (HGB) and 22/26 (linear) cells, with a mean difference of about 0. The gain
   is measured as balanced accuracy of an argmax ridge classifier. On imbalanced targets
   (credit-g 70/30, bank 88/12) that metric stays flat when a single weak feature is added,
   so many features get a +0.0 adjustment. The bonus is also capped at +15/−10 points, and
   only the first 20 heuristic-ranked features are validated. The ridge check can even
   mislead: California `AveOccup` receives the maximum −10 penalty because outliers hurt
   ridge, while tree models find it useful (AutoViz selects it).

## What this suggested for scoring

- Put every association on one scale, for example sqrt(eta²) or R²-equivalents everywhere,
  before it enters the 0–100 score.
- Add a nonlinear, type-agnostic relevance measure (mutual information, or a
  cross-validated shallow-tree/PPS-style score) alongside Pearson/eta²/Cramér's V.
- Measure predictive gain with a threshold-free metric (ROC-AUC or log-loss) and,
  optionally, a tree baseline, rather than balanced accuracy of an argmax ridge.
- Detect categorical↔numeric re-encodings (one-to-one level mapping) as redundancy.
- Consider conditional (greedy forward) gain so that complementary features and
  interactions are rewarded.

These changes should be re-tested with this benchmark. If VizJudge does not beat mutual
information afterwards, the project's value claim should rest on explanation and workflow,
not on selection quality.

## After the scoring changes

All five suggestions were implemented (details in [Scoring](scoring.md)). Every number
below comes from the same protocol, datasets, seeds and comparison selectors as the
v0.5.0 run. Only VizJudge's code changed. Full run: 5 seeds, 20k-row cap, 201 s on 4
workers. Quick run: 2 seeds, 2k-row cap, 40 s on 4 workers.

1. **Comparable scale.** Box charts score eta = sqrt(eta²), not eta².
2. **Nonlinear relevance.** A bias-corrected, binned mutual-information *information
   coefficient* on the |r| scale is blended 50/50 with the linear measure and reported
   as `information_coefficient` and `relevance`.
3. **Threshold-free validation.** Classification uses ROC-AUC. The ridge baseline
   winsorizes numeric features and adds 8 quantile-bin indicators (an additive,
   tree-like step model). Gains map to the |r| scale, and the adjustment moves
   relevance halfway towards them (±15 points instead of +15/−10).
4. **Re-encodings.** One-to-one level mappings are flagged as redundant (adult's
   `education`/`education-num`).
5. **Conditional gain.** A greedy forward selection over the validated features records
   `conditional_gain`. Scores average marginal and conditional held-out strength.

Two unplanned fixes came out of inspecting the selections. Quantile binning merged
heavy-tie tails: `capital-loss`, which is mostly zeros, got an information coefficient
of 0. The flat −15 sparse-group penalty now scales with the share of sparse groups;
before, one rare `occupation` level cost the whole feature 15 points.

The blend weights were fixed at 0.5/0.5 up front. Before forward selection was added, a
quick-mode sweep of other weights (linear weight 0, 0.5 or 1; predictive weight 0.5 or 1)
kept HGB Δ vs MI between −0.006 and −0.010. No weight was tuned to the benchmark.

### Average rank (1 = best, 23 cells where every selector ran)

| run | model | vizjudge | vizjudge_heuristic | mutual_info | ppscore | dabl | autoviz | random |
|---|---|---|---|---|---|---|---|---|
| full, before | HGB | 4.17 | 4.04 | 3.13 | 3.48 | 4.09 | **2.91** | 6.17 |
| full, after | HGB | 3.24 | 3.50 | 3.33 | 4.07 | 4.57 | **3.09** | 6.22 |
| full, before | linear | 3.52 | 3.30 | 3.00 | 4.87 | 4.09 | **2.96** | 6.26 |
| full, after | linear | **2.76** | 3.59 | 3.11 | 4.98 | 4.13 | 3.09 | 6.35 |
| quick, before | HGB | 4.17 | 4.26 | 3.17 | 4.22 | 3.85 | **2.67** | 5.65 |
| quick, after | HGB | 3.00 | 3.74 | 3.54 | 4.65 | 4.39 | **2.76** | 5.91 |
| quick, before | linear | 3.22 | 3.13 | 3.17 | 4.96 | 4.02 | 3.41 | 6.09 |
| quick, after | linear | **2.48** | 3.09 | 3.37 | 5.09 | 4.30 | 3.46 | 6.22 |

The other selectors' code did not change. Their ranks moved only because VizJudge's
position among them changed.

### Head-to-head, wins/ties/losses over 26 cells (mean Δ in ROC-AUC / R²)

| run | model | vizjudge vs MI | vizjudge vs AutoViz | vizjudge vs random | heuristic-only vs MI | vizjudge vs heuristic-only |
|---|---|---|---|---|---|---|
| full, before | HGB | 9/1/16 (−0.012) | 7/2/14 (−0.018) | 22/0/4 (+0.078) | 9/2/15 (−0.012) | 4/18/4 (−0.001) |
| full, after | HGB | 9/4/13 (+0.003) | 9/2/12 (−0.000) | 23/2/1 (+0.094) | 7/7/12 (+0.000) | 9/11/6 (+0.003) |
| full, before | linear | 11/4/11 (−0.001) | 8/4/11 (+0.006) | 23/1/2 (+0.070) | 12/4/10 (+0.002) | 2/22/2 (−0.003) |
| full, after | linear | 12/6/8 (+0.009) | 10/2/11 (+0.017) | 24/1/1 (+0.080) | 11/6/9 (+0.003) | 8/14/4 (+0.005) |
| quick, before | HGB | 7/3/16 (−0.011) | 7/1/15 (−0.024) | 19/1/6 (+0.073) | 8/1/17 (−0.011) | 3/19/4 (−0.000) |
| quick, after | HGB | 12/5/9 (+0.013) | 9/2/12 (−0.001) | 22/1/3 (+0.097) | 11/3/12 (+0.001) | 7/17/2 (+0.011) |
| quick, before | linear | 11/4/11 (−0.001) | 11/3/9 (+0.009) | 24/1/1 (+0.063) | 12/3/11 (−0.001) | 2/22/2 (−0.000) |
| quick, after | linear | 15/6/5 (+0.015) | 12/3/8 (+0.027) | 25/1/0 (+0.078) | 15/2/9 (+0.007) | 8/16/2 (+0.007) |

AutoViz cells exclude red wine, where AutoViz crashes (23 cells).

`vizjudge_heuristic` (`validation=False`) includes changes 1, 2 and 4 only. Changes 3
and 5 run only with validation. A quick-mode ablation compared the same 24 cells with
and without forward selection; titanic was missing because of a since-fixed crash.

- Without forward selection: HGB 10/2/12 (−0.006) and linear 12/3/9 (+0.001) vs MI.
- With forward selection: HGB 10/5/9 (+0.003) and linear 13/6/5 (+0.009) vs MI.

So most of the improvement over heuristic-only comes from conditional gain.

### The v0.5.0 losing cases

| case | before | after | MI | best other |
|---|---|---|---|---|
| California housing, K=3, HGB (R²) | 0.627 | **0.819** | 0.819 | 0.819 (MI, ppscore) |
| Bike sharing, K=5, HGB (R²) | 0.687 | 0.769 | 0.696 | 0.895 (AutoViz) |
| Adult, K=5, HGB (ROC-AUC) | 0.867 | 0.877 | 0.907 | 0.910 (AutoViz) |
| Bank marketing, K=10, HGB (ROC-AUC) | 0.733 | 0.749 | 0.776 | 0.785 (AutoViz); random 0.763 |
| Credit-g, K=5, HGB (ROC-AUC) | 0.729 | 0.753 | 0.756 | 0.756 (MI) |
| Adult, K=3, linear (ROC-AUC) *(a v0.5.0 win)* | 0.863 | 0.815 | 0.826 | 0.847 (dabl) |

How the selections changed (seed 0, full run):

- **California.** Picks `MedInc`, `Latitude`, `Longitude`.
- **Credit-g.** `duration` ranks 2nd; before, it ranked below `savings_status` and
  `purpose`.
- **Bike sharing.** `temp` drops behind its near-duplicate `feel_temp`, and `year`
  moves up.
- **Adult.** `education` and `education-num` are no longer both in the top 5. The
  linear K=3 win was lost because the top 3 changed from `relationship`,
  `marital-status`, `education-num` to `relationship`, `marital-status`, `age`.

### What still limits VizJudge

- **Interactions.** The validation model is additive, so `hour × workingday` in bike
  sharing is still invisible. AutoViz's XGBoost finds it (0.895 vs 0.769).
- **Chart-level diversity penalties.** The benchmark reads feature order from the
  chart ranking. A target chart loses 12 points when the same feature's distribution
  chart was already selected. Features whose distribution chart has not been shown
  yet therefore jump ahead: adult's `native-country` (relevance 0.10) reaches the top
  5 at full size. This comes from chart-ranking design, not feature scoring, and it
  hurts adult at K=5.
- **Type inference.** On bank marketing, `month` is typed as a date and never scored.
  `campaign` and `previous` are typed categorical (low-cardinality integers), so their
  association is computed over sparse levels. VizJudge still trails random at K=10 there.
- **Noise.** In most cells, the HGB difference from MI is within about one seed
  standard deviation. No significance test was run.

## Caveats

- **Proxy.** VizJudge ranks *charts*, not features. The benchmark takes the feature order of
  the target-relationship charts as a proxy for "what VizJudge prioritizes". Distribution
  charts carry no feature and are ignored.
- **Comparison selectors are adapted.** dabl has no single cross-type ranking, so its
  continuous and categorical rankings are interleaved; this merge is ours. AutoViz 0.1.905
  needed a shim: categoricals are label-encoded first, because its encoder calls a pandas API
  removed in 2.0 and silently falls back to column order. AutoViz also infers red wine
  quality (integers 3–8) as multiclass and then crashes in XGBoost, so it has no wine
  results. AutoViz is an XGBoost-importance selector, which favours it when the downstream
  model is also a tree ensemble.
- **Scope.** 10 datasets, 5 seeds, at most 20k rows, IID splits. Differences under about
  0.01 are within seed noise (see the ± columns), and no significance tests were run. Several
  small datasets saturate at K=10.
- **Tool fairness.** Every selector runs with its default settings; none was tuned.
