# Downstream-model benchmark

**Question.** VizJudge claims that the charts it ranks highest are the ones most useful for
machine learning. If so, a model trained only on the features behind its top-K charts should
beat one trained on features chosen by other EDA tools or at random.

**Short answer (v0.4.0).** Partly. VizJudge's feature order is clearly better than random
(it wins 22 of 26 non-saturated cells with gradient boosting and 23 of 26 with linear models).
It is roughly on par with mutual information, ppscore and dabl for linear models. It is
**behind** mutual information and AutoViz when the downstream model is gradient boosting.
Its own predictive-validation step has almost no effect on the ranking. The current claim
holds against random selection, but the benchmark does not show an advantage over simple
baselines. The causes below are specific and fixable.

Reproduce with `pip install -e ".[benchmark]"` then `python benchmarks/run.py --quick`
(~2–4 min) or `python benchmarks/run.py --jobs 4` (full, ~4 min on 4 workers). The protocol
and selector definitions are in [`benchmarks/README.md`](../benchmarks/README.md). The full
tables are in [`benchmarks/results/summary_full.md`](../benchmarks/results/summary_full.md).

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

## Results (full run: 5 seeds)

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

### Where VizJudge does well

- **Adult, K=3.** Its top 3 (`relationship`, `marital-status`, `education-num`) give the best
  linear ROC-AUC of any selector (0.863 vs 0.826 for MI/AutoViz) and the second-best HGB
  ROC-AUC.
- **Bike sharing, K=3 (linear).** R² is 0.329 vs 0.285 for MI/ppscore/dabl.
- **Breast cancer, abalone, wine, titanic at small K.** VizJudge is level with the best
  selector, within one standard deviation.
- **Against random.** VizJudge's feature order is reliably informative: +0.07–0.08 mean
  ROC-AUC/R² over random.

### Where VizJudge loses

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

## What this suggests for scoring

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
