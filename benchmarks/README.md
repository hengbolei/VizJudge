# Downstream-model selection benchmark

Tests VizJudge's core claim: the features behind the charts VizJudge ranks highest should
make better models than features picked by other EDA tools or at random.
Results and interpretation are in [`docs/benchmark.md`](../docs/benchmark.md).

```bash
pip install -e ".[benchmark]"
python benchmarks/run.py --quick          # about 2–3 minutes
python benchmarks/run.py --jobs 4         # full run: 5 seeds, 20k-row cap, 10 random draws
```

Useful flags: `--datasets adult titanic`, `--methods vizjudge mutual_info`, `--seeds`,
`--max-rows`, `--ks 3 5 10`, `--out-dir`, `--verbose` (prints selector tracebacks).

## Protocol

For every dataset and seed:

1. Seeded subsample to the row cap, then a 70/30 train/test split (stratified for
   classification).
2. Every selector sees **only the training split** and returns its top-K features
   (K = 3, 5, 10). The test split is never used for selection.
3. `HistGradientBoosting` and a linear model (logistic regression / ridge with median
   imputation, scaling and one-hot encoding) are fit on the training split restricted to
   those features and scored on the test split: ROC-AUC and balanced accuracy for
   classification, R² and RMSE for regression.
4. `all_features` is reported as a reference ceiling, not ranked.

Cells where K ≥ the number of available features are marked *saturated* and excluded
from ranks and win counts.

## Selectors

| name | what it does |
|---|---|
| `vizjudge` | `analyze_frame(..., validation=True)`; feature order of the target charts in `ranked_charts` |
| `vizjudge_heuristic` | same with `validation=False` (heuristic scores only), an ablation of the ridge check |
| `mutual_info` | scikit-learn mutual information (categoricals as discrete codes) |
| `ppscore` | `ppscore.predictors` order |
| `dabl` | dabl's own plot-ranking criteria (ANOVA F / \|Spearman\| for continuous, discrete MI for categorical) on `dabl.detect_types`, interleaved because dabl ranks the two groups separately |
| `autoviz` | AutoViz `classify_print_vars` with `max_cols_analyzed=K` (XGBoost gain + correlation pruning); may return fewer than K |
| `random` | seeded random order, averaged over draws within each seed |

Compatibility shims, needed for current pandas, are documented in `methods.py`: AutoViz gets
label-encoded categoricals because its own encoder calls a removed pandas API, and dabl
gets integer-valued float columns cast to int because its type detection crashes on them.

## Outputs

- `results/raw_{quick,full}.csv`: one row per dataset × seed × selector × K × model,
  with the selected features.
- `results/summary_{quick,full}.csv`: mean/std across seeds.
- `results/summary_{quick,full}.md`: rank, head-to-head and per-dataset tables.

Datasets are downloaded through scikit-learn/OpenML into `benchmarks/.cache/` (ignored by
git). See `datasets.py` for sources, licences and dropped leakage columns.
