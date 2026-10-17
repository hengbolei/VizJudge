# Breast-cancer example (runnable, no download)

`breast_cancer.csv` is the Breast Cancer Wisconsin (Diagnostic) dataset: 569 tumours,
30 numeric cell-nucleus measurements and a `diagnosis` target (`malignant` / `benign`).
It is committed here so the example runs offline. Source: UCI Machine Learning
Repository, <https://doi.org/10.24432/C5DW2B>, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); exported from the copy bundled
with scikit-learn by [`export_dataset.py`](export_dataset.py).

From the repository root:

```bash
uv run vizjudge agent examples/breast_cancer/breast_cancer.csv --target diagnosis --task classification --seed 42 --output outputs/breast-cancer
```

The committed result of this command is [`outputs/sample_report`](../../outputs/sample_report).
Expect target box plots for the concave-point and perimeter features (high confidence,
positive held-out gain in every fold) plus skewed `*_error` histograms flagged for
transformation. The 30 features are strongly collinear (radius, perimeter and area),
which makes this a useful check of the redundancy and diversity penalties. It is a
smoke test of the workflow, not a clinical or modeling benchmark.
