"""Does ranking features the way VizJudge ranks charts produce better downstream models?

Protocol per (dataset, seed):
  1. Seeded subsample (``--max-rows``), then a stratified/shuffled train/test split.
  2. Every selector sees only the training split and returns its top-K features.
  3. HistGradientBoosting and a linear model (logistic / ridge) are fit on the
     training split restricted to those K features and scored on the untouched test split.

Usage:
    python benchmarks/run.py --quick        # ~5 minutes, 2 seeds, 2k-row cap
    python benchmarks/run.py                # full: 5 seeds, 20k-row cap
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import datasets as benchmark_datasets  # noqa: E402
from methods import methods as build_methods  # noqa: E402

PRIMARY = {"classification": "roc_auc", "regression": "r2"}
MODELS = ("hgb", "linear")


def _prepare(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = frame[features].copy()
    categorical = [name for name in features if not pd.api.types.is_numeric_dtype(data[name])]
    for name in categorical:
        data[name] = data[name].fillna("__missing__").astype(str)
    numeric = [name for name in features if name not in categorical]
    return data, numeric, categorical


def _model(kind: str, task: str, numeric: list[str], categorical: list[str], seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    if kind == "hgb":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=250
        )
        columns = ColumnTransformer(
            [("cat", encoder, categorical), ("num", "passthrough", numeric)]
        )
        mask = [True] * len(categorical) + [False] * len(numeric)
        estimator = (
            HistGradientBoostingClassifier
            if task == "classification"
            else HistGradientBoostingRegressor
        )(categorical_features=mask, random_state=seed)
    else:
        columns = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                      sparse_output=False), categorical),
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                      StandardScaler()), numeric),
            ]
        )  # fmt: skip
        estimator = (
            LogisticRegression(max_iter=2000) if task == "classification" else Ridge(alpha=1.0)
        )
    return make_pipeline(columns, estimator)


def evaluate(
    train: pd.DataFrame, test: pd.DataFrame, target: str, task: str, features: list[str],
    kind: str, seed: int,
) -> dict[str, float]:  # fmt: skip
    from sklearn import metrics
    from sklearn.dummy import DummyClassifier, DummyRegressor

    y_train, y_test = train[target], test[target]
    if features:
        x_train, numeric, categorical = _prepare(train, features)
        x_test, _, _ = _prepare(test, features)
        model = _model(kind, task, numeric, categorical, seed)
    else:
        x_train, x_test = np.zeros((len(train), 1)), np.zeros((len(test), 1))
        model = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        if task == "classification":
            proba = model.predict_proba(x_test)
            classes = list(model.classes_)
            if len(classes) == 2:
                auc = metrics.roc_auc_score(y_test == classes[1], proba[:, 1])
            else:
                auc = metrics.roc_auc_score(y_test, proba, multi_class="ovr", labels=classes)
            return {
                "roc_auc": float(auc),
                "balanced_accuracy": float(metrics.balanced_accuracy_score(y_test, prediction)),
            }
    return {
        "r2": float(metrics.r2_score(y_test, prediction)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, prediction))),
    }


def run_one(name: str, seed: int, args: argparse.Namespace) -> list[dict]:
    from sklearn.model_selection import train_test_split

    frame, target, task = benchmark_datasets.load(name)
    if len(frame) > args.max_rows:
        frame = frame.sample(args.max_rows, random_state=seed).reset_index(drop=True)
    train, test = train_test_split(
        frame, test_size=args.test_size, random_state=seed,
        stratify=frame[target] if task == "classification" else None,
    )  # fmt: skip
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    available = [column for column in frame.columns if column != target]
    selections: dict[tuple[str, int], tuple[list[str], float, str]] = {}
    for method, select in build_methods(args.random_draws).items():
        if args.methods and method.split("#")[0] not in args.methods:
            continue
        started = time.perf_counter()
        try:
            chosen = select(train, target, task, args.ks, seed)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - one broken selector must not stop the run
            chosen, status = {k: None for k in args.ks}, f"error: {type(exc).__name__}: {exc}"
            if args.verbose:
                traceback.print_exc()
        elapsed = time.perf_counter() - started
        for k in args.ks:
            selections[(method, k)] = (chosen[k], elapsed, status)
    for k in args.ks:
        selections[("all_features", k)] = (available, 0.0, "ok")

    rows, cache = [], {}
    for (method, k), (features, elapsed, status) in selections.items():
        for kind in MODELS:
            scores: dict[str, float] = {}
            if features is not None:
                key = (frozenset(features), kind)
                if key not in cache:
                    cache[key] = evaluate(train, test, target, task, features, kind, seed)
                scores = cache[key]
            rows.append({
                "dataset": name, "task": task, "seed": seed, "method": method, "k": k,
                "model": kind, "n_available": len(available),
                "n_selected": len(features) if features is not None else 0,
                "saturated": k >= len(available), "selection_seconds": round(elapsed, 4),
                "status": status, "features": ";".join(features or []), **scores,
            })  # fmt: skip
    print(f"  done {name} seed={seed} rows={len(frame)}", flush=True)
    return rows


def summarize(raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    ok = raw[raw["status"] == "ok"].copy()
    ok["method"] = ok["method"].str.split("#").str[0]
    metric_cols = [c for c in ("roc_auc", "balanced_accuracy", "r2", "rmse") if c in ok]
    keys = ["dataset", "task", "k", "model", "method"]
    # Random draws are averaged within a seed first; std is then across seeds for every method.
    per_seed = ok.groupby([*keys, "seed"], as_index=False).agg(
        {**{c: "mean" for c in metric_cols}, "n_selected": "mean", "selection_seconds": "mean",
         "saturated": "first", "n_available": "first"}
    )  # fmt: skip
    agg = per_seed.groupby(keys, as_index=False).agg(
        **{f"{c}_{s}": (c, s) for c in metric_cols for s in ("mean", "std")},
        n_seeds=("seed", "nunique"), n_selected=("n_selected", "mean"),
        selection_seconds=("selection_seconds", "mean"), saturated=("saturated", "first"),
        n_available=("n_available", "first"),
    )  # fmt: skip
    agg["primary"] = agg["task"].map(PRIMARY)
    for stat in ("mean", "std"):
        agg[f"primary_{stat}"] = [
            agg.at[index, f"{metric}_{stat}"] for index, metric in agg["primary"].items()
        ]
    return agg, _markdown(agg, raw, args)


def _markdown(agg: pd.DataFrame, raw: pd.DataFrame, args: argparse.Namespace) -> str:
    selectors = [m for m in ("vizjudge", "vizjudge_heuristic", "mutual_info", "ppscore", "dabl",
                             "autoviz", "random") if m in set(agg["method"])]  # fmt: skip
    lines = [
        f"# VizJudge selection benchmark ({args.mode})",
        "",
        f"- Seeds: {args.seeds}; row cap: {args.max_rows}; test fraction: {args.test_size}; "
        f"random draws per seed: {args.random_draws}; K: {', '.join(map(str, args.ks))}",
        "- Primary metric: ROC-AUC (classification) / R² (regression) on the held-out test "
        "split, mean ± std across seeds. Higher is better.",
        "- `saturated` cells (K ≥ available features) are excluded from ranks and win counts: "
        "every selector picks all features there.",
        "",
    ]
    failures = raw[raw["status"] != "ok"].drop_duplicates(["dataset", "seed", "method"])
    if len(failures):
        lines += ["## Selector failures", ""]
        for row in failures.itertuples():
            lines.append(f"- {row.dataset} seed {row.seed} `{row.method}`: {row.status[:160]}")
        lines.append("")
    live = agg[~agg["saturated"] & agg["method"].isin(selectors)].copy()
    cell = ["dataset", "k", "model"]
    # Ranks are only comparable on cells where every selector produced a result.
    complete = live[live.groupby(cell)["method"].transform("nunique") == len(selectors)].copy()
    complete["rank"] = complete.groupby(cell)["primary_mean"].rank(
        ascending=False, method="average"
    )
    lines += ["## Average rank (1 = best) over non-saturated cells where every selector ran", ""]
    lines += ["| model | " + " | ".join(selectors) + " | cells |",
              "|---|" + "---|" * (len(selectors) + 1)]  # fmt: skip
    for model in MODELS:
        part = complete[complete["model"] == model]
        ranks = part.groupby("method")["rank"].mean()
        cells = part.groupby(["dataset", "k"]).ngroups
        lines.append(f"| {model} | " + " | ".join(f"{ranks.get(m, np.nan):.2f}" for m in selectors)
                     + f" | {cells} |")  # fmt: skip
    lines.append("")
    lines += ["## VizJudge head-to-head: wins / ties / losses (|Δ| ≤ 0.002 ties), mean Δ", ""]
    others = [m for m in selectors if m != "vizjudge"]
    lines += ["| model | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
    for model in MODELS:
        pivot = live[live["model"] == model].pivot_table(
            index=["dataset", "k"], columns="method", values="primary_mean"
        )
        cells = []
        for other in others:
            delta = (pivot["vizjudge"] - pivot[other]).dropna()
            cells.append(f"{(delta > 0.002).sum()}/{(delta.abs() <= 0.002).sum()}/"
                         f"{(delta < -0.002).sum()} (Δ {delta.mean():+.3f})")  # fmt: skip
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    columns = [*selectors, "all_features"]
    for model in MODELS:
        lines += [f"## Held-out primary metric, {model}", ""]
        lines += ["| dataset | metric | K | " + " | ".join(columns) + " |",
                  "|---|---|---|" + "---|" * len(columns)]  # fmt: skip
        part = agg[agg["model"] == model]
        for (dataset, k), cell in part.groupby(["dataset", "k"], sort=True):
            values = cell.set_index("method")
            best = values.loc[values.index.isin(selectors), "primary_mean"].max()
            saturated = bool(values["saturated"].iloc[0])
            entries = []
            for method in columns:
                if method not in values.index:
                    entries.append("—")
                    continue
                mean, std = values.loc[method, "primary_mean"], values.loc[method, "primary_std"]
                text = f"{mean:.3f} ± {0 if pd.isna(std) else std:.3f}"
                if method in selectors and not saturated and mean >= best - 1e-12:
                    text = f"**{text}**"
                entries.append(text)
            label = f"{k} (sat.)" if saturated else str(k)
            lines.append(f"| {dataset} | {values['primary'].iloc[0]} | {label} | "
                         + " | ".join(entries) + " |")  # fmt: skip
        lines.append("")
    timing = agg[agg["model"] == "hgb"].groupby("method")["selection_seconds"].mean()
    lines += ["## Mean selection time per dataset split (seconds)", ""]
    lines += [
        "| " + " | ".join(selectors) + " |",
        "|" + "---|" * len(selectors),
        "| " + " | ".join(f"{timing.get(m, np.nan):.2f}" for m in selectors) + " |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="2 seeds, 2k rows, 3 random draws")
    parser.add_argument("--datasets", nargs="*", default=list(benchmark_datasets.DATASETS))
    parser.add_argument("--methods", nargs="*", default=None, help="subset of selectors")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-draws", type=int, default=None)
    parser.add_argument("--ks", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--jobs", type=int, default=1, help="parallel (dataset, seed) workers")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.mode = "quick" if args.quick else "full"
    defaults = {"seeds": (2, 5), "max_rows": (2000, 20000), "random_draws": (3, 10)}
    for name, (quick, full) in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, quick if args.quick else full)
    unknown = set(args.datasets) - set(benchmark_datasets.DATASETS)
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")

    started = time.perf_counter()
    jobs = [(name, seed) for name in args.datasets for seed in range(args.seeds)]
    print(f"Running {len(jobs)} dataset × seed jobs ({args.mode} mode)", flush=True)
    if args.jobs == 1:
        batches = [run_one(name, seed, args) for name, seed in jobs]
    else:
        from joblib import Parallel, delayed

        batches = Parallel(n_jobs=args.jobs)(delayed(run_one)(n, s, args) for n, s in jobs)
    raw = pd.DataFrame([row for batch in batches for row in batch])
    agg, markdown = summarize(raw, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / f"raw_{args.mode}.csv", index=False)
    agg.to_csv(args.out_dir / f"summary_{args.mode}.csv", index=False)
    elapsed = time.perf_counter() - started
    markdown += f"\nTotal runtime: {elapsed:.0f} s.\n"
    (args.out_dir / f"summary_{args.mode}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
