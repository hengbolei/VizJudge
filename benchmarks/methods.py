"""Feature-selection methods compared by the benchmark.

Each selector receives only the selection (training) split and returns, for every
requested K, an ordered list of at most K feature names. Selectors that produce a
full ranking are truncated; AutoViz is called once per K because it selects a set.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

Selector = Callable[[pd.DataFrame, str, str, list[int], int], dict[int, list[str]]]


def _features(frame: pd.DataFrame, target: str) -> list[str]:
    return [name for name in frame.columns if name != target]


def _truncate(order: list[str], ks: list[int]) -> dict[int, list[str]]:
    order = list(dict.fromkeys(str(name) for name in order))
    return {k: order[:k] for k in ks}


def _encoded(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ordinal-encode categoricals, median-impute numerics; return matrix and discrete mask."""
    columns, discrete = [], []
    for name in features:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float, na_value=np.nan)
            median = np.nanmedian(values) if np.isfinite(values).any() else 0.0
            columns.append(np.nan_to_num(values, nan=median))
            discrete.append(False)
        else:
            columns.append(pd.factorize(series.fillna("__missing__"))[0].astype(float))
            discrete.append(True)
    return np.column_stack(columns), np.array(discrete)


def vizjudge_selector(validation: bool) -> Selector:
    def select(frame, target, task, ks, seed):
        from vizjudge import analyze_frame

        budget = 2 * len(frame.columns) + 2
        report = analyze_frame(
            frame,
            target=target,
            task=task,
            render=False,
            validation=validation,
            seed=seed % (2**32),
            top_k=budget,
            max_candidates=budget,
        )
        order = [chart["feature"] for chart in report["ranked_charts"] if chart["feature"]]
        return _truncate(order, ks)

    return select


def random_selector(draw: int) -> Selector:
    def select(frame, target, task, ks, seed):
        rng = np.random.default_rng([seed, draw])
        return _truncate(list(rng.permutation(_features(frame, target))), ks)

    return select


def mutual_info(frame, target, task, ks, seed):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    features = _features(frame, target)
    matrix, discrete = _encoded(frame, features)
    if task == "classification":
        scores = mutual_info_classif(
            matrix, frame[target], discrete_features=discrete, random_state=seed
        )
    else:
        scores = mutual_info_regression(
            matrix, frame[target].to_numpy(float), discrete_features=discrete, random_state=seed
        )
    order = [features[index] for index in np.argsort(-scores, kind="stable")]
    return _truncate(order, ks)


def ppscore(frame, target, task, ks, seed):
    import ppscore as pps

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pps.predictors(frame, target, random_seed=seed, sorted=True)
    return _truncate(result["x"].tolist(), ks)


def dabl_style(frame, target, task, ks, seed):
    """dabl's own plot-ranking criteria, merged across feature types.

    dabl ranks continuous features (ANOVA F for classification, |Spearman| for
    regression) and categorical features (discrete mutual information) in separate
    plot groups. It has no single ranking, so the two lists are interleaved,
    continuous first, matching dabl's plot order. Features dabl types as
    ``useless``/``free_string``/``date`` are never shown, so they are never selected.
    """
    import dabl
    from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
    from sklearn.impute import SimpleImputer

    typed = frame.drop(columns=[target])
    for name in typed.columns:
        # dabl 0.3.2 calls np.bincount on integer-valued float columns and crashes.
        values = typed[name]
        if pd.api.types.is_float_dtype(values) and values.notna().all():
            if np.array_equal(values, np.round(values)):
                typed[name] = values.astype("int64")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        types = dabl.detect_types(typed)

    def pick(*kinds: str) -> list[str]:
        return [
            name
            for name in types.index
            if name != target
            and any(kind in types and bool(types.loc[name, kind]) for kind in kinds)
        ]

    continuous = pick("continuous", "dirty_float", "low_card_int_ordinal")
    categorical = [
        name for name in pick("categorical", "low_card_int_categorical") if name not in continuous
    ]
    y = frame[target]
    ranked_cont: list[str] = []
    if continuous:
        numeric = frame[continuous].apply(pd.to_numeric, errors="coerce")
        if task == "classification":
            imputed = SimpleImputer().fit_transform(numeric)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = np.nan_to_num(f_classif(imputed, y)[0], nan=0.0)
        else:
            scores = numeric.corrwith(y, method="spearman").abs().fillna(0.0).to_numpy()
        ranked_cont = [continuous[i] for i in np.argsort(-scores, kind="stable")]
    ranked_cat: list[str] = []
    if categorical:
        codes = np.column_stack(
            [pd.factorize(frame[name].astype(str))[0] for name in categorical]
        ).astype(float)
        mask = np.ones(len(categorical), dtype=bool)
        if task == "classification":
            scores = mutual_info_classif(codes, y, discrete_features=mask, random_state=seed)
        else:
            scores = mutual_info_regression(
                codes, y.to_numpy(float), discrete_features=mask, random_state=seed
            )
        ranked_cat = [categorical[i] for i in np.argsort(-scores, kind="stable")]
    order: list[str] = []
    for index in range(max(len(ranked_cont), len(ranked_cat))):
        order += [group[index] for group in (ranked_cont, ranked_cat) if index < len(group)]
    return _truncate(order, ks)


def autoviz(frame, target, task, ks, seed):
    """AutoViz's own selection step (``classify_print_vars``) with max_cols_analyzed=K.

    AutoViz only selects (XGBoost gain + correlation pruning) when the feature count
    reaches the column limit; below it every usable feature is kept. The XGBoost
    seed is hard-coded inside AutoViz, so ``seed`` only changes the data split.

    Compatibility shim: AutoViz 0.1.905 label-encodes string columns with
    ``DataFrame.dtypes.iteritems`` (removed in pandas 2) inside a bare ``except``,
    so with categoricals its XGBoost step silently falls back to column order.
    Categoricals are therefore label-encoded here first (missing → -1), which is the
    encoding AutoViz applies itself.
    """
    from autoviz.AutoViz_Utils import classify_print_vars

    out: dict[int, list[str]] = {}
    features = set(_features(frame, target))
    data = frame.copy()
    for name in features:
        if not pd.api.types.is_numeric_dtype(data[name]):
            data[name] = pd.factorize(data[name], sort=True)[0].astype("int64")
    if task == "classification":
        # AutoViz detects classification from integer/object targets.
        data[target] = pd.factorize(data[target], sort=True)[0]
    for k in ks:
        log = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(log):
            warnings.simplefilter("ignore")
            result = classify_print_vars(
                data, ",", max(len(data), 1) + 1, k, depVar=target, verbose=0
            )
        if "crashing" in log.getvalue():
            raise RuntimeError("AutoViz XGBoost feature selection crashed")
        selected = [name for name in result[0].columns if name in features]
        out[k] = selected[:k]
    return out


def methods(random_draws: int) -> dict[str, Selector]:
    table: dict[str, Selector] = {
        "vizjudge": vizjudge_selector(validation=True),
        "vizjudge_heuristic": vizjudge_selector(validation=False),
        "mutual_info": mutual_info,
        "ppscore": ppscore,
        "dabl": dabl_style,
        "autoviz": autoviz,
    }
    for draw in range(random_draws):
        table[f"random#{draw}"] = random_selector(draw)
    return table
