# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A fair, multi-seed head-to-head harness: omnibias-tab vs gradient boosting (LightGBM).

The empirical-validation gate for ``tab`` is *best-in-class* -- **match or beat a named
classical baseline** (LightGBM) on a fair benchmark (same split, same seeds), never an
asserted claim. This module is the reusable engine behind both:

* the deterministic CPU-smoke example (``docs/examples/tab_validate.py``, wired as a CI
  smoke), and
* the heavier cluster sweep (``packages/omnibias-tab/bench/sweep.py``), whose summary is
  transcribed into ``docs/benchmarks.md``.

Heavy / optional dependencies (``scikit-learn``, ``lightgbm``, ``torch``) are imported
lazily inside the functions that need them, so importing :mod:`omnibias.tab` stays light and
backend-free. Datasets are ``scikit-learn`` built-ins (bundled, offline, deterministic) so
the smoke needs no network; the network-only datasets (``california_housing`` via a cached
download, ``adult`` / ``higgs`` via OpenML) are guarded and skipped when unavailable.

Every metric is reported so **higher is better** (``accuracy`` for classification,
``-rmse`` for regression) -- the benchmark's ``>=`` gate -- alongside secondary metrics
(AUC / log-loss / R^2) for context.

Terminology: ``tab``'s split gate ``sigmoid(beta (w.x - t))`` hardens as ``beta -> inf``
(the feasibility / temperature sense of "collapse"), distinct from the founding
``delta -> 0`` bias collapse. This module trains and *scores* models; it invokes no
collapse limit itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------- #
# Datasets (scikit-learn built-ins are bundled / offline / deterministic).     #
# ---------------------------------------------------------------------------- #

# name -> task; the bundled ones need no network, the rest are guarded.
SMOKE_SUITE: tuple[str, ...] = ("breast_cancer", "diabetes", "wine")
FULL_SUITE: tuple[str, ...] = (
    "breast_cancer",
    "wine",
    "digits",
    "diabetes",
    "california_housing",
    "adult",
    "higgs",
)

# Eight binary public datasets for theory 05-02 G3 (arrangement vs LightGBM).
# Only breast_cancer is guaranteed offline; the rest use OpenML and may skip.
ARRANGEMENT_PUBLIC_SUITE: tuple[str, ...] = (
    "breast_cancer",
    "adult",
    "higgs",
    "banknote",
    "blood_transfusion",
    "ionosphere",
    "sonar",
    "spambase",
)

# Default row caps for the arrangement public suite (None = use all rows).
ARRANGEMENT_PUBLIC_MAX_ROWS: dict[str, int | None] = {
    "breast_cancer": None,
    "adult": 20_000,
    "higgs": 20_000,
    "banknote": None,
    "blood_transfusion": None,
    "ionosphere": None,
    "sonar": None,
    "spambase": None,
}

# OpenML (name_or_id, version) for the arrangement public suite extras.
_OPENML_BINARY: dict[str, tuple[str, int]] = {
    "adult": ("adult", 2),
    "higgs": ("higgs", 1),
    "banknote": ("banknote-authentication", 1),
    "blood_transfusion": ("blood-transfusion-service-center", 1),
    "ionosphere": ("ionosphere", 1),
    "sonar": ("sonar", 1),
    "spambase": ("spambase", 1),
}


@dataclass
class Dataset:
    r"""A loaded tabular dataset."""

    name: str
    X: np.ndarray
    y: np.ndarray
    task: str  # "binary" | "multiclass" | "regression"
    n_outputs: int


def _encode_openml_xy(data: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenML ``data`` / ``target`` to float64 ``X`` and binary ``{0,1}`` ``y``.

    Object / string feature columns are ordinal-encoded (sorted category order).
    Works without pandas (``as_frame=False`` path); OpenML adult v2 already ships
    numeric features under that path.
    """
    raw = np.asarray(data)
    if raw.dtype == object or raw.dtype.kind in "OUS":
        cols: list[np.ndarray] = []
        for j in range(raw.shape[1]):
            col = raw[:, j]
            if col.dtype == object or getattr(col, "dtype", None) is not None and col.dtype.kind in "OUS":
                # Mixed object column: try numeric, else factorize strings.
                try:
                    cols.append(np.asarray(col, dtype=np.float64))
                except (TypeError, ValueError):
                    classes, codes = np.unique(col.astype(str), return_inverse=True)
                    cols.append(codes.astype(np.float64))
            else:
                cols.append(np.asarray(col, dtype=np.float64))
        X = np.column_stack(cols)
    else:
        X = np.asarray(raw, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    yt = np.asarray(target)
    if yt.dtype.kind in "OU" or yt.dtype == object:
        classes, y = np.unique(yt.astype(str), return_inverse=True)
    else:
        classes, y = np.unique(np.asarray(yt, dtype=np.float64), return_inverse=True)
    if classes.size != 2:
        raise ValueError(f"expected binary target, got {classes.size} classes: {classes}")
    return X, y.astype(np.float64)


def _fetch_openml_binary(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch a known OpenML binary set; raise ``RuntimeError`` on failure."""
    if name not in _OPENML_BINARY:
        raise ValueError(f"unknown OpenML binary dataset {name!r}")
    oml_name, version = _OPENML_BINARY[name]
    try:
        from sklearn.datasets import fetch_openml

        # as_frame=False keeps the dependency surface free of pandas; adult v2
        # is already numerically encoded on this path.
        d = fetch_openml(
            oml_name,
            version=version,
            as_frame=False,
            parser="liac-arff",
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not fetch OpenML {name}: {exc}") from exc
    try:
        return _encode_openml_xy(d.data, d.target)
    except Exception as exc:
        raise RuntimeError(f"could not encode OpenML {name}: {exc}") from exc


def load_dataset(name: str, *, max_rows: int | None = None, seed: int = 0) -> Dataset:
    r"""Load a benchmark dataset by name (see :data:`FULL_SUITE` / :data:`ARRANGEMENT_PUBLIC_SUITE`).

    ``max_rows`` optionally subsamples (deterministically, by ``seed``) for a faster loop;
    the network-only datasets raise :class:`RuntimeError` when they cannot be fetched so the
    caller can skip them.
    """
    from sklearn import datasets as skds

    if name == "breast_cancer":
        d = skds.load_breast_cancer()
        X, y, task, k = d.data, d.target, "binary", 1
    elif name == "wine":
        d = skds.load_wine()
        X, y, task, k = d.data, d.target, "multiclass", 3
    elif name == "digits":
        d = skds.load_digits()
        X, y, task, k = d.data, d.target, "multiclass", 10
    elif name == "diabetes":
        d = skds.load_diabetes()
        X, y, task, k = d.data, d.target, "regression", 1
    elif name == "california_housing":
        try:
            d = skds.fetch_california_housing()
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"could not fetch california_housing: {exc}") from exc
        X, y, task, k = d.data, d.target, "regression", 1
    elif name in _OPENML_BINARY:
        X, y = _fetch_openml_binary(name)
        task, k = "binary", 1
    else:
        known = sorted(set(FULL_SUITE) | set(ARRANGEMENT_PUBLIC_SUITE))
        raise ValueError(f"unknown dataset {name!r}; choose from {known}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.default_rng(seed).permutation(X.shape[0])[:max_rows]
        X, y = X[idx], y[idx]
    return Dataset(name=name, X=X, y=y, task=task, n_outputs=k)


# ---------------------------------------------------------------------------- #
# The fair split + preprocessing (fit on train only; scaling helps the gates). #
# ---------------------------------------------------------------------------- #


def train_test_split(
    ds: Dataset, *, seed: int, test_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Deterministic (stratified for classification) train/test split, standardized.

    Standardization is fit on the *train* split only and applied to both -- soft gates
    ``sigmoid(beta (w.x - t))`` need scaled inputs, while LightGBM is scale-invariant, so
    both models see the identical split and neither is advantaged.
    """
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    stratify = ds.y if ds.task != "regression" else None
    Xtr, Xte, ytr, yte = _split(
        ds.X, ds.y, test_size=test_frac, random_state=seed, stratify=stratify
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def train_val_test_split(
    ds: Dataset,
    *,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, np.ndarray]:
    """Stratified 60/20/20 (by default) split with StandardScaler fit on train only.

    Returns keys ``Xtr``, ``ytr``, ``Xva``, ``yva``, ``Xte``, ``yte``.
    """
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    if ds.task == "regression":
        raise ValueError("train_val_test_split is for classification tasks")
    test_frac = 1.0 - float(train_frac) - float(val_frac)
    if test_frac <= 0.0 or train_frac <= 0.0 or val_frac <= 0.0:
        raise ValueError("train_frac, val_frac, and test_frac must all be positive")
    X_rest, Xte, y_rest, yte = _split(
        ds.X,
        ds.y,
        test_size=test_frac,
        random_state=seed,
        stratify=ds.y,
    )
    # val share of the remaining mass
    val_of_rest = float(val_frac) / (float(train_frac) + float(val_frac))
    Xtr, Xva, ytr, yva = _split(
        X_rest,
        y_rest,
        test_size=val_of_rest,
        random_state=seed + 17,
        stratify=y_rest,
    )
    scaler = StandardScaler().fit(Xtr)
    return {
        "Xtr": scaler.transform(Xtr),
        "ytr": np.asarray(ytr, dtype=np.float64),
        "Xva": scaler.transform(Xva),
        "yva": np.asarray(yva, dtype=np.float64),
        "Xte": scaler.transform(Xte),
        "yte": np.asarray(yte, dtype=np.float64),
    }


# ---------------------------------------------------------------------------- #
# Models: omnibias-tab (boosted / joint) and the LightGBM baseline.            #
# ---------------------------------------------------------------------------- #


@dataclass
class TabConfig:
    r"""Knobs for the ``tab`` model in the head-to-head (defaults are the tuned smoke set)."""

    method: str = "boost"  # "boost" (GBM-mirror) | "joint" (exact 2nd-order)
    n_stages: int = 60
    learning_rate: float = 0.3
    depth: int = 2
    trees_per_stage: int = 1
    inner_steps: int = 40
    inner_lr: float = 0.06
    beta_final: float = 8.0
    # joint-mode knobs
    n_trees: int = 64
    optimizer: str = "trust_region"
    max_steps: int = 60


def fit_predict_tab(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    task: str,
    n_outputs: int,
    cfg: TabConfig | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Fit an omnibias-tab model and return ``(predictions, probabilities-or-None)``."""
    from omnibias.tab._core.config import SoftTreeConfig

    c = cfg or TabConfig()
    if c.method == "boost":
        from omnibias.tab.torch.boosting import fit_boosted

        stage_cfg = SoftTreeConfig(
            n_features=Xtr.shape[1],
            n_trees=c.trees_per_stage,
            depth=c.depth,
            task=task,
            n_outputs=n_outputs,
            beta_final=c.beta_final,
            seed=seed,
        )
        model, _ = fit_boosted(
            Xtr, ytr, stage_cfg,
            n_stages=c.n_stages, learning_rate=c.learning_rate,
            inner_steps=c.inner_steps, inner_lr=c.inner_lr,
        )
    elif c.method == "joint":
        import torch
        from omnibias.tab.torch.model import SoftTreeEnsemble
        from omnibias.tab.torch.train import fit_second_order

        joint_cfg = SoftTreeConfig(
            n_features=Xtr.shape[1],
            n_trees=c.n_trees,
            depth=c.depth,
            task=task,
            n_outputs=n_outputs,
            beta_final=c.beta_final,
            seed=seed,
        )
        torch.manual_seed(seed)
        model = SoftTreeEnsemble(joint_cfg)
        fit_second_order(model, Xtr, ytr, optimizer=c.optimizer, steps=c.max_steps)
    else:
        raise ValueError(f"unknown tab method {c.method!r}; choose 'boost' or 'joint'")

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte) if task != "regression" else None
    return pred, prob


def fit_predict_lightgbm(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    task: str,
    n_outputs: int,
    seed: int = 0,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Fit a LightGBM baseline and return ``(predictions, probabilities-or-None)``."""
    import warnings

    import lightgbm as lgb

    common = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=seed,
        verbose=-1,
        n_jobs=1,
        **kwargs,
    )
    # LightGBM's sklearn wrapper emits a benign feature-name UserWarning on numpy input;
    # silence it here so the harness is safe under a warnings-as-errors test config.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if task == "regression":
            est = lgb.LGBMRegressor(**common)
            est.fit(Xtr, ytr)
            return est.predict(Xte), None
        est = lgb.LGBMClassifier(**common)
        est.fit(Xtr, ytr.astype(np.int64))
        pred = est.predict(Xte).astype(np.float64)
        proba = est.predict_proba(Xte)
        prob = proba[:, 1] if task == "binary" else proba
    return pred, prob


# ---------------------------------------------------------------------------- #
# Scoring + multi-seed head-to-head.                                           #
# ---------------------------------------------------------------------------- #


def score_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray | None,
    task: str,
) -> dict[str, float]:
    r"""Metrics for one fit; ``primary`` is higher-is-better (accuracy / ``-rmse``)."""
    yv = np.asarray(y_true, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    if task == "regression":
        rmse = float(np.sqrt(np.mean((np.asarray(pred).reshape(-1) - yv) ** 2)))
        from sklearn.metrics import r2_score

        out["rmse"] = rmse
        out["r2"] = float(r2_score(yv, np.asarray(pred).reshape(-1)))
        out["primary"] = -rmse
        return out
    acc = float(np.mean(np.asarray(pred).reshape(-1) == yv))
    out["accuracy"] = acc
    out["primary"] = acc
    if task == "binary" and prob is not None:
        from sklearn.metrics import log_loss, roc_auc_score

        try:
            out["auc"] = float(roc_auc_score(yv, np.asarray(prob).reshape(-1)))
        except ValueError:  # pragma: no cover - degenerate single-class split
            out["auc"] = float("nan")
        out["logloss"] = float(log_loss(yv, np.clip(np.asarray(prob).reshape(-1), 1e-9, 1 - 1e-9)))
    return out


@dataclass
class HeadToHead:
    r"""Aggregated multi-seed comparison on one dataset."""

    dataset: str
    task: str
    seeds: list[int]
    tab: list[dict[str, float]] = field(default_factory=list)
    lgbm: list[dict[str, float]] = field(default_factory=list)

    def _arr(self, who: list[dict[str, float]], key: str) -> np.ndarray:
        return np.array([m[key] for m in who], dtype=np.float64)

    def mean(self, who: str, key: str = "primary") -> float:
        return float(self._arr(self.tab if who == "tab" else self.lgbm, key).mean())

    def std(self, who: str, key: str = "primary") -> float:
        return float(self._arr(self.tab if who == "tab" else self.lgbm, key).std(ddof=0))

    @property
    def seed_noise(self) -> float:
        r"""The baseline's across-seed std -- the honest tolerance band for ``>=``."""
        return self.std("lgbm")

    @property
    def not_worse(self) -> bool:
        r"""``True`` iff tab's mean primary is within LightGBM's own seed noise (or better)."""
        return self.mean("tab") >= self.mean("lgbm") - self.seed_noise

    def summary(self) -> dict[str, Any]:
        key = "rmse" if self.task == "regression" else "accuracy"
        return {
            "dataset": self.dataset,
            "task": self.task,
            "n_seeds": len(self.seeds),
            "metric": key,
            "tab_mean_primary": self.mean("tab"),
            "tab_std_primary": self.std("tab"),
            "lgbm_mean_primary": self.mean("lgbm"),
            "lgbm_std_primary": self.std("lgbm"),
            "seed_noise": self.seed_noise,
            "not_worse": self.not_worse,
        }


def head_to_head(
    name: str,
    *,
    seeds: list[int] | int = 5,
    tab_cfg: TabConfig | None = None,
    lgbm_kwargs: dict[str, Any] | None = None,
    max_rows: int | None = None,
    on_seed: Callable[[int], None] | None = None,
) -> HeadToHead:
    r"""Run the fair, multi-seed tab-vs-LightGBM comparison on one dataset."""
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    ds = load_dataset(name, max_rows=max_rows, seed=seed_list[0])
    h2h = HeadToHead(dataset=name, task=ds.task, seeds=seed_list)
    for s in seed_list:
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=s)
        tp, tpr = fit_predict_tab(
            Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs, cfg=tab_cfg, seed=s
        )
        lp, lpr = fit_predict_lightgbm(
            Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs, seed=s, **(lgbm_kwargs or {})
        )
        h2h.tab.append(score_predictions(yte, tp, tpr, ds.task))
        h2h.lgbm.append(score_predictions(yte, lp, lpr, ds.task))
        if on_seed is not None:
            on_seed(s)
    return h2h


def run_suite(
    names: tuple[str, ...] = SMOKE_SUITE,
    *,
    seeds: list[int] | int = 5,
    tab_cfg: TabConfig | None = None,
    max_rows: int | None = None,
    skip_unavailable: bool = True,
) -> list[HeadToHead]:
    r"""Run :func:`head_to_head` over a suite; skip network-only datasets when unavailable."""
    results: list[HeadToHead] = []
    for name in names:
        try:
            results.append(
                head_to_head(name, seeds=seeds, tab_cfg=tab_cfg, max_rows=max_rows)
            )
        except RuntimeError:
            if not skip_unavailable:
                raise
    return results


__all__ = [
    "ARRANGEMENT_PUBLIC_MAX_ROWS",
    "ARRANGEMENT_PUBLIC_SUITE",
    "Dataset",
    "FULL_SUITE",
    "HeadToHead",
    "SMOKE_SUITE",
    "TabConfig",
    "fit_predict_lightgbm",
    "fit_predict_tab",
    "head_to_head",
    "load_dataset",
    "run_suite",
    "score_predictions",
    "train_test_split",
    "train_val_test_split",
]
