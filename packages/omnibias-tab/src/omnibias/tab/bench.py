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

# Nine public regression sets for theory 05-03 G4/G5 (>= 6 required; the extra three are
# headroom against an occasional OpenML fetch failure -- run_suite's skip_unavailable=True
# tolerates that already). Every entry was verified, before being added here, to fetch a
# genuinely continuous float64 target (not a binarized classification re-release of the
# same base dataset -- several OpenML "regression-looking" names, e.g. plain ``servo`` /
# ``abalone`` / ``house_16H`` / ``elevators``, default to exactly that trap and were excluded).
_OPENML_REGRESSION: dict[str, tuple[str, int]] = {
    "kin8nm": ("kin8nm", 1),
    "wine_quality": ("wine_quality", 1),
    "energy_efficiency": ("energy_efficiency", 1),
    "auto_mpg": ("autoMpg", 1),
    "forest_fires": ("forest_fires", 1),
    "house_sales": ("house_sales", 3),
    "bank32nh": ("bank32nh", 1),
    "bike_sharing": ("Bike_Sharing_Demand", 2),
    "miami_housing": ("MiamiHousing2016", 1),
}

NOISE_PUBLIC_SUITE: tuple[str, ...] = tuple(_OPENML_REGRESSION.keys())

# Row caps (deterministic subsample, see load_dataset) for the larger public sets, keeping
# a full multi-seed x multi-baseline (LightGBM/CatBoost/RealMLP/TabM) sweep tractable --
# RealMLP/TabM fit cost grows with n in a way LightGBM/CatBoost barely notice.
NOISE_PUBLIC_MAX_ROWS: dict[str, int | None] = {
    "kin8nm": None,  # 8192
    "wine_quality": None,  # 6497
    "energy_efficiency": None,  # 768
    "auto_mpg": None,  # 398
    "forest_fires": None,  # 517
    "house_sales": 8000,  # 21613 rows natively
    "bank32nh": None,  # 8192
    "bike_sharing": 8000,  # 17379 rows natively
    "miami_housing": 8000,  # 13932 rows natively
}


@dataclass
class Dataset:
    r"""A loaded tabular dataset.

    ``f_clean`` / ``s_true`` are populated **only** by the synthetic noise-uncertainty
    generators (:func:`saw_wave_2d`, :func:`mlp_heteroscedastic_20d`; ``None``
    everywhere else).     ``s_true`` is the *true* per-row noise std used to generate
    ``y = f_clean + s_true * N(0, 1)`` -- kept around for theory 05-03 gate G1's
    reference-validity check and the Figure-5-style diagnostic panel. It is never
    available on real data, which is exactly why an independent estimator
    (:func:`fit_predict_catboost_uncertainty`) is needed there.

    ``df_dx`` (``(n, n_features)``) is the clean target's exact gradient, populated only
    by :func:`mlp_heteroscedastic_20d` (whose generator is a frozen ReLU MLP, so its
    gradient is cheap and exact) for theory 05-03 gate G2's
    ``local_target_consistency_loss``. :func:`saw_wave_2d`'s target is piecewise-constant
    (gradient zero almost everywhere), so it is ``None`` there too -- spec section 10
    explains why G2 is not run on it.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    task: str  # "binary" | "multiclass" | "regression"
    n_outputs: int
    f_clean: np.ndarray | None = None
    s_true: np.ndarray | None = None
    df_dx: np.ndarray | None = None


def _encode_openml_X(data: Any) -> np.ndarray:
    """Ordinal-encode object/string feature columns (sorted category order), cast the
    rest to float64, and zero-fill missing values. Shared by the binary-classification
    and regression OpenML loaders below; works without pandas (``as_frame=False``).
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
                    _classes, codes = np.unique(col.astype(str), return_inverse=True)
                    cols.append(codes.astype(np.float64))
            else:
                cols.append(np.asarray(col, dtype=np.float64))
        X = np.column_stack(cols)
    else:
        X = np.asarray(raw, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def _encode_openml_xy(data: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenML ``data`` / ``target`` to float64 ``X`` and binary ``{0,1}`` ``y``.

    OpenML adult v2 already ships numeric features under the ``as_frame=False`` path.
    """
    X = _encode_openml_X(data)
    yt = np.asarray(target)
    if yt.dtype.kind in "OU" or yt.dtype == object:
        classes, y = np.unique(yt.astype(str), return_inverse=True)
    else:
        classes, y = np.unique(np.asarray(yt, dtype=np.float64), return_inverse=True)
    if classes.size != 2:
        raise ValueError(f"expected binary target, got {classes.size} classes: {classes}")
    return X, y.astype(np.float64)


def _encode_openml_regression_xy(data: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenML ``data`` / ``target`` to float64 ``X`` and a **continuous** float64
    ``y`` -- the regression twin of :func:`_encode_openml_xy`, with no class-count check.
    """
    X = _encode_openml_X(data)
    y = np.asarray(target, dtype=np.float64)
    return X, y


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


def _fetch_openml_regression(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch a known OpenML regression set (:data:`NOISE_PUBLIC_SUITE`); raise
    ``RuntimeError`` on failure, matching :func:`_fetch_openml_binary`'s convention.
    """
    if name not in _OPENML_REGRESSION:
        raise ValueError(f"unknown OpenML regression dataset {name!r}")
    oml_name, version = _OPENML_REGRESSION[name]
    try:
        from sklearn.datasets import fetch_openml

        d = fetch_openml(oml_name, version=version, as_frame=False, parser="liac-arff")
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not fetch OpenML {name}: {exc}") from exc
    try:
        return _encode_openml_regression_xy(d.data, d.target)
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
    elif name in _OPENML_REGRESSION:
        X, y = _fetch_openml_regression(name)
        task, k = "regression", 1
    else:
        known = sorted(set(FULL_SUITE) | set(ARRANGEMENT_PUBLIC_SUITE) | set(NOISE_PUBLIC_SUITE))
        raise ValueError(f"unknown dataset {name!r}; choose from {known}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if max_rows is not None and X.shape[0] > max_rows:
        idx = np.random.default_rng(seed).permutation(X.shape[0])[:max_rows]
        X, y = X[idx], y[idx]
    return Dataset(name=name, X=X, y=y, task=task, n_outputs=k)


# ---------------------------------------------------------------------------- #
# Synthetic noise-uncertainty generators (theory 05-03), known s(x) exactly.   #
# ---------------------------------------------------------------------------- #

# Both carry a known f_clean / s_true (Dataset's optional fields) for gate G1's
# reference-validity + skill sub-gates and the Figure-5-style diagnostic panel.
NOISE_UNCERTAINTY_SUITE: tuple[str, ...] = ("saw_wave_2d", "mlp_heteroscedastic_20d")


def saw_wave_f(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    r"""The clean :func:`saw_wave_2d` target, vectorized over arbitrary ``(x1, x2)``.

    ``1`` inside the five wedges (see :func:`saw_wave_2d`), else ``0``. Shared by the
    dataset generator and the Figure-5-style diagnostic panel so the ground-truth
    formula is defined exactly once.
    """
    x1a = np.asarray(x1, dtype=np.float64)
    x2a = np.asarray(x2, dtype=np.float64)
    period = np.floor(x2a / 2.0)
    local = x2a - 2.0 * period  # in [0, 2)
    boundary = np.where(local <= 1.0, local, 2.0 - local)
    return (x1a <= boundary).astype(np.float64)


def saw_wave_s(x2: np.ndarray) -> np.ndarray:
    r"""The :func:`saw_wave_2d` noise std ``s(x2) = x2**6 / 62500`` (known exactly)."""
    x2a = np.asarray(x2, dtype=np.float64)
    return (x2a**6) / 62500.0


def saw_wave_2d(n: int, *, seed: int) -> Dataset:
    r"""Kartashev, Rubachev & Babenko (arXiv:2509.04430) subsection 4.2 / Figure 5.

    ``x1 ~ U[0, 1]`` (horizontal) and ``x2 ~ U[0, 10]`` (vertical); the clean target is
    ``1`` inside five right-pointing wedges with ``(x2, x1)`` vertices
    ``(2i, 0), (2i+1, 1), (2i+2, 0)`` for ``i = 0..4`` stacked up the ``x2`` axis, and
    ``0`` outside them. The noise std is ``s(x) = x2**6 / 62500`` -- **known exactly**
    (about ``1e-3`` at ``x2=2``, ``0.25`` at ``x2=5``, ``16`` at ``x2=10``), which is
    what makes ``s_true`` usable as gate G1's reference-validity ranking.

    The paper's own text is internally inconsistent here: subsection 4.2 gives
    ``x2**6/62500`` while Appendix D gives ``x1**6/4`` with the axis names swapped.
    Only the subsection-4.2 form reproduces the paper's Figure 5 (crisp below
    ``x2~3``, degrading through ``5``, hopeless above ``7``), so that is the formula
    used here; this discrepancy is recorded rather than silently resolved.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, size=n)
    x2 = rng.uniform(0.0, 10.0, size=n)
    f_clean = saw_wave_f(x1, x2)
    s_true = saw_wave_s(x2)
    y = f_clean + s_true * rng.standard_normal(n)
    X = np.column_stack([x1, x2])
    return Dataset(
        name="saw_wave_2d", X=X, y=y, task="regression", n_outputs=1,
        f_clean=f_clean, s_true=s_true,
    )


def _frozen_relu_mlp(
    widths: tuple[int, ...], rng: np.random.Generator
) -> Callable[[np.ndarray], np.ndarray]:
    r"""A He-initialised, **frozen** (never trained) ReLU MLP, ``(n, widths[0]) -> (n,)``.

    The returned callable also carries a ``.grad`` attribute: ``forward.grad(x)`` is the
    exact Jacobian ``d forward(x)/dx``, ``(n, widths[0])``, computed by the plain chain
    rule (piecewise-constant, since every hidden layer is ``ReLU``) rather than autodiff --
    this module has no torch dependency. Used by :func:`mlp_heteroscedastic_20d` to expose
    a known-exact ``df/dx`` for theory 05-03 gate G2.
    """
    layers = [
        (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in), np.zeros(fan_out))
        for fan_in, fan_out in zip(widths[:-1], widths[1:], strict=False)
    ]

    def forward(x: np.ndarray) -> np.ndarray:
        h = x
        for idx, (W, b) in enumerate(layers):
            h = h @ W + b
            if idx < len(layers) - 1:
                h = np.maximum(h, 0.0)
        out: np.ndarray = h[:, 0]
        return out

    def grad(x: np.ndarray) -> np.ndarray:
        r"""Exact ``d forward(x)/dx``, ``(n, widths[0])``, reusing ``forward``'s own
        pre-activation masks (a second forward pass, then one reverse chain-rule pass --
        cheap relative to the CatBoost / omnibias fits this feeds into)."""
        h = x
        masks: list[np.ndarray] = []
        for idx, (W, b) in enumerate(layers):
            h = h @ W + b
            if idx < len(layers) - 1:
                masks.append((h > 0.0).astype(np.float64))
                h = np.maximum(h, 0.0)
        g = np.ones((x.shape[0], 1), dtype=np.float64)  # d(out)/d(h_last); out = h_last[:, 0]
        for idx in range(len(layers) - 1, -1, -1):
            W, _b = layers[idx]
            if idx < len(layers) - 1:
                g = g * masks[idx]
            g = g @ W.T
        return g

    forward.grad = grad  # type: ignore[attr-defined]
    return forward


def mlp_heteroscedastic_20d(n: int, *, seed: int, d: int = 20) -> Dataset:
    r"""Kartashev et al. subsection 4.1 / Appendix F: 20-D Gaussian features, target and
    log-noise both randomly-initialised, frozen ReLU MLPs.

    ``x ~ N(0, I_d)``; ``f`` is a 3-layer ReLU MLP (``d -> 32 -> 16 -> 1``), ``g`` a
    2-layer ReLU MLP (``d -> 16 -> 1``), both He-initialised and frozen at
    construction (never trained). ``s(x) = e^{g(x)}``, ``y = f(x) + s(x) N(0, 1)``.
    Higher-dimensional and not plottable, so this dataset guards gate G1 against the
    conclusion being an artifact of :func:`saw_wave_2d`'s hand-drawn 2-D geometry.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    f_fn = _frozen_relu_mlp((d, 32, 16, 1), rng)
    g_fn = _frozen_relu_mlp((d, 16, 1), rng)
    f_clean = f_fn(X)
    s_true = np.exp(g_fn(X))
    y = f_clean + s_true * rng.standard_normal(n)
    df_dx = f_fn.grad(X)  # type: ignore[attr-defined]
    return Dataset(
        name="mlp_heteroscedastic_20d", X=X, y=y, task="regression", n_outputs=1,
        f_clean=f_clean, s_true=s_true, df_dx=df_dx,
    )


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
    """Stratified (classification) / plain-random (regression) 60/20/20 split, scaled.

    Returns keys ``Xtr``, ``ytr``, ``Xva``, ``yva``, ``Xte``, ``yte``. ``StandardScaler``
    is fit on the train split only. Regression has no class labels to stratify by, so
    the split is plain random for ``ds.task == "regression"`` and stratified otherwise.
    """
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    test_frac = 1.0 - float(train_frac) - float(val_frac)
    if test_frac <= 0.0 or train_frac <= 0.0 or val_frac <= 0.0:
        raise ValueError("train_frac, val_frac, and test_frac must all be positive")
    is_regression = ds.task == "regression"
    X_rest, Xte, y_rest, yte = _split(
        ds.X,
        ds.y,
        test_size=test_frac,
        random_state=seed,
        stratify=None if is_regression else ds.y,
    )
    # val share of the remaining mass
    val_of_rest = float(val_frac) / (float(train_frac) + float(val_frac))
    Xtr, Xva, ytr, yva = _split(
        X_rest,
        y_rest,
        test_size=val_of_rest,
        random_state=seed + 17,
        stratify=None if is_regression else y_rest,
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


def fit_predict_catboost_uncertainty(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int = 0,
    iterations: int = 500,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The **frozen, independent** noise estimator theory 05-03 section 4(c) requires.

    Fits ``CatBoostRegressor(loss_function="RMSEWithUncertainty")`` on ``Xtr``/``ytr``
    only, then returns ``(mean_pred, log_scale_pred)`` on ``Xte``. Under this loss,
    ``CatBoost.predict`` returns a raw ``(n, 2)`` array whose columns are
    ``(mean, w)`` with variance ``s**2 = e**w``, so ``log_scale = w / 2``.

    This must be the noise estimate passed to ``fit_noise_aware(..., log_scale=...)``
    and to gate G1's decile ranking -- **never** an ``omnibias.tab`` model's own
    output, or the anti-circularity the gate depends on is lost (section 4(c) / 10).
    The estimate is ``GuaranteeKind.MODEL_BASED`` (``omnibias.core.uncertainty``), not
    a sound enclosure, and must never be passed to ``certify_tab``.
    """
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        iterations=iterations,
        random_seed=seed,
        verbose=False,
        **kwargs,
    )
    model.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64))
    raw = np.asarray(model.predict(np.asarray(Xte, dtype=np.float64)), dtype=np.float64)
    mean_pred = raw[:, 0]
    log_scale_pred = raw[:, 1] / 2.0
    return mean_pred, log_scale_pred


# ---------------------------------------------------------------------------- #
# Tuned baselines for theory 05-03 G4 (public-suite honesty report): plain     #
# CatBoost (not the uncertainty head above) plus RealMLP / TabM via pytabkit.  #
# ---------------------------------------------------------------------------- #


def fit_predict_catboost(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    task: str,
    n_outputs: int,
    seed: int = 0,
    iterations: int = 500,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""A tuned, plain CatBoost baseline (RMSE / Logloss) -- the fourth named baseline in
    theory 05-03 gate G4, alongside LightGBM. Distinct from
    :func:`fit_predict_catboost_uncertainty`: this fits ``CatBoostRegressor`` /
    ``CatBoostClassifier`` under CatBoost's ordinary loss, not
    ``RMSEWithUncertainty`` -- there is no ``log_scale`` output here, only a point
    prediction, matching :func:`fit_predict_lightgbm`'s ``(pred, prob_or_None)`` contract.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor

    common = dict(iterations=iterations, random_seed=seed, verbose=False, **kwargs)
    if task == "regression":
        reg = CatBoostRegressor(**common)
        reg.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64))
        pred = np.asarray(reg.predict(np.asarray(Xte, dtype=np.float64)), dtype=np.float64).reshape(-1)
        return pred, None
    clf = CatBoostClassifier(**common)
    clf.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.int64))
    pred = np.asarray(clf.predict(np.asarray(Xte, dtype=np.float64)), dtype=np.float64).reshape(-1)
    proba = np.asarray(clf.predict_proba(np.asarray(Xte, dtype=np.float64)), dtype=np.float64)
    prob = proba[:, 1] if task == "binary" else proba
    return pred, prob


def fit_predict_realmlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int = 0,
    device: str = "cpu",
    n_threads: int = 1,
    **kwargs: Any,
) -> tuple[np.ndarray, None]:
    r"""Tuned RealMLP-TD regression baseline (Holzmuller, Grinsztajn & Steinwart,
    "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data",
    NeurIPS 2024), via ``pytabkit.RealMLP_TD_Regressor`` -- its meta-tuned default
    hyperparameters, no search performed here. **Regression only** (theory 05-03
    section 10; ``pytabkit`` also ships a classifier, out of scope for this spec).
    """
    import warnings

    from pytabkit import RealMLP_TD_Regressor

    model = RealMLP_TD_Regressor(device=device, random_state=seed, n_threads=n_threads, verbosity=0, **kwargs)
    # pytabkit's internals lag the sklearn / torch versions pinned here on two benign
    # points -- the sklearn<1.6 kwarg name ``force_all_finite``, and reading the
    # pre-2.9 ``torch.backends.cuda.matmul.allow_tf32`` getter -- neither an omnibias
    # concern, but this repo's test config runs under filterwarnings=error, so silence
    # both the same way fit_predict_lightgbm silences LightGBM's feature-name warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'force_all_finite' was renamed", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Please use the new API settings", category=UserWarning)
        model.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64).reshape(-1))
        pred = np.asarray(model.predict(np.asarray(Xte, dtype=np.float64)), dtype=np.float64).reshape(-1)
    return pred, None


def fit_predict_tabm(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int = 0,
    device: str = "cpu",
    n_threads: int = 1,
    **kwargs: Any,
) -> tuple[np.ndarray, None]:
    r"""Tuned TabM regression baseline (Gorishniy, Kotelnikov & Babenko, "TabM: Advancing
    Tabular Deep Learning with Parameter-Efficient Ensembling", ICLR 2025), via
    ``pytabkit.TabM_D_Regressor`` -- its default hyperparameters, no search performed
    here. **Regression only** (theory 05-03 section 10). Noticeably slower to fit than
    :func:`fit_predict_realmlp` at comparable ``n`` (TabM trains a parameter-efficient
    ensemble, not a single MLP), which is why the ``--full`` public-suite sweep runs as a
    cluster job (see ``benchmarks/tabular_uncertainty.py``), not on a login node.
    """
    import warnings

    from pytabkit import TabM_D_Regressor

    model = TabM_D_Regressor(device=device, random_state=seed, n_threads=n_threads, verbosity=0, **kwargs)
    with warnings.catch_warnings():
        # See fit_predict_realmlp: same pytabkit/sklearn/torch deprecation shims.
        warnings.filterwarnings("ignore", message="'force_all_finite' was renamed", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Please use the new API settings", category=UserWarning)
        model.fit(np.asarray(Xtr, dtype=np.float64), np.asarray(ytr, dtype=np.float64).reshape(-1))
        pred = np.asarray(model.predict(np.asarray(Xte, dtype=np.float64)), dtype=np.float64).reshape(-1)
    return pred, None


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
    "NOISE_PUBLIC_MAX_ROWS",
    "NOISE_PUBLIC_SUITE",
    "NOISE_UNCERTAINTY_SUITE",
    "SMOKE_SUITE",
    "TabConfig",
    "fit_predict_catboost",
    "fit_predict_catboost_uncertainty",
    "fit_predict_lightgbm",
    "fit_predict_realmlp",
    "fit_predict_tab",
    "fit_predict_tabm",
    "head_to_head",
    "load_dataset",
    "mlp_heteroscedastic_20d",
    "run_suite",
    "saw_wave_2d",
    "saw_wave_f",
    "saw_wave_s",
    "score_predictions",
    "train_test_split",
    "train_val_test_split",
]
