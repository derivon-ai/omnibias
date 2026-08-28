# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Greedy axis-aligned oblivious tree with closed-form Newton leaves (theory 05-04 / 05-05).

For each depth, scan a column subsample and empirical quantiles, score the
closed-form leaf loss at the current depth, keep the minimizer. Temperature
collapse: the gate is ``sigmoid(beta (x[f] - t))``.

Theory 05-05 adds (a) **grouped** source-feature picks so band-token columns
that came from one original feature are not treated as independent, and (b) a
**pairwise** depth-2 warm-start that scores both gates jointly.
"""

from __future__ import annotations

import numpy as np
from omnibias.tab._core.config import SoftTreeConfig
from omnibias.tab._core.forward import gate_activations_arrays, memberships_from_gates
from omnibias.tab._core.leaves import closed_form_leaves, newton_leaf_loss
from omnibias.tab._core.params import TabParams


def _n_cand(n_features: int, colsample: float) -> int:
    return max(1, min(n_features, int(round(float(colsample) * n_features))))


def token_feature_groups(
    n_source: int,
    *,
    n_bins: int,
    concat_raw: bool,
    use_embed: bool,
) -> list[np.ndarray]:
    r"""Column index groups: one group per original (post-preprocess) feature.

    Band tokens occupy ``n_bins + 1`` columns per source feature; optional raw
    scalars are concatenated after the embedding block.
    """
    groups: list[np.ndarray] = []
    if not use_embed:
        return [np.array([j], dtype=np.int64) for j in range(int(n_source))]
    width = int(n_bins) + 1
    raw_off = int(n_source) * width
    for f in range(int(n_source)):
        cols = list(range(f * width, (f + 1) * width))
        if concat_raw:
            cols.append(raw_off + f)
        groups.append(np.asarray(cols, dtype=np.int64))
    return groups


def _score_partial(
    W: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    depth_now: int,
    beta: float,
    leaf_l2: float,
    tree: int,
) -> float:
    G = gate_activations_arrays(
        W[tree : tree + 1, :depth_now], t[tree : tree + 1, :depth_now], X, beta
    )
    P = memberships_from_gates(G, depth_now)
    leaves = closed_form_leaves(P, residual, weight, leaf_l2)
    return float(newton_leaf_loss(P, residual, weight, leaves))


def fit_axis_tree(
    X: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    depth: int,
    beta: float,
    leaf_l2: float,
    n_quantiles: int,
    colsample: float,
    rng: np.random.Generator,
    n_trees: int = 1,
    feature_groups: list[np.ndarray] | None = None,
    pairwise: bool = False,
) -> TabParams:
    r"""Grow ``n_trees`` oblivious axis trees on the Newton residual; ``b0 = 0``."""
    Xv = np.asarray(X, dtype=np.float64)
    rv = np.asarray(residual, dtype=np.float64)
    hv = np.asarray(weight, dtype=np.float64)
    n, d = Xv.shape
    rv = rv.reshape(n, -1)
    hv = hv.reshape(n, -1)
    k = int(rv.shape[1])
    D = int(depth)
    T = int(n_trees)
    W = np.zeros((T, D, d), dtype=np.float64)
    t = np.zeros((T, D), dtype=np.float64)
    qs = np.linspace(1.0 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
    groups = feature_groups
    n_units = len(groups) if groups is not None else d
    n_cand = _n_cand(n_units, colsample)
    use_pair = bool(pairwise) and D == 2
    for m in range(T):
        if use_pair:
            _fill_pairwise(
                W, t, m, Xv, rv, hv, qs, beta, leaf_l2, rng, n_cand, groups
            )
            continue
        for j in range(D):
            units = rng.choice(n_units, size=n_cand, replace=False)
            best_loss = np.inf
            best_f = 0
            best_t = 0.0
            for u in units:
                cols = groups[int(u)] if groups is not None else np.array([int(u)], dtype=np.int64)
                for f in cols:
                    f = int(f)
                    cand_t = np.unique(np.quantile(Xv[:, f], qs))
                    for tv in cand_t:
                        W[m, j, :] = 0.0
                        W[m, j, f] = 1.0
                        t[m, j] = float(tv)
                        loss = _score_partial(
                            W, t, Xv, rv, hv, depth_now=j + 1, beta=beta, leaf_l2=leaf_l2, tree=m
                        )
                        if loss < best_loss:
                            best_loss = loss
                            best_f = f
                            best_t = float(tv)
            W[m, j, :] = 0.0
            W[m, j, best_f] = 1.0
            t[m, j] = best_t
    G = gate_activations_arrays(W, t, Xv, beta)
    P = memberships_from_gates(G, D)
    leaves = closed_form_leaves(P, rv, hv, leaf_l2)
    cfg = SoftTreeConfig(
        n_features=d,
        n_trees=T,
        depth=D,
        split_kind="axis",
        task="regression",
        n_outputs=k,
        beta_final=float(beta),
        leaf_l2=float(leaf_l2),
        seed=0,
    )
    return TabParams(cfg, W, t, leaves, np.zeros(k, dtype=np.float64))


def _fill_pairwise(
    W: np.ndarray,
    t: np.ndarray,
    m: int,
    X: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    qs: np.ndarray,
    beta: float,
    leaf_l2: float,
    rng: np.random.Generator,
    n_cand: int,
    groups: list[np.ndarray] | None,
) -> None:
    r"""Score depth-2 feature pairs jointly (05-05 H5)."""
    d = int(X.shape[1])
    n_units = len(groups) if groups is not None else d
    units = rng.choice(n_units, size=min(n_cand, n_units), replace=False)
    col_sets: list[np.ndarray] = []
    for u in units:
        col_sets.append(groups[int(u)] if groups is not None else np.array([int(u)], dtype=np.int64))
    cols = np.unique(np.concatenate(col_sets)) if col_sets else np.arange(d)
    best_loss = np.inf
    best: tuple[int, int, float, float] = (int(cols[0]), int(cols[min(1, cols.size - 1)]), 0.0, 0.0)
    for i, f0 in enumerate(cols):
        f0 = int(f0)
        q0 = np.unique(np.quantile(X[:, f0], qs))
        for f1 in cols[i + 1 :] if cols.size > 1 else cols:
            f1 = int(f1)
            q1 = np.unique(np.quantile(X[:, f1], qs))
            for t0 in q0:
                for t1 in q1:
                    W[m, :, :] = 0.0
                    W[m, 0, f0] = 1.0
                    W[m, 1, f1] = 1.0
                    t[m, 0] = float(t0)
                    t[m, 1] = float(t1)
                    loss = _score_partial(
                        W, t, X, residual, weight, depth_now=2, beta=beta, leaf_l2=leaf_l2, tree=m
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best = (f0, f1, float(t0), float(t1))
    W[m, :, :] = 0.0
    W[m, 0, best[0]] = 1.0
    W[m, 1, best[1]] = 1.0
    t[m, 0] = best[2]
    t[m, 1] = best[3]


__all__ = ["fit_axis_tree", "token_feature_groups"]
