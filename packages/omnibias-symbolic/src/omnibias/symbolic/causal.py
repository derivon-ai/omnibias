# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Causal *parent-ranking* of candidate terms for symbolic discovery.

Sparse regression tells you *which* library terms enter an equation; it does not
tell you which observed quantities **drive** which.  This module adds a light,
numpy-only causal-discovery layer on top of the discovery engine, built from two
honest ingredients:

1. **Mutual-information screening** -- reuses the Miller-Madow-corrected
   estimator :func:`omnibias.symbolic.diagnostics.feature_residual_mutual_information`
   to rank candidate terms by their (nonlinear) dependence on a target.
   MI is *symmetric*, so on its own it gives an undirected relevance ranking.

2. **NOTEARS-lite continuous acyclicity** -- a small implementation of the
   smooth acyclicity functional ``h(W) = tr(e^{W∘W}) - d`` of Zheng et al.
   (NeurIPS 2018) and a proximal-gradient learner of a linear-SEM weighted
   adjacency under an ``ell_1`` + acyclicity penalty.  Reading the weights into a
   node gives a *directed* parent ranking.

.. warning::
   This is a **ranking**, not a certified DAG.  Linear-Gaussian SEMs are only
   direction-identifiable under assumptions (e.g. equal noise variances on
   *un-standardised* data); MI is model-free but undirected.  Treat the output
   as a prioritised hypothesis list for the discovery library, not a proof of
   causation.  Nothing here imports a backend; it is pure numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.symbolic.diagnostics import feature_residual_mutual_information

__all__ = [
    "causal_discovery_report",
    "mutual_information_matrix",
    "notears_acyclicity",
    "notears_lite",
    "term_parent_ranking",
]


def _expm(matrix: np.ndarray) -> np.ndarray:
    """Dependency-free matrix exponential (scaling-and-squaring + Taylor series).

    Adequate for the small nonnegative matrices ``W∘W`` arising in the acyclicity
    functional; avoids a SciPy dependency for :mod:`omnibias.core`'s downstream
    pure-numpy contract.
    """
    a = np.asarray(matrix, dtype=float)
    d = a.shape[0]
    norm = float(np.abs(a).sum(axis=1).max()) if d else 0.0
    s = max(0, int(np.ceil(np.log2(norm + 1e-12))))
    s = min(s, 60)
    scaled = a / (2.0**s)
    result = np.eye(d, dtype=float)
    term = np.eye(d, dtype=float)
    for k in range(1, 40):
        term = term @ scaled / k
        result = result + term
        if float(np.abs(term).max()) < 1e-18:
            break
    for _ in range(s):
        result = result @ result
    return np.asarray(result, dtype=float)


def notears_acyclicity(weights: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Smooth acyclicity functional ``h(W) = tr(e^{W∘W}) - d`` and its gradient.

    ``h(W) >= 0`` with equality **iff** the weighted digraph with adjacency
    ``W`` (entry ``W[k, j]`` = edge ``k -> j``) is acyclic.  The gradient is
    ``dh/dW = (e^{W∘W})^T ∘ 2W``.

    Returns ``(h, grad)``.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"weights must be a square matrix, got shape {w.shape}")
    d = w.shape[0]
    expw = _expm(w * w)
    h = float(np.trace(expw) - d)
    grad = np.asarray(expw.T * (2.0 * w), dtype=float)
    return h, grad


def mutual_information_matrix(
    data: np.ndarray, *, bins: int = 16, bias_correction: bool = True
) -> np.ndarray:
    r"""Symmetric pairwise mutual-information matrix of the columns of ``data``.

    ``M[i, j] = I(X_i; X_j)`` (nats) via the bias-corrected 2-D histogram
    estimator; the diagonal is set to ``0`` (self-information is not a useful
    edge score).  This is the model-free, *undirected* relevance backbone.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D (n_samples, n_vars), got shape {x.shape}")
    d = x.shape[1]
    out = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(i + 1, d):
            mi = feature_residual_mutual_information(
                x[:, i], x[:, j], bins=bins, bias_correction=bias_correction
            )
            out[i, j] = mi
            out[j, i] = mi
    return out


def _soft_threshold(a: np.ndarray, thresh: float) -> np.ndarray:
    out = np.sign(a) * np.maximum(np.abs(a) - thresh, 0.0)
    np.fill_diagonal(out, 0.0)
    return np.asarray(out, dtype=float)


def notears_lite(
    data: np.ndarray,
    *,
    lambda1: float = 0.05,
    w_threshold: float = 0.3,
    max_outer: int = 80,
    max_inner: int = 200,
    h_tol: float = 1e-10,
    rho_max: float = 1e16,
    standardize: bool = False,
) -> dict[str, Any]:
    r"""Learn a linear-SEM weighted adjacency by NOTEARS-lite.

    Minimises ``0.5/n ||X - X W||_F^2 + lambda1 ||W||_1`` subject to the
    acyclicity constraint ``h(W) = 0`` via an augmented-Lagrangian outer loop and
    a proximal-gradient (ISTA with backtracking) inner solve.  ``W[k, j]`` is the
    linear weight of parent ``k`` in the structural equation for child ``j``
    (``X_j ≈ sum_k W[k, j] X_k``); the diagonal is pinned to zero.

    By default the data is mean-centred but **not** standardised: linear-Gaussian
    direction identifiability needs the raw scales (equal-noise case), and
    standardising can flip edge directions.  Set ``standardize=True`` only when
    you care about the undirected skeleton.

    Returns a dict with ``weights`` (``W``), ``support`` (boolean adjacency after
    thresholding at ``w_threshold``), ``acyclicity`` (``h(W)``),
    ``outer_iterations`` and ``standardized``.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D (n_samples, n_vars), got shape {x.shape}")
    x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        std = x.std(axis=0, keepdims=True)
        x = x / np.where(std > 0.0, std, 1.0)
    n, d = x.shape
    if n < 2:
        raise ValueError("notears_lite needs at least 2 samples")

    w = np.zeros((d, d), dtype=float)
    rho = 1.0
    alpha = 0.0
    h_prev = float("inf")

    def smooth(mat: np.ndarray) -> float:
        resid = x - x @ mat
        h_val, _ = notears_acyclicity(mat)
        loss = 0.5 * float((resid * resid).sum()) / n
        return float(loss + 0.5 * rho * h_val * h_val + alpha * h_val)

    def smooth_grad(mat: np.ndarray) -> np.ndarray:
        resid = x - x @ mat
        g_loss = -(x.T @ resid) / n
        h_val, g_h = notears_acyclicity(mat)
        return np.asarray(g_loss + (rho * h_val + alpha) * g_h, dtype=float)

    outer = 0
    while outer < max_outer:
        outer += 1
        lipschitz = 1.0
        for _ in range(max_inner):
            f_w = smooth(w)
            grad = smooth_grad(w)
            while True:
                w_new = _soft_threshold(w - grad / lipschitz, lambda1 / lipschitz)
                delta = w_new - w
                quad = f_w + float((grad * delta).sum()) + 0.5 * lipschitz * float((delta * delta).sum())
                if smooth(w_new) <= quad + 1e-18 or lipschitz > 1e14:
                    break
                lipschitz *= 2.0
            step = float(np.sqrt((( w_new - w) ** 2).sum()))
            w = w_new
            if step < 1e-11:
                break
            lipschitz = max(lipschitz * 0.9, 1e-6)
        h_val, _ = notears_acyclicity(w)
        alpha += rho * h_val
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10.0, rho_max)
        h_prev = h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    h_final, _ = notears_acyclicity(w)
    support = np.abs(w) >= w_threshold
    np.fill_diagonal(support, False)
    return {
        "weights": w,
        "support": support,
        "acyclicity": float(h_final),
        "outer_iterations": int(outer),
        "standardized": bool(standardize),
        "n_edges": int(support.sum()),
    }


def _rank(names: Sequence[str], scores: np.ndarray) -> list[tuple[str, float]]:
    order = np.argsort(-np.asarray(scores, dtype=float))
    return [(str(names[i]), float(scores[i])) for i in order]


def term_parent_ranking(
    candidates: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    target_name: str = "target",
    bins: int = 16,
    lambda1: float = 0.05,
    w_threshold: float = 0.3,
    standardize: bool = False,
) -> dict[str, Any]:
    r"""Rank candidate terms as causal *parents* of ``target``.

    Combines the two ingredients: a model-free MI relevance ranking
    ``I(term; target)`` and a directed NOTEARS-lite ranking from the learned
    weight of each candidate into the ``target`` node.  The ``combined`` ranking
    is the mean of the two min-max-normalised scores.

    ``candidates`` is the ``(n_samples, n_terms)`` library matrix, ``target`` the
    ``(n_samples,)`` quantity whose parents are sought, and ``names`` the
    ``n_terms`` candidate names aligned to the ``candidates`` columns.

    The returned dict carries ``mi_ranking``, ``notears_ranking`` and
    ``combined_ranking`` (each a list of ``(name, score)`` sorted descending),
    plus the raw NOTEARS ``weights`` and an honesty note.
    """
    x = np.asarray(candidates, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"candidates must be 2-D, got shape {x.shape}")
    p = x.shape[1]
    if len(names) != p:
        raise ValueError(f"need {p} names, got {len(names)}")
    if y.shape[0] != x.shape[0]:
        raise ValueError("candidates and target must share the sample axis")

    mi_scores = np.array(
        [feature_residual_mutual_information(x[:, j], y, bins=bins) for j in range(p)],
        dtype=float,
    )

    full = np.column_stack([x, y])
    learned = notears_lite(
        full, lambda1=lambda1, w_threshold=w_threshold, standardize=standardize
    )
    weights = learned["weights"]
    parent_scores = np.abs(weights[:p, p])  # edges (candidate_k -> target)

    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        lo, hi = float(v.min()), float(v.max())
        if hi - lo <= 0.0:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    combined = 0.5 * (_norm(mi_scores) + _norm(parent_scores))

    return {
        "target_name": target_name,
        "mi_ranking": _rank(names, mi_scores),
        "notears_ranking": _rank(names, parent_scores),
        "combined_ranking": _rank(names, combined),
        "notears_parent_weights": {str(names[j]): float(weights[j, p]) for j in range(p)},
        "acyclicity": float(learned["acyclicity"]),
        "note": (
            "MI is undirected and model-free; NOTEARS-lite gives a direction under "
            "linear-SEM assumptions. This is a parent RANKING, not a certified DAG."
        ),
    }


def causal_discovery_report(
    data: np.ndarray,
    names: Sequence[str],
    *,
    bins: int = 16,
    lambda1: float = 0.05,
    w_threshold: float = 0.3,
    standardize: bool = False,
) -> dict[str, Any]:
    r"""Full pairwise-MI + NOTEARS-lite structure report over named variables.

    Returns the MI matrix, the learned weighted adjacency / support, the directed
    edge list (ranked by ``|weight|``), and a per-node parent ranking.  Honest:
    the edges are a ranked hypothesis, not a certified causal graph.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D (n_samples, n_vars), got shape {x.shape}")
    d = x.shape[1]
    if len(names) != d:
        raise ValueError(f"need {d} names, got {len(names)}")

    mi = mutual_information_matrix(x, bins=bins)
    learned = notears_lite(x, lambda1=lambda1, w_threshold=w_threshold, standardize=standardize)
    weights = learned["weights"]
    support = learned["support"]

    edges: list[tuple[str, str, float]] = []
    for i in range(d):
        for j in range(d):
            if support[i, j]:
                edges.append((str(names[i]), str(names[j]), float(weights[i, j])))
    edges.sort(key=lambda e: -abs(e[2]))

    parents = {
        str(names[j]): _rank(names, np.abs(weights[:, j])) for j in range(d)
    }

    return {
        "names": list(names),
        "mutual_information_matrix": mi,
        "weights": weights,
        "support": support,
        "edges": edges,
        "parents": parents,
        "acyclicity": float(learned["acyclicity"]),
        "n_edges": int(learned["n_edges"]),
        "note": (
            "Pairwise MI is undirected; NOTEARS-lite edges are direction hypotheses "
            "under linear-SEM assumptions, ranked by weight -- not a certified DAG."
        ),
    }
