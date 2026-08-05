# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Model-order selection for discovered sparse equations.

Validation RMSE alone rewards the *largest* library that still generalises a
little; it does not answer "is this extra term worth its complexity?".  This
module adds principled answers:

* information criteria -- :func:`aic`, :func:`aicc`, :func:`bic`, :func:`mdl`
  (all on the common deviance scale ``-2 log L + penalty``, lower is better) and
  the dispatcher :func:`information_criterion`;
* :func:`kfold_select` -- K-fold cross-validation over an ``(alpha, threshold)``
  grid, returning the most predictive configuration;
* :func:`stability_selection` -- Meinshausen--Buhlmann subsampling that ranks
  candidate terms by how often they survive the sparsity threshold.

All criteria assume a homoscedastic Gaussian residual model, for which the
profiled log-likelihood is a function of the residual sum of squares alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation, rmse

CRITERIA: tuple[str, ...] = ("aic", "aicc", "bic", "mdl")

_TINY_RSS = 1e-300


def gaussian_log_likelihood(n: int, rss: float) -> float:
    r"""Profiled Gaussian log-likelihood at the MLE variance ``sigma^2 = rss/n``.

    ``log L = -n/2 (log(2 pi) + log(rss/n) + 1)``.  A floor on ``rss`` keeps the
    value finite for (near-)exact fits.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    sigma2 = max(float(rss), _TINY_RSS) / n
    return -0.5 * n * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0)


def aic(n: int, rss: float, k: int) -> float:
    """Akaike information criterion ``2k - 2 log L`` (lower is better)."""
    return 2.0 * k - 2.0 * gaussian_log_likelihood(n, rss)


def aicc(n: int, rss: float, k: int) -> float:
    """Small-sample-corrected AIC; ``inf`` when ``n <= k + 1``."""
    denom = n - k - 1
    if denom <= 0:
        return float("inf")
    return aic(n, rss, k) + (2.0 * k * (k + 1)) / denom


def bic(n: int, rss: float, k: int) -> float:
    """Bayesian (Schwarz) information criterion ``k log n - 2 log L``."""
    return k * math.log(n) - 2.0 * gaussian_log_likelihood(n, rss)


def _log_binom(p: int, k: int) -> float:
    if k < 0 or k > p:
        return 0.0
    return math.lgamma(p + 1) - math.lgamma(k + 1) - math.lgamma(p - k + 1)


def mdl(n: int, rss: float, k: int, *, n_candidates: int | None = None) -> float:
    r"""Minimum description length on the deviance scale.

    Uses the BIC core ``k log n - 2 log L`` plus, when ``n_candidates`` (the
    library size ``p``) is given, a model-structure code length
    ``2 log C(p, k)`` for *which* ``k`` of ``p`` terms are active.  That extra
    combinatorial penalty -- absent from AIC/BIC -- makes MDL prefer sparser
    models when the candidate library is large.
    """
    value = bic(n, rss, k)
    if n_candidates is not None:
        value += 2.0 * _log_binom(int(n_candidates), int(k))
    return value


def information_criterion(
    name: str,
    n: int,
    rss: float,
    k: int,
    *,
    n_candidates: int | None = None,
) -> float:
    """Dispatch to :func:`aic` / :func:`aicc` / :func:`bic` / :func:`mdl` by name."""
    key = name.lower()
    if key == "aic":
        return aic(n, rss, k)
    if key == "aicc":
        return aicc(n, rss, k)
    if key == "bic":
        return bic(n, rss, k)
    if key == "mdl":
        return mdl(n, rss, k, n_candidates=n_candidates)
    raise ValueError(f"unknown selection criterion {name!r}; expected one of {CRITERIA}")


def equation_information_criterion(
    equation: SparseEquation,
    design: np.ndarray,
    target: np.ndarray,
    *,
    name: str = "bic",
    count_intercept: bool = True,
    n_candidates: int | None = None,
) -> float:
    """Information criterion of a fitted ``equation`` on ``(design, target)``.

    The parameter count ``k`` is the number of active terms (plus the intercept
    when ``count_intercept``); the residual sum of squares is measured on the
    supplied data.
    """
    pred = equation.predict(design)
    resid = np.asarray(target, dtype=float) - pred
    rss = float(resid @ resid)
    n = int(np.asarray(target).shape[0])
    k = len(equation.active_terms()) + (1 if count_intercept else 0)
    if n_candidates is None:
        n_candidates = len(equation.term_names)
    return information_criterion(name, n, rss, k, n_candidates=n_candidates)


@dataclass(frozen=True)
class KFoldSelection:
    """Result of :func:`kfold_select`."""

    alpha: float
    threshold: float
    cv_rmse: float
    table: tuple[dict[str, float], ...]


def kfold_select(
    design: np.ndarray,
    target: np.ndarray,
    term_names: list[str],
    *,
    alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4),
    thresholds: tuple[float, ...] = (1e-6, 1e-4, 1e-3, 1e-2),
    k: int = 5,
    seed: int = 0,
    max_iter: int = 8,
) -> KFoldSelection:
    """Select ``(alpha, threshold)`` by ``k``-fold cross-validated RMSE.

    Returns the configuration with the lowest mean held-out RMSE together with
    the full scored table (sorted best-first).
    """
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    n = x.shape[0]
    if k < 2 or k > n:
        raise ValueError("k must satisfy 2 <= k <= n_samples")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    table: list[dict[str, float]] = []
    for alpha in alphas:
        for threshold in thresholds:
            fold_rmse: list[float] = []
            for f in range(k):
                test_idx = folds[f]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != f])
                eq = fit_sparse_equation(
                    x[train_idx], y[train_idx], list(term_names),
                    alpha=alpha, threshold=threshold, max_iter=max_iter,
                )
                fold_rmse.append(rmse(y[test_idx], eq.predict(x[test_idx])))
            table.append(
                {
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "cv_rmse": float(np.mean(fold_rmse)),
                    "cv_rmse_std": float(np.std(fold_rmse)),
                }
            )
    table.sort(key=lambda row: row["cv_rmse"])
    best = table[0]
    return KFoldSelection(
        alpha=best["alpha"],
        threshold=best["threshold"],
        cv_rmse=best["cv_rmse"],
        table=tuple(table),
    )


def stability_selection(
    design: np.ndarray,
    target: np.ndarray,
    term_names: list[str],
    *,
    alpha: float = 1e-8,
    threshold: float = 1e-4,
    n_resample: int = 100,
    sample_fraction: float = 0.5,
    seed: int = 0,
    max_iter: int = 8,
) -> dict[str, Any]:
    """Meinshausen--Buhlmann stability selection.

    Refits the STLSQ on ``n_resample`` random subsamples (drawn *without*
    replacement, each of size ``sample_fraction * n``) and reports, per term, the
    fraction of subsamples in which it is active.  Returns ``term_names``,
    ``selection_frequency`` (aligned), and ``ranking`` (term/frequency pairs,
    most stable first).
    """
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    n, p = x.shape
    if p != len(term_names):
        raise ValueError("term_names must match design width")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1]")
    m = max(2, int(round(sample_fraction * n)))
    rng = np.random.default_rng(seed)
    counts = np.zeros(p, dtype=float)
    for _ in range(n_resample):
        idx = rng.choice(n, size=m, replace=False)
        eq = fit_sparse_equation(
            x[idx], y[idx], list(term_names),
            alpha=alpha, threshold=threshold, max_iter=max_iter,
        )
        counts += np.asarray(eq.active_mask, dtype=float)
    freq = counts / float(n_resample)
    order = np.argsort(-freq)
    ranking = tuple((term_names[int(i)], float(freq[int(i)])) for i in order)
    return {
        "term_names": tuple(term_names),
        "selection_frequency": freq,
        "ranking": ranking,
        "n_resample": int(n_resample),
        "sample_fraction": float(sample_fraction),
    }


__all__ = [
    "CRITERIA",
    "KFoldSelection",
    "aic",
    "aicc",
    "bic",
    "equation_information_criterion",
    "gaussian_log_likelihood",
    "information_criterion",
    "kfold_select",
    "mdl",
    "stability_selection",
]
