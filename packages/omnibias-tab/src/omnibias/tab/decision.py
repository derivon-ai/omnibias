# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias.tab.decision: certify the discrete leaf-routing decision of a soft tree.

A thin adapter (a bridge on omnibias-struct) that turns a soft tree's per-tree leaf-routing
distribution into a **certified discrete decision** -- *which leaf the sample commits to* --
without touching tab's trained forward. It reuses
:func:`~omnibias.tab.leaf_memberships` and
:func:`omnibias.struct.certify_argmax`.

Because the routing distribution ``P[i, m, :]`` is already a probability vector over the
``2**depth`` leaves that sums to one, the logits ``s = log(P) / beta`` reproduce ``P`` exactly
as a Gibbs law ``softmax(beta s) = P``. Feeding them to
:func:`~omnibias.struct.certify_argmax` yields a faithful
:class:`~omnibias.struct.SelectionCertificate` of tab's own ``beta -> inf`` routing collapse:
which leaf dominates, the value gap ``<= log(2**depth)/beta``, the mode-mass concentration
lower bound, and (with ``eps``) the ``L^inf`` argmax-stability radius of the routing decision.

Terminology: the gate / routing ``beta -> inf`` hardening is the **feasibility / temperature**
sense of "collapse", distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import numpy as np
from omnibias.struct._core.select import SelectionCertificate, certify_argmax
from omnibias.tab._core.forward import leaf_memberships
from omnibias.tab._core.params import TabParams

_TINY = 1e-300  # floor so log of a numerically-zero membership stays finite


def leaf_logits(params: TabParams, x: np.ndarray, beta: float) -> np.ndarray:
    r"""Per-tree leaf logits ``s`` with ``softmax(beta * s) = P`` (the tab routing distribution).

    ``P = leaf_memberships(params, x, beta)`` sums to one over the leaves, so ``s = log(P)/beta``
    is the (shift-equivalent) logit set whose Gibbs law at inverse temperature ``beta`` is
    exactly ``P``. Shape ``(n, n_trees, 2**depth)``.
    """
    p = leaf_memberships(params, np.asarray(x, dtype=float), beta)
    logits: np.ndarray = np.log(np.clip(p, _TINY, None)) / beta
    return logits


def certified_leaf_decision(
    params: TabParams,
    x: np.ndarray,
    *,
    beta: float | None = None,
    tree: int = 0,
    eps: float | None = None,
) -> tuple[np.ndarray, SelectionCertificate]:
    r"""Certify which leaf a single sample routes to, for one tree of the ensemble.

    Parameters
    ----------
    params:
        The soft-tree ensemble parameters.
    x:
        A single sample of shape ``(n_features,)`` (or ``(1, n_features)``).
    beta:
        Gate sharpness (defaults to the config ``beta_final``). The certificate's value gap is
        ``<= log(2**depth) / beta`` and shrinks as ``beta -> inf``.
    tree:
        Which tree of the ensemble to certify the routing of (default the first).
    eps:
        Optional ``L^inf`` perturbation radius for the argmax-stability sub-claim.

    Returns
    -------
    (routing, certificate):
        ``routing`` is the leaf probability vector ``P[tree]`` of shape ``(2**depth,)``;
        ``certificate`` is the :class:`SelectionCertificate` of the ``beta -> inf`` collapse.
    """
    b = float(params.config.beta_final if beta is None else beta)
    xarr = np.asarray(x, dtype=float).reshape(1, -1)
    n_trees = int(params.config.n_trees)
    if not (0 <= tree < n_trees):
        raise ValueError(f"tree {tree} out of range for an ensemble of {n_trees} trees")
    routing = leaf_memberships(params, xarr, b)[0, tree]  # (L,)
    logits = np.log(np.clip(routing, _TINY, None)) / b
    cert = certify_argmax(logits, b, eps=eps)
    return routing, cert


def certified_leaf_decisions(
    params: TabParams,
    X: np.ndarray,
    *,
    beta: float | None = None,
    tree: int = 0,
    eps: float | None = None,
) -> list[tuple[np.ndarray, SelectionCertificate]]:
    r"""Batched :func:`certified_leaf_decision` over the rows of ``X`` (one result per sample)."""
    Xv = np.asarray(X, dtype=float)
    if Xv.ndim != 2:
        raise ValueError(f"X must be 2D (n, n_features), got shape {Xv.shape}")
    return [
        certified_leaf_decision(params, Xv[i], beta=beta, tree=tree, eps=eps)
        for i in range(Xv.shape[0])
    ]


__all__ = [
    "certified_leaf_decision",
    "certified_leaf_decisions",
    "leaf_logits",
]
