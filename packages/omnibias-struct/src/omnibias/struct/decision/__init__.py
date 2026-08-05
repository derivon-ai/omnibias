# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.struct.decision: an embeddable, certified predict-then-optimize decision layer.

A bridge that makes omnibias-struct's :func:`~omnibias.struct.certify_argmax` **embeddable in a
network** as a differentiable *decision layer*:

* forward (relaxed decision): ``softmax(beta * scores)`` -- differentiable, so a downstream
  decision-regret loss backpropagates into whatever produced the scores;
* the ``beta -> inf`` hard decision: ``argmax(scores)``;
* the certificate: the closed-form :class:`~omnibias.struct.SelectionCertificate` -- the value
  gap ``max <= lse_beta <= max + log(N)/beta``, the Gibbs mode-mass concentration lower bound,
  and (with ``eps``) the ``L^inf`` argmax-stability radius ``margin / 2``.

The backend-neutral numpy helpers (:func:`certified_decision`, :func:`decision_regret`,
:func:`best_index`) live here; the trainable :class:`~omnibias.struct.decision.torch.DecisionLayer`
lives in :mod:`omnibias.struct.decision.torch` (needs the ``torch`` extra).

Terminology: the ``beta -> inf`` annealing is the **feasibility / temperature** sense of
"collapse" (the same axis as ``omnibias-discrete`` / ``omnibias-qubo``), **not** the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form derivative
``sigma^(K-1)``) -- the derivative tower is only the engine that differentiates ``lse_beta``
exactly. This is the measure/mode face of that feasibility axis (a Gibbs law collapsing onto
its mode).
"""

from __future__ import annotations

import numpy as np
from omnibias.struct._core.select import SelectionCertificate, certify_argmax


def best_index(scores: np.ndarray, *, axis: int = -1) -> np.ndarray:
    r"""The hard decision ``argmax(scores)`` along ``axis`` (the ``beta -> inf`` limit)."""
    idx: np.ndarray = np.asarray(np.argmax(np.asarray(scores, dtype=float), axis=axis))
    return idx


def certified_decision(
    scores: np.ndarray, beta: float, *, eps: float | None = None
) -> SelectionCertificate:
    r"""Certify the ``beta -> inf`` decision collapse of ``softmax(beta * scores)`` (1-D scores).

    Thin wrapper over :func:`omnibias.struct.certify_argmax`: returns the closed-form
    :class:`SelectionCertificate` for a single decision over ``N`` options.
    """
    return certify_argmax(np.asarray(scores, dtype=float).reshape(-1), beta, eps=eps)


def decision_regret(scores_hat: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    r"""Per-sample realized regret of deciding ``argmax(scores_hat)`` under true ``rewards``.

    ``regret_i = max_j rewards[i, j] - rewards[i, argmax_j scores_hat[i, j]]`` (``>= 0``, ``0``
    when the predicted-best option is truly best). ``scores_hat`` and ``rewards`` are
    ``(n, N)``; returns the ``(n,)`` regret vector (mean it for the headline number).
    """
    s = np.asarray(scores_hat, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if s.shape != r.shape or s.ndim != 2:
        raise ValueError("scores_hat and rewards must be the same (n, N) shape")
    picked = np.argmax(s, axis=1)
    realized = r[np.arange(r.shape[0]), picked]
    regret: np.ndarray = np.max(r, axis=1) - realized
    return regret


__all__ = [
    "SelectionCertificate",
    "best_index",
    "certified_decision",
    "decision_regret",
]
