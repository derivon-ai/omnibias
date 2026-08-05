# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable sparse ``l_p -> l_0`` relaxation (JAX) over the shared temperature-collapse core.

Bit-identical twin of :mod:`omnibias.discrete.sparse.torch.relaxation` (float64). The
relaxed landscape replaces the exact linear cardinality penalty ``lambda 1^T z`` with the
concave ``l_p`` surrogate ``lambda sum_i (x_i + eps)^p`` (``0 < p <= 1``), so the
closed-form gradient handed to :func:`omnibias.discrete.jax.anneal_descent` is

.. math::
    \nabla_x E_{\text{relax}}(x)
      = A^\top A\, x - A^\top b
      + \lambda\, p\, (x + \varepsilon)^{p-1},

where ``A^T A x - A^T b`` is the exact least-squares data gradient and the
``(x + eps)^{p-1}`` factor is the sparsity-promoting reweighting spike (it blows up as
``x -> 0`` for ``p < 1``, pushing small entries to the ``0`` vertex). ``anneal_descent``
chains the Riccati sigmoid Jacobian ``beta x (1 - x)`` internally and hardens
``x = sigmoid(beta theta)`` onto ``{0, 1}^n`` as ``beta -> inf``.

``p`` is a per-call knob (sweep ``p: 1 -> 0.5 -> 0.1`` toward ``l_0``); at ``p = 1`` the
penalty gradient is the constant ``lambda`` and this reduces to the exact QUBO relaxation
of :class:`~omnibias.discrete.sparse.problem.SupportSelectionProblem`. The exact
``energy`` / ``to_polynomial`` / ``certify_gap`` are ``p``-independent, so the ``l_p``
homotopy is a pure *relaxation* knob and the certified seal stays sound.

Terminology (two distinct senses of "collapse"): both the ``beta -> inf`` sigmoid
hardening and the ``l_p -> l_0`` penalty-exponent homotopy here are the **feasibility** /
temperature sense (a soft object becoming a hard ``0/1`` selection). Neither is the
**founding bias collapse** -- the multi-bias ``delta -> 0`` limit of an ``OMBU`` to the
closed-form derivative ``sigma^(K-1)`` (see ``docs/theory.md`` and
:mod:`omnibias.jax.jet`).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.jax import anneal_descent
from omnibias.discrete.sparse.problem import SupportSelectionProblem


def sparse_relaxation(
    problem: SupportSelectionProblem,
    *,
    p: float = 1.0,
    eps: float = 1e-3,
    schedule: AnnealSchedule | None = None,
) -> Array:
    r"""Differentiable ``l_p`` annealed relaxation of a support-selection instance.

    Parameters
    ----------
    problem:
        The :class:`~omnibias.discrete.sparse.problem.SupportSelectionProblem` (its
        ``A^T A`` / ``A^T b`` / ``lambda`` define the relaxed energy).
    p:
        Penalty exponent ``0 < p <= 1``; ``p -> 0`` approaches the ``l_0`` count, ``p = 1``
        is the exact linear penalty. Swept toward ``0`` in the reweighting homotopy.
    eps:
        Small positive offset stabilising ``(x + eps)^{p-1}`` near ``x = 0``.
    schedule:
        The :class:`AnnealSchedule` (defaults to ``AnnealSchedule()``).

    Returns
    -------
    The soft assignment ``x in (0, 1)^n`` (decode with :func:`omnibias.discrete.decode`).
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must satisfy 0 < p <= 1, got {p}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    sched = schedule or AnnealSchedule()
    n = problem.n
    gram = jnp.asarray(problem.gram_matrix, dtype=jnp.float64)
    corr = jnp.asarray(problem.correlation, dtype=jnp.float64)
    lam = float(problem.lam)
    gram_norm = float(jnp.linalg.norm(gram, 2)) if n else 1.0
    scale = gram_norm + lam * p * (eps ** (p - 1.0)) + 1.0

    def grad_x_fn(x: Array) -> Array:
        data = gram @ x - corr
        penalty = lam * p * jnp.power(x + eps, p - 1.0)
        out: Array = data + penalty
        return out

    result: Array = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["sparse_relaxation"]
