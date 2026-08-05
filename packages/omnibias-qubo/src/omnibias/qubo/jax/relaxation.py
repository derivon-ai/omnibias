# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable QUBO relaxation layer (JAX) via the annealed temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.qubo.torch.relaxation` (float64). The layer
parametrises a soft assignment ``x = sigmoid(beta theta) in (0, 1)^n`` and descends the
**closed-form** energy gradient

.. math::
    \nabla_\theta E = (2 Q x + c)\odot\bigl(\beta\, x (1 - x)\bigr),

by unrolled gradient descent along a geometric ``beta`` homotopy; as ``beta`` grows the
soft assignment collapses onto a binary vertex. This is a thin QUBO front-end over the
shared substrate core :func:`omnibias.discrete.jax.anneal_descent`: one ``jit``-able,
``jax.grad``-friendly call, so a model that predicts ``Q`` / ``c`` can be trained
*through* the relaxation. The returned soft assignment is decoded with
:func:`omnibias.qubo.decode_qubo` and the gap certified with
:func:`omnibias.qubo.certify_qubo_gap`.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array
from omnibias.discrete.jax import anneal_descent
from omnibias.qubo.problem import AnnealSchedule


def _unpack(problem_or_Q: Any, c: Any) -> tuple[Any, Any]:
    if hasattr(problem_or_Q, "Q") and hasattr(problem_or_Q, "c"):
        return problem_or_Q.Q, problem_or_Q.c
    return problem_or_Q, c


def qubo_relaxation(
    problem_or_Q: Any,
    c: Any = None,
    schedule: AnnealSchedule | None = None,
) -> Array:
    r"""Differentiable annealed relaxation of a QUBO -> soft assignment ``x in (0, 1)^n``.

    Accepts a :class:`~omnibias.qubo.problem.QUBOProblem` or an array-like ``Q`` (with an
    optional linear ``c``); pass tensors to differentiate through ``Q`` / ``c``.
    """
    sched = schedule or AnnealSchedule()
    Q_in, c_in = _unpack(problem_or_Q, c)
    Q = jnp.asarray(Q_in, dtype=jnp.float64)
    Q = 0.5 * (Q + Q.T)
    n = int(Q.shape[0])
    cvec = jnp.zeros(n, dtype=jnp.float64) if c_in is None else jnp.asarray(c_in, dtype=jnp.float64)

    # Frobenius norm upper-bounds the spectral norm -> a safe (smaller) descent step;
    # anneal_descent detaches it so the step size does not perturb the grad w.r.t. Q / c.
    scale = 2.0 * jnp.sqrt(jnp.sum(Q * Q)) + jnp.max(jnp.abs(cvec))

    def grad_x_fn(x: Array) -> Array:
        grad: Array = 2.0 * (Q @ x) + cvec
        return grad

    result: Array = anneal_descent(grad_x_fn, scale, n, sched)
    return result


__all__ = ["qubo_relaxation"]
