# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable TSP relaxation layers (JAX) via the unrolled temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.routing.torch.relaxation` (float64). Each layer
solves a poly-size convex TSP relaxation (:mod:`omnibias.routing._core.relax_systems`)
with the omnibias temperature-collapse penalty *unrolled* for differentiability: a small
quadratic regulariser ``reg`` (strong convexity -> smooth cost-sensitive gradients),
the hard-hinge exterior penalty ``mu/2 relu(A_ineq x - b_ineq)^2`` and the quadratic
equality penalty ``mu/2 ||A_eq x - b_eq||^2``, both with the **closed-form** gradient

.. math::
    \nabla F = c + \mathrm{reg}\, x + \mu A_{eq}^\top (A_{eq} x - b_{eq})
             + \mu A_{ineq}^\top \mathrm{relu}(A_{ineq} x - b_{ineq}),

minimised by accelerated (Nesterov) gradient descent along a geometric ``mu``
homotopy with the closed-form Lipschitz step ``eta = safety / (reg + mu(||A_eq||_F^2
+ ||A_ineq||_F^2))``. One batched ``jit`` call, ``jax.grad`` friendly -- backprop
flows through the tour relaxation into the cost model (decision-focused routing).

Returns the fractional **arc-use matrix** ``(n, n)`` (a heatmap); decode it with
:func:`omnibias.routing.decode_tour` and certify the gap with
:func:`omnibias.routing.certify_tour_gap`.

Terminology: "temperature-collapse penalty" here is the feasibility sense of
"collapse" (a hard-hinge constraint force), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to
``sigma^(K-1)``, a derivative; see ``docs/theory.md`` and
:mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.routing._core.relax_systems import RelaxSystem, build_system
from omnibias.routing.problem import RelaxationSchedule


def _descent(
    cvec: Array,
    A_eq: Array,
    b_eq: Array,
    A_ineq: Array,
    b_ineq: Array,
    aeq2: float,
    ain2: float,
    sched: RelaxationSchedule,
) -> Array:
    reg = sched.reg
    steps = sched.steps

    def stage(x0: Array, mu: float) -> Array:
        eta = sched.step_safety / (reg + mu * (aeq2 + ain2) + 1e-30)

        def body(_: Array, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            x, y, t = carry
            req = y @ A_eq.T - b_eq
            rin = jnp.maximum(y @ A_ineq.T - b_ineq, 0.0)
            grad = cvec + reg * y + mu * (req @ A_eq) + mu * (rin @ A_ineq)
            x_next = y - eta * grad
            t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            y_next = x_next + ((t - 1.0) / t_next) * (x_next - x)
            return x_next, y_next, t_next

        one = jnp.asarray(1.0, dtype=cvec.dtype)
        x_final, _, _ = jax.lax.fori_loop(0, steps, body, (x0, x0, one))
        return cast(Array, x_final)

    x = jnp.zeros_like(cvec)
    for mu in sched.mus():
        x = stage(x, mu)
    return x


def _relaxation(cost: Array, kind: str, schedule: RelaxationSchedule | None) -> Array:
    sched = schedule or RelaxationSchedule()
    cost_j = jnp.asarray(cost, dtype=jnp.float64)
    single = cost_j.ndim == 2
    if single:
        cost_j = cost_j[None]
    batch, n, _ = cost_j.shape
    sys: RelaxSystem = build_system(n, kind)
    arc_src = jnp.asarray(np.array([i * n + j for (i, j) in sys.arcs]))
    A_eq = jnp.asarray(sys.A_eq)
    b_eq = jnp.asarray(sys.b_eq)
    A_ineq = jnp.asarray(sys.A_ineq)
    b_ineq = jnp.asarray(sys.b_ineq)
    aeq2 = float(np.sum(sys.A_eq * sys.A_eq))
    ain2 = float(np.sum(sys.A_ineq * sys.A_ineq))

    flat = cost_j.reshape(batch, n * n)
    cx = flat[:, arc_src]
    cx = cx / (jnp.mean(cx, axis=1, keepdims=True) + 1e-12)  # scale-invariant conditioning
    pad = sys.n_vars - sys.n_arcs
    cvec = cx if pad == 0 else jnp.concatenate([cx, jnp.zeros((batch, pad))], axis=1)

    x = _descent(cvec, A_eq, b_eq, A_ineq, b_ineq, aeq2, ain2, sched)
    arc = x[:, : sys.n_arcs]
    mat = jnp.zeros((batch, n * n)).at[:, arc_src].set(arc).reshape(batch, n, n)
    return mat[0] if single else mat


def assignment_relaxation(cost: Array, schedule: RelaxationSchedule | None = None) -> Array:
    """Differentiable degree-constrained (assignment) relaxation -> arc-use ``(n, n)``."""
    return _relaxation(cost, "assignment", schedule)


def flow_relaxation(cost: Array, schedule: RelaxationSchedule | None = None) -> Array:
    """Differentiable single-commodity-flow (subtour-free) relaxation -> arc-use."""
    return _relaxation(cost, "flow", schedule)


def held_karp_layer(cost: Array, schedule: RelaxationSchedule | None = None) -> Array:
    r"""Differentiable multicommodity-flow (Held-Karp) relaxation -> arc-use.

    Tightest bound but ``O(n^3)`` variables (dense); intended for small ``n`` /
    single instances. A matrix-free operator is the staged scalability follow-up.
    """
    return _relaxation(cost, "held_karp", schedule)


__all__ = ["assignment_relaxation", "flow_relaxation", "held_karp_layer"]
