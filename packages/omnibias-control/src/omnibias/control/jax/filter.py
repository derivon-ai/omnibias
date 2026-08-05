# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable, batched, per-sample CBF-QP safety filter (jax).

Bit-identical twin of :mod:`omnibias.control.torch.filter` (the arithmetic is the
same elementwise reductions, so the two agree to ``rtol=1e-12`` in float64).

Terminology: the hinge below belongs to **temperature collapse** -- the
``beta -> inf`` axis that sharpens one gate into a 0/1 feasibility indicator.
That is a different limit from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to ``sigma^(K-1)``, a derivative; see
:mod:`omnibias.torch.unit`). Note this filter takes the hard clamp directly
rather than annealing, so it sits at the ``beta = inf`` endpoint of that axis.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.control.problem import FilterSchedule, SafeAction


def cbf_residual(G: Array, h: Array, a: Array) -> Array:
    r"""Per-sample worst constraint residual ``max_i (G_i a - h_i)`` (``(B,)``)."""
    return jnp.max(jnp.einsum("bmd,bd->bm", G, a) - h, axis=1)


def cbf_filter(
    a_nom: Array, G: Array, h: Array, schedule: FilterSchedule | None = None
) -> Array:
    r"""Project ``a_nom`` onto the *per-sample* polytope ``{a : G_i a <= h_i}``.

    The state-dependent generalisation of a batched Euclidean projection: each sample
    ``i`` has its own constraint block ``(G_i (m,d), h_i (m,))``. Solves

    .. math::
        a_i^\star = \arg\min_a \tfrac12\lVert a - a_{\text{nom},i}\rVert^2
        \quad\text{s.t.}\quad G_i a \le h_i

    with the exterior hard-hinge penalty ``mu/2 sum relu(G_i a - h_i)^2`` -- closed-form
    gradient ``(a - a_nom) + mu G_i^T relu(G_i a - h_i)`` (the temperature-collapse unit) and
    per-sample closed-form Lipschitz step (Frobenius bound ``||G_i||_2^2 <= ||G_i||_F^2``,
    elementwise so it is bit-identical across backends), minimised by accelerated
    (Nesterov) gradient descent over a short ``mu`` homotopy. Pure / ``jit`` / ``grad``
    friendly, so a policy can be trained *through* the filter.

    Parameters
    ----------
    a_nom:
        Nominal (task) actions, shape ``(B, d)``.
    G, h:
        Per-sample constraint block, shapes ``(B, m, d)`` and ``(B, m)``.
    schedule:
        :class:`~omnibias.control.problem.FilterSchedule`; ``None`` uses the
        eval-quality default (use :meth:`FilterSchedule.fast` when training).

    Returns
    -------
    The filtered actions ``a*``, shape ``(B, d)``.
    """
    sched = schedule if schedule is not None else FilterSchedule()
    a2 = jnp.sum(G * G, axis=(1, 2))                       # (B,) per-sample ||G||_F^2

    def descent(x0: Array, mu: float, steps: int) -> Array:
        eta = sched.safety / (1.0 + mu * a2 + 1e-30)       # (B,)

        def body(_: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            x, y, t = carry
            u = jnp.einsum("bmd,bd->bm", G, y) - h          # (B, m)
            gate = jnp.maximum(u, 0.0)                      # hard temperature-collapse unit
            grad = (y - a_nom) + mu * jnp.einsum("bmd,bm->bd", G, gate)
            x_next = y - eta[:, None] * grad
            t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
            y_next = x_next + ((t - 1.0) / t_next) * (x_next - x)
            return x_next, y_next, t_next

        one = jnp.asarray(1.0, dtype=a_nom.dtype)
        x, _, _ = jax.lax.fori_loop(0, steps, body, (x0, x0, one))
        return cast(Array, x)

    x = a_nom
    mu = sched.mu0
    for _ in range(sched.stages):
        x = descent(x, mu, sched.steps)
        mu *= sched.mu_growth
    return x


def filter_action(
    a_nom: Array, G: Array, h: Array, schedule: FilterSchedule | None = None
) -> SafeAction[Array]:
    """:func:`cbf_filter` wrapped in a :class:`SafeAction` with the residual diagnostic."""
    a = cbf_filter(a_nom, G, h, schedule)
    return SafeAction(action=a, nominal=a_nom, residual=cbf_residual(G, h, a))


__all__ = ["cbf_filter", "cbf_residual", "filter_action"]
