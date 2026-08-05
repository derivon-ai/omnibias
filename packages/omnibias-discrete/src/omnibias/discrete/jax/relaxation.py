# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable annealed relaxation core (JAX) via the temperature-collapse penalty.

Bit-identical twin of :mod:`omnibias.discrete.torch.relaxation` (float64). Given a
caller-supplied **closed-form** energy gradient ``grad_x E(x)``, the layer parametrises a
soft assignment ``x = sigmoid(beta theta) in (0, 1)^n`` and descends

.. math::
    \nabla_\theta E = \bigl(\nabla_x E\bigr)\odot\bigl(\beta\, x (1 - x)\bigr),

by unrolled gradient descent along a geometric ``beta`` homotopy; as ``beta`` grows the
soft assignment collapses onto a binary vertex. One ``jit``-able, ``jax.grad``-friendly
call, so a model that predicts the problem coefficients can be trained *through* the
relaxation. Consumers (``omnibias.qubo``, ``omnibias.discrete.maxsat``) supply the
gradient and a step ``scale``; the returned soft assignment is decoded with
:func:`omnibias.discrete.decode` and the gap certified with
:func:`omnibias.discrete.certify_gap`.

Terminology: the ``beta -> inf`` hardening of ``sigmoid`` here is the feasibility /
temperature sense of "collapse" (a soft indicator becoming a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to ``sigma^(K-1)``,
a derivative; see ``docs/theory.md`` and :mod:`omnibias.torch.unit`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.discrete._core.relax import initial_theta
from omnibias.discrete._core.schedule import AnnealSchedule


def anneal_descent(
    grad_x_fn: Callable[[Array], Array],
    scale: Any,
    n: int,
    schedule: AnnealSchedule | None = None,
) -> Array:
    r"""Anneal ``x = sigmoid(beta theta)`` to a vertex by descending ``grad_x_fn``.

    Parameters
    ----------
    grad_x_fn:
        The closed-form energy gradient ``x -> grad_x E(x)`` (an ``(n,)`` array). It may
        close over differentiable coefficients so the whole descent is
        ``jax.grad``-differentiable through them.
    scale:
        A (detached) Lipschitz-like magnitude for the energy gradient, setting the step
        size ``eta = step_safety / (beta * 0.25 * scale)``. Larger is safer (slower).
    n:
        Number of variables.
    schedule:
        The :class:`AnnealSchedule` (defaults to ``AnnealSchedule()``).
    """
    sched = schedule or AnnealSchedule()
    theta = jnp.asarray(initial_theta(n), dtype=jnp.float64)
    # Detached so the step size never perturbs the gradient w.r.t. the coefficients.
    scale_c = jax.lax.stop_gradient(jnp.asarray(scale, dtype=jnp.float64))

    def stage(theta: Array, beta: float) -> Array:
        eta = sched.step_safety / (beta * 0.25 * scale_c + 1e-30)

        def body(_: int, th: Array) -> Array:
            x = jax.nn.sigmoid(beta * th)
            grad_x = grad_x_fn(x)
            grad_theta = grad_x * (beta * x * (1.0 - x))
            return cast(Array, th - eta * grad_theta)

        return cast(Array, jax.lax.fori_loop(0, sched.steps, body, theta))

    betas = sched.betas()
    for beta in betas:
        theta = stage(theta, beta)
    result: Array = jax.nn.sigmoid(betas[-1] * theta)
    return result


__all__ = ["anneal_descent"]
