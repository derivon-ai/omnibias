# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable safe closed-loop rollout (jax). Bit-identical twin."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.control.jax.filter import cbf_filter, cbf_residual
from omnibias.control.problem import FilterSchedule


def safe_rollout(
    policy: Callable[[Array], Array],
    step: Callable[[Array, Array], Array],
    rows_fn: Callable[[Array], tuple[Array, Array]],
    x0: Array,
    *,
    horizon: int,
    schedule: FilterSchedule | None = None,
) -> tuple[Array, Array, Array]:
    r"""Roll the safe closed loop ``horizon`` steps (differentiable via ``lax.scan``).

    At each step: ``a_nom = policy(x)``, build the state-dependent rows
    ``G, h = rows_fn(x)``, filter ``a = cbf_filter(a_nom, G, h)``, and advance
    ``x = step(x, a)``. Because every piece is differentiable, gradients flow through
    the *whole* safe loop -- train the policy to anticipate the filter.

    Parameters
    ----------
    policy:
        ``x (B,n) -> a_nom (B,d)``.
    step:
        Dynamics ``(x (B,n), a (B,d)) -> x_next (B,n)`` (analytic or learned).
    rows_fn:
        ``x (B,n) -> (G (B,m,d), h (B,m))`` CBF + actuator rows (e.g. a
        :mod:`~omnibias.control.jax.builders` partial).
    x0:
        Initial states, shape ``(B, n)``.
    horizon:
        Number of steps ``T``.
    schedule:
        Filter homotopy (``None`` -> eval-quality default; use
        :meth:`FilterSchedule.fast` for training).

    Returns
    -------
    ``(X (T,B,n), A (T,B,d), residual (T,B))`` -- post-step states, applied safe
    actions, and the per-step worst CBF residual.
    """

    def scan_step(x: Array, _: None) -> tuple[Array, tuple[Array, Array, Array]]:
        a_nom = policy(x)
        G, h = rows_fn(x)
        a = cbf_filter(a_nom, G, h, schedule)
        x_next = step(x, a)
        return x_next, (x_next, a, cbf_residual(G, h, a))

    _, (X, A, resid) = jax.lax.scan(scan_step, x0, None, length=horizon)
    return X, A, resid


def barrier_trace(barrier: Callable[[Array], Array], X: Array) -> Array:
    r"""Barrier value ``h(x)`` at every ``(t, sample)`` of a rollout, shape ``(T, B)``."""
    return jax.vmap(jax.vmap(barrier))(X)


def min_barrier(barrier: Callable[[Array], Array], X: Array) -> Array:
    r"""Worst (minimum-over-time) barrier per sample, shape ``(B,)`` (safe iff ``>= 0``)."""
    return jnp.min(barrier_trace(barrier, X), axis=0)


__all__ = ["barrier_trace", "min_barrier", "safe_rollout"]
