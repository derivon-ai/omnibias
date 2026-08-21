# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact jet line search (theory 03-12), JAX twin.

Bit-identical partner of :mod:`omnibias.torch.line_search` on the same
directional derivatives. The polynomial solve is host-side (eager); do
not ``jit`` the driver. Nested ``jax.grad`` of the scalar restriction is
the jet; :func:`mlp_jet` / :func:`compose_jet` handle the input-ray path.

Enable 64-bit JAX before the first array if you need torch parity
(:mod:`omnibias.jax.precision`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from omnibias.core.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    run_model_line_search,
)
from omnibias.core.spec import ActivationSpec
from omnibias.jax.jet import compose_jet, jet_to_tower, mlp_jet

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree

ScalarFn = Callable[[Any], Array]
Layer = tuple[Array, Array | None, ActivationSpec[Array] | str | None]


def _flat_pair(params: Any, direction: Any) -> tuple[Array, Array, Callable[[Array], Any]]:
    theta, unravel = ravel_pytree(params)
    direction_flat, _ = ravel_pytree(direction)
    if theta.shape != direction_flat.shape:
        raise ValueError(
            f"params and direction must ravel to the same shape, got {tuple(theta.shape)} "
            f"vs {tuple(direction_flat.shape)}"
        )
    return theta, direction_flat, unravel


def directional_derivatives(
    loss_fn: ScalarFn,
    params: Any,
    direction: Any,
    order: int,
) -> list[float]:
    """``phi^(k)(0)`` for ``k = 0 .. order`` via nested ``jax.grad``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    theta, direction_flat, unravel = _flat_pair(params, direction)

    def phi(step: Array) -> Array:
        return loss_fn(unravel(theta + step * direction_flat))

    s0 = jnp.zeros((), dtype=theta.dtype)
    derivs = [float(phi(s0))]
    current: Callable[[Array], Array] = phi
    for _ in range(order):
        current = jax.grad(current)
        derivs.append(float(current(s0)))
    return derivs


def quadratic_loss_tower(u0: Array, target: Array, order: int) -> Array:
    r"""Derivative tower of ``0.5 (u - t)^2`` at ``u0`` (orders ``0 .. order``)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    rows = [0.5 * (u0 - target) ** 2]
    if order >= 1:
        rows.append(u0 - target)
    if order >= 2:
        rows.append(jnp.ones_like(u0))
    rows.extend(jnp.zeros_like(u0) for _ in range(max(0, order - 2)))
    return jnp.stack(rows[: order + 1], axis=0)


def mse_loss_jet(output_jet: Array, target: Array) -> Array:
    """Compose ``0.5 ||y - t||^2`` onto an output jet via :func:`compose_jet`."""
    jet = jnp.asarray(output_jet)
    tgt = jnp.asarray(target, dtype=jet.dtype)
    if jet.ndim == 1:
        jet = jet[:, None]
    if jet.ndim != 2:
        raise ValueError(f"output_jet must be (N+1,) or (N+1, C), got {tuple(jet.shape)}")
    order = int(jet.shape[0]) - 1
    width = int(jet.shape[1])
    tgt = tgt.reshape(-1)
    if int(tgt.shape[0]) != width:
        raise ValueError(f"target length {int(tgt.shape[0])} != output width {width}")
    acc: Array | None = None
    for idx in range(width):
        u = jet[:, idx]
        tower = quadratic_loss_tower(u[0], tgt[idx], order)
        part = compose_jet(u, tower)
        acc = part if acc is None else acc + part
    assert acc is not None
    return acc


def jet_line_search(
    loss_fn: ScalarFn,
    params: Any,
    direction: Any,
    *,
    config: JetLineSearchConfig | None = None,
    next_derivative_bound: float | None = None,
) -> LineSearchResult:
    """Parameter-direction jet line search with optional certified radius."""
    cfg = config if config is not None else JetLineSearchConfig()
    extra = 1 if (cfg.trust_radius == "certified" and next_derivative_bound is None) else 0
    derivs = directional_derivatives(loss_fn, params, direction, cfg.order + extra)
    bound = next_derivative_bound
    if extra:
        bound = None
    theta, direction_flat, unravel = _flat_pair(params, direction)

    def actual(step: float) -> float:
        s = jnp.asarray(step, dtype=theta.dtype)
        return float(loss_fn(unravel(theta + s * direction_flat)))

    return run_model_line_search(
        derivs[: cfg.order + 1],
        config=cfg,
        next_derivative_bound=bound,
        actual_fn=actual if cfg.verify else None,
    )


def jet_line_search_on_ray(
    layers: Sequence[Layer],
    x0: Array,
    direction: Array,
    target: Array,
    *,
    config: JetLineSearchConfig | None = None,
    next_derivative_bound: float | None = None,
) -> LineSearchResult:
    """Input-ray line search: ``mlp_jet`` then quadratic ``compose_jet``."""
    cfg = config if config is not None else JetLineSearchConfig()
    y_jet = mlp_jet(x0, direction, layers, cfg.order)
    loss_jet = mse_loss_jet(y_jet, target)
    tower = jet_to_tower(loss_jet)
    derivs = [float(jnp.reshape(tower[k], (-1,))[0]) for k in range(cfg.order + 1)]

    def actual(step: float) -> float:
        moved = x0 + float(step) * direction
        y = mlp_jet(moved, direction, layers, 0)[0]
        tgt = jnp.asarray(target, dtype=y.dtype)
        residual = jnp.reshape(y, (-1,)) - jnp.reshape(tgt, (-1,))
        return float(0.5 * jnp.dot(residual, residual))

    return run_model_line_search(
        derivs,
        config=cfg,
        next_derivative_bound=next_derivative_bound,
        actual_fn=actual if cfg.verify else None,
    )


__all__ = [
    "JetLineSearchConfig",
    "Layer",
    "LineSearchResult",
    "ScalarFn",
    "directional_derivatives",
    "jet_line_search",
    "jet_line_search_on_ray",
    "mse_loss_jet",
    "quadratic_loss_tower",
]
