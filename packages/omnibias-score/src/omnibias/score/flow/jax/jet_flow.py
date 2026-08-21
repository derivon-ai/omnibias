# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Coupling jet-flow (jax; theory 09-06).

Closed-form ``log|det|`` from ``sigma'`` (founding bias collapse
``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. Do not conflate the two. Not ``integrate_cnf``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core import coupling_flow as core

DISCLAIMER = core.DISCLAIMER
JetFlowConfig = core.JetFlowConfig
honesty_payload = core.honesty_payload


def jet_flow_forward(x: Array, scale: Array, shift: Array | None = None) -> tuple[Array, Array]:
    t = jnp.zeros((), dtype=x.dtype) if shift is None else shift
    pre = scale * x + t
    y = jnp.tanh(pre)
    log_det = jnp.log(jnp.abs(scale) * (1.0 - y * y))
    return y, log_det


def jet_flow_inverse(
    y: Array,
    scale: Array,
    shift: Array | None = None,
    *,
    config: JetFlowConfig | None = None,
    tol: float = 1e-12,
) -> Array:
    cfg = core.DEFAULT_CONFIG if config is None else config
    floor = cfg.sigma_floor
    t = jnp.zeros((), dtype=y.dtype) if shift is None else shift

    def cond(state: tuple[Array, Array, Array]) -> Array:
        _z, i, err = state
        return (err > tol) & (i < cfg.newton_max)

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        z, i, _err = state
        pre = scale * z + t
        pred = jnp.tanh(pre)
        deriv = scale * (1.0 - pred * pred)
        deriv = jnp.where(jnp.abs(deriv) < floor, floor, deriv)
        z_next = z - (pred - y) / deriv
        return z_next, i + 1, jnp.abs(pred - y)

    z0 = jnp.zeros_like(y)
    z, _, _ = jax.lax.while_loop(cond, body, (z0, jnp.asarray(0), jnp.asarray(1.0)))
    return z


def worked_example() -> dict[str, float]:
    return core.worked_example()


def jet_flow_forward_params(
    x: Sequence[float],
    params: Sequence[tuple[float, float]],
    *,
    config: JetFlowConfig | None = None,
) -> tuple[list[float], float]:
    return core.jet_flow_forward(x, params, config=config)


__all__ = [
    "DISCLAIMER",
    "JetFlowConfig",
    "honesty_payload",
    "jet_flow_forward",
    "jet_flow_forward_params",
    "jet_flow_inverse",
    "worked_example",
]
