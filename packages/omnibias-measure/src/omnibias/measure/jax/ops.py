# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable (jax) measure-integral primitives.

Bit-identical twin of :mod:`omnibias.measure._core.integrate` and
:mod:`omnibias.measure.torch.ops`. ``nodes`` come from the numpy
:class:`~omnibias.measure._core.measure.Measure`; ``weights`` and ``beta`` may be
passed as ``jax.Array`` s so gradients flow through them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.measure._core.integrate import SimpleFunctionApprox
from omnibias.measure._core.measure import Measure

Integrand = Callable[[Array], Array]

_TRAPEZOID: Callable[..., Any] = getattr(jnp, "trapezoid", None) or jnp.trapz  # type: ignore[attr-defined]


def _nodes_weights(
    measure: Measure | None,
    nodes: Array | None,
    weights: Array | None,
) -> tuple[Array, Array]:
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`/`weights`")
        nodes = jnp.asarray(measure.nodes)
    if weights is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `weights`")
        w = jnp.asarray(measure.weights)
    else:
        w = weights
    if w.shape[0] != nodes.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != n_nodes {nodes.shape[0]}")
    return nodes, w


def lebesgue_integral(
    f: Integrand,
    measure: Measure | None = None,
    *,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> Array:
    """Measure integral ``int f dmu = sum_i weights_i * f(nodes_i)`` (jax)."""
    nd, w = _nodes_weights(measure, nodes, weights)
    vals = f(nd)
    if vals.shape[0] != nd.shape[0]:
        raise ValueError("f must return one value per node")
    return jnp.tensordot(w, vals, axes=([0], [0]))


def importance_expectation(
    f: Integrand,
    measure: Measure | None = None,
    log_weight: Integrand | None = None,
    *,
    self_normalized: bool = True,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> Array:
    """(Self-normalized) importance-sampling expectation of ``f`` (jax)."""
    if log_weight is None:
        raise ValueError("importance_expectation requires a `log_weight` callable")
    nd, w = _nodes_weights(measure, nodes, weights)
    vals = f(nd)
    lw = log_weight(nd)
    if lw.shape != (nd.shape[0],):
        raise ValueError(f"log_weight must return shape ({nd.shape[0]},)")
    if self_normalized:
        shift = jax.lax.stop_gradient(jnp.max(lw))
        iw = w * jnp.exp(lw - shift)
        num = jnp.tensordot(iw, vals, axes=([0], [0]))
        den = jnp.sum(iw)
        return num / den
    iw = w * jnp.exp(lw)
    return jnp.tensordot(iw, vals, axes=([0], [0]))


def superlevel_measure(
    f: Integrand,
    measure: Measure | None = None,
    levels: Array | None = None,
    *,
    beta: float | Array = 50.0,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> Array:
    """Soft superlevel-set measure ``G_k = mu({f > level_k})`` (jax)."""
    if levels is None:
        raise ValueError("superlevel_measure requires `levels`")
    nd, w = _nodes_weights(measure, nodes, weights)
    fx = f(nd).reshape(-1)
    t = jnp.asarray(levels).reshape(-1)
    diff = beta * (fx[None, :] - t[:, None])
    return jnp.sum(w[None, :] * jax.nn.sigmoid(diff), axis=1)


def layer_cake_integral(
    f: Integrand,
    measure: Measure | None = None,
    *,
    t_grid: Array | None = None,
    t_max: float | None = None,
    num_t: int = 256,
    beta: float | Array = 50.0,
    signed: bool = True,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> Array:
    r"""Layer-cake (distribution-function) measure integral of scalar ``f`` (jax)."""
    nd, w = _nodes_weights(measure, nodes, weights)
    fx = f(nd).reshape(-1)
    if t_grid is None:
        if t_max is None:
            t_max = float(jnp.max(jnp.abs(fx))) * 1.05 + 1e-6
        tg = jnp.linspace(0.0, float(t_max), int(num_t), dtype=fx.dtype)
    else:
        tg = jnp.asarray(t_grid).reshape(-1)
    diff_pos = beta * (fx[None, :] - tg[:, None])
    s_pos = jnp.sum(w[None, :] * jax.nn.sigmoid(diff_pos), axis=1)
    integrand = s_pos
    if signed:
        diff_neg = beta * (-fx[None, :] - tg[:, None])
        s_neg = jnp.sum(w[None, :] * jax.nn.sigmoid(diff_neg), axis=1)
        integrand = s_pos - s_neg
    out: Array = _TRAPEZOID(integrand, tg)
    return out


def simple_function_approx(
    f: Integrand,
    measure: Measure | None = None,
    *,
    levels: Array,
    beta: float | Array = 50.0,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> SimpleFunctionApprox[Array]:
    """Monotone from-below simple-function approximation of ``int f dmu`` (jax)."""
    t = jnp.sort(jnp.asarray(levels).reshape(-1))
    g = superlevel_measure(f, measure, t, beta=beta, nodes=nodes, weights=weights)
    zero = jnp.zeros((1,), dtype=g.dtype)
    g_next = jnp.concatenate([g[1:], zero])
    band_masses = g - g_next
    prev = jnp.concatenate([zero, t[:-1].astype(g.dtype)])
    integral = jnp.sum((t.astype(g.dtype) - prev) * g)
    return SimpleFunctionApprox(
        integral=integral,
        levels=t,
        superlevel_measures=g,
        band_masses=band_masses,
    )


__all__ = [
    "Integrand",
    "importance_expectation",
    "layer_cake_integral",
    "lebesgue_integral",
    "simple_function_approx",
    "superlevel_measure",
]
