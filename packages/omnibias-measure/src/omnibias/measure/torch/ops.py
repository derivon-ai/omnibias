# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable (torch) measure-integral primitives.

Bit-identical twin of :mod:`omnibias.measure._core.integrate`. The measure's
``nodes`` come from the numpy :class:`~omnibias.measure._core.measure.Measure`
(fixed positions), while ``weights`` and the softness ``beta`` may be passed as
tensors so gradients flow through them -- enabling learnable measures and
learnable level-set softness. The integrand ``f`` is any torch callable
``nodes -> values`` (typically a sub-network), so autograd carries gradients
through its parameters too.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from omnibias.measure._core.integrate import SimpleFunctionApprox
from omnibias.measure._core.measure import Measure
from torch import Tensor

Integrand = Callable[[Tensor], Tensor]


def _nodes_weights(
    measure: Measure | None,
    nodes: Tensor | None,
    weights: Tensor | None,
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    dt = dtype if dtype is not None else torch.get_default_dtype()
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`/`weights`")
        nodes = torch.as_tensor(measure.nodes, dtype=dt, device=device)
    if weights is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `weights`")
        w = torch.as_tensor(measure.weights, dtype=dt, device=device)
    else:
        w = weights
    if w.shape[0] != nodes.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != n_nodes {nodes.shape[0]}")
    return nodes, w


def lebesgue_integral(
    f: Integrand,
    measure: Measure | None = None,
    *,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Measure integral ``int f dmu = sum_i weights_i * f(nodes_i)`` (torch)."""
    nd, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    vals = f(nd)
    if vals.shape[0] != nd.shape[0]:
        raise ValueError("f must return one value per node")
    out: Tensor = torch.tensordot(w, vals, dims=([0], [0]))
    return out


def importance_expectation(
    f: Integrand,
    measure: Measure | None = None,
    log_weight: Integrand | None = None,
    *,
    self_normalized: bool = True,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """(Self-normalized) importance-sampling expectation of ``f`` (torch)."""
    if log_weight is None:
        raise ValueError("importance_expectation requires a `log_weight` callable")
    nodes, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    vals = f(nodes)
    lw = log_weight(nodes)
    if lw.shape != (nodes.shape[0],):
        raise ValueError(f"log_weight must return shape ({nodes.shape[0]},)")
    if self_normalized:
        shift = torch.max(lw).detach()
        iw = w * torch.exp(lw - shift)
        num: Tensor = torch.tensordot(iw, vals, dims=([0], [0]))
        den = torch.sum(iw)
        return num / den
    iw = w * torch.exp(lw)
    plain: Tensor = torch.tensordot(iw, vals, dims=([0], [0]))
    return plain


def superlevel_measure(
    f: Integrand,
    measure: Measure | None = None,
    levels: Tensor | None = None,
    *,
    beta: float | Tensor = 50.0,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Soft superlevel-set measure ``G_k = mu({f > level_k})`` (torch)."""
    if levels is None:
        raise ValueError("superlevel_measure requires `levels`")
    nodes, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    fx = f(nodes).reshape(-1)
    t = levels.reshape(-1).to(dtype=fx.dtype)
    diff = beta * (fx[None, :] - t[:, None])
    return torch.sum(w[None, :] * torch.sigmoid(diff), dim=1)


def layer_cake_integral(
    f: Integrand,
    measure: Measure | None = None,
    *,
    t_grid: Tensor | None = None,
    t_max: float | None = None,
    num_t: int = 256,
    beta: float | Tensor = 50.0,
    signed: bool = True,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Layer-cake (distribution-function) measure integral of scalar ``f`` (torch).

    ``int f dmu = int_0^inf [mu({f>t}) - mu({f<-t})] dt`` with soft superlevel
    measures; differentiable through ``f``, the weights and ``beta``.
    """
    nodes, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    fx = f(nodes).reshape(-1)
    if t_grid is None:
        if t_max is None:
            t_max = float(torch.max(torch.abs(fx)).detach()) * 1.05 + 1e-6
        tg = torch.linspace(0.0, float(t_max), int(num_t), dtype=fx.dtype, device=fx.device)
    else:
        tg = t_grid.reshape(-1).to(dtype=fx.dtype)
    diff_pos = beta * (fx[None, :] - tg[:, None])
    s_pos = torch.sum(w[None, :] * torch.sigmoid(diff_pos), dim=1)
    integrand = s_pos
    if signed:
        diff_neg = beta * (-fx[None, :] - tg[:, None])
        s_neg = torch.sum(w[None, :] * torch.sigmoid(diff_neg), dim=1)
        integrand = s_pos - s_neg
    return torch.trapezoid(integrand, tg)


def simple_function_approx(
    f: Integrand,
    measure: Measure | None = None,
    *,
    levels: Tensor,
    beta: float | Tensor = 50.0,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> SimpleFunctionApprox[Tensor]:
    """Monotone from-below simple-function approximation of ``int f dmu`` (torch)."""
    t = torch.sort(levels.reshape(-1)).values
    g = superlevel_measure(
        f, measure, t, beta=beta, nodes=nodes, weights=weights, dtype=dtype, device=device
    )
    zero = torch.zeros(1, dtype=g.dtype, device=g.device)
    g_next = torch.cat([g[1:], zero])
    band_masses = g - g_next
    prev = torch.cat([zero, t[:-1].to(dtype=g.dtype)])
    integral = torch.sum((t.to(dtype=g.dtype) - prev) * g)
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
