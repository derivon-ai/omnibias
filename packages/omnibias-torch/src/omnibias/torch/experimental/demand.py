# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The extended demand vocabulary served by the stacked stream engine (see the JAX module of the same name)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.core.demand import (
    Demand,
    Quantities,
    QuantitySpec,
    Residual,
    block_order,
    partial_requests,
)
from omnibias.torch.experimental._chain import points_2d
from omnibias.torch.experimental.streams import partial_derivatives, second_order_streams

import torch
from torch import Tensor

__all__ = ["QuantitySpec", "streams_quantities", "streams_residual"]


def streams_quantities(
    x: Tensor,
    layers: Sequence[Any],
    demand: Residual | Demand,
    spec: QuantitySpec | None = None,
    *,
    diffusion: Tensor | None = None,
    point_diffusion: Tensor | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Tensor | None = None,
    weights: Tensor | None = None,
    form: str = "auto",
    allow_almost_everywhere: bool = False,
) -> Quantities:
    """Every quantity of ``demand`` at ``x`` from the stacked stream engine, differentiable in the layers.

    The second-order group (``grad_sq``, ``lap``, ``directional``, and ``grad`` through the full Jacobian) is one stacked
    pass, ``hess_form`` a second pass carrying ``A``, and the partials (``mixed``, ``third``, ``fourth``, ``lap_grad``,
    ``bilap``) share one pass of directional Taylor streams. Metric arguments and ``directions`` (``(D, q)`` or per-point
    ``(B, D, q)``) are those of :func:`~omnibias.torch.experimental.streams.second_order_streams`.
    """
    names = demand.demand if isinstance(demand, Residual) else demand
    spec = QuantitySpec() if spec is None else spec
    a, _ = points_2d(x)
    dim = a.shape[1]
    spec.check(names, dim)
    metric: dict[str, Any] = {
        "diffusion": diffusion,
        "point_diffusion": point_diffusion,
        "coordinates": coordinates,
        "weights": weights,
    }
    plain = all(v is None for v in metric.values())
    if "bilap" in names and (diffusion is not None or point_diffusion is not None):
        raise ValueError(
            "bilap through the stream engine is the plain Laplacian squared over `coordinates`"
        )
    value: Tensor | None = None
    grad = grad_sq = lap = directional = None
    flags = {k: k in names for k in ("grad", "grad_sq", "lap", "directional")}

    if flags["grad"]:
        full = second_order_streams(
            x,
            layers,
            directions=directions if plain else None,
            full_jacobian=True,
            allow_almost_everywhere=allow_almost_everywhere,
        )
        value, grad = full.value, full.jacobian
        if plain:
            grad_sq, lap, directional = full.grad_sq, full.laplacian, full.directional
    if (flags["grad_sq"] or flags["lap"] or flags["directional"]) and (
        grad_sq is None or not plain
    ):
        res = second_order_streams(
            x,
            layers,
            directions=directions,
            form=form,
            allow_almost_everywhere=allow_almost_everywhere,
            **metric,
        )
        value = res.value if value is None else value
        grad_sq, lap, directional = res.grad_sq, res.laplacian, res.directional
    hess = None
    if "hess_form" in names:
        res = second_order_streams(
            x,
            layers,
            coordinates=coordinates,
            hess_form=torch.as_tensor(spec.hess_form, dtype=a.dtype, device=a.device),
            form=form,
            allow_almost_everywhere=allow_almost_everywhere,
        )
        value = res.value if value is None else value
        hess = res.laplacian
    coords = None if coordinates is None else [int(c) for c in np.asarray(coordinates)]
    alphas, where, bilap_w = partial_requests(names, spec, dim, coords)
    parts: dict[str, Tensor] = {}
    if alphas:
        v, d = partial_derivatives(
            x, layers, alphas, allow_almost_everywhere=allow_almost_everywhere
        )
        value = v if value is None else value
        for name, (lo, hi) in where.items():
            parts[name] = d[..., lo:hi, :]
    if value is None:
        value = second_order_streams(
            x, layers, allow_almost_everywhere=allow_almost_everywhere
        ).value
    batch = tuple(value.shape[:-1])
    out: dict[str, Any] = {}
    if "lap_grad" in parts:
        p = parts["lap_grad"]
        out["lap_grad"] = p.reshape(*batch, dim, dim, p.shape[-1]).sum(dim=-2)
    if "bilap" in parts:
        w = torch.as_tensor(bilap_w, dtype=value.dtype, device=value.device)
        out["bilap"] = torch.einsum("r,...rc->...c", w, parts["bilap"])
    for name in ("mixed", "third", "fourth"):
        if name in parts:
            out[name] = parts[name]
    return Quantities(
        u=value,
        x=x,
        grad=grad,
        grad_sq=grad_sq if flags["grad_sq"] else None,
        lap=lap if flags["lap"] else None,
        directional=directional if flags["directional"] else None,
        hess_form=hess,
        **out,
    )


def streams_residual(
    x: Tensor,
    layers: Sequence[Any],
    residual: Residual,
    spec: QuantitySpec | None = None,
    **options: Any,
) -> Tensor:
    """``residual(streams_quantities(...))``: the residual rows of a field, differentiable in the layers."""
    block_order(residual, None, "streams")
    return torch.as_tensor(residual(streams_quantities(x, layers, residual, spec, **options)))
