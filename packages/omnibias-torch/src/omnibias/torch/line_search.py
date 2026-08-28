# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact jet line search (theory 03-12), PyTorch twin.

``jet_line_search`` minimises the directional Taylor model of a scalar
``phi(s) = loss(params + s * direction)``. Coefficients come from nested
:func:`torch.func.grad` of the restriction (exact for an omnibias PINN
loss whose residual jet is closed form). Model roots, the certified
Lagrange radius, and the never-worse verify live in
:mod:`omnibias.core.line_search` so the jax twin is bit-identical on the
same derivatives.

``jet_line_search_on_ray`` is the input-direction variant: :func:`mlp_jet`
plus :func:`compose_jet` through a quadratic loss, with no new jet
arithmetic.

The driver is eager (host-side root isolation). Do not wrap it in
``torch.compile`` and expect the polynomial solve to stay on device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from omnibias.core.line_search import (
    JetLineSearchConfig,
    LineSearchResult,
    run_model_line_search,
)
from omnibias.core.spec import ActivationSpec
from omnibias.torch.jet import compose_jet, jet_to_tower, mlp_jet

import torch
from torch import Tensor
from torch.func import grad

ScalarFn = Callable[[Tensor], Tensor]
Layer = tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]


def directional_derivatives(
    loss_fn: ScalarFn,
    params: Tensor,
    direction: Tensor,
    order: int,
) -> list[float]:
    """``phi^(k)(0)`` for ``k = 0 .. order`` via nested reverse-mode.

    ``params`` and ``direction`` are flat tensors of the same shape.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if params.shape != direction.shape:
        raise ValueError(
            f"params and direction must share shape, got {tuple(params.shape)} "
            f"vs {tuple(direction.shape)}"
        )

    def phi(step: Tensor) -> Tensor:
        return loss_fn(params + step * direction)

    s0 = torch.zeros((), dtype=params.dtype, device=params.device)
    derivs = [float(phi(s0))]
    current: Callable[[Tensor], Tensor] = phi
    for _ in range(order):
        current = grad(current)
        derivs.append(float(cast(Tensor, current(s0))))
    return derivs


def quadratic_loss_tower(u0: Tensor, target: Tensor, order: int) -> Tensor:
    r"""Derivative tower of ``0.5 (u - t)^2`` at ``u0`` (orders ``0 .. order``).

    ``sigma' = u - t``, ``sigma'' = 1``, ``sigma^(k) = 0`` for ``k > 2``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    rows = [0.5 * (u0 - target) ** 2]
    if order >= 1:
        rows.append(u0 - target)
    if order >= 2:
        rows.append(torch.ones_like(u0))
    rows.extend(torch.zeros_like(u0) for _ in range(max(0, order - 2)))
    return torch.stack(rows[: order + 1], dim=0)


def mse_loss_jet(output_jet: Tensor, target: Tensor) -> Tensor:
    """Compose ``0.5 ||y - t||^2`` onto an output jet via :func:`compose_jet`."""
    jet = torch.as_tensor(output_jet)
    tgt = torch.as_tensor(target, dtype=jet.dtype, device=jet.device)
    if jet.ndim == 1:
        jet = jet.unsqueeze(-1)
    if jet.ndim != 2:
        raise ValueError(f"output_jet must be (N+1,) or (N+1, C), got {tuple(jet.shape)}")
    order = int(jet.shape[0]) - 1
    width = int(jet.shape[1])
    tgt = tgt.reshape(-1)
    if int(tgt.shape[0]) != width:
        raise ValueError(f"target length {int(tgt.shape[0])} != output width {width}")
    acc: Tensor | None = None
    for idx in range(width):
        u = jet[:, idx]
        tower = quadratic_loss_tower(u[0], tgt[idx], order)
        part = compose_jet(u, tower)
        acc = part if acc is None else acc + part
    assert acc is not None
    return acc


def jet_line_search(
    loss_fn: ScalarFn,
    params: Tensor,
    direction: Tensor,
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
        # |phi^(N+1)(0)| is a point value, not a sound max on [0, r].
        bound = None
    params_d = params.detach()
    direction_d = direction.detach()

    def actual(step: float) -> float:
        s = torch.as_tensor(step, dtype=params_d.dtype, device=params_d.device)
        return float(loss_fn(params_d + s * direction_d))

    return run_model_line_search(
        derivs[: cfg.order + 1],
        config=cfg,
        next_derivative_bound=bound,
        actual_fn=actual if cfg.verify else None,
    )


def jet_line_search_on_ray(
    layers: Sequence[Layer],
    x0: Tensor,
    direction: Tensor,
    target: Tensor,
    *,
    config: JetLineSearchConfig | None = None,
    next_derivative_bound: float | None = None,
) -> LineSearchResult:
    """Input-ray line search: ``mlp_jet`` then quadratic ``compose_jet``."""
    cfg = config if config is not None else JetLineSearchConfig()
    y_jet = mlp_jet(x0, direction, layers, cfg.order)
    loss_jet = mse_loss_jet(y_jet, target)
    tower = jet_to_tower(loss_jet)
    derivs = [float(tower[k].reshape(-1)[0]) for k in range(cfg.order + 1)]

    def actual(step: float) -> float:
        moved = x0 + float(step) * direction
        y = mlp_jet(moved, direction, layers, 0)[0]
        tgt = torch.as_tensor(target, dtype=y.dtype, device=y.device)
        residual = y.reshape(-1) - tgt.reshape(-1)
        return float(0.5 * torch.dot(residual, residual))

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
