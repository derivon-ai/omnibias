# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Live torch realization geometry using the shared parameter jet substrate."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from omnibias.core.realization import RealizationSpec
from omnibias.torch.realization import evaluate, parameter_jet
from torch import Tensor

from .geometry import Realization


def dense_realization(
    points: Tensor, spec: RealizationSpec, *, max_coefficients: int = 1_000_000
) -> Realization[Tensor]:
    """Finite value observations of an explicit dense six-role architecture."""

    def value(theta: Tensor) -> Tensor:
        return evaluate(points, theta, spec).reshape(-1)

    def jvp(theta: Tensor, direction: Tensor) -> Tensor:
        return parameter_jet(
            points, theta, spec, direction.reshape(-1, 1), 1, max_coefficients=max_coefficients
        )[1].reshape(-1)

    def vjp(theta: Tensor, cotangent: Tensor) -> Tensor:
        pullback = torch.func.vjp(value, theta)[1]
        return cast(Tensor, pullback(cotangent)[0])

    def jet(theta: Tensor, basis: Tensor, order: int) -> Tensor:
        result = parameter_jet(points, theta, spec, basis, order, max_coefficients=max_coefficients)
        return result.reshape(result.shape[0], -1)

    return Realization(value, jvp, vjp, jet, spec)


def observation_realization(
    value: Callable[[Tensor], Tensor],
    *,
    jet: Callable[[Tensor, Tensor, int], Tensor],
    scope: str = "finite_observations",
    derivative_provenance: str = "supplied_analytic_jets",
) -> Realization[Tensor]:
    """Callback observations with an explicit jet provider; no ignored model."""

    def jvp(theta: Tensor, direction: Tensor) -> Tensor:
        return jet(theta, direction.reshape(-1, 1), 1)[1]

    def vjp(theta: Tensor, cotangent: Tensor) -> Tensor:
        pullback = torch.func.vjp(value, theta)[1]
        return cast(Tensor, pullback(cotangent)[0])

    return Realization(
        value, jvp, vjp, jet, observation_scope=scope, derivative_provenance=derivative_provenance
    )


def dense_jacobian(
    realization: Realization[Tensor], theta: Tensor, *, max_entries: int = 1_000_000
) -> Tensor:
    """Explicit, budgeted dense Jacobian; metric_action avoids this allocation."""
    if theta.ndim != 1 or theta.shape[0] * realization.value(theta).shape[0] > max_entries:
        raise ValueError("dense observation Jacobian budget exceeded")
    basis = torch.eye(theta.shape[0], dtype=theta.dtype, device=theta.device)
    return torch.stack([realization.jvp(theta, basis[i]) for i in range(theta.shape[0])], dim=1)


__all__ = ["dense_jacobian", "dense_realization", "observation_realization"]
