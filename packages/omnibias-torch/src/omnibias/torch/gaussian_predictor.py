# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable one-pass large-age predictors for Gaussian heat inverses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .gaussian_heat import _tensor


class GaussianHeatPredictorResult(NamedTuple):
    """Numerical predictor, analytical truncation estimate and validity flags.

    The error estimate does not include floating-point evaluation error and
    is not an outward-rounded certificate. Both it and ``valid`` have the
    target batch shape; ``contraction_bound`` is a shared scalar.
    """

    value: Tensor
    approximation_error_bound: Tensor
    contraction_bound: Tensor
    valid: Tensor


def gaussian_heat_inverse_predictor(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int = 1,
) -> GaussianHeatPredictorResult:
    r"""Evaluate an order-0 or order-1 Gaussian inverse predictor without a solve.

    For relative centers r=a-a0, let p=softmax(b-b0+r@target/(2*nu)),
    mu=sum(p*r), and T=sum(p*(r-mu)*|r-mu|²). Order zero returns
    ``time*target+a0+mu``; order one subtracts ``T/(4*nu*time)``.
    These are explicit approximations, so autograd differentiates the predictor
    itself, including higher derivatives, rather than an implicit exact root.

    Target shape is ``(...,d)`` with float32/64 dtype. Shared centers ``(m,d)``,
    biases ``(m,)`` and positive scalar viscosity/time are cast to its dtype
    and device with gradients preserved. Bias and center gauges are removed
    before computing logits. ``order`` is the static integer 0 or 1.

    In exact arithmetic, with D the center diameter and q=D²/(8*nu*time)<1,
    physical Euclidean truncation bounds are D*q/2 and D*q² respectively.
    Returned bounds are numerical evaluations of these analytical formulas;
    they exclude rounding error and are NOT rigorous interval certificates.
    Diagnostics are stopped-gradient. Invalid/nonfinite arithmetic or q>=1
    gives NaN value/batch error and false validity. Finite q>=1 is retained
    for diagnosis; invalid shared geometry gives NaN q. The value uses O(m*d)
    work per target; the shared diameter costs O(m²*d), with no d-by-d matrix.
    """
    if not isinstance(target, Tensor) or target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be a float32 or float64 Tensor")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(order) is not int:
        raise TypeError("order must be the integer 0 or 1")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    locations, offsets = _tensor(centers, target, "centers"), _tensor(biases, target, "biases")
    nu, age = _tensor(viscosity, target, "viscosity"), _tensor(time, target, "time")
    if locations.ndim != 2 or locations.shape[0] < 1 or locations.shape[1] != target.shape[-1]:
        raise ValueError("centers must have shape (positive mode count, target dimension)")
    if offsets.shape != (locations.shape[0],):
        raise ValueError("biases must have one entry per center")
    if nu.ndim or age.ndim:
        raise ValueError("viscosity and time must be scalars")
    shared = (torch.isfinite(locations).all() & torch.isfinite(offsets).all()
              & torch.isfinite(nu) & (nu > 0) & torch.isfinite(age) & (age > 0))
    locations = torch.where(torch.isfinite(locations), locations, torch.zeros_like(locations))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    nu = torch.where(torch.isfinite(nu) & (nu > 0), nu, torch.ones_like(nu))
    age = torch.where(torch.isfinite(age) & (age > 0), age, torch.ones_like(age))
    relative, offsets = locations - locations[0], offsets - offsets[0]
    shared = shared & torch.isfinite(relative).all() & torch.isfinite(offsets).all()
    relative = torch.where(torch.isfinite(relative), relative, torch.zeros_like(relative))
    offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
    # Geometry is a diagnostic, so do not retain pairwise-distance tapes for
    # the trainable predictor's backward pass.
    with torch.no_grad():
        squared_diameter = torch.stack([(relative - row).square().sum(-1).amax() for row in relative]).amax()
    denominator = 8 * nu * age
    shared = shared & torch.isfinite(squared_diameter) & torch.isfinite(denominator) & (denominator > 0)
    safe_denominator = torch.where(torch.isfinite(denominator) & (denominator > 0), denominator, torch.ones_like(denominator))
    q = squared_diameter / safe_denominator
    shared = shared & torch.isfinite(q)
    valid = shared & (q < 1) & torch.isfinite(target).all(-1)
    query = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    logits = query @ relative.mT / (2 * nu) + offsets
    valid = valid & torch.isfinite(logits).all(-1)
    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    probabilities = torch.softmax(logits, dim=-1)
    mean = probabilities @ relative
    value = age * query + locations[0] + mean
    if order:
        centered = relative - mean[..., None, :]
        scaled_squared = centered.square().sum(-1) / (safe_denominator / 2)
        correction = ((probabilities * scaled_squared)[..., :, None] * centered).sum(-2)
        value = value - correction
        valid = valid & torch.isfinite(scaled_squared).all(-1) & torch.isfinite(correction).all(-1)
    # Diagnostics are deliberately independent of the differentiable predictor.
    diameter = torch.sqrt(squared_diameter.detach())
    diagnostic_q = q.detach()
    estimate = diameter * (diagnostic_q / 2 if order == 0 else diagnostic_q.square())
    valid = valid & torch.isfinite(value).all(-1) & torch.isfinite(estimate)
    return GaussianHeatPredictorResult(
        torch.where(valid[..., None], value, torch.full_like(value, float("nan"))),
        torch.where(valid, estimate, torch.full_like(valid, float("nan"), dtype=target.dtype)),
        torch.where(shared, diagnostic_q, torch.full_like(diagnostic_q, float("nan"))),
        valid,
    )


__all__ = ["GaussianHeatPredictorResult", "gaussian_heat_inverse_predictor"]
