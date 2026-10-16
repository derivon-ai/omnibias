# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable forward aging series for Gaussian heat inverses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .gaussian_heat import _tensor


class GaussianHeatAgingResult(NamedTuple):
    """Truncated inverse, ordinary correction coefficients and diagnostics.

    The error estimate does not include floating-point evaluation error and
    is not an outward-rounded certificate. Both it and ``valid`` have the
    target batch shape; ``contraction_bound`` is a shared scalar.
    Coefficients have shape ``(order+1, ..., dimension)`` with zero order zero.
    """

    value: Tensor
    correction_coefficients: Tensor
    approximation_error_bound: Tensor
    contraction_bound: Tensor
    valid: Tensor


def gaussian_heat_inverse_aging(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, order: int = 8,
) -> GaussianHeatAgingResult:
    r"""Evaluate the Gaussian inverse aging series without iteration or solves.

    With p=softmax(b+a@target/(2*nu)), mu=sum(p*a), the returned coefficients
    c_n describe ``X(target,time/t)-(time/t)*target-mu = sum(c_n*t**n)``.
    They are ordinary Taylor coefficients, with structural c_0=0. ``value``
    evaluates the truncated series at t=1, adding ``time*target+mu``. Only
    the centered correction is analytic at t=0. The coefficient recurrence
    depends on previously computed orders and needs no implicit linear solve.
    Autograd differentiates this finite formula, including higher derivatives;
    these are not implicit derivatives of the exact inverse.

    Target shape is ``(...,d)`` with float32/64 dtype. Shared centers ``(m,d)``,
    biases ``(m,)`` and positive scalar viscosity/time are cast to its dtype
    and device with gradients preserved. Bias and center gauges are removed
    before computing logits. ``order`` is a static nonnegative integer.

    With D the center diameter, require q=D²/(8*nu*time)<1/3. The analytical
    Euclidean tail bound is ``D/2*(3*q)**(order+1)/(1-3*q)``. Orders zero and
    one use the minimum with D*q/2 and D*q², respectively. Identical centers
    give an exact affine result with zero correction. Numerical bounds
    exclude rounding error and are NOT rigorous interval certificates.
    Diagnostics are stopped-gradient. Invalid/nonfinite arithmetic or q>=1/3
    gives NaN value/coefficients/batch error and false validity. Finite q is
    retained for diagnosis; invalid shared geometry gives NaN q. For N orders
    each target uses O((m+d)*N²+m*d*N) arithmetic and O((m+d)*N+m*d) forward
    storage; the shared diameter costs O(m²*d). Autograd may retain more storage.
    """
    if not isinstance(target, Tensor) or target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be a float32 or float64 Tensor")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(order) is not int:
        raise TypeError("order must be a nonnegative integer")
    if order < 0:
        raise ValueError("order must be nonnegative")
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
    valid = shared & (q < 1 / 3) & torch.isfinite(target).all(-1)
    query = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    logits = query @ relative.mT / (2 * nu) + offsets
    valid = valid & torch.isfinite(logits).all(-1)
    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    probabilities = torch.softmax(logits, dim=-1)
    mean = probabilities @ relative
    centered = relative - mean[..., None, :]
    coefficients = [torch.zeros_like(query)]
    logarithms = [torch.zeros_like(probabilities)]
    exponentials = [torch.ones_like(probabilities)]
    partitions = [torch.ones_like(valid, dtype=target.dtype)]
    for n in range(1, order + 1):
        logarithm = (-centered.square().sum(-1) / (safe_denominator / 2) if n == 1
                     else (centered * coefficients[n - 1][..., None, :]).sum(-1) / (safe_denominator / 4))
        logarithms.append(logarithm)
        exponential = torch.zeros_like(probabilities)
        for k in range(1, n + 1):
            exponential = exponential + (k / n) * logarithms[k] * exponentials[n - k]
        exponentials.append(exponential)
        weighted = probabilities * exponential
        partition = weighted.sum(-1)
        partitions.append(partition)
        coefficient = (weighted[..., :, None] * centered).sum(-2)
        for k in range(1, n + 1):
            coefficient = coefficient - partitions[k][..., None] * coefficients[n - k]
        coefficients.append(coefficient)
        valid = (valid & torch.isfinite(logarithm).all(-1) & torch.isfinite(exponential).all(-1)
                 & torch.isfinite(partition) & torch.isfinite(coefficient).all(-1))
    stacked = torch.stack(coefficients)
    value = age * query + locations[0] + mean + stacked.sum(0)
    # Diagnostics are deliberately independent of the differentiable predictor.
    diameter = torch.sqrt(squared_diameter.detach())
    diagnostic_q = q.detach()
    safe_q = torch.where(shared & (diagnostic_q < 1 / 3), diagnostic_q, torch.zeros_like(diagnostic_q))
    estimate = diameter / 2 * (3 * safe_q).pow(order + 1) / (1 - 3 * safe_q)
    if order == 0:
        estimate = torch.minimum(estimate, diameter * safe_q / 2)
    elif order == 1:
        estimate = torch.minimum(estimate, diameter * safe_q.square())
    valid = valid & torch.isfinite(value).all(-1) & torch.isfinite(estimate)
    return GaussianHeatAgingResult(
        torch.where(valid[..., None], value, torch.full_like(value, float("nan"))),
        torch.where(valid[None, ..., None], stacked, torch.full_like(stacked, float("nan"))),
        torch.where(valid, estimate, torch.full_like(valid, float("nan"), dtype=target.dtype)),
        torch.where(shared, diagnostic_q, torch.full_like(diagnostic_q, float("nan"))),
        valid,
    )


__all__ = ["GaussianHeatAgingResult", "gaussian_heat_inverse_aging"]
