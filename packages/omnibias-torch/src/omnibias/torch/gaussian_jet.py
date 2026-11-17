# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward Taylor coefficients of numerical Gaussian heat-velocity inverses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from .gaussian_heat import GaussianHeatInverseResult, _parameters, _tensor, gaussian_heat_inverse


class GaussianHeatInverseJet(NamedTuple):
    """Ordinary Taylor coefficients and the actual numerical expansion anchor."""

    coefficients: Tensor
    base_result: GaussianHeatInverseResult
    anchor_target: Tensor
    base_target_residual: Tensor


class _Factor(NamedTuple):
    lower: Tensor
    b: Tensor
    low_rank: bool
    valid: Tensor


def _factor(probabilities: Tensor, centered: Tensor, nu: Tensor, time: Tensor,
            valid: Tensor, mode: str) -> _Factor:
    # The masked square root also has finite derivatives when softmax underflows.
    positive = probabilities > 0
    root = torch.sqrt(torch.where(positive, probabilities, torch.ones_like(probabilities)))
    root = torch.where(positive, root, torch.zeros_like(root))
    denominator = 2 * nu * time
    valid = valid & torch.isfinite(denominator) & (denominator > 0)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    b = root[..., :, None] * centered / torch.sqrt(denominator)
    valid = valid & torch.isfinite(b).all(dim=(-2, -1))
    b = torch.where(valid[..., None, None], b, torch.zeros_like(b))
    dimension, modes = centered.shape[-1], centered.shape[-2]
    low_rank = mode == "low_rank" or (mode == "auto" and modes < dimension)
    size = modes if low_rank else dimension
    gram = b @ b.mT if low_rank else b.mT @ b
    matrix = torch.eye(size, dtype=b.dtype, device=b.device) - gram
    lower, info = torch.linalg.cholesky_ex(matrix)
    valid = valid & (info == 0) & torch.isfinite(lower).all(dim=(-2, -1))
    lower = torch.where(valid[..., None, None], lower, torch.eye(size, dtype=b.dtype, device=b.device))
    return _Factor(lower, b, low_rank, valid)


def _solve(factor: _Factor, right: Tensor) -> Tensor:
    if factor.low_rank:
        projected = factor.b @ right[..., None]
        correction = torch.cholesky_solve(projected, factor.lower)
        return right + (factor.b.mT @ correction).squeeze(-1)
    return torch.cholesky_solve(right[..., None], factor.lower).squeeze(-1)


def gaussian_heat_inverse_jet(
    target_coefficients: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, max_iterations: int = 128,
    tolerance: float | None = None, adjoint: str = "auto",
) -> GaussianHeatInverseJet:
    r"""Propagate an ordinary target Taylor series through a Gaussian inverse.

    Input and physical output coefficients have shape ``(N+1,...,d)``. The
    order axis stores ordinary coefficients: the nth raw derivative is n!
    times coefficient n. Shared parameters and static inverse options follow
    :func:`gaussian_heat_inverse`. All coefficients and physical parameters
    retain autograd, including mixed and higher parameter derivatives.

    The base point x0 is obtained numerically. Higher coefficients solve the
    formal inverse series at ``anchor_target=U(x0)``, with the supplied
    positive-order target coefficients. ``base_target_residual`` is the vector
    ``anchor_target-target_coefficients[0]``. Thus the series is anchored at
    the reported target, which need not equal the requested constant term.

    A single Cholesky factorization of I-Cov(a)/(2*nu*time), or its smaller
    mode-space Woodbury matrix, is reused across all positive orders. The
    forward recurrence uses no repeated spatial differentiation. ``low_rank``
    never constructs a d-by-d matrix. Order zero needs no factorization.

    A refused base has NaN coefficients, anchor and residual. Arithmetic
    failure at a higher order gives NaN from that order onward; base_result
    describes only the base solve. These are numerical coefficients, not
    outward-rounded enclosures or convergence-radius certificates.
    """
    if not isinstance(target_coefficients, Tensor) or target_coefficients.dtype not in (torch.float32, torch.float64):
        raise TypeError("target_coefficients must be a float32 or float64 Tensor")
    if target_coefficients.ndim < 2 or target_coefficients.shape[0] < 1 or target_coefficients.shape[-1] < 1:
        raise ValueError("target_coefficients must have shape (positive order count,...,positive dimension)")
    target = target_coefficients[0]
    locations, offsets = _tensor(centers, target, "centers"), _tensor(biases, target, "biases")
    nu, s = _tensor(viscosity, target, "viscosity"), _tensor(time, target, "time")
    base = gaussian_heat_inverse(target, locations, offsets, nu, s, max_iterations=max_iterations,
                                 tolerance=tolerance, adjoint=adjoint)
    clean_centers, relative, _, phase, nu, s, _ = _parameters(locations, offsets, nu, s)
    value = torch.where(base.converged[..., None], base.value, torch.zeros_like(base.value))
    z = (value - clean_centers[0]) / s
    logits = z @ relative.mT / (2 * nu) + phase
    valid = base.converged & torch.isfinite(z).all(-1) & torch.isfinite(logits).all(-1)
    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    probabilities = torch.softmax(logits, dim=-1)
    mean = probabilities @ relative
    anchor = z - mean / s
    valid = valid & torch.isfinite(anchor).all(-1)
    anchor = torch.where(valid[..., None], anchor, torch.full_like(anchor, float("nan")))
    residual = anchor - target
    order = target_coefficients.shape[0] - 1
    if order == 0:
        return GaussianHeatInverseJet(base.value[None], base, anchor, residual)

    factor = _factor(probabilities, relative - mean[..., None, :], nu, s, valid, adjoint)
    rates = relative / (2 * nu * s)
    valid = factor.valid & torch.isfinite(rates).all()
    rates = torch.where(torch.isfinite(rates), rates, torch.zeros_like(rates))
    coefficients = [base.value]
    exponentials = [torch.ones_like(probabilities)]
    log_coefficients = [torch.zeros_like(probabilities)]
    partitions = [torch.ones_like(probabilities[..., 0])]
    means = [mean]
    for n in range(1, order + 1):
        exponential_hat = torch.zeros_like(probabilities)
        for k in range(1, n):
            exponential_hat = exponential_hat + (k / n) * log_coefficients[k] * exponentials[n - k]
        weighted_hat = probabilities * exponential_hat
        partition_hat = weighted_hat.sum(-1)
        mean_hat = weighted_hat @ relative - partition_hat[..., None] * mean
        for k in range(1, n):
            mean_hat = mean_hat - partitions[k][..., None] * means[n - k]
        requested = target_coefficients[n]
        valid = valid & torch.isfinite(requested).all(-1)
        requested = torch.where(torch.isfinite(requested), requested, torch.zeros_like(requested))
        coefficient = _solve(factor, s * requested + mean_hat)
        log_coefficient = coefficient @ rates.mT
        exponential = exponential_hat + log_coefficient
        weighted = probabilities * exponential
        partition = weighted.sum(-1)
        mean_coefficient = weighted @ relative - partition[..., None] * mean
        for k in range(1, n):
            mean_coefficient = mean_coefficient - partitions[k][..., None] * means[n - k]
        valid = (valid & torch.isfinite(coefficient).all(-1) & torch.isfinite(exponential).all(-1)
                 & torch.isfinite(partition) & torch.isfinite(mean_coefficient).all(-1))
        coefficients.append(torch.where(valid[..., None], coefficient, torch.full_like(coefficient, float("nan"))))
        log_coefficients.append(log_coefficient)
        exponentials.append(exponential)
        partitions.append(partition)
        means.append(mean_coefficient)
    return GaussianHeatInverseJet(torch.stack(coefficients), base, anchor, residual)


__all__ = ["GaussianHeatInverseJet", "gaussian_heat_inverse_jet"]
