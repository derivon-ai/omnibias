# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable Gaussian heat-velocity inverses with implicit Torch gradients."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor


class GaussianHeatInverseResult(NamedTuple):
    """Numerical inverse and diagnostics; these are not interval certificates."""

    value: Tensor
    converged: Tensor
    residual_bound: Tensor
    contraction_bound: Tensor


def _parameters(centers: Tensor, biases: Tensor, viscosity: Tensor, time: Tensor) -> tuple[Tensor, ...]:
    valid = (torch.isfinite(centers).all() & torch.isfinite(biases).all()
             & torch.isfinite(viscosity) & (viscosity > 0) & torch.isfinite(time) & (time > 0))
    nu = torch.where(torch.isfinite(viscosity) & (viscosity > 0), viscosity, torch.ones_like(viscosity))
    s = torch.where(torch.isfinite(time) & (time > 0), time, torch.ones_like(time))
    clean_centers = torch.where(torch.isfinite(centers), centers, torch.zeros_like(centers))
    clean_biases = torch.where(torch.isfinite(biases), biases, torch.zeros_like(biases))
    relative_raw, bias_raw = clean_centers - clean_centers[0], clean_biases - clean_biases[0]
    relative = torch.where(torch.isfinite(relative_raw), relative_raw, torch.zeros_like(relative_raw))
    offsets = torch.where(torch.isfinite(bias_raw), bias_raw, torch.zeros_like(bias_raw))
    squares_raw = relative.square().sum(-1)
    squares = torch.where(torch.isfinite(squares_raw), squares_raw, torch.zeros_like(squares_raw))
    denominator_raw = 4 * nu * s
    valid = (valid & torch.isfinite(relative_raw).all() & torch.isfinite(bias_raw).all()
             & torch.isfinite(squares_raw).all() & torch.isfinite(denominator_raw) & (denominator_raw > 0))
    denominator = torch.where(torch.isfinite(denominator_raw) & (denominator_raw > 0), denominator_raw,
                              torch.ones_like(denominator_raw))
    phase_raw = offsets - squares / denominator
    valid = valid & torch.isfinite(phase_raw).all()
    phase = torch.where(torch.isfinite(phase_raw), phase_raw, torch.zeros_like(phase_raw))
    return clean_centers, relative, squares, phase, nu, s, valid


def _step(z: Tensor, target: Tensor, relative: Tensor, phase: Tensor, nu: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    logits = z @ relative.mT / (2 * nu) + phase
    valid = torch.isfinite(logits).all(-1)
    safe_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    mean = torch.softmax(safe_logits, dim=-1) @ relative
    proposed = target + mean / time
    valid = valid & torch.isfinite(proposed).all(-1)
    return torch.where(torch.isfinite(proposed), proposed, torch.zeros_like(proposed)), valid


def _norm(value: Tensor) -> Tensor:
    scale = value.abs().amax(dim=-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * torch.sqrt((value / safe_scale[..., None]).square().sum(dim=-1))


def _solve_adjoint(probabilities: Tensor, centered: Tensor, nu: Tensor, time: Tensor,
                   gradient: Tensor, valid: Tensor, mode: str) -> Tensor:
    # Avoid sqrt'(0) contaminating higher derivatives after softmax underflow.
    positive = probabilities > 0
    root = torch.sqrt(torch.where(positive, probabilities, torch.ones_like(probabilities)))
    root = torch.where(positive, root, torch.zeros_like(root))
    b = root[..., :, None] * centered / torch.sqrt(2 * nu * time)
    b = torch.where(valid[..., None, None], b, torch.zeros_like(b))
    dimension, modes = centered.shape[-1], centered.shape[-2]
    low_rank = mode == "low_rank" or (mode == "auto" and modes < dimension)
    if low_rank:
        matrix = torch.eye(modes, dtype=b.dtype, device=b.device) - b @ b.mT
        right = (b @ gradient[..., None])
        solved, info = torch.linalg.solve_ex(matrix, right)
        value = gradient + (b.mT @ solved).squeeze(-1)
    else:
        matrix = torch.eye(dimension, dtype=b.dtype, device=b.device) - b.mT @ b
        solved, info = torch.linalg.solve_ex(matrix, gradient[..., None])
        value = solved.squeeze(-1)
    return torch.where((info == 0)[..., None] & torch.isfinite(value), value, torch.full_like(value, float("nan")))


class _GaussianInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, target: Tensor, centers: Tensor, biases: Tensor, viscosity: Tensor, time: Tensor,
                iterations: int, tolerance: float, adjoint: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        clean_centers, relative, _, phase, nu, s, shared_valid = _parameters(centers, biases, viscosity, time)
        # No d-by-d allocation is needed to compute the exact finite-set diameter.
        distances = torch.stack([(relative - row).square().sum(-1).amax() for row in relative])
        q_denominator = 8 * nu * s
        q_raw = distances.amax() / q_denominator
        q_valid = shared_valid & torch.isfinite(q_denominator) & (q_denominator > 0) & torch.isfinite(q_raw)
        q = torch.where(q_valid, q_raw, torch.full_like(q_raw, float("inf")))
        valid = shared_valid & (q < 1) & torch.isfinite(target).all(-1)
        safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        z = safe_target
        for _ in range(iterations):
            z, step_valid = _step(z, safe_target, relative, phase, nu, s)
            valid = valid & step_valid
        image, step_valid = _step(z, safe_target, relative, phase, nu, s)
        denominator = torch.where(q < 1, 1 - q, torch.ones_like(q))
        estimate = s * _norm(z - image) / denominator
        physical = clean_centers[0] + s * z
        valid = valid & step_valid & torch.isfinite(estimate) & torch.isfinite(physical).all(-1)
        residual = torch.where(valid, estimate, torch.full_like(estimate, float("inf")))
        converged = valid & (residual <= tolerance)
        value = torch.where(converged[..., None], physical, torch.full_like(physical, float("nan")))
        ctx.save_for_backward(value, target, centers, biases, viscosity, time, converged)
        ctx.adjoint = adjoint
        ctx.mark_non_differentiable(converged, residual, q)
        return value, converged, residual, q

    @staticmethod
    def backward(ctx: Any, gradient: Tensor, _converged: Tensor | None, _residual: Tensor | None,
                 _contraction: Tensor | None) -> tuple[Tensor | None, ...]:
        value, target, centers, biases, viscosity, time, converged = ctx.saved_tensors
        clean_centers, relative, squares, phase, nu, s, _ = _parameters(centers, biases, viscosity, time)
        safe_value = torch.where(converged[..., None], value, torch.zeros_like(value))
        z_raw = (safe_value - clean_centers[0]) / s
        z = torch.where(torch.isfinite(z_raw), z_raw, torch.zeros_like(z_raw))
        logits = z @ relative.mT / (2 * nu) + phase
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        probabilities = torch.softmax(logits, dim=-1)
        mean = probabilities @ relative
        centered = relative - mean[..., None, :]
        adjoint = _solve_adjoint(probabilities, centered, nu, s, gradient, converged, ctx.adjoint)
        projection = (centered * adjoint[..., None, :]).sum(-1)
        bias_gradient = probabilities * projection
        center_gradient = (probabilities[..., :, None] * adjoint[..., None, :]
                           + bias_gradient[..., :, None] * (z[..., None, :] - relative / s) / (2 * nu))
        # Subtract the common squared distance analytically. Its coefficient
        # sum is zero, and retaining it would amplify cancellation/overflow.
        energy = squares / (4 * nu * s) - z @ relative.mT / (2 * nu)
        energy_gradient = (bias_gradient * energy).sum(-1)
        target_gradient = s * adjoint
        viscosity_gradient = energy_gradient / nu
        safe_target = torch.where(converged[..., None], target, torch.zeros_like(target))
        time_gradient = (adjoint * safe_target).sum(-1) + energy_gradient / s
        # A masked-out failed sample contributes zero; differentiating a used
        # failed value is undefined and is reported explicitly as NaN.
        failed = ~converged & (gradient != 0).any(-1)
        target_gradient = torch.where(failed[..., None], torch.full_like(target_gradient, float("nan")), target_gradient)
        center_gradient = torch.where(failed[..., None, None], torch.full_like(center_gradient, float("nan")), center_gradient)
        bias_gradient = torch.where(failed[..., None], torch.full_like(bias_gradient, float("nan")), bias_gradient)
        viscosity_gradient = torch.where(failed, torch.full_like(viscosity_gradient, float("nan")), viscosity_gradient)
        time_gradient = torch.where(failed, torch.full_like(time_gradient, float("nan")), time_gradient)
        return (target_gradient, center_gradient.sum_to_size(centers.shape), bias_gradient.sum_to_size(biases.shape),
                viscosity_gradient.sum(), time_gradient.sum(), None, None, None)


def _tensor(value: Any, reference: Tensor, name: str) -> Tensor:
    original = torch.as_tensor(value)
    if original.dtype == torch.bool or original.is_complex():
        raise TypeError(f"{name} must contain real non-boolean numbers")
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def gaussian_heat_inverse(
    target: Tensor, centers: Tensor | Sequence[Sequence[float]], biases: Tensor | Sequence[float],
    viscosity: Tensor | float = 1.0, time: Tensor | float = 1.0, *, max_iterations: int = 128,
    tolerance: float | None = None, adjoint: str = "auto",
) -> GaussianHeatInverseResult:
    r"""Invert U(x)=(x-E[a])/s numerically, with implicit parameter gradients.

    ``target`` has shape ``(...,d)`` and dtype float32/float64. Shared centers
    ``(m,d)``, biases ``(m,)`` and scalar positive viscosity/time are converted
    to its dtype/device without detaching parameter gradients. Mixture weights
    are ``exp(biases)``. Forward iteration uses centered ``z=(x-a0)/s``;
    its tape is discarded. Backward solves ``(I-Cov(a)/(2*nu*s))*lambda=g``
    and supports double backward by recomputing from the saved solution.
    These are implicit solution derivatives, not derivatives of truncated
    iteration; their accuracy depends on the numerical forward residual.

    ``adjoint='auto'`` chooses the smaller of d-dimensional dense and m-mode
    Woodbury solves. ``'low_rank'`` never forms a d-by-d matrix. All parameters
    are differentiable; diagnostics are not. Failed values have undefined
    gradients. Forward-mode torch.func transforms are not supplied here.

    The numerical global bound is q=max|a_i-a_j|²/(8*nu*s). A nonfinite input,
    nonpositive scalar, q>=1, arithmetic failure or exhausted tolerance gives
    ``converged=False`` and NaN value. ``residual_bound`` is the numerical
    physical L2 residual/(1-q) estimate, or infinity for invalid arithmetic.
    It is NOT an outward-rounded certificate. Default absolute tolerance is
    ``64*finfo(dtype).eps``. ``max_iterations`` is a static nonnegative bound.
    """
    if not isinstance(target, Tensor) or target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be a float32 or float64 Tensor")
    if target.ndim < 1 or target.shape[-1] < 1:
        raise ValueError("target must have shape (..., positive dimension)")
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if adjoint not in ("auto", "dense", "low_rank"):
        raise ValueError("adjoint must be auto, dense or low_rank")
    if tolerance is None:
        tolerance = 64 * torch.finfo(target.dtype).eps
    elif type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive scalar")
    locations = _tensor(centers, target, "centers")
    offsets = _tensor(biases, target, "biases")
    nu, s = _tensor(viscosity, target, "viscosity"), _tensor(time, target, "time")
    if locations.ndim != 2 or locations.shape[0] < 1 or locations.shape[1] != target.shape[-1]:
        raise ValueError("centers must have shape (positive mode count, target dimension)")
    if offsets.shape != (locations.shape[0],):
        raise ValueError("biases must have one entry per center")
    if nu.ndim or s.ndim:
        raise ValueError("viscosity and time must be scalars")
    result = cast(tuple[Tensor, Tensor, Tensor, Tensor],
                  _GaussianInverse.apply(target, locations, offsets, nu, s, max_iterations, float(tolerance), adjoint))
    return GaussianHeatInverseResult(*result)


__all__ = ["GaussianHeatInverseResult", "gaussian_heat_inverse"]
