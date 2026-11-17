# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable Gaussian weak measurements of planar distributional fields."""

from __future__ import annotations

from math import pi, sqrt

from omnibias.core.polynomials import hermite_coeffs
from omnibias.core.ridge_distribution import RidgeDistribution, gaussian_pairing_orders

import torch
from torch import Tensor


def gaussian_weak(
    field: RidgeDistribution,
    mean: Tensor,
    variance: float | Tensor,
    bias: float | Tensor | None = None,
) -> Tensor:
    r"""Pair the field with normalized Gaussian probes; never sample a Dirac.

    For a probe N(mu,Sigma), supply ``mean=w@mu`` and
    ``variance=w@Sigma@w`` using the field's direction w. The default bias
    is the field metadata; a scalar tensor override remains trainable.
    Mean/variance broadcast, producing shape ``broadcast_shape+(outputs,)``.
    Projecting probe parameters is the caller's responsibility. In particular,
    no extra surface Jacobian is applied: delta(w.x+b) already has its
    pullback normalization, so rescaling w changes the measured density.

    The recurrence is analytic in real arithmetic. This floating computation
    is not an enclosure; high orders, extreme tails and tiny variances can
    overflow or lose relative accuracy (regular moments can cancel).
    Variance must be finite and positive; nonpositive values produce NaNs
    without a data-dependent host branch. Inputs must be real and finite.
    Bias gradients and arbitrary-order weak derivatives remain connected.
    This integrates over R^d, not a bounded domain or its boundary.
    """
    regular_order, _ = gaussian_pairing_orders(field)
    if not isinstance(mean, Tensor) or not mean.is_floating_point():
        raise TypeError("mean must be a real floating tensor")
    var = torch.as_tensor(variance, dtype=mean.dtype, device=mean.device)
    offset = torch.as_tensor(float(field.bias) if bias is None else bias,
                             dtype=mean.dtype, device=mean.device)
    if offset.ndim != 0:
        raise ValueError("bias must be scalar")
    m: Tensor
    m, var = torch.broadcast_tensors(mean + offset, var)  # type: ignore[no-untyped-call]
    var = torch.where((var > 0) & torch.isfinite(var), var, torch.full_like(var, float("nan")))
    scale = torch.sqrt(var)
    standardized = m / scale
    density = torch.exp(-0.5 * standardized**2) / (sqrt(2 * pi) * scale)
    regular: list[Tensor] = []
    if regular_order >= 0:
        regular.append(0.5 * torch.erfc(-standardized / sqrt(2)))
    if regular_order >= 1:
        regular.append(m * regular[0] + var * density)
    for degree in range(2, regular_order + 1):
        regular.append((m * regular[-1] + var * regular[-2]) / degree)
    # Include runtime inputs even for the identically zero field.
    result: Tensor = (m * 0 + var * 0)[..., None].expand(*m.shape, field.outputs)
    for index, row in field.terms:
        if index >= 0:
            value = regular[index]
        else:
            order = -index - 1
            polynomial = torch.zeros_like(m)
            for coefficient in reversed(hermite_coeffs(order)):
                polynomial = polynomial * standardized + coefficient
            value = (-1)**order * polynomial * density / scale**order
        coefficient_row = mean.new_tensor([float(coefficient) for coefficient in row])
        result = result + value[..., None] * coefficient_row
    return result


__all__ = ["gaussian_weak"]
