# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numerical Clenshaw evaluation of a fixed Gaussian inverse approximation."""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from omnibias.core.verified.gaussian_chebyshev import GaussianInverseApproximation


class GaussianChebyshevResult(NamedTuple):
    """Numerical physical values and domain/arithmetic flags, not certificates."""

    value: Tensor
    valid: Tensor


def _float(value: Fraction | float) -> float:
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _boundary(width: Fraction, single_precision: bool) -> float:
    """Largest selected-dtype number <= exact positive width, if representable."""
    rounded = _float(width)
    if single_precision:
        try:
            rounded = struct.unpack("f", struct.pack("f", rounded))[0]
        except OverflowError:
            return math.inf
    if not math.isfinite(rounded) or rounded <= 0 or Fraction.from_float(rounded) <= width:
        return rounded
    if single_precision:
        bits = struct.unpack("I", struct.pack("f", rounded))[0]
        return float(struct.unpack("f", struct.pack("I", bits - 1))[0])
    return math.nextafter(rounded, 0.0)


def evaluate_gaussian_inverse_approximation(
    parameter: Tensor, approximation: GaussianInverseApproximation,
) -> GaussianChebyshevResult:
    r"""Evaluate the fixed surrogate along ``v(t)=base_target+t*direction``.

    ``parameter`` is a float32/float64 Tensor with arbitrary batch shape.
    Output ``value`` has shape ``(*parameter.shape, dimension)`` and matches
    its dtype/device. Clenshaw evaluates the ordinary Chebyshev coefficients
    of the correction, with no endpoint half factors, at t/half_width. The
    affine part is ``time*(base_target+t*direction)+reference_center``.

    Parameter autograd, including higher derivatives, remains connected.
    Approximation coefficients and model metadata are fixed host constants;
    this function does not differentiate their original construction or model
    parameters. Coefficients are converted to the query dtype on each call.

    ``valid`` requires a finite query inside the exact closed fit interval,
    finite converted metadata and finite output. Invalid values are NaN.
    This is a numerical flag, not a certificate: floating conversion and
    Clenshaw rounding errors are not covered by the core uniform error bound.
    Use the core artifact's rigorous enclosure API for certified evaluation.
    """
    from omnibias.core.verified.gaussian_chebyshev import GaussianInverseApproximation

    if not isinstance(approximation, GaussianInverseApproximation):
        raise TypeError("approximation must be a GaussianInverseApproximation")
    if not isinstance(parameter, Tensor) or parameter.dtype not in (torch.float32, torch.float64):
        raise TypeError("parameter must be a float32 or float64 Tensor")
    base = parameter.new_tensor([_float(value) for value in approximation.base_target])
    direction = parameter.new_tensor([_float(value) for value in approximation.direction])
    reference = parameter.new_tensor([_float(value) for value in approximation.tube.reference_center])
    coefficients = parameter.new_tensor([[_float(value) for value in row] for row in approximation.coefficients])
    time = parameter.new_tensor(_float(approximation.tube.model.time))
    width = parameter.new_tensor(_float(approximation.half_width))
    boundary = parameter.new_tensor(_boundary(approximation.half_width, parameter.dtype == torch.float32))
    metadata_valid = (torch.isfinite(base).all() & torch.isfinite(direction).all() & torch.isfinite(reference).all()
                      & torch.isfinite(coefficients).all() & torch.isfinite(time) & torch.isfinite(width) & (width > 0))
    valid = metadata_valid & torch.isfinite(parameter) & (parameter.abs() <= boundary)
    query = torch.where(valid, parameter, torch.zeros_like(parameter))
    safe_width = torch.where(torch.isfinite(width) & (width > 0), width, torch.ones_like(width))
    scaled = (query / safe_width)[..., None]
    following = parameter.new_zeros((*parameter.shape, len(approximation.base_target)))
    after_following = torch.zeros_like(following)
    for coefficient in reversed(coefficients[1:]):
        current = 2 * scaled * following - after_following + coefficient
        after_following, following = following, current
    correction = scaled * following - after_following + coefficients[0]
    value = time * (base + query[..., None] * direction) + reference + correction
    valid = valid & torch.isfinite(value).all(-1)
    value = torch.where(valid[..., None], value, torch.full_like(value, float("nan")))
    return GaussianChebyshevResult(value, valid)


__all__ = ["GaussianChebyshevResult", "evaluate_gaussian_inverse_approximation"]
