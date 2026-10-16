# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible Clenshaw evaluation of fixed Gaussian inverse approximations."""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from omnibias.core.verified.gaussian_chebyshev import GaussianInverseApproximation


class GaussianChebyshevResult(NamedTuple):
    """Numerical physical values and domain/arithmetic flags, not certificates."""

    value: Array
    valid: Array


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
    parameter: Array, approximation: GaussianInverseApproximation,
) -> GaussianChebyshevResult:
    r"""Evaluate the fixed surrogate along ``v(t)=base_target+t*direction``.

    ``parameter`` is float32/float64 with arbitrary batch shape. Output value
    has shape ``(*parameter.shape, dimension)`` and the same dtype. Clenshaw
    evaluates the ordinary Chebyshev correction at t/half_width, with no
    endpoint half factors, then adds the affine physical inverse term.

    Close over ``approximation`` when using ``jit``. The query stays a runtime
    argument supporting ``vmap`` and higher parameter derivatives. Coefficients
    and model metadata are fixed host constants; gradients through their
    original construction or the original model parameters are not supplied.

    ``valid`` requires a finite query inside the exact closed fit interval,
    finite converted metadata and finite output. Invalid values are NaN.
    This is a numerical flag, not a certificate: floating conversion and
    Clenshaw rounding errors are not covered by the core uniform error bound.
    Use the core artifact's rigorous enclosure API for certified evaluation.
    """
    from omnibias.core.verified.gaussian_chebyshev import GaussianInverseApproximation

    if not isinstance(approximation, GaussianInverseApproximation):
        raise TypeError("approximation must be a GaussianInverseApproximation")
    parameter = jnp.asarray(parameter)
    if parameter.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("parameter must have float32 or float64 dtype")
    base = jnp.asarray([_float(value) for value in approximation.base_target], dtype=parameter.dtype)
    direction = jnp.asarray([_float(value) for value in approximation.direction], dtype=parameter.dtype)
    reference = jnp.asarray([_float(value) for value in approximation.tube.reference_center], dtype=parameter.dtype)
    coefficients = jnp.asarray([[_float(value) for value in row] for row in approximation.coefficients], dtype=parameter.dtype)
    time = jnp.asarray(_float(approximation.tube.model.time), dtype=parameter.dtype)
    width = jnp.asarray(_float(approximation.half_width), dtype=parameter.dtype)
    boundary = jnp.asarray(_boundary(approximation.half_width, parameter.dtype == jnp.float32), dtype=parameter.dtype)
    metadata_valid = (jnp.isfinite(base).all() & jnp.isfinite(direction).all() & jnp.isfinite(reference).all()
                      & jnp.isfinite(coefficients).all() & jnp.isfinite(time) & jnp.isfinite(width) & (width > 0))
    valid = metadata_valid & jnp.isfinite(parameter) & (jnp.abs(parameter) <= boundary)
    query = jnp.where(valid, parameter, 0.0)
    safe_width = jnp.where(jnp.isfinite(width) & (width > 0), width, 1.0)
    scaled = (query / safe_width)[..., None]
    following = jnp.zeros((*parameter.shape, len(approximation.base_target)), dtype=parameter.dtype)
    after_following = jnp.zeros_like(following)
    for coefficient in reversed(coefficients[1:]):
        current = 2 * scaled * following - after_following + coefficient
        after_following, following = following, current
    correction = scaled * following - after_following + coefficients[0]
    value = time * (base + query[..., None] * direction) + reference + correction
    valid = valid & jnp.isfinite(value).all(-1)
    value = jnp.where(valid[..., None], value, jnp.nan)
    return GaussianChebyshevResult(value, valid)


__all__ = ["GaussianChebyshevResult", "evaluate_gaussian_inverse_approximation"]
