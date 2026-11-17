# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigorous Gaussian posterior moments throughout a physical coordinate box.

This internal helper returns finite interval enclosures, not a standalone
certificate. Its consumer owns the resulting mathematical claim and sealing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _finite, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import _probabilities, _strict

Vector = tuple[Interval, ...]
Matrix = tuple[Vector, ...]
_ZERO = Interval.point(0.0)


@dataclass(frozen=True)
class GaussianMoments:
    """Posterior and physical-center moments at every point in the input box.

    ``relative_mean`` encloses E[a-a0], where a0=model.centers[0]. Covariance
    is E[(a-mu)(a-mu)^T], and ``third_trace`` is the vector
    E[(a-mu)*||a-mu||_2²]. The latter contracts two third-moment indices
    without allocating a dimension-cubed tensor. All tuples are immutable.
    """

    posterior: Vector
    relative_mean: Vector
    covariance: Matrix
    third_trace: Vector


def _zero(value: Interval) -> bool:
    return value.lo == value.hi == 0.0


def _add(left: Interval, right: Interval) -> Interval:
    if _zero(left):
        return right
    if _zero(right):
        return left
    return _finite(left+right)


def _multiply(left: Interval, right: Interval) -> Interval:
    if _zero(left) or _zero(right):
        return _ZERO
    return _finite(left*right)


def _square(value: Interval) -> Interval:
    if _zero(value):
        return _ZERO
    result = _finite(value**2)
    return Interval(max(0.0, result.lo), result.hi)


def _posterior_moments(posterior: Vector, converted: Matrix, *, covariance: bool = True) -> GaussianMoments:
    """Moments of enclosed relative centers/weights; caller preflights resources.

    Exact positive weights represented by ``posterior`` must sum to one.
    With covariance=False, its returned matrix is empty and work is O(m*d).
    """
    dimension = len(converted[0])
    mean = []
    for axis in range(dimension):
        total = _ZERO
        for probability, row in zip(posterior, converted, strict=True):
            total = _add(total, _multiply(probability, row[axis]))
        hull = Interval(min(row[axis].lo for row in converted), max(row[axis].hi for row in converted))
        mean.append(total.intersect(hull))
    cov = [[_ZERO]*dimension for _ in range(dimension)] if covariance else []
    third_trace = [_ZERO]*dimension
    for probability, row in zip(posterior, converted, strict=True):
        centered = tuple(_ZERO if value.lo == value.hi == center.lo == center.hi else _finite(value-center)
                         for value, center in zip(row, mean, strict=True))
        squares = tuple(_square(value) for value in centered)
        norm_squared = _ZERO
        for value in squares:
            norm_squared = _add(norm_squared, value)
        norm_squared = Interval(max(0.0, norm_squared.lo), norm_squared.hi)
        trace_factor = _multiply(probability, norm_squared)
        for a in range(dimension):
            third_trace[a] = _add(third_trace[a], _multiply(trace_factor, centered[a]))
            if covariance:
                for b in range(a, dimension):
                    product = squares[a] if a == b else _multiply(centered[a], centered[b])
                    cov[a][b] = _add(cov[a][b], _multiply(probability, product))
    if covariance:
        for a in range(dimension):
            diagonal = cov[a][a]
            cov[a][a] = Interval(max(0.0, diagonal.lo), diagonal.hi)
            for b in range(a+1, dimension):
                cov[b][a] = cov[a][b]
    return GaussianMoments(posterior, tuple(mean), tuple(tuple(row) for row in cov), tuple(third_trace))


def gaussian_moments(
    model: GaussianHeatMap, box: Sequence[Interval], *, arithmetic: _Arithmetic | None = None,
) -> GaussianMoments:
    """Enclose posterior, relative mean, covariance and third trace uniformly.

    Coordinates are physical x, with finite binary64 interval endpoints (or
    exactly representable integers). Model metadata stays exact: cached
    logits use (a-a0).(x-a0)/(2*nu*s)+(b-b0)-|a-a0|²/(4*nu*s), removing
    common biases and forming center differences before binary conversion.
    Existing rigorous softmax normalization handles underflow of lower bounds.

    An optional shared arithmetic meter must use the model's budget. Storage
    is O(m*d+d²) and work O(m*d²), with aggregate preflight and checked exact
    intermediates. Nonfinite enclosures raise ArithmeticError, budget failures
    raise ValueError, and a missing rigorous backend raises RuntimeError.
    Strict backend state is restored and sticky fallback history is preserved.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(box, Sequence):
        raise TypeError("box must be a finite sequence of Interval objects")
    if arithmetic is not None and not isinstance(arithmetic, _Arithmetic):
        raise TypeError("arithmetic must be a shared _Arithmetic meter")
    if arithmetic is not None and arithmetic.limits != model.budget:
        raise ValueError("shared arithmetic must use the model budget")
    meter = _Arithmetic(model.budget) if arithmetic is None else arithmetic
    count, dimension = model.mode_count, model.dimension
    meter._entries(16*count*dimension + 16*count + 16*dimension*dimension + 32*dimension + 64)
    meter._work(128*(count+1)*(dimension+1)**2)
    frozen = model._box(box)
    relative = tuple(tuple(meter.add(value, -reference) for value, reference in zip(row, model.centers[0], strict=True))
                     for row in model.centers)
    converted = tuple(tuple(_rational(value) for value in row) for row in relative)
    with _strict():
        # Identical centers make the posterior independent of x; avoid an
        # unnecessary conversion of a possibly enormous common center.
        coordinates = (_ZERO,)*dimension if model.diameter_squared == 0 else model._centered(frozen, physical=True)
        posterior = _probabilities(model._mean_map._logits(coordinates))
        return _posterior_moments(posterior, converted)


__all__ = ["GaussianMoments", "gaussian_moments"]
