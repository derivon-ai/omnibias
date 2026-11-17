# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified inverse finite exponential means and global convex minimizers.

For ``Phi(x)=log(sum(exp(k_i.x+a_i)))-target.x``, the gradient is the
exponential mean minus the target and the Hessian is its covariance. Exact
full affine rank of the modes makes this Hessian positive definite everywhere.
A certified gradient zero is consequently the unique GLOBAL minimizer, without
assuming that Newton iteration converges or that every target has an inverse.

The proof combines controlled rigorous interval callbacks with the existing
Krawczyk test. It is not a Lean-kernel theorem or a general PDE existence claim.
Numerical failure is inconclusive; it never establishes infeasibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.linalg import _float_inverse
from omnibias.core.verified.mean_map import MeanMap
from omnibias.core.verified.transcend import (
    backend_name,
    require_rigorous_backend,
    set_strict_backend,
)

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

MeanInverseStatus = Literal["certified", "not_certified", "rank_deficient"]


@dataclass(frozen=True)
class MeanInverseCertificate:
    """A unique global minimizer lying in the rigorous ``enclosure``.

    ``point`` is the approximate tested center, not an asserted exact zero.
    ``box`` contains the actual binary64 endpoints tested by Krawczyk; the
    nominal radius need not give those endpoints exactly in real arithmetic.
    The sealed certificate binds model, rank, and interval source operands.
    Its digest records integrity, not a theorem-prover verification.
    """

    point: tuple[float, ...]
    box: tuple[Interval, ...]
    enclosure: tuple[Interval, ...]
    affine_rank: int
    kappa: float
    certificate: Cert


@dataclass(frozen=True)
class MeanInverseResult:
    """Bounded proposal search; only ``certified`` establishes an inverse."""

    status: MeanInverseStatus
    point: tuple[float, ...]
    iterations: int
    certificate: MeanInverseCertificate | None


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a binary64 float or exactly representable integer")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite and exactly representable") from exc
    if not math.isfinite(result) or result != value:
        raise ValueError(f"{label} must be finite and exactly representable")
    return result


def _point(values: Sequence[float], dimension: int, label: str) -> tuple[float, ...]:
    if len(values) != dimension:
        raise ValueError(f"{label} dimension must match the mean map")
    return tuple(_finite_float(value, label) for value in values)


def _model(model: MeanMap) -> None:
    if not isinstance(model, MeanMap):
        raise TypeError("model must be a MeanMap")


def _matrix(values: Sequence[Sequence[float]], dimension: int) -> tuple[tuple[float, ...], ...]:
    if len(values) != dimension or any(len(row) != dimension for row in values):
        raise ValueError("preconditioner must be square with the mean-map dimension")
    return tuple(_point(row, dimension, "preconditioner") for row in values)


def _radius(value: float) -> float:
    result = _finite_float(value, "radius")
    if result <= 0.0:
        raise ValueError("radius must be positive")
    return result


class _Work:
    def __init__(self, model: MeanMap):
        self.model = model
        self.used = 0

    def charge(self, count: int) -> None:
        self.used += count
        self.model.budget._work(self.used)

    def evaluation(self) -> None:
        self.charge(self.model._evaluation_work())


def _rational(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _finite_krawczyk_arithmetic(
    point: tuple[float, ...], box: tuple[Interval, ...],
    inverse: tuple[tuple[float, ...], ...], residual: tuple[Interval, ...],
    jacobian: tuple[tuple[Interval, ...], ...],
) -> None:
    """Conservative magnitude preflight before any possible infinity reuse.

    Existing interval multiplication is not defined for 0*infinity. Bound
    every intermediate of the existing Krawczyk computation before invoking
    it; outward positive bounds may safely refuse some otherwise usable data.
    """
    def upper(value: float) -> float:
        result = math.nextafter(value, math.inf)
        if not math.isfinite(result):
            raise ArithmeticError("nonfinite Krawczyk arithmetic bound")
        return result

    def add(left: float, right: float) -> float:
        return upper(left + right)

    def multiply(left: float, right: float) -> float:
        return upper(left * right)

    offsets = tuple(axis - Interval.point(center) for axis, center in zip(box, point, strict=True))
    if any(not math.isfinite(value) for axis in offsets for value in (axis.lo, axis.hi)):
        raise ArithmeticError("nonfinite Krawczyk offset")
    for i, row in enumerate(inverse):
        af = 0.0
        mdiff = 0.0
        row_norm = 0.0
        for a, f in zip(row, residual, strict=True):
            af = add(af, multiply(abs(a), max(abs(f.lo), abs(f.hi))))
        for j, offset in enumerate(offsets):
            aj = 0.0
            for a, derivative in zip(row, (entry[j] for entry in jacobian), strict=True):
                aj = add(aj, multiply(abs(a), max(abs(derivative.lo), abs(derivative.hi))))
            entry_bound = add(float(i == j), aj)
            row_norm = add(row_norm, upper(entry_bound))
            mdiff = add(mdiff, multiply(entry_bound, max(abs(offset.lo), abs(offset.hi))))
        add(add(abs(point[i]), af), mdiff)


def _certify(
    model: MeanMap, point: tuple[float, ...], preconditioner: tuple[tuple[float, ...], ...],
    radius: float, work: _Work,
) -> MeanInverseCertificate | None:
    if model.affine_rank != model.dimension:
        return None
    dimension = model.dimension
    try:
        work.charge(4 * dimension + 1)
    except ValueError:
        return None
    # Krawczyk uses these rounded-nearest endpoints. They are a valid tested
    # box, but must not be confused with the exact real center +/- radius.
    endpoints = tuple((coordinate - radius, coordinate + radius) for coordinate in point)
    if any(not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi for lo, hi in endpoints):
        return None
    box = tuple(Interval(lo, hi) for lo, hi in endpoints)
    require_rigorous_backend()
    previous = set_strict_backend(True)
    try:
        try:
            model.budget._entries(4 * model.mode_count * (dimension + 1) + 12 * dimension**2 + 16 * dimension)
            work.charge(12 * dimension**3 + 24 * dimension**2 + 4 * model.mode_count * (dimension + 1))
            center_box = tuple(Interval.point(value) for value in point)
            work.evaluation()
            residual = model.residual(center_box)
            work.evaluation()
            jacobian = model.jacobian(box)
            if any(not math.isfinite(endpoint) for interval in (*residual, *(entry for row in jacobian for entry in row))
                   for endpoint in (interval.lo, interval.hi)):
                return None
            _finite_krawczyk_arithmetic(point, box, preconditioner, residual, jacobian)

            def func(requested: list[Interval]) -> tuple[Interval, ...]:
                if tuple(requested) != center_box:
                    raise ValueError("unexpected Krawczyk residual evaluation box")
                return residual

            def derivative(requested: list[Interval]) -> tuple[tuple[Interval, ...], ...]:
                if tuple(requested) != box:
                    raise ValueError("unexpected Krawczyk Jacobian evaluation box")
                return jacobian

            meta = {"transcend_backend": backend_name()}
            local = krawczyk_certificate(func, derivative, point, preconditioner, radius, certificate_meta=meta)
            if local is None:
                return None
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            enclosure = tuple(Interval(lo, hi) for lo, hi in local.enclosure)
            differences = [[_rational(value - model.modes[0][axis]) for axis, value in enumerate(mode)]
                           for mode in model.modes[1:]]
            certificate = make_certificate(
                claim="Unique global minimizer of the finite exponential potential and unique mean-map inverse.",
                payload={
                    "type": "mean_inverse",
                    "modes": [[_rational(value) for value in mode] for mode in model.modes],
                    "biases": [_rational(value) for value in model.biases],
                    "target": [_rational(value) for value in model.target],
                    "affine_rank": model.affine_rank,
                    "affine_differences": differences,
                    "center": list(point),
                    "tested_box": [encode_interval(interval) for interval in box],
                    "preconditioner": [list(row) for row in preconditioner],
                    "residual_at_center": [encode_interval(interval) for interval in residual],
                    "jacobian_on_box": [[encode_interval(interval) for interval in row] for row in jacobian],
                    "local_krawczyk": local.certificate["payload"],
                    "global_argument": "Full affine rank and positive exponential weights imply a positive-definite covariance Hessian everywhere; a gradient zero is the unique global minimum.",
                },
                honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta=meta,
            )
            return MeanInverseCertificate(point, box, enclosure, model.affine_rank, local.kappa, certificate)
        except (ArithmeticError, ValueError):
            # Every caller-owned type/shape has already been validated. These
            # are numerical or resource failures, never infeasibility evidence.
            return None
    finally:
        set_strict_backend(previous)


def certify_mean_inverse(
    model: MeanMap, point: Sequence[float], preconditioner: Sequence[Sequence[float]], radius: float,
) -> MeanInverseCertificate | None:
    """Certify one proposed inverse, or return an inconclusive ``None``.

    Invalid caller metadata raises. Arithmetic/resource limits, insufficient
    affine rank, a degenerate representable box, and failed Krawczyk inequalities
    are inconclusive. The rigorous transcendental backend is required for an
    attempted full-rank proof; its absence raises ``RuntimeError``. Prior strict
    mode and sticky libm provenance are preserved on every exit.
    """
    _model(model)
    center = _point(point, model.dimension, "point")
    inverse = _matrix(preconditioner, model.dimension)
    extent = _radius(radius)
    return _certify(model, center, inverse, extent, _Work(model))


def inverse_mean_map(
    model: MeanMap, initial: Sequence[float] | None = None, *, max_iterations: int = 64,
    radii: Sequence[float] = (1e-8, 1e-6, 1e-4, 1e-2),
) -> MeanInverseResult:
    """Propose an inverse by bounded damped Newton steps, then certify it.

    No residual tolerance produces success: only a fresh interval certificate
    does. A finite search may fail even when a unique inverse exists. Rank
    deficiency is reported separately and says nothing about feasibility.
    Iterations count accepted Newton steps. Backtracking is capped at 24 trials
    per step; all evaluations, inversions and proof attempts share the model's
    cumulative work budget. Requested search limits are preflighted as metadata.
    """
    _model(model)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be a nonnegative integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if not radii:
        raise ValueError("radii must be nonempty")
    model.budget._entries(len(radii))
    model.budget._work(max_iterations + len(radii))
    extents = tuple(_radius(value) for value in radii)
    point = (0.0,) * model.dimension if initial is None else _point(initial, model.dimension, "initial point")
    if model.affine_rank != model.dimension:
        return MeanInverseResult("rank_deficient", point, 0, None)
    work = _Work(model)
    iterations = 0
    try:
        work.charge(2 * len(extents) + model.dimension)
        for _ in range(max_iterations + 1):
            work.evaluation()
            current = model.float_evaluate(point)
            work.charge(8 * model.dimension**3 + 16 * model.dimension**2)
            proposed = _float_inverse(current.covariance)
            if proposed is None or any(not math.isfinite(value) for row in proposed for value in row):
                break
            inverse = tuple(tuple(row) for row in proposed)
            for radius in extents:
                certified = _certify(model, point, inverse, radius, work)
                model.budget._work(work.used)
                if certified is not None:
                    return MeanInverseResult("certified", point, iterations, certified)
            if iterations >= max_iterations:
                break
            work.charge(2 * model.dimension**2 + 2 * model.dimension)
            step = tuple(math.fsum(a * b for a, b in zip(row, current.residual, strict=True)) for row in inverse)
            decrease = math.fsum(a * b for a, b in zip(current.residual, step, strict=True))
            if not math.isfinite(decrease) or decrease <= 0.0 or not all(math.isfinite(value) for value in step):
                break
            accepted = False
            scale = 1.0
            residual_norm = max(abs(value) for value in current.residual)
            for _ in range(24):
                work.charge(4 * model.dimension + 2)
                trial = tuple(x - scale * dx for x, dx in zip(point, step, strict=True))
                if trial == point:
                    break
                if all(math.isfinite(value) for value in trial):
                    work.evaluation()
                    try:
                        candidate = model.float_evaluate(trial)
                    except ArithmeticError:
                        scale *= 0.5
                        continue
                    if (candidate.reduced_objective <= current.reduced_objective - 1e-4 * scale * decrease or
                            (candidate.reduced_objective <= current.reduced_objective and
                             max(abs(value) for value in candidate.residual) < residual_norm)):
                        point = trial
                        iterations += 1
                        accepted = True
                        break
                scale *= 0.5
            if not accepted:
                break
    except (ArithmeticError, ValueError):
        pass
    return MeanInverseResult("not_certified", point, iterations, None)


__all__ = ["MeanInverseCertificate", "MeanInverseResult", "MeanInverseStatus", "certify_mean_inverse", "inverse_mean_map"]
