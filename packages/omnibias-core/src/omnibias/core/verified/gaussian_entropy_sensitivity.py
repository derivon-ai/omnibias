# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Target derivatives of locally stable, globally selected Gaussian decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, _posterior_moments
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_entropy_face import (
    GaussianEntropyFace,
    _face_intact,
    _face_source_slots,
)
from omnibias.core.verified.gaussian_entropy_residual import _residual_source_slots
from omnibias.core.verified.gaussian_heat import _finite, _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)


def _matrix_payload(matrix: IntervalMatrix) -> list[list[dict[str, str]]]:
    from omnibias.core.proof.certificate import encode_interval

    return [[encode_interval(value) for value in row] for row in matrix]


def _source_slots(face: GaussianEntropyFace) -> int:
    model = face.residual.unique.domain.model
    return (_face_source_slots(face)+_residual_source_slots(face.residual.unique)
            +256*(model.dimension+1)**2+128*(model.mode_count+1)*(model.dimension+1)+4096)


def _sensitivity_intact(sensitivity: GaussianEntropySensitivity) -> bool:
    from omnibias.core.proof.certificate import certificate_is_unconditional

    if (not _intact(sensitivity.certificate) or not _face_intact(sensitivity.face)
            or not certificate_is_unconditional(sensitivity.certificate)
            or not certificate_is_unconditional(sensitivity.face.certificate)):
        return False
    payload = sensitivity.certificate.get("payload")
    return (isinstance(payload, dict) and payload.get("face_certificate") == sensitivity.face.certificate
            and payload.get("terms") == sensitivity.terms
            and payload.get("projected_covariance") == _matrix_payload(sensitivity.projected_covariance)
            and payload.get("contraction_bound") == _pair(sensitivity.contraction_bound)
            and payload.get("polynomial") == _matrix_payload(sensitivity.polynomial)
            and payload.get("tail_norm_bound") == _pair(sensitivity.tail_norm_bound)
            and payload.get("target_jacobian") == _matrix_payload(sensitivity.target_jacobian))


def _power_step(power: IntervalMatrix, operator: IntervalMatrix) -> IntervalMatrix:
    """The true powers commute with the symmetric operator, so mirror entries."""
    dimension = len(power)
    result = [[_ZERO]*dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i, dimension):
            value = _ZERO
            for k in range(dimension):
                value = _add(value, _multiply(power[i][k], operator[k][j]))
            if i == j:
                value = Interval(max(0.0, value.lo), value.hi)
            result[i][j] = result[j][i] = value
    return tuple(tuple(row) for row in result)


def _response(
    projector: Matrix, polynomial: IntervalMatrix, time: Fraction, tail: Fraction,
    vector: Vector, arithmetic: _Arithmetic,
) -> Box:
    """Exact normal annihilation, then interval polynomial and tangent tail."""
    tangent = _linear(projector, vector, arithmetic)
    if not any(tangent):
        return (_ZERO,)*len(vector)
    norm_squared = _dot(tangent, tangent, arithmetic)
    tail_squared = arithmetic.mul(tail, tail)
    result = []
    for i, row in enumerate(polynomial):
        value = _ZERO
        for entry, coordinate in zip(row, tangent, strict=True):
            if coordinate:
                value = _add(value, _multiply(entry, _rational(arithmetic.mul(time, coordinate))))
        squared_radius = arithmetic.mul(arithmetic.mul(tail_squared, projector[i][i]), norm_squared)
        if squared_radius:
            arithmetic._work(64)
            radius = _sqrt_upper(squared_radius, arithmetic.limits)
            value = _finite(value+Interval(-radius, radius))
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, init=False)
class GaussianEntropySensitivity:
    r"""Target Jacobian of a unique GLOBAL Gaussian minimizer on a stable face.

    Exact geometry is in face.tangent_projector R. The projected covariance is
    formed from exact R*(a_j-a_0), retaining structural normal cancellations.
    For T=R*Cov(a)*R/(2*nu*s), ``polynomial`` encloses sum(T**k,k=0..terms-1)
    at the true optimum and ``contraction_bound`` bounds ||T||_2 below one.
    ``tail_norm_bound`` is an exact rational upper bound on
    s*q**terms/(1-q), with outward interval powers avoiding huge exact powers.
    ``target_jacobian`` encloses J=s*sum(T**k,k>=0)*R.

    The derivative belongs to the exact optimizer, not the numerical solver.
    Strict face margins and positive root curvature imply a smooth global
    optimizer branch on some open target neighborhood. No quantitative radius
    or claimed derivative accuracy follows from the number of terms alone.
    """

    face: GaussianEntropyFace
    terms: int
    projected_covariance: IntervalMatrix
    contraction_bound: Fraction
    polynomial: IntervalMatrix
    tail_norm_bound: Fraction
    target_jacobian: IntervalMatrix
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_sensitivity")

    def vjp(self, cotangent: Sequence[Rational]) -> Box:
        """Enclose the target VJP, with exact zeros for normal cotangents.

        The exact Jacobian is symmetric. Contract the rational projector
        before conversion, then use the stored interval polynomial and tail.
        No new transcendental evaluation, solve or certificate seal is needed;
        a cached unconditional proof remains usable without mpmath. Damaged
        proofs, invalid inputs and exhausted resources raise. Strict/backend
        history is preserved. Work after proof scans is O(d²).
        """
        model, projector = self.face.residual.unique.domain.model, self.face.residual.unique.domain.projector
        _validate_vector(cotangent, model.dimension, "cotangent")
        arithmetic = _Arithmetic(_combined_budget(model.budget, projector.budget))
        slots = _source_slots(self.face)
        arithmetic._entries(slots)
        arithmetic._work(4*slots+256*(model.dimension+1)**2)
        if not _sensitivity_intact(self):
            raise ValueError("entropy sensitivity proof or nested dependency is damaged or conditional")
        vector = tuple(arithmetic.exact(value) for value in cotangent)
        return _response(self.face.tangent_projector, self.polynomial, model.time, self.tail_norm_bound,
                         vector, arithmetic)


def certify_gaussian_entropy_sensitivity(
    face: GaussianEntropyFace, *, terms: int = 16,
) -> GaussianEntropySensitivity | None:
    r"""Bound a stable globally selected decision's target Jacobian by a series.

    Face stability and the source positive curvature justify local implicit
    differentiation. Exact projected modes and original posterior enclosures
    give a symmetric PSD T with norm q<1. Finite interval matrix powers and a
    geometric operator tail enclose s*(I-T)^(-1)*R. Remainder component bounds
    exploit tangent support, and exact R*g contraction preserves normal zeros.
    Interval dependency and a q near one can make the bounds coarse.

    Invalid metadata raises. Conditional/damaged dependencies, exhausted
    combined source budgets, nonfinite arithmetic or failed sufficient bounds
    return None. Rigorous backend and untainted history are required for this
    new certificate. Cached VJP queries use only the existing proof and exact/
    interval arithmetic. No ordinary derivative is inferred at a weak switch.
    """
    if not isinstance(face, GaussianEntropyFace):
        raise TypeError("face must be a GaussianEntropyFace")
    if type(terms) is not int:
        raise TypeError("terms must be an integer excluding bool")
    if terms < 1:
        raise ValueError("terms must be positive")
    model, projector = face.residual.unique.domain.model, face.residual.unique.domain.projector
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    dimension = model.dimension
    try:
        slots = _source_slots(face)
        a._entries(slots)
        a.limits._work(4*slots+512*(terms+1)*(dimension+1)**3
                       +256*(model.mode_count+1)*(dimension+1)**2+4096)
        a._work(4*slots)
        a.exact(terms)
        if not _face_intact(face):
            return None
        from omnibias.core.proof.certificate import certificate_is_unconditional, make_certificate

        if not certificate_is_unconditional(face.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy sensitivity refuses prior conditional libm fallback history")
        tangent = face.tangent_projector
        relative = tuple(tuple(a.add(value, -origin) for value, origin in zip(row, model.centers[0], strict=True))
                         for row in model.centers)
        projected = tuple(_linear(tangent, row, a) for row in relative)
        a._work(256*(model.mode_count+1)*(dimension+1)**2)
        threshold = a.mul(a.mul(Fraction(2), model.viscosity), model.time)
        with _strict():
            converted = tuple(tuple(_rational(value) for value in row) for row in projected)
            covariance = _posterior_moments(face.posterior, converted).covariance
            inverse_threshold = _rational(a._coefficient(1/threshold))
            operator = tuple(tuple(_multiply(value, inverse_threshold) for value in row) for row in covariance)
            trace = a.sum(tuple(a.exact(Fraction(operator[i][i].hi)) for i in range(dimension)))
            contraction = min(face.residual.contraction_bound, trace)
            if not 0 <= contraction < 1:
                return None
            identity = tuple(tuple(_ONE if i == j else _ZERO for j in range(dimension)) for i in range(dimension))
            polynomial = power = identity
            if contraction:
                for _ in range(1, terms):
                    a._work(128*(dimension+1)**3)
                    power = _power_step(power, operator)
                    polynomial = tuple(tuple(_add(left, right) for left, right in zip(row, other, strict=True))
                                       for row, other in zip(polynomial, power, strict=True))
                factor = _rational(contraction).intersect(Interval(0.0, 1.0))
                power_bound = _ONE
                for _ in range(terms):
                    a._work(8)
                    raw = _multiply(power_bound, factor)
                    power_bound = Interval(max(0.0, raw.lo), min(1.0, raw.hi))
                tail = a._coefficient(a.mul(model.time, a.exact(Fraction(power_bound.hi)))
                                      /a.add(Fraction(1), -contraction))
            else:
                tail = Fraction(0)
            columns = tuple(_response(tangent, polynomial, model.time, tail,
                                      tuple(Fraction(int(i == j)) for i in range(dimension)), a)
                            for j in range(dimension))
            jacobian = tuple(tuple(column[i] for column in columns) for i in range(dimension))
            certificate = make_certificate(
                claim="The target Jacobian of the unique global Gaussian decision on its certified stable active face.",
                payload={
                    "type": "gaussian_entropy_sensitivity", "face_certificate": deepcopy(face.certificate),
                    "terms": terms, "projected_covariance": _matrix_payload(covariance),
                    "contraction_bound": _pair(contraction), "polynomial": _matrix_payload(polynomial),
                    "tail_norm_bound": _pair(tail), "target_jacobian": _matrix_payload(jacobian),
                    "argument": "The source face proof gives a locally affine Euclidean projection with exact orthogonal tangent projector R at the true global optimizer. The source covariance bound gives positive spatial curvature at that point. Gaussian smoothness extends positive curvature to an ambient open neighborhood, even if the point is on the boundary of the certified physical box. Strict face margins and the implicit-function theorem give a smooth local target branch. Uniform coercivity for bounded nearby targets and uniqueness at the base force every nearby global minimizer into that branch neighborhood, so this is the unique global optimizer branch on some open target neighborhood. For fixed model and constraints, differentiating tangent stationarity gives J=Z*(Z^T*H*Z)^(-1)*Z^T for any full-rank tangent basis Z, H=I/s-Cov/(2nu*s²). Equivalently J=s*sum_{k>=0}T^k*R with T=R*Cov*R/(2nu*s). Project original relative centers exactly before interval posterior moments to enclose this projected covariance and preserve normal cancellations. Its PSD trace and the source contraction bound give ||T||<=q<1. Finite interval powers enclose the first terms; outward powers followed by exact division give the recorded tail norm bound at least s*q^terms/(1-q). Symmetry and tangent support imply entry errors at most tail*sqrt(R_ii*R_jj); for a query w=R*g they imply component errors at most tail*sqrt(R_ii*||w||²). Exact-comparison square-root bounds are used. The Jacobian is symmetric, so the same expression encloses its VJP; an exact normal cotangent gives w=0 and an exact zero result without numerical cancellation.",
                    "scope": "Model and polyhedral constraints are fixed; only target derivatives of the exact selected optimizer are asserted. Stability holds in an existential open target neighborhood; no numerical perturbation radius, derivative of the finite solver, guaranteed enclosure width from term count, arbitrary parameter gradients, derivative at weak switches or external formal theorem-prover verification is claimed.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropySensitivity)
        for name, value in (("face", face), ("terms", terms), ("projected_covariance", covariance),
                            ("contraction_bound", contraction), ("polynomial", polynomial),
                            ("tail_norm_bound", tail), ("target_jacobian", jacobian), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropySensitivity", "certify_gaussian_entropy_sensitivity"]
