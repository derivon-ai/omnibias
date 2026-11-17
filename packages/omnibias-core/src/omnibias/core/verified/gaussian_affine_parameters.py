# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified original-parameter adjoints for affine Gaussian optimization."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply
from omnibias.core.verified.gaussian_affine_adjoint import GaussianAffineAdjointGeometry
from omnibias.core.verified.gaussian_affine_sensitivity import (
    GaussianAffineTargetVJPCertificate,
    _binary_vector,
    certify_gaussian_affine_target_vjp,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import (
    GaussianInverseCertificate,
    certify_gaussian_inverse,
)
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
ExactMatrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)


def _sum(values: Sequence[Interval]) -> Interval:
    total = _ZERO
    for value in values:
        total = _add(total, value)
    return total


def _dot(left: Box, right: Box) -> Interval:
    return _sum(tuple(_multiply(a, b) for a, b in zip(left, right, strict=True)))


def _action(matrix: ExactMatrix, vector: Box) -> Box:
    return tuple(_dot(tuple(_rational(value) for value in row), vector) for row in matrix)


def _shift(box: Box, reference: Vector, arithmetic: _Arithmetic) -> Box:
    """Subtract exact references before rounding endpoints, retaining zeros."""
    return tuple(Interval(_rational(arithmetic.add(Fraction(value.lo), -offset)).lo,
                          _rational(arithmetic.add(Fraction(value.hi), -offset)).hi)
                 for value, offset in zip(box, reference, strict=True))


def _mean(posterior: Box, rows: ExactMatrix) -> Box:
    return tuple(_sum(tuple(_multiply(p, _rational(row[i])) for p, row in zip(posterior, rows, strict=True)))
                 .intersect(Interval(_rational(min(row[i] for row in rows)).lo,
                                     _rational(max(row[i] for row in rows)).hi))
                 for i in range(len(rows[0])))


@dataclass(frozen=True, init=False)
class GaussianAffineParameterVJP:
    """Enclose all first derivatives of g·X for the fixed declared upstream g.

    Gradients refer to ORIGINAL centers, biases, viscosity, heat age, target,
    constraint matrix and right-hand side. Both proposals are untrusted;
    this certifies derivatives of the true unique constrained optimizer.
    Geometry has full row rank. Boxes can lose correlations and need not
    themselves satisfy exact gradient identities. No component accuracy
    tolerance or higher derivative is claimed.

    ``primal_multiplier`` is None and ``posterior`` is empty when exact
    P*g=0 makes them unnecessary. They are not asserted to be zero. The
    root remains certified, even on this branch. ``root_certificate`` is
    absent only for an exactly affine restricted map, whose root is exact.
    """

    geometry: GaussianAffineAdjointGeometry
    target_certificate: GaussianAffineTargetVJPCertificate
    root_certificate: GaussianInverseCertificate | None
    root_enclosure: Box
    posterior: Box
    primal_multiplier: Box | None
    adjoint_multiplier: Box
    target_gradient: Box
    center_gradient: Matrix
    bias_gradient: Box
    constraint_gradient: Matrix
    rhs_gradient: Box
    viscosity_gradient: Interval
    time_gradient: Interval
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_affine_parameter_vjp")


def certify_gaussian_affine_parameter_vjp(
    geometry: GaussianAffineAdjointGeometry, target: Sequence[int | Fraction], scaled_point: Sequence[float],
    upstream: Sequence[float], proposal: Sequence[float],
) -> GaussianAffineParameterVJP | None:
    r"""Certify seven original-parameter VJP blocks without a query-time solve.

    Use precompiled L=(C*C.T)^(-1)*C. For A=D_x U, the exact root and
    target adjoint z satisfy U-v+C.T*lambda=0, C*x=d and
    A*z+C.T*eta=g, C*z=0. Certified root/target-adjoint boxes plus outward
    Gaussian moments enclose lambda, eta and all parameter derivatives.
    Only the tangent Hessian must be positive: require q_T<1. The ambient
    Hessian may be indefinite. Full row rank C is a compilation precondition
    for derivatives with respect to arbitrary entries of C and d.

    Target entries are exact int/Fraction; scaled_point=x/s, upstream and
    proposal are finite binary64 vectors, as in the target-VJP certificate.
    Upstream is held fixed. Derivatives of an external loss's upstream,
    explicit parameter dependence of that loss and higher derivatives are
    not certified. No inverse iteration, dense covariance or query-time
    matrix factorization occurs. Work/storage after compilation are
    O(m*d+d²+r*d), including certificate composition and the dense projector.

    Invalid inputs raise; exhausted budgets, q_T>=1, damaged dependencies
    or nonfinite bounds return None. A missing rigorous transcendental
    backend raises when a posterior/root evaluation is necessary. Sealed
    evidence is not a theorem-prover verification.
    """
    if not isinstance(geometry, GaussianAffineAdjointGeometry):
        raise TypeError("geometry must be a GaussianAffineAdjointGeometry")
    restriction = geometry.restriction
    model, source = restriction.model, restriction.source_model
    if not isinstance(target, Sequence):
        raise TypeError("target must be a finite sequence of exact integers or Fractions")
    if len(target) != model.dimension:
        raise ValueError("target dimension must match the restriction")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in target):
        raise TypeError("target entries must be exact integers or Fractions, excluding bool")
    point = _binary_vector(scaled_point, model.dimension, "scaled_point")
    direction = _binary_vector(upstream, model.dimension, "upstream")
    proposed = _binary_vector(proposal, model.dimension, "proposal")
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, count, rows = model.dimension, model.mode_count, len(restriction.constraints)
        # Aggregate reservations include the composed target certificate,
        # its residual/root evaluations, another root evaluation if needed,
        # the posterior and all retained dependency/parameter-output data.
        arithmetic._entries(512*(count+rows+1)*(dimension+1) + 512*(dimension+1)**2 + 2048)
        arithmetic._work(4096*(count+rows+1)*(dimension+1) + 4096*(dimension+1)**2
                         + 8*model._evaluation_work() + 8192)
        if not all(_intact(cert) for cert in (geometry.certificate, restriction.certificate)):
            return None
        frozen = tuple(arithmetic.exact(value) for value in target)
        sensitivity = certify_gaussian_affine_target_vjp(restriction, frozen, point, direction, proposed)
        if sensitivity is None or not _intact(sensitivity.certificate):
            return None
        root = None if sensitivity.base_certificate is None else sensitivity.base_certificate.root_certificate
        if model.contraction_bound == 0:
            exact_root = tuple(arithmetic.add(arithmetic.mul(model.time, value), reference)
                               for value, reference in zip(sensitivity.projected_target, model.centers[0], strict=True))
            root_box = tuple(_rational(value) for value in exact_root)
        else:
            if root is None:
                root = certify_gaussian_inverse(model, sensitivity.projected_target, sensitivity.scaled_point)
            if root is None or not _intact(root.certificate):
                return None
            root_box = root.enclosure
        z = sensitivity.enclosure
        exact_g = tuple(checked(Fraction(value)) for value in sensitivity.upstream)
        eta0 = _linear(geometry.left_inverse, exact_g, arithmetic)
        eta = tuple(_rational(value) for value in eta0)
        posterior: Box = ()
        primal: Box | None = None
        center_gradient: Matrix = ((_ZERO,)*dimension,)*count
        bias_gradient: Box = (_ZERO,)*count
        nu_gradient = time_gradient = _ZERO
        if any(sensitivity.projected_upstream):
            s, nu = model.time, model.viscosity
            inv_s = checked(1/s)
            inv_2nus2 = checked(1/arithmetic.mul(arithmetic.mul(Fraction(2), nu), arithmetic.mul(s, s)))
            relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                                   zip(row, source.centers[0], strict=True)) for row in source.centers)
            tangent = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                                  zip(row, model.centers[0], strict=True)) for row in model.centers)
            # Exact center differences remove common spatial/bias gauges.
            # On the feasible root, transformed and original posteriors agree.
            position = _shift(root_box, model.centers[0], arithmetic)
            coordinates = tuple(_multiply(value, _rational(inv_s)) for value in position)
            if model.contraction_bound == 0:
                coordinates = (_ZERO,)*dimension
            with _strict():
                posterior = _probabilities(model._mean_map._logits(coordinates))
            mean = _mean(posterior, relative)
            tangent_mean = _mean(posterior, tangent)
            # Center relative rows by their enclosed means. Structural
            # singleton equality is exact; all other subtractions round out.
            centered = tuple(tuple(_add(_rational(value), -average) for value, average in zip(row, mean, strict=True))
                             for row in relative)
            tangent_centered = tuple(tuple(_add(_rational(value), -average) for value, average in
                                           zip(row, tangent_mean, strict=True)) for row in tangent)
            contractions = tuple(_dot(z, row) for row in tangent_centered)
            weighted = tuple(_multiply(p, value) for p, value in zip(posterior, contractions, strict=True))
            covariance_action = tuple(_sum(tuple(_multiply(weight, row[i]) for weight, row in
                                                  zip(weighted, centered, strict=True))) for i in range(dimension))
            eta_correction = _action(geometry.left_inverse, covariance_action)
            eta = tuple(_add(base, _multiply(_rational(inv_2nus2), correction))
                        for base, correction in zip(eta, eta_correction, strict=True))
            base = tuple(arithmetic.add(value, arithmetic.mul(inv_s, arithmetic.add(reference, -anchor)))
                         for value, reference, anchor in zip(sensitivity.target, source.centers[0],
                                                              restriction.anchor, strict=True))
            lambda0 = _linear(geometry.left_inverse, base, arithmetic)
            lambda_correction = _action(geometry.left_inverse, mean)
            primal = tuple(_add(_rational(value), _multiply(_rational(inv_s), correction))
                           for value, correction in zip(lambda0, lambda_correction, strict=True))
            bias_gradient = tuple(_multiply(weight, _rational(inv_s)) for weight in weighted)
            centers = []
            distance_covariance = _ZERO
            for p, weight, original, tangent_row, cost in zip(posterior, weighted, source.centers,
                                                              tangent, restriction.normal_cost_differences, strict=True):
                displacement = _shift(root_box, original, arithmetic)
                centers.append(tuple(_add(_multiply(_multiply(p, value), _rational(inv_s)),
                                          _multiply(_multiply(weight, delta), _rational(inv_2nus2)))
                                     for value, delta in zip(z, displacement, strict=True)))
                square = arithmetic.sum(tuple(arithmetic.mul(value, value) for value in tangent_row))
                constant = arithmetic.add(square, cost)
                distance = _add(_rational(constant), _multiply(Interval.point(-2),
                    _dot(tuple(_rational(value) for value in tangent_row), position)))
                distance_covariance = _add(distance_covariance, _multiply(weight, distance))
            center_gradient = tuple(centers)
            inv_4nu2s2 = checked(inv_2nus2/arithmetic.mul(Fraction(2), nu))
            inv_4nus3 = checked(inv_2nus2/arithmetic.mul(Fraction(2), s))
            nu_gradient = _multiply(distance_covariance, _rational(inv_4nu2s2))
            time_gradient = _add(_multiply(_dot(z, tuple(_rational(value) for value in sensitivity.projected_target)),
                                           _rational(inv_s)), _multiply(distance_covariance, _rational(inv_4nus3)))
        # When P*g=0, z=0 exactly. Model/target gradients vanish, but
        # eta=L*g and -eta*x.T generally do not; no posterior is needed.
        constraint_gradient = tuple(tuple(-_add(_multiply(multiplier, coordinate),
                                                  _ZERO if primal is None else _multiply(primal[i], adjoint))
                                            for coordinate, adjoint in zip(root_box, z, strict=True))
                                    for i, multiplier in enumerate(eta))
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        def encode_box(box: Box) -> list[dict[str, str]]:
            return [encode_interval(value) for value in box]

        certificate = make_certificate(
            claim="All original-parameter VJP enclosures for the unique affine-constrained Gaussian optimum.",
            payload={
                "type": "gaussian_affine_parameter_vjp", "geometry_certificate": deepcopy(geometry.certificate),
                "target_certificate": deepcopy(sensitivity.certificate),
                "root_certificate": None if root is None else deepcopy(root.certificate),
                "root_enclosure": encode_box(root_box), "posterior": encode_box(posterior),
                "primal_multiplier": None if primal is None else encode_box(primal),
                "adjoint_multiplier": encode_box(eta), "target_gradient": encode_box(z),
                "center_gradient": [encode_box(row) for row in center_gradient], "bias_gradient": encode_box(bias_gradient),
                "constraint_gradient": [encode_box(row) for row in constraint_gradient], "rhs_gradient": encode_box(eta),
                "viscosity_gradient": encode_interval(nu_gradient), "time_gradient": encode_interval(time_gradient),
                "argument": "Full row rank C and positive tangent Hessian make the original KKT matrix invertible even when the ambient Hessian is indefinite. The certified target adjoint z solves A*z+C.T*eta=g, C*z=0. Exact compiled L=(C*C.T)^-1*C gives lambda=L*(v+(mu-anchor)/s), eta=L*g+L*Cov(a)*z/(2*nu*s^2). Differentiating the original Gaussian KKT equations yields grad_v=z, grad_d=eta, grad_C=-lambda*z.T-eta*x.T, grad_bj=pj*z.(aj-mu)/s, grad_aj=pj*z/s+pj*(z.(aj-mu))*(x-aj)/(2*nu*s^2), grad_nu=z.Cov(a,|x-a|^2)/(4*nu^2*s^2), grad_s=z.v/s+z.Cov(a,|x-a|^2)/(4*nu*s^3). Outward posterior intervals on the certified root box and the certified tangent adjoint enclose these formulas. Original center differences, tangent contractions and exact normal-cost differences remove common gauges. When P*g=0, z=0 and all model derivatives vanish, while eta=L*g and grad_C=-eta*x.T still apply.",
                "scope": "First derivatives of g.X with g fixed to the declared binary64 upstream. Original independent centers, biases, positive viscosity/age, target and all entries of full-row-rank C and rhs are varied. No external-upstream error, direct loss parameter dependence, higher derivative or derivative of a finite approximation is certified.",
            },
            honesty={"unproven_claim": False, "unconditional_transcendentals": bool(posterior) or root is not None,
                     "continuum_pde_claim": False},
            meta={"transcend_backend": backend_name() if posterior else (
                "not_used" if root is None else root.certificate["meta"]["transcend_backend"])},
        )
        result = object.__new__(GaussianAffineParameterVJP)
        for name, stored in (("geometry", geometry), ("target_certificate", sensitivity), ("root_certificate", root),
                             ("root_enclosure", root_box), ("posterior", posterior), ("primal_multiplier", primal),
                             ("adjoint_multiplier", eta), ("target_gradient", z), ("center_gradient", center_gradient),
                             ("bias_gradient", bias_gradient), ("constraint_gradient", constraint_gradient), ("rhs_gradient", eta),
                             ("viscosity_gradient", nu_gradient), ("time_gradient", time_gradient), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAffineParameterVJP", "certify_gaussian_affine_parameter_vjp"]
