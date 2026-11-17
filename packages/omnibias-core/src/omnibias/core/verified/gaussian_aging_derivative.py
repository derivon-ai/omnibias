# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified physical-age derivatives of Gaussian inverse aging expansions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._chebyshev_calculus import _binomial, _power
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply
from omnibias.core.verified.gaussian_aging import GaussianInverseAging
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Box = tuple[Interval, ...]


def _rising_tail(ratio: Fraction, start: int, order: int, arithmetic: _Arithmetic) -> Fraction:
    r"""Positive exact sum of n-rising-factorial(order)*ratio**n, n>=start.

    The caller validates 0<=ratio<1, start>=1 and nonnegative order within
    its budget. Vandermonde gives k!*r**K times
    sum_{j=0}^k binom(K+k-1,k-j)*r**j/(1-r)**(j+1).
    Streaming the binomial and geometric factors takes O(k+log K) work.
    """
    checked = arithmetic._coefficient
    if not ratio:
        return Fraction(0)
    complement = arithmetic.add(Fraction(1), -ratio)
    geometric = checked(1/complement)
    step = checked(ratio/complement)
    binomial = _binomial(start+order-1, order, arithmetic)
    total = Fraction(0)
    factorial = Fraction(1)
    for j in range(order+1):
        total = arithmetic.add(total, arithmetic.mul(binomial, geometric))
        if j < order:
            factorial = arithmetic.mul(factorial, arithmetic.exact(j+1))
            binomial = checked(arithmetic.mul(binomial, arithmetic.exact(order-j))/(start+j))
            geometric = arithmetic.mul(geometric, step)
    return arithmetic.mul(arithmetic.mul(factorial, _power(ratio, start, arithmetic)), total)


@dataclass(frozen=True, init=False)
class GaussianAgingDerivative:
    r"""Raw d**order X(v,s)/ds**order, holding v, centers, biases and nu fixed.

    ``polynomial_enclosure`` encloses the derivative of the finite aging
    formula at the source age, throughout its target box. ``enclosure`` also
    includes a Euclidean analytic tail, ``approximation_error_bound``. This
    bound comes from the complex aging disk, not differentiation of a value
    error estimate. Derivatives remain nonzero at orders above the truncation
    degree, because the finite formula is a polynomial in inverse age.
    """

    aging: GaussianInverseAging
    order: int
    polynomial_enclosure: Box
    approximation_error_bound: Fraction
    enclosure: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_aging_derivative")


def certify_gaussian_aging_derivative(
    aging: GaussianInverseAging, order: int = 1,
) -> GaussianAgingDerivative | None:
    r"""Certify raw physical-age derivatives without inverse or matrix solves.

    If c_n are the source's scaled inverse-age coefficients, the kth raw age
    derivative of the correction is (-1)**k/s**k * sum n-rising-factorial(k)
    *c_n. Add v at k=1. With r=3*q, the analytic error is at most D/(2*s**k)
    *sum_{n>N} n-rising-factorial(k)*r**n. A positive finite rational formula
    evaluates that tail without cancellation. At k=0 retain the source bound.

    Only the source heat age is enclosed, not an interval of ages. At k=1 this
    certifies the inverse Burgers/hodograph evolution X_s; subtracting v gives
    its nonlinear diffusion. It does not certify target derivatives or an
    independently rounded tensor evaluation. Construction performs no new
    transcendental evaluation and retains the source certificate's provenance.

    Invalid types/orders raise. Exhausted budgets, damaged dependencies or
    nonfinite bounds return None. A certificate digest is not a formal proof.
    """
    if not isinstance(aging, GaussianInverseAging):
        raise TypeError("aging must be a GaussianInverseAging")
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0:
        raise ValueError("order must be nonnegative")
    model = aging.model
    if order > model.budget.max_order:
        return None
    try:
        arithmetic = _Arithmetic(model.budget)
        checked = arithmetic._coefficient
        dimension, degree = model.dimension, aging.order
        arithmetic._entries(16*(degree+order+2)*(dimension+1) + 128)
        arithmetic._work(64*(degree+order+2)*(dimension+1))
        from omnibias.core.proof.certificate import (
            encode_interval,
            make_certificate,
            verify_certificate_digest,
        )

        try:
            intact = verify_certificate_digest(aging.certificate)
        except (TypeError, ValueError, RecursionError):
            return None
        if not intact:
            return None
        dependency = deepcopy(aging.certificate)
        metadata = dependency.get("meta")
        if not isinstance(metadata, Mapping):
            return None
        backend = metadata.get("transcend_backend")
        if backend not in ("not_used", "mpmath", "arb", "mpfr"):
            return None
        if order == 0:
            polynomial, error, enclosure = aging.predictor_enclosure, aging.approximation_error_bound, aging.enclosure
        else:
            zero = Interval.point(0.0)
            polynomial = aging.target_box if order == 1 else (zero,)*dimension
            error = Fraction(0)
            if model.contraction_bound:
                scale = Fraction(1)
                weight = Fraction(1)
                for k in range(1, order+1):
                    scale = checked(scale/model.time)
                    weight = arithmetic.mul(weight, arithmetic.exact(k))
                weight = arithmetic.mul(weight, scale)
                if order % 2:
                    weight = -weight
                for n in range(1, degree+1):
                    factor = _rational(weight)
                    polynomial = tuple(_add(total, _multiply(factor, value))
                                       for total, value in zip(polynomial, aging.correction_coefficients[n], strict=True))
                    if n < degree:
                        weight = checked(arithmetic.mul(weight, arithmetic.exact(n+order))/n)
                radius = checked(Fraction(_sqrt_upper(model.diameter_squared, model.budget))/2)
                tail = _rising_tail(arithmetic.mul(Fraction(3), model.contraction_bound), degree+1, order, arithmetic)
                error = arithmetic.mul(arithmetic.mul(radius, scale), tail)
            upper = _rational(error).hi
            enclosure = tuple(_add(value, Interval(-upper, upper)) for value in polynomial)
        certificate = make_certificate(
            claim="Uniform raw physical-age derivative enclosure for the Gaussian heat inverse.",
            payload={
                "type": "gaussian_inverse_aging_derivative", "aging_certificate": dependency, "order": order,
                "polynomial_enclosure": [encode_interval(value) for value in polynomial],
                "approximation_error_bound": _pair(error), "enclosure": [encode_interval(value) for value in enclosure],
                "derivative": "raw d^order X(v,s)/ds^order at the source age; target, centers, biases and viscosity fixed",
                "argument": "The complex aging disk bounds ||h_n|| by (D/2)*Z^(-n). Since c_n=h_n/s^n, termwise kth physical-age differentiation gives (-1)^k*n-rising-factorial(k)*c_n/s^k. Absolute convergence on a neighborhood of the source age justifies differentiation. Vandermonde and geometric binomial sums evaluate the positive derivative tail exactly. The finite formula is enclosed with outward interval arithmetic. The affine trend contributes v for k=1 and zero for higher orders. Order zero retains the source value enclosure.",
                "error_scope": "Euclidean analytic error plus outward evaluation on the source target box; fixed heat age, no target-derivative or separate tensor-roundoff claim.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": backend, "construction_arithmetic": "rational_and_outward_interval",
                  "transcendental_provenance": "inherited_from_aging"},
        )
        result = object.__new__(GaussianAgingDerivative)
        for name, stored in (("aging", aging), ("order", order), ("polynomial_enclosure", polynomial),
                             ("approximation_error_bound", error), ("enclosure", enclosure), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianAgingDerivative", "certify_gaussian_aging_derivative"]
