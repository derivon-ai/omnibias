# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rational analytic-domain and Taylor-tail proofs for Gaussian inverse jets.

The proof uses a complex contraction bound; positive real mixture weights do
not imply an entire inverse. Exact stored series data and the complete target
polynomial are bound into the certificate. No tensor or transcendental backend
is used, and serialization is not formal verification by a theorem prover.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.gaussian_jet import GaussianInverseSeries
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction


def _exact(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("analytic parameters must be exact int or Fraction, excluding bool")
    return Fraction(value)


def _power(value: Fraction, exponent: int, budget: RidgeBudget) -> Fraction:
    result = Fraction(1)
    for _ in range(exponent):
        result = budget._coefficient(result * value)
    return result


@dataclass(frozen=True, init=False)
class GaussianSeriesCertificate:
    """Analytic radius and exact remainder formula for an immutable series.

    ``parameter_radius=None`` denotes an entire polynomial inverse (identical
    centers or a constant target path). Otherwise ``spatial_radius`` bounds
    the displacement from the exact base point on the certified complex disk.
    Only :func:`certify_gaussian_inverse_series` constructs these records.
    """

    series: GaussianInverseSeries
    parameter_radius: Fraction | None
    spatial_radius: Fraction | None
    complex_contraction_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_series")

    def tail_bound(self, radius: Rational) -> Fraction:
        """Exact Euclidean remainder bound on ``|t| <= radius``.

        The finite analytic case requires ``0 <= radius < parameter_radius``.
        Exact coefficient/work-budget failures raise rather than return an
        unbounded or uncertified value.
        """
        rho = _exact(radius)
        if rho < 0:
            raise ValueError("evaluation radius must be nonnegative")
        budget = self.series.budget
        budget._coefficient(rho)
        budget._work(16 * (self.series.order + 1) + 32)
        if self.parameter_radius is None:
            return Fraction(0)
        if rho >= self.parameter_radius:
            raise ValueError("evaluation radius must lie strictly inside the certified parameter disk")
        assert self.spatial_radius is not None
        checked = budget._coefficient
        ratio = checked(rho / self.parameter_radius)
        power = _power(ratio, self.series.order + 1, budget)
        return checked(checked(self.spatial_radius * power) / checked(1 - ratio))

    def enclose(self, parameter: Rational) -> tuple[Interval, ...]:
        """Enclose the true physical inverse at one exact real parameter.

        The exact rational polynomial is evaluated before adding the exact
        Euclidean tail to each coordinate. Endpoint conversion rounds outward;
        overflow or exhausted budgets raise explicitly.
        """
        value = _exact(parameter)
        dimension = len(self.series.base_point)
        self.series.budget._entries(8 * dimension + 16)
        self.series.budget._work(32 * (dimension + 1) * (self.series.order + 1) + 64)
        tail = self.tail_bound(abs(value))
        polynomial = self.series.evaluate(value)
        checked = self.series.budget._coefficient
        return tuple(Interval(_rational(checked(center - tail)).lo,
                              _rational(checked(center + tail)).hi) for center in polynomial)


def certify_gaussian_inverse_series(
    series: GaussianInverseSeries, *, parameter_radius: Rational | None = None,
) -> GaussianSeriesCertificate | None:
    r"""Prove a complex analytic disk and Taylor remainder for exact inverse jets.

    ``series.target_coefficients`` defines a polynomial, with no unspecified
    higher coefficients. Write D for the center diameter and q for the real
    contraction bound. For 0<q<1, choose a checked upper B on sqrt(D²*(1-q))
    and set r=2*nu*s*(1-q)/B. On the complex spatial ball of radius r,
    the phase half-width eta=D*||Im(h)||/(4*nu*s) obeys eta²<=(1-q)/4.
    Positive weights after any real shift give a rotated partition modulus
    at least cos(eta). A pairwise covariance identity and Cauchy--Schwarz
    then bound the complex derivative by q/cos²(eta). Since
    cos²(eta)>=1-eta²>=(3+q)/4, kappa=4*q/(3+q)<1 is rational and rigorous.
    A supplied positive rational R is accepted only if
    s*sum_{n>=1} ||v_n||_upper*R^n <= (1-kappa)*r.

    The automatic radius is min(1,(1-kappa)*r/(s*sum ||v_n||_upper)). Identical
    centers (q=0) and constant target paths have exact polynomial inverses and
    receive no finite domain restriction. Explicit R is validated but unnecessary
    in these entire cases. Invalid types/domains raise; inapplicable contraction,
    arithmetic/resource failure or an unproved supplied radius returns None.
    """
    from omnibias.core.gaussian_jet import GaussianInverseSeries

    if not isinstance(series, GaussianInverseSeries):
        raise TypeError("series must be a GaussianInverseSeries")
    requested = None if parameter_radius is None else _exact(parameter_radius)
    if requested is not None and requested <= 0:
        raise ValueError("parameter_radius must be strictly positive")
    q = series.contraction_bound
    if q >= 1:
        return None
    budget = series.budget
    checked = budget._coefficient
    try:
        count, dimension, order = len(series.centers), len(series.base_point), series.order
        # Include the retained exact artifact, rational-pair serialization,
        # scratch norms, and returned proof metadata in the aggregate preflight.
        entries = count * dimension + count + dimension + 2 * (order + 1) * dimension
        budget._entries(4 * entries + 16 * (order + 1) + 64)
        budget._work(64 * (count + order + 2) * (dimension + 1) + 128 * (order + 1))
        if requested is not None:
            checked(requested)
        entire = q == 0 or not any(any(row) for row in series.target_coefficients[1:])
        norms: list[Fraction] = []
        denominator_squared: Fraction | None = None
        denominator_upper: Fraction | None = None
        phase_squared_bound: Fraction | None = None
        spatial: Fraction | None = None
        radius: Fraction | None = None
        kappa, target_growth, available = Fraction(0), Fraction(0), Fraction(0)
        if not entire:
            margin = checked(1 - q)
            denominator_squared = checked(series.diameter_squared * margin)
            denominator_upper = checked(Fraction(_sqrt_upper(denominator_squared, budget)))
            scale = checked(checked(2 * series.viscosity) * series.time)
            spatial = checked(checked(scale * margin) / denominator_upper)
            phase_squared_bound = checked(margin / 4)
            kappa = checked(checked(4 * q) / checked(3 + q))
            if not 0 <= kappa < 1 or spatial <= 0:
                return None
            total = Fraction(0)
            for row in series.target_coefficients[1:]:
                squared = Fraction(0)
                for value in row:
                    squared = checked(squared + checked(value * value))
                norm = checked(Fraction(_sqrt_upper(squared, budget)))
                norms.append(norm)
                total = checked(total + norm)
            available = checked(checked(1 - kappa) * spatial)
            radius = requested if requested is not None else min(
                Fraction(1), checked(available / checked(series.time * total)))
            power = Fraction(1)
            for norm in norms:
                power = checked(power * radius)
                target_growth = checked(target_growth + checked(norm * power))
            target_growth = checked(series.time * target_growth)
            if target_growth > available:
                return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Analytic parameter domain and Euclidean Taylor remainder for an exact Gaussian heat inverse series.",
            payload={
                "type": "gaussian_inverse_series_analytic",
                "centers": [[_pair(value) for value in row] for row in series.centers],
                "posterior": [_pair(value) for value in series.posterior],
                "base_point": [_pair(value) for value in series.base_point],
                "viscosity": _pair(series.viscosity), "time": _pair(series.time),
                "target_coefficients": [[_pair(value) for value in row] for row in series.target_coefficients],
                "coefficients": [[_pair(value) for value in row] for row in series.coefficients],
                "diameter_squared": _pair(series.diameter_squared), "real_contraction_bound": _pair(q),
                "strip_denominator_squared": None if denominator_squared is None else _pair(denominator_squared),
                "strip_denominator_upper": None if denominator_upper is None else _pair(denominator_upper),
                "phase_half_width_squared_bound": None if phase_squared_bound is None else _pair(phase_squared_bound),
                "parameter_radius": None if radius is None else _pair(radius),
                "spatial_radius": None if spatial is None else _pair(spatial),
                "complex_contraction_bound": _pair(kappa),
                "target_coefficient_norm_bounds": [_pair(value) for value in norms],
                "target_growth_bound": _pair(target_growth), "available_displacement": _pair(available),
                "entire_polynomial": entire,
                "family": "b_j=log(posterior_j)+|base_point-center_j|^2/(4*viscosity*time), up to a common bias",
                "tail_formula": "0 if entire; otherwise spatial_radius*(rho/parameter_radius)^(order+1)/(1-rho/parameter_radius), 0<=rho<parameter_radius",
                "analytic_argument": "After any real spatial shift, normalized Gaussian weights are positive. Their phases span at most 2*eta with eta=D*||Im(h)||/(4*nu*s), so a common-phase rotation gives |Z|>=cos(eta). The pairwise complex covariance identity and Cauchy-Schwarz against those positive weights give ||C_complex||<=D^2/(4*cos(eta)^2), uniformly in the real shift. The checked upper B>=sqrt(D^2*(1-q)) gives r=2*nu*s*(1-q)/B and eta^2<=(1-q)/4<1/4. Thus cos(eta)^2>=1-eta^2>=(3+q)/4, and the complex fixed-point contraction bound is kappa=4*q/(3+q)<1. The target-growth inequality makes the spatial ball invariant. Uniform holomorphic contraction and Cauchy's coefficient bound give the stated Taylor tail.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianSeriesCertificate)
        for name, stored in (("series", series), ("parameter_radius", radius), ("spatial_radius", spatial),
                            ("complex_contraction_bound", kappa), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianSeriesCertificate", "certify_gaussian_inverse_series"]
