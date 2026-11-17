# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact analytic disks and Taylor tails for joint Gaussian bias/target paths."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_bias_jet import GaussianBiasInverseSeries
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine_sensitivity import _norm_upper
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]


def _exact(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be exact int or Fraction, excluding bool")
    return Fraction(value)


def _preflight(series: GaussianBiasInverseSeries, arithmetic: _Arithmetic) -> None:
    """Reserve retained series/model evidence, hashing, norms and evaluation."""
    count, dimension, order = series.mode_count, series.dimension, series.order
    slots = count*dimension + count + dimension + (order+1)*(2*dimension+3*count)
    arithmetic._entries(8*slots + 128*(order+1) + 256)
    arithmetic._work(128*(count+dimension+1)*(order+1) + 64*(count*dimension+1) + 512)


def _tail_bound(certificate: GaussianBiasSeriesCertificate, rho: Fraction, arithmetic: _Arithmetic) -> Fraction:
    """Shared exact tail arithmetic; caller validates domain, proof and budget."""
    checked_rho = arithmetic._coefficient(rho)
    ratio = arithmetic._coefficient(checked_rho/certificate.parameter_radius)
    power = Fraction(1)
    for _ in range(certificate.series.order+1):
        power = arithmetic.mul(power, ratio)
    return arithmetic._coefficient(arithmetic.mul(certificate.spatial_radius, power)
                                    /arithmetic.add(Fraction(1), -ratio))


def _enclose(
    certificate: GaussianBiasSeriesCertificate, parameter: Fraction, arithmetic: _Arithmetic,
) -> tuple[Interval, ...]:
    """Shared Horner/tail enclosure; caller validates domain, proof and budget."""
    value = arithmetic._coefficient(parameter)
    tail = _tail_bound(certificate, abs(value), arithmetic)
    polynomial = certificate.series.coefficients[-1]
    for row in reversed(certificate.series.coefficients[:-1]):
        polynomial = tuple(arithmetic.add(coefficient, arithmetic.mul(value, component))
                           for coefficient, component in zip(row, polynomial, strict=True))
    return tuple(Interval(_rational(arithmetic.add(center, -tail)).lo,
                          _rational(arithmetic.add(center, tail)).hi) for center in polynomial)


@dataclass(frozen=True, init=False)
class GaussianBiasSeriesCertificate:
    r"""A finite holomorphic disk and Euclidean inverse-series remainder.

    On ``|t|<parameter_radius`` the original model with the declared polynomial
    bias and target paths has a holomorphic inverse branch lying within
    ``spatial_radius`` of the exact base point. It is the unique fixed point
    in that ball, and agrees with the unique real inverse for real t.

    ``target_growth_bound`` is V=sum ||v_n||_upper R^n, without the age factor.
    ``bias_growth_bound`` is B=sum range(b_n) R^n. The phase half-width is
    eta=D_upper*r/(4*nu*s)+B/2, and kappa=q/(1-eta²) bounds the complex
    fixed-point derivative. ``forcing_bound`` is s*V+D_upper*B/[4*(1-eta²)],
    at most ``available_displacement=(1-kappa)*r``. All these fields are exact
    rationals. D_upper is an outward bound, not a fitted center diameter.

    No entire shortcut is taken, even for identical centers or a constant
    target. The finite disk also controls partition nonvanishing under the
    declared bias path. Integrity checks do not constitute formal verification.
    """

    series: GaussianBiasInverseSeries
    parameter_radius: Fraction
    spatial_radius: Fraction
    diameter_bound: Fraction
    phase_half_width_bound: Fraction
    complex_contraction_bound: Fraction
    target_growth_bound: Fraction
    bias_growth_bound: Fraction
    forcing_bound: Fraction
    available_displacement: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_bias_inverse_series")

    def tail_bound(self, radius: Rational) -> Fraction:
        """Bound the Euclidean tail on |t|<=radius, with 0<=radius<R.

        The bound is r*(radius/R)^(N+1)/(1-radius/R). Invalid domains,
        damaged evidence and exhausted exact arithmetic budgets raise.
        """
        rho = _exact(radius, "evaluation radius")
        if not 0 <= rho < self.parameter_radius:
            raise ValueError("evaluation radius must satisfy 0<=radius<parameter_radius")
        arithmetic = _Arithmetic(self.series.budget)
        _preflight(self.series, arithmetic)
        if not _intact(self.certificate):
            raise ValueError("analytic series certificate is damaged")
        return _tail_bound(self, rho, arithmetic)

    def enclose(self, parameter: Rational) -> tuple[Interval, ...]:
        """Enclose the real inverse at one exact |parameter|<R.

        Evaluate the exact finite polynomial with Horner arithmetic and add
        the Euclidean tail to each coordinate before outward conversion.
        This enclosure concerns the continuum inverse, not just the finite
        polynomial. Invalid domains, damaged evidence, overflow and resource
        failures raise. No new backend or certificate is constructed.
        """
        value = _exact(parameter, "parameter")
        if abs(value) >= self.parameter_radius:
            raise ValueError("parameter must lie strictly inside the certified disk")
        arithmetic = _Arithmetic(self.series.budget)
        _preflight(self.series, arithmetic)
        if not _intact(self.certificate):
            raise ValueError("analytic series certificate is damaged")
        return _enclose(self, value, arithmetic)


def _certify_bias_series(
    series: GaussianBiasInverseSeries, parameter_radius: Fraction, spatial_radius: Fraction, arithmetic: _Arithmetic,
) -> GaussianBiasSeriesCertificate | None:
    """Certify with a caller's cumulative meter after public validation."""
    if arithmetic.limits != series.budget:
        raise ValueError("shared arithmetic and series budgets must agree")
    try:
        _preflight(series, arithmetic)
        radius, spatial = arithmetic._coefficient(parameter_radius), arithmetic._coefficient(spatial_radius)
        model = series.model
        q = arithmetic._coefficient(model.contraction_bound)
        if not 0 <= q < 1:
            return None
        diameter_squared = arithmetic._coefficient(model.diameter_squared)
        diameter = arithmetic._coefficient(Fraction(_sqrt_upper(diameter_squared, arithmetic.limits)))
        norms, ranges = [], []
        target_growth = bias_growth = Fraction(0)
        power = Fraction(1)
        for target, biases in zip(series.target_coefficients[1:], series.relative_bias_coefficients[1:], strict=True):
            norm = _norm_upper(target, arithmetic)
            bias_range = arithmetic.add(max(biases), -min(biases))
            norms.append(norm)
            ranges.append(bias_range)
            power = arithmetic.mul(power, radius)
            target_growth = arithmetic.add(target_growth, arithmetic.mul(norm, power))
            bias_growth = arithmetic.add(bias_growth, arithmetic.mul(bias_range, power))
        scale = arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity), model.time)
        phase = arithmetic.add(arithmetic._coefficient(arithmetic.mul(diameter, spatial)/scale),
                               arithmetic._coefficient(bias_growth/2))
        phase_squared = arithmetic.mul(phase, phase)
        if phase_squared >= arithmetic.add(Fraction(1), -q):
            return None
        complement = arithmetic.add(Fraction(1), -phase_squared)
        kappa = arithmetic._coefficient(q/complement)
        forcing = arithmetic.add(arithmetic.mul(model.time, target_growth), arithmetic._coefficient(
            arithmetic.mul(diameter, bias_growth)/arithmetic.mul(Fraction(4), complement)))
        available = arithmetic.mul(arithmetic.add(Fraction(1), -kappa), spatial)
        if forcing > available:
            return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Holomorphic parameter disk and Euclidean Taylor tail for an exact joint Gaussian bias/target inverse series.",
            payload={
                "type": "gaussian_bias_inverse_series_analytic",
                "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases],
                "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "base_point": [_pair(value) for value in series.base_point],
                "target_coefficients": [[_pair(value) for value in row] for row in series.target_coefficients],
                "bias_coefficients": [[_pair(value) for value in row] for row in series.bias_coefficients],
                "relative_bias_coefficients": [[_pair(value) for value in row] for row in series.relative_bias_coefficients],
                "coefficients": [[_pair(value) for value in row] for row in series.coefficients],
                "posterior_coefficients": [[_pair(value) for value in row] for row in series.posterior_coefficients],
                "diameter_squared": _pair(diameter_squared), "diameter_bound": _pair(diameter),
                "real_contraction_bound": _pair(q), "parameter_radius": _pair(radius), "spatial_radius": _pair(spatial),
                "phase_half_width_bound": _pair(phase), "phase_half_width_squared_bound": _pair(phase_squared),
                "complex_contraction_bound": _pair(kappa),
                "target_coefficient_norm_bounds": [_pair(value) for value in norms],
                "bias_coefficient_ranges": [_pair(value) for value in ranges],
                "target_growth_bound": _pair(target_growth), "bias_growth_bound": _pair(bias_growth),
                "forcing_bound": _pair(forcing), "available_displacement": _pair(available),
                "tail_formula": "r*(rho/R)^(order+1)/(1-rho/R), 0<=rho<R; Euclidean norm",
                "analytic_argument": "For |t|<=R and ||h||<=r, original mode phases span at most 2*eta, eta=D_upper*r/(4*nu*s)+sum_n range(b_n)*R^n/2. Absorb real parts into positive normalized weights and rotate the common phase: the normalized partition modulus is at least cos(eta)>0. A centered covariance factorization and the oblique-projection norm bound, together with real diameter variance and Cauchy-Schwarz, give ||Cov_complex(a)||<=D^2/(4*cos(eta)^2) and ||Cov_complex(a,b_n)||<=D*range(b_n)/(4*cos(eta)^2) for each REAL coefficient row b_n. Since cos(eta)^2>=1-eta^2 and eta^2<1-q, the spatial fixed-point map has derivative norm at most kappa=q/(1-eta^2)<1. At the base point integrate the bias perturbation theta*sum_n b_n*t^n for 0<=theta<=1. Apply the mixed covariance estimate coefficientwise to the real rows and sum |t|^n; this bounds its mean displacement by D_upper*B/[4*(1-eta^2)]. Adding the target displacement s*V gives forcing_bound. The inequality forcing_bound<=(1-kappa)*r makes the spatial ball invariant. Uniform holomorphic contraction gives the unique inverse branch in this ball, agreeing with the real global inverse. Vector-valued Cauchy bounds for displacement at most r imply the stated dimension-free Euclidean Taylor tail.",
                "scope": "The complete stored target and original-bias polynomials, with centers, viscosity and age fixed. No unspecified higher path coefficients, automatic radius, entire shortcut, global complex-root uniqueness, multiplier control or ordinary finite-polynomial exactness is asserted. Serialization is not a theorem-prover proof.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianBiasSeriesCertificate)
        for name, value in (("series", series), ("parameter_radius", radius), ("spatial_radius", spatial),
                            ("diameter_bound", diameter), ("phase_half_width_bound", phase),
                            ("complex_contraction_bound", kappa), ("target_growth_bound", target_growth),
                            ("bias_growth_bound", bias_growth), ("forcing_bound", forcing),
                            ("available_displacement", available), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


def certify_gaussian_bias_inverse_series(
    series: GaussianBiasInverseSeries, *, parameter_radius: Rational, spatial_radius: Rational,
) -> GaussianBiasSeriesCertificate | None:
    r"""Certify the supplied positive exact parameter and spatial radii.

    Set V=sum ||v_n||_upper*R^n and B=sum range(b_n)*R^n, summing orders
    n>=1 of the complete stored polynomial paths. With D_upper>=sqrt(D²),
    eta=D_upper*r/(4*nu*s)+B/2 must satisfy eta²<1-q. Set
    kappa=q/(1-eta²), and require
    s*V+D_upper*B/[4*(1-eta²)] <= (1-kappa)*r.
    These sufficient rational inequalities prove a holomorphic inverse disk
    and a Taylor remainder; their failure makes no nonexistence claim.

    No radius is chosen automatically and no entire shortcut is taken. Invalid
    types or nonpositive radii raise before preflight. Unproved bounds and
    exhausted exact arithmetic resources return None. There is no exponential,
    numerical root solve or factorization. Shared strict certificate-sealing
    policy still applies, and sticky fallback history is preserved.
    """
    if not isinstance(series, GaussianBiasInverseSeries):
        raise TypeError("series must be a GaussianBiasInverseSeries")
    radius = _exact(parameter_radius, "parameter_radius")
    spatial = _exact(spatial_radius, "spatial_radius")
    if radius <= 0 or spatial <= 0:
        raise ValueError("parameter_radius and spatial_radius must be strictly positive")
    return _certify_bias_series(series, radius, spatial, _Arithmetic(series.budget))


__all__ = ["GaussianBiasSeriesCertificate", "certify_gaussian_bias_inverse_series"]
