# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact global complex-tube bounds for Gaussian heat velocity inverses.

The original rational Gaussian family is retained; no exact posterior or
inverse base point is required. The inverse is unique within a certified
spatial tube, not among all complex roots. These finite analytic arguments
are sealed as evidence and do not constitute a theorem-prover verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction


@dataclass(frozen=True, init=False)
class GaussianInverseTubeCertificate:
    r"""A holomorphic inverse on ``||Im(target)||_2 < target_radius``.

    For every such target, its inverse X is the unique solution in the closed
    tube ``||Im(X)||_2 <= spatial_radius``. All real targets are included,
    with no restriction on their magnitude. The Euclidean complex correction
    satisfies ``||X-time*target-reference_center||_2**2 <=
    correction_bound_squared``. ``None`` radii denote the entire affine case
    of identical centers, where that correction is exactly zero.

    Construction is restricted to :func:`certify_gaussian_inverse_tube`.
    """

    model: GaussianHeatMap
    spatial_radius: Fraction | None
    target_radius: Fraction | None
    complex_contraction_bound: Fraction
    correction_bound_squared: Fraction
    reference_center: tuple[Fraction, ...]
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_inverse_tube")


def certify_gaussian_inverse_tube(
    model: GaussianHeatMap, *, spatial_radius: Rational | None = None,
) -> GaussianInverseTubeCertificate | None:
    r"""Prove a global inverse tube using bounded rational inequalities.

    Write D for the center diameter, nu for viscosity, s for time, and
    q=D²/(8*nu*s). A supplied positive rational r is accepted when
    beta=D²*r²/(16*nu²*s²)<1-q. Otherwise the default is
    r=2*nu*s*(1-q)/sqrt_upper(D²*(1-q)). The resulting exact bounds are
    kappa=q/(1-beta) and B=(r/s)*(1-q*(1+beta/(3*(1-beta)))).

    After any real spatial shift the Gaussian weights are positive. Their
    phases lie in an arc of half-width w<=D*r/(4*nu*s), so a rotated
    partition has modulus at least cos(w). The pairwise covariance identity
    and Cauchy--Schwarz bound the fixed-point derivative by q*sec(w)**2,
    hence by kappa<1. Integrating along the imaginary direction and using
    tan(w)<=w+w³/(3*(1-w²)) proves that each target with ||Im(v)||<B maps
    the entire closed spatial tube into itself. Banach gives uniqueness
    there; the holomorphic implicit function theorem gives holomorphy.

    Type/domain errors raise. q>=1, an unproved requested radius, or exhausted
    arithmetic/resource bounds return None, without asserting nonexistence.
    No transcendental evaluations occur. Certificate sealing retains the
    shared strict-backend policy and may raise RuntimeError under that policy.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    requested: Fraction | None = None
    if spatial_radius is not None:
        if isinstance(spatial_radius, bool) or not isinstance(spatial_radius, (int, Fraction)):
            raise TypeError("spatial_radius must be exact int or Fraction, excluding bool")
        requested = Fraction(spatial_radius)
        if requested <= 0:
            raise ValueError("spatial_radius must be strictly positive")
    q = model.contraction_bound
    if q >= 1:
        return None
    budget = model.budget
    checked = budget._coefficient
    try:
        # Include retained model metadata, pair serialization, proof fields,
        # scratch rational arithmetic and the returned immutable record.
        count, dimension = model.mode_count, model.dimension
        budget._entries(8 * (count * dimension + count + dimension) + 128)
        budget._work(256 * (count + 1) * (dimension + 1) + 512)
        if requested is not None:
            checked(requested)
        entire = model.diameter_squared == 0
        radius: Fraction | None = None
        target: Fraction | None = None
        denominator_squared: Fraction | None = None
        denominator_upper: Fraction | None = None
        beta, kappa, correction, imaginary_factor = (Fraction(0),) * 4
        if not entire:
            margin = checked(1 - q)
            scale = checked(checked(4 * model.viscosity) * model.time)
            if requested is None:
                denominator_squared = checked(model.diameter_squared * margin)
                denominator_upper = checked(Fraction(_sqrt_upper(denominator_squared, budget)))
                numerator = checked(checked(scale / 2) * margin)
                radius = checked(numerator / denominator_upper)
            else:
                radius = requested
            normalized = checked(radius / scale)
            beta = checked(model.diameter_squared * checked(normalized * normalized))
            if radius <= 0 or beta >= margin:
                return None
            complement = checked(1 - beta)
            kappa = checked(q / complement)
            tangent_factor = checked(1 + checked(beta / checked(3 * complement)))
            imaginary_factor = checked(q * tangent_factor)
            target = checked(checked(radius / model.time) * checked(1 - imaginary_factor))
            correction = checked(model.diameter_squared / complement)
            if not 0 <= kappa < 1 or target <= 0:
                return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Holomorphic Gaussian heat velocity inverse on a global target tube, unique within its certified spatial tube.",
            payload={
                "type": "gaussian_heat_inverse_tube",
                "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases],
                "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "diameter_squared": _pair(model.diameter_squared),
                "real_contraction_bound": _pair(q),
                "reference_center": [_pair(value) for value in model.centers[0]],
                "default_radius_denominator_squared": None if denominator_squared is None else _pair(denominator_squared),
                "default_radius_denominator_upper": None if denominator_upper is None else _pair(denominator_upper),
                "spatial_radius": None if radius is None else _pair(radius),
                "target_radius": None if target is None else _pair(target),
                "phase_half_width_squared_bound": _pair(beta),
                "complex_contraction_bound": _pair(kappa),
                "imaginary_mean_factor": _pair(imaginary_factor),
                "correction_bound_squared": _pair(correction),
                "entire_affine": entire,
                "target_domain": "all v in C^d with ||Im(v)||_2 < target_radius; all C^d when entire_affine",
                "uniqueness_domain": "closed tube ||Im(X)||_2 <= spatial_radius; all C^d when entire_affine",
                "analytic_argument": "Positive normalized weights after any real shift give a rotated complex partition modulus >=cos(w), w<=D*r/(4*nu*s). The pairwise covariance identity and complex Cauchy-Schwarz give derivative norm <=q*sec(w)^2<=q/(1-beta)=kappa<1. Since tan(w)-w=int_0^w tan(t)^2 dt <=w^3/(3*(1-w^2)), integration along each imaginary segment gives ||Im(mu(X))||<=q*r*(1+beta/(3*(1-beta))). Thus ||Im(v)||<B makes X->s*v+mu(X) preserve the closed complete convex spatial tube. Banach gives its unique fixed point; strict interior margin and the invertible I-Dmu give a holomorphic branch agreeing with the real inverse. Finally ||mu(X)-a0||<=D/|Z|<=D/sqrt(1-beta). Identical centers instead give X=s*v+a0 identically.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianInverseTubeCertificate)
        for name, stored in (("model", model), ("spatial_radius", radius), ("target_radius", target),
                             ("complex_contraction_bound", kappa), ("correction_bound_squared", correction),
                             ("reference_center", model.centers[0]), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianInverseTubeCertificate", "certify_gaussian_inverse_tube"]
