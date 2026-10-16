# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global Gaussian uniqueness from a convex posterior domain and entropy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_posterior_domain import (
    GaussianPosteriorDomain,
    _domain_intact,
    _domain_source_slots,
)
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.transcend import libm_fallback_used, strict_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert
    from omnibias.core.verified.interval import Interval

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Bounds = tuple[Fraction, Fraction]


def _unique_source_slots(unique: GaussianEntropyUnique) -> int:
    """Reserve the retained uniqueness dependency and exact metadata."""
    model = unique.domain.model
    return 2*_domain_source_slots(unique.domain)+128*(model.mode_count+1)*(model.dimension+1)+4096


def _unique_intact(unique: GaussianEntropyUnique) -> bool:
    """Check digests and snapshot bindings after the caller reserves scans."""
    if not _intact(unique.certificate) or not _domain_intact(unique.domain):
        return False
    payload = unique.certificate.get("payload")
    if not isinstance(payload, dict):
        return False
    return (payload.get("domain_certificate") == unique.domain.certificate
            and payload.get("reference") == [_pair(value) for value in unique.reference]
            and payload.get("squared_radii") == [_pair(value) for value in unique.squared_radii]
            and payload.get("expectation_bounds") == [_pair(value) for value in unique.expectation_bounds]
            and payload.get("variance_upper_bound") == _pair(unique.variance_upper_bound)
            and payload.get("uniqueness_margin") == _pair(unique.uniqueness_margin)
            and payload.get("entropy_monotonicity_lower_bound") == _pair(unique.entropy_monotonicity_lower_bound))


def _expectation_bounds(values: Vector, box: tuple[Interval, ...], arithmetic: _Arithmetic) -> Bounds:
    """Exact linear expectation extrema on a rectangular stick-domain image."""
    lower = upper = values[-1]
    for j in reversed(range(len(values)-1)):
        endpoints = arithmetic.exact(Fraction(box[j].lo)), arithmetic.exact(Fraction(box[j].hi))
        candidates = tuple(arithmetic.add(tail, arithmetic.mul(u, arithmetic.add(values[j], -tail)))
                           for u in endpoints for tail in (lower, upper))
        lower, upper = min(candidates), max(candidates)
    return lower, upper


@dataclass(frozen=True, init=False)
class GaussianEntropyUnique:
    r"""Exactly one original constrained global minimizer at the fixed data.

    The source posterior stick box maps onto a convex polytope P of probability
    weights: L_j*sum_{i>=j}p_i <= p_j <= U_j*sum_{i>=j}p_i. Every global
    optimizer's strictly positive posterior lies in P, even when P touches
    the simplex boundary and domain.gradient_safe is false.

    ``expectation_bounds`` are the exact minimum and maximum on P of
    sum_j p_j*||a_j-reference||². The upper endpoint B bounds the covariance
    operator on P; ``uniqueness_margin`` is 2*nu*s-B > 0. The entropy gradient
    satisfies a strong monotonicity inequality with coefficient
    ``entropy_monotonicity_lower_bound`` = 2*nu-B/s, multiplying the symmetric
    KL divergence. No ordinary Hessian at a projection switch is needed.

    The bound depends on the reference; refusal at one reference does not
    imply failure at another or nonuniqueness. This certificate does not locate
    the unique point, differentiate it or establish a perturbation radius.
    Convexity in posterior weights does not assert convexity in stick variables
    or throughout physical space.
    """

    domain: GaussianPosteriorDomain
    reference: Vector
    squared_radii: Vector
    expectation_bounds: Bounds
    variance_upper_bound: Fraction
    uniqueness_margin: Fraction
    entropy_monotonicity_lower_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_unique")


def certify_gaussian_entropy_unique(
    domain: GaussianPosteriorDomain, *, reference: Sequence[Rational] | None = None,
) -> GaussianEntropyUnique | None:
    r"""Prove global uniqueness when the posterior covariance bound is small.

    Choose an exact rational reference (default: first center). A linear tail
    recurrence computes B=max_P sum p_j*||a_j-reference||² exactly, where P
    is the convex image of the source posterior-domain box. B<2*nu*s proves
    strict monotonicity of the entropy envelope gradient between any two
    positive points of P, without differentiating the polyhedral projection.
    All global posteriors lie in P, so precisely one global optimizer exists.

    Uses O(m*d) rational work after checking the entire proof dependency; no
    optimization, projection query or new transcendental evaluation is needed.
    Both source budgets apply. Invalid argument types/shapes raise. A failed
    sufficient inequality (including equality), damaged/conditional dependency
    or exhausted budget returns None and does not disprove uniqueness. Exact
    postprocessing preserves the source backend provenance and shared strict
    sealing policy, without clearing conditional-libm history.
    """
    if not isinstance(domain, GaussianPosteriorDomain):
        raise TypeError("domain must be a GaussianPosteriorDomain")
    model = domain.model
    if reference is not None:
        _validate_vector(reference, model.dimension, "reference")
    arithmetic = _Arithmetic(_combined_budget(model.budget, domain.projector.budget))
    try:
        source = 2*_domain_source_slots(domain)+128*(model.mode_count+1)*(model.dimension+1)+4096
        arithmetic._entries(source)
        arithmetic.limits._work(4*source+128*(model.mode_count+1)*(model.dimension+1)+4096)
        arithmetic._work(4*source)
        if not _domain_intact(domain):
            return None
        from omnibias.core.proof.certificate import (
            certificate_is_unconditional,
            certificate_transcend_backend,
            make_certificate,
        )

        if not certificate_is_unconditional(domain.certificate):
            return None
        center = tuple(arithmetic.exact(value) for value in (model.centers[0] if reference is None else reference))
        radii = []
        for row in model.centers:
            delta = tuple(arithmetic.add(value, -origin) for value, origin in zip(row, center, strict=True))
            radii.append(_dot(delta, delta, arithmetic))
        squared_radii = tuple(radii)
        expectation = _expectation_bounds(squared_radii, domain.parameter_box, arithmetic)
        bound = expectation[1]
        threshold = arithmetic.mul(arithmetic.mul(Fraction(2), model.viscosity), model.time)
        margin = arithmetic.add(threshold, -bound)
        if margin <= 0:
            return None
        monotonicity = arithmetic._coefficient(margin/model.time)
        if strict_backend() and libm_fallback_used():
            raise RuntimeError("strict entropy uniqueness sealing refuses prior conditional libm fallback history")
        certificate = make_certificate(
            claim="The original finite Gaussian objective has exactly one global minimizer on its polyhedron.",
            payload={
                "type": "gaussian_entropy_unique", "domain_certificate": deepcopy(domain.certificate),
                "reference": [_pair(value) for value in center],
                "squared_radii": [_pair(value) for value in squared_radii],
                "expectation_bounds": [_pair(value) for value in expectation],
                "variance_upper_bound": _pair(bound), "uniqueness_margin": _pair(margin),
                "entropy_monotonicity_lower_bound": _pair(monotonicity),
                "argument": "Write T_j=sum_{i>=j}p_i. The image P of the certified stick box L_j<=z_j<=U_j is exactly the simplex polytope L_j*T_j<=p_j<=U_j*T_j. For T_j>0 the inverse coordinate is p_j/T_j; at a zero tail every remaining probability vanishes and arbitrary allowed coordinates suffice. Thus P is convex and contains the positive posterior of every original global minimizer. Each linear expected squared radius has exact extrema from the backward recurrence R_j=z_j*r_j+(1-z_j)*R_{j+1}, testing both coordinate and tail endpoints. Its upper bound B gives Cov_p(a)<=B*I throughout P by subtracting the nonnegative squared mean offset from the expected squared radius. For two positive weights p,p' in P, put u=p-p' and A*u=sum_j u_j*a_j. The Gibbs envelope has x_p=P_C(s*v+A*p) and gradient |a_j|²/(2s)-a_j.x_p/s+2nu*(log(p_j)+1-b_j), modulo a common simplex-normal offset. Nonexpansiveness gives u.(grad(E)(p)-grad(E)(p'))>=2nu*J-|A*u|²/s, where J=sum_j u_j*(log(p_j)-log(p'_j)). On the entire segment p_t=p'+t*u, weighted Cauchy-Schwarz gives |A*u|²<=B*sum_j u_j²/(p_t)_j. Integration over t gives |A*u|²<=B*J. Hence the gradient difference is at least (2nu-B/s)*J>0 for distinct p,p'. Every original global minimizer corresponds to a global minimum of the Gibbs envelope with a positive posterior, so its simplex-tangent gradient vanishes. Two distinct such posteriors contradict strict monotonicity. Coercivity and the source nonempty polyhedron give existence, and each posterior has exactly one projected spatial optimizer. Therefore the original global optimizer is unique.",
                "scope": "Fixed source model, constraints and target; arbitrary finite center geometry. The covariance bound is sufficient, not necessary, and depends on the supplied reference. Boundary-inclusive source domains are allowed because actual global posteriors remain positive. This does not locate the optimum, assert global physical convexity or convexity after stick parametrization, differentiate the selected optimizer, establish a perturbation radius or supply external formal theorem-prover verification.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": certificate_transcend_backend(domain.certificate)},
        )
        result = object.__new__(GaussianEntropyUnique)
        for name, value in (("domain", domain), ("reference", center), ("squared_radii", squared_radii),
                            ("expectation_bounds", expectation), ("variance_upper_bound", bound),
                            ("uniqueness_margin", margin), ("entropy_monotonicity_lower_bound", monotonicity),
                            ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyUnique", "certify_gaussian_entropy_unique"]
