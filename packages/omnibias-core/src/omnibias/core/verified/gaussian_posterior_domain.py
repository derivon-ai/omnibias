# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified posterior stick-coordinate domains for constrained Gaussian minima."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget, _ProjectionOracle
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.polyhedral_projection import (
    PolyhedralProjector,
    _project_exact,
    _query_work_bound,
    _source_slots,
)
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Box = tuple[Interval, ...]
Bounds = tuple[tuple[Fraction, Fraction], ...]


def _domain_source_slots(domain: GaussianPosteriorDomain) -> int:
    """Conservative retained source/domain/proof snapshot scalar slots."""
    dimension, modes = domain.model.dimension, domain.model.mode_count
    return 2*_source_slots(domain.projector)+256*(modes+1)*(dimension+1)+128*(dimension+1)**2+4096


def _domain_intact(domain: GaussianPosteriorDomain) -> bool:
    """Replay digest and source snapshot bindings; caller preflights scans."""
    from omnibias.core.proof.certificate import encode_interval

    if not _intact(domain.certificate) or not _intact(domain.projector.certificate):
        return False
    payload = domain.certificate.get("payload")
    if not isinstance(payload, dict):
        return False
    model = domain.model
    return (payload.get("projector_certificate") == domain.projector.certificate
            and payload.get("centers") == [[_pair(value) for value in row] for row in model.centers]
            and payload.get("biases") == [_pair(value) for value in model.biases]
            and payload.get("viscosity") == _pair(model.viscosity)
            and payload.get("time") == _pair(model.time)
            and payload.get("target") == [_pair(value) for value in domain.target]
            and payload.get("physical_bounds") == [[_pair(value) for value in pair] for pair in domain.physical_bounds]
            and payload.get("parameter_box") == [encode_interval(value) for value in domain.parameter_box]
            and payload.get("gradient_safe") is domain.gradient_safe)


@dataclass(frozen=True, init=False)
class GaussianPosteriorDomain:
    r"""Posterior stick-coordinate cover of EVERY original constrained minimum.

    ``physical_bounds`` encloses P_C(s*target+conv(centers)) in exact rational
    coordinates. At every constrained global minimum, its original posterior
    weights have stick coordinates z_j=p_j/(p_j+...+p_last) in ``parameter_box``.
    This box can include posteriors that do not describe a stationary point.

    ``gradient_safe`` means every represented coordinate has strict binary64
    bounds 0<lo<=hi<1. Positive real posteriors can round to a boundary-inclusive
    enclosure; such sound bounds are retained with gradient_safe=False. No
    artificial clipping excludes underflowed probabilities. One mode has a
    dummy singleton [0,0] and gradient_safe=True, with no genuine stick variable.
    The certificate neither selects a minimum nor proves uniqueness.
    """

    model: GaussianHeatMap
    projector: PolyhedralProjector
    target: Vector
    physical_bounds: Bounds
    parameter_box: Box
    gradient_safe: bool
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_posterior_domain")


def certify_gaussian_posterior_domain(
    model: GaussianHeatMap, projector: PolyhedralProjector, target: Sequence[Rational],
) -> GaussianPosteriorDomain | None:
    """Enclose all minimizing posteriors by bounded exact logit contrasts.

    Invalid types/shapes raise. A damaged projector, no feasible projection,
    exhausted combined resources or nonfinite enclosures return None. Strict
    mpmath evaluation is required, including for the one-mode dummy domain;
    missing backend or prior conditional-libm history raises RuntimeError and
    never clears that history. Exact geometry and affine contrast bounds
    precede binary64 conversion. Posterior work is O(m²*d), preflighted in full.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(projector, PolyhedralProjector):
        raise TypeError("projector must be a PolyhedralProjector")
    if model.dimension != projector.dimension:
        raise ValueError("model and projector dimensions must match")
    _validate_vector(target, model.dimension, "target")
    arithmetic = _Arithmetic(_combined_budget(model.budget, projector.budget))
    dimension, modes = model.dimension, model.mode_count
    try:
        source = 2*_source_slots(projector)+256*(modes+1)*(dimension+1)+128*(dimension+1)**2+4096
        arithmetic._entries(source+256*(modes+1)*(dimension+1)+4096)
        arithmetic.limits._work(4*source+2*_query_work_bound(projector)
                               +1024*(modes+1)**2*(dimension+1)+4096)
        arithmetic._work(4*source)
        if not _intact(projector.certificate):
            return None
        goal = tuple(arithmetic.exact(value) for value in target)
        oracle = _ProjectionOracle(model, projector, goal, arithmetic)
        if _project_exact(projector, oracle.shift, arithmetic) is None:
            return None
        cube = (Interval(0.0, 1.0),)*(modes-1) if modes > 1 else (Interval.point(0.0),)
        physical = oracle.image(cube)
        centered = tuple((arithmetic.add(lower, -reference), arithmetic.add(upper, -reference))
                         for (lower, upper), reference in zip(physical, model.centers[0], strict=True))
        if libm_fallback_used():
            raise RuntimeError("Gaussian posterior domain refuses prior conditional libm fallback history")
        with _strict():
            parameters = []
            for j in range(modes-1):
                arithmetic._work(256*(modes-j)*(dimension+1))
                contrasts = [Interval.point(0.0)]
                for i in range(j+1, modes):
                    lower = upper = arithmetic.add(oracle.offsets[i], -oracle.offsets[j])
                    for ri, rj, (left, right) in zip(oracle.rates[i], oracle.rates[j], centered, strict=True):
                        rate = arithmetic.add(ri, -rj)
                        products = arithmetic.mul(rate, left), arithmetic.mul(rate, right)
                        lower = arithmetic.add(lower, min(products))
                        upper = arithmetic.add(upper, max(products))
                    contrasts.append(Interval(_rational(lower).lo, _rational(upper).hi))
                enclosure = _probabilities(tuple(contrasts))[0]
                parameters.append(Interval(max(0.0, enclosure.lo), min(1.0, enclosure.hi)))
            parameter_box = tuple(parameters) if modes > 1 else (Interval.point(0.0),)
            gradient_safe = modes == 1 or all(0.0 < value.lo <= value.hi < 1.0 for value in parameter_box)
            from omnibias.core.proof.certificate import encode_interval, make_certificate

            certificate = make_certificate(
                claim="A posterior stick-coordinate box covering every original constrained Gaussian global minimizer.",
                payload={
                    "type": "gaussian_posterior_domain", "projector_certificate": deepcopy(projector.certificate),
                    "centers": [[_pair(value) for value in row] for row in model.centers],
                    "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                    "target": [_pair(value) for value in goal],
                    "physical_bounds": [[_pair(value) for value in pair] for pair in physical],
                    "parameter_box": [encode_interval(value) for value in parameter_box], "gradient_safe": gradient_safe,
                    "argument": "A feasible exact projection establishes a nonempty polyhedron. Coercivity gives an original global minimum. Every constrained minimum satisfies x=P_C(s*target+E[a](x)), so it lies in the projected full center hull. Exact stick-breaking source bounds and nonexpansiveness of Euclidean projection enclose that entire image by the recorded rational physical bounds. For original logits ell_j=b_j-|x-a_j|²/(4nu*s), each contrast ell_i-ell_j is affine in x. Compute those affine bounds exactly before numerical conversion. For every j<m-1, z_j=p_j/(p_j+...+p_last)=1/(1+sum_{i>j}exp(ell_i-ell_j)). Rigorous shared softmax bounds on (0,contrasts) enclose z_j throughout the physical bounds and hence at every global minimum. Bounds are intersected only with the true closed range [0,1]; strict numerical interior is separately recorded. One mode needs no posterior coordinate and uses a dummy singleton.",
                    "scope": "Fixed exact finite Gaussian model, target and polyhedral constraints. No stationary-point selection, uniqueness, quantitative perturbation radius, general convergence or external formal proof-checker claim. Boundary-inclusive probability enclosures remain sound but are not safe for an unguarded interior entropy gradient.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianPosteriorDomain)
        for name, value in (("model", model), ("projector", projector), ("target", goal), ("physical_bounds", physical),
                            ("parameter_box", parameter_box), ("gradient_safe", gradient_safe), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPosteriorDomain", "certify_gaussian_posterior_domain"]
