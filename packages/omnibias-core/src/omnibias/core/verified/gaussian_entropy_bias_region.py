# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Uniform Gaussian global uniqueness and decision bounds under finite bias changes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_entropy_unique import (
    GaussianEntropyUnique,
    _unique_intact,
    _unique_source_slots,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import exp_iv, libm_fallback_used, tanh_iv

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

_UNIT = Interval(0.0, 1.0)


def _tilt_decay(oscillation: Fraction) -> Interval:
    """Outward exp(-oscillation); caller owns validation, work and strict mode."""
    if not oscillation:
        return Interval.point(1.0)
    if oscillation > 1024:
        return Interval(0.0, exp_iv(Interval.point(-1024.0)).hi).intersect(_UNIT)
    return exp_iv(-_rational(oscillation)).intersect(_UNIT)


def _variation_upper(oscillation: Fraction, arithmetic: _Arithmetic) -> Fraction:
    """Upper bound on half-L1 total variation under bounded log-weight tilt."""
    if not oscillation:
        return Fraction(0)
    if oscillation > 1024:
        return Fraction(1)
    enclosed = tanh_iv(_rational(arithmetic._coefficient(oscillation/4))).intersect(_UNIT)
    return arithmetic.exact(Fraction(enclosed.hi))


def _tilted_bound(maximum: Fraction, base_bound: Fraction, decay: Interval, arithmetic: _Arithmetic) -> Fraction:
    """Sharp radius-moment transfer using a rigorous lower decay endpoint."""
    if not base_bound or maximum == base_bound:
        return base_bound
    denominator = arithmetic.add(base_bound, arithmetic.mul(arithmetic.add(maximum, -base_bound),
                                                            arithmetic.exact(Fraction(decay.lo))))
    return arithmetic._coefficient(arithmetic.mul(maximum, base_bound)/denominator)


def _source_slots(unique: GaussianEntropyUnique) -> int:
    model = unique.domain.model
    return 2*_unique_source_slots(unique)+256*(model.mode_count+1)*(model.dimension+1)+8192


def _region_source_slots(region: GaussianEntropyBiasRegion) -> int:
    """Reserve the retained region and its bound source snapshots."""
    return _source_slots(region.unique)


def _region_intact(region: GaussianEntropyBiasRegion) -> bool:
    from omnibias.core.proof.certificate import certificate_is_unconditional, encode_interval

    if (not _intact(region.certificate) or not _unique_intact(region.unique)
            or not certificate_is_unconditional(region.certificate)
            or not certificate_is_unconditional(region.unique.certificate)):
        return False
    payload = region.certificate.get("payload")
    if not isinstance(payload, dict) or payload.get("unique_certificate") != region.unique.certificate:
        return False
    for name in ("oscillation", "max_squared_radius", "variance_upper_bound", "uniqueness_margin",
                 "contraction_bound", "strong_convexity_bound", "total_variation_bound",
                 "mean_shift_squared_bound", "displacement_squared_bound"):
        if payload.get(name) != _pair(getattr(region, name)):
            return False
    return (payload.get("tilt_decay") == encode_interval(region.tilt_decay)
            and payload.get("displacement_bound") == region.displacement_bound.hex())


@dataclass(frozen=True, init=False)
class GaussianEntropyBiasRegion:
    r"""Uniform global guarantees for every log-weight change of bounded spread.

    The original Gaussian centers, viscosity, time, target and polyhedron
    stay fixed. All real bias changes delta with max(delta)-min(delta) at
    most ``oscillation`` are covered, including unrestricted common shifts.
    No particular perturbed model or nearby posterior-domain solve is needed.

    ``max_squared_radius`` is D=max_j||a_j-c||² at the source reference.
    The source has E_pi||a-c||²<=B on its whole physical box K. Tilting by
    bounded-spread log weights yields the sharp distribution-independent
    bound D*B/(B+(D-B)*exp(-oscillation)). ``variance_upper_bound`` is its
    outward rational upper bound; ``tilt_decay`` encloses that exponential.
    Its strict margin below H=2*nu*s gives global uniqueness and spatial
    curvature ``strong_convexity_bound``=(H-bound)/(H*s) on the same K.

    ``total_variation_bound`` bounds tanh(oscillation/4), where total variation
    is half the L1 distance. With center diameter Delta and ORIGINAL rho=B/H,
    ``mean_shift_squared_bound`` bounds (Delta*TV)², and
    ``displacement_squared_bound`` bounds (Delta*TV/(1-rho))².
    ``displacement_bound`` is an outward square root. It bounds both the
    optimizer's movement and its posterior projection input's movement from
    their exact original values, not from a numerical proposal.

    Active-face persistence is a separate certificate. Neither this bound
    nor a base Jacobian asserts a finite-step Taylor approximation or an
    ordinary derivative through a constraint switch.
    """

    unique: GaussianEntropyUnique
    oscillation: Fraction
    max_squared_radius: Fraction
    tilt_decay: Interval
    variance_upper_bound: Fraction
    uniqueness_margin: Fraction
    contraction_bound: Fraction
    strong_convexity_bound: Fraction
    total_variation_bound: Fraction
    mean_shift_squared_bound: Fraction
    displacement_squared_bound: Fraction
    displacement_bound: float
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_bias_region")


def certify_gaussian_entropy_bias_region(
    unique: GaussianEntropyUnique, oscillation: int | Fraction,
) -> GaussianEntropyBiasRegion | None:
    r"""Prove global uniqueness and displacement for a finite log-bias region.

    For X=||a-c||² in [0,D], E[X]<=B, and likelihood tilt in [1,t],
    the maximum possible tilted expectation is D*t*B/(D+(t-1)*B).
    The compiler evaluates its equivalent negative-exponential form using
    an outward lower bound on exp(-oscillation), avoiding positive exponent
    overflow. A strict upper bound below 2*nu*s is required, even if the
    mathematical sufficient margin exists but numerical underflow loses it.

    Zero oscillation preserves the source bound and has exact zero movement.
    For oscillations above 1024, monotonicity gives a conservative exponential
    enclosure [0,exp(-1024).hi] and TV upper bound one. Thus very large exact
    finite inputs can still certify when D<2*nu*s. D=0 or B=0 gives zero
    transferred radius; B=D stays D. A failed sufficient test is inconclusive.

    Invalid types or negative oscillation raise. Damaged/conditional sources,
    nonfinite bounds or exhausted combined budgets return None. New sealing
    requires the rigorous backend and no prior conditional libm history;
    strict state is restored and history is never cleared. No projection,
    root iteration, dense factorization or posterior recomputation occurs.
    """
    if not isinstance(unique, GaussianEntropyUnique):
        raise TypeError("unique must be a GaussianEntropyUnique")
    if isinstance(oscillation, bool) or not isinstance(oscillation, (int, Fraction)):
        raise TypeError("oscillation must be an exact integer or Fraction excluding bool")
    if oscillation < 0:
        raise ValueError("oscillation must be nonnegative")
    model, projector = unique.domain.model, unique.domain.projector
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    try:
        slots = _source_slots(unique)
        a._entries(slots)
        a.limits._work(4*slots+1024*(model.mode_count+1)*(model.dimension+1)+8192)
        a._work(4*slots)
        if not _unique_intact(unique):
            return None
        from omnibias.core.proof.certificate import (
            certificate_is_unconditional,
            encode_interval,
            make_certificate,
        )

        if not certificate_is_unconditional(unique.certificate):
            return None
        omega = a.exact(oscillation)
        maximum = a.exact(max(unique.squared_radii))
        base_bound = a.exact(unique.variance_upper_bound)
        threshold = a.mul(a.mul(Fraction(2), model.viscosity), model.time)
        if not 0 <= base_bound <= maximum or base_bound >= threshold:
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy bias region refuses prior conditional libm fallback history")
        a._work(512*(model.mode_count+1)*(model.dimension+1)+2048)
        with _strict():
            decay = _tilt_decay(omega)
            variation = _variation_upper(omega, a)
            bound = _tilted_bound(maximum, base_bound, decay, a)
            margin = a.add(threshold, -bound)
            if margin <= 0:
                return None
            contraction = a._coefficient(bound/threshold)
            curvature = a._coefficient(margin/a.mul(threshold, model.time))
            base_margin = a._coefficient(a.add(threshold, -base_bound)/threshold)
            shift_squared = a.mul(model.diameter_squared, a.mul(variation, variation))
            drift_squared = a._coefficient(shift_squared/a.mul(base_margin, base_margin))
            a._work(64)
            drift = _sqrt_upper(drift_squared, a.limits)
            certificate = make_certificate(
                claim="Uniform global Gaussian uniqueness and decision displacement for every original bias change with the declared oscillation bound.",
                payload={
                    "type": "gaussian_entropy_bias_region", "unique_certificate": deepcopy(unique.certificate),
                    "oscillation": _pair(omega), "max_squared_radius": _pair(maximum),
                    "tilt_decay": encode_interval(decay), "variance_upper_bound": _pair(bound),
                    "uniqueness_margin": _pair(margin), "contraction_bound": _pair(contraction),
                    "strong_convexity_bound": _pair(curvature), "total_variation_bound": _pair(variation),
                    "mean_shift_squared_bound": _pair(shift_squared), "displacement_squared_bound": _pair(drift_squared),
                    "displacement_bound": drift.hex(),
                    "argument": "All original and perturbed stationary/global decisions satisfy x=P_C(s*v+E_pi[a]), so they lie in the same bias-independent physical box K enclosing the full projected center hull. The source bounds E_pi[X]<=B throughout K for X=|a-c|² in [0,D]. Every bias perturbation with oscillation at most omega tilts these original probabilities by weights that, after a common positive rescaling, lie in [1,t], t=exp(omega). Put q=D*t*B/(D+(t-1)*B). The convex function phi(X)=max(X-q,t*(X-q)) lies below its endpoint chord on [0,D]. Its expectation is at most -q+(B/D)*(t*D-(t-1)*q)=0, hence E[w*(X-q)]<=0 and the tilted expected radius is at most q. D=0, B=0 and B=D have the recorded direct limits. The negative-exponential formula with a rigorous lower endpoint bounds q from above. Covariance is at most this expected radius times I on all K. Its strict margin below H=2nu*s yields a spatial Hessian at least (H-q)/(H*s) I on K. Every global minimizer lies in K, so strong monotonicity of the objective gradient plus the polyhedral normal cone gives uniqueness; coercivity supplies existence. Original and tilted posteriors at a fixed point have TV (half-L1) at most tanh(omega/4). Their means differ by at most Delta*TV, with Delta the center diameter. The ORIGINAL mean map is rho0=B/H Lipschitz on convex K. For any perturbed stationary point, nonexpansiveness of projection gives |x_new-x_base|<=rho0*|x_new-x_base|+Delta*TV. Thus optimizer displacement is at most Delta*TV/(1-rho0). The projection inputs q=s*v+mean obey the same bound by adding the original spatial mean change and the fixed-point tilt change. Exact endpoint arithmetic bounds the squared displacement, and an exact-comparison outward square root gives the recorded radius. Common bias shifts have zero oscillation and exactly zero drift. No shared active face is assumed.",
                    "scope": "Only the original logarithmic Gaussian mixture weights vary. Target, centers, positive age/viscosity and the nonempty closed polyhedron remain fixed. This is a uniform finite-region uniqueness and displacement claim, including unrestricted common bias shifts. Active-face persistence needs a separate strict witness replay. No finite-step Taylor remainder, base Jacobian valid at changed parameters, moving-geometry derivative, universal PDE solver, novelty theorem or external formal theorem-prover verification is asserted.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyBiasRegion)
        for name, value in (("unique", unique), ("oscillation", omega), ("max_squared_radius", maximum),
                            ("tilt_decay", decay), ("variance_upper_bound", bound), ("uniqueness_margin", margin),
                            ("contraction_bound", contraction), ("strong_convexity_bound", curvature),
                            ("total_variation_bound", variation), ("mean_shift_squared_bound", shift_squared),
                            ("displacement_squared_bound", drift_squared), ("displacement_bound", drift),
                            ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyBiasRegion", "certify_gaussian_entropy_bias_region"]
