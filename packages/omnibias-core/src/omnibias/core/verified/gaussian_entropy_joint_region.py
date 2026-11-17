# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite joint target/log-weight regions for globally unique Gaussian decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _validate_vector
from omnibias.core.verified.gaussian_entropy_bias_region import (
    _tilt_decay,
    _tilted_bound,
    _variation_upper,
)
from omnibias.core.verified.gaussian_entropy_unique import (
    GaussianEntropyUnique,
    _unique_intact,
    _unique_source_slots,
)
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Bounds = tuple[tuple[Fraction, Fraction], ...]


def _source_slots(unique: GaussianEntropyUnique) -> int:
    model = unique.domain.model
    return 2*_unique_source_slots(unique)+512*(model.mode_count+1)*(model.dimension+1)+12288


def _joint_region_source_slots(region: GaussianEntropyJointRegion) -> int:
    return _source_slots(region.unique)


def _joint_region_intact(region: GaussianEntropyJointRegion) -> bool:
    from omnibias.core.proof.certificate import certificate_is_unconditional, encode_interval

    if (not _intact(region.certificate) or not _unique_intact(region.unique)
            or not certificate_is_unconditional(region.certificate)
            or not certificate_is_unconditional(region.unique.certificate)):
        return False
    payload = region.certificate.get("payload")
    if not isinstance(payload, dict) or payload.get("unique_certificate") != region.unique.certificate:
        return False
    for name in ("bias_oscillation", "target_radius", "diameter_bound", "expansion_radius", "spatial_oscillation",
                 "total_oscillation", "max_squared_radius", "spatial_variance_upper_bound", "variance_upper_bound",
                 "spatial_contraction_bound", "contraction_bound", "uniqueness_margin", "strong_convexity_bound",
                 "total_variation_bound", "forcing_bound", "displacement_squared_bound"):
        if payload.get(name) != _pair(getattr(region, name)):
            return False
    return (payload.get("expanded_coordinate_bounds") == [[_pair(value) for value in pair]
                                                          for pair in region.expanded_coordinate_bounds]
            and payload.get("spatial_tilt_decay") == encode_interval(region.spatial_tilt_decay)
            and payload.get("tilt_decay") == encode_interval(region.tilt_decay)
            and payload.get("displacement_bound") == region.displacement_bound.hex())


@dataclass(frozen=True, init=False)
class GaussianEntropyJointRegion:
    r"""Uniform global guarantees on a target ball times a relative-bias region.

    All real changes with ||delta_v||_2<=target_radius and
    osc(delta_b)<=bias_oscillation are covered. Centers, positive heat age s,
    viscosity nu and the polyhedron stay fixed; common bias shifts are free.

    The curvature domain is the convex Euclidean expansion
    M=K+B_2(expansion_radius), expansion_radius=s*target_radius, of the source
    physical box K. ``expanded_coordinate_bounds`` is ONLY a coordinate
    enclosure of M. Its corners can lie outside the certified curvature
    domain. ``contains_physical`` checks membership in M by exact distance.

    ``diameter_bound`` is an outward rational upper bound on the center
    diameter Delta. Spatial logit spread is at most
    eta=diameter_bound*target_radius/(2*nu). ``spatial_variance_upper_bound``
    transfers the ORIGINAL radius moment B from K to M using eta.
    ``variance_upper_bound`` also includes the bias spread. Their ratios to
    H=2*nu*s are the spatial and joint contraction bounds. The joint bound's
    strict positive margin proves global uniqueness for every allowed change.

    Both exact optimizer and posterior projection-input movements are bounded
    by (s*V+Delta*tanh(omega/4))/(1-rho_spatial). ``forcing_bound`` encloses
    the numerator, ``displacement_squared_bound`` bounds the squared ratio,
    and ``displacement_bound`` is its outward square root. These movements
    are measured from exact original values, not a numerical proposal.

    No active face or finite-step accuracy of a base Jacobian is assumed.
    Membership queries use existing unconditional evidence without a new
    backend, solve or seal; invalid data, damaged proofs and resource failures
    raise. Their strict/backend history is preserved.
    """

    unique: GaussianEntropyUnique
    bias_oscillation: Fraction
    target_radius: Fraction
    diameter_bound: Fraction
    expansion_radius: Fraction
    expanded_coordinate_bounds: Bounds
    spatial_oscillation: Fraction
    total_oscillation: Fraction
    spatial_tilt_decay: Interval
    tilt_decay: Interval
    max_squared_radius: Fraction
    spatial_variance_upper_bound: Fraction
    variance_upper_bound: Fraction
    spatial_contraction_bound: Fraction
    contraction_bound: Fraction
    uniqueness_margin: Fraction
    strong_convexity_bound: Fraction
    total_variation_bound: Fraction
    forcing_bound: Fraction
    displacement_squared_bound: Fraction
    displacement_bound: float
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_joint_region")

    def _arithmetic(self) -> _Arithmetic:
        domain = self.unique.domain
        a = _Arithmetic(_combined_budget(domain.model.budget, domain.projector.budget))
        slots = _source_slots(self.unique)
        a._entries(slots)
        a._work(6*slots+512*(domain.model.mode_count+domain.model.dimension+1)+4096)
        if not _joint_region_intact(self):
            raise ValueError("joint Gaussian region proof or nested dependency is damaged or conditional")
        return a

    def contains_physical(self, point: Sequence[Rational]) -> bool:
        """Test exact membership in M, not feasibility in the polyhedron.

        The squared distance to a box is the sum of squared excesses from its
        coordinate intervals. This avoids treating outer-box corners as part
        of the Euclidean curvature domain. Work after proof scans is O(d).
        """
        _validate_vector(point, self.unique.domain.model.dimension, "point")
        a = self._arithmetic()
        exact = tuple(a.exact(value) for value in point)
        radius_squared = a.mul(self.expansion_radius, self.expansion_radius)
        distance_squared = Fraction(0)
        for value, (lower, upper) in zip(exact, self.unique.domain.physical_bounds, strict=True):
            excess = a.add(lower, -value) if value < lower else a.add(value, -upper) if value > upper else Fraction(0)
            if excess > self.expansion_radius:
                return False
            distance_squared = a.add(distance_squared, a.mul(excess, excess))
            if distance_squared > radius_squared:
                return False
        return True

    def contains_parameters(
        self, target_change: Sequence[Rational], bias_change: Sequence[Rational],
    ) -> bool:
        """Test exact target-ball and relative-bias membership; common shifts cancel."""
        model = self.unique.domain.model
        _validate_vector(target_change, model.dimension, "target_change")
        _validate_vector(bias_change, model.mode_count, "bias_change")
        a = self._arithmetic()
        target = tuple(a.exact(value) for value in target_change)
        bias = tuple(a.exact(value) for value in bias_change)
        if a.add(max(bias), -min(bias)) > self.bias_oscillation:
            return False
        norm_squared = Fraction(0)
        radius_squared = a.mul(self.target_radius, self.target_radius)
        for value in target:
            if abs(value) > self.target_radius:
                return False
            norm_squared = a.add(norm_squared, a.mul(value, value))
            if norm_squared > radius_squared:
                return False
        return True


def certify_gaussian_entropy_joint_region(
    unique: GaussianEntropyUnique, bias_oscillation: Rational, target_radius: Rational,
) -> GaussianEntropyJointRegion | None:
    r"""Certify simultaneous finite target/log-weight perturbations.

    A target change of norm at most V moves each projected center-hull point
    by at most s*V, hence all new stationary/global decisions lie in
    M=K+B_2(s*V). For x in M, a nearby y in K changes original logit contrasts
    by at most Delta*V/(2*nu). Transfer the source expected-radius bound once
    for this spatial spread, and once for spatial plus declared bias spread.
    A strict joint margin below 2*nu*s yields positive curvature on convex M
    and global uniqueness. The original field's spatial-only bound is used
    in the decision displacement denominator.

    Coordinate bounds enclose M but do NOT carry its curvature claim.
    Centers, time, viscosity and constraints remain fixed. Zero target radius
    recovers the bias-only theorem; numerical outward bounds may differ.

    Invalid types/negative radii raise. Conditional/damaged sources, exhausted
    budgets, insufficient margins and nonfinite bounds return None. Rigorous
    backend and untainted history are required for a new seal. No posterior
    recomputation, optimizer/projection search or matrix factorization occurs.
    """
    if not isinstance(unique, GaussianEntropyUnique):
        raise TypeError("unique must be a GaussianEntropyUnique")
    for name, value in (("bias_oscillation", bias_oscillation), ("target_radius", target_radius)):
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError(f"{name} must be an exact integer or Fraction excluding bool")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    model, projector = unique.domain.model, unique.domain.projector
    a = _Arithmetic(_combined_budget(model.budget, projector.budget))
    try:
        slots = _source_slots(unique)
        a._entries(slots)
        a.limits._work(4*slots+2048*(model.mode_count+1)*(model.dimension+1)+16384)
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
        omega, radius = a.exact(bias_oscillation), a.exact(target_radius)
        maximum, base_bound = a.exact(max(unique.squared_radii)), a.exact(unique.variance_upper_bound)
        threshold = a.mul(a.mul(Fraction(2), model.viscosity), model.time)
        if not 0 <= base_bound <= maximum or base_bound >= threshold:
            return None
        if libm_fallback_used():
            raise RuntimeError("joint Gaussian entropy region refuses prior conditional libm fallback history")
        a._work(1024*(model.mode_count+1)*(model.dimension+1)+4096)
        a._work(64)
        diameter = a.exact(Fraction(_sqrt_upper(model.diameter_squared, a.limits)))
        expansion = a.mul(model.time, radius)
        coordinate_bounds = tuple((a.add(lower, -expansion), a.add(upper, expansion))
                                  for lower, upper in unique.domain.physical_bounds)
        spatial_oscillation = a._coefficient(a.mul(diameter, radius)/a.mul(Fraction(2), model.viscosity))
        total_oscillation = a.add(spatial_oscillation, omega)
        with _strict():
            spatial_decay = _tilt_decay(spatial_oscillation)
            joint_decay = spatial_decay if not omega else _tilt_decay(total_oscillation)
            spatial_bound = _tilted_bound(maximum, base_bound, spatial_decay, a)
            bound = max(spatial_bound, _tilted_bound(maximum, base_bound, joint_decay, a))
            margin = a.add(threshold, -bound)
            if margin <= 0:
                return None
            spatial_contraction = a._coefficient(spatial_bound/threshold)
            contraction = a._coefficient(bound/threshold)
            curvature = a._coefficient(margin/a.mul(threshold, model.time))
            variation = _variation_upper(omega, a)
            forcing = a.add(expansion, a.mul(diameter, variation))
            spatial_margin = a.add(Fraction(1), -spatial_contraction)
            drift_squared = a._coefficient(a.mul(forcing, forcing)/a.mul(spatial_margin, spatial_margin))
            a._work(64)
            drift = _sqrt_upper(drift_squared, a.limits)
            certificate = make_certificate(
                claim="Uniform global Gaussian uniqueness and displacement throughout a finite joint target and log-weight region.",
                payload={
                    "type": "gaussian_entropy_joint_region", "unique_certificate": deepcopy(unique.certificate),
                    "bias_oscillation": _pair(omega), "target_radius": _pair(radius), "diameter_bound": _pair(diameter),
                    "expansion_radius": _pair(expansion),
                    "expanded_coordinate_bounds": [[_pair(value) for value in pair] for pair in coordinate_bounds],
                    "spatial_oscillation": _pair(spatial_oscillation), "total_oscillation": _pair(total_oscillation),
                    "spatial_tilt_decay": encode_interval(spatial_decay), "tilt_decay": encode_interval(joint_decay),
                    "max_squared_radius": _pair(maximum), "spatial_variance_upper_bound": _pair(spatial_bound),
                    "variance_upper_bound": _pair(bound), "spatial_contraction_bound": _pair(spatial_contraction),
                    "contraction_bound": _pair(contraction), "uniqueness_margin": _pair(margin),
                    "strong_convexity_bound": _pair(curvature), "total_variation_bound": _pair(variation),
                    "forcing_bound": _pair(forcing), "displacement_squared_bound": _pair(drift_squared),
                    "displacement_bound": drift.hex(),
                    "argument": "Let K be the source convex physical box enclosing P_C(s*v+conv{a_j}). For any perturbed stationary/global point x=P_C(s*(v+delta_v)+sum p_j*a_j), compare the projection of the SAME convex-center point at the old target. Nonexpansiveness places x within s*V of K. Thus every new global minimizer lies in the convex Minkowski expansion M=K+B_2(s*V), independently of biases. At any x in M choose y in K within s*V. The original Gaussian logit contrast between modes i,j changes by (a_i-a_j).(x-y)/(2nu*s), with oscillation at most Delta*V/(2nu). Original covariance on M is therefore bounded by the sharp transferred squared-radius expectation B_eta=D*B/[B+(D-B)*exp(-eta)]. Adding a bias change of spread at most omega bounds the joint logit spread by eta+omega and gives B_joint on the same M. Rigorous negative-exponential endpoints and exact arithmetic give the recorded upper bounds; a conservative joint maximum also dominates the spatial bound. Its strict margin below H=2nu*s yields positive spatial Hessian on convex M. Coercivity gives existence and normal-cone strong monotonicity on M excludes two global minimizers. At the SAME point, bias changes move the posterior mean by at most Delta*tanh(omega/4), with TV interpreted as half-L1. The ORIGINAL mean map is rho_spatial=B_eta/H Lipschitz on M. Comparing fixed-point equations gives |delta_x|<=s*V+rho_spatial*|delta_x|+Delta*TV, hence the recorded displacement radius. The posterior projection-input difference satisfies the identical bound. No equality of active faces is assumed. Exact-distance membership tests use M itself. The coordinate box K plus/minus s*V only encloses M and is NOT assigned the curvature guarantee.",
                    "scope": "All real target perturbations of Euclidean norm at most target_radius and all original log-weight perturbations with spread at most bias_oscillation. Centers, positive age/viscosity and polyhedral geometry remain fixed. Common bias shifts are unrestricted. Active-face persistence needs a separate proof. No curvature claim on outer-box corners outside M, finite-step Taylor accuracy of a base Jacobian, moving-geometry derivatives, universal PDE solver, novelty theorem or external formal theorem-prover verification is asserted.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyJointRegion)
        for name, stored in (("unique", unique), ("bias_oscillation", omega), ("target_radius", radius),
                             ("diameter_bound", diameter), ("expansion_radius", expansion),
                             ("expanded_coordinate_bounds", coordinate_bounds), ("spatial_oscillation", spatial_oscillation),
                             ("total_oscillation", total_oscillation), ("spatial_tilt_decay", spatial_decay), ("tilt_decay", joint_decay),
                             ("max_squared_radius", maximum), ("spatial_variance_upper_bound", spatial_bound),
                             ("variance_upper_bound", bound), ("spatial_contraction_bound", spatial_contraction),
                             ("contraction_bound", contraction), ("uniqueness_margin", margin), ("strong_convexity_bound", curvature),
                             ("total_variation_bound", variation), ("forcing_bound", forcing),
                             ("displacement_squared_bound", drift_squared), ("displacement_bound", drift), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyJointRegion", "certify_gaussian_entropy_joint_region"]
