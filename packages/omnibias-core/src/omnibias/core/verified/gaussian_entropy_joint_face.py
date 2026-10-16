# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable global Gaussian faces under finite target and log-weight changes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_entropy_bias_face import _propagate_face
from omnibias.core.verified.gaussian_entropy_face import (
    GaussianEntropyFace,
    _face_intact,
    _face_source_slots,
)
from omnibias.core.verified.gaussian_entropy_joint_region import (
    GaussianEntropyJointRegion,
    _joint_region_intact,
    _joint_region_source_slots,
)
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Bounds = tuple[tuple[Fraction, Fraction], ...]
Box = tuple[Interval, ...]


def _source_slots(face: GaussianEntropyFace, region: GaussianEntropyJointRegion) -> int:
    model, projector = face.residual.unique.domain.model, face.residual.unique.domain.projector
    return (_face_source_slots(face)+_joint_region_source_slots(region)
            +128*(len(projector.inequalities)+1)*(model.dimension+1)+256*(model.dimension+1)**2+4096)


def _joint_face_source_slots(result: GaussianEntropyJointFace) -> int:
    """Retained joint-face/source/certificate reservation for later queries."""
    return _source_slots(result.face, result.region)


def _joint_face_intact(result: GaussianEntropyJointFace) -> bool:
    """Replay source snapshots and data bindings after reserving source scans."""
    from omnibias.core.proof.certificate import encode_interval

    if not _intact(result.certificate) or not _face_intact(result.face) or not _joint_region_intact(result.region):
        return False
    payload = result.certificate.get("payload")
    if not isinstance(payload, dict):
        return False
    return (payload.get("face_certificate") == result.face.certificate
            and payload.get("region_certificate") == result.region.certificate
            and result.face.residual.unique.certificate == result.region.unique.certificate
            and payload.get("projection_input_bounds") == [[_pair(value) for value in pair] for pair in result.projection_input_bounds]
            and payload.get("optimizer_bounds") == [[_pair(value) for value in pair] for pair in result.optimizer_bounds]
            and payload.get("projection_multipliers") == [encode_interval(value) for value in result.projection_multipliers]
            and payload.get("slacks") == [encode_interval(value) for value in result.slacks])


@dataclass(frozen=True, init=False)
class GaussianEntropyJointFace:
    r"""One stable face throughout a finite joint target and bias region.

    The source region supplies uniqueness and displacement bounds for the
    declared Euclidean target ball and bias-oscillation ball. All resulting
    original global optimizers have exactly the active rows in ``face``;
    Euclidean projection at each true posterior input is locally affine with
    the same tangent projector. Multiplier representations need not be unique.

    Exact coordinate bounds use the region's expanded containment box. They
    are not clipped to the old target's physical box. Curvature is certified
    only on the region's Euclidean enlargement, not everywhere in its larger
    coordinate enclosure. The geometry propagation needs containment only.
    Centers, viscosity, time and constraints remain fixed.
    """

    face: GaussianEntropyFace
    region: GaussianEntropyJointRegion
    projection_input_bounds: Bounds
    optimizer_bounds: Bounds
    projection_multipliers: Box
    slacks: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_joint_face")


def certify_gaussian_entropy_joint_face(
    face: GaussianEntropyFace, region: GaussianEntropyJointRegion,
) -> GaussianEntropyJointFace | None:
    """Propagate the cached stable face over a joint finite perturbation region.

    Both sources must bind the same unconditional uniqueness certificate.
    The region bounds displacement of the true optimizer and its posterior
    projection input. The shared affine propagation kernel requires strict
    active multiplier and inactive slack margins over that entire enclosure.
    Target variation uses region.expanded_coordinate_bounds for containment.

    Invalid record types raise. Mismatched/damaged/conditional proofs, failed
    sufficient margins, exhausted combined resources or nonfinite arithmetic
    return None. New strict sealing requires a rigorous backend and refuses
    prior conditional-libm history without clearing it. No projection query,
    factorization, posterior evaluation or optimizer search is performed.
    """
    if not isinstance(face, GaussianEntropyFace):
        raise TypeError("face must be a GaussianEntropyFace")
    if not isinstance(region, GaussianEntropyJointRegion):
        raise TypeError("region must be a GaussianEntropyJointRegion")
    domain, other = face.residual.unique.domain, region.unique.domain
    limits = _combined_budget(_combined_budget(domain.model.budget, domain.projector.budget),
                              _combined_budget(other.model.budget, other.projector.budget))
    a = _Arithmetic(limits)
    dimension, count = domain.model.dimension, len(domain.projector.inequalities)
    try:
        slots = _source_slots(face, region)
        a._entries(slots)
        a.limits._work(4*slots+256*(dimension+count+1)*(dimension+1)**2+4096)
        a._work(4*slots)
        if not _face_intact(face) or not _joint_region_intact(region):
            return None
        if face.residual.unique.certificate != region.unique.certificate:
            return None
        from omnibias.core.proof.certificate import (
            certificate_is_unconditional,
            encode_interval,
            make_certificate,
        )

        if not certificate_is_unconditional(face.certificate) or not certificate_is_unconditional(region.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy joint-face sealing refuses prior conditional libm fallback history")
        with _strict():
            drift = a.exact(Fraction(region.displacement_bound))
            propagated = _propagate_face(face, drift, region.expanded_coordinate_bounds, a)
            if propagated is None:
                return None
            inputs, optimizer, multipliers, slacks = propagated
            certificate = make_certificate(
                claim="The globally selected Gaussian optimizer keeps the same stable projection face throughout the certified joint target-and-bias region.",
                payload={
                    "type": "gaussian_entropy_joint_face", "face_certificate": deepcopy(face.certificate),
                    "region_certificate": deepcopy(region.certificate),
                    "projection_input_bounds": [[_pair(value) for value in pair] for pair in inputs],
                    "optimizer_bounds": [[_pair(value) for value in pair] for pair in optimizer],
                    "projection_multipliers": [encode_interval(value) for value in multipliers],
                    "slacks": [encode_interval(value) for value in slacks],
                    "argument": "The sources bind the same exact base uniqueness proof. For every declared target and bias perturbation, the joint region proves a unique original global optimizer and bounds both its spatial displacement and its true projection-input displacement by d. Inflate the base true input bounds by [-d,d] coordinatewise. The shared face kernel uses the exact sample x=P_C(q0), independent normal rows E, their cached Gram inverse M, tangent projector R and full-positive active-row witness lambda0. For delta=q-q0, xhat=x+R*delta and lambda=lambda0 plus M*E*delta on the E coordinates satisfy q-xhat=G^T*lambda. Every active slack vanishes structurally; all active multiplier and inactive slack enclosures remain strictly positive. Original projection KKT therefore gives xhat=P_C(q), which equals the perturbed global optimizer, with exactly the same active rows. Strict affine margins make projection locally affine with derivative R at every such input. The optimizer enclosure intersects the affine image, the displaced base optimizer bounds and the region's expanded coordinate containment bounds. With target radius V, the relevant geometric domain is M=K+B_2(s*V); the coordinate expansion encloses M and all new optimizer images, but no curvature over the coordinate expansion's corners is assumed. Only containment is used by this face kernel.",
                    "scope": "Joint finite target and original log-weight changes in the source region; fixed centers, time, viscosity and polyhedron. Supports redundant, opposite and zero constraint rows. Common bias gauge shifts are unrestricted. Refusal is inconclusive; no new projection solve, posterior evaluation, factorization, search or external formal theorem-prover verification is asserted.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyJointFace)
        for name, stored in (("face", face), ("region", region), ("projection_input_bounds", inputs),
                             ("optimizer_bounds", optimizer), ("projection_multipliers", multipliers),
                             ("slacks", slacks), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyJointFace", "certify_gaussian_entropy_joint_face"]
