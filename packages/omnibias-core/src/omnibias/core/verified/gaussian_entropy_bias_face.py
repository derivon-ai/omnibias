# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite log-weight regions preserving a globally selected projection face."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot
from omnibias.core.verified.gaussian_entropy_bias_region import (
    GaussianEntropyBiasRegion,
    _region_intact,
    _region_source_slots,
)
from omnibias.core.verified.gaussian_entropy_face import (
    GaussianEntropyFace,
    _face_intact,
    _face_source_slots,
    _interval,
    _linear_bounds,
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
_ZERO = Interval.point(0.0)


def _source_slots(face: GaussianEntropyFace, region: GaussianEntropyBiasRegion) -> int:
    model, projector = face.residual.unique.domain.model, face.residual.unique.domain.projector
    return (_face_source_slots(face)+_region_source_slots(region)
            +128*(len(projector.inequalities)+1)*(model.dimension+1)+256*(model.dimension+1)**2+4096)


def _bias_face_source_slots(result: GaussianEntropyBiasFace) -> int:
    """Reservation for both bound dependencies and the finite face outputs."""
    return _source_slots(result.face, result.region)


def _bias_face_intact(result: GaussianEntropyBiasFace) -> bool:
    """Check snapshots and data bindings after the caller reserves all scans."""
    from omnibias.core.proof.certificate import encode_interval

    if not _intact(result.certificate) or not _face_intact(result.face) or not _region_intact(result.region):
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


def _propagate_face(
    face: GaussianEntropyFace, drift: Fraction, domain_bounds: Bounds, arithmetic: _Arithmetic,
) -> tuple[Bounds, Bounds, Box, Box] | None:
    """Replay affine face margins on a proved displacement/containment region.

    The caller validates the source, reserves aggregate storage and work, and
    guarantees both optimizer and true projection-input displacement <=drift.
    ``domain_bounds`` must contain all new optimizers. This kernel makes no
    curvature assertion on that box and no certificate/backend operation.
    """
    a, domain = arithmetic, face.residual.unique.domain
    dimension, count = domain.model.dimension, len(domain.projector.inequalities)
    a._work(64*(dimension+count+1)*(dimension+1)**2)
    inputs = tuple((a.add(lower, -drift), a.add(upper, drift))
                   for lower, upper in face.projection_input_bounds)
    delta = tuple((a.add(lower, -source), a.add(upper, -source))
                  for (lower, upper), source in zip(inputs, face.residual.projection_source, strict=True))
    columns = tuple(tuple(row[j] for row in face.normal_rows) for j in range(dimension))
    left = tuple(tuple(_dot(row, column, a) for column in columns) for row in face.gram_inverse)
    multipliers = [(_ZERO if not value else _interval((value, value))) for value in face.witness_multipliers]
    for index, row in zip(face.basis_indices, left, strict=True):
        lower, upper = _linear_bounds(row, delta, a)
        base = face.witness_multipliers[index]
        multipliers[index] = _interval((a.add(base, lower), a.add(base, upper)))
    active = set(face.active_indices)
    if any(multipliers[index].lo <= 0 for index in active):
        return None
    tangent_columns = tuple(tuple(row[j] for row in face.tangent_projector) for j in range(dimension))
    slacks = []
    for index, (row, bound) in enumerate(zip(domain.projector.inequalities, domain.projector.upper_bounds, strict=True)):
        if index in active:
            slacks.append(_ZERO)
            continue
        initial = a.add(bound, -_dot(row, face.residual.point, a))
        projected_row = tuple(_dot(row, column, a) for column in tangent_columns)
        change = _linear_bounds(projected_row, delta, a)
        slack = _interval((a.add(initial, -change[1]), a.add(initial, -change[0])))
        if slack.lo <= 0:
            return None
        slacks.append(slack)
    optimizer = []
    for axis, row in enumerate(face.tangent_projector):
        change = _linear_bounds(row, delta, a)
        sample = face.residual.point[axis]
        base_lower, base_upper = face.optimizer_bounds[axis]
        physical_lower, physical_upper = domain_bounds[axis]
        lower = max(a.add(sample, change[0]), a.add(base_lower, -drift), physical_lower)
        upper = min(a.add(sample, change[1]), a.add(base_upper, drift), physical_upper)
        if lower > upper:
            return None
        optimizer.append((lower, upper))
    return inputs, tuple(optimizer), tuple(multipliers), tuple(slacks)


@dataclass(frozen=True, init=False)
class GaussianEntropyBiasFace:
    r"""One stable face for every bias change in a certified oscillation ball.

    For every real delta-b with max(delta-b)-min(delta-b)<=region.oscillation,
    the original constrained global optimizer is unique and has exactly the
    active rows recorded in ``face``. Common bias shifts are unrestricted.
    The Euclidean projection at each true posterior input is locally affine
    with the same tangent projector. Strict full-row multiplier witnesses
    allow redundant, opposite and zero rows without claiming a unique dual.

    ``projection_input_bounds`` and ``optimizer_bounds`` are exact rational
    coordinate bounds over the whole bias region. Multiplier and slack boxes
    retain all original row coordinates; active slacks and inactive
    multipliers are structural zero. The original target, centers, viscosity,
    time and constraints are fixed. This is a finite bias stability result,
    not a statement about derivatives of a numerical solver.
    """

    face: GaussianEntropyFace
    region: GaussianEntropyBiasRegion
    projection_input_bounds: Bounds
    optimizer_bounds: Bounds
    projection_multipliers: Box
    slacks: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_bias_face")


def certify_gaussian_entropy_bias_face(
    face: GaussianEntropyFace, region: GaussianEntropyBiasRegion,
) -> GaussianEntropyBiasFace | None:
    """Propagate a finite bias-region displacement through cached face data.

    Both records must bind the same unconditional uniqueness certificate.
    Inflate the existing true projection-input enclosure by the region's
    Euclidean displacement bound, then replay the cached affine primal/dual
    formulas. All active multipliers and inactive slacks must remain strictly
    positive. Refusal is conservative and does not prove a face change.

    No projection query, new factorization, optimizer search, posterior or
    transcendental evaluation is performed. All source budgets are combined;
    aggregate source/copy/work reservations precede integrity scans and exact
    arithmetic. Invalid argument types raise. Failed margins, mismatched or
    damaged sources, exhausted resources and nonfinite enclosures return None.
    New sealing requires the rigorous backend and refuses sticky conditional
    history, without changing or clearing that history.
    """
    if not isinstance(face, GaussianEntropyFace):
        raise TypeError("face must be a GaussianEntropyFace")
    if not isinstance(region, GaussianEntropyBiasRegion):
        raise TypeError("region must be a GaussianEntropyBiasRegion")
    domain = face.residual.unique.domain
    other = region.unique.domain
    limits = _combined_budget(_combined_budget(domain.model.budget, domain.projector.budget),
                              _combined_budget(other.model.budget, other.projector.budget))
    a = _Arithmetic(limits)
    dimension, count = domain.model.dimension, len(domain.projector.inequalities)
    try:
        source = _source_slots(face, region)
        a._entries(source)
        a.limits._work(4*source+256*(dimension+count+1)*(dimension+1)**2+4096)
        a._work(4*source)
        if not _face_intact(face) or not _region_intact(region):
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
            raise RuntimeError("Gaussian entropy bias-face sealing refuses prior conditional libm fallback history")
        with _strict():
            drift = a.exact(Fraction(region.displacement_bound))
            propagated = _propagate_face(face, drift, domain.physical_bounds, a)
            if propagated is None:
                return None
            inputs, optimizer, multipliers, slacks = propagated
            certificate = make_certificate(
                claim="The globally selected Gaussian optimizer keeps the same stable projection face throughout the certified finite bias-oscillation region.",
                payload={
                    "type": "gaussian_entropy_bias_face", "face_certificate": deepcopy(face.certificate),
                    "region_certificate": deepcopy(region.certificate),
                    "projection_input_bounds": [[_pair(value) for value in pair] for pair in inputs],
                    "optimizer_bounds": [[_pair(value) for value in pair] for pair in optimizer],
                    "projection_multipliers": [encode_interval(value) for value in multipliers],
                    "slacks": [encode_interval(value) for value in slacks],
                    "argument": "Both sources bind the same exact uniqueness certificate and hence the same fixed model, target and polyhedron. The region certificate gives a unique global optimizer for every real bias change with oscillation at most omega, and bounds both optimizer and true posterior projection-input displacement from their base values by d. Inflating the base true q enclosure by [-d,d] in each coordinate therefore encloses every perturbed q. Relative to the exact original sample q0 and x=P_C(q0), use the cached independent normal rows E, Gram inverse M, tangent projector R and strictly positive full-active-row witness lambda0. For delta=q-q0, xhat=x+R*delta and lambda=lambda0 with M*E*delta added only on the selected E coordinates satisfy q-xhat=G^T*lambda. All recorded active rows have structural zero slack. The strict multiplier and inactive-slack enclosures prove every original projection KKT condition throughout the region, so xhat=P_C(q) is the perturbed global optimizer and has exactly the same active rows. Strict affine margins also make projection locally affine with derivative R at every such q. The returned exact optimizer box intersects the affine image with the unchanged source physical box and the displaced base optimizer box, each independently valid. Redundant, opposite and zero rows retain their original coordinates; no uniqueness of multiplier representation is needed.",
                    "scope": "Finite changes of original Gaussian log weights only, with fixed target, centers, time, viscosity and constraints. Common bias gauge shifts are unrestricted. Refusal is inconclusive. No posterior recomputation, projection query, factorization, numerical search, arbitrary constraint derivative or external formal theorem-prover verification is asserted.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyBiasFace)
        for name, stored in (("face", face), ("region", region), ("projection_input_bounds", inputs),
                             ("optimizer_bounds", tuple(optimizer)), ("projection_multipliers", tuple(multipliers)),
                             ("slacks", tuple(slacks)), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyBiasFace", "certify_gaussian_entropy_bias_face"]
