# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gaussian KKT binding from exact projected centroids of logit groups."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_polyhedral import GaussianPolyhedralFace, _face_intact
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
_ZERO = Interval.point(0.0)


def _mixture(values: Vector, probabilities: Box, arithmetic: _Arithmetic) -> Interval:
    """Enclose a normalized mixture, retaining exact constant coefficients."""
    reference = values[0]
    result = _rational(reference)
    for value, probability in zip(values[1:], probabilities[1:], strict=True):
        difference = arithmetic.add(value, -reference)
        if difference:
            result = _add(result, _multiply(_rational(difference), probability))
    return result.intersect(Interval(_rational(min(values)).lo, _rational(max(values)).hi))


def _box_payload(box: Box) -> list[list[float]]:
    return [[value.lo, value.hi] for value in box]


def _matrix_payload(matrix: Matrix) -> list[list[list[int]]]:
    return [[_pair(value) for value in row] for row in matrix]


def _abel_suffixes(
    coefficients: Vector, groups: tuple[tuple[int, ...], ...], order: tuple[int, ...], arithmetic: _Arithmetic,
) -> Vector:
    """Cardinality-weighted suffix sums in ascending exact logit order."""
    suffixes = []
    total = Fraction(0)
    for index in reversed(order):
        total = arithmetic.add(total, arithmetic.mul(Fraction(len(groups[index])), coefficients[index]))
        suffixes.append(total)
    return tuple(reversed(suffixes))


@dataclass(frozen=True, init=False)
class GaussianProjectedEquilibrium:
    r"""Exact original KKT point with rigorously enclosed group weights.

    ``groups`` partition original modes by equal rational original logits at
    ``point``, in first-appearance order. Group centroids may differ in selected
    face normal directions. Each exact group force
    ``target-(point-group_centroid)/time`` is replayed as E.T times its row of
    ``group_multipliers``, where E contains original equalities then selected
    inequalities. The true multipliers are normalized group-weight mixtures.

    ``centered_centers[j]`` is center j minus its OWN group's centroid.
    ``group_centroid_offsets`` are relative to the first group's centroid;
    they are retained exactly, including normal offsets. Posterior weights
    follow original modes; group probabilities include group cardinalities.
    ``projected_centroid`` is the first centroid projected onto the original
    selected face tangent space; groupwise replay makes it common to all groups.

    All tight inequalities are retained. A selected row is weak only when all
    exact group multiplier coefficients vanish; it is strong when the exact
    cardinality-weighted Abel suffix sums in ascending logit order are all
    nonnegative and some positive, or a rigorous mixture lower bound is
    positive. This includes nonnegative coefficients with a positive one.
    Exact positivity remains valid if an enclosed
    floating lower endpoint underflows to zero. Omitted multipliers are zero.
    Labels refer to this valid multiplier representation. This record asserts
    neither a Hessian nor an ordinary/directional derivative.
    """

    face: GaussianPolyhedralFace
    target: Vector
    point: Vector
    groups: tuple[tuple[int, ...], ...]
    group_logits: Vector
    centered_centers: Matrix
    group_centroids: Matrix
    group_centroid_offsets: Matrix
    projected_centroid: Vector
    group_multipliers: Matrix
    posterior: Box
    group_probabilities: Box
    equality_multipliers: Box
    inequality_multipliers: Box
    slacks: Vector
    active_indices: tuple[int, ...]
    strong_indices: tuple[int, ...]
    weak_indices: tuple[int, ...]
    strong_convexity_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_projected_equilibrium")


def certify_gaussian_projected_equilibrium(
    face: GaussianPolyhedralFace, target: Sequence[int | Fraction], point: Sequence[int | Fraction],
) -> GaussianProjectedEquilibrium | None:
    r"""Certify one exact original constrained Gaussian equilibrium.

    Original logit differences and group centroids use source-relative exact
    arithmetic, avoiding common spatial squares and bias offsets. Every group
    force must belong to the selected face normal space, with its multipliers
    obtained by the precompiled left inverse and stationarity replayed exactly.
    This is a sufficient model-binding identity, not fitted probabilities.
    Selected face feasibility, all inequalities and q_base<1 are required.

    Rigorously enclose per-mode softmax weights, then multiply by cardinalities
    for group masses. A single group has an exact uniform-weight shortcut.
    Sort exact group logits alpha_g ascending, and let a_g be group size
    times its multiplier coefficient. For suffix sums S_k=sum_{j>=k} a_j,
    Abel's identity gives sum_g a_g*exp(alpha_g)=S_0*exp(alpha_0)
    +sum_{k>=1} S_k*(exp(alpha_k)-exp(alpha_{k-1})). Distinct exact logits
    make every exponential difference positive. Nonnegative suffix sums,
    with at least one positive, therefore prove strict positivity even for
    mixed coefficients or arbitrarily close logits. Otherwise a strictly
    positive rigorous mixture lower bound is required. A mixture merely
    containing zero remains inconclusive; no floating zero is classified as
    an exact weak row. The certificate stores the order and suffix evidence.

    Invalid types/shapes raise before preflight. Damaged dependencies, failed
    identities/signs, exhausted resources and nonfinite numerical enclosures
    return None without an infeasibility claim. Missing rigorous backend raises
    RuntimeError; strict sealing policy and sticky fallback history persist.
    Bounded O(m*d²+m*log(m+1)+k*d) work and retained exact/interval evidence are charged
    cumulatively. No solve, factorization, root iteration or model rebuild is
    performed. A sealed digest establishes integrity, not formal verification.
    """
    if not isinstance(face, GaussianPolyhedralFace):
        raise TypeError("face must be a GaussianPolyhedralFace")
    source = face.base_restriction.source_model
    dimension, count = source.dimension, source.mode_count
    _validate_vector(target, dimension, "target")
    _validate_vector(point, dimension, "point")
    rows, inequalities = len(face.base_restriction.constraints), len(face.inequalities)
    try:
        arithmetic = _Arithmetic(source.budget)
        arithmetic._entries(512*(count+rows+inequalities+1)*(dimension+1) + 128*(dimension+1)**2 + 1024)
        arithmetic._work(1024*(count+rows+inequalities+1)*(dimension+1)**2 + 512*(dimension+1)**2 + 4096)
        if not _face_intact(face) or face.base_restriction.model.contraction_bound >= 1:
            return None
        frozen_target = tuple(arithmetic.exact(value) for value in target)
        frozen_point = tuple(arithmetic.exact(value) for value in point)
        restriction = face.geometry.restriction
        if _linear(restriction.constraints, frozen_point, arithmetic) != restriction.rhs:
            return None
        slacks = tuple(arithmetic.add(upper, -_dot(row, frozen_point, arithmetic))
                       for row, upper in zip(face.inequalities, face.upper_bounds, strict=True))
        if any(value < 0 for value in slacks):
            return None
        relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                               zip(center, source.centers[0], strict=True)) for center in source.centers)
        position = tuple(arithmetic.add(value, -reference) for value, reference in
                         zip(frozen_point, source.centers[0], strict=True))
        denominator = arithmetic.mul(arithmetic.mul(Fraction(4), source.viscosity), source.time)
        partition: dict[Fraction, list[int]] = {}
        for index, (row, bias) in enumerate(zip(relative, source.biases, strict=True)):
            contrast = arithmetic.add(_dot(row, row, arithmetic),
                                      -arithmetic.mul(Fraction(2), _dot(row, position, arithmetic)))
            logit = arithmetic.add(arithmetic.add(bias, -source.biases[0]),
                                   -arithmetic._coefficient(contrast/denominator))
            partition.setdefault(logit, []).append(index)
        groups = tuple(tuple(indices) for indices in partition.values())
        logits = tuple(partition)
        relative_centroids = tuple(tuple(arithmetic._coefficient(arithmetic.sum(tuple(relative[index][axis] for index in group))
                                                                 /len(group))
                                         for axis in range(dimension)) for group in groups)
        centroids = tuple(tuple(arithmetic.add(reference, value) for reference, value in
                                zip(source.centers[0], centroid, strict=True)) for centroid in relative_centroids)
        offsets = tuple(tuple(arithmetic.add(value, -reference) for value, reference in
                              zip(centroid, relative_centroids[0], strict=True)) for centroid in relative_centroids)
        projected_centroid = _linear(restriction.projector, centroids[0], arithmetic)
        centered_list = [(Fraction(0),)*dimension]*count
        for group, centroid in zip(groups, relative_centroids, strict=True):
            for index in group:
                centered_list[index] = tuple(arithmetic.add(value, -average) for value, average in
                                             zip(relative[index], centroid, strict=True))
        centered = tuple(centered_list)
        inverse_time = arithmetic._coefficient(1/source.time)
        transpose = tuple(tuple(row[i] for row in restriction.constraints) for i in range(dimension))
        multipliers = []
        for centroid in relative_centroids:
            velocity = tuple(arithmetic.mul(inverse_time, arithmetic.add(value, -average))
                             for value, average in zip(position, centroid, strict=True))
            force = tuple(arithmetic.add(goal, -value) for goal, value in zip(frozen_target, velocity, strict=True))
            coefficients = _linear(face.geometry.left_inverse, force, arithmetic)
            if _linear(transpose, coefficients, arithmetic) != force:
                return None
            multipliers.append(coefficients)
        group_multipliers = tuple(multipliers)
        arithmetic._work(len(groups)*max(1, len(groups).bit_length()))
        logit_order = tuple(sorted(range(len(groups)), key=logits.__getitem__))
        if len(groups) == 1:
            posterior = (_rational(arithmetic._coefficient(Fraction(1, count))),)*count
            probabilities: Box = (Interval.point(1.0),)
            provenance = "not_used"
        else:
            mode_logits = [Fraction(0)]*count
            maximum = max(logits)
            for group, logit in zip(groups, logits, strict=True):
                shifted = arithmetic.add(logit, -maximum)
                for index in group:
                    mode_logits[index] = shifted
            with _strict():
                posterior = _probabilities(tuple(_rational(value) for value in mode_logits))
            probabilities = tuple(_multiply(_rational(Fraction(len(group))), posterior[group[0]])
                                  .intersect(Interval(0.0, 1.0)) for group in groups)
            provenance = backend_name()
        selected = tuple(_mixture(tuple(row[index] for row in group_multipliers), probabilities, arithmetic)
                         for index in range(len(restriction.constraints)))
        inequality_multipliers = [_ZERO]*inequalities
        positive = set()
        sign_proofs = []
        for index, column in zip(face.active_indices, range(rows, len(selected)), strict=True):
            coefficients = tuple(row[column] for row in group_multipliers)
            suffixes = _abel_suffixes(coefficients, groups, logit_order, arithmetic)
            enclosure = selected[column]
            if all(value == 0 for value in coefficients):
                enclosure = _ZERO
                proof_kind = "exact_zero"
            elif all(value >= 0 for value in suffixes) and any(value > 0 for value in suffixes):
                positive.add(index)
                enclosure = Interval(max(0.0, enclosure.lo), enclosure.hi)
                proof_kind = "abel_positive"
            elif enclosure.lo > 0:
                positive.add(index)
                proof_kind = "interval_positive"
            else:
                return None
            inequality_multipliers[index] = enclosure
            sign_proofs.append({"inequality_index": index, "kind": proof_kind,
                                "weighted_suffixes": [_pair(value) for value in suffixes]})
        active = tuple(index for index, value in enumerate(slacks) if value == 0)
        strong = tuple(index for index in active if index in positive)
        weak = tuple(index for index in active if index not in positive)
        modulus = arithmetic.mul(inverse_time, arithmetic.add(Fraction(1), -face.base_restriction.model.contraction_bound))
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Original Gaussian polyhedral equilibrium from exact groupwise projected stationarity and rigorous multiplier signs.",
            payload={
                "type": "gaussian_projected_equilibrium", "face_certificate": deepcopy(face.certificate),
                "target": [_pair(value) for value in frozen_target], "point": [_pair(value) for value in frozen_point],
                "groups": [list(group) for group in groups], "group_logits": [_pair(value) for value in logits],
                "ascending_group_logit_order": list(logit_order), "selected_multiplier_sign_proofs": sign_proofs,
                "centered_centers": _matrix_payload(centered), "group_centroids": _matrix_payload(centroids),
                "group_centroid_offsets": _matrix_payload(offsets), "group_multipliers": _matrix_payload(group_multipliers),
                "projected_centroid": [_pair(value) for value in projected_centroid],
                "posterior": _box_payload(posterior), "group_probabilities": _box_payload(probabilities),
                "equality_multipliers": _box_payload(selected[:rows]),
                "inequality_multipliers": _box_payload(tuple(inequality_multipliers)),
                "slacks": [_pair(value) for value in slacks], "active_indices": list(active),
                "strong_indices": list(strong), "weak_indices": list(weak), "strong_convexity_bound": _pair(modulus),
                "identities": ["original relative logits equal within each group", "C*point=rhs", "G*point<=upper_bounds",
                               "E.T*group_multiplier_g=target-(point-group_centroid_g)/time",
                               "mu=sum(group_probability_g*group_multiplier_g)", "mu>=0", "mu_i*slack_i=0"],
                "argument": "Each exact original logit group has uniform conditional mode weights and an exact centroid. Precompiled L gives groupwise multipliers, and E.T times each multiplier is replayed as that group's force. Their true normalized positive group masses therefore give original KKT stationarity. Rigorous per-mode softmax enclosures include group cardinalities when computing group masses. Selected inequality multipliers vanish only when every exact group coefficient vanishes. For exact ascending distinct logits alpha_g and coefficients a_g=group_size_g*multiplier_g, stored suffix sums S_k=sum_{j>=k} a_j satisfy the Abel identity sum_g a_g*exp(alpha_g)=S_0*exp(alpha_0)+sum_{k>=1} S_k*(exp(alpha_k)-exp(alpha_{k-1})). All exponential factors and differences are strictly positive, so nonnegative suffix sums with at least one positive prove a strictly positive multiplier, even when logits nearly coincide or interval lower bounds underflow. If this sufficient criterion is inconclusive, a strictly positive rigorous mixture lower bound is required. Omitted multipliers are zero, and all tight rows are retained. Feasibility and the base-tangent strong convexity bound certify the unique global polyhedral optimizer.",
                "scope": "Sufficient exact model-binding identity at the declared point. Group centroids may differ normally. No fitted weights, necessity assertion, neighborhood equilibrium, Hessian, ordinary derivative or directional derivative claim. Strong/weak labels refer to the recorded valid multiplier representation.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": provenance},
        )
        result = object.__new__(GaussianProjectedEquilibrium)
        for name, stored in (("face", face), ("target", frozen_target), ("point", frozen_point), ("groups", groups),
                             ("group_logits", logits), ("centered_centers", centered), ("group_centroids", centroids),
                             ("group_centroid_offsets", offsets), ("group_multipliers", group_multipliers),
                             ("projected_centroid", projected_centroid),
                             ("posterior", posterior), ("group_probabilities", probabilities),
                             ("equality_multipliers", selected[:rows]), ("inequality_multipliers", tuple(inequality_multipliers)),
                             ("slacks", slacks), ("active_indices", active), ("strong_indices", strong), ("weak_indices", weak),
                             ("strong_convexity_bound", modulus), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianProjectedEquilibrium", "certify_gaussian_projected_equilibrium"]
