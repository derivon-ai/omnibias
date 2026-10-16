# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Gaussian KKT points from equal centroids of rational logit groups."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_polyhedral import GaussianPolyhedralFace, _face_intact
from omnibias.core.verified.inverse_gaussian_affine import _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True, init=False)
class GaussianGroupedEquilibrium:
    r"""Exact original KKT point from a sufficient equal-centroid identity.

    ``groups`` partitions original mode indices by exactly equal original
    logits at ``point``. Groups and ``group_logits`` follow first appearance;
    logits are relative to the first mode. Each group's unweighted center
    centroid equals ``centroid``, proving that the Gaussian posterior mean
    also equals this exact rational vector, without evaluating exponentials.
    ``centered_centers`` are original centers minus that common centroid.

    Group probabilities are generally unequal. No posterior or Hessian is
    asserted by this record, and the centroid identity is only at the bound
    point; it is not a constant-mean claim in a neighborhood.

    All exactly tight inequalities appear in ``active_indices``. Positive
    KKT multipliers define ``strong_indices``; all tight zero-multiplier rows,
    including omitted/redundant rows, appear in ``weak_indices``. These labels
    refer to the recorded valid multiplier representation. No ordinary or
    directional derivative is asserted by this artifact.
    """

    face: GaussianPolyhedralFace
    target: Vector
    point: Vector
    groups: tuple[tuple[int, ...], ...]
    group_logits: Vector
    centroid: Vector
    centered_centers: Matrix
    velocity: Vector
    equality_multipliers: Vector
    inequality_multipliers: Vector
    slacks: Vector
    active_indices: tuple[int, ...]
    strong_indices: tuple[int, ...]
    weak_indices: tuple[int, ...]
    strong_convexity_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_grouped_equilibrium")


def certify_gaussian_grouped_equilibrium(
    face: GaussianPolyhedralFace, target: Sequence[int | Fraction], point: Sequence[int | Fraction],
) -> GaussianGroupedEquilibrium | None:
    r"""Bind an exact grouped-logit equilibrium of the original Gaussian model.

    The supplied model, target and point remain exact rational data. Compute
    original relative logits (b_j-b_0)-(|r_j|²-2*r_j.(point-a_0))/(4*nu*time),
    r_j=a_j-a_0, then partition by exact equality. If every group's unweighted
    centroid is c, its contribution to the posterior first moment is
    |group|*exp(logit_group)*c; dividing by the partition sum gives mean c.
    This is a sufficient identity, not fitted probabilities or an evaluation
    of those exponential weights. Groups may have different cardinalities.

    Require q_base<1, selected face feasibility, every inequality, and exact
    original KKT stationarity. Precompiled L maps target-U to the selected
    multipliers; that stationarity identity is replayed and inequality
    multipliers must be nonnegative. Omitted multipliers are zero and every
    tight row is retained. Target need not equal the original velocity.

    Invalid exact types/shapes raise. Unequal group centroids, failed KKT or
    feasibility, q_base>=1, damaged dependencies and resource exhaustion
    return None without an infeasibility claim. Source-relative arithmetic
    avoids common spatial squares and bias offsets. Preflight covers grouping,
    O(m*d+k*d+d²) arithmetic, exact coefficient growth and retained evidence.
    No numerical solve, factorization or transcendental evaluation occurs;
    shared strict sealing policy still applies. A digest is not a formal proof.
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
        arithmetic._entries(256*(count+rows+inequalities+1)*(dimension+1) + 128*(dimension+1)**2 + 512)
        arithmetic._work(512*(count+rows+inequalities+1)*(dimension+1) + 512*(dimension+1)**2 + 2048)
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
        relative_centroid = relative_centroids[0]
        if any(value != relative_centroid for value in relative_centroids[1:]):
            return None
        centroid = tuple(arithmetic.add(reference, value) for reference, value in
                         zip(source.centers[0], relative_centroid, strict=True))
        centered = tuple(tuple(arithmetic.add(value, -average) for value, average in zip(row, relative_centroid, strict=True))
                         for row in relative)
        inverse_time = arithmetic._coefficient(1/source.time)
        velocity = tuple(arithmetic.mul(inverse_time, arithmetic.add(value, -average))
                         for value, average in zip(position, relative_centroid, strict=True))
        force = tuple(arithmetic.add(goal, -value) for goal, value in zip(frozen_target, velocity, strict=True))
        selected_multipliers = _linear(face.geometry.left_inverse, force, arithmetic)
        transpose = tuple(tuple(row[i] for row in restriction.constraints) for i in range(dimension))
        if _linear(transpose, selected_multipliers, arithmetic) != force:
            return None
        multipliers = [Fraction(0)]*inequalities
        for index, value in zip(face.active_indices, selected_multipliers[rows:], strict=True):
            multipliers[index] = value
        if any(value < 0 for value in multipliers):
            return None
        active = tuple(index for index, value in enumerate(slacks) if value == 0)
        strong = tuple(index for index in active if multipliers[index] > 0)
        weak = tuple(index for index in active if multipliers[index] == 0)
        modulus = arithmetic.mul(inverse_time, arithmetic.add(Fraction(1), -face.base_restriction.model.contraction_bound))
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact original Gaussian polyhedral equilibrium from common centroids of exact logit groups.",
            payload={
                "type": "gaussian_grouped_equilibrium", "face_certificate": deepcopy(face.certificate),
                "target": [_pair(value) for value in frozen_target], "point": [_pair(value) for value in frozen_point],
                "groups": [list(group) for group in groups], "group_logits": [_pair(value) for value in logits],
                "centroid": [_pair(value) for value in centroid],
                "centered_centers": [[_pair(value) for value in row] for row in centered],
                "velocity": [_pair(value) for value in velocity],
                "equality_multipliers": [_pair(value) for value in selected_multipliers[:rows]],
                "inequality_multipliers": [_pair(value) for value in multipliers],
                "slacks": [_pair(value) for value in slacks], "active_indices": list(active),
                "strong_indices": list(strong), "weak_indices": list(weak), "strong_convexity_bound": _pair(modulus),
                "identities": ["relative_original_logits equal within each group", "mean(center_j in group)=centroid",
                               "C*point=rhs", "G*point<=upper_bounds",
                               "U_source(point)-target+C^T*lambda+G^T*mu=0", "mu>=0", "mu_i*slack_i=0"],
                "argument": "Partition original rational logit contrasts by exact equality at the declared point. Every group has the same exact unweighted center centroid c. Its total first-moment contribution is group_size*exp(group_logit)*c, so the normalized posterior mean equals c regardless of group weights. Exact U=(point-c)/time follows without exponentials. Selected face feasibility and all inequalities hold. Compiled L gives selected multipliers, original stationarity is replayed and omitted multipliers are zero. Nonnegative inequality multipliers and q_base<1 certify the unique global polyhedral optimizer. Every tight inequality, including omitted redundant zero-multiplier rows, is retained.",
                "scope": "Sufficient exact model-binding identity at the declared point. No fitted posterior weights, necessity assertion, posterior/Hessian enclosure, constant-mean neighborhood, ordinary derivative or directional derivative claim. Strong/weak labels refer to the recorded valid multiplier representation.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianGroupedEquilibrium)
        for name, stored in (("face", face), ("target", frozen_target), ("point", frozen_point), ("groups", groups),
                             ("group_logits", logits), ("centroid", centroid), ("centered_centers", centered),
                             ("velocity", velocity), ("equality_multipliers", selected_multipliers[:rows]),
                             ("inequality_multipliers", tuple(multipliers)), ("slacks", slacks),
                             ("active_indices", active), ("strong_indices", strong), ("weak_indices", weak),
                             ("strong_convexity_bound", modulus), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianGroupedEquilibrium", "certify_gaussian_grouped_equilibrium"]
