# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact polyhedral Gaussian equilibria at rational equal-logit points."""

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
class GaussianPolyhedralEquilibrium:
    r"""Exact KKT data for the unique global polyhedral Gaussian optimum.

    Every original Gaussian logit agrees exactly at ``point``, so posterior
    probabilities and the original velocity/Hessian are rational. ``hessian``
    is D_x U_source, including normal directions; it may be indefinite off
    the original equality tangent space. ``strong_convexity_bound`` applies
    on that tangent space, using the original base restriction's q<1.

    ``active_indices`` contains ALL exactly tight original inequalities,
    not just the face's proposed rows. ``strong_indices`` have positive KKT
    multipliers; ``weak_indices`` are tight with zero multipliers, including
    omitted or redundant tight rows. These are the particular valid KKT
    multipliers obtained from the selected independent face representation;
    uniqueness of multipliers across redundant tight rows is not asserted.

    The certificate makes no ordinary derivative or active-set stability
    claim at weak constraints. Construct through the factory below.
    """

    face: GaussianPolyhedralFace
    target: Vector
    point: Vector
    posterior: Vector
    velocity: Vector
    hessian: Matrix
    equality_multipliers: Vector
    inequality_multipliers: Vector
    slacks: Vector
    active_indices: tuple[int, ...]
    strong_indices: tuple[int, ...]
    weak_indices: tuple[int, ...]
    strong_convexity_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_equilibrium")


def certify_gaussian_polyhedral_equilibrium(
    face: GaussianPolyhedralFace, target: Sequence[int | Fraction], point: Sequence[int | Fraction],
) -> GaussianPolyhedralEquilibrium | None:
    r"""Certify an exact feasible equal-logit KKT point without exponentials.

    Target and point entries must be exact integers/Fractions, excluding
    bool. Require q_base<1 on the original equality tangent space. The point
    must satisfy the selected face equations and every original inequality.
    With r_j=a_j-a_0 and xrel=point-a_0, equality of original logits is checked
    using (b_j-b_0)-(|r_j|²-2*r_j.xrel)/(4*nu*time)=0. Common absolute spatial
    and bias offsets are removed before any square or numerical conversion.

    Uniform posterior gives exact U and H=(I-Cov(a)/(2*nu*time))/time.
    The precompiled L supplies selected KKT multipliers L*(target-U); the
    original stationarity identity is replayed exactly. Target need not be
    U: active constraints can balance a nonzero target-minus-velocity.
    Omitted inequality multipliers are zero and all exactly tight rows are
    retained in the active/weak sets. Nonnegative multipliers and strong
    convexity certify the unique global constrained optimizer.

    Invalid types/shapes raise. Unequal logits, failed feasibility or KKT
    identities, negative inequality multipliers, q_base>=1, damaged evidence
    and exhausted exact budgets return None, without an infeasibility claim.
    Preflight bounds O(m*d²+k*d+d²) exact arithmetic and retained metadata.
    No iteration, factorization or transcendental evaluation occurs. Strict
    sealing retains the shared backend policy; a digest is not a formal proof.
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
        arithmetic._work(256*(count+rows+inequalities+1)*(dimension+1)**2 + 512*(dimension+1)**2 + 1024)
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
        for row, bias in zip(relative, source.biases, strict=True):
            contrast = arithmetic.add(_dot(row, row, arithmetic),
                                      -arithmetic.mul(Fraction(2), _dot(row, position, arithmetic)))
            logit = arithmetic.add(arithmetic.add(bias, -source.biases[0]),
                                   -arithmetic._coefficient(contrast/denominator))
            if logit:
                return None
        probability = arithmetic._coefficient(Fraction(1, count))
        posterior = (probability,)*count
        mean = tuple(arithmetic.mul(probability, arithmetic.sum(tuple(row[axis] for row in relative)))
                     for axis in range(dimension))
        inverse_time = arithmetic._coefficient(1/source.time)
        velocity = tuple(arithmetic.mul(inverse_time, arithmetic.add(value, -average))
                         for value, average in zip(position, mean, strict=True))
        centered = tuple(tuple(arithmetic.add(value, -average) for value, average in zip(row, mean, strict=True))
                         for row in relative)
        covariance_scale = arithmetic._coefficient(1/arithmetic.mul(arithmetic.mul(Fraction(2), source.viscosity), source.time))
        hessian = tuple(tuple(arithmetic.mul(inverse_time, arithmetic.add(Fraction(int(i == j)),
                            -arithmetic.mul(covariance_scale, arithmetic.mul(probability, arithmetic.sum(tuple(
                                arithmetic.mul(row[i], row[j]) for row in centered))))))
                              for j in range(dimension)) for i in range(dimension))
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
            claim="Exact original Gaussian KKT equilibrium and all tight inequalities at the unique polyhedral minimum.",
            payload={
                "type": "gaussian_polyhedral_equilibrium", "face_certificate": deepcopy(face.certificate),
                "target": [_pair(value) for value in frozen_target], "point": [_pair(value) for value in frozen_point],
                "posterior": [_pair(value) for value in posterior], "velocity": [_pair(value) for value in velocity],
                "hessian": [[_pair(value) for value in row] for row in hessian],
                "equality_multipliers": [_pair(value) for value in selected_multipliers[:rows]],
                "inequality_multipliers": [_pair(value) for value in multipliers],
                "slacks": [_pair(value) for value in slacks], "active_indices": list(active),
                "strong_indices": list(strong), "weak_indices": list(weak), "strong_convexity_bound": _pair(modulus),
                "identities": ["original_logit_j-original_logit_0=0", "C*point=rhs", "G*point<=upper_bounds",
                               "U_source(point)-target+C^T*lambda+G^T*mu=0", "mu>=0", "mu_i*slack_i=0"],
                "argument": "Exact original logit contrasts establish uniform posterior. Exact moments yield the original Gaussian velocity and full ambient Hessian. The point satisfies selected face equations and every inequality. Precompiled L yields selected multipliers, original stationarity is replayed, and omitted multipliers are zero. Nonnegative inequality multipliers give KKT sufficiency under q_base<1 and global strong convexity on the original equality tangent. Every exactly tight inequality is retained, including unselected or redundant rows with zero multipliers. Thus the point is the unique global polyhedral optimizer.",
                "scope": "Exact equal-logit equilibrium only. The full ambient Hessian need not be positive definite. Strong/weak labels refer to this valid multiplier representation; redundant tight-row multipliers need not be unique. No ordinary derivative, strict active-set stability, directional derivative or neighborhood radius is asserted.",
            },
            honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralEquilibrium)
        for name, stored in (("face", face), ("target", frozen_target), ("point", frozen_point), ("posterior", posterior),
                            ("velocity", velocity), ("hessian", hessian),
                            ("equality_multipliers", selected_multipliers[:rows]), ("inequality_multipliers", tuple(multipliers)),
                            ("slacks", slacks), ("active_indices", active), ("strong_indices", strong), ("weak_indices", weak),
                            ("strong_convexity_bound", modulus), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralEquilibrium", "certify_gaussian_polyhedral_equilibrium"]
