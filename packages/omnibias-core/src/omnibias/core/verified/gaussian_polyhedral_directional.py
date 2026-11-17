# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact critical-cone calculus at Gaussian decision switching boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite, solve_affine
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_polyhedral import _face_intact
from omnibias.core.verified.gaussian_polyhedral_equilibrium import GaussianPolyhedralEquilibrium
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _equilibrium_intact(equilibrium: GaussianPolyhedralEquilibrium) -> bool:
    return _intact(equilibrium.certificate) and _face_intact(equilibrium.face)


@dataclass(frozen=True, init=False)
class GaussianCriticalFace:
    """Compiled linear response and exact cone of valid target directions.

    The critical cone uses all exactly tight inequalities at ``equilibrium``.
    Its positive-multiplier rows are equalities; its zero-multiplier rows
    remain inequalities. ``active_weak_indices`` proposes which of those
    weak rows to impose as equalities for this directional response.

    ``constraints`` stacks original equalities, strong rows, then selected
    weak rows. ``response`` maps direction xi to the equality-QP solution;
    ``multiplier_response`` maps xi to its multipliers. This solution is
    the true directional derivative when the certificate query
    verifies ``region_inequalities @ xi >= 0``. No ordinary Jacobian of the
    original optimizer at the base point is asserted. This is a sufficient
    direction region, not necessarily the largest region on which B is valid.
    """

    equilibrium: GaussianPolyhedralEquilibrium
    active_weak_indices: tuple[int, ...]
    constraints: Matrix
    response: Matrix
    multiplier_response: Matrix
    region_inequalities: Matrix
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_critical_face")


def compile_gaussian_critical_face(
    equilibrium: GaussianPolyhedralEquilibrium, active_weak_indices: Sequence[int] = (),
) -> GaussianCriticalFace | None:
    r"""Compile one exact linear region of the directional decision map.

    Select original inequality indices from equilibrium.weak_indices. The
    stacked rows E=[C;G_strong;G_selected_weak] must be independent; redundant
    selected rows raise ValueError. Unselected redundant weak rows are allowed
    and remain in the direction-feasibility check. No active-set enumeration
    or numerical QP proposal is trusted.

    With exact nullspace basis N and original Hessian H, compile
    B=N*(N.T*H*N)^(-1)*N.T, and M=(E*E.T)^(-1)*E*(I-H*B).
    Only the reduced Hessian must be positive definite. Reuse one reduced
    LDLT and one row-Gram LDLT across all right-hand sides. Replay E*B=0,
    H*B+E.T*M=I and B.T=B exactly. For a zero-dimensional nullspace B=0.

    Work/storage include dense responses, exact elimination and dependencies,
    under the source budget. Invalid input raises; damaged dependencies,
    exhausted arithmetic/resources or failed replay return None. No
    transcendental is evaluated; the shared strict sealing policy applies.
    """
    if not isinstance(equilibrium, GaussianPolyhedralEquilibrium):
        raise TypeError("equilibrium must be a GaussianPolyhedralEquilibrium")
    if not isinstance(active_weak_indices, Sequence) or isinstance(active_weak_indices, (str, bytes)):
        raise TypeError("active_weak_indices must be a finite sequence of integers")
    if any(type(index) is not int for index in active_weak_indices):
        raise TypeError("active weak indices must be integers, excluding bool")
    if len(active_weak_indices) > len(equilibrium.weak_indices):
        raise ValueError("selected indices must be distinct weak inequalities")
    source = equilibrium.face.base_restriction.source_model
    dimension, count, inequalities = source.dimension, source.mode_count, len(equilibrium.face.inequalities)
    if len(equilibrium.face.base_restriction.constraints)+len(equilibrium.strong_indices)+len(active_weak_indices) > dimension:
        # Independent selected rows cannot exceed the ambient dimension.
        # Check before the O(selected_count*weak_count) membership scan.
        raise ValueError("selected critical-face rows must have full row rank")
    arithmetic = _Arithmetic(source.budget)
    try:
        arithmetic._entries(1024*(count+inequalities+1)*(dimension+1) + 512*(dimension+1)**2 + 4096)
        arithmetic._work(2048*(count+inequalities+dimension+1)*(dimension+1)**2 + 2048*(dimension+1)**3 + 8192)
        if not _equilibrium_intact(equilibrium):
            return None
    except (ArithmeticError, ValueError):
        return None
    if len(set(active_weak_indices)) != len(active_weak_indices):
        raise ValueError("active weak indices must be distinct")
    if any(index not in equilibrium.weak_indices for index in active_weak_indices):
        raise ValueError("selected index is not an exactly weak active inequality")
    active = tuple(sorted(active_weak_indices))
    constraints = (equilibrium.face.base_restriction.constraints
                   + tuple(equilibrium.face.inequalities[i] for i in equilibrium.strong_indices)
                   + tuple(equilibrium.face.inequalities[i] for i in active))
    rows = len(constraints)
    try:
        if constraints:
            solved = solve_affine(constraints, None, arithmetic)
            basis = solved.homogeneous_basis
            rank = solved.rank
        else:
            basis = tuple(tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension))
            rank = 0
    except (ArithmeticError, ValueError):
        return None
    if rank != rows:
        raise ValueError("selected critical-face rows must have full row rank")
    try:
        if basis:
            h_basis = tuple(_linear(equilibrium.hessian, row, arithmetic) for row in basis)
            reduced = tuple(tuple(_dot(row, image, arithmetic) for image in h_basis) for row in basis)
            factor = factor_positive_definite(reduced, arithmetic)
            columns = tuple(factor.solve(tuple(row[axis] for row in basis), arithmetic) for axis in range(dimension))
            response = tuple(tuple(_dot(tuple(row[i] for row in basis), column, arithmetic) for column in columns)
                             for i in range(dimension))
        else:
            response = ((Fraction(0),)*dimension,)*dimension
        response_columns = tuple(tuple(row[j] for row in response) for j in range(dimension))
        h_response_columns = tuple(_linear(equilibrium.hessian, column, arithmetic) for column in response_columns)
        remainder_columns = tuple(tuple(arithmetic.add(Fraction(int(i == j)), -value) for i, value in enumerate(column))
                                  for j, column in enumerate(h_response_columns))
        if constraints:
            gram = tuple(tuple(_dot(left, right, arithmetic) for right in constraints) for left in constraints)
            factor = factor_positive_definite(gram, arithmetic)
            multiplier_columns = tuple(factor.solve(_linear(constraints, column, arithmetic), arithmetic)
                                       for column in remainder_columns)
            multipliers = tuple(tuple(column[i] for column in multiplier_columns) for i in range(rows))
        else:
            multipliers = ()
            multiplier_columns = ((),)*dimension
        if response_columns != response:
            return None
        for j, column in enumerate(response_columns):
            if any(_linear(constraints, column, arithmetic)):
                return None
            for i in range(dimension):
                normal = _dot(tuple(row[i] for row in constraints), multiplier_columns[j], arithmetic)
                if arithmetic.add(h_response_columns[j][i], normal) != int(i == j):
                    return None
        weak_offset = len(equilibrium.face.base_restriction.constraints)+len(equilibrium.strong_indices)
        region = tuple(multipliers[weak_offset+i] for i in range(len(active)))
        unselected = tuple(i for i in equilibrium.weak_indices if i not in active)
        region += tuple(tuple(-_dot(equilibrium.face.inequalities[i], column, arithmetic) for column in response_columns)
                        for i in unselected)
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact linear response and direction-cone inequalities for a Gaussian critical-cone QP face.",
            payload={
                "type": "gaussian_critical_face", "equilibrium_certificate": deepcopy(equilibrium.certificate),
                "active_weak_indices": list(active), "constraints": [[_pair(value) for value in row] for row in constraints],
                "nullspace_basis": [[_pair(value) for value in row] for row in basis],
                "response": [[_pair(value) for value in row] for row in response],
                "multiplier_response": [[_pair(value) for value in row] for row in multipliers],
                "region_inequalities": [[_pair(value) for value in row] for row in region],
                "identities": ["E*B=0", "H*B+E^T*M=I", "B^T=B"],
                "argument": "Global strong convexity on the original equality space implies N^T*H*N is positive definite for the selected critical-face nullspace. Exact reused-factor solves give B=N*(N^T*H*N)^-1*N^T and M=(E*E^T)^-1*E*(I-H*B). Homogeneous feasibility, stationarity for every target direction and symmetry are replayed exactly. Selected weak multipliers and unselected weak slacks give the rows of R, defining R*xi>=0. Strong-row derivative multipliers have no sign restriction.",
                "scope": "One compiled direction region, not active-set search or a single ordinary Jacobian of the original optimizer. A later query must check its exact region inequalities.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianCriticalFace)
        for name, value in (("equilibrium", equilibrium), ("active_weak_indices", active), ("constraints", constraints),
                            ("response", response), ("multiplier_response", multipliers),
                            ("region_inequalities", region), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


@dataclass(frozen=True, init=False)
class GaussianPolyhedralDirectional:
    """Exact one-sided Hadamard target derivative, with source/geometry fixed.

    ``value`` is lim_{t down to 0}(X(v+t*direction)-X(v))/t. This derivative
    can be nonlinear in direction, despite each compiled region being linear.
    ``equality_multipliers`` are QP multipliers in compiled E row order,
    including the selected weak rows. ``weak_multipliers`` and ``weak_slacks``
    follow ALL original inequality indices, with structural zeros at nonweak
    rows; only weak-row entries of these two arrays have a QP meaning.

    This gives no finite-step remainder, ordinary Jacobian/VJP, switching
    radius, parameter-changing-geometry derivative or higher derivative.
    """

    face: GaussianCriticalFace
    direction: Vector
    value: Vector
    equality_multipliers: Vector
    weak_multipliers: Vector
    weak_slacks: Vector
    quadratic_value: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_directional")


def certify_gaussian_polyhedral_directional(
    face: GaussianCriticalFace, direction: Sequence[Rational],
) -> GaussianPolyhedralDirectional | None:
    r"""Certify the exact directional derivative by matrix actions and signs.

    Fixed source/geometry and q_base<1 imply the derivative is the unique
    minimizer of .5*w.T*H*w-direction.T*w on the critical cone. Exact tight
    slacks and primal multipliers define that cone; no interval-containing-zero
    test is substituted for this classification. Validate selected weak
    multipliers and all weak slacks, then replay KKT/complementarity exactly.
    Strong-row derivative multipliers are unrestricted.

    No nonlinear iteration, active-set enumeration or linear solve occurs at
    query time. All input entries are exact int/Fraction. Invalid types/shapes
    raise; a wrong proposed face, damaged dependencies, exhausted resources
    or failed identity replay return None. Query work/storage are quadratic
    in dimension plus inequality checks and retained certificate data.
    """
    if not isinstance(face, GaussianCriticalFace):
        raise TypeError("face must be a GaussianCriticalFace")
    equilibrium = face.equilibrium
    source = equilibrium.face.base_restriction.source_model
    dimension, inequalities, count = source.dimension, len(equilibrium.face.inequalities), source.mode_count
    _validate_vector(direction, dimension, "direction")
    try:
        arithmetic = _Arithmetic(source.budget)
        arithmetic._entries(2048*(count+inequalities+1)*(dimension+1) + 1024*(dimension+1)**2 + 8192)
        arithmetic._work(4096*(count+inequalities+dimension+1)*(dimension+1)**2 + 16384)
        if not _intact(face.certificate) or not _equilibrium_intact(equilibrium):
            return None
        frozen = tuple(arithmetic.exact(value) for value in direction)
        if any(value < 0 for value in _linear(face.region_inequalities, frozen, arithmetic)):
            return None
        value = _linear(face.response, frozen, arithmetic)
        multipliers = _linear(face.multiplier_response, frozen, arithmetic)
        weak_slacks = [Fraction(0)]*inequalities
        weak_multipliers = [Fraction(0)]*inequalities
        for index in equilibrium.weak_indices:
            weak_slacks[index] = -_dot(equilibrium.face.inequalities[index], value, arithmetic)
        offset = len(equilibrium.face.base_restriction.constraints)+len(equilibrium.strong_indices)
        for index, multiplier in zip(face.active_weak_indices, multipliers[offset:], strict=True):
            weak_multipliers[index] = multiplier
        if any(slack < 0 for slack in weak_slacks) or any(multiplier < 0 for multiplier in weak_multipliers):
            return None
        if any(arithmetic.mul(a, b) for a, b in zip(weak_slacks, weak_multipliers, strict=True)):
            return None
        if any(_linear(face.constraints, value, arithmetic)):
            return None
        h_value = _linear(equilibrium.hessian, value, arithmetic)
        for i in range(dimension):
            normal = _dot(tuple(row[i] for row in face.constraints), multipliers, arithmetic)
            if arithmetic.add(h_value[i], normal) != frozen[i]:
                return None
        quadratic = arithmetic.add(arithmetic.mul(Fraction(1, 2), _dot(value, h_value, arithmetic)),
                                   -_dot(frozen, value, arithmetic))
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact one-sided Hadamard target derivative of the unique polyhedral Gaussian decision.",
            payload={
                "type": "gaussian_polyhedral_directional", "critical_face_certificate": deepcopy(face.certificate),
                "direction": [_pair(component) for component in frozen], "value": [_pair(component) for component in value],
                "equality_multipliers": [_pair(component) for component in multipliers],
                "weak_multipliers": [_pair(component) for component in weak_multipliers],
                "weak_slacks": [_pair(component) for component in weak_slacks], "quadratic_value": _pair(quadratic),
                "argument": "The exact equilibrium proves all tight rows and one nonnegative primal multiplier vector. Its critical cone is C*w=0, G_strong*w=0, G_weak*w<=0: on the feasible tangent cone, the zero first-order objective condition is equivalent to equality in every positive-multiplier row. Strong monotonicity bounds target difference quotients. Their second-order variational expansion forces every limit to minimize .5*w^T*H*w-xi^T*w on this cone. Positive tangent curvature gives a unique minimizer, hence the Hadamard directional limit. Exact QP feasibility, stationarity, nonnegative weak multipliers and complementarity prove that the returned B*xi is that minimizer.",
                "scope": "Fixed original Gaussian source, age, viscosity and all constraints; only target varies along the declared direction. No ordinary Jacobian/VJP, finite-step error bound, switching radius, higher derivative or derivative under moving geometry is asserted. Only weak-row entries of weak_slacks/weak_multipliers have QP meaning.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralDirectional)
        for name, stored in (("face", face), ("direction", frozen), ("value", value), ("equality_multipliers", multipliers),
                             ("weak_multipliers", tuple(weak_multipliers)), ("weak_slacks", tuple(weak_slacks)),
                             ("quadratic_value", quadratic), ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianCriticalFace", "GaussianPolyhedralDirectional", "certify_gaussian_polyhedral_directional",
           "compile_gaussian_critical_face"]
