# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global Gaussian decisions and locally stable derivatives on polyhedra."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import (
    GaussianAffineRestriction,
    InconsistentAffineConstraints,
    _dot,
    _validate_vector,
    restrict_gaussian_heat,
)
from omnibias.core.verified.gaussian_affine_adjoint import (
    GaussianAffineAdjointGeometry,
    compile_gaussian_affine_adjoint,
)
from omnibias.core.verified.gaussian_affine_parameters import (
    GaussianAffineParameterVJP,
    certify_gaussian_affine_parameter_vjp,
)
from omnibias.core.verified.gaussian_affine_sensitivity import _binary_vector
from omnibias.core.verified.gaussian_affine_stationarity import (
    GaussianAffineStationarity,
    certify_gaussian_affine_stationarity,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
_ZERO = Interval.point(0.0)


@dataclass(frozen=True, init=False)
class GaussianPolyhedralFace:
    """Exact geometry of a proposed active set for C*x=d, G*x<=h.

    ``base_restriction`` represents only the original equalities. ``geometry``
    stacks these with active inequality rows, all against the ORIGINAL
    Gaussian family. Stacked rows must be independent. Active indices are
    sorted, distinct and zero-based. Compilation certifies neither optimality
    nor feasibility of the inactive inequalities, and performs no search.

    ``slack_directions`` are P_face*G_i and ``slack_offsets`` are
    h_i-G_i*a'_0, where a'_0 is the first transformed face center. These
    exact data evaluate slacks with normal directions removed before rounding.
    """

    base_restriction: GaussianAffineRestriction
    inequalities: Matrix
    upper_bounds: Vector
    active_indices: tuple[int, ...]
    geometry: GaussianAffineAdjointGeometry
    slack_directions: Matrix
    slack_offsets: Vector
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_polyhedral_face")


def compile_gaussian_polyhedral_face(
    base_restriction: GaussianAffineRestriction, inequalities: Sequence[Sequence[Rational]],
    upper_bounds: Sequence[Rational], active_indices: Sequence[int] = (),
) -> GaussianPolyhedralFace | None:
    """Compile a proposed face; do not select or trust its active set.

    All geometric data must be exact int/Fraction. Full row rank of the
    original equalities and stacked active rows is required. Redundant rows
    raise ValueError; inconsistent active equations raise
    InconsistentAffineConstraints. Invalid shapes/types/indices raise.
    Exhausted budgets or damaged dependencies return None. No contraction
    condition is imposed at compilation. Work includes exact restriction,
    reused-factor adjoint compilation and O(k*d²) slack projection; all
    operations share an aggregate resource reservation.
    """
    if not isinstance(base_restriction, GaussianAffineRestriction):
        raise TypeError("base_restriction must be a GaussianAffineRestriction")
    if not isinstance(inequalities, Sequence) or isinstance(inequalities, (str, bytes)):
        raise TypeError("inequalities must be a finite exact matrix")
    dimension = base_restriction.dimension
    for row in inequalities:
        _validate_vector(row, dimension, "inequality row")
    _validate_vector(upper_bounds, len(inequalities), "upper_bounds")
    if not isinstance(active_indices, Sequence) or isinstance(active_indices, (str, bytes)):
        raise TypeError("active_indices must be a finite sequence of integers")
    if any(type(index) is not int for index in active_indices):
        raise TypeError("active indices must be integers, excluding bool")
    if len(active_indices) > len(inequalities):
        raise ValueError("active indices must be distinct and within the inequality matrix")
    try:
        # Bound the duplicate-check allocation before constructing a set.
        # The larger aggregate reservation below also covers this linear work.
        base_restriction.source_model.budget._entries(len(active_indices))
        base_restriction.source_model.budget._work(len(active_indices))
    except ValueError:
        return None
    if len(set(active_indices)) != len(active_indices):
        raise ValueError("active indices must be distinct")
    if any(index < 0 or index >= len(inequalities) for index in active_indices):
        raise ValueError("active index outside the inequality matrix")
    source = base_restriction.source_model
    rows, count, inequalities_count = len(base_restriction.constraints), source.mode_count, len(inequalities)
    arithmetic = _Arithmetic(source.budget)
    try:
        arithmetic._entries(512*(count+rows+inequalities_count+1)*(dimension+1) + 128*(dimension+1)**2 + 2048)
        arithmetic._work(2048*(count+rows+inequalities_count+dimension+1)*(dimension+1)**2
                         + 2048*(dimension+1)**3 + 4096)
        if active_indices:
            # Restriction rebuilds the source Gaussian model and its MeanMap;
            # their diameter comparisons require quadratic work in mode count.
            arithmetic._work(512*(count+1)*(dimension+1)**2 + 128*count*count*dimension)
        if not _intact(base_restriction.certificate):
            return None
        matrix = tuple(tuple(arithmetic.exact(value) for value in row) for row in inequalities)
        upper = tuple(arithmetic.exact(value) for value in upper_bounds)
        active = tuple(sorted(active_indices))
    except (ArithmeticError, ValueError):
        return None
    if base_restriction.constraint_rank != rows:
        raise ValueError("polyhedral face compilation requires full row rank of the base equalities")
    try:
        # Re-restrict the original family. Treating the already transformed
        # base model as the source would change original-parameter derivatives.
        restriction = base_restriction if not active else restrict_gaussian_heat(
            source, base_restriction.constraints+tuple(matrix[i] for i in active),
            base_restriction.rhs+tuple(upper[i] for i in active),
        )
    except InconsistentAffineConstraints:
        raise
    except (ArithmeticError, ValueError):
        return None
    if restriction is None:
        return None
    if restriction.constraint_rank != rows+len(active):
        raise ValueError("stacked active constraints must have full row rank")
    try:
        geometry = compile_gaussian_affine_adjoint(restriction)
        if geometry is None:
            return None
        directions = tuple(_linear(restriction.projector, row, arithmetic) for row in matrix)
        offsets = tuple(arithmetic.add(bound, -_dot(row, restriction.model.centers[0], arithmetic))
                        for row, bound in zip(matrix, upper, strict=True))
        if any(any(directions[index]) or offsets[index] for index in active):
            return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact compiled active-face geometry and slack identities for Gaussian polyhedral decisions.",
            payload={
                "type": "gaussian_polyhedral_face", "base_certificate": deepcopy(base_restriction.certificate),
                "geometry_certificate": deepcopy(geometry.certificate), "active_indices": list(active),
                "inequalities": [[_pair(value) for value in row] for row in matrix],
                "upper_bounds": [_pair(value) for value in upper],
                "slack_directions": [[_pair(value) for value in row] for row in directions],
                "slack_offsets": [_pair(value) for value in offsets],
                "argument": "Stack original equalities and proposed active inequalities against the original source Gaussian family. Exact restriction and adjoint geometry certify independent stacked rows. For every point x on the face, h_i-G_i*x=(h_i-G_i*a'_0)-(P_face*G_i).(x-a'_0). Active rows have identically zero offset and direction, replayed exactly.",
                "scope": "Geometry only. No active-set selection, inactive feasibility, multiplier sign or optimality is asserted.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianPolyhedralFace)
        for name, value in (("base_restriction", base_restriction), ("inequalities", matrix), ("upper_bounds", upper),
                            ("active_indices", active), ("geometry", geometry), ("slack_directions", directions),
                            ("slack_offsets", offsets), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


def _face_intact(face: GaussianPolyhedralFace) -> bool:
    return all(_intact(cert) for cert in (face.certificate, face.base_restriction.certificate,
                                         face.geometry.certificate, face.geometry.restriction.certificate))


def _slack(face: GaussianPolyhedralFace, index: int, box: Box, arithmetic: _Arithmetic) -> Interval:
    """Evaluate a face slack with exact rational endpoint arithmetic."""
    lower = upper = face.slack_offsets[index]
    for coefficient, reference, value in zip(face.slack_directions[index],
                                             face.geometry.restriction.model.centers[0], box, strict=True):
        if coefficient:
            a = arithmetic.mul(coefficient, arithmetic.add(Fraction(value.lo), -reference))
            b = arithmetic.mul(coefficient, arithmetic.add(Fraction(value.hi), -reference))
            lower = arithmetic.add(lower, -max(a, b))
            upper = arithmetic.add(upper, -min(a, b))
    return Interval(_rational(lower).lo, _rational(upper).hi)


@dataclass(frozen=True, init=False)
class GaussianPolyhedralMinimum:
    """Unique GLOBAL optimizer, with separate active-set stability evidence.

    The certified root satisfies every original equality and inequality.
    Its enclosing box need not consist entirely of feasible points. Inactive
    multipliers and active slacks are identically zero. A false
    ``active_set_stable`` means strict margins were not certified; it does
    not weaken the global-minimum claim or prove nondifferentiability.
    No neighborhood radius or accuracy tolerance is asserted.
    """

    face: GaussianPolyhedralFace
    stationarity: GaussianAffineStationarity
    root_enclosure: Box
    equality_multipliers: Box
    inequality_multipliers: Box
    slacks: Box
    strong_convexity_bound: Fraction
    active_set_stable: bool
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_minimum")


def certify_gaussian_polyhedral_minimum(
    face: GaussianPolyhedralFace, target: Sequence[Rational], scaled_point: Sequence[float],
) -> GaussianPolyhedralMinimum | None:
    r"""Certify a proposed active-face optimizer over the whole polyhedron.

    Require q_base<1 on the ORIGINAL equality tangent space, not merely on
    the smaller proposed face. Then m=(1-q_base)/s is a global strong
    convexity modulus. Affine stationarity, nonnegative active multipliers
    and nonnegative inactive slacks imply F(y)-F(X)>=m*||y-X||²/2 for every
    feasible y. No Slater condition is needed for this sufficiency argument.

    Strict positive active multipliers AND inactive slacks additionally
    certify a locally stable active set, enabling the derivative wrapper.
    Both checks use outward interval endpoints without feasibility tolerances.
    No inverse iteration, active-set search or matrix factorization occurs
    in this query. Invalid inputs raise; failed gates, damaged dependencies,
    exhausted budgets and nonfinite arithmetic return None, never a claim
    that the problem is infeasible. A required rigorous backend must exist.
    """
    if not isinstance(face, GaussianPolyhedralFace):
        raise TypeError("face must be a GaussianPolyhedralFace")
    model = face.geometry.restriction.model
    _validate_vector(target, model.dimension, "target")
    point = _binary_vector(scaled_point, model.dimension, "scaled_point")
    try:
        arithmetic = _Arithmetic(model.budget)
        dimension, count = model.dimension, model.mode_count
        rows, inequalities = len(face.base_restriction.constraints), len(face.inequalities)
        arithmetic._entries(1024*(count+rows+inequalities+1)*(dimension+1) + 256*(dimension+1)**2 + 4096)
        arithmetic._work(8192*(count+rows+inequalities+1)*(dimension+1)**2 + 8*model._evaluation_work() + 8192)
        if not _face_intact(face) or face.base_restriction.model.contraction_bound >= 1:
            return None
        frozen = tuple(arithmetic.exact(value) for value in target)
        stationary = certify_gaussian_affine_stationarity(face.geometry, frozen, point)
        if stationary is None or not _intact(stationary.certificate):
            return None
        active = set(face.active_indices)
        slack = tuple(_ZERO if i in active else _slack(face, i, stationary.root_enclosure, arithmetic)
                      for i in range(inequalities))
        multipliers = [_ZERO]*inequalities
        for index, multiplier in zip(face.active_indices, stationary.primal_multiplier[rows:], strict=True):
            multipliers[index] = multiplier
        if any(slack[i].lo < 0 for i in range(inequalities) if i not in active):
            return None
        if any(multipliers[i].lo < 0 for i in active):
            return None
        stable = all(multipliers[i].lo > 0 for i in active) and all(slack[i].lo > 0 for i in range(inequalities) if i not in active)
        modulus = arithmetic._coefficient(arithmetic.add(Fraction(1), -face.base_restriction.model.contraction_bound)/model.time)
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="Unique global Gaussian optimum on the declared polyhedron, with optional strict active-set margins.",
            payload={
                "type": "gaussian_polyhedral_minimum", "face_certificate": deepcopy(face.certificate),
                "stationarity_certificate": deepcopy(stationary.certificate),
                "root_enclosure": [encode_interval(value) for value in stationary.root_enclosure],
                "equality_multipliers": [encode_interval(value) for value in stationary.primal_multiplier[:rows]],
                "inequality_multipliers": [encode_interval(value) for value in multipliers],
                "slacks": [encode_interval(value) for value in slack], "strong_convexity_bound": _pair(modulus),
                "active_set_stable": stable,
                "argument": "The original equality tangent contraction q_base<1 gives global strong convexity m=(1-q_base)/s on that affine space. The certified face root obeys original-objective stationarity and exact active equations. Nonnegative active multipliers and inactive slacks prove primal/dual feasibility and complementarity. For any feasible y, F(y)-F(X)>=-mu.(G_active*y-h_active)+m*||y-X||^2/2>=m*||y-X||^2/2. Independent stacked rows, strict active multipliers and strict inactive slacks imply a nonsingular equality KKT system and a locally unchanged active set when all strict margins hold.",
                "scope": "The root is globally optimal, not every point of the enclosure. active_set_stable certifies existence of a parameter neighborhood, not its radius. Non-strict margins still certify optimality but do not justify the derivative wrapper. Failed certification is inconclusive; face contraction alone is insufficient.",
            }, honesty={"unproven_claim": False, "unconditional_transcendentals": bool(stationary.posterior)
                        or stationary.root_certificate.root_certificate is not None, "continuum_pde_claim": False},
            meta={"transcend_backend": stationary.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianPolyhedralMinimum)
        for name, value in (("face", face), ("stationarity", stationary), ("root_enclosure", stationary.root_enclosure),
                            ("equality_multipliers", stationary.primal_multiplier[:rows]),
                            ("inequality_multipliers", tuple(multipliers)), ("slacks", slack),
                            ("strong_convexity_bound", modulus), ("active_set_stable", stable), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


@dataclass(frozen=True, init=False)
class GaussianPolyhedralParameterVJP:
    """Original-parameter VJP enclosures inside a certified stable active set.

    Inactive inequality coefficients and bounds have exactly zero local
    derivatives. Fields follow the original row order, independently of the
    active-set proposal order. This does not define derivatives at a switch,
    a finite perturbation radius, or error in an external loss upstream.
    """

    minimum: GaussianPolyhedralMinimum
    base_vjp: GaussianAffineParameterVJP
    target_gradient: Box
    center_gradient: IntervalMatrix
    bias_gradient: Box
    viscosity_gradient: Interval
    time_gradient: Interval
    equality_gradient: IntervalMatrix
    equality_rhs_gradient: Box
    inequality_gradient: IntervalMatrix
    inequality_rhs_gradient: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_polyhedral_parameter_vjp")


def certify_gaussian_polyhedral_parameter_vjp(
    minimum: GaussianPolyhedralMinimum, upstream: Sequence[float], proposal: Sequence[float],
) -> GaussianPolyhedralParameterVJP | None:
    """Certify original parameter derivatives only after strict-margin proof.

    Reuse the certified minimum's target and scaled root proposal. ``proposal``
    is an untrusted target adjoint for the declared fixed binary64 upstream.
    Map the existing all-parameter affine VJP back to original equality and
    inequality rows. Invalid inputs raise; non-strict margins, damaged
    dependencies, numerical or budget failure return None. No derivative
    at an active-set switch is inferred from a face derivative.
    """
    if not isinstance(minimum, GaussianPolyhedralMinimum):
        raise TypeError("minimum must be a GaussianPolyhedralMinimum")
    face = minimum.face
    model = face.geometry.restriction.model
    direction = _binary_vector(upstream, model.dimension, "upstream")
    proposed = _binary_vector(proposal, model.dimension, "proposal")
    try:
        arithmetic = _Arithmetic(model.budget)
        dimension, count = model.dimension, model.mode_count
        rows, inequalities = len(face.base_restriction.constraints), len(face.inequalities)
        arithmetic._entries(2048*(count+rows+inequalities+1)*(dimension+1) + 512*(dimension+1)**2 + 8192)
        arithmetic._work(16384*(count+rows+inequalities+1)*(dimension+1)**2 + 16*model._evaluation_work() + 16384)
        stationary = minimum.stationarity
        dependencies = (minimum.certificate, stationary.certificate, stationary.root_certificate.certificate)
        if not minimum.active_set_stable or not _face_intact(face) or not all(_intact(cert) for cert in dependencies):
            return None
        root = stationary.root_certificate.root_certificate
        if root is not None and not _intact(root.certificate):
            return None
        # An affine root has no floating proposal; any validated zero vector
        # is sufficient because the existing derivative checker solves that
        # affine case exactly. Otherwise reuse the declared residual proposal.
        point = (0.0,)*dimension if root is None else root.scaled_point
        base = certify_gaussian_affine_parameter_vjp(face.geometry, stationary.target, point, direction, proposed)
        if base is None or not _intact(base.certificate):
            return None
        inequality_gradient = [(_ZERO,)*dimension]*inequalities
        inequality_rhs_gradient = [_ZERO]*inequalities
        for offset, index in enumerate(face.active_indices, start=rows):
            inequality_gradient[index] = base.constraint_gradient[offset]
            inequality_rhs_gradient[index] = base.rhs_gradient[offset]
        from omnibias.core.proof.certificate import encode_interval, make_certificate

        certificate = make_certificate(
            claim="True original-parameter VJP enclosures for a globally optimal Gaussian decision with a locally stable active set.",
            payload={
                "type": "gaussian_polyhedral_parameter_vjp", "minimum_certificate": deepcopy(minimum.certificate),
                "affine_parameter_certificate": deepcopy(base.certificate),
                "equality_gradient": [[encode_interval(value) for value in row] for row in base.constraint_gradient[:rows]],
                "equality_rhs_gradient": [encode_interval(value) for value in base.rhs_gradient[:rows]],
                "inequality_gradient": [[encode_interval(value) for value in row] for row in inequality_gradient],
                "inequality_rhs_gradient": [encode_interval(value) for value in inequality_rhs_gradient],
                "argument": "Strict inactive slacks, strict active multipliers, independent stacked rows and positive base-equality tangent Hessian imply a locally unchanged active set and nonsingular KKT system. Hence the polyhedral optimizer locally equals the original-source affine optimizer on the stacked active equations. Its certified parameter VJP maps to original equality/inequality row order; all inactive inequality derivatives vanish identically in this neighborhood.",
                "scope": "Fixed declared binary64 upstream, first derivatives at the certified model and target only. No perturbation radius, switching derivative, higher derivative, direct loss parameter dependence or external-upstream error bound is asserted.",
            }, honesty={"unproven_claim": False, "unconditional_transcendentals": bool(base.posterior)
                        or base.root_certificate is not None or bool(stationary.posterior)
                        or root is not None, "continuum_pde_claim": False},
            meta={"transcend_backend": stationary.certificate["meta"]["transcend_backend"]
                  if base.certificate["meta"]["transcend_backend"] == "not_used" else base.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianPolyhedralParameterVJP)
        for name, value in (("minimum", minimum), ("base_vjp", base), ("target_gradient", base.target_gradient),
                            ("center_gradient", base.center_gradient), ("bias_gradient", base.bias_gradient),
                            ("viscosity_gradient", base.viscosity_gradient), ("time_gradient", base.time_gradient),
                            ("equality_gradient", base.constraint_gradient[:rows]), ("equality_rhs_gradient", base.rhs_gradient[:rows]),
                            ("inequality_gradient", tuple(inequality_gradient)),
                            ("inequality_rhs_gradient", tuple(inequality_rhs_gradient)), ("certificate", certificate)):
            object.__setattr__(result, name, value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianPolyhedralFace", "GaussianPolyhedralMinimum", "GaussianPolyhedralParameterVJP",
           "certify_gaussian_polyhedral_minimum", "certify_gaussian_polyhedral_parameter_vjp", "compile_gaussian_polyhedral_face"]
