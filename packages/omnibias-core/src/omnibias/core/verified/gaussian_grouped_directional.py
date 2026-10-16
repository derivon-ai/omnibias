# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual-certified switching derivatives with nonuniform Gaussian weights."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite, solve_affine
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _add, _multiply, _square, _zero
from omnibias.core.verified.gaussian_affine import _check_geometry, _dot, _validate_vector
from omnibias.core.verified.gaussian_affine_adjoint import _check_identities
from omnibias.core.verified.gaussian_affine_sensitivity import _binary_vector, _norm_upper
from omnibias.core.verified.gaussian_grouped_equilibrium import GaussianGroupedEquilibrium
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_polyhedral import _face_intact
from omnibias.core.verified.gaussian_projected_equilibrium import GaussianProjectedEquilibrium
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair, _sqrt_upper
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.transcend import backend_name

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
Equilibrium = GaussianGroupedEquilibrium | GaussianProjectedEquilibrium
_ZERO = Interval.point(0.0)


def _equilibrium_intact(equilibrium: Equilibrium) -> bool:
    return _intact(equilibrium.certificate) and _face_intact(equilibrium.face)


def _interval_action(matrix: IntervalMatrix, vector: Box) -> Box:
    output = []
    for row in matrix:
        total = _ZERO
        for a, b in zip(row, vector, strict=True):
            total = _add(total, _multiply(a, b))
        output.append(total)
    return tuple(output)


def _box_payload(box: Box) -> list[list[float]]:
    return [[value.lo, value.hi] for value in box]


def _matrix_payload(matrix: Matrix) -> list[list[list[int]]]:
    return [[_pair(value) for value in row] for row in matrix]


def _moment(left: Matrix, right: Matrix, group: tuple[int, ...], arithmetic: _Arithmetic) -> Matrix:
    return tuple(tuple(arithmetic._coefficient(arithmetic.sum(tuple(arithmetic.mul(left[k][i], right[k][j]) for k in group))
                                               /len(group)) for j in range(len(right[0]))) for i in range(len(left[0])))


def _mix(matrices: tuple[Matrix, ...], probabilities: Box, arithmetic: _Arithmetic) -> IntervalMatrix:
    """Use sum pi=1 before rounding, preserving constant entries and zeros."""
    if not matrices[0]:
        return ()
    result = []
    for i, row in enumerate(matrices[0]):
        output = []
        for j, reference in enumerate(row):
            value = _rational(reference)
            for matrix, probability in zip(matrices[1:], probabilities[1:], strict=True):
                difference = arithmetic.add(matrix[i][j], -reference)
                if difference:
                    value = _add(value, _multiply(_rational(difference), probability))
            hull = Interval(_rational(min(matrix[i][j] for matrix in matrices)).lo,
                            _rational(max(matrix[i][j] for matrix in matrices)).hi)
            output.append(value.intersect(hull))
        result.append(tuple(output))
    return tuple(result)


def _between_group_covariance(offsets: Matrix, probabilities: Box, arithmetic: _Arithmetic) -> IntervalMatrix:
    """Enclose centroid covariance in O(groups*dimension²) bounded work.

    Offsets are exact centroid differences from the first group. Shared
    spatial gauges are removed before interval conversion. This full-space
    covariance is generally nonzero even when every projected offset is zero.
    """
    means = _mix(tuple((row,) for row in offsets), probabilities, arithmetic)[0]
    seconds = tuple(tuple(tuple(arithmetic.mul(a, b) for b in row) for a in row) for row in offsets)
    second = _mix(seconds, probabilities, arithmetic)
    covariance = [[_ZERO]*len(means) for _ in means]
    for i, left in enumerate(means):
        for j in range(i, len(means)):
            product = _square(left) if i == j else _multiply(left, means[j])
            value = _add(second[i][j], Interval(-product.hi, -product.lo))
            if i == j:
                value = Interval(max(0.0, value.lo), value.hi)
            covariance[i][j] = covariance[j][i] = value
    return tuple(tuple(row) for row in covariance)


@dataclass(frozen=True, init=False)
class GaussianGroupedCriticalFace:
    r"""Exact critical-face geometry and rigorous original Hessian enclosures.

    E stacks original equalities, positive-primal-multiplier rows and selected
    weak rows. ``projector`` P projects onto ker(E); ``left_inverse`` is
    L=(E E.T)^(-1)E. Group weights include each group's cardinality.
    ``hessian``, ``projected_hessian`` and ``multiplier_hessian`` enclose H,
    P H P and L H P at the bound original equilibrium. Exact within-group
    centered moments and transformations precede interval conversion. With
    unequal full group centroids, H additionally includes their covariance;
    that between-group term vanishes in P H P and L H P after an exact
    projection check. No response inverse is stored. The query needs a
    proposed response and verifies its residual.
    """

    equilibrium: Equilibrium
    active_weak_indices: tuple[int, ...]
    constraints: Matrix
    projector: Matrix
    left_inverse: Matrix
    posterior: Box
    group_probabilities: Box
    hessian: IntervalMatrix
    projected_hessian: IntervalMatrix
    multiplier_hessian: IntervalMatrix
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use compile_gaussian_grouped_critical_face")


def compile_gaussian_grouped_critical_face(
    equilibrium: Equilibrium, active_weak_indices: Sequence[int] = (),
) -> GaussianGroupedCriticalFace | None:
    r"""Compile exact geometry and interval curvature for one critical face.

    Select original weak inequality indices; E must have full row rank.
    Redundant unselected weak rows remain in subsequent feasibility checks.
    One exact row-Gram LDLT and reused solves construct L and P=I-E.T*L;
    exact geometry identities are replayed. Group posterior weights are
    enclosed using rigorous softmax, with per-mode rather than per-group
    normalization. A projected equilibrium's sealed weights are reused. A
    single logit group has exact uniform posterior weights.

    With each exact group mean c_g, delta_j=a_j-c_g and gamma=1/(2*nu*time²),
    H_g=I/time-gamma*mean_group(delta_j*delta_j.T). Transform delta_j by
    P and L before rounding to compute P H_g P and L H_g P. The normalized
    mixture H_0+sum_{g>0} pi_g*(H_g-H_0) preserves constant entries. When
    full group centroids differ, subtract gamma*Cov_pi(c_g) from the full
    H enclosure. Exact P*(c_g-c_0)=0 is replayed, proving that this extra
    covariance vanishes in both projected blocks. Strong/weak labels from
    the equilibrium's exact or rigorous sign proof are retained, including
    strict positivity whose interval lower endpoint underflows to zero.
    Dense interval matrices are stored; this is not matrix-free.

    Aggregate bounded work includes O(m*d²+d³+k*d) exact arithmetic, posterior
    evaluation and retained evidence; storage includes group moment matrices.
    Invalid types/indices or dependent selected rows raise. Damaged evidence,
    exhausted resources or unrepresentable finite bounds return None. Missing
    rigorous backend raises RuntimeError; prior fallback history is preserved.
    A sealed digest is not a formal theorem-prover verification.
    """
    if not isinstance(equilibrium, (GaussianGroupedEquilibrium, GaussianProjectedEquilibrium)):
        raise TypeError("equilibrium must be a GaussianGroupedEquilibrium or GaussianProjectedEquilibrium")
    if not isinstance(active_weak_indices, Sequence) or isinstance(active_weak_indices, (str, bytes)):
        raise TypeError("active_weak_indices must be a finite sequence of integers")
    if any(type(index) is not int for index in active_weak_indices):
        raise TypeError("active weak indices must be integers, excluding bool")
    if len(active_weak_indices) > len(equilibrium.weak_indices):
        raise ValueError("selected indices must be distinct weak inequalities")
    source = equilibrium.face.base_restriction.source_model
    dimension, count, inequalities = source.dimension, source.mode_count, len(equilibrium.face.inequalities)
    rows = len(equilibrium.face.base_restriction.constraints)+len(equilibrium.strong_indices)+len(active_weak_indices)
    if rows > dimension:
        raise ValueError("selected critical-face rows must have full row rank")
    arithmetic = _Arithmetic(source.budget)
    try:
        arithmetic._entries(512*(count+inequalities+1)*(dimension+1) + 128*(count+1)*(dimension+1)**2 + 4096)
        arithmetic._work(2048*(count+inequalities+dimension+1)*(dimension+1)**2 + 2048*(dimension+1)**3 + 8192)
        if isinstance(equilibrium, GaussianProjectedEquilibrium):
            # Includes retained group/multiplier evidence, centroid second
            # moments, their serialization and exact projection replay.
            arithmetic._entries(512*(count+inequalities+1)*(dimension+1)
                                + 256*(count+1)*(dimension+1)**2 + 8192)
            arithmetic._work(256*(count+1)*(dimension+1)**2 + 2048)
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
    try:
        rank = solve_affine(constraints, None, arithmetic).rank if constraints else 0
    except (ArithmeticError, ValueError):
        return None
    if rank != rows:
        raise ValueError("selected critical-face rows must have full row rank")
    try:
        if constraints:
            gram = tuple(tuple(_dot(a, b, arithmetic) for b in constraints) for a in constraints)
            factor = factor_positive_definite(gram, arithmetic)
            columns = tuple(factor.solve(tuple(row[j] for row in constraints), arithmetic) for j in range(dimension))
            left = tuple(tuple(column[i] for column in columns) for i in range(rows))
        else:
            left = ()
            columns = ((),)*dimension
        projector = tuple(tuple(arithmetic.add(Fraction(int(i == j)),
                                                -_dot(tuple(row[i] for row in constraints), columns[j], arithmetic))
                                for j in range(dimension)) for i in range(dimension))
        _check_geometry(constraints, (Fraction(0),)*rows, rows, (Fraction(0),)*dimension, projector, arithmetic)
        _check_identities(constraints, left, projector, arithmetic)
        groups = equilibrium.groups
        if isinstance(equilibrium, GaussianProjectedEquilibrium):
            if any(any(_linear(projector, row, arithmetic)) for row in equilibrium.group_centroid_offsets):
                return None
            posterior = equilibrium.posterior
            probabilities: Box = equilibrium.group_probabilities
            provenance = equilibrium.certificate["meta"]["transcend_backend"]
        elif len(groups) == 1:
            posterior = (_rational(arithmetic._coefficient(Fraction(1, count))),)*count
            probabilities = (Interval.point(1.0),)
            provenance = "not_used"
        else:
            logits = [Fraction(0)]*count
            maximum = max(equilibrium.group_logits)
            for group, logit in zip(groups, equilibrium.group_logits, strict=True):
                shifted = arithmetic.add(logit, -maximum)
                for index in group:
                    logits[index] = shifted
            with _strict():
                posterior = _probabilities(tuple(_rational(value) for value in logits))
            probabilities = tuple(_multiply(_rational(Fraction(len(group))), posterior[group[0]])
                                  .intersect(Interval(0.0, 1.0)) for group in groups)
            provenance = backend_name()
        centered = equilibrium.centered_centers
        tangent = tuple(_linear(projector, row, arithmetic) for row in centered)
        normal = tuple(_linear(left, row, arithmetic) for row in centered)
        inverse_time = arithmetic._coefficient(1/source.time)
        denominator = arithmetic.mul(arithmetic.mul(Fraction(2), source.viscosity), source.time)
        scale = arithmetic._coefficient(inverse_time/denominator)
        all_h, all_p, all_l = [], [], []
        for group in groups:
            covariance = _moment(centered, centered, group, arithmetic)
            tangent_covariance = _moment(tangent, tangent, group, arithmetic)
            cross_covariance = _moment(normal, tangent, group, arithmetic)
            all_h.append(tuple(tuple(arithmetic.add(inverse_time if i == j else Fraction(0),
                                                     -arithmetic.mul(scale, covariance[i][j])) for j in range(dimension))
                               for i in range(dimension)))
            all_p.append(tuple(tuple(arithmetic.add(arithmetic.mul(inverse_time, projector[i][j]),
                                                     -arithmetic.mul(scale, tangent_covariance[i][j])) for j in range(dimension))
                               for i in range(dimension)))
            all_l.append(tuple(tuple(-arithmetic.mul(scale, value) for value in row) for row in cross_covariance))
        hessian = _mix(tuple(all_h), probabilities, arithmetic)
        if isinstance(equilibrium, GaussianProjectedEquilibrium):
            between = _between_group_covariance(equilibrium.group_centroid_offsets, probabilities, arithmetic)
            hessian = tuple(tuple(_add(value, _multiply(_rational(-scale), correction))
                                  for value, correction in zip(row, between_row, strict=True))
                            for row, between_row in zip(hessian, between, strict=True))
        projected_hessian = _mix(tuple(all_p), probabilities, arithmetic)
        multiplier_hessian = _mix(tuple(all_l), probabilities, arithmetic)
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Exact critical-face geometry and rigorous nonuniform Gaussian Hessian enclosures.",
            payload={
                "type": "gaussian_grouped_critical_face", "equilibrium_certificate": deepcopy(equilibrium.certificate),
                "active_weak_indices": list(active), "constraints": _matrix_payload(constraints),
                "projector": _matrix_payload(projector), "left_inverse": _matrix_payload(left),
                "posterior": _box_payload(posterior), "group_probabilities": _box_payload(probabilities),
                "hessian": [_box_payload(row) for row in hessian],
                "projected_hessian": [_box_payload(row) for row in projected_hessian],
                "multiplier_hessian": [_box_payload(row) for row in multiplier_hessian],
                "identities": ["E*P=0", "P*P=P", "P.T=P", "L*E.T=I", "E.T*L=I-P"],
                "argument": "Exact independent-row Gram solves give L and P and replay their identities. The bound equilibrium proves exact group centroids and a common projected centroid on this critical face; unequal centroid offsets are projected to zero again before curvature evaluation. This follows because group force coefficients vanish identically in every weak primal row, leaving group forces in the span of original equality and strong-active normals. Per-mode rigorous softmax or the projected equilibrium's sealed weights enclose each original probability and each cardinality-weighted group mass. Covariance decomposes into weighted within-group covariance plus centroid covariance. The full H retains both terms. The latter vanishes after right multiplication by P, so exact within-group centered moments, transformed by P and L before rounding, suffice for P*H*P and L*H*P. Normalized first-group-plus-differences mixtures preserve constant entries and remove common spatial offsets.",
                "scope": "One proposed critical face at one exact original equilibrium. Dense curvature enclosures, no inverse response, direction-region maximality, active-set search or derivative claim until a query succeeds.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": provenance},
        )
        result = object.__new__(GaussianGroupedCriticalFace)
        for name, stored in (("equilibrium", equilibrium), ("active_weak_indices", active), ("constraints", constraints),
                             ("projector", projector), ("left_inverse", left), ("posterior", posterior),
                             ("group_probabilities", probabilities), ("hessian", hessian),
                             ("projected_hessian", projected_hessian), ("multiplier_hessian", multiplier_hessian),
                             ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


def _center_box(value: Vector, radii: Vector, arithmetic: _Arithmetic) -> Box:
    return tuple(Interval(_rational(arithmetic.add(center, -radius)).lo,
                          _rational(arithmetic.add(center, radius)).hi)
                 for center, radius in zip(value, radii, strict=True))


def _exact_minus_box(value: Vector, box: Box, arithmetic: _Arithmetic) -> Box:
    return tuple(Interval(_rational(arithmetic.add(exact, -Fraction(interval.hi))).lo,
                          _rational(arithmetic.add(exact, -Fraction(interval.lo))).hi)
                 for exact, interval in zip(value, box, strict=True))


def _exact_plus_box(value: Vector, box: Box, arithmetic: _Arithmetic) -> Box:
    return _exact_minus_box(value, tuple(Interval(-component.hi, -component.lo) for component in box), arithmetic)


@dataclass(frozen=True)
class _CriticalResponse:
    value: Vector
    enclosure: Box
    error_bound: Fraction
    coordinate_error_bounds: Vector
    residual_enclosure: Box
    residual_norm_bound: Fraction
    equality_multipliers: Box
    weak_multipliers: Box
    weak_slacks: Box


def _critical_response(
    face: GaussianGroupedCriticalFace, projected: Vector, forcing: Vector, normal_forcing: Vector,
    arithmetic: _Arithmetic, *, forcing_remainder: Box, normal_remainder: Box,
) -> _CriticalResponse | None:
    """Shared residual/sign proof for exactly referenced enclosed forcing.

    The caller validates evidence and inputs and reserves aggregate resources.
    ``forcing + forcing_remainder`` must enclose P*zeta, while
    ``normal_forcing + normal_remainder`` encloses L*zeta for the SAME true
    QP forcing zeta. Reference scalars stay rational until endpoint conversion.
    Exact zero references and singleton-zero remainders prove zero forcing;
    an interval merely containing zero never triggers that shortcut.
    """
    equilibrium = face.equilibrium
    source = equilibrium.face.base_restriction.source_model
    dimension, inequalities = source.dimension, len(equilibrium.face.inequalities)
    if not any(forcing) and all(_zero(component) for component in forcing_remainder):
        value = (Fraction(0),)*dimension
        residual = (_ZERO,)*dimension
        residual_norm = error = Fraction(0)
        radii = value
        enclosure = residual
        multipliers = _exact_plus_box(normal_forcing, normal_remainder, arithmetic)
    else:
        value = projected
        action = _interval_action(face.projected_hessian, tuple(_rational(component) for component in value))
        # Negating the residual does not alter its norm; orient the stored
        # enclosure as P*H*P*value-P*zeta for the declared QP forcing.
        negative_residual = _exact_minus_box(forcing, action, arithmetic)
        negative_residual = tuple(_add(a, b) for a, b in zip(negative_residual, forcing_remainder, strict=True))
        residual = tuple(Interval(-component.hi, -component.lo) for component in negative_residual)
        magnitudes = tuple(arithmetic._coefficient(max(abs(Fraction(component.lo)), abs(Fraction(component.hi)))) for component in residual)
        residual_norm = _norm_upper(magnitudes, arithmetic)
        error = arithmetic._coefficient(residual_norm/equilibrium.strong_convexity_bound)
        radii = tuple(arithmetic.mul(error, arithmetic._coefficient(Fraction(_sqrt_upper(face.projector[i][i], source.budget))))
                      for i in range(dimension))
        enclosure = _center_box(value, radii, arithmetic)
        multipliers = _exact_minus_box(normal_forcing, _interval_action(face.multiplier_hessian, enclosure), arithmetic)
        multipliers = tuple(_add(a, b) for a, b in zip(multipliers, normal_remainder, strict=True))
    weak_slacks = [_ZERO]*inequalities
    weak_multipliers = [_ZERO]*inequalities
    active = set(face.active_weak_indices)
    for index in equilibrium.weak_indices:
        if index in active:
            continue
        row = equilibrium.face.inequalities[index]
        nominal = -_dot(row, value, arithmetic)
        radius = arithmetic.mul(_norm_upper(_linear(face.projector, row, arithmetic), arithmetic), error)
        weak_slacks[index] = _center_box((nominal,), (radius,), arithmetic)[0]
    offset = len(equilibrium.face.base_restriction.constraints)+len(equilibrium.strong_indices)
    for index, multiplier in zip(face.active_weak_indices, multipliers[offset:], strict=True):
        weak_multipliers[index] = multiplier
    if any(component.lo < 0 for component in weak_slacks) or any(component.lo < 0 for component in weak_multipliers):
        return None
    return _CriticalResponse(value, enclosure, error, radii, residual, residual_norm,
                             multipliers, tuple(weak_multipliers), tuple(weak_slacks))


@dataclass(frozen=True, init=False)
class GaussianGroupedDirectional:
    r"""Enclose a one-sided Hadamard target derivative and its proposal error.

    ``value`` is normally exact P*proposal, with Euclidean error at most
    ``error_bound`` from the true derivative. ``enclosure`` uses the sharper
    coordinate radii sqrt(P_ii)*error_bound. If P*direction=0, a successful
    sign check proves the response is exactly zero regardless of proposal;
    ``projected_proposal`` still records that projection.

    Multipliers follow E row order including selected weak rows. Weak arrays
    follow every original inequality index, with structural zeros outside
    weak rows. Only weak-row entries of these arrays have QP meaning.
    """

    face: GaussianGroupedCriticalFace
    direction: Vector
    proposal: tuple[float, ...]
    projected_proposal: Vector
    value: Vector
    enclosure: Box
    error_bound: Fraction
    coordinate_error_bounds: Vector
    residual_enclosure: Box
    residual_norm_bound: Fraction
    equality_multipliers: Box
    weak_multipliers: Box
    weak_slacks: Box
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_grouped_directional")


def certify_gaussian_grouped_directional(
    face: GaussianGroupedCriticalFace, direction: Sequence[int | Fraction], proposal: Sequence[float],
) -> GaussianGroupedDirectional | None:
    r"""Certify a switching derivative by residual bounds and weak-row signs.

    Only the target varies; the original source, age, viscosity and geometry
    remain fixed. The exact equilibrium determines the critical cone from all
    tight rows and one valid primal multiplier representation. Its unique QP
    minimizer is the one-sided Hadamard derivative. For selected face E, let
    w_hat=P*proposal and r=P*H*P*w_hat-P*direction. The equality-QP solution
    w_E satisfies ||w_E-w_hat||<=||r||/m, with m=(1-q_base)/time>0.

    Enclose lambda=L*direction-L*H*P*w_E and unselected weak slacks using
    -G_i*w_hat +/- ||P*G_i||*error. Selected weak slacks vanish identically.
    Nonnegative selected weak multipliers and all weak slacks prove w_E is
    the critical-cone minimizer. Strong-row derivative multipliers are free.
    P*direction=0 is an exact zero-response shortcut, still subject to signs.

    Query work uses dense matrix actions and bounded exact arithmetic, with
    no solve, factorization, iteration, exponentials or active-set search.
    Invalid types/shapes raise even on shortcuts. Wrong face, inconclusive
    signs, damaged evidence, exhausted resources or nonfinite bounds return
    None; refusal says nothing about derivative existence. No ordinary
    Jacobian/VJP, finite-step remainder, switching radius or moving-geometry
    derivative is asserted. A digest does not establish a formal proof.
    """
    if not isinstance(face, GaussianGroupedCriticalFace):
        raise TypeError("face must be a GaussianGroupedCriticalFace")
    equilibrium = face.equilibrium
    source = equilibrium.face.base_restriction.source_model
    dimension, count, inequalities = source.dimension, source.mode_count, len(equilibrium.face.inequalities)
    _validate_vector(direction, dimension, "direction")
    binary_proposal = _binary_vector(proposal, dimension, "proposal")
    try:
        arithmetic = _Arithmetic(source.budget)
        arithmetic._entries(1024*(count+inequalities+1)*(dimension+1) + 256*(count+1)*(dimension+1)**2 + 8192)
        arithmetic._work(4096*(count+inequalities+dimension+1)*(dimension+1)**2 + 16384)
        if isinstance(equilibrium, GaussianProjectedEquilibrium):
            arithmetic._entries(1024*(count+inequalities+1)*(dimension+1)
                                + 512*(count+1)*(dimension+1)**2 + 16384)
            arithmetic._work(256*(count+inequalities+1)*(dimension+1)**2 + 4096)
        if not _intact(face.certificate) or not _equilibrium_intact(equilibrium):
            return None
        frozen = tuple(arithmetic.exact(component) for component in direction)
        projected = _linear(face.projector, tuple(arithmetic._coefficient(Fraction(component)) for component in binary_proposal), arithmetic)
        forcing = _linear(face.projector, frozen, arithmetic)
        normal_forcing = _linear(face.left_inverse, frozen, arithmetic)
        response = _critical_response(face, projected, forcing, normal_forcing, arithmetic,
                                      forcing_remainder=(_ZERO,)*dimension,
                                      normal_remainder=(_ZERO,)*len(normal_forcing))
        if response is None:
            return None
        value, enclosure = response.value, response.enclosure
        error, radii = response.error_bound, response.coordinate_error_bounds
        residual, residual_norm = response.residual_enclosure, response.residual_norm_bound
        multipliers = response.equality_multipliers
        weak_multipliers, weak_slacks = response.weak_multipliers, response.weak_slacks
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Rigorous enclosure and proposal error for a nonuniform Gaussian switching derivative.",
            payload={
                "type": "gaussian_grouped_directional", "critical_face_certificate": deepcopy(face.certificate),
                "direction": [_pair(component) for component in frozen], "proposal": list(binary_proposal),
                "projected_proposal": [_pair(component) for component in projected], "value": [_pair(component) for component in value],
                "enclosure": _box_payload(enclosure), "error_bound": _pair(error),
                "coordinate_error_bounds": [_pair(component) for component in radii],
                "residual_enclosure": _box_payload(residual), "residual_norm_bound": _pair(residual_norm),
                "equality_multipliers": _box_payload(multipliers), "weak_multipliers": _box_payload(tuple(weak_multipliers)),
                "weak_slacks": _box_payload(tuple(weak_slacks)),
                "argument": "Exact equilibrium and q_base<1 give the unique global decision and its critical cone: original equalities and positive-multiplier tight rows are equalities; all zero-multiplier tight rows remain inequalities. Strong monotonicity bounds target difference quotients. Second-order variational expansion forces each limit to minimize .5*w.T*H*w-direction.T*w on that cone; tangent strong convexity makes the minimizer unique and yields the Hadamard directional limit. On the proposed equality face, tangent coercivity m bounds proposal error by the enclosed projected residual norm divided by m. P_ii and projected weak normals sharpen coordinate and slack errors. Enclosed selected weak multipliers and all unselected weak slacks are nonnegative, so exact equality-QP stationarity and structural complementarity prove its solution is the critical-cone minimizer. If P*direction=0, zero solves the equality QP exactly, but its weak multiplier signs are still checked.",
                "scope": "Fixed original source, age, viscosity and all constraint data; only the target varies. Enclosure of a one-sided Hadamard derivative, not a finite-step approximation error, ordinary Jacobian/VJP, higher derivative, switching radius or derivative under moving geometry. Only weak-row entries of weak arrays have QP meaning. No query solve or transcendental evaluation.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False},
            meta={"transcend_backend": face.certificate["meta"]["transcend_backend"]},
        )
        result = object.__new__(GaussianGroupedDirectional)
        for name, stored in (("face", face), ("direction", frozen), ("proposal", binary_proposal),
                             ("projected_proposal", projected), ("value", value), ("enclosure", enclosure),
                             ("error_bound", error), ("coordinate_error_bounds", radii), ("residual_enclosure", residual),
                             ("residual_norm_bound", residual_norm), ("equality_multipliers", multipliers),
                             ("weak_multipliers", tuple(weak_multipliers)), ("weak_slacks", tuple(weak_slacks)),
                             ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianGroupedCriticalFace", "GaussianGroupedDirectional", "certify_gaussian_grouped_directional",
           "compile_gaussian_grouped_critical_face"]
