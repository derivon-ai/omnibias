# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stable polyhedral projection faces for certified Gaussian global minima."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified._gaussian_moments import _posterior_moments
from omnibias.core.verified.gaussian_affine import _dot
from omnibias.core.verified.gaussian_entropy_residual import (
    GaussianEntropyResidual,
    _residual_intact,
    _residual_source_slots,
)
from omnibias.core.verified.gaussian_heat import _rational
from omnibias.core.verified.gaussian_projection_minimum import _combined_budget, _ProjectionOracle
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact
from omnibias.core.verified.inverse_gaussian_heat import _pair
from omnibias.core.verified.mean_map import _probabilities, _strict
from omnibias.core.verified.polyhedral_projection import _project_exact, _query_work_bound
from omnibias.core.verified.transcend import libm_fallback_used

if TYPE_CHECKING:
    from omnibias.core.proof.certificate import Cert

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]
Box = tuple[Interval, ...]
IntervalMatrix = tuple[Box, ...]
Pair = tuple[Fraction, Fraction]
Bounds = tuple[Pair, ...]
_ZERO = Interval.point(0.0)


def _source_slots(residual: GaussianEntropyResidual) -> int:
    model, projector = residual.unique.domain.model, residual.unique.domain.projector
    return (_residual_source_slots(residual.unique)+256*(model.mode_count+len(projector.inequalities)+1)
            *(model.dimension+1)+128*(model.dimension+1)**2+4096)


def _face_source_slots(face: GaussianEntropyFace) -> int:
    """Additive reservation for the retained face and its proof dependencies."""
    return _source_slots(face.residual)


def _face_intact(face: GaussianEntropyFace) -> bool:
    """Replay digests and public-data bindings after reserving source scans."""
    from omnibias.core.proof.certificate import encode_interval

    if not _intact(face.certificate) or not _residual_intact(face.residual):
        return False
    payload = face.certificate.get("payload")
    if not isinstance(payload, dict) or payload.get("residual_certificate") != face.residual.certificate:
        return False
    for name in ("active_indices", "basis_indices"):
        if payload.get(name) != list(getattr(face, name)):
            return False
    for name in ("normal_rows", "gram_inverse", "tangent_projector"):
        if payload.get(name) != [[_pair(value) for value in row] for row in getattr(face, name)]:
            return False
    for name in ("optimizer_bounds", "projection_input_bounds"):
        if payload.get(name) != [[_pair(value) for value in pair] for pair in getattr(face, name)]:
            return False
    for name in ("posterior", "projection_multipliers", "slacks"):
        if payload.get(name) != [encode_interval(value) for value in getattr(face, name)]:
            return False
    return (payload.get("covariance") == [[encode_interval(value) for value in row] for row in face.covariance]
            and payload.get("witness_epsilon") == _pair(face.witness_epsilon)
            and payload.get("witness_multipliers") == [_pair(value) for value in face.witness_multipliers]
            and payload.get("strong_convexity_bound") == _pair(face.strong_convexity_bound))


def _linear_bounds(row: Vector, bounds: Bounds, arithmetic: _Arithmetic) -> Pair:
    lower = upper = Fraction(0)
    for coefficient, pair in zip(row, bounds, strict=True):
        if coefficient:
            products = tuple(arithmetic.mul(coefficient, endpoint) for endpoint in pair)
            lower, upper = arithmetic.add(lower, min(products)), arithmetic.add(upper, max(products))
    return lower, upper


def _interval(bounds: Pair) -> Interval:
    return Interval(_rational(bounds[0]).lo, _rational(bounds[1]).hi)


@dataclass(frozen=True, init=False)
class GaussianEntropyFace:
    r"""A stable projection face at the unique original Gaussian optimizer.

    ``active_indices`` includes ALL original rows tight at the source sample,
    including redundant and zero rows. ``basis_indices`` selects a maximal
    independent subset with rows E. The exact orthogonal tangent projector is
    R=I-E^T*(E*E^T)^(-1)*E. Existing projection factors are reused.

    Positive ``projection_multipliers`` on every active row and strictly
    positive inactive ``slacks`` certify that the true optimizer has this
    face and that Euclidean projection is locally affine with derivative R
    at its posterior projection input. Opposite rows encoding equalities are
    supported. No particular multiplier representation is asserted unique.

    ``optimizer_bounds`` are exact physical bounds formed before numerical
    centering. ``posterior`` and ``covariance`` enclose the original family
    throughout those bounds. Local stability is qualitative; no numerical
    parameter perturbation radius or arbitrary constraint derivative is given.
    """

    residual: GaussianEntropyResidual
    active_indices: tuple[int, ...]
    basis_indices: tuple[int, ...]
    normal_rows: Matrix
    gram_inverse: Matrix
    tangent_projector: Matrix
    optimizer_bounds: Bounds
    posterior: Box
    covariance: IntervalMatrix
    projection_multipliers: Box
    slacks: Box
    strong_convexity_bound: Fraction
    witness_epsilon: Fraction
    witness_multipliers: Vector
    projection_input_bounds: Bounds
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_entropy_face")


def certify_gaussian_entropy_face(
    residual: GaussianEntropyResidual, *, max_trials: int = 32,
) -> GaussianEntropyFace | None:
    r"""Certify a stable projection face, including redundant/equality rows.

    At the exact source point x, let J contain every tight row. Trial epsilon
    values 1,1/2,... test P_C(q0-epsilon*sum(G_j,j in J))=x. A successful
    projection supplies a strictly positive full-J normal representation at
    q0. Exact affine interval bounds then require that its multiplier and
    inactive-slack margins persist at the true posterior projection input.

    All requested trials and moment work are preflighted against both source
    budgets before queries. Invalid argument types/domains raise. An unstable
    or unresolved face, damaged dependency, exhausted budget or nonfinite
    arithmetic returns None. Missing rigorous backend or prior conditional
    libm history raises RuntimeError and never clears that history. No new
    factorization, active-set enumeration, optimizer search or differentiating
    a projection routine is performed.
    """
    if not isinstance(residual, GaussianEntropyResidual):
        raise TypeError("residual must be a GaussianEntropyResidual")
    if type(max_trials) is not int:
        raise TypeError("max_trials must be an integer excluding bool")
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    domain = residual.unique.domain
    model, projector = domain.model, domain.projector
    dimension, count = model.dimension, len(projector.inequalities)
    arithmetic = _Arithmetic(_combined_budget(model.budget, projector.budget))
    a = arithmetic
    try:
        slots = _source_slots(residual)
        a._entries(slots)
        work = (4*slots+max_trials*_query_work_bound(projector)
                +1024*(model.mode_count+count+1)*(dimension+1)**2+128*(dimension+1)**3)
        a.limits._work(work)
        a._work(4*slots)
        a.exact(max_trials)
        if not _residual_intact(residual):
            return None
        from omnibias.core.proof.certificate import (
            certificate_is_unconditional,
            encode_interval,
            make_certificate,
        )

        if not certificate_is_unconditional(residual.certificate):
            return None
        if libm_fallback_used():
            raise RuntimeError("Gaussian entropy face refuses prior conditional libm fallback history")
        with _strict():
            a._work(256*(model.mode_count+count+1)*(dimension+1)**2)
            initial_slacks = tuple(a.add(bound, -_dot(row, residual.point, a))
                                   for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
            active = tuple(index for index, value in enumerate(initial_slacks) if value == 0)
            active_set = set(active)
            basis = max((item for item in projector.bases if all(index in active_set for index in item.active_indices)),
                        key=lambda item: len(item.active_indices))
            basis_indices = basis.active_indices
            rows = tuple(projector.inequalities[index] for index in basis_indices)
            rank = len(rows)
            inverse: Matrix = ()
            if basis.factor is not None:
                columns = tuple(basis.factor.solve(tuple(Fraction(int(i == j)) for i in range(rank)), a)
                                for j in range(rank))
                inverse = tuple(tuple(column[i] for column in columns) for i in range(rank))
            row_columns = tuple(tuple(row[j] for row in rows) for j in range(dimension))
            left = tuple(tuple(_dot(row, column, a) for column in row_columns) for row in inverse)
            left_columns = tuple(tuple(row[j] for row in left) for j in range(dimension))
            tangent = tuple(tuple(a.add(Fraction(int(i == j)), -_dot(row_columns[i], left_columns[j], a))
                                  for j in range(dimension)) for i in range(dimension))
            # Replay exact annihilation for every tight original row, not just
            # the independent basis. Structural zero slacks never round away.
            tangent_columns = tuple(tuple(row[j] for row in tangent) for j in range(dimension))
            if any(_dot(projector.inequalities[index], column, a) for index in active for column in tangent_columns):
                return None
            radius = a.exact(Fraction(residual.error_bound))
            optimizer_bounds = tuple((max(a.add(x, -radius), limits[0]), min(a.add(x, radius), limits[1]))
                                     for x, limits in zip(residual.point, domain.physical_bounds, strict=True))
            projection = _ProjectionOracle(model, projector, domain.target, a)
            centered = tuple((a.add(lower, -origin), a.add(upper, -origin))
                             for (lower, upper), origin in zip(optimizer_bounds, projection.origin, strict=True))
            logits = tuple(tuple(a.add(offset, endpoint) for endpoint in _linear_bounds(row, centered, a))
                           for row, offset in zip(projection.rates, projection.offsets, strict=True))
            shift = max(pair[1] for pair in logits)
            converted_logits = tuple(_interval((a.add(lower, -shift), a.add(upper, -shift))) for lower, upper in logits)
            posterior = _probabilities(converted_logits)
            moments = _posterior_moments(posterior, tuple(tuple(_rational(value) for value in row) for row in projection.relative))
            exact_mean = tuple(a.sum(tuple(a.mul(weight, row[j])
                                          for weight, row in zip(residual.probabilities, projection.relative, strict=True)))
                               for j in range(dimension))
            delta = tuple((a.add(a.exact(Fraction(bound.lo)), -mean), a.add(a.exact(Fraction(bound.hi)), -mean))
                          for bound, mean in zip(moments.relative_mean, exact_mean, strict=True))
            input_bounds = tuple((a.add(source, lower), a.add(source, upper))
                                 for source, (lower, upper) in zip(residual.projection_source, delta, strict=True))
            corrections = tuple(_linear_bounds(row, delta, a) for row in left)
            slacks = []
            for index, (row, initial) in enumerate(zip(projector.inequalities, initial_slacks, strict=True)):
                if index in active_set:
                    slacks.append(_ZERO)
                    continue
                projected_row = tuple(_dot(row, column, a) for column in tangent_columns)
                change = _linear_bounds(projected_row, delta, a)
                bound = _interval((a.add(initial, -change[1]), a.add(initial, -change[0])))
                if bound.lo <= 0:
                    return None
                slacks.append(bound)
            witness: Vector = (Fraction(0),)*count
            multiplier_bounds: Box = (_ZERO,)*count
            epsilon = Fraction(0)
            if active:
                row_sum = tuple(a.sum(tuple(projector.inequalities[index][j] for index in active)) for j in range(dimension))
                epsilon = Fraction(1)
                found = False
                for _ in range(max_trials):
                    trial = tuple(a.add(source, -a.mul(epsilon, normal))
                                  for source, normal in zip(residual.projection_source, row_sum, strict=True))
                    solved = _project_exact(projector, trial, a)
                    if solved is not None and solved[0] == residual.point:
                        lifted = [epsilon if index in active_set else Fraction(0) for index in range(count)]
                        for index, value in zip(solved[1], solved[2], strict=True):
                            lifted[index] = a.add(lifted[index], value)
                        exact_bounds = [(value, value) for value in lifted]
                        for index, (lower, upper) in zip(basis_indices, corrections, strict=True):
                            exact_bounds[index] = a.add(lifted[index], lower), a.add(lifted[index], upper)
                        bounds = tuple(_interval(pair) for pair in exact_bounds)
                        if all(bounds[index].lo > 0 for index in active):
                            witness, multiplier_bounds, found = tuple(lifted), bounds, True
                            break
                    epsilon = a._coefficient(epsilon/2)
                if not found:
                    return None
            certificate = make_certificate(
                claim="The unique constrained Gaussian global minimizer lies on the recorded stable polyhedral projection face.",
                payload={
                    "type": "gaussian_entropy_face", "residual_certificate": deepcopy(residual.certificate),
                    "active_indices": list(active), "basis_indices": list(basis_indices),
                    "normal_rows": [[_pair(value) for value in row] for row in rows],
                    "gram_inverse": [[_pair(value) for value in row] for row in inverse],
                    "tangent_projector": [[_pair(value) for value in row] for row in tangent],
                    "optimizer_bounds": [[_pair(value) for value in pair] for pair in optimizer_bounds],
                    "projection_input_bounds": [[_pair(value) for value in pair] for pair in input_bounds],
                    "posterior": [encode_interval(value) for value in posterior],
                    "covariance": [[encode_interval(value) for value in row] for row in moments.covariance],
                    "projection_multipliers": [encode_interval(value) for value in multiplier_bounds],
                    "slacks": [encode_interval(value) for value in slacks],
                    "strong_convexity_bound": _pair(residual.strong_convexity_bound),
                    "witness_epsilon": _pair(epsilon), "witness_multipliers": [_pair(value) for value in witness],
                    "argument": "The source residual certificate encloses the unique global optimizer x*. Reconstruct its exact coordinate bounds from the exact sample plus/minus its outward radius, intersected with the source physical box, before subtracting the first center. Exact affine logit bounds and rigorous shared posterior moments enclose pi(x*) and Cov(a)(x*). Let q0=s*v+A*p and x=P_C(q0) be the source exact projection, J all original rows tight at x, E a maximal independent stored basis within J, and R=I-E^T*(E*E^T)^(-1)*E. Stored Gram factors construct the inverse without refactorization, and G_J*R=0 is replayed exactly. Projecting q0-epsilon*sum(G_j,j in J) back to x gives nonnegative projection multipliers. Adding epsilon to every J coordinate supplies a strictly positive full-J representation lambda0 of q0-x, also for redundant, opposite or zero rows. For the true input q*=s*v+A*pi(x*), delta=q*-q0 is enclosed by centered posterior means minus the exact proposal mean. Set xhat=x+R*delta and eta=(E*E^T)^(-1)*E*delta. Then q*-xhat=G^T*lambda with lambda=lambda0 plus eta on the E rows; all J slacks are structural zero. The strict tested multiplier and inactive-slack margins prove all original projection KKT conditions, so xhat=P_C(q*)=x*. Since these affine inequalities have strict margins, the same formula holds on a neighborhood of q*, and the Euclidean projection derivative there is R. Empty J has R=I and needs no witness trial. This certifies a stable geometric face without assigning uniqueness to its multiplier representation.",
                    "scope": "Fixed original model, target and polyhedron. Supports redundant/equality/zero rows and lower-dimensional feasible sets. Refusal is inconclusive, including weak one-sided switches and insufficient source accuracy. No numerical perturbation radius, arbitrary constraint derivative, new optimizer search or external formal theorem-prover verification claim.",
                }, honesty={"unproven_claim": False, "unconditional_transcendentals": True, "continuum_pde_claim": False},
                meta={"transcend_backend": "mpmath"},
            )
        result = object.__new__(GaussianEntropyFace)
        for name, stored_value in (("residual", residual), ("active_indices", active), ("basis_indices", basis_indices),
                            ("normal_rows", rows), ("gram_inverse", inverse), ("tangent_projector", tangent),
                            ("optimizer_bounds", optimizer_bounds), ("posterior", posterior), ("covariance", moments.covariance),
                            ("projection_multipliers", multiplier_bounds), ("slacks", tuple(slacks)),
                            ("strong_convexity_bound", residual.strong_convexity_bound), ("witness_epsilon", epsilon),
                            ("witness_multipliers", witness), ("projection_input_bounds", input_bounds), ("certificate", certificate)):
            object.__setattr__(result, name, stored_value)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianEntropyFace", "certify_gaussian_entropy_face"]
