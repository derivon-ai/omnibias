# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Route-2 solver: polygonal barriers to exact coefficient feasibility.

Assembles ``M alpha >= 1`` from declared annuli and searches signed coefficient
vectors.  The floating LP is only a proposer: acceptance always rescales and
replays over exact ``Q``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, isfinite
from typing import Literal

import torch
from omnibias.convex.torch import InfeasibleProblemError, solve_lp
from omnibias.core.realization.polynomial import Rational, SparsePolynomial
from omnibias.geometry.algebraic import (
    BarrierAnnulus,
    HomogeneousPlaneCurve,
    PolygonalAnnulus,
    ProjectiveCurveCertificate,
    ProjectiveCurveFormalVerification,
    RationalPolygon,
    _nesting,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
    verify_curve_certificate_formally,
)
from omnibias.geometry.algebraic_layout import (
    LayoutParams,
    parents_to_rooted_tree,
    square_polygon_xy,
    validate_annulus_layout,
)
from omnibias.geometry.algebraic_layout import (
    layout_annuli_for_tree as _tree_layout_annuli,
)
from omnibias.geometry.patchwork import RootedTree
from omnibias.geometry.patchwork_height_lp import (
    RationalFarkasCertificate,
    RationalFeasibilityCertificate,
    RationalInequalitySystem,
    certify_rational_feasibility,
    certify_rational_infeasibility,
    propose_farkas_infeasibility,
    verify_rational_feasibility,
    verify_rational_infeasibility,
)
from omnibias.sos.certify import DEFAULT_DENOMINATORS

OcticSymmetry = Literal["d4", "klein", "central", "full"]

__all__ = [
    "LayoutParams",
    "OcticSymmetry",
    "Route2LpReport",
    "Route2SearchReport",
    "assemble_barrier_system",
    "layout_annuli_for_tree",
    "nested_annulus_chain",
    "octic_basis_orbits",
    "octic_monomial_indices",
    "parents_to_rooted_tree",
    "certify_route2_realization",
    "seal_route2_witness",
    "search_route2_coefficients",
    "search_route2_full_coefficients",
    "search_route2_lp",
    "signs_by_nesting_depth",
    "square_polygon",
    "validate_annulus_layout",
]


def octic_monomial_indices() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (i, j, 8 - i - j)
        for i in range(9)
        for j in range(9 - i)
    )


def octic_basis_orbits(
    symmetry: OcticSymmetry = "full",
) -> tuple[tuple[int, ...], ...]:
    """Return coefficient-basis orbits inside the 45-dimensional octic space."""
    indices = octic_monomial_indices()
    positions = {index: position for position, index in enumerate(indices)}
    if symmetry == "full":
        return tuple((position,) for position in range(len(indices)))
    if symmetry == "central":
        return tuple(
            (position,)
            for position, (i, j, _k) in enumerate(indices)
            if (i + j) % 2 == 0
        )
    klein = tuple(
        position
        for position, (i, j, _k) in enumerate(indices)
        if i % 2 == 0 and j % 2 == 0
    )
    if symmetry == "klein":
        return tuple((position,) for position in klein)
    if symmetry != "d4":
        raise ValueError(f"unknown octic symmetry {symmetry!r}")
    seen: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for position in klein:
        if position in seen:
            continue
        i, j, k = indices[position]
        swapped = positions[(j, i, k)]
        orbit = tuple(sorted({position, swapped}))
        seen.update(orbit)
        orbits.append(orbit)
    return tuple(orbits)


def signs_by_nesting_depth(
    annuli: Sequence[BarrierAnnulus],
    *,
    exterior_sign: int = 1,
) -> tuple[int, ...]:
    """Assign each outer boundary the sign of its parent complement region."""
    if exterior_sign not in (-1, 1):
        raise ValueError("exterior_sign must be -1 or 1")
    barriers = tuple(annuli)
    parents = _nesting(barriers)
    signs: list[int] = []
    for index in range(len(parents)):
        depth = 0
        parent = parents[index]
        seen: set[int] = set()
        while parent >= 0:
            if parent in seen:
                raise ValueError("annulus parent relation contains a cycle")
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        signs.append(exterior_sign if depth % 2 == 0 else -exterior_sign)
    return tuple(signs)


def layout_annuli_for_tree(
    tree: RootedTree,
    *,
    params: LayoutParams | None = None,
    base_half: Fraction = Fraction(1, 16),
    growth: Fraction = Fraction(5, 4),
) -> tuple[PolygonalAnnulus, ...]:
    """Generate a tree-directed annulus layout validated against ``tree``."""
    return _tree_layout_annuli(
        tree,
        params=params,
        base_half=base_half,
        growth=growth,
    )


def nested_annulus_chain(
    count: int,
    *,
    base_half: Fraction = Fraction(1, 8),
    growth: Fraction = Fraction(3, 2),
) -> tuple[PolygonalAnnulus, ...]:
    """Build a correctly nested chain of square annuli for Route-2 searches."""
    if count < 1:
        raise ValueError("count must be positive")
    annuli: list[PolygonalAnnulus] = []
    half = base_half
    for index in range(count):
        offset = index * 2
        inner = square_polygon(offset, offset, half)
        outer = square_polygon(offset, offset, half * growth)
        annuli.append(PolygonalAnnulus(inner, outer))
        half *= growth
    return tuple(annuli)


def square_polygon(cx: int, cy: int, half: Fraction) -> RationalPolygon:
    return square_polygon_xy(Fraction(cx), Fraction(cy), half)


def _monomial_t_coefficients(
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
    exponent: tuple[int, int],
    *,
    poly_degree: int,
) -> tuple[Fraction, ...]:
    i, j = exponent
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    coeffs = [Fraction(0)] * (poly_degree + 1)
    for ix in range(i + 1):
        for jx in range(j + 1):
            t_power = ix + jx
            coeffs[t_power] += (
                Fraction(comb(i, ix))
                * Fraction(comb(j, jx))
                * start[0] ** (i - ix)
                * dx**ix
                * start[1] ** (j - jx)
                * dy**jx
            )
    return tuple(coeffs)


def _segment_polynomial_rows(
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
    sign: int,
    degree: int,
) -> list[tuple[tuple[Fraction, ...], Fraction]]:
    """Bernstein sign rows ``sign * beta_j(alpha) >= 1`` on one affine segment."""
    indices = octic_monomial_indices()
    nvars = len(indices)
    rows: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for j in range(degree + 1):
        row = [Fraction(0)] * nvars
        for mon_index, (i, jy, _k) in enumerate(indices):
            t_coeffs = _monomial_t_coefficients(start, end, (i, jy), poly_degree=degree)
            beta = sum(
                t_coeffs[t] * Fraction(comb(j, t), comb(degree, t))
                for t in range(j + 1)
            )
            if beta != 0:
                row[mon_index] += sign * beta
        if any(entry != 0 for entry in row):
            rows.append((tuple(row), Fraction(1)))
    return rows


def _dyadic_segments(
    start: tuple[Fraction, Fraction],
    end: tuple[Fraction, Fraction],
    depth: int,
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]:
    segments = ((start, end),)
    for _ in range(depth):
        refined = []
        for left, right in segments:
            midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            refined.extend(((left, midpoint), (midpoint, right)))
        segments = tuple(refined)
    return segments


def _reduce_system(
    system: RationalInequalitySystem,
    symmetry: OcticSymmetry,
) -> RationalInequalitySystem:
    orbits = octic_basis_orbits(symmetry)
    matrix = tuple(
        tuple(sum((row[position] for position in orbit), Fraction(0)) for orbit in orbits)
        for row in system.matrix
    )
    return RationalInequalitySystem(matrix, system.rhs)


def assemble_barrier_system(
    annuli: Sequence[BarrierAnnulus],
    signs: Sequence[int],
    *,
    degree: int = 8,
    symmetry: OcticSymmetry = "full",
    subdivision_depth: int = 0,
) -> RationalInequalitySystem:
    """Assemble ``M alpha >= 1`` from opposite boundary signs on each annulus."""
    if type(subdivision_depth) is not int or not 0 <= subdivision_depth <= 8:
        raise ValueError("subdivision_depth must be an integer from zero to eight")
    if len(annuli) != len(signs):
        raise ValueError("annuli and signs must have the same length")
    rows: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for annulus, sign in zip(annuli, signs, strict=True):
        if not isinstance(annulus, PolygonalAnnulus):
            raise TypeError("only PolygonalAnnulus barriers are supported in Route 2")
        for polygon, edge_sign in ((annulus.inner, -sign), (annulus.outer, sign)):
            for start, end in polygon.edges:
                for left, right in _dyadic_segments(start, end, subdivision_depth):
                    rows.extend(_segment_polynomial_rows(left, right, edge_sign, degree))
    if not rows:
        raise ValueError("no barrier rows were assembled")
    matrix = tuple(row for row, _ in rows)
    rhs = tuple(rhs for _, rhs in rows)
    return _reduce_system(RationalInequalitySystem(matrix, rhs), symmetry)


@dataclass(frozen=True)
class Route2LpReport:
    """Max-margin LP proposal with exact rational acceptance."""

    annulus_count: int
    constraint_count: int
    margin_float: float
    candidate_found: bool
    feasibility: RationalFeasibilityCertificate | None
    curve_certificate_passed: bool
    detail: str
    coefficients: tuple[Fraction, ...] | None = None
    converged: bool = False
    symmetry: OcticSymmetry = "full"
    farkas: RationalFarkasCertificate | None = None


@dataclass(frozen=True)
class Route2SearchReport:
    annulus_count: int
    constraint_count: int
    candidate_found: bool
    feasibility: RationalFeasibilityCertificate | None
    curve_certificate_passed: bool
    detail: str


def _candidate_values(
    indices: tuple[tuple[int, int, int], ...],
    active: Sequence[int],
    scale: Fraction,
) -> tuple[Fraction, ...]:
    return tuple(scale if pos in active else Fraction(0) for pos in range(len(indices)))


def _try_candidate(
    system: RationalInequalitySystem,
    annuli: Sequence[BarrierAnnulus],
    indices: tuple[tuple[int, int, int], ...],
    values: tuple[Fraction, ...],
    *,
    symmetry: OcticSymmetry = "full",
) -> Route2SearchReport | None:
    if not any(value != 0 for value in values):
        return None
    try:
        feasibility = certify_rational_feasibility(system, values)
    except ValueError:
        return None
    if not verify_rational_feasibility(system, feasibility):
        return None
    full_values = _expand_basis_values(values, symmetry)
    terms = {
        idx: full_values[pos]
        for pos, idx in enumerate(indices)
        if full_values[pos] != 0
    }
    curve = HomogeneousPlaneCurve(SparsePolynomial(3, terms))
    witness = find_smoothness_witness(curve, max_multiplier_degree=12)
    if witness is None:
        return None
    try:
        certificate = certify_curve(curve, witness, annuli)
    except (ArithmeticError, TypeError, ValueError):
        return None
    passed = replay_curve_certificate(curve, certificate)
    return Route2SearchReport(
        len(annuli),
        system.n_constraints,
        True,
        feasibility,
        passed,
        "exact rational coefficient vector passed barrier feasibility",
    )


def _expand_basis_values(
    values: Sequence[Fraction],
    symmetry: OcticSymmetry,
) -> tuple[Fraction, ...]:
    orbits = octic_basis_orbits(symmetry)
    if len(values) != len(orbits):
        raise ValueError("coefficient vector does not match the symmetry basis")
    expanded = [Fraction(0) for _ in octic_monomial_indices()]
    for value, orbit in zip(values, orbits, strict=True):
        for position in orbit:
            expanded[position] = value
    return tuple(expanded)


def _round_and_rescale(
    system: RationalInequalitySystem,
    values: Sequence[float],
    *,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
) -> tuple[Fraction, ...] | None:
    for denominator in denominators:
        rounded = tuple(Fraction(value).limit_denominator(denominator) for value in values)
        raw = tuple(
            sum((entry * value for entry, value in zip(row, rounded, strict=True)), Fraction(0))
            for row in system.matrix
        )
        minimum = min(raw)
        if minimum <= 0:
            continue
        scale = Fraction(1) if minimum >= 1 else Fraction(1) / minimum
        candidate = tuple(scale * value for value in rounded)
        if all(residual >= 0 for residual in system.residuals(candidate)):
            return candidate
    return None


def search_route2_lp(
    annuli: Sequence[BarrierAnnulus],
    signs: Sequence[int],
    *,
    coeff_bound: float = 1.0,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
    symmetry: OcticSymmetry = "full",
    subdivision_depth: int = 0,
    certify_infeasible: bool = False,
    anchor: tuple[int, int] | None = None,
    barrier_system: RationalInequalitySystem | None = None,
) -> Route2LpReport:
    """Maximize normalized strict margin; accept only after exact ``Q`` replay."""
    if not isfinite(coeff_bound) or coeff_bound <= 0:
        raise ValueError("coeff_bound must be positive and finite")
    if anchor is not None and coeff_bound != 1.0:
        raise ValueError("anchored projective charts require coeff_bound=1")
    system = barrier_system or assemble_barrier_system(
        annuli, signs, symmetry=symmetry, subdivision_depth=subdivision_depth
    )
    normalized_rows: list[tuple[Fraction, ...]] = []
    seen_normalized: set[tuple[Fraction, ...]] = set()
    for row_index, row in enumerate(system.matrix):
        row_scale = sum((abs(value) for value in row), Fraction(0))
        if row_scale <= 0:
            farkas = None
            if certify_infeasible:
                multiplier = tuple(
                    Fraction(1) if index == row_index else Fraction(0)
                    for index in range(system.n_constraints)
                )
                try:
                    farkas = certify_rational_infeasibility(system, multiplier)
                except ValueError:
                    pass
            return Route2LpReport(
                len(annuli), system.n_constraints, -1.0, False, None, False,
                "symmetry-reduced barrier system contains an impossible zero row",
                converged=True,
                symmetry=symmetry,
                farkas=farkas,
            )
        normalized = tuple(value / row_scale for value in row)
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            normalized_rows.append(normalized)
    matrix = torch.tensor(normalized_rows, dtype=torch.float64)
    basis_width = matrix.shape[1]
    anchor_index: int | None = None
    anchor_sign = 0
    if anchor is not None:
        anchor_index, anchor_sign = anchor
        if not 0 <= anchor_index < basis_width or anchor_sign not in (-1, 1):
            raise ValueError("anchor must be a valid basis index and sign")
        free_positions = tuple(index for index in range(basis_width) if index != anchor_index)
        lp_matrix = matrix[:, free_positions]
        barrier_bounds = matrix[:, anchor_index] * float(anchor_sign)
    else:
        free_positions = tuple(range(basis_width))
        lp_matrix = matrix
        barrier_bounds = torch.zeros(matrix.shape[0], dtype=torch.float64)
    n_vars = len(free_positions)
    rows: list[torch.Tensor] = []
    bounds: list[float] = []
    for row_index in range(matrix.shape[0]):
        constraint = torch.zeros(n_vars + 1, dtype=torch.float64)
        constraint[:n_vars] = -lp_matrix[row_index]
        constraint[-1] = 1.0
        rows.append(constraint)
        bounds.append(float(barrier_bounds[row_index].item()))
    for var_index in range(n_vars):
        upper = torch.zeros(n_vars + 1, dtype=torch.float64)
        upper[var_index] = 1.0
        rows.append(upper)
        bounds.append(coeff_bound)
        lower = torch.zeros(n_vars + 1, dtype=torch.float64)
        lower[var_index] = -1.0
        rows.append(lower)
        bounds.append(coeff_bound)
    nonneg_margin = torch.zeros(n_vars + 1, dtype=torch.float64)
    nonneg_margin[-1] = -1.0
    rows.append(nonneg_margin)
    bounds.append(2.0 if anchor is not None else 1.0)
    max_margin = torch.zeros(n_vars + 1, dtype=torch.float64)
    max_margin[-1] = 1.0
    rows.append(max_margin)
    bounds.append(1.0)
    objective = torch.zeros(n_vars + 1, dtype=torch.float64)
    objective[-1] = -1.0
    x0 = torch.zeros(n_vars + 1, dtype=torch.float64)
    x0[-1] = -1.5 if anchor is not None else -0.5
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        try:
            solution = solve_lp(
                objective,
                torch.stack(rows),
                torch.tensor(bounds, dtype=torch.float64),
                x0=x0,
            )
        finally:
            torch.set_num_threads(previous_threads)
    except InfeasibleProblemError as error:
        return Route2LpReport(
            len(annuli), system.n_constraints, -1.0, False, None, False,
            f"normalized barrier LP is infeasible: {error}",
            converged=False,
            symmetry=symmetry,
        )
    free_coeffs = solution.x[:n_vars].detach().cpu().tolist()
    if anchor_index is None:
        coeffs = free_coeffs
    else:
        coeffs = []
        free_cursor = 0
        for index in range(basis_width):
            if index == anchor_index:
                coeffs.append(float(anchor_sign))
            else:
                coeffs.append(float(free_coeffs[free_cursor]))
                free_cursor += 1
    margin = float(solution.x[-1].item())
    if (
        not solution.converged
        or not isfinite(margin)
        or not all(isfinite(value) for value in coeffs)
    ):
        return Route2LpReport(
            len(annuli), system.n_constraints, margin, False, None, False,
            "normalized max-margin LP did not converge",
            converged=False,
            symmetry=symmetry,
        )
    if margin <= 0:
        farkas = None
        if certify_infeasible:
            try:
                candidate = propose_farkas_infeasibility(system)
                if verify_rational_infeasibility(system, candidate):
                    farkas = candidate
            except (RuntimeError, ValueError):
                pass
        return Route2LpReport(
            len(annuli), system.n_constraints, margin, False, None, False,
            "normalized max-margin is nonpositive",
            converged=True,
            symmetry=symmetry,
            farkas=farkas,
        )
    rounded = _round_and_rescale(system, coeffs, denominators=denominators)
    if rounded is None:
        return Route2LpReport(
            len(annuli),
            system.n_constraints,
            margin,
            False,
            None,
            False,
            "positive float margin did not survive exact rational lifting",
            converged=True,
            symmetry=symmetry,
        )
    try:
        feasibility = certify_rational_feasibility(system, rounded)
    except ValueError:
        feasibility = None
    if feasibility is None or not verify_rational_feasibility(system, feasibility):
        return Route2LpReport(
            len(annuli),
            system.n_constraints,
            margin,
            False,
            None,
            False,
            "rounded LP coefficients failed exact rational feasibility replay",
            rounded,
            True,
            symmetry,
        )
    report = _try_candidate(
        system,
        annuli,
        octic_monomial_indices(),
        rounded,
        symmetry=symmetry,
    )
    if report is None:
        return Route2LpReport(
            len(annuli),
            system.n_constraints,
            margin,
            True,
            feasibility,
            False,
            "exact rational lift passed; curve smoothness is not yet sealed",
            rounded,
            True,
            symmetry,
        )
    return Route2LpReport(
        len(annuli),
        system.n_constraints,
        margin,
        True,
        report.feasibility,
        report.curve_certificate_passed,
        report.detail,
        rounded,
        True,
        symmetry,
    )


def seal_route2_witness(certificate: ProjectiveCurveCertificate) -> ProjectiveCurveFormalVerification:
    """Replay the Route-2 curve certificate through ``polynomial_identity_q``."""
    return verify_curve_certificate_formally(certificate)


def certify_route2_realization(
    annuli: Sequence[BarrierAnnulus],
    signs: Sequence[int],
    values: Sequence[Rational],
    *,
    max_multiplier_degree: int = 12,
    subdivision_depth: int = 8,
    fixed_axis: int = 2,
    symmetry: OcticSymmetry = "full",
    barrier_subdivision_depth: int = 0,
) -> ProjectiveCurveCertificate:
    """Seal a Route-2 witness through smoothness and curve replay."""
    system = assemble_barrier_system(
        annuli,
        signs,
        symmetry=symmetry,
        subdivision_depth=barrier_subdivision_depth,
    )
    feasibility = certify_rational_feasibility(system, values)
    if not verify_rational_feasibility(system, feasibility):
        raise ValueError("rational feasibility certificate does not replay")
    indices = octic_monomial_indices()
    full_values = _expand_basis_values(tuple(values), symmetry)
    terms = {
        indices[pos]: full_values[pos]
        for pos in range(len(indices))
        if full_values[pos] != 0
    }
    curve = HomogeneousPlaneCurve(SparsePolynomial(3, terms))
    witness = find_smoothness_witness(curve, max_multiplier_degree=max_multiplier_degree)
    if witness is None:
        raise ValueError("no bounded complex-smoothness witness for the candidate curve")
    certificate = certify_curve(
        curve,
        witness,
        annuli,
        fixed_axis=fixed_axis,
        subdivision_depth=subdivision_depth,
    )
    if not replay_curve_certificate(curve, certificate):
        raise ValueError("curve certificate replay failed")
    return certificate


def search_route2_coefficients(
    annuli: Sequence[BarrierAnnulus],
    signs: Sequence[int],
    *,
    template_scales: Sequence[Fraction] = (Fraction(1), Fraction(1, 2), Fraction(2)),
    max_pair_support: int = 32,
) -> Route2SearchReport:
    """Exact-Q semidecision search over sparse positive coefficient templates.

    Tries every single-monomial template across all 45 octic coefficients,
    then a bounded family of two-monomial supports. LP/SMT may propose, but
    acceptance always replays ``M alpha >= 1`` exactly.
    """
    system = assemble_barrier_system(annuli, signs)
    indices = octic_monomial_indices()
    for scale in template_scales:
        for pos in range(len(indices)):
            report = _try_candidate(system, annuli, indices, _candidate_values(indices, (pos,), scale))
            if report is not None:
                return report
        pair_budget = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if pair_budget >= max_pair_support:
                    break
                pair_budget += 1
                report = _try_candidate(
                    system, annuli, indices, _candidate_values(indices, (i, j), scale)
                )
                if report is not None:
                    return report
    return Route2SearchReport(
        len(annuli),
        system.n_constraints,
        False,
        None,
        False,
        "no sparse template in the searched family satisfied M alpha >= 1",
    )


def search_route2_full_coefficients(
    annuli: Sequence[BarrierAnnulus],
    signs: Sequence[int],
    *,
    template_scales: Sequence[Fraction] = (Fraction(1), Fraction(1, 2), Fraction(2)),
    max_support: int = 3,
    support_pool: int = 12,
) -> Route2SearchReport:
    """Exact-Q search over all 45 coefficients with bounded support size."""
    from itertools import combinations

    system = assemble_barrier_system(annuli, signs)
    indices = octic_monomial_indices()
    pool = tuple(range(min(support_pool, len(indices))))
    for support_size in range(1, max_support + 1):
        for active in combinations(pool, support_size):
            for scale in template_scales:
                report = _try_candidate(system, annuli, indices, _candidate_values(indices, active, scale))
                if report is not None:
                    return report
    return Route2SearchReport(
        len(annuli),
        system.n_constraints,
        False,
        None,
        False,
        "no bounded-support coefficient vector satisfied M alpha >= 1",
    )
