# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact-``Q`` regular-height and Farkas certificates for patchworks.

Floating optimization is a proposer only.  A regular triangulation is accepted
only after every lower-hull inequality has been multiplied out over
:class:`fractions.Fraction`.  Infeasibility is accepted only from the exact
Farkas identity ``lambda >= 0``, ``lambda^T M = 0``,
``lambda^T rhs > 0``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Any

import numpy as np
from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lift import as_fraction
from omnibias.geometry.patchwork import (
    LatticePoint,
    Triangulation,
    lattice_points,
)

Rational = int | Fraction


def _q(value: Rational) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, Fraction):
        return value
    raise TypeError("exact rational values are required; floats are proposer-only")


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _det(a: LatticePoint, b: LatticePoint, c: LatticePoint) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@dataclass(frozen=True)
class RationalInequalitySystem:
    """A finite exact system ``matrix @ x >= rhs``."""

    matrix: tuple[tuple[Fraction, ...], ...]
    rhs: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(_q(value) for value in row) for row in self.matrix)
        rhs = tuple(_q(value) for value in self.rhs)
        if not rows or not rows[0]:
            raise ValueError("a nonempty inequality system is required")
        if len(rows) != len(rhs) or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("matrix and right-hand-side shapes do not match")
        object.__setattr__(self, "matrix", rows)
        object.__setattr__(self, "rhs", rhs)

    @property
    def n_variables(self) -> int:
        return len(self.matrix[0])

    @property
    def n_constraints(self) -> int:
        return len(self.matrix)

    def residuals(self, values: Sequence[Rational]) -> tuple[Fraction, ...]:
        vector = tuple(_q(value) for value in values)
        if len(vector) != self.n_variables:
            raise ValueError("candidate width does not match the inequality system")
        return tuple(
            sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction(0))
            - rhs
            for row, rhs in zip(self.matrix, self.rhs, strict=True)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "matrix": [[_fraction_payload(value) for value in row] for row in self.matrix],
            "rhs": [_fraction_payload(value) for value in self.rhs],
        }

    @property
    def digest(self) -> str:
        raw = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class RationalFeasibilityCertificate:
    """An exact feasible point and every checked residual."""

    system_digest: str
    values: tuple[Fraction, ...]
    residuals: tuple[Fraction, ...]
    seal: Cert


@dataclass(frozen=True)
class RationalFarkasCertificate:
    """An exact infeasibility witness for ``matrix @ x >= rhs``."""

    system_digest: str
    multipliers: tuple[Fraction, ...]
    annihilator: tuple[Fraction, ...]
    contradiction: Fraction
    seal: Cert


def certify_rational_feasibility(
    system: RationalInequalitySystem,
    values: Sequence[Rational],
    *,
    claim: str = "exact rational linear feasibility",
) -> RationalFeasibilityCertificate:
    """Accept ``values`` iff every exact residual is nonnegative."""
    vector = tuple(_q(value) for value in values)
    residuals = system.residuals(vector)
    if any(value < 0 for value in residuals):
        raise ValueError("candidate violates an exact rational inequality")
    payload = {
        "type": "rational_linear_feasible",
        "system_digest": system.digest,
        "values": [_fraction_payload(value) for value in vector],
        "residuals": [_fraction_payload(value) for value in residuals],
    }
    seal = make_certificate(
        claim=claim,
        payload=payload,
        honesty={
            "finite_rational_feasibility_verified": True,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return RationalFeasibilityCertificate(system.digest, vector, residuals, seal)


def verify_rational_feasibility(
    system: RationalInequalitySystem,
    certificate: RationalFeasibilityCertificate,
) -> bool:
    """Replay a rational feasibility certificate from its source system."""
    if not verify_certificate_digest(certificate.seal):
        return False
    try:
        expected = certify_rational_feasibility(system, certificate.values)
    except (TypeError, ValueError):
        return False
    return expected == certificate


def certify_rational_infeasibility(
    system: RationalInequalitySystem,
    multipliers: Sequence[Rational],
    *,
    claim: str = "exact rational Farkas infeasibility",
) -> RationalFarkasCertificate:
    """Accept an exact Farkas alternative for ``matrix @ x >= rhs``."""
    vector = tuple(_q(value) for value in multipliers)
    if len(vector) != system.n_constraints:
        raise ValueError("one Farkas multiplier per inequality is required")
    if any(value < 0 for value in vector):
        raise ValueError("Farkas multipliers must be nonnegative")
    annihilator = tuple(
        sum(
            (vector[row] * system.matrix[row][column] for row in range(system.n_constraints)),
            Fraction(0),
        )
        for column in range(system.n_variables)
    )
    contradiction = sum(
        (value * rhs for value, rhs in zip(vector, system.rhs, strict=True)),
        Fraction(0),
    )
    if any(annihilator) or contradiction <= 0:
        raise ValueError("multipliers do not furnish an exact Farkas contradiction")
    payload = {
        "type": "rational_farkas_infeasible",
        "system_digest": system.digest,
        "multipliers": [_fraction_payload(value) for value in vector],
        "annihilator": [_fraction_payload(value) for value in annihilator],
        "contradiction": _fraction_payload(contradiction),
    }
    seal = make_certificate(
        claim=claim,
        payload=payload,
        honesty={
            "finite_rational_infeasibility_verified": True,
            "global_nonrealizability_claim": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return RationalFarkasCertificate(
        system.digest,
        vector,
        annihilator,
        contradiction,
        seal,
    )


def verify_rational_infeasibility(
    system: RationalInequalitySystem,
    certificate: RationalFarkasCertificate,
) -> bool:
    """Replay a Farkas certificate from its source system."""
    if not verify_certificate_digest(certificate.seal):
        return False
    try:
        expected = certify_rational_infeasibility(system, certificate.multipliers)
    except (TypeError, ValueError):
        return False
    return expected == certificate


@dataclass(frozen=True)
class RegularHeightSystem:
    """The strict lower-hull system attached to one triangulation."""

    triangulation: Triangulation
    margin: Fraction
    points: tuple[LatticePoint, ...]
    inequalities: RationalInequalitySystem


@dataclass(frozen=True)
class RegularHeightCertificate:
    """Exact regularity evidence for a triangulation."""

    triangulation: Triangulation
    margin: Fraction
    heights: tuple[Fraction, ...]
    feasibility: RationalFeasibilityCertificate


def lower_hull_inequalities(
    points: Sequence[LatticePoint],
    triangles: Sequence[tuple[LatticePoint, LatticePoint, LatticePoint]],
    *,
    margin: Rational = 1,
) -> RationalInequalitySystem:
    """Assemble strict lower-hull inequalities for a planar point configuration.

    Unlike :func:`regular_height_system`, this accepts a general finite planar
    configuration.  It is useful for exact replay of nonregular examples, whose
    infeasibility is certified by :func:`certify_rational_infeasibility`.
    """
    gap = _q(margin)
    if gap <= 0:
        raise ValueError("the strict lower-hull margin must be positive")
    vertices = tuple(points)
    if len(vertices) < 3 or len(set(vertices)) != len(vertices):
        raise ValueError("at least three distinct configuration points are required")
    if any(
        len(point) != 2 or any(type(coordinate) is not int for coordinate in point)
        for point in vertices
    ):
        raise TypeError("point configurations require integer planar coordinates")
    point_index = {point: i for i, point in enumerate(vertices)}
    faces = tuple(tuple(triangle) for triangle in triangles)
    if not faces:
        raise ValueError("at least one declared lower face is required")
    rows: list[tuple[Fraction, ...]] = []
    for triangle in faces:
        if (
            len(triangle) != 3
            or len(set(triangle)) != 3
            or any(point not in point_index for point in triangle)
        ):
            raise ValueError("lower faces require three distinct configuration points")
        a, b, c = triangle
        determinant = _det(a, b, c)
        if determinant == 0:
            raise ValueError("declared lower faces must be nondegenerate")
        for point in vertices:
            if point in triangle:
                continue
            barycentric = (
                Fraction(_det(point, b, c), determinant),
                Fraction(_det(a, point, c), determinant),
                Fraction(_det(a, b, point), determinant),
            )
            row = [Fraction(0) for _ in vertices]
            row[point_index[point]] = 1
            for vertex, weight in zip(triangle, barycentric, strict=True):
                row[point_index[vertex]] -= weight
            rows.append(tuple(row))
    return RationalInequalitySystem(tuple(rows), (gap,) * len(rows))


def regular_height_system(
    triangulation: Triangulation,
    *,
    margin: Rational = 1,
) -> RegularHeightSystem:
    """Assemble every exact strict lower-hull inequality."""
    gap = _q(margin)
    if gap <= 0:
        raise ValueError("the strict lower-hull margin must be positive")
    points = lattice_points(triangulation.degree)
    if any(_det(*triangle) != 1 for triangle in triangulation.triangles):
        raise ArithmeticError("triangulation orientation/unimodularity invariant failed")
    inequalities = lower_hull_inequalities(
        points,
        triangulation.triangles,
        margin=gap,
    )
    return RegularHeightSystem(triangulation, gap, points, inequalities)


def certify_regular_heights(
    triangulation: Triangulation,
    heights: Mapping[LatticePoint, Rational] | Sequence[Rational],
    *,
    margin: Rational = 1,
) -> RegularHeightCertificate:
    """Certify that exact heights induce every declared lower face."""
    system = regular_height_system(triangulation, margin=margin)
    if isinstance(heights, Mapping):
        if set(heights) != set(system.points):
            raise ValueError("height mapping must cover exactly the Newton lattice")
        vector = tuple(_q(heights[point]) for point in system.points)
    else:
        vector = tuple(_q(value) for value in heights)
    feasibility = certify_rational_feasibility(
        system.inequalities,
        vector,
        claim="declared patchwork triangulation has exact rational regular heights",
    )
    return RegularHeightCertificate(triangulation, system.margin, vector, feasibility)


def verify_regular_height_certificate(certificate: RegularHeightCertificate) -> bool:
    """Rebuild the lower-hull system and replay all exact inequalities."""
    try:
        expected = certify_regular_heights(
            certificate.triangulation,
            certificate.heights,
            margin=certificate.margin,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate


def propose_regular_heights(
    triangulation: Triangulation,
    *,
    bound: float = 10_000.0,
    denominator_bound: int = 1_000_000,
) -> RegularHeightCertificate:
    """Use the float64 convex solver as proposer, then certify over ``Q``.

    Three corner heights are fixed to zero after quotienting the affine-height
    lineality.  Finite box constraints make the zero-objective barrier problem
    bounded.  The returned object exists only after exact rescaling and replay.
    """
    if not np.isfinite(bound) or bound <= 1:
        raise ValueError("the proposer bound must be finite and greater than one")
    if type(denominator_bound) is not int or denominator_bound < 1:
        raise ValueError("denominator_bound must be a positive integer")
    system = regular_height_system(triangulation)
    fixed = {
        system.points.index((0, 0)),
        system.points.index((triangulation.degree, 0)),
        system.points.index((0, triangulation.degree)),
    }
    free = tuple(i for i in range(len(system.points)) if i not in fixed)
    reduced = np.asarray(
        [[float(row[i]) for i in free] for row in system.inequalities.matrix],
        dtype=float,
    )
    lower = -reduced
    a = np.concatenate((lower, np.eye(len(free)), -np.eye(len(free))), axis=0)
    b = np.concatenate((
        -np.ones(system.inequalities.n_constraints),
        np.full(len(free), bound),
        np.full(len(free), bound),
    ))
    from omnibias.convex.torch.solver import solve_lp

    solution = solve_lp(np.zeros(len(free)), a, b)
    proposed = [
        as_fraction(float(value), denom_bound=denominator_bound)
        for value in solution.x.detach().cpu().tolist()
    ]
    vector = [Fraction(0) for _ in system.points]
    for index, value in zip(free, proposed, strict=True):
        vector[index] = value
    raw = tuple(
        sum((entry * value for entry, value in zip(row, vector, strict=True)), Fraction(0))
        for row in system.inequalities.matrix
    )
    minimum = min(raw)
    if minimum <= 0:
        raise ArithmeticError("float proposer did not lift to a positive exact margin")
    scale = system.margin / minimum if minimum < system.margin else Fraction(1)
    return certify_regular_heights(
        triangulation,
        tuple(scale * value for value in vector),
    )


def propose_farkas_infeasibility(
    system: RationalInequalitySystem,
    *,
    denominator_bound: int = 1_000_000,
) -> RationalFarkasCertificate:
    """Propose a nonnegative Farkas vector with a float LP, then check in ``Q``."""
    if type(denominator_bound) is not int or denominator_bound < 1:
        raise ValueError("denominator_bound must be a positive integer")
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - extension environment
        raise RuntimeError("the optional SciPy proposer is unavailable") from exc
    matrix = np.asarray([[float(value) for value in row] for row in system.matrix])
    rhs = np.asarray([float(value) for value in system.rhs])
    result = linprog(
        np.zeros(system.n_constraints),
        A_ub=np.asarray([-rhs]),
        b_ub=np.asarray([-1.0]),
        A_eq=matrix.T,
        b_eq=np.zeros(system.n_variables),
        bounds=[(0.0, None)] * system.n_constraints,
        method="highs",
    )
    if not result.success:
        raise ValueError("float proposer did not find a Farkas witness")
    snapped = tuple(
        as_fraction(float(value), denom_bound=denominator_bound)
        for value in result.x
    )
    return certify_rational_infeasibility(system, snapped)


def clear_denominators(
    values: Sequence[Rational],
) -> tuple[int, ...]:
    """Return the common-denominator integer scaling of exact values."""
    vector = tuple(_q(value) for value in values)
    denominator = 1
    for value in vector:
        denominator = lcm(denominator, value.denominator)
    return tuple(int(value * denominator) for value in vector)


__all__ = [
    "RationalFarkasCertificate",
    "RationalFeasibilityCertificate",
    "RationalInequalitySystem",
    "RegularHeightCertificate",
    "RegularHeightSystem",
    "certify_rational_feasibility",
    "certify_rational_infeasibility",
    "certify_regular_heights",
    "clear_denominators",
    "lower_hull_inequalities",
    "propose_farkas_infeasibility",
    "propose_regular_heights",
    "regular_height_system",
    "verify_rational_feasibility",
    "verify_rational_infeasibility",
    "verify_regular_height_certificate",
]
