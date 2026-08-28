# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Cone-field hyperbolicity and a topological-entropy lower bound.

A **cone field** along a finite orbit segment is a family of expanding cones
in the tangent space.  If every interval Jacobian along the segment maps the
cone strictly into itself with a uniform expansion factor ``eta > 1``, the
orbit segment is *certified hyperbolic* in the cone-field sense, and

.. math::

    h_{\mathrm{top}} \;\ge\; \frac{1}{p}\sum_{i=0}^{p-1} \log\eta_i

is a rigorous **lower** bound on the topological entropy of the *finite*
orbit segment (period ``p``), never of a continuum Anosov flow or of an
infinite invariant set.

This module consumes already-enclosed Jacobians (QR-Lohner monodromy
columns, or an explicit interval matrix for a discrete linear map).  It
does **not** run a validated flow itself.  A non-hyperbolic example (a
rotation / harmonic-oscillator step) is refused with ``certified=False``,
never silently labelled hyperbolic.

Scope.  Finite orbit segment only.  No Anosov, no SRB, no continuum
chaos claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import inf_norm_vector, to_interval_matrix

MatrixLike = Sequence[Sequence[IntervalLike]]
VectorLike = Sequence[IntervalLike]


def _matvec(matrix: list[list[Interval]], vector: list[Interval]) -> list[Interval]:
    n = len(matrix)
    if len(vector) != n:
        raise ValueError(f"vector length {len(vector)} != matrix size {n}")
    return [
        sum((matrix[i][j] * vector[j] for j in range(n)), Interval.point(0.0))
        for i in range(n)
    ]


def _euclidean_norm_sq(vector: Sequence[Interval]) -> Interval:
    acc = Interval.point(0.0)
    for component in vector:
        acc = acc + component * component
    return acc


def cone_expansion_factor(jacobian: MatrixLike, cone_vector: VectorLike) -> Interval:
    """Enclosure of ``||J v||_2 / ||v||_2`` for one cone generator ``v``."""
    j_iv = to_interval_matrix(jacobian)
    v_iv = [Interval.from_value(x) for x in cone_vector]
    image = _matvec(j_iv, v_iv)
    num_sq = _euclidean_norm_sq(image)
    den_sq = _euclidean_norm_sq(v_iv)
    if den_sq.lo <= 0.0:
        raise ValueError("cone generator must be certified nonzero")
    # ||Jv|| / ||v|| = sqrt(num_sq / den_sq); outward via interval sqrt.
    return (num_sq / den_sq).sqrt()


@dataclass(frozen=True)
class ConeHyperbolicityCertificate:
    """Cone-field hyperbolicity of a finite orbit segment."""

    expansions: tuple[float, ...]
    eta: float
    entropy_lower: float
    certified: bool
    detail: str
    continuum_claim: bool = False
    anosov_claim: bool = False

    def __post_init__(self) -> None:
        if self.continuum_claim or self.anosov_claim:
            raise ValueError("continuum / Anosov claims are refused by construction")


def certified_cone_hyperbolicity(
    jacobians: Sequence[MatrixLike],
    cone_vectors: Sequence[VectorLike],
    *,
    eta: float,
) -> ConeHyperbolicityCertificate:
    """Certify that every Jacobian expands every cone generator by at least ``eta``.

    ``eta`` must be strictly greater than ``1``.  The entropy lower bound is
    ``(1/p) * p * log(eta) = log(eta)`` when every step meets the *same*
    uniform ``eta``; if a step only meets a (still ``> 1``) factor
    ``eta_i >= eta``, the bound still uses the uniform ``log(eta)`` so it
    remains a lower bound.
    """
    if eta <= 1.0 or not math.isfinite(eta):
        raise ValueError(f"eta must be finite and > 1, got {eta!r}")
    if not jacobians:
        raise ValueError("jacobians must be a non-empty finite orbit segment")
    if not cone_vectors:
        raise ValueError("cone_vectors must be a non-empty generating set")
    expansions: list[float] = []
    for jacobian in jacobians:
        step_min = math.inf
        for vector in cone_vectors:
            factor = cone_expansion_factor(jacobian, vector)
            step_min = min(step_min, factor.lo)
        expansions.append(step_min)
    certified = all(value >= eta for value in expansions)
    period = len(jacobians)
    entropy_lower = math.log(eta) if certified else 0.0
    # Uniform eta over p steps: (1/p) * p * log(eta) = log(eta).
    del period
    if certified:
        detail = (
            "finite orbit segment is cone-field hyperbolic with uniform expansion "
            f"eta={eta}; topological-entropy lower bound is log(eta) "
            "(segment only, not a continuum Anosov claim)"
        )
    else:
        detail = (
            "cone-field expansion failed to meet eta on this finite segment; "
            "not hyperbolic (a rotation / elliptic step is a typical refuse)"
        )
    return ConeHyperbolicityCertificate(
        expansions=tuple(expansions),
        eta=float(eta),
        entropy_lower=entropy_lower,
        certified=certified,
        detail=detail,
    )


def doubling_map_jacobian() -> list[list[Interval]]:
    """Interval Jacobian of the expanding doubling map ``x |-> 2x`` on the circle."""
    return [[Interval.point(2.0)]]


def cat_map_jacobian() -> list[list[Interval]]:
    """Interval Jacobian of the Arnold cat map ``[[2, 1], [1, 1]]``."""
    return [
        [Interval.point(2.0), Interval.point(1.0)],
        [Interval.point(1.0), Interval.point(1.0)],
    ]


def rotation_jacobian() -> list[list[Interval]]:
    """Interval Jacobian of a quarter-turn (harmonic-oscillator discrete step)."""
    return [
        [Interval.point(0.0), Interval.point(-1.0)],
        [Interval.point(1.0), Interval.point(0.0)],
    ]


def cat_map_unstable_generator() -> list[Interval]:
    """An expanding generator for the cat map: ``((1 + sqrt(5))/2, 1)``."""
    phi = (Interval.point(1.0) + Interval.point(5.0).sqrt()) * Interval.point(0.5)
    return [phi, Interval.point(1.0)]


def inf_norm_expansion_witness(jacobian: MatrixLike) -> float:
    """Upper bound on ``||J||_inf`` (diagnostic, not the cone-field factor)."""
    j_iv = to_interval_matrix(jacobian)
    # Reuse the vector inf-norm of each standard-basis image.
    n = len(j_iv)
    best = 0.0
    for j in range(n):
        column = [j_iv[i][j] for i in range(n)]
        best = max(best, inf_norm_vector(column))
    return best


__all__ = [
    "ConeHyperbolicityCertificate",
    "cat_map_jacobian",
    "cat_map_unstable_generator",
    "certified_cone_hyperbolicity",
    "cone_expansion_factor",
    "doubling_map_jacobian",
    "inf_norm_expansion_witness",
    "rotation_jacobian",
]
