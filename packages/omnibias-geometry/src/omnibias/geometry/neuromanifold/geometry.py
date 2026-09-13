# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Function-observation geometry with explicit rank and quotient provenance.

Numerical diagnostics in this module are not certificates. Finite observations
are not global function equality. Exact affine quotients below are quotients
of the supplied linear map, with scope recorded in the chart.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from omnibias.core.realization import RealizationSpec

Array = NDArray[np.float64]
T = TypeVar("T")


@dataclass(frozen=True)
class Realization(Generic[T]):
    """A realization of explicitly declared observations and live linear actions."""

    value: Callable[[T], T]
    jvp: Callable[[T, T], T]
    vjp: Callable[[T, T], T]
    jet: Callable[[T, T, int], T]
    spec: RealizationSpec | None = None
    observation_scope: str = "finite_observations"
    derivative_provenance: str = "supplied_analytic_jets"

    def metric_action(self, theta: T, direction: T, weight: Callable[[T], T]) -> T:
        """J^T W J v without materializing a parameter Hessian."""
        return self.vjp(theta, weight(self.jvp(theta, direction)))


@dataclass(frozen=True)
class RankReport:
    numerical_rank: int
    singular_values: tuple[float, ...]
    threshold: float
    certified_lower: int | None = None
    certified_upper: int | None = None
    explanation: str = "numerical observation rank; intrinsic image rank is unresolved"


def rank_report(jacobian: Array, *, rtol: float = 1e-10, atol: float = 1e-12) -> RankReport:
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or not np.isfinite(j).all() or rtol < 0 or atol < 0:
        raise ValueError("a finite matrix and nonnegative tolerances are required")
    s = np.linalg.svd(j, compute_uv=False)
    threshold = max(atol, rtol * float(s[0]) if s.size else atol)
    return RankReport(int(np.sum(s > threshold)), tuple(float(x) for x in s), threshold)


@dataclass(frozen=True)
class ObservationMetric:
    """Fixed positive observation weighting, with its integration semantics.

    Quadrature and empirical weights are finite approximations unless their
    own enclosure/remainder witness is supplied to a downstream verifier.
    """

    weight: Array
    provenance: Literal["empirical", "quadrature", "sobolev", "residual", "measure"] = "empirical"

    def __post_init__(self) -> None:
        w = np.asarray(self.weight, dtype=float)
        if w.ndim == 1:
            w = np.diag(w)
        if (
            w.ndim != 2
            or w.shape[0] != w.shape[1]
            or not np.isfinite(w).all()
            or not np.allclose(w, w.T, rtol=0, atol=1e-14)
        ):
            raise ValueError("weight must be a finite symmetric matrix or positive vector")
        # Quadratic observation losses depend on the symmetric part. Remove
        # accepted last-bit asymmetry so actions and projections use that metric.
        w = 0.5 * w + 0.5 * w.T
        np.linalg.cholesky(w)
        object.__setattr__(self, "weight", w.copy())

    def action(self, jacobian: Array, direction: Array) -> Array:
        return np.asarray(jacobian.T @ (self.weight @ (jacobian @ direction)))

    def dense(self, jacobian: Array) -> Array:
        return np.asarray(jacobian.T @ self.weight @ jacobian)


@dataclass(frozen=True)
class ExtrinsicGeometry:
    tangent_projection: Array
    normal_projection: Array
    second_fundamental: Array | None
    numerical_rank: int


def extrinsic_geometry(
    jacobian: Array, metric: ObservationMetric, hessian: Array | None = None, *, rtol: float = 1e-10
) -> ExtrinsicGeometry:
    """Weighted projectors and II on a regular, full-column-rank chart.

    Hessian shape is (observations, chart_dim, chart_dim). A deficient Jacobian
    is refused; its pseudoinverse would not by itself establish a quotient.
    """
    j = np.asarray(jacobian, dtype=float)
    report = rank_report(j, rtol=rtol)
    if report.numerical_rank != j.shape[1]:
        raise ValueError("regular full-column-rank chart required; rank event invalidates chart")
    g = metric.dense(j)
    tangent = j @ np.linalg.solve(g, j.T @ metric.weight)
    normal = np.eye(j.shape[0]) - tangent
    ii = None
    if hessian is not None:
        h = np.asarray(hessian, dtype=float)
        if h.shape != (j.shape[0], j.shape[1], j.shape[1]):
            raise ValueError("observation Hessian has incompatible shape")
        ii = np.einsum("ab,bij->aij", normal, h)
    return ExtrinsicGeometry(tangent, normal, ii, report.numerical_rank)


def normal_acceleration(jacobian: Array, acceleration: Array, metric: ObservationMetric) -> Array:
    return np.asarray(extrinsic_geometry(jacobian, metric).normal_projection @ acceleration)


def least_squares_hessian(
    jacobian: Array, observation_hessian: Array, residual: Array, metric: ObservationMetric
) -> Array:
    """Hessian of 1/2 r^T W r, including realization curvature; W is fixed."""
    h = np.asarray(observation_hessian)
    if h.shape != (jacobian.shape[0], jacobian.shape[1], jacobian.shape[1]):
        raise ValueError("observation Hessian has incompatible shape")
    return cast(Array, metric.dense(jacobian) + np.einsum("a,aij->ij", metric.weight @ residual, h))


@dataclass(frozen=True)
class VisibilityReport:
    first_visible_order: int | None
    leading_vector: tuple[float, ...] | None
    status: str


def higher_order_visibility(
    derivatives: Sequence[Array], *, atol: float = 1e-12
) -> VisibilityReport:
    """Finite-order visibility along one direction; never a full stratum classifier."""
    if atol < 0:
        raise ValueError("atol must be nonnegative")
    for order, derivative in enumerate(derivatives, 1):
        d = np.asarray(derivative, dtype=float).reshape(-1)
        if np.linalg.norm(d) > atol:
            return VisibilityReport(order, tuple(float(x) for x in d), "visible")
    return VisibilityReport(None, None, "inconclusive_at_supplied_order")


def escape_candidates(
    directions: Array, loss_derivatives: Array, *, atol: float = 1e-12
) -> tuple[Array, ...]:
    """Directions with a finite-order descent term, including odd-order sign flips."""
    if loss_derivatives.ndim != 2 or directions.shape[0] != loss_derivatives.shape[0]:
        raise ValueError("one derivative row per direction is required")
    result = []
    for v, row in zip(directions, loss_derivatives, strict=False):
        for order, coefficient in enumerate(row, 1):
            if abs(coefficient) <= atol:
                continue
            if coefficient < 0:
                result.append(v.copy())
            elif order % 2:
                result.append(-v)
            break
    return tuple(result)


def _rref(matrix: Sequence[Sequence[Fraction]]) -> tuple[list[list[Fraction]], tuple[int, ...]]:
    a = [list(row) for row in matrix]
    pivots: list[int] = []
    row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(row, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        divisor = a[row][col]
        a[row] = [v / divisor for v in a[row]]
        for i in range(len(a)):
            if i != row:
                factor = a[i][col]
                a[i] = [v - factor * w for v, w in zip(a[i], a[row], strict=False)]
        pivots.append(col)
        row += 1
        if row == len(a):
            break
    return a, tuple(pivots)


@dataclass(frozen=True)
class RegularQuotientChart:
    """An exact affine local slice of a declared linear realization.

    The source rational matrix is retained so exact kernel and right-inverse
    relations can be replayed. This does not infer a nonlinear neural symmetry
    from a sampled nullspace. Chart coordinates are independent columns.
    """

    source: tuple[tuple[Fraction, ...], ...]
    independent_columns: tuple[int, ...]
    projection: tuple[tuple[Fraction, ...], ...]
    kernel: tuple[tuple[Fraction, ...], ...]
    observation_scope: str

    def verify(self) -> bool:
        """Replay factorization, right inverse, independent image, and full kernel."""
        a, p, r = self.source, self.independent_columns, self.projection
        if not a or not a[0] or not p:
            return False
        n, d = len(a[0]), len(p)
        if (
            len(set(p)) != d
            or any(j < 0 or j >= n for j in p)
            or len(r) != d
            or any(len(row) != n for row in (*a, *r))
            or any(not isinstance(v, Fraction) for row in (*a, *r) for v in row)
        ):
            return False
        if any(r[i][p[j]] != int(i == j) for i in range(d) for j in range(d)):
            return False
        if any(
            sum((row[p[k]] * r[k][j] for k in range(d)), Fraction()) != row[j]
            for row in a
            for j in range(n)
        ):
            return False
        if len(_rref([[row[j] for j in p] for row in a])[1]) != d:
            return False
        expected = []
        for j in range(n):
            if j in p:
                continue
            vector = [Fraction(0)] * n
            vector[j] = Fraction(1)
            for i, pivot in enumerate(p):
                vector[pivot] = -r[i][j]
            expected.append(tuple(vector))
        return tuple(expected) == self.kernel

    @property
    def parameter_dim(self) -> int:
        return len(self.source[0])

    @property
    def dim(self) -> int:
        return len(self.independent_columns)

    def embed(self, coordinates: Array) -> Array:
        if not self.verify() or coordinates.shape != (self.dim,):
            raise ValueError("wrong coordinate dimension")
        theta = np.zeros(self.parameter_dim)
        theta[list(self.independent_columns)] = coordinates
        return theta

    def retract(self, theta: Array) -> Array:
        if not self.verify() or theta.shape != (self.parameter_dim,):
            raise ValueError("wrong parameter dimension")
        return np.asarray(self.projection, dtype=float) @ theta

    def valid_at(self, jacobian: Array, *, rtol: float = 1e-10) -> bool:
        """Numerical guard supplements the exact, constant-rank source witness."""
        source = np.asarray(self.source, dtype=float)
        return (
            self.verify()
            and jacobian.shape == source.shape
            and np.array_equal(jacobian, source)
            and rank_report(jacobian, rtol=rtol).numerical_rank == self.dim
        )


def affine_quotient(
    matrix: Sequence[Sequence[float | int | Fraction]],
    *,
    observation_scope: str = "finite_observations",
    max_entries: int = 1_000_000,
) -> RegularQuotientChart:
    """Construct an exact quotient for the affine map with this linear part."""
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("nonempty rectangular matrix required")
    if max(len(matrix) * len(matrix[0]), len(matrix[0]) ** 2) > max_entries:
        raise ValueError("explicit quotient coefficient budget exceeded")
    source = tuple(tuple(Fraction(v) for v in row) for row in matrix)
    reduced, pivots = _rref(source)
    if not pivots:
        raise ValueError("constant realization has no regular positive-dimensional chart")
    projection = tuple(tuple(row) for row in reduced[: len(pivots)])
    free = [j for j in range(len(source[0])) if j not in pivots]
    kernel = []
    for j in free:
        vector = [Fraction(0)] * len(source[0])
        vector[j] = Fraction(1)
        for i, pivot in enumerate(pivots):
            vector[pivot] = -projection[i][j]
        kernel.append(tuple(vector))
    return RegularQuotientChart(source, pivots, projection, tuple(kernel), observation_scope)


@dataclass(frozen=True)
class ReducedStep:
    coordinates: Array
    accepted: bool
    step_size: float
    loss: float


def quotient_step(
    coordinates: Array,
    gradient: Array,
    metric: Array,
    objective: Callable[[Array], float],
    *,
    damping: float = 0.0,
    retraction: Callable[[Array], Array] | None = None,
    max_backtracks: int = 20,
    solve: Callable[[Array, Array], Array] | None = None,
) -> ReducedStep:
    """Reduced-coordinate natural step with Armijo backtracking.

    Coordinates must already belong to a justified quotient chart. Damping
    regularizes its solve and does not create a quotient from a nullspace.
    ``solve(matrix, rhs)`` can delegate to an existing backend linear solver;
    the default uses NumPy. The matrix passed to it already includes damping.
    """
    if damping < 0 or max_backtracks < 1:
        raise ValueError("invalid damping or backtrack count")
    initial = float(objective(coordinates))
    linear_solve = np.linalg.solve if solve is None else solve
    direction = -np.asarray(linear_solve(metric + damping * np.eye(len(coordinates)), gradient))
    if direction.shape != coordinates.shape:
        raise ValueError("linear solver must return one direction per reduced coordinate")
    slope = float(gradient @ direction)
    if not np.isfinite(slope) or slope >= 0:
        return ReducedStep(coordinates.copy(), False, 0.0, initial)
    for k in range(max_backtracks):
        alpha = 0.5**k
        proposed = coordinates + alpha * direction
        if retraction is not None:
            proposed = retraction(proposed)
        value = float(objective(proposed))
        if np.isfinite(value) and value <= initial + 1e-4 * alpha * slope:
            return ReducedStep(proposed, True, alpha, value)
    return ReducedStep(coordinates.copy(), False, 0.0, initial)


__all__ = [
    "ExtrinsicGeometry",
    "ObservationMetric",
    "RankReport",
    "Realization",
    "ReducedStep",
    "RegularQuotientChart",
    "VisibilityReport",
    "affine_quotient",
    "escape_candidates",
    "extrinsic_geometry",
    "higher_order_visibility",
    "least_squares_hessian",
    "normal_acceleration",
    "quotient_step",
    "rank_report",
]
