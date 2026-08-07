# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Constrained-expression algebra for hard boundary / initial conditions.

A *constrained expression* embeds a finite set of linear conditions into the
architecture, so they hold for **every** parameter value rather than being
chased by a penalty. For conditions :math:`C_k[u] = t_k`, pick ``k`` linearly
independent *support functions* :math:`s_j`, form the **support matrix**
:math:`M_{kj} = C_k[s_j]`, and define *switching functions*
:math:`\varphi_i = \sum_j (M^{-1})_{ji}\, s_j`, which by construction satisfy
:math:`C_k[\varphi_i] = \delta_{ki}`. Then for a free function ``g``

.. math:: u = g + \sum_k \varphi_k \,\bigl(t_k - C_k[g]\bigr)

satisfies :math:`C_m[u] = C_m[g] + (t_m - C_m[g]) = t_m` exactly, for *any*
``g``. Nothing is fitted: the condition is an algebraic identity. This is the
Theory of Functional Connections (Mortari 2017; Leake and Mortari,
*Mathematics* **8**\ (8):1303, 2020), and the multi-axis case follows by
applying the univariate form once per axis.

Two facts govern whether the construction is legitimate, and both are checked
rather than assumed:

* **The support matrix must be invertible.** Singularity depends on the
  *combination* of support functions and conditions, not on either alone -- the
  classic trap is a condition set that a given support family cannot
  interpolate. :func:`certify_support_matrix` seals a rigorous enclosure of
  :math:`\lambda_{\min}(M^{\mathsf{T}}M) > 0`, using the Gram matrix because
  ``M`` is generally non-symmetric. It is a finite rational obligation, so it is
  in scope for the Lean kernel.
* **Multi-axis data must agree where axes meet.** Applying axis ``a``'s
  operator to axis ``b``'s target must equal axis ``b``'s operator applied to
  axis ``a``'s target. That is a condition on the *data* -- physically, the
  initial state must agree with the boundary state at ``t = t0`` -- not on the
  method, and violating it leaves an order-one residual rather than a small one.
  :func:`corner_pairs` enumerates every pair that has to agree, over any number
  of axes; evaluating them needs user target callables, so the arithmetic lives
  in the backend cages, sharing :func:`apply_constraint` and
  :func:`compatibility_sample` so the two cannot disagree about what was checked.

A constraint is not restricted to a single point: :func:`periodic` ties two
faces together as the *relative* functional ``u(hi) - u(lo) = 0``, which is how
periodicity becomes structural rather than a penalty.

This module is *pure Python*: no torch, no jax, no numpy. Both backend cages
import the switching coefficients from here, so they cannot disagree about the
geometry of the ansatz.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import factorial
from typing import Any

#: Reject a support matrix whose certified spectral condition number exceeds
#: this. The switching functions are built by inverting ``M``, so its
#: conditioning is exactly the amplification applied to the condition data; a
#: badly conditioned ``M`` yields an ansatz that is exact in theory and noise in
#: float64. Well-posed sets sit far below: a four-condition Hermite family on
#: the unit interval certifies at ``kappa(M) <= 24``.
SUPPORT_CONDITION_LIMIT = 1.0e8


@dataclass(frozen=True)
class ConstraintTerm:
    """One ``coef * d^order u / dx^order`` evaluated at ``point``."""

    coef: float
    point: float
    order: int = 0

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"constraint order must be >= 0, got {self.order}")


@dataclass(frozen=True)
class LinearConstraint:
    """A linear functional ``C[u] = sum_terms coef * u^(order)(point)`` on one axis.

    Dirichlet, Neumann, Robin and an initial velocity are all this one type with
    different terms, which is why the cage needs no per-kind branching.
    """

    terms: tuple[ConstraintTerm, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("a linear constraint needs at least one term")

    @property
    def max_order(self) -> int:
        """Highest derivative order appearing in the functional."""
        return max(t.order for t in self.terms)

    @property
    def points(self) -> tuple[float, ...]:
        """The distinct evaluation points, in first-seen order."""
        seen: list[float] = []
        for t in self.terms:
            if t.point not in seen:
                seen.append(t.point)
        return tuple(seen)


def dirichlet(point: float, *, label: str = "") -> LinearConstraint:
    """``u(point) = value``."""
    return LinearConstraint((ConstraintTerm(1.0, point, 0),), label or f"u({point})")


def neumann(point: float, *, outward: float = 1.0, label: str = "") -> LinearConstraint:
    """``du/dn(point) = value``; ``outward`` is ``+1`` on a hi face, ``-1`` on a lo face.

    Folding the outward sign into the coefficient means the cage enforces the
    *normal* derivative the user asked for, not the axis derivative, with no
    sign convention left implicit at call time.
    """
    return LinearConstraint(
        (ConstraintTerm(outward, point, 1),), label or f"du/dn({point})"
    )


def robin(
    point: float,
    *,
    alpha: float,
    beta: float,
    outward: float = 1.0,
    label: str = "",
) -> LinearConstraint:
    """``alpha * u(point) + beta * du/dn(point) = value``."""
    if alpha == 0.0 and beta == 0.0:
        raise ValueError("a Robin condition needs alpha or beta nonzero")
    return LinearConstraint(
        (ConstraintTerm(alpha, point, 0), ConstraintTerm(beta * outward, point, 1)),
        label or f"{alpha}*u + {beta}*du/dn ({point})",
    )


def derivative_at(point: float, order: int, *, label: str = "") -> LinearConstraint:
    """``d^order u / dx^order (point) = value`` -- an initial velocity when ``order == 1``."""
    return LinearConstraint(
        (ConstraintTerm(1.0, point, order),), label or f"d^{order}u({point})"
    )


def periodic(lo: float, hi: float, *, order: int = 0, label: str = "") -> LinearConstraint:
    """``d^order u(hi) - d^order u(lo) = 0`` -- periodicity as a *relative* constraint.

    Every other helper here pins a value at one point; this one ties two points
    together without fixing either, which the switching form handles as readily
    because a linear functional may reference several points. ``order=0`` matches
    the value across the seam and ``order=1`` matches the slope; a smooth
    periodic solution satisfies both, and matching only the value leaves a
    solution free to have a kink exactly where nothing is watching.

    The target of a relative constraint is ``0`` and is not attached to either
    face, so it must be a constant -- a callable target would have no
    well-defined face to be evaluated on.
    """
    if hi == lo:
        raise ValueError(f"a periodic constraint needs two distinct points, got {lo}")
    prefix = "u" if order == 0 else f"d^{order}u"
    return LinearConstraint(
        (ConstraintTerm(1.0, hi, order), ConstraintTerm(-1.0, lo, order)),
        label or f"{prefix}({hi}) - {prefix}({lo})",
    )


def is_relative(constraint: LinearConstraint) -> bool:
    """Whether the functional ties several points together rather than pinning one."""
    return len(constraint.points) > 1


def face_point(constraint: LinearConstraint) -> float | None:
    """The single axis coordinate a constraint's target lives on.

    ``None`` for a relative constraint, whose target belongs to no single face.
    """
    points = constraint.points
    return points[0] if len(points) == 1 else None


def apply_constraint(
    constraint: LinearConstraint, evaluate: Callable[[float, int], Any]
) -> Any:
    """Apply ``constraint`` to whatever ``evaluate(point, order)`` returns.

    Deliberately generic over the value type: pure-Python floats in the tests,
    tensors in the torch cage, arrays in the jax cage. Both backends route their
    constraint operators through this one function, so a sign or coefficient can
    never drift between them.
    """
    total: Any = None
    for term in constraint.terms:
        piece = evaluate(term.point, term.order) * term.coef
        total = piece if total is None else total + piece
    return total


def _falling(n: int, k: int) -> float:
    """``n (n-1) ... (n-k+1)`` as a float; zero when ``k > n``."""
    if k > n:
        return 0.0
    return float(factorial(n) // factorial(n - k))


def support_derivative(index: int, xi: float, order: int, length: float) -> float:
    """``d^order/dx^order`` of the ``index``-th support monomial, at normalized ``xi``.

    Support functions are monomials in the affinely normalized coordinate
    ``xi = (x - lo) / length``, so each ``d/dx`` contributes a ``1 / length``.
    Normalizing first is what keeps the support matrix well conditioned on a
    domain that is not the unit interval.
    """
    if order > index:
        return 0.0
    return _falling(index, order) * xi ** (index - order) / length**order


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[Sequence[float]]) -> list[list[float]]:
    """Gauss-Jordan solve ``matrix @ X = rhs`` with partial pivoting (pure Python).

    Small dense systems only -- the support matrix is one row per condition on a
    single axis, so ``n`` is a handful.
    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    if len(rhs) != n:
        raise ValueError("rhs must have one row per matrix row")
    m = len(rhs[0])
    aug = [[float(v) for v in matrix[i]] + [float(v) for v in rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if aug[pivot][col] == 0.0:
            raise ValueError(
                "singular support matrix: the chosen support functions cannot "
                "interpolate this condition set"
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f != 0.0:
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [row[n : n + m] for row in aug]


#: How far above the minimum the monomial ladder is searched when picking a
#: support family. Every extra degree the search may reach is a degree the
#: switching polynomials may reach, so this stays small deliberately.
_DEGREE_HEADROOM = 3


@dataclass(frozen=True)
class AxisConstraints:
    """Every hard condition carried by one axis, plus the derived switching functions.

    ``switching[i]`` holds the monomial coefficients of :math:`\\varphi_i` in the
    normalized coordinate, so a backend evaluates it with a plain polynomial
    contraction and never re-derives the algebra.

    ``degrees`` records which monomials the support family uses. It is *not*
    always ``0 .. n-1``: a set of pure derivative conditions annihilates the low
    monomials -- two Neumann conditions see nothing of the constant, which is
    the well-posed pure-Neumann problem, not an ill-posed one -- so the degrees
    are chosen by rank instead of assumed.
    """

    axis: int
    lo: float
    hi: float
    constraints: tuple[LinearConstraint, ...]
    switching: tuple[tuple[float, ...], ...] = field(default=(), compare=False)
    degrees: tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("an axis with no constraints should not be caged")
        if self.hi <= self.lo:
            raise ValueError(f"axis bounds must satisfy lo < hi, got ({self.lo}, {self.hi})")
        if not self.degrees:
            object.__setattr__(self, "degrees", _select_support_degrees(self))
        if not self.switching:
            object.__setattr__(self, "switching", switching_coefficients(self))

    @property
    def length(self) -> float:
        """Axis extent, the scale that normalizes the support monomials."""
        return self.hi - self.lo

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    def normalize(self, x: float) -> float:
        """Map an axis coordinate into ``[0, 1]``."""
        return (x - self.lo) / self.length

    def projection_points(self) -> tuple[float, ...]:
        """Distinct axis coordinates the cage must evaluate the base field at."""
        seen: list[float] = []
        for c in self.constraints:
            for p in c.points:
                if p not in seen:
                    seen.append(p)
        return tuple(seen)


def _support_column(axis: AxisConstraints, degree: int) -> list[float]:
    """``[C_k[s_degree] for k]`` -- what one candidate monomial contributes."""
    return [
        float(
            apply_constraint(
                c,
                lambda p, o, _d=degree: support_derivative(  # type: ignore[misc]
                    _d, axis.normalize(p), o, axis.length
                ),
            )
        )
        for c in axis.constraints
    ]


def _select_support_degrees(axis: AxisConstraints) -> tuple[int, ...]:
    """Lowest monomial degrees whose support matrix has full rank.

    Walks the ladder ``1, x, x^2, ...`` and keeps a degree only when it adds a
    direction the already-kept ones do not span, by modified Gram-Schmidt. The
    tolerance here only decides *which* family to try; whether the resulting
    matrix is good enough is settled rigorously by
    :func:`certify_support_matrix`, so a borderline accept cannot slip through
    into an inexact ansatz.
    """
    n = axis.n_constraints
    ceiling = n + max(c.max_order for c in axis.constraints) + _DEGREE_HEADROOM
    chosen: list[int] = []
    basis: list[list[float]] = []
    for degree in range(ceiling + 1):
        col = _support_column(axis, degree)
        scale = max((abs(v) for v in col), default=0.0)
        if scale == 0.0:
            continue
        residual = list(col)
        for b in basis:
            proj = sum(r * v for r, v in zip(residual, b, strict=True))
            residual = [r - proj * v for r, v in zip(residual, b, strict=True)]
        norm = sum(v * v for v in residual) ** 0.5
        if norm <= 1e-10 * scale:
            continue
        basis.append([v / norm for v in residual])
        chosen.append(degree)
        if len(chosen) == n:
            return tuple(chosen)
    raise ValueError(
        f"singular support matrix: no monomial family up to degree {ceiling} can "
        f"interpolate these {n} conditions on axis {axis.axis}. They are linearly "
        "dependent as functionals, so no exact constrained expression exists"
    )


def support_matrix(axis: AxisConstraints) -> list[list[float]]:
    """``M[k][j] = C_k[s_j]`` over the chosen support degrees."""
    columns = [_support_column(axis, d) for d in axis.degrees]
    return [[col[k] for col in columns] for k in range(axis.n_constraints)]


def switching_coefficients(axis: AxisConstraints) -> tuple[tuple[float, ...], ...]:
    """Monomial coefficients of every switching function, in the normalized coordinate.

    Solves ``M alpha = I``; ``alpha[j][i]`` is the weight of support monomial
    ``degrees[j]`` in :math:`\\varphi_i`, which is exactly the statement
    ``C_k[phi_i] = delta_ki``. Returned densely by power, zero-padded over the
    degrees the family skipped, so a backend evaluates one plain polynomial and
    never has to know the family was sparse.
    """
    n = axis.n_constraints
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    alpha = solve_linear(support_matrix(axis), identity)
    width = max(axis.degrees) + 1
    out: list[tuple[float, ...]] = []
    for i in range(n):
        dense = [0.0] * width
        for j, degree in enumerate(axis.degrees):
            dense[degree] = alpha[j][i]
        out.append(tuple(dense))
    return tuple(out)


def switching_derivative_coeffs(
    coeffs: Sequence[float], order: int, length: float
) -> tuple[float, ...]:
    """Coefficients of ``d^order phi / dx^order``, still in the normalized monomial basis.

    Returned as a plain tuple so a backend evaluates
    ``sum_k out[k] * xi**k`` on a tensor with no Python loop over the batch.
    """
    if order == 0:
        return tuple(float(c) for c in coeffs)
    scale = length ** (-order)
    return tuple(
        float(coeffs[j]) * _falling(j, order) * scale for j in range(order, len(coeffs))
    )


def _certified_gram_spectrum(axis: AxisConstraints) -> tuple[Any, Any]:
    """``(lambda_min, lambda_max)`` enclosures for the **exact** Gram ``M^T M``.

    The float eigen-enclosure is a statement about the float Gram; Weyl's
    inequality widens it into a statement about the exact one, which is the
    version the certificate is allowed to claim.
    """
    from omnibias.core.verified.conditioning import (
        certified_max_eigenvalue,
        certified_min_eigenvalue,
    )
    from omnibias.core.verified.interval import Interval

    gram, radius = _gram_with_radius(axis)
    pad = Interval(-radius, radius)
    return certified_min_eigenvalue(gram) + pad, certified_max_eigenvalue(gram) + pad


def support_matrix_condition(axis: AxisConstraints) -> Any:
    """Rigorous enclosure of ``kappa(M)``, via the Gram matrix ``M^T M``.

    ``M`` is generally non-symmetric, and the certified eigenvalue machinery
    reasons about symmetric matrices, so the honest object is the Gram: ``M`` is
    invertible exactly when ``lambda_min(M^T M) > 0``, and
    ``kappa(M) = sqrt(kappa(M^T M))``. An unbounded upper endpoint is the honest
    signal that invertibility could not be certified -- note that squaring into
    the Gram costs half the available digits, so this fires at roughly
    ``kappa(M) ~ 1/sqrt(eps)``.
    """
    from omnibias.core.verified.interval import Interval

    lam_min, lam_max = _certified_gram_spectrum(axis)
    if lam_min.lo <= 0.0:
        return Interval(1.0, float("inf"))
    return (lam_max * lam_min.reciprocal()).sqrt()


def _gram_with_radius(axis: AxisConstraints) -> tuple[list[list[float]], float]:
    """``(midpoint of M^T M, sound Frobenius bound on its error)``.

    The support-matrix entries are built in outward-rounded interval arithmetic,
    so the Gram carries a radius. Weyl's inequality turns that radius into an
    additive bound on the eigenvalue error, which is how a float eigen-enclosure
    is made sound for the *exact* Gram rather than only for its float image.
    """
    from omnibias.core.verified.interval import Interval

    n = axis.n_constraints
    m_iv: list[list[Interval]] = []
    length = Interval.point(axis.length)
    for c in axis.constraints:
        row: list[Interval] = []
        for j in axis.degrees:
            acc = Interval.point(0.0)
            for term in c.terms:
                if term.order > j:
                    continue
                xi = (Interval.point(term.point) - Interval.point(axis.lo)) / length
                entry = (
                    Interval.point(term.coef)
                    * Interval.point(_falling(j, term.order))
                    * xi.pow_int(j - term.order)
                )
                if term.order:
                    entry = entry / length.pow_int(term.order)
                acc = acc + entry
            row.append(acc)
        m_iv.append(row)

    mid: list[list[float]] = []
    sq_rad = 0.0
    for i in range(n):
        out_row: list[float] = []
        for j in range(n):
            acc = Interval.point(0.0)
            for k in range(n):
                acc = acc + m_iv[k][i] * m_iv[k][j]
            out_row.append(acc.mid)
            sq_rad += acc.rad**2
        mid.append(out_row)
    return mid, sq_rad**0.5


def certify_support_matrix(
    axis: AxisConstraints,
    *,
    condition_limit: float = SUPPORT_CONDITION_LIMIT,
    claim: str | None = None,
) -> dict[str, Any]:
    """Seal a certificate that this condition set admits an exact constrained expression.

    Encloses ``lambda_min(M^T M)`` rigorously -- widened by the Weyl bound so the
    statement is about the exact Gram, not its float image -- and refuses when
    positive definiteness cannot be certified or the condition number exceeds
    ``condition_limit``. A refusal is the correct outcome for a linearly
    dependent condition set: there is no exact ansatz to build, and silently
    returning an approximate one would be the failure this whole construction
    exists to avoid.
    """
    from omnibias.core.proof.certificate import encode_interval, make_certificate

    _, radius = _gram_with_radius(axis)
    lam_min, lam_max = _certified_gram_spectrum(axis)
    if lam_min.lo <= 0.0:
        raise ValueError(
            "support matrix is not certified invertible "
            f"(lambda_min(M^T M) enclosed by [{lam_min.lo:.3e}, {lam_min.hi:.3e}]); "
            "the conditions are linearly dependent or the support family cannot "
            "interpolate them, so no exact constrained expression exists"
        )
    kappa = (lam_max * lam_min.reciprocal()).sqrt()
    if kappa.hi > condition_limit:
        raise ValueError(
            f"support matrix condition number kappa(M) <= {kappa.hi:.3e} exceeds the "
            f"limit {condition_limit:.3e}; the switching functions would amplify the "
            "condition data beyond float64 resolution"
        )
    payload = {
        "type": "constrained_expression_support",
        "axis": axis.axis,
        "bounds": [axis.lo, axis.hi],
        "n_constraints": axis.n_constraints,
        "labels": [c.label for c in axis.constraints],
        "gram_lambda_min": encode_interval(lam_min),
        "gram_lambda_max": encode_interval(lam_max),
        "condition_number": encode_interval(kappa),
        "gram_radius": radius,
    }
    return dict(
        make_certificate(
            claim=claim
            or (
                f"support matrix of {axis.n_constraints} hard condition(s) on axis "
                f"{axis.axis} is certified invertible"
            ),
            payload=payload,
            honesty={"unproven_claim": False},
            meta={"scope": "gram_matrix", "construction": "constrained_expression"},
        )
    )


@dataclass(frozen=True)
class HardCondition:
    """One condition to enforce structurally: component, axis, functional, target.

    ``target`` is a constant or a backend callable ``target(coords) -> array``
    evaluated on the face, so it may depend on the *other* coordinates. Kept
    ``Any`` because the two backends carry different array types; everything
    else about the condition is shared.
    """

    component: str
    axis: int
    constraint: LinearConstraint
    target: Any = 0.0


@dataclass(frozen=True)
class AxisPlan:
    """One recursion step: every condition a component carries on a single axis."""

    constraints: AxisConstraints
    targets: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.targets) != self.constraints.n_constraints:
            raise ValueError(
                f"axis {self.constraints.axis} has {self.constraints.n_constraints} "
                f"condition(s) but {len(self.targets)} target(s)"
            )


def group_hard_conditions(
    conditions: Sequence[HardCondition],
    bounds: Sequence[tuple[float, float]],
) -> dict[str, tuple[AxisPlan, ...]]:
    """Group conditions into per-component, per-axis recursion steps.

    Axes are ordered by index so the recursion is deterministic and the two
    backends apply the steps in the same order. Ordering does not change the
    result when the data is compatible, which is precisely what the
    compatibility gate checks -- but a fixed order keeps float rounding
    identical across backends.
    """
    grouped: dict[str, dict[int, list[HardCondition]]] = {}
    for cond in conditions:
        if not 0 <= cond.axis < len(bounds):
            raise ValueError(
                f"condition on axis {cond.axis} is outside the {len(bounds)} "
                "axes of the domain"
            )
        grouped.setdefault(cond.component, {}).setdefault(cond.axis, []).append(cond)

    plans: dict[str, tuple[AxisPlan, ...]] = {}
    for name, per_axis in grouped.items():
        steps: list[AxisPlan] = []
        for axis in sorted(per_axis):
            group = per_axis[axis]
            lo, hi = bounds[axis]
            steps.append(
                AxisPlan(
                    constraints=AxisConstraints(
                        axis=axis,
                        lo=lo,
                        hi=hi,
                        constraints=tuple(c.constraint for c in group),
                    ),
                    targets=tuple(c.target for c in group),
                )
            )
        plans[name] = tuple(steps)
    return plans


@dataclass(frozen=True)
class CornerPair:
    """Two conditions on different axes, whose data must agree where the axes meet.

    The construction embeds each axis in turn, so the *last* axis applied always
    wins; if the data disagrees, the earlier axis quietly loses. Enumerating the
    pairs explicitly is what turns that into a detected refusal rather than a
    solution that satisfies three of the four conditions it was given.
    """

    component: str
    axis_a: int
    constraint_a: LinearConstraint
    target_a: Any
    axis_b: int
    constraint_b: LinearConstraint
    target_b: Any

    @property
    def label(self) -> str:
        return (
            f"{self.constraint_a.label!r} on axis {self.axis_a} vs "
            f"{self.constraint_b.label!r} on axis {self.axis_b}"
        )


def corner_pairs(plans: dict[str, tuple[AxisPlan, ...]]) -> tuple[CornerPair, ...]:
    """Every (component, condition, condition) pair spanning two different axes.

    With ``A`` axes carrying ``n_a`` conditions each this is
    ``sum_{a<b} n_a n_b`` pairs -- quadratic in the conditions, not exponential
    in the axes, so checking all of them stays cheap as the recursion depth
    grows. Conditions on the *same* axis are excluded: they are embedded
    simultaneously by one support matrix, whose invertibility is the certificate's
    business rather than a data question.
    """
    out: list[CornerPair] = []
    for name, steps in plans.items():
        for i, first in enumerate(steps):
            for second in steps[i + 1 :]:
                for k, ck in enumerate(first.constraints.constraints):
                    for m, cm in enumerate(second.constraints.constraints):
                        out.append(
                            CornerPair(
                                component=name,
                                axis_a=first.constraints.axis,
                                constraint_a=ck,
                                target_a=first.targets[k],
                                axis_b=second.constraints.axis,
                                constraint_b=cm,
                                target_b=second.targets[m],
                            )
                        )
    return tuple(out)


#: Irrational multipliers for the compatibility sample -- square roots of the
#: first primes, whose fractional parts equidistribute (Weyl). Any axis beyond
#: this many reuses the list scaled by the repeat index, which keeps the points
#: distinct without needing a prime table.
_SAMPLE_ROOTS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def compatibility_sample(
    bounds: Sequence[tuple[float, float]], count: int = 16
) -> tuple[tuple[float, ...], ...]:
    """A small deterministic point set spanning ``bounds``, in pure Python.

    Used to evaluate condition *data* -- the targets, not the field -- when
    checking that conditions on different axes agree. A Kronecker lattice rather
    than an RNG draw, so the two backends see literally the same coordinates and
    a refusal is reproducible rather than seed-dependent. Compatibility is an
    identity in the data when it holds, so a handful of points suffices to
    expose it when it does not.
    """
    if count < 1:
        raise ValueError(f"compatibility sample needs at least one point, got {count}")
    alphas = [
        _SAMPLE_ROOTS[d % len(_SAMPLE_ROOTS)] * (1 + d // len(_SAMPLE_ROOTS))
        for d in range(len(bounds))
    ]
    out: list[tuple[float, ...]] = []
    for i in range(count):
        row: list[float] = []
        for d, (lo, hi) in enumerate(bounds):
            frac = ((i + 0.5) * alphas[d] ** 0.5) % 1.0
            row.append(lo + (hi - lo) * frac)
        out.append(tuple(row))
    return tuple(out)


def projection_cost(plans: dict[str, tuple[AxisPlan, ...]]) -> int:
    """Base-field evaluations one forward pass costs.

    The recursion pins each constrained axis in turn, so the distinct projected
    coordinate sets are the *product* over axes of ``1 + #projection points``,
    not the sum: a face carrying both a value and a slope still costs one, and a
    second constrained axis multiplies. Components share the cache, so a pin
    combination two components both need is paid for once.

    Worth computing before absorbing every face of a 3-D box, which is why it is
    a public number rather than a docstring claim.
    """
    combos: set[tuple[tuple[int, float], ...]] = set()
    for steps in plans.values():
        per_axis: list[list[tuple[int, float] | None]] = []
        for step in steps:
            points: list[tuple[int, float] | None] = [None]
            points.extend(
                (step.constraints.axis, p) for p in _axis_projection_points(step)
            )
            per_axis.append(points)
        stack: list[tuple[tuple[int, float], ...]] = [()]
        for points in per_axis:
            stack = [
                pins if pin is None else (*pins, pin) for pins in stack for pin in points
            ]
        combos.update(stack)
    return len(combos)


def _axis_projection_points(step: AxisPlan) -> tuple[float, ...]:
    """The distinct coordinates on one axis the recursion has to evaluate at."""
    seen: list[float] = []
    for constraint in step.constraints.constraints:
        for point in constraint.points:
            if point not in seen:
                seen.append(point)
    return tuple(seen)


__all__ = [
    "AxisConstraints",
    "AxisPlan",
    "ConstraintTerm",
    "CornerPair",
    "HardCondition",
    "LinearConstraint",
    "SUPPORT_CONDITION_LIMIT",
    "apply_constraint",
    "certify_support_matrix",
    "compatibility_sample",
    "corner_pairs",
    "derivative_at",
    "dirichlet",
    "face_point",
    "group_hard_conditions",
    "is_relative",
    "neumann",
    "periodic",
    "projection_cost",
    "robin",
    "solve_linear",
    "support_derivative",
    "support_matrix",
    "support_matrix_condition",
    "switching_coefficients",
    "switching_derivative_coeffs",
]
