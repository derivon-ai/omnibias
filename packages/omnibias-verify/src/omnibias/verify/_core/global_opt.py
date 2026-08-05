# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified global minimization by interval branch-and-bound (Moore--Skelboe).

Gradient descent finds a *local* minimum and can never certify it is the global
one -- for a non-convex objective the true global minimum could sit in an
unexplored basin.  Interval branch-and-bound answers the global question with a
**proof**: it maintains a rigorous enclosure ``[f_lower, f_upper]`` of the global
minimum over a box, where

* ``f_upper`` is the value at a concrete feasible point (an upper bound on the
  global minimum), and
* ``f_lower`` is the smallest *sound lower bound* over all sub-boxes still able to
  contain the minimizer,

so ``f_lower <= min_{x in box} f(x) <= f_upper`` holds **unconditionally**, and the
search stops once the gap ``f_upper - f_lower`` drops below ``tol``.  The frontier
is a priority queue keyed by each sub-box's certified lower bound, so effort is
spent only where the minimum can still hide; a box whose lower bound already
exceeds the incumbent is discarded (soundly -- it cannot hold the minimizer).

This is the rigorous register of omnibias's optimisation stack, built on the
verified ``Interval`` substrate.  Two omnibias-specific accelerators use *exact*
derivative information (supplied as interval enclosures -- e.g. from the
closed-form :mod:`omnibias.core.verified.jet_mv` for network/field objectives, or
written by hand):

* **monotonicity test** -- if ``df/dx_i`` has a constant sign over a box, ``f`` is
  monotone in ``x_i`` there, so the minimum lies on a face; the axis is collapsed
  to the min-achieving endpoint (a rigorous volume reduction);
* **mean-value (centered) form** -- ``f(box) subset f(c) + sum_i g_i(box)(x_i-c_i)``
  is intersected with the natural interval extension, usually a much tighter
  lower bound (it cancels the first-order dependency overestimation).

The complementary :func:`certify_strict_local_min` certifies (via the interval
``LDL^T`` inertia in :mod:`omnibias.core.verified.eig_operator`) that the returned
point sits in a region where the Hessian is positive definite -- i.e. it is a
strict local minimizer -- which together with the global gap upgrades the answer
from "best found" to "certified global minimizer to tolerance ``tol``".

Scope / honesty: interval B&B is *sound for any dimension* but its cost grows
exponentially in the box dimension in the worst case (the curse of dimensionality
is real -- this is for low-dimensional global problems, not million-parameter
training).  It is *complete* only in the limit of infinite refinement; within a
finite ``max_boxes`` budget it returns the best rigorous enclosure obtained (which
is always sound, just possibly wider than ``tol``).
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.core.verified.eig_operator import is_positive_definite
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.verify._core.newton import krawczyk_contract

Box = tuple[Interval, ...]
#: A scalar objective as an *interval extension*: it maps a box (one Interval per
#: axis) to an Interval enclosing ``{f(x) : x in box}``.  Called on degenerate
#: point-boxes to read a feasible value.  Write it with :class:`Interval` algebra
#: and the ``*_iv`` transcendentals from :mod:`omnibias.core.verified.transcend`.
ObjectiveFn = Callable[[tuple[Interval, ...]], IntervalLike]
#: An interval gradient enclosure ``box -> (df/dx_0, ..., df/dx_{d-1})``.
GradFn = Callable[[tuple[Interval, ...]], Sequence[IntervalLike]]
#: An interval Hessian enclosure ``box -> [[d2f/dx_i dx_j]]``.
HessianFn = Callable[[tuple[Interval, ...]], Sequence[Sequence[IntervalLike]]]


def _as_box(box: Sequence[object]) -> Box:
    out: list[Interval] = []
    for b in box:
        if isinstance(b, Interval):
            out.append(b)
        elif isinstance(b, tuple | list) and len(b) == 2:
            out.append(Interval(float(b[0]), float(b[1])))
        else:
            out.append(Interval.from_value(b))  # type: ignore[arg-type]
    return tuple(out)


def _widest_axis(box: Box) -> int:
    return max(range(len(box)), key=lambda i: box[i].width)


def _split(box: Box, axis: int) -> tuple[Box, Box]:
    mid = box[axis].mid
    left = tuple(Interval(iv.lo, mid) if i == axis else iv for i, iv in enumerate(box))
    right = tuple(Interval(mid, iv.hi) if i == axis else iv for i, iv in enumerate(box))
    return left, right


def _center(box: Box) -> tuple[float, ...]:
    return tuple(iv.mid for iv in box)


def _point_box(point: Sequence[float]) -> Box:
    return tuple(Interval.point(x) for x in point)


def _clamp_into(point: Sequence[float], box: Box) -> tuple[float, ...]:
    """Clamp a concrete point per-axis into ``box`` so it is a feasible incumbent."""
    if len(point) != len(box):
        raise ValueError(f"seed has {len(point)} coords but the box has {len(box)} axes")
    return tuple(
        min(max(float(x), iv.lo), iv.hi) for x, iv in zip(point, box, strict=True)
    )


@dataclass(frozen=True)
class GlobalMinResult:
    r"""A rigorous enclosure of the global minimum and the work spent to get it.

    The guarantee is unconditional regardless of ``converged``:
    ``f_lower <= min_{x in box} f(x) <= f_upper`` and ``f(x_argmin) <= f_upper``.
    """

    x: tuple[float, ...]
    f_upper: float
    f_lower: float
    tol: float
    boxes_explored: int
    boxes_remaining: int

    @property
    def gap(self) -> float:
        return self.f_upper - self.f_lower

    @property
    def enclosure(self) -> Interval:
        return Interval(self.f_lower, self.f_upper)

    @property
    def converged(self) -> bool:
        """``True`` when the certified gap has reached ``tol``."""
        return self.gap <= self.tol

    #: Alias -- the enclosure is a *certificate* of global optimality to ``tol``.
    @property
    def certified(self) -> bool:
        return self.converged


def _enclose(
    f: ObjectiveFn, box: Box, grad: GradFn | None, hess: HessianFn | None = None
) -> Interval:
    r"""Sound enclosure of ``f`` over ``box`` -- the intersection of up to three
    unconditionally sound forms (each contains ``{f(x) : x in box}``):

    * **natural** interval extension ``f(box)``;
    * **mean-value (centered) form** ``f(c) + sum_i g_i(box)(x_i - c_i)`` -- cancels
      the first-order dependency overestimation (needs ``grad``);
    * **second-order (Taylor) form**
      ``f(c) + g(c)·(x-c) + 1/2 (x-c)^T H(box) (x-c)`` -- converges *quadratically*
      as the box shrinks, so it dominates on refined boxes where the first-order
      form plateaus (needs ``grad`` and ``hess``).

    Intersecting sound enclosures is itself sound, so adding a form can only
    tighten -- never invalidate -- the bound.
    """
    natural = Interval.from_value(f(box))
    if grad is None:
        return natural
    center = _center(box)
    cbox = _point_box(center)
    fc = Interval.from_value(f(cbox))
    dx = [iv - Interval.point(ci) for iv, ci in zip(box, center, strict=True)]
    mv = fc
    for gi, dxi in zip(grad(box), dx, strict=True):
        mv = mv + Interval.from_value(gi) * dxi
    lo, hi = max(natural.lo, mv.lo), min(natural.hi, mv.hi)
    if hess is not None:
        so = fc
        for gi, dxi in zip(grad(cbox), dx, strict=True):  # gradient at the centre
            so = so + Interval.from_value(gi) * dxi
        h = hess(box)
        for i, row in enumerate(h):
            for j, hij in enumerate(row):
                so = so + Interval.point(0.5) * Interval.from_value(hij) * dx[i] * dx[j]
        lo, hi = max(lo, so.lo), min(hi, so.hi)
    return Interval(lo, hi)


def _interior_full_dim(box: Box, domain: Box) -> bool:
    r"""``True`` iff ``box`` is full-dimensional and strictly inside ``domain``.

    Soundness gate for Krawczyk contraction: a global minimizer that lives in a
    *strictly interior*, full-dimensional sub-box is necessarily a stationary
    point of the unconstrained problem (an interior local min), hence a root of
    ``grad f`` -- so it is preserved by ``box ∩ K(box)``.  A box touching the
    domain boundary may hold a *boundary* minimizer (where ``grad f != 0``), which
    contraction could drop; such boxes are left to bisection instead.  A box with a
    collapsed axis (a face produced by the monotonicity test) is likewise excluded,
    since full-space stationarity does not apply on a lower-dimensional face.
    """
    for i in range(len(box)):
        if box[i].lo <= domain[i].lo or box[i].hi >= domain[i].hi:
            return False
        if box[i].lo >= box[i].hi:  # collapsed axis -> not full-dimensional
            return False
    return True


def _monotone_reduce(box: Box, grad: GradFn | None) -> Box:
    """Collapse every axis in which ``f`` is certified monotone over ``box``.

    If ``df/dx_i > 0`` on the whole box the minimum is at ``x_i = lo``; if
    ``df/dx_i < 0`` it is at ``x_i = hi``.  Sound coordinate-wise, hence jointly.
    """
    if grad is None:
        return box
    g = grad(box)
    reduced = list(box)
    for i, gi_like in enumerate(g):
        gi = Interval.from_value(gi_like)
        if gi.lo > 0.0:
            reduced[i] = Interval.point(box[i].lo)
        elif gi.hi < 0.0:
            reduced[i] = Interval.point(box[i].hi)
    return tuple(reduced)


def certified_minimize(
    f: ObjectiveFn,
    box: Sequence[object],
    *,
    tol: float = 1e-6,
    max_boxes: int = 100_000,
    grad: GradFn | None = None,
    hess: HessianFn | None = None,
    use_newton: bool = True,
    min_width: float = 1e-12,
    seeds: Sequence[Sequence[float]] | None = None,
) -> GlobalMinResult:
    r"""Rigorously enclose ``min_{x in box} f(x)`` by interval branch-and-bound.

    ``f`` must be an *interval extension*: given one :class:`Interval` per axis it
    returns an Interval enclosing the objective over that box (and, on a degenerate
    point-box, a tight enclosure of the value there).  Optional accelerators, all
    using *exact* derivative enclosures and all preserving the unconditional
    guarantee ``f_lower <= min f <= f_upper``:

    * ``grad`` -- enables the monotonicity test and the mean-value lower bound;
    * ``hess`` -- adds the **second-order (Taylor) lower bound** (quadratically
      convergent on refined boxes) and, unless ``use_newton`` is ``False``, the
      **interval-Newton / Krawczyk contractor**, which contracts a strictly
      interior sub-box to the part that can still hold a stationary point and
      discards it outright when none can exist there.  The contractor is applied
      only to full-dimensional boxes strictly inside the domain (see
      :func:`_interior_full_dim`), so a boundary minimizer is never lost.
    * ``seeds`` -- optional concrete warm-start point(s) (each clamped into the
      box) used to initialise the incumbent ``f_upper``.  A cheap upper-bound
      accelerator: a good incumbent prunes sub-boxes sooner, so the certified gap
      closes in fewer boxes.  Because evaluating a feasible point can only *lower*
      ``f_upper`` and never touches the sound lower bound, seeds cannot change the
      certified enclosure -- correctness never depends on them.  A closed-form
      gradient-descent seed from :mod:`omnibias.verify.torch.warm_start` /
      :mod:`omnibias.verify.jax.warm_start` is the intended source.

    Returns a :class:`GlobalMinResult` whose ``[f_lower, f_upper]`` always encloses
    the true global minimum; ``converged`` reports whether the certified gap reached
    ``tol`` within the ``max_boxes`` budget.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_boxes < 1:
        raise ValueError(f"max_boxes must be >= 1, got {max_boxes}")
    domain = _as_box(box)
    box0 = _monotone_reduce(domain, grad)
    contract = use_newton and hess is not None

    counter = itertools.count()
    root_lo = _enclose(f, box0, grad, hess).lo
    x_star = _center(box0)
    f_upper = Interval.from_value(f(_point_box(x_star))).hi
    # Warm-start incumbents: evaluating any feasible point only lowers f_upper, so
    # seeds tighten the incumbent (fewer boxes) without ever affecting the sound
    # lower bound -- the certified enclosure never depends on them.
    for seed in seeds or ():
        pt = _clamp_into(seed, domain)
        fu = Interval.from_value(f(_point_box(pt))).hi
        if fu < f_upper:
            f_upper, x_star = fu, pt
    # heap entries: (lower_bound, tie, box) -- pop the box owning the global min.
    heap: list[tuple[float, int, Box]] = [(root_lo, next(counter), box0)]
    explored = 1

    while heap and explored < max_boxes:
        best_lo = heap[0][0]
        if f_upper - best_lo <= tol:
            break  # certified gap reached
        _, _, parent = heapq.heappop(heap)
        if parent[_widest_axis(parent)].width <= min_width:
            heapq.heappush(heap, (best_lo, next(counter), parent))  # cannot refine further
            break
        axis = _widest_axis(parent)
        for child in _split(parent, axis):
            child = _monotone_reduce(child, grad)
            if contract and _interior_full_dim(child, domain):
                assert grad is not None and hess is not None
                contracted = krawczyk_contract(grad, hess, child)
                if contracted is None:
                    explored += 1
                    continue  # certified: no stationary point in this interior box
                child = contracted
            enc = _enclose(f, child, grad, hess)
            explored += 1
            center = _center(child)
            fu = Interval.from_value(f(_point_box(center))).hi
            if fu < f_upper:
                f_upper, x_star = fu, center
            if enc.lo <= f_upper:  # else prune: cannot beat the incumbent (sound)
                heapq.heappush(heap, (enc.lo, next(counter), child))

    f_lower = heap[0][0] if heap else f_upper
    return GlobalMinResult(
        x=x_star,
        f_upper=f_upper,
        f_lower=f_lower,
        tol=tol,
        boxes_explored=explored,
        boxes_remaining=len(heap),
    )


def certify_strict_local_min(
    hessian: HessianFn,
    box: Sequence[object],
) -> bool:
    r"""Certify ``f`` is strictly convex on ``box`` (Hessian positive definite there).

    Uses the interval ``LDL^T`` inertia (:func:`omnibias.core.verified.eig_operator.
    is_positive_definite`): ``True`` guarantees *every* point matrix in the enclosed
    Hessian is positive definite, so any interior stationary point of ``f`` in ``box``
    is a strict local minimizer (and the unique one).  Pair it with a
    :func:`certified_minimize` gap and a ``0 in grad(box)`` check to upgrade the
    incumbent to a certified strict global-in-``box`` minimizer.
    """
    bx = _as_box(box)
    h_like = hessian(bx)
    h = [[Interval.from_value(hij) for hij in row] for row in h_like]
    return is_positive_definite(h)


__all__ = [
    "GlobalMinResult",
    "GradFn",
    "HessianFn",
    "ObjectiveFn",
    "certified_minimize",
    "certify_strict_local_min",
]
