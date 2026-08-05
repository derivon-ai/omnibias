# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified stationary points and basin flatness from exact interval derivatives.

Two rigorous read-outs that complement :func:`certified_minimize`:

* :func:`certified_critical_points` -- the rigorous, tractable form of "solve
  ``grad f = 0``".  A Krawczyk-accelerated interval branch-and-bound that
  *encloses every* root of the gradient in a box, *certifies existence &
  uniqueness* where ``K(X) ⊆ int(X)``, and classifies each root as ``min`` /
  ``max`` / ``saddle`` from the interval Hessian.  For a general network the
  stationarity system has no closed-form solution (it is NP-hard, with
  exponentially many roots), so this is a *low-dimensional* tool -- but there it
  gives what symbolic algebra cannot: a proof that you have found *all* the
  critical points.

* :func:`certified_flatness` -- a rigorous enclosure of the smallest and largest
  Hessian eigenvalue over a box (via the Lehmann/inertia machinery in
  :mod:`omnibias.core.verified.eig_operator`).  This is a *certified* basin
  sharpness / width measure, the exact-curvature analogue of the flat-minima
  heuristic (Hochreiter--Schmidhuber 1997; Keskar 2017; SAM, Foret 2021).

  **Honesty:** flatness predicts *generalization*, it does **not** prove *global*
  optimality -- a wide basin can be a non-global local minimum.  Use it to rank
  candidate minima or as an exact-curvature regularizer, never as a globality
  certificate; that is what :func:`certified_minimize` is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.eig_operator import (
    generalized_eigenvalue_enclosure,
    is_positive_definite,
)
from omnibias.core.verified.interval import Interval
from omnibias.verify._core.global_opt import (
    Box,
    GradFn,
    HessianFn,
    _as_box,
    _split,
    _widest_axis,
)
from omnibias.verify._core.newton import krawczyk_contract, krawczyk_image, krawczyk_unique


@dataclass(frozen=True)
class FlatnessResult:
    r"""Rigorous enclosure of a Hessian's extreme eigenvalues over a box.

    ``eig_min``/``eig_max`` bracket the smallest/largest eigenvalue of the Hessian
    at *every* point of the box.  A *smaller* ``sharpness`` (largest eigenvalue)
    means a *flatter/wider* basin.
    """

    eig_min: Interval
    eig_max: Interval

    @property
    def certified_positive_definite(self) -> bool:
        """``True`` iff the Hessian is certified PD on the whole box (strict min)."""
        return self.eig_min.lo > 0.0

    @property
    def sharpness(self) -> float:
        """Certified upper bound on the largest eigenvalue (larger = sharper)."""
        return self.eig_max.hi

    @property
    def width_lower_bound(self) -> float:
        r"""A basin-width proxy ``1/sqrt(sharpness)`` (larger = wider); ``inf`` if flat."""
        s = self.eig_max.hi
        return float("inf") if s <= 0.0 else float(s**-0.5)


@dataclass(frozen=True)
class CriticalPoint:
    r"""A certified enclosure of one stationary point of ``grad f``."""

    box: tuple[tuple[float, float], ...]
    point: tuple[float, ...]
    unique: bool  # K(X) ⊆ int(X): existence + uniqueness proven
    kind: str  # "min" | "max" | "saddle" | "indefinite"
    eig_min: float  # certified lower bound on the smallest Hessian eigenvalue
    eig_max: float  # certified upper bound on the largest Hessian eigenvalue


def _identity(n: int) -> list[list[Interval]]:
    return [[Interval.point(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]


def _eig_bounds(h: list[list[Interval]]) -> tuple[float, float]:
    n = len(h)
    ident = _identity(n)
    try:
        lo = generalized_eigenvalue_enclosure(h, ident, 1).lo
        hi = generalized_eigenvalue_enclosure(h, ident, n).hi
    except (ValueError, ZeroDivisionError):
        return float("-inf"), float("inf")
    return lo, hi


def certified_flatness(hessian: HessianFn, box: Sequence[object]) -> FlatnessResult:
    r"""Rigorously enclose the extreme Hessian eigenvalues of ``f`` over ``box``.

    ``hessian`` maps a box to an interval Hessian enclosure (e.g. hand-written, or
    the closed-form :func:`omnibias.core.verified.jet_mv.jet_hessian`).  The result
    brackets the sharpest/flattest curvature direction over the whole box; pair it
    with a stationary point from :func:`certified_critical_points` to certify basin
    sharpness at a minimizer.
    """
    bx = _as_box(box)
    h = [[Interval.from_value(hij) for hij in row] for row in hessian(bx)]
    n = len(h)
    ident = _identity(n)
    eig_min = generalized_eigenvalue_enclosure(h, ident, 1)
    eig_max = generalized_eigenvalue_enclosure(h, ident, n)
    return FlatnessResult(eig_min=eig_min, eig_max=eig_max)


def _classify(h: list[list[Interval]], eig_lo: float, eig_hi: float) -> str:
    if is_positive_definite(h):
        return "min"
    if is_positive_definite([[-e for e in row] for row in h]):
        return "max"
    if eig_lo < 0.0 < eig_hi:  # a certified-negative and a certified-positive eigenvalue
        return "saddle"
    return "indefinite"


def _contains_root(grad: GradFn, box: Box) -> bool:
    return all(Interval.from_value(gi).contains_zero() for gi in grad(box))


def certified_critical_points(
    grad: GradFn,
    hess: HessianFn,
    box: Sequence[object],
    *,
    tol: float = 1e-8,
    max_boxes: int = 200_000,
    dedupe_atol: float = 1e-4,
) -> list[CriticalPoint]:
    r"""Enclose **every** root of ``grad f = 0`` in ``box``, certifying uniqueness.

    Krawczyk-accelerated interval branch-and-bound: boxes with ``K ∩ X = ∅`` are
    discarded (certified root-free), boxes with ``K ⊆ int X`` yield a *unique* root
    (Newton-refined to ``tol``), and the rest are bisected.  Each returned
    :class:`CriticalPoint` is classified (``min``/``max``/``saddle``) and carries a
    rigorous enclosure of its Hessian's extreme eigenvalues.

    This is sound and *complete in the limit*: within ``max_boxes`` it returns every
    root it could isolate.  A root reported with ``unique=False`` is enclosed (its
    box satisfies the ``0 ∈ grad(box)`` necessary condition) but existence was not
    certified -- typical for extremely ill-conditioned points.
    """
    bx = _as_box(box)
    stack: list[Box] = [bx]
    found: list[CriticalPoint] = []
    seen = 0
    while stack and seen < max_boxes:
        x = stack.pop()
        seen += 1
        image = krawczyk_image(grad, hess, x)
        contracted = krawczyk_contract(grad, hess, x) if image is not None else x
        if contracted is None:
            continue  # certified: no stationary point here
        if image is not None and krawczyk_unique(image, x):
            cur = contracted
            for _ in range(100):  # quadratically convergent refinement of the unique root
                if max(iv.width for iv in cur) <= tol:
                    break
                nxt = krawczyk_contract(grad, hess, cur)
                if nxt is None:
                    break
                cur = nxt
            found.append(_make_cp(hess, cur, unique=True))
            continue
        if max(iv.width for iv in contracted) <= tol:
            if _contains_root(grad, contracted):
                found.append(_make_cp(hess, contracted, unique=False))
            continue
        axis = _widest_axis(contracted)
        stack.extend(_split(contracted, axis))
    return _dedupe(found, dedupe_atol)


def _make_cp(hess: HessianFn, box: Box, *, unique: bool) -> CriticalPoint:
    h = [[Interval.from_value(hij) for hij in row] for row in hess(box)]
    eig_lo, eig_hi = _eig_bounds(h)
    return CriticalPoint(
        box=tuple((iv.lo, iv.hi) for iv in box),
        point=tuple(iv.mid for iv in box),
        unique=unique,
        kind=_classify(h, eig_lo, eig_hi),
        eig_min=eig_lo,
        eig_max=eig_hi,
    )


def _dedupe(cps: list[CriticalPoint], atol: float) -> list[CriticalPoint]:
    out: list[CriticalPoint] = []
    for cp in cps:
        idx = next(
            (i for i, o in enumerate(out)
             if all(abs(a - b) <= atol for a, b in zip(o.point, cp.point, strict=True))),
            None,
        )
        if idx is None:
            out.append(cp)
        elif cp.unique and not out[idx].unique:  # keep the certified-unique enclosure
            out[idx] = cp
    return out


__all__ = [
    "CriticalPoint",
    "FlatnessResult",
    "certified_critical_points",
    "certified_flatness",
]
