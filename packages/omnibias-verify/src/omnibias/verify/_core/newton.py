# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Interval-Newton (Krawczyk) primitives for the certified optimizer.

The Krawczyk operator is the rigorous, tractable form of "solve ``grad f = 0``".
For a box ``X`` with midpoint ``c`` and any real preconditioner ``Y`` (soundness
holds for *any* ``Y``; contraction is best when ``Y ~ inv(mid J)``):

    K(X) = c - Y·g(c) + (I - Y·J(X))·(X - c)

with ``g`` the gradient map (root system) and ``J`` its Jacobian -- i.e. the
Hessian of the objective.  Two theorems make it useful:

* **enclosure** -- every root of ``g`` in ``X`` lies in ``K(X)``, so
  ``roots(g, X) ⊆ K(X) ∩ X``.  Hence ``K(X) ∩ X = ∅`` *certifies there is no
  stationary point in* ``X``;
* **existence + uniqueness** -- if ``K(X) ⊆ int(X)`` there is exactly one root in
  ``X`` (Krawczyk / Kantorovich).

These power both the branch-and-bound contractor in
:mod:`omnibias.verify._core.global_opt` and the certified critical-point
enumeration in :mod:`omnibias.verify._core.stationary`.  Pure Python on the
verified :class:`Interval`; no numpy, so the preconditioner is a small
Gauss--Jordan inverse (dimensions here are low).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from omnibias.core.verified.interval import Interval, IntervalLike

Box = tuple[Interval, ...]
GradFn = Callable[[tuple[Interval, ...]], Sequence[IntervalLike]]
HessianFn = Callable[[tuple[Interval, ...]], Sequence[Sequence[IntervalLike]]]


def float_inverse(a: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Gauss--Jordan inverse of a small real matrix; ``None`` if (near-)singular.

    Used only as a *preconditioner*; any returned matrix keeps the Krawczyk
    operator rigorous, so an approximate float inverse is fine.
    """
    n = len(a)
    aug = [[float(a[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-300:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv_pivot = 1.0 / aug[col][col]
        for j in range(2 * n):
            aug[col][j] *= inv_pivot
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                for j in range(2 * n):
                    aug[r][j] -= factor * aug[col][j]
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def _dot_real_interval(reals: Sequence[float], ivs: Sequence[Interval]) -> Interval:
    acc = Interval.point(0.0)
    for r, iv in zip(reals, ivs, strict=True):
        acc = acc + Interval.point(r) * iv
    return acc


def krawczyk_image(grad: GradFn, hess: HessianFn, box: Box) -> Box | None:
    r"""The Krawczyk image ``K(X)`` of the root map ``grad`` over ``box``.

    Returns ``None`` when the midpoint Hessian is numerically singular (no
    preconditioner -> the caller should fall back to bisection).
    """
    n = len(box)
    center = tuple(iv.mid for iv in box)
    cbox = tuple(Interval.point(ci) for ci in center)
    gc = [Interval.from_value(gi) for gi in grad(cbox)]
    jac = [[Interval.from_value(hij) for hij in row] for row in hess(box)]
    y = float_inverse([[jac[i][j].mid for j in range(n)] for i in range(n)])
    if y is None:
        return None
    xc = [box[i] - Interval.point(center[i]) for i in range(n)]
    image: list[Interval] = []
    for i in range(n):
        acc = Interval.point(center[i]) - _dot_real_interval(y[i], gc)
        for j in range(n):
            yj = _dot_real_interval(y[i], [jac[k][j] for k in range(n)])  # (Y J)_ij
            m_ij = (Interval.point(1.0) if i == j else Interval.point(0.0)) - yj
            acc = acc + m_ij * xc[j]
        image.append(acc)
    return tuple(image)


def krawczyk_contract(grad: GradFn, hess: HessianFn, box: Box) -> Box | None:
    r"""Contract ``box`` to ``box ∩ K(box)``.

    * returns a (possibly much smaller) box that still contains **every** root of
      ``grad`` that was in ``box``;
    * returns ``None`` when ``K(box) ∩ box = ∅`` -- a *certificate* that ``box``
      holds no stationary point (the caller may discard it);
    * returns ``box`` unchanged when no preconditioner is available.
    """
    image = krawczyk_image(grad, hess, box)
    if image is None:
        return box
    out: list[Interval] = []
    for i in range(len(box)):
        lo = max(image[i].lo, box[i].lo)
        hi = min(image[i].hi, box[i].hi)
        if lo > hi:
            return None
        out.append(Interval(lo, hi))
    return tuple(out)


def krawczyk_unique(image: Box, box: Box) -> bool:
    """``True`` when ``K(box) ⊆ int(box)`` -- existence & uniqueness of a root."""
    return all(box[i].lo < image[i].lo and image[i].hi < box[i].hi for i in range(len(box)))


__all__ = [
    "float_inverse",
    "krawczyk_contract",
    "krawczyk_image",
    "krawczyk_unique",
]
