# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous Poincare-section return map.

A **Poincare section** is a hyperplane :math:`\Sigma = \{y : \langle \hat n, y
\rangle = c\}`; the return map sends a point of :math:`\Sigma` to the next point
at which the flow crosses :math:`\Sigma` (in a chosen direction).  Detecting and
localising that crossing *rigorously* is the crux.

The strategy here is sound by construction:

1. Flow the (boxed) initial set with the QR-Lohner step; track the section
   functional ``g(y) = <n, y> - c`` as an interval at each grid time.
2. A crossing is **guaranteed** on a step when ``g`` is strictly one sign at the
   start and strictly the opposite sign at the end (intermediate-value theorem
   applied to *every* trajectory in the bundle), with the sign change matching the
   requested direction.
3. Localise: re-step the crossing interval into sub-steps to bracket the crossing
   time, take the a-priori enclosure of the crossing sub-step (which provably
   contains the whole trajectory segment, hence the crossing point) and *flatten*
   it onto :math:`\Sigma` by solving ``g = 0`` for the dominant coordinate.

The returned :class:`PoincareCrossing` encloses the true crossing point of every
trajectory that started in the initial box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.lohner import JacobianEnclosure, LohnerSet, lohner_step
from omnibias.core.verified.ode import VectorField, _apriori_enclosure


@dataclass(frozen=True)
class PoincareSection:
    r"""The hyperplane ``<normal, y> = offset`` with a crossing ``direction``.

    ``direction`` is ``+1`` to keep only crossings where ``g = <normal,y> - offset``
    increases through zero, ``-1`` for decreasing crossings, ``0`` for either.
    """

    normal: tuple[float, ...]
    offset: float
    direction: int = 1

    def g(self, point: Sequence[Interval]) -> Interval:
        """Section functional ``<normal, y> - offset`` as an interval enclosure."""
        acc = Interval.point(-self.offset)
        for ni, yi in zip(self.normal, point, strict=True):
            acc = acc + Interval.point(ni) * yi
        return acc


@dataclass(frozen=True)
class PoincareCrossing:
    """The rigorous enclosure of a return-map crossing (or a no-crossing report)."""

    crossed: bool
    enclosure: tuple[Interval, ...]
    step_index: int
    time_bracket: tuple[float, float]


def _strict_sign(g: Interval) -> int:
    """``-1`` / ``+1`` when ``g`` is strictly signed, ``0`` when it straddles zero."""
    if g.hi < 0.0:
        return -1
    if g.lo > 0.0:
        return 1
    return 0


def _direction_ok(sign_after: int, direction: int) -> bool:
    """Whether a crossing whose post-sign is ``sign_after`` matches the request."""
    if direction > 0:
        return sign_after > 0
    if direction < 0:
        return sign_after < 0
    return True


def _guaranteed_cross(g_prev: Interval, g_next: Interval, direction: int) -> bool:
    """Whether a transversal crossing is *certain* on the step (matching direction)."""
    up = g_prev.hi < 0.0 and g_next.lo > 0.0
    down = g_prev.lo > 0.0 and g_next.hi < 0.0
    if direction > 0:
        return up
    if direction < 0:
        return down
    return up or down


def _flatten(box: Sequence[Interval], section: PoincareSection) -> tuple[Interval, ...]:
    """Intersect a box with ``{g = 0}`` by solving for the dominant coordinate."""
    p = max(range(len(section.normal)), key=lambda i: abs(section.normal[i]))
    if section.normal[p] == 0.0:
        return tuple(box)
    acc = Interval.point(section.offset)
    for j, nj in enumerate(section.normal):
        if j != p:
            acc = acc - Interval.point(nj) * box[j]
    xp = (acc * Interval.point(section.normal[p]).reciprocal()).intersect(box[p])
    out = list(box)
    out[p] = xp
    return tuple(out)


def _localize(
    field: VectorField,
    jac: JacobianEnclosure,
    section: PoincareSection,
    start: LohnerSet,
    h: float,
    order: int,
    refine: int,
) -> tuple[Interval, ...]:
    """Tighten the crossing enclosure by sub-stepping then flattening onto Sigma."""
    sub = start
    g_prev = section.g(sub.to_box())
    hh = h / refine
    for _ in range(refine):
        nxt = lohner_step(field, jac, sub, hh, order)
        g_next = section.g(nxt.to_box())
        if _guaranteed_cross(g_prev, g_next, section.direction):
            z = _apriori_enclosure(field, sub.to_box(), hh)
            return _flatten(z, section)
        sub, g_prev = nxt, g_next
    z = _apriori_enclosure(field, start.to_box(), h)
    return _flatten(z, section)


def poincare_map(
    field: VectorField,
    jac: JacobianEnclosure,
    section: PoincareSection,
    y0: Sequence[IntervalLike],
    h: float,
    *,
    max_steps: int = 10000,
    order: int = 12,
    skip_initial_steps: int = 1,
    refine: int = 8,
) -> PoincareCrossing:
    r"""Enclose the first transversal return of the flow to ``section``.

    ``skip_initial_steps`` ignores crossings in the first few steps so a start
    point *on* the section is not reported as its own image.  Returns a
    :class:`PoincareCrossing` with ``crossed=False`` if no guaranteed crossing is
    found within ``max_steps``.
    """
    if h <= 0.0:
        raise ValueError("step size h must be positive")
    state = LohnerSet.from_box(y0)
    anchor_state = state
    anchor_t = 0.0
    anchor_sign = _strict_sign(section.g(state.to_box()))
    t = 0.0
    for k in range(max_steps):
        nxt = lohner_step(field, jac, state, h, order)
        t1 = t + h
        sign_next = _strict_sign(section.g(nxt.to_box()))
        if sign_next != 0:
            crossing = anchor_sign != 0 and sign_next == -anchor_sign
            if crossing and k + 1 > skip_initial_steps and _direction_ok(sign_next, section.direction):
                enclosure = _localize(
                    field, jac, section, anchor_state, t1 - anchor_t, order, refine
                )
                return PoincareCrossing(True, enclosure, k, (anchor_t, t1))
            anchor_state, anchor_t, anchor_sign = nxt, t1, sign_next
        state, t = nxt, t1
    return PoincareCrossing(False, tuple(state.to_box()), max_steps, (0.0, t))


__all__ = [
    "PoincareCrossing",
    "PoincareSection",
    "poincare_map",
]
