# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Winding / argument collapse: a certified zero count on a circle.

The surviving object is a **winding number** (an integer), not a
derivative, not a 0/1 step, and not a scalar point-plus-proof. The
moving parameter is the argument along a circular contour. Collapse
fires when the enclosed ``Δarg / 2π`` contains exactly one integer.

A box that contains the origin, or that crosses the principal branch
cut, is ``Inconclusive``. This is not a blow-up proof and not a
continuum PDE claim. Do not conflate with founding bias collapse
(``delta -> 0``), temperature collapse (``beta -> inf``), or Enclosure
Collapse of a value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, atan_iv, cos_iv, sin_iv

WINDING_SPEC = CollapseSpec(
    name="winding",
    parameter="argument",
    limit="2πZ",
    surviving_object="winding_number",
    failure="Inconclusive",
    home="omnibias.core.collapse.winding",
    register="verified",
)

CoeffC = int | float | complex


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="winding")
    payload["winding_collapse"] = True
    payload["continuum_parent_inferred"] = False
    return payload


def cis_iv(theta: Interval) -> ComplexInterval:
    """Sound rectangular enclosure of ``exp(i theta)``."""

    return ComplexInterval(cos_iv(theta), sin_iv(theta))


def contains_origin(z: ComplexInterval) -> bool:
    return z.re.contains_zero() and z.im.contains_zero()


def crosses_branch_cut(z: ComplexInterval) -> bool:
    """Whether the rectangle meets the principal cut ``(-inf, 0]``."""

    if contains_origin(z):
        return True
    return z.re.hi <= 0.0 and z.im.contains_zero()


def _atan2_point(y: float, x: float) -> Interval:
    if x == 0.0 and y == 0.0:
        raise ValueError("atan2 is undefined at the origin")
    yi = Interval.point(y)
    xi = Interval.point(x)
    half = PI_IV / Interval.from_rational(2)
    if x > 0.0:
        return atan_iv(yi / xi)
    if x < 0.0 and y >= 0.0:
        return atan_iv(yi / xi) + PI_IV
    if x < 0.0 and y < 0.0:
        return atan_iv(yi / xi) - PI_IV
    if y > 0.0:
        return half
    return -half


def arg_iv(z: ComplexInterval) -> Interval | None:
    """Principal-argument enclosure, or ``None`` if it is not well-defined."""

    if contains_origin(z) or crosses_branch_cut(z):
        return None
    corners = (
        (z.re.lo, z.im.lo),
        (z.re.lo, z.im.hi),
        (z.re.hi, z.im.lo),
        (z.re.hi, z.im.hi),
    )
    pieces = [_atan2_point(im, re) for re, im in corners]
    return Interval(min(p.lo for p in pieces), max(p.hi for p in pieces))


def horner_complex(
    coeffs: Sequence[CoeffC],
    z: ComplexInterval,
) -> ComplexInterval:
    if not coeffs:
        return ComplexInterval.zero()
    acc = ComplexInterval.from_value(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        acc = acc * z + ComplexInterval.from_value(coeff)
    return acc


def integers_in(box: Interval) -> tuple[int, ...]:
    """Integers that the enclosure may contain (no false negatives)."""

    start = math.ceil(box.lo)
    end = math.floor(box.hi)
    if start > end:
        return ()
    return tuple(range(start, end + 1))


def winding_enclosure(
    coeffs: Sequence[CoeffC],
    center: complex,
    radius: float,
    *,
    segments: int = 64,
) -> Interval | None:
    """Sound enclosure of the winding number, or ``None`` if inconclusive."""

    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if segments < 4:
        raise ValueError(f"segments must be >= 4, got {segments}")
    if not coeffs:
        return None
    two_pi = PI_IV * Interval.from_rational(2)
    origin = ComplexInterval.point(center)
    acc = Interval.point(0.0)
    for index in range(segments):
        t0 = two_pi * Interval.from_rational(Fraction(index, segments))
        t1 = two_pi * Interval.from_rational(Fraction(index + 1, segments))
        arc = Interval(t0.lo, t1.hi)
        z_arc = origin + cis_iv(arc) * Interval.point(radius)
        image = horner_complex(coeffs, z_arc)
        if contains_origin(image):
            return None
        z0 = origin + cis_iv(t0) * Interval.point(radius)
        z1 = origin + cis_iv(t1) * Interval.point(radius)
        p0 = horner_complex(coeffs, z0)
        p1 = horner_complex(coeffs, z1)
        if contains_origin(p0) or contains_origin(p1):
            return None
        delta = arg_iv(p1 / p0)
        if delta is None:
            return None
        acc = acc + delta
    return acc / two_pi


def _blocked(detail: str, residual: Interval | None) -> ObligationVerdict:
    outcome = CollapseOutcome(
        status="inconclusive",
        spec_name="winding",
        surviving=None,
        residual=residual,
        detail=detail,
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=True,
        evaluated=0,
        complete=False,
        detail=detail,
    )


def winding_collapse(
    coeffs: Sequence[CoeffC],
    center: complex = 0j,
    radius: float = 1.0,
    *,
    expected: int | None = None,
    segments: int = 32,
    max_segments: int = 256,
) -> ObligationVerdict:
    """Collapse when the winding enclosure contains exactly one integer."""

    if max_segments < segments:
        raise ValueError("max_segments must be >= segments")
    current = segments
    last: Interval | None = None
    while current <= max_segments:
        last = winding_enclosure(
            coeffs, center, radius, segments=current
        )
        if last is not None:
            hits = integers_in(last)
            if len(hits) == 1:
                n = hits[0]
                if expected is None or expected == n:
                    outcome = CollapseOutcome(
                        status="collapsed",
                        spec_name="winding",
                        surviving=n,
                        residual=last,
                        detail=f"winding enclosure collapsed onto {n}",
                        honesty=_honesty(),
                    )
                    return ObligationVerdict(
                        status="PROVED",
                        outcome=outcome,
                        existential=True,
                        evaluated=current,
                        complete=True,
                        detail=outcome.detail,
                    )
                outcome = CollapseOutcome(
                    status="excluded",
                    spec_name="winding",
                    surviving=n,
                    residual=last,
                    detail=f"winding is {n}, not expected {expected}",
                    honesty=_honesty(),
                )
                return ObligationVerdict(
                    status="DISPROVED",
                    outcome=outcome,
                    existential=True,
                    evaluated=current,
                    complete=True,
                    detail=outcome.detail,
                )
        current *= 2
    return _blocked(
        "winding enclosure did not isolate a unique integer",
        last,
    )


def _reseed() -> None:
    try:
        get_collapse("winding")
    except KeyError:
        register_collapse(WINDING_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "WINDING_SPEC",
    "arg_iv",
    "cis_iv",
    "contains_origin",
    "crosses_branch_cut",
    "horner_complex",
    "integers_in",
    "winding_collapse",
    "winding_enclosure",
]
