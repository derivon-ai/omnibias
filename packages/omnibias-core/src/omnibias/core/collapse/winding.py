# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Winding / argument collapse: a certified zero count on a closed contour.

The surviving object is a **winding number** (an integer), not a
derivative, not a 0/1 step, and not a scalar point-plus-proof. The
moving parameter is the argument along a circular or rectangular contour. Collapse
fires when the enclosed ``Δarg / 2π`` contains exactly one integer.

A box that contains the origin, or that crosses the principal branch
cut, is ``Inconclusive``. This is not a blow-up proof and not a
continuum PDE claim. Do not conflate with founding bias collapse
(``delta -> 0``), temperature collapse (``beta -> inf``), or Enclosure
Collapse of a value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Literal

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
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

CoeffC = ComplexLike
Contour = Literal["circle", "rectangle"]
ComplexEnclosureFn = Callable[[ComplexInterval], ComplexInterval]


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


def _circle_segments(
    center: complex, radius: float, segments: int
) -> list[tuple[ComplexInterval, ComplexInterval, ComplexInterval]]:
    """Return ``(image domain, start, end)`` parameter segments on a circle."""
    two_pi = PI_IV * Interval.from_rational(2)
    origin = ComplexInterval.point(center)
    result: list[tuple[ComplexInterval, ComplexInterval, ComplexInterval]] = []
    for index in range(segments):
        t0 = two_pi * Interval.from_rational(Fraction(index, segments))
        t1 = two_pi * Interval.from_rational(Fraction(index + 1, segments))
        arc = Interval(t0.lo, t1.hi)
        result.append(
            (
                origin + cis_iv(arc) * Interval.point(radius),
                origin + cis_iv(t0) * Interval.point(radius),
                origin + cis_iv(t1) * Interval.point(radius),
            )
        )
    return result


def _rectangle_segments(
    center: complex,
    half_width: float,
    half_height: float,
    segments: int,
) -> list[tuple[ComplexInterval, ComplexInterval, ComplexInterval]]:
    """Return counter-clockwise line segments on an axis-aligned rectangle."""
    if segments % 4:
        raise ValueError("rectangle contours require a segment count divisible by 4")
    per_edge = segments // 4
    vertices = (
        center + complex(-half_width, -half_height),
        center + complex(half_width, -half_height),
        center + complex(half_width, half_height),
        center + complex(-half_width, half_height),
    )
    result: list[tuple[ComplexInterval, ComplexInterval, ComplexInterval]] = []
    for edge, start in enumerate(vertices):
        end = vertices[(edge + 1) % len(vertices)]
        direction = ComplexInterval.point(end - start)
        origin = ComplexInterval.point(start)
        for index in range(per_edge):
            t0 = Interval.from_rational(Fraction(index, per_edge))
            t1 = Interval.from_rational(Fraction(index + 1, per_edge))
            segment = Interval(t0.lo, t1.hi)
            result.append(
                (
                    origin + direction * ComplexInterval.from_value(segment),
                    origin + direction * ComplexInterval.from_value(t0),
                    origin + direction * ComplexInterval.from_value(t1),
                )
            )
    return result


def contour_parameter_segments(
    center: complex,
    radius: float,
    *,
    segments: int = 64,
    contour: str = "circle",
    half_width: float | None = None,
    half_height: float | None = None,
) -> tuple[ComplexInterval, ...]:
    """Return sound complex boxes covering a circular or rectangular contour.

    This is the contour domain used by :func:`winding_enclosure`.  It is useful
    when a caller must independently prove that an analytic function has no
    zero on every contour segment before interpreting its winding number.
    """
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError(f"radius must be positive, got {radius!r}")
    if segments < 4:
        raise ValueError(f"segments must be >= 4, got {segments}")
    if contour == "circle":
        path = _circle_segments(center, radius, segments)
    elif contour == "rectangle":
        width = radius if half_width is None else float(half_width)
        height = radius if half_height is None else float(half_height)
        if width <= 0.0 or not math.isfinite(width):
            raise ValueError(f"rectangle half_width must be positive, got {width!r}")
        if height <= 0.0 or not math.isfinite(height):
            raise ValueError(f"rectangle half_height must be positive, got {height!r}")
        path = _rectangle_segments(center, width, height, segments)
    else:
        raise ValueError("contour must be 'circle' or 'rectangle'")
    return tuple(domain for domain, _, _ in path)


def winding_enclosure(
    coeffs: Sequence[CoeffC],
    center: complex,
    radius: float,
    *,
    segments: int = 64,
    contour: str = "circle",
    half_width: float | None = None,
    half_height: float | None = None,
) -> Interval | None:
    """Sound contour-winding enclosure, or ``None`` if it is inconclusive.

    ``contour="circle"`` preserves the historical circle centred at ``center``
    with the supplied ``radius``.  ``contour="rectangle"`` traces an
    axis-aligned rectangle counter-clockwise; its half-width and half-height
    default to ``radius`` so existing radius-only calls can be switched to a
    square contour without changing scale.
    """

    if not coeffs:
        return None
    path = contour_parameter_segments(
        center,
        radius,
        segments=segments,
        contour=contour,
        half_width=half_width,
        half_height=half_height,
    )

    two_pi = PI_IV * Interval.from_rational(2)
    acc = Interval.point(0.0)
    if contour == "circle":
        endpoints = _circle_segments(center, radius, segments)
    else:
        width = radius if half_width is None else float(half_width)
        height = radius if half_height is None else float(half_height)
        endpoints = _rectangle_segments(center, width, height, segments)
    for z_arc, (_, z0, z1) in zip(path, endpoints, strict=True):
        image = horner_complex(coeffs, z_arc)
        if contains_origin(image):
            return None
        p0 = horner_complex(coeffs, z0)
        p1 = horner_complex(coeffs, z1)
        if contains_origin(p0) or contains_origin(p1):
            return None
        delta = arg_iv(p1 / p0)
        if delta is None:
            return None
        acc = acc + delta
    return acc / two_pi


def winding_enclosure_function(
    function: ComplexEnclosureFn,
    center: complex,
    radius: float,
    *,
    segments: int = 64,
    contour: str = "circle",
    half_width: float | None = None,
    half_height: float | None = None,
) -> Interval | None:
    """Enclose the winding of an interval-evaluable function on a contour.

    ``function`` must enclose its analytic target over every supplied
    :class:`ComplexInterval`.  Returning ``None`` means that the image of a
    contour segment could contain zero, or that the principal-argument
    increment was not isolatable; it is never interpreted as a zero count.
    """
    path = contour_parameter_segments(
        center,
        radius,
        segments=segments,
        contour=contour,
        half_width=half_width,
        half_height=half_height,
    )
    if contour == "circle":
        endpoints = _circle_segments(center, radius, segments)
    else:
        width = radius if half_width is None else float(half_width)
        height = radius if half_height is None else float(half_height)
        endpoints = _rectangle_segments(center, width, height, segments)

    two_pi = PI_IV * Interval.from_rational(2)
    acc = Interval.point(0.0)
    for domain, (_, z0, z1) in zip(path, endpoints, strict=True):
        image = function(domain)
        if contains_origin(image):
            return None
        p0 = function(z0)
        p1 = function(z1)
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
    contour: str = "circle",
    half_width: float | None = None,
    half_height: float | None = None,
) -> ObligationVerdict:
    """Collapse when a circle or rectangle winding isolates one integer."""

    if max_segments < segments:
        raise ValueError("max_segments must be >= segments")
    current = segments
    last: Interval | None = None
    while current <= max_segments:
        last = winding_enclosure(
            coeffs,
            center,
            radius,
            segments=current,
            contour=contour,
            half_width=half_width,
            half_height=half_height,
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
    "ComplexEnclosureFn",
    "Contour",
    "WINDING_SPEC",
    "arg_iv",
    "cis_iv",
    "contains_origin",
    "contour_parameter_segments",
    "crosses_branch_cut",
    "horner_complex",
    "integers_in",
    "winding_collapse",
    "winding_enclosure",
    "winding_enclosure_function",
]
