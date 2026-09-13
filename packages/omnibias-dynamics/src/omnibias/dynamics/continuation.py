# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform finite-dimensional continuation certificates.

Callbacks must provide sound enclosures of the named C1 residual and its
derivatives. A segment quantifies over an entire parameter interval. Event
certificates additionally require named nondegeneracy enclosures over the
augmented root box; point estimates and sampled residuals are not accepted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    IntervalJac,
    IntervalMap,
    KrawczykCertificate,
    krawczyk_certificate,
)

FamilyMap = Callable[[list[Interval], Interval], Sequence[Interval]]
FamilyJacobian = Callable[[list[Interval], Interval], Sequence[Sequence[Interval]]]
ConditionBounds = Callable[[list[Interval]], Mapping[str, Interval]]


def _inverse(a: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(a)
    rows = [list(row) + [float(i == j) for j in range(n)] for i, row in enumerate(a)]
    if any(len(row) != 2 * n for row in rows):
        raise ValueError("Jacobian must be square")
    for i in range(n):
        k = max(range(i, n), key=lambda j: abs(rows[j][i]))
        rows[i], rows[k] = rows[k], rows[i]
        pivot = rows[i][i]
        if not math.isfinite(pivot) or pivot == 0:
            raise ValueError("singular midpoint Jacobian")
        rows[i] = [v / pivot for v in rows[i]]
        for j in range(n):
            if j != i:
                factor = rows[j][i]
                rows[j] = [v - factor * w for v, w in zip(rows[j], rows[i], strict=True)]
    return [row[n:] for row in rows]


@dataclass(frozen=True)
class SegmentCertificate:
    parameter: Interval
    enclosure: tuple[Interval, ...]
    root: KrawczykCertificate | None
    status: str

    @property
    def certified(self) -> bool:
        return self.root is not None


def certify_segment(
    value: FamilyMap,
    jacobian: FamilyJacobian,
    parameter: Interval,
    center: Sequence[float],
    *,
    slope: Sequence[float] | None = None,
    radius: float = 1e-4,
) -> SegmentCertificate:
    """Prove a unique correction w(s) for x(s)=center+slope*(s-mid)+w.

    Uniform contraction certifies a continuous solution for each s in the
    parameter interval, under the supplied C1 enclosure contract. No PDE
    discretization or tail theorem is inferred from this finite system.
    """
    if not center or radius <= 0 or not math.isfinite(radius):
        raise ValueError("nonempty center and positive finite radius required")
    if not all(math.isfinite(v) for v in (*center, parameter.lo, parameter.hi)):
        raise ValueError("finite center and parameter interval required")
    tangent = [0.0] * len(center) if slope is None else list(slope)
    if len(tangent) != len(center) or not all(math.isfinite(v) for v in tangent):
        raise ValueError("slope must match center and be finite")
    path = [
        Interval.point(c) + v * (parameter - parameter.mid)
        for c, v in zip(center, tangent, strict=True)
    ]
    box = tuple(p + Interval(-radius, radius) for p in path)

    def f(w: list[Interval]) -> Sequence[Interval]:
        return value([p + x for p, x in zip(path, w, strict=True)], parameter)

    def j(w: list[Interval]) -> Sequence[Sequence[Interval]]:
        return jacobian([p + x for p, x in zip(path, w, strict=True)], parameter)

    try:
        inv = _inverse(
            [
                [v.mid for v in row]
                for row in jacobian(
                    [Interval.point(c) for c in center], Interval.point(parameter.mid)
                )
            ]
        )
        cert = krawczyk_certificate(f, j, [0.0] * len(center), inv, radius)
    except (ValueError, ZeroDivisionError, OverflowError):
        cert = None
    return SegmentCertificate(
        parameter, box, cert, "certified_uniform_segment" if cert else "inconclusive"
    )


@dataclass(frozen=True)
class EventCertificate:
    kind: str
    root: KrawczykCertificate | None
    conditions: Mapping[str, Interval]
    status: str

    @property
    def certified(self) -> bool:
        return self.root is not None and self.status == "certified_event"


def certify_event(
    kind: str,
    augmented_value: IntervalMap,
    augmented_jacobian: IntervalJac,
    center: Sequence[float],
    conditions: ConditionBounds,
    *,
    radius: float = 1e-5,
) -> EventCertificate:
    """Validate a fold/Hopf augmented root and its nondegeneracy bounds.

    The caller's condition provider encloses the defining normal-form
    quantities of this augmented system on the full root box. Required keys:
    fold: transversality, quadratic, complement_gap; Hopf: transversality,
    lyapunov, frequency, complement_gap, resonance_gap. Gaps/frequency must
    be strictly positive; other quantities must exclude zero. This function
    does not infer those mathematical identities from arbitrary callbacks.
    """
    required = {
        "fold": ("transversality", "quadratic", "complement_gap"),
        "hopf": ("transversality", "lyapunov", "frequency", "complement_gap", "resonance_gap"),
    }
    if kind not in required or not center or radius <= 0 or not math.isfinite(radius):
        raise ValueError("valid fold/Hopf kind, center and finite positive radius required")
    points = [Interval.point(v) for v in center]
    box = [Interval(v - radius, v + radius) for v in center]
    bounds = dict(conditions(box))
    if any(k not in bounds or not isinstance(bounds[k], Interval) for k in required[kind]):
        raise ValueError("missing sound interval nondegeneracy conditions")
    passed = all(
        bounds[k].lo > 0
        if k.endswith("gap") or k == "frequency"
        else (bounds[k].lo > 0 or bounds[k].hi < 0)
        for k in required[kind]
    )
    try:
        inv = _inverse(
            [[Interval.from_value(v).mid for v in row] for row in augmented_jacobian(points)]
        )
        root = krawczyk_certificate(augmented_value, augmented_jacobian, list(center), inv, radius)
    except (ValueError, ZeroDivisionError, OverflowError):
        root = None
    return EventCertificate(
        kind, root, bounds, "certified_event" if root and passed else "inconclusive"
    )


def certify_join(
    value: FamilyMap,
    jacobian: FamilyJacobian,
    left: SegmentCertificate,
    right: SegmentCertificate,
) -> bool:
    """Prove adjacent certified segments coincide at a common endpoint.

    A fresh uniqueness box contains both entire segment enclosures at that
    endpoint; mere intersection of two numerical boxes would not suffice.
    """
    if not left.certified or not right.certified or left.parameter.hi != right.parameter.lo:
        return False
    if len(left.enclosure) != len(right.enclosure):
        return False
    hull = [
        Interval(min(a.lo, b.lo), max(a.hi, b.hi))
        for a, b in zip(left.enclosure, right.enclosure, strict=True)
    ]
    center = [v.mid for v in hull]
    radius = math.nextafter(
        max(max(c - v.lo, v.hi - c) for c, v in zip(center, hull, strict=True)), math.inf
    )
    return certify_segment(
        value, jacobian, Interval.point(left.parameter.hi), center, radius=radius
    ).certified


__all__ = [
    "ConditionBounds",
    "EventCertificate",
    "FamilyJacobian",
    "FamilyMap",
    "SegmentCertificate",
    "certify_event",
    "certify_join",
    "certify_segment",
]
