# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-bundle vocabulary helpers (theory 01-10).

This is a **reformulation**, not a discovery: a tower of numbers is a jet
only when it is holonomic (annihilates the contact ideal). There is no
``omnibias-jetbundle`` package.

A 1-D jet at a point is ``(u, u', ..., u^{(N)})``. The contact residual of
order ``k`` is ``u^{(k)}(x+h) - u^{(k)}(x) - h u^{(k+1)}(x)``, which is
``O(h^2)`` for a genuine prolongation and ``O(h)`` for a corrupted tower.

Founding ``delta -> 0`` produces fiber coordinates. Temperature collapse
(``beta -> inf``) acts on the base stratification, not on the fiber.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Jet1D = Sequence[float]
SectionSampler = Callable[[float], Jet1D]


def contact_residual(
    jet_at_x: Jet1D,
    jet_at_x_plus_h: Jet1D,
    *,
    h: float,
) -> tuple[float, ...]:
    """Residuals of ``du^{(k)} - u^{(k+1)} dx`` for a 1-D jet.

    Should be ``O(h^2)`` for a holonomic jet (Taylor remainder) and
    ``O(h)`` when a coordinate is overwritten.
    """
    if h == 0.0:
        raise ValueError("h must be nonzero")
    a = tuple(float(v) for v in jet_at_x)
    b = tuple(float(v) for v in jet_at_x_plus_h)
    if len(a) != len(b):
        raise ValueError("jets must have equal length")
    if len(a) < 2:
        raise ValueError("contact residual needs order >= 1 (at least two coordinates)")
    return tuple(b[k] - a[k] - h * a[k + 1] for k in range(len(a) - 1))


def max_abs_contact_residual(
    jet_at_x: Jet1D,
    jet_at_x_plus_h: Jet1D,
    *,
    h: float,
) -> float:
    res = contact_residual(jet_at_x, jet_at_x_plus_h, h=h)
    return max(abs(v) for v in res)


def is_holonomic(
    section_sampler: SectionSampler,
    x: float,
    *,
    h: float,
    rtol: float = 1e-6,
) -> bool:
    """Rate test: halve ``h`` and check the residual drops by about ``4x``.

    A holonomic 1-D jet has second-order contact remainder, so
    ``r(h/2) / r(h) ~ 1/4``. Corrupted towers stay near ``1/2``.
    """
    if h == 0.0:
        raise ValueError("h must be nonzero")
    j0 = section_sampler(x)
    r_h = max_abs_contact_residual(j0, section_sampler(x + h), h=h)
    h2 = 0.5 * h
    r_h2 = max_abs_contact_residual(j0, section_sampler(x + h2), h=h2)
    # Machine-zero remainder is holonomic. Do not scale the floor by |jet|:
    # a corrupted first derivative of size O(1) produces r ~ |h|, which must
    # not be swallowed by rtol * max|jet|.
    floor = 1e-14
    if r_h <= floor and r_h2 <= floor:
        return True
    if r_h <= 0.0:
        return r_h2 <= floor
    ratio = r_h2 / r_h
    # Second-order: ~0.25; first-order: ~0.5. Split at ~0.38.
    return ratio <= 0.38


__all__ = [
    "contact_residual",
    "is_holonomic",
    "max_abs_contact_residual",
]
