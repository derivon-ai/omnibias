# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Planted Newton-Kantorovich / Krawczyk existence certificates.

A ``nk_existence`` payload names the locked quadratic plant ``x² - 2`` and a
route (``radii`` or ``krawczyk``). The Mathlib bridge re-derives the rational
contraction facts and emits Lean that applies the corresponding
``OmnibiasAnalytic.Check`` unique-zero theorem.

This is a unique real root of a named quadratic on a compact interval. It is
not a continuum PDE claim, a Navier-Stokes regularity theorem, a mass gap,
or any asymptotic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_NK_FAMILIES: tuple[str, ...] = ("quadratic",)
LEGAL_NK_ROUTES: tuple[str, ...] = ("radii", "krawczyk")

NKRoute = Literal["radii", "krawczyk"]

# Locked plant: f(x) = x² - 2, c = 3/2, A = 1/3, r = 1/4.
PLANT_CENTER = Fraction(3, 2)
PLANT_RADIUS = Fraction(1, 4)
PLANT_A = Fraction(1, 3)
PLANT_Y0 = Fraction(1, 12)
PLANT_Z0 = Fraction(0, 1)
PLANT_Z1 = Fraction(0, 1)
PLANT_Z2 = Fraction(2, 3)

# Krawczyk image of the locked plant: [11/8, 35/24] ⊂ (5/4, 7/4).
PLANT_KRAWCZYK_LO = Fraction(11, 8)
PLANT_KRAWCZYK_HI = Fraction(35, 24)
PLANT_KRAWCZYK_KAPPA = Fraction(1, 6)

_LEAN_THMS: dict[str, str] = {
    "radii": "quadratic_plant_radii_unique_zero",
    "krawczyk": "quadratic_plant_krawczyk_unique_zero",
}


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _as_frac(value: Any) -> Fraction | None:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 2
    ):
        return None
    num, den = value
    if (
        isinstance(num, int)
        and not isinstance(num, bool)
        and isinstance(den, int)
        and not isinstance(den, bool)
        and den != 0
    ):
        return Fraction(num, den)
    return None


def plant_radii_poly() -> Fraction:
    """``p(r) = Y0 + (Z0+Z1)r + Z2 r² - r`` at the locked plant."""
    r = PLANT_RADIUS
    return PLANT_Y0 + (PLANT_Z0 + PLANT_Z1) * r + PLANT_Z2 * r * r - r


def plant_radii_kappa() -> Fraction:
    """``κ(r) = Z0 + Z1 + 2 Z2 r`` at the locked plant."""
    return PLANT_Z0 + PLANT_Z1 + 2 * PLANT_Z2 * PLANT_RADIUS


def plant_box() -> tuple[Fraction, Fraction]:
    return PLANT_CENTER - PLANT_RADIUS, PLANT_CENTER + PLANT_RADIUS


def locked_plant_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked quadratic plant rationals."""
    expected = {
        "center": PLANT_CENTER,
        "radius": PLANT_RADIUS,
        "A": PLANT_A,
        "Y0": PLANT_Y0,
        "Z0": PLANT_Z0,
        "Z1": PLANT_Z1,
        "Z2": PLANT_Z2,
    }
    return all(_as_frac(payload.get(key)) == value for key, value in expected.items())


def lean_nk_theorem(route: str) -> str:
    """Lean Check theorem applied by the ``nk_existence`` generator."""
    try:
        return _LEAN_THMS[route]
    except KeyError as exc:
        raise ValueError(f"unknown NK route {route!r}") from exc


def nk_existence_certificate(
    route: NKRoute,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``nk_existence`` certificate for the Mathlib bridge.

    Rationals are stored as ``[numerator, denominator]`` pairs so the bridge
    can re-derive them exactly (binary floats cannot represent ``1/3``).
    """
    if route not in LEGAL_NK_ROUTES:
        raise ValueError(f"unknown NK route {route!r}; expected one of {LEGAL_NK_ROUTES}")
    return make_certificate(
        claim=f"unique root of x^2 - 2 in [5/4, 7/4] via {route}",
        payload={
            "type": "nk_existence",
            "family": "quadratic",
            "route": route,
            "center": _pair(PLANT_CENTER),
            "radius": _pair(PLANT_RADIUS),
            "A": _pair(PLANT_A),
            "Y0": _pair(PLANT_Y0),
            "Z0": _pair(PLANT_Z0),
            "Z1": _pair(PLANT_Z1),
            "Z2": _pair(PLANT_Z2),
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_NK_FAMILIES",
    "LEGAL_NK_ROUTES",
    "PLANT_A",
    "PLANT_CENTER",
    "PLANT_KRAWCZYK_HI",
    "PLANT_KRAWCZYK_KAPPA",
    "PLANT_KRAWCZYK_LO",
    "PLANT_RADIUS",
    "PLANT_Y0",
    "PLANT_Z0",
    "PLANT_Z1",
    "PLANT_Z2",
    "lean_nk_theorem",
    "locked_plant_matches",
    "nk_existence_certificate",
    "plant_box",
    "plant_radii_kappa",
    "plant_radii_poly",
]
