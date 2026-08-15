# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named unique-zero certificates for planted polynomial maps.

A ``named_zero`` payload names one locked family (``circle_line``,
``hopf_radial``, ``ccf_chebyshev``) and the rational center / radius of its
compact box. The Mathlib bridge re-derives those rationals and emits Lean
that applies the matching ``OmnibiasAnalytic.Check`` unique-zero theorem.

Each family is a unique root of a named polynomial on an explicit compact
box. It is not a continuum PDE, not a Lohner time-``2π`` return map, and
not a continuum CCF / Euler blow-up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_NAMED_FAMILIES: tuple[str, ...] = (
    "circle_line",
    "hopf_radial",
    "ccf_chebyshev",
)

NamedFamily = Literal["circle_line", "hopf_radial", "ccf_chebyshev"]


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


# Locked plants: (center, radius, A, Y0, K) over ℚ. The certificate stores
# only family / center / radius; A, Y0, K are the 1-D Newton constants the
# Lean theorems use (circle_line after the y = x reduction).
_LOCKED: dict[str, dict[str, Fraction]] = {
    "circle_line": {
        "center": Fraction(3, 4),
        "radius": Fraction(1, 8),
        "A": Fraction(1, 3),
        "Y0": Fraction(1, 24),
        "K": Fraction(1, 6),
    },
    "hopf_radial": {
        "center": Fraction(1, 1),
        "radius": Fraction(1, 4),
        "A": Fraction(-1, 2),
        "Y0": Fraction(0, 1),
        "K": Fraction(27, 32),
    },
    "ccf_chebyshev": {
        "center": Fraction(7, 8),
        "radius": Fraction(1, 8),
        "A": Fraction(16, 99),
        "Y0": Fraction(7, 792),
        "K": Fraction(5, 11),
    },
}

_LEAN_THMS: dict[str, str] = {
    "circle_line": "circle_line_unique_zero",
    "hopf_radial": "hopf_radial_unique_zero",
    "ccf_chebyshev": "ccf_chebyshev_unique_zero",
}

_CLAIMS: dict[str, str] = {
    "circle_line": "unique intersection of the unit circle and y = x in [5/8, 7/8]^2",
    "hopf_radial": "unique root of r(1-r^2) in [3/4, 5/4]",
    "ccf_chebyshev": "unique root of T3 in [3/4, 1]",
}


def locked_named_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked center / radius for its family."""
    family = payload.get("family")
    if family not in _LOCKED:
        return False
    locked = _LOCKED[family]
    return (
        _as_frac(payload.get("center")) == locked["center"]
        and _as_frac(payload.get("radius")) == locked["radius"]
    )


def family_selfmap_holds(family: str) -> bool:
    """Re-derive ``K < 1`` and ``Y0 + K r ≤ r`` for the locked plant."""
    locked = _LOCKED[family]
    y0, kappa, radius = locked["Y0"], locked["K"], locked["radius"]
    return kappa < 1 and y0 + kappa * radius <= radius


def lean_named_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``named_zero`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown named-zero family {family!r}") from exc


def named_zero_certificate(
    family: NamedFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``named_zero`` certificate for the Mathlib bridge.

    Rationals are stored as ``[numerator, denominator]`` pairs so the bridge
    can re-derive them exactly.
    """
    if family not in LEGAL_NAMED_FAMILIES:
        raise ValueError(
            f"unknown named-zero family {family!r}; expected one of {LEGAL_NAMED_FAMILIES}"
        )
    locked = _LOCKED[family]
    return make_certificate(
        claim=_CLAIMS[family],
        payload={
            "type": "named_zero",
            "family": family,
            "center": _pair(locked["center"]),
            "radius": _pair(locked["radius"]),
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_NAMED_FAMILIES",
    "family_selfmap_holds",
    "lean_named_theorem",
    "locked_named_matches",
    "named_zero_certificate",
]
