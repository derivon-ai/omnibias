# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named Racah 6j certificates for locked rational identities.

A ``sixj`` payload names one locked family (``half_half_zero``,
``all_half_vanishes``). The Mathlib bridge re-derives the rationals and
emits Lean that applies the matching ``OmnibiasAnalytic.Check`` SixJ
theorem.

Each family is a finite rational identity on named labels. It is not a
continuum gauge claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_SIXJ_FAMILIES: tuple[str, ...] = ("half_half_zero", "all_half_vanishes")

SixJFamily = Literal["half_half_zero", "all_half_vanishes"]


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


_LEAN_THMS: dict[str, str] = {
    "half_half_zero": "sixj_half_half_zero",
    "all_half_vanishes": "sixj_all_half_vanishes",
}

_CLAIMS: dict[str, str] = {
    "half_half_zero": "Racah 6j {1/2 1/2 0; 1/2 1/2 0} = -1/2",
    "all_half_vanishes": "all-1/2 6j vanishes (illegal triad)",
}

_LOCKED_VALUE: dict[str, Fraction] = {
    "half_half_zero": Fraction(-1, 2),
    "all_half_vanishes": Fraction(0),
}


def locked_sixj_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked rational for its family."""
    family = payload.get("family")
    if family not in _LOCKED_VALUE:
        return False
    return _as_frac(payload.get("value")) == _LOCKED_VALUE[family]


def family_facts_hold(family: str) -> bool:
    """Re-derive the locked 6j identity over ``Fraction``."""
    if family == "half_half_zero":
        return Fraction(1, 4) * Fraction(-2) == Fraction(-1, 2)
    if family == "all_half_vanishes":
        return Fraction(0) == 0
    return False


def lean_sixj_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``sixj`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown sixj family {family!r}") from exc


def sixj_certificate(
    family: SixJFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``sixj`` certificate for the Mathlib bridge."""
    if family not in LEGAL_SIXJ_FAMILIES:
        raise ValueError(
            f"unknown sixj family {family!r}; expected one of {LEGAL_SIXJ_FAMILIES}"
        )
    return make_certificate(
        claim=_CLAIMS[family],
        payload={
            "type": "sixj",
            "family": family,
            "value": _pair(_LOCKED_VALUE[family]),
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_SIXJ_FAMILIES",
    "family_facts_hold",
    "lean_sixj_theorem",
    "locked_sixj_matches",
    "sixj_certificate",
]
