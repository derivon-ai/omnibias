# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Compact-box residual and finite-matrix gap certificates.

A ``compact_box`` payload names one locked family (``ns_box``,
``transfer_2x2``). The Mathlib bridge re-derives the rationals and emits Lean
that applies the matching ``OmnibiasAnalytic.Check`` compact theorem.

``ns_box`` is a residual lower bound of a named incompressible polynomial
field on ``[1/2, 1]²``. ``transfer_2x2`` is the characteristic polynomial
of a named rational matrix with ratio ``5/8``. Neither is a continuum
regularity theorem or a continuum gauge claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_COMPACT_FAMILIES: tuple[str, ...] = ("ns_box", "transfer_2x2")

CompactFamily = Literal["ns_box", "transfer_2x2"]


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


NS_BOX_LO = Fraction(1, 2)
NS_BOX_HI = Fraction(1, 1)
NS_RESIDUAL_LO = Fraction(1, 2)

TRANSFER_A00 = Fraction(13, 2)
TRANSFER_A01 = Fraction(3, 2)
TRANSFER_RATIO = Fraction(5, 8)
TRANSFER_GAP = Fraction(3, 1)

_LEAN_THMS: dict[str, str] = {
    "ns_box": "ns_box_div_and_residual",
    "transfer_2x2": "transfer_plant_charpoly_and_gap",
}

_CLAIMS: dict[str, str] = {
    "ns_box": "named incompressible residual lo on [1/2, 1]^2",
    "transfer_2x2": "char-poly gap of [[13/2, 3/2], [3/2, 13/2]]",
}


def locked_compact_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked rationals for its family."""
    family = payload.get("family")
    if family == "ns_box":
        return (
            _as_frac(payload.get("lo")) == NS_BOX_LO
            and _as_frac(payload.get("hi")) == NS_BOX_HI
            and _as_frac(payload.get("residual_lo")) == NS_RESIDUAL_LO
        )
    if family == "transfer_2x2":
        return (
            _as_frac(payload.get("a00")) == TRANSFER_A00
            and _as_frac(payload.get("a01")) == TRANSFER_A01
            and _as_frac(payload.get("a10")) == TRANSFER_A01
            and _as_frac(payload.get("a11")) == TRANSFER_A00
            and _as_frac(payload.get("ratio")) == TRANSFER_RATIO
        )
    return False


def family_facts_hold(family: str) -> bool:
    """Re-derive the locked compact-box facts over ``Fraction``."""
    if family == "ns_box":
        return NS_BOX_LO <= NS_RESIDUAL_LO and NS_RESIDUAL_LO <= NS_BOX_HI
    if family == "transfer_2x2":
        return abs(TRANSFER_RATIO) < 1 and TRANSFER_GAP > 0
    return False


def lean_compact_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``compact_box`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown compact-box family {family!r}") from exc


def compact_box_certificate(
    family: CompactFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``compact_box`` certificate for the Mathlib bridge.

    Rationals are stored as ``[numerator, denominator]`` pairs so the bridge
    can re-derive them exactly.
    """
    if family not in LEGAL_COMPACT_FAMILIES:
        raise ValueError(
            f"unknown compact-box family {family!r}; expected one of {LEGAL_COMPACT_FAMILIES}"
        )
    if family == "ns_box":
        payload: dict[str, Any] = {
            "type": "compact_box",
            "family": family,
            "lo": _pair(NS_BOX_LO),
            "hi": _pair(NS_BOX_HI),
            "residual_lo": _pair(NS_RESIDUAL_LO),
        }
    else:
        payload = {
            "type": "compact_box",
            "family": family,
            "a00": _pair(TRANSFER_A00),
            "a01": _pair(TRANSFER_A01),
            "a10": _pair(TRANSFER_A01),
            "a11": _pair(TRANSFER_A00),
            "ratio": _pair(TRANSFER_RATIO),
        }
    return make_certificate(
        claim=_CLAIMS[family],
        payload=payload,
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_COMPACT_FAMILIES",
    "NS_BOX_HI",
    "NS_BOX_LO",
    "NS_RESIDUAL_LO",
    "TRANSFER_A00",
    "TRANSFER_A01",
    "TRANSFER_GAP",
    "TRANSFER_RATIO",
    "compact_box_certificate",
    "family_facts_hold",
    "lean_compact_theorem",
    "locked_compact_matches",
]
