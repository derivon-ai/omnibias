# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named quadratic-Casimir certificates for locked SU(2) / SU(3) partitions.

A ``casimir`` payload names one locked family (``su2_fund``,
``su3_fund``). The Mathlib bridge re-derives the rationals and emits Lean
that applies the matching ``OmnibiasAnalytic.Check`` Casimir theorem.

The Freudenthal formula here is the same closed rational used by
``omnibias.geometry.gauge.quadratic_casimir``. Each family is a finite
rational identity on a named partition. It is not a continuum gauge claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_CASIMIR_FAMILIES: tuple[str, ...] = ("su2_fund", "su3_fund")

CasimirFamily = Literal["su2_fund", "su3_fund"]


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


def casimir_of(n: int, partition: Sequence[int]) -> Fraction:
    """Freudenthal ``C2`` on a length-``n`` partition (``lambda_n = 0``).

    Matches ``omnibias.geometry.gauge.quadratic_casimir``:
    ``C2 = 1/2 [ Σ_i λ_i (λ_i + n + 1 - 2(i+1)) - |λ|² / n ]``.
    """
    if n < 2:
        raise ValueError(f"su(N) needs N >= 2, got N={n}")
    if len(partition) != n:
        raise ValueError(f"partition must have length {n}, got {len(partition)}")
    total = sum(partition)
    acc = Fraction(0)
    for index, part in enumerate(partition):
        acc += part * (part + n + 1 - 2 * (index + 1))
    return Fraction(1, 2) * (acc - Fraction(total * total, n))


SU2_TRIVIAL = casimir_of(2, (0, 0))
SU2_FUND = casimir_of(2, (1, 0))
SU2_ADJOINT = casimir_of(2, (2, 0))
SU2_FUND_GAP = SU2_FUND - SU2_TRIVIAL
SU3_FUND = casimir_of(3, (1, 0, 0))

_LEAN_THMS: dict[str, str] = {
    "su2_fund": "su2_casimir_fund_gap",
    "su3_fund": "su3_casimir_fund",
}

_CLAIMS: dict[str, str] = {
    "su2_fund": "SU(2) Casimir gap C2(1)-C2(0)=3/4",
    "su3_fund": "SU(3) fundamental Casimir C2(1,0)=4/3",
}

_LOCKED_VALUE: dict[str, Fraction] = {
    "su2_fund": SU2_FUND_GAP,
    "su3_fund": SU3_FUND,
}


def locked_casimir_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked rational for its family."""
    family = payload.get("family")
    if family not in _LOCKED_VALUE:
        return False
    return _as_frac(payload.get("value")) == _LOCKED_VALUE[family]


def family_facts_hold(family: str) -> bool:
    """Re-derive the locked Casimir identity over ``Fraction``."""
    if family == "su2_fund":
        return (
            SU2_TRIVIAL == 0
            and SU2_FUND == Fraction(3, 4)
            and SU2_ADJOINT == 2
            and SU2_FUND_GAP == Fraction(3, 4)
        )
    if family == "su3_fund":
        return SU3_FUND == Fraction(4, 3)
    return False


def lean_casimir_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``casimir`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown casimir family {family!r}") from exc


def casimir_certificate(
    family: CasimirFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``casimir`` certificate for the Mathlib bridge.

    Rationals are stored as ``[numerator, denominator]`` pairs so the bridge
    can re-derive them exactly.
    """
    if family not in LEGAL_CASIMIR_FAMILIES:
        raise ValueError(
            f"unknown casimir family {family!r}; expected one of {LEGAL_CASIMIR_FAMILIES}"
        )
    return make_certificate(
        claim=_CLAIMS[family],
        payload={
            "type": "casimir",
            "family": family,
            "value": _pair(_LOCKED_VALUE[family]),
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_CASIMIR_FAMILIES",
    "SU2_ADJOINT",
    "SU2_FUND",
    "SU2_FUND_GAP",
    "SU2_TRIVIAL",
    "SU3_FUND",
    "casimir_certificate",
    "casimir_of",
    "family_facts_hold",
    "lean_casimir_theorem",
    "locked_casimir_matches",
]
