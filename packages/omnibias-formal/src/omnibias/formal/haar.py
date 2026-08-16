# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Named Weyl-volume prefactor certificates.

A ``haar_volume`` payload names the locked integer identity ``6 * 4 = 24``.
The Mathlib bridge re-derives it and applies ``haar_weyl_prefactor_24``.

This is finite arithmetic on the Weyl prefactor. It is not a continuum
Haar theorem and not 4-D SU(3) Yang-Mills.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from omnibias.core.proof.certificate import Cert, make_certificate

LEGAL_HAAR_FAMILIES: tuple[str, ...] = ("weyl_prefactor_24",)

HaarFamily = Literal["weyl_prefactor_24"]

_LEAN_THMS: dict[str, str] = {
    "weyl_prefactor_24": "haar_weyl_prefactor_24",
}

_CLAIMS: dict[str, str] = {
    "weyl_prefactor_24": "Weyl volume prefactor 6*4=24",
}

_LOCKED_VALUE: dict[str, int] = {
    "weyl_prefactor_24": 24,
}


def locked_haar_matches(payload: Mapping[str, Any]) -> bool:
    """``True`` iff ``payload`` carries the locked integer for its family."""
    family = payload.get("family")
    if family not in _LOCKED_VALUE:
        return False
    value = payload.get("value")
    return isinstance(value, int) and not isinstance(value, bool) and value == _LOCKED_VALUE[family]


def family_facts_hold(family: str) -> bool:
    """Re-derive the locked Weyl-prefactor identity over ``int``."""
    if family == "weyl_prefactor_24":
        return 6 * 4 == 24
    return False


def lean_haar_theorem(family: str) -> str:
    """Lean Check theorem applied by the ``haar_volume`` generator."""
    try:
        return _LEAN_THMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown haar family {family!r}") from exc


def haar_certificate(
    family: HaarFamily,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a planted ``haar_volume`` certificate for the Mathlib bridge."""
    if family not in LEGAL_HAAR_FAMILIES:
        raise ValueError(
            f"unknown haar family {family!r}; expected one of {LEGAL_HAAR_FAMILIES}"
        )
    return make_certificate(
        claim=_CLAIMS[family],
        payload={
            "type": "haar_volume",
            "family": family,
            "value": _LOCKED_VALUE[family],
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_HAAR_FAMILIES",
    "family_facts_hold",
    "haar_certificate",
    "lean_haar_theorem",
    "locked_haar_matches",
]
