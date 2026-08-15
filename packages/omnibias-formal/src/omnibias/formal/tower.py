# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Exact derivative-tower coefficient certificates for the Mathlib track.

A ``tower_coeffs`` payload records the integer coefficients of one family
(``sigmoid`` / ``tanh`` / ``sech`` / ``hermite``) at a finite order ``n``.
The Mathlib bridge re-derives the list from
:mod:`omnibias.core.verified.coeffs` and emits a Lean obligation that those
integers equal the ``OmnibiasAnalytic.Tower`` recurrence.

This is a finite coefficient identity. It does not state a finite-difference
collapse, a continuum PDE claim, or any asymptotic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from omnibias.core.proof.certificate import Cert, make_certificate
from omnibias.core.verified.coeffs import (
    hermite_coeffs_exact,
    sech_poly_coeffs_exact,
    sigmoid_poly_coeffs_exact,
    tanh_poly_coeffs_exact,
)

LEGAL_TOWER_FAMILIES: tuple[str, ...] = ("sigmoid", "tanh", "sech", "hermite")

_COEFF_FNS: dict[str, Callable[[int], tuple[int, ...]]] = {
    "sigmoid": sigmoid_poly_coeffs_exact,
    "tanh": tanh_poly_coeffs_exact,
    "sech": sech_poly_coeffs_exact,
    "hermite": hermite_coeffs_exact,
}

#: Lean ``ℕ → ℤ[X]`` name for each family (``OmnibiasAnalytic.Tower``).
TOWER_POLY_NAMES: dict[str, str] = {
    "sigmoid": "sigmoidPoly",
    "tanh": "tanhPoly",
    "sech": "sechPoly",
    "hermite": "hermitePoly",
}

#: Lean ``ℕ → List ℤ`` name for each family.
TOWER_COEFF_LIST_NAMES: dict[str, str] = {
    "sigmoid": "sigmoidCoeffList",
    "tanh": "tanhCoeffList",
    "sech": "sechCoeffList",
    "hermite": "hermiteCoeffList",
}


def tower_coeffs(family: str, n: int) -> tuple[int, ...]:
    """Exact integer coefficients of ``family`` at order ``n``.

    Raises:
        ValueError: unknown family or negative order (the verified generators
            also reject ``n < 0``).
    """
    if family not in _COEFF_FNS:
        raise ValueError(
            f"unknown tower family {family!r}; expected one of {LEGAL_TOWER_FAMILIES}"
        )
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    return _COEFF_FNS[family](n)


def tower_coeffs_certificate(
    family: str,
    n: int,
    *,
    honesty: Mapping[str, bool] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Cert:
    """Seal a ``tower_coeffs`` certificate for the Mathlib bridge.

    The payload stores the exact integer list from
    :func:`omnibias.core.verified.coeffs`; the bridge refuses a list that does
    not reproduce that generator.
    """
    coeffs = list(tower_coeffs(family, n))
    return make_certificate(
        claim=f"exact {family} tower coefficients at order {n}",
        payload={
            "type": "tower_coeffs",
            "family": family,
            "n": n,
            "coeffs": coeffs,
        },
        honesty=honesty,
        meta=meta,
    )


__all__ = [
    "LEGAL_TOWER_FAMILIES",
    "TOWER_COEFF_LIST_NAMES",
    "TOWER_POLY_NAMES",
    "tower_coeffs",
    "tower_coeffs_certificate",
]
