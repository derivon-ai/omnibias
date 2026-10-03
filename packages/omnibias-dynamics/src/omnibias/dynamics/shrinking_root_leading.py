# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Leading two-root I-map for the shrinking-root chart ``r1 -> 0``.

On the slow line with distinct roots the height logarithm has derivative
``x / ((x-r1)(x-r2))``, so

    dx/dkappa = (x-r1)(x-r2)/x.

As ``r1 -> 0`` at fixed ``r2`` and ``x`` bounded away from zero this
tends to ``x-r2``, with exact remainder ``-r1 (x-r2)/x``. The
partial-fraction coefficient ``C = r2/(r2-r1)`` tends to 1.

These identities do not restore a uniform physical wall ``a_min``, do
not prove outgoing first-hit of the large first-root section, and do
not pass G1, G4, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ShrinkingRootLeadingReport",
    "identity_verdicts",
    "report",
    "residual_c_limit",
    "residual_r1_limit_dx",
    "residual_rescaled_b",
    "residual_two_root_dx",
    "residual_xi_limit",
    "two_root_dx_dkappa",
]


def _honesty() -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def two_root_dx_dkappa(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``dx/dκ`` along the two-root slow-line I-map."""
    if x == 0:
        raise ValueError("two_root_dx_dkappa requires nonzero x")
    return (x - r1) * (x - r2) / x


def residual_two_root_dx(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    return two_root_dx_dkappa(x, r1, r2) * x - (x - r1) * (x - r2)


def residual_r1_limit_dx(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``(x-r1)(x-r2) - x (x-r2) + r1 (x-r2)``; the ``r1 -> 0`` remainder."""
    return (x - r1) * (x - r2) - x * (x - r2) + r1 * (x - r2)


def residual_c_limit(r1: Fraction, r2: Fraction) -> Fraction:
    """``C - 1 - r1/(r2-r1)`` for ``C = r2/(r2-r1)``."""
    if r1 == r2:
        raise ValueError("residual_c_limit requires distinct roots")
    coeff_c = r2 / (r2 - r1)
    return coeff_c - 1 - r1 / (r2 - r1)


def residual_rescaled_b(xi: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``B_-(r1 xi) - r1^2 (xi-1)(xi - r2/r1)``."""
    if r1 == 0:
        raise ValueError("residual_rescaled_b requires nonzero r1")
    x = r1 * xi
    return (x - r1) * (x - r2) - r1**2 * (xi - 1) * (xi - r2 / r1)


def residual_xi_limit(xi: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """Limiting rescaled ODE remainder ``r1(ξ-1)(ξ-r2/r1) - r2(1-ξ) - r1 ξ (ξ-1)``."""
    if r1 == 0:
        raise ValueError("residual_xi_limit requires nonzero r1")
    return r1 * (xi - 1) * (xi - r2 / r1) - r2 * (1 - xi) - r1 * xi * (xi - 1)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    x, r1, r2 = Fraction(2), Fraction(1, 5), Fraction(4)
    return {
        "two_root_dx": _verdict(residual_two_root_dx(x, r1, r2)),
        "r1_limit_dx": _verdict(residual_r1_limit_dx(x, r1, r2)),
        "c_limit": _verdict(residual_c_limit(r1, r2)),
        "rescaled_b": _verdict(residual_rescaled_b(Fraction(2), r1, r2)),
        "xi_limit": _verdict(residual_xi_limit(Fraction(2), r1, r2)),
    }


@dataclass(frozen=True)
class ShrinkingRootLeadingReport:
    """Finite replay of the shrinking-root I-map. Not G1."""

    identities: Mapping[str, str]
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-shrinking-root-leading-v1",
            "identities": dict(self.identities),
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact two-root I-map and r1->0 remainder. Not outgoing "
                "first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> ShrinkingRootLeadingReport:
    return ShrinkingRootLeadingReport(
        identities=identity_verdicts(),
        outgoing_first_hit=False,
        honesty=_honesty(),
    )
