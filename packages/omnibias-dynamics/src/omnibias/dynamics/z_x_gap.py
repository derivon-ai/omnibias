# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Unfrozen-Z first-log-derivative gap including Z_x.

Along the slow line, ``dx/dkappa = B/x`` with
``B = (x-r)^2 + mu + eps^2 x^3 Z``. If ``Z`` depends on ``x``,

    (x B_x - B) - (x B0_x - B0) = eps^2 (2 x^3 Z + x^4 Z_x).

The frozen-Z identities are physical_c2. These identities do not bound
``Z_x``, do not give ``sep > 0``, first-hit, G1, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.physical_c2 import frozen_log_d1_gap

__all__ = [
    "ZXGapReport",
    "identity_verdicts",
    "report",
    "residual_unfrozen_log_d1_gap",
    "residual_unfrozen_recovers_frozen",
    "residual_zx_extra",
    "unfrozen_log_d1_gap",
]


def _honesty() -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        physical_c2_remainder=False,
        z_x_bound=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def unfrozen_log_d1_gap(
    x: Fraction,
    rstar: Fraction,
    mu: Fraction,
    eps: Fraction,
    z_jet: Fraction,
    z_x: Fraction,
) -> Fraction:
    """``(x B_x - B) - (x B0_x - B0)`` with ``Z`` depending on ``x``."""
    b0 = (x - rstar) ** 2 + mu
    b0x = 2 * (x - rstar)
    rho = eps**2 * x**3 * z_jet
    bx = b0x + eps**2 * (3 * x**2 * z_jet + x**3 * z_x)
    wall = b0 + rho
    return (x * bx - wall) - (x * b0x - b0)


def residual_unfrozen_log_d1_gap(
    x: Fraction,
    rstar: Fraction,
    mu: Fraction,
    eps: Fraction,
    z_jet: Fraction,
    z_x: Fraction,
) -> Fraction:
    """Unfrozen first-log-derivative gap minus ``eps^2 (2 x^3 Z + x^4 Z_x)``."""
    return unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, z_x) - eps**2 * (
        2 * x**3 * z_jet + x**4 * z_x
    )


def residual_unfrozen_recovers_frozen(
    x: Fraction,
    rstar: Fraction,
    mu: Fraction,
    eps: Fraction,
    z_jet: Fraction,
) -> Fraction:
    """``Z_x = 0`` recovers the frozen-Z first-log-derivative gap."""
    return unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, Fraction(0)) - frozen_log_d1_gap(
        x, rstar, mu, eps, z_jet
    )


def residual_zx_extra(
    x: Fraction,
    rstar: Fraction,
    mu: Fraction,
    eps: Fraction,
    z_jet: Fraction,
    z_x: Fraction,
) -> Fraction:
    """Unfrozen minus frozen gap minus ``eps^2 x^4 Z_x``."""
    return (
        unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, z_x)
        - frozen_log_d1_gap(x, rstar, mu, eps, z_jet)
        - eps**2 * x**4 * z_x
    )


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    x, rstar, mu, eps, z_jet, z_x = (
        Fraction(2),
        Fraction(3, 2),
        Fraction(1, 4),
        Fraction(1, 5),
        Fraction(7, 2),
        Fraction(5, 2),
    )
    return {
        "unfrozen_log_d1_gap": _verdict(
            residual_unfrozen_log_d1_gap(x, rstar, mu, eps, z_jet, z_x)
        ),
        "unfrozen_recovers_frozen": _verdict(
            residual_unfrozen_recovers_frozen(x, rstar, mu, eps, z_jet)
        ),
        "zx_extra": _verdict(residual_zx_extra(x, rstar, mu, eps, z_jet, z_x)),
    }


@dataclass(frozen=True)
class ZXGapReport:
    """Unfrozen-Z first-log-derivative identities. Not a Z_x bound or G1."""

    identities: Mapping[str, str]
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-z-x-gap-v1",
            "identities": dict(self.identities),
            "honesty": dict(self.honesty),
            "scope": (
                "Unfrozen-Z first-log-derivative gap identities including "
                "Z_x versus the lifted fold. Not a bound on Z_x, sep>0, "
                "a uniform-in-eps majorant, G1, or Hilbert XVI."
            ),
        }


def report() -> ZXGapReport:
    """Replay unfrozen-Z identities. Parent flags stay false."""
    return ZXGapReport(identities=identity_verdicts(), honesty=_honesty())
