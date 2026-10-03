# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Frozen-Z C2 remainder of log D' versus the lifted fold map.

Along the slow line, ``dx/dkappa = B/x`` and
``(log D')_kappa = (x B_x - B)/x^2``. Splitting
``B = (x-r)^2 + mu + eps^2 x^3 Z`` with ``Z`` frozen in ``x`` yields an
exact first-derivative gap ``2 eps^2 x Z`` and an exact C2 gap
``2 eps^2 Z B / x``.

These identities do not consume a uniform-in-eps Cauchy majorant, do not
cover ``Z_x``, ``sep > 0``, first-hit, G1, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "PhysicalC2Report",
    "frozen_c2_gap",
    "frozen_log_d1_gap",
    "identity_verdicts",
    "report",
    "residual_frozen_c2_gap",
    "residual_frozen_log_d1_gap",
    "residual_lift_cubic_mismatch",
]


def _honesty() -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        inner_z_compact_bound=False,
        fold_z_compact_bound=False,
        physical_c2_remainder=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_lift_cubic_mismatch(
    x: Fraction, rstar: Fraction, beta0: Fraction, eps: Fraction
) -> Fraction:
    """Linear-in-``eps`` gap of the lift versus ``beta0 eps x^3``."""
    b_lead = (x - rstar) ** 2 + beta0 * eps * x**3
    b0 = (x - rstar) ** 2 + beta0 * eps * rstar**3
    return b_lead - b0 - beta0 * eps * (x**3 - rstar**3)


def frozen_log_d1_gap(
    x: Fraction, rstar: Fraction, mu: Fraction, eps: Fraction, z_jet: Fraction
) -> Fraction:
    """``(x B_x - B) - (x B0_x - B0)`` with ``Z`` frozen in ``x``."""
    b0 = (x - rstar) ** 2 + mu
    b0x = 2 * (x - rstar)
    rho = eps**2 * x**3 * z_jet
    bx = b0x + 3 * eps**2 * x**2 * z_jet
    wall = b0 + rho
    return (x * bx - wall) - (x * b0x - b0)


def residual_frozen_log_d1_gap(
    x: Fraction, rstar: Fraction, mu: Fraction, eps: Fraction, z_jet: Fraction
) -> Fraction:
    """Frozen first-log-derivative gap minus ``2 eps^2 x^3 Z``."""
    return frozen_log_d1_gap(x, rstar, mu, eps, z_jet) - 2 * eps**2 * x**3 * z_jet


def frozen_c2_gap(
    x: Fraction, rstar: Fraction, mu: Fraction, eps: Fraction, z_jet: Fraction
) -> Fraction:
    """Frozen C2 remainder ``2 eps^2 Z B / x``."""
    if x == 0:
        raise ValueError("frozen_c2_gap requires nonzero x")
    b0 = (x - rstar) ** 2 + mu
    wall = b0 + eps**2 * x**3 * z_jet
    return 2 * eps**2 * z_jet * wall / x


def residual_frozen_c2_gap(
    x: Fraction, rstar: Fraction, mu: Fraction, eps: Fraction, z_jet: Fraction
) -> Fraction:
    """``(2 eps^2 Z B / x) * x - 2 eps^2 Z B``."""
    if x == 0:
        raise ValueError("residual_frozen_c2_gap requires nonzero x")
    b0 = (x - rstar) ** 2 + mu
    wall = b0 + eps**2 * x**3 * z_jet
    return frozen_c2_gap(x, rstar, mu, eps, z_jet) * x - 2 * eps**2 * z_jet * wall


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    x, rstar, mu, eps, z_jet, beta0 = (
        Fraction(2),
        Fraction(3, 2),
        Fraction(1, 4),
        Fraction(1, 5),
        Fraction(7, 2),
        Fraction(1, 3),
    )
    return {
        "lift_cubic_mismatch": _verdict(
            residual_lift_cubic_mismatch(x, rstar, beta0, eps)
        ),
        "frozen_log_d1_gap": _verdict(
            residual_frozen_log_d1_gap(x, rstar, mu, eps, z_jet)
        ),
        "frozen_c2_gap": _verdict(residual_frozen_c2_gap(x, rstar, mu, eps, z_jet)),
    }


@dataclass(frozen=True)
class PhysicalC2Report:
    """Frozen-Z C2 identities. Not a physical remainder or G1."""

    identities: Mapping[str, str]
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-physical-c2-v1",
            "identities": dict(self.identities),
            "honesty": dict(self.honesty),
            "scope": (
                "Frozen-Z C2 remainder identities versus the lifted fold. "
                "Not Z_x, sep>0, a uniform-in-eps majorant, G1, or Hilbert XVI."
            ),
        }


def report() -> PhysicalC2Report:
    """Replay frozen-Z C2 identities. Parent flags stay false."""
    return PhysicalC2Report(identities=identity_verdicts(), honesty=_honesty())
