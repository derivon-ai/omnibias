# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Actual-versus-comparison T_h gap on the C=0 chart.

The first-root height continuation uses ``T_h = q/h + k``. The C=0
comparison is ``T_h = 1``. Their difference splits exactly as

    (q/h + k) - (1 + C eps T/h + C eps^3 / h)
        = (k - 1) + (q - C eps (T + eps^2)) / h.

When ``q`` touches the comparison flux ``C eps (T + eps^2)``, the gap
collapses to ``k - 1``. On ``C=0``, ``A=1``, ``|k-1| <= 3 nu`` for
``|v| <= 2`` and ``nu <= 1/8``.

This is the algebraic gap, not a validated integral of ``T-h`` along an
actual ``(V,h)`` orbit, not height-section first-hit, G1, or Hilbert
XVI. The ``C!=0`` mixing is height_mix; the slow-line ``Z`` bound is
cancelled_n.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.k_zeta_remainder import k_normal

__all__ = [
    "OrbitThReport",
    "identity_verdicts",
    "k_gap_on_box",
    "report",
    "residual_th_split",
    "residual_th_touching",
]

_SAMPLE_Q = Fraction(3, 16)
_SAMPLE_H = Fraction(1, 4)
_SAMPLE_K = Fraction(17, 16)
_SAMPLE_C = Fraction(2)
_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_T = Fraction(1, 8)
_SAMPLE_NU = Fraction(1, 16)
_SAMPLE_VMAX = Fraction(2)


def _honesty(*, orbit_th_c0: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        height_mix_gh=False,
        cancelled_n_usable_z=False,
        orbit_th_c0=orbit_th_c0,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_th_split(
    q: Fraction,
    height: Fraction,
    kay: Fraction,
    c: Fraction,
    eps: Fraction,
    kinetic: Fraction,
) -> Fraction:
    """Actual ``T_h`` minus comparison minus ``(k-1) + (q - C eps(T+eps^2))/h``."""
    if height == 0:
        raise ValueError("residual_th_split requires nonzero h")
    actual = q / height + kay
    comparison = 1 + c * eps * kinetic / height + c * eps**3 / height
    gap = (kay - 1) + (q - c * eps * (kinetic + eps * eps)) / height
    return actual - comparison - gap


def residual_th_touching(
    c: Fraction, eps: Fraction, kinetic: Fraction, height: Fraction, kay: Fraction
) -> Fraction:
    """When ``q = C eps (T + eps^2)`` the gap equals ``k-1``."""
    q = c * eps * (kinetic + eps * eps)
    actual = q / height + kay
    comparison = 1 + c * eps * kinetic / height + c * eps**3 / height
    return (actual - comparison) - (kay - 1)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    q, h, kay = _SAMPLE_Q, _SAMPLE_H, _SAMPLE_K
    c, eps, kinetic = _SAMPLE_C, _SAMPLE_EPS, _SAMPLE_T
    return {
        "th_split": _verdict(residual_th_split(q, h, kay, c, eps, kinetic)),
        "th_touching": _verdict(residual_th_touching(c, eps, kinetic, h, kay)),
    }


def k_gap_on_box() -> dict[str, float | bool]:
    """``|k_normal-1| <= 3 nu`` on ``|v|<=2``, ``nu=1/n`` for ``n>=8``."""
    samples: list[float] = []
    ok = True
    for n in (8, 16, 32, 64):
        nu = Fraction(1, n)
        for v in (_SAMPLE_VMAX, -_SAMPLE_VMAX, Fraction(0), Fraction(1)):
            gap = abs(k_normal(Fraction(1), nu, v) - 1)
            samples.append(float(gap / nu))
            if gap > 3 * nu:
                ok = False
    return {"ratio_hi": max(samples), "within_three": ok}


@dataclass(frozen=True)
class OrbitThReport:
    """C=0 actual-versus-comparison T_h gap. Not integrated T-h or G1."""

    identities: Mapping[str, str]
    k_bound: Mapping[str, float | bool]
    orbit_th_c0: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-orbit-th-v1",
            "identities": dict(self.identities),
            "k_bound": dict(self.k_bound),
            "orbit_th_c0": self.orbit_th_c0,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Actual-versus-comparison T_h gap equals (k-1) at flux "
                "touching. Not an integrated T-h orbit, first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> OrbitThReport:
    """Replay the T_h gap identities and the C=0 |k-1| box."""
    identities = identity_verdicts()
    k_bound = k_gap_on_box()
    sealed = all(status == "PROVED" for status in identities.values()) and bool(
        k_bound["within_three"]
    )
    return OrbitThReport(
        identities=identities,
        k_bound=k_bound,
        orbit_th_c0=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(orbit_th_c0=sealed),
    )
