# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Incoming GRAZING comparison speed bound for E_sigma.

On the kill line L = 0, lambda1 = -2, nu = eps, the reversed cubic
field starting at V = 0, h = 4 eps^3 has

    Vdot_rev = -(f + h g) >= -F(V, eps),

where F = f + 4 eps^3 g is the same comparison polynomial as
e_out_speed. On V in [0, 1], phi(0) = -2 eps + 4 eps^3 < 0 for
eps in (0, 1/16], so F_V < 0, F is decreasing, and

    F(V, eps) <= F(0, eps) = -4 eps^3 (1 + eps) < 0.

Hence Vdot_rev >= 4 eps^3 (1 + eps) while h >= h(0) and g < 0, and
the comparison time to cross Delta V = 1 is at most
1 / (4 eps^3 (1 + eps)). Lohner wrapping refuses a certified
E_sigma first-hit on this compact. This is an O(1/eps^3)
comparison majorant, not GRAZING first-hit, not G1, and not
Hilbert XVI. The outgoing comparison is e_out_speed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_speed import cmp_F, enclose_g, phi
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ESigmaSpeedReport",
    "enclose_phi_zero",
    "identity_verdicts",
    "report",
    "residual_F_zero",
    "residual_T_in",
    "residual_phi_zero",
    "residual_phi_zero_factor",
    "t_in_majorant",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_T = Fraction(16384, 17)


def _honesty(*, e_sigma_speed_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_speed_bound=e_sigma_speed_bound,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_F_zero(eps: Fraction) -> Fraction:
    """``F(0, eps) + 4 eps^3 (1 + eps)``."""
    return cmp_F(Fraction(0), eps) + 4 * eps**3 * (1 + eps)


def residual_phi_zero(eps: Fraction) -> Fraction:
    """``phi(0, eps) - (-2 eps + 4 eps^3)``."""
    return phi(Fraction(0), eps) - (-2 * eps + 4 * eps**3)


def residual_phi_zero_factor(eps: Fraction) -> Fraction:
    """``phi(0, eps) + 2 eps (1 - 2 eps^2)``."""
    return phi(Fraction(0), eps) + 2 * eps * (1 - 2 * eps * eps)


def t_in_majorant(eps: Fraction) -> Fraction:
    """``T = 1 / (4 eps^3 (1 + eps))``."""
    if eps == 0:
        raise ValueError("eps must be nonzero")
    return 1 / (4 * eps**3 * (1 + eps))


def residual_T_in(eps: Fraction, majorant: Fraction) -> Fraction:
    """Cleared ``T = 1/(4 eps^3 (1+eps))``: ``4 eps^3 (1+eps) T - 1``."""
    return 4 * eps**3 * (1 + eps) * majorant - 1


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    eps, majorant = _SAMPLE_EPS, _SAMPLE_T
    return {
        "F_zero": _verdict(residual_F_zero(eps)),
        "phi_zero": _verdict(residual_phi_zero(eps)),
        "phi_zero_factor": _verdict(residual_phi_zero_factor(eps)),
        "T_in": _verdict(residual_T_in(eps, majorant)),
    }


def enclose_phi_zero(*, eps: Fraction = _SAMPLE_EPS) -> dict[str, float | bool]:
    """Point enclosure of ``phi(0, eps)``. Refuses a collapsed origin and ``eps=1``."""
    if eps == 0:
        return {"lo": 0.0, "hi": 0.0, "strictly_negative": False}
    value = phi(Fraction(0), eps)
    box = Interval.from_rational(value)
    return {
        "lo": box.lo,
        "hi": box.hi,
        "strictly_negative": box.hi < 0.0,
    }


@dataclass(frozen=True)
class ESigmaSpeedReport:
    """Incoming GRAZING comparison speed. Not E_sigma first-hit or G1."""

    identities: Mapping[str, str]
    phi_zero: Mapping[str, float | bool]
    g_box: Mapping[str, float | bool]
    origin: Mapping[str, float | bool]
    wide: Mapping[str, float | bool]
    e_sigma_speed_bound: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-speed-v1",
            "identities": dict(self.identities),
            "phi_zero": dict(self.phi_zero),
            "g_box": dict(self.g_box),
            "origin": dict(self.origin),
            "wide": dict(self.wide),
            "e_sigma_speed_bound": self.e_sigma_speed_bound,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Incoming reverse cubic has Vdot_rev >= 4 eps^3 (1+eps) "
                "on V in [0, 1] because F is decreasing and g<0 with "
                "h>=h(0). Comparison time to cross Delta V=1 is at most "
                "1/(4 eps^3 (1+eps)). Lohner wrapping refuses a certified "
                "E_sigma first-hit. O(1/eps^3) comparison majorant, not "
                "GRAZING first-hit, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaSpeedReport:
    """Replay incoming comparison identities; refuse origin and eps=1."""
    identities = identity_verdicts()
    phi_zero = enclose_phi_zero()
    g_box = enclose_g(eps=_SAMPLE_EPS, v_lo=Fraction(0), v_hi=Fraction(1))
    origin = enclose_phi_zero(eps=Fraction(0))
    wide = enclose_phi_zero(eps=Fraction(1))
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(phi_zero["strictly_negative"])
        and bool(g_box["strictly_negative"])
        and origin["strictly_negative"] is False
        and wide["strictly_negative"] is False
    )
    return ESigmaSpeedReport(
        identities=identities,
        phi_zero=phi_zero,
        g_box=g_box,
        origin=origin,
        wide=wide,
        e_sigma_speed_bound=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_speed_bound=sealed),
    )
