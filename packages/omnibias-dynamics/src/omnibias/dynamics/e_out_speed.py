# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line comparison speed bound for matching-chart E_out.

On L = 0, lambda1 = -2, nu = eps, the cubic field with h >= 4 eps^3
obeys

    Vdot <= F(V, eps) := f(V) + 4 eps^3 g(V).

The V-derivative is F_V = eps phi with phi = V^2 - 2 V - 2 eps + 4 eps^3.
On V in [-rho, -eps], phi is decreasing in V and its right-end value is
eps^2 (1 + 4 eps) > 0, so F_V > 0, F is increasing, and

    F(V, eps) <= F(-eps, eps) = - (3 eps^3 + (13/3) eps^4 + 4 eps^5) < 0.

Hence -Vdot >= 3 eps^3 and the comparison hitting time of V = -rho is
at most (rho - eps) / (3 eps^3), uniformly for every eps in (0, 1/16].
This is an O(1/eps^3) comparison majorant, not a Lohner first-hit for
every eps, not GRAZING E_sigma, not G1, and not Hilbert XVI. The finite
shrinking pack is e_out_eps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.vh_orbit import field_f, field_g

__all__ = [
    "EOutSpeedReport",
    "cmp_F",
    "cmp_F_V",
    "enclose_g",
    "enclose_init_speed",
    "enclose_phi",
    "identity_verdicts",
    "phi",
    "report",
    "residual_F_V_phi",
    "residual_F_expand",
    "residual_T_cubic",
    "residual_phi_right",
    "t_cubic_majorant",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(-1, 8)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_T = Fraction(256)


def _honesty(*, e_out_speed_bound: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_out_speed_bound=e_out_speed_bound,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def cmp_F(v_coord: Fraction, eps: Fraction) -> Fraction:
    """Comparison ``f(V) + 4 eps^3 g(V)`` on the kill line."""
    return field_f(Fraction(0), Fraction(-2), eps, v_coord) + (4 * eps**3) * field_g(
        eps, v_coord
    )


def F_expanded(v_coord: Fraction, eps: Fraction) -> Fraction:
    """Expanded comparison polynomial."""
    return (
        -2 * eps**2 * v_coord
        - eps * v_coord * v_coord
        + (eps / 3) * v_coord**3
        - 4 * eps**3
        - 4 * eps**4
        + 4 * eps**4 * v_coord
    )


def cmp_F_V(v_coord: Fraction, eps: Fraction) -> Fraction:
    """``dF/dV = -2 eps^2 - 2 eps V + eps V^2 + 4 eps^4``."""
    return -2 * eps**2 - 2 * eps * v_coord + eps * v_coord * v_coord + 4 * eps**4


def phi(v_coord: Fraction, eps: Fraction) -> Fraction:
    """``V^2 - 2 V - 2 eps + 4 eps^3``; ``F_V = eps phi``."""
    return v_coord * v_coord - 2 * v_coord - 2 * eps + 4 * eps**3


def residual_F_expand(v_coord: Fraction, eps: Fraction) -> Fraction:
    """``(f + 4 eps^3 g) - expanded F``."""
    return cmp_F(v_coord, eps) - F_expanded(v_coord, eps)


def residual_F_V_phi(v_coord: Fraction, eps: Fraction) -> Fraction:
    """``F_V - eps phi``."""
    return cmp_F_V(v_coord, eps) - eps * phi(v_coord, eps)


def residual_phi_right(eps: Fraction) -> Fraction:
    """``phi(-eps, eps) - (eps^2 + 4 eps^3)``."""
    return phi(-eps, eps) - (eps * eps + 4 * eps**3)


def t_cubic_majorant(eps: Fraction, rho: Fraction) -> Fraction:
    """``T = (rho - eps) / (3 eps^3)``."""
    if eps == 0:
        raise ValueError("eps must be nonzero")
    return (rho - eps) / (3 * eps**3)


def residual_T_cubic(eps: Fraction, rho: Fraction, majorant: Fraction) -> Fraction:
    """Cleared ``T = (rho-eps)/(3 eps^3)``: ``3 eps^3 T - (rho-eps)``."""
    return 3 * eps**3 * majorant - (rho - eps)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    eps, v_coord, rho, majorant = _SAMPLE_EPS, _SAMPLE_V, _SAMPLE_RHO, _SAMPLE_T
    return {
        "F_expand": _verdict(residual_F_expand(v_coord, eps)),
        "F_V_phi": _verdict(residual_F_V_phi(v_coord, eps)),
        "phi_right": _verdict(residual_phi_right(eps)),
        "T_cubic": _verdict(residual_T_cubic(eps, rho, majorant)),
    }


def enclose_phi(
    *,
    eps: Fraction = _SAMPLE_EPS,
    v_lo: Fraction = -_SAMPLE_RHO,
    v_hi: Fraction | None = None,
) -> dict[str, float | bool]:
    """Interval enclosure of ``phi`` on a ``V`` box. Refuses a collapsed origin."""
    if v_hi is None:
        v_hi = -eps
    if eps == 0 and v_hi >= 0:
        return {"lo": 0.0, "hi": 0.0, "strictly_positive": False}
    V = Interval.hull(Interval.from_rational(v_lo), Interval.from_rational(v_hi))
    e = Interval.from_rational(eps)
    two = Interval.from_rational(2)
    four = Interval.from_rational(4)
    box = (V * V) - two * V - two * e + four * (e**3)
    exact_min = eps * eps * (1 + 4 * eps)
    return {
        "lo": box.lo,
        "hi": box.hi,
        "exact_min": float(exact_min),
        "strictly_positive": box.lo > 0.0 and exact_min > 0,
    }


def enclose_g(
    *,
    eps: Fraction = _SAMPLE_EPS,
    v_lo: Fraction = -_SAMPLE_RHO,
    v_hi: Fraction | None = None,
) -> dict[str, float | bool]:
    """Interval enclosure of ``g = -1 + eps (V-1)`` on the outgoing box."""
    if v_hi is None:
        v_hi = -eps
    if eps == 0:
        return {"lo": -1.0, "hi": -1.0, "strictly_negative": True}
    V = Interval.hull(Interval.from_rational(v_lo), Interval.from_rational(v_hi))
    e = Interval.from_rational(eps)
    one = Interval.from_rational(1)
    box = Interval.from_rational(-1) + e * (V - one)
    return {
        "lo": box.lo,
        "hi": box.hi,
        "strictly_negative": box.hi < 0.0,
    }


def enclose_init_speed(*, eps: Fraction = _SAMPLE_EPS) -> dict[str, float | bool]:
    """Exact kill-line matching ``-Vdot = 3 eps^3 + (13/3) eps^4 + 4 eps^5``."""
    speed = 3 * eps**3 + Fraction(13, 3) * eps**4 + 4 * eps**5
    box = Interval.from_rational(speed)
    cubic = 3 * eps**3
    return {
        "lo": box.lo,
        "hi": box.hi,
        "above_cubic": speed >= cubic,
        "cubic": float(cubic),
    }


@dataclass(frozen=True)
class EOutSpeedReport:
    """Kill-line comparison speed bound. Not uniform Lohner, GRAZING, or G1."""

    identities: Mapping[str, str]
    phi_box: Mapping[str, float | bool]
    g_box: Mapping[str, float | bool]
    speed: Mapping[str, float | bool]
    origin: Mapping[str, float | bool]
    e_out_speed_bound: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-out-speed-v1",
            "identities": dict(self.identities),
            "phi_box": dict(self.phi_box),
            "g_box": dict(self.g_box),
            "speed": dict(self.speed),
            "origin": dict(self.origin),
            "e_out_speed_bound": self.e_out_speed_bound,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison Vdot <= F with F increasing and "
                "-Vdot >= 3 eps^3 on the kill-line cubic, so the "
                "hitting time of V=-rho is at most (rho-eps)/(3 eps^3) "
                "for every eps in (0, 1/16]. O(1/eps^3) comparison "
                "majorant, not Lohner for every eps, not GRAZING "
                "E_sigma, not G1, or Hilbert XVI."
            ),
        }


def report() -> EOutSpeedReport:
    """Replay comparison identities and enclose phi > 0 on the outgoing box."""
    identities = identity_verdicts()
    phi_box = enclose_phi()
    g_box = enclose_g()
    speed = enclose_init_speed()
    origin = enclose_phi(eps=Fraction(0), v_lo=Fraction(0), v_hi=Fraction(0))
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(phi_box["strictly_positive"])
        and bool(g_box["strictly_negative"])
        and bool(speed["above_cubic"])
        and origin["strictly_positive"] is False
    )
    return EOutSpeedReport(
        identities=identities,
        phi_box=phi_box,
        g_box=g_box,
        speed=speed,
        origin=origin,
        e_out_speed_bound=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_out_speed_bound=sealed),
    )
