# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite shrinking-eps pack of matching-chart E_out first-hit.

On the kill line L = 0, lambda1 = -2, rho = 1/4, C = 2, matching
V(0) = -eps, h(0) = 4 eps^3, certify_stopped_event hits E_out for
eps = 1/n with n in {16, 20, 25} inside the declared majorant horizon
T = n^2 / 8. A short horizon at n = 16 does not certify.

This is a finite shrinking pack, not a uniform-in-eps theorem, not
GRAZING E_sigma, not G1, and not Hilbert XVI. The L-pack at one eps
is e_out_section.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import certify_e_out
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.vh_orbit import field_f, field_g

__all__ = [
    "EPS_PACK",
    "EOutEpsReport",
    "horizon_majorant",
    "horizon_steps",
    "identity_verdicts",
    "report",
    "residual_h0_matching",
    "residual_horizon_n2",
    "residual_v0_matching",
    "residual_vdot_kill_init",
]

EPS_PACK = (16, 20, 25)
_SAMPLE_N = 16
_SAMPLE_EPS = Fraction(1, _SAMPLE_N)
_KILL_L = Fraction(0)
_SHORT_STEPS = 32
_HIT_ORDER = 6


def _honesty(*, e_out_eps_pack: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_out_eps_pack=e_out_eps_pack,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_v0_matching(v0: Fraction, eps: Fraction) -> Fraction:
    """Matching ``V(0) + eps``."""
    return v0 + eps


def residual_h0_matching(h0: Fraction, eps: Fraction) -> Fraction:
    """Matching ``h(0) - 4 eps^3``."""
    return h0 - 4 * eps**3


def residual_horizon_n2(n: Fraction, majorant: Fraction) -> Fraction:
    """Declared majorant ``T = n^2 / 8`` cleared: ``n^2 - 8 T``."""
    return n * n - 8 * majorant


def residual_vdot_kill_init(vdot: Fraction, eps: Fraction) -> Fraction:
    """Kill-line matching ``Vdot + 3 eps^3 + (13/3) eps^4 + 4 eps^5``."""
    return vdot + 3 * eps**3 + Fraction(13, 3) * eps**4 + 4 * eps**5


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    eps = _SAMPLE_EPS
    n = Fraction(_SAMPLE_N)
    v0 = -eps
    h0 = 4 * eps**3
    eff = field_f(Fraction(0), Fraction(-2), eps, v0)
    gee = field_g(eps, v0)
    vdot = eff + h0 * gee
    return {
        "v0_matching": _verdict(residual_v0_matching(v0, eps)),
        "h0_matching": _verdict(residual_h0_matching(h0, eps)),
        "horizon_n2": _verdict(residual_horizon_n2(n, horizon_majorant(n))),
        "vdot_kill_init": _verdict(residual_vdot_kill_init(vdot, eps)),
    }


def horizon_majorant(n: Fraction | int) -> Fraction:
    """Declared hit-time majorant ``T = n^2 / 8``."""
    value = n if isinstance(n, Fraction) else Fraction(n)
    return value * value / 8


def horizon_steps(n: int, *, step: float) -> int:
    """Integer step count covering ``T = n^2 / 8``."""
    return int(float(horizon_majorant(n)) / step) + 2


def _step_for(n: int) -> float:
    return 0.5 if n <= 16 else 1.0


def certify_pack_member(n: int) -> dict[str, float | bool | str | int]:
    """E_out first-hit at ``eps = 1/n``, ``L = 0``, inside ``T = n^2/8``."""
    step = _step_for(n)
    hit = certify_e_out(
        L=_KILL_L,
        eps=Fraction(1, n),
        max_steps=horizon_steps(n, step=step),
        step=step,
        order=_HIT_ORDER,
    )
    return {**hit, "n": n, "step": step}


@dataclass(frozen=True)
class EOutEpsReport:
    """Finite shrinking-eps E_out pack. Not uniform eps->0, GRAZING, or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str | int]]
    short: Mapping[str, float | bool | str]
    e_out_eps_pack: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-out-eps-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "e_out_eps_pack": self.e_out_eps_pack,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified matching-chart E_out first-hit on the finite "
                "shrinking pack eps=1/n for n in {16, 20, 25} at L=0 "
                "inside T=n^2/8. Not a uniform-in-eps theorem, not "
                "GRAZING E_sigma, not G1, or Hilbert XVI."
            ),
        }


def report() -> EOutEpsReport:
    """Replay matching identities, certify the shrinking pack, refuse a short horizon."""
    identities = identity_verdicts()
    pack = {str(n): certify_pack_member(n) for n in EPS_PACK}
    short = certify_e_out(
        L=_KILL_L,
        eps=_SAMPLE_EPS,
        max_steps=_SHORT_STEPS,
        step=0.5,
        order=_HIT_ORDER,
    )
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_negative"])
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and pack_ok
        and short["status"] != "certified"
    )
    return EOutEpsReport(
        identities=identities,
        pack=pack,
        short=short,
        e_out_eps_pack=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_out_eps_pack=sealed),
    )
