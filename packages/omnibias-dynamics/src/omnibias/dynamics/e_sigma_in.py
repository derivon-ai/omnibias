# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Incoming GRAZING-chart V=1/4 first-hit on the reverse cubic.

GRAZING starts at V = 0, h = 4 eps^3. The reversed kill-line cubic
(L in {9/25, 1/16, 0}, lambda1 = -2, nu = eps) has unique transverse
first-hit of the declared wall V = 1/4. A short horizon does not
certify. Lohner wrapping still refuses E_sigma = V-1+... near V = 1.

Not certified E_sigma first-hit, not G1, and not Hilbert XVI. The
incoming comparison speed bound is e_sigma_speed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)
from omnibias.dynamics.vh_orbit import cubic_polynomial_flow, field_f, field_g

__all__ = [
    "ESigmaInReport",
    "KILL_L_PACK",
    "certify_vin_wall",
    "identity_verdicts",
    "report",
    "residual_hdot_rev",
    "residual_v0_grazing",
    "residual_vin_wall",
    "residual_vdot_rev_init",
    "reverse_cubic_flow",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(1, 8)
_SAMPLE_H = Fraction(1)
_SAMPLE_WALL = Fraction(1, 4)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 1.0
_HIT_MAX_STEPS = 80
_HIT_ORDER = 6
_SHORT_STEPS = 50


def _honesty(*, incoming_vwall_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        incoming_vwall_hit=incoming_vwall_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def reverse_cubic_flow(
    *, L: Fraction | None = None, eps: Fraction | None = None
) -> PolynomialFlow:
    """Time-reversed cubic ``(V,h)`` field."""
    fwd = cubic_polynomial_flow(L=L, eps=eps)
    return PolynomialFlow(tuple(-comp for comp in fwd.components), fwd.parameter_count)


def residual_v0_grazing(v0: Fraction) -> Fraction:
    """GRAZING start ``V(0)``."""
    return v0


def residual_hdot_rev(hdot_rev: Fraction, v_coord: Fraction, height: Fraction) -> Fraction:
    """Reverse ``hdot = V h``."""
    return hdot_rev - v_coord * height


def residual_vdot_rev_init(vdot_rev: Fraction, eps: Fraction) -> Fraction:
    """Kill-line reverse matching ``Vdot_rev - 4 eps^3 (1 + eps)`` at ``V=0``."""
    return vdot_rev - 4 * eps**3 * (1 + eps)


def residual_vin_wall(v_coord: Fraction, wall: Fraction) -> Fraction:
    """Declared incoming wall ``V - 1/4``."""
    return v_coord - wall


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    eps = _SAMPLE_EPS
    v0 = Fraction(0)
    h0 = 4 * eps**3
    vdot_fwd = field_f(Fraction(0), Fraction(-2), eps, v0) + h0 * field_g(eps, v0)
    vdot_rev = -vdot_fwd
    return {
        "v0_grazing": _verdict(residual_v0_grazing(v0)),
        "hdot_rev": _verdict(residual_hdot_rev(_SAMPLE_V * _SAMPLE_H, _SAMPLE_V, _SAMPLE_H)),
        "vdot_rev_init": _verdict(residual_vdot_rev_init(vdot_rev, eps)),
        "vin_wall": _verdict(residual_vin_wall(_SAMPLE_WALL, _SAMPLE_WALL)),
    }


def certify_vin_wall(
    *,
    L: Fraction = Fraction(0),
    wall: Fraction = _SAMPLE_WALL,
    eps: Fraction | None = None,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``V = wall`` on the reverse cubic."""
    eps = _SAMPLE_EPS if eps is None else eps
    v_p = SparsePolynomial.variable(3, 0)
    request = StoppedEventRequest(
        flow=reverse_cubic_flow(L=L, eps=eps),
        initial=(
            SparsePolynomial.constant(0, Fraction(0)),
            SparsePolynomial.constant(0, 4 * eps**3),
        ),
        parameters=(),
        target=PolynomialEvent(v_p - wall, direction=1, name="Vin"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    v_lo = float("nan")
    h_lo = float("nan")
    nvel_lo = float("nan")
    if result.return_box is not None:
        v_b, h_b = result.return_box
        v_lo = v_b.lo
        h_lo = h_b.lo
    if result.normal_velocity is not None:
        nvel_lo = result.normal_velocity.lo
    return {
        "status": result.status,
        "replayed": replay,
        "v_lo": v_lo,
        "h_lo": h_lo,
        "nvel_lo": nvel_lo,
        "transverse_positive": nvel_lo > 0.0 if nvel_lo == nvel_lo else False,
    }


@dataclass(frozen=True)
class ESigmaInReport:
    """Incoming V=1/4 first-hit. Not E_sigma, GRAZING first-hit, or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    incoming_vwall_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-in-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "incoming_vwall_hit": self.incoming_vwall_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of V=1/4 on the "
                "reverse cubic from the GRAZING start V=0, h=4 eps^3, "
                "on L in {9/25, 1/16, 0}. A short horizon does not "
                "certify. Lohner wrapping still refuses E_sigma near "
                "V=1. Not GRAZING first-hit, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaInReport:
    """Replay reverse-cubic identities, certify the L-pack, refuse a short horizon."""
    identities = identity_verdicts()
    pack = {str(L): certify_vin_wall(L=L) for L in _KILL_L_PACK}
    short = certify_vin_wall(L=Fraction(0), max_steps=_SHORT_STEPS)
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and pack_ok
        and short["status"] != "certified"
    )
    return ESigmaInReport(
        identities=identities,
        pack=pack,
        short=short,
        incoming_vwall_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(incoming_vwall_hit=sealed),
    )
