# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite shrinking-eps pack of Stage-C matching-chart Lohner first-hits.

On lambda1 = -2, sep = 0, the matching-chart height flow from
(x, y) = (1/4, 1) has unique transverse first-hit of x = rho/eps = n/4
at eps = 1/n for n in {16, 20, 25} with step = 1/20 and declared
horizons T = 6, 7, 8. A short horizon of 120 steps at n = 20 does
not certify.

This is a finite shrinking pack, not a uniform-in-eps theorem, not
chart O, C2, dx_e off the kill line, G1, or Hilbert XVI. The sep-pack
at one eps is stage_c_oneshot.
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
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)
from omnibias.dynamics.stage_c_oneshot import matching_height_flow

__all__ = [
    "EPS_PACK",
    "StageCOneshotEpsReport",
    "horizon_steps",
    "identity_verdicts",
    "report",
    "residual_ceps_T25",
    "residual_ceps_n20_sec",
    "residual_ceps_n25_sec",
]

EPS_PACK = (16, 20, 25)
_RHO = Fraction(1, 4)
_X0 = Fraction(1, 4)
_Y1 = Fraction(1)
_HIT_STEP = 0.05
_HIT_STEP_Q = Fraction(1, 20)
_HIT_ORDER = 6
_SHORT_STEPS = 120
_HORIZON_STEPS = {16: 120, 20: 140, 25: 160}
_X_SEC_20 = Fraction(5)
_X_SEC_25 = Fraction(25, 4)
_HORIZON_25 = Fraction(8)


def _honesty(*, stage_c_oneshot_eps: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_c_oneshot=False,
        stage_c_oneshot_eps=stage_c_oneshot_eps,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_ceps_n20_sec() -> Fraction:
    """``(1/4) / (1/20) = 5``: matching-chart image of E_out at n=20."""
    return _RHO / Fraction(1, 20) - _X_SEC_20


def residual_ceps_n25_sec() -> Fraction:
    """``(1/4) / (1/25) = 25/4``: matching-chart image of E_out at n=25."""
    return _RHO / Fraction(1, 25) - _X_SEC_25


def residual_ceps_T25() -> Fraction:
    """``160 * (1/20) = 8``: compact Lohner horizon at n=25."""
    return Fraction(_HORIZON_STEPS[25]) * _HIT_STEP_Q - _HORIZON_25


def horizon_steps(n: int) -> int:
    """Integer step count for the declared compact at ``eps=1/n``."""
    return _HORIZON_STEPS[n]


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "ceps_n20_sec": _verdict(residual_ceps_n20_sec()),
        "ceps_n25_sec": _verdict(residual_ceps_n25_sec()),
        "ceps_T25": _verdict(residual_ceps_T25()),
    }


def certify_stage_c_eout_n(
    *,
    n: int,
    max_steps: int | None = None,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of matching-chart ``x = n/4`` at ``eps=1/n``."""
    eps = Fraction(1, n)
    x_sec = _RHO / eps
    request = StoppedEventRequest(
        flow=matching_height_flow(sep=Fraction(0), eps=eps),
        initial=(
            SparsePolynomial.constant(0, _X0),
            SparsePolynomial.constant(0, _Y1),
        ),
        parameters=(),
        target=PolynomialEvent(
            SparsePolynomial.variable(3, 0) - x_sec,
            direction=1,
            name="Xsec",
        ),
        step=step,
        max_steps=horizon_steps(n) if max_steps is None else max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    x_lo = float("nan")
    x_hi = float("nan")
    y_lo = float("nan")
    nvel_lo = float("nan")
    if result.return_box is not None:
        x_b, y_b = result.return_box
        x_lo = x_b.lo
        x_hi = x_b.hi
        y_lo = y_b.lo
    if result.normal_velocity is not None:
        nvel_lo = result.normal_velocity.lo
    x_sec_f = float(x_sec)
    return {
        "status": result.status,
        "replayed": replay,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "nvel_lo": nvel_lo,
        "transverse_positive": nvel_lo > 0.0 if nvel_lo == nvel_lo else False,
        "hits_section": (
            x_lo <= x_sec_f <= x_hi if x_lo == x_lo and x_hi == x_hi else False
        ),
    }


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        and bool(row["hits_section"])
    )


@dataclass(frozen=True)
class StageCOneshotEpsReport:
    """Shrinking-eps Stage-C Lohner pack. Not every eps or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    stage_c_oneshot_eps: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-oneshot-eps-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "stage_c_oneshot_eps": self.stage_c_oneshot_eps,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of matching-chart "
                "x=rho/eps=n/4 from Stage-C start (x,y)=(1/4,1) at "
                "eps=1/n for n in {16, 20, 25}, sep=0, step=1/20. "
                "A short horizon of 120 steps at n=20 does not certify. "
                "Not uniform in eps, chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCOneshotEpsReport:
    """Replay identities, certify the n-pack, refuse a short horizon at n=20."""
    identities = identity_verdicts()
    pack = {str(n): certify_stage_c_eout_n(n=n) for n in EPS_PACK}
    short = certify_stage_c_eout_n(n=20, max_steps=_SHORT_STEPS)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
    )
    return StageCOneshotEpsReport(
        identities=identities,
        pack=pack,
        short=short,
        stage_c_oneshot_eps=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_oneshot_eps=sealed),
    )
