# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line Stage-C Lohner first-hit of matching-chart x = rho/eps.

On lambda1 = -2 the matching-chart height flow with independent
sigma, dy/d sigma = x y and

    dx/d sigma = eps y + eps (x-r1)(x-r2),

is polynomial. The section x = rho/eps = 4 is the leading image of
E_out. From the declared Stage-C start (x, y) = (1/4, 1),
certify_stopped_event hits x = 4 uniquely and transversely on
sep in {0, 3/5, 1} with step = 1/20 and max_steps = 120. A short
horizon of 80 steps does not certify at sep = 0.

This is a finite Lohner pack from Stage-C start, not a uniform-in-eps
theorem, not chart O, C2, dx_e off the kill line, G1, or Hilbert XVI.
The comparison first-hit of E_out is stage_c_sec. GRAZING-start Lohner
E_out is e_out_section.
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
from omnibias.dynamics.stage_b import r1_kill

__all__ = [
    "SEP_PACK",
    "StageCOneshotReport",
    "certify_stage_c_eout",
    "identity_verdicts",
    "matching_height_flow",
    "report",
    "residual_hit_T",
    "residual_short_T",
    "residual_x_section",
]

_EPS = Fraction(1, 16)
_RHO = Fraction(1, 4)
_X_SEC = Fraction(4)
_X0 = Fraction(1, 4)
_Y1 = Fraction(1)
_HIT_STEP = 0.05
_HIT_STEP_Q = Fraction(1, 20)
_HIT_MAX_STEPS = 120
_SHORT_STEPS = 80
_HIT_ORDER = 6
_HORIZON = Fraction(6)
_SHORT_HORIZON = Fraction(4)
_SEP_PACK = (Fraction(0), Fraction(3, 5), Fraction(1))
SEP_PACK = _SEP_PACK


def _honesty(*, stage_c_oneshot: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        orbit_continuation_on_kill=False,
        stage_a_wall=False,
        chi_b_bound=False,
        dx_e_leading=False,
        dx_e_unif=False,
        stage_c_amin=False,
        stage_c_exit=False,
        stage_c_th=False,
        stage_c_gap=False,
        stage_c_env=False,
        stage_c_if=False,
        stage_c_int=False,
        stage_c_lo=False,
        stage_c_k=False,
        stage_c_boot=False,
        stage_c_rect=False,
        stage_c_hit=False,
        stage_c_sec=False,
        stage_c_oneshot=stage_c_oneshot,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_x_section() -> Fraction:
    """``(1/4) / (1/16) = 4``: matching-chart image of E_out."""
    return _RHO / _EPS - _X_SEC


def residual_hit_T() -> Fraction:
    """``120 * (1/20) = 6``: compact Lohner horizon from x=1/4."""
    return Fraction(_HIT_MAX_STEPS) * _HIT_STEP_Q - _HORIZON


def residual_short_T() -> Fraction:
    """``80 * (1/20) = 4``: short horizon that does not certify."""
    return Fraction(_SHORT_STEPS) * _HIT_STEP_Q - _SHORT_HORIZON


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "x_section": _verdict(residual_x_section()),
        "hit_T": _verdict(residual_hit_T()),
        "short_T": _verdict(residual_short_T()),
    }


def matching_height_flow(*, sep: Fraction, eps: Fraction = _EPS) -> PolynomialFlow:
    """Polynomial matching-chart height flow on the kill line."""
    x_p = SparsePolynomial.variable(3, 0)
    y_p = SparsePolynomial.variable(3, 1)
    r1 = r1_kill(sep)
    r2 = Fraction(2) - r1
    prod = (x_p - r1) * (x_p - r2)
    dx = SparsePolynomial.constant(3, eps) * y_p + SparsePolynomial.constant(3, eps) * prod
    dy = x_p * y_p
    return PolynomialFlow((dx, dy), 0)


def certify_stage_c_eout(
    *,
    sep: Fraction = Fraction(0),
    x0: Fraction = _X0,
    y0: Fraction = _Y1,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of matching-chart ``x = 4`` from Stage-C start."""
    request = StoppedEventRequest(
        flow=matching_height_flow(sep=sep),
        initial=(
            SparsePolynomial.constant(0, x0),
            SparsePolynomial.constant(0, y0),
        ),
        parameters=(),
        target=PolynomialEvent(
            SparsePolynomial.variable(3, 0) - _X_SEC,
            direction=1,
            name="Xsec",
        ),
        step=step,
        max_steps=max_steps,
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
    return {
        "status": result.status,
        "replayed": replay,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "nvel_lo": nvel_lo,
        "transverse_positive": nvel_lo > 0.0 if nvel_lo == nvel_lo else False,
        "hits_section": (
            x_lo <= float(_X_SEC) <= x_hi if x_lo == x_lo and x_hi == x_hi else False
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
class StageCOneshotReport:
    """Stage-C Lohner first-hit of matching-chart x=4. Not every eps or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    stage_c_oneshot: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-oneshot-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "stage_c_oneshot": self.stage_c_oneshot,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of matching-chart "
                "x=rho/eps=4 from Stage-C start (x,y)=(1/4,1) on "
                "sep in {0, 3/5, 1} at eps=1/16, step=1/20, "
                "max_steps=120. A short horizon of 80 steps does not "
                "certify. Not uniform in eps, chart O, C2, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> StageCOneshotReport:
    """Replay identities, certify the sep-pack, refuse a short horizon."""
    identities = identity_verdicts()
    pack = {str(sep): certify_stage_c_eout(sep=sep) for sep in _SEP_PACK}
    short = certify_stage_c_eout(sep=Fraction(0), max_steps=_SHORT_STEPS)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
    )
    return StageCOneshotReport(
        identities=identities,
        pack=pack,
        short=short,
        stage_c_oneshot=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_oneshot=sealed),
    )
