# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Compact parametric-eps Stage-C matching-chart Lohner cover of [23/400, 1/16].

On lambda1 = -2, sep = 0, with eps a PolynomialFlow parameter, the
matching-chart height flow from (x, y) = (1/4, 1) has unique
transverse first-hit of 4 eps x - 1 = 0 (the image of E_out) on
four slabs of width 1/800 covering [23/400, 1/16]. A single slab
over the whole compact is unresolved.

This is enclosure continuation of a declared eps compact containing
eps = 1/16, not a uniform-in-eps theorem for every eps, not chart O,
C2, G1, or Hilbert XVI. The finite shrinking point pack is
stage_c_oneshot_eps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    StoppedEventResult,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "StageCEpsSpanReport",
    "certify_stage_c_eps_span",
    "enclose_cover",
    "identity_verdicts",
    "matching_height_flow_eps_param",
    "report",
    "residual_cspan_join",
    "residual_cspan_n800",
    "residual_cspan_slabs",
]

_EPS_HI = Fraction(1, 16)
_N_SLABS = 4
_SLAB = Fraction(1, 800)
_WIDTH = Fraction(1, 200)
_EPS_LO = Fraction(23, 400)
_X0 = Fraction(1, 4)
_Y1 = Fraction(1)
_HIT_STEP = 0.05
_HIT_MAX_STEPS = 140
_HIT_ORDER = 6
_COARSE_SLABS = 1


def _honesty(*, stage_c_eps_span: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_oneshot_eps=False,
        stage_c_eps_span=stage_c_eps_span,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_cspan_join() -> Fraction:
    """``23/400 + 1/200 = 1/16``: declared cover closes at the compact edge."""
    return _EPS_LO + _WIDTH - _EPS_HI


def residual_cspan_slabs() -> Fraction:
    """``4 * (1/800) = 1/200``: four equal slabs fill the cover."""
    return Fraction(_N_SLABS) * _SLAB - _WIDTH


def residual_cspan_n800() -> Fraction:
    """``800 * (1/16) = 50``: the compact edge in units of the slab."""
    return Fraction(800) * _EPS_HI - Fraction(50)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "cspan_join": _verdict(residual_cspan_join()),
        "cspan_slabs": _verdict(residual_cspan_slabs()),
        "cspan_n800": _verdict(residual_cspan_n800()),
    }


def matching_height_flow_eps_param() -> PolynomialFlow:
    """Kill-line sep=0 matching-chart height flow with one parameter slot for eps."""
    nvars = 4
    x_p = SparsePolynomial.variable(nvars, 0)
    y_p = SparsePolynomial.variable(nvars, 1)
    e_p = SparsePolynomial.variable(nvars, 2)
    dx = e_p * y_p + e_p * (x_p - 1) * (x_p - 1)
    dy = x_p * y_p
    return PolynomialFlow((dx, dy), 1)


def _x_section_poly() -> SparsePolynomial:
    """``4 eps x - 1``: matching-chart image of E_out."""
    x_p = SparsePolynomial.variable(4, 0)
    e_p = SparsePolynomial.variable(4, 2)
    return SparsePolynomial.constant(4, 4) * e_p * x_p - 1


def enclose_cover() -> dict[str, Fraction]:
    """Declared four-slab cover of ``[23/400, 1/16]``."""
    return {
        "lo": _EPS_LO,
        "hi": _EPS_HI,
        "width": _WIDTH,
        "slab": _SLAB,
        "n_slabs": Fraction(_N_SLABS),
    }


def _slabs(n_slabs: int) -> Sequence[tuple[Fraction, Fraction]]:
    return tuple((_EPS_LO + i * _SLAB, _EPS_LO + (i + 1) * _SLAB) for i in range(n_slabs))


def _hit_payload(result: StoppedEventResult) -> dict[str, float | bool | str]:
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
    }


def certify_stage_c_eps_span(
    *,
    eps_lo: Fraction = _EPS_LO,
    eps_hi: Fraction = _EPS_HI,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``4 eps x = 1`` on a closed initial-eps interval."""
    request = StoppedEventRequest(
        flow=matching_height_flow_eps_param(),
        initial=(
            SparsePolynomial.constant(1, _X0),
            SparsePolynomial.constant(1, _Y1),
        ),
        parameters=(Interval.hull(Interval.from_rational(eps_lo), Interval.from_rational(eps_hi)),),
        target=PolynomialEvent(_x_section_poly(), direction=1, name="Xsec"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    return _hit_payload(certify_stopped_event(request))


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class StageCEpsSpanReport:
    """Parametric-eps Stage-C Lohner cover. Not every eps or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    coarse: Mapping[str, float | bool | str]
    stage_c_eps_span: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        cover = enclose_cover()
        return {
            "schema": "hilbert16-stage-c-eps-span-v1",
            "identities": dict(self.identities),
            "cover": {key: str(value) for key, value in cover.items()},
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "stage_c_eps_span": self.stage_c_eps_span,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of matching-chart "
                "4 eps x - 1 = 0 from Stage-C start (x,y)=(1/4,1) on "
                "four slabs of width 1/800 covering [23/400, 1/16] at "
                "sep=0. A single slab over that compact is unresolved. "
                "Not uniform in eps, chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCEpsSpanReport:
    """Replay identities, certify four slabs, refuse a coarse one-slab cover."""
    identities = identity_verdicts()
    pack = {
        f"{lo}-{hi}": certify_stage_c_eps_span(eps_lo=lo, eps_hi=hi)
        for lo, hi in _slabs(_N_SLABS)
    }
    coarse = certify_stage_c_eps_span()
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and coarse["status"] != "certified"
        and len(pack) == _N_SLABS
        and _COARSE_SLABS == 1
    )
    return StageCEpsSpanReport(
        identities=identities,
        pack=pack,
        coarse=coarse,
        stage_c_eps_span=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_eps_span=sealed),
    )
