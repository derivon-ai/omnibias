# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Nearer-interface parametric-sep matching-chart Lohner cover of [7/4, 2].

On lambda1 = -2, with sep a PolynomialFlow parameter, the matching-chart
height flow from (x, y) = (1/8, 1) has unique transverse first-hit of
x = 4 on eight slabs of width 1/32 covering [7/4, 2] (r1 from 1/8 down
to 0). At the left endpoint r1 = 1/8 equals the start x. A single slab
over the whole compact is unresolved.

This is enclosure continuation of a declared sep compact containing
chart-O sep = 2 from a start closer to the first root than x = 1/4,
not every r1, not the shrinking interface x = r1(1+theta), not complete
first-hit on chart O, C2, G1, or Hilbert XVI. The x = 1/4 cover of
[3/2, 2] is stage_c_origin_span.
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
    StoppedEventRequest,
    StoppedEventResult,
    certify_stopped_event,
    verify_stopped_event,
)
from omnibias.dynamics.stage_c_origin_span import matching_height_flow_sep_param

__all__ = [
    "StageCOriginIfaceReport",
    "certify_stage_c_origin_iface",
    "enclose_cover",
    "identity_verdicts",
    "report",
    "residual_oiface_join",
    "residual_oiface_n32",
    "residual_oiface_slabs",
]

_SEP_LO = Fraction(7, 4)
_SEP_HI = Fraction(2)
_N_SLABS = 8
_SLAB = Fraction(1, 32)
_WIDTH = Fraction(1, 4)
_X0 = Fraction(1, 8)
_Y1 = Fraction(1)
_X_SEC = Fraction(4)
_HIT_STEP = 0.05
_HIT_MAX_STEPS = 160
_HIT_ORDER = 6
_COARSE_SLABS = 1


def _honesty(*, stage_c_origin_iface: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_origin_span=False,
        stage_c_origin_iface=stage_c_origin_iface,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_oiface_join() -> Fraction:
    """``7/4 + 1/4 = 2``: declared cover closes at chart-O sep."""
    return _SEP_LO + _WIDTH - _SEP_HI


def residual_oiface_slabs() -> Fraction:
    """``8 * (1/32) = 1/4``: eight equal slabs fill the cover."""
    return Fraction(_N_SLABS) * _SLAB - _WIDTH


def residual_oiface_n32() -> Fraction:
    """``32 * (1/4) = 8``: the cover width in units of the slab."""
    return Fraction(32) * _WIDTH - Fraction(8)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "oiface_join": _verdict(residual_oiface_join()),
        "oiface_slabs": _verdict(residual_oiface_slabs()),
        "oiface_n32": _verdict(residual_oiface_n32()),
    }


def enclose_cover() -> dict[str, Fraction]:
    """Declared eight-slab cover of ``[7/4, 2]`` from ``x = 1/8``."""
    return {
        "lo": _SEP_LO,
        "hi": _SEP_HI,
        "width": _WIDTH,
        "slab": _SLAB,
        "n_slabs": Fraction(_N_SLABS),
    }


def _slabs(n_slabs: int) -> Sequence[tuple[Fraction, Fraction]]:
    return tuple((_SEP_LO + i * _SLAB, _SEP_LO + (i + 1) * _SLAB) for i in range(n_slabs))


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
        "hits_section": (
            x_lo <= float(_X_SEC) <= x_hi if x_lo == x_lo and x_hi == x_hi else False
        ),
    }


def certify_stage_c_origin_iface(
    *,
    sep_lo: Fraction = _SEP_LO,
    sep_hi: Fraction = _SEP_HI,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``x = 4`` from ``x = 1/8`` on a sep interval."""
    request = StoppedEventRequest(
        flow=matching_height_flow_sep_param(),
        initial=(
            SparsePolynomial.constant(1, _X0),
            SparsePolynomial.constant(1, _Y1),
        ),
        parameters=(Interval.hull(Interval.from_rational(sep_lo), Interval.from_rational(sep_hi)),),
        target=PolynomialEvent(
            SparsePolynomial.variable(4, 0) - _X_SEC,
            direction=1,
            name="Xsec",
        ),
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
        and bool(row["hits_section"])
    )


@dataclass(frozen=True)
class StageCOriginIfaceReport:
    """Nearer-interface chart-O Lohner cover. Not every r1 or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    coarse: Mapping[str, float | bool | str]
    stage_c_origin_iface: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        cover = enclose_cover()
        return {
            "schema": "hilbert16-stage-c-origin-iface-v1",
            "identities": dict(self.identities),
            "cover": {key: str(value) for key, value in cover.items()},
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "stage_c_origin_iface": self.stage_c_origin_iface,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of matching-chart "
                "x=4 from start (x,y)=(1/8,1) on eight slabs of width "
                "1/32 covering [7/4, 2] (r1 from 1/8 down to 0) at "
                "eps=1/16. A single slab over that compact is unresolved. "
                "Not every r1, not the shrinking interface, not complete "
                "first-hit on chart O, C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCOriginIfaceReport:
    """Replay identities, certify eight slabs, refuse a coarse one-slab cover."""
    identities = identity_verdicts()
    pack = {
        f"{lo}-{hi}": certify_stage_c_origin_iface(sep_lo=lo, sep_hi=hi)
        for lo, hi in _slabs(_N_SLABS)
    }
    coarse = certify_stage_c_origin_iface()
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and coarse["status"] != "certified"
        and len(pack) == _N_SLABS
        and _COARSE_SLABS == 1
    )
    return StageCOriginIfaceReport(
        identities=identities,
        pack=pack,
        coarse=coarse,
        stage_c_origin_iface=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_origin_iface=sealed),
    )
