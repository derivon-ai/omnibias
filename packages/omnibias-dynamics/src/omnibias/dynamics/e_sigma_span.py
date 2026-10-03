# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""L=0 whole-wall h-span GRAZING E_sigma cover.

The GRAZING reverse cubic from V=0, h=4 eps^3 has a certified
first-hit of V=1/4. On L=0 that return h-box sits inside the
declared rational span [19/1000, 1/25]. Split into twenty-one
equal slabs of width 1/1000, certify_stopped_event hits GRAZING
E_sigma uniquely and transversely on every slab. A single slab
over the whole span is unresolved. The GRAZING start V=0 still
excludes E_sigma on this compact horizon.

This is enclosure continuation of a declared span containing the
whole L=0 wall box, not the L in {9/25, 1/16} walls (those extend
below 19/1000), not a single Lohner run from V=0, not uniform in
eps, not G1, and not Hilbert XVI. The interior sub-box cover is
e_sigma_box. The remaining L-pack walls are e_sigma_pack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_sigma_box import enclose_cover
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.e_sigma_wall import certify_align_wall
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ESigmaSpanReport",
    "identity_verdicts",
    "report",
    "residual_span_align_hi",
    "residual_span_align_lo",
    "residual_span_hi",
    "residual_span_width",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_H = Fraction(1, 40)
_H_LO = Fraction(19, 1000)
_H_HI = Fraction(1, 25)
_N_SLABS = 21
_SLAB = Fraction(1, 1000)
_ALIGN_LO = Fraction(6, 1000)
_ALIGN_HI = Fraction(15, 1000)
_COARSE_SLABS = 1
_HIT_STEP = 0.125
_HIT_MAX_STEPS = 80
_HIT_ORDER = 8


def _honesty(*, e_sigma_span_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_span_hit=e_sigma_span_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_span_hi(h_hi_mil: Fraction, quarter: Fraction = Fraction(1, 25)) -> Fraction:
    """Declared span upper endpoint ``40/1000 - 1/25``."""
    return h_hi_mil - quarter


def residual_span_width(
    h_lo: Fraction,
    h_hi: Fraction,
    n_slabs: int = _N_SLABS,
    slab: Fraction = _SLAB,
) -> Fraction:
    """Twenty-one slabs of ``1/1000`` fill ``[19/1000, 1/25]``."""
    return n_slabs * slab - (h_hi - h_lo)


def residual_span_align_lo(
    align: Fraction,
    h_lo: Fraction,
    gap: Fraction = _ALIGN_LO,
) -> Fraction:
    """``1/40 - 19/1000 - 6/1000``."""
    return (align - h_lo) - gap


def residual_span_align_hi(
    h_hi: Fraction,
    align: Fraction,
    gap: Fraction = _ALIGN_HI,
) -> Fraction:
    """``1/25 - 1/40 - 15/1000``."""
    return (h_hi - align) - gap


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "span_hi": _verdict(residual_span_hi(Fraction(40, 1000))),
        "span_width": _verdict(residual_span_width(_H_LO, _H_HI)),
        "span_align_lo": _verdict(residual_span_align_lo(_SAMPLE_H, _H_LO)),
        "span_align_hi": _verdict(residual_span_align_hi(_H_HI, _SAMPLE_H)),
    }


def _as_float_slabs(payload: Mapping[str, object]) -> dict[str, object]:
    slabs = payload["slabs"]
    assert isinstance(slabs, Sequence)
    return {
        "n_slabs": payload["n_slabs"],
        "all_certified": payload["all_certified"],
        "slabs": [dict(row) for row in slabs],
    }


def _wall_inside_span(wall: Mapping[str, float | bool | str]) -> bool:
    return (
        wall["status"] == "certified"
        and bool(wall["replayed"])
        and float(_H_LO) <= float(wall["h_lo"])
        and float(wall["h_hi"]) <= float(_H_HI)
    )


@dataclass(frozen=True)
class ESigmaSpanReport:
    """L=0 whole-wall h-span E_sigma cover. Not Lohner-from-V=0 or G1."""

    identities: Mapping[str, str]
    wall: Mapping[str, float | bool | str]
    cover: Mapping[str, object]
    coarse: Mapping[str, object]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_span_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-span-v1",
            "identities": dict(self.identities),
            "wall": dict(self.wall),
            "cover": dict(self.cover),
            "coarse": dict(self.coarse),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_span_hit": self.e_sigma_span_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma on twenty-one h-slabs covering [19/1000, 1/25] "
                "at V=1/4, L=0, a declared span containing the whole "
                "L=0 V=1/4 wall box. A single slab over the whole span "
                "is unresolved. The GRAZING start V=0 still excludes "
                "E_sigma on this compact horizon. Not the L in {9/25, "
                "1/16} walls, not a single Lohner run from V=0, not G1, "
                "or Hilbert XVI."
            ),
        }


def report() -> ESigmaSpanReport:
    """Replay span identities, enclose twenty-one slabs, refuse coarse and V=0."""
    identities = identity_verdicts()
    wall = certify_align_wall(L=Fraction(0))
    cover = _as_float_slabs(enclose_cover(n_slabs=_N_SLABS, h_lo=_H_LO, h_hi=_H_HI))
    coarse = _as_float_slabs(
        enclose_cover(n_slabs=_COARSE_SLABS, h_lo=_H_LO, h_hi=_H_HI)
    )
    from_start = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
        step=_HIT_STEP,
        max_steps=_HIT_MAX_STEPS,
        order=_HIT_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and _wall_inside_span(wall)
        and bool(cover["all_certified"])
        and not bool(coarse["all_certified"])
        and from_start["status"] != "certified"
    )
    return ESigmaSpanReport(
        identities=identities,
        wall=wall,
        cover=cover,
        coarse=coarse,
        from_grazing_start=from_start,
        e_sigma_span_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_span_hit=sealed),
    )
