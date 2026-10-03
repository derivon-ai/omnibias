# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""L in {9/25, 1/16} wall-span GRAZING E_sigma cover.

The GRAZING reverse cubic from V=0, h=4 eps^3 has a certified
first-hit of V=1/4. On L in {9/25, 1/16} those return h-boxes sit
inside the declared rational span [17/1000, 7/200]. Split into
eighteen equal slabs of width 1/1000, certify_stopped_event hits
GRAZING E_sigma uniquely and transversely on every slab at both
L. A single slab over the whole span is unresolved. The GRAZING
start V=0 still excludes E_sigma on this compact horizon.

This is enclosure continuation of a declared span containing the
remaining L-pack wall boxes, not a single Lohner run from V=0,
not uniform in eps, not G1, and not Hilbert XVI. The L=0 whole
wall is e_sigma_span.
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
    "ESigmaPackReport",
    "KILL_L_PACK",
    "identity_verdicts",
    "report",
    "residual_pack_align",
    "residual_pack_hi",
    "residual_pack_lo",
    "residual_pack_width",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_H = Fraction(1, 40)
_H_LO = Fraction(17, 1000)
_H_HI = Fraction(7, 200)
_H_LO_MIL = Fraction(34, 2000)
_H_HI_MIL = Fraction(35, 1000)
_N_SLABS = 18
_SLAB = Fraction(1, 1000)
_ALIGN_GAP = Fraction(8, 1000)
_COARSE_SLABS = 1
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.125
_HIT_MAX_STEPS = 80
_HIT_ORDER = 8


def _honesty(*, e_sigma_pack_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_pack_hit=e_sigma_pack_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_pack_lo(h_lo: Fraction, mil: Fraction = _H_LO_MIL) -> Fraction:
    """Declared pack span lower endpoint ``17/1000 - 34/2000``."""
    return h_lo - mil


def residual_pack_hi(h_hi_mil: Fraction, seventh: Fraction = _H_HI) -> Fraction:
    """Declared pack span upper endpoint ``35/1000 - 7/200``."""
    return h_hi_mil - seventh


def residual_pack_width(
    h_lo: Fraction,
    h_hi: Fraction,
    n_slabs: int = _N_SLABS,
    slab: Fraction = _SLAB,
) -> Fraction:
    """Eighteen slabs of ``1/1000`` fill ``[17/1000, 7/200]``."""
    return n_slabs * slab - (h_hi - h_lo)


def residual_pack_align(
    align: Fraction,
    h_lo: Fraction,
    gap: Fraction = _ALIGN_GAP,
) -> Fraction:
    """``1/40 - 17/1000 - 8/1000``."""
    return (align - h_lo) - gap


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "pack_lo": _verdict(residual_pack_lo(_H_LO)),
        "pack_hi": _verdict(residual_pack_hi(_H_HI_MIL)),
        "pack_width": _verdict(residual_pack_width(_H_LO, _H_HI)),
        "pack_align": _verdict(residual_pack_align(_SAMPLE_H, _H_LO)),
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
class ESigmaPackReport:
    """L in {9/25, 1/16} wall-span E_sigma cover. Not Lohner-from-V=0 or G1."""

    identities: Mapping[str, str]
    walls: Mapping[str, Mapping[str, float | bool | str]]
    pack: Mapping[str, Mapping[str, object]]
    coarse: Mapping[str, object]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_pack_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-pack-v1",
            "identities": dict(self.identities),
            "walls": {key: dict(value) for key, value in self.walls.items()},
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_pack_hit": self.e_sigma_pack_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma on eighteen h-slabs covering [17/1000, 7/200] "
                "at V=1/4 on L in {9/25, 1/16}, a declared span "
                "containing both remaining L-pack V=1/4 wall boxes. A "
                "single slab over the whole span is unresolved. The "
                "GRAZING start V=0 still excludes E_sigma on this "
                "compact horizon. Not a single Lohner run from V=0, "
                "not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaPackReport:
    """Replay pack identities, enclose eighteen slabs at both L, refuse coarse and V=0."""
    identities = identity_verdicts()
    walls = {str(L): certify_align_wall(L=L) for L in _KILL_L_PACK}
    pack = {
        str(L): _as_float_slabs(
            enclose_cover(L=L, n_slabs=_N_SLABS, h_lo=_H_LO, h_hi=_H_HI)
        )
        for L in _KILL_L_PACK
    }
    coarse = _as_float_slabs(
        enclose_cover(
            L=Fraction(9, 25),
            n_slabs=_COARSE_SLABS,
            h_lo=_H_LO,
            h_hi=_H_HI,
        )
    )
    from_start = certify_e_sigma(
        L=Fraction(9, 25),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
        step=_HIT_STEP,
        max_steps=_HIT_MAX_STEPS,
        order=_HIT_ORDER,
    )
    walls_ok = all(_wall_inside_span(row) for row in walls.values())
    pack_ok = all(bool(row["all_certified"]) for row in pack.values())
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and walls_ok
        and pack_ok
        and not bool(coarse["all_certified"])
        and from_start["status"] != "certified"
    )
    return ESigmaPackReport(
        identities=identities,
        walls=walls,
        pack=pack,
        coarse=coarse,
        from_grazing_start=from_start,
        e_sigma_pack_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_pack_hit=sealed),
    )
