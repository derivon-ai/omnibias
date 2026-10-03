# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Lower aligned parametric-eps GRAZING E_sigma cover of [1/64, 1/16].

On the reverse cubic with eps a PolynomialFlow parameter, the aligned
restart (V, h) = (1/4, 1/40) has unique transverse first-hit of
GRAZING E_sigma on six equal slabs of width 1/128 covering
[1/64, 1/16] at L = 0. That compact contains the previous
[1/25, 1/16] span and {1/16, 1/20, 1/25, 1/32, 1/64}. The last
slab containing eps = 1/16 certifies on L in {9/25, 1/16}. A
single slab over the whole compact is unresolved. The GRAZING
start V = 0, h = 4 eps^3 still excludes E_sigma on the last slab.

This is enclosure continuation of a declared lower eps compact from
the aligned restart, not a uniform-in-eps theorem for every eps,
not Lohner from V = 0 on that compact, not Z_x C2, not G1, and not
Hilbert XVI. The [1/25, 1/16] three-slab cover is e_sigma_eps_span.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_sigma_eps_span import (
    KILL_L_PACK,
    certify_e_sigma_eps_span,
    enclose_cover,
)
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ESigmaEpsLoReport",
    "identity_verdicts",
    "report",
    "residual_elo_contains_32",
    "residual_elo_lo",
    "residual_elo_pack_L",
    "residual_elo_width",
]

_EPS_LO = Fraction(1, 64)
_EPS_HI = Fraction(1, 16)
_EPS_MID = Fraction(1, 32)
_N_SLABS = 6
_SLAB = Fraction(1, 128)
_CONTAINS = Fraction(1, 2048)
_PACK_L = Fraction(169, 400)
_COARSE_SLABS = 1
_GRAZE_STEP = 0.25
_GRAZE_MAX_STEPS = 280
_GRAZE_ORDER = 8


def _honesty(*, e_sigma_eps_lo_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_eps_lo_hit=e_sigma_eps_lo_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_elo_lo(eps_lo: Fraction, declared: Fraction = _EPS_LO) -> Fraction:
    """Declared compact lower endpoint ``1/64``."""
    return eps_lo - declared


def residual_elo_width(
    eps_lo: Fraction,
    eps_hi: Fraction,
    n_slabs: int = _N_SLABS,
    slab: Fraction = _SLAB,
) -> Fraction:
    """Six slabs of ``1/128`` fill ``[1/64, 1/16]``."""
    return n_slabs * slab - (eps_hi - eps_lo)


def residual_elo_contains_32(
    mid: Fraction,
    eps_lo: Fraction,
    eps_hi: Fraction,
    product: Fraction = _CONTAINS,
) -> Fraction:
    """``(1/32-1/64)*(1/16-1/32) - 1/2048``."""
    return (mid - eps_lo) * (eps_hi - mid) - product


def residual_elo_pack_L(
    a: Fraction,
    b: Fraction,
    total: Fraction = _PACK_L,
) -> Fraction:
    """Declared L-pack ``9/25 + 1/16 - 169/400``."""
    return (a + b) - total


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "elo_lo": _verdict(residual_elo_lo(_EPS_LO)),
        "elo_width": _verdict(residual_elo_width(_EPS_LO, _EPS_HI)),
        "elo_contains_32": _verdict(residual_elo_contains_32(_EPS_MID, _EPS_LO, _EPS_HI)),
        "elo_pack_L": _verdict(residual_elo_pack_L(Fraction(9, 25), Fraction(1, 16))),
    }


def _as_float_slabs(payload: Mapping[str, object]) -> dict[str, object]:
    slabs = payload["slabs"]
    assert isinstance(slabs, Sequence)
    return {
        "n_slabs": payload["n_slabs"],
        "all_certified": payload["all_certified"],
        "slabs": [dict(row) for row in slabs],
    }


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class ESigmaEpsLoReport:
    """Lower aligned parametric-eps E_sigma cover. Not every eps or G1."""

    identities: Mapping[str, str]
    cover: Mapping[str, object]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    coarse: Mapping[str, object]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_eps_lo_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-eps-lo-v1",
            "identities": dict(self.identities),
            "cover": dict(self.cover),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_eps_lo_hit": self.e_sigma_eps_lo_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma on six eps-slabs covering [1/64, 1/16] from "
                "aligned (V,h)=(1/4,1/40) at L=0, a declared compact "
                "containing the previous [1/25, 1/16] span and "
                "{1/16, 1/20, 1/25, 1/32, 1/64}. The last slab "
                "containing eps=1/16 certifies on L in {9/25, 1/16}. "
                "A single slab over the whole compact is unresolved. "
                "The GRAZING start V=0 still excludes E_sigma on the "
                "last slab. Not a uniform-in-eps theorem for every eps, "
                "not Lohner from V=0 on this compact, not Z_x C2, not "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaEpsLoReport:
    """Replay lower-compact identities, enclose six slabs, refuse coarse and V=0."""
    identities = identity_verdicts()
    cover = _as_float_slabs(
        enclose_cover(n_slabs=_N_SLABS, eps_lo=_EPS_LO, eps_hi=_EPS_HI)
    )
    last_lo = _EPS_HI - _SLAB
    pack = {
        str(L): certify_e_sigma_eps_span(L=L, eps_lo=last_lo, eps_hi=_EPS_HI)
        for L in KILL_L_PACK
    }
    coarse = _as_float_slabs(
        enclose_cover(n_slabs=_COARSE_SLABS, eps_lo=_EPS_LO, eps_hi=_EPS_HI)
    )
    from_start = certify_e_sigma_eps_span(
        L=Fraction(0),
        eps_lo=last_lo,
        eps_hi=_EPS_HI,
        graze=True,
        step=_GRAZE_STEP,
        max_steps=_GRAZE_MAX_STEPS,
        order=_GRAZE_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(cover["all_certified"])
        and all(_hit_ok(row) for row in pack.values())
        and not bool(coarse["all_certified"])
        and from_start["status"] != "certified"
    )
    return ESigmaEpsLoReport(
        identities=identities,
        cover=cover,
        pack=pack,
        coarse=coarse,
        from_grazing_start=from_start,
        e_sigma_eps_lo_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_eps_lo_hit=sealed),
    )
