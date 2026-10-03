# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite shrinking-eps pack of aligned GRAZING E_sigma first-hit.

On the kill line L = 0, lambda1 = -2, the aligned restart
(V, h) = (1/4, 1/40) lies in the certified GRAZING-from-V=0 V=1/4
return box at eps = 1/n for n in {16, 20, 25}. From that point,
certify_stopped_event hits GRAZING E_sigma uniquely and
transversely inside the declared compact T = 10. A short horizon
at n = 16 does not certify. The GRAZING start V=0 still excludes
E_sigma on this compact horizon.

This is a finite shrinking pack of an aligned-point restart, not a
wall-span cover at every n, not a single Lohner run from V=0, not
a uniform-in-eps theorem, not G1, and not Hilbert XVI. The L-pack
wall-span at one eps is e_sigma_pack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.e_sigma_wall import certify_align_wall
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "EPS_PACK",
    "ESigmaEpsReport",
    "identity_verdicts",
    "incoming_steps",
    "report",
    "residual_sigma_h0_16",
    "residual_sigma_h0_20",
    "residual_sigma_horizon",
    "residual_sigma_pack_sum",
]

EPS_PACK = (16, 20, 25)
_SAMPLE_N = 16
_SAMPLE_V = Fraction(1, 4)
_SAMPLE_H = Fraction(1, 40)
_KILL_L = Fraction(0)
_HIT_STEP = 0.125
_HIT_MAX_STEPS = 80
_HIT_ORDER = 8
_SHORT_STEPS = 16
_SHORT_STEP = 0.25
_HORIZON = Fraction(10)
_PACK_SUM = Fraction(61, 400)


def _honesty(*, e_sigma_eps_pack: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_eps_pack=e_sigma_eps_pack,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_sigma_pack_sum(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    total: Fraction = _PACK_SUM,
) -> Fraction:
    """Declared pack ``1/16 + 1/20 + 1/25 - 61/400``."""
    return (a + b + c) - total


def residual_sigma_h0_16(h0: Fraction, eps: Fraction = Fraction(1, 16)) -> Fraction:
    """GRAZING start ``h(0) - 4 eps^3`` at ``n = 16``."""
    return h0 - 4 * eps**3


def residual_sigma_h0_20(h0: Fraction, eps: Fraction = Fraction(1, 20)) -> Fraction:
    """GRAZING start ``h(0) - 4 eps^3`` at ``n = 20``."""
    return h0 - 4 * eps**3


def residual_sigma_horizon(
    steps: Fraction,
    step: Fraction,
    horizon: Fraction = _HORIZON,
) -> Fraction:
    """Aligned compact ``T = 80 * (1/8) = 10``."""
    return steps * step - horizon


def incoming_steps(n: int) -> int:
    """Integer step count covering the incoming ``V=1/4`` hit at ``eps=1/n``."""
    return n**3 // 50 + 16


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "sigma_pack_sum": _verdict(
            residual_sigma_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25))
        ),
        "sigma_h0_16": _verdict(residual_sigma_h0_16(Fraction(1, 1024))),
        "sigma_h0_20": _verdict(residual_sigma_h0_20(Fraction(1, 2000))),
        "sigma_horizon": _verdict(
            residual_sigma_horizon(Fraction(_HIT_MAX_STEPS), Fraction(1, 8))
        ),
    }


def _wall_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["contains_align"])
    )


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class ESigmaEpsReport:
    """Shrinking-eps aligned E_sigma pack. Not uniform, Lohner-from-V=0, or G1."""

    identities: Mapping[str, str]
    walls: Mapping[str, Mapping[str, float | bool | str]]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_eps_pack: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-eps-v1",
            "identities": dict(self.identities),
            "walls": {key: dict(value) for key, value in self.walls.items()},
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_eps_pack": self.e_sigma_eps_pack,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma from the aligned point (V,h)=(1/4,1/40) at "
                "eps=1/n for n in {16, 20, 25}, L=0, inside T=10. Each "
                "GRAZING-from-V=0 V=1/4 return box contains that point. "
                "A short horizon at n=16 is unresolved. The GRAZING "
                "start V=0 still excludes E_sigma on this compact "
                "horizon. Not a wall-span cover at every n, not a "
                "single Lohner run from V=0, not uniform in eps, not "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaEpsReport:
    """Replay pack identities, certify aligned E_sigma at three n, refuse short and V=0."""
    identities = identity_verdicts()
    walls = {
        str(n): certify_align_wall(
            L=_KILL_L,
            eps=Fraction(1, n),
            max_steps=incoming_steps(n),
        )
        for n in EPS_PACK
    }
    pack = {
        str(n): certify_e_sigma(
            L=_KILL_L,
            v0=_SAMPLE_V,
            h0=_SAMPLE_H,
            eps=Fraction(1, n),
            step=_HIT_STEP,
            max_steps=_HIT_MAX_STEPS,
            order=_HIT_ORDER,
        )
        for n in EPS_PACK
    }
    short = certify_e_sigma(
        L=_KILL_L,
        v0=_SAMPLE_V,
        h0=_SAMPLE_H,
        eps=Fraction(1, _SAMPLE_N),
        step=_SHORT_STEP,
        max_steps=_SHORT_STEPS,
        order=_HIT_ORDER,
    )
    from_start = certify_e_sigma(
        L=_KILL_L,
        v0=Fraction(0),
        h0=4 * Fraction(1, _SAMPLE_N) ** 3,
        step=_HIT_STEP,
        max_steps=_HIT_MAX_STEPS,
        order=_HIT_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_wall_ok(row) for row in walls.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
        and from_start["status"] != "certified"
    )
    return ESigmaEpsReport(
        identities=identities,
        walls=walls,
        pack=pack,
        short=short,
        from_grazing_start=from_start,
        e_sigma_eps_pack=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_eps_pack=sealed),
    )
