# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite shrinking-eps pack of one-shot Lohner GRAZING E_sigma from V=0.

On the kill line L = 0, the reverse cubic from GRAZING V=0,
h = 4 eps^3, with step = 1/4, has unique transverse first-hit of
GRAZING E_sigma at eps = 1/n for n in {16, 20, 25} inside the
declared horizons T = 70, 100, 250. A short horizon at n = 16
does not certify.

This is a finite shrinking pack of one-shot runs, not a
uniform-in-eps theorem, not Z_x C2, not G1, and not Hilbert XVI.
The L-pack at one eps is e_sigma_oneshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "EPS_PACK",
    "ESigmaOneshotEpsReport",
    "horizon_steps",
    "identity_verdicts",
    "report",
    "residual_oeps_T20",
    "residual_oeps_T25",
    "residual_oeps_h0_25",
    "residual_oeps_pack_sum",
]

EPS_PACK = (16, 20, 25)
_KILL_L = Fraction(0)
_HIT_STEP = 0.25
_HIT_ORDER = 8
_SHORT_STEPS = 200
_HORIZON_STEPS = {16: 280, 20: 400, 25: 1000}
_PACK_SUM = Fraction(61, 400)


def _honesty(*, e_sigma_oneshot_eps: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_oneshot_eps=e_sigma_oneshot_eps,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_oeps_pack_sum(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    total: Fraction = _PACK_SUM,
) -> Fraction:
    """Declared pack ``1/16 + 1/20 + 1/25 - 61/400``."""
    return (a + b + c) - total


def residual_oeps_T20(
    steps: Fraction,
    step: Fraction,
    horizon: Fraction = Fraction(100),
) -> Fraction:
    """n=20 compact ``T = 400 * (1/4) = 100``."""
    return steps * step - horizon


def residual_oeps_T25(
    steps: Fraction,
    step: Fraction,
    horizon: Fraction = Fraction(250),
) -> Fraction:
    """n=25 compact ``T = 1000 * (1/4) = 250``."""
    return steps * step - horizon


def residual_oeps_h0_25(h0: Fraction, eps: Fraction = Fraction(1, 25)) -> Fraction:
    """GRAZING start ``h(0) - 4 eps^3`` at ``n = 25``."""
    return h0 - 4 * eps**3


def horizon_steps(n: int) -> int:
    """Integer step count for the declared one-shot compact at ``eps=1/n``."""
    return _HORIZON_STEPS[n]


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    step = Fraction(1, 4)
    return {
        "oeps_pack_sum": _verdict(
            residual_oeps_pack_sum(Fraction(1, 16), Fraction(1, 20), Fraction(1, 25))
        ),
        "oeps_T20": _verdict(residual_oeps_T20(Fraction(400), step)),
        "oeps_T25": _verdict(residual_oeps_T25(Fraction(1000), step)),
        "oeps_h0_25": _verdict(residual_oeps_h0_25(Fraction(4, 15625))),
    }


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class ESigmaOneshotEpsReport:
    """Shrinking-eps one-shot E_sigma pack. Not uniform, Z_x, or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    e_sigma_oneshot_eps: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-oneshot-eps-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "e_sigma_oneshot_eps": self.e_sigma_oneshot_eps,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma from GRAZING V=0, h=4 eps^3, at eps=1/n for "
                "n in {16, 20, 25}, L=0, in a single "
                "certify_stopped_event per n. A short horizon at "
                "n=16 is unresolved. Not uniform in eps, not Z_x C2, "
                "not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaOneshotEpsReport:
    """Replay identities, certify one-shot E_sigma at three n, refuse a short horizon."""
    identities = identity_verdicts()
    pack = {
        str(n): certify_e_sigma(
            L=_KILL_L,
            v0=Fraction(0),
            h0=4 * Fraction(1, n) ** 3,
            eps=Fraction(1, n),
            step=_HIT_STEP,
            max_steps=horizon_steps(n),
            order=_HIT_ORDER,
        )
        for n in EPS_PACK
    }
    short = certify_e_sigma(
        L=_KILL_L,
        v0=Fraction(0),
        h0=4 * Fraction(1, 16) ** 3,
        eps=Fraction(1, 16),
        step=_HIT_STEP,
        max_steps=_SHORT_STEPS,
        order=_HIT_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
    )
    return ESigmaOneshotEpsReport(
        identities=identities,
        pack=pack,
        short=short,
        e_sigma_oneshot_eps=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_oneshot_eps=sealed),
    )
