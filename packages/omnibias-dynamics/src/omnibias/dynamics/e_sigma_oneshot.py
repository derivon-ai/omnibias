# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""One-shot Lohner GRAZING E_sigma from V=0 at eps=1/16.

The reverse cubic from the GRAZING start V=0, h=4 eps^3, with
step=1/4 and max_steps=280, has unique transverse first-hit of
GRAZING E_sigma on L in {9/25, 1/16, 0}. A short horizon of 200
steps does not certify. Coarser step=1/2 wraps.

This is a single certify_stopped_event from V=0 at one eps, not a
uniform-in-eps theorem, not Z_x C2, not G1, and not Hilbert XVI.
The shrinking-eps aligned restart pack is e_sigma_eps.
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
    "ESigmaOneshotReport",
    "KILL_L_PACK",
    "identity_verdicts",
    "report",
    "residual_oneshot_T",
    "residual_oneshot_h0",
    "residual_oneshot_pack_L",
    "residual_oneshot_short",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_H0 = Fraction(1, 1024)
_HIT_STEP = 0.25
_HIT_MAX_STEPS = 280
_HIT_ORDER = 8
_SHORT_STEPS = 200
_HORIZON = Fraction(70)
_SHORT_HORIZON = Fraction(50)
_PACK_L = Fraction(169, 400)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK


def _honesty(*, e_sigma_oneshot_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_oneshot_hit=e_sigma_oneshot_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_oneshot_h0(h0: Fraction, eps: Fraction = _SAMPLE_EPS) -> Fraction:
    """GRAZING start ``h(0) - 4 eps^3``."""
    return h0 - 4 * eps**3


def residual_oneshot_T(
    steps: Fraction,
    step: Fraction,
    horizon: Fraction = _HORIZON,
) -> Fraction:
    """One-shot compact ``T = 280 * (1/4) = 70``."""
    return steps * step - horizon


def residual_oneshot_short(
    steps: Fraction,
    step: Fraction,
    horizon: Fraction = _SHORT_HORIZON,
) -> Fraction:
    """Short horizon ``T = 200 * (1/4) = 50``."""
    return steps * step - horizon


def residual_oneshot_pack_L(
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
    step = Fraction(1, 4)
    return {
        "oneshot_h0": _verdict(residual_oneshot_h0(_SAMPLE_H0)),
        "oneshot_T": _verdict(residual_oneshot_T(Fraction(_HIT_MAX_STEPS), step)),
        "oneshot_short": _verdict(residual_oneshot_short(Fraction(_SHORT_STEPS), step)),
        "oneshot_pack_L": _verdict(residual_oneshot_pack_L(Fraction(9, 25), Fraction(1, 16))),
    }


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class ESigmaOneshotReport:
    """One-shot Lohner E_sigma from V=0. Not uniform, Z_x, or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    e_sigma_oneshot_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-oneshot-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "e_sigma_oneshot_hit": self.e_sigma_oneshot_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma from the GRAZING start V=0, h=4 eps^3, on "
                "L in {9/25, 1/16, 0} at eps=1/16 in a single "
                "certify_stopped_event with step=1/4 and max_steps=280. "
                "A short horizon of 200 steps does not certify. Not "
                "uniform in eps, not Z_x C2, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaOneshotReport:
    """Replay identities, certify one-shot E_sigma on the L-pack, refuse a short horizon."""
    identities = identity_verdicts()
    h0 = 4 * _SAMPLE_EPS**3
    pack = {
        str(L): certify_e_sigma(
            L=L,
            v0=Fraction(0),
            h0=h0,
            eps=_SAMPLE_EPS,
            step=_HIT_STEP,
            max_steps=_HIT_MAX_STEPS,
            order=_HIT_ORDER,
        )
        for L in _KILL_L_PACK
    }
    short = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=h0,
        eps=_SAMPLE_EPS,
        step=_HIT_STEP,
        max_steps=_SHORT_STEPS,
        order=_HIT_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
    )
    return ESigmaOneshotReport(
        identities=identities,
        pack=pack,
        short=short,
        e_sigma_oneshot_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_oneshot_hit=sealed),
    )
