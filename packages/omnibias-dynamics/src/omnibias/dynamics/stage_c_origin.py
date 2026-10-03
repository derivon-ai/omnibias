# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Kill-line matching-chart Lohner pack toward chart O (r1 -> 0).

On lambda1 = -2, r1 = 1 - sep/2, the origin layer is sep -> 2,
r1 -> 0. From the compact post-corridor start (x, y) = (1/4, 1),
certify_stopped_event hits matching-chart x = 4 uniquely and
transversely on sep in {3/2, 7/4, 2} (r1 in {1/4, 1/8, 0}) with
step = 1/20 and max_steps = 160. A short horizon of 120 steps does
not certify at sep = 2.

This is a finite origin pack from a compact x-start, not every r1,
not the shrinking interface x = r1(1+theta), not complete first-hit
on chart O, C2, G1, or Hilbert XVI. The sep in {0, 3/5, 1} pack is
stage_c_oneshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.stage_b import r1_kill
from omnibias.dynamics.stage_c_oneshot import certify_stage_c_eout

__all__ = [
    "SEP_O_PACK",
    "StageCOriginReport",
    "identity_verdicts",
    "report",
    "residual_r1_o_eighth",
    "residual_r1_o_quarter",
    "residual_r1_origin",
]

_SEP_O_PACK = (Fraction(3, 2), Fraction(7, 4), Fraction(2))
SEP_O_PACK = _SEP_O_PACK
_HIT_MAX_STEPS = 160
_SHORT_STEPS = 120


def _honesty(*, stage_c_origin: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        stage_c_oneshot=False,
        stage_c_eps_span=False,
        stage_c_origin=stage_c_origin,
        outgoing_first_hit=False,
        hk_theorem_24_used=False,
    )


def residual_r1_origin() -> Fraction:
    """``1 - 2/2 = 0``: chart-O limit ``r1 = 0`` at ``sep = 2``."""
    return r1_kill(Fraction(2))


def residual_r1_o_quarter() -> Fraction:
    """``1 - (3/2)/2 = 1/4``: pack member ``sep = 3/2``."""
    return r1_kill(Fraction(3, 2)) - Fraction(1, 4)


def residual_r1_o_eighth() -> Fraction:
    """``1 - (7/4)/2 = 1/8``: pack member ``sep = 7/4``."""
    return r1_kill(Fraction(7, 4)) - Fraction(1, 8)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "r1_origin": _verdict(residual_r1_origin()),
        "r1_o_quarter": _verdict(residual_r1_o_quarter()),
        "r1_o_eighth": _verdict(residual_r1_o_eighth()),
    }


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        and bool(row["hits_section"])
    )


@dataclass(frozen=True)
class StageCOriginReport:
    """Chart-O Lohner pack at r1 in {1/4, 1/8, 0}. Not complete first-hit or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    stage_c_origin: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-stage-c-origin-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "stage_c_origin": self.stage_c_origin,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of matching-chart "
                "x=4 from compact start (x,y)=(1/4,1) on sep in "
                "{3/2, 7/4, 2} (r1 in {1/4, 1/8, 0}) at eps=1/16, "
                "step=1/20, max_steps=160. A short horizon of 120 steps "
                "at sep=2 does not certify. Not every r1, not the "
                "shrinking interface, not complete first-hit on chart O, "
                "C2, G1, or Hilbert XVI."
            ),
        }


def report() -> StageCOriginReport:
    """Replay identities, certify the origin sep-pack, refuse a short horizon."""
    identities = identity_verdicts()
    pack = {
        str(sep): certify_stage_c_eout(sep=sep, max_steps=_HIT_MAX_STEPS)
        for sep in _SEP_O_PACK
    }
    short = certify_stage_c_eout(sep=Fraction(2), max_steps=_SHORT_STEPS)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and all(_hit_ok(row) for row in pack.values())
        and short["status"] != "certified"
    )
    return StageCOriginReport(
        identities=identities,
        pack=pack,
        short=short,
        stage_c_origin=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(stage_c_origin=sealed),
    )
