# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Orbit-aligned E_sigma first-hit from the certified V=1/4 wall box.

The GRAZING reverse cubic from V=0, h=4 eps^3 has a certified
first-hit of V=1/4 (e_sigma_in). On L in {9/25, 1/16, 0} that
return box contains the rational point (V, h)=(1/4, 1/40).
From that aligned restart, certify_stopped_event hits GRAZING
E_sigma uniquely and transversely. A short horizon does not
certify. The GRAZING start V=0 still excludes E_sigma on this
compact horizon.

This is a two-segment chain: certified wall from V=0, then a
point in that wall box to E_sigma. Not enclosure continuation
of the whole h-box, not a single Lohner run from V=0, not
uniform in eps, not G1, and not Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.e_sigma_in import reverse_cubic_flow
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "ESigmaWallReport",
    "KILL_L_PACK",
    "certify_align_wall",
    "identity_verdicts",
    "report",
    "residual_align_h",
    "residual_align_v",
    "residual_e_align_nu0",
    "residual_e_align_start",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(1, 4)
_SAMPLE_H = Fraction(1, 40)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_SAMPLE_E_START = Fraction(-619519, 819200)
_SAMPLE_E_NU0 = Fraction(-121, 160)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.25
_HIT_MAX_STEPS = 40
_HIT_ORDER = 8
_SHORT_STEPS = 16
_WALL_STEP = 1.0
_WALL_MAX_STEPS = 80
_WALL_ORDER = 6


def _honesty(*, e_sigma_wall_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_wall_hit=e_sigma_wall_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_align_v(v0: Fraction, wall: Fraction = _SAMPLE_V) -> Fraction:
    """Orbit-aligned restart ``V - 1/4``."""
    return v0 - wall


def residual_align_h(height: Fraction, start: Fraction = _SAMPLE_H) -> Fraction:
    """Orbit-aligned restart ``h - 1/40``."""
    return height - start


def residual_e_align_nu0(
    v_coord: Fraction, height: Fraction, sigma: Fraction, rho: Fraction
) -> Fraction:
    """``E_sigma`` at ``nu=0`` plus ``121/160`` at the aligned restart."""
    return e_sigma(v_coord, height, sigma, rho, Fraction(0), _SAMPLE_C) - _SAMPLE_E_NU0


def residual_e_align_start(
    v_coord: Fraction,
    height: Fraction,
    sigma: Fraction,
    rho: Fraction,
    nu: Fraction,
    c: Fraction,
) -> Fraction:
    """``E_sigma`` plus ``619519/819200`` at the aligned restart."""
    return e_sigma(v_coord, height, sigma, rho, nu, c) - _SAMPLE_E_START


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    v_coord, height = _SAMPLE_V, _SAMPLE_H
    rho, nu, c, sigma = _SAMPLE_RHO, _SAMPLE_EPS, _SAMPLE_C, _SAMPLE_SIGMA
    return {
        "align_v": _verdict(residual_align_v(v_coord)),
        "align_h": _verdict(residual_align_h(height)),
        "e_align_nu0": _verdict(residual_e_align_nu0(v_coord, height, sigma, rho)),
        "e_align_start": _verdict(
            residual_e_align_start(v_coord, height, sigma, rho, nu, c)
        ),
    }


def certify_align_wall(
    *,
    L: Fraction = Fraction(0),
    eps: Fraction | None = None,
    max_steps: int = _WALL_MAX_STEPS,
    step: float = _WALL_STEP,
    order: int = _WALL_ORDER,
) -> dict[str, float | bool | str]:
    """V=1/4 wall from GRAZING V=0, and whether it contains (1/4, 1/40)."""
    eps = _SAMPLE_EPS if eps is None else eps
    v_p = SparsePolynomial.variable(3, 0)
    request = StoppedEventRequest(
        flow=reverse_cubic_flow(L=L, eps=eps),
        initial=(
            SparsePolynomial.constant(0, Fraction(0)),
            SparsePolynomial.constant(0, 4 * eps**3),
        ),
        parameters=(),
        target=PolynomialEvent(v_p - _SAMPLE_V, direction=1, name="Vin"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    v_lo = float("nan")
    v_hi = float("nan")
    h_lo = float("nan")
    h_hi = float("nan")
    contains_align = False
    if result.return_box is not None:
        v_b, h_b = result.return_box
        v_lo, v_hi = v_b.lo, v_b.hi
        h_lo, h_hi = h_b.lo, h_b.hi
        contains_align = bool(v_b.contains(float(_SAMPLE_V)) and h_b.contains(float(_SAMPLE_H)))
    return {
        "status": result.status,
        "replayed": replay,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "h_lo": h_lo,
        "h_hi": h_hi,
        "contains_align": contains_align,
    }


@dataclass(frozen=True)
class ESigmaWallReport:
    """Orbit-aligned E_sigma from the V=1/4 wall box. Not Lohner-from-V=0 or G1."""

    identities: Mapping[str, str]
    walls: Mapping[str, Mapping[str, float | bool | str]]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_wall_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-wall-v1",
            "identities": dict(self.identities),
            "walls": {key: dict(value) for key, value in self.walls.items()},
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_wall_hit": self.e_sigma_wall_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma from the orbit-aligned point (V,h)=(1/4,1/40) "
                "inside the certified V=1/4 return box on L in "
                "{9/25, 1/16, 0}. A short horizon does not certify. "
                "The GRAZING start V=0 still excludes E_sigma on this "
                "compact horizon. Not enclosure continuation of the "
                "whole h-box, not a single Lohner run from V=0, not "
                "G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaWallReport:
    """Replay identities, contain 1/40 in the wall box, certify E_sigma, refuse short and V=0."""
    identities = identity_verdicts()
    walls = {str(L): certify_align_wall(L=L) for L in _KILL_L_PACK}
    pack = {
        str(L): certify_e_sigma(
            L=L,
            v0=_SAMPLE_V,
            h0=_SAMPLE_H,
            step=_HIT_STEP,
            max_steps=_HIT_MAX_STEPS,
            order=_HIT_ORDER,
        )
        for L in _KILL_L_PACK
    }
    short = certify_e_sigma(
        L=Fraction(0),
        v0=_SAMPLE_V,
        h0=_SAMPLE_H,
        step=_HIT_STEP,
        max_steps=_SHORT_STEPS,
        order=_HIT_ORDER,
    )
    from_start = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
        step=_HIT_STEP,
        max_steps=_HIT_MAX_STEPS,
        order=_HIT_ORDER,
    )
    walls_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["contains_align"])
        for row in walls.values()
    )
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and walls_ok
        and pack_ok
        and short["status"] != "certified"
        and from_start["status"] != "certified"
    )
    return ESigmaWallReport(
        identities=identities,
        walls=walls,
        pack=pack,
        short=short,
        from_grazing_start=from_start,
        e_sigma_wall_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_wall_hit=sealed),
    )
