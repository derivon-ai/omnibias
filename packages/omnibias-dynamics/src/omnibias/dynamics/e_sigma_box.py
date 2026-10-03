# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Enclosure continuation of a wall-box h-interval to GRAZING E_sigma.

The GRAZING reverse cubic from V=0, h=4 eps^3 has a certified
first-hit of V=1/4 whose h-boxes on L in {9/25, 1/16, 0} all
contain the rational interval [1/50, 4/125]. Split into twelve
equal slabs of width 1/1000, certify_stopped_event hits GRAZING
E_sigma uniquely and transversely on every slab at L=0, and on
the aligned slab containing h=1/40 at L in {9/25, 1/16}. A single
slab over the whole interval is unresolved. The GRAZING start
V=0 still excludes E_sigma on this compact horizon.

This is enclosure continuation of a declared rational sub-box of
the wall intersection, not the whole wall h-interval, not a
single Lohner run from V=0, not uniform in eps, not G1, and not
Hilbert XVI. The orbit-aligned point restart is e_sigma_wall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.e_sigma_wall import certify_align_wall
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "ESigmaBoxReport",
    "KILL_L_PACK",
    "certify_e_sigma_box",
    "enclose_cover",
    "identity_verdicts",
    "report",
    "residual_cover_contains",
    "residual_cover_hi",
    "residual_cover_lo",
    "residual_cover_width",
    "reverse_cubic_flow_param",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(1, 4)
_SAMPLE_H = Fraction(1, 40)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_H_LO = Fraction(1, 50)
_H_HI = Fraction(4, 125)
_H_LO_MIL = Fraction(20, 1000)
_H_HI_MIL = Fraction(32, 1000)
_N_SLABS = 12
_SLAB = Fraction(1, 1000)
_COVER_CONTAINS = Fraction(7, 200000)
_COARSE_SLABS = 1
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.125
_HIT_MAX_STEPS = 80
_HIT_ORDER = 8


def _honesty(*, e_sigma_box_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_box_hit=e_sigma_box_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_cover_lo(h_lo: Fraction, mil: Fraction = _H_LO_MIL) -> Fraction:
    """Declared cover lower endpoint ``1/50 - 20/1000``."""
    return h_lo - mil


def residual_cover_hi(h_hi: Fraction, mil: Fraction = _H_HI_MIL) -> Fraction:
    """Declared cover upper endpoint ``4/125 - 32/1000``."""
    return h_hi - mil


def residual_cover_width(
    h_lo: Fraction,
    h_hi: Fraction,
    n_slabs: int = _N_SLABS,
    slab: Fraction = _SLAB,
) -> Fraction:
    """Twelve slabs of ``1/1000`` fill ``[1/50, 4/125]``."""
    return n_slabs * slab - (h_hi - h_lo)


def residual_cover_contains(
    align: Fraction,
    h_lo: Fraction,
    h_hi: Fraction,
    product: Fraction = _COVER_CONTAINS,
) -> Fraction:
    """``(1/40-1/50)*(4/125-1/40) - 7/200000``."""
    return (align - h_lo) * (h_hi - align) - product


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    return {
        "cover_lo": _verdict(residual_cover_lo(_H_LO)),
        "cover_hi": _verdict(residual_cover_hi(_H_HI)),
        "cover_width": _verdict(residual_cover_width(_H_LO, _H_HI)),
        "cover_contains": _verdict(residual_cover_contains(_SAMPLE_H, _H_LO, _H_HI)),
    }


def reverse_cubic_flow_param(
    *, L: Fraction | None = None, eps: Fraction | None = None
) -> PolynomialFlow:
    """Time-reversed cubic ``(V,h)`` field with one parameter slot for initial ``h``."""
    nvars = 4
    v_p = SparsePolynomial.variable(nvars, 0)
    h_p = SparsePolynomial.variable(nvars, 1)
    eps = _SAMPLE_EPS if eps is None else eps
    nu = eps
    lam1 = Fraction(-2)
    L = Fraction(0) if L is None else L
    v2 = v_p * v_p
    v3 = v2 * v_p
    field_f = (
        SparsePolynomial.constant(nvars, -L * eps**3)
        + SparsePolynomial.constant(nvars, lam1 * eps**2) * v_p
        + SparsePolynomial.constant(nvars, -eps) * v2
        + SparsePolynomial.constant(nvars, eps / 3) * v3
    )
    field_g = SparsePolynomial.constant(nvars, -1) + SparsePolynomial.constant(nvars, nu) * (
        v_p - 1
    )
    vdot = field_f + h_p * field_g
    hdot = (SparsePolynomial.constant(nvars, -1) * v_p) * h_p
    return PolynomialFlow((-vdot, -hdot), 1)


def _e_sigma_poly4(*, nu: Fraction | None = None) -> SparsePolynomial:
    v_p = SparsePolynomial.variable(4, 0)
    h_p = SparsePolynomial.variable(4, 1)
    rho, c, sigma = _SAMPLE_RHO, _SAMPLE_C, _SAMPLE_SIGMA
    nu = _SAMPLE_EPS if nu is None else nu
    return (
        v_p
        - 1
        + SparsePolynomial.constant(4, sigma * rho) * h_p
        + SparsePolynomial.constant(4, nu * rho * rho) * (h_p * h_p)
        + SparsePolynomial.constant(4, c * nu * nu * sigma * rho) * (h_p * h_p)
    )


def certify_e_sigma_box(
    *,
    L: Fraction = Fraction(0),
    h_lo: Fraction = _H_LO,
    h_hi: Fraction = _H_HI,
    eps: Fraction | None = None,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``E_sigma`` on a closed initial-``h`` interval."""
    eps = _SAMPLE_EPS if eps is None else eps
    hpar = SparsePolynomial.variable(1, 0)
    request = StoppedEventRequest(
        flow=reverse_cubic_flow_param(L=L, eps=eps),
        initial=(SparsePolynomial.constant(1, _SAMPLE_V), hpar),
        parameters=(Interval.hull(Interval.from_rational(h_lo), Interval.from_rational(h_hi)),),
        target=PolynomialEvent(_e_sigma_poly4(nu=eps), direction=1, name="Esigma"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    v_lo = float("nan")
    h_lo_out = float("nan")
    nvel_lo = float("nan")
    if result.return_box is not None:
        v_b, h_b = result.return_box
        v_lo = v_b.lo
        h_lo_out = h_b.lo
    if result.normal_velocity is not None:
        nvel_lo = result.normal_velocity.lo
    return {
        "status": result.status,
        "replayed": replay,
        "v_lo": v_lo,
        "h_lo": h_lo_out,
        "nvel_lo": nvel_lo,
        "transverse_positive": nvel_lo > 0.0 if nvel_lo == nvel_lo else False,
    }


def enclose_cover(
    *,
    L: Fraction = Fraction(0),
    n_slabs: int = _N_SLABS,
    h_lo: Fraction = _H_LO,
    h_hi: Fraction = _H_HI,
) -> dict[str, object]:
    """Split ``[h_lo, h_hi]`` into equal slabs; each must certify ``E_sigma``."""
    slabs: list[dict[str, float | bool | str]] = []
    all_certified = True
    width = (h_hi - h_lo) / n_slabs
    for index in range(n_slabs):
        lo = h_lo + index * width
        hi = h_lo + (index + 1) * width
        row = certify_e_sigma_box(L=L, h_lo=lo, h_hi=hi)
        slabs.append(row)
        if not (
            row["status"] == "certified"
            and bool(row["replayed"])
            and bool(row["transverse_positive"])
        ):
            all_certified = False
    return {
        "n_slabs": n_slabs,
        "all_certified": all_certified,
        "slabs": slabs,
    }


def _as_float_slabs(payload: Mapping[str, object]) -> dict[str, object]:
    slabs = payload["slabs"]
    assert isinstance(slabs, Sequence)
    return {
        "n_slabs": payload["n_slabs"],
        "all_certified": payload["all_certified"],
        "slabs": [dict(row) for row in slabs],
    }


def _cover_inside_wall(wall: Mapping[str, float | bool | str]) -> bool:
    return (
        wall["status"] == "certified"
        and bool(wall["replayed"])
        and float(wall["h_lo"]) <= float(_H_LO)
        and float(_H_HI) <= float(wall["h_hi"])
    )


@dataclass(frozen=True)
class ESigmaBoxReport:
    """Wall-box h-interval E_sigma cover. Not Lohner-from-V=0 or G1."""

    identities: Mapping[str, str]
    walls: Mapping[str, Mapping[str, float | bool | str]]
    cover: Mapping[str, object]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    coarse: Mapping[str, object]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_box_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-box-v1",
            "identities": dict(self.identities),
            "walls": {key: dict(value) for key, value in self.walls.items()},
            "cover": dict(self.cover),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_box_hit": self.e_sigma_box_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma on twelve h-slabs covering [1/50, 4/125] at "
                "V=1/4, L=0, a declared rational sub-box of every "
                "L-pack V=1/4 wall box. The aligned slab containing "
                "h=1/40 certifies on L in {9/25, 1/16}. A single slab "
                "over the whole interval is unresolved. The GRAZING "
                "start V=0 still excludes E_sigma on this compact "
                "horizon. Not the whole wall h-interval, not a single "
                "Lohner run from V=0, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaBoxReport:
    """Replay cover identities, enclose twelve slabs, refuse coarse and V=0."""
    identities = identity_verdicts()
    walls = {str(L): certify_align_wall(L=L) for L in _KILL_L_PACK}
    cover = _as_float_slabs(enclose_cover())
    align_lo = _SAMPLE_H
    align_hi = _SAMPLE_H + _SLAB
    pack = {
        str(L): certify_e_sigma_box(L=L, h_lo=align_lo, h_hi=align_hi)
        for L in (Fraction(9, 25), Fraction(1, 16))
    }
    coarse = _as_float_slabs(enclose_cover(n_slabs=_COARSE_SLABS))
    from_start = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
        step=_HIT_STEP,
        max_steps=_HIT_MAX_STEPS,
        order=_HIT_ORDER,
    )
    walls_ok = all(_cover_inside_wall(row) for row in walls.values())
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and walls_ok
        and bool(cover["all_certified"])
        and pack_ok
        and not bool(coarse["all_certified"])
        and from_start["status"] != "certified"
    )
    return ESigmaBoxReport(
        identities=identities,
        walls=walls,
        cover=cover,
        pack=pack,
        coarse=coarse,
        from_grazing_start=from_start,
        e_sigma_box_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_box_hit=sealed),
    )
