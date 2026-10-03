# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Compact aligned parametric-eps GRAZING E_sigma cover of [1/25, 1/16].

On the reverse cubic with eps a PolynomialFlow parameter, the aligned
restart (V, h) = (1/4, 1/40) has unique transverse first-hit of
GRAZING E_sigma on three equal slabs of width 3/400 covering
[1/25, 1/16] at L = 0. The last slab containing eps = 1/16 certifies
on L in {9/25, 1/16}. A single slab over the whole compact is
unresolved. The GRAZING start V = 0, h = 4 eps^3 still excludes
E_sigma on the last slab of this compact.

This is enclosure continuation of a declared eps compact containing
{1/16, 1/20, 1/25} from the aligned restart, not a uniform-in-eps
theorem for every eps, not Lohner from V = 0 on that compact, not
Z_x C2, not G1, and not Hilbert XVI. The finite shrinking one-shot
pack from V = 0 is e_sigma_oneshot_eps.
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
    PolynomialFlow,
    StoppedEventRequest,
    StoppedEventResult,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "ESigmaEpsSpanReport",
    "KILL_L_PACK",
    "certify_e_sigma_eps_span",
    "enclose_cover",
    "identity_verdicts",
    "report",
    "residual_espan_contains",
    "residual_espan_lo",
    "residual_espan_pack_L",
    "residual_espan_width",
    "reverse_cubic_flow_eps_param",
]

_SAMPLE_V = Fraction(1, 4)
_SAMPLE_H = Fraction(1, 40)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_EPS_LO = Fraction(1, 25)
_EPS_HI = Fraction(1, 16)
_EPS_MID = Fraction(1, 20)
_N_SLABS = 3
_SLAB = Fraction(3, 400)
_CONTAINS = Fraction(1, 8000)
_PACK_L = Fraction(169, 400)
_COARSE_SLABS = 1
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.125
_HIT_MAX_STEPS = 80
_HIT_ORDER = 8
_GRAZE_STEP = 0.25
_GRAZE_MAX_STEPS = 280


def _honesty(*, e_sigma_eps_span_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_eps_span_hit=e_sigma_eps_span_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_espan_lo(eps_lo: Fraction, declared: Fraction = _EPS_LO) -> Fraction:
    """Declared compact lower endpoint ``1/25``."""
    return eps_lo - declared


def residual_espan_width(
    eps_lo: Fraction,
    eps_hi: Fraction,
    n_slabs: int = _N_SLABS,
    slab: Fraction = _SLAB,
) -> Fraction:
    """Three slabs of ``3/400`` fill ``[1/25, 1/16]``."""
    return n_slabs * slab - (eps_hi - eps_lo)


def residual_espan_contains(
    mid: Fraction,
    eps_lo: Fraction,
    eps_hi: Fraction,
    product: Fraction = _CONTAINS,
) -> Fraction:
    """``(1/20-1/25)*(1/16-1/20) - 1/8000``."""
    return (mid - eps_lo) * (eps_hi - mid) - product


def residual_espan_pack_L(
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
        "espan_lo": _verdict(residual_espan_lo(_EPS_LO)),
        "espan_width": _verdict(residual_espan_width(_EPS_LO, _EPS_HI)),
        "espan_contains": _verdict(residual_espan_contains(_EPS_MID, _EPS_LO, _EPS_HI)),
        "espan_pack_L": _verdict(residual_espan_pack_L(Fraction(9, 25), Fraction(1, 16))),
    }


def reverse_cubic_flow_eps_param(*, L: Fraction | None = None) -> PolynomialFlow:
    """Time-reversed cubic ``(V,h)`` field with one parameter slot for ``eps``."""
    nvars = 4
    v_p = SparsePolynomial.variable(nvars, 0)
    h_p = SparsePolynomial.variable(nvars, 1)
    e_p = SparsePolynomial.variable(nvars, 2)
    lam1 = Fraction(-2)
    L = Fraction(0) if L is None else L
    v2 = v_p * v_p
    v3 = v2 * v_p
    e2 = e_p * e_p
    e3 = e2 * e_p
    field_f = (
        SparsePolynomial.constant(nvars, -L) * e3
        + SparsePolynomial.constant(nvars, lam1) * e2 * v_p
        + SparsePolynomial.constant(nvars, -1) * e_p * v2
        + SparsePolynomial.constant(nvars, Fraction(1, 3)) * e_p * v3
    )
    field_g = SparsePolynomial.constant(nvars, -1) + e_p * (v_p - 1)
    vdot = field_f + h_p * field_g
    hdot = (SparsePolynomial.constant(nvars, -1) * v_p) * h_p
    return PolynomialFlow((-vdot, -hdot), 1)


def _e_sigma_poly4() -> SparsePolynomial:
    v_p = SparsePolynomial.variable(4, 0)
    h_p = SparsePolynomial.variable(4, 1)
    e_p = SparsePolynomial.variable(4, 2)
    rho, c, sigma = _SAMPLE_RHO, _SAMPLE_C, _SAMPLE_SIGMA
    h2 = h_p * h_p
    e2 = e_p * e_p
    return (
        v_p
        - 1
        + SparsePolynomial.constant(4, sigma * rho) * h_p
        + e_p * SparsePolynomial.constant(4, rho * rho) * h2
        + e2 * SparsePolynomial.constant(4, c * sigma * rho) * h2
    )


def _hit_payload(result: StoppedEventResult) -> dict[str, float | bool | str]:
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


def certify_e_sigma_eps_span(
    *,
    L: Fraction = Fraction(0),
    eps_lo: Fraction = _EPS_LO,
    eps_hi: Fraction = _EPS_HI,
    v0: Fraction = _SAMPLE_V,
    h0: Fraction | None = _SAMPLE_H,
    graze: bool = False,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``E_sigma`` on a closed initial-``eps`` interval."""
    epar = SparsePolynomial.variable(1, 0)
    if graze:
        initial = (
            SparsePolynomial.constant(1, 0),
            SparsePolynomial.constant(1, 4) * (epar**3),
        )
    else:
        height = _SAMPLE_H if h0 is None else h0
        initial = (
            SparsePolynomial.constant(1, v0),
            SparsePolynomial.constant(1, height),
        )
    request = StoppedEventRequest(
        flow=reverse_cubic_flow_eps_param(L=L),
        initial=initial,
        parameters=(Interval.hull(Interval.from_rational(eps_lo), Interval.from_rational(eps_hi)),),
        target=PolynomialEvent(_e_sigma_poly4(), direction=1, name="Esigma"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    return _hit_payload(certify_stopped_event(request))


def enclose_cover(
    *,
    L: Fraction = Fraction(0),
    n_slabs: int = _N_SLABS,
    eps_lo: Fraction = _EPS_LO,
    eps_hi: Fraction = _EPS_HI,
) -> dict[str, object]:
    """Split ``[eps_lo, eps_hi]`` into equal slabs; each must certify ``E_sigma``."""
    slabs: list[dict[str, float | bool | str]] = []
    all_certified = True
    width = (eps_hi - eps_lo) / n_slabs
    for index in range(n_slabs):
        lo = eps_lo + index * width
        hi = eps_lo + (index + 1) * width
        row = certify_e_sigma_eps_span(L=L, eps_lo=lo, eps_hi=hi)
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


def _hit_ok(row: Mapping[str, float | bool | str]) -> bool:
    return (
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
    )


@dataclass(frozen=True)
class ESigmaEpsSpanReport:
    """Compact aligned parametric-eps E_sigma cover. Not every eps or G1."""

    identities: Mapping[str, str]
    cover: Mapping[str, object]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    coarse: Mapping[str, object]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_eps_span_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-eps-span-v1",
            "identities": dict(self.identities),
            "cover": dict(self.cover),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "coarse": dict(self.coarse),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_eps_span_hit": self.e_sigma_eps_span_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma on three eps-slabs covering [1/25, 1/16] from "
                "aligned (V,h)=(1/4,1/40) at L=0, a declared compact "
                "containing {1/16, 1/20, 1/25}. The last slab containing "
                "eps=1/16 certifies on L in {9/25, 1/16}. A single slab "
                "over the whole compact is unresolved. The GRAZING start "
                "V=0 still excludes E_sigma on the last slab. Not a "
                "uniform-in-eps theorem for every eps, not Lohner from "
                "V=0 on this compact, not Z_x C2, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaEpsSpanReport:
    """Replay compact identities, enclose three slabs, refuse coarse and V=0."""
    identities = identity_verdicts()
    cover = _as_float_slabs(enclose_cover())
    last_lo = _EPS_HI - _SLAB
    pack = {
        str(L): certify_e_sigma_eps_span(L=L, eps_lo=last_lo, eps_hi=_EPS_HI)
        for L in _KILL_L_PACK
    }
    coarse = _as_float_slabs(enclose_cover(n_slabs=_COARSE_SLABS))
    from_start = certify_e_sigma_eps_span(
        L=Fraction(0),
        eps_lo=last_lo,
        eps_hi=_EPS_HI,
        graze=True,
        step=_GRAZE_STEP,
        max_steps=_GRAZE_MAX_STEPS,
        order=_HIT_ORDER,
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(cover["all_certified"])
        and all(_hit_ok(row) for row in pack.values())
        and not bool(coarse["all_certified"])
        and from_start["status"] != "certified"
    )
    return ESigmaEpsSpanReport(
        identities=identities,
        cover=cover,
        pack=pack,
        coarse=coarse,
        from_grazing_start=from_start,
        e_sigma_eps_span_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_eps_span_hit=sealed),
    )
