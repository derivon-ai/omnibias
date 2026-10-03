# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Comparison GRAZING E_sigma first-hit from V=0 on the reverse cubic.

On the kill-line reverse cubic (L >= 0, lambda1 = -2, nu = eps),
Vdot_rev = -f + h (1+nu-nu V). For V in [0, 6/5], f <= 0 and
1+nu-nu V > 0, so

    dh/dV <= V / (1+nu-nu V).

The integral is exact:

    int_0^V t/(1+nu-nu t) dt
        = -V/nu + (1+nu)/nu^2 log((1+nu)/(1+nu-nu V)).

The log is majorized by the cubic Taylor remainder identity
log(1+x) < x - x^2/2 + x^3/3 (x>0), which is rational at
V* = 6/5, eps = 1/16 (x = 6/79). The resulting height majorant
h_up makes E_sigma(V*, h_up) > 0 while E_sigma(0, 4 eps^3) < 0.
Along the tube, dE/dV >= 1 - rho V*/(1+nu-nu V*) = 55/79 > 0,
so E_sigma has a unique increasing zero. Lohner wrapping still
refuses certify_stopped_event of E_sigma from V=0.

This is a comparison first-hit from the GRAZING start, not a
Lohner event, not uniform in eps, not G1, and not Hilbert XVI.
The declared-point Lohner hit is e_sigma_hit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_hit import certify_e_sigma
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.vh_orbit import field_f, field_g

__all__ = [
    "ESigmaFrom0Report",
    "KILL_L_PACK",
    "enclose_from0",
    "height_majorant",
    "identity_verdicts",
    "log_taylor_upper",
    "report",
    "residual_de_sigma_lo",
    "residual_f_cubic_factor",
    "residual_log_taylor_num",
    "residual_ratio_split",
    "residual_vdot_rev_split",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_H = Fraction(1, 8)
_SAMPLE_VSTAR = Fraction(6, 5)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_SAMPLE_X = Fraction(6, 79)
_SAMPLE_DE = Fraction(55, 79)
_SAMPLE_DEN = Fraction(79, 80)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK


def _honesty(*, e_sigma_from0_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_from0_hit=e_sigma_from0_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_f_cubic_factor(
    v_coord: Fraction, eps: Fraction
) -> Fraction:
    """Kill-line ``f`` at ``L=0`` minus ``(eps/3) V (V^2-3V-6 eps)``."""
    factored = (eps / 3) * v_coord * (v_coord * v_coord - 3 * v_coord - 6 * eps)
    return field_f(Fraction(0), Fraction(-2), eps, v_coord) - factored


def residual_vdot_rev_split(
    v_coord: Fraction, height: Fraction, eps: Fraction
) -> Fraction:
    """Reverse ``Vdot + f - h(1+nu-nu V)`` with ``nu=eps``."""
    nu = eps
    f_val = field_f(Fraction(0), Fraction(-2), eps, v_coord)
    g_val = field_g(eps, v_coord)
    vdot_rev = -(f_val + height * g_val)
    split = -f_val + height * (1 + nu - nu * v_coord)
    return vdot_rev - split


def residual_ratio_split(v_coord: Fraction, nu: Fraction) -> Fraction:
    """``V/(1+nu-nu V) - ((1+nu)/nu /(1+nu-nu V) - 1/nu)``."""
    den = 1 + nu - nu * v_coord
    lhs = v_coord / den
    rhs = (1 + nu) / nu / den - 1 / nu
    return lhs - rhs


def residual_log_taylor_num(x: Fraction) -> Fraction:
    """``(1-x+x^2)(1+x)-1 - x^3``; derivative numerator of the log gap."""
    return (1 - x + x * x) * (1 + x) - 1 - x**3


def residual_de_sigma_lo(
    de: Fraction, den: Fraction, rho: Fraction, vstar: Fraction
) -> Fraction:
    """Cleared ``dE/dV >= 1 - rho V*/den``: ``de * den - (den - rho V*)``."""
    return de * den - (den - rho * vstar)


def log_taylor_upper(x: Fraction) -> Fraction:
    """Rational majorant ``x - x^2/2 + x^3/3`` of ``log(1+x)`` for ``x>0``."""
    return x - x * x / 2 + x**3 / 3


def height_majorant(
    v_coord: Fraction, eps: Fraction = _SAMPLE_EPS
) -> Fraction:
    """Taylor-majorized incoming ``h(V)`` from ``h(0)=4 eps^3``."""
    if eps == 0:
        raise ValueError("eps must be nonzero")
    nu = eps
    den = 1 + nu - nu * v_coord
    if den <= 0:
        raise ValueError("1+nu-nu V must be positive")
    x = (nu * v_coord) / den
    return 4 * eps**3 - v_coord / nu + (1 + nu) / (nu * nu) * log_taylor_upper(x)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    eps, v_coord, height = _SAMPLE_EPS, _SAMPLE_V, _SAMPLE_H
    return {
        "f_cubic_factor": _verdict(residual_f_cubic_factor(v_coord, eps)),
        "vdot_rev_split": _verdict(residual_vdot_rev_split(v_coord, height, eps)),
        "ratio_split": _verdict(residual_ratio_split(v_coord, eps)),
        "log_taylor_num": _verdict(residual_log_taylor_num(_SAMPLE_X)),
        "de_sigma_lo": _verdict(
            residual_de_sigma_lo(_SAMPLE_DE, _SAMPLE_DEN, _SAMPLE_RHO, _SAMPLE_VSTAR)
        ),
    }


def enclose_from0(
    *,
    eps: Fraction = _SAMPLE_EPS,
    vstar: Fraction = _SAMPLE_VSTAR,
    L: Fraction = Fraction(0),
) -> dict[str, float | bool]:
    """Comparison tube: E starts negative, E(V*,h_up)>0, dE/dV>0, f<=0."""
    if eps == 0:
        return {
            "e_start_hi": 0.0,
            "e_end_lo": 0.0,
            "de_lo": 0.0,
            "f_hi": 0.0,
            "den_lo": 0.0,
            "sign_change": False,
            "transverse": False,
            "f_nonpositive": False,
        }
    h0 = 4 * eps**3
    h_up = height_majorant(vstar, eps)
    e_start = Interval.from_rational(
        e_sigma(Fraction(0), h0, _SAMPLE_SIGMA, _SAMPLE_RHO, eps, _SAMPLE_C)
    )
    e_end = Interval.from_rational(
        e_sigma(vstar, h_up, _SAMPLE_SIGMA, _SAMPLE_RHO, eps, _SAMPLE_C)
    )
    den = 1 + eps - eps * vstar
    de = Interval.from_rational(1 - _SAMPLE_RHO * vstar / den)
    f_box = Interval.from_rational(field_f(L, Fraction(-2), eps, vstar))
    den_box = Interval.from_rational(den)
    return {
        "e_start_hi": e_start.hi,
        "e_end_lo": e_end.lo,
        "de_lo": de.lo,
        "f_hi": f_box.hi,
        "den_lo": den_box.lo,
        "sign_change": e_start.hi < 0.0 and e_end.lo > 0.0,
        "transverse": de.lo > 0.0,
        "f_nonpositive": f_box.hi < 0.0 and den_box.lo > 0.0,
    }


@dataclass(frozen=True)
class ESigmaFrom0Report:
    """Comparison GRAZING E_sigma from V=0. Not Lohner or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool]]
    refuse_v1: Mapping[str, float | bool]
    lohner_from_start: Mapping[str, float | bool | str]
    e_sigma_from0_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-from0-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "refuse_v1": dict(self.refuse_v1),
            "lohner_from_start": dict(self.lohner_from_start),
            "e_sigma_from0_hit": self.e_sigma_from0_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Comparison tube from GRAZING V=0, h=4 eps^3: unique "
                "increasing E_sigma zero on L in {9/25, 1/16, 0} at "
                "eps=1/16, V*=6/5. V*=1 does not yet change sign. "
                "Lohner wrapping still refuses certify_stopped_event "
                "from V=0. Not uniform eps, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaFrom0Report:
    """Replay identities, enclose the L-pack, refuse V*=1 and Lohner-from-0."""
    identities = identity_verdicts()
    pack = {str(L): enclose_from0(L=L) for L in _KILL_L_PACK}
    refuse_v1 = enclose_from0(vstar=Fraction(1))
    from_start = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
    )
    pack_ok = all(
        row["sign_change"] and row["transverse"] and row["f_nonpositive"]
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and pack_ok
        and refuse_v1["sign_change"] is False
        and from_start["status"] != "certified"
    )
    return ESigmaFrom0Report(
        identities=identities,
        pack=pack,
        refuse_v1=refuse_v1,
        lohner_from_start=from_start,
        e_sigma_from0_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_from0_hit=sealed),
    )
