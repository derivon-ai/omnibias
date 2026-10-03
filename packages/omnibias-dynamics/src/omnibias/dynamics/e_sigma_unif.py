# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uniform-in-eps comparison GRAZING E_sigma from V=0.

The Taylor height majorant of e_sigma_from0 is pole-free after
cancellation:

    I = V^2/den - (1+eps) V^2/(2 den^2) + (1+eps) eps V^3/(3 den^3),

    den = 1+eps-eps V,   h_up = 4 eps^3 + I.

On V* = 6/5 and eps in [0, 1/8], eight equal Interval slabs each
have E_sigma(V*, h_up) > 0 and dE/dV >= 1 - rho V*/den > 0. A single
slab over the whole interval wraps and does not separate 0. V*=1
does not change sign at the sample. This is a uniform comparison
tube, not a Lohner event for every eps, not G1, and not Hilbert XVI.
The sample-eps comparison is e_sigma_from0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_from0 import height_majorant
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "ESigmaUnifReport",
    "I_cancel",
    "enclose_slab",
    "enclose_uniform",
    "identity_verdicts",
    "report",
    "residual_E_eps0",
    "residual_I_cancel",
    "residual_I_limit",
    "residual_de_eps0",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_VSTAR = Fraction(6, 5)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_SAMPLE_DE0 = Fraction(7, 10)
_EPS_HI = Fraction(1, 8)
_N_SLABS = 8
_COARSE_SLABS = 1


def _honesty(*, e_sigma_unif_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_unif_hit=e_sigma_unif_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def I_cancel(v_coord: Fraction, eps: Fraction) -> Fraction:
    """Pole-free incoming height integral majorant."""
    den = 1 + eps - eps * v_coord
    return (
        (v_coord * v_coord) / den
        - (1 + eps) * v_coord * v_coord / (2 * den * den)
        + (1 + eps) * eps * v_coord**3 / (3 * den**3)
    )


def residual_I_cancel(v_coord: Fraction, eps: Fraction) -> Fraction:
    """Taylor majorant minus the cancelled rational form."""
    return height_majorant(v_coord, eps) - 4 * eps**3 - I_cancel(v_coord, eps)


def residual_I_limit(v_coord: Fraction) -> Fraction:
    """``I(V, 0) - V^2/2``."""
    return I_cancel(v_coord, Fraction(0)) - v_coord * v_coord / 2


def residual_E_eps0(
    v_coord: Fraction, height: Fraction, sigma: Fraction, rho: Fraction, c: Fraction
) -> Fraction:
    """``E_sigma`` at ``nu=0`` minus ``V-1+sigma rho h``."""
    return e_sigma(v_coord, height, sigma, rho, Fraction(0), c) - (
        v_coord - 1 + sigma * rho * height
    )


def residual_de_eps0(rho: Fraction, vstar: Fraction, de0: Fraction) -> Fraction:
    """``(1 - rho V*) - 7/10`` at the sample ``V*=6/5``."""
    return (1 - rho * vstar) - de0


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    vstar, eps, rho, sigma, c = (
        _SAMPLE_VSTAR,
        _SAMPLE_EPS,
        _SAMPLE_RHO,
        _SAMPLE_SIGMA,
        _SAMPLE_C,
    )
    height = vstar * vstar / 2
    return {
        "I_cancel": _verdict(residual_I_cancel(vstar, eps)),
        "I_limit": _verdict(residual_I_limit(vstar)),
        "E_eps0": _verdict(residual_E_eps0(vstar, height, sigma, rho, c)),
        "de_eps0": _verdict(residual_de_eps0(rho, vstar, _SAMPLE_DE0)),
    }


def enclose_slab(
    eps_lo: Fraction,
    eps_hi: Fraction,
    *,
    vstar: Fraction = _SAMPLE_VSTAR,
) -> dict[str, float | bool]:
    """Interval enclosure of the cancelled tube on an eps slab."""
    V = Interval.from_rational(vstar)
    rho = Interval.from_rational(_SAMPLE_RHO)
    eps = Interval.hull(Interval.from_rational(eps_lo), Interval.from_rational(eps_hi))
    one = Interval.from_rational(1)
    two = Interval.from_rational(2)
    three = Interval.from_rational(3)
    den = one + eps * (one - V)
    integral = (
        (V * V) / den
        - (one + eps) * V * V / (two * den * den)
        + (one + eps) * eps * (V * V * V) / (three * den * den * den)
    )
    h = Interval.from_rational(4) * (eps**3) + integral
    E = (
        V
        - one
        - rho * h
        + eps * (rho * rho) * (h * h)
        - two * (eps * eps) * rho * (h * h)
    )
    de = one - rho * V / den
    return {
        "e_lo": E.lo,
        "e_hi": E.hi,
        "de_lo": de.lo,
        "den_lo": den.lo,
        "positive": E.lo > 0.0 and de.lo > 0.0 and den.lo > 0.0,
    }


def enclose_uniform(
    *,
    n_slabs: int = _N_SLABS,
    eps_hi: Fraction = _EPS_HI,
    vstar: Fraction = _SAMPLE_VSTAR,
) -> dict[str, object]:
    """Split ``[0, eps_hi]`` into equal slabs; each must keep E>0."""
    slabs: list[dict[str, float | bool]] = []
    all_positive = True
    for index in range(n_slabs):
        lo = eps_hi * Fraction(index, n_slabs)
        hi = eps_hi * Fraction(index + 1, n_slabs)
        row = enclose_slab(lo, hi, vstar=vstar)
        slabs.append(row)
        if not row["positive"]:
            all_positive = False
    return {
        "n_slabs": n_slabs,
        "all_positive": all_positive,
        "e_lo_min": min(row["e_lo"] for row in slabs) if slabs else float("-inf"),
        "slabs": slabs,
    }


def _as_float_slabs(payload: Mapping[str, object]) -> dict[str, object]:
    slabs = payload["slabs"]
    assert isinstance(slabs, Sequence)
    return {
        "n_slabs": payload["n_slabs"],
        "all_positive": payload["all_positive"],
        "e_lo_min": payload["e_lo_min"],
        "slabs": [dict(row) for row in slabs],
    }


@dataclass(frozen=True)
class ESigmaUnifReport:
    """Uniform comparison GRAZING E_sigma on eps in [0, 1/8]. Not Lohner or G1."""

    identities: Mapping[str, str]
    uniform: Mapping[str, object]
    coarse: Mapping[str, object]
    refuse_v1: Mapping[str, object]
    e_sigma_unif_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-unif-v1",
            "identities": dict(self.identities),
            "uniform": dict(self.uniform),
            "coarse": dict(self.coarse),
            "refuse_v1": dict(self.refuse_v1),
            "e_sigma_unif_hit": self.e_sigma_unif_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Uniform comparison tube from GRAZING V=0 on eps in "
                "[0, 1/8] at V*=6/5: eight Interval slabs each keep "
                "E_sigma>0 and dE/dV>0. A single slab wraps. V*=1 "
                "does not change sign. Not a Lohner event for every "
                "eps, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaUnifReport:
    """Replay cancelled identities, enclose eight slabs, refuse coarse and V*=1."""
    identities = identity_verdicts()
    uniform = _as_float_slabs(enclose_uniform())
    coarse = _as_float_slabs(enclose_uniform(n_slabs=_COARSE_SLABS))
    refuse_v1 = _as_float_slabs(enclose_uniform(n_slabs=_N_SLABS, vstar=Fraction(1)))
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(uniform["all_positive"])
        and not bool(coarse["all_positive"])
        and not bool(refuse_v1["all_positive"])
    )
    return ESigmaUnifReport(
        identities=identities,
        uniform=uniform,
        coarse=coarse,
        refuse_v1=refuse_v1,
        e_sigma_unif_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_unif_hit=sealed),
    )
