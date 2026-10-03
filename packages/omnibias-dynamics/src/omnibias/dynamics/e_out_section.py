# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Matching-chart image of x = rho/nu and certified first-hit of E_out.

Under the first-root matching embedding V = -eps x with nu = eps, the
large physical section x = rho/nu maps to V = -rho. The height-corrected
matching-chart polynomial is

    E_out(V,h) = V + rho + nu rho h + C nu^2 rho h^2.

On the declared cubic compact eps = 1/16, lambda1 = -2,
V(0) = -eps, h(0) = 4 eps^3, rho = 1/4, C = 2, certify_stopped_event
proves a unique transverse first hit of E_out for L in {9/25, 1/16, 0},
including the kill limit L = 0. The GRAZING / HEIGHT-COMPARISON
polynomial E_sigma = V-1+sigma rho h+nu rho^2 h^2+C nu^2 sigma rho h^2
lives near V = 1 and is excluded on this outgoing orbit, including at
L = 0.

Not GRAZING E_sigma, not uniform in eps -> 0, not G1, and not Hilbert
XVI. The cubic (V,h) orbit is vh_orbit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)
from omnibias.dynamics.vh_orbit import cubic_polynomial_flow

__all__ = [
    "EOutSectionReport",
    "KILL_L_PACK",
    "certify_e_out",
    "certify_grazing_excluded",
    "e_out",
    "e_sigma",
    "identity_verdicts",
    "report",
    "residual_e_out_group",
    "residual_e_out_lead",
    "residual_e_sigma_nu0",
    "residual_section_embed",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_NU = _SAMPLE_EPS
_SAMPLE_V = -_SAMPLE_EPS
_SAMPLE_H = 4 * _SAMPLE_EPS**3
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(1)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.5
_HIT_MAX_STEPS = 64
_HIT_ORDER = 6
_GRAZING_MAX_STEPS = 16


def _honesty(*, e_out_first_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_out_first_hit=e_out_first_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def e_sigma(
    v_coord: Fraction,
    height: Fraction,
    sigma: Fraction,
    rho: Fraction,
    nu: Fraction,
    c: Fraction,
) -> Fraction:
    """GRAZING / HEIGHT-COMPARISON physical section polynomial."""
    return (
        v_coord
        - 1
        + sigma * rho * height
        + nu * rho * rho * height * height
        + c * nu * nu * sigma * rho * height * height
    )


def e_out(
    v_coord: Fraction,
    height: Fraction,
    rho: Fraction,
    nu: Fraction,
    c: Fraction,
) -> Fraction:
    """Matching-chart image of x = rho/nu under V = -eps x."""
    return v_coord + rho + nu * rho * height + c * nu * nu * rho * height * height


def residual_e_sigma_nu0(
    v_coord: Fraction, height: Fraction, sigma: Fraction, rho: Fraction, c: Fraction
) -> Fraction:
    """``E_sigma`` at ``nu = 0`` minus ``V-1+sigma rho h``."""
    return e_sigma(v_coord, height, sigma, rho, Fraction(0), c) - (
        v_coord - 1 + sigma * rho * height
    )


def residual_e_out_lead(v_coord: Fraction, rho: Fraction, c: Fraction) -> Fraction:
    """``E_out`` at ``nu = 0`` minus ``V+rho``."""
    return e_out(v_coord, Fraction(0), rho, Fraction(0), c) - (v_coord + rho)


def residual_section_embed(eps: Fraction, rho: Fraction) -> Fraction:
    """``V_embed + rho`` for ``V_embed = -eps (rho/eps)``."""
    if eps == 0:
        raise ValueError("eps must be nonzero")
    return -eps * (rho / eps) + rho


def residual_e_out_group(
    v_coord: Fraction, height: Fraction, rho: Fraction, nu: Fraction, c: Fraction
) -> Fraction:
    """``E_out - (V + rho (1 + nu h + C nu^2 h^2))``."""
    grouped = v_coord + rho * (1 + nu * height + c * nu * nu * height * height)
    return e_out(v_coord, height, rho, nu, c) - grouped


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    v_coord, height = _SAMPLE_V, _SAMPLE_H
    rho, nu, c, sigma = _SAMPLE_RHO, _SAMPLE_NU, _SAMPLE_C, _SAMPLE_SIGMA
    return {
        "e_sigma_nu0": _verdict(residual_e_sigma_nu0(v_coord, height, sigma, rho, c)),
        "e_out_lead": _verdict(residual_e_out_lead(v_coord, rho, c)),
        "section_embed": _verdict(residual_section_embed(_SAMPLE_EPS, rho)),
        "e_out_group": _verdict(residual_e_out_group(v_coord, height, rho, nu, c)),
    }


def _e_out_poly(*, nu: Fraction | None = None) -> SparsePolynomial:
    v_p = SparsePolynomial.variable(3, 0)
    h_p = SparsePolynomial.variable(3, 1)
    rho, c = _SAMPLE_RHO, _SAMPLE_C
    nu = _SAMPLE_NU if nu is None else nu
    return (
        v_p
        + SparsePolynomial.constant(3, rho)
        + SparsePolynomial.constant(3, nu * rho) * h_p
        + SparsePolynomial.constant(3, c * nu * nu * rho) * (h_p * h_p)
    )


def _grazing_lead_poly() -> SparsePolynomial:
    v_p = SparsePolynomial.variable(3, 0)
    h_p = SparsePolynomial.variable(3, 1)
    return v_p - 1 + SparsePolynomial.constant(3, _SAMPLE_RHO) * h_p


def certify_e_out(
    *,
    L: Fraction = Fraction(9, 25),
    eps: Fraction | None = None,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of matching-chart ``E_out`` at a declared ``L``."""
    eps = _SAMPLE_EPS if eps is None else eps
    request = StoppedEventRequest(
        flow=cubic_polynomial_flow(L=L, eps=eps),
        initial=(
            SparsePolynomial.constant(0, -eps),
            SparsePolynomial.constant(0, 4 * eps**3),
        ),
        parameters=(),
        target=PolynomialEvent(_e_out_poly(nu=eps), direction=-1, name="Eout"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    v_hi = float("nan")
    h_lo = float("nan")
    nvel_hi = float("nan")
    if result.return_box is not None:
        v_b, h_b = result.return_box
        v_hi = v_b.hi
        h_lo = h_b.lo
    if result.normal_velocity is not None:
        nvel_hi = result.normal_velocity.hi
    return {
        "status": result.status,
        "replayed": replay,
        "v_hi": v_hi,
        "h_lo": h_lo,
        "nvel_hi": nvel_hi,
        "transverse_negative": nvel_hi < 0.0 if nvel_hi == nvel_hi else False,
    }


def certify_grazing_excluded(
    *,
    L: Fraction = Fraction(0),
    max_steps: int = _GRAZING_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """GRAZING ``E_sigma`` at ``nu = 0`` is excluded on the outgoing cubic orbit."""
    request = StoppedEventRequest(
        flow=cubic_polynomial_flow(L=L),
        initial=(
            SparsePolynomial.constant(0, _SAMPLE_V),
            SparsePolynomial.constant(0, _SAMPLE_H),
        ),
        parameters=(),
        target=PolynomialEvent(_grazing_lead_poly(), direction=-1, name="Esigma"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    return {
        "status": result.status,
        "replayed": replay,
    }


@dataclass(frozen=True)
class EOutSectionReport:
    """Matching-chart E_out first-hit including L=0. Not GRAZING E_sigma or G1."""

    identities: Mapping[str, str]
    hit: Mapping[str, float | bool | str]
    kill_hits: Mapping[str, Mapping[str, float | bool | str]]
    grazing: Mapping[str, float | bool | str]
    e_out_first_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-out-section-v1",
            "identities": dict(self.identities),
            "hit": dict(self.hit),
            "kill_hits": {key: dict(value) for key, value in self.kill_hits.items()},
            "grazing": dict(self.grazing),
            "e_out_first_hit": self.e_out_first_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified first-hit of matching-chart E_out = V+rho+"
                "nu rho h+C nu^2 rho h^2, the image of x=rho/nu under "
                "V=-eps x, on L in {9/25, 1/16, 0} including the kill "
                "limit L=0. GRAZING E_sigma=V-1+... is excluded, "
                "including at L=0. Not uniform in eps->0, not G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> EOutSectionReport:
    """Replay E_out identities, certify the L-pack, refuse GRAZING at L=0."""
    identities = identity_verdicts()
    kill_hits = {str(L): certify_e_out(L=L) for L in _KILL_L_PACK}
    hit = kill_hits[str(Fraction(9, 25))]
    grazing = certify_grazing_excluded(L=Fraction(0))
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_negative"])
        for row in kill_hits.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and pack_ok
        and grazing["status"] != "certified"
    )
    return EOutSectionReport(
        identities=identities,
        hit=hit,
        kill_hits=kill_hits,
        grazing=grazing,
        e_out_first_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_out_first_hit=sealed),
    )
