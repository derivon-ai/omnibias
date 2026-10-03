# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified E_sigma first-hit from a declared incoming compact point.

On the reverse cubic at eps = 1/16, lambda1 = -2, from the declared
incoming point (V, h) = (3/4, 1/4), certify_stopped_event hits the
GRAZING / HEIGHT-COMPARISON polynomial

    E_sigma = V-1+sigma rho h+nu rho^2 h^2+C nu^2 sigma rho h^2

with sigma = -1, rho = 1/4, C = 2, uniquely and transversely, for
L in {9/25, 1/16, 0}. A short horizon does not certify. The GRAZING
start V = 0, h = 4 eps^3 still excludes E_sigma on this compact
horizon (Lohner wrapping on the long incoming path).

This is a declared incoming-point first-hit, not the GRAZING band from
V = 0, not uniform in eps, not G1, and not Hilbert XVI. The incoming
V = 1/4 wall from V = 0 is e_sigma_in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.e_out_section import e_sigma
from omnibias.dynamics.e_sigma_in import reverse_cubic_flow
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "ESigmaHitReport",
    "KILL_L_PACK",
    "certify_e_sigma",
    "identity_verdicts",
    "report",
    "residual_e_sigma_in_group",
    "residual_e_sigma_in_lead",
    "residual_restart_h",
    "residual_restart_v",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_V = Fraction(3, 4)
_SAMPLE_H = Fraction(1, 4)
_SAMPLE_RHO = Fraction(1, 4)
_SAMPLE_C = Fraction(2)
_SAMPLE_SIGMA = Fraction(-1)
_KILL_L_PACK = (Fraction(9, 25), Fraction(1, 16), Fraction(0))
KILL_L_PACK = _KILL_L_PACK
_HIT_STEP = 0.25
_HIT_MAX_STEPS = 16
_HIT_ORDER = 8
_SHORT_STEPS = 3


def _honesty(*, e_sigma_first_hit: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        e_sigma_first_hit=e_sigma_first_hit,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def residual_restart_v(v0: Fraction, wall: Fraction = _SAMPLE_V) -> Fraction:
    """Declared incoming restart ``V - 3/4``."""
    return v0 - wall


def residual_restart_h(height: Fraction, start: Fraction = _SAMPLE_H) -> Fraction:
    """Declared incoming restart ``h - 1/4``."""
    return height - start


def residual_e_sigma_in_lead(
    v_coord: Fraction, height: Fraction, sigma: Fraction, rho: Fraction, c: Fraction
) -> Fraction:
    """``E_sigma`` at ``nu = 0`` minus ``V-1+sigma rho h``."""
    return e_sigma(v_coord, height, sigma, rho, Fraction(0), c) - (
        v_coord - 1 + sigma * rho * height
    )


def residual_e_sigma_in_group(
    v_coord: Fraction,
    height: Fraction,
    sigma: Fraction,
    rho: Fraction,
    nu: Fraction,
    c: Fraction,
) -> Fraction:
    """``E_sigma`` minus the grouped HEIGHT-COMPARISON polynomial."""
    grouped = (
        v_coord
        - 1
        + sigma * rho * (height + c * nu * nu * height * height)
        + nu * rho * rho * height * height
    )
    return e_sigma(v_coord, height, sigma, rho, nu, c) - grouped


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    v_coord, height = _SAMPLE_V, _SAMPLE_H
    rho, nu, c, sigma = _SAMPLE_RHO, _SAMPLE_EPS, _SAMPLE_C, _SAMPLE_SIGMA
    return {
        "restart_v": _verdict(residual_restart_v(v_coord)),
        "restart_h": _verdict(residual_restart_h(height)),
        "e_sigma_in_lead": _verdict(residual_e_sigma_in_lead(v_coord, height, sigma, rho, c)),
        "e_sigma_in_group": _verdict(
            residual_e_sigma_in_group(v_coord, height, sigma, rho, nu, c)
        ),
    }


def _e_sigma_poly(*, nu: Fraction | None = None) -> SparsePolynomial:
    v_p = SparsePolynomial.variable(3, 0)
    h_p = SparsePolynomial.variable(3, 1)
    rho, c, sigma = _SAMPLE_RHO, _SAMPLE_C, _SAMPLE_SIGMA
    nu = _SAMPLE_EPS if nu is None else nu
    return (
        v_p
        - 1
        + SparsePolynomial.constant(3, sigma * rho) * h_p
        + SparsePolynomial.constant(3, nu * rho * rho) * (h_p * h_p)
        + SparsePolynomial.constant(3, c * nu * nu * sigma * rho) * (h_p * h_p)
    )


def certify_e_sigma(
    *,
    L: Fraction = Fraction(0),
    v0: Fraction = _SAMPLE_V,
    h0: Fraction = _SAMPLE_H,
    eps: Fraction | None = None,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``E_sigma`` from a declared incoming point."""
    eps = _SAMPLE_EPS if eps is None else eps
    request = StoppedEventRequest(
        flow=reverse_cubic_flow(L=L, eps=eps),
        initial=(
            SparsePolynomial.constant(0, v0),
            SparsePolynomial.constant(0, h0),
        ),
        parameters=(),
        target=PolynomialEvent(_e_sigma_poly(nu=eps), direction=1, name="Esigma"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    v_lo = float("nan")
    h_lo = float("nan")
    nvel_lo = float("nan")
    if result.return_box is not None:
        v_b, h_b = result.return_box
        v_lo = v_b.lo
        h_lo = h_b.lo
    if result.normal_velocity is not None:
        nvel_lo = result.normal_velocity.lo
    return {
        "status": result.status,
        "replayed": replay,
        "v_lo": v_lo,
        "h_lo": h_lo,
        "nvel_lo": nvel_lo,
        "transverse_positive": nvel_lo > 0.0 if nvel_lo == nvel_lo else False,
    }


@dataclass(frozen=True)
class ESigmaHitReport:
    """Declared-point E_sigma first-hit. Not the GRAZING V=0 band or G1."""

    identities: Mapping[str, str]
    pack: Mapping[str, Mapping[str, float | bool | str]]
    short: Mapping[str, float | bool | str]
    from_grazing_start: Mapping[str, float | bool | str]
    e_sigma_first_hit: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-e-sigma-hit-v1",
            "identities": dict(self.identities),
            "pack": {key: dict(value) for key, value in self.pack.items()},
            "short": dict(self.short),
            "from_grazing_start": dict(self.from_grazing_start),
            "e_sigma_first_hit": self.e_sigma_first_hit,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "Certified unique transverse first-hit of GRAZING "
                "E_sigma from the declared incoming point (V,h)=(3/4,1/4) "
                "on L in {9/25, 1/16, 0}. A short horizon does not "
                "certify. The GRAZING start V=0 still excludes E_sigma "
                "on this compact horizon. Not the GRAZING band from V=0, "
                "not uniform eps, not G1, or Hilbert XVI."
            ),
        }


def report() -> ESigmaHitReport:
    """Replay E_sigma identities, certify the L-pack, refuse short and V=0 start."""
    identities = identity_verdicts()
    pack = {str(L): certify_e_sigma(L=L) for L in _KILL_L_PACK}
    short = certify_e_sigma(L=Fraction(0), max_steps=_SHORT_STEPS)
    from_start = certify_e_sigma(
        L=Fraction(0),
        v0=Fraction(0),
        h0=4 * _SAMPLE_EPS**3,
    )
    pack_ok = all(
        row["status"] == "certified"
        and bool(row["replayed"])
        and bool(row["transverse_positive"])
        for row in pack.values()
    )
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and pack_ok
        and short["status"] != "certified"
        and from_start["status"] != "certified"
    )
    return ESigmaHitReport(
        identities=identities,
        pack=pack,
        short=short,
        from_grazing_start=from_start,
        e_sigma_first_hit=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(e_sigma_first_hit=sealed),
    )
