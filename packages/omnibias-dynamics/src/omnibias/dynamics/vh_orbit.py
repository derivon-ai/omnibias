# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""QR-Lohner cubic (V,h) orbit and certified first-hit of V = -1/4.

The cubic truncation of the first-root normal field is the polynomial

    Vdot = f(V) + h g(V),   hdot = -V h,
    f(V) = -L eps^3 + lam1 eps^2 V - eps V^2 + (eps/3) V^3,
    g(V) = -1 + nu (V-1).

On the declared matching compact ``eps = 1/16``, ``lambda1 = -2``,
``L = 9/25``, ``V(0) = -eps``, ``h(0) = 4 eps^3``, a QR-Lohner prefix
keeps ``V < 0``, ``h > h(0)``, and ``T-h < 9 eps``, and
``certify_stopped_event`` proves a unique transverse first hit of the
declared section ``V = -1/4``. Matching-chart ``E_out`` is
e_out_section. Not GRAZING ``E_sigma``, not uniform in ``r1 -> 0``, not
G1, and not Hilbert XVI. The comparison-bootstrap integral is
th_integral.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.lohner import lohner_flow
from omnibias.core.verified.ode import TaylorSeries
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.k_zeta_remainder import q_cubic
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "VhOrbitReport",
    "certify_vwall",
    "cubic_field",
    "cubic_jac",
    "cubic_polynomial_flow",
    "enclose_lohner_prefix",
    "field_f",
    "field_g",
    "identity_verdicts",
    "report",
    "residual_Vdot_split",
    "residual_g_jet_vh",
    "residual_hdot_vh",
    "residual_q_plus_f",
]

_SAMPLE_EPS = Fraction(1, 16)
_SAMPLE_NU = _SAMPLE_EPS
_SAMPLE_L = Fraction(9, 25)
_SAMPLE_LAM1 = Fraction(-2)
_SAMPLE_Y0 = Fraction(4)
_SAMPLE_XSTAR = Fraction(1)
_SAMPLE_V = -_SAMPLE_EPS * _SAMPLE_XSTAR
_SAMPLE_H = _SAMPLE_EPS**3 * _SAMPLE_Y0
_SAMPLE_W = -_SAMPLE_V
_WALL = Fraction(1, 4)
_MARGIN = 9.0
_LOHNER_STEP = 0.5
_LOHNER_STEPS = 16
_LOHNER_ORDER = 6
_HIT_STEP = 0.5
_HIT_MAX_STEPS = 64
_HIT_ORDER = 6


def _honesty(*, vh_orbit_certified: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        th_integral_majorant=False,
        vh_orbit_certified=vh_orbit_certified,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def field_f(L: Fraction, lam1: Fraction, eps: Fraction, v_coord: Fraction) -> Fraction:
    """Cubic ``f(V) = -L eps^3 + lam1 eps^2 V - eps V^2 + (eps/3) V^3``."""
    return (
        -L * eps**3
        + lam1 * eps**2 * v_coord
        - eps * v_coord * v_coord
        + (eps / 3) * v_coord**3
    )


def field_g(nu: Fraction, v_coord: Fraction) -> Fraction:
    """First-order ``g = -1 + nu (V-1)``."""
    return -1 + nu * (v_coord - 1)


def residual_q_plus_f(
    L: Fraction, lam1: Fraction, eps: Fraction, w: Fraction
) -> Fraction:
    """``q_cubic(-w) + f(-w)``; the height flux is ``-f`` on ``h=0``."""
    return q_cubic(L, lam1, eps, w) + field_f(L, lam1, eps, -w)


def residual_hdot_vh(hdot: Fraction, v_coord: Fraction, height: Fraction) -> Fraction:
    """``hdot + V h``."""
    return hdot + v_coord * height


def residual_g_jet_vh(gee: Fraction, nu: Fraction, v_coord: Fraction) -> Fraction:
    """``g - (-1 + nu (V-1))``."""
    return gee - field_g(nu, v_coord)


def residual_Vdot_split(
    vdot: Fraction, eff: Fraction, height: Fraction, gee: Fraction
) -> Fraction:
    """``Vdot - (f + h g)``."""
    return vdot - (eff + height * gee)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    L, lam1, eps = _SAMPLE_L, _SAMPLE_LAM1, _SAMPLE_EPS
    nu, v_coord, height = _SAMPLE_NU, _SAMPLE_V, _SAMPLE_H
    w = _SAMPLE_W
    eff = field_f(L, lam1, eps, v_coord)
    gee = field_g(nu, v_coord)
    vdot = eff + height * gee
    hdot = -v_coord * height
    return {
        "q_plus_f": _verdict(residual_q_plus_f(L, lam1, eps, w)),
        "hdot_vh": _verdict(residual_hdot_vh(hdot, v_coord, height)),
        "g_jet_vh": _verdict(residual_g_jet_vh(gee, nu, v_coord)),
        "Vdot_split": _verdict(residual_Vdot_split(vdot, eff, height, gee)),
    }


def cubic_field(series: Sequence[TaylorSeries]) -> list[TaylorSeries]:
    """Taylor vector field of the cubic ``(V,h)`` truncation."""
    v_s, h_s = series
    order = v_s.order
    eps = _SAMPLE_EPS
    nu = _SAMPLE_NU
    L = _SAMPLE_L
    lam1 = _SAMPLE_LAM1
    f = TaylorSeries.constant(-L * eps**3, order)
    f = f + v_s * (lam1 * eps**2)
    f = f + (v_s * v_s) * (-eps)
    f = f + (v_s * v_s * v_s) * (eps / 3)
    g = TaylorSeries.constant(-1, order) + (v_s - 1) * nu
    return [f + h_s * g, (v_s * h_s) * (-1)]


def cubic_jac(box: Sequence[Interval]) -> list[list[Interval]]:
    """Jacobian enclosure of the cubic field on a ``(V,h)`` box."""
    v_b, h_b = box
    e = Interval.from_rational(_SAMPLE_EPS)
    n = Interval.from_rational(_SAMPLE_NU)
    l1 = Interval.from_rational(_SAMPLE_LAM1)
    dv_dv = l1 * (e**2) + (Interval.point(-2.0) * e) * v_b + e * (v_b * v_b) + h_b * n
    dv_dh = Interval.point(-1.0) + n * (v_b - Interval.point(1.0))
    return [[dv_dv, dv_dh], [-h_b, -v_b]]


def enclose_lohner_prefix(
    *,
    n_steps: int = _LOHNER_STEPS,
    step: float = _LOHNER_STEP,
    order: int = _LOHNER_ORDER,
    margin: float = _MARGIN,
) -> dict[str, float | bool]:
    """QR-Lohner prefix from matching. Refuses a step that cannot be enclosed."""
    y0 = [
        Interval.from_rational(_SAMPLE_V),
        Interval.from_rational(_SAMPLE_H),
    ]
    try:
        state = lohner_flow(cubic_field, cubic_jac, y0, h=step, n_steps=n_steps, order=order)
        v_b, h_b = state.to_box()
        kinetic = (v_b * v_b) / Interval.point(2.0)
        gap = kinetic - h_b
        return {
            "v_hi": v_b.hi,
            "h_lo": h_b.lo,
            "gap_hi": gap.hi,
            "width": state.width(),
            "v_negative": v_b.hi < 0.0,
            "h_above_exit": h_b.lo > float(_SAMPLE_H),
            "below_margin": gap.hi < margin * float(_SAMPLE_EPS),
        }
    except (ValueError, RuntimeError, ZeroDivisionError):
        return {
            "v_hi": float("nan"),
            "h_lo": float("nan"),
            "gap_hi": float("inf"),
            "width": float("inf"),
            "v_negative": False,
            "h_above_exit": False,
            "below_margin": False,
        }


def cubic_polynomial_flow(
    *, L: Fraction | None = None, eps: Fraction | None = None
) -> PolynomialFlow:
    """Exact cubic ``(V,h)`` polynomial field. Defaults are the sample compact."""
    v_p = SparsePolynomial.variable(3, 0)
    h_p = SparsePolynomial.variable(3, 1)
    eps = _SAMPLE_EPS if eps is None else eps
    nu = eps
    lam1 = _SAMPLE_LAM1
    L = _SAMPLE_L if L is None else L
    v2 = v_p * v_p
    v3 = v2 * v_p
    f = (
        SparsePolynomial.constant(3, -L * eps**3)
        + SparsePolynomial.constant(3, lam1 * eps**2) * v_p
        + SparsePolynomial.constant(3, -eps) * v2
        + SparsePolynomial.constant(3, eps / 3) * v3
    )
    g = SparsePolynomial.constant(3, -1) + SparsePolynomial.constant(3, nu) * (v_p - 1)
    vdot = f + h_p * g
    hdot = (SparsePolynomial.constant(3, -1) * v_p) * h_p
    return PolynomialFlow((vdot, hdot), 0)


def certify_vwall(
    *,
    wall: Fraction = _WALL,
    max_steps: int = _HIT_MAX_STEPS,
    step: float = _HIT_STEP,
    order: int = _HIT_ORDER,
) -> dict[str, float | bool | str]:
    """Unique transverse first-hit of ``V = -wall`` on the cubic field."""
    flow = cubic_polynomial_flow()
    v_p = SparsePolynomial.variable(3, 0)
    request = StoppedEventRequest(
        flow=flow,
        initial=(
            SparsePolynomial.constant(0, _SAMPLE_V),
            SparsePolynomial.constant(0, _SAMPLE_H),
        ),
        parameters=(),
        target=PolynomialEvent(v_p + wall, direction=-1, name="Vwall"),
        step=step,
        max_steps=max_steps,
        order=order,
        derivative_order=0,
    )
    result = certify_stopped_event(request)
    replay = bool(result.certified) and verify_stopped_event(result)
    gap_hi = float("inf")
    v_hi = float("nan")
    h_lo = float("nan")
    if result.return_box is not None:
        v_b, h_b = result.return_box
        kinetic = (v_b * v_b) / Interval.point(2.0)
        gap_hi = (kinetic - h_b).hi
        v_hi = v_b.hi
        h_lo = h_b.lo
    return {
        "status": result.status,
        "replayed": replay,
        "v_hi": v_hi,
        "h_lo": h_lo,
        "gap_hi": gap_hi,
        "below_margin": gap_hi < _MARGIN * float(_SAMPLE_EPS),
        "v_negative": v_hi < 0.0 if v_hi == v_hi else False,
    }


@dataclass(frozen=True)
class VhOrbitReport:
    """Cubic (V,h) Lohner prefix and V=-1/4 first-hit. Not physical E_sigma or G1."""

    identities: Mapping[str, str]
    lohner: Mapping[str, float | bool]
    hit: Mapping[str, float | bool | str]
    vh_orbit_certified: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-vh-orbit-v1",
            "identities": dict(self.identities),
            "lohner": dict(self.lohner),
            "hit": dict(self.hit),
            "vh_orbit_certified": self.vh_orbit_certified,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "QR-Lohner prefix of the cubic (V,h) field plus certified "
                "first-hit of V=-1/4. Matching-chart E_out is "
                "e_out_section. Not GRAZING E_sigma, not uniform in "
                "r1->0, not G1, or Hilbert XVI."
            ),
        }


def report() -> VhOrbitReport:
    """Replay cubic identities, a Lohner prefix, and the V=-1/4 first-hit."""
    identities = identity_verdicts()
    lohner = enclose_lohner_prefix()
    hit = certify_vwall()
    wide = certify_vwall(wall=Fraction(2), max_steps=16)
    sealed = (
        all(status == "PROVED" for status in identities.values())
        and bool(lohner["v_negative"])
        and bool(lohner["h_above_exit"])
        and bool(lohner["below_margin"])
        and hit["status"] == "certified"
        and bool(hit["replayed"])
        and bool(hit["below_margin"])
        and wide["status"] != "certified"
    )
    return VhOrbitReport(
        identities=identities,
        lohner=lohner,
        hit=hit,
        vh_orbit_certified=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(vh_orbit_certified=sealed),
    )
