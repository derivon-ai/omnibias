# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Cancelled-N holomorphic Z on the lambda1=-2 kill compact.

The cubic pieces of

    N = Vdot/eps - eps^2 L - eps lambda1 V + V^2 - V^3 / 3

cancel at nu = 0, so a rectangular Cauchy box on N is fat. Factoring
that cancellation gives holomorphic formulas for Z0 (lambda=0) and
for the full slow-line Z (lambda1, L arbitrary). On the declared real
compact

    nu in [0, 0.02],  v in [-0.5, 1.5],  L in [0, 1],  lambda1 = -2

Picard inclusion holds around k0 = 3 v0 / l, the holomorphic enclosure
of |Z| is O(1), and 2 eps |V| |Z| < 1. That is a usable C=2+delta
prefactor on the slow line. It is not T-h along the actual (V,h)
orbit, not C!=0 |g_h|, not height-section first-hit, G1, or Hilbert
XVI. The rectangular Cauchy majorant is kill_zeta.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.canonical_zeta import k_lambda0, slow_line_V, zeta_closed
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "CancelledNReport",
    "enclose_holomorphic_z",
    "identity_verdicts",
    "q1_holomorphic",
    "report",
    "residual_delta_slow",
    "residual_ell_V_nu",
    "residual_n_cancelled",
    "residual_t_L",
    "residual_t_lambda",
    "residual_z0_holomorphic",
    "z0_holomorphic",
]

_SAMPLE_NU = Fraction(5, 16)
_SAMPLE_V0 = Fraction(4, 5)
_SAMPLE_V = Fraction(1, 2)
_SAMPLE_K = Fraction(2)
_SAMPLE_L = Fraction(0)
_SAMPLE_LAM1 = Fraction(288, 125)
_NU_LO = 0.0
_NU_HI = 0.02
_VCHART_LO = -0.5
_VCHART_HI = 1.5
_L_LO = 0.0
_L_HI = 1.0
_K_PAD = 0.08
_KILL_LAM1 = -2.0
_SAMPLE_KILL_NU = Fraction(1, 64)
_SAMPLE_KILL_R1 = Fraction(1, 5)
_SAMPLE_KILL_V = Fraction(1, 2)


def _honesty(*, cancelled_n_usable_z: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        kill_z_compact_bound=False,
        cancelled_n_usable_z=cancelled_n_usable_z,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def _ell(nu: Fraction, v: Fraction) -> Fraction:
    return 1 + 2 * nu * v


def _slow0(v: Fraction, v0: Fraction) -> Fraction:
    return -2 * v0**3 + 3 * v0**2 * v - v**3


def _unfolding_slow(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    ell0 = _ell(nu, v0)
    if ell0 == 0 or kay == 0:
        raise ValueError("unfolding_slow requires nonzero l and k")
    f0 = (nu**2 * kay**3 * lam0) / ell0
    f1 = -nu * kay**2 * lam1 - (2 * nu**3 * kay**3 * lam0) / ell0**2
    ptilde = 3 * v0**2 + f1
    mtilde = f0 - ptilde * v0 + v0**3
    slow = mtilde + ptilde * v - v**3
    return slow, f0, f1, ell0


def field_N(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> Fraction:
    """Cubic embedding numerator ``N`` on the slow line."""
    slow, _, _, _ = _unfolding_slow(nu, v, v0, kay, lam0, lam1)
    ell = _ell(nu, v)
    vdot_over_eps = ell * slow / kay
    eps = nu * kay
    v_coord = slow_line_V(nu, v)
    return (
        vdot_over_eps
        - eps**2 * lam0
        - eps * lam1 * v_coord
        + v_coord * v_coord
        - v_coord**3 / 3
    )


def cancelled_N(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> Fraction:
    """``N`` after exact cubic / ``T_lambda`` / ``T_L`` cancellation."""
    ell = _ell(nu, v)
    ell0 = _ell(nu, v0)
    if ell0 == 0 or kay == 0 or v0 == 0:
        raise ValueError("cancelled_N requires nonzero l, k, v0")
    k0 = 3 * v0 / ell0
    v_coord = slow_line_V(nu, v)
    cubic = v_coord * v_coord - v_coord**3 / 3
    c0 = ell * _slow0(v, v0) / k0 + cubic
    gap = v - v0
    t_lambda = -(nu**2) * kay * lam1 * gap * gap
    t_L = -4 * nu**4 * kay**2 * lam0 * gap * gap / ell0**2
    return (k0 / kay) * c0 + ((kay - k0) / kay) * cubic + t_lambda + t_L


def q1_holomorphic(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Holomorphic factor of ``zeta+1-V/3`` after cancelling ``nu`` and ``V``."""
    s = v0 + v
    return (
        4 * v0
        - 2 * v0 * v0
        - 3 * v0 * v
        + nu * (4 * v0 * v0 - 3 * v0 * s * s)
        - nu * nu * v0 * s * s * s
        + (v0 - v) * (-2 - nu * v0)
    )


def z0_holomorphic(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """``Z`` on the ``lambda=0`` slice with no division by ``nu``."""
    ell0 = _ell(nu, v0)
    wall = 1 + nu * (v0 + v)
    if v0 == 0 or wall == 0:
        raise ValueError("z0_holomorphic requires nonzero v0 and D")
    return ell0 * q1_holomorphic(nu, v, v0) / (9 * v0 * v0 * wall**3)


def residual_delta_slow(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> Fraction:
    """``slow - slow0 - f0 - f1 (v-v0)``."""
    slow, f0, f1, _ = _unfolding_slow(nu, v, v0, kay, lam0, lam1)
    return slow - _slow0(v, v0) - f0 - f1 * (v - v0)


def residual_ell_V_nu(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """``ell (v-v0) + V - nu (v-v0)^2``; vanishes on the slow-line root."""
    ell = _ell(nu, v)
    v_coord = slow_line_V(nu, v)
    return ell * (v - v0) + v_coord - nu * (v - v0) ** 2


def residual_t_lambda(
    nu: Fraction, v: Fraction, v0: Fraction, kay: Fraction, lam1: Fraction
) -> Fraction:
    """Lambda source after the ``O(nu)`` cancellation."""
    if kay == 0:
        raise ValueError("residual_t_lambda requires nonzero k")
    ell = _ell(nu, v)
    v_coord = slow_line_V(nu, v)
    gap = v - v0
    t_lambda = (ell / kay) * (-nu * kay**2 * lam1) * gap - (nu * kay) * lam1 * v_coord
    closed = -(nu**2) * kay * lam1 * gap * gap
    return t_lambda - closed


def residual_t_L(
    nu: Fraction, v: Fraction, v0: Fraction, kay: Fraction, lam0: Fraction
) -> Fraction:
    """``L`` source after the ``O(nu^2)`` cancellation."""
    ell = _ell(nu, v)
    ell0 = _ell(nu, v0)
    if kay == 0 or ell0 == 0:
        raise ValueError("residual_t_L requires nonzero k and l")
    gap = v - v0
    term_f0 = (ell / kay) * (nu**2 * kay**3 * lam0 / ell0)
    term_f1 = (ell / kay) * (-2 * nu**3 * kay**3 * lam0 / ell0**2) * gap
    t_L = term_f0 + term_f1 - (nu * kay) ** 2 * lam0
    closed = -4 * nu**4 * kay**2 * lam0 * gap * gap / ell0**2
    return t_L - closed


def residual_n_cancelled(
    nu: Fraction,
    v: Fraction,
    v0: Fraction,
    kay: Fraction,
    lam0: Fraction,
    lam1: Fraction,
) -> Fraction:
    """Field ``N`` versus the cancelled decomposition."""
    return field_N(nu, v, v0, kay, lam0, lam1) - cancelled_N(
        nu, v, v0, kay, lam0, lam1
    )


def residual_z0_holomorphic(nu: Fraction, v: Fraction, v0: Fraction) -> Fraction:
    """Direct ``Z0 = (zeta+1-V/3)/(eps0 V)`` versus the holomorphic formula."""
    v_coord = slow_line_V(nu, v)
    kay = k_lambda0(nu, v0)
    eps0 = nu * kay
    if eps0 == 0 or v_coord == 0:
        raise ValueError("residual_z0_holomorphic requires nonzero eps0 V")
    direct = (zeta_closed(nu, v, v0) + 1 - v_coord / 3) / (eps0 * v_coord)
    return direct - z0_holomorphic(nu, v, v0)


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v0, v, kay = _SAMPLE_NU, _SAMPLE_V0, _SAMPLE_V, _SAMPLE_K
    lam0, lam1 = _SAMPLE_L, _SAMPLE_LAM1
    return {
        "delta_slow": _verdict(residual_delta_slow(nu, v, v0, kay, lam0, lam1)),
        "ell_V_nu": _verdict(residual_ell_V_nu(nu, v, v0)),
        "t_lambda": _verdict(residual_t_lambda(nu, v, v0, kay, lam1)),
        "t_L": _verdict(residual_t_L(nu, v, v0, kay, lam0)),
        "n_cancelled": _verdict(residual_n_cancelled(nu, v, v0, kay, lam0, lam1)),
        "z0_holomorphic": _verdict(residual_z0_holomorphic(nu, v, v0)),
    }


def _contains(outer: Interval, inner: Interval) -> bool:
    return outer.lo <= inner.lo and inner.hi <= outer.hi


def enclose_holomorphic_z(
    *,
    nu_lo: float = _NU_LO,
    nu_hi: float = _NU_HI,
    v_lo: float = _VCHART_LO,
    v_hi: float = _VCHART_HI,
    L_lo: float = _L_LO,
    L_hi: float = _L_HI,
    k_pad: float = _K_PAD,
) -> dict[str, float | bool]:
    """Holomorphic ``Z`` enclosure on a declared real kill compact."""
    if nu_hi < nu_lo or v_hi < v_lo or L_lo < 0.0 or L_hi < L_lo or k_pad <= 0.0:
        raise ValueError("enclose_holomorphic_z requires a declared compact")
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    three = Interval.point(3.0)
    four = Interval.point(4.0)
    nine = Interval.point(9.0)
    lam1 = Interval.point(_KILL_LAM1)
    nu = Interval(nu_lo, nu_hi)
    v = Interval(v_lo, v_hi)
    lam0 = Interval(L_lo, L_hi)
    v0 = two / (one + (one + four * nu).sqrt())
    ell0 = one + two * nu * v0
    k0 = (three * v0) / ell0
    guess = k0 + Interval(-k_pad, k_pad)
    phi = k0 + (nu * nu * guess * guess * lam1) / (ell0 * ell0) + (
        four * nu * nu * nu * nu * guess * guess * guess * lam0
    ) / (ell0 * ell0 * ell0 * ell0)
    included = _contains(guess, phi)
    kay = phi
    wall = one + nu * (v0 + v)
    s = v0 + v
    q1 = (
        four * v0
        - two * v0 * v0
        - three * v0 * v
        + nu * (four * v0 * v0 - three * v0 * s * s)
        - nu * nu * v0 * s * s * s
        + (v0 - v) * (-two - nu * v0)
    )
    z0 = ell0 * q1 / (nine * v0 * v0 * wall * wall * wall)
    z_box = (
        (k0 / kay) ** 2 * z0
        - nu * lam1 / (three * ell0 * ell0)
        - four * (nu**3) * kay * lam0 / (three * ell0**4)
        - (nu**2) * lam1 * (wall + ell0) / (ell0 * ell0 * wall * wall * wall)
        - four * (nu**4) * kay * lam0 * (wall + ell0) / (ell0**4 * wall**3)
    )
    v_coord = one - v - nu * v * v
    z_bound = max(abs(z_box.lo), abs(z_box.hi)) if included else float("inf")
    v_abs = max(abs(v_coord.lo), abs(v_coord.hi))
    eps_max = nu_hi * kay.hi if included else float("inf")
    remainder = 2.0 * eps_max * v_abs * z_bound if included else float("inf")
    return {
        "nu_lo": nu_lo,
        "nu_hi": nu_hi,
        "v_lo": v_lo,
        "v_hi": v_hi,
        "L_lo": L_lo,
        "L_hi": L_hi,
        "picard_included": included,
        "z_bound": z_bound,
        "v_abs_max": v_abs,
        "eps_max": eps_max,
        "remainder_majorant": remainder,
        "usable_c2_delta": bool(included) and remainder < 1.0,
        "finite": bool(included) and z_bound < float("inf"),
    }


def sample_z_cancelled() -> Interval:
    """Sound real-line enclosure of holomorphic ``Z`` at a kill sample."""
    nu = Interval.from_rational(_SAMPLE_KILL_NU)
    r1 = Interval.from_rational(_SAMPLE_KILL_R1)
    v = Interval.from_rational(_SAMPLE_KILL_V)
    lam0 = r1 * (Interval.from_rational(Fraction(2)) - r1)
    lam1 = Interval.point(_KILL_LAM1)
    one = Interval.from_rational(Fraction(1))
    two = Interval.from_rational(Fraction(2))
    three = Interval.from_rational(Fraction(3))
    four = Interval.from_rational(Fraction(4))
    nine = Interval.from_rational(Fraction(9))
    v0 = two / (one + (one + four * nu).sqrt())
    ell0 = one + two * nu * v0
    k0 = (three * v0) / ell0
    pad = Interval(-0.05, 0.05)
    guess = k0 + pad
    phi = (
        k0
        + (nu * nu * guess * guess * lam1) / (ell0 * ell0)
        + (four * nu * nu * nu * nu * guess * guess * guess * lam0)
        / (ell0 * ell0 * ell0 * ell0)
    )
    if not _contains(guess, phi):
        raise ValueError("sample_z_cancelled Picard box does not contain its image")
    kay = phi
    wall = one + nu * (v0 + v)
    s = v0 + v
    q1 = (
        four * v0
        - two * v0 * v0
        - three * v0 * v
        + nu * (four * v0 * v0 - three * v0 * s * s)
        - nu * nu * v0 * s * s * s
        + (v0 - v) * (-two - nu * v0)
    )
    z0 = ell0 * q1 / (nine * v0 * v0 * wall * wall * wall)
    return (
        (k0 / kay) ** 2 * z0
        - nu * lam1 / (three * ell0 * ell0)
        - four * (nu**3) * kay * lam0 / (three * ell0**4)
        - (nu**2) * lam1 * (wall + ell0) / (ell0 * ell0 * wall * wall * wall)
        - four * (nu**4) * kay * lam0 * (wall + ell0) / (ell0**4 * wall**3)
    )


@dataclass(frozen=True)
class CancelledNReport:
    """Cancelled-N holomorphic Z on the kill compact. Not first-hit or G1."""

    identities: Mapping[str, str]
    sample_abs_z: float
    enclosure: Mapping[str, float | bool]
    sample_below_bound: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-cancelled-n-v1",
            "identities": dict(self.identities),
            "sample_abs_z": self.sample_abs_z,
            "enclosure": dict(self.enclosure),
            "sample_below_bound": self.sample_below_bound,
            "kill_compact": "lambda1=-2, L in [0, 1], nu in [0, 0.02], v in [-0.5, 1.5]",
            "honesty": dict(self.honesty),
            "scope": (
                "Cancelled-N holomorphic Z on the lambda1=-2 kill compact "
                "with 2 eps |V| |Z| < 1. Slow-line only. Not T-h along the "
                "orbit, C!=0 |g_h|, first-hit, G1, or Hilbert XVI."
            ),
        }


def report() -> CancelledNReport:
    """Replay cancelled-N identities and enclose a usable slow-line Z bound."""
    identities = identity_verdicts()
    sample = sample_z_cancelled()
    sample_abs = sample.abs().hi
    try:
        enclosure = enclose_holomorphic_z()
        sealed = (
            bool(enclosure["finite"])
            and bool(enclosure["usable_c2_delta"])
            and all(status == "PROVED" for status in identities.values())
        )
        z_bound = float(enclosure["z_bound"])
        below = sealed and sample_abs <= z_bound
        flag = sealed and below
    except (ValueError, ZeroDivisionError):
        enclosure = {
            "nu_lo": _NU_LO,
            "nu_hi": _NU_HI,
            "v_lo": _VCHART_LO,
            "v_hi": _VCHART_HI,
            "L_lo": _L_LO,
            "L_hi": _L_HI,
            "picard_included": False,
            "z_bound": float("inf"),
            "v_abs_max": float("inf"),
            "eps_max": float("inf"),
            "remainder_majorant": float("inf"),
            "usable_c2_delta": False,
            "finite": False,
        }
        below = False
        flag = False
    return CancelledNReport(
        identities=identities,
        sample_abs_z=sample_abs,
        enclosure=enclosure,
        sample_below_bound=below,
        honesty=_honesty(cancelled_n_usable_z=flag),
    )
