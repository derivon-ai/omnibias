# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""k=1+O(eps) and cubic remainder prefactor on a compact (V,h) rectangle.

On the ``C=0`` normal chart the exact height coefficient is

    k_normal = (1 + 2 nu v)(1 - A nu v).

At ``A=1`` this is ``1 + nu v - 2 nu^2 v^2``. The first-order
normal-coordinate jet ``k_lead = 1 - nu (V-1)`` matches it through
order ``nu`` once ``V = 1 - v - nu v^2``, with exact remainder
``k_normal - k_lead = -3 nu^2 v^2``. On ``|v| <= 2`` and
``nu <= 1/8``, ``|k_normal - 1| <= 3 nu``.

The cubic correction past the two-root leading term is

    q_cubic = eps^3 (x-r1)(x-r2) + eps^4 x^3 / 3

at ``V = -eps x``. Its size relative to the comparison
``eps (eps^2 + T)`` with ``T = V^2 / 2`` is at most ``2 eps |V|``
times a holomorphic ``Z`` factor, because

    2 (eps^2 + T) - w^2 = 2 eps^2.

This is not a bound on ``Z``, not ``T-h = O(eps)`` along the actual
orbit, not the transversal event, G1, or Hilbert XVI. The ``C != 0``
height mixing ``|g_h| = O(eps^2)`` is not this slice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "KZetaRemainderReport",
    "identity_verdicts",
    "k_lead",
    "k_normal",
    "k_turn",
    "q_cubic",
    "report",
    "residual_cubic_vs_tworoot",
    "residual_k_box_gap",
    "residual_k_lead_embed",
    "residual_k_normal",
    "residual_k_normal_vs_lead",
    "residual_k_turn",
    "residual_relative_prefactor",
]

_SAMPLE_NU = Fraction(1, 16)
_SAMPLE_VCHART = Fraction(1, 2)
_SAMPLE_A = Fraction(1)
_SAMPLE_C = Fraction(0)
_SAMPLE_R1 = Fraction(1, 5)
_SAMPLE_R2 = Fraction(9, 5)
_SAMPLE_X = Fraction(1)
_SAMPLE_VMAX = Fraction(2)


def _honesty(*, k_zeta_compact: bool) -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        fold_leading_c2_remainder=False,
        k_zeta_compact=k_zeta_compact,
        outgoing_first_hit=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def k_turn(a: Fraction, nu: Fraction, v: Fraction, c: Fraction) -> Fraction:
    """Unfolding turning-geometry ``k = 1 - A nu v + C nu^2 v^2``."""
    return 1 - a * nu * v + c * nu**2 * v * v


def k_normal(a: Fraction, nu: Fraction, v: Fraction) -> Fraction:
    """``C=0`` normal-coordinate ``k = ell (1 - A nu v)``."""
    ell = 1 + 2 * nu * v
    return ell * (1 - a * nu * v)


def k_lead(nu: Fraction, v_coord: Fraction) -> Fraction:
    """First-order jet ``1 - nu (V-1)``."""
    return 1 - nu * (v_coord - 1)


def q_cubic(
    L: Fraction, lam1: Fraction, eps: Fraction, w: Fraction
) -> Fraction:
    """Cubic ``q(-w)`` with ``zeta = -1 + V/3`` and ``V = -w``."""
    return L * eps**3 + lam1 * eps**2 * w + eps * w * w * (1 + w / 3)


def residual_k_turn(
    kay: Fraction, a: Fraction, nu: Fraction, v: Fraction, c: Fraction
) -> Fraction:
    """``k - (1 - A nu v + C nu^2 v^2)``."""
    return kay - k_turn(a, nu, v, c)


def residual_k_normal(a: Fraction, nu: Fraction, v: Fraction) -> Fraction:
    """``ell (1-A nu v)`` versus the expanded polynomial at ``C=0``."""
    expanded = 1 + (2 - a) * nu * v - 2 * a * nu**2 * v * v
    return k_normal(a, nu, v) - expanded


def residual_k_lead_embed(nu: Fraction, v: Fraction) -> Fraction:
    """``k_lead`` versus ``1 + nu v + nu^2 v^2`` on ``V = 1 - v - nu v^2``."""
    v_coord = 1 - v - nu * v * v
    return k_lead(nu, v_coord) - 1 - nu * v - nu**2 * v * v


def residual_k_normal_vs_lead(nu: Fraction, v: Fraction) -> Fraction:
    """``k_normal - k_lead + 3 nu^2 v^2`` at ``A=1``, ``C=0``."""
    v_coord = 1 - v - nu * v * v
    return k_normal(Fraction(1), nu, v) - k_lead(nu, v_coord) + 3 * nu**2 * v * v


def residual_k_box_gap(v_coord: Fraction) -> Fraction:
    """``9 - (V-1)^2`` versus ``8 + 2V - V^2``; nonnegative on ``|V|<=2``."""
    return (9 - (v_coord - 1) ** 2) - (8 + 2 * v_coord - v_coord * v_coord)


def residual_cubic_vs_tworoot(
    L: Fraction,
    lam1: Fraction,
    eps: Fraction,
    x: Fraction,
    r1: Fraction,
    r2: Fraction,
) -> Fraction:
    """Cubic ``q`` versus two-root leading plus ``eps^4 x^3 / 3``."""
    w = eps * x
    return q_cubic(L, lam1, eps, w) - eps**3 * (x - r1) * (x - r2) - eps**4 * x**3 / 3


def residual_relative_prefactor(eps: Fraction, w: Fraction) -> Fraction:
    """``2(eps^2 + T) - w^2 - 2 eps^2`` with ``T = w^2 / 2``."""
    kinetic = w * w / 2
    return 2 * (eps**2 + kinetic) - w * w - 2 * eps**2


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    nu, v, a, c = _SAMPLE_NU, _SAMPLE_VCHART, _SAMPLE_A, _SAMPLE_C
    r1, r2, x = _SAMPLE_R1, _SAMPLE_R2, _SAMPLE_X
    L, lam1 = r1 * r2, -(r1 + r2)
    kay = k_turn(a, nu, v, c)
    return {
        "k_turn": _verdict(residual_k_turn(kay, a, nu, v, c)),
        "k_normal": _verdict(residual_k_normal(a, nu, v)),
        "k_lead_embed": _verdict(residual_k_lead_embed(nu, v)),
        "k_normal_vs_lead": _verdict(residual_k_normal_vs_lead(nu, v)),
        "k_box_gap": _verdict(residual_k_box_gap(_SAMPLE_VMAX)),
        "cubic_vs_tworoot": _verdict(
            residual_cubic_vs_tworoot(L, lam1, nu, x, r1, r2)
        ),
        "relative_prefactor": _verdict(residual_relative_prefactor(nu, nu * x)),
    }


def _k_bound_on_box() -> dict[str, float | bool]:
    """``|k_normal-1| <= 3 nu`` on ``|v|<=2``, ``nu=1/n`` for ``n>=8``."""
    samples: list[float] = []
    ok = True
    for n in (8, 16, 32, 64):
        nu = Fraction(1, n)
        for v in (_SAMPLE_VMAX, -_SAMPLE_VMAX, Fraction(0), Fraction(1)):
            gap = abs(k_normal(Fraction(1), nu, v) - 1)
            samples.append(float(gap / nu))
            if gap > 3 * nu:
                ok = False
    return {"ratio_hi": max(samples), "within_three": ok}


@dataclass(frozen=True)
class KZetaRemainderReport:
    """k=1+O(eps) C=0 jet and cubic remainder prefactor. Not first-hit or G1."""

    identities: Mapping[str, str]
    k_bound: Mapping[str, float | bool]
    k_zeta_compact: bool
    outgoing_first_hit: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-k-zeta-remainder-v1",
            "identities": dict(self.identities),
            "k_bound": dict(self.k_bound),
            "k_zeta_compact": self.k_zeta_compact,
            "outgoing_first_hit": self.outgoing_first_hit,
            "honesty": dict(self.honesty),
            "scope": (
                "On C=0, A=1 the normal k is 1+O(nu) with exact O(nu^2) "
                "remainder versus the V-jet; the cubic correction past "
                "the two-root leading term is eps^4 x^3/3. Not a Z bound, "
                "T-h along the orbit, height-section first-hit, G1, or "
                "Hilbert XVI."
            ),
        }


def report() -> KZetaRemainderReport:
    """Replay k=1+O(eps) identities on the C=0 compact rectangle."""
    identities = identity_verdicts()
    try:
        k_bound = _k_bound_on_box()
        sealed = (
            all(status == "PROVED" for status in identities.values())
            and bool(k_bound["within_three"])
        )
    except (ValueError, ZeroDivisionError):
        k_bound = {"ratio_hi": float("inf"), "within_three": False}
        sealed = False
    return KZetaRemainderReport(
        identities=identities,
        k_bound=k_bound,
        k_zeta_compact=sealed,
        outgoing_first_hit=False,
        honesty=_honesty(k_zeta_compact=sealed),
    )
