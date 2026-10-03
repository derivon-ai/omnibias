# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Leading slow-line entry-exit algebra for the coalescing quadratic passage.

The scale dichotomy of :mod:`omnibias.dynamics.scale_dichotomy` compares
``sigma * kappa`` to a W-ratio against a *frozen* outgoing height. That is a
dichotomy of one matching strategy. The first-root / coalescing variation
already supplies a prefactor ``exp(Psi_pre) = O(sep^2)``. The product

    sep^2 * (h_1 / (epsilon^3 * mu * sep^2))^(C epsilon)
        = sep^(2 - 2 C epsilon) * (h_1 / (epsilon^3 * mu))^(C epsilon)

is bounded for small ``epsilon``, uniformly in ``sep in (0, 1]``. On the
named kill sequence the sep-power tends to zero faster than any
``exp(O(1/epsilon))`` outgoing factor can grow.

This module certifies those exact identities and the slow-line partial-
fraction antiderivative. It does not prove a C2 remainder, first-hit
completeness, the ``sep = 0`` saddle-node chart, the shrinking-root chart
``L -> 0``, G1, G4, or Hilbert XVI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.scale_dichotomy import KILL_EPS, KILL_LAMBDA1, kill_sep, rstar

__all__ = [
    "EntryExitLeadingReport",
    "dx_dy_y_dominated",
    "height_inflation_delta_x",
    "kill_tracked_log",
    "partial_fraction_coeffs",
    "report",
    "residual_double_root_entry_exit",
    "residual_height_inflation_dx_dy",
    "residual_partial_fraction_numerators",
    "residual_product_alpha_one",
    "tracked_outgoing_product",
    "tracked_product_identity_log",
]


def _honesty() -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        scale_dichotomy_c2_remainder=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
        new_closing_map=False,
    )


def partial_fraction_coeffs(r1: Fraction, r2: Fraction) -> tuple[Fraction, Fraction]:
    """Coefficients of ``x/((x-r1)(x-r2)) = A/(x-r1) + C/(x-r2)``."""
    if r1 == r2:
        raise ValueError("partial_fraction_coeffs requires distinct roots")
    return r1 / (r1 - r2), r2 / (r2 - r1)


def residual_partial_fraction_numerators(x: Fraction, r1: Fraction, r2: Fraction) -> Fraction:
    """``A (x-r2) + C (x-r1) - x``; vanishes iff the partial-fraction identity holds."""
    coeff_a, coeff_c = partial_fraction_coeffs(r1, r2)
    return coeff_a * (x - r2) + coeff_c * (x - r1) - x


def residual_double_root_entry_exit(x: Fraction, rstar_value: Fraction) -> Fraction:
    """Numerator of ``x/(x-r)^2 - 1/(x-r) - r/(x-r)^2``, namely ``x - (x-r) - r``."""
    return x - (x - rstar_value) - rstar_value


def residual_product_alpha_one(
    sep: Fraction, height: Fraction, eps: Fraction, mu: Fraction
) -> Fraction:
    """``sep^2 * h / (eps^3 mu sep^2) - h / (eps^3 mu)`` at outgoing exponent 1."""
    if sep == 0 or eps == 0 or mu == 0:
        raise ValueError("residual_product_alpha_one requires nonzero sep, eps, mu")
    left = sep**2 * height / (eps**3 * mu * sep**2)
    right = height / (eps**3 * mu)
    return left - right


def residual_height_inflation_dx_dy(
    eps: Fraction, kappa_field: Fraction, wall: Fraction, height: Fraction
) -> Fraction:
    """``eps (kappa_field * height) / (wall * height) - eps * kappa_field / wall``.

    If the slow-fast remainder is height-dominated (``D = kappa_field * y``),
    the physical ``dx/dy`` equals ``epsilon * k / x`` independently of height.
    """
    if wall == 0 or height == 0:
        raise ValueError("residual_height_inflation_dx_dy requires nonzero wall and height")
    left = eps * (kappa_field * height) / (wall * height)
    right = eps * kappa_field / wall
    return left - right


def tracked_product_identity_log(
    sep: float, height: float, eps: float, mu: float, alpha: float
) -> tuple[float, float]:
    """Logarithm of both sides of the tracked outgoing product identity.

    Evaluated in the log domain so ``sep = exp(-1/eps^2)`` does not underflow
    ``sep^2`` to zero in float64.
    """
    if not all(math.isfinite(v) and v > 0 for v in (sep, height, eps, mu)):
        raise ValueError("tracked_product_identity_log requires positive finite inputs")
    log_sep = math.log(sep)
    log_h = math.log(height)
    log_core = 3.0 * math.log(eps) + math.log(mu)
    left = 2.0 * log_sep + alpha * (log_h - log_core - 2.0 * log_sep)
    right = (2.0 - 2.0 * alpha) * log_sep + alpha * (log_h - log_core)
    return left, right


def tracked_outgoing_product(
    sep: float, height: float, eps: float, mu: float, c_eps: float
) -> float:
    """``sep^2 * (h / (eps^3 mu sep^2))^(c_eps)`` evaluated in the log domain."""
    left, _right = tracked_product_identity_log(sep, height, eps, mu, c_eps)
    return math.exp(left)


def kill_tracked_log(*, eps: float = KILL_EPS, c_factor: float = 2.0) -> float:
    """Sep-power log ``(2 - 2 C eps) log sep`` on ``sep = exp(-1/eps^2)``.

    Uses the exact ``log sep = -1/eps^2`` so the evaluation does not depend
    on whether ``exp(-1/eps^2)`` underflows.
    """
    if eps <= 0.0:
        raise ValueError("kill_tracked_log requires positive epsilon")
    return (2.0 - 2.0 * c_factor * eps) * (-1.0 / (eps * eps))


def dx_dy_y_dominated(eps: float, kappa_field: float, wall: float) -> float:
    """Leading ``dx/dy`` after the height dominates ``B``."""
    if wall == 0.0:
        raise ValueError("dx_dy_y_dominated requires a nonzero wall")
    return eps * kappa_field / wall


def height_inflation_delta_x(
    eps: float, kappa_field: float, wall: float, y_lo: float, y_hi: float
) -> float:
    """Integrated ``dx = (eps k / x) dy`` from ``y_lo`` to ``y_hi``."""
    return dx_dy_y_dominated(eps, kappa_field, wall) * (y_hi - y_lo)


def identity_verdicts() -> dict[str, str]:
    names = {
        "partial_fraction_numerators": residual_partial_fraction_numerators(
            Fraction(2), Fraction(1), Fraction(4)
        ),
        "double_root_entry_exit": residual_double_root_entry_exit(Fraction(5, 2), Fraction(3, 2)),
        "product_alpha_one": residual_product_alpha_one(
            Fraction(2), Fraction(3), Fraction(1, 2), Fraction(5)
        ),
        "height_inflation_dx_dy": residual_height_inflation_dx_dy(
            Fraction(1, 3), Fraction(7, 2), Fraction(5), Fraction(4)
        ),
    }
    out: dict[str, str] = {}
    for name, residual in names.items():
        box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
        out[name] = adjudicate_residual(box, existential=False).status
    return out


@dataclass(frozen=True)
class EntryExitLeadingReport:
    """Finite replay of the tracked product and slow-line leading map."""

    identities: Mapping[str, str]
    kill_log: float
    kill_product: float
    height_inflation_dx: float
    product_bounded_on_unit_sep: bool
    kill_log_negative: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-entry-exit-leading-v1",
            "identities": dict(self.identities),
            "kill_log": self.kill_log,
            "kill_product": self.kill_product,
            "height_inflation_dx": self.height_inflation_dx,
            "product_bounded_on_unit_sep": self.product_bounded_on_unit_sep,
            "kill_log_negative": self.kill_log_negative,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact slow-line partial fractions and the tracked "
                "sep^2 * outgoing-factor product. Not a C2 remainder, G1, "
                "or Hilbert XVI."
            ),
        }


def report(*, eps: float = KILL_EPS, c_factor: float = 2.0) -> EntryExitLeadingReport:
    """Replay the leading identities on the named kill sequence."""
    sep = kill_sep(eps)
    mu = 1.0
    height = eps**3  # Stage-C matching height h_1 = Theta(eps^3), y1 = 1
    kill_log = kill_tracked_log(eps=eps, c_factor=c_factor)
    # The kill-sequence product is exp(kill_log) times the sep-independent
    # (h_1 / (eps^3 mu))^{C eps} = 1 factor used here.
    product = math.exp(kill_log) if kill_log > -745.0 else 0.0
    wall = rstar(KILL_LAMBDA1)
    y_lo = max(sep * sep, math.exp(-700.0))
    dx = height_inflation_delta_x(eps, 1.0, wall, y_lo, 1.0)
    unit_seps = (1.0, 0.5, 0.1)
    unit_products = [
        tracked_outgoing_product(s, height, eps, mu, c_factor * eps) for s in unit_seps
    ]
    honesty = _honesty()
    return EntryExitLeadingReport(
        identities=identity_verdicts(),
        kill_log=kill_log,
        kill_product=product,
        height_inflation_dx=dx,
        product_bounded_on_unit_sep=all(value <= 1.0 + 1e-12 for value in unit_products)
        and kill_log < 0.0,
        kill_log_negative=kill_log < 0.0,
        honesty=honesty,
    )
