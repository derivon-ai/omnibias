# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Leading fold (sep=0) I-map: first derivative and C2 of log sensitivity.

Gronwall ``exp(C * sigma * kappa)`` on a frozen blow-up scale is not the
leading derivative at coalescence. The slow-line antiderivative

    I(x) = log|x - r| - r / (x - r)

inverted at a sep-independent exit height determines a gap
``delta = r - x > 0`` with ``I(r - delta) ~ kappa``. Implicit
differentiation then gives the exact algebraic identities

    dx/dkappa = delta^2 / (r - delta),
    d2x/dkappa2 = -delta^3 (2 r - delta) / (r - delta)^3,
    (log x')_kappa = -delta (2 r - delta) / (r - delta)^2,
    (log x')_kappa kappa = 2 r^2 delta^2 / (r - delta)^4.

The reciprocal gap ``delta = r / kappa`` yields
``dx/dkappa = r / (kappa^2 - kappa)`` and
``(log x')_kappa kappa = 2 kappa^2 / (kappa - 1)^4``.

These identities do not prove a physical remainder, first-hit
completeness, chart O, G1, G4, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.verdict import adjudicate_residual
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.honesty_inventory import build_honesty

__all__ = [
    "FoldLeadingReport",
    "chi_log_second_difference",
    "fold_d2x_dkappa2",
    "fold_dx_dkappa",
    "fold_log_d2_dkappa2",
    "fold_log_deriv_dkappa",
    "fold_reciprocal_dx",
    "fold_reciprocal_log_c2",
    "relative_remainder",
    "report",
    "residual_chi_log_second",
    "residual_fold_d2x",
    "residual_fold_dx",
    "residual_fold_log_d1",
    "residual_fold_log_d2",
    "residual_fold_reciprocal",
    "residual_fold_reciprocal_log_c2",
    "residual_interface_relative",
    "residual_relative_remainder",
    "residual_zeta_rho",
    "residual_lifted_dx",
    "residual_lifted_d2x",
    "residual_lifted_log_d1",
    "residual_lifted_log_c2",
    "residual_z_relative_on_lift",
]


def _honesty() -> dict[str, object]:
    return build_honesty(
        g1_passed=False,
        g4_passed=False,
        full_graphic_cyclicity_proved=False,
        full_hilbert16_solved=False,
        physical_return_membership_proved=False,
        scale_dichotomy_c2_remainder=False,
        fold_leading_c2_remainder=False,
        inner_z_compact_bound=False,
        orbit_continuation_on_kill=False,
        hk_theorem_24_used=False,
    )


def fold_dx_dkappa(delta: Fraction, rstar: Fraction) -> Fraction:
    """``dx/dκ`` for ``x = r - delta`` along ``I(r - delta) = κ``."""
    wall = rstar - delta
    if wall == 0:
        raise ValueError("fold_dx_dkappa requires delta != rstar")
    return delta**2 / wall


def fold_d2x_dkappa2(delta: Fraction, rstar: Fraction) -> Fraction:
    """Second κ-derivative of the same implicit I-map."""
    wall = rstar - delta
    if wall == 0:
        raise ValueError("fold_d2x_dkappa2 requires delta != rstar")
    return -(delta**3 * (2 * rstar - delta)) / wall**3


def fold_log_deriv_dkappa(delta: Fraction, rstar: Fraction) -> Fraction:
    """First κ-derivative of ``log(dx/dκ)`` along the I-map."""
    wall = rstar - delta
    if wall == 0:
        raise ValueError("fold_log_deriv_dkappa requires delta != rstar")
    return -(delta * (2 * rstar - delta)) / wall**2


def fold_log_d2_dkappa2(delta: Fraction, rstar: Fraction) -> Fraction:
    """Second κ-derivative of ``log(dx/dκ)`` along the I-map."""
    wall = rstar - delta
    if wall == 0:
        raise ValueError("fold_log_d2_dkappa2 requires delta != rstar")
    return (2 * rstar**2 * delta**2) / wall**4


def fold_reciprocal_dx(rstar: Fraction, kappa: Fraction) -> Fraction:
    """``dx/dκ`` at the reciprocal gap ``delta = r / kappa``."""
    if kappa in (0, 1):
        raise ValueError("fold_reciprocal_dx requires kappa not in {0, 1}")
    return rstar / (kappa**2 - kappa)


def fold_reciprocal_log_c2(kappa: Fraction) -> Fraction:
    """Exact ``(log x')_{κκ}`` at ``delta = r / kappa``."""
    if kappa in (0, 1):
        raise ValueError("fold_reciprocal_log_c2 requires kappa not in {0, 1}")
    return (2 * kappa**2) / (kappa - 1) ** 4


def residual_fold_dx(delta: Fraction, rstar: Fraction) -> Fraction:
    return fold_dx_dkappa(delta, rstar) * (rstar - delta) - delta**2


def residual_fold_d2x(delta: Fraction, rstar: Fraction) -> Fraction:
    return fold_d2x_dkappa2(delta, rstar) * (rstar - delta) ** 3 + delta**3 * (
        2 * rstar - delta
    )


def residual_fold_log_d1(delta: Fraction, rstar: Fraction) -> Fraction:
    return fold_log_deriv_dkappa(delta, rstar) * (rstar - delta) ** 2 + delta * (
        2 * rstar - delta
    )


def residual_fold_log_d2(delta: Fraction, rstar: Fraction) -> Fraction:
    return fold_log_d2_dkappa2(delta, rstar) * (rstar - delta) ** 4 - (
        2 * rstar**2 * delta**2
    )


def residual_fold_reciprocal(rstar: Fraction, kappa: Fraction) -> Fraction:
    delta = rstar / kappa
    return fold_dx_dkappa(delta, rstar) - fold_reciprocal_dx(rstar, kappa)


def residual_fold_reciprocal_log_c2(rstar: Fraction, kappa: Fraction) -> Fraction:
    delta = rstar / kappa
    return fold_log_d2_dkappa2(delta, rstar) - fold_reciprocal_log_c2(kappa)


def relative_remainder(delta: Fraction, eps: Fraction, rho: Fraction) -> Fraction:
    """Leading relative remainder ``(B_eps - B)/B = eps * rho / delta^2``."""
    if delta == 0:
        raise ValueError("relative_remainder requires nonzero delta")
    return eps * rho / delta**2


def residual_relative_remainder(
    delta: Fraction, eps: Fraction, rho: Fraction, b_eps: Fraction
) -> Fraction:
    """``(B_eps - delta^2) * delta^2 - eps * rho * delta^2``."""
    wall = delta**2
    return (b_eps - wall) * wall - eps * rho * wall


def residual_interface_relative(
    delta: Fraction, eps: Fraction, rho: Fraction, margin: Fraction
) -> Fraction:
    """At ``delta^2 = M * eps`` the relative remainder equals ``rho / M``."""
    if margin == 0:
        raise ValueError("residual_interface_relative requires nonzero margin")
    return relative_remainder(delta, eps, rho) - rho / margin


def residual_zeta_rho(
    x: Fraction,
    lam0: Fraction,
    lam1: Fraction,
    eps: Fraction,
    beta0: Fraction,
    z_jet: Fraction,
) -> Fraction:
    """``B_eps - B_- - beta0 eps x^3 - eps^2 x^3 Z`` from the zeta split."""
    wall = -eps * x
    zeta = -1 + beta0 * wall + eps * wall * z_jet
    b_eps = lam0 + lam1 * x - x**2 * zeta
    b_minus = lam0 + lam1 * x + x**2
    return b_eps - b_minus - beta0 * eps * x**3 - eps**2 * x**3 * z_jet


def lifted_b(x: Fraction, rstar: Fraction, mu: Fraction) -> Fraction:
    """Inner leading field ``(x-r)^2 + mu`` with ``mu = beta0 eps r^3`` frozen."""
    return (x - rstar) ** 2 + mu


def residual_lifted_dx(x: Fraction, rstar: Fraction, mu: Fraction) -> Fraction:
    """``(dx/dκ) x - ((x-r)^2 + mu)`` along the lifted I-map."""
    if x == 0:
        raise ValueError("residual_lifted_dx requires nonzero x")
    dx = lifted_b(x, rstar, mu) / x
    return dx * x - lifted_b(x, rstar, mu)


def residual_lifted_d2x(x: Fraction, rstar: Fraction, mu: Fraction) -> Fraction:
    """Second κ-derivative identity for the lifted I-map."""
    if x == 0:
        raise ValueError("residual_lifted_d2x requires nonzero x")
    wall = lifted_b(x, rstar, mu)
    d2x = wall * (2 * (x - rstar) * x - wall) / x**3
    return d2x * x**3 - wall * (2 * (x - rstar) * x - wall)


def residual_lifted_log_d1(x: Fraction, rstar: Fraction, mu: Fraction) -> Fraction:
    """First κ-derivative of ``log(dx/dκ)`` on the lifted I-map."""
    if x == 0:
        raise ValueError("residual_lifted_log_d1 requires nonzero x")
    wall = lifted_b(x, rstar, mu)
    log_d1 = (2 * (x - rstar) * x - wall) / x**2
    return log_d1 * x**2 - (2 * (x - rstar) * x - wall)


def residual_lifted_log_c2(x: Fraction, rstar: Fraction, mu: Fraction) -> Fraction:
    """Leading C2 of ``log(dx/dκ)`` on the lifted I-map."""
    if x == 0:
        raise ValueError("residual_lifted_log_c2 requires nonzero x")
    wall = lifted_b(x, rstar, mu)
    log_c2 = 2 * (rstar**2 + mu) * wall / x**4
    return log_c2 * x**4 - 2 * (rstar**2 + mu) * wall


def residual_z_relative_on_lift(eps: Fraction, beta0: Fraction, z_jet: Fraction) -> Fraction:
    """On the lift ``B >= beta0 eps x^3``, the Z remainder is ``O(eps)`` relatively."""
    if beta0 == 0 or eps == 0:
        raise ValueError("residual_z_relative_on_lift requires nonzero beta0 and eps")
    return (eps**2 * z_jet) / (beta0 * eps) - (eps * z_jet) / beta0


def residual_chi_log_second(
    coeff: Fraction, sep: Fraction, r1: Fraction, kappa0: Fraction, shift: Fraction
) -> Fraction:
    """Second κ-difference of the leading ``log D' = -coeff * (sep/r1) * κ``."""

    def leading(kappa: Fraction) -> Fraction:
        return -coeff * (sep / r1) * kappa

    plus = leading(kappa0 + shift) - leading(kappa0)
    minus = leading(kappa0) - leading(kappa0 - shift)
    return plus - minus


def chi_log_second_difference(
    coeff: float, sep: float, r1: float, kappa: float, shift: float
) -> float:
    def leading(k: float) -> float:
        return -coeff * (sep / r1) * k

    return (leading(kappa + shift) - leading(kappa)) - (
        leading(kappa) - leading(kappa - shift)
    )


def _verdict(residual: Fraction) -> str:
    box = Interval.point(float(residual)) if residual == 0 else Interval.from_rational(residual)
    return adjudicate_residual(box, existential=False).status


def identity_verdicts() -> dict[str, str]:
    delta, rstar, kappa = Fraction(1), Fraction(3), Fraction(3)
    return {
        "fold_dx": _verdict(residual_fold_dx(delta, rstar)),
        "fold_d2x": _verdict(residual_fold_d2x(delta, rstar)),
        "fold_log_d1": _verdict(residual_fold_log_d1(delta, rstar)),
        "fold_log_d2": _verdict(residual_fold_log_d2(delta, rstar)),
        "fold_reciprocal": _verdict(residual_fold_reciprocal(rstar, kappa)),
        "fold_reciprocal_log_c2": _verdict(residual_fold_reciprocal_log_c2(rstar, kappa)),
        "relative_remainder": _verdict(
            residual_relative_remainder(Fraction(2), Fraction(1, 4), Fraction(3), Fraction(19, 4))
        ),
        "interface_relative": _verdict(
            residual_interface_relative(Fraction(2), Fraction(1), Fraction(3), Fraction(4))
        ),
        "zeta_rho": _verdict(
            residual_zeta_rho(
                Fraction(2),
                Fraction(9, 4),
                Fraction(-3),
                Fraction(1, 5),
                Fraction(1, 3),
                Fraction(7, 2),
            )
        ),
        "lifted_dx": _verdict(residual_lifted_dx(Fraction(2), Fraction(3), Fraction(1, 4))),
        "lifted_d2x": _verdict(residual_lifted_d2x(Fraction(2), Fraction(3), Fraction(1, 4))),
        "lifted_log_d1": _verdict(
            residual_lifted_log_d1(Fraction(2), Fraction(3), Fraction(1, 4))
        ),
        "lifted_log_c2": _verdict(
            residual_lifted_log_c2(Fraction(2), Fraction(3), Fraction(1, 4))
        ),
        "z_relative_on_lift": _verdict(
            residual_z_relative_on_lift(Fraction(1, 5), Fraction(1, 3), Fraction(7, 2))
        ),
        "chi_log_second": _verdict(
            residual_chi_log_second(
                Fraction(2), Fraction(1, 5), Fraction(3, 2), Fraction(4), Fraction(1, 3)
            )
        ),
    }


@dataclass(frozen=True)
class FoldLeadingReport:
    """Finite replay of the fold I-map derivatives. Not G1."""

    identities: Mapping[str, str]
    large_kappa_dx: float
    large_kappa_log_c2: float
    dx_decays: bool
    chi_c2_vanishes: bool
    inner_c2: float
    inner_c2_o_eps: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-fold-leading-v1",
            "identities": dict(self.identities),
            "large_kappa_dx": self.large_kappa_dx,
            "large_kappa_log_c2": self.large_kappa_log_c2,
            "dx_decays": self.dx_decays,
            "chi_c2_vanishes": self.chi_c2_vanishes,
            "inner_c2": self.inner_c2,
            "inner_c2_o_eps": self.inner_c2_o_eps,
            "honesty": dict(self.honesty),
            "scope": (
                "Exact fold I-map, lifted inner map, and leading C2 of "
                "log sensitivity. Not a physical remainder, G1, or Hilbert XVI."
            ),
        }


def report(*, rstar: float = 1.5, kappa: float = 20.0, eps: float = 0.05) -> FoldLeadingReport:
    """Evaluate the reciprocal-gap asymptote and the lifted inner C2."""
    if kappa <= 1.0 or rstar <= 0.0 or eps <= 0.0:
        raise ValueError("report requires kappa > 1, positive rstar, and positive eps")
    dx = rstar / (kappa * kappa - kappa)
    log_c2 = 2.0 * (kappa * kappa) / (kappa - 1.0) ** 4
    mu = (1.0 / 3.0) * eps * (rstar**3)
    inner_c2 = 2.0 * mu * (rstar * rstar + mu) / (rstar**4)
    honesty = _honesty()
    return FoldLeadingReport(
        identities=identity_verdicts(),
        large_kappa_dx=dx,
        large_kappa_log_c2=log_c2,
        dx_decays=0.0 < dx < 2.0 * rstar / (kappa * kappa),
        chi_c2_vanishes=abs(chi_log_second_difference(2.0, 0.2, 1.5, 4.0, 0.3)) < 1e-15,
        inner_c2=inner_c2,
        inner_c2_o_eps=0.0 < inner_c2 < 10.0 * eps,
        honesty=honesty,
    )
