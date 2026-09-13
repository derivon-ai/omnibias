#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite algebra for the height-parametrized passage; not an analytic verifier."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def identities() -> list[str]:
    V, h, f, g, gh = sp.symbols("V h f g gh")
    denominator = f + h * g
    R = -V * h / denominator
    checks = {
        "normal_height_variation": sp.diff(R, h)
        + sp.diff(R, g) * gh
        + V * (f - h**2 * gh) / denominator**2,
    }
    eps, a, H, gamma = sp.symbols("epsilon a H gamma", positive=True)
    u, w, lam0, lam1, zeta = sp.symbols("u w lambda0 lambda1 zeta")
    F = eps**3 * lam0 + eps**2 * lam1 * V + eps * V**2 * zeta + h * g
    checks["scaled_slow_equation"] = F.subs({V: a * u, h: a**2 * w}) / a**2 - (
        eps**3 * lam0 / a**2 + eps**2 * lam1 * u / a + eps * u**2 * zeta + w * g
    )
    checks["scaled_height_equation"] = (-V * h).subs({V: a * u, h: a**2 * w}) / a**3 + u * w
    checks["compact_scale_parameters"] = H / (H + eps**3) + eps**3 / (H + eps**3) - 1
    checks["linear_perturbation_squared_margin"] = (
        eps - eps**4 / (H + eps**3) - eps * H / (H + eps**3)
    )
    checks["center_height_lower_margin"] = H / (H + eps**3) - gamma / (1 + gamma) - (
        H - gamma * eps**3
    ) / ((H + eps**3) * (1 + gamma))
    c, eta = sp.symbols("c eta", positive=True)
    W = sp.symbols("W", positive=True)
    B = u * W / (W + c)
    invariant = u**2 - 2 * W - 2 * c * sp.log(W)
    checks["limiting_first_integral"] = sp.diff(invariant, u) + sp.diff(invariant, W) * B
    J = (1 + c / eta) / (1 + c / W)
    checks["frozen_scale_variation"] = sp.diff(J, W) * B - sp.diff(B, W) * J
    checks["variation_initial_value"] = J.subs(W, eta) - 1
    checks["limiting_core_reflection"] = B.subs(u, -u) + B
    checks["physical_core_multiplier"] = 1 - lam0 * (eps**3 / a**2) / (H / a**2) - (
        1 - lam0 * eps**3 / H
    )
    delta, U = sp.symbols("delta U", positive=True)
    zero_slope = -u / (lam0 + delta * lam1 * u - delta**2 * u**2)
    checks["zero_height_log_multiplier_first_coefficient"] = sp.integrate(
        sp.diff(zero_slope, delta).subs(delta, 0), (u, U, -U)
    ) + 2 * lam1 * U**3 / (3 * lam0**2)

    # The physical endpoints move along v=sigma*rho*h. J is varied at
    # fixed V, and the analysis scale a stays frozen at the base trajectory.
    v, nu, C, rho, sigma, ell, RR, JJ = sp.symbols("v nu C rho sigma ell R J")
    vV = -1 / ell
    vh = -C * nu**2 * v / ell
    event_dV = -(vh - sigma * rho) * JJ / (vV + (vh - sigma * rho) * RR)
    event_dh = JJ + RR * event_dV
    K = (2 * sigma * rho * (v - 1) - 2) / (1 + (C * nu**2 * v + sigma * rho * ell) * RR)
    checks["physical_section_event_factor"] = (2 * sigma * rho * (v - 1) - 2) * event_dh - K * JJ
    checks["event_factor_at_zero_parameter"] = K.subs({nu: 0, ell: 1, RR: 1 - v}) + 2
    ell_actual = 1 + 2 * nu * v + C * nu**2 * h
    normal_coordinate = 1 - v - nu * v**2 - C * nu**2 * v * h
    checks["inverse_coordinate_V_derivative"] = sp.diff(normal_coordinate, v) * (-1 / ell_actual) - 1
    checks["inverse_coordinate_height_derivative"] = sp.diff(normal_coordinate, h) + sp.diff(
        normal_coordinate, v
    ) * (-C * nu**2 * v / ell_actual)
    for sign in (-1, 1):
        event = V - 1 + sign * rho * h + nu * rho**2 * h**2 + C * nu**2 * sign * rho * h**2
        checks[f"physical_event_height_derivative_{sign}"] = sp.diff(event, h) - (
            sign * rho * ell_actual + C * nu**2 * v
        ).subs(v, sign * rho * h)
    z, z0 = sp.symbols("z z0", positive=True)
    shift = -2 * eps**3 * (z - z0 - lam0 * sp.log(z / z0))
    checks["anchored_physical_log_derivative"] = sp.diff(shift, z) + 2 * eps**3 * (1 - lam0 / z)
    checks["tail_cubic_primitive"] = sp.diff(-eps**3 / (2 * V**2), V) - eps**3 / V**3
    checks["tail_quadratic_primitive"] = sp.diff(-eps**2 / V, V) - eps**2 / V**2
    checks["tail_logarithmic_primitive"] = sp.diff(eps * sp.log(V), V) - eps / V
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def rational_examples() -> dict[str, object]:
    # Illustrative gates for declared constants; the analytic theorem must
    # supply actual coefficient bounds and a small-parameter cutoff.
    eta_min, cmax, U = Q(1, 4), Q(2), Q(64)
    root_bound = Q(3)
    require(root_bound**2 * eta_min >= 2, "sqrt(2/eta_min) upper bound")
    core_lower = Q(1, 2) - cmax * root_bound / U
    core_upper = Q(1, 2) + 1 / U**2
    require(Q(3, 8) < core_lower <= core_upper < Q(5, 8), "core enters strict tail barriers")
    coefficient_bound, delta = Q(4), Q(1, 100000)
    eps = delta**2
    quotient_error = coefficient_bound * (1 / U**2 + delta / U + eps)
    require(quotient_error < Q(1, 8), "declared denominator perturbation")
    require(Q(1, 2) < 1 / (1 + quotient_error), "lower height barrier points inward")
    require(1 / (1 - quotient_error) < Q(3, 2), "upper height barrier points inward")
    theta = Q(13, 14)
    required_slope_error = (1 - theta) / 2
    require(1 - required_slope_error > theta, "eventual displacement slope separates")
    return {
        "core_lower_ratio": str(core_lower),
        "core_upper_ratio": str(core_upper),
        "declared_quotient_error": str(quotient_error),
        "regular_slope_upper": str(theta),
        "required_singular_slope_error": str(required_slope_error),
        "scope": "rational sufficient-inequality examples; no physical cutoff is certified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "symbolic_checks": identities(),
        "rational_examples": rational_examples(),
        "honesty": {
            "finite_algebra_replayed": True,
            "analytic_theorem_formally_verified": False,
            "physical_small_parameter_cutoff_certified": False,
            "all_grazing_scales_covered": False,
            "full_graphic_cyclicity_proved": False,
            "hilbert16_solved": False,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
