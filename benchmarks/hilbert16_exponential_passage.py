#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite algebra and limiting-profile gates for the exponential passage."""

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
    x, y, eps = sp.symbols("x y epsilon", positive=True)
    B, Bx, k, kx, ky, kxy = sp.symbols("B B_x k k_x k_y k_xy")
    denominator = B + k * y
    radial_slope = x * y / (eps * denominator)
    height_variation = sp.diff(radial_slope, y) + sp.diff(radial_slope, k) * ky
    exponent_integrand = (Bx + y * kx) / denominator
    total_denominator_derivative = Bx + y * kx + (k + y * ky) * radial_slope
    checks = {
        "exact_radial_height_variation": height_variation
        - x * (B - y**2 * ky) / (eps * denominator**2),
        "compensated_variation_identity": radial_slope / y
        - total_denominator_derivative / denominator
        - height_variation
        + exponent_integrand,
    }

    integrand_height_derivative = (
        (kx + y * kxy) * denominator - (Bx + y * kx) * (k + y * ky)
    ) / denominator**2
    numerator = Bx * (k + y * ky) - B * kx - B * y * kxy - k * y**2 * kxy + y**2 * kx * ky
    checks["actual_height_dependent_kernel"] = (
        integrand_height_derivative * denominator**2 + numerator
    )
    D0, expPsi = sp.symbols("D0 expPsi")
    height_kappa = -D0 * y * expPsi / (eps * denominator)
    coordinate_derivative = eps * denominator / (x * y)
    checks["kappa_derivative_change_of_variable_and_sign"] = (
        integrand_height_derivative * height_kappa * coordinate_derivative
        - D0 * expPsi * numerator / (x * denominator**2)
    )

    Bpositive = sp.symbols("Bpositive", positive=True)
    checks["limiting_level_integral"] = (
        sp.integrate(1 / (Bpositive + y) ** 2, (y, 0, sp.oo)) - 1 / Bpositive
    )
    L, eta = sp.symbols("L eta", positive=True)
    label_kappa = (-2) * (L / eta) * (Bpositive / L) * (-eps**2 * eta)
    checks["physical_label_derivative_sign_and_coefficient"] = label_kappa - 2 * eps**2 * Bpositive

    lam1 = sp.symbols("lambda1", real=True)
    for sigma in (-1, 1):
        polynomial = L - sigma * lam1 * x + x**2
        inverse_primitive = sp.Function(f"I_{sigma}")(x)
        escape_primitive = (
            sp.log(polynomial / L) + sigma * lam1 * inverse_primitive
        ) / 2
        checks[f"escape_primitive_derivative_{sigma}"] = sp.diff(escape_primitive, x).subs(
            sp.diff(inverse_primitive, x), 1 / polynomial
        ) - x / polynomial
        checks[f"implicit_escape_derivative_{sigma}"] = (
            x / polynomial * (polynomial / x) - 1
        )

    Bplus, Bminus = sp.symbols("Bplus Bminus", positive=True)
    Iplus, Iminus = sp.symbols("Iplus Iminus")
    Splus = (sp.log(Bplus) - sp.log(L) + lam1 * Iplus) / 2
    Sminus = (sp.log(Bminus) - sp.log(L) - lam1 * Iminus) / 2
    checks["equal_escape_log_ratio_primitive"] = (
        2 * (Sminus - Splus) - (sp.log(Bminus) - sp.log(Bplus)) + lam1 * (Iminus + Iplus)
    )

    xplus, xminus = sp.symbols("xplus xminus", positive=True)
    log_ratio_derivative = (2 * xminus + lam1) / xminus - (2 * xplus - lam1) / xplus
    checks["limiting_log_slope_derivative"] = log_ratio_derivative - lam1 * (
        1 / xminus + 1 / xplus
    )
    R, Rprime = sp.symbols("R Rprime")
    checks["limiting_profile_derivative"] = (
        R * log_ratio_derivative - R * lam1 * (1 / xminus + 1 / xplus)
    )
    checks["physical_scaled_curvature_chain_rule"] = (
        eps**2 * Rprime / (2 * eps**2 * Bplus) - Rprime / (2 * Bplus)
    )

    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def rational_examples() -> dict[str, object]:
    # These bound limiting profiles and declared error margins. They do not
    # supply the analytic theorem's finite-epsilon convergence cutoff.
    Lmin, Lmax, lam_abs = Q(1), Q(2), Q(1)
    small_radius = Q(1, 100)
    small_Bmin = Lmin - lam_abs * small_radius
    small_Bmax = Lmax + lam_abs * small_radius + small_radius**2
    require(small_Bmin > 0, "near-zero profile denominator")
    kappa_upper = Q(1, 100000)
    escape_integral_lower = small_radius**2 / (2 * small_Bmax)
    require(kappa_upper < escape_integral_lower, "near-zero escape lies inside the radius")
    # Equal escape integrals imply log R=lambda1*(Iminus+Iplus).
    # Each I is at most radius/Bmin, and exp(-a)>=1-a.
    log_ratio_abs_bound = 2 * lam_abs * small_radius / small_Bmin
    limiting_slope_lower = 1 - log_ratio_abs_bound
    regular_slope_upper = Q(13, 14)
    require(limiting_slope_lower > regular_slope_upper, "near-zero limiting slope separation")
    slope_error_budget = (limiting_slope_lower - regular_slope_upper) / 2
    require(
        limiting_slope_lower - slope_error_budget > regular_slope_upper,
        "declared finite-parameter slope tolerance",
    )

    radius_min, radius_max = Q(1, 10), Q(1, 5)
    Bmin = Lmin - lam_abs * radius_max
    Bmax = Lmax + lam_abs * radius_max + radius_max**2
    lambda_negative_margin = Q(1, 2)
    require(Bmin > 0, "negative-curvature profile denominator")
    kappa_min, kappa_max = Q(1, 140), Q(1, 120)
    first_integral_upper = radius_min**2 / (2 * Bmin)
    last_integral_lower = radius_max**2 / (2 * Bmax)
    require(
        first_integral_upper < kappa_min < kappa_max < last_integral_lower,
        "both escape radii stay in the declared compact interval",
    )
    ratio_lower = Bmin / Bmax
    negative_log_derivative_margin = 2 * lambda_negative_margin / radius_max
    scaled_curvature_margin = ratio_lower * negative_log_derivative_margin / (2 * Bmax)
    require(scaled_curvature_margin > 0, "strict negative limiting scaled curvature")
    curvature_error_budget = scaled_curvature_margin / 2
    declared_regular_curvature_bound = Q(2)
    candidate_epsilon = Q(1, 10)
    displacement_scaled_margin = (
        scaled_curvature_margin
        - curvature_error_budget
        - declared_regular_curvature_bound * candidate_epsilon**2
    )
    require(displacement_scaled_margin > 0, "declared displacement curvature margin")
    return {
        "coefficient_box": {
            "L_min": str(Lmin),
            "L_max": str(Lmax),
            "lambda1_absolute_upper": str(lam_abs),
        },
        "near_zero_limiting_profile": {
            "radial_upper": str(small_radius),
            "B_lower": str(small_Bmin),
            "B_upper": str(small_Bmax),
            "kappa_upper": str(kappa_upper),
            "escape_integral_lower": str(escape_integral_lower),
            "log_ratio_absolute_bound": str(log_ratio_abs_bound),
            "limiting_slope_lower": str(limiting_slope_lower),
            "regular_slope_upper": str(regular_slope_upper),
            "required_actual_slope_error_upper": str(slope_error_budget),
        },
        "negative_curvature_limiting_profile": {
            "lambda1_upper": str(-lambda_negative_margin),
            "radial_lower": str(radius_min),
            "radial_upper": str(radius_max),
            "B_lower": str(Bmin),
            "B_upper": str(Bmax),
            "kappa_lower": str(kappa_min),
            "kappa_upper": str(kappa_max),
            "escape_integral_at_lower_radius_upper": str(first_integral_upper),
            "escape_integral_at_upper_radius_lower": str(last_integral_lower),
            "ratio_lower": str(ratio_lower),
            "negative_log_derivative_margin": str(negative_log_derivative_margin),
            "negative_scaled_curvature_margin": str(scaled_curvature_margin),
            "required_actual_scaled_curvature_error_upper": str(curvature_error_budget),
            "declared_regular_curvature_absolute_bound": str(declared_regular_curvature_bound),
            "candidate_epsilon": str(candidate_epsilon),
            "conditional_scaled_displacement_margin": str(displacement_scaled_margin),
        },
        "scope": (
            "rational limiting-profile and conditional error gates; actual convergence errors "
            "and a physical small-parameter cutoff are not certified"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    symbolic_checks = identities()
    report = {
        "symbolic_checks": symbolic_checks,
        "symbolic_check_count": len(symbolic_checks),
        "rational_examples": rational_examples(),
        "honesty": {
            "finite_algebra_replayed": True,
            "analytic_formally_verified": False,
            "analytic_convergence_certified_by_benchmark": False,
            "physical_small_parameter_cutoff_certified": False,
            "all_singular_parameter_layers_covered": False,
            "full_graphic_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
