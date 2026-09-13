#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact algebra for the rapid zero-fiber passage, not an analytic verifier."""

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
    h, a, b, n, c = sp.symbols("h a b n c")
    R = h * (n + c * h) / (b * h - a)
    checks = {
        "height_variation_first": sp.diff(R, h)
        - (b * c * h**2 - a * n - 2 * a * c * h) / (b * h - a) ** 2,
        "height_variation_second": sp.diff(R, h, 2) - 2 * a * (b * n + a * c) / (b * h - a) ** 3,
    }
    v, nu, A, C = sp.symbols("v nu A C")
    s = v - 1
    a1 = -(s**2) * (v + 2)
    actual = R.subs(
        {a: nu * a1, b: 1 - A * nu * v + C * nu**2 * v**2, n: s + nu * v**2, c: C * nu**2 * v}
    )
    R1 = (A - 1) * v**2 - (A + 2) * v + 4
    checks["first_order_zero_fiber_slope"] = sp.diff(actual, nu).subs(nu, 0).subs(h, s**2 / 2) - R1
    U = (A - 1) * v**3 / 3 - (A + 2) * v**2 / 2 + 4 * v
    checks["rapid_primitive"] = sp.diff(U, v) - R1
    vi = sp.symbols("vi")
    checks["grazing_threshold"] = 2 * (U.subs({A: 1, v: 1}) - U.subs({A: 1, v: vi})) - (
        3 * vi - 5
    ) * (vi - 1)
    v0, F0, F1, ptilde, mtilde = sp.symbols("v0 F0 F1 ptilde mtilde")
    slow = nu * (mtilde + ptilde * v - v**3)
    checks["canonical_slow_factorization"] = slow.subs(
        {
            ptilde: 3 * v0**2 + F1,
            mtilde: F0 - (3 * v0**2 + F1) * v0 + v0**3,
        }
    ) - nu * (-((v - v0) ** 2) * (v + 2 * v0) + F1 * (v - v0) + F0)
    rho, z, t = sp.symbols("rho z t", positive=True)
    for sigma in (-1, 1):
        D = rho**2 + 2 * sigma * nu * rho + C * nu**2
        physical_z = (2 / h - rho**2 + C * nu**2) / D
        checks[f"exact_regular_coordinate_{sigma}"] = (
            physical_z.subs(h, 2 / (rho**2 * (1 + z))) - (rho**2 * z + C * nu**2) / D
        )
        section_t = (v - 1) ** 2 - 2 * v / (sigma * rho)
        checks[f"section_label_derivative_{sigma}"] = sp.diff(section_t, v) - 2 * (
            v - 1 - 1 / (sigma * rho)
        )
        Rbase = v - 1
        L = sp.diff(section_t, v)
        checks[f"moving_input_variation_{sigma}"] = (1 / (sigma * rho) - Rbase) / L + sp.Rational(
            1, 2
        )
    V = sp.symbols("V", positive=True)
    rho_i = 2 * (V - 1) / V**2
    z_i = V**2 / (V - 1) ** 2 - 1
    Y0 = 2 / V**2
    vdot_parameter = -(rho_i**2) * z_i / Y0**2
    hdot_parameter = rho_i * z_i / Y0**2
    eta_i = V * (2 * V - 1) * (V - 2) / (V - 1)
    checks["old_zero_label_shift"] = 2 * (-V) * vdot_parameter - 2 * hdot_parameter - eta_i
    checks["old_zero_inside_rapid_band"] = (
        (3 * (1 - V) - 5) * ((1 - V) - 1) - eta_i - (V**2 + 5 * V + V / (V - 1))
    )
    # The first-order correction is integrated in v; its denominator loss
    # is only logarithmic under the proved h >= k*((v-1)^2+nu) bootstrap.
    checks["logarithmic_majorant_primitive"] = sp.diff(sp.log(t**2 + nu) / 2, t) - t / (t**2 + nu)
    L = 2 * (v - 1 - 1 / rho)
    event_first = -sp.Rational(1, 2) / (1 / rho - (v - 1))
    event_second = event_first**2 / (1 / rho - (v - 1))
    checks["base_event_second_derivative_cancels"] = 2 * event_first**2 + L * event_second
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def curvature_rescalings() -> list[dict[str, str]]:
    rows = [
        (1, 4, 3, Q(1, 2)),
        (2, 2, 3, Q(1, 2)),
        (3, 1, 3, Q(1)),
        (4, 0, 3, Q(3, 2)),
        (2, 4, 4, Q(1, 2)),
        (3, 3, 4, Q(1)),
        (1, 3, 3, Q(0)),
    ]
    report = []
    for p, q, r, expected in rows:
        exponent = Q(p) + Q(q, 2) + Q(1, 2) - r
        require(exponent == expected, "curvature integral rescaling")
        require(q - 2 * r < -1, "rescaled absolute integrand is integrable at infinity")
        report.append(
            {"integrand": f"nu^{p}*abs(s)^{q}/(s^2+nu)^{r}", "integral_power": str(exponent)}
        )
    return report


def rational_examples() -> dict[str, object]:
    V = Q(20)
    rho_section = 2 * (V - 1) / V**2
    tau_c = 3 * V**2 + 2 * V
    eta_i = V * (2 * V - 1) * (V - 2) / (V - 1)
    margin = V**2 + 5 * V + V / (V - 1)
    require(tau_c - eta_i == margin > 0, "strict zero-fiber rapid margin")
    theta, rho, zeta = Q(6, 7), Q(1, 1000), Q(1, 10000)
    nu = rho / 100000
    cmax = Q(3)
    # Bounds the transformed inverse coordinate for |zbar|<=2*zeta.
    factor = (1 + 2 * zeta) / (1 - 2 * zeta) ** 3
    factor *= 4 / ((1 - 2 * rho) * (2 - 3 * rho) ** 2)
    factor *= (rho**2 + 2 * nu * rho + cmax * nu**2) / (rho**2 - 2 * nu * rho)
    regular_bound = theta * factor
    theta_h = (1 + theta) / 2
    require(regular_bound < theta_h < 1, "regular derivative margin")
    singular_input_bound = (1 + theta_h) / 2
    require(singular_input_bound > theta_h, "eventual rapid derivative separates")
    return {
        "coordinate_example": {
            "rho": str(rho_section),
            "tau_c": str(tau_c),
            "old_zero_shift": str(eta_i),
            "strict_margin": str(margin),
        },
        "derivative_example": {
            "regular_upper": str(regular_bound),
            "theta_h": str(theta_h),
            "declared_rapid_lower": str(singular_input_bound),
        },
        "scope": "rational algebra examples; actual asymptotic cutoffs are not computed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "symbolic_checks": identities(),
        "curvature_rescalings": curvature_rescalings(),
        "rational_examples": rational_examples(),
        "honesty": {
            "finite_algebra_replayed": True,
            "analytic_theorem_formally_verified": False,
            "physical_small_parameter_cutoff_certified": False,
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
