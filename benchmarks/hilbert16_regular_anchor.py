#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay exact identities for the sharp regular-passage multiplier anchor.

This checks finite symbolic algebra. It does not verify the analytic BVP,
uniform remainder estimates, physical cutoffs, or any full cyclicity claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def require_zero(residual: object, label: str) -> None:
    # Only an exactly reduced zero passes. An unresolved symbolic expression
    # is a failed check, even if some sample substitutions would vanish.
    require(sp.cancel(residual) == 0, label)


def identities() -> dict[str, object]:
    s, k, m, C, x, y, nu = sp.symbols("s k m C x y nu")
    denominator = C * m**2 + 12 * k**2 + 16 * k + 2 * m + 4
    conic_denominator = denominator - 6 * C * m * k**2 - 2 * C * m * k
    f0 = 2 * C * (1 + 3 * k) / conic_denominator
    a = -(1 + 3 * k) * (m + 6 * k**2 + 8 * k + 2) / conic_denominator
    A = (2 + C * m - 2 * C * k * (3 * k + 1)) / (2 * (1 + 3 * k))
    p = (-m + 8 * k**2 + 2 * k) / 2
    r = m - 3 * k**2 - k
    mu1 = k * (-m + 6 * k**2 + 2 * k) / 2
    jacobian = 1 + k * f0 * m
    conic = a * (x + k * y)**2 - f0 * m * x + y + f0
    field_x = A * x - y + x**2 + (p + r) * x * y + mu1 * y**2
    field_y = C * x + x**2 + x * y + r * y**2
    cofactor = 2 * (1 + k) * x + m * y
    xs = (s + k * a * s**2 + k * f0) / jacobian
    ys = (f0 * m * s - a * s**2 - f0) / jacobian
    on_curve = {x: xs, y: ys}
    speed = sp.cancel((field_x + k * field_y).subs(on_curve, simultaneous=True))
    restricted_cofactor = sp.cancel(cofactor.subs(on_curve, simultaneous=True))
    speed_poly, cofactor_poly = sp.Poly(speed, s), sp.Poly(restricted_cofactor, s)
    cubic_coefficient = (
        -(1 + 3 * k) * (p - 3 * k**2) * (6 * k**2 + 8 * k + m + 2) / denominator
    )
    checks = {
        "affine_parameter_inverse": xs + k * ys - s,
        "parameterized_parabola": conic.subs(on_curve, simultaneous=True),
        "Darboux_invariance": sp.diff(conic, x) * field_x
        + sp.diff(conic, y) * field_y - cofactor * conic,
        "exact_cubic_speed_coefficient": speed_poly.coeff_monomial(s**3) - cubic_coefficient,
        "cofactor_quadratic_cancellation": cofactor_poly.coeff_monomial(s**2)
        - 2 * cubic_coefficient,
        "base_reference_speed": speed.subs({k: 0, m: 0}) - (s**2 + 2 * s + C) / 2,
        "base_reference_cofactor": restricted_cofactor.subs({k: 0, m: 0}) - 2 * s,
    }
    require(speed_poly.degree() == 3, "reference speed is cubic")
    require(cofactor_poly.degree() == 2, "reference cofactor is quadratic")

    # This endpoint identity also holds before imposing the conic parameters.
    independent_a = sp.symbols("a")
    x_generic = s - k * y
    conic_y = 1 + 2 * independent_a * k * s
    endpoint_numerator = (nu * x_generic + 1) * y - x_generic**2
    expected_numerator = (
        2 * y - s**2 + (nu + 2 * k * (1 + independent_a)) * s * y
        - k * (nu + k) * y**2
    )
    checks["exact_endpoint_numerator_cancellation"] = (
        endpoint_numerator + y * conic_y - expected_numerator
    )
    signed_label = (x / (nu * y) - 1)**2 - 2 / (nu**2 * y)
    checks["actual_signed_endpoint_derivative"] = (
        sp.diff(signed_label, y) - 2 * ((nu * x + 1) * y - x**2) / (nu**2 * y**3)
    )

    # Finite coefficient identities for p=3nu^2+O(nu^3), r=-nu.
    # Analytic uniformity of the actual canonical embedding is a separate premise.
    p3 = sp.symbols("p3")
    k_series = -nu + nu**2 + (2 * p3 + 10) * nu**3
    p_series = 3 * nu**2 + p3 * nu**3
    m_series = 8 * k_series**2 + 2 * k_series - 2 * p_series
    checks["canonical_parabola_inverse_through_cubic"] = sp.series(
        k_series + 5 * k_series**2 - (2 * p_series - nu), nu, 0, 4,
    ).removeO()
    checks["canonical_reference_mu1_through_cubic"] = sp.series(
        mu1.subs({k: k_series, m: m_series}, simultaneous=True) + 2 * nu**3,
        nu, 0, 4,
    ).removeO()
    checks["canonical_reference_A_through_quadratic"] = sp.series(
        A.subs({k: k_series, m: m_series}, simultaneous=True)
        - (1 + 3 * nu + (6 - 2 * C) * nu**2), nu, 0, 3,
    ).removeO()

    for label, residual in checks.items():
        require_zero(residual, label)

    # A wrong cofactor factor must fail through the same checker as the claims.
    mutated_residual = cofactor_poly.coeff_monomial(s**2) - 3 * cubic_coefficient
    try:
        require_zero(mutated_residual, "mutated cofactor factor three")
    except ArithmeticError:
        mutated_rejected = True
    else:
        raise ArithmeticError("the false cofactor mutation was accepted")

    # An unresolved expression cannot silently earn a successful identity flag.
    try:
        require_zero(sp.Function("unresolved")(s), "unresolved residual")
    except ArithmeticError:
        unresolved_rejected = True
    else:
        raise ArithmeticError("an unresolved residual was accepted")

    return {
        "symbolic_checks": list(checks),
        "symbolic_check_count": len(checks),
        "degree_checks": ["cubic_speed", "quadratic_cofactor"],
        "degree_check_count": 2,
        "negative_checks": {
            "mutated_cofactor_identity_rejected": mutated_rejected,
            "unresolved_symbolic_expression_rejected": unresolved_rejected,
        },
        "algebraic_scope": (
            "rational identities wherever their denominators are nonzero; "
            "finite series coefficients do not certify a uniform growing-domain remainder"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        **identities(),
        "honesty": {
            "finite_algebra_replayed": True,
            "actual_canonical_embedding_uniformity_certified_by_benchmark": False,
            "matched_BVP_sensitivity_estimates_certified_by_benchmark": False,
            "uniform_regular_multiplier_remainder_certified_by_benchmark": False,
            "physical_small_parameter_cutoffs_certified": False,
            "analytic_formally_verified": False,
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
