#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay finite algebra for actual singular and regular kappa derivatives.

The physical epsilon is fixed in every kappa derivative. The checks cover
the off-line normal coefficient, scale generators, radial height derivatives,
moving cuts, and regular coordinate/chain identities. They do not certify uniform analytic estimates,
physical passage domains, a cycle bound, or the full Hilbert XVI problem.
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
    # An unresolved expression fails; sampled zeros cannot replace an identity.
    require(sp.simplify(residual) == 0, label)


def normal_coefficient_checks() -> dict[str, object]:
    """Derive the spatial coefficient premise from the exact off-line field."""
    nu, epsilon, V, h, C, abar = sp.symbols("nu epsilon V h C Abar")
    m0, m1, p0, p1 = sp.symbols("m0 m1 p0 p1")
    w = 1 - V
    v = w - nu * w**2 + nu**2 * (2 * w**3 - C * w * h)
    ell = 1 + 2 * nu * v + C * nu**2 * h
    slow_constant = nu * (m0 + nu * m1)
    slow_linear = nu * (p0 + nu * p1)
    fast_field = (
        slow_constant + slow_linear * v - nu * v**3 - h
        + (1 + nu * abar) * nu * v * h - C * nu**2 * v**2 * h
    )
    normal_field = sp.series(
        ell * fast_field + C * nu**2 * v * V * h, nu, 0, 3,
    ).removeO().expand()
    g = sp.cancel((normal_field - normal_field.subs(h, 0)) / h).expand()
    g2 = (
        -2 * C * V**2 + 3 * C * V - C * h - C
        + 3 * V**2 - abar * V - 6 * V + abar + 3
    )
    checks = {
        "inverse_normal_coordinate_through_nu_squared": sp.series(
            -1 + v + nu * v**2 + C * nu**2 * v * h + V, nu, 0, 3,
        ).removeO(),
        "normal_g_through_nu_squared": g - (-1 + nu * (V - 1) + nu**2 * g2),
        "first_fast_coefficient_height_independent": sp.diff(g, nu, h).subs(nu, 0),
        "second_fast_coefficient_height_derivative": sp.diff(g, h).coeff(nu, 2) + C,
        "slow_coefficients_cancel_from_fast_two_jet": sum(
            sp.diff(g, symbol)**2 for symbol in (m0, m1, p0, p1)
        ),
    }
    # The exact canonical scale has this first jet; its higher terms cannot
    # change the height independence of the first epsilon coefficient.
    v0 = 1 - nu + 2 * nu**2
    scale_first_jet = sp.series(3 * v0 / (1 + 2 * nu * v0), nu, 0, 2).removeO()
    checks["canonical_scale_first_jet"] = scale_first_jet - (3 - 9 * nu)
    inverse_nu = epsilon / 3 + epsilon**2 / 3
    checks["inverse_epsilon_conversion"] = sp.series(
        3 * inverse_nu - 9 * inverse_nu**2 - epsilon, epsilon, 0, 3,
    ).removeO()
    k_epsilon = sp.series((-g).subs(nu, inverse_nu), epsilon, 0, 3).removeO()
    checks["height_derivative_in_canonical_parameter"] = (
        sp.diff(k_epsilon, h) - C * epsilon**2 / 9
    )
    return checks


def identities() -> dict[str, object]:
    X, Y, epsilon, kappa = sp.symbols("X Y epsilon kappa", positive=True)
    omega = sp.exp(-kappa)
    u = epsilon / omega
    base = sp.Function("B")
    b = omega**2 * base(X / omega)
    kfun = sp.Function("k")
    khat = kfun(u * X, epsilon * u**2 * Y)
    checks = {
        **normal_coefficient_checks(),
        "exact_fixed_epsilon_b_generator": sp.diff(b, kappa)
        - X * sp.diff(b, X) + 2 * b,
        "exact_fixed_epsilon_k_generator": sp.diff(khat, kappa)
        - X * sp.diff(khat, X) - 2 * Y * sp.diff(khat, Y),
        "phase_parameter_integrand_boundary": -X * sp.diff(b, kappa) / b**2
        - sp.diff(X**2 / b, X),
    }

    s, h = sp.symbols("s h", positive=True)
    q = sp.Function("q")(s)
    k = sp.Function("k")(s, h)
    denominator = q + k * h
    radial = s * h / denominator
    numerator = q - h**2 * sp.diff(k, h)
    checks["physical_radial_first_height_derivative"] = (
        sp.diff(radial, h) - s * numerator / denominator**2
    )
    second = s * (
        (-2 * h * sp.diff(k, h) - h**2 * sp.diff(k, h, 2)) * denominator
        - 2 * numerator * (k + h * sp.diff(k, h))
    ) / denominator**3
    checks["physical_radial_second_height_derivative"] = (
        sp.diff(radial, h, 2) - second
    )

    # An arbitrary polynomial second jet exercises every second Leibniz term.
    # This is finite jet algebra, not an analytic differentiation-under-integral
    # certificate for an arbitrary family of integrands.
    lower = sp.Function("a")(kappa)
    coefficients = sp.symbols("c0:6")
    monomials = (1, s, kappa, s**2, s * kappa, kappa**2)
    integrand = sum(
        coefficient * monomial
        for coefficient, monomial in zip(coefficients, monomials, strict=True)
    )
    upper = sp.symbols("upper", positive=True)
    integral = sp.Integral(integrand, (s, lower, upper))
    expected = (
        sp.Integral(sp.diff(integrand, kappa, 2), (s, lower, upper))
        - integrand.subs(s, lower) * sp.diff(lower, kappa, 2)
        - sp.diff(integrand, s).subs(s, lower) * sp.diff(lower, kappa)**2
        - 2 * sp.diff(integrand, kappa).subs(s, lower) * sp.diff(lower, kappa)
    )
    checks["moving_cut_second_leibniz_on_arbitrary_second_jet"] = (
        sp.diff(integral, kappa, 2) - expected
    ).doit()

    # Exact regular signed coordinates on their principal positive-root branch.
    rho = sp.symbols("rho", positive=True)
    t, z = sp.symbols("t z", real=True)
    root = sp.sqrt(1 - 2 * rho + rho**2 * t)
    coordinate_denominator = 1 - rho + root
    entrance = 2 / coordinate_denominator - 1
    entrance_first = -rho**2 / (root * coordinate_denominator**2)
    entrance_second = rho**4 * (
        1 / (2 * root**3 * coordinate_denominator**2)
        + 1 / (root**2 * coordinate_denominator**3)
    )
    entrance_third = -rho**6 * (
        3 / (4 * root**5 * coordinate_denominator**2)
        + 3 / (2 * root**4 * coordinate_denominator**3)
        + 3 / (2 * root**3 * coordinate_denominator**4)
    )
    checks["regular_entrance_coordinate_first_derivative"] = (
        sp.diff(entrance, t) - entrance_first
    )
    checks["regular_entrance_coordinate_second_derivative"] = (
        sp.diff(entrance, t, 2) - entrance_second
    )
    checks["regular_entrance_coordinate_third_derivative"] = (
        sp.diff(entrance, t, 3) - entrance_third
    )
    exit_coordinate = 1 + 4 / rho**2 * (
        (1 + z)**-2 - (1 + rho) / (1 + z)
    )
    checks["regular_exit_coordinate_third_derivative"] = (
        sp.diff(exit_coordinate, z, 3)
        - 4 / rho**2 * (-24 / (1 + z)**5 + 6 * (1 + rho) / (1 + z)**4)
    )

    # Arbitrary local cubic jets preserve every term in the third chain rule.
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = sp.symbols(
        "a1 a2 a3 b1 b2 b3 c1 c2 c3",
    )
    inner = a1 * t + a2 * t**2 / 2 + a3 * t**3 / 6
    middle = b1 * z + b2 * z**2 / 2 + b3 * z**3 / 6
    outer = c1 * z + c2 * z**2 / 2 + c3 * z**3 / 6
    composed = outer.subs(z, middle.subs(z, inner))
    second_chain = c2 * (b1 * a1)**2 + c1 * (b2 * a1**2 + b1 * a2)
    third_chain = (
        c3 * (b1 * a1)**3
        + 3 * c2 * (b1 * a1) * (b2 * a1**2 + b1 * a2)
        + c1 * (b3 * a1**3 + 3 * b2 * a1 * a2 + b1 * a3)
    )
    checks["regular_composite_second_section_jet"] = (
        sp.diff(composed, t, 2).subs(t, 0) - second_chain
    )
    checks["regular_composite_third_section_jet"] = (
        sp.diff(composed, t, 3).subs(t, 0) - third_chain
    )

    # Alpha and the physical field are fixed: only the entrance label varies.
    H1 = sp.symbols("H1", positive=True)
    H2, H3, ti1, ti2 = sp.symbols("H2 H3 ti1 ti2", real=True)
    local_parameter = sp.symbols("local_parameter", real=True)
    regular = H1 * t + H2 * t**2 / 2 + H3 * t**3 / 6
    entrance_label = ti1 * local_parameter + ti2 * local_parameter**2 / 2
    log_multiplier = sp.log(sp.diff(regular, t).subs(t, entrance_label))
    logarithmic_second = (H3 / H1 - (H2 / H1)**2) * ti1**2 + H2 / H1 * ti2
    checks["fixed_field_log_multiplier_first_kappa_jet"] = (
        sp.diff(log_multiplier, local_parameter).subs(local_parameter, 0)
        - H2 / H1 * ti1
    )
    checks["fixed_field_log_multiplier_second_kappa_jet"] = (
        sp.diff(log_multiplier, local_parameter, 2).subs(local_parameter, 0)
        - logarithmic_second
    )

    for label, residual in checks.items():
        require_zero(residual, label)

    # Omitting the entrance label's second derivative must fail the same gate.
    mutated_second = (H3 / H1 - (H2 / H1)**2) * ti1**2
    try:
        require_zero(logarithmic_second - mutated_second, "missing entrance second jet")
    except ArithmeticError:
        mutation_rejected = True
    else:
        raise ArithmeticError("the omitted second entrance jet was accepted")

    try:
        require_zero(sp.Function("unresolved")(s), "unresolved symbolic residual")
    except ArithmeticError:
        unresolved_rejected = True
    else:
        raise ArithmeticError("an unresolved symbolic residual was accepted")

    return {
        "symbolic_checks": list(checks),
        "symbolic_check_count": len(checks),
        "negative_checks": {
            "missing_entrance_second_jet_rejected": mutation_rejected,
            "unresolved_symbolic_expression_rejected": unresolved_rejected,
        },
        "algebraic_scope": (
            "exact identities on smooth branches with nonzero denominators; "
            "Leibniz and composition replays use arbitrary finite local jets"
        ),
        "derivative_convention": (
            "physical epsilon and field fixed; omega=exp(-kappa), "
            "u=epsilon/omega; the regular matching parameter alpha is fixed"
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
            "uniform_singular_C2_remainder_certified_by_benchmark": False,
            "actual_label_derivative_bounds_certified_by_benchmark": False,
            "uniform_regular_section_jet_bounds_certified_by_benchmark": False,
            "physical_small_parameter_cutoffs_certified": False,
            "connected_passage_domain_certified_by_benchmark": False,
            "analytic_formally_verified": False,
            "actual_displacement_zero_bound_certified_by_benchmark": False,
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
