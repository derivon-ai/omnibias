#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Reuse verified Taylor models and quadrature for a limiting Hilbert tail."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp  # type: ignore[import-untyped]
import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.quadrature import trapezoid_integral
from omnibias.core.verified.taylor_model import TaylorModel

L_MIN, L_MAX = Q(1), Q(2)
LAMBDA_MIN, LAMBDA_MAX = Q(-1), Q(-1, 2)
S_MAX = Q(1, 8)


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def rational_interval(lower: Q, upper: Q) -> Interval:
    require(lower <= upper, "ordered rational interval")
    return Interval(Interval.from_rational(lower).lo, Interval.from_rational(upper).hi)


def contains_rational(enclosure: Interval, value: Q) -> bool:
    return Q.from_float(enclosure.lo) <= value <= Q.from_float(enclosure.hi)


def interval_payload(enclosure: Interval) -> dict[str, object]:
    return {
        "lo": enclosure.lo,
        "hi": enclosure.hi,
        "lo_exact": str(Q.from_float(enclosure.lo)),
        "hi_exact": str(Q.from_float(enclosure.hi)),
    }


def identities() -> list[str]:
    s, x, z = sp.symbols("s x z", positive=True)
    L, lam = sp.symbols("L lambda", real=True)
    d, dm, dp = sp.symbols("d dminus dplus", positive=True)
    checks = {}
    for sigma in (-1, 1):
        denominator = 1 - sigma * lam * s + L * s**2
        numerator = -sigma * lam + L * s
        function = numerator / denominator
        first = L / denominator - numerator * sp.diff(denominator, s) / denominator**2
        second = (
            -2 * L * (sp.diff(denominator, s) + numerator) / denominator**2
            + 2 * numerator * sp.diff(denominator, s) ** 2 / denominator**3
        )
        checks[f"explicit_first_derivative_{sigma}"] = sp.diff(function, s) - first
        checks[f"explicit_second_derivative_{sigma}"] = sp.diff(function, s, 2) - second
        checks[f"node_monotonicity_L_{sigma}"] = sp.diff(function, L) - s / denominator**2
        checks[f"node_monotonicity_lambda_{sigma}"] = (
            sp.diff(function, lam) + sigma / denominator**2
        )

        B = x**2 - sigma * lam * x + L
        checks[f"primitive_derivative_{sigma}"] = (
            (2 * x - sigma * lam) / (2 * B) + sigma * lam / (2 * B) - x / B
        )
        checks[f"reciprocal_derivative_{sigma}"] = -1 / (s * denominator) + 1 / s - function
        inverse = d / z + sigma * lam - L * z / (2 * d) + sigma * lam * L * z**2 / (3 * d**2)
        checks[f"inverse_flow_series_{sigma}"] = sp.series(
            -z * sp.diff(inverse, z) - inverse + sigma * lam - L / inverse,
            z, 0, 3,
        ).removeO()
        expected_B = d**2 / z**2 + sigma * lam * d / z + sigma * lam * L * z / (6 * d)
        checks[f"quadratic_series_{sigma}"] = sp.series(
            sp.expand(inverse**2 - sigma * lam * inverse + L) - expected_B,
            z, 0, 2,
        ).removeO()

    normalized_minus = dm**2 - lam * dm * z - lam * L * z**3 / (6 * dm)
    normalized_plus = dp**2 + lam * dp * z + lam * L * z**3 / (6 * dp)
    ratio = sp.series(normalized_minus / normalized_plus * dp**2 / dm**2, z, 0, 3).removeO()
    expected_ratio = (
        1 - lam * (1 / dm + 1 / dp) * z
        + lam**2 * (1 / dp**2 + 1 / (dp * dm)) * z**2
    )
    checks["ratio_series"] = ratio - expected_ratio
    xm, xp = sp.symbols("xminus xplus", positive=True)
    checks["log_ratio_derivative"] = (
        (2 * xm + lam) / xm - (2 * xp - lam) / xp - lam * (1 / xm + 1 / xp)
    )
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def exact_values(s: Q, L: Q, lam: Q, sigma: int) -> tuple[Q, Q, Q]:
    denominator = 1 - sigma * lam * s + L * s * s
    require(denominator > 0, "positive exact denominator")
    numerator = -sigma * lam + L * s
    derivative = -sigma * lam + 2 * L * s
    return (
        numerator / denominator,
        L / denominator - numerator * derivative / denominator**2,
        -2 * L * (derivative + numerator) / denominator**2
        + 2 * numerator * derivative**2 / denominator**3,
    )


def model_enclosures(sigma: int, order: int) -> tuple[Interval, tuple[Interval, Interval, Interval]]:
    center = float(S_MAX / 2)
    radius = center
    s = TaylorModel.identity(center, radius, order)
    L = rational_interval(L_MIN, L_MAX)
    lam = rational_interval(LAMBDA_MIN, LAMBDA_MAX)

    def constant(value: Interval | int) -> TaylorModel:
        return TaylorModel.constant(value, center, radius, order)

    denominator = constant(1) + s * (-sigma * lam) + s * s * L
    numerator = constant(-sigma * lam) + s * L
    derivative = constant(-sigma * lam) + s * (2 * L)
    denominator_bound = denominator.bound()
    require(denominator_bound.lo > 0, "Taylor denominator excludes zero")
    reciprocal = denominator.reciprocal()
    inverse_square = reciprocal * reciprocal
    function = numerator * reciprocal
    first = reciprocal * L - numerator * derivative * inverse_square
    # Construct each derivative as a separate exact rational expression.
    # Differentiation of the flat value remainder would not be justified.
    second = (
        (derivative + numerator) * (-2 * L) * inverse_square
        + numerator * derivative * derivative * inverse_square * reciprocal * 2
    )
    return denominator_bound, (function.bound(), first.bound(), second.bound())


def validate_values(
    sigma: int, enclosures: tuple[Interval, Interval, Interval], *, random_samples: int
) -> dict[str, object]:
    count = 0
    observed_min: list[Q | None] = [None, None, None]
    observed_max: list[Q | None] = [None, None, None]

    def check_point(s: Q, L: Q, lam: Q) -> None:
        nonlocal count
        for degree, value in enumerate(exact_values(s, L, lam, sigma)):
            require(contains_rational(enclosures[degree], value), f"rational containment derivative {degree}")
            old_min, old_max = observed_min[degree], observed_max[degree]
            observed_min[degree] = value if old_min is None else min(old_min, value)
            observed_max[degree] = value if old_max is None else max(old_max, value)
        count += 1

    for i in range(9):
        L = L_MIN + (L_MAX - L_MIN) * Q(i, 8)
        for j in range(9):
            lam = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * Q(j, 8)
            for step in range(65):
                check_point(S_MAX * Q(step, 64), L, lam)
    deterministic_count = count
    rng = random.Random(161600 + sigma)
    for _ in range(random_samples):
        check_point(
            S_MAX * Q(rng.randrange(1025), 1024),
            L_MIN + (L_MAX - L_MIN) * Q(rng.randrange(1025), 1024),
            LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * Q(rng.randrange(1025), 1024),
        )
    return {
        "deterministic_points": deterministic_count,
        "seeded_rational_random_points": random_samples,
        "derivative_comparisons": 3 * count,
        "sampled_exact_minima": [str(value) for value in observed_min],
        "sampled_exact_maxima": [str(value) for value in observed_max],
        "scope": "independent containment validation; the uniform bound comes from Taylor-model arithmetic",
    }


def node_enclosure(s: Q, sigma: int) -> Interval:
    # The checked identities give f_L=s/Q^2>=0 and f_lambda=-sigma/Q^2.
    # Therefore these exact rational corner values enclose the entire box.
    values = [
        exact_values(s, L, lam, sigma)[0]
        for L in (L_MIN, L_MAX)
        for lam in (LAMBDA_MIN, LAMBDA_MAX)
    ]
    return rational_interval(min(values), max(values))


def secondary_integral_checks(sigma: int, enclosure: Interval) -> dict[str, object]:
    parameters = [
        (L_MIN + (L_MAX - L_MIN) * Q(i, 2), LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * Q(j, 2))
        for i in range(3) for j in range(3)
    ]
    rng = random.Random(161650 + sigma)
    parameters.extend(
        (
            L_MIN + (L_MAX - L_MIN) * Q(rng.randrange(1025), 1024),
            LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * Q(rng.randrange(1025), 1024),
        )
        for _ in range(8)
    )
    rows = []
    with mp.workdps(80):
        for L, lam in parameters:
            ell = mp.mpf(L.numerator) / L.denominator
            parameter = mp.mpf(lam.numerator) / lam.denominator
            value = mp.quad(
                lambda s, p=parameter, c=ell: (-sigma * p + c * s)
                / (1 - sigma * p * s + c * s * s),
                [0, mp.mpf(S_MAX.numerator) / S_MAX.denominator],
            )
            require(mp.mpf(enclosure.lo) <= value <= mp.mpf(enclosure.hi), "secondary integral containment")
            rows.append({"L": str(L), "lambda": str(lam), "integral": str(mp.nstr(value, 35))})
    return {"precision_decimal_digits": 80, "points": rows, "scope": "secondary numerical validation only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--random-samples", type=int, default=512)
    args = parser.parse_args()
    require(args.order >= 2, "Taylor order at least two")
    require(args.panels >= 1, "positive quadrature panel count")
    require(args.random_samples >= 1, "positive random validation count")
    checks = identities()
    rational_denominator_lower = 1 - max(abs(LAMBDA_MIN), abs(LAMBDA_MAX)) * S_MAX
    require(rational_denominator_lower > 0, "uniform rational denominator gate")
    complex_radius = Q(1, 32)
    complex_denominator_error = complex_radius + L_MAX * complex_radius**2
    complex_derivative_upper = (1 + L_MAX * complex_radius) / (1 - complex_denominator_error)
    complex_contraction_upper = complex_radius * complex_derivative_upper
    require(complex_denominator_error == Q(17, 512), "complex denominator perturbation")
    require(complex_derivative_upper == Q(544, 495), "complex primitive derivative upper bound")
    require(complex_contraction_upper < Q(1, 16), "common complex inversion disk margin")
    rows = []
    for sigma in (-1, 1):
        denominator, bounds = model_enclosures(sigma, args.order)
        nodes = [node_enclosure(S_MAX * Q(j, args.panels), sigma) for j in range(args.panels + 1)]
        integral = trapezoid_integral(nodes, 0.0, float(S_MAX), bounds[2])
        rows.append({
            "sigma": sigma,
            "denominator": interval_payload(denominator),
            "function_and_derivative_bounds": [interval_payload(bound) for bound in bounds],
            "integral": interval_payload(integral),
            "validation": validate_values(sigma, bounds, random_samples=args.random_samples),
            "secondary_integral_validation": secondary_integral_checks(sigma, integral),
        })
    report = {
        "symbolic_checks": checks,
        "symbolic_check_count": len(checks),
        "parameter_box": {
            "L": [str(L_MIN), str(L_MAX)],
            "lambda": [str(LAMBDA_MIN), str(LAMBDA_MAX)],
            "s": ["0", str(S_MAX)],
            "rational_denominator_lower": str(rational_denominator_lower),
        },
        "taylor_order": args.order,
        "quadrature_panels": args.panels,
        "complex_inversion_domain_gates": {
            "radius": str(complex_radius),
            "denominator_error_upper": str(complex_denominator_error),
            "primitive_derivative_upper": str(complex_derivative_upper),
            "radius_times_derivative_upper": str(complex_contraction_upper),
            "scope": "finite sufficient inequalities for the separate written limiting-family inversion argument",
        },
        "signs": rows,
        "honesty": {
            "existing_verified_components_reused": True,
            "limiting_compactified_tail_enclosed": True,
            "value_remainder_differentiated": False,
            "analytic_formally_verified": False,
            "actual_expanding_domain_passage_certified": False,
            "actual_large_kappa_matching_proved_by_benchmark": False,
            "physical_small_parameter_cutoff_certified": False,
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
