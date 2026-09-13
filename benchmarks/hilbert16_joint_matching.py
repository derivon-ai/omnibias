#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite identities and interval multiplier gaps for joint Hilbert matching.

The interval certificates concern limiting multipliers on coefficient boxes.
The two-scale checks concern exact coefficients, anchor/coordinate identities,
and a leading quadratic model. They do not certify actual passage errors,
cutoffs, or an analytic cycle count.
"""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp  # type: ignore[import-untyped]
import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, exp_iv


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
    r, u, v, Y, x, epsilon = sp.symbols("r u v Y x epsilon", positive=True)
    L, lam, Z, k, sigma = sp.symbols("L lambda Z k sigma", real=True)
    B0 = L - sigma * lam * x + x**2
    Bepsilon = B0 - sigma * epsilon * x**3 * Z
    scaled_B = L * r**2 - sigma * lam * r * v + v**2 - sigma * u * v**3 * Z
    substituted_B = Bepsilon.subs({x: v / r, epsilon: r * u}, simultaneous=True)
    radial_y = x * sp.Symbol("y") / (epsilon * (Bepsilon + k * sp.Symbol("y")))
    # Y=r^2*y and v=r*x imply dY/dv=r*dy/dx, with r frozen.
    scaled_slope = r * radial_y.subs(
        {x: v / r, sp.Symbol("y"): Y / r**2, epsilon: r * u}, simultaneous=True,
    )
    expected_slope = v * Y / (r * u * (scaled_B + k * Y))
    checks = {
        "exact_radial_denominator_scaling": r**2 * substituted_B - scaled_B,
        "frozen_radial_height_chain_rule": scaled_slope - expected_slope,
        "phase_error_integrand": x / Bepsilon - x / B0
        - sigma * epsilon * x**4 * Z / (B0 * Bepsilon),
        "scaled_phase_integrand": (v / r**2) / substituted_B - v / scaled_B,
        "exact_logarithmic_height_loss": r * u * expected_slope / Y
        - v / scaled_B + r * u * k * expected_slope / scaled_B,
        "exact_squared_radial_anchor_derivative": v - r * u * expected_slope
        - v * (scaled_B + (k - 1) * Y) / (scaled_B + k * Y),
        "physical_squared_label_scaling": u**2 * v**2 - 2 * epsilon * u**2 * Y
        - 2 * u**2 * (v**2 / 2 - epsilon * Y),
    }

    kappa = sp.symbols("kappa", real=True)
    dm, dp, positive_L = sp.symbols("dminus dplus positive_L", positive=True)
    exponent_difference = (
        2 * kappa + sp.log(dm**2 / positive_L)
        - (2 * kappa + sp.log(dp**2 / positive_L))
    )
    checks["common_large_exponent_cancellation"] = sp.expand_log(
        exponent_difference - 2 * sp.log(dm / dp), force=True,
    )

    a, R = sp.symbols("a R", positive=True)
    C = a**2 + 1
    q = (x**2 + 2 * x + C) / 2
    log_D = 4 * (sp.atan((x + 1) / a) - sp.pi / 2) / a
    checks["regular_variational_log_primitive"] = (
        2 * sp.diff(q, x) / q - sp.diff(log_D, x) - 2 * x / q
    )
    qminus, qplus = q.subs(x, -R), q.subs(x, R)
    log_g0 = sp.log(qplus / qminus) + log_D.subs(x, -R) - log_D.subs(x, R)
    checks["normalized_regular_log_multiplier_derivative"] = (
        sp.diff(log_g0, R) + (3 * R**2 + C) / (qplus * qminus)
    )
    checks["regular_multiplier_limit"] = sp.limit(log_g0, R, sp.oo) + 4 * sp.pi / a

    rho, nu = sp.symbols("rho nu", positive=True)
    t, zbar, gp = sp.symbols("t zbar Gprime", real=True)
    root_minus = sp.sqrt(1 - 2 * rho + rho**2 * t)
    Zminus = 2 / (1 - rho + root_minus) - 1
    Kplus = 1 + 4 / rho**2 * ((1 + zbar)**-2 - (1 + rho) / (1 + zbar))
    delta_minus = rho**2 - 2 * nu * rho + C * nu**2
    delta_plus = rho**2 + 2 * nu * rho + C * nu**2
    affine_minus, affine_plus = rho**2 / delta_minus, rho**2 / delta_plus
    actual_chain = sp.diff(Kplus, zbar) * gp * affine_minus * sp.diff(Zminus, t) / affine_plus
    inverse_factor = (2 - (1 + rho) * (1 + zbar)) / (1 + zbar)**3
    entrance_factor = 4 / (root_minus * (1 - rho + root_minus)**2)
    checks["actual_affine_section_coordinate_derivative"] = (
        actual_chain - delta_plus / delta_minus * inverse_factor * gp * entrance_factor
    )

    b = sp.symbols("b", positive=True)
    # Here a=sqrt(C-1), b=sqrt(Delta), L=(b^2+lambda^2)/4.
    resonance_numerator = 4 * (b**2 + lam**2) - (a**2 + 4) * lam**2
    direct_gap = 2 * sp.pi * (lam / b + 2 / a)
    rationalized_gap = 2 * sp.pi * resonance_numerator / (a * b * (2 * b - lam * a))
    checks["rationalized_resonance_gap"] = direct_gap - rationalized_gap
    checks["joint_logarithmic_error_decomposition"] = sp.expand_log(
        r * u * sp.log(1 / (r * u))
        - r * (u * sp.log(1 / u)) - u * (r * sp.log(1 / r)), force=True,
    )

    # A common auxiliary scale extracts the full linear part in (r,u),
    # without inferring derivatives of the actual analytic error.
    scale, d, beta = sp.symbols("scale d beta", positive=True)
    evaluated_logs = {}
    for sign in (-1, 1):
        X = d + sign * lam * scale * r - sign * beta * d**2 * scale * u
        phase = sp.log(X / d) - sign * lam * scale * r / X + sign * beta * scale * u * X
        evaluated_B = (
            X**2 - sign * lam * scale * r * X + L * scale**2 * r**2
            - sign * beta * scale * u * X**3
        )
        linear_log = sign * lam * r / d - 3 * sign * beta * d * u
        checks[f"two_scale_escape_linear_part_{sign}"] = sp.diff(phase, scale).subs(scale, 0)
        checks[f"two_scale_evaluated_log_linear_part_{sign}"] = (
            sp.diff(sp.log(evaluated_B), scale).subs(scale, 0) - linear_log
        )
        evaluated_logs[sign] = sp.log(evaluated_B).subs(d, dm if sign == -1 else dp)
    checks["two_scale_outgoing_minus_incoming_linear_part"] = (
        sp.diff(evaluated_logs[-1] - evaluated_logs[1], scale).subs(scale, 0)
        + lam * (1 / dm + 1 / dp) * r - 3 * beta * (dm + dp) * u
    )

    baseline, perturbation = sp.symbols("baseline perturbation", positive=True)
    checks["exact_reciprocal_remainder"] = (
        1 / (baseline + perturbation) - 1 / baseline + perturbation / baseline**2
        - perturbation**2 / (baseline**2 * (baseline + perturbation))
    )
    for sign in (-1, 1):
        denominator = x**2 - sign * lam * x + L
        checks[f"quadratic_phase_tail_remainder_{sign}"] = (
            x / denominator - 1 / x - sign * lam / x**2
            - ((lam**2 - L) / x**3 - sign * lam * L / x**4)
            / (1 - sign * lam / x + L / x**2)
        )

    mobius_a = dm / dp
    mobius_b = beta * (mobius_a + 1)
    squared_label = sp.symbols("squared_label", positive=True)
    squared_mobius = mobius_a**2 * squared_label / (1 - mobius_b * sp.sqrt(squared_label))**2
    checks["two_scale_mobius_coefficient_consistency"] = 3 * mobius_b * dp - 3 * beta * (dm + dp)
    checks["squared_label_mobius_derivative"] = (
        sp.diff(squared_mobius, squared_label)
        - mobius_a**2 / (1 - mobius_b * sp.sqrt(squared_label))**3
    )
    checks["positive_base_squared_label_second_derivative"] = (
        sp.diff(squared_mobius, squared_label, 2)
        - 3 * mobius_a**2 * mobius_b
        / (2 * sp.sqrt(squared_label) * (1 - mobius_b * sp.sqrt(squared_label))**4)
    )

    # These are finite anchor and coordinate identities. The actual error
    # bounds and physical branch/pole margins remain analytic premises.
    for sign in (-1, 1):
        anchor_cut = d / (1 + sign * beta * d * u)
        anchor_label = (u * anchor_cut)**2
        checks[f"retained_cubic_phase_derivative_{sign}"] = (
            sp.diff(sp.log(v) - sp.log(1 - sign * beta * u * v), v)
            - 1 / (v * (1 - sign * beta * u * v))
        )
        checks[f"rational_anchor_phase_root_{sign}"] = (
            anchor_cut - d * (1 - sign * beta * u * anchor_cut)
        )
        checks[f"rational_anchor_cubic_expansion_{sign}"] = (
            sp.series(anchor_label, u, 0, 4).removeO()
            - d**2 * u**2 + 2 * sign * beta * d**3 * u**3
        )
    incoming_anchor = u * dp / (1 + beta * dp * u)
    outgoing_anchor = u * dm / (1 - beta * dm * u)
    checks["rational_anchors_share_mobius_graph"] = (
        outgoing_anchor - mobius_a * incoming_anchor / (1 - mobius_b * incoming_anchor)
    )
    checks["rational_anchor_pole_factor"] = (
        1 - mobius_b * incoming_anchor - (1 - beta * dm * u) / (1 + beta * dp * u)
    )

    radius, height, physical_q = sp.symbols("radius height physical_q", positive=True)
    checks["physical_squared_label_tail_derivative"] = (
        2 * radius - 2 * radius * height / (physical_q + k * height)
        - 2 * radius * (physical_q + (k - 1) * height) / (physical_q + k * height)
    )
    physical_v = sp.symbols("physical_v", real=True)
    coordinate_shift = nu * physical_v**2 + C * nu**2 * physical_v * height
    physical_V = 1 - physical_v - coordinate_shift
    checks["exact_physical_signed_label_shift"] = (
        (physical_v - 1)**2 - 2 * height - (physical_V**2 - 2 * height)
        - 2 * physical_V * coordinate_shift - coordinate_shift**2
    )
    M1, M2 = sp.symbols("M1 M2", nonnegative=True)
    regular_second_bound = (
        320 / rho**2 * (M1 * 4 * rho**2)**2
        + 40 / rho**2 * (M2 * (4 * rho**2)**2 + M1 * 16 * rho**4)
    )
    checks["actual_regular_second_derivative_bound_arithmetic"] = (
        regular_second_bound - (5120 * M1**2 + 640 * M2 + 640 * M1) * rho**2
    )

    # This is a formal first-variation integrand, not a uniform large-R
    # expansion of an actual regular orbit.
    y_base = (x**2 - C) / 2
    compensation = -x**3 / 2 - sp.Rational(3, 2) * x**2 - C * x / 2 - C
    h_coefficient = 3 * x - x * y_base
    b_coefficient = -3 * x**2 + x**2 * y_base - y_base**2
    first_integrand = C / q + (
        -2 * x * h_coefficient + 4 * x * compensation + b_coefficient
    ) / q**2
    checks["formal_compensated_variation_tail"] = sp.limit(first_integrand, x, sp.oo) + 3

    ti, to = sp.symbols("ti to", real=True)
    radicand_plus, radicand_minus = 1 + 2 * rho + to * rho**2, 1 - 2 * rho + ti * rho**2
    section_plus = 1 + rho + sp.sqrt(radicand_plus)
    section_minus = 1 - rho + sp.sqrt(radicand_minus)
    section_ratio_log = (
        sp.log(radicand_plus) / 2 - sp.log(radicand_minus) / 2
        + 2 * sp.log(section_plus) - 2 * sp.log(section_minus)
    )
    checks["signed_coordinate_log_ratio_second_order"] = (
        sp.series(section_ratio_log, rho, 0, 3).removeO() - 6 * rho - (to - ti) * rho**2
    )

    model_a, model_b = sp.symbols("model_a model_b", positive=True)
    detuning = sp.symbols("detuning", real=True)
    correction = model_a * r + model_b * epsilon / r
    polynomial = model_a * r**2 + detuning * r + model_b * epsilon
    checks["leading_slope_positive_denominator_clearing"] = r * (detuning + correction) - polynomial
    discriminant = detuning**2 - 4 * model_a * model_b * epsilon
    checks["leading_slope_complete_square"] = (
        4 * model_a * polynomial - (2 * model_a * r + detuning)**2 + discriminant
    )
    roots = [(-detuning + sign * sp.sqrt(discriminant)) / (2 * model_a) for sign in (-1, 1)]
    checks["leading_slope_root_sum"] = sum(roots) + detuning / model_a
    checks["leading_slope_root_product"] = roots[0] * roots[1] - model_b * epsilon / model_a
    minimizing_r = sp.sqrt(model_b * epsilon / model_a)
    checks["two_scale_minimum_value"] = correction.subs(r, minimizing_r) - 2 * sp.sqrt(
        model_a * model_b * epsilon,
    )
    checks["two_scale_minimum_stationarity"] = sp.diff(correction, r).subs(r, minimizing_r)
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def limiting_enclosures(C: Interval, L: Interval, lam: Interval) -> dict[str, Interval]:
    require(C.lo > 1, "regular coefficient domain C>1")
    require(L.lo > 0 and lam.hi < 0, "negative-lambda resonance domain")
    delta = 4 * L - lam**2
    require(delta.lo > 0, "strict discriminant margin")
    a, b = (C - 1).sqrt(), delta.sqrt()
    numerator = 16 * L - (C + 3) * lam**2
    denominator = a * b * (2 * b - lam * a)
    require(denominator.lo > 0, "positive rationalized-gap denominator")
    gap = 2 * PI_IV * numerator / denominator
    singular_log = 2 * PI_IV * lam / b
    regular_log = -4 * PI_IV / a
    return {
        "delta": delta,
        "resonance_numerator": numerator,
        "resonance_denominator": denominator,
        "log_multiplier_gap": gap,
        "limiting_singular_log_multiplier": singular_log,
        "limiting_regular_log_multiplier": regular_log,
        "limiting_singular_multiplier": exp_iv(singular_log),
        "limiting_regular_multiplier": exp_iv(regular_log),
        "limiting_multiplier_ratio": exp_iv(gap),
    }


def validate_enclosures(
    parameters: tuple[tuple[Q, Q], tuple[Q, Q], tuple[Q, Q]],
    bounds: dict[str, Interval], *, seed: int, random_samples: int,
) -> dict[str, object]:
    count = 0
    observed_gap: list[str] = []

    def check_point(C: Q, L: Q, lam: Q) -> None:
        nonlocal count
        delta = 4 * L - lam**2
        numerator = 16 * L - (C + 3) * lam**2
        require(contains_rational(bounds["delta"], delta), "exact discriminant containment")
        require(contains_rational(bounds["resonance_numerator"], numerator), "exact numerator containment")
        c_mp = mp.mpf(C.numerator) / C.denominator
        l_mp = mp.mpf(L.numerator) / L.denominator
        lam_mp = mp.mpf(lam.numerator) / lam.denominator
        a, b = mp.sqrt(c_mp - 1), mp.sqrt(4 * l_mp - lam_mp**2)
        denominator = a * b * (2 * b - lam_mp * a)
        singular_log, regular_log = 2 * mp.pi * lam_mp / b, -4 * mp.pi / a
        gap = singular_log - regular_log
        values = {
            "resonance_denominator": denominator,
            "log_multiplier_gap": gap,
            "limiting_singular_log_multiplier": singular_log,
            "limiting_regular_log_multiplier": regular_log,
            "limiting_singular_multiplier": mp.exp(singular_log),
            "limiting_regular_multiplier": mp.exp(regular_log),
            "limiting_multiplier_ratio": mp.exp(gap),
        }
        for name, value in values.items():
            enclosure = bounds[name]
            require(mp.mpf(enclosure.lo) <= value <= mp.mpf(enclosure.hi), f"secondary {name} containment")
        if count < 2:
            observed_gap.append(str(mp.nstr(gap, 35)))
        count += 1

    grids = [
        [lower] if lower == upper else [lower + (upper - lower) * Q(i, 8) for i in range(9)]
        for lower, upper in parameters
    ]
    with mp.workdps(80):
        for C in grids[0]:
            for L in grids[1]:
                for lam in grids[2]:
                    check_point(C, L, lam)
        deterministic_count = count
        rng = random.Random(seed)
        for _ in range(random_samples):
            C, L, lam = [
                lower + (upper - lower) * Q(rng.randrange(65537), 65536)
                for lower, upper in parameters
            ]
            check_point(C, L, lam)
    return {
        "deterministic_grid_points": deterministic_count,
        "seeded_rational_random_points": random_samples,
        "exact_rational_containment_checks": 2 * count,
        "secondary_transcendental_containment_checks": 7 * count,
        "secondary_precision_decimal_digits": 80,
        "first_two_sample_log_gaps": observed_gap,
        "scope": "sample validation is secondary; continuum signs follow from interval arithmetic",
    }


def coefficient_examples(random_samples: int) -> list[dict[str, object]]:
    cases = (
        ("positive_gap", ((Q(2), Q(3)), (Q(1), Q(2)), (Q(-1, 2), Q(-1, 4))), 1),
        ("negative_gap", ((Q(2), Q(3)), (Q(1), Q(1)), (Q(-199, 100), Q(-195, 100))), -1),
    )
    results = []
    for index, (name, parameters, sign) in enumerate(cases):
        C, L, lam = (rational_interval(*pair) for pair in parameters)
        bounds = limiting_enclosures(C, L, lam)
        gap, ratio = bounds["log_multiplier_gap"], bounds["limiting_multiplier_ratio"]
        require(gap.lo > 0 if sign > 0 else gap.hi < 0, f"{name} continuum log-gap sign")
        require(ratio.lo > 1 if sign > 0 else ratio.hi < 1, f"{name} continuum multiplier separation")
        lambda_abs = max(abs(value) for value in parameters[2])
        chi = 4 * parameters[1][0] - lambda_abs**2
        require(chi > 0, "exact rational discriminant lower bound")
        # B=(x-sigma*lambda/2)^2+Delta/4. The inequality
        # x^2 <= 2*(x-sigma*lambda/2)^2 + lambda^2/2 gives B>=b*(1+x^2).
        reciprocal_b = 2 + (4 + 2 * lambda_abs**2) / chi
        require(reciprocal_b >= 2, "quadratic-square coefficient domination")
        require(reciprocal_b * chi / 4 >= 1 + lambda_abs**2 / 2, "quadratic constant domination")
        results.append({
            "name": name,
            "coefficient_box": {
                key: [str(value) for value in pair]
                for key, pair in zip(("C", "L", "lambda"), parameters, strict=True)
            },
            "expected_gap_sign": sign,
            "rational_discriminant_lower": str(chi),
            "rational_quadratic_lower_coefficient": str(1 / reciprocal_b),
            "quadratic_lower_scope": "B_sigma(x)>=b*(1+x^2) for every real x and both signs",
            "limiting_enclosures": {key: interval_payload(value) for key, value in bounds.items()},
            "validation": validate_enclosures(
                parameters, bounds, seed=161670 + index, random_samples=random_samples,
            ),
        })
    return results


def logarithmic_absorption_gate() -> dict[str, object]:
    inverse_e = exp_iv(Interval.point(-1))
    rational_maximum = Q(3, 8)
    require(Q.from_float(inverse_e.hi) < rational_maximum, "certified exp(-1)<3/8")
    r_coefficient = 2 * rational_maximum
    u_coefficient = 2 * rational_maximum + 1
    total_coefficient = max(r_coefficient, u_coefficient)
    require(total_coefficient == Q(7, 4), "joint logarithm absorption coefficient")
    return {
        "domain": "0<r,u<1; epsilon=r*u; kappa=log(1/r)",
        "analytic_premise": (
            "x*log(1/x)<=1/e on 0<x<1, from differentiation and the maximum at exp(-1); "
            "this elementary analytic premise is not proved by sampling or formal replay"
        ),
        "exp_minus_one": interval_payload(inverse_e),
        "rational_maximum_upper": str(rational_maximum),
        "absorbed_expression": (
            "epsilon*kappa + epsilon*log(1/epsilon) + epsilon*log(1/u) + epsilon^2/u"
        ),
        "r_coefficient_upper": str(r_coefficient),
        "u_coefficient_upper": str(u_coefficient),
        "coefficient_of_r_plus_u_upper": str(total_coefficient),
        "unknown_analytic_error_constants_absorbed_numerically": False,
    }


def leading_model_examples() -> list[dict[str, object]]:
    """Classify rational quadratics only; no actual slope remainder is used."""
    model_a, model_b, epsilon = Q(1), Q(1), Q(1, 10000)
    cases = (
        ("two_positive_roots", Q(-3, 100), 2, 1),
        ("double_positive_root", Q(-2, 100), 1, 2),
        ("no_real_roots", Q(-1, 100), 0, 0),
        ("two_negative_roots", Q(3, 100), 0, 0),
    )
    results = []
    for name, detuning, count, multiplicity in cases:
        discriminant = detuning**2 - 4 * model_a * model_b * epsilon
        root_sum = -detuning / model_a
        root_product = model_b * epsilon / model_a
        require(root_product > 0, "positive leading-model root product")
        if name == "two_positive_roots":
            require(discriminant > 0 and root_sum > 0, name)
        elif name == "double_positive_root":
            require(discriminant == 0 and root_sum > 0, name)
        elif name == "no_real_roots":
            require(discriminant < 0, name)
        else:
            require(discriminant > 0 and root_sum < 0, name)
        results.append({
            "name": name,
            "a": str(model_a),
            "b": str(model_b),
            "epsilon": str(epsilon),
            "detuning": str(detuning),
            "discriminant": str(discriminant),
            "root_sum": str(root_sum),
            "root_product": str(root_product),
            "positive_distinct_roots": count,
            "positive_root_multiplicity": multiplicity,
            "scope": "exact leading quadratic model on omega>0, not actual displacement or slope",
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-samples", type=int, default=512)
    args = parser.parse_args()
    require(args.random_samples >= 1, "positive random validation count")
    symbolic_checks = identities()
    report = {
        "symbolic_checks": symbolic_checks,
        "symbolic_check_count": len(symbolic_checks),
        "coefficient_examples": coefficient_examples(args.random_samples),
        "logarithmic_absorption_gate": logarithmic_absorption_gate(),
        "leading_two_scale_model_examples": leading_model_examples(),
        "honesty": {
            "finite_algebra_replayed": True,
            "finite_two_scale_expansion_coefficients_replayed": True,
            "finite_anchor_and_physical_coordinate_identities_replayed": True,
            "finite_positive_base_derivative_identities_replayed": True,
            "leading_quadratic_model_examples_checked_exactly": True,
            "limiting_multiplier_box_separation_certified": True,
            "actual_joint_passage_convergence_certified_by_benchmark": False,
            "actual_joint_error_constant_certified": False,
            "actual_two_scale_error_certified_by_benchmark": False,
            "actual_positive_anchor_error_certified_by_benchmark": False,
            "joined_positive_base_cycle_count_certified_by_benchmark": False,
            "actual_regular_resonance_error_certified": False,
            "actual_resonance_slope_zero_count_certified": False,
            "actual_resonance_cycle_count_certified": False,
            "physical_small_parameter_cutoffs_certified": False,
            "kappa_derivative_matching_certified": False,
            "analytic_formally_verified": False,
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
