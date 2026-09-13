#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact algebra and rational constants for regular-passage Schwarzian persistence.

The analytic proof is in packages/omnibias-dynamics/HILBERT16-REGULAR-JETS.md.
This script checks finite identities and sufficient rational conditions,
including capture from a fixed coarse tube.
"""

from __future__ import annotations

import argparse
import json
import runpy
from fractions import Fraction as F
from pathlib import Path

import sympy as sp


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def exact_checks() -> dict[str, str]:
    n, n1, n2, p, p1, p2 = sp.symbols("n n1 n2 p p1 p2")

    def dw(expression):
        return (
            n1 * sp.diff(expression, n)
            + n2 * sp.diff(expression, n1)
            + p1 * sp.diff(expression, p)
            + p2 * sp.diff(expression, p1)
        )

    quotient3 = (
        -3 * n2 * p1 / p**2
        - 3 * n1 * p2 / p**2
        + 6 * n1 * p1**2 / p**3
        + 6 * n * p1 * p2 / p**3
        - 6 * n * p1**3 / p**4
    )
    require(
        sp.cancel(dw(dw(dw(n / p))) - quotient3) == 0,
        "third derivative of a quadratic-over-quadratic quotient",
    )

    derivatives = [n / p]
    for k in range(1, 5):
        derivatives.append(dw(derivatives[-1]))
        source = [n, n1, n2, 0, 0][k]
        product_identity = p * derivatives[k] + k * p1 * derivatives[k - 1] - source
        if k >= 2:
            product_identity += sp.binomial(k, 2) * p2 * derivatives[k - 2]
        require(sp.cancel(product_identity) == 0, f"quotient recurrence order {k}")

    x, w, C, eps, pp, r, m, alpha = sp.symbols("x w C eps pp r m alpha")
    q = (x**2 + 2 * x + C) / 2
    y0 = (x**2 - C) / 2
    y = y0 + w
    P = (1 + alpha) * x - y + x**2 + eps * (pp + r) * x * y + eps**2 * m * y**2
    Qphys = C * x + x**2 + x * y + eps * r * y**2
    N = sp.expand(Qphys - x * P)
    require(sp.expand(sp.diff(N, w, 2) - 2 * eps * r + 2 * eps**2 * m * x) == 0, "exact N_ww")
    require(sp.expand(sp.diff(P, w, 2) - 2 * eps**2 * m) == 0, "exact P_ww")
    require(
        sp.diff(N, w, 3) == 0 and sp.diff(P, w, 3) == 0,
        "both numerator and denominator are quadratic in height",
    )
    tw = -eps * pp * x**2 - C * eps * r + 2 * eps * r * w - 2 * eps**2 * m * x * (y0 + w)
    require(sp.expand(sp.diff(N, w) - 2 * x - tw) == 0, "exact T_w")
    base_quotient = (N / P).subs({eps: 0, alpha: 0})
    require(
        sp.cancel(sp.diff(base_quotient, w, 3).subs(w, 0) - 12 * x / q**3) == 0,
        "base F_www at height zero",
    )

    a, b, c, f1, f2, f3 = sp.symbols("a b c f1 f2 f3")
    schwarz = c / a - sp.Rational(3, 2) * (b / a) ** 2
    derivative = (
        sp.diff(schwarz, a) * (f1 * a)
        + sp.diff(schwarz, b) * (f2 * a**2 + f1 * b)
        + sp.diff(schwarz, c) * (f3 * a**3 + 3 * f2 * a * b + f1 * c)
    )
    require(sp.cancel(derivative - f3 * a**2) == 0, "Schwarzian derivative along the scalar flow")

    d, f4 = sp.symbols("d f4")
    flow_a = f1 * a
    flow_b = f2 * a**2 + f1 * b
    flow_c = f3 * a**3 + 3 * f2 * a * b + f1 * c
    flow_d = f4 * a**4 + 6 * f3 * a**2 * b + f2 * (4 * a * c + 3 * b**2) + f1 * d
    z2, z3, z4 = sp.symbols("z2 z3 z4")
    for order, vn, expected in (
        (2, b / a, f2 * a),
        (3, c / a, 3 * f2 * a * z2 + f3 * a**2),
        (4, d / a, f2 * a * (4 * z3 + 3 * z2**2) + 6 * f3 * a**2 * z2 + f4 * a**3),
    ):
        derivative = sum(
            sp.diff(vn, var) * flow
            for var, flow in ((a, flow_a), (b, flow_b), (c, flow_c), (d, flow_d))
        )
        require(
            sp.cancel(derivative.subs({b: a * z2, c: a * z3, d: a * z4}) - expected) == 0,
            f"normalized Bell variation order {order}",
        )

    qm = q.subs(x, -x)
    require(
        sp.cancel((x - 3) / q + (-x - 3) / qm + (5 * x**2 + 3 * C) / (q * qm)) == 0,
        "paired logarithmic derivative for q/D^2",
    )
    R = sp.symbols("R", positive=True)
    require(
        sp.cancel(1 / (1 + R / 2) - 1 / (1 + R) - R / ((R + 1) * (R + 2))) == 0,
        "tail integral expression",
    )
    require(
        sp.expand((R + 1) * (R + 2) - 6 * R - (R - 1) * (R - 2)) == 0,
        "tail integral maximum for R>=2",
    )
    require(F(12) * F(15, 64) * F(16, 81) == F(5, 9), "base margin coefficient")
    return {"status": "exact quotient, flow-Schwarzian, base and tail identities passed"}


def constant_checks(cmin: F, cmax: F, L: F) -> dict[str, object]:
    existing = runpy.run_path(str(Path(__file__).with_name("hilbert16_endpoint_passage.py")))
    report = existing["rational_constants"](cmin, cmax, L)
    v = {label: F(value) for label, value in report["exact_constants"].items()}
    kappa, Q, Y = v["kappa"], v["Q"], v["Y"]
    p0, n0, t1, d0, Kw = v["p0"], v["n0"], v["t1"], v["d0"], v["K_w"]
    kp = 2 * L * (1 + Y)
    n1 = 2 + t1
    parts = {
        "Nww_term": 48 * L * p0 / kappa**2,
        "Pww_term": 24 * L * n1 / kappa**2,
        "Nw_perturbation": 48 * t1 * p0**2 / kappa**3,
        "Pw_perturbation": 96 * kp * (p0 + 1) / kappa**3,
        "P_inverse_cube_perturbation": 456 * d0 / kappa**4,
        "N_Pw_Pww_term": 96 * n0 * p0 * L / kappa**3,
        "N_Pw_cubed_term": 96 * n0 * p0**3 / kappa**4,
    }
    K3 = sum(parts.values(), F(0))
    variation_constant = Q**2 / kappa
    KS = variation_constant**2 * (F(8, 5) * K3 + 48 * Kw / kappa**3)
    exponent_bound = 1 / (3 * kappa)
    exponent_integer = (
        exponent_bound.numerator + exponent_bound.denominator - 1
    ) // exponent_bound.denominator
    tail_ratio = F(1, 3**exponent_integer)
    gamma = v["gamma"]
    paired_gap = 3 * gamma / (1 + 3 * gamma)
    sigma = F(5, 9) * kappa / Q**2 * tail_ratio**2 * paired_gap
    eta_s = min(v["eta_p"], sigma / (2 * KS))
    constants = {
        "k_p": kp,
        "n_1": n1,
        "K_3": K3,
        "input_variation_constant": variation_constant,
        "K_S": KS,
        "tail_exponent_bound": exponent_bound,
        "tail_ratio_lower": tail_ratio,
        "gamma": gamma,
        "paired_gap_lower": paired_gap,
        "sigma_base": sigma,
        "eta_existing": v["eta_p"],
        "eta_s": eta_s,
    }
    quotient_bounds = [2 * n0 / kappa]
    quotient_bounds.append((2 / kappa) * (n1 + p0 * quotient_bounds[0]))
    quotient_bounds.append(
        (2 / kappa) * (4 * L + 2 * p0 * quotient_bounds[1] + 2 * L * quotient_bounds[0])
    )
    for k in range(3, 5):
        quotient_bounds.append(
            (2 / kappa)
            * (k * p0 * quotient_bounds[k - 1] + k * (k - 1) * L * quotient_bounds[k - 2])
        )
    integral_bounds = {
        k: quotient_bounds[k] * (2 * variation_constant) ** (k - 1) / (k - 1) for k in range(2, 5)
    }
    T2, T3, T4 = (integral_bounds[k] for k in (2, 3, 4))
    theta = v["theta"]
    section_bounds = {
        1: theta,
        2: theta * T2,
        3: theta * (3 * T2**2 + T3),
        4: theta * (15 * T2**3 + 10 * T2 * T3 + T4),
    }
    require(
        all(isinstance(value, F) and value > 0 for value in quotient_bounds),
        "rational positive quotient derivative bounds",
    )
    require(
        all(isinstance(value, F) and value > 0 for value in section_bounds.values()),
        "rational positive section derivative bounds",
    )
    for k in range(2, 5):
        require(1 - 2 * k + 4 * (k - 1) == 2 * k - 3, f"weighted spatial exponent order {k}")
    constants.update({f"C_{k}": value for k, value in enumerate(quotient_bounds)})
    constants.update({f"T_{k}": value for k, value in integral_bounds.items()})
    constants.update({f"G_derivative_{k}_upper": value for k, value in section_bounds.items()})
    dc = 2 + 2 * L * Y + L * Y**2
    Tc = v["b0"] + 2 * L + 2 * L * Q
    Ec = 2 * Tc + 6 * dc / kappa
    Fc = 2 * dc / kappa
    Gc = v["F1"] + Fc * (v["F0"] + v["F1"])
    eta_c = min(F(1), kappa / (2 * dc), 1 / (2 * v["K"] * Ec), 1 / Gc)
    constants.update(
        {
            "d_coarse": dc,
            "T_coarse": Tc,
            "E_coarse": Ec,
            "F_coarse": Fc,
            "G_coarse": Gc,
            "eta_coarse": eta_c,
        }
    )
    gates = {
        "all_rational": all(isinstance(value, F) for value in constants.values()),
        "all_positive": all(value > 0 for value in constants.values()),
        "tail_exponent_covered": exponent_integer >= exponent_bound,
        "existing_passage_sector": eta_s <= v["eta_p"],
        "squared_kernel_exponent": 4 * Kw * eta_s <= 1,
        "schwarzian_error_at_most_half_margin": KS * eta_s <= sigma / 2,
        "strict_negative_upper_bound": -sigma + KS * eta_s < 0,
        "coarse_unit_sector": eta_c <= 1,
        "coarse_positive_denominator": dc * eta_c / kappa <= F(1, 2),
        "coarse_norm_absorption": v["K"] * Ec * eta_c <= F(1, 2),
        "coarse_forcing_bound": Gc * eta_c <= 1,
        "coarse_capture_amplitude": 2 * (v["K_b"] + v["K"] * (v["F0"] + 1)) <= v["M_e"],
    }
    for label, condition in gates.items():
        require(condition, label)
    return {
        "parameter_box": {"C_min": str(cmin), "C_max": str(cmax), "L": str(L)},
        "K3_parts": {label: str(value) for label, value in parts.items()},
        "tail_exponent_integer": exponent_integer,
        "exact_constants": {label: str(value) for label, value in constants.items()},
        "gates": gates,
        "decimal_displays_only": {
            "sigma": f"{float(sigma):.12e}",
            "K_S": f"{float(KS):.12e}",
            "eta_s": f"{float(eta_s):.12e}",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/regular_jets.json")
    )
    args = parser.parse_args()
    output = exact_checks()
    output["constant_checks"] = [
        constant_checks(F(2), F(3), F(1)),
        constant_checks(F(5, 4), F(7, 4), F(2)),
    ]
    output["scope"] = (
        "Finite symbolic and rational checks; analytic proof is not formally verified."
    )
    serialized = json.dumps(output, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
