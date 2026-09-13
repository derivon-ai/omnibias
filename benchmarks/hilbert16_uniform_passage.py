#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite exact checks for the uniform nonlinear compensated-passage proof.

The Banach fixed-point and asymptotic arguments are in
packages/omnibias-dynamics/HILBERT16-UNIFORM-PASSAGE.md. These checks verify
finite algebra and rational inequalities, not the infinite analytic theorem.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path

import sympy as sp


def require_check(condition: object, label: str) -> None:
    """Keep every finite check active under Python optimization."""
    if not condition:
        raise ArithmeticError(label)


def algebra_checks() -> dict[str, object]:
    x, y, w, alpha, eps = sp.symbols("x y w alpha eps")
    C, p, r, m = sp.symbols("C p r m")
    y0 = (x**2 - C) / 2
    q = (x**2 + 2 * x + C) / 2
    b2, b3, b1 = (-(x**2) * y0, y0**2 - x**2 * y0, -x * y0**2)
    f4 = p * b2 + r * b3
    Slin = -alpha * x**2 + eps * f4 + eps**2 * m * b1
    T = (-eps * p * x**2 - C * eps * r) * w + eps * r * w**2
    T -= eps**2 * m * x * (2 * y0 * w + w**2)
    d = -w + alpha * x + eps * (p + r) * x * (y0 + w)
    d += eps**2 * m * (y0 + w) ** 2
    Z = Slin + 2 * x * w
    P = (1 + alpha) * x - y + x**2 + eps * (p + r) * x * y + eps**2 * m * y**2
    Q = C * x + x**2 + x * y + eps * r * y**2
    require_check(
        sp.expand(P.subs(y, y0 + w) - q - d) == 0, "sp.expand(P.subs(y, y0 + w) - q - d) == 0"
    )
    require_check(
        sp.expand((Q - x * P).subs(y, y0 + w) - 2 * x * w - Slin - T) == 0,
        "sp.expand((Q - x * P).subs(y, y0 + w) - 2 * x * w - Slin - T) == 0",
    )
    E = (q * T - d * Z) / (q + d)
    # Substitute the actual x-time slope into L_C w=q*w'-2*x*w.
    require_check(
        sp.cancel(q * (2 * x * w + Slin + T) / (q + d) - 2 * x * w - Slin - E) == 0,
        "sp.cancel(q * (2 * x * w + Slin + T) / (q + d) - 2 * x * w - Slin - E) == 0",
    )
    alpha1 = -(C + 6) * p - 3 * r
    U = p * (x**3 + 3 * x**2 + 3 * C / 2)
    U += r * (x**3 / 2 + 3 * x**2 / 2 + C * x / 2 + C)
    h = alpha1 * x + (p + r) * x * y0
    B = -p * x**2 - C * r
    F2 = sp.expand((U - h) * sp.diff(U, x) + B * U)
    require_check(
        sp.expand(q * sp.diff(U, x) - 2 * x * U - f4 + alpha1 * x**2) == 0,
        "sp.expand(q * sp.diff(U, x) - 2 * x * U - f4 + alpha1 * x ** 2) == 0",
    )
    require_check(
        sp.factor(F2.coeff(x, 5) - p * (2 * p + r) / 4) == 0,
        "sp.factor(F2.coeff(x, 5) - p * (2 * p + r) / 4) == 0",
    )
    ref = {w: eps * U, alpha: eps * alpha1}
    remainder_num = sp.Poly(sp.expand((q * T - d * Z - eps**2 * F2 * (q + d)).subs(ref)), eps)
    nonzero_degrees = []
    for (n,), coefficient in remainder_num.terms():
        require_check(3 <= n <= 6, "3 <= n <= 6")
        # Division by P>=kappa*j^2/2 leaves eps^n*j^(n+3),
        # bounded by eps^3*j^6 on eps*j<=1.
        degree_x = sp.Poly(coefficient, x).degree()
        require_check(degree_x <= n + 5, "degree_x <= n + 5")
        nonzero_degrees.append({"epsilon_degree": n, "x_degree_bound": int(degree_x)})
    # Exact quotient difference used by both nonlinear Lipschitz estimates.
    qv, d_a, d_b, T_a, T_b, Z_a, Z_b = sp.symbols("qv d_a d_b T_a T_b Z_a Z_b")
    E_a = (qv * T_a - d_a * Z_a) / (qv + d_a)
    E_b = (qv * T_b - d_b * Z_b) / (qv + d_b)
    difference = (qv * (T_a - T_b) - (d_a - d_b) * Z_a - d_b * (Z_a - Z_b) - E_b * (d_a - d_b)) / (
        qv + d_a
    )
    require_check(sp.cancel(E_a - E_b - difference) == 0, "sp.cancel(E_a - E_b - difference) == 0")
    return {
        "status": "all exact field, projection-source and nonlinear-remainder identities passed",
        "reference_remainder_numerator_degrees": nonzero_degrees,
        "actual_logarithmic_factor": str(p * (2 * p + r) - m),
    }


def rational_constants(cmin: F, cmax: F, direction_bound: F) -> dict[str, object]:
    """Check explicit constants using exact rational arithmetic throughout."""
    require_check(1 < cmin <= cmax and direction_bound >= 1, "admissible compact parameter box")
    L = direction_bound
    Q = cmax / 2
    kappa = min(F(1), cmin - 1) / 16
    W0 = kappa ** (-3)
    J0 = 8 / (9 * (cmax + 8) ** 3)
    A4 = 2 * W0 / J0
    B4 = 81 * Q**2 * W0 * (1 + A4 / 3)
    K = max(A4, B4)
    F0, F1 = (L * (Q**2 + 2 * Q), L * Q**2)
    U0, A0 = (L * (6 + 3 * cmax), L * (cmax + 9))
    M = max(F(1), 2 * K * (F0 + 1), 2 * U0, 2 * A0)
    Y = Q + 1
    d0 = 2 * M + 2 * L * Y + L * Y**2
    z0 = 3 * M + F0 + F1
    b0 = L * (1 + cmax)
    t0 = b0 * M + L * M + 2 * L * Q + L
    e0 = 2 * t0 + 2 * d0 * z0 / kappa
    d1 = 2 + 2 * L + 2 * L * Y
    t1 = b0 + 4 * L + 2 * L * Q
    e1 = 2 * t1 + 2 * d1 * z0 / kappa + 6 * d0 / kappa + 2 * e0 * d1 / kappa
    eta0 = min(F(1), 1 / M, kappa / (2 * d0), (F0 + 1) / (F1 + e0), 1 / (2 * K * e1))
    constants = {
        "Q": Q,
        "kappa": kappa,
        "W0": W0,
        "J0": J0,
        "A4": A4,
        "B4": B4,
        "K": K,
        "F0": F0,
        "F1": F1,
        "U0": U0,
        "A0": A0,
        "M": M,
        "Y": Y,
        "d0": d0,
        "z0": z0,
        "b0": b0,
        "t0": t0,
        "e0": e0,
        "d1": d1,
        "t1": t1,
        "e1": e1,
        "eta0": eta0,
    }
    gates = {
        "fraction_arithmetic": all(isinstance(value, F) for value in constants.values()),
        "positive_constants": all(value > 0 for value in constants.values()),
        "unit_sector": eta0 <= 1,
        "height_amplitude": M * eta0 <= 1,
        "positive_denominator": d0 * eta0 / kappa <= F(1, 2),
        "forcing_sector": eta0 * (F1 + e0) <= F0 + 1,
        "self_map_strict": K * (F0 + eta0 * (F1 + e0)) < M,
        "contraction_at_most_half": K * eta0 * e1 <= F(1, 2),
        "reference_amplitudes": M >= 2 * max(U0, A0),
    }
    for label, condition in gates.items():
        require_check(condition, label)
    return {
        "parameter_box": {"C_min": str(cmin), "C_max": str(cmax), "L": str(L)},
        "exact_constants": {label: str(value) for label, value in constants.items()},
        "gates": gates,
        "scope": "Finite exact rational checks for the displayed compact parameter box.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/uniform_passage.json")
    )
    args = parser.parse_args()
    output = algebra_checks()
    output["constant_checks"] = [
        rational_constants(F(2), F(3), F(1)),
        rational_constants(F(5, 4), F(7, 4), F(2)),
    ]
    output["scope"] = (
        "Finite symbolic identities and rational constant inequalities. The companion analytic proof is not formally verified; no singular closing-map or full-return zero bound is checked."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
