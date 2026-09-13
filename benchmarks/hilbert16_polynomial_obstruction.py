#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact reduction modulo L=(x^2+2*x+C)/2*d/dx-2*x.

The output proves finite polynomial identities, not convergence of an
infinite formal invariant graph. Requires SymPy only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sympy as sp


def require_check(condition: object, label: str) -> None:
    """Keep exact obligations active even when Python runs with -O."""
    if not condition:
        raise ArithmeticError(label)


x, C = sp.symbols("x C")
q = (x**2 + 2 * x + C) / 2


def L(polynomial: sp.Expr) -> sp.Expr:
    return sp.expand(q * sp.diff(polynomial, x) - 2 * x * polynomial)


M = sp.Matrix([[L(x**j).coeff(x, i) for j in range(5)] for i in range(5)])
M_inverse = M.inv()


def reduce_forcing(forcing: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Return the unique (V,b) with forcing=L(V)+b*x^5 over Q(C)[x].

    Additional symbolic coefficient parameters are also allowed. Fixed-C
    specialization of this formula requires C!=0,-3 and defined coefficients.
    """
    residual = sp.Poly(sp.expand(forcing), x)
    primitive = sp.S.Zero
    while not residual.is_zero and residual.degree() >= 6:
        n = residual.degree()
        term = 2 * residual.LC() * x ** (n - 1) / (n - 5)
        primitive += term
        residual = sp.Poly(sp.expand(residual.as_expr() - L(term)), x)
    obstruction = sp.cancel(residual.nth(5))
    low = residual.as_expr() - obstruction * x**5
    low_coefficients = sp.Matrix([sp.expand(low).coeff(x, j) for j in range(5)])
    solution = M_inverse * low_coefficients
    primitive += sum(sp.cancel(solution[j]) * x**j for j in range(5))
    primitive = sp.cancel(primitive)
    require_check(
        sp.cancel(forcing - L(primitive) - obstruction * x**5) == 0,
        "exact identity check at source line 45",
    )
    return primitive, obstruction


def monomial_obstructions(max_degree: int) -> list[sp.Expr]:
    """b(x^n), n=0..max_degree, from the exact scalar recurrence."""
    out = [sp.S.Zero] * (max_degree + 1)
    if max_degree >= 5:
        out[5] = sp.S.One
    for n in range(6, max_degree + 1):
        out[n] = sp.expand(-(n - 1) * (2 * out[n - 1] + C * out[n - 2]) / (n - 5))
    return out


def exact_checks() -> dict[str, object]:
    require_check(sp.factor(M.det()) == 8 * C * (C + 3), "exact identity check at source line 60")
    obstructions = monomial_obstructions(20)
    for n in range(21):
        _, b = reduce_forcing(x**n)
        require_check(sp.expand(b - obstructions[n]) == 0, "exact identity check at source line 64")

    # Recover independently supplied primitives and obstructions, including
    # high degrees, rational coefficients and parameter dependence.
    rng = random.Random(731)
    for degree in (0, 1, 4, 5, 6, 9, 13, 18):
        V = sum(
            sp.Rational(rng.randint(-8, 8), rng.randint(1, 7)) * (1 + C * (j % 3)) * x**j
            for j in range(degree + 1)
        )
        b_expected = sp.Rational(rng.randint(-8, 8), rng.randint(1, 7)) * (C + 1)
        V_found, b_found = reduce_forcing(L(V) + b_expected * x**5)
        require_check(sp.cancel(V_found - V) == 0, "exact identity check at source line 74")
        require_check(
            sp.cancel(b_found - b_expected) == 0, "exact identity check at source line 75"
        )

    p, r, m1 = sp.symbols("p r m1")
    alpha1 = -(C + 6) * p - 3 * r
    U = p * (x**3 + 3 * x**2 + 3 * C / 2)
    U += r * (x**3 / 2 + 3 * x**2 / 2 + C * x / 2 + C)
    h = alpha1 * x + (p + r) * x * (x**2 - C) / 2
    B = -p * x**2 - C * r
    forcing = sp.expand((U - h) * sp.diff(U, x) + B * U - m1 * x * (x**2 - C) ** 2 / 4)
    _, b_second = reduce_forcing(forcing)
    require_check(
        sp.factor(b_second - (p * (2 * p + r) - m1) / 4) == 0,
        "exact identity check at source line 86",
    )

    # Formal integrating factor: Theta'(t)=-4*Theta/(1+2t+C*t^2), Theta(0)=1.
    # Generate enough terms without transcendental evaluation and independently
    # check that Theta/(1+2t+C*t^2)^3 generates the monomial obstruction.
    t = sp.symbols("t")
    order = 12
    theta_coeffs = [sp.S.One]
    for n in range(order):
        previous = theta_coeffs[n - 1] if n >= 1 else sp.S.Zero
        theta_coeffs.append(
            sp.expand((-(2 * n + 4) * theta_coeffs[n] - C * (n - 1) * previous) / (n + 1))
        )
    theta = sum(theta_coeffs[n] * t**n for n in range(order + 1))
    generating = sp.series(theta / (1 + 2 * t + C * t**2) ** 3, t, 0, order + 1).removeO()
    for k in range(order + 1):
        require_check(
            sp.expand(generating.coeff(t, k) - obstructions[k + 5]) == 0,
            "exact identity check at source line 102",
        )

    return {
        "status": "all exact reductions, reconstruction tests and generating coefficients passed",
        "matrix": str(M),
        "determinant": str(sp.factor(M.det())),
        "monomial_obstructions_0_to_12": [str(item) for item in obstructions[:13]],
        "second_order_obstruction": str(sp.factor(b_second)),
        "scope": "finite polynomial algebra; no formal-series convergence or Hilbert-16 claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hilbert16/polynomial-obstruction.json")
    )
    args = parser.parse_args()
    output = exact_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
