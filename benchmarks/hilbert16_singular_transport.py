#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact algebra checks for the actual singular-section coordinate transport.

Requires SymPy. The analytic argument is in
packages/omnibias-dynamics/HILBERT16-SINGULAR-TRANSPORT.md. This script checks
finite identities; it does not verify the analytic entry-exit theorem,
passive-parameter uniformity, or a Hilbert-16 conclusion.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TypedDict

import sympy as sp  # type: ignore[import-untyped]


class IdentityCheck(TypedDict):
    passed: bool
    residual: str


def record_identity(checks: dict[str, IdentityCheck], name: str, expression: sp.Expr) -> None:
    """Reject unresolved or nonzero residuals, including under Python -O."""
    if name in checks:
        raise ValueError(f"Duplicate identity label: {name}")
    residual = sp.factor(sp.simplify(expression))
    if residual != sp.S.Zero:
        raise ArithmeticError(f"Identity {name} did not simplify to zero: {residual}")
    checks[name] = {"passed": True, "residual": str(residual)}


def check_rejection() -> None:
    """Exercise the same checker with false and unresolved obligations."""
    for residual in (sp.S.One, sp.Symbol("unresolved")):
        try:
            record_identity({}, "intentional_rejection", residual)
        except ArithmeticError:
            continue
        raise ArithmeticError("The identity checker accepted a nonzero or unresolved residual")


def exact_checks() -> dict[str, object]:
    checks: dict[str, IdentityCheck] = {}

    def check(name: str, expression: sp.Expr) -> None:
        record_identity(checks, name, expression)

    check_rejection()
    nu, v, yf, r, A, C, m, p = sp.symbols("nu v yf r A C m p")
    V = -(r + v + nu * v**2 + C * nu**2 * v * yf)
    ell = 1 + 2 * nu * v + C * nu**2 * yf
    F = m + p * v - nu * v**3 - yf + A * nu * v * yf - C * nu**2 * v**2 * yf
    # Reverse time in the family chart: vdot=-F and yfdot=-V*yf.
    check(
        "exact_reversed_V_field",
        -sp.diff(V, v) * F - sp.diff(V, yf) * V * yf - (ell * F + C * nu**2 * v * V * yf),
    )
    fast_v = sp.Symbol("fast_v")
    invariant = fast_v**2 - 2 * yf
    check(
        "fast_invariant",
        sp.diff(invariant, fast_v) * (-yf) + sp.diff(invariant, yf) * (-fast_v * yf),
    )

    v0, k, lambda0, lambda1 = sp.symbols("v0 k lambda0 lambda1")
    ell_v = 1 + 2 * nu * v
    coefficients = [ell_v * (m + p * v - nu * v**3)]
    # d/dV=-(1/ell_v)d/dv. Division by j generates Taylor coefficients.
    for j in range(1, 4):
        coefficients.append(sp.factor(-sp.diff(coefficients[-1], v) / (j * ell_v)))
    ell0 = 1 + 2 * nu * v0
    F0 = nu**2 * k**3 * lambda0 / ell0
    F1 = -nu * k**2 * lambda1 - 2 * nu**3 * k**3 * lambda0 / ell0**2
    p_tilde = 3 * v0**2 + F1
    m_tilde = F0 - p_tilde * v0 + v0**3
    embedded = [
        sp.factor(coefficient.subs({v: v0, m: nu * m_tilde, p: nu * p_tilde}))
        for coefficient in coefficients
    ]
    check("canonical_constant", embedded[0] - (nu * k) ** 3 * lambda0)
    check("canonical_linear", embedded[1] - (nu * k) ** 2 * lambda1)
    k_rhs = 3 * v0 / ell0 + nu**2 * k**2 * lambda1 / ell0**2 + 4 * nu**4 * k**3 * lambda0 / ell0**4
    check("canonical_quadratic_IFT", embedded[2] + nu * k_rhs)
    beta = (
        1 / (k * ell0**2) - 2 * nu**3 * k * lambda1 / ell0**4 - 8 * nu**5 * k**2 * lambda0 / ell0**6
    )
    check("canonical_cubic_beta", embedded[3] - nu * k * beta)

    rho, z, t = sp.symbols("rho z t", positive=True)
    for sigma in (-1, 1):
        radical = sp.sqrt(1 - 2 * sigma * r * rho + rho**2 * t)
        Z = 2 / (1 - sigma * r * rho + radical) - 1
        K_of_Z = r**2 + 4 * sigma * r / (rho * (1 + Z)) - 4 * Z / (rho**2 * (1 + Z) ** 2)
        check(f"section_inverse_{sigma}", K_of_Z - t)
        expansion = sp.series(Z, rho, 0, 3).removeO()
        check(
            f"section_expansion_{sigma}",
            expansion - (sigma * r * rho + (5 * r**2 - t) * rho**2 / 4),
        )
        v_section = sigma * 2 / (rho * (1 + Z))
        V_section = -r - v_section
        check(f"section_orientation_formula_{sigma}", V_section + sigma * (1 + radical) / rho)
        Y = rho**2 * (1 + z) / 2 + sigma * nu * rho * z + C * nu**2 * (z - 1) / 2
        x = sigma * rho / nu
        q = (x**2 + 2 * x + C) / 2
        y = (x**2 - C) / 2 + z * q
        check(f"exact_moving_height_{sigma}", nu**2 * y - Y)

    u, a, b = sp.symbols("u a b", positive=True)
    H = sp.Function("H")
    M = a * u / (1 - b * u)
    displacement = H(u**2) - M**2
    expected = (
        4 * u * sp.Subs(sp.Derivative(H(t), (t, 2)), t, u**2) - 6 * a**2 * b / (1 - b * u) ** 4
    )
    check("weighted_two_zero_identity", sp.diff(sp.diff(displacement, u) / u, u) - expected)

    if len(checks) != 15 or not all(check["passed"] for check in checks.values()):
        raise ArithmeticError("The expected complete set of 15 identities did not pass")
    return {
        "status": "all 15 exact singular-transport identities passed",
        "identity_count": len(checks),
        "all_passed": True,
        "rejects_nonzero_and_unresolved": True,
        "checks": checks,
        "scope": (
            "Finite symbolic identities only. Neither the analytic entry-exit theorem, "
            "passive-parameter uniformity, nor a Hilbert-16 conclusion is verified here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "singular_transport.json",
    )
    args = parser.parse_args()
    output = exact_checks()
    serialized = json.dumps(output, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
