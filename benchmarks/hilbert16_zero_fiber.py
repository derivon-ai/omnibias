#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Check exact algebra for the actual zero-fiber section obstruction.

The default report contains eleven symbolic identities. Optional --diagnostics
adds ordinary floating-point trajectories, which are not validated orbit
certificates. The analytic proof and parameter domains are documented in
packages/omnibias-dynamics/HILBERT16-ZERO-FIBER.md.
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


class DiagnosticRecord(TypedDict):
    nu: float
    signed_output_label: float
    label_over_nu: float
    relative_error_to_first_order: float
    z_out_minus_zero_label: float
    central_minimum_h_sample: float


class DiagnosticsReport(TypedDict):
    status: str
    parameters: dict[str, float]
    predicted_shift_coefficient: float
    records: list[DiagnosticRecord]


class ZeroFiberReport(TypedDict):
    status: str
    identity_count: int
    all_passed: bool
    rejects_nonzero_and_unresolved: bool
    checks: dict[str, IdentityCheck]
    diagnostics: DiagnosticsReport | None
    scope: str


def record_identity(checks: dict[str, IdentityCheck], name: str, expression: sp.Expr) -> None:
    """Reject unresolved, false, or duplicate checks, including under Python -O."""
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
        raise ArithmeticError("The identity checker accepted a false or unresolved obligation")


def exact_checks() -> dict[str, IdentityCheck]:
    checks: dict[str, IdentityCheck] = {}

    def check(name: str, expression: sp.Expr) -> None:
        record_identity(checks, name, expression)

    check_rejection()
    nu, V, h, r, A, C = sp.symbols("nu V h r A C")
    v = -r - V - nu * (r + V) ** 2
    F = 2 * nu * r**3 + 3 * nu * r**2 * v - nu * v**3 - h + A * nu * v * h - C * nu**2 * v**2 * h
    N = (1 + 2 * nu * v + C * nu**2 * h) * F + C * nu**2 * v * V * h
    leading = -h + nu * (V**3 + 3 * r * V**2 + (2 - A) * (r + V) * h)
    check("actual_normal_first_order", sp.series(N - leading, nu, 0, 2).removeO())
    L = -2 * (4 - A) * V**2 - 2 * (8 - A) * r * V
    k0 = -3 * r
    beta = 1 / k0
    g1 = (2 - A) * (r + V) / k0
    check("V_time_first_variation", L - k0 * (-4 * V * (-1 + beta * V) - 2 * V * g1))
    p = -sp.Rational(2, 3) * (4 - A) * V**3 - (8 - A) * r * V**2
    check("first_variation_primitive", sp.diff(p, V) - L)

    rho = sp.symbols("rho", nonzero=True)
    for sigma in (-1, 1):
        z = 4 / (rho**2 * V**2) - 1
        Y0 = 2 / V**2
        Y1 = sigma * rho * z
        vv0 = sigma * rho / Y0
        vv1 = -vv0 * Y1 / Y0
        hh1 = -Y1 / Y0**2
        VV1 = -vv1 - vv0**2
        e = 2 * V**3 - rho**2 * V**5 + 2 * sigma * V**2 / rho - sigma * rho * V**4 / 2
        check(f"Gamma_first_order_{sigma}", 2 * V * VV1 - 2 * hh1 - e)
        constrained = (e - p.subs(A, 1)).subs(rho, -2 * sigma * (r + V) / V**2)
        target = r * V * (V**2 - 4 * r * V - 4 * r**2) / (V + r)
        check(f"endpoint_first_order_primitive_{sigma}", constrained - target)

    a, b = sp.symbols("a b", positive=True)
    Vi, Vo = (1 + a) / rho, -(1 + b) / rho
    E = r * V**2 - 5 * r**2 * V + r**3 - r**4 / (V + r)
    kappa = E.subs(V, Vi) - E.subs(V, Vo)
    negative_form = r**2 * (2 + a + b) / rho * (4 / (a + b) - 5) - 2 * r**4 * rho * (
        1 / (1 + a) ** 2 + 1 / (1 + b) ** 2
    )
    # Check the rational identity modulo the two radical equations. The proof
    # separately establishes nonzero denominators and the strict negative sign.
    numerator = sp.together(kappa - negative_form).as_numer_denom()[0]
    basis = sp.groebner(
        [a**2 - 1 - 2 * r * rho, b**2 - 1 + 2 * r * rho],
        a,
        b,
        r,
        rho,
        order="lex",
    )
    reduced = sp.reduced(numerator, basis.polys, a, b, r, rho)[1]
    check("strict_negative_kappa_formula", reduced.as_expr())

    Vi = sp.symbols("V_in", positive=True)
    rho_i = 2 * (Vi - 1) / Vi**2
    z_i = 4 / (rho_i**2 * Vi**2) - 1
    Y1_i = -rho_i * z_i
    eta_i = Y1_i * Vi**3 * (1 - Vi / 2)
    eta_expected = Vi * (2 * Vi - 1) * (Vi - 2) / (Vi - 1)
    check("old_B_zero_new_label", eta_i - eta_expected)
    tau_c = 3 * Vi**2 + 2 * Vi
    margin = Vi**2 + 5 * Vi + Vi / (Vi - 1)
    check("strict_rapid_corridor_margin", tau_c - eta_expected - margin)

    u, lam0 = sp.symbols("u lambda0", real=True)
    w = sp.symbols("w", positive=True)
    J = u**2 - 2 * w + 2 * lam0 * sp.log(w)
    check("deep_core_first_integral", sp.diff(J, u) * (lam0 - w) + sp.diff(J, w) * (-u * w))

    if len(checks) != 11 or not all(check["passed"] for check in checks.values()):
        raise ArithmeticError("The expected complete set of eleven identities did not pass")
    return checks


def numerical_diagnostics() -> DiagnosticsReport:
    """Run optional, explicitly nonvalidated floating-point trajectories."""
    import numpy as np
    import numpy.typing as npt
    from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
    from scipy.optimize import root_scalar  # type: ignore[import-untyped]

    r, rho, C, A = -1.0, 0.1, 2.0, 1.0
    Vi = (1 + np.sqrt(1 + 2 * r * rho)) / rho
    Vo = -(1 + np.sqrt(1 - 2 * r * rho)) / rho

    def E(V: float) -> float:
        return r * V * (V * V - 4 * r * V - 4 * r * r) / (V + r)

    predicted = E(float(Vi)) - E(float(Vo))
    zout0 = 2 / (1 - r * rho + np.sqrt(1 - 2 * r * rho)) - 1
    zin0 = 2 / (1 + r * rho + np.sqrt(1 + 2 * r * rho)) - 1
    records: list[DiagnosticRecord] = []
    for nu in (1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
        lam0, lam1 = -1.0, 0.0
        v0 = -2 * r / (1 + np.sqrt(1 - 4 * nu * r))
        ell = 1 + 2 * nu * v0

        def normalization(
            k: float,
            nu_value: float = nu,
            v0_value: float = float(v0),
            ell_value: float = float(ell),
            lambda0: float = lam0,
            lambda1: float = lam1,
        ) -> float:
            return (
                k
                - 3 * v0_value / ell_value
                - nu_value**2 * k * k * lambda1 / ell_value**2
                - 4 * nu_value**4 * k**3 * lambda0 / ell_value**4
            )

        root = root_scalar(normalization, bracket=(1.0, 5.0))
        k = float(root.root)
        if not root.converged or not np.isfinite(k) or k <= 0:
            raise ArithmeticError("The diagnostic canonical parameter solve did not converge")
        F0 = nu**2 * k**3 * lam0 / ell
        F1 = -nu * k * k * lam1 - 2 * nu**3 * k**3 * lam0 / ell**2
        pt = 3 * v0 * v0 + F1
        mt = F0 - pt * v0 + v0**3
        Y = rho * rho * (1 + zin0) / 2 - nu * rho * zin0 + C * nu * nu * (zin0 - 1) / 2
        vin, hin = -rho / Y, 1 / Y

        def rhs(
            v: float,
            values: npt.NDArray[np.float64],
            nu_value: float = nu,
            mt_value: float = float(mt),
            pt_value: float = float(pt),
        ) -> list[float]:
            h = float(values[0])
            FF = (
                nu_value * mt_value
                + nu_value * pt_value * v
                - nu_value * v**3
                - h
                + A * nu_value * v * h
                - C * nu_value**2 * v * v * h
            )
            NN = r + v + nu_value * v * v + C * nu_value**2 * v * h
            return [float(-NN * h / FF)]

        def event(v: float, values: npt.NDArray[np.float64]) -> float:
            return v - rho * float(values[0])

        # SciPy supports these documented attributes on an event callback.
        event.direction = -1  # type: ignore[attr-defined]
        event.terminal = True  # type: ignore[attr-defined]
        out = solve_ivp(
            rhs,
            (vin, 40),
            [hin],
            method="DOP853",
            rtol=2e-13,
            atol=1e-13,
            events=event,
            max_step=0.1,
        )
        if not out.success or len(out.t_events[0]) != 1:
            raise ArithmeticError(f"Diagnostic passage did not terminate: {out.message}")
        hout = float(out.y_events[0][0, 0])
        if not np.isfinite(hout) or hout <= 0:
            raise ArithmeticError("The diagnostic outgoing height is not finite and positive")
        zout = (2 / hout - rho * rho + C * nu * nu) / (rho * rho + 2 * nu * rho + C * nu * nu)
        signed_t = r * r + 4 * r / (rho * (1 + zout)) - 4 * zout / (rho * rho * (1 + zout) ** 2)
        records.append(
            {
                "nu": nu,
                "signed_output_label": signed_t,
                "label_over_nu": signed_t / nu,
                "relative_error_to_first_order": abs(signed_t / nu - predicted) / abs(predicted),
                "z_out_minus_zero_label": float(zout - zout0),
                "central_minimum_h_sample": float(np.min(out.y[0])),
            }
        )
    return {
        "status": "ordinary floating-point diagnostics, not a certificate",
        "parameters": {"r": r, "rho": rho, "C": C, "A": A, "lambda0": -1.0, "lambda1": 0.0},
        "predicted_shift_coefficient": predicted,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
        / "hilbert16"
        / "zero_fiber.json",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="also run nonvalidated floating-point trajectories (requires NumPy and SciPy)",
    )
    args = parser.parse_args()
    checks = exact_checks()
    report: ZeroFiberReport = {
        "status": "all eleven exact zero-fiber identities passed",
        "identity_count": len(checks),
        "all_passed": True,
        "rejects_nonzero_and_unresolved": True,
        "checks": checks,
        "diagnostics": numerical_diagnostics() if args.diagnostics else None,
        "scope": (
            "Finite symbolic identities only. The analytic limits, domain inequalities, "
            "and cyclicity conclusions are not formally verified. Optional numerical "
            "trajectories are diagnostics, not validated orbit certificates."
        ),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
