#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite identities and inequality examples for the thin-height arguments."""

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
    V, h, q, qV, k, kV, kh = sp.symbols("V h q q_V k k_V k_h")
    denominator = q + k * h
    R = V * h / denominator
    Rh = sp.diff(R, h) + sp.diff(R, k) * kh
    logarithmic_derivative = R / h - (qV + h * kV + (k + h * kh) * R) / denominator
    checks = {
        "exact_height_variation": Rh - V * (q - h**2 * kh) / denominator**2,
        "compensated_logarithmic_variation": logarithmic_derivative
        - Rh
        + (qV + h * kV) / denominator,
    }
    F, FV, H = sp.symbols("F F_V H", nonzero=True)
    checks["inverse_height_equation"] = sp.diff(V**2 / 2, V) * (-F / (V * h)) + F / h
    checks["inverse_height_variation_coefficient"] = (-FV / h) / V + FV / (h * V)
    time, F0 = sp.symbols("time F0", nonzero=True)
    checks["center_square_root_leading_coefficient"] = (F0 * time) ** 2 / 2 - (
        -F0 / H
    ) * (-F0 * H * time**2 / 2)

    eps, lam0, lam1, beta = sp.symbols("epsilon lambda0 lambda1 beta")
    Z = sp.Function("Z")(V)
    gV = sp.symbols("g_V")
    zeta = -1 + V * Z
    f = eps**3 * lam0 + eps**2 * lam1 * V + eps * V**2 * zeta
    envelope = -2 * eps + (eps**2 * lam1 + h * gV) / V + eps * V * (
        3 * Z + V * sp.diff(Z, V)
    )
    checks["exact_drift_envelope"] = (sp.diff(f, V) + h * gV) / V - envelope
    checks["limiting_positive_cubic_factor"] = (3 * Z + V * sp.diff(Z, V)).subs(
        {Z: beta, sp.diff(Z, V): 0}
    ) - 3 * beta
    checks["envelope_integral"] = sp.diff(2 * eps * sp.log(h / H), h) - 2 * eps / h

    rho, nu, C, Xi, Th = sp.symbols("rho nu C Xi T_h")
    for sigma in (-1, 1):
        event = V - 1 + sigma * rho * h + nu * rho**2 * h**2 + C * nu**2 * sigma * rho * h**2
        label = (sigma * rho * h - 1) ** 2 - 2 * h
        event_h = sp.diff(event, h)
        event_dh = -(Xi / V) / (event_h + Th / V)
        factor = -sp.diff(label, h) / (Th + V * event_h)
        checks[f"moving_height_event_{sigma}"] = sp.diff(label, h) * event_dh - factor * Xi
        checks[f"base_height_event_factor_{sigma}"] = factor.subs(
            {nu: 0, Th: 1, h: (1 - V) / (sigma * rho)}
        ) - 2

    x, w, eta, B, Bmax = sp.symbols("x w eta B Bmax", positive=True)
    Phi = Bmax * sp.log(w / eta) + 2 * (w - eta)
    radial_slope = x * w / (B + k * w)
    checks["radial_escape_comparison"] = sp.diff(Phi, w) * radial_slope - x - x * (
        Bmax - B + (2 - k) * w
    ) / (B + k * w)
    Kcap, scale = sp.symbols("K A", positive=True)
    delta_general = eps * (1 + Kcap)
    core_radius_squared = eps**3 * scale * (1 + Kcap)
    checks["general_core_scale"] = core_radius_squared / eps**2 - scale * delta_general
    checks["general_tail_rate_margin"] = delta_general / scale - eps**4 / core_radius_squared - (
        eps * Kcap * (Kcap + 2) / (scale * (1 + Kcap))
    )
    primitive = eps * sp.log(V) - eps**2 / V
    checks["compensated_tail_primitive"] = sp.diff(primitive, V) - eps / V - eps**2 / V**2
    U, L, delta = sp.symbols("U L delta", nonzero=True)
    ui = U - delta * lam1 * U**2 / (3 * L)
    uo = U + delta * lam1 * U**2 / (3 * L)
    apparent_term = (ui**2 - uo**2) / (2 * L) + delta * lam1 * (ui**3 + uo**3) / (3 * L**2)
    checks["unequal_escape_first_order_cancellation"] = sp.diff(apparent_term, delta).subs(delta, 0)
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def rational_examples() -> dict[str, object]:
    Bmax = Q(5)
    A = 256 * (1 + Bmax) ** 2
    require(A >= 16, "growing-core scale")
    require(A >= 16 * Bmax, "escape log-height contribution")
    # log A = log(256)+2log(1+Bmax) < 6+2Bmax.
    require(A >= 8 * Bmax * (6 + 2 * Bmax), "escape logarithmic scale contribution")
    endpoint_error, eps, ratio = Q(1, 10000), Q(1, 100000), Q(4)
    # exp(-x)>=1-x and log(ratio)<=ratio-1 give a rational lower gate.
    drift_lower = (2 - endpoint_error) / (2 + endpoint_error) * (1 - 2 * eps * (ratio - 1))
    theta = Q(13, 14)
    require(drift_lower > theta, "nonnegative-drift derivative separates from regular map")
    declared_constant, delta0, kappa_star = Q(10), Q(1, 100), Q(1, 1000000)
    margin = (1 - theta) / 2
    require(2 * kappa_star < delta0, "declared initial exponential sector is inside delta gate")
    require(declared_constant**2 * 2 * kappa_star < margin**2, "declared exponential slope margin")
    exponent_rows = []
    for p in (Q(1, 4), Q(1, 2), Q(3, 4)):
        rate = (1 - p) / 2
        orders = {"core_linear": rate, "core_quadratic": 1 - p, "tail_inverse": (1 + p) / 2}
        require(all(order >= rate > 0 for order in orders.values()), "thin-band powers vanish")
        require(1 - rate > 0, "epsilon log is smaller than the reported rate")
        require(3 - p > 2, "core location error is smaller than epsilon squared")
        exponent_rows.append({"p": str(p), **{name: str(value) for name, value in orders.items()}})
    return {
        "escape_scale_A": str(A),
        "declared_drift_lower": str(drift_lower),
        "regular_slope_upper": str(theta),
        "initial_exponential_example": {
            "declared_error_constant": str(declared_constant),
            "declared_delta0": str(delta0),
            "kappa_star": str(kappa_star),
            "epsilon_upper": str(kappa_star),
        },
        "rescaling_examples": exponent_rows,
        "scope": "finite sufficient-inequality examples; actual coefficient bounds and cutoffs are not certified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "symbolic_checks": identities(),
        "rational_examples": rational_examples(),
        "honesty": {
            "finite_algebra_replayed": True,
            "analytic_theorems_formally_verified": False,
            "physical_small_parameter_cutoff_certified": False,
            "all_singular_parameter_layers_covered": False,
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
