#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay exact algebra and declared-bound gates for Hilbert boundary reductions.

See packages/omnibias-dynamics/HILBERT16-BOUNDARY-REDUCTIONS.md. These finite
checks do not measure the error or prove an analytic cutoff for an ODE.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]


def require(condition: object, message: str) -> None:
    if not condition:
        raise ArithmeticError(message)


def symbolic_checks() -> list[str]:
    u, s, delta = sp.symbols("u s delta", real=True)
    B, U, a, beta, E = sp.symbols("B U a beta E", positive=True)
    kappa, v, l0, l1 = sp.symbols("kappa v l0 l1", nonzero=True)
    Bmatch = 1 / (beta + a * E * (1 / U + beta))
    checks = {
        "completed_square": (-s**2 - delta**2) + 2 * s * u - u**2
        + (u - s) ** 2 + delta**2,
        "exact_pole_margin": (
            1 - beta * (a + 1) * B - B * a * (E / U + beta * (E - 1))
        ).subs(B, Bmatch),
        "inverse_leading_map": (a * B / (1 - beta * (a + 1) * B)).subs(
            B, U / (a + beta * (a + 1) * U)
        ) - U,
        "secondary_polynomial_chart": kappa**2 * l0 + kappa * l1 * (kappa * v)
        - (kappa * v) ** 2 - kappa**2 * (l0 + l1 * v - v**2),
    }
    for name, residual in checks.items():
        require(sp.simplify(residual) == 0, name)
    return list(checks)


def bottleneck_gate(
    *, s0: Q, S: Q, K: Q, eta: Q, endpoint_log_bound: Q
) -> dict[str, str | bool]:
    """Evaluate the sufficient inequality using declared rational bounds."""
    require(0 < s0 <= S < K, "positive nonzero fold interval")
    require(0 < eta < s0 / 2 and eta < (K - S) / 2, "bottleneck domain")
    require(endpoint_log_bound >= 0, "nonnegative endpoint logarithm bound")
    forced = s0 / (3 * eta)
    opposite = 2 * K**2 / s0**2
    crossing_slack = (K - S) ** 2 - eta**2
    require(crossing_slack > 0, "inward boundary crossing with small E")
    return {
        "eta": str(eta),
        "forced_log_change": str(forced),
        "opposite_half_bound": str(opposite),
        "endpoint_log_bound": str(endpoint_log_bound),
        "crossing_slack": str(crossing_slack),
        "strict_margin": str(forced - opposite - endpoint_log_bound),
        "sufficient_gate": forced > opposite + endpoint_log_bound,
    }


def pole_gate(*, betamax: Q, Umax: Q) -> dict[str, str]:
    require(betamax > 0 and Umax > 0, "positive compact pole inputs")
    d0 = min(Q(1, 4), 1 / (4 * betamax * Umax))
    bracket = (1 - d0) / Umax - betamax * d0
    require(bracket >= 1 / (2 * Umax), "pole bracket lower bound")
    return {
        "d0": str(d0),
        "bracket_lower": str(bracket),
        "required_lower": str(1 / (2 * Umax)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    good = bottleneck_gate(
        s0=Q(1), S=Q(2), K=Q(4), eta=Q(1, 200), endpoint_log_bound=Q(4)
    )
    inconclusive = bottleneck_gate(
        s0=Q(1), S=Q(2), K=Q(4), eta=Q(1, 20), endpoint_log_bound=Q(4)
    )
    require(good["sufficient_gate"], "small bottleneck gate passes")
    require(not inconclusive["sufficient_gate"], "larger bottleneck is inconclusive")
    rejected = []
    try:
        bottleneck_gate(s0=Q(0), S=Q(2), K=Q(4), eta=Q(1, 200), endpoint_log_bound=Q(4))
    except ArithmeticError:
        rejected.append("zero_fold_center_margin")
    require(len(rejected) == 1, "zero-margin input rejection remains active")
    try:
        pole_gate(betamax=Q(1, 3), Umax=Q(0))
    except ArithmeticError:
        rejected.append("zero_output_bound")
    require(len(rejected) == 2, "zero-output input rejection remains active")
    report = {
        "symbolic_checks": symbolic_checks(),
        "bottleneck_examples": [good, inconclusive],
        "pole_example": pole_gate(betamax=Q(1, 3), Umax=Q(10)),
        "rejected_inputs": rejected,
        "honesty": {
            "finite_algebra_replayed": True,
            "example_bounds_measured_on_actual_field": False,
            "analytic_cutoff_computed": False,
            "analytic_theorem_formally_verified": False,
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
