#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay χ-scale identities and a concrete frozen-exponent failure.

The written atlas and G1/G4 verdicts live in
packages/omnibias-dynamics/HILBERT16-COALESCING-CAPTURE.md.
This benchmark does not prove a physical passage remainder, G1, G4, or
Hilbert XVI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]

SEED = 160027


def require(condition: object, label: str) -> None:
    if not condition:
        raise ArithmeticError(label)


def record_identity(checks: list[str], name: str, residual: sp.Expr) -> None:
    if name in checks:
        raise ValueError(f"duplicate identity: {name}")
    if sp.simplify(residual) != 0:
        raise ArithmeticError(f"nonzero or unresolved identity: {name}")
    checks.append(name)


def identities() -> list[str]:
    sep, r, kappa, chi, C, gamma, K, theta = sp.symbols(
        "sep r kappa chi C gamma K theta", nonzero=True
    )
    L, lam1, x, eps, B, Bx = sp.symbols("L lam1 x eps B Bx")
    checks: list[str] = []
    chi_def = (sep / r) * kappa
    record_identity(
        checks,
        "linear_exit_is_exp_neg_chi",
        sp.exp(-(sep / r) * kappa) - sp.exp(-chi).subs(chi, chi_def),
    )
    record_identity(
        checks,
        "chi_derivative_of_linear_exit",
        sp.diff(sp.exp(-chi), chi) + sp.exp(-chi),
    )
    record_identity(
        checks,
        "kappa_derivative_of_linear_exit",
        sp.diff(sp.exp(-(sep / r) * kappa), kappa)
        + (sep / r) * sp.exp(-(sep / r) * kappa),
    )
    ratio = ((sep / r) * sp.exp(-(sep / r) * kappa)) / (
        C * sp.exp(-gamma * kappa)
    )
    record_identity(
        checks,
        "kappa_sensitivity_ratio",
        ratio - (sep / (r * C)) * sp.exp((gamma - sep / r) * kappa),
    )
    record_identity(
        checks,
        "chi_of_reciprocal_sep",
        (sep / r) * (K / sep) - K / r,
    )
    record_identity(
        checks,
        "chi_of_scaled_kappa",
        (sep / r) * (chi * r / sep) - chi,
    )
    r1 = (-lam1 - sep) / 2
    r2 = (-lam1 + sep) / 2
    rstar = -lam1 / 2
    record_identity(checks, "root_sum", r1 + r2 + lam1)
    record_identity(checks, "root_product", r1 * r2 - (lam1**2 - sep**2) / 4)
    Bminus = (x - r1) * (x - r2)
    record_identity(
        checks,
        "limiting_outgoing_quadratic",
        sp.expand(Bminus - (x**2 + lam1 * x + (lam1**2 - sep**2) / 4)),
    )
    record_identity(
        checks,
        "limiting_outgoing_quadratic_from_L",
        sp.expand(x**2 + lam1 * x + L - (x - r1) * (x - r2)).subs(
            L, (lam1**2 - sep**2) / 4
        ),
    )
    record_identity(
        checks,
        "B_prime_is_twice_offset",
        sp.diff(Bminus, x) - 2 * (x - rstar),
    )
    a = r1 - theta * sep
    record_identity(
        checks,
        "shrinking_wall_value",
        sp.expand(Bminus.subs(x, a) - theta * (1 + theta) * sep**2),
    )
    record_identity(
        checks,
        "slow_line_B_at_origin",
        sp.expand(Bminus.subs(x, 0) - r1 * r2),
    )
    record_identity(
        checks,
        "slow_line_psi_pre_ratio",
        sp.expand(
            Bminus.subs(x, a) * r1 * r2
            - theta * (1 + theta) * sep**2 * Bminus.subs(x, 0)
        ),
    )
    eta = (eps**2 * (-Bx)) / (eps * r1)
    record_identity(
        checks,
        "linearized_saddle_exponent",
        eta - eps * (-Bx) / r1,
    )
    record_identity(
        checks,
        "first_root_eta_leading_term",
        eta.subs(Bx, -sep) - eps * sep / r1,
    )
    return checks


def frozen_gamma_failure() -> dict[str, object]:
    # Concrete instance of the Lean obstruction: C=1, gamma=1, r=1, s0=1.
    C, gamma, r = 1.0, 1.0, 1.0
    sep = min(1.0, r * gamma / 2.0)
    alpha = gamma - sep / r
    kappa = (r * C / sep + 1.0) / alpha
    left = C * math.exp(-gamma * kappa)
    right = (sep / r) * math.exp(-(sep / r) * kappa)
    require(left < right, "frozen gamma must fail on the Lean witness")
    require(sep > 0 and kappa > 0, "positive obstruction coordinates")
    # A still-smaller separation at the same frozen gamma diverges further.
    sep_small = sep / 4.0
    kappa_large = kappa + 8.0
    ratio = ((sep_small / r) * math.exp(-(sep_small / r) * kappa_large)) / (
        C * math.exp(-gamma * kappa_large)
    )
    require(ratio > 10.0, "smaller separation makes the frozen ratio large")
    return {
        "C": C,
        "gamma": gamma,
        "r": r,
        "sep": sep,
        "kappa": kappa,
        "left": left,
        "right": right,
        "smaller_sep_ratio": ratio,
    }


def rejection_checks() -> list[str]:
    rejected: list[str] = []
    try:
        record_identity([], "nonzero_residual", sp.Integer(1))
    except ArithmeticError:
        rejected.append("nonzero_residual")
    try:
        identities_dup = identities()
        record_identity(identities_dup, identities_dup[0], sp.Integer(0))
    except ValueError:
        rejected.append("duplicate_identity")
    require(len(rejected) == 2, "negative self-checks remain active")
    return rejected


def honesty() -> dict[str, bool]:
    return {
        "finite_algebra_replayed": True,
        "linear_chi_identities_replayed": True,
        "frozen_exponent_obstruction_witnessed": True,
        "actual_epsilon_cutoff_certified": False,
        "analytic_passage_theorem_verified": False,
        "analytic_theorem_formally_verified": False,
        "theorem_prover_verified": False,
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "hilbert16-coalescing-capture-finite-replay-v1",
        "seed": SEED,
        "identities": identities(),
        "frozen_gamma_failure": frozen_gamma_failure(),
        "negative_self_checks": rejection_checks(),
        "honesty": honesty(),
        "scope": (
            "Exact linear χ-identities, first-root eigenvalue algebra, "
            "and a concrete frozen-exponent failure. Not G1, G4, or Hilbert XVI."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
