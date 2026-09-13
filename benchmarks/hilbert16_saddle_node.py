#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay double-root identities and the fold-scale tension.

The written argument is in packages/omnibias-dynamics/HILBERT16-SADDLE-NODE.md.
This benchmark does not prove a C2 remainder, G1, or Hilbert XVI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp  # type: ignore[import-untyped]

SEED = 160028


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
    rstar, x, L, lam1, sep, sigma, chi, r1, kappa = sp.symbols(
        "rstar x L lam1 sep sigma chi r1 kappa", nonzero=True
    )
    checks: list[str] = []
    record_identity(
        checks,
        "double_root_quadratic",
        sp.expand(rstar**2 + (-2 * rstar) * x + x**2 - (x - rstar) ** 2),
    )
    record_identity(checks, "wall_at_equilibrium", (rstar - rstar) ** 2)
    record_identity(
        checks,
        "vanishing_sep_linear_exit",
        sp.exp(sp.Integer(0)) - 1,
    )
    record_identity(
        checks,
        "chi_of_zero_sep",
        (sp.Integer(0) / r1) * kappa,
    )
    record_identity(
        checks,
        "sigma_kappa_on_chi_locus",
        sigma * (chi * r1 / sep) - (sigma / sep) * chi * r1,
    )
    record_identity(
        checks,
        "first_root_product",
        sp.expand((-lam1 - sep) * (-lam1 + sep) - (lam1**2 - sep**2)),
    )
    record_identity(
        checks,
        "first_root_from_L",
        sp.expand((-lam1 - sep) / 2 * (-lam1 + sep) - 2 * L).subs(
            L, (lam1**2 - sep**2) / 4
        ),
    )
    record_identity(
        checks,
        "r1_linear_in_L",
        (((-lam1 - sep) / 2) / L - 2 / (-lam1 + sep)).subs(
            L, (lam1**2 - sep**2) / 4
        ),
    )
    eps_pos = sp.symbols("eps_pos", positive=True)
    record_identity(
        checks,
        "fold_exit_height_independent_of_sep",
        eps_pos**3 * (sp.sqrt(eps_pos) ** 2) - eps_pos**4,
    )
    return checks


def scale_tension_witness() -> dict[str, object]:
    eps = 0.05
    sep = math.exp(-1.0 / (eps * eps))
    kappa = 1.0 / sep
    chi, r1 = 1.0, 1.0
    fold = math.sqrt(eps)
    sep_scale = sep
    fold_product = fold * kappa
    sep_product = sep_scale * kappa
    # Evaluate epsilon |log h_e| from the exponents; sep^2 underflows in float64.
    outgoing_fold = abs(-4.0 * math.log(eps)) * eps
    outgoing_sep = abs(-3.0 * math.log(eps) - 2.0 * math.log(sep)) * eps
    require(fold_product > 10.0, "fold scale times kappa unbounded on kill sequence")
    require(math.isclose(sep_product, 1.0), "separation scale keeps sigma*kappa of order one")
    require(outgoing_sep > outgoing_fold + 1.0, "sep-scale outgoing log factor larger")
    require(not (fold_product <= 2.0 and outgoing_sep <= 1.0),
            "no single scale makes both products small")
    return {
        "epsilon": eps,
        "sep": sep,
        "kappa": kappa,
        "chi": chi,
        "r1": r1,
        "fold_sigma_kappa": fold_product,
        "sep_sigma_kappa": sep_product,
        "outgoing_eps_log_fold": outgoing_fold,
        "outgoing_eps_log_sep": outgoing_sep,
    }


def rejection_checks() -> list[str]:
    rejected: list[str] = []
    try:
        record_identity([], "nonzero_residual", sp.Integer(1))
    except ArithmeticError:
        rejected.append("nonzero_residual")
    try:
        names = identities()
        record_identity(names, names[0], sp.Integer(0))
    except ValueError:
        rejected.append("duplicate_identity")
    require(len(rejected) == 2, "negative self-checks remain active")
    return rejected


def honesty() -> dict[str, bool]:
    return {
        "finite_algebra_replayed": True,
        "double_root_identities_replayed": True,
        "scale_tension_witnessed": True,
        "saddle_node_c2_remainder": False,
        "super_small_sep_absorbed": False,
        "actual_epsilon_cutoff_certified": False,
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
        "schema": "hilbert16-saddle-node-finite-replay-v1",
        "seed": SEED,
        "identities": identities(),
        "scale_tension": scale_tension_witness(),
        "negative_self_checks": rejection_checks(),
        "honesty": honesty(),
        "scope": (
            "Exact double-root algebra and a concrete fold-versus-separation "
            "scale tension. Not a C2 remainder, G1, or Hilbert XVI."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
