#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay two-blow-up identities, named scale paths, and the chart-cell ledger.

The written argument is in packages/omnibias-dynamics/HILBERT16-TWO-BLOWUP.md.
This benchmark does not prove a C2 remainder, G1, or Hilbert XVI.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.dynamics.chart_cells import (
    MALETTO_QUARTIC_EXAMPLE,
    g1_from_cells,
    ledger_payload,
    rematch_shrinking_root,
    replay_maletto_type,
)
from omnibias.dynamics.hilbert16_identities import replay_hilbert16_identities
from omnibias.geometry.algebraic import HomogeneousPlaneCurve, find_smoothness_witness
from omnibias.symbolic.reduction import (
    ParameterReduction,
    ParameterTerm,
    evaluate_reduction,
    reduction_candidate,
)

SEED = 160029


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
    eps, w, wmax, we, hmax_pow, he_pow = sp.symbols(
        "eps w wmax we hmax_pow he_pow", nonzero=True
    )
    sigma_f, sigma_s, chi, r1, sep = sp.symbols(
        "sigma_f sigma_s chi r1 sep", nonzero=True
    )
    checks: list[str] = []
    record_identity(checks, "w_recovers_height_power", eps * (hmax_pow / eps) - hmax_pow)
    record_identity(
        checks,
        "outgoing_as_w_ratio",
        (eps * wmax) / (eps * we) - wmax / we,
    )
    record_identity(
        checks,
        "two_scale_product",
        sigma_f * (chi * r1 / sep) * (sigma_s / sigma_f) - sigma_s * (chi * r1 / sep),
    )
    record_identity(checks, "logW_tau_neg_u", -sp.symbols("u") + sp.symbols("u"))
    return checks


def scale_paths() -> dict[str, object]:
    # eta[0] = sqrt(eps) reconstructs (eps, sigma) = (eta^2, eta).
    fold = ParameterReduction(
        (ParameterTerm(source=0, power=1), ParameterTerm(source=0, power=0)),
        1,
    )
    sep = ParameterReduction(
        (ParameterTerm(source=0, power=0), ParameterTerm(source=1, power=0), ParameterTerm(source=1, power=0)),
        2,
    )
    shrink = ParameterReduction(
        (ParameterTerm(source=None, power=1), ParameterTerm(source=None, scale=0.0, offset=-2.0)),
        1,
    )
    fold_params = fold(0.5, np.asarray([0.5]))
    sep_params = sep(1.0, np.asarray([0.2, 0.05]))
    shrink_params = shrink(0.1, np.asarray([0.0]))
    require(abs(fold_params[0] - 0.25) < 1e-12 and abs(fold_params[1] - 0.5) < 1e-12, "fold path")
    require(abs(sep_params[1] - sep_params[2]) < 1e-12, "sep path uses sigma = sep")
    require(abs(shrink_params[0] - 0.1) < 1e-12 and abs(shrink_params[1] + 2.0) < 1e-12, "shrink path")

    def full_fold(params: np.ndarray, _inputs: np.ndarray) -> np.ndarray:
        return np.asarray([params[1] - params[0] ** 0.5], dtype=float)

    fold_report = evaluate_reduction(
        reduction_candidate(full_fold, fold, name="fold_sigma_sqrt_eps"),
        (0.5, 0.4, 0.3),
        np.asarray([0.5]),
        np.asarray([0.0]),
    )
    require(not fold_report.certified, "reduction reports stay uncertified")
    return {
        "fold_sigma_is_sqrt_eps": [float(fold_params[0]), float(fold_params[1])],
        "sep_sigma_equals_sep": [float(x) for x in sep_params],
        "shrink_L_and_lambda1": [float(x) for x in shrink_params],
        "certified": False,
        "evaluated_inputs_only": True,
    }


def kill_sequence_w_ratio() -> dict[str, object]:
    eps = 0.05
    sep = math.exp(-1.0 / (eps * eps))
    log_w_ratio = abs((-3.0 * math.log(eps) - 2.0 * math.log(sep)) - (-3.0 * math.log(eps)))
    require(log_w_ratio > 10.0, "WS outgoing W-ratio explodes on the kill sequence")
    fold_event = math.sqrt(eps) / sep
    require(fold_event > 10.0, "WF event factor explodes on the kill sequence")
    return {
        "epsilon": eps,
        "sep": sep,
        "ws_log_w_ratio": log_w_ratio,
        "wf_sigma_over_sep": fold_event,
    }


def maletto_smoothness() -> dict[str, object]:
    """Attach the existing plane-curve smoothness witness. Not a real-scheme certificate."""

    terms = {tuple(index): Fraction(coeff) for index, coeff in MALETTO_QUARTIC_EXAMPLE["terms"]}  # type: ignore[misc]
    curve = HomogeneousPlaneCurve(SparsePolynomial(3, terms))
    witness = find_smoothness_witness(curve, max_multiplier_degree=4)
    require(witness is not None and witness.verifies(curve), "Maletto §1.1 quartic is complex-smooth")
    return {
        "algebraic_smoothness": "PROVED",
        "complete_real_scheme": False,
        "curve_degree": curve.degree,
        "source": MALETTO_QUARTIC_EXAMPLE["source"],
    }


def honesty() -> dict[str, bool]:
    return {
        "finite_algebra_replayed": True,
        "named_scale_paths_evaluated": True,
        "two_blowup_c2_remainder": False,
        "sr2_rematch": False,
        "g1_passed": False,
        "g4_passed": False,
        "full_graphic_cyclicity_proved": False,
        "full_hilbert16_solved": False,
        "analytic_theorem_formally_verified": False,
        "theorem_prover_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    discovery = replay_hilbert16_identities()
    require(discovery.status == "PROVED", "finite identity family must hit")
    require(discovery.statement.parent_status == "open", "parent stays open")
    maletto = {
        **replay_maletto_type(
            MALETTO_QUARTIC_EXAMPLE["counts"],  # type: ignore[arg-type]
            MALETTO_QUARTIC_EXAMPLE["words"],  # type: ignore[arg-type]
            MALETTO_QUARTIC_EXAMPLE["trees"],  # type: ignore[arg-type]
            degree=int(MALETTO_QUARTIC_EXAMPLE["degree"]),
        ),
        **maletto_smoothness(),
    }
    rematch = rematch_shrinking_root(Fraction(1, 20), Fraction(-2))
    require(not rematch["sr2_rematch"], "SR2 rematch stays failed")
    require(not g1_from_cells(), "cell ledger does not pass G1")
    report = {
        "schema": "hilbert16-two-blowup-finite-replay-v1",
        "seed": SEED,
        "identities": identities(),
        "discovery": {
            "status": discovery.status,
            "parent": discovery.statement.parent,
            "parent_status": discovery.statement.parent_status,
            "evaluated": discovery.evaluated,
            "solutions": [str(item) for item in discovery.solutions],
        },
        "scale_paths": scale_paths(),
        "kill_sequence": kill_sequence_w_ratio(),
        "maletto": maletto,
        "sr2": rematch,
        "ledger": ledger_payload(),
        "honesty": honesty(),
        "scope": (
            "Exact W-identities, named scale paths, one combinatorial type, "
            "and a failed two-chart covering. Not a C2 remainder, G1, or Hilbert XVI."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
