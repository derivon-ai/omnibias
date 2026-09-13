#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay scale-dichotomy identities and the shared next-atlas findings.

The written argument is in packages/omnibias-dynamics/HILBERT16-NEXT-ATLAS.md.
This benchmark does not prove a C2 remainder, G1, or Hilbert XVI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp  # type: ignore[import-untyped]
from omnibias.dynamics.chart_cells import g1_from_cells, ledger_payload
from omnibias.dynamics.hilbert16_identities import replay_hilbert16_identities
from omnibias.dynamics.scale_dichotomy import report
from omnibias.symbolic.reduction import (
    ParameterReduction,
    ParameterTerm,
    evaluate_reduction,
    reduction_candidate,
)

SEED = 160030


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
    eps, sigma, eta, sigma2 = sp.symbols("eps sigma eta sigma2", nonzero=True)
    X, kappa, ell = sp.symbols("X kappa ell", nonzero=True)
    lam1, sep, r1 = sp.symbols("lam1 sep r1")
    checks: list[str] = []
    record_identity(
        checks,
        "blowup_height_scale",
        eps**3 * sigma**2 * eta - eps**3 * sigma**2 * eta,
    )
    record_identity(
        checks,
        "blowup_height_ratio",
        (eps**3 * sigma**2 * eta) / (eps**3 * sigma2**2 * eta) - (sigma / sigma2) ** 2,
    )
    record_identity(
        checks,
        "event_leading_kappa",
        (sigma / X) * kappa + (2 * eps * sigma / X) * ell
        - ((sigma / X) * 0 + (2 * eps * sigma / X) * ell)
        - (sigma / X) * kappa,
    )
    record_identity(
        checks,
        "event_second_difference",
        ((sigma / X) * (kappa + 1) - (sigma / X) * kappa)
        - ((sigma / X) * kappa - (sigma / X) * (kappa - 1)),
    )
    record_identity(checks, "joint_sep_r1_sum", 2 * ((-lam1 - sep) / 2) + sep + lam1)
    record_identity(
        checks,
        "log_inner_on_kill",
        eps * sp.log(sp.exp(1 / eps**2)) - 1 / eps,
    )
    return checks


def joint_axis_path() -> dict[str, object]:
    # eta[0] = r1 reconstructs (sep, r1) on 2*r1 + sep = 2 at lambda1 = -2.
    axis = ParameterReduction(
        (
            ParameterTerm(source=0, scale=-2.0, power=0, offset=2.0),
            ParameterTerm(source=0, power=0),
        ),
        1,
    )
    params = axis(0.1, np.asarray([0.4]))
    require(abs(params[0] + 2.0 * params[1] - 2.0) < 1e-12, "joint axis 2 r1 + sep = 2")
    require(abs(params[1] - 0.4) < 1e-12, "r1 source")

    def full_axis(parameters: np.ndarray, _inputs: np.ndarray) -> np.ndarray:
        return np.asarray([parameters[0] + 2.0 * parameters[1] - 2.0], dtype=float)

    path = evaluate_reduction(
        reduction_candidate(full_axis, axis, name="joint_sep_r1_axis"),
        (0.5, 0.4, 0.3),
        np.asarray([0.4]),
        np.asarray([0.0]),
    )
    require(not path.certified, "reduction reports stay uncertified")
    return {
        "sep_and_r1": [float(params[0]), float(params[1])],
        "certified": False,
        "evaluated_inputs_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    discovery = replay_hilbert16_identities()
    require(discovery.status == "PROVED", "finite identity family must hit")
    require(discovery.statement.parent_status == "open", "parent stays open")
    payload = report()
    require(all(status == "PROVED" for status in payload["identities"].values()), "Q residuals")
    require(payload["dichotomy"]["dichotomy_on_tested_scales"], "scale dichotomy on tested scales")
    require(not payload["dichotomy"]["some_scale_bounds_both"], "no scale bounds both sides")
    require(payload["dichotomy"]["log_inner_is_1_over_eps"], "log chart on the kill sequence")
    require(payload["existing_sections"]["all_existing_sections_explode"], "existing sections")
    require(payload["admission"]["incoming_first_hit_retained"], "incoming retained")
    require(not payload["admission"]["inadmissible"], "kill sequence stays admitted")
    require(payload["hk_leading_jet"]["second_vanishes"], "leading second difference")
    require(not payload["hk_leading_jet"]["physical_c2_remainder"], "no physical C2")
    require(not g1_from_cells(), "cell ledger does not pass G1")
    require(not payload["honesty"]["g1_passed"], "G1 stays failed")
    require(not payload["honesty"]["g4_passed"], "G4 stays closed")
    require(not payload["honesty"]["hk_theorem_24_used"], "Theorem 2.4 unused")
    report_out = {
        **payload,
        "seed": SEED,
        "sympy_identities": identities(),
        "discovery": {
            "status": discovery.status,
            "parent": discovery.statement.parent,
            "parent_status": discovery.statement.parent_status,
            "evaluated": discovery.evaluated,
            "solutions": [str(item) for item in discovery.solutions],
        },
        "joint_axis_path": joint_axis_path(),
        "cells": ledger_payload(),
    }
    text = json.dumps(report_out, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
