# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: tanh-method solitons (theory 02-09).

G4 PINN init-win is leftover-recorded (leftover #22) and stays
``--full``. Algebraic solve / residual cost is reported, not in CI
``all_passed``. Tanh algebra, not a collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

COST_WARMUP = 1
COST_REPEATS = 3
COST_NAMES = ("burgers", "kdv", "mkdv")


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _coeff_l1(coeffs: tuple[Any, ...]) -> float:
    return float(sum(abs(float(c)) for c in coeffs))


def _run_cost() -> dict[str, Any]:
    """Named leftover: algebraic solve wall; G4 PINN init-win stays --full."""
    from omnibias.core.tanh_method import (
        G1_NAMES,
        TravellingWaveAnsatz,
        classical_pdes,
        published_ansatz,
        solve_ansatz,
        substitute,
        verify_exact,
    )

    pdes = classical_pdes()

    def _verify_all() -> None:
        for name in G1_NAMES:
            verify_exact(pdes[name], published_ansatz(name))

    def _solve_all() -> None:
        for name in G1_NAMES:
            solve_ansatz(pdes[name])

    verify_wall = _median_seconds(_verify_all, warmup=COST_WARMUP, repeats=COST_REPEATS)
    solve_wall = _median_seconds(_solve_all, warmup=COST_WARMUP, repeats=COST_REPEATS)
    rows: list[dict[str, Any]] = []
    for name in COST_NAMES:
        pub = published_ansatz(name)
        pub_l1 = _coeff_l1(substitute(pdes[name], pub))
        cold_coeffs = tuple(Fraction(1) for _ in range(pub.degree + 1))
        cold = TravellingWaveAnsatz(pub.degree, cold_coeffs, pub.wavenumber, pub.frequency)
        cold_l1 = _coeff_l1(substitute(pdes[name], cold))
        rows.append(
            {
                "name": name,
                "published_residual_l1": pub_l1,
                "cold_residual_l1": cold_l1,
            }
        )
    return {
        "name": "cost_algebraic_vs_init_win",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "verify_exact_wall_seconds": float(verify_wall),
        "solve_ansatz_wall_seconds": float(solve_wall),
        "n_g1": len(G1_NAMES),
        "rows": rows,
        "g4_init_win": {
            "earned": False,
            "reported": True,
            "leftover_recorded": True,
            "leftover_id": 22,
            "leftover_tick": 64,
            "stays_full": True,
            "need": "5x fewer PINN steps vs cold start on a perturbed problem, 5 seeds",
            "reason": (
                "Leftover #22 leftover-recorded: no PINN training loop "
                "is wired. Algebraic verify_exact / solve_ansatz and "
                "residual L1 vs a same-degree cold ansatz are recorded; "
                "they are not the named 5x step-count win."
            ),
        },
        "note": (
            "Leftover #22 leftover-recorded: G1 algebraic wall plus "
            "published-vs-cold residual L1. G4 init-win is a 5-seed "
            "PINN study under $OMNIBIAS_SCRATCH, not CI. Previous "
            "g4_init_win passed=True / --full-only stub with no timing "
            "withdrawn. Tanh algebra, not a collapse. Not in CI "
            "all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.tanh_method import (
        G1_NAMES,
        PDESpec,
        PDETerm,
        TermKind,
        classical_pdes,
        published_ansatz,
        solve_ansatz,
        verify_exact,
    )

    pdes = classical_pdes()
    g1 = all(verify_exact(pdes[n], published_ansatz(n)) for n in G1_NAMES)
    heat = PDESpec("heat", (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)))
    entries: list[dict[str, Any]] = [
        {"name": "g1_symbolic_exact", "passed": g1, "n": len(G1_NAMES), "in_ci_all_passed": True},
        {"name": "g5_heat_negative", "passed": solve_ansatz(heat) == (), "in_ci_all_passed": True},
    ]
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.soliton_tanh_method.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g5"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["honesty"] = {
        "algebra": "tanh polynomial, not a collapse",
        "n_soliton": False,
        "temperature_collapse": False,
        "founding_bias_collapse": False,
        "g4_init_win_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 22,
        "g4_leftover_tick": 64,
        "g4_stays_full": True,
        "cost_earned": False,
        "cost_reported": True,
        "cost_in_ci_all_passed": False,
    }
    if args.full:
        dest = SCRATCH / "soliton" / "soliton_tanh_method.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('soliton_tanh_method_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
