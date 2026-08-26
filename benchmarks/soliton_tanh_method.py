# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: tanh-method solitons (theory 02-09).

G2 is ``balance_degree`` vs published ``M``. G3 is ``exact_residual``
on a dense float64 grid. G4 PINN init-win is leftover-recorded
(leftover #22) and stays ``--full``. Algebraic solve / residual cost
is reported, not in CI ``all_passed``. Tanh algebra, not a collapse.
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


def _run_g2() -> dict[str, Any]:
    from omnibias.core.tanh_method import G1_NAMES, balance_degree, classical_pdes, published_ansatz

    pdes = classical_pdes()
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for name in G1_NAMES:
        ans = published_ansatz(name)
        if ans.kind != "tanh_poly":
            rows.append({"name": name, "kind": ans.kind, "skipped": True})
            continue
        got = int(balance_degree(pdes[name]))
        expect = int(ans.degree)
        ok = got == expect
        if not ok:
            mismatches += 1
        rows.append({"name": name, "balance": got, "published_M": expect, "passed": ok})
    passed = mismatches == 0
    return {
        "name": "g2_balance_degree",
        "passed": passed,
        "in_ci_all_passed": passed,
        "mismatches": int(mismatches),
        "rows": rows,
    }


def _run_g3() -> dict[str, Any]:
    import torch
    from omnibias.core.tanh_method import G1_NAMES, classical_pdes, published_ansatz
    from omnibias.pinn.travelling.torch import SolitonField

    torch.set_default_dtype(torch.float64)
    pdes = classical_pdes()
    xs = torch.linspace(-1.5, 1.5, 21, dtype=torch.float64)
    ts = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)
    x = xs.repeat(ts.numel())
    t = ts.repeat_interleave(xs.numel())
    worst_rel = 0.0
    violations = 0
    rows: list[dict[str, Any]] = []
    for name in G1_NAMES:
        ans = published_ansatz(name)
        field = SolitonField((ans,), dtype=torch.float64)
        res = field.exact_residual(x, t, pdes[name])
        mag = float(field(x, t).abs().max().clamp_min(1e-16).detach())
        peak = float(res.abs().max().detach())
        rel = peak / max(mag, 1.0)
        ok = peak <= 1e-14 * max(mag, 1.0)
        if not ok:
            violations += 1
        worst_rel = max(worst_rel, rel)
        rows.append({"name": name, "max_abs": peak, "mag": mag, "rel": rel, "passed": ok})
    passed = violations == 0
    return {
        "name": "g3_numerical_residual",
        "passed": passed,
        "in_ci_all_passed": passed,
        "violations": int(violations),
        "worst_rel": float(worst_rel),
        "n_grid": int(x.numel()),
        "rows": rows,
    }


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
        "leftover_recorded": True,
        "leftover_id": 44,
        "leftover_tick": 88,
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
            "Leftover #44 leftover-recorded: G1 algebraic wall plus "
            "published-vs-cold residual L1. G4 init-win is leftover "
            "#22 and stays --full under $OMNIBIAS_SCRATCH, not CI. Previous "
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
    g2 = _run_g2()
    g3 = _run_g3()
    heat = PDESpec("heat", (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)))
    entries: list[dict[str, Any]] = [
        {"name": "g1_symbolic_exact", "passed": g1, "n": len(G1_NAMES), "in_ci_all_passed": True},
        {
            "name": "g2_balance_degree",
            "passed": bool(g2["passed"]),
            "in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        },
        {
            "name": "g3_numerical_residual",
            "passed": bool(g3["passed"]),
            "in_ci_all_passed": bool(g3["in_ci_all_passed"]),
        },
        {"name": "g5_heat_negative", "passed": solve_ansatz(heat) == (), "in_ci_all_passed": True},
    ]
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.soliton_tanh_method.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g3", "g5"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g2"] = g2
    payload["g3"] = g3
    payload["cost"] = cost
    payload["honesty"] = {
        "algebra": "tanh polynomial, not a collapse",
        "n_soliton": False,
        "temperature_collapse": False,
        "founding_bias_collapse": False,
        "g2_earned": bool(g2["passed"]),
        "g3_earned": bool(g3["passed"]),
        "g4_init_win_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 22,
        "g4_leftover_tick": 64,
        "g4_stays_full": True,
        "cost_earned": False,
        "cost_reported": True,
        "cost_leftover_recorded": True,
        "cost_leftover_id": 44,
        "cost_leftover_tick": 88,
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
