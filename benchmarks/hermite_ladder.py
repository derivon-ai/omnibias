# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: Hermite ladder (theory 02-10).

G4 many-body FermiNet variance is leftover-recorded (leftover #21) and
stays ``--full``. Exact-ladder orbital cost is reported, not in CI
``all_passed``. G5 anharmonic lose/win is leftover-recorded (leftover
#26). The raw tower is not the QHO eigenbasis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

COST_NS = (2, 4, 8)
COST_N_POINTS = 64
COST_WARMUP = 1
COST_REPEATS = 3
COST_FD_H = 1e-6


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_cost() -> dict[str, Any]:
    """Named leftover: exact-ladder orbital wall vs FD; G4 many-body stays --full."""
    from omnibias.ferminet.hermite import apply_ladder, oscillator_phi, vmc_comparison_report

    xs = np.linspace(-2.0, 2.0, COST_N_POINTS)
    rows: list[dict[str, Any]] = []
    for n in COST_NS:

        def _exact(order=n, grid=xs) -> None:
            for x in grid:
                apply_ladder(order, float(x), "lower", normalization="oscillator")

        def _fd(order=n, grid=xs) -> None:
            h = COST_FD_H
            for x in grid:
                xf = float(x)
                (
                    oscillator_phi(order, xf + h) - oscillator_phi(order, xf - h)
                ) / (2.0 * h)

        exact_wall = _median_seconds(_exact, warmup=COST_WARMUP, repeats=COST_REPEATS)
        fd_wall = _median_seconds(_fd, warmup=COST_WARMUP, repeats=COST_REPEATS)
        rows.append(
            {
                "n": int(n),
                "n_points": COST_N_POINTS,
                "exact_wall_seconds": float(exact_wall),
                "fd_wall_seconds": float(fd_wall),
                "fd_over_exact": float(fd_wall / max(exact_wall, 1e-18)),
            }
        )
    ratios = [float(row["fd_over_exact"]) for row in rows]
    vmc = vmc_comparison_report(seeds=5)
    return {
        "name": "cost_exact_vs_fd_orbitals",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "rows": rows,
        "median_fd_over_exact": float(np.median(np.asarray(ratios))),
        "g4_many_body": {
            "earned": False,
            "reported": True,
            "leftover_recorded": True,
            "leftover_id": 21,
            "leftover_tick": 65,
            "stays_full": True,
            "need": "2x variational-energy variance on a small FermiNet system, 5 seeds",
            "one_d_no_improvement": bool(vmc["no_improvement"]),
            "reason": (
                "Leftover #21 leftover-recorded: "
                + str(vmc["reason"])
            ),
        },
        "note": (
            "Leftover #21 leftover-recorded: exact apply_ladder orbital "
            "derivatives vs central FD on the named 1-D oscillator. G4 "
            "many-body 2x variance is a FermiNet run under "
            "$OMNIBIAS_SCRATCH, not CI. 1-D QHO envelope already contains "
            "the ground state (no improvement). Previous G4 passed=True "
            "/ --full-only stub with no timing withdrawn. Not in CI "
            "all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    """Named leftover: anharmonic Rayleigh vs FD grid; lose is allowed."""
    from omnibias.ferminet.hermite import oscillator_phi

    lam = 1.0
    xs = np.linspace(-4.0, 4.0, 81)
    dx = float(xs[1] - xs[0])
    phi = np.asarray([oscillator_phi(0, float(x)) for x in xs], dtype=np.float64)
    # H_qho phi_0 = (1/2) phi_0; H_anh = H_qho + λ x^4.
    hphi = 0.5 * phi + lam * (xs**4) * phi
    osc_e = float(np.trapezoid(hphi * phi, xs) / np.trapezoid(phi * phi, xs))
    pot = 0.5 * xs**2 + lam * xs**4
    ham = np.diag(pot + 1.0 / (dx * dx))
    off = -0.5 / (dx * dx)
    idx = np.arange(xs.size - 1)
    ham[idx, idx + 1] = off
    ham[idx + 1, idx] = off
    grid_e = float(np.linalg.eigvalsh(ham)[0])
    lost = bool(grid_e < osc_e)
    return {
        "name": "g5_anharmonic",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "lambda_x4": lam,
        "n_grid": int(xs.size),
        "oscillator_rayleigh": osc_e,
        "fd_grid_ground": grid_e,
        "lost_to_grid": lost,
        "leftover_recorded": True,
        "leftover_id": 26,
        "leftover_tick": 66,
        "note": (
            "Leftover #26 leftover-recorded: strongly anharmonic well "
            "V = x^2/2 + x^4. Oscillator ground Rayleigh versus a "
            "Dirichlet FD grid on the same box. G5 is honesty: the "
            "basis is allowed to lose. Previous passed=True stub with "
            "no measured lose/win withdrawn. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.ladder import (
        Normalization,
        hermite_function,
        number_operator_apply,
        tower_raise,
    )

    x = 0.7
    g1 = abs(tower_raise(3, x) - hermite_function(4, x, normalization=Normalization.TOWER)) < 1e-12
    h = hermite_function(3, x, normalization=Normalization.TOWER)
    g2 = abs(number_operator_apply(3, x) - 3 * h) < 1e-12
    entries: list[dict[str, Any]] = [
        {"name": "g1_raise", "passed": g1, "in_ci_all_passed": True},
        {"name": "g2_number", "passed": g2, "in_ci_all_passed": True},
    ]
    cost = _run_cost()
    g5 = _run_g5()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.hermite_ladder.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "g5_in_all_passed": False,
            "gates_in_scope": ["g1", "g2"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["g5"] = g5
    payload["honesty"] = {
        "raw_tower_is_qho": False,
        "rodrigues_required": True,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "g4_many_body_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 21,
        "g4_leftover_tick": 65,
        "g4_stays_full": True,
        "g5_anharmonic_earned": False,
        "g5_anharmonic_reported": True,
        "g5_leftover_recorded": True,
        "g5_leftover_id": 26,
        "g5_leftover_tick": 66,
        "g5_in_ci_all_passed": False,
        "cost_earned": False,
        "cost_reported": True,
        "cost_in_ci_all_passed": False,
        "many_body_solution_claim": False,
    }
    if args.full:
        dest = SCRATCH / "ladder" / "hermite_ladder.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('hermite_ladder_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
