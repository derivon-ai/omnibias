# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: equality locus layer (theory 02-12). Not a PDE solver.

G4 Burgers RH is leftover-recorded (leftover #29). Clean
Rankine–Hugoniot speed is measured from ``affine_locus``. The noisy-data
skill versus contour extraction stays ``--full``: units are published,
not fit from samples. Founding ``delta -> 0`` only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

RH_SPEED = 0.5
RH_TOL = 1e-10
NOISE_SIGMA = 0.15
N_SEEDS = 5
N_TIMES = 8


def _burgers_units() -> Any:
    from omnibias.core.locus import EqualitySystem, UnitTerm

    # Spec 02-12: w_L = (1, -1), w_R = (1, 0) => mirror branch x = 0.5 t.
    return EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, -1.0), 0.0), UnitTerm(1, 1.0, (1.0, 0.0), 0.0))
    )


def _recovered_speed(planes: Any) -> float | None:
    for plane in planes:
        if int(plane.branch) != -1:
            continue
        nx, nt = float(plane.normal[0]), float(plane.normal[1])
        if abs(nx) <= 1e-15:
            continue
        # n . (x, t) + offset = 0 => x = -(nt / nx) t - offset / nx
        if abs(float(plane.offset)) > 1e-15:
            continue
        return float(-nt / nx)
    return None


def _contour_shock_x(xs: np.ndarray, u: np.ndarray, *, mid: float = 0.5) -> float:
    for i in range(int(xs.shape[0]) - 1):
        left, right = float(u[i]), float(u[i + 1])
        if (left - mid) * (right - mid) <= 0.0:
            du = right - left
            if abs(du) <= 1e-18:
                return float(xs[i])
            w = (mid - left) / du
            return float(xs[i] + w * (xs[i + 1] - xs[i]))
    return float(xs[int(np.argmin(np.abs(u - mid)))])


def _run_g4() -> dict[str, Any]:
    """Named leftover: clean RH from affine_locus; noisy contour stays --full."""
    from omnibias.core.locus import affine_locus, residual

    sys = _burgers_units()
    planes = affine_locus(sys)
    recovered = _recovered_speed(planes) if planes is not None else None
    rh_err = float("inf") if recovered is None else abs(float(recovered) - RH_SPEED)
    rh_match = bool(recovered is not None and rh_err <= RH_TOL)
    residual_ok = abs(residual(sys, (RH_SPEED, 1.0))[0]) <= 1e-12

    wins = 0
    skills: list[float] = []
    rows: list[dict[str, Any]] = []
    times = np.linspace(0.25, 2.0, N_TIMES)
    xs = np.linspace(-1.0, 2.0, 81)
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        pred_loc: list[float] = []
        pred_con: list[float] = []
        truth: list[float] = []
        for t in times:
            tt = float(t)
            x_true = RH_SPEED * tt
            u = np.where(xs < x_true, 1.0, 0.0) + NOISE_SIGMA * rng.normal(size=xs.shape)
            pred_loc.append(float(recovered) * tt if recovered is not None else 0.0)
            pred_con.append(_contour_shock_x(xs, u))
            truth.append(x_true)
        y = np.asarray(truth, dtype=np.float64)
        loc = np.asarray(pred_loc, dtype=np.float64)
        con = np.asarray(pred_con, dtype=np.float64)
        mse_loc = float(np.mean((loc - y) ** 2))
        mse_con = float(np.mean((con - y) ** 2))
        mse_zero = float(np.mean(y**2))
        skill = 1.0 - mse_loc / max(mse_zero, 1e-18)
        skills.append(skill)
        win = bool(mse_loc < mse_con)
        wins += int(win)
        rows.append(
            {
                "seed": int(seed),
                "mse_locus": mse_loc,
                "mse_contour": mse_con,
                "skill_vs_zero": skill,
                "locus_wins": win,
            }
        )
    return {
        "name": "g4_burgers_rh",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 29,
        "leftover_tick": 69,
        "in_ci_all_passed": False,
        "rh_match": rh_match,
        "rh_abs_err": float(rh_err),
        "rh_tol": RH_TOL,
        "recovered_speed": None if recovered is None else float(recovered),
        "residual_at_shock": residual_ok,
        "wins": int(wins),
        "n_seeds": N_SEEDS,
        "median_skill_vs_zero": float(np.median(np.asarray(skills))),
        "rows": rows,
        "units_fit_from_data": False,
        "stays_full": True,
        "note": (
            "Leftover #29 leftover-recorded: clean Rankine-Hugoniot "
            "speed from affine_locus on the published Burgers units "
            "(need |s-0.5|<=1e-10). Noisy contour extraction is "
            "measured on five seeds; the locus uses those published "
            "units, not a fit from samples. Named G4 needs a data-fit "
            "that beats contour with skill > 0. Previous "
            "g4_burgers_rh passed=True smoke-geometry stub withdrawn. "
            "Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.locus import EqualitySystem, UnitTerm, affine_locus, residual

    sys_eq = EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, -0.5), 0.0), UnitTerm(1, 1.0, (1.0, 0.0), 0.0))
    )
    planes = affine_locus(sys_eq)
    # Mirror branch 2x - 0.5 t = 0 => x = 0.25 t  (RH for c=0.5 would be 0.25)
    shock_ok = False
    if planes is not None:
        for pt in ((0.25, 1.0), (0.5, 2.0)):
            if abs(residual(sys_eq, pt)[0]) <= 1e-12:
                shock_ok = True
                break
    entries: list[dict[str, Any]] = [
        {"name": "g1_locus_residual", "passed": shock_ok, "in_ci_all_passed": True},
    ]
    g4 = _run_g4()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equality_intersection.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g4_in_all_passed": False,
            "gates_in_scope": ["g1"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g4"] = g4
    payload["honesty"] = {
        "level3_general_solver": False,
        "returns": "branch / condition / converged",
        "g4_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 29,
        "g4_leftover_tick": 69,
        "g4_in_ci_all_passed": False,
        "g4_units_fit_from_data": False,
        "temperature_collapse": False,
        "founding_bias_collapse": True,
    }
    if args.full:
        dest = SCRATCH / "locus" / "equality_intersection.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('equality_intersection_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
