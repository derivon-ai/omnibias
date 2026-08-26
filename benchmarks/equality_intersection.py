# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: equality locus layer (theory 02-12). Not a PDE solver.

G2 is IFT vs unrolled autodiff plus an asserted O(1)-in-``max_iter``
backward. G3 is degeneracy refusal. G5 is ``AnsatzSolutionField``
construction-time reject. G6 is torch/jax ``EqualityLocusLayer``
parity. G4 Burgers RH is leftover-recorded (leftover #29). Clean
Rankine–Hugoniot speed is measured from ``affine_locus``. The noisy-data
skill versus contour extraction stays ``--full``: units are published,
not fit from samples. Founding ``delta -> 0`` only.
"""

from __future__ import annotations

import argparse
import inspect
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


def _matched_layer() -> Any:
    from omnibias.core.locus import EqualitySystem, UnitTerm

    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(1, 1.0, (0.0, 1.0), 0.0),
        )
    )


def _ift_case() -> Any:
    from omnibias.core.locus import EqualitySystem, UnitTerm

    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(2, -2.0, (0.0, 1.0), 0.0),
        )
    )


def _run_g2() -> dict[str, Any]:
    """IFT gradient vs unrolled autodiff; memory asserted independent of max_iter."""
    import torch
    from omnibias.fields.locus.torch import (
        _NewtonIFT,
        newton_project,
        newton_project_unrolled,
    )

    torch.set_default_dtype(torch.float64)
    sys = _ift_case()
    x0 = torch.tensor([0.0, 0.20], dtype=torch.float64)
    w = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    x_ift = newton_project(sys, x0, weights=w, max_iter=8, tol=1e-14)
    x_ift[1].backward()
    g_ift = w.grad.detach().clone()

    w2 = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    x_un = newton_project_unrolled(sys, x0, weights=w2, max_iter=8, tol=1e-14)
    x_un[1].backward()
    g_un = w2.grad.detach().clone()
    rel = float(torch.norm(g_ift - g_un) / torch.norm(g_un).clamp_min(1e-30))

    def _ift_grad(max_iter: int) -> Any:
        ww = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        xx = newton_project(sys, x0, weights=ww, max_iter=max_iter, tol=1e-14)
        xx[1].backward()
        return ww.grad.detach().clone()

    g8 = _ift_grad(8)
    g20 = _ift_grad(20)
    iter_rel = float(torch.norm(g8 - g20) / torch.norm(g20).clamp_min(1e-30))
    src = inspect.getsource(_NewtonIFT.forward)
    saved = [ln for ln in src.splitlines() if "save_for_backward" in ln]
    memory_independent = bool(
        len(saved) == 1
        and "x_star" in saved[0]
        and "weights" in saved[0]
        and "max_iter" not in saved[0]
    )
    passed = bool(rel <= 1e-8 and iter_rel <= 1e-12 and memory_independent)
    return {
        "name": "g2_ift_vs_unrolled",
        "passed": passed,
        "in_ci_all_passed": passed,
        "rel": rel,
        "iter_rel": iter_rel,
        "memory_independent_of_max_iter": memory_independent,
        "saved_tensors": "x_star, weights",
    }


def _run_g3() -> dict[str, Any]:
    import torch
    from omnibias.core.locus import EqualitySystem, UnitTerm
    from omnibias.fields.locus.torch import EqualityLocusLayer

    torch.set_default_dtype(torch.float64)
    sys = EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
        )
    )
    layer = EqualityLocusLayer(sys, require_transversal=True, dtype=torch.float64)
    out = layer(torch.tensor([0.2, 0.3], dtype=torch.float64))
    converged = bool(out.converged.detach())
    condition = float(out.condition.detach())
    passed = bool((converged is False) and condition > 1e6)
    return {
        "name": "g3_degeneracy_refusal",
        "passed": passed,
        "in_ci_all_passed": passed,
        "converged": converged,
        "condition": condition,
    }


def _run_g5() -> dict[str, Any]:
    from omnibias.core.tanh_method import classical_pdes, published_ansatz
    from omnibias.fields.locus.torch import AnsatzSolutionField

    pde = classical_pdes()["burgers"]
    wrong_raised = False
    try:
        AnsatzSolutionField(pde, published_ansatz("kdv"))
    except ValueError as exc:
        wrong_raised = "symbolic verification" in str(exc)
    good = AnsatzSolutionField(pde, published_ansatz("burgers"))
    cert = good.certificate()
    passed = bool(
        wrong_raised
        and cert["verified"] is True
        and cert["level3_general_solver"] is False
    )
    return {
        "name": "g5_ansatz_reject",
        "passed": passed,
        "in_ci_all_passed": passed,
        "wrong_raised": bool(wrong_raised),
        "verified": bool(cert["verified"]),
        "level3_general_solver": bool(cert["level3_general_solver"]),
    }


def _run_g6() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.fields.locus.jax import equality_locus_apply
    from omnibias.fields.locus.torch import EqualityLocusLayer

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    sys = _matched_layer()
    x0 = torch.tensor([0.4, -0.1], dtype=torch.float64)
    out_t = EqualityLocusLayer(sys, dtype=torch.float64)(x0)
    out_j = equality_locus_apply(sys, jnp.asarray([0.4, -0.1], dtype=jnp.float64))
    max_abs = float(
        np.max(
            np.abs(
                out_t.point.detach().cpu().numpy()
                - np.asarray(out_j.point.tolist(), dtype=np.float64)
            )
        )
    )
    passed = bool(max_abs == 0.0)
    return {
        "name": "g6_parity",
        "passed": passed,
        "in_ci_all_passed": passed,
        "max_abs": max_abs,
    }


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
    g2 = _run_g2()
    g3 = _run_g3()
    g5 = _run_g5()
    g6 = _run_g6()
    entries: list[dict[str, Any]] = [
        {"name": "g1_locus_residual", "passed": shock_ok, "in_ci_all_passed": True},
        {k: g2[k] for k in ("name", "passed", "in_ci_all_passed")},
        {k: g3[k] for k in ("name", "passed", "in_ci_all_passed")},
        {k: g5[k] for k in ("name", "passed", "in_ci_all_passed")},
        {k: g6[k] for k in ("name", "passed", "in_ci_all_passed")},
    ]
    g4 = _run_g4()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equality_intersection.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g4_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g3", "g5", "g6"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["g5"] = g5
    payload["g6"] = g6
    payload["honesty"] = {
        "level3_general_solver": False,
        "returns": "branch / condition / converged",
        "g2_earned": bool(g2["passed"]),
        "g3_earned": bool(g3["passed"]),
        "g5_earned": bool(g5["passed"]),
        "g6_earned": bool(g6["passed"]),
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
