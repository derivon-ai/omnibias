# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Calibrate physics weight / FD step for DeepONet residual bake-offs.

Short-budget sweeps (stated *before* the run) that pick the knobs for
``operator_deeponet.py`` and ``operator_ks_bakeoff.py`` from measured curves,
not from guesses:

* Burgers: physics weight ``lambda`` in ``{0.1, 1, 10}`` at coarse ``n_times``,
  then ``n_times`` in ``{11, 6, 3}`` at the winning ``lambda``.
* KS: stencil step ``h`` in ``{1e-1, 1e-2, 1e-3}`` at ``lambda=1``.

Decision rule: pick the config that maximises ``median(B) - median(C)`` among
configs where every arm clears the train-rel-L2 guard; if none separate, pick
the config with the largest absolute gap and record that the gap is negative.

Run::

    uv run python benchmarks/operator_residual_calibration.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.operator.torch import (  # noqa: E402
    build_deeponet,
    burgers_residual_loss,
    data_loss,
    ks_residual_loss,
    ks_residual_loss_fd,
    make_burgers_slab,
    make_ks_slab,
)
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(int(s) for s in os.environ.get("OP_CAL_SEEDS", "0,1,2").split(","))
STEPS = int(os.environ.get("OP_CAL_STEPS", "500"))
OUT_NAME = os.environ.get("OP_CAL_OUT", "operator_residual_calibration.json")
TRAIN_REL_L2_MAX = 0.5  # looser than final bake-off; short budget
LAMBDAS = (0.1, 1.0, 10.0)
N_TIMES_LIST = (11, 6, 3)
KS_HS = (1e-1, 1e-2, 1e-3)
VISCOSITY = 0.05


def _rel_l2(op: Any, slab: Any) -> float:
    with torch.no_grad():
        field = op.condition(slab.sensors)
        state = field.on_grid(slab.coords)
        pred = tops.value(state, "u").reshape(slab.values.shape[0], -1)
        target = slab.values[..., 0]
        return float(
            (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()
        )


def _burgers_fd_residual(op: Any, slab: Any, dt: float) -> torch.Tensor:
    field = op.condition(slab.sensors)
    state = field.on_grid(slab.coords)
    u = tops.value(state, "u")
    u_x = tops.derivative(state, "u", axis=0, order=1)
    u_xx = tops.derivative(state, "u", axis=0, order=2)
    n_x = int(torch.unique(slab.coords[:, 0]).numel())
    n_t = int(torch.unique(slab.coords[:, 1]).numel())
    F = slab.sensors.shape[0]
    u_b = u.reshape(F, n_t, n_x)
    u_x_b = u_x.reshape(F, n_t, n_x)
    u_xx_b = u_xx.reshape(F, n_t, n_x)
    u_t_fd = (u_b[:, 1:, :] - u_b[:, :-1, :]) / float(dt)
    resid = u_t_fd + u_b[:, 1:, :] * u_x_b[:, 1:, :] - VISCOSITY * u_xx_b[:, 1:, :]
    return torch.mean(resid**2)


def _run_burgers_cell(payload: tuple[Any, ...]) -> dict[str, Any]:
    seed, lam, n_times = payload
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    train = make_burgers_slab(
        n_samples=8,
        n_grid=64,
        n_sensors=32,
        n_modes=2,
        amplitude=0.5,
        viscosity=VISCOSITY,
        t_final=0.5,
        n_times=n_times,
        seed=seed,
    )
    hold = make_burgers_slab(
        n_samples=4,
        n_grid=64,
        n_sensors=32,
        n_modes=2,
        amplitude=0.5,
        viscosity=VISCOSITY,
        t_final=0.5,
        n_times=n_times,
        seed=seed + 100,
    )
    times = torch.unique(train.coords[:, 1], sorted=True)
    dt = float(times[1] - times[0])
    cs = CoordinateSpec(("x", "t"))
    comps = ComponentSpec(("u",))
    out: dict[str, Any] = {
        "kind": "burgers",
        "seed": seed,
        "lambda": lam,
        "n_times": n_times,
        "dt": dt,
    }
    for arm in ("B", "C"):
        torch.manual_seed(seed)
        op = build_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=32,
            trunk_width=32,
            trunk_hidden=64,
            trunk_depth=3,
            branch_hidden=64,
            branch_depth=3,
            jet_order=2,
        )
        opt = torch.optim.Adam(op.parameters(), lr=1e-3)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = data_loss(op, train)
            if arm == "B":
                loss = loss + float(lam) * _burgers_fd_residual(op, train, dt)
            else:
                loss = loss + float(lam) * burgers_residual_loss(
                    op, train.sensors, train.coords, viscosity=VISCOSITY
                )
            loss.backward()
            opt.step()
        out[f"{arm}_train"] = _rel_l2(op, train)
        out[f"{arm}_hold"] = _rel_l2(op, hold)
    return out


def _run_ks_cell(payload: tuple[Any, ...]) -> dict[str, Any]:
    seed, lam, h = payload
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    train = make_ks_slab(
        n_samples=8,
        n_grid=64,
        n_sensors=32,
        n_modes=2,
        amplitude=0.3,
        t_final=0.5,
        n_times=6,
        seed=seed,
    )
    hold = make_ks_slab(
        n_samples=4,
        n_grid=64,
        n_sensors=32,
        n_modes=2,
        amplitude=0.3,
        t_final=0.5,
        n_times=6,
        seed=seed + 100,
    )
    cs = CoordinateSpec(("x", "t"), time_axis="t")
    comps = ComponentSpec(("u",))
    out: dict[str, Any] = {
        "kind": "ks",
        "seed": seed,
        "lambda": lam,
        "h": h,
    }
    for arm in ("B", "C"):
        torch.manual_seed(seed)
        op = build_deeponet(
            coordinate_spec=cs,
            components=comps,
            n_sensors=32,
            trunk_width=32,
            trunk_hidden=64,
            trunk_depth=3,
            branch_hidden=64,
            branch_depth=3,
            jet_order=4,
        )
        opt = torch.optim.Adam(op.parameters(), lr=3e-3)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = data_loss(op, train)
            if arm == "B":
                loss = loss + float(lam) * ks_residual_loss_fd(
                    op, train.sensors, train.coords, h=float(h)
                )
            else:
                loss = loss + float(lam) * ks_residual_loss(
                    op, train.sensors, train.coords
                )
            loss.backward()
            opt.step()
        out[f"{arm}_train"] = _rel_l2(op, train)
        out[f"{arm}_hold"] = _rel_l2(op, hold)
    return out


def _aggregate(
    cells: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for c in cells:
        k = tuple(c[f] for f in key_fields)
        groups.setdefault(k, []).append(c)
    rows = []
    for k, group in sorted(groups.items(), key=lambda kv: kv[0]):
        b = [g["B_hold"] for g in group if g["B_hold"] == g["B_hold"]]
        c = [g["C_hold"] for g in group if g["C_hold"] == g["C_hold"]]
        cleared = all(
            g["B_train"] <= TRAIN_REL_L2_MAX and g["C_train"] <= TRAIN_REL_L2_MAX
            for g in group
        )
        med_b = float(statistics.median(b)) if b else float("nan")
        med_c = float(statistics.median(c)) if c else float("nan")
        row = {f: v for f, v in zip(key_fields, k, strict=True)}
        row.update(
            {
                "median_B_hold": med_b,
                "median_C_hold": med_c,
                "gap_B_minus_C": med_b - med_c,
                "c_beats_b_seeds": sum(
                    1 for g in group if g["C_hold"] < g["B_hold"]
                ),
                "n_seeds": len(group),
                "train_cleared": cleared,
            }
        )
        rows.append(row)
    return rows


def _pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cleared = [r for r in rows if r["train_cleared"]]
    pool = cleared if cleared else rows
    # Prefer positive gap (C better); else largest absolute gap with note.
    positive = [r for r in pool if r["gap_B_minus_C"] > 0]
    if positive:
        best = max(positive, key=lambda r: r["gap_B_minus_C"])
        best = dict(best)
        best["selection"] = "max_positive_gap_among_cleared"
        return best
    best = max(pool, key=lambda r: abs(r["gap_B_minus_C"]))
    best = dict(best)
    best["selection"] = "max_abs_gap_no_positive_separation"
    return best


def main() -> None:
    t0 = time.perf_counter()
    n_workers = min(max(len(SEEDS), 1), os.cpu_count() or 1)

    # Burgers: lambda sweep at coarsest n_times, then n_times at best lambda.
    burgers_jobs = [(s, lam, 3) for lam in LAMBDAS for s in SEEDS]
    burgers_cells: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_burgers_cell, j): j for j in burgers_jobs}
        for fut in as_completed(futs):
            burgers_cells.append(fut.result())
    burgers_lambda_rows = _aggregate(burgers_cells, ("lambda", "n_times"))
    best_lam_row = _pick(burgers_lambda_rows)
    best_lam = float(best_lam_row["lambda"])

    burgers_nt_jobs = [(s, best_lam, nt) for nt in N_TIMES_LIST for s in SEEDS]
    # Skip re-running n_times=3 at best_lam (already have those cells).
    burgers_nt_cells = [
        c for c in burgers_cells if c["lambda"] == best_lam and c["n_times"] == 3
    ]
    remaining = [j for j in burgers_nt_jobs if j[2] != 3]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_burgers_cell, j): j for j in remaining}
        for fut in as_completed(futs):
            burgers_nt_cells.append(fut.result())
    burgers_nt_rows = _aggregate(burgers_nt_cells, ("lambda", "n_times"))
    best_burgers = _pick(burgers_nt_rows)

    # KS: h sweep at lambda=1.
    ks_jobs = [(s, 1.0, h) for h in KS_HS for s in SEEDS]
    ks_cells: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_ks_cell, j): j for j in ks_jobs}
        for fut in as_completed(futs):
            ks_cells.append(fut.result())
    ks_rows = _aggregate(ks_cells, ("lambda", "h"))
    best_ks = _pick(ks_rows)

    payload = provenance(
        schema="operator_residual_calibration/v1",
        config={
            "seeds": list(SEEDS),
            "steps": STEPS,
            "train_rel_l2_max": TRAIN_REL_L2_MAX,
            "lambdas": list(LAMBDAS),
            "n_times_list": list(N_TIMES_LIST),
            "ks_hs": list(KS_HS),
            "ks_slab": {
                "n_grid": 64,
                "n_sensors": 32,
                "n_modes": 2,
                "amplitude": 0.3,
                "t_final": 0.5,
                "n_times": 6,
                "lr": 3e-3,
            },
            "decision_rule": (
                "maximise median(B_hold)-median(C_hold) among train-cleared "
                "configs; else max abs gap"
            ),
        },
    )
    payload["burgers_lambda_rows"] = burgers_lambda_rows
    payload["burgers_n_times_rows"] = burgers_nt_rows
    payload["ks_h_rows"] = ks_rows
    payload["chosen"] = {
        "burgers": {
            "lambda": best_burgers["lambda"],
            "n_times": best_burgers["n_times"],
            "gap_B_minus_C": best_burgers["gap_B_minus_C"],
            "selection": best_burgers["selection"],
        },
        "ks": {
            "lambda": best_ks["lambda"],
            "h": best_ks["h"],
            "gap_B_minus_C": best_ks["gap_B_minus_C"],
            "selection": best_ks["selection"],
        },
    }
    seen = {
        (c["lambda"], c["n_times"], c["seed"]) for c in burgers_cells
    }
    burgers_all = list(burgers_cells)
    for c in burgers_nt_cells:
        key = (c["lambda"], c["n_times"], c["seed"])
        if key not in seen:
            burgers_all.append(c)
            seen.add(key)
    payload["cells"] = {
        "burgers": sorted(
            burgers_all, key=lambda r: (r["lambda"], r["n_times"], r["seed"])
        ),
        "ks": sorted(ks_cells, key=lambda r: (r["h"], r["seed"])),
    }
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")
    print(f"chosen burgers: {payload['chosen']['burgers']}")
    print(f"chosen ks: {payload['chosen']['ks']}")


if __name__ == "__main__":
    main()
