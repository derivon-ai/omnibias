# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""KS DeepONet bake-off: data-only vs FD-u_xxxx residual vs closed-form residual.

Three arms at identical architecture, parameter count, seed and step budget on
periodic Kuramoto-Sivashinsky (the equation that puts ``u_xxxx`` in the
residual, where the measured FD floor is 4.5e-6):

* **A** data-only Adam
* **B** physics-informed, ``u_xxxx`` by 5-point FD (``ks_residual_loss_fd``)
* **C** physics-informed, closed-form trunk jet (``ks_residual_loss``)

Decision rule (stated *before* the run):

* C beats B on median held-out rel-L2 **and** on at least 6 of 8 seeds -> claim
  the 4th-order FD-floor win in training.
* Otherwise report the honest negative and keep only the structural claims
  (exact order-4 derivatives, one-jet residual, measured FD floor).

Beating data-only (A) is **not** required and is not claimed.

Run::

    uv run python benchmarks/operator_ks_bakeoff.py
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
    data_loss,
    ks_residual_loss,
    ks_residual_loss_fd,
    make_ks_slab,
)
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(
    int(s) for s in os.environ.get("OP_KS_SEEDS", "0,1,2,3,4,5,6,7").split(",")
)
STEPS = int(os.environ.get("OP_KS_STEPS", "2000"))
OUT_NAME = os.environ.get("OP_KS_OUT", "operator_ks_bakeoff.json")
# Calibration maximised C-vs-B gap at lambda=1 / h=0.01, but lambda=1 failed the
# train-rel-L2 guard on some seeds; lambda=0.1 clears the guard while keeping h.
PHYS_WEIGHT = float(os.environ.get("OP_KS_PHYS_WEIGHT", "0.1"))
FD_H = float(os.environ.get("OP_KS_FD_H", "0.01"))
TRAIN_REL_L2_MAX = float(os.environ.get("OP_KS_TRAIN_REL_MAX", "0.35"))
LR = float(os.environ.get("OP_KS_LR", "0.003"))
N_SAMPLES_TRAIN = 8
N_SAMPLES_HOLD = 4
N_GRID = 64
N_SENSORS = 32
N_MODES = 2
AMPLITUDE = 0.3
T_FINAL = 0.5
N_TIMES = 6
TRUNK_WIDTH = 32
HIDDEN = 64
DEPTH = 3


def _rel_l2(op: Any, slab: Any) -> float:
    with torch.no_grad():
        field = op.condition(slab.sensors)
        state = field.on_grid(slab.coords)
        pred = tops.value(state, "u").reshape(slab.values.shape[0], -1)
        target = slab.values[..., 0]
        return float(
            (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()
        )


def _fd_vs_closed_u_xxxx(op: Any, sensors: torch.Tensor, coords: torch.Tensor, h: float) -> float:
    """Mechanism diagnostic: max |FD u_xxxx - closed-form u_xxxx| on a query grid."""
    field = op.condition(sensors[:1])
    state = field.on_grid(coords)
    closed = tops.derivative(state, "u", axis=0, order=4)

    def _val(dx: float) -> torch.Tensor:
        shifted = coords.clone()
        shifted[:, 0] = shifted[:, 0] + float(dx)
        return tops.value(field.on_grid(shifted), "u")

    u0 = tops.value(state, "u")
    hh = float(h)
    fd = (
        _val(-2 * hh) - 4 * _val(-hh) + 6 * u0 - 4 * _val(hh) + _val(2 * hh)
    ) / (hh**4)
    return float((fd - closed).abs().max().detach())


def _build_op(seed: int) -> Any:
    torch.manual_seed(seed)
    return build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=N_SENSORS,
        trunk_width=TRUNK_WIDTH,
        trunk_hidden=HIDDEN,
        trunk_depth=DEPTH,
        branch_hidden=HIDDEN,
        branch_depth=DEPTH,
        jet_order=4,
    )


def _run_seed(seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    train = make_ks_slab(
        n_samples=N_SAMPLES_TRAIN,
        n_grid=N_GRID,
        n_sensors=N_SENSORS,
        n_modes=N_MODES,
        amplitude=AMPLITUDE,
        t_final=T_FINAL,
        n_times=N_TIMES,
        seed=seed,
    )
    hold = make_ks_slab(
        n_samples=N_SAMPLES_HOLD,
        n_grid=N_GRID,
        n_sensors=N_SENSORS,
        n_modes=N_MODES,
        amplitude=AMPLITUDE,
        t_final=T_FINAL,
        n_times=N_TIMES,
        seed=seed + 100,
    )

    def train_arm(arm: str) -> dict[str, float]:
        op = _build_op(seed)
        opt = torch.optim.Adam(op.parameters(), lr=LR)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            opt.zero_grad()
            loss = data_loss(op, train)
            if arm == "B":
                loss = loss + PHYS_WEIGHT * ks_residual_loss_fd(
                    op, train.sensors, train.coords, h=FD_H
                )
            elif arm == "C":
                loss = loss + PHYS_WEIGHT * ks_residual_loss(
                    op, train.sensors, train.coords
                )
            loss.backward()
            opt.step()
        wall = time.perf_counter() - t0
        train_err = _rel_l2(op, train)
        hold_err = _rel_l2(op, hold)
        if train_err > TRAIN_REL_L2_MAX:
            raise RuntimeError(
                f"arm {arm} seed {seed}: train rel-L2 {train_err:.4f} > "
                f"{TRAIN_REL_L2_MAX} after {STEPS} steps"
            )
        mech = _fd_vs_closed_u_xxxx(op, train.sensors, train.coords[:32], FD_H)
        return {
            "hold_rel_l2": hold_err,
            "train_rel_l2": train_err,
            "wall_seconds": wall,
            "fd_vs_closed_u_xxxx": mech,
        }

    a = train_arm("A")
    b = train_arm("B")
    c = train_arm("C")
    return {
        "seed": seed,
        "A_data_only": a["hold_rel_l2"],
        "B_fd_residual": b["hold_rel_l2"],
        "C_closed_form": c["hold_rel_l2"],
        "A_train_rel_l2": a["train_rel_l2"],
        "B_train_rel_l2": b["train_rel_l2"],
        "C_train_rel_l2": c["train_rel_l2"],
        "A_wall_seconds": a["wall_seconds"],
        "B_wall_seconds": b["wall_seconds"],
        "C_wall_seconds": c["wall_seconds"],
        "B_fd_vs_closed_u_xxxx": b["fd_vs_closed_u_xxxx"],
        "C_fd_vs_closed_u_xxxx": c["fd_vs_closed_u_xxxx"],
    }


def _iqr(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    qs = statistics.quantiles(xs, n=4)
    return float(qs[2] - qs[0])


def _decide(cells: list[dict[str, Any]]) -> dict[str, Any]:
    a = [c["A_data_only"] for c in cells]
    b = [c["B_fd_residual"] for c in cells]
    c = [c["C_closed_form"] for c in cells]
    med = {
        "A": float(statistics.median(a)),
        "B": float(statistics.median(b)),
        "C": float(statistics.median(c)),
    }
    c_beats_b = sum(
        1 for row in cells if row["C_closed_form"] < row["B_fd_residual"]
    )
    a_beats_both = sum(
        1
        for row in cells
        if row["A_data_only"] < row["B_fd_residual"]
        and row["A_data_only"] < row["C_closed_form"]
    )
    n = len(cells)
    if med["C"] < med["B"] and c_beats_b >= 6:
        verdict = (
            "closed-form residual (C) beats FD residual (B) on median held-out "
            f"rel-L2 and on {c_beats_b}/{n} seeds; the 4th-order FD floor is "
            "visible in trained operator error under this budget"
        )
    elif med["C"] < med["B"]:
        verdict = (
            "closed-form residual (C) beats FD residual (B) on median held-out "
            f"rel-L2 but only on {c_beats_b}/{n} seeds -- seed-fragile; keep "
            "the structural claims (exact order-4 derivatives, one-jet residual, "
            "measured FD floor) and do not advertise a seed-stable training win"
        )
    else:
        verdict = (
            "closed-form residual (C) did not beat FD residual (B) on median "
            "held-out rel-L2 under this budget; keep only the structural claims "
            "(exact order-4 derivatives, one-jet residual, measured FD floor)"
        )
    return {
        "median_rel_l2": med,
        "iqr_rel_l2": {
            "A": _iqr(a),
            "B": _iqr(b),
            "C": _iqr(c),
        },
        "median_wall_seconds": {
            "A": float(statistics.median([c["A_wall_seconds"] for c in cells])),
            "B": float(statistics.median([c["B_wall_seconds"] for c in cells])),
            "C": float(statistics.median([c["C_wall_seconds"] for c in cells])),
        },
        "median_fd_vs_closed_u_xxxx": {
            "B": float(
                statistics.median([c["B_fd_vs_closed_u_xxxx"] for c in cells])
            ),
            "C": float(
                statistics.median([c["C_fd_vs_closed_u_xxxx"] for c in cells])
            ),
        },
        "c_beats_b_seeds": c_beats_b,
        "a_beats_both_seeds": a_beats_both,
        "n_seeds": n,
        "verdict": verdict,
    }


def main() -> None:
    t0 = time.perf_counter()
    cells: list[dict[str, Any]] = []
    n_workers = min(len(SEEDS), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_seed, s): s for s in SEEDS}
        for fut in as_completed(futs):
            cells.append(fut.result())
    cells.sort(key=lambda r: r["seed"])
    decision = _decide(cells)
    payload = provenance(
        schema="operator_ks_bakeoff/v1",
        config={
            "seeds": list(SEEDS),
            "steps": STEPS,
            "phys_weight": PHYS_WEIGHT,
            "fd_h": FD_H,
            "lr": LR,
            "train_rel_l2_max": TRAIN_REL_L2_MAX,
            "n_samples_train": N_SAMPLES_TRAIN,
            "n_samples_hold": N_SAMPLES_HOLD,
            "n_grid": N_GRID,
            "n_sensors": N_SENSORS,
            "n_modes": N_MODES,
            "amplitude": AMPLITUDE,
            "t_final": T_FINAL,
            "n_times": N_TIMES,
            "trunk_width": TRUNK_WIDTH,
            "hidden": HIDDEN,
            "depth": DEPTH,
            "jet_order": 4,
            "budget_kind": "step_count",
            "decision_rule": (
                "C beats B on median held-out rel-L2 AND on >=6/8 seeds -> "
                "claim the 4th-order FD-floor training win; else honest "
                "negative / structural-only. Beating A is not required."
            ),
            "smoke_note": (
                "KS slab shortened vs house stiff-test (n_grid=64, t_final=0.5, "
                "amplitude=0.3) so a DeepONet clears train-rel-L2<=0.35 at "
                "lr=3e-3 / 2000 steps on CPU. Calibration picked h=0.01 and "
                "lambda=1 for max C-vs-B gap, but lambda=1 failed the train "
                "guard on seed 4 (arm B train=0.58); bake-off uses lambda=0.1 "
                "which clears the guard. Mechanism diagnostic still uses the "
                "measured FD u_xxxx floor."
            ),
        },
    )
    payload["cells"] = cells
    payload["decision"] = decision
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")
    print(f"median rel-L2: {decision['median_rel_l2']}")
    print(f"c_beats_b_seeds: {decision['c_beats_b_seeds']}/{decision['n_seeds']}")
    print(f"verdict: {decision['verdict']}")


if __name__ == "__main__":
    main()
