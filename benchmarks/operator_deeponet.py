# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepONet operator learning: data-only vs FD-residual vs closed-form residual.

Three arms at identical architecture, parameter count, seed and **step budget**
on periodic viscous Burgers, with ground truth from the spectral MOL reference:

* **A** data-only DeepONet + Adam (Lu et al. classical baseline)
* **B** physics-informed, ``u_t`` by finite difference (§3a convention)
* **C** physics-informed, closed-form trunk jet (omnibias)

Arms B and C share the same order-2 jet and differ only in how ``u_t`` is
obtained, so they are near iso-cost. The comparison is step-budget matched, not
wall-clock matched; per-arm wall clock is recorded separately. The FD time
step is derived from the slab's time column (never a hardcoded ``DT``).

Decision rules (stated *before* the run):

* The C-vs-B comparison is the whole claim. If closed-form residuals beat FD
  residuals on held-out operator error at equal step count, the FD truncation
  error in the physics loss is real; if not, say so and keep only the
  structural claims. An honest negative on Burgers is acceptable: the KS
  bake-off carries the 4th-order claim.
* If A beats both, report it -- the honest-negative pattern of §3a.
* A convergence guard fails loudly if any arm's train rel-L2 stays above the
  threshold: we refuse to compare untrained models.

Run::

    uv run python benchmarks/operator_deeponet.py
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
    make_burgers_slab,
)
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(
    int(s) for s in os.environ.get("OP_SEEDS", "0,1,2,3,4,5,6,7").split(",")
)
STEPS = int(os.environ.get("OP_STEPS", "1500"))
OUT_NAME = os.environ.get("OP_OUT", "operator_deeponet.json")
VISCOSITY = 0.05
# Defaults match docs/benchmarks/operator_residual_calibration.json (chosen.burgers).
PHYS_WEIGHT = float(os.environ.get("OP_PHYS_WEIGHT", "0.1"))
N_TIMES = int(os.environ.get("OP_N_TIMES", "11"))
TRAIN_REL_L2_MAX = float(os.environ.get("OP_TRAIN_REL_MAX", "0.35"))
N_SAMPLES_TRAIN = 16
N_SAMPLES_HOLD = 8
N_GRID = 64
N_SENSORS = 32
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


def _slab_dt(slab: Any) -> float:
    times = torch.unique(slab.coords[:, 1], sorted=True)
    if times.numel() < 2:
        raise ValueError("slab must have at least two distinct times for FD dt")
    return float(times[1] - times[0])


def _burgers_fd_residual(op: Any, slab: Any, dt: float) -> torch.Tensor:
    """FD-in-time Burgers residual on the product grid (arm B)."""
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


def _build_op(seed: int) -> Any:
    torch.manual_seed(seed)
    return build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=N_SENSORS,
        trunk_width=TRUNK_WIDTH,
        trunk_hidden=HIDDEN,
        trunk_depth=DEPTH,
        branch_hidden=HIDDEN,
        branch_depth=DEPTH,
        jet_order=2,
    )


def _run_seed(seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    train = make_burgers_slab(
        n_samples=N_SAMPLES_TRAIN,
        n_grid=N_GRID,
        n_sensors=N_SENSORS,
        n_modes=2,
        amplitude=0.5,
        viscosity=VISCOSITY,
        t_final=0.5,
        n_times=N_TIMES,
        seed=seed,
    )
    hold = make_burgers_slab(
        n_samples=N_SAMPLES_HOLD,
        n_grid=N_GRID,
        n_sensors=N_SENSORS,
        n_modes=2,
        amplitude=0.5,
        viscosity=VISCOSITY,
        t_final=0.5,
        n_times=N_TIMES,
        seed=seed + 100,
    )
    dt = _slab_dt(train)

    def train_arm(arm: str) -> dict[str, float]:
        op = _build_op(seed)
        opt = torch.optim.Adam(op.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            opt.zero_grad()
            loss = data_loss(op, train)
            if arm == "B":
                loss = loss + PHYS_WEIGHT * _burgers_fd_residual(op, train, dt)
            elif arm == "C":
                loss = loss + PHYS_WEIGHT * burgers_residual_loss(
                    op, train.sensors, train.coords, viscosity=VISCOSITY
                )
            loss.backward()
            opt.step()
        wall = time.perf_counter() - t0
        train_err = _rel_l2(op, train)
        hold_err = _rel_l2(op, hold)
        if train_err > TRAIN_REL_L2_MAX:
            raise RuntimeError(
                f"arm {arm} seed {seed}: train rel-L2 {train_err:.4f} > "
                f"{TRAIN_REL_L2_MAX} after {STEPS} steps -- refusing to compare "
                f"untrained models; raise OP_STEPS or lower OP_TRAIN_REL_MAX"
            )
        return {
            "hold_rel_l2": hold_err,
            "train_rel_l2": train_err,
            "wall_seconds": wall,
        }

    a = train_arm("A")
    b = train_arm("B")
    c = train_arm("C")
    return {
        "seed": seed,
        "dt": dt,
        "A_data_only": a["hold_rel_l2"],
        "B_fd_residual": b["hold_rel_l2"],
        "C_closed_form": c["hold_rel_l2"],
        "A_train_rel_l2": a["train_rel_l2"],
        "B_train_rel_l2": b["train_rel_l2"],
        "C_train_rel_l2": c["train_rel_l2"],
        "A_wall_seconds": a["wall_seconds"],
        "B_wall_seconds": b["wall_seconds"],
        "C_wall_seconds": c["wall_seconds"],
    }


def _iqr(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    qs = statistics.quantiles(xs, n=4)
    return float(qs[2] - qs[0])


def _decide(cells: list[dict[str, Any]]) -> dict[str, Any]:
    def _finite(xs: list[float]) -> list[float]:
        return [x for x in xs if x == x]

    a = _finite([c["A_data_only"] for c in cells])
    b = _finite([c["B_fd_residual"] for c in cells])
    c = _finite([c["C_closed_form"] for c in cells])
    if not a or not b or not c:
        return {
            "median_rel_l2": {"A": None, "B": None, "C": None},
            "c_beats_b_seeds": 0,
            "a_beats_both_seeds": 0,
            "n_seeds": len(cells),
            "n_finite": {"A": len(a), "B": len(b), "C": len(c)},
            "verdict": "insufficient finite seeds; regenerate with a stabler IC budget",
        }
    med = {
        "A": float(statistics.median(a)),
        "B": float(statistics.median(b)),
        "C": float(statistics.median(c)),
    }
    c_beats_b = sum(
        1
        for row in cells
        if row["C_closed_form"] == row["C_closed_form"]
        and row["B_fd_residual"] == row["B_fd_residual"]
        and row["C_closed_form"] < row["B_fd_residual"]
    )
    a_beats_both = sum(
        1
        for row in cells
        if row["A_data_only"] == row["A_data_only"]
        and row["B_fd_residual"] == row["B_fd_residual"]
        and row["C_closed_form"] == row["C_closed_form"]
        and row["A_data_only"] < row["B_fd_residual"]
        and row["A_data_only"] < row["C_closed_form"]
    )
    n = len(cells)
    if a_beats_both == n and med["C"] < med["B"]:
        verdict = (
            "honest negative on the physics residual at this CPU budget: "
            "data-only Adam (A) beats both physics-informed arms on every "
            "seed (mirroring docs/benchmarks.md §3a). Separately, closed-form "
            f"(C) beats FD (B) on {c_beats_b}/{n} seeds / median; the KS "
            "bake-off carries the 4th-order claim. Structural claims stand"
        )
    elif a_beats_both == n:
        verdict = (
            "data-only Adam (A) beats both physics-informed arms on every seed "
            "-- the honest-negative pattern of docs/benchmarks.md §3a. Keep "
            "the structural claims; the KS bake-off carries the 4th-order claim"
        )
    elif med["C"] < med["B"]:
        verdict = (
            "closed-form residual (C) beats FD residual (B) on median held-out "
            f"rel-L2 ({c_beats_b}/{n} seeds); the FD truncation error in the "
            "physics loss is real under this budget"
        )
    else:
        verdict = (
            "closed-form residual (C) did not beat FD residual (B) on median "
            "held-out rel-L2 under this budget; keep the structural claims and "
            "let the KS bake-off carry the 4th-order claim"
        )
    return {
        "median_rel_l2": med,
        "iqr_rel_l2": {"A": _iqr(a), "B": _iqr(b), "C": _iqr(c)},
        "median_wall_seconds": {
            "A": float(statistics.median([c["A_wall_seconds"] for c in cells])),
            "B": float(statistics.median([c["B_wall_seconds"] for c in cells])),
            "C": float(statistics.median([c["C_wall_seconds"] for c in cells])),
        },
        "c_beats_b_seeds": c_beats_b,
        "a_beats_both_seeds": a_beats_both,
        "n_seeds": n,
        "n_finite": {"A": len(a), "B": len(b), "C": len(c)},
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
        schema="operator_deeponet/v1",
        config={
            "seeds": list(SEEDS),
            "steps": STEPS,
            "viscosity": VISCOSITY,
            "phys_weight": PHYS_WEIGHT,
            "n_times": N_TIMES,
            "dt_source": "derived_from_slab_time_column",
            "n_samples_train": N_SAMPLES_TRAIN,
            "n_samples_hold": N_SAMPLES_HOLD,
            "n_grid": N_GRID,
            "n_sensors": N_SENSORS,
            "n_modes": 2,
            "amplitude": 0.5,
            "trunk_width": TRUNK_WIDTH,
            "hidden": HIDDEN,
            "depth": DEPTH,
            "train_rel_l2_max": TRAIN_REL_L2_MAX,
            "budget_kind": "step_count",
            "decision_rule": (
                "C beats B on median held-out rel-L2 -> advertise the FD "
                "truncation win; otherwise keep only structural claims / KS "
                "bake-off. If A beats both, report the honest negative. Refuse "
                "the run if any arm fails the train-rel-L2 convergence guard."
            ),
        },
    )
    payload["cells"] = cells
    payload["decision"] = decision
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")
    print(f"median rel-L2: {decision['median_rel_l2']}")
    print(f"median wall s: {decision.get('median_wall_seconds')}")
    print(f"c_beats_b_seeds: {decision['c_beats_b_seeds']}/{decision['n_seeds']}")
    print(f"verdict: {decision['verdict']}")


if __name__ == "__main__":
    main()
