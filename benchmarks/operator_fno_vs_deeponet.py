# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FNO vs DeepONet on the same Burgers operator target (matched params + steps).

Both arms train on spectral-MOL Burgers slabs at equal step budget. Parameter
counts are matched approximately by choosing FNO width / modes against the
DeepONet trunk/branch sizes. The artifact records the honest asymmetry: FNO
supplies no off-grid ``u_t`` or ``u_xxxx`` at all -- its spatial derivatives
are FFT-based and grid-bound -- so the closed-form operator claim does not
transfer.

Run::

    uv run python benchmarks/operator_fno_vs_deeponet.py
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
    build_fno1d,
    data_loss,
    make_burgers_slab,
)
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(int(s) for s in os.environ.get("OP_FNO_SEEDS", "0,1,2,3,4").split(","))
STEPS = int(os.environ.get("OP_FNO_STEPS", "1500"))
OUT_NAME = os.environ.get("OP_FNO_OUT", "operator_fno_vs_deeponet.json")
VISCOSITY = 0.05
N_GRID = 64
N_SENSORS = 32
N_SAMPLES_TRAIN = 16
N_SAMPLES_HOLD = 8
# DeepONet size (same as operator_deeponet.py bake-off).
TRUNK_WIDTH = 32
HIDDEN = 64
DEPTH = 3
# FNO size chosen for roughly matched parameter count (~20k vs DeepONet ~23k).
FNO_MODES = 8
FNO_WIDTH = 20
FNO_LAYERS = 3


def _deeponet_rel_l2(op: Any, slab: Any) -> float:
    with torch.no_grad():
        field = op.condition(slab.sensors)
        state = field.on_grid(slab.coords)
        pred = tops.value(state, "u").reshape(slab.values.shape[0], -1)
        target = slab.values[..., 0]
        return float(
            (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()
        )


def _fno_targets(slab: Any) -> torch.Tensor:
    """Final-time spatial snapshot per sample: shape ``(F, n_grid)``."""
    n_x = slab.grid.n
    n_t = int(torch.unique(slab.coords[:, 1]).numel())
    u = slab.values[..., 0].reshape(slab.values.shape[0], n_t, n_x)
    return u[:, -1, :]


def _fno_rel_l2(fno: Any, sensors: torch.Tensor, target: torch.Tensor) -> float:
    # Sensors are a subsample of the fine grid; upsample by nearest index for FNO
    # input on the full grid when n_sensors != n_grid.
    with torch.no_grad():
        if sensors.shape[-1] == target.shape[-1]:
            u0 = sensors
        else:
            # Tile / interpolate sensors onto n_grid by linear upsample.
            u0 = torch.nn.functional.interpolate(
                sensors[:, None, :],
                size=target.shape[-1],
                mode="linear",
                align_corners=True,
            )[:, 0, :]
        pred = fno(u0)[..., 0]
        return float(
            (torch.linalg.norm(pred - target) / torch.linalg.norm(target)).item()
        )


def _count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


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
        n_times=11,
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
        n_times=11,
        seed=seed + 100,
    )
    train_fno_y = _fno_targets(train)
    hold_fno_y = _fno_targets(hold)

    # DeepONet: full space-time operator.
    torch.manual_seed(seed)
    deeponet = build_deeponet(
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
    n_deeponet = _count_params(deeponet)
    opt_d = torch.optim.Adam(deeponet.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt_d.zero_grad()
        data_loss(deeponet, train).backward()
        opt_d.step()
    deeponet_wall = time.perf_counter() - t0
    deeponet_hold = _deeponet_rel_l2(deeponet, hold)

    # FNO: final-time map u0(x) -> u(x, t_final) on the periodic grid.
    torch.manual_seed(seed)
    fno = build_fno1d(
        modes=FNO_MODES, width=FNO_WIDTH, n_layers=FNO_LAYERS, dtype=torch.float64
    )
    n_fno = _count_params(fno)
    opt_f = torch.optim.Adam(fno.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt_f.zero_grad()
        u0 = torch.nn.functional.interpolate(
            train.sensors[:, None, :],
            size=N_GRID,
            mode="linear",
            align_corners=True,
        )[:, 0, :]
        pred = fno(u0)[..., 0]
        loss = torch.mean((pred - train_fno_y) ** 2)
        loss.backward()
        opt_f.step()
    fno_wall = time.perf_counter() - t0
    fno_hold = _fno_rel_l2(fno, hold.sensors, hold_fno_y)

    return {
        "seed": seed,
        "deeponet_hold_rel_l2": deeponet_hold,
        "fno_hold_rel_l2": fno_hold,
        "deeponet_params": n_deeponet,
        "fno_params": n_fno,
        "deeponet_wall_seconds": deeponet_wall,
        "fno_wall_seconds": fno_wall,
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
    med_d = float(statistics.median([c["deeponet_hold_rel_l2"] for c in cells]))
    med_f = float(statistics.median([c["fno_hold_rel_l2"] for c in cells]))
    payload = provenance(
        schema="operator_fno_vs_deeponet/v1",
        config={
            "seeds": list(SEEDS),
            "steps": STEPS,
            "viscosity": VISCOSITY,
            "n_samples_train": N_SAMPLES_TRAIN,
            "n_samples_hold": N_SAMPLES_HOLD,
            "n_grid": N_GRID,
            "n_sensors": N_SENSORS,
            "deeponet": {
                "trunk_width": TRUNK_WIDTH,
                "hidden": HIDDEN,
                "depth": DEPTH,
            },
            "fno": {
                "modes": FNO_MODES,
                "width": FNO_WIDTH,
                "n_layers": FNO_LAYERS,
            },
            "asymmetry": (
                "FNO evaluates only on its periodic grid and supplies no "
                "off-grid u_t or u_xxxx; DeepONet query derivatives are "
                "closed-form from one trunk jet"
            ),
            "budget_kind": "step_count",
            "target_note": (
                "DeepONet is scored on full space-time slabs; FNO is scored on "
                "the final-time spatial snapshot (the natural FNO target)"
            ),
        },
    )
    payload["cells"] = cells
    payload["decision"] = {
        "median_hold_rel_l2": {"deeponet": med_d, "fno": med_f},
        "median_params": {
            "deeponet": int(statistics.median([c["deeponet_params"] for c in cells])),
            "fno": int(statistics.median([c["fno_params"] for c in cells])),
        },
        "n_seeds": len(cells),
        "verdict": (
            f"median hold rel-L2: DeepONet={med_d:.4f}, FNO={med_f:.4f}. "
            "FNO cannot supply off-grid u_t or u_xxxx; that asymmetry is the "
            "structural point of the comparison, independent of which arm wins "
            "on grid-restricted final-time error."
        ),
    }
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")
    print(payload["decision"]["verdict"])


if __name__ == "__main__":
    main()
